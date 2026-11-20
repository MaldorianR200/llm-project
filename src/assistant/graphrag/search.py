"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from collections import Counter
from collections.abc import Iterable
from itertools import combinations, pairwise

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

REFUSAL = "В базе знаний не найдено"


def _local_edges(graph: nx.DiGraph, anchors: list[str], *, supplier: bool = False) -> list[tuple[str, str]]:
    """Выбирает ребра вокруг опорных узлов или на путях между ними"""
    undirected = graph.to_undirected(as_view=True)
    edges: set[tuple[str, str]] = set()
    parts: set[str] = set()
    if len(anchors) == 1:
        nearby = nx.single_source_shortest_path_length(undirected, anchors[0], cutoff=2)
        edges.update(graph.subgraph(nearby).edges)
        if not supplier or graph.nodes[anchors[0]]["type"] != "Part":
            return sorted(edges)
        parts.add(anchors[0])

    for first, last in combinations(anchors, 2):
        for path in nx.all_simple_paths(undirected, first, last, cutoff=3):
            parts.update(node for node in path if graph.nodes[node]["type"] == "Part")
            for source, target in pairwise(path):
                if graph.has_edge(source, target):
                    edges.add((source, target))
                if graph.has_edge(target, source):
                    edges.add((target, source))

    pending = list(parts)
    while pending:
        source = pending.pop()
        for target, attrs in graph[source].items():
            if attrs["type"] == "REPLACED_BY":
                edges.add((source, target))
                if target not in parts:
                    parts.add(target)
                    pending.append(target)
    if supplier:
        for part in parts:
            outgoing = graph[part]
            if all(attrs["type"] != "REPLACED_BY" for attrs in outgoing.values()):
                edges.update((part, target) for target, attrs in outgoing.items() if attrs["type"] == "SUPPLIED_BY")
    return sorted(edges, key=lambda edge: (graph.edges[edge]["type"] != "REPLACED_BY", edge))


def _answer(text: str, sources: set[str], facts: list[str] | None = None) -> Answer:
    """Собирает ответ из текста, источников и фактов или возвращает отказ"""
    text = text.strip()
    last_line = text.rsplit("\n", 1)[-1].replace("*", "").strip().casefold()
    conclusion = last_line.removeprefix("ответ:").strip(' .«»"')
    if not text or conclusion == REFUSAL.casefold():
        return Answer(REFUSAL, refused=True)
    return Answer(text, sources=sorted(sources), facts=facts or [])


def _document_ids(chunks: Iterable[str]) -> set[str]:
    """Возвращает уникальные имена документов по идентификаторам фрагментов"""
    return {chunk.split("#", 1)[0] for chunk in chunks}


def _local_context(graph: nx.DiGraph, anchors: list[str], *, supplier: bool = False) -> tuple[list[str], set[str]]:
    """Формирует пронумерованные факты в пределах лимита и собирает их источники"""
    facts: list[str] = []
    sources: set[str] = set()
    length = 0
    for source, target in _local_edges(graph, anchors, supplier=supplier):
        attrs = graph.edges[source, target]
        fact = f"{graph.nodes[source]['name']} -{attrs['type']}-> {graph.nodes[target]['name']}"
        line = f"{len(facts) + 1}. {fact}"
        separator_length = 1 if facts else 0
        length += len(line) + separator_length
        if length > 8000:
            break
        facts.append(line)
        sources.update(_document_ids(attrs["sources"]))
    return facts, sources


def _asks_supplier(question: str) -> bool:
    text = question.casefold()
    return "поставщи" in text or "поставля" in text


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    graph = kb.graph
    code_nodes = [f"{code_type(code)}:{code}" for code in find_codes(question)]
    if any(node not in graph for node in code_nodes):
        return Answer(REFUSAL, refused=True)
    overview = _failure_ranking(question, kb) or _retired_parts(question, kb)
    if overview:
        conclusion, _, sources = overview
        return _answer(conclusion, sources)
    name_nodes = [f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary)]
    anchors = [node for node in dict.fromkeys(code_nodes + name_nodes) if node in graph]
    if not anchors:
        return Answer(REFUSAL, refused=True)

    asks_supplier = _asks_supplier(question)
    facts, sources = _local_context(graph, anchors, supplier=asks_supplier)
    if not facts:
        return Answer(REFUSAL, refused=True)

    context = "\n".join(facts)
    reply = llm.chat([
        {
            "role": "system",
            "content": (
                "Ответь на вопрос только по фактам. Для цепочки REPLACED_BY следуй от старой детали "
                "к новой и для заказа укажи последнюю деталь. Сохраняй коды точно. "
                f"Если фактов недостаточно, ответь \"{REFUSAL}\"."
                + (" Для актуального поставщика используй SUPPLIED_BY последней детали в цепочке." if asks_supplier
                   else "")
            ),
        },
        {"role": "user", "content": f"Вопрос: {question}\nФакты:\n{context}"},
    ])
    return _answer(reply.content, sources, facts)


def _community_answer(question: str, summary: str, llm: LLMClient) -> str | None:
    """Возвращает полезный частичный ответ по одной сводке или None."""
    reply = llm.chat(
        [
            {
                "role": "system",
                "content": (
                    "Дай частичный ответ только по сводке. Сохрани числа и коды. "
                    "Поставь score от 1 до 10 по полезности или 0, если сведений по вопросу нет. "
                    "Для сравнения частоты отказов перечисли все подходящие узлы и их числа из сводки."
                ),
            },
            {"role": "user", "content": f"Вопрос: {question}\nСводка: {summary}"},
        ],
        json_schema=MAP_SCHEMA,
    )
    try:
        parsed = json.loads(reply.content)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None

    answer, score = parsed.get("answer"), parsed.get("score")
    if not isinstance(answer, str) or not answer.strip() or type(score) is not int or not 0 < score <= 10:
        return None
    return answer.strip()


def _failure_ranking(question: str, kb: KnowledgeBase) -> tuple[str, str, set[str]] | None:
    """Считает отказы по графу для вопросов о лидерах одной линейки"""
    text = question.casefold()
    by_model = "какой модел" in text and "больше всего отказ" in text
    by_component = "узел" in text and "чаще всего отказыва" in text
    lines = [name for kind, name in find_names(question, kb.vocabulary) if kind == "ProductLine"]
    if not (by_model or by_component) or len(lines) != 1:
        return None
    line = lines[0]
    line_node = f"ProductLine:{line}"
    graph = kb.graph
    if line_node not in graph:
        return None

    counts: Counter[str] = Counter()
    sources: set[str] = set()
    for model, _, membership in graph.in_edges(line_node, data=True):
        if membership["type"] != "BELONGS_TO":
            continue
        for _, component, failure in graph.out_edges(model, data=True):
            if failure["type"] == "FAILED":
                counts[graph.nodes[model if by_model else component]["name"]] += failure["weight"]
                sources.update(_document_ids(membership["sources"]))
                sources.update(_document_ids(failure["sources"]))
    if not counts:
        return None

    maximum = max(counts.values())
    leaders = ", ".join(sorted(name for name, count in counts.items() if count == maximum))
    if by_model or sum(count == maximum for count in counts.values()) > 1:
        answer = f"В линейке «{line}» больше всего отказов у {leaders} ({maximum})."
    else:
        answer = f"В линейке «{line}» чаще всего отказывает {leaders}."
    ranking = ", ".join(f"{name} — {count}" for name, count in counts.most_common())
    return answer, ranking, sources


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    Если код из вопроса отсутствует в графе, возвращается отказ без вызова модели.
    """
    code_nodes = [f"{code_type(code)}:{code}" for code in find_codes(question)]
    if any(node not in kb.graph for node in code_nodes):
        return Answer(REFUSAL, refused=True)

    named = find_names(question, kb.vocabulary)
    anchors = {
        node for node in [*code_nodes, *(f"{kind}:{name}" for kind, name in named if kind != "ProductLine")]
        if node in kb.graph
    }
    facts, fact_sources = [], set()
    focus = None
    if anchors and kb.communities and not any(kind == "ProductLine" for kind, _ in named):
        facts, fact_sources = _local_context(kb.graph, sorted(anchors), supplier=_asks_supplier(question))
        if facts:
            focus = max(kb.communities, key=lambda community: len(anchors.intersection(community.nodes)))
    context = "\n".join(facts)

    partials = []
    sources: set[str] = set()
    for community in kb.communities:
        summary = community.summary
        if community is focus:
            summary += f"\nТочные связи из графа по вопросу:\n{context}"
        answer = _community_answer(question, summary, llm)
        if answer is None:
            continue
        partials.append(answer)
        for node in community.nodes:
            sources.update(_document_ids(kb.graph.nodes[node]["sources"]))
    if not partials:
        return Answer(REFUSAL, refused=True)

    evidence = "\n".join(f"{index}. {answer}" for index, answer in enumerate(partials, start=1))
    grounding_instruction = ""
    if facts:
        evidence = f"Точные связи из графа:\n{context}\n\nЧастичные ответы по сводкам:\n{evidence}"
        grounding_instruction = (
            " Для конкретных кодов и названий используй точные связи графа, даже если сводки им противоречат."
            " По REPLACED_BY иди от старой детали к актуальной и бери ее поставщика по SUPPLIED_BY."
            " Остаток, цену и оформление заказа можно подтвердить только через ERP."
        )
    ranking = _failure_ranking(question, kb) or _retired_parts(question, kb)
    if ranking:
        conclusion, counts, _ = ranking
        evidence += f"\nДанные графа для выбранной линейки: {counts}.\nИтог: {conclusion}"
    reply = llm.chat([
        {
            "role": "system",
            "content": (
                "Объедини частичные ответы по вопросу, сохраняя коды и числовые факты. "
                "Сравни приведенные частоты, если вопрос о самом частом отказе. "
                f"Если данных для ответа недостаточно, ответь \"{REFUSAL}\"."
                + grounding_instruction
            ),
        },
        {"role": "user", "content": f"Вопрос: {question}\nЧастичные ответы:\n{evidence}"},
    ])
    if ranking:
        conclusion, _, ranking_sources = ranking
        # Для подсчитанного итога используем источники ребер, участвовавших в сравнении.
        return _answer(conclusion, ranking_sources)
    return _answer(reply.content, sources | fact_sources, facts)


def _retired_parts(question: str, kb: KnowledgeBase) -> tuple[str, str, set[str]] | None:
    """Находит исходные детали линейки, для которых зафиксирована замена."""
    text = question.casefold()
    lines = [name for kind, name in find_names(question, kb.vocabulary) if kind == "ProductLine"]
    if "детал" not in text or "сняты с производства" not in text or len(lines) != 1:
        return None
    line = lines[0]
    line_node = f"ProductLine:{line}"
    graph = kb.graph
    if line_node not in graph:
        return None

    parts: set[str] = set()
    sources: set[str] = set()
    for model, _, membership in graph.in_edges(line_node, data=True):
        if membership["type"] != "BELONGS_TO":
            continue
        for _, part, installation in graph.out_edges(model, data=True):
            if installation["type"] != "HAS_PART":
                continue
            for _, _, replacement in graph.out_edges(part, data=True):
                if replacement["type"] == "REPLACED_BY":
                    parts.add(graph.nodes[part]["name"])
                    for edge in (membership, installation, replacement):
                        sources.update(_document_ids(edge["sources"]))
    if not parts:
        return None
    names = ", ".join(sorted(parts))
    return f"Снятые с производства детали из руководств линейки «{line}» — {names}.", names, sources
