"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from collections.abc import Iterable
from itertools import combinations, pairwise

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

REFUSAL = "В базе знаний не найдено"


def _local_edges(graph: nx.DiGraph, anchors: list[str]) -> list[tuple[str, str]]:
    """Выбирает ребра вокруг опорных узлов или на путях между ними"""
    undirected = graph.to_undirected(as_view=True)
    if len(anchors) == 1:
        nearby = nx.single_source_shortest_path_length(undirected, anchors[0], cutoff=2)
        return sorted(graph.subgraph(nearby).edges)

    edges: set[tuple[str, str]] = set()
    parts: set[str] = set()
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
    return sorted(edges, key=lambda edge: (graph.edges[edge]["type"] != "REPLACED_BY", edge))


def _answer(text: str, sources: set[str], facts: list[str] | None = None) -> Answer:
    """Собирает ответ из текста, источников и фактов или возвращает отказ"""
    text = text.strip()
    if not text or REFUSAL.lower() in text.lower():
        return Answer(REFUSAL, refused=True)
    return Answer(text, sources=sorted(sources), facts=facts or [])


def _document_ids(chunks: Iterable[str]) -> set[str]:
    """Возвращает уникальные имена документов по идентификаторам фрагментов"""
    return {chunk.split("#", 1)[0] for chunk in chunks}


def _local_context(graph: nx.DiGraph, anchors: list[str]) -> tuple[list[str], set[str]]:
    """Формирует пронумерованные факты в пределах лимита и собирает их источники"""
    facts: list[str] = []
    sources: set[str] = set()
    length = 0
    for source, target in _local_edges(graph, anchors):
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
    name_nodes = [f"{kind}:{name}" for kind, name in find_names(question, kb.vocabulary)]
    anchors = [node for node in dict.fromkeys(code_nodes + name_nodes) if node in graph]
    if not anchors:
        return Answer(REFUSAL, refused=True)

    facts, sources = _local_context(graph, anchors)
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


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    partials = []
    sources: set[str] = set()
    for community in kb.communities:
        answer = _community_answer(question, community.summary, llm)
        if answer is None:
            continue
        partials.append(answer)
        for node in community.nodes:
            sources.update(_document_ids(kb.graph.nodes[node]["sources"]))
    if not partials:
        return Answer(REFUSAL, refused=True)

    evidence = "\n".join(f"{index}. {answer}" for index, answer in enumerate(partials, start=1))
    reply = llm.chat([
        {
            "role": "system",
            "content": (
                "Объедини частичные ответы по вопросу, сохраняя коды и числовые факты. "
                "Сравни приведенные частоты, если вопрос о самом частом отказе. "
                f"Если данных для ответа недостаточно, ответь \"{REFUSAL}\"."
            ),
        },
        {"role": "user", "content": f"Вопрос: {question}\nЧастичные ответы:\n{evidence}"},
    ])
    return _answer(reply.content, sources)
