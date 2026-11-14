"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter

import networkx as nx

from assistant.infra.llm import SEED, LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph))

    for subject_id, object_id, data in sorted(graph.edges(data=True)):
        weight = data.get("weight", 1)
        if undirected.has_edge(subject_id, object_id):
            undirected.edges[subject_id, object_id]["weight"] += weight
        else:
            undirected.add_edge(subject_id, object_id, weight=weight)

    if not undirected:
        return []

    partition = nx.community.louvain_communities(undirected, weight="weight", seed=SEED)
    ordered_groups = sorted((sorted(nodes) for nodes in partition), key=lambda nodes: nodes[0])
    return [Community(community_id, nodes) for community_id, nodes in enumerate(ordered_groups)]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    nodes = [node_id for node_id in community.nodes if node_id in graph]
    node_set = set(nodes)
    names = nx.get_node_attributes(graph, "name")
    entity_counts = Counter(graph.nodes[node_id].get("type", "Unknown") for node_id in nodes)
    relation_counts: Counter[str] = Counter()
    failure_counts: Counter[tuple[str, str]] = Counter()
    facts: list[str] = []

    for subject_id, object_id, data in sorted(graph.edges(data=True)):
        relation_type = data.get("type", "UNKNOWN")
        weight = int(data.get("weight", 1))
        subject_name = names.get(subject_id, subject_id)
        object_name = names.get(object_id, object_id)

        if relation_type == "FAILED" and object_id in node_set:
            # Привязываем отказ к линейке модели кодом, чтобы LLM не смешивала линейки при сравнении.
            product_lines = [
                names.get(target_id, target_id)
                for target_id, edge in graph[subject_id].items()
                if edge.get("type") == "BELONGS_TO"
            ]
            for product_line in product_lines or ["линейка неизвестна"]:
                failure_counts[product_line, object_name] += weight
            relation_counts[relation_type] += weight
            facts.append(f"{subject_name} -FAILED-> {object_name} (вес: {weight})")
        elif relation_type != "FAILED" and subject_id in node_set and object_id in node_set:
            relation_counts[relation_type] += weight
            facts.append(f"{subject_name} -{relation_type}-> {object_name} (вес: {weight})")

    entity_stats = _format_counts(entity_counts)
    relation_stats = _format_counts(relation_counts)
    failure_stats = _format_failures(failure_counts)
    member_names = ", ".join(names.get(node_id, node_id) for node_id in nodes) or "нет"
    fact_list = "\n".join(f"{number}. {fact}" for number, fact in enumerate(facts, start=1))
    if not fact_list:
        fact_list = "Внутренних связей нет."

    messages = [
        {
            "role": "system",
            "content": (
                "Ты составляешь краткие сводки сообществ графа знаний сервисной службы. "
                "Используй только переданные факты и уже рассчитанные числа. Не пересчитывай и не придумывай числа. "
                "Сохраняй названия линеек и не смешивай показатели разных линеек. "
                "Ответь по-русски, не более чем 150 словами."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Сообщество {community.community_id}.\n"
                f"Узлов: {len(nodes)}.\n"
                f"Узлы: {member_names}.\n"
                f"Сущности по типам: {entity_stats}.\n"
                f"Суммарный вес связей по типам: {relation_stats}.\n"
                f"Итоги по отказам для каждой линейки и узла: {failure_stats}.\n"
                "Опиши, какие сущности входят в сообщество, как они связаны и что встречается чаще всего.\n"
                f"Факты:\n{fact_list}"
            ),
        },
    ]
    response = llm.chat(messages, max_tokens=300).content
    words = response.split()
    return " ".join(words[:150]) if words else "Сводка не сформирована."


def _format_counts(counts: Counter[str]) -> str:
    """Форматирует рассчитанные кодом показатели в стабильном порядке."""
    if not counts:
        return "нет"
    return ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))


def _format_failures(counts: Counter[tuple[str, str]]) -> str:
    """Форматирует взвешенные частоты отказов отдельно по линейкам."""
    if not counts:
        return "нет"
    return ", ".join(
        f"{product_line} — {component}: {count}"
        for (product_line, component), count in sorted(counts.items())
    )
