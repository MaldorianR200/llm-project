"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    # TODO: снять направление рёбер, разбить с учётом weight, вернуть Community с номерами по порядку
    undirected = nx.Graph()
    undirected.add_nodes_from(sorted(graph))
    for source, target, attrs in sorted(graph.edges(data=True)):
        previous = undirected.get_edge_data(source, target, default={})
        undirected.add_edge(source, target, weight=previous.get("weight", 0) + attrs.get("weight", 1))
    groups = nx.community.louvain_communities(undirected, weight="weight", seed=7) if undirected else []
    return [Community(index, nodes) for index, nodes in enumerate(sorted(map(sorted, groups)))]


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    # TODO: собрать факты и подсчёты сообщества, попросить модель написать сводку
    members = set(community.nodes)
    names = nx.get_node_attributes(graph, "name")
    failures: Counter[tuple[str, str]] = Counter()
    facts = []
    for source, target, attrs in sorted(graph.edges(data=True)):
        kind = attrs["type"]
        weight = attrs["weight"]
        if kind == "FAILED" and target in members:
            # Каждый узел оборудования относится к одному сообществу, поэтому отказы не дублируются
            lines = [
                names[node] for node, edge in graph[source].items() if edge["type"] == "BELONGS_TO"
            ]
            for line in lines or ["линейка неизвестна"]:
                failures[line, names[target]] += weight
            facts.append(f"{names[source]} -FAILED-> {names[target]}: {weight}")
        # Для остальных связей оба узла должны входить в сообщество
        elif kind != "FAILED" and source in members and target in members:
            facts.append(f"{names[source]} -{kind}-> {names[target]}")

    counts = "\n".join(f"{line}, {component}: {count}" for (line, component), count in sorted(failures.items()))
    context = "\n".join(facts)
    members_text = ", ".join(names[node] for node in community.nodes)
    reply = llm.chat([
        {
            "role": "system",
            "content": (
                "Составь сводку сообщества до 150 слов. Укажи основные сущности и связи. "
                "Если даны итоги по отказам, сохрани линейки, названия узлов и числа без изменений."
            ),
        },
        {
            "role": "user",
            "content": f"Итоги по отказам:\n{counts}\nУзлы: {members_text}\nФакты:\n{context}",
        },
    ])
    return " ".join(reply.content.split()[:150])
