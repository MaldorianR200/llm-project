"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    graph = nx.DiGraph(conflicts=0) # Создаём направленный граф
    edge_sources: dict[tuple[str, str, str], set[str]] = {} # Словарь для накопления источников рёбер

    for result in results:
        for entity in result.entities:
            _add_node_source(graph, entity.node_id, entity.type, entity.name, result.chunk_id)

        for relation in result.relations:
            if not relation.matches_schema():
                continue

            subject = relation.subject
            object_ = relation.object
            _add_node_source(graph, subject.node_id, subject.type, subject.name, result.chunk_id)
            _add_node_source(graph, object_.node_id, object_.type, object_.name, result.chunk_id)

            key = (subject.node_id, relation.type, object_.node_id)
            edge_sources.setdefault(key, set()).add(result.chunk_id)

    for (subject_id, relation_type, object_id), sources in edge_sources.items():
        graph.add_edge(
            subject_id,
            object_id,
            type=relation_type,
            weight=len(sources),
            sources=sorted(sources),
        )

    conflicts: set[frozenset[str]] = set() # Конфликтующие пары
    for subject_id, object_id, data in graph.edges(data=True):
        if subject_id == object_id or data["type"] != "REPLACED_BY":
            continue
        if graph.has_edge(object_id, subject_id) and graph.edges[object_id, subject_id]["type"] == "REPLACED_BY":
            conflicts.add(frozenset((subject_id, object_id)))

    for conflict in conflicts:
        first, second = sorted(conflict)
        first_weight = graph.edges[first, second]["weight"]
        second_weight = graph.edges[second, first]["weight"]

        if first_weight <= second_weight:
            graph.remove_edge(first, second)
        if second_weight <= first_weight:
            graph.remove_edge(second, first)

    graph.graph["conflicts"] = len(conflicts)
    return graph


def _add_node_source(
    graph: nx.DiGraph,
    node_id: str,
    entity_type: str,
    name: str,
    chunk_id: str,
) -> None:
    """Добавляет узел или дополняет его список уникальных источников."""
    if node_id not in graph:
        graph.add_node(node_id, type=entity_type, name=name, sources=[])

    sources = set(graph.nodes[node_id]["sources"])
    sources.add(chunk_id)
    graph.nodes[node_id]["sources"] = sorted(sources)
