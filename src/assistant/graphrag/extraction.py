"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import json

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, find_codes
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import EXTRACTION_SCHEMA, Chunk, Entity, ExtractionResult, Relation

_SYSTEM_PROMPT = """Ты извлекаешь граф знаний из русскоязычного документа сервисной службы.
Верни только сущности и связи по переданной JSON-схеме.

Допустимые направления связей:
- Model -BELONGS_TO-> ProductLine
- Model -HAS_PART-> Part
- Model -FAILED-> Component
- Part -IS_A-> Component
- Part -REPLACED_BY-> Part (старая деталь -> новая деталь)
- Part -SUPPLIED_BY-> Supplier
- ErrorCode -INDICATES-> Component
- ErrorCode -FIXED_BY-> Procedure

Не добавляй факты, которых нет в тексте. Пример: фраза «P-1042 снята с производства,
вместо неё устанавливается P-1042B» даёт сущности Part P-1042 и Part P-1042B,
а также связь P-1042 -REPLACED_BY-> P-1042B.
"""


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": f"Извлеки граф знаний из этого чанка:\n\n{chunk.text}"},
    ]
    raw: dict | None = None
    for _ in range(2):
        try:
            candidate = json.loads(llm.chat(messages, json_schema=EXTRACTION_SCHEMA).content)
            if not _valid_payload(candidate):
                raise ValueError("ответ не соответствует ожидаемой структуре")
            raw = candidate
            break
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    if raw is None:
        return ExtractionResult(chunk.chunk_id, chunk.doc_id)

    vocab = vocabulary()
    confirmed_codes = set(find_codes(chunk.text))
    entities: dict[str, Entity] = {}
    dropped = 0

    def checked_entity(entity_type: str, name: str) -> Entity | None:
        canonical = canonical_name(entity_type, name, vocab)
        if canonical is None or (entity_type in CODE_TYPES and canonical not in confirmed_codes):
            return None
        return Entity(entity_type, canonical)

    for item in raw["entities"]:
        entity = checked_entity(item["type"], item["name"])
        if entity is None:
            dropped += 1
        else:
            entities[entity.node_id] = entity

    relations: list[Relation] = []
    seen_relations: set[tuple[str, str, str]] = set()
    for item in raw["relations"]:
        subject = checked_entity(item["subject_type"], item["subject"])
        object_ = checked_entity(item["object_type"], item["object"])
        relation = Relation(subject, item["type"], object_) if subject and object_ else None
        if relation is None or not relation.matches_schema():
            dropped += 1
            continue
        key = (relation.subject.node_id, relation.type, relation.object.node_id)
        if key not in seen_relations:
            relations.append(relation)
            seen_relations.add(key)
        entities[subject.node_id] = subject
        entities[object_.node_id] = object_

    return ExtractionResult(chunk.chunk_id, chunk.doc_id, list(entities.values()), relations, dropped)


def _valid_payload(value: object) -> bool:
    """Минимальная проверка структуры до обращения к полям ответа модели."""
    if not isinstance(value, dict) or not isinstance(value.get("entities"), list) or not isinstance(
        value.get("relations"), list
    ):
        return False
    entity_keys = {"type", "name"}
    relation_keys = {"subject_type", "subject", "type", "object_type", "object"}
    return all(
        isinstance(item, dict)
        and entity_keys <= item.keys()
        and all(isinstance(item[key], str) for key in entity_keys)
        for item in value["entities"]
    ) and all(
        isinstance(item, dict)
        and relation_keys <= item.keys()
        and all(isinstance(item[key], str) for key in relation_keys)
        for item in value["relations"]
    )
