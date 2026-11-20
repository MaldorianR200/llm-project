"""Извлечение сущностей и связей из чанка. Требование FR-1."""

import re

from pydantic import BaseModel, ValidationError

from assistant.infra.config import vocabulary
from assistant.infra.grounding import CODE_TYPES, canonical_name, code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import EXTRACTION_SCHEMA, Chunk, Entity, ExtractionResult, Relation

SYSTEM = """Извлеки сущности и связи из фрагмента сервисного документа.
Верни JSON с массивами entities и relations. В entities перечисли все упомянутые сущности,
включая оба конца каждой связи. Типы: Model, ProductLine, Part, Component, ErrorCode, Procedure, Supplier.
Коды бери из текста. Используй канонические названия узлов, линеек и поставщиков из словаря.

Направления связей от subject к object:
- BELONGS_TO: Model → ProductLine (модель входит в линейку).
- HAS_PART: Model → Part (деталь установлена в модели).
- IS_A: Part → Component (деталь относится к виду узла).
- INDICATES: ErrorCode → Component (ошибка указывает на узел).
- FIXED_BY: ErrorCode → Procedure (ошибка устраняется по регламенту).
- REPLACED_BY: Part → Part (старая деталь → новая).
- SUPPLIED_BY: Part → Supplier (деталь → её поставщик).
- FAILED: Model → Component (в отчёте зафиксирован отказ узла на этой модели).
Извлекай только подтверждённые текстом факты. Учитывай заголовок документа и строки таблиц или списков.

Пример для фразы «E-17 означает отказ пускового реле»:
{"entities":[{"type":"ErrorCode","name":"E-17"},{"type":"Component","name":"пусковое реле"}],
 "relations":[{"subject_type":"ErrorCode","subject":"E-17","type":"INDICATES",
               "object_type":"Component","object":"пусковое реле"}]}

Пример для заголовка «Руководство AR-210» и строки
«пусковое реле — деталь P-1042, поставщик ХолодКомплект»:
{"entities":[{"type":"Model","name":"AR-210"},{"type":"Part","name":"P-1042"},
             {"type":"Component","name":"пусковое реле"},
             {"type":"Supplier","name":"ХолодКомплект"}],
 "relations":[{"subject_type":"Model","subject":"AR-210","type":"HAS_PART",
               "object_type":"Part","object":"P-1042"},
              {"subject_type":"Part","subject":"P-1042","type":"IS_A",
               "object_type":"Component","object":"пусковое реле"},
              {"subject_type":"Part","subject":"P-1042","type":"SUPPLIED_BY",
               "object_type":"Supplier","object":"ХолодКомплект"}]}
"""

_MANUAL_SECTION = re.compile(r"^## Состав узлов\s*\n(.*?)(?=^## |\Z)", re.MULTILINE | re.DOTALL)
_PART_CODE = r"P-[0-9]{4}[A-Z]?"
_PART_ROW = re.compile(rf"^- .+? [–—-] деталь ({_PART_CODE})(?=,|$)", re.MULTILINE)
_REPLACEMENT = re.compile(
    rf"\bДеталь ({_PART_CODE})(?: \([^\n)]+\))? снята с производства\.\s+"
    rf"Вместо не[её] устанавливается деталь ({_PART_CODE})\."
)
_PROCEDURE_APPLICATION = re.compile(r"^Регламент (SP-[0-9]{3}) применяется при ошибке (E-[0-9]{2})\.", re.MULTILINE)


def _manual_parts(text: str) -> list[Relation]:
    title = text.partition("\n")[0]
    models = [code for code in find_codes(title) if code_type(code) == "Model"]
    section = _MANUAL_SECTION.search(text)
    if not title.startswith("# Руководство ") or len(models) != 1 or section is None:
        return []
    model = Entity("Model", models[0])
    return [Relation(model, "HAS_PART", Entity("Part", part_id)) for part_id in _PART_ROW.findall(section[1])]


class _EntityFields(BaseModel):
    """Поля сущности в JSON-ответе модели."""

    type: str
    name: str


class _RelationFields(BaseModel):
    """Поля связи и её концов в JSON-ответе модели."""

    subject_type: str
    subject: str
    type: str
    object_type: str
    object: str


class _ExtractionFields(BaseModel):
    """Структура JSON-ответа с сущностями и связями."""

    entities: list[_EntityFields]
    relations: list[_RelationFields]


def _canonical_extracted_name(kind: str, raw: str, names: dict[str, list[str]]) -> str | None:
    """Приводит значение из ответа модели к каноническому виду."""
    if kind in CODE_TYPES:
        # Исправляем повторный дефис перед цифрами, например P--1042.
        raw = re.sub(r"(?<=[A-Za-zА-Яа-я])-{2,}(?=\d)", "-", raw)
    return canonical_name(kind, raw, names)


def _request_extraction(chunk: Chunk, llm: LLMClient, names: dict[str, list[str]]) -> _ExtractionFields | None:
    """Запрашивает и проверяет JSON, повторяя изменённый запрос при ошибке разбора."""
    dictionary = "\n".join(f"{kind}: {', '.join(values)}" for kind, values in names.items())
    messages = [
        {"role": "system", "content": f"{SYSTEM}\nСловарь:\n{dictionary}"},
        {"role": "user", "content": chunk.text},
    ]
    retry_instruction = {
        "role": "user",
        "content": "Повтори извлечение из фрагмента. Верни только полный JSON по заданной схеме без пояснений.",
    }

    for request in (messages, [*messages, retry_instruction]):
        reply = llm.chat(request, json_schema=EXTRACTION_SCHEMA, max_tokens=2048)
        try:
            return _ExtractionFields.model_validate_json(reply.content, strict=True)
        except ValidationError:
            continue
    return None


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    result = ExtractionResult(chunk.chunk_id, chunk.doc_id)
    names = vocabulary()
    extracted = _request_extraction(chunk, llm, names)
    if extracted is None:
        return result

    codes = set(find_codes(chunk.text))
    mentioned_names = {Entity(kind, name) for kind, name in find_names(chunk.text, names)}
    # Ключи словарей убирают дубликаты, сохраняя порядок элементов из ответа.
    entities: dict[Entity, None] = {}

    for item in extracted.entities:
        name = _canonical_extracted_name(item.type, item.name, names)
        if name is None or (item.type in CODE_TYPES and name not in codes):
            result.dropped += 1
        else:
            entities[Entity(item.type, name)] = None

    replacements = set(_REPLACEMENT.findall(chunk.text))
    relations: dict[Relation, None] = {}
    for item in extracted.relations:
        subject = _canonical_extracted_name(item.subject_type, item.subject, names)
        target = _canonical_extracted_name(item.object_type, item.object, names)
        if subject is None or target is None:
            result.dropped += 1
            continue

        start = Entity(item.subject_type, subject)
        end = Entity(item.object_type, target)
        relation = Relation(start, item.type, end)
        # Если модель указала словарное имя только внутри связи, подтверждаем его по тексту чанка.
        if not relation.matches_schema() or any(
            entity not in entities and entity not in mentioned_names for entity in (start, end)
        ):
            result.dropped += 1
            continue
        if relation.type == "REPLACED_BY" and (target, subject) in replacements:
            relation = Relation(end, relation.type, start)
        entities[start] = None
        entities[end] = None
        relations[relation] = None

    explicit_relations = _manual_parts(chunk.text)
    explicit_relations.extend(
        Relation(Entity("ErrorCode", error), "FIXED_BY", Entity("Procedure", procedure))
        for procedure, error in _PROCEDURE_APPLICATION.findall(chunk.text)
    )
    for relation in explicit_relations:
        entities[relation.subject] = None
        entities[relation.object] = None
        relations[relation] = None

    result.entities = list(entities)
    result.relations = list(relations)
    return result
