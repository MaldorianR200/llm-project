"""Инструменты агента. Требование FR-6."""

import re
from collections.abc import Callable

from assistant.graphrag.search import global_search, local_search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

WAREHOUSES = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]
PART_ID_PATTERN = r"^P-[0-9]{4}[A-Z]?$"
PART_ID = {"type": "string", "pattern": PART_ID_PATTERN, "description": "Номер детали, например P-1042B"}
WAREHOUSE = {"type": "string", "enum": WAREHOUSES}
QUESTION = {"type": "string", " description": "Вопрос для поиска по базе знаний"}


def _tool_schema(name: str, description: str, properties: dict) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
# TODO: заполните список. У параметра warehouse четыре склада задаются перечислением.
TOOL_SCHEMAS: list[dict] = [
    _tool_schema("search_knowledge", "Найти факты о конкретной модели, детали или ошибке", {"question": QUESTION}),
    _tool_schema("summarize_topic", "Ответить на обзорный вопрос по сводкам", {"question": QUESTION}),
    _tool_schema("check_stock", "Проверить остаток детали на складе", {"part_id": PART_ID, "warehouse": WAREHOUSE}),
    _tool_schema("get_price", "Узнать цену детали в рублях", {"part_id": PART_ID}),
    _tool_schema(
        "create_service_order",
        "Создать сервисный заказ на деталь",
        {
            "part_id": PART_ID,
            "warehouse": WAREHOUSE,
            "qty": {"type": "integer", "minimum": 1},
            "model": {"type": "string", "description": "Модель оборудования, например AR-210"},
        },
    ),
]


def _call_erp(method: Callable[..., dict], part_id: str, **arguments) -> dict:
    if not isinstance(part_id, str) or not re.fullmatch(PART_ID_PATTERN, part_id):
        return {"error": "Укажи номер детали в формате P-1042 или P-1042B."}
    if "warehouse" in arguments and arguments["warehouse"] not in WAREHOUSES:
        return {"error": f"Неизвестный склад. Допустимые склады: {', '.join(WAREHOUSES)}."}
    if "qty" in arguments and (type(arguments["qty"]) is not int or arguments["qty"] <= 0):
        return {"error": "Количество должно быть положительным целым числом."}
    if "model" in arguments and (not isinstance(arguments["model"], str) or not arguments["model"].strip()):
        return {"error": "Укажи модель оборудования."}

    try:
        return method(part_id=part_id, **arguments)
    except ErpNotFound as exc:
        return {"error": f"Данные в ERP не найдены: {exc}"}
    except ErpRejected as exc:
        return {"error": f"ERP отклонила запрос: {exc}"}


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    # TODO: пять функций поверх local_search, global_search и клиента ERP
    def search_knowledge(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Укажи вопрос для поиска."}
        answer = local_search(question, kb, llm)
        return {"answer": answer.text, "facts": answer.facts, "sources": answer.sources, "refused": answer.refused}

    def summarize_topic(question: str) -> dict:
        if not isinstance(question, str) or not question.strip():
            return {"error": "Укажи вопрос для поиска."}
        answer = global_search(question, kb, llm)
        return {"answer": answer.text, "sources": answer.sources, "refused": answer.refused}

    def check_stock(part_id: str, warehouse: str) -> dict:
        return _call_erp(erp.get_stock, part_id, warehouse=warehouse)

    def get_price(part_id: str) -> dict:
        return _call_erp(erp.get_price, part_id)

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> dict:
        return _call_erp(erp.create_order, part_id, warehouse=warehouse, qty=qty, model=model)

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }
