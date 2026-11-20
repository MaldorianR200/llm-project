"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable
from typing import Any

from assistant.agent.tools import PART_ID_PATTERN, TOOL_SCHEMAS, WAREHOUSES, validate_erp_arguments
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import code_type, find_codes
from assistant.infra.llm import LLMClient, ToolCall, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

REFUSAL = "В базе знаний не найдено"
_TOOL_ARGUMENTS = {
    tool["function"]["name"]: set(tool["function"]["parameters"]["required"])
    for tool in TOOL_SCHEMAS
}
_NUMBER = re.compile(r"(?<![\w-])(\d+)(?![\w-])")
_WITH_UNIT = re.compile(
    r"(?<![\w-])([+−-]?\d+(?:[ \u00a0\u202f]\d{3})*(?:[.,]\d+)?)\s*(руб\w*|₽|шт\w*)(?!\w)", re.I
)
_PRICE_REQUEST = re.compile(
    r"^\s*(?:сколько\s+(?:будет\s+стоить|стоит)|какова\s+цена|подскажи(?:те)?\s+цену)\s+", re.I
)
_PRODUCT_LINE_LOOKUP = re.compile(r"\b(?:в какую|в какой|к какой|какая)\s+линейк", re.I)
_WAREHOUSE_NAMES = {
    "Москва": r"\bмоскв\w*",
    "Санкт-Петербург": r"\bсанкт-петербург\w*",
    "Казань": r"\bказан(?:ь|и|ью)?\b",
    "Новосибирск": r"\bновосибирск\w*",
}
_ORDER_REQUEST = re.compile(r"^\s*(?:пожалуйста,?\s+)?(?:оформи(?:те)?\s+заказ|закажи(?:те)?)\b", re.I)
_ORDER_QTY = re.compile(r"(?<![\w-])([1-9]\d*)\s*(?:штук\w*|шт\.?)(?!\w)", re.I)
_ORDER_CREATED = re.compile(
    r"\bзаказ\b[^.!?\n]*?(?<!не )\b(?:оформлен|создан|размещ[её]н)\b"
    r"|(?<!не )\b(?:оформил[аи]?|создал[аи]?|разместил[аи]?)\s+заказ\b",
    re.I,
)

_SYSTEM = (
    "Ты помощник диспетчера сервисной службы. Отвечай только по результатам инструментов. "
    "Если номера детали нет в вопросе, сначала вызови search_knowledge. "
    "На обзорный вопрос вызывай summarize_topic. Остаток бери из check_stock, цену из get_price. "
    "Не придумывай коды, цены и остатки. Если поиск ничего не нашел, ответь: В базе знаний не найдено. "
    "Заказ предлагай только когда человек сам назвал количество. "
    "Пока заказ не подтвержден, не пиши, что он уже создан. "
    "Коды и числа в ответе оставляй как в наблюдениях."
)


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле: рассуждение, вызов инструмента, наблюдение.

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    messages = [{"role": "system", "content": _SYSTEM}, {"role": "user", "content": question}]
    steps: list[ToolStep] = []
    seen = [question]
    cache: dict[str, dict] = {}

    while True:
        if len(steps) >= max_steps:
            return AgentResult(_interrupted(steps), steps=steps, sources=_sources(steps))

        reply = llm.chat(messages, tools=TOOL_SCHEMAS)
        if not reply.tool_calls:
            needs_local_lookup = (
                bool(steps) and all(step.tool == "summarize_topic" for step in steps)
                and _PRODUCT_LINE_LOOKUP.search(question)
                and any(code_type(code) == "Model" for code in find_codes(question))
            )
            if _is_refusal(reply.content or "") or needs_local_lookup:
                fallback = _lookup_after_refusal(question, tools, steps, max_steps)
                if fallback is not None:
                    return fallback
            return _accept(reply.content, question, steps, messages, llm, confirm=confirm)

        messages.append(reply.as_message())
        for call in reply.tool_calls:
            if len(steps) >= max_steps:
                return AgentResult(_interrupted(steps), steps=steps, sources=_sources(steps))
            outcome = _take_call(call, question, tools, confirm, steps, messages, seen, cache)
            if outcome is not None:
                return outcome


def _take_call(call, question, tools, confirm, steps, messages, seen, cache):
    if "_raw" in call.arguments or call.name not in tools:
        _reject(call, steps, messages, "Не удалось разобрать вызов инструмента.")
        return None
    required = _TOOL_ARGUMENTS.get(call.name)
    if required is not None and call.arguments.keys() != required:
        _reject(call, steps, messages, f"Ожидаемые параметры: {', '.join(sorted(required))}.")
        return None

    if _unknown_part(call.arguments, seen):
        _reject(
            call,
            steps,
            messages,
            "Этого номера детали не было в вопросе и в прошлых результатах. Вызови search_knowledge.",
        )
        return None

    if call.name == "create_service_order":
        error = validate_erp_arguments(**call.arguments)
        if error:
            _reject(call, steps, messages, error)
            return None
        if not _qty_named(question, call.arguments):
            steps.append(ToolStep(call.name, call.arguments, error="В вопросе не названо количество."))
            return AgentResult("Сколько штук заказать?", steps=steps, sources=_sources(steps))

        if not confirm:
            return _pending_order(call.arguments, steps)

    key = call.name + "\n" + json.dumps(call.arguments, ensure_ascii=False, sort_keys=True)
    if key in cache:
        result = cache[key]
    else:
        result = _run_tool(tools[call.name], call.arguments)
        cache[key] = result

    error = result.get("error") if isinstance(result, dict) else "Инструмент вернул не словарь."
    if not isinstance(result, dict):
        result = {"error": error}
    steps.append(ToolStep(call.name, call.arguments, result, error or None))
    if call.name == "create_service_order" and not error and result.get("order_id"):
        return AgentResult(f"Заказ {result['order_id']} создан.", steps=steps, sources=_sources(steps))
    messages.append(tool_message(call, result))
    seen.append(json.dumps(result, ensure_ascii=False))
    return None


def _reject(call: ToolCall, steps: list[ToolStep], messages: list[dict], text: str) -> None:
    steps.append(ToolStep(call.name, call.arguments, error=text))
    messages.append(tool_message(call, {"error": text}))


def _run_tool(fn: Callable[..., dict[str, Any]], arguments: dict[str, Any]) -> dict[str, Any]:
    error = "ERP недоступна"
    for _ in range(3):
        try:
            return fn(**arguments)
        except ErpUnavailable as exc:
            error = str(exc)
    return {"error": error}


def _unknown_part(arguments, seen) -> bool:
    raw = json.dumps(arguments, ensure_ascii=False)
    allowed = set(find_codes("\n".join(seen)))
    return any(code_type(code) == "Part" and code not in allowed for code in find_codes(raw))


def _qty_named(question, arguments) -> bool:
    qty = arguments.get("qty")
    if isinstance(qty, bool) or not isinstance(qty, int) or qty <= 0:
        return False
    return str(qty) in _NUMBER.findall(question)


def _pending_order(arguments: dict, steps: list[ToolStep]) -> AgentResult:
    steps.append(ToolStep("create_service_order", arguments, error="Нужно подтверждение."))
    return AgentResult(
        f"Готов заказать {arguments['part_id']}, подтвердите.",
        steps=steps,
        sources=_sources(steps),
        needs_confirmation=True,
        pending_action={"tool": "create_service_order", "args": arguments},
    )


def _order_from_observations(question: str, steps: list[ToolStep]) -> dict | None:
    """Восстанавливает параметры явного заказа по запросу и проверенному остатку."""
    quantities = _ORDER_QTY.findall(question)
    codes = find_codes(question)
    models = [code for code in codes if code_type(code) == "Model"]
    requested_parts = {code for code in codes if code_type(code) == "Part"}
    warehouses = [
        name for name in WAREHOUSES if re.search(rf"(?<!\w){re.escape(name)}(?!\w)", question, re.I)
    ]
    if len(quantities) != 1 or len(models) != 1 or len(warehouses) != 1 or len(requested_parts) > 1:
        return None

    warehouse = warehouses[0]
    confirmed_parts = {
        step.args.get("part_id")
        for step in steps
        if step.tool == "check_stock"
        and not step.error
        and step.result
        and step.args.get("part_id") == step.result.get("part_id")
        and step.result.get("warehouse") == warehouse
        and step.result.get("discontinued") is False
    }
    if requested_parts:
        confirmed_parts &= requested_parts
    else:
        # Остаток подтверждает наличие детали, а связь с запросом берем из фактов поиска.
        found_parts = {
            code
            for step in steps
            if step.tool == "search_knowledge" and not step.error and step.result and not step.result.get("refused")
            for fact in step.result.get("facts", [])
            for code in find_codes(fact)
        }
        confirmed_parts &= found_parts
    if len(confirmed_parts) != 1:
        return None
    part_id = confirmed_parts.pop()
    if not isinstance(part_id, str) or not re.fullmatch(PART_ID_PATTERN, part_id):
        return None
    return {"part_id": part_id, "warehouse": warehouse, "qty": int(quantities[0]), "model": models[0]}


def _accept(answer, question, steps, messages, llm, *, confirm: bool):
    sources = _sources(steps)
    if not confirm and _ORDER_REQUEST.search(question):
        arguments = _order_from_observations(question, steps)
        if arguments:
            return _pending_order(arguments, steps)
    text = _restore_erp_facts((answer or "").strip(), steps)
    if _ok(text, question, steps):
        return AgentResult(text, steps, sources, refused=_is_refusal(text))

    messages.append({"role": "assistant", "content": text})
    messages.append(
        {
            "role": "user",
            "content": (
                "В ответе есть код, цена или остаток, которых не было в вопросе и в результатах инструментов. "
                "Исправь ответ и оставь только проверенные факты."
            ),
        }
    )
    fixed = _restore_erp_facts((llm.chat(messages, tools=TOOL_SCHEMAS).content or "").strip(), steps)
    if _ok(fixed, question, steps):
        return AgentResult(fixed, steps, sources, refused=_is_refusal(fixed))
    return AgentResult(REFUSAL, steps, sources, refused=True)


def _ok(text, question, steps) -> bool:
    if not text or _ORDER_CREATED.search(text):
        return False
    if _is_refusal(text):
        return True
    blob = "\n".join([question, *[json.dumps(step.result, ensure_ascii=False) for step in steps if step.result]])
    allowed = set(find_codes(blob))
    if any(code not in allowed for code in find_codes(text)):
        return False
    return _erp_amounts_match(text, question, steps)


def _erp_amounts_match(text: str, question: str, steps: list[ToolStep]) -> bool:
    """Сверяет деньги и количество с соответствующими полями ERP."""
    observations: dict[str, dict[tuple[str, str | None], int]] = {"get_price": {}, "check_stock": {}}
    for step in steps:
        if step.error or not step.result or step.tool not in observations:
            continue
        result = step.result
        part = step.args.get("part_id")
        warehouse = step.args.get("warehouse")
        if not part or result.get("part_id") != part or result.get("warehouse") != warehouse:
            continue
        value = result.get("price_rub" if step.tool == "get_price" else "qty")
        if type(value) is int:
            observations[step.tool][part, warehouse] = value

    question_parts = {code for code in find_codes(question) if code_type(code) == "Part"}
    for sentence in re.split(r"[.!?](?=\s)|\n", text):
        parts = {code for code in find_codes(sentence) if code_type(code) == "Part"} or question_parts
        warehouses = {name for name, pattern in _WAREHOUSE_NAMES.items() if re.search(pattern, sentence, re.I)}
        if not warehouses:
            warehouses = {name for name, pattern in _WAREHOUSE_NAMES.items() if re.search(pattern, question, re.I)}
        for raw, unit in _WITH_UNIT.findall(sentence):
            whole, _, fraction = re.sub(r"\s+", "", raw).replace("−", "-").replace(",", ".").partition(".")
            tool = "check_stock" if unit.casefold().startswith("шт") else "get_price"
            candidates = {
                value
                for (part, warehouse), value in observations[tool].items()
                if (not parts or part in parts) and (tool == "get_price" or not warehouses or warehouse in warehouses)
            }
            if candidates != {int(whole)} or fraction.strip("0"):
                return False
    return True


def _is_refusal(text: str) -> bool:
    return text.strip().rstrip(".").casefold() == REFUSAL.casefold()


def _lookup_after_refusal(question, tools, steps, max_steps) -> AgentResult | None:
    """При отказе проверяет граф и при необходимости запрашивает цену по подтвержденному коду."""
    if "search_knowledge" not in tools or _ORDER_REQUEST.search(question):
        return None
    codes = find_codes(question)
    if not codes:
        return None

    price_request = _PRICE_REQUEST.match(question)
    models = [code for code in codes if code_type(code) == "Model"]
    resolve_price = (
        price_request
        and len(models) == 1
        and all(code_type(code) != "Part" for code in codes)
        and "get_price" in tools
        and len(steps) + 2 <= max_steps
        and all(step.tool != "get_price" for step in steps)
    )
    if resolve_price:
        lookup = f"Какой номер детали? {question[price_request.end():]}"
    elif not steps or all(step.tool == "summarize_topic" for step in steps):
        lookup = question
    else:
        return None

    args = {"question": lookup}
    if any(step.tool == "search_knowledge" and step.args == args for step in steps):
        return None
    result = _run_tool(tools["search_knowledge"], args)
    steps.append(ToolStep("search_knowledge", args, result, result.get("error")))
    if result.get("error") or result.get("refused"):
        return None
    sources = _sources(steps)
    if not resolve_price:
        answer = result.get("answer", "")
        return AgentResult(answer, steps, sources) if _ok(answer, question, steps) else None

    raw_facts: object = result.get("facts")
    if not isinstance(raw_facts, list):
        return None
    facts = [fact for fact in raw_facts if isinstance(fact, str)]
    part_id = _installed_part(question, models[0], facts)
    if part_id is None:
        return None
    args = {"part_id": part_id}
    price = _run_tool(tools["get_price"], args)
    steps.append(ToolStep("get_price", args, price, price.get("error")))
    value: object = price.get("price_rub")
    if not price.get("error") and price.get("part_id") == part_id and type(value) is int:
        return AgentResult(f"Цена детали {part_id} — {value} руб.", steps, sources)
    return None


def _installed_part(question: str, model: str, facts: list[str]) -> str | None:
    """Находит единственную деталь, связанную с моделью и названным узлом."""
    facts = [fact.partition(". ")[2] for fact in facts]
    installed = {fact.split(" -HAS_PART-> ")[1] for fact in facts if fact.startswith(f"{model} -HAS_PART-> ")}
    candidates = set()
    for fact in facts:
        part, separator, component = fact.partition(" -IS_A-> ")
        if (
            separator and part in installed and re.fullmatch(PART_ID_PATTERN, part)
            and component and component.casefold() in question.casefold()
        ):
            candidates.add(part)
    return next(iter(candidates)) if len(candidates) == 1 else None


def _restore_erp_facts(text: str, steps: list[ToolStep]) -> str:
    """Подставляет подтвержденные цены и остатки, если модель сообщила об их отсутствии."""
    if REFUSAL.casefold() not in text.casefold():
        return text

    lines: dict[tuple, str] = {}
    for step in steps:
        if step.error or step.result is None:
            continue
        result = step.result
        part_id = step.args.get("part_id")
        if not part_id or result.get("part_id") != part_id:
            continue
        if step.tool == "get_price" and type(result.get("price_rub")) is int:
            lines[step.tool, part_id] = f"Цена детали {part_id} — {result['price_rub']} руб."
        if step.tool == "check_stock" and result.get("warehouse") == step.args.get("warehouse"):
            warehouse, qty = result.get("warehouse"), result.get("qty")
            if isinstance(warehouse, str) and type(qty) is int:
                lines[step.tool, part_id, warehouse] = f"Остаток детали {part_id} (склад {warehouse}) — {qty} шт."
    return "\n".join(lines.values()) if lines else text


def _sources(steps: list[ToolStep]) -> list[str]:
    found: list[str] = []
    for step in steps:
        if not isinstance(step.result, dict):
            continue
        for src in step.result.get("sources") or []:
            if src not in found:
                found.append(src)
    return found


def _interrupted(steps: list[ToolStep]) -> str:
    bits = ["Работа прервана."]
    for step in steps:
        if step.error or not isinstance(step.result, dict):
            continue
        bits.append(json.dumps(step.result, ensure_ascii=False))
    return " ".join(bits)
