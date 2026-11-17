"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""

import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import code_type, find_codes
from assistant.infra.llm import LLMClient, ToolCall, tool_message
from assistant.infra.schemas import AgentResult, ToolStep

REFUSAL = "В базе знаний не найдено"
_NUMBER = re.compile(r"(?<![\w-])(\d+)(?![\w-])")
_WITH_UNIT = re.compile(r"(?<![\w-])(\d+)\s*(?:руб|шт)")

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
            return _accept(reply.content, question, steps, messages, llm)

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

    if _unknown_part(call.arguments, seen):
        _reject(
            call,
            steps,
            messages,
            "Этого номера детали не было в вопросе и в прошлых результатах. Вызови search_knowledge.",
        )
        return None

    if call.name == "create_service_order" and not _qty_named(question, call.arguments):
        steps.append(ToolStep(call.name, call.arguments, error="В вопросе не названо количество."))
        return AgentResult("Сколько штук заказать?", steps=steps, sources=_sources(steps))

    if call.name == "create_service_order" and not confirm:
        steps.append(ToolStep(call.name, call.arguments, error="Нужно подтверждение."))
        part = call.arguments.get("part_id", "")
        return AgentResult(
            f"Готов заказать {part}, подтвердите.",
            steps=steps,
            sources=_sources(steps),
            needs_confirmation=True,
            pending_action={"tool": call.name, "args": call.arguments},
        )

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
    messages.append(tool_message(call, result))
    seen.append(json.dumps(result, ensure_ascii=False))
    return None


def _reject(call: ToolCall, steps: list[ToolStep], messages: list[dict], text: str) -> None:
    steps.append(ToolStep(call.name, call.arguments, error=text))
    messages.append(tool_message(call, {"error": text}))


def _run_tool(fn, arguments):
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


def _accept(answer, question, steps, messages, llm):
    sources = _sources(steps)
    text = (answer or "").strip()
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
    fixed = (llm.chat(messages, tools=TOOL_SCHEMAS).content or "").strip()
    if _ok(fixed, question, steps):
        return AgentResult(fixed, steps, sources, refused=_is_refusal(fixed))
    return AgentResult(REFUSAL, steps, sources, refused=True)


def _ok(text, question, steps) -> bool:
    if not text:
        return False
    if _is_refusal(text):
        return True
    blob = "\n".join([question, *[json.dumps(step.result, ensure_ascii=False) for step in steps if step.result]])
    allowed = set(find_codes(blob))
    if any(code not in allowed for code in find_codes(text)):
        return False
    numbers = set(re.findall(r"\d+", blob))
    return all(match.group(1) in numbers for match in _WITH_UNIT.finditer(text))


def _is_refusal(text: str) -> bool:
    return REFUSAL.lower() in text.lower()


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
