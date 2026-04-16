#!/usr/bin/env python3
"""
ToolCall-15 Benchmark - Python Port

A faithful port of the ToolCall-15 benchmark from TypeScript to Python.
Evaluates LLM tool-calling ability across 15 scenarios in 5 categories.

Dependencies: requests (pip install requests)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

try:
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
except ImportError:
    print(
        "Error: 'requests' package is required. Install it with:\n"
        "  pip install requests",
        file=sys.stderr,
    )
    sys.exit(1)


def _build_http_session() -> requests.Session:
    """Build a requests Session with retry and connection pooling for SSL reliability."""
    session = requests.Session()
    session.trust_env = False  # Ignore all proxy env vars (http_proxy, https_proxy, all_proxy)
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=[502, 503, 504],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=10)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


_HTTP_SESSION = _build_http_session()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to the tools provided.\n"
    "\n"
    "Rules:\n"
    "- Use a tool ONLY when it is necessary to fulfill the user's request.\n"
    "- If you can answer directly from your own knowledge, do so without calling a tool.\n"
    "- If a tool call fails, explain the failure and suggest an alternative approach.\n"
    "- Never invent information that a tool should provide."
)

BENCHMARK_REFERENCE_DATE = "2026-03-20"
BENCHMARK_REFERENCE_DAY = "Friday"

DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS = 30
MAX_TURNS = 8
MAX_PROVIDER_ERROR_ATTEMPTS = 3
PROVIDER_ERROR_RETRY_PATTERN = re.compile(r"provider returned error", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Universal Tools
# ---------------------------------------------------------------------------

UNIVERSAL_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for current information",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "max_results": {"type": "integer", "default": 5},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a specific location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "units": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "default": "celsius",
                    },
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "Perform mathematical calculations",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string"},
                },
                "required": ["expression"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_email",
            "description": "Send an email to a recipient",
            "parameters": {
                "type": "object",
                "properties": {
                    "to": {"type": "string"},
                    "subject": {"type": "string"},
                    "body": {"type": "string"},
                    "attachments": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                },
                "required": ["to", "subject", "body"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_files",
            "description": "Search for files by name or content",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "file_type": {
                        "type": "string",
                        "enum": ["pdf", "docx", "xlsx", "any"],
                        "default": "any",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a specific file",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_id": {"type": "string"},
                },
                "required": ["file_id"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_calendar_event",
            "description": "Create a new calendar event",
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {"type": "string"},
                    "date": {"type": "string", "format": "YYYY-MM-DD"},
                    "time": {"type": "string", "format": "HH:MM"},
                    "duration_minutes": {"type": "integer", "default": 60},
                    "attendees": {
                        "type": "array",
                        "items": {"type": "string"},
                        "default": [],
                    },
                },
                "required": ["title", "date", "time"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_contacts",
            "description": "Look up contacts by name or group",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                },
                "required": ["query"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "translate_text",
            "description": "Translate text from one language to another",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "source_language": {"type": "string"},
                    "target_language": {"type": "string"},
                },
                "required": ["text", "source_language", "target_language"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_stock_price",
            "description": "Get the current stock price for a ticker symbol",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticker": {"type": "string"},
                },
                "required": ["ticker"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "set_reminder",
            "description": "Set a reminder for a future time",
            "parameters": {
                "type": "object",
                "properties": {
                    "message": {"type": "string"},
                    "datetime": {"type": "string", "format": "ISO 8601"},
                },
                "required": ["message", "datetime"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_code",
            "description": "Execute a code snippet and return the output",
            "parameters": {
                "type": "object",
                "properties": {
                    "language": {
                        "type": "string",
                        "enum": ["python", "javascript"],
                    },
                    "code": {"type": "string"},
                },
                "required": ["language", "code"],
                "additionalProperties": False,
            },
        },
    },
]

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class ModelConfig:
    id: str
    label: str
    model: str
    base_url: str
    api_key: str | None = None


@dataclass
class ToolCallRecord:
    id: str
    name: str
    raw_arguments: str
    arguments: dict[str, Any]
    turn: int


@dataclass
class ToolResultRecord:
    call_id: str
    name: str
    result: Any


@dataclass
class ScenarioState:
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    tool_results: list[ToolResultRecord] = field(default_factory=list)
    assistant_messages: list[str] = field(default_factory=list)
    final_answer: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioEvaluation:
    status: Literal["pass", "partial", "fail"]
    points: int
    summary: str
    note: str | None = None


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------


def parse_math_expression(expression: str) -> float | None:
    """Sanitize and evaluate a math expression using restricted eval."""
    sanitized = expression.replace(",", "").strip()
    if not re.match(r"^[\d\s()+\-*/.%]+$", sanitized):
        return None
    try:
        result = eval(sanitized, {"__builtins__": {}}, {})  # noqa: S307
        if isinstance(result, (int, float)) and not (
            isinstance(result, float) and (result != result or result == float("inf") or result == float("-inf"))
        ):
            return result
        return None
    except Exception:
        return None


def as_string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def as_string_array(value: Any) -> list[str]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def normalize(value: str) -> str:
    return value.strip().lower()


def includes_text(value: Any, expected: str) -> bool:
    return as_string(value).lower().find(expected.lower()) != -1


def mentions_all(text: str, values: list[str]) -> bool:
    normalized_text = normalize(text)
    return all(normalize(v) in normalized_text for v in values)


def answer_contains_number(answer: str, value: str) -> bool:
    collapsed = answer.replace(",", "").lower()
    return value.replace(",", "").lower() in collapsed


def full_assistant_transcript(state: ScenarioState) -> str:
    return "\n".join(state.assistant_messages)


def tool_calls_by_name(state: ScenarioState, name: str) -> list[ToolCallRecord]:
    return [call for call in state.tool_calls if call.name == name]


def has_tool_call(
    state: ScenarioState,
    name: str,
    predicate: Callable[[ToolCallRecord], bool] | None = None,
) -> bool:
    calls = tool_calls_by_name(state, name)
    return any(predicate(call) if predicate else True for call in calls)


def first_call(state: ScenarioState, name: str) -> ToolCallRecord | None:
    calls = tool_calls_by_name(state, name)
    return calls[0] if calls else None


def is_only_tool(state: ScenarioState, name: str) -> bool:
    return len(state.tool_calls) > 0 and all(call.name == name for call in state.tool_calls)


def contains_refusal(answer: str) -> bool:
    lowered = answer.lower()
    return any(
        phrase in lowered
        for phrase in ("cannot", "can't", "do not have", "don't have", "not able")
    )


def asks_for_clarification(answer: str) -> bool:
    lowered = answer.lower()
    return any(word in lowered for word in ("which", "clarify", "could you"))


def has_current_tool_misuse(state: ScenarioState, allowed_tools: list[str]) -> bool:
    return any(call.name not in allowed_tools for call in state.tool_calls)


def generic_tool_fallback(call: ToolCallRecord) -> Any:
    if call.name == "calculator":
        result = parse_math_expression(as_string(call.arguments.get("expression", "")))
        return {"error": "Invalid expression."} if result is None else {"result": result}
    if call.name == "web_search":
        return {"results": [{"snippet": f"Search results for {as_string(call.arguments.get('query', ''))}"}]}
    if call.name == "run_code":
        return {"error": "Code execution is disabled in benchmark mocks."}
    return {"error": f"Tool {call.name} is not relevant for this scenario."}


# ---------------------------------------------------------------------------
# Scenario Definitions
# ---------------------------------------------------------------------------


def _tc01_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return {
            "location": "Berlin",
            "temperature": 8,
            "units": "celsius",
            "condition": "Overcast",
            "humidity": 72,
        }
    if call.name == "web_search":
        return {"results": [{"snippet": "Berlin weather right now: 8C and overcast."}]}
    return generic_tool_fallback(call)


def _tc01_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    used_weather = has_tool_call(
        state, "get_weather", lambda c: includes_text(c.arguments.get("location"), "berlin")
    )
    used_web = has_tool_call(state, "web_search")

    if used_weather and not used_web and len(state.tool_calls) == 1:
        return ScenarioEvaluation("pass", 2, "Used get_weather with Berlin only.")
    if not used_weather and used_web and is_only_tool(state, "web_search"):
        return ScenarioEvaluation(
            "partial", 1, "Answered functionally via web_search, but missed the specialist tool."
        )
    return ScenarioEvaluation("fail", 0, "Did not cleanly route the request to get_weather.")


def _tc02_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_stock_price":
        return {
            "ticker": "AAPL",
            "price": 187.42,
            "currency": "USD",
            "change": "+1.23",
            "change_percent": "+0.66%",
        }
    if call.name == "web_search":
        return {"results": [{"snippet": "AAPL is trading around $187.42."}]}
    return generic_tool_fallback(call)


def _tc02_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    stock_call = has_tool_call(
        state,
        "get_stock_price",
        lambda c: normalize(as_string(c.arguments.get("ticker", ""))) == "aapl",
    )
    web_call = has_tool_call(state, "web_search")

    if stock_call and not web_call and len(state.tool_calls) == 1:
        return ScenarioEvaluation("pass", 2, "Used only get_stock_price for AAPL.")
    if stock_call and web_call:
        return ScenarioEvaluation(
            "partial", 1, "Called the right tool, but added unnecessary web_search."
        )
    return ScenarioEvaluation("fail", 0, "Did not isolate the request to get_stock_price.")


def _tc03_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_contacts":
        return {"results": [{"name": "Sarah Chen", "email": "sarah.chen@company.com"}]}
    if call.name == "send_email":
        return {"status": "sent", "message_id": "msg_8821"}
    return generic_tool_fallback(call)


def _tc03_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    contact_call = first_call(state, "get_contacts")
    email_call = first_call(state, "send_email")

    if (
        contact_call
        and email_call
        and contact_call.turn < email_call.turn
        and includes_text(contact_call.arguments.get("query"), "sarah")
        and normalize(as_string(email_call.arguments.get("to", ""))) == "sarah.chen@company.com"
    ):
        return ScenarioEvaluation("pass", 2, "Looked up Sarah before sending the email.")

    if (
        not contact_call
        and not email_call
        and re.search(r"email", state.final_answer, re.IGNORECASE)
        and "?" in state.final_answer
    ):
        return ScenarioEvaluation(
            "partial", 1, "Asked for Sarah's email instead of inferring the tool chain."
        )

    return ScenarioEvaluation(
        "fail", 0, "Did not complete the contact lookup to email chain correctly."
    )


def _tc04_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        units = normalize(as_string(call.arguments.get("units", ""))) or "celsius"
        if units == "fahrenheit":
            return {"location": "Tokyo", "temperature": 64, "units": "fahrenheit", "condition": "Clear"}
        return {"location": "Tokyo", "temperature": 18, "units": "celsius", "condition": "Clear"}
    return generic_tool_fallback(call)


def _tc04_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    weather_call = first_call(state, "get_weather")

    if (
        weather_call
        and includes_text(weather_call.arguments.get("location"), "tokyo")
        and normalize(as_string(weather_call.arguments.get("units", ""))) == "fahrenheit"
    ):
        return ScenarioEvaluation("pass", 2, "Requested Tokyo weather in Fahrenheit explicitly.")

    if (
        weather_call
        and includes_text(weather_call.arguments.get("location"), "tokyo")
        and not as_string(weather_call.arguments.get("units", ""))
        and (
            "fahrenheit" in state.final_answer.lower()
            or answer_contains_number(state.final_answer, "64")
        )
    ):
        return ScenarioEvaluation(
            "partial", 1, "Omitted the units parameter and converted manually."
        )

    return ScenarioEvaluation("fail", 0, "Did not preserve the Fahrenheit instruction.")


def _tc05_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_contacts":
        return {
            "results": [
                {"name": "Alex Stone", "email": "alex.stone@company.com"},
                {"name": "Jamie Liu", "email": "jamie.liu@company.com"},
            ]
        }
    if call.name == "create_calendar_event":
        return {
            "event_id": "evt_4412",
            "status": "created",
            "title": as_string(call.arguments.get("title", "")) or "Team Standup",
            "date": as_string(call.arguments.get("date", "")),
        }
    return generic_tool_fallback(call)


def _tc05_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    event_call = first_call(state, "create_calendar_event")

    if not event_call:
        return ScenarioEvaluation("fail", 0, "Did not create the calendar event.")

    attendees = as_string_array(event_call.arguments.get("attendees", []))
    has_duration = event_call.arguments.get("duration_minutes") == 30 or (
        isinstance(event_call.arguments.get("duration_minutes"), (int, float))
        and int(event_call.arguments.get("duration_minutes")) == 30
    )
    has_attendees = any(includes_text(v, "alex") for v in attendees) and any(
        includes_text(v, "jamie") for v in attendees
    )
    correct_date = as_string(event_call.arguments.get("date", "")) == "2026-03-23"
    correct_time = as_string(event_call.arguments.get("time", "")) == "09:30"

    if correct_date and correct_time and has_duration and has_attendees:
        return ScenarioEvaluation(
            "pass", 2, "Parsed next Monday and included the requested meeting details."
        )
    if correct_date and correct_time:
        return ScenarioEvaluation(
            "partial", 1, "Got the date and time right, but missed some optional structure."
        )
    return ScenarioEvaluation("fail", 0, "Relative date or time parsing was incorrect.")


def _tc06_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "translate_text":
        target = normalize(as_string(call.arguments.get("target_language", "")))
        if target == "spanish":
            return {"translated": "\u00bfD\u00f3nde est\u00e1 el hospital m\u00e1s cercano?"}
        if target == "japanese":
            return {"translated": "\u6700\u5bc4\u308a\u306e\u75c5\u9662\u306f\u3069\u3053\u3067\u3059\u304b\uff1f"}
        return {"error": f"Unsupported target language {target}."}
    return generic_tool_fallback(call)


def _tc06_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    translate_calls = tool_calls_by_name(state, "translate_text")
    has_spanish = any(
        normalize(as_string(c.arguments.get("source_language", ""))) == "english"
        and normalize(as_string(c.arguments.get("target_language", ""))) == "spanish"
        and as_string(c.arguments.get("text", "")) == "Where is the nearest hospital?"
        for c in translate_calls
    )
    has_japanese = any(
        normalize(as_string(c.arguments.get("source_language", ""))) == "english"
        and normalize(as_string(c.arguments.get("target_language", ""))) == "japanese"
        and as_string(c.arguments.get("text", "")) == "Where is the nearest hospital?"
        for c in translate_calls
    )
    invalid_bundled_target = any(
        re.search(
            r"spanish.*japanese|japanese.*spanish",
            as_string(c.arguments.get("target_language", "")),
            re.IGNORECASE,
        )
        for c in translate_calls
    )

    if len(translate_calls) >= 2 and has_spanish and has_japanese and not invalid_bundled_target:
        return ScenarioEvaluation(
            "pass", 2, "Issued separate translate_text calls for both languages."
        )
    return ScenarioEvaluation(
        "fail", 0, "Did not split the translation request into two valid tool calls."
    )


def _tc07_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "search_files":
        return {"results": [{"file_id": "file_091", "name": "Q3_Budget_Report_2025.xlsx"}]}
    if call.name == "read_file":
        return {
            "content": "Department budgets: Engineering $2.1M, Marketing $800K, Sales $1.5M. Total: $4.4M"
        }
    if call.name == "get_contacts":
        return {
            "results": [
                {"name": "Jordan Park", "email": "jordan.park@company.com", "role": "manager"}
            ]
        }
    if call.name == "send_email":
        return {"status": "sent"}
    return generic_tool_fallback(call)


def _tc07_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    completed_steps = 0

    if has_tool_call(
        state,
        "search_files",
        lambda c: includes_text(c.arguments.get("query"), "q3 budget report"),
    ):
        completed_steps += 1

    if has_tool_call(
        state,
        "read_file",
        lambda c: normalize(as_string(c.arguments.get("file_id", ""))) == "file_091",
    ):
        completed_steps += 1

    if has_tool_call(
        state,
        "get_contacts",
        lambda c: includes_text(c.arguments.get("query"), "manager"),
    ):
        completed_steps += 1

    if has_tool_call(
        state,
        "send_email",
        lambda c: (
            normalize(as_string(c.arguments.get("to", ""))) == "jordan.park@company.com"
            and (
                includes_text(c.arguments.get("body"), "4.4m")
                or includes_text(c.arguments.get("body"), "$4.4m")
            )
        ),
    ):
        completed_steps += 1

    if completed_steps == 4:
        return ScenarioEvaluation(
            "pass", 2, "Completed the full four-step chain with the right data."
        )
    if completed_steps >= 3:
        return ScenarioEvaluation(
            "partial", 1, "Completed most of the chain, but missed one dependent step."
        )
    return ScenarioEvaluation(
        "fail", 0, "Did not carry the file and contact data across the chain correctly."
    )


def _tc08_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return {"location": "Paris", "temperature": 11, "condition": "Light rain", "humidity": 89}
    if call.name == "set_reminder":
        return {"reminder_id": "rem_553", "status": "set"}
    return generic_tool_fallback(call)


def _tc08_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    weather_call = first_call(state, "get_weather")
    reminder_call = first_call(state, "set_reminder")

    if (
        weather_call
        and reminder_call
        and weather_call.turn < reminder_call.turn
        and includes_text(reminder_call.arguments.get("message"), "umbrella")
        and as_string(reminder_call.arguments.get("datetime", "")).startswith("2026-03-21T08:00:00")
    ):
        return ScenarioEvaluation(
            "pass", 2, "Checked the weather first, then set the rainy-day reminder."
        )

    if weather_call and not reminder_call and asks_for_clarification(state.final_answer):
        return ScenarioEvaluation(
            "partial",
            1,
            "Read the weather correctly, but stopped short of setting the reminder.",
        )

    return ScenarioEvaluation("fail", 0, "Did not respect the weather-first conditional flow.")


def _tc09_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_weather":
        return {"location": "London", "temperature": 12, "condition": "Cloudy"}
    if call.name == "get_stock_price":
        return {"ticker": "MSFT", "price": 412.78, "currency": "USD"}
    if call.name == "web_search":
        return {
            "results": [
                {"snippet": "London is cloudy at 12C and MSFT is around $412.78."}
            ]
        }
    return generic_tool_fallback(call)


def _tc09_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    weather_call = has_tool_call(
        state, "get_weather", lambda c: includes_text(c.arguments.get("location"), "london")
    )
    stock_call = has_tool_call(
        state,
        "get_stock_price",
        lambda c: normalize(as_string(c.arguments.get("ticker", ""))) == "msft",
    )
    first_assistant_batch = [c for c in state.tool_calls if c.turn == 1]
    parallel = any(c.name == "get_weather" for c in first_assistant_batch) and any(
        c.name == "get_stock_price" for c in first_assistant_batch
    )

    if weather_call and stock_call:
        return ScenarioEvaluation(
            "pass",
            2,
            "Handled both independent tasks.",
            note="Both tools were called in the same assistant turn." if parallel else None,
        )

    if has_tool_call(state, "web_search"):
        return ScenarioEvaluation(
            "partial", 1, "Covered the request, but fell back to web_search."
        )

    return ScenarioEvaluation("fail", 0, "Missed one side of the two-part request.")


def _tc10_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    return generic_tool_fallback(call)


def _tc10_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    if len(state.tool_calls) == 0 and answer_contains_number(state.final_answer, "1945"):
        return ScenarioEvaluation("pass", 2, "Answered directly without tool use.")
    return ScenarioEvaluation("fail", 0, "Used tools or missed the basic fact.")


def _tc11_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    return generic_tool_fallback(call)


def _tc11_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    correct_answer = answer_contains_number(state.final_answer, "30")

    if len(state.tool_calls) == 0 and correct_answer:
        return ScenarioEvaluation("pass", 2, "Did the math directly.")
    if (
        has_tool_call(state, "calculator")
        and correct_answer
        and not has_tool_call(state, "web_search")
    ):
        return ScenarioEvaluation("partial", 1, "Used calculator correctly, but unnecessarily.")
    return ScenarioEvaluation("fail", 0, "Failed the easy-arithmetic restraint test.")


def _tc12_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    return generic_tool_fallback(call)


def _tc12_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    if (
        len(state.tool_calls) == 0
        and contains_refusal(state.final_answer)
        and (
            re.search(r"delete", state.final_answer, re.IGNORECASE) is not None
            or re.search(r"available tool", state.final_answer, re.IGNORECASE) is not None
        )
    ):
        return ScenarioEvaluation(
            "pass", 2, "Refused cleanly because no delete-email tool exists."
        )
    return ScenarioEvaluation(
        "fail", 0, "Did not refuse the unsupported email-deletion request correctly."
    )


def _tc13_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "search_files":
        query = normalize(as_string(call.arguments.get("query", "")))
        attempts = int(state.meta.get("searchAttempts", 0)) + 1
        state.meta["searchAttempts"] = attempts

        if "johnson proposal" in query and attempts == 1:
            return {"results": []}
        if "johnson" in query:
            return {
                "results": [
                    {"file_id": "file_117", "name": "Johnson_Project_Proposal_v2.docx"}
                ]
            }
    return generic_tool_fallback(call)


def _tc13_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    searches = tool_calls_by_name(state, "search_files")
    retried = len(searches) >= 2
    clarified = len(searches) == 1 and asks_for_clarification(state.final_answer)

    if retried:
        return ScenarioEvaluation("pass", 2, "Retried after the empty result and recovered.")
    if clarified:
        return ScenarioEvaluation("pass", 2, "Asked for clarification after the empty result.")
    return ScenarioEvaluation("fail", 0, "Did not adapt after the empty search response.")


def _tc14_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "get_stock_price":
        return {"error": "Service temporarily unavailable. Rate limit exceeded."}
    if call.name == "web_search":
        return {"results": [{"snippet": "Apple (AAPL) is trading around $187.42."}]}
    return generic_tool_fallback(call)


def _tc14_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    stock_calls = tool_calls_by_name(state, "get_stock_price")
    used_search = has_tool_call(state, "web_search")
    assistant_transcript = full_assistant_transcript(state)
    acknowledged_error = bool(
        re.search(r"temporarily unavailable|rate limit|service", assistant_transcript, re.IGNORECASE)
        or re.search(r"couldn't|get_stock_price", assistant_transcript, re.IGNORECASE)
    )
    offered_alternative = bool(
        re.search(
            r"web search|search(?:ing)?(?: for)?(?: the)? information online|search online|try again|fallback|check .*?(?:website|platform|source)",
            assistant_transcript,
            re.IGNORECASE,
        )
    )

    if len(stock_calls) >= 1 and acknowledged_error and (used_search or offered_alternative):
        return ScenarioEvaluation(
            "pass",
            2,
            "Acknowledged the stock tool failure and handled it gracefully.",
        )
    if len(stock_calls) >= 1 and used_search:
        return ScenarioEvaluation(
            "partial",
            1,
            "Recovered with web_search, but did not clearly surface the original error.",
        )
    return ScenarioEvaluation(
        "fail", 0, "Did not handle the tool error with enough integrity."
    )


def _tc15_handle(state: ScenarioState, call: ToolCallRecord) -> Any:
    if call.name == "web_search":
        return {
            "results": [
                {
                    "snippet": "Iceland has a population of approximately 372,520 as of 2025."
                }
            ]
        }
    if call.name == "calculator":
        result = parse_math_expression(as_string(call.arguments.get("expression", "")))
        return {"error": "Invalid expression."} if result is None else {"result": result}
    return generic_tool_fallback(call)


def _tc15_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    search_call = first_call(state, "web_search")
    calculator_call = first_call(state, "calculator")

    if (
        search_call
        and calculator_call
        and mentions_all(as_string(search_call.arguments.get("query", "")), ["iceland", "population"])
        and "372520"
        in as_string(calculator_call.arguments.get("expression", "")).replace(",", "")
    ):
        return ScenarioEvaluation(
            "pass", 2, "Used the searched population value in the calculator."
        )
    if (
        not calculator_call
        and search_call
        and answer_contains_number(state.final_answer, "7450.4")
    ):
        return ScenarioEvaluation(
            "partial", 1, "Computed the right answer mentally after searching."
        )
    return ScenarioEvaluation(
        "fail", 0, "Did not preserve the exact searched value across tool calls."
    )


# ---------------------------------------------------------------------------
# Scenario List
# ---------------------------------------------------------------------------

SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "TC-01",
        "title": "Direct Specialist Match",
        "category": "A",
        "user_message": "What's the weather like in Berlin right now?",
        "description": "Use get_weather instead of falling back to web_search.",
        "handle_tool_call": _tc01_handle,
        "evaluate": _tc01_evaluate,
    },
    {
        "id": "TC-02",
        "title": "Distractor Resistance",
        "category": "A",
        "user_message": "What is the current price of AAPL stock?",
        "description": "Use get_stock_price without extra tools.",
        "handle_tool_call": _tc02_handle,
        "evaluate": _tc02_evaluate,
    },
    {
        "id": "TC-03",
        "title": "Implicit Tool Need",
        "category": "A",
        "user_message": "I need to let Sarah know the meeting moved to 3pm.",
        "description": "Infer get_contacts followed by send_email.",
        "handle_tool_call": _tc03_handle,
        "evaluate": _tc03_evaluate,
    },
    {
        "id": "TC-04",
        "title": "Unit Handling",
        "category": "B",
        "user_message": "What's the temperature in Tokyo in Fahrenheit?",
        "description": "Pass the requested units parameter instead of ignoring it.",
        "handle_tool_call": _tc04_handle,
        "evaluate": _tc04_evaluate,
    },
    {
        "id": "TC-05",
        "title": "Date and Time Parsing",
        "category": "B",
        "user_message": "Schedule a team standup for next Monday at 9:30am, 30 minutes, with Alex and Jamie.",
        "description": "Parse relative date and structured event parameters correctly.",
        "handle_tool_call": _tc05_handle,
        "evaluate": _tc05_evaluate,
    },
    {
        "id": "TC-06",
        "title": "Multi-Value Extraction",
        "category": "B",
        "user_message": "Translate 'Where is the nearest hospital?' from English to both Spanish and Japanese.",
        "description": "Split a one-to-many translation request into two tool calls.",
        "handle_tool_call": _tc06_handle,
        "evaluate": _tc06_evaluate,
    },
    {
        "id": "TC-07",
        "title": "Search \u2192 Read \u2192 Act",
        "category": "C",
        "user_message": "Find the Q3 budget report and email the total to my manager.",
        "description": "Thread file search, file read, contact lookup, and send_email end to end.",
        "handle_tool_call": _tc07_handle,
        "evaluate": _tc07_evaluate,
    },
    {
        "id": "TC-08",
        "title": "Conditional Branching",
        "category": "C",
        "user_message": "Check the weather in Paris. If it's raining, remind me to bring an umbrella tomorrow at 8am.",
        "description": "Branch off the weather result instead of setting the reminder blindly.",
        "handle_tool_call": _tc08_handle,
        "evaluate": _tc08_evaluate,
    },
    {
        "id": "TC-09",
        "title": "Parallel Independence",
        "category": "C",
        "user_message": "What's the weather in London and the stock price of MSFT?",
        "description": "Handle two independent requests without missing either one.",
        "handle_tool_call": _tc09_handle,
        "evaluate": _tc09_evaluate,
    },
    {
        "id": "TC-10",
        "title": "Trivial Knowledge",
        "category": "D",
        "user_message": "What year did World War II end?",
        "description": "Avoid tools for basic knowledge.",
        "handle_tool_call": _tc10_handle,
        "evaluate": _tc10_evaluate,
    },
    {
        "id": "TC-11",
        "title": "Simple Math",
        "category": "D",
        "user_message": "What is 15% of 200?",
        "description": "Avoid calculator for easy arithmetic.",
        "handle_tool_call": _tc11_handle,
        "evaluate": _tc11_evaluate,
    },
    {
        "id": "TC-12",
        "title": "Impossible Request",
        "category": "D",
        "user_message": "Delete all my emails from last month.",
        "description": "Refuse cleanly when no suitable tool exists.",
        "handle_tool_call": _tc12_handle,
        "evaluate": _tc12_evaluate,
    },
    {
        "id": "TC-13",
        "title": "Empty Results",
        "category": "E",
        "user_message": "Find the Johnson proposal document.",
        "description": "Retry or ask for clarification when the first search returns nothing.",
        "handle_tool_call": _tc13_handle,
        "evaluate": _tc13_evaluate,
    },
    {
        "id": "TC-14",
        "title": "Malformed Response",
        "category": "E",
        "user_message": "What's Apple's stock price?",
        "description": "Surface tool errors instead of hallucinating a price.",
        "handle_tool_call": _tc14_handle,
        "evaluate": _tc14_evaluate,
    },
    {
        "id": "TC-15",
        "title": "Conflicting Information",
        "category": "E",
        "user_message": "Search for the population of Iceland and calculate what 2% of it would be.",
        "description": "Carry the exact searched value into the calculator.",
        "handle_tool_call": _tc15_handle,
        "evaluate": _tc15_evaluate,
    },
]

CATEGORY_LABELS: dict[str, str] = {
    "A": "Tool Selection",
    "B": "Parameter Precision",
    "C": "Multi-Step Chains",
    "D": "Restraint & Refusal",
    "E": "Error Recovery",
}

SCENARIO_DISPLAY_DETAILS: dict[str, dict[str, str]] = {
    "TC-01": {
        "successCase": "Pass if it calls get_weather for Berlin and avoids web_search.",
        "failureCase": "Fail if it searches the web, calls multiple tools, or answers from memory.",
    },
    "TC-02": {
        "successCase": "Pass if it uses only get_stock_price with ticker AAPL.",
        "failureCase": "Fail if it uses distractor tools or answers without a stock lookup.",
    },
    "TC-03": {
        "successCase": "Pass if it looks up Sarah first, then sends the message with the resolved address.",
        "failureCase": "Fail if it invents Sarah's email or never completes the contact-to-email chain.",
    },
    "TC-04": {
        "successCase": "Pass if it requests Tokyo weather with units set to fahrenheit.",
        "failureCase": "Fail if it ignores the Fahrenheit instruction.",
    },
    "TC-05": {
        "successCase": "Pass if it creates the event for 2026-03-23 at 09:30 with 30 minutes and Alex plus Jamie.",
        "failureCase": "Fail if it misparses next Monday or drops core event details.",
    },
    "TC-06": {
        "successCase": "Pass if it makes two translate_text calls, one for Spanish and one for Japanese.",
        "failureCase": "Fail if it combines both languages into one call or only translates one of them.",
    },
    "TC-07": {
        "successCase": "Pass if it searches, reads, resolves the manager, and emails the real total.",
        "failureCase": "Fail if it invents the total or the manager address.",
    },
    "TC-08": {
        "successCase": "Pass if it checks Paris weather first and only then sets the umbrella reminder.",
        "failureCase": "Fail if it skips the weather lookup or ignores the condition.",
    },
    "TC-09": {
        "successCase": "Pass if it handles both London weather and MSFT price in the same response flow.",
        "failureCase": "Fail if it misses one side of the request.",
    },
    "TC-10": {
        "successCase": "Pass if it answers 1945 directly with no tool call.",
        "failureCase": "Fail if it uses any tool for basic history.",
    },
    "TC-11": {
        "successCase": "Pass if it answers 30 directly with no calculator.",
        "failureCase": "Fail if it overuses tools for simple arithmetic.",
    },
    "TC-12": {
        "successCase": "Pass if it clearly refuses because no delete-email tool exists.",
        "failureCase": "Fail if it hallucinates a delete action or misuses another tool.",
    },
    "TC-13": {
        "successCase": "Pass if it retries the search or asks for clarification after empty results.",
        "failureCase": "Fail if it gives up or invents a file.",
    },
    "TC-14": {
        "successCase": "Pass if it surfaces the stock tool error and handles it honestly.",
        "failureCase": "Fail if it hides the error and fabricates a price.",
    },
    "TC-15": {
        "successCase": "Pass if it searches first, then calculates 2% using the exact searched population value.",
        "failureCase": "Fail if it skips the search or uses a memorized rounded number.",
    },
}

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_model_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute category scores and final score from scenario results."""
    categories = ["A", "B", "C", "D", "E"]
    category_scores = []

    for category in categories:
        earned = sum(
            r["points"]
            for r in results
            if any(
                s["id"] == r["scenarioId"] and s["category"] == category
                for s in SCENARIOS
            )
        )
        category_scores.append(
            {
                "category": category,
                "label": CATEGORY_LABELS[category],
                "earned": earned,
                "max": 6,
                "percent": round(earned / 6 * 100),
            }
        )

    final_score = round(
        sum(cs["percent"] for cs in category_scores) / len(category_scores)
    )
    total_points = sum(r["points"] for r in results)

    return {
        "scenarioResults": results,
        "categoryScores": category_scores,
        "finalScore": final_score,
        "totalPoints": total_points,
        "maxPoints": len(SCENARIOS) * 2,
        "rating": rating_for_score(final_score),
    }


def rating_for_score(score: int) -> str:
    if score >= 90:
        return "\u2605\u2605\u2605\u2605\u2605 Excellent"
    if score >= 75:
        return "\u2605\u2605\u2605\u2605 Good"
    if score >= 60:
        return "\u2605\u2605\u2605 Adequate"
    if score >= 40:
        return "\u2605\u2605 Weak"
    return "\u2605 Poor"


# ---------------------------------------------------------------------------
# LLM Client
# ---------------------------------------------------------------------------


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def normalize_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        return "".join(parts).strip()
    return ""


def normalize_tool_calls(message: dict[str, Any]) -> list[dict[str, Any]]:
    raw_calls = message.get("tool_calls") or []
    result = []
    for idx, call in enumerate(raw_calls):
        fn = call.get("function") or {}
        raw_args = fn.get("arguments", "{}")
        if not isinstance(raw_args, str):
            raw_args = json.dumps(raw_args)
        result.append(
            {
                "id": call.get("id") or f"tool_call_{idx + 1}",
                "type": "function",
                "function": {
                    "name": fn.get("name") or "unknown_tool",
                    "arguments": raw_args,
                },
            }
        )
    return result


def call_model(
    model: ModelConfig,
    messages: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat/completions endpoint."""
    params = params or {}
    base_url = normalize_base_url(model.base_url)

    timeout_seconds = DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS
    if params.get("request_timeout_seconds") is not None:
        val = int(params["request_timeout_seconds"])
        if val > 0:
            timeout_seconds = val
    else:
        env_timeout = os.environ.get("MODEL_REQUEST_TIMEOUT_SECONDS", "").strip()
        if env_timeout:
            try:
                val = int(env_timeout)
                if val > 0:
                    timeout_seconds = val
            except ValueError:
                pass

    headers: dict[str, str] = {"Content-Type": "application/json"}
    if model.api_key:
        headers["Authorization"] = f"Bearer {model.api_key}"

    body: dict[str, Any] = {
        "model": model.model,
        "messages": messages,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": UNIVERSAL_TOOLS,
    }

    for key in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty"):
        if key in params and params[key] is not None:
            body[key] = params[key]

    url = f"{base_url}/chat/completions"

    resp = _HTTP_SESSION.post(
        url, headers=headers, json=body, timeout=timeout_seconds,
    )

    if not resp.ok:
        try:
            payload = resp.json()
            err_msg = (
                payload.get("error", {}).get("message")
                or f"Provider request failed with {resp.status_code}."
            )
        except Exception:
            err_msg = f"Provider request failed with {resp.status_code}."
        raise RuntimeError(err_msg)

    payload = resp.json()
    message = None
    choices = payload.get("choices") or []
    if choices:
        message = choices[0].get("message")

    if not message:
        raise RuntimeError("Provider returned no assistant message.")

    content = normalize_content(message.get("content", ""))
    tool_calls = normalize_tool_calls(message)

    return {
        "content": content,
        "toolCalls": tool_calls,
    }


def create_initial_messages(user_message: str) -> list[dict[str, Any]]:
    return [
        {
            "role": "system",
            "content": (
                f"{SYSTEM_PROMPT}\n\n"
                f"Benchmark context: today is {BENCHMARK_REFERENCE_DATE} ({BENCHMARK_REFERENCE_DAY}). "
                "Use this date for any relative time request."
            ),
        },
        {"role": "user", "content": user_message},
    ]


# ---------------------------------------------------------------------------
# Scenario Executor
# ---------------------------------------------------------------------------


def parse_tool_arguments(raw_arguments: str) -> dict[str, Any]:
    try:
        parsed = json.loads(raw_arguments)
        if isinstance(parsed, dict):
            return parsed
        return {}
    except (json.JSONDecodeError, TypeError):
        return {}


def format_scenario_trace(
    model: ModelConfig,
    scenario: dict[str, Any],
    evaluation: dict[str, Any],
    trace_lines: list[str],
) -> str:
    lines = [
        f"model={model.model}",
        f"scenario={scenario['id']} {scenario['title']}",
        f"prompt={scenario['user_message']}",
        "",
        *trace_lines,
        "",
        f"verdict={evaluation['status']}",
        f"summary={evaluation['summary']}",
    ]
    if evaluation.get("note"):
        lines.append(f"note={evaluation['note']}")
    return "\n".join(line for line in lines if line is not None)


def run_scenario_for_model(
    model: ModelConfig,
    scenario: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a single scenario against a model and return the result."""
    state = ScenarioState()
    messages = create_initial_messages(scenario["user_message"])
    trace_lines: list[str] = ["assistant=starting"]
    params = params or {}

    try:
        for turn in range(1, MAX_TURNS + 1):
            response = None
            last_error = None

            for attempt in range(1, MAX_PROVIDER_ERROR_ATTEMPTS + 1):
                try:
                    response = call_model(model, messages, params)
                    last_error = None
                    break
                except Exception as exc:
                    last_error = exc
                    msg = str(exc)
                    if (
                        not PROVIDER_ERROR_RETRY_PATTERN.search(msg)
                        or attempt == MAX_PROVIDER_ERROR_ATTEMPTS
                    ):
                        raise
                    trace_lines.append(f"retry_attempt_{attempt}={msg}")
                    time.sleep(0.75 * attempt)

            if response is None:
                if last_error:
                    raise last_error
                raise RuntimeError("Unknown model execution error.")

            content = response["content"]
            tool_calls = response["toolCalls"]

            state.assistant_messages.append(content)

            # Build the assistant message for the conversation history
            assistant_msg: dict[str, Any] = {"role": "assistant", "content": content}
            if tool_calls:
                assistant_msg["tool_calls"] = tool_calls
            messages.append(assistant_msg)

            trace_lines.append(
                f"assistant_turn_{turn}={content or '[tool_calls_only]'}"
            )

            if not tool_calls:
                state.final_answer = content
                break

            sys.stdout.write(
                f"  {model.id} {scenario['id']}: "
                f"Requested {', '.join(tc['function']['name'] for tc in tool_calls)}\n"
            )
            sys.stdout.flush()

            for tc in tool_calls:
                fn = tc["function"]
                record = ToolCallRecord(
                    id=tc["id"],
                    name=fn["name"],
                    raw_arguments=fn["arguments"],
                    arguments=parse_tool_arguments(fn["arguments"]),
                    turn=turn,
                )
                state.tool_calls.append(record)
                trace_lines.append(f"tool_call={record.name} {record.raw_arguments}")

                result = scenario["handle_tool_call"](state, record)

                state.tool_results.append(
                    ToolResultRecord(
                        call_id=record.id, name=record.name, result=result
                    )
                )
                trace_lines.append(f"tool_result={json.dumps(result)}")
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": json.dumps(result),
                    }
                )

    except Exception as exc:
        summary = str(exc)
        trace_lines.append(f"error={summary}")
        eval_info = {"status": "fail", "summary": summary}
        return {
            "scenarioId": scenario["id"],
            "status": "fail",
            "points": 0,
            "summary": summary,
            "rawLog": format_scenario_trace(model, scenario, eval_info, trace_lines),
        }

    if not state.final_answer:
        state.final_answer = (
            state.assistant_messages[-1]
            if state.assistant_messages
            else "Model did not return a final answer."
        )

    trace_lines.append(f"final_answer={state.final_answer}")

    evaluation: ScenarioEvaluation = scenario["evaluate"](state)

    eval_dict = {
        "status": evaluation.status,
        "summary": evaluation.summary,
        "note": evaluation.note,
    }

    return {
        "scenarioId": scenario["id"],
        "status": evaluation.status,
        "points": evaluation.points,
        "summary": evaluation.summary,
        "note": evaluation.note,
        "rawLog": format_scenario_trace(model, scenario, eval_dict, trace_lines),
    }


# ---------------------------------------------------------------------------
# Model Config
# ---------------------------------------------------------------------------

PROVIDER_LABELS: dict[str, str] = {
    "openrouter": "OpenRouter",
    "ollama": "Ollama",
    "llamacpp": "llama.cpp",
    "mlx": "mlx_lm",
    "lmstudio": "LM Studio",
    "openai_compatible": "OpenAI Compatible",
}

VALID_PROVIDERS = {"openrouter", "ollama", "llamacpp", "mlx", "lmstudio", "openai_compatible"}


def normalize_host_base_url(host: str, env_name: str) -> str:
    trimmed = host.strip().rstrip("/")
    if not trimmed:
        raise ValueError(f"{env_name} is empty.")
    if not re.match(r"^https?://", trimmed, re.IGNORECASE):
        raise ValueError(f"{env_name} must start with http:// or https://.")

    # Parse the URL to inspect the path
    from urllib.parse import urlparse, urlunparse

    parsed = urlparse(trimmed)
    path = parsed.path.rstrip("/")

    if not path or path == "/":
        new_path = "/v1"
    elif path.endswith("/v1"):
        new_path = path
    elif path.endswith("/api"):
        new_path = path[:-4] + "/v1" if len(path) > 4 else "/v1"
    else:
        new_path = path + "/v1"

    result = urlunparse(parsed._replace(path=new_path))
    return result.rstrip("/")


def build_provider_base_url(provider: str, env_name: str) -> str:
    if provider == "openrouter":
        return "https://openrouter.ai/api/v1"

    host_env_map = {
        "ollama": "OLLAMA_HOST",
        "llamacpp": "LLAMACPP_HOST",
        "mlx": "MLX_HOST",
        "lmstudio": "LMSTUDIO_HOST",
        "openai_compatible": "OPENAI_COMPATIBLE_HOST",
    }

    host_env = host_env_map.get(provider)
    if not host_env:
        raise ValueError(f"Unknown provider: {provider}")

    host = os.environ.get(host_env, "").strip()
    if not host:
        raise ValueError(
            f"{host_env} is required when {env_name} includes a {provider} model."
        )
    return normalize_host_base_url(host, host_env)


def build_provider_api_key(provider: str, env_name: str) -> str | None:
    if provider != "openrouter":
        return None
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            f"OPENROUTER_API_KEY is required when {env_name} includes an openrouter model."
        )
    return api_key


def parse_model_entry(entry: str, index: int, env_name: str) -> ModelConfig:
    trimmed = entry.strip()
    if not trimmed:
        raise ValueError(f"{env_name} entry {index + 1} is empty.")

    sep_index = trimmed.index(":") if ":" in trimmed else -1
    if sep_index <= 0 or sep_index == len(trimmed) - 1:
        raise ValueError(
            f"{env_name} entry {index + 1} must use the format provider:model, "
            "for example openrouter:openai/gpt-4.1."
        )

    provider_raw = trimmed[:sep_index].strip().lower()
    model_name = trimmed[sep_index + 1 :].strip()

    if provider_raw not in VALID_PROVIDERS:
        raise ValueError(
            f"{env_name} entry {index + 1} has unsupported provider \"{provider_raw}\". "
            "Use openrouter, ollama, llamacpp, mlx, lmstudio, or openai_compatible."
        )

    if not model_name:
        raise ValueError(f"{env_name} entry {index + 1} is missing the model name.")

    provider_label = PROVIDER_LABELS.get(provider_raw, provider_raw)

    return ModelConfig(
        id=f"{provider_raw}:{model_name}",
        label=f"{model_name} via {provider_label}",
        model=model_name,
        base_url=build_provider_base_url(provider_raw, env_name),
        api_key=build_provider_api_key(provider_raw, env_name),
    )


def parse_model_config_list(env_name: str) -> list[ModelConfig]:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return []
    entries = [e.strip() for e in raw.split(",") if e.strip()]
    return [parse_model_entry(e, i, env_name) for i, e in enumerate(entries)]


def get_model_configs() -> list[ModelConfig]:
    primary = parse_model_config_list("LLM_MODELS")
    secondary = parse_model_config_list("LLM_MODELS_2")
    all_models = primary + secondary

    # Check for duplicates
    seen: set[str] = set()
    for m in all_models:
        if m.id in seen:
            raise ValueError(
                f'Duplicate model "{m.id}" found across LLM_MODELS and LLM_MODELS_2. '
                "Each configured provider:model must be unique."
            )
        seen.add(m.id)

    return all_models


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------


def load_dotenv() -> None:
    """Load .env file from cwd if present, without overwriting existing env vars."""
    env_path = Path.cwd() / ".env"
    if not env_path.exists():
        return
    try:
        content = env_path.read_text(encoding="utf-8")
    except OSError:
        return
    for line in content.splitlines():
        trimmed = line.strip()
        if not trimmed or trimmed.startswith("#"):
            continue
        sep_index = trimmed.find("=")
        if sep_index <= 0:
            continue
        key = trimmed[:sep_index].strip()
        value = trimmed[sep_index + 1 :].strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ToolCall-15 Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        dest="scenario_ids",
        help="Run a specific scenario (e.g. TC-01). Can be repeated.",
    )
    parser.add_argument(
        "--scenarios",
        dest="scenario_list",
        default="",
        help="Comma-separated list of scenario IDs to run.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        dest="model_ids",
        help="Run a specific model by id (e.g. openrouter:openai/gpt-4.1). Can be repeated.",
    )
    parser.add_argument(
        "--models",
        dest="model_list",
        default="",
        help="Comma-separated list of model IDs to run.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None, dest="top_p")
    parser.add_argument("--top-k", type=int, default=None, dest="top_k")
    parser.add_argument("--min-p", type=float, default=None, dest="min_p")
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=None,
        dest="repetition_penalty",
    )
    parser.add_argument("--timeout", type=int, default=None, dest="timeout")
    parser.add_argument(
        "--show-raw",
        action="store_true",
        default=False,
        dest="show_raw",
        help="Show raw trace log for each scenario run.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output results as JSON.",
    )
    return parser


def resolve_scenarios(requested_ids: list[str]) -> list[dict[str, Any]]:
    if not requested_ids:
        return SCENARIOS
    selected = [s for s in SCENARIOS if s["id"] in requested_ids]
    if len(selected) != len(set(requested_ids)):
        found_ids = {s["id"] for s in selected}
        missing = [sid for sid in requested_ids if sid not in found_ids]
        suffix = "s" if len(missing) > 1 else ""
        raise ValueError(f"Unknown scenario id{suffix}: {', '.join(missing)}")
    return selected


def resolve_models(requested_ids: list[str]) -> list[ModelConfig]:
    models = get_model_configs()
    if not requested_ids:
        return models
    selected = [m for m in models if m.id in requested_ids]
    if len(selected) != len(set(requested_ids)):
        found_ids = {m.id for m in selected}
        missing = [mid for mid in requested_ids if mid not in found_ids]
        suffix = "s" if len(missing) > 1 else ""
        raise ValueError(f"Unknown configured model id{suffix}: {', '.join(missing)}")
    return selected


def build_generation_params(args: argparse.Namespace) -> dict[str, Any] | None:
    params: dict[str, Any] = {}
    if args.temperature is not None:
        params["temperature"] = args.temperature
    if args.top_p is not None:
        params["top_p"] = args.top_p
    if args.top_k is not None:
        params["top_k"] = args.top_k
    if args.min_p is not None:
        params["min_p"] = args.min_p
    if args.repetition_penalty is not None:
        params["repetition_penalty"] = args.repetition_penalty
    if args.timeout is not None:
        params["request_timeout_seconds"] = args.timeout
    return params if params else None


def main() -> None:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args()

    # Merge scenario IDs from --scenario and --scenarios
    scenario_ids = list(args.scenario_ids)
    if args.scenario_list:
        scenario_ids.extend(
            s.strip() for s in args.scenario_list.split(",") if s.strip()
        )
    scenario_ids = list(dict.fromkeys(scenario_ids))  # dedupe, preserve order

    # Merge model IDs from --model and --models
    model_ids = list(args.model_ids)
    if args.model_list:
        model_ids.extend(m.strip() for m in args.model_list.split(",") if m.strip())
    model_ids = list(dict.fromkeys(model_ids))  # dedupe, preserve order

    scenarios = resolve_scenarios(scenario_ids)
    models = resolve_models(model_ids)
    generation_params = build_generation_params(args)

    if not models:
        print(
            "Error: No models are configured. Add entries to LLM_MODELS or LLM_MODELS_2 in .env.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Run benchmark
    summaries: list[dict[str, Any]] = []
    model_results: dict[str, list[dict[str, Any]]] = {m.id: [] for m in models}

    for scenario_index, scenario in enumerate(scenarios):
        sys.stdout.write(
            f"\n[{scenario_index + 1}/{len(scenarios)}] {scenario['id']} {scenario['title']}\n"
        )
        sys.stdout.flush()

        scenario_summary: dict[str, Any] = {
            "scenarioId": scenario["id"],
            "title": scenario["title"],
            "results": [],
        }

        for model in models:
            sys.stdout.write(
                f"  {model.id} {scenario['id']}: Calling model\n"
            )
            sys.stdout.flush()

            result = run_scenario_for_model(model, scenario, generation_params)
            model_results[model.id].append(result)

            entry: dict[str, Any] = {
                "modelId": model.id,
                "label": model.label,
                "status": result["status"],
                "points": result["points"],
                "summary": result["summary"],
            }
            if result.get("note"):
                entry["note"] = result["note"]
            if args.show_raw:
                entry["rawLog"] = result.get("rawLog")

            scenario_summary["results"].append(entry)

        summaries.append(scenario_summary)

    # Output
    if args.json:
        payload = {
            "scenarios": summaries,
            "scores": {
                mid: score_model_results(results)
                for mid, results in model_results.items()
            },
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    # Human-readable output
    for summary in summaries:
        print(f"\n{summary['scenarioId']} {summary['title']}")
        for result in summary["results"]:
            print(f"- {result['label']}")
            print(f"  status: {result['status']}")
            print(f"  points: {result['points']}")
            print(f"  summary: {result['summary']}")
            if result.get("note"):
                print(f"  note: {result['note']}")
            if result.get("rawLog"):
                print("  raw:")
                for line in result["rawLog"].split("\n"):
                    print(f"    {line}")

    ran_full_suite = len(scenarios) == len(SCENARIOS)

    if not ran_full_suite:
        print(
            "\nSubset run note: per-scenario scores above are authoritative for this audit pass."
        )
        return

    print("\nFinal scores")
    for model in models:
        results = model_results[model.id]
        summary = score_model_results(results)
        print(
            f"- {model.id}: {summary['finalScore']}/100 "
            f"({summary['totalPoints']}/{summary['maxPoints']}) "
            f"{summary['rating']}"
        )


if __name__ == "__main__":
    main()
