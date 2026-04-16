#!/usr/bin/env python3
"""
BugFind-15 Benchmark - Python Port

A faithful port of the BugFind-15 benchmark from TypeScript to Python.
Evaluates LLM bug-finding ability across 15 scenarios in 5 categories.

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
    session.trust_env = False
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
    "You are an expert software debugger. The user will show you code that may have a bug.\n"
    "\n"
    "Your job:\n"
    "1. IDENTIFY the bug \u2014 point to the exact line(s) and explain what is wrong.\n"
    "2. EXPLAIN why it causes the observed behavior.\n"
    "3. FIX the code \u2014 provide the corrected version.\n"
    "\n"
    "Rules:\n"
    "- Do not rewrite the entire program. Only change what is necessary.\n"
    "- If the code looks correct and the described behavior seems impossible, say so.\n"
    "- If you need more information to diagnose the bug, ask specific questions.\n"
    "- Do not introduce new functionality. Only fix the bug.\n"
    "- Your final answer may include brief explanation text outside the solution block, but it must include exactly one machine-readable solution block using this format:\n"
    '  <solution language="python|javascript|rust|go" verdict="fix">\n'
    "  corrected code here\n"
    "  </solution>\n"
    '- Inside <solution verdict="fix">, include raw valid source code only.\n'
    "- Do not put explanations, bullet points, placeholders, XML/HTML tags, or markdown fences such as ``` inside the <solution> block.\n"
    '- Do not write phrases like "corrected code here" or "fixed code".\n'
    "- Do not wrap the answer in tags like <response>, <analysis>, <fixed_code>, or <parameter>.\n"
    '- Use verdict="fix" only when you are actually providing corrected code.\n'
    "- For trap scenarios where there is no bug, put your explanation outside the block and use exactly this empty block:\n"
    '  <solution language="python|javascript|rust|go" verdict="no_bug"></solution>\n'
    '- When verdict="no_bug", the <solution> block must be completely empty.\n'
    "- If you ask a clarification question first, do not include a <solution> block yet. Once you give the final answer, include exactly one <solution> block."
)

BENCHMARK_REFERENCE_DATE = "2026-03-20"
BENCHMARK_REFERENCE_DAY = "Friday"

DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS = 60
MAX_ASSISTANT_TURNS = 3
MAX_PROVIDER_ERROR_ATTEMPTS = 3
PROVIDER_ERROR_RETRY_PATTERN = re.compile(r"provider returned error", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

AxisScore = int  # 0, 1, or 2
MultiTurnQuality = Literal["targeted", "generic", "irrelevant", "none"]


@dataclass
class ModelConfig:
    id: str
    label: str
    model: str
    base_url: str
    api_key: str | None = None


@dataclass
class ConversationMessage:
    role: str  # "system", "user", "assistant"
    content: str


@dataclass
class ScenarioState:
    assistant_messages: list[str] = field(default_factory=list)
    final_answer: str = ""
    conversation: list[ConversationMessage] = field(default_factory=list)
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class ScenarioEvaluation:
    status: Literal["pass", "partial", "fail"]
    score: int
    summary: str
    note: str | None = None
    axes: dict[str, AxisScore] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------

AXIS_WEIGHTS: dict[str, float] = {
    "identification": 0.35,
    "fixQuality": 0.40,
    "discipline": 0.25,
}

CATEGORY_LABELS: dict[str, str] = {
    "A": "Syntax & Surface",
    "B": "Logic & Algorithmic",
    "C": "Subtle & Tricky",
    "D": "Red Herring Resistance",
    "E": "Multi-Turn Debugging",
}

CATEGORY_WEIGHTS: dict[str, int] = {
    "A": 15,
    "B": 25,
    "C": 25,
    "D": 20,
    "E": 15,
}


def normalize(value: str) -> str:
    return value.strip().lower()


def includes_any(text: str, needles: list[str]) -> bool:
    source = normalize(text)
    return any(normalize(needle) in source for needle in needles)


def includes_all(text: str, needles: list[str]) -> bool:
    source = normalize(text)
    return all(normalize(needle) in source for needle in needles)


def matches_any(text: str, patterns: list[re.Pattern]) -> bool:
    return any(p.search(text) for p in patterns)


def strip_code_fences(text: str) -> str:
    return re.sub(r"```[\s\S]*?```", " ", text)


def strip_solution_blocks(text: str) -> str:
    return re.sub(r"<solution\b[\s\S]*?</solution>", " ", text, flags=re.IGNORECASE)


def prose_only_text(text: str) -> str:
    return strip_solution_blocks(strip_code_fences(text))


def contains_question(text: str) -> bool:
    stripped = strip_code_fences(text)
    if "?" in stripped:
        return True
    return bool(re.search(r"(^|\s)(what|which|could|can|are|is|do|does|did|where|when|why)\b", stripped, re.IGNORECASE))


def status_for_score(score: int) -> Literal["pass", "partial", "fail"]:
    if score >= 85:
        return "pass"
    if score >= 60:
        return "partial"
    return "fail"


def score_axes(axes: dict[str, AxisScore], adjustment: int = 0) -> int:
    weighted = (
        axes["identification"] * AXIS_WEIGHTS["identification"]
        + axes["fixQuality"] * AXIS_WEIGHTS["fixQuality"]
        + axes["discipline"] * AXIS_WEIGHTS["discipline"]
    )
    base = round((weighted / 2) * 100)
    return max(0, min(100, base + adjustment))


def make_evaluation(
    axes: dict[str, AxisScore],
    summary: str,
    state: ScenarioState | None = None,
    note: str | None = None,
) -> ScenarioEvaluation:
    quality: MultiTurnQuality = (state.meta.get("multiTurnQuality") or "none") if state else "none"
    adjustment = 10 if quality == "targeted" else (-5 if quality == "irrelevant" else 0)
    adjusted_axes = dict(axes)

    # No sandbox in standalone mode, skip execution result adjustments

    quality_note: str | None = None
    if quality == "targeted":
        quality_note = "Targeted clarification question earned the multi-turn bonus."
    elif quality == "irrelevant":
        quality_note = "Irrelevant clarification question incurred the multi-turn penalty."

    combined_note = " ".join(filter(None, [note, quality_note])).strip() or None
    final_score = score_axes(adjusted_axes, adjustment)

    return ScenarioEvaluation(
        status=status_for_score(final_score),
        score=final_score,
        summary=summary,
        note=combined_note,
        axes=adjusted_axes,
    )


def combined_assistant_text(state: ScenarioState) -> str:
    return "\n\n".join(state.assistant_messages)


def first_assistant_text(state: ScenarioState) -> str:
    return state.assistant_messages[0] if state.assistant_messages else ""


def mentions_no_bug(text: str) -> bool:
    return (
        includes_any(text, [
            "no bug", "code is correct", "this compiles fine", "nothing is wrong",
            "looks correct", "this code is valid", "there isn't a bug",
        ])
        or matches_any(text, [re.compile(r"no\s+issue", re.IGNORECASE), re.compile(r"works?\s+correctly", re.IGNORECASE)])
    )


def suggests_unnecessary_trap_fix(text: str) -> bool:
    return includes_any(text, ["clone()", ".clone()", "&name", "append(key)", "result.append(key)", "change to key"])


def mentions_minimal_rewrite(text: str) -> bool:
    return includes_any(text, ["rewrite", "from scratch", "completely rewrite", "new algorithm", "different approach"])


def mentions_code_pattern(text: str, patterns: list[str]) -> bool:
    source = re.sub(r"\s+", " ", text)
    return any(p in source for p in patterns)


def uses_fix_verdict(text: str) -> bool:
    return bool(re.search(r'<solution\b[^>]*verdict\s*=\s*["\']fix["\']', text, re.IGNORECASE))


def note_if_mentioned(text: str, patterns: list[re.Pattern], note: str) -> str | None:
    return note if matches_any(text, patterns) else None


# ---------------------------------------------------------------------------
# Multi-Turn Classifiers
# ---------------------------------------------------------------------------


def classify_bf13_question(text: str) -> MultiTurnQuality:
    if not contains_question(text):
        return "none"
    if matches_any(text, [
        re.compile(r"what output", re.IGNORECASE),
        re.compile(r"what (are|is).*(expect|expected)", re.IGNORECASE),
        re.compile(r"what do you see", re.IGNORECASE),
        re.compile(r"what are you getting", re.IGNORECASE),
        re.compile(r"bob.*(last|first)", re.IGNORECASE),
    ]):
        return "targeted"
    if matches_any(text, [
        re.compile(r"python version", re.IGNORECASE),
        re.compile(r"how many users", re.IGNORECASE),
        re.compile(r"framework", re.IGNORECASE),
        re.compile(r"library", re.IGNORECASE),
    ]):
        return "irrelevant"
    return "generic"


def classify_bf14_question(text: str) -> MultiTurnQuality:
    if not contains_question(text):
        return "none"
    if matches_any(text, [
        re.compile(r"production data", re.IGNORECASE),
        re.compile(r"logs", re.IGNORECASE),
        re.compile(r"shipping[_ ]address", re.IGNORECASE),
        re.compile(r"pickup", re.IGNORECASE),
        re.compile(r"missing field", re.IGNORECASE),
        re.compile(r"what does .*order.*look like", re.IGNORECASE),
    ]):
        return "targeted"
    if matches_any(text, [
        re.compile(r"node version", re.IGNORECASE),
        re.compile(r"framework", re.IGNORECASE),
        re.compile(r"npm", re.IGNORECASE),
    ]):
        return "irrelevant"
    return "generic"


def classify_bf15_question(text: str) -> MultiTurnQuality:
    if not contains_question(text):
        return "none"
    if matches_any(text, [
        re.compile(r"multi-core", re.IGNORECASE),
        re.compile(r"multi core", re.IGNORECASE),
        re.compile(r"how many cores", re.IGNORECASE),
        re.compile(r"race", re.IGNORECASE),
        re.compile(r"concurrency", re.IGNORECASE),
        re.compile(r"production server", re.IGNORECASE),
    ]):
        return "targeted"
    if matches_any(text, [
        re.compile(r"go version", re.IGNORECASE),
        re.compile(r"goroutine count", re.IGNORECASE),
        re.compile(r"memory", re.IGNORECASE),
    ]):
        return "irrelevant"
    return "generic"


def build_multi_turn_follow_up(
    classifier: Callable[[str], MultiTurnQuality],
    clarification: str,
) -> Callable[[ScenarioState, str], str | None]:
    def follow_up(state: ScenarioState, assistant_message: str) -> str | None:
        if state.meta.get("followUpSent"):
            return None
        quality = classifier(assistant_message)
        if quality == "none":
            state.meta["multiTurnQuality"] = "none"
            return None
        state.meta["multiTurnQuality"] = quality
        state.meta["followUpSent"] = True
        return clarification
    return follow_up


# ---------------------------------------------------------------------------
# Scenario Definitions
# ---------------------------------------------------------------------------


def _bf01_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    answer = combined_assistant_text(state)
    identifies_bug = (
        includes_any(answer, ["off-by-one", "range(1", "len(numbers) + 1", "starts at 1", "index 3", "out of bounds"])
        or matches_any(answer, [re.compile(r"skip.*first", re.IGNORECASE), re.compile(r"index\s*0", re.IGNORECASE)])
    )
    minimal_fix = (
        includes_any(answer, ["range(len(numbers))", "for num in numbers", "start from 0"])
        or matches_any(answer, [re.compile(r"for\s+\w+\s+in\s+numbers", re.IGNORECASE)])
    )
    uses_enumerate = includes_any(answer, ["enumerate("])

    axes = {
        "identification": 2 if identifies_bug else 0,
        "fixQuality": 2 if minimal_fix else (1 if identifies_bug else 0),
        "discipline": 1 if uses_enumerate else (2 if identifies_bug else 0),
    }
    return make_evaluation(axes, "Expected the model to isolate the off-by-one loop bounds bug.", state)


def _bf02_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    answer = combined_assistant_text(state)
    missing_empty_string = (
        includes_any(answer, ["empty string", '""', "missing case", "missing check"])
        and not includes_any(answer, ["== vs ===", "=== issue"])
    )
    good_fix = includes_any(answer, ['!== ""', "if (input)", "falsy"])
    weak_fix = includes_any(answer, ["input.length"])
    bad_diagnosis = includes_any(answer, ["== vs ===", "type coercion bug"])

    axes = {
        "identification": 2 if missing_empty_string else (0 if bad_diagnosis else 1),
        "fixQuality": 2 if good_fix else (1 if weak_fix else 0),
        "discipline": 0 if bad_diagnosis else (1 if weak_fix else (2 if missing_empty_string else 0)),
    }
    return make_evaluation(axes, "Expected the model to identify the missing empty-string case.", state)


def _bf03_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    answer = combined_assistant_text(state)
    identifies_trap = mentions_no_bug(answer) or includes_any(answer, ["format! borrows", "does not move", "compiles fine"])
    asks_for_real_code = includes_any(answer, ["double-check", "actual error", "real code", "different code"])
    false_positive = suggests_unnecessary_trap_fix(answer)

    axes = {
        "identification": 2 if (identifies_trap and not false_positive) else 0,
        "fixQuality": 2 if (identifies_trap and not false_positive) else (1 if asks_for_real_code else 0),
        "discipline": 0 if false_positive else (2 if asks_for_real_code else (2 if identifies_trap else 0)),
    }
    return make_evaluation(axes, "Expected the model to recognize that the code is already correct.", state)


def _bf04_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    answer = combined_assistant_text(state)
    identifies_bug = includes_any(answer, [
        "dictionary changed size", "modify a dictionary while iterating",
        "mutating the dictionary", "during iteration",
    ])
    good_fix = (
        includes_any(answer, ["list(users.items())", "dict comprehension", "to_remove", "collect", "return {"])
        or matches_any(answer, [re.compile(r"for .* in .*to_remove", re.IGNORECASE)])
    )
    bad_fix = includes_any(answer, ["try/except"])

    axes = {
        "identification": 2 if identifies_bug else 0,
        "fixQuality": 2 if good_fix else (1 if identifies_bug else 0),
        "discipline": 0 if bad_fix else (2 if identifies_bug else 0),
    }
    return make_evaluation(axes, "Expected the model to diagnose and fix dictionary mutation during iteration.", state)


def _bf05_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    answer = combined_assistant_text(state)
    identifies_bug = includes_any(answer, [
        "captures the variable", "closure captures", "same final value",
        "i equals 5", "loop variable",
    ])
    good_fix = includes_any(answer, ["go func(n int)", "}(i)", "i := i"])
    bad_fix = includes_any(answer, ["time.sleep", "sleep("])
    note = note_if_mentioned(answer, [
        re.compile(r"go 1\.22", re.IGNORECASE),
        re.compile(r"go1\.22", re.IGNORECASE),
        re.compile(r"1\.22.*changed", re.IGNORECASE),
    ], "Mentioned the Go 1.22 loop-variable semantic change.")

    axes = {
        "identification": 2 if identifies_bug else 0,
        "fixQuality": 2 if good_fix else (1 if identifies_bug else 0),
        "discipline": 0 if bad_fix else (2 if identifies_bug else 0),
    }
    return make_evaluation(axes, "Expected the model to diagnose loop-variable capture in Go 1.21.", state, note)


def _bf06_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    answer = combined_assistant_text(state)
    prose = prose_only_text(answer)
    identifies_promise = (
        includes_any(prose, ["response is a promise", "pending promise", "promise object"])
        or matches_any(prose, [re.compile(r"fetch\W*(returns?|is)\W+(an?\W+)?promise", re.IGNORECASE)])
    )
    await_fetch = includes_any(answer, ["await fetch"])
    await_json = includes_any(answer, ["await response.json"])
    then_chain = includes_any(answer, [".then("])

    axes = {
        "identification": 2 if identifies_promise else 0,
        "fixQuality": 2 if (await_fetch and await_json) else (1 if (await_fetch or await_json) else 0),
        "discipline": 1 if then_chain else (2 if identifies_promise else 0),
    }
    return make_evaluation(axes, "Expected the model to add both missing awaits and fix the Promise mismatch.", state)


def _bf07_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    answer = combined_assistant_text(state)
    identifies_bug = includes_any(answer, [
        "mutable default", "default argument", "evaluated once",
        "shared list", "same list object",
    ])
    good_fix = includes_any(answer, ["item_list=None", "if item_list is None", "item_list = []"])
    contract_change = includes_any(answer, ["remove the default", "require item_list"])

    axes = {
        "identification": 2 if identifies_bug else 0,
        "fixQuality": 2 if good_fix else (1 if identifies_bug else 0),
        "discipline": 1 if contract_change else (2 if identifies_bug else 0),
    }
    return make_evaluation(axes, "Expected the model to diagnose and fix the mutable default argument bug.", state)


def _bf08_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    answer = combined_assistant_text(state)
    identifies_overflow = includes_any(answer, ["integer overflow", "overflows u64", "u64::MAX", "25!"])
    checked_fix = includes_any(answer, ["checked_mul", "Option<u64>", "Result<u64", "big integer", "u128"])
    bad_fix = includes_any(answer, ["f64", "float"])

    axes = {
        "identification": 2 if identifies_overflow else 0,
        "fixQuality": 2 if (checked_fix and not bad_fix) else (1 if identifies_overflow else 0),
        "discipline": 0 if bad_fix else (2 if identifies_overflow else 0),
    }
    return make_evaluation(axes, "Expected the model to diagnose overflow handling and provide a safe factorial fix.", state)


def _bf09_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    answer = combined_assistant_text(state)
    identifies_alias = includes_any(answer, [
        "shared backing array", "same underlying array", "alias",
        "nums[:0]", "same backing memory",
    ])
    good_fix = includes_any(answer, ["make([]int, 0)", "make([]int,0)"])
    unnecessary_rewrite = includes_any(answer, ["two loops", "separate loops"])
    bad_fix = includes_any(answer, ["copy("])

    axes = {
        "identification": 2 if identifies_alias else 0,
        "fixQuality": 2 if good_fix else (1 if identifies_alias else 0),
        "discipline": 0 if bad_fix else (1 if unnecessary_rewrite else (2 if identifies_alias else 0)),
    }
    return make_evaluation(axes, "Expected the model to diagnose slice aliasing and allocate independent slices.", state)


def _bf10_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    answer = combined_assistant_text(state)
    identifies_trap = mentions_no_bug(answer) or includes_any(answer, ["preserve original", "first occurrence", "normalized key"])
    false_positive = (
        suggests_unnecessary_trap_fix(answer)
        or uses_fix_verdict(answer)
        or (
            not identifies_trap
            and (
                includes_any(answer, ["the bug is", "clear bug", "real bug", "actual bug", "the issue is", "the problem is"])
                or matches_any(answer, [re.compile(r"line\s+\d+", re.IGNORECASE), re.compile(r"\bfix\b", re.IGNORECASE)])
            )
        )
    )
    vague = includes_any(answer, ["probably fine", "naming could be better", "style"])

    axes = {
        "identification": 2 if (identifies_trap and not false_positive) else (0 if false_positive else 1),
        "fixQuality": 2 if (identifies_trap and not false_positive) else 0,
        "discipline": 0 if false_positive else (1 if vague else (2 if identifies_trap else 0)),
    }
    return make_evaluation(axes, "Expected the model to resist the red herring and confirm the code is correct.", state)


def _bf11_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    answer = combined_assistant_text(state)
    prose = prose_only_text(answer)
    identifies_silent_failure = (
        includes_any(prose, [
            "silently returns", "silent failure", "invalid input", "invalid discount",
            "110%", "range error", "explicit signal",
        ])
        or matches_any(prose, [
            re.compile(r"returns?\s+the\s+original\s+price", re.IGNORECASE),
            re.compile(r"silently\s+accept", re.IGNORECASE),
            re.compile(r"should\s+signal\s+an\s+error", re.IGNORECASE),
        ])
    )
    confirms_math = includes_any(prose, ["math is correct", "rounding is fine", "valid discounts work"])
    good_fix = includes_any(answer, ["throw new RangeError", "throw", "return null", "error object", "explicit"])
    bad_focus = includes_any(prose, ["math.round", "rounding bug", "rounding issue", "rounding problem"])
    says_fine = includes_any(prose, ["fine as-is", "looks fine"])
    note = "Explicitly confirmed that the valid-discount math already works." if confirms_math else None

    axes = {
        "identification": 2 if identifies_silent_failure else (1 if says_fine else 0),
        "fixQuality": 2 if (good_fix and not bad_focus) else (1 if identifies_silent_failure else 0),
        "discipline": 0 if bad_focus else (1 if says_fine else (2 if identifies_silent_failure else 0)),
    }
    return make_evaluation(axes, "Expected the model to address the silent invalid-input path, not the rounding math.", state, note)


def _bf12_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    answer = combined_assistant_text(state)
    prose = prose_only_text(answer)
    current_val_issue = includes_any(answer, [
        "current_val", "current streak", "compare against max_val",
        "historical best", "current run",
    ])
    final_check_issue = includes_any(answer, [
        "final streak", "last streak", "after the loop", "end of the loop",
    ])
    good_fix = current_val_issue and final_check_issue
    cosmetic_only = includes_any(prose, ["&[i32]", "&vec"])
    rewrite = mentions_minimal_rewrite(prose)

    axes = {
        "identification": 2 if good_fix else (1 if (current_val_issue or final_check_issue) else 0),
        "fixQuality": 2 if good_fix else (1 if (current_val_issue or final_check_issue) else 0),
        "discipline": 0 if cosmetic_only else (1 if rewrite else (2 if good_fix else (1 if (current_val_issue or final_check_issue) else 0))),
    }
    return make_evaluation(axes, "Expected the model to diagnose both the missing current-value tracking and the missing final comparison.", state)


def _bf13_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    answer = combined_assistant_text(state)
    identifies_bug = includes_any(answer, [
        "strings", "string", "lexicographic", "lexicographically", "age values are strings",
    ])
    good_fix = includes_any(answer, ['int(u["age"])', "int(u['age'])", "convert age to int"])
    bad_diagnosis = includes_any(answer, ["reverse=True", "reverse = true"])

    axes = {
        "identification": 2 if identifies_bug else 0,
        "fixQuality": 2 if good_fix else (1 if identifies_bug else 0),
        "discipline": 0 if bad_diagnosis else (2 if identifies_bug else 0),
    }
    return make_evaluation(axes, "Expected the model to diagnose lexicographic string sorting in the age field.", state)


def _bf14_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    answer = combined_assistant_text(state)
    prose = prose_only_text(answer)
    identifies_bug = (
        includes_any(prose, [
            "shipping_address could be undefined", "missing shipping_address",
            "missing shipping address", "order.shipping_address is undefined",
            "optional chaining", "pickup orders",
        ])
        or matches_any(prose, [
            re.compile(r"reading ['\"`]city['\"`].*undefined", re.IGNORECASE),
            re.compile(r"shipping[_ ]address.*undefined", re.IGNORECASE),
            re.compile(r"shipping[_ ]address.*missing", re.IGNORECASE),
        ])
    )
    good_fix = (
        includes_any(answer, [
            "shipping_address?.city", "order?.shipping_address?.city",
            "if (!city)", "optional chaining",
        ])
        or matches_any(answer, [
            re.compile(r"shipping_address\?\.\s*city", re.IGNORECASE),
            re.compile(r"order\?\.\s*shipping_address\?\.\s*city", re.IGNORECASE),
        ])
    )
    bad_fix = includes_any(prose, ["try/catch", "wrap the function"])

    axes = {
        "identification": 2 if identifies_bug else 0,
        "fixQuality": 2 if good_fix else (1 if identifies_bug else 0),
        "discipline": 0 if bad_fix else (2 if identifies_bug else 0),
    }
    return make_evaluation(axes, "Expected the model to diagnose production-only missing `shipping_address` data.", state)


def _bf15_evaluate(state: ScenarioState) -> ScenarioEvaluation:
    answer = combined_assistant_text(state)
    identifies_bug = includes_any(answer, [
        "data race", "race condition", "non-atomic", "read-modify-write", "count++",
    ])
    good_fix = includes_any(answer, ["sync.Mutex", "mu.Lock", "atomic.AddInt64", "sync/atomic"])
    bad_fix = includes_any(answer, ["WaitGroup fixes", "reduce goroutine count"])

    axes = {
        "identification": 2 if identifies_bug else 0,
        "fixQuality": 2 if good_fix else (1 if identifies_bug else 0),
        "discipline": 0 if bad_fix else (2 if identifies_bug else 0),
    }
    return make_evaluation(axes, "Expected the model to isolate the `count++` data race under concurrent load.", state)


# ---------------------------------------------------------------------------
# Scenario List
# ---------------------------------------------------------------------------

SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "BF-01",
        "title": "Off-by-One in Python Loop",
        "category": "A",
        "language": "Python",
        "difficulty": "Easy",
        "user_message": (
            "This function should return the sum of all elements in a list, but it's throwing an IndexError. Can you find the bug?\n\n"
            "```python\n"
            "def sum_list(numbers):\n"
            "    total = 0\n"
            "    for i in range(1, len(numbers) + 1):\n"
            "        total += numbers[i]\n"
            "    return total\n"
            "\n"
            "# sum_list([1, 2, 3])  -> IndexError: list index out of range\n"
            "# sum_list([10])       -> IndexError: list index out of range\n"
            "```"
        ),
        "description": "Tests whether the model catches an off-by-one range bug instead of hand-waving at the IndexError.",
        "success_case": "Identify the `range(1, len(numbers) + 1)` bug, explain the skipped first element, and fix the loop minimally.",
        "failure_case": "Miss the off-by-one, or fix it with an unnecessary rewrite that avoids explaining what is actually wrong.",
        "evaluate": _bf01_evaluate,
    },
    {
        "id": "BF-02",
        "title": "JavaScript Equality Trap",
        "category": "A",
        "language": "JavaScript",
        "difficulty": "Easy",
        "user_message": (
            'This validation function should reject empty strings, but `validateInput("")` returns `true`. What\'s wrong?\n\n'
            "```javascript\n"
            "function validateInput(input) {\n"
            "    if (input !== null && input !== undefined && input !== false) {\n"
            "        return true;\n"
            "    }\n"
            "    return false;\n"
            "}\n"
            "\n"
            '// validateInput("")         -> true\n'
            "// validateInput(null)       -> false\n"
            "// validateInput(undefined)  -> false\n"
            '// validateInput("hello")    -> true\n'
            "```"
        ),
        "description": "Checks whether the model finds the missing empty-string case rather than inventing a coercion issue.",
        "success_case": 'Call out the missing `""` check, explain why `""` passes, and fix it with either an explicit check or a proper falsy guard.',
        "failure_case": "Misdiagnose the issue as `==`/`===` or vague type coercion without addressing the actual missing case.",
        "evaluate": _bf02_evaluate,
    },
    {
        "id": "BF-03",
        "title": "Rust Trap Scenario",
        "category": "A",
        "language": "Rust",
        "difficulty": "Easy",
        "user_message": (
            'This Rust code won\'t compile. The compiler says "value used here after move" but I don\'t understand why. Can you help?\n\n'
            "```rust\n"
            "fn main() {\n"
            '    let name = String::from("Alice");\n'
            '    let greeting = format!("Hello, {}", name);\n'
            '    println!("{}", greeting);\n'
            '    println!("Name was: {}", name);\n'
            "}\n"
            "```"
        ),
        "description": "Trap scenario. The code is correct, so the model has to resist inventing an ownership bug.",
        "success_case": "Explain that `format!` borrows its arguments, the code compiles, and the user likely has a different error elsewhere.",
        "failure_case": "Invent a fix like `.clone()` or `&name` for a move bug that is not actually present.",
        "evaluate": _bf03_evaluate,
    },
    {
        "id": "BF-04",
        "title": "Python Dictionary Mutation During Iteration",
        "category": "B",
        "language": "Python",
        "difficulty": "Medium",
        "user_message": (
            'This function should remove all users with "inactive" status, but it crashes with '
            "`RuntimeError: dictionary changed size during iteration`. What am I doing wrong?\n\n"
            "```python\n"
            "def remove_inactive_users(users):\n"
            "    for user_id, status in users.items():\n"
            '        if status == "inactive":\n'
            "            del users[user_id]\n"
            "    return users\n"
            "\n"
            'users = {"u1": "active", "u2": "inactive", "u3": "active", "u4": "inactive"}\n'
            "print(remove_inactive_users(users))\n"
            "```"
        ),
        "description": "Measures whether the model explains mutation-during-iteration rather than suggesting a band-aid.",
        "success_case": "Explain why deleting from a dict during iteration raises `RuntimeError`, then fix it with a snapshot or two-pass approach.",
        "failure_case": "Wrap the deletion in `try/except` or avoid explaining why the dictionary view is the problem.",
        "evaluate": _bf04_evaluate,
    },
    {
        "id": "BF-05",
        "title": "Go Goroutine Loop Variable Capture",
        "category": "B",
        "language": "Go",
        "difficulty": "Medium",
        "user_message": (
            "I'm running Go 1.21. I'm trying to print numbers 0-4 using goroutines, but the output is always "
            "`5 5 5 5 5` instead. What's going on?\n\n"
            "```go\n"
            "package main\n"
            "\n"
            "import (\n"
            '    "fmt"\n'
            '    "sync"\n'
            ")\n"
            "\n"
            "func main() {\n"
            "    var wg sync.WaitGroup\n"
            "    for i := 0; i < 5; i++ {\n"
            "        wg.Add(1)\n"
            "        go func() {\n"
            "            defer wg.Done()\n"
            "            fmt.Println(i)\n"
            "        }()\n"
            "    }\n"
            "    wg.Wait()\n"
            "}\n"
            "```"
        ),
        "description": "Checks whether the model knows pre-Go-1.22 loop-variable capture semantics and fixes the closure correctly.",
        "success_case": "Explain that the goroutine closes over the loop variable `i`, then pass `i` as a parameter or shadow it per iteration.",
        "failure_case": "Offer timing hacks like `time.Sleep` instead of fixing the captured loop variable.",
        "evaluate": _bf05_evaluate,
    },
    {
        "id": "BF-06",
        "title": "JavaScript Async/Await Missing Await",
        "category": "B",
        "language": "JavaScript",
        "difficulty": "Medium",
        "user_message": (
            "This function should fetch a user and return their name, but it throws "
            "`TypeError: response.json is not a function`. The API is definitely working \u2014 I checked in the browser.\n\n"
            "```javascript\n"
            "async function getUserName(userId) {\n"
            "    const response = fetch(`/api/users/${userId}`);\n"
            "    const data = response.json();\n"
            "    return data.name;\n"
            "}\n"
            "```"
        ),
        "description": "Checks whether the model catches both missing `await` calls instead of fixing only one layer.",
        "success_case": "Explain that `fetch()` returns a Promise, then add `await` to both `fetch()` and `response.json()`.",
        "failure_case": "Patch only the `fetch()` call or switch to `.then()` chains without explaining the actual Promise issue.",
        "evaluate": _bf06_evaluate,
    },
    {
        "id": "BF-07",
        "title": "Python Mutable Default Argument",
        "category": "C",
        "language": "Python",
        "difficulty": "Hard",
        "user_message": (
            "Every time I call `add_item`, items from previous calls show up in the list. "
            "The first call works fine but subsequent calls are broken. What's happening?\n\n"
            "```python\n"
            "def add_item(item, item_list=[]):\n"
            "    item_list.append(item)\n"
            "    return item_list\n"
            "\n"
            'print(add_item("apple"))\n'
            'print(add_item("banana"))\n'
            'print(add_item("cherry"))\n'
            "```"
        ),
        "description": "Tests whether the model understands Python's one-time evaluation of mutable default arguments.",
        "success_case": "Call out the shared default list, explain why it persists across calls, and switch to the `None` sentinel pattern.",
        "failure_case": "Treat it like a random Python bug or change the function contract instead of fixing the default argument behavior.",
        "evaluate": _bf07_evaluate,
    },
    {
        "id": "BF-08",
        "title": "Rust Integer Overflow in Release",
        "category": "C",
        "language": "Rust",
        "difficulty": "Hard",
        "user_message": (
            "This function calculates factorial. It works perfectly in debug mode, but in release mode "
            "`factorial(25)` returns a clearly wrong number instead of the correct astronomical value. "
            "No error, no crash \u2014 just a silently wrong answer. What's going on?\n\n"
            "```rust\n"
            "fn factorial(n: u64) -> u64 {\n"
            "    let mut result: u64 = 1;\n"
            "    for i in 1..=n {\n"
            "        result *= i;\n"
            "    }\n"
            "    result\n"
            "}\n"
            "\n"
            "fn main() {\n"
            '    println!("{}", factorial(20));\n'
            '    println!("{}", factorial(25));\n'
            "}\n"
            "```"
        ),
        "description": "Tests whether the model knows Rust's overflow behavior difference between debug and release builds.",
        "success_case": "Identify integer overflow, explain debug panic vs release wrapping, and fix it with `checked_mul` or another explicit overflow strategy.",
        "failure_case": "Only say `use u128` without explaining why release behaves differently, or suggest floating point.",
        "evaluate": _bf08_evaluate,
    },
    {
        "id": "BF-09",
        "title": "Go Slice Aliasing",
        "category": "C",
        "language": "Go",
        "difficulty": "Hard",
        "user_message": (
            "I have a function that should return two separate filtered slices from the same input. "
            "But the results are wrong \u2014 the positive slice seems to get corrupted. What's going on?\n\n"
            "```go\n"
            "package main\n"
            "\n"
            'import "fmt"\n'
            "\n"
            "func filterPositiveAndNegative(nums []int) ([]int, []int) {\n"
            "    positive := nums[:0]\n"
            "    negative := nums[:0]\n"
            "\n"
            "    for _, n := range nums {\n"
            "        if n > 0 {\n"
            "            positive = append(positive, n)\n"
            "        } else if n < 0 {\n"
            "            negative = append(negative, n)\n"
            "        }\n"
            "    }\n"
            "    return positive, negative\n"
            "}\n"
            "```"
        ),
        "description": "Checks whether the model understands shared backing arrays in Go slices.",
        "success_case": "Explain that `nums[:0]` aliases the same backing array for both slices and allocate independent slices with `make`.",
        "failure_case": "Suggest `copy()` or unrelated rewrites without explaining the shared backing array.",
        "evaluate": _bf09_evaluate,
    },
    {
        "id": "BF-10",
        "title": "Python Red-Herring Trap",
        "category": "D",
        "language": "Python",
        "difficulty": "Medium",
        "user_message": (
            "My coworker wrote this and I'm pretty sure there's a bug. The naming is weird and the logic looks off. "
            "Can you find the issue?\n\n"
            "```python\n"
            "def process(data):\n"
            "    result = []\n"
            "    seen = set()\n"
            "    for item in data:\n"
            "        key = item.lower().strip()\n"
            "        if key not in seen:\n"
            "            seen.add(key)\n"
            "            result.append(item)\n"
            "    return result\n"
            "```"
        ),
        "description": "Trap scenario. The code is correct and intentionally preserves the first original spelling while deduplicating by normalized key.",
        "success_case": "Confirm there is no bug and explain why appending `item` while deduplicating on `key` is the intended design.",
        "failure_case": "Invent a bug by changing `result.append(item)` to append the normalized key instead.",
        "evaluate": _bf10_evaluate,
    },
    {
        "id": "BF-11",
        "title": "JavaScript Silent Failure on Invalid Discount",
        "category": "D",
        "language": "JavaScript",
        "difficulty": "Hard",
        "user_message": (
            "My discount calculator seems unreliable. For valid discounts like 15% off $100, it works fine. "
            "But I just realized that `applyDiscount(50, 110)` returns `50` instead of throwing an error. "
            "Same with `applyDiscount(50, -5)`. Is there a better way to handle this?\n\n"
            "```javascript\n"
            "function applyDiscount(price, discountPercent) {\n"
            "    if (discountPercent < 0 || discountPercent > 100) {\n"
            "        return price;\n"
            "    }\n"
            "    const discounted = price * (1 - discountPercent / 100);\n"
            "    return Math.round(discounted * 100) / 100;\n"
            "}\n"
            "```"
        ),
        "description": "Checks whether the model finds the design-level bug in silent invalid-input handling instead of touching the already-correct math.",
        "success_case": "Call out the silent return for invalid inputs, keep the valid-discount math alone, and make invalid input handling explicit.",
        "failure_case": "Try to fix `Math.round` or claim the function is already correct as-is.",
        "evaluate": _bf11_evaluate,
    },
    {
        "id": "BF-12",
        "title": "Rust Longest-Streak Double Bug",
        "category": "D",
        "language": "Rust",
        "difficulty": "Hard",
        "user_message": (
            "I'm getting wrong results from my longest-streak function. For the input `[2, 2, 1, 1, 1]`, "
            "it returns `(2, 2)` instead of `(1, 3)`. I think there might be more than one issue but I can't "
            "figure out what's wrong.\n\n"
            "```rust\n"
            "fn longest_streak(data: &Vec<i32>) -> (i32, usize) {\n"
            "    let mut max_val = data[0];\n"
            "    let mut max_count: usize = 1;\n"
            "    let mut current_count: usize = 1;\n"
            "\n"
            "    for i in 1..data.len() {\n"
            "        if data[i] == max_val {\n"
            "            current_count += 1;\n"
            "        } else if current_count > max_count {\n"
            "            max_count = current_count;\n"
            "            max_val = data[i - 1];\n"
            "            current_count = 1;\n"
            "        } else {\n"
            "            current_count = 1;\n"
            "        }\n"
            "    }\n"
            "    (max_val, max_count)\n"
            "}\n"
            "```"
        ),
        "description": "Tests whether the model finds both related bugs instead of stopping at the obvious final-streak issue.",
        "success_case": "Identify both the missing `current_val` tracking and the missing final-streak check, then patch both without overhauling the algorithm.",
        "failure_case": "Only fix one of the two issues or focus on cosmetic advice like `&[i32]` instead of the real logic bug.",
        "evaluate": _bf12_evaluate,
    },
    {
        "id": "BF-13",
        "title": "Python Ambiguous Behavior Report",
        "category": "E",
        "language": "Python",
        "difficulty": "Medium",
        "user_message": (
            "My sorting function doesn't work. It's supposed to sort users by age but the output is wrong.\n\n"
            "```python\n"
            "def sort_users(users):\n"
            '    return sorted(users, key=lambda u: u["age"])\n'
            "\n"
            "users = [\n"
            '    {"name": "Alice", "age": "30"},\n'
            '    {"name": "Bob", "age": "5"},\n'
            '    {"name": "Charlie", "age": "25"},\n'
            "]\n"
            "print(sort_users(users))\n"
            "```"
        ),
        "description": "Multi-turn scenario. A strong model either spots string sorting immediately or asks for the observed-vs-expected output.",
        "success_case": "Either identify lexicographic string sorting immediately or ask for the wrong output first, then fix by converting ages to integers.",
        "failure_case": "Ask irrelevant questions or misdiagnose it as needing `reverse=True`.",
        "get_follow_up": build_multi_turn_follow_up(
            classify_bf13_question,
            "It outputs Bob last instead of first. Bob is 5, he should be youngest and sorted first.",
        ),
        "evaluate": _bf13_evaluate,
    },
    {
        "id": "BF-14",
        "title": "JavaScript Environment-Dependent Bug",
        "category": "E",
        "language": "JavaScript",
        "difficulty": "Hard",
        "user_message": (
            "This code works on my machine but crashes in production with "
            "`Cannot read properties of undefined (reading 'city')`. I have no idea what's different.\n\n"
            "```javascript\n"
            "function getShippingZone(order) {\n"
            "    const city = order.shipping_address.city;\n"
            "    const zones = {\n"
            '        "New York": "east",\n'
            '        "Los Angeles": "west",\n'
            '        "Chicago": "central",\n'
            "    };\n"
            '    return zones[city] || "standard";\n'
            "}\n"
            "```"
        ),
        "description": "Multi-turn scenario. The strongest responses ask about production data or immediately infer missing `shipping_address` in production-only orders.",
        "success_case": "Ask about production payload differences or directly identify the missing null check on `order.shipping_address`, then fix it with a safe guard.",
        "failure_case": "Ask about Node versions or wrap the whole function in `try/catch` instead of fixing the nullable property access.",
        "get_follow_up": build_multi_turn_follow_up(
            classify_bf14_question,
            "Hmm, I just checked the production logs. Some orders from our mobile app don't have a shipping_address field at all \u2014 they're pickup orders.",
        ),
        "evaluate": _bf14_evaluate,
    },
    {
        "id": "BF-15",
        "title": "Go Race Condition Under Load",
        "category": "E",
        "language": "Go",
        "difficulty": "Expert",
        "user_message": (
            "My counter service works fine in testing but gives wrong totals under heavy load. "
            "Sometimes the count is lower than expected. I'm using goroutines. Here's the code:\n\n"
            "```go\n"
            "package main\n"
            "\n"
            "import (\n"
            '    "fmt"\n'
            '    "sync"\n'
            ")\n"
            "\n"
            "type Counter struct {\n"
            "    count int\n"
            "}\n"
            "\n"
            "func (c *Counter) Increment() {\n"
            "    c.count++\n"
            "}\n"
            "\n"
            "func (c *Counter) GetCount() int {\n"
            "    return c.count\n"
            "}\n"
            "\n"
            "func main() {\n"
            "    counter := &Counter{}\n"
            "    var wg sync.WaitGroup\n"
            "\n"
            "    for i := 0; i < 1000; i++ {\n"
            "        wg.Add(1)\n"
            "        go func() {\n"
            "            defer wg.Done()\n"
            "            counter.Increment()\n"
            "        }()\n"
            "    }\n"
            "\n"
            "    wg.Wait()\n"
            '    fmt.Println("Final count:", counter.GetCount())\n'
            "}\n"
            "```"
        ),
        "description": "Multi-turn scenario. The strongest responses identify the data race on `count++` or ask a sharp concurrency question before fixing it.",
        "success_case": "Diagnose `c.count++` as a non-atomic read-modify-write race and fix it with a mutex or `sync/atomic`.",
        "failure_case": "Claim `WaitGroup` solves it or suggest reducing goroutine count instead of synchronizing the counter.",
        "get_follow_up": build_multi_turn_follow_up(
            classify_bf15_question,
            "Yes, it's a 16-core production server. Locally on my laptop it sometimes gives 1000 correctly.",
        ),
        "evaluate": _bf15_evaluate,
    },
]

SCENARIO_DISPLAY_DETAILS: dict[str, dict[str, str]] = {
    "BF-01": {
        "successCase": "Identify the `range(1, len(numbers) + 1)` bug, explain the skipped first element, and fix the loop minimally.",
        "failureCase": "Miss the off-by-one, or fix it with an unnecessary rewrite that avoids explaining what is actually wrong.",
    },
    "BF-02": {
        "successCase": 'Call out the missing `""` check, explain why `""` passes, and fix it with either an explicit check or a proper falsy guard.',
        "failureCase": "Misdiagnose the issue as `==`/`===` or vague type coercion without addressing the actual missing case.",
    },
    "BF-03": {
        "successCase": "Explain that `format!` borrows its arguments, the code compiles, and the user likely has a different error elsewhere.",
        "failureCase": "Invent a fix like `.clone()` or `&name` for a move bug that is not actually present.",
    },
    "BF-04": {
        "successCase": "Explain why deleting from a dict during iteration raises `RuntimeError`, then fix it with a snapshot or two-pass approach.",
        "failureCase": "Wrap the deletion in `try/except` or avoid explaining why the dictionary view is the problem.",
    },
    "BF-05": {
        "successCase": "Explain that the goroutine closes over the loop variable `i`, then pass `i` as a parameter or shadow it per iteration.",
        "failureCase": "Offer timing hacks like `time.Sleep` instead of fixing the captured loop variable.",
    },
    "BF-06": {
        "successCase": "Explain that `fetch()` returns a Promise, then add `await` to both `fetch()` and `response.json()`.",
        "failureCase": "Patch only the `fetch()` call or switch to `.then()` chains without explaining the actual Promise issue.",
    },
    "BF-07": {
        "successCase": "Call out the shared default list, explain why it persists across calls, and switch to the `None` sentinel pattern.",
        "failureCase": "Treat it like a random Python bug or change the function contract instead of fixing the default argument behavior.",
    },
    "BF-08": {
        "successCase": "Identify integer overflow, explain debug panic vs release wrapping, and fix it with `checked_mul` or another explicit overflow strategy.",
        "failureCase": "Only say `use u128` without explaining why release behaves differently, or suggest floating point.",
    },
    "BF-09": {
        "successCase": "Explain that `nums[:0]` aliases the same backing array for both slices and allocate independent slices with `make`.",
        "failureCase": "Suggest `copy()` or unrelated rewrites without explaining the shared backing array.",
    },
    "BF-10": {
        "successCase": "Confirm there is no bug and explain why appending `item` while deduplicating on `key` is the intended design.",
        "failureCase": "Invent a bug by changing `result.append(item)` to append the normalized key instead.",
    },
    "BF-11": {
        "successCase": "Call out the silent return for invalid inputs, keep the valid-discount math alone, and make invalid input handling explicit.",
        "failureCase": "Try to fix `Math.round` or claim the function is already correct as-is.",
    },
    "BF-12": {
        "successCase": "Identify both the missing `current_val` tracking and the missing final-streak check, then patch both without overhauling the algorithm.",
        "failureCase": "Only fix one of the two issues or focus on cosmetic advice like `&[i32]` instead of the real logic bug.",
    },
    "BF-13": {
        "successCase": "Either identify lexicographic string sorting immediately or ask for the wrong output first, then fix by converting ages to integers.",
        "failureCase": "Ask irrelevant questions or misdiagnose it as needing `reverse=True`.",
    },
    "BF-14": {
        "successCase": "Ask about production payload differences or directly identify the missing null check on `order.shipping_address`, then fix it with a safe guard.",
        "failureCase": "Ask about Node versions or wrap the whole function in `try/catch` instead of fixing the nullable property access.",
    },
    "BF-15": {
        "successCase": "Diagnose `c.count++` as a non-atomic read-modify-write race and fix it with a mutex or `sync/atomic`.",
        "failureCase": "Claim `WaitGroup` solves it or suggest reducing goroutine count instead of synchronizing the counter.",
    },
}

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_model_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute weighted category scores and final score from scenario results."""
    categories = ["A", "B", "C", "D", "E"]
    category_scores = []

    for category in categories:
        cat_results = [
            r for r in results
            if any(s["id"] == r["scenarioId"] and s["category"] == category for s in SCENARIOS)
        ]
        avg_score = (
            round(sum(r["score"] for r in cat_results) / len(cat_results))
            if cat_results else 0
        )
        category_scores.append({
            "category": category,
            "label": CATEGORY_LABELS[category],
            "weight": CATEGORY_WEIGHTS[category],
            "averageScore": avg_score,
            "percent": avg_score,
        })

    final_score = round(
        sum(cs["averageScore"] * (cs["weight"] / 100) for cs in category_scores)
    )
    total_score = sum(r["score"] for r in results)

    return {
        "scenarioResults": results,
        "categoryScores": category_scores,
        "finalScore": final_score,
        "totalScore": total_score,
        "maxScore": len(SCENARIOS) * 100,
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


def call_model(
    model: ModelConfig,
    messages: list[dict[str, Any]],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Call an OpenAI-compatible chat/completions endpoint (no tools)."""
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
    return {"content": content}


def create_initial_messages(user_message: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


# ---------------------------------------------------------------------------
# Scenario Executor
# ---------------------------------------------------------------------------


def format_scenario_trace(
    model: ModelConfig,
    scenario: dict[str, Any],
    evaluation: dict[str, Any],
    trace_lines: list[str],
) -> str:
    lines = [
        f"model={model.model}",
        f"scenario={scenario['id']} {scenario['title']}",
        f"prompt={scenario['user_message'][:100]}...",
        "",
        *trace_lines,
        "",
        f"verdict={evaluation['status']}",
        f"score={evaluation.get('score', 0)}",
        f"summary={evaluation['summary']}",
    ]
    if evaluation.get("note"):
        lines.append(f"note={evaluation['note']}")
    if evaluation.get("axes"):
        lines.append(f"axes={json.dumps(evaluation['axes'])}")
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
    get_follow_up = scenario.get("get_follow_up")

    try:
        for turn in range(1, MAX_ASSISTANT_TURNS + 1):
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
            state.assistant_messages.append(content)
            messages.append({"role": "assistant", "content": content})
            trace_lines.append(f"assistant_turn_{turn}={content[:200]}")

            # Multi-turn: check if there's a follow-up
            if get_follow_up and turn < MAX_ASSISTANT_TURNS:
                follow_up_msg = get_follow_up(state, content)
                if follow_up_msg:
                    sys.stdout.write(
                        f"  {model.id} {scenario['id']}: Multi-turn follow-up (turn {turn + 1})\n"
                    )
                    sys.stdout.flush()
                    messages.append({"role": "user", "content": follow_up_msg})
                    trace_lines.append(f"follow_up_turn_{turn}={follow_up_msg[:200]}")
                    continue

            # No follow-up or no more turns, we're done
            state.final_answer = content
            break

    except Exception as exc:
        summary = str(exc)
        trace_lines.append(f"error={summary}")
        eval_info = {"status": "fail", "summary": summary, "score": 0}
        return {
            "scenarioId": scenario["id"],
            "status": "fail",
            "score": 0,
            "summary": summary,
            "rawLog": format_scenario_trace(model, scenario, eval_info, trace_lines),
        }

    if not state.final_answer:
        state.final_answer = (
            state.assistant_messages[-1]
            if state.assistant_messages
            else "Model did not return a final answer."
        )

    trace_lines.append(f"final_answer={state.final_answer[:300]}")

    evaluation: ScenarioEvaluation = scenario["evaluate"](state)

    eval_dict = {
        "status": evaluation.status,
        "score": evaluation.score,
        "summary": evaluation.summary,
        "note": evaluation.note,
        "axes": evaluation.axes,
    }

    return {
        "scenarioId": scenario["id"],
        "status": evaluation.status,
        "score": evaluation.score,
        "summary": evaluation.summary,
        "note": evaluation.note,
        "axes": evaluation.axes,
        "rawLog": format_scenario_trace(model, scenario, eval_dict, trace_lines),
    }


# ---------------------------------------------------------------------------
# Model Config (reused from toolcall15.py pattern)
# ---------------------------------------------------------------------------

PROVIDER_LABELS_MAP: dict[str, str] = {
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
        raise ValueError(f"{host_env} is required when {env_name} includes a {provider} model.")
    return normalize_host_base_url(host, host_env)


def build_provider_api_key(provider: str, env_name: str) -> str | None:
    if provider != "openrouter":
        return None
    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        raise ValueError(f"OPENROUTER_API_KEY is required when {env_name} includes an openrouter model.")
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
    model_name = trimmed[sep_index + 1:].strip()

    if provider_raw not in VALID_PROVIDERS:
        raise ValueError(
            f'{env_name} entry {index + 1} has unsupported provider "{provider_raw}". '
            "Use openrouter, ollama, llamacpp, mlx, lmstudio, or openai_compatible."
        )

    if not model_name:
        raise ValueError(f"{env_name} entry {index + 1} is missing the model name.")

    provider_label = PROVIDER_LABELS_MAP.get(provider_raw, provider_raw)

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
        value = trimmed[sep_index + 1:].strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BugFind-15 Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario", action="append", default=[], dest="scenario_ids",
        help="Run a specific scenario (e.g. BF-01). Can be repeated.",
    )
    parser.add_argument(
        "--scenarios", dest="scenario_list", default="",
        help="Comma-separated list of scenario IDs to run.",
    )
    parser.add_argument(
        "--model", action="append", default=[], dest="model_ids",
        help="Run a specific model by id (e.g. openrouter:openai/gpt-4.1). Can be repeated.",
    )
    parser.add_argument(
        "--models", dest="model_list", default="",
        help="Comma-separated list of model IDs to run.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None, dest="top_p")
    parser.add_argument("--top-k", type=int, default=None, dest="top_k")
    parser.add_argument("--min-p", type=float, default=None, dest="min_p")
    parser.add_argument("--repetition-penalty", type=float, default=None, dest="repetition_penalty")
    parser.add_argument("--timeout", type=int, default=None, dest="timeout")
    parser.add_argument(
        "--show-raw", action="store_true", default=False, dest="show_raw",
        help="Show raw trace log for each scenario run.",
    )
    parser.add_argument(
        "--json", action="store_true", default=False,
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

    scenario_ids = list(args.scenario_ids)
    if args.scenario_list:
        scenario_ids.extend(s.strip() for s in args.scenario_list.split(",") if s.strip())
    scenario_ids = list(dict.fromkeys(scenario_ids))

    model_ids = list(args.model_ids)
    if args.model_list:
        model_ids.extend(m.strip() for m in args.model_list.split(",") if m.strip())
    model_ids = list(dict.fromkeys(model_ids))

    scenarios = resolve_scenarios(scenario_ids)
    models = resolve_models(model_ids)
    generation_params = build_generation_params(args)

    if not models:
        print(
            "Error: No models are configured. Add entries to LLM_MODELS or LLM_MODELS_2 in .env.",
            file=sys.stderr,
        )
        sys.exit(1)

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
            sys.stdout.write(f"  {model.id} {scenario['id']}: Calling model\n")
            sys.stdout.flush()

            result = run_scenario_for_model(model, scenario, generation_params)
            model_results[model.id].append(result)

            entry: dict[str, Any] = {
                "modelId": model.id,
                "label": model.label,
                "status": result["status"],
                "score": result["score"],
                "summary": result["summary"],
            }
            if result.get("note"):
                entry["note"] = result["note"]
            if result.get("axes"):
                entry["axes"] = result["axes"]
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
            print(f"  score: {result['score']}")
            print(f"  summary: {result['summary']}")
            if result.get("note"):
                print(f"  note: {result['note']}")
            if result.get("axes"):
                print(f"  axes: {result['axes']}")
            if result.get("rawLog"):
                print("  raw:")
                for line in result["rawLog"].split("\n"):
                    print(f"    {line}")

    ran_full_suite = len(scenarios) == len(SCENARIOS)

    if not ran_full_suite:
        print("\nSubset run note: per-scenario scores above are authoritative for this audit pass.")
        return

    print("\nFinal scores")
    for model in models:
        results = model_results[model.id]
        summary = score_model_results(results)
        print(
            f"- {model.id}: {summary['finalScore']}/100 "
            f"({summary['totalScore']}/{summary['maxScore']}) "
            f"{summary['rating']}"
        )


if __name__ == "__main__":
    main()
