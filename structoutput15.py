#!/usr/bin/env python3
"""
StructOutput-15 Benchmark - Python Port

A faithful port of the StructOutput-15 benchmark from TypeScript to Python.
Evaluates LLM structured output generation across 15 scenarios in 5 categories.

Dependencies: requests (pip install requests)
Optional: pyyaml (pip install pyyaml) for YAML validation
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

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

try:
    import yaml

    HAS_YAML = True
except ImportError:
    HAS_YAML = False

try:
    import tomllib  # Python 3.11+

    HAS_TOML = True
except ImportError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]

        HAS_TOML = True
    except ImportError:
        HAS_TOML = False

import xml.etree.ElementTree as ET


def _build_http_session() -> requests.Session:
    """Build a requests Session with retry and connection pooling."""
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
    "You are a helpful assistant that generates structured output.\n"
    "\n"
    "Rules:\n"
    "- Output ONLY the requested format.\n"
    "- Do not add explanations, markdown fences, labels, or surrounding prose.\n"
    "- Follow the exact schema or structure described in the request.\n"
    "- Use proper escaping for special characters in the target format.\n"
    "- Preserve required data types exactly.\n"
    "- If a field has no value, use the format's standard null/empty representation."
)

DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS = 30
MAX_PROVIDER_ERROR_ATTEMPTS = 3
PROVIDER_ERROR_RETRY_PATTERN = re.compile(r"provider returned error", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Axis scoring
# ---------------------------------------------------------------------------

AXIS_WEIGHTS: dict[str, float] = {
    "parseable": 0.40,
    "correctness": 0.35,
    "discipline": 0.25,
}

CATEGORY_LABELS: dict[str, str] = {
    "A": "Basic Single-Format",
    "B": "Less Common Formats",
    "C": "Complex Structures",
    "D": "Conversion & Multi-Format",
    "E": "Adversarial Edge Cases",
    "S": "Supplemental",
}

CATEGORY_WEIGHTS: dict[str, int] = {
    "A": 15,
    "B": 20,
    "C": 25,
    "D": 20,
    "E": 20,
    "S": 0,
}

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
class ScenarioState:
    assistant_messages: list[str] = field(default_factory=list)
    final_answer: str = ""
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class AxisScores:
    parseable: int = 0  # 0, 1, or 2
    correctness: int = 0
    discipline: int = 0


@dataclass
class ScenarioEvaluation:
    status: Literal["pass", "partial", "fail"]
    score: int
    summary: str
    note: str | None = None
    axes: AxisScores = field(default_factory=AxisScores)


# ---------------------------------------------------------------------------
# Scoring helpers
# ---------------------------------------------------------------------------


def score_axes(axes: AxisScores) -> int:
    weighted = (
        axes.parseable * AXIS_WEIGHTS["parseable"]
        + axes.correctness * AXIS_WEIGHTS["correctness"]
        + axes.discipline * AXIS_WEIGHTS["discipline"]
    )
    return round((weighted / 2) * 100)


def status_for_score(score: int) -> Literal["pass", "partial", "fail"]:
    if score >= 85:
        return "pass"
    if score >= 60:
        return "partial"
    return "fail"


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


def make_eval(axes: AxisScores, summary: str, note: str | None = None) -> ScenarioEvaluation:
    sc = score_axes(axes)
    return ScenarioEvaluation(
        status=status_for_score(sc),
        score=sc,
        summary=summary,
        note=note,
        axes=axes,
    )


# ---------------------------------------------------------------------------
# Discipline helper: check for unwanted wrappers/prose
# ---------------------------------------------------------------------------


def discipline_score(text: str, format_hint: str = "") -> int:
    """2 = clean output only, 1 = minor wrapper, 0 = lots of prose."""
    stripped = text.strip()
    has_fence = bool(re.search(r"^```", stripped, re.MULTILINE))
    # Check for leading/trailing prose lines outside the actual content
    lines = stripped.splitlines()
    prose_lines = 0
    for line in lines:
        l = line.strip()
        if not l:
            continue
        # heuristic: if line looks like natural language sentence
        if re.match(r"^(Here|Below|This|The |I |Note|Sure|Of course|Certainly)", l, re.IGNORECASE):
            prose_lines += 1
    if has_fence and prose_lines > 0:
        return 0
    if has_fence or prose_lines > 1:
        return 1
    if prose_lines == 1:
        return 1
    return 2


def strip_fences(text: str) -> str:
    """Remove markdown code fences if present."""
    stripped = text.strip()
    m = re.match(r"^```\w*\n(.*?)```\s*$", stripped, re.DOTALL)
    if m:
        return m.group(1).strip()
    return stripped


# ---------------------------------------------------------------------------
# Evaluators for each scenario
# ---------------------------------------------------------------------------


def evaluate_so01(state: ScenarioState) -> ScenarioEvaluation:
    """SO-01: Simple JSON Object"""
    raw = state.final_answer
    content = strip_fences(raw)
    disc = discipline_score(raw)

    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return make_eval(AxisScores(0, 0, disc), "Invalid JSON.")

    if not isinstance(obj, dict):
        return make_eval(AxisScores(1, 0, disc), "Parsed but not an object.")

    correctness = 2
    checks = [
        obj.get("title") == "The Great Gatsby",
        obj.get("author") == "F. Scott Fitzgerald",
        obj.get("year") == 1925,
        obj.get("genre") == "Novel",
        obj.get("in_print") is True,
    ]
    passed = sum(checks)
    if passed < 3:
        correctness = 0
    elif passed < 5:
        correctness = 1

    return make_eval(AxisScores(2, correctness, disc), f"JSON parsed, {passed}/5 fields correct.")


def evaluate_so02(state: ScenarioState) -> ScenarioEvaluation:
    """SO-02: CSV with Headers"""
    raw = state.final_answer
    content = strip_fences(raw)
    disc = discipline_score(raw)

    try:
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
    except Exception:
        return make_eval(AxisScores(0, 0, disc), "Could not parse CSV.")

    if len(rows) < 1:
        return make_eval(AxisScores(0, 0, disc), "Empty CSV.")

    headers = [h.strip().lower() for h in rows[0]]
    expected_headers = ["name", "age", "city", "email"]
    header_ok = headers == expected_headers

    data_rows = [r for r in rows[1:] if any(c.strip() for c in r)]
    row_count_ok = len(data_rows) == 3

    correctness = 0
    if header_ok and row_count_ok:
        # Check content
        names = [r[0].strip() for r in data_rows] if all(len(r) >= 4 for r in data_rows) else []
        if "Alice Johnson" in names and "Bob Smith" in names and "Carol White" in names:
            correctness = 2
        else:
            correctness = 1
    elif header_ok or row_count_ok:
        correctness = 1

    parse_score = 2 if header_ok else (1 if len(rows) > 1 else 0)
    return make_eval(AxisScores(parse_score, correctness, disc), f"CSV headers={'ok' if header_ok else 'wrong'}, rows={len(data_rows)}.")


def evaluate_so03(state: ScenarioState) -> ScenarioEvaluation:
    """SO-03: YAML Configuration"""
    raw = state.final_answer
    content = strip_fences(raw)
    disc = discipline_score(raw)

    if not HAS_YAML:
        # Best-effort: check key patterns
        has_host = "host:" in content and "0.0.0.0" in content
        has_port = "port:" in content and "8080" in content
        has_db = "database:" in content or "db:" in content
        parse_ok = has_host and has_port
        correctness = 2 if (has_host and has_port and has_db) else (1 if parse_ok else 0)
        return make_eval(
            AxisScores(2 if parse_ok else 1, correctness, disc),
            "YAML checked by pattern (pyyaml not installed).",
        )

    try:
        obj = yaml.safe_load(content)
    except Exception:
        return make_eval(AxisScores(0, 0, disc), "Invalid YAML.")

    if not isinstance(obj, dict):
        return make_eval(AxisScores(1, 0, disc), "YAML parsed but not a mapping.")

    checks = [
        obj.get("host") == "0.0.0.0",
        obj.get("port") == 8080,
        obj.get("debug") is False,
        isinstance(obj.get("allowed_origins"), list) and len(obj.get("allowed_origins", [])) == 2,
        isinstance(obj.get("database"), dict),
    ]
    passed = sum(checks)
    correctness = 2 if passed >= 4 else (1 if passed >= 2 else 0)
    return make_eval(AxisScores(2, correctness, disc), f"YAML parsed, {passed}/5 checks passed.")


def evaluate_so04(state: ScenarioState) -> ScenarioEvaluation:
    """SO-04: TOML Configuration"""
    raw = state.final_answer
    content = strip_fences(raw)
    disc = discipline_score(raw)

    if not HAS_TOML:
        has_package = "[package]" in content
        has_deps = "[dependencies]" in content or "serde" in content
        has_name = '"my_cli"' in content or "'my_cli'" in content
        parse_ok = has_package and has_name
        correctness = 2 if (has_package and has_deps and has_name) else (1 if parse_ok else 0)
        return make_eval(
            AxisScores(2 if parse_ok else 1, correctness, disc),
            "TOML checked by pattern (tomllib not available).",
        )

    try:
        obj = tomllib.loads(content)
    except Exception:
        return make_eval(AxisScores(0, 0, disc), "Invalid TOML.")

    pkg = obj.get("package", {})
    deps = obj.get("dependencies", {})
    checks = [
        pkg.get("name") == "my_cli",
        pkg.get("version") == "0.1.0",
        pkg.get("edition") == "2021",
        isinstance(pkg.get("authors"), list),
        "serde" in deps or any("serde" in str(k) for k in deps),
    ]
    passed = sum(checks)
    correctness = 2 if passed >= 4 else (1 if passed >= 2 else 0)
    return make_eval(AxisScores(2, correctness, disc), f"TOML parsed, {passed}/5 checks passed.")


def evaluate_so05(state: ScenarioState) -> ScenarioEvaluation:
    """SO-05: SQL CREATE + INSERT"""
    raw = state.final_answer
    content = strip_fences(raw).upper()
    disc = discipline_score(raw)

    has_create = bool(re.search(r"CREATE\s+TABLE", content))
    has_employees = "EMPLOYEES" in content
    has_insert = content.count("INSERT") >= 2 or (
        "INSERT" in content and ("VALUES" in content)
    )
    # Check for both names
    content_lower = strip_fences(raw).lower()
    has_alice = "alice chen" in content_lower
    has_bob = "bob park" in content_lower

    parse_score = 2 if (has_create and has_employees) else (1 if has_create else 0)
    checks = [has_create, has_employees, has_insert, has_alice, has_bob]
    passed = sum(checks)
    correctness = 2 if passed >= 4 else (1 if passed >= 2 else 0)
    return make_eval(AxisScores(parse_score, correctness, disc), f"SQL structure {passed}/5 checks passed.")


def evaluate_so06(state: ScenarioState) -> ScenarioEvaluation:
    """SO-06: ICS Calendar Event"""
    raw = state.final_answer
    content = strip_fences(raw)
    disc = discipline_score(raw)

    has_vcalendar = "BEGIN:VCALENDAR" in content and "END:VCALENDAR" in content
    has_vevent = "BEGIN:VEVENT" in content and "END:VEVENT" in content
    has_summary = "Q2 Planning Session" in content or "Q2 Planning" in content
    has_location = "Conference Room B" in content
    has_organizer = "alice@company.com" in content
    # Check for April 15 2026 date
    has_date = bool(re.search(r"20260415", content))

    parse_score = 2 if (has_vcalendar and has_vevent) else (1 if has_vevent else 0)
    checks = [has_vcalendar, has_vevent, has_summary, has_location, has_organizer, has_date]
    passed = sum(checks)
    correctness = 2 if passed >= 5 else (1 if passed >= 3 else 0)
    return make_eval(AxisScores(parse_score, correctness, disc), f"ICS structure {passed}/6 checks passed.")


def evaluate_so07(state: ScenarioState) -> ScenarioEvaluation:
    """SO-07: Nested JSON with Arrays and Nulls"""
    raw = state.final_answer
    content = strip_fences(raw)
    disc = discipline_score(raw)

    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return make_eval(AxisScores(0, 0, disc), "Invalid JSON.")

    if not isinstance(obj, dict):
        return make_eval(AxisScores(1, 0, disc), "Parsed but not an object.")

    checks = [
        obj.get("id") == 42,
        obj.get("username") == "j_doe",
        obj.get("email") is None,
        isinstance(obj.get("roles"), list) and "editor" in obj.get("roles", []),
        isinstance(obj.get("address"), dict) and obj.get("address", {}).get("city") == "Springfield",
        isinstance(obj.get("phone_numbers"), list) and len(obj.get("phone_numbers", [])) == 2,
    ]
    # Check nested null in phone_numbers
    phones = obj.get("phone_numbers", [])
    if isinstance(phones, list) and len(phones) >= 2:
        work_phone = next((p for p in phones if isinstance(p, dict) and p.get("type") == "work"), None)
        if work_phone and work_phone.get("number") is None:
            checks.append(True)
        else:
            checks.append(False)
    else:
        checks.append(False)

    # Check nested_null
    nested = obj.get("nested_null")
    if isinstance(nested, dict) and nested.get("a") is None:
        checks.append(True)
    else:
        checks.append(False)

    # Check metadata
    meta = obj.get("metadata")
    if isinstance(meta, dict) and meta.get("login_count") == 847:
        checks.append(True)
    else:
        checks.append(False)

    passed = sum(checks)
    correctness = 2 if passed >= 7 else (1 if passed >= 4 else 0)
    return make_eval(AxisScores(2, correctness, disc), f"Nested JSON {passed}/9 checks passed.")


def evaluate_so08(state: ScenarioState) -> ScenarioEvaluation:
    """SO-08: CSV with Special Characters"""
    raw = state.final_answer
    content = strip_fences(raw)
    disc = discipline_score(raw)

    try:
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
    except Exception:
        return make_eval(AxisScores(0, 0, disc), "Could not parse CSV.")

    data_rows = rows[1:] if len(rows) > 1 else rows
    # Filter empties
    data_rows = [r for r in data_rows if any(c.strip() for c in r)]

    parse_score = 2 if len(data_rows) >= 3 else (1 if len(data_rows) >= 1 else 0)

    # Check for key content
    all_text = " ".join(" ".join(r) for r in data_rows)
    has_acme = "Acme" in all_text
    has_obrien = "Brien" in all_text
    has_japanese = any(ord(c) > 0x3000 for c in all_text)

    passed = sum([has_acme, has_obrien, has_japanese])
    correctness = 2 if passed == 3 else (1 if passed >= 1 else 0)
    return make_eval(AxisScores(parse_score, correctness, disc), f"CSV special chars {passed}/3 content checks passed.")


def evaluate_so09(state: ScenarioState) -> ScenarioEvaluation:
    """SO-09: XML with Namespaces and Attributes"""
    raw = state.final_answer
    content = strip_fences(raw)
    disc = discipline_score(raw)

    has_declaration = content.strip().startswith("<?xml")

    try:
        root = ET.fromstring(content)
        parsed = True
    except ET.ParseError:
        parsed = False

    if not parsed:
        return make_eval(AxisScores(0, 0, disc), "Invalid XML.")

    # Check namespace (may be in tag)
    has_ns = "http://example.com/books" in content
    has_version = 'version="2.0"' in content or "version='2.0'" in content

    # Count book elements (namespace-aware)
    books = root.findall(".//{*}book") or root.findall(".//book")
    if not books:
        # try with namespace prefix
        books = root.findall(".//{http://example.com/books}book")
    book_count = len(books)

    has_bk101 = 'bk101' in content
    has_bk102 = 'bk102' in content
    has_jpn = any(ord(c) > 0x3000 for c in content)

    parse_score = 2 if (parsed and has_declaration) else (1 if parsed else 0)
    checks = [has_ns, has_version, book_count >= 2, has_bk101, has_bk102, has_jpn]
    passed = sum(checks)
    correctness = 2 if passed >= 5 else (1 if passed >= 3 else 0)
    return make_eval(AxisScores(parse_score, correctness, disc), f"XML {passed}/6 checks passed.")


def evaluate_so10(state: ScenarioState) -> ScenarioEvaluation:
    """SO-10: JSON to Markdown Table"""
    raw = state.final_answer
    content = strip_fences(raw)
    disc = discipline_score(raw)

    lines = [l.strip() for l in content.strip().splitlines() if l.strip()]

    # Check for table structure
    has_header = bool(lines) and "|" in lines[0]
    has_separator = len(lines) > 1 and bool(re.match(r"^[\|\s\-:]+$", lines[1]))
    data_lines = [l for l in lines[2:] if "|" in l] if len(lines) > 2 else []

    parse_score = 2 if (has_header and has_separator and len(data_lines) >= 4) else (
        1 if has_header else 0
    )

    # Check content
    all_text = " ".join(lines)
    checks = [
        "Alice" in all_text,
        "Bob" in all_text,
        "Carol" in all_text,
        "Dave" in all_text,
        "95" in all_text,
        "82" in all_text,
    ]
    passed = sum(checks)
    correctness = 2 if passed >= 5 else (1 if passed >= 3 else 0)
    return make_eval(AxisScores(parse_score, correctness, disc), f"Markdown table {passed}/6 content checks.")


def evaluate_so11(state: ScenarioState) -> ScenarioEvaluation:
    """SO-11: Mermaid Flowchart"""
    raw = state.final_answer
    content = strip_fences(raw)
    disc = discipline_score(raw)

    # Check for flowchart keyword
    has_flowchart = bool(re.search(r"(flowchart|graph)\s+(TD|TB|LR|RL|BT)", content, re.IGNORECASE))
    has_arrow = "-->" in content
    # Check for branching (conditional)
    has_branch = bool(re.search(r"\{.*\}", content)) or "Yes" in content or "No" in content or "valid" in content.lower()

    # Key concepts
    all_lower = content.lower()
    has_submit = "submit" in all_lower or "form" in all_lower
    has_validate = "valid" in all_lower
    has_save = "save" in all_lower or "database" in all_lower
    has_error = "error" in all_lower or "invalid" in all_lower

    parse_score = 2 if (has_flowchart and has_arrow) else (1 if has_arrow else 0)
    checks = [has_flowchart, has_arrow, has_branch, has_submit, has_validate, has_save, has_error]
    passed = sum(checks)
    correctness = 2 if passed >= 6 else (1 if passed >= 3 else 0)
    return make_eval(AxisScores(parse_score, correctness, disc), f"Mermaid flowchart {passed}/7 checks.")


def evaluate_so12(state: ScenarioState) -> ScenarioEvaluation:
    """SO-12: HTML Table with Semantic Markup"""
    raw = state.final_answer
    content = strip_fences(raw)
    disc = discipline_score(raw)

    lower = content.lower()
    has_thead = "<thead" in lower and "</thead>" in lower
    has_tbody = "<tbody" in lower and "</tbody>" in lower
    has_th = "<th" in lower
    has_td = "<td" in lower
    has_caption = "<caption" in lower and "2025" in content

    # Data checks
    has_q1 = "Q1" in content or "q1" in lower
    has_q4 = "Q4" in content or "q4" in lower
    has_revenue = "$1.2M" in content or "1.2M" in content
    has_growth = "+63.6%" in content or "63.6" in content

    parse_score = 2 if (has_thead and has_tbody and has_th and has_td) else (
        1 if (has_th or has_td) else 0
    )
    checks = [has_thead, has_tbody, has_caption, has_q1, has_q4, has_revenue, has_growth]
    passed = sum(checks)
    correctness = 2 if passed >= 6 else (1 if passed >= 3 else 0)
    return make_eval(AxisScores(parse_score, correctness, disc), f"HTML table {passed}/7 checks.")


def evaluate_so13(state: ScenarioState) -> ScenarioEvaluation:
    """SO-13: JSON with Tricky Values"""
    raw = state.final_answer
    content = strip_fences(raw)
    disc = discipline_score(raw)

    try:
        obj = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return make_eval(AxisScores(0, 0, disc), "Invalid JSON.")

    if not isinstance(obj, dict):
        return make_eval(AxisScores(1, 0, disc), "Parsed but not an object.")

    checks = [
        obj.get("empty_string") == "",
        obj.get("null_value") is None,
        obj.get("zero") == 0 and obj.get("zero") is not False,
        obj.get("false_value") is False,
        obj.get("empty_array") == [],
        obj.get("empty_object") == {},
        isinstance(obj.get("special_chars"), str) and "\\" in obj["special_chars"] and '"' in obj["special_chars"],
        isinstance(obj.get("nested_null"), dict)
        and obj["nested_null"].get("a") is None
        and isinstance(obj["nested_null"].get("b"), list)
        and None in obj["nested_null"].get("b", []),
    ]
    passed = sum(checks)
    correctness = 2 if passed >= 7 else (1 if passed >= 4 else 0)
    return make_eval(AxisScores(2, correctness, disc), f"Tricky JSON {passed}/8 checks passed.")


def evaluate_so14(state: ScenarioState) -> ScenarioEvaluation:
    """SO-14: CSV Adversarial"""
    raw = state.final_answer
    content = strip_fences(raw)
    disc = discipline_score(raw)

    try:
        reader = csv.reader(io.StringIO(content))
        rows = list(reader)
    except Exception:
        return make_eval(AxisScores(0, 0, disc), "Could not parse CSV.")

    # Expect header + 2 data rows
    data_rows = rows[1:] if len(rows) > 1 else rows
    data_rows = [r for r in data_rows if any(c.strip() for c in r)]

    parse_score = 2 if len(data_rows) >= 2 else (1 if len(data_rows) >= 1 else 0)

    all_text = " ".join(" ".join(r) for r in rows)
    has_hello_world = "hello, world" in all_text.lower() or "hello,world" in all_text.lower()
    has_sum = "SUM" in all_text or "sum" in all_text
    has_empty_notes = False
    if data_rows and len(data_rows) >= 2:
        last_row = data_rows[-1]
        if len(last_row) >= 4 and last_row[-1].strip() == "":
            has_empty_notes = True

    checks = [len(data_rows) >= 2, has_hello_world, has_sum, has_empty_notes]
    passed = sum(checks)
    correctness = 2 if passed >= 3 else (1 if passed >= 2 else 0)
    return make_eval(AxisScores(parse_score, correctness, disc), f"Adversarial CSV {passed}/4 checks.")


def evaluate_so15(state: ScenarioState) -> ScenarioEvaluation:
    """SO-15: BSON Boundary Check"""
    raw = state.final_answer
    disc = discipline_score(raw)

    lower = raw.lower()
    explains_binary = any(
        phrase in lower
        for phrase in ("binary", "cannot be represented as text", "not a text format",
                       "binary format", "cannot generate bson", "can't produce bson",
                       "not possible to generate bson as text", "hex", "byte")
    )
    offers_alternative = any(
        phrase in lower
        for phrase in ("json", "alternative", "instead", "here is", "here's",
                       "python", "code", "bson library")
    )

    if explains_binary and offers_alternative:
        # For this scenario: parseable=2 (correctly identified), correctness=2, discipline special
        # discipline: 0 is fine since explanation IS the expected output
        return make_eval(
            AxisScores(2, 2, 2),
            "Correctly explained BSON is binary and offered alternative.",
        )
    if explains_binary:
        return make_eval(
            AxisScores(2, 1, 2),
            "Explained binary nature but didn't offer clear alternative.",
        )
    # Check if model just output JSON pretending it's BSON
    try:
        obj = json.loads(strip_fences(raw))
        if isinstance(obj, dict) and "name" in obj:
            return make_eval(
                AxisScores(0, 0, 0),
                "Output JSON but mislabeled or pretended it was BSON.",
            )
    except Exception:
        pass
    return make_eval(AxisScores(0, 0, 1), "Did not recognize BSON as a binary format.")


# ---------------------------------------------------------------------------
# Scenario Definitions
# ---------------------------------------------------------------------------

SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "SO-01",
        "title": "Simple JSON Object",
        "category": "A",
        "description": "Baseline JSON output with exact scalar types.",
        "user_message": 'Generate a JSON object for a book with these details: title "The Great Gatsby", author "F. Scott Fitzgerald", year 1925, genre "Novel", in_print true.',
        "successCase": "Emits a valid JSON object with all requested fields and types, without wrappers.",
        "failureCase": "Uses the wrong types, invalid JSON syntax, or extra prose.",
        "evaluate": evaluate_so01,
    },
    {
        "id": "SO-02",
        "title": "CSV with Headers",
        "category": "A",
        "description": "Basic CSV generation with exact headers and row values.",
        "user_message": "Generate a CSV file with headers: name, age, city, email. Include these 3 records:\n- Alice Johnson, 32, Portland, alice@example.com\n- Bob Smith, 45, Chicago, bob@example.com\n- Carol White, 28, Austin, carol@example.com",
        "successCase": "Produces standard comma-separated CSV with the requested columns and rows only.",
        "failureCase": "Changes delimiters, drops rows, or adds unrequested columns.",
        "evaluate": evaluate_so02,
    },
    {
        "id": "SO-03",
        "title": "YAML Configuration",
        "category": "A",
        "description": "Nested YAML with lists, booleans, and integer fields.",
        "user_message": 'Generate a YAML configuration for a web server with these settings: host "0.0.0.0", port 8080, debug false, allowed_origins is a list containing "https://example.com" and "https://app.example.com", and database has nested fields: host "localhost", port 5432, name "myapp_db".',
        "successCase": "Produces parseable YAML with the correct nesting and types.",
        "failureCase": "Breaks indentation, types, or the requested structure.",
        "evaluate": evaluate_so03,
    },
    {
        "id": "SO-04",
        "title": "TOML Configuration",
        "category": "B",
        "description": "TOML package metadata plus dependency declarations with features.",
        "user_message": 'Generate a TOML configuration file for a Rust project with: package name "my_cli", version "0.1.0", edition "2021", authors list with "Alice <alice@example.com>". Dependencies section: serde version "1.0" with features ["derive"], clap version "4.5".',
        "successCase": "Uses TOML section syntax and correct dependency shapes.",
        "failureCase": "Falls back to JSON/YAML syntax or breaks the TOML tables.",
        "evaluate": evaluate_so04,
    },
    {
        "id": "SO-05",
        "title": "SQL CREATE + INSERT",
        "category": "B",
        "description": "SQLite DDL plus DML for a single table and two rows.",
        "user_message": 'Generate SQL to create a table called "employees" with columns: id (integer, primary key, auto increment), name (varchar 100, not null), department (varchar 50), salary (decimal 10,2), hire_date (date). Then insert 2 rows: "Alice Chen", "Engineering", 95000.00, "2023-06-15" and "Bob Park", "Marketing", 78500.50, "2024-01-10".',
        "successCase": "Executes cleanly in SQLite and creates the requested table plus the two rows.",
        "failureCase": "Uses non-SQLite syntax, fails to insert both rows, or injects explicit ids.",
        "evaluate": evaluate_so05,
    },
    {
        "id": "SO-06",
        "title": "ICS Calendar Event",
        "category": "B",
        "description": "A minimally conformant calendar event with the right instant and duration.",
        "user_message": 'Generate an ICS calendar event for a team meeting on April 15, 2026, from 2:00 PM to 3:30 PM Eastern Time. Title: "Q2 Planning Session". Location: "Conference Room B". Description: "Quarterly planning meeting - bring your project updates". Organizer: alice@company.com.',
        "successCase": "Builds a valid VCALENDAR/VEVENT wrapper with the required properties.",
        "failureCase": "Omits required calendar blocks or gets the event time wrong.",
        "evaluate": evaluate_so06,
    },
    {
        "id": "SO-07",
        "title": "Nested JSON with Arrays and Nulls",
        "category": "C",
        "description": "Deeply nested JSON with arrays of objects and required null values.",
        "user_message": 'Generate a JSON object representing an API response for a user profile. The user has: id 42, username "j_doe", email null (not verified yet), roles array with "editor" and "viewer", address object with street "123 Main St", city "Springfield", state "IL", zip "62704", and phone_numbers array containing two objects: one with type "mobile", number "+1-555-0123", primary true, and one with type "work", number null, primary false. Include a metadata object with last_login "2026-03-15T10:30:00Z" and login_count 847.',
        "successCase": "Keeps the full nested structure and preserves types, especially nulls and strings.",
        "failureCase": "Turns nulls into strings, changes scalar types, or drops nested keys.",
        "evaluate": evaluate_so07,
    },
    {
        "id": "SO-08",
        "title": "CSV with Special Characters",
        "category": "C",
        "description": "RFC 4180 CSV escaping with commas, quotes, apostrophes, and Unicode.",
        "user_message": "Generate a CSV with headers: company, description, revenue, ceo. Include these 3 records:\n- Acme, Inc., \"Makes everything, from anvils to rockets\", $1.2B, Jane \"JJ\" Smith\n- O'Brien & Sons, \"Family-owned since 1952\", $45M, Patrick O'Brien\n- \u682a\u5f0f\u4f1a\u793e\u30c6\u30b9\u30c8 (Test Corp), \"Japanese tech company\", \u00a5500B, \u7530\u4e2d\u592a\u90ce",
        "successCase": "Quotes and escapes fields correctly while preserving Unicode.",
        "failureCase": "Breaks the CSV parser on commas or embedded quotes.",
        "evaluate": evaluate_so08,
    },
    {
        "id": "SO-09",
        "title": "XML with Namespaces and Attributes",
        "category": "C",
        "description": "XML declaration, default namespace, element attributes, and nested elements.",
        "user_message": 'Generate an XML document representing a book catalog. The root element is "catalog" with namespace "http://example.com/books" and attribute version="2.0". Include 2 books, each with attributes id and lang. Book 1: id="bk101", lang="en", title "Rust Programming", author "Steve Klabnik", price 39.99 with currency attribute "USD". Book 2: id="bk102", lang="ja", title "\u30d7\u30ed\u30b0\u30e9\u30df\u30f3\u30b0Rust", author "Steve Klabnik", price 4500 with currency attribute "JPY". Include an XML declaration.',
        "successCase": "Produces parseable XML with the right namespace, attributes, and book payloads.",
        "failureCase": "Drops the declaration, namespace, or required attributes.",
        "evaluate": evaluate_so09,
    },
    {
        "id": "SO-10",
        "title": "JSON to Markdown Table",
        "category": "D",
        "description": "Format conversion from structured JSON into a deterministic Markdown table.",
        "user_message": 'Convert this JSON array into a Markdown table:\n[{"name": "Alice", "score": 95, "grade": "A"}, {"name": "Bob", "score": 82, "grade": "B+"}, {"name": "Carol", "score": 78, "grade": "C+"}, {"name": "Dave", "score": 91, "grade": "A-"}]',
        "successCase": "Outputs a well-formed Markdown table with the requested rows and columns.",
        "failureCase": "Misses the table shape or mutates the provided values.",
        "evaluate": evaluate_so10,
    },
    {
        "id": "SO-11",
        "title": "Natural Language to Mermaid Diagram",
        "category": "D",
        "description": "Flowchart conversion with branching control flow and diagram syntax.",
        "user_message": "Generate a Mermaid flowchart for this process: User submits a form. System validates the input. If valid, save to database and send confirmation email, then show success page. If invalid, show error message and return to form.",
        "successCase": "Creates a Mermaid flowchart with the required valid and invalid branches.",
        "failureCase": "Skips a branch or emits prose instead of diagram syntax.",
        "evaluate": evaluate_so11,
    },
    {
        "id": "SO-12",
        "title": "HTML Table with Semantic Markup",
        "category": "D",
        "description": "Semantic HTML table output with caption, thead, tbody, th, and td usage.",
        "user_message": 'Generate an HTML table showing quarterly revenue. Headers: Quarter, Revenue, Growth. Data: Q1 $1.2M +5%, Q2 $1.4M +16.7%, Q3 $1.1M -21.4%, Q4 $1.8M +63.6%. Use thead, tbody, and th elements properly. Add a caption "2025 Quarterly Revenue".',
        "successCase": "Produces semantic table markup with the caption and all four rows.",
        "failureCase": "Uses plain text, misses the semantic wrappers, or drops cells.",
        "evaluate": evaluate_so12,
    },
    {
        "id": "SO-13",
        "title": "JSON with Tricky Values",
        "category": "E",
        "description": "JSON escaping with empty values, booleans, nulls, nested nulls, and control characters.",
        "user_message": "Generate a JSON object with these exact key-value pairs: empty_string should be \"\" (an empty string), null_value should be null, zero should be 0, false_value should be false, empty_array should be [], empty_object should be {}, special_chars should be the string containing a backslash, a double quote, a newline, and a tab: `\\\\\\\"\\n\\t`, nested_null should be an object with key \"a\" set to null and key \"b\" set to an array containing null and 1.",
        "successCase": "Emits valid JSON with the exact control characters and nested null structure.",
        "failureCase": "Mis-escapes the control characters or coerces the empty values into the wrong types.",
        "evaluate": evaluate_so13,
    },
    {
        "id": "SO-14",
        "title": "CSV Adversarial \u2014 The Delimiter Minefield",
        "category": "E",
        "description": "Adversarial CSV with embedded commas, quotes, formulas, real newlines, and an empty trailing field.",
        "user_message": 'Generate a CSV with headers: id, description, formula, notes. Include these 2 records:\nRow 1: id 1, description is: He said "hello, world" and left, formula is: =SUM(A1,B1), notes is: Line one\\nLine two (actual newline in the value)\nRow 2: id 2, description is: Simple value, formula is: =IF(A1>0,"yes","no"), notes is empty',
        "successCase": "Preserves the quoted newline and quote-doubling so the CSV still parses into two rows.",
        "failureCase": "Breaks row alignment or uses the wrong escaping convention.",
        "evaluate": evaluate_so14,
    },
    {
        "id": "SO-15",
        "title": "Binary Format Boundary \u2014 BSON",
        "category": "S",
        "description": "Supplemental boundary-awareness check for a non-text binary format.",
        "user_message": 'Generate a BSON document with fields name "Alice" and age 30.',
        "successCase": "Explains BSON is binary and offers a practical text-friendly alternative such as JSON or code.",
        "failureCase": "Hallucinates a textual BSON format or mislabels plain JSON as BSON.",
        "evaluate": evaluate_so15,
    },
]

SCENARIO_DISPLAY_DETAILS: dict[str, dict[str, str]] = {
    s["id"]: {"successCase": s["successCase"], "failureCase": s["failureCase"]}
    for s in SCENARIOS
}

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_model_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute weighted final score from scenario results."""
    categories = ["A", "B", "C", "D", "E", "S"]
    category_scores = []

    for category in categories:
        cat_results = [
            r for r in results
            if any(s["id"] == r["scenarioId"] and s["category"] == category for s in SCENARIOS)
        ]
        avg_score = (
            round(sum(r["score"] for r in cat_results) / len(cat_results))
            if cat_results
            else 0
        )
        category_scores.append(
            {
                "category": category,
                "label": CATEGORY_LABELS[category],
                "weight": CATEGORY_WEIGHTS[category],
                "averageScore": avg_score,
                "percent": avg_score,
            }
        )

    core = [cs for cs in category_scores if cs["category"] != "S"]
    final_score = round(
        sum(cs["averageScore"] * (cs["weight"] / 100) for cs in core)
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
) -> str:
    """Call an OpenAI-compatible chat/completions endpoint. Returns text content."""
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

    return normalize_content(message.get("content", ""))


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
        f"prompt={scenario['user_message'][:80]}...",
        "",
        *trace_lines,
        "",
        f"verdict={evaluation['status']}",
        f"score={evaluation['score']}",
        f"summary={evaluation['summary']}",
    ]
    if evaluation.get("note"):
        lines.append(f"note={evaluation['note']}")
    return "\n".join(lines)


def run_scenario_for_model(
    model: ModelConfig,
    scenario: dict[str, Any],
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a single scenario against a model and return the result."""
    state = ScenarioState()
    messages = create_initial_messages(scenario["user_message"])
    trace_lines: list[str] = []
    params = params or {}

    try:
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

        state.assistant_messages.append(response)
        state.final_answer = response
        trace_lines.append(f"response={response[:200]}...")

    except Exception as exc:
        summary = str(exc)
        trace_lines.append(f"error={summary}")
        eval_info = {"status": "fail", "score": 0, "summary": summary}
        return {
            "scenarioId": scenario["id"],
            "status": "fail",
            "score": 0,
            "summary": summary,
            "rawLog": format_scenario_trace(model, scenario, eval_info, trace_lines),
        }

    evaluation: ScenarioEvaluation = scenario["evaluate"](state)

    eval_dict = {
        "status": evaluation.status,
        "score": evaluation.score,
        "summary": evaluation.summary,
        "note": evaluation.note,
    }

    return {
        "scenarioId": scenario["id"],
        "status": evaluation.status,
        "score": evaluation.score,
        "summary": evaluation.summary,
        "note": evaluation.note,
        "axes": {
            "parseable": evaluation.axes.parseable,
            "correctness": evaluation.axes.correctness,
            "discipline": evaluation.axes.discipline,
        },
        "rawLog": format_scenario_trace(model, scenario, eval_dict, trace_lines),
    }


# ---------------------------------------------------------------------------
# Model Config (shared with toolcall15.py)
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
    model_name = trimmed[sep_index + 1:].strip()

    if provider_raw not in VALID_PROVIDERS:
        raise ValueError(
            f'{env_name} entry {index + 1} has unsupported provider "{provider_raw}". '
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
        description="StructOutput-15 Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        dest="scenario_ids",
        help="Run a specific scenario (e.g. SO-01). Can be repeated.",
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

    # Merge scenario IDs
    scenario_ids = list(args.scenario_ids)
    if args.scenario_list:
        scenario_ids.extend(s.strip() for s in args.scenario_list.split(",") if s.strip())
    scenario_ids = list(dict.fromkeys(scenario_ids))

    # Merge model IDs
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
            if result.get("axes"):
                entry["axes"] = result["axes"]
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
            print(f"  score: {result['score']}")
            if result.get("axes"):
                ax = result["axes"]
                print(f"  axes: parseable={ax['parseable']} correctness={ax['correctness']} discipline={ax['discipline']}")
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
            f"({summary['totalScore']}/{summary['maxScore']}) "
            f"{summary['rating']}"
        )


if __name__ == "__main__":
    main()
