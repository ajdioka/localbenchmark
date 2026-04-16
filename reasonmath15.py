#!/usr/bin/env python3
"""
ReasonMath-15 Benchmark - Python Port

A faithful port of the ReasonMath-15 benchmark from TypeScript to Python.
Evaluates LLM reasoning ability across 15 scenarios in 5 categories.
Single-turn, pure text completion (no tool calling).

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
    "You are a helpful assistant that solves practical reasoning problems.\n"
    "\n"
    "Rules:\n"
    "- Show concise visible work using equations, short bullet points, or numbered steps.\n"
    "- Do not write a long essay.\n"
    '- End with exactly one line that starts with "ANSWER: ".\n'
    "- If the question asks for more than one value, format the final line as semicolon-separated key=value pairs.\n"
    "- Use exact arithmetic when possible.\n"
    "- Round only the final result when the problem context requires it.\n"
    "- If the constraints are inconsistent, say so explicitly in the final answer."
)

DEFAULT_MODEL_REQUEST_TIMEOUT_SECONDS = 30
MAX_PROVIDER_ERROR_ATTEMPTS = 3
PROVIDER_ERROR_RETRY_PATTERN = re.compile(r"provider returned error", re.IGNORECASE)

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
class ScenarioEvaluation:
    status: Literal["pass", "partial", "fail"]
    score: int
    summary: str
    note: str | None = None


@dataclass
class ScenarioSpec:
    id: str
    title: str
    category: str
    description: str
    user_message: str
    success_case: str
    failure_case: str
    canonical_answer: str
    accepted_answers: list[str] = field(default_factory=list)
    partial_answers: list[str] = field(default_factory=list)
    checkpoints: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Evaluation Helpers
# ---------------------------------------------------------------------------


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip()).lower()


def normalize_checkpoint_label(value: str) -> str:
    return normalize_text(value.replace("_", " "))


def extract_answer_line(answer: str) -> str:
    lines = answer.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    matches = [line for line in lines if line.startswith("ANSWER: ")]
    return matches[-1] if matches else ""


def answer_payload(answer_line: str) -> str:
    return re.sub(r"^ANSWER:\s*", "", answer_line, flags=re.IGNORECASE).strip()


def try_single_value_match(canonical_answer: str, answer_line: str) -> bool:
    canonical_pl = answer_payload(canonical_answer)
    answer_pl = answer_payload(answer_line)

    if ";" in canonical_pl or "=" not in canonical_pl:
        return False

    sep_index = canonical_pl.index("=")
    expected_key = normalize_checkpoint_label(canonical_pl[:sep_index])
    expected_value = normalize_text(canonical_pl[sep_index + 1:])
    actual_normalized = normalize_text(answer_pl)

    if actual_normalized == expected_value:
        return True

    actual_without_label = re.sub(r"^[a-z_][a-z0-9_ ]*=\s*", "", actual_normalized, flags=re.IGNORECASE).strip()
    if actual_without_label == expected_value:
        return True

    return expected_key in actual_normalized and expected_value in actual_normalized


def evaluate_answer_axis(spec: ScenarioSpec, raw_answer: str) -> dict[str, Any]:
    answer_line = extract_answer_line(raw_answer)
    if not answer_line:
        return {"points": 0, "note": 'Missing final "ANSWER: " line.'}

    normalized = normalize_text(answer_line)
    canonical = normalize_text(spec.canonical_answer)
    accepted = [normalize_text(a) for a in spec.accepted_answers]
    partial = [normalize_text(p) for p in spec.partial_answers]

    if normalized == canonical or normalized in accepted:
        return {"points": 2}

    if try_single_value_match(spec.canonical_answer, answer_line):
        return {"points": 2}

    if normalized in partial:
        return {"points": 1, "note": "Matched a scenario-defined partial answer."}

    return {"points": 0, "note": f"Unexpected final line: {answer_line}"}


def evaluate_trace_axis(spec: ScenarioSpec, raw_answer: str) -> dict[str, Any]:
    normalized = normalize_text(raw_answer)
    matched = []
    for checkpoint in spec.checkpoints:
        normalized_cp = normalize_text(checkpoint)
        if normalized_cp in normalized:
            matched.append(checkpoint)
            continue

        sep_index = checkpoint.find("=")
        if sep_index == -1:
            continue

        left = normalize_checkpoint_label(checkpoint[:sep_index])
        right = normalize_text(checkpoint[sep_index + 1:])
        if left in normalized and right in normalized:
            matched.append(checkpoint)

    if len(matched) == len(spec.checkpoints):
        return {"points": 2}

    if len(matched) > 0:
        return {"points": 1, "note": f"Matched {len(matched)}/{len(spec.checkpoints)} checkpoints."}

    return {"points": 0, "note": "No published checkpoints matched."}


def score_scenario(answer_points: int, trace_points: int) -> int:
    return round(100 * (0.7 * (answer_points / 2) + 0.3 * (trace_points / 2)))


def status_for_score(score: int) -> Literal["pass", "partial", "fail"]:
    if score >= 85:
        return "pass"
    if score >= 60:
        return "partial"
    return "fail"


# ---------------------------------------------------------------------------
# Scenario Definitions
# ---------------------------------------------------------------------------

SCENARIO_SPECS: list[ScenarioSpec] = [
    ScenarioSpec(
        id="RM-01",
        title="Bill Splitting with Tax and Tip",
        category="A",
        description="Multi-step percentage arithmetic with an explicit tip base and a per-person total.",
        user_message="Three friends had dinner. The food total was $84.00 before tax. Tax is 8.5%. They want to leave a 20% tip calculated on the pre-tax amount. How much does each person owe in total?",
        success_case="Uses the right tax and tip bases, then ends with the published per-person answer line.",
        failure_case="Calculates tip on the wrong base or omits the required final answer line.",
        canonical_answer="ANSWER: per_person=$35.98",
        accepted_answers=["ANSWER: $35.98", "ANSWER: 35.98", "ANSWER: per_person=35.98"],
        checkpoints=["7.14", "16.80", "107.94", "35.98"],
    ),
    ScenarioSpec(
        id="RM-02",
        title="Unit Conversion Chain",
        category="A",
        description="Two-step conversion from cups to grams to kilograms with a published partial-credit endpoint.",
        user_message="A recipe calls for 2.5 cups of flour. I only have a kitchen scale. If 1 cup of flour weighs approximately 125 grams, how many kilograms of flour do I need?",
        success_case="Converts to kilograms and ends with the required answer format.",
        failure_case="Stops at grams or omits the answer line entirely.",
        canonical_answer="ANSWER: kg=0.3125",
        accepted_answers=["ANSWER: kg=0.313"],
        partial_answers=["ANSWER: grams=312.5"],
        checkpoints=["grams=312.5", "kg=0.3125"],
    ),
    ScenarioSpec(
        id="RM-03",
        title="Percentage Change \u2014 Not What You Think",
        category="A",
        description="A percentage-increase trap where decrease and increase are not symmetric.",
        user_message=(
            "A shirt was originally $80. It went on sale for 25% off. After you bought it on sale, the store raised the original price by 25%. What is the new original price, and did you save money compared to the new price?\n\n"
            "Return the final line exactly in this format:\n"
            "ANSWER: new_original_price=<money>; saved_money=<yes|no>"
        ),
        success_case="Separates the sale price from the later new original price and reports both requested outputs.",
        failure_case="Claims the price simply returns to $80 or misses one of the requested values.",
        canonical_answer="ANSWER: new_original_price=$100.00; saved_money=yes",
        accepted_answers=[
            "ANSWER: new_original_price=$100; saved_money=yes",
            "ANSWER: new_original_price=100; saved_money=yes",
        ],
        checkpoints=["sale_price=60", "new_original_price=100", "saved_money=yes"],
    ),
    ScenarioSpec(
        id="RM-04",
        title="Constraint Consistency Check",
        category="B",
        description="An intentionally unsatisfiable logic puzzle that should be rejected cleanly.",
        user_message=(
            "There are 5 houses in a row. Each house is painted a different color: red, blue, green, yellow, white. Given these clues:\n"
            "1. The red house is immediately to the left of the blue house.\n"
            "2. The green house is in the middle (position 3).\n"
            "3. The yellow house is not next to the green house.\n"
            "4. The white house is at one of the ends (position 1 or 5).\n\n"
            "What is the order of the houses from left to right?"
        ),
        success_case="Detects that the clues are inconsistent and ends with `ANSWER: status=unsat` or an accepted equivalent.",
        failure_case="Forces an invalid arrangement instead of reporting the contradiction.",
        canonical_answer="ANSWER: status=unsat",
        accepted_answers=[
            "ANSWER: status=no valid arrangement",
            "ANSWER: status=no solution",
            "ANSWER: the constraints are inconsistent.",
            "ANSWER: constraints are inconsistent",
            "ANSWER: inconsistent",
        ],
        checkpoints=["green at position 3", "yellow must be at position 1 or 5", "white is at an end", "red and blue cannot be placed"],
    ),
    ScenarioSpec(
        id="RM-05",
        title="Scheduling Conflict Resolution",
        category="B",
        description="Scheduling arithmetic with buffers, a fixed lunch block, and a max-fit conclusion.",
        user_message=(
            "I need to schedule 4 meetings today. Each meeting is 45 minutes long with 15 minutes of buffer between meetings. My available time starts at 9:00 AM and ends at 1:00 PM. I also have a fixed lunch break from 12:00 PM to 12:30 PM that cannot be moved.\n\n"
            "Can I fit all 4 meetings? If yes, what are the time slots? If no, how many can I fit?\n\n"
            "Return the final line exactly in this format:\n"
            "ANSWER: fit=<yes|no>; max_meetings=<integer>\n\n"
            "Do not include the slots in the final ANSWER line."
        ),
        success_case="Counts the usable minutes correctly and reports that only three meetings fit.",
        failure_case="Ignores buffers or lunch and overstates capacity.",
        canonical_answer="ANSWER: fit=no; max_meetings=3",
        accepted_answers=[
            "ANSWER: fit_all_4=no; max_meetings=3",
            "ANSWER: can_fit_4=no; max_meetings=3",
            "ANSWER: canfit4=no; maxmeetings=3",
            "ANSWER: fit_all=no; max_meetings=3",
            "ANSWER: all_fit=no; max_meetings=3",
            "ANSWER: can_fit_4=false; max_meetings=3",
            "ANSWER: fit_all=false; max_meetings=3",
            "ANSWER: all_fit=false; max_meetings=3",
            "ANSWER: canfit=no; maxmeetings=3",
            "ANSWER: fit_4=false; max_meetings=3",
            "ANSWER: fit_4=no; max_meetings=3",
            "ANSWER: canfit=false; maxmeetings=3",
        ],
        checkpoints=["240", "30", "210", "225", "k = 3"],
    ),
    ScenarioSpec(
        id="RM-06",
        title="Monty Hall Variant with Explicit Host Rule",
        category="B",
        description="Conditional probability with a nonstandard four-door host rule.",
        user_message=(
            "You're on a game show with 4 doors. Behind one door is a car. Behind the other 3 doors are goats. You pick Door 1.\n\n"
            "The host knows where the car is. The host always opens exactly 2 goat doors among the 3 doors you did not choose, leaving exactly 1 unopened alternative door besides your original choice. If the host has more than one valid pair of goat doors to open, the host chooses uniformly at random among those valid pairs.\n\n"
            "In this run, the host opens Door 3 and Door 4, both goats, and offers you the chance to switch to Door 2.\n\n"
            "What is the probability of winning the car if you switch to Door 2? What if you stay with Door 1?"
        ),
        success_case="Applies the stated host rule and reports the 75%/25% split.",
        failure_case="Treats the host choice as random noise or collapses to the wrong probability.",
        canonical_answer="ANSWER: switch=75%; stay=25%",
        accepted_answers=["ANSWER: switch=3/4; stay=1/4"],
        checkpoints=["p(c_1) = 1/4", "p(h | c_2) = 1", "p(h) = 1/3", "p(c_2 | h) = 3/4"],
    ),
    ScenarioSpec(
        id="RM-07",
        title="Speed, Distance, Time with a Twist",
        category="C",
        description="Round-trip average speed where arithmetic must be done over total distance and total time.",
        user_message=(
            "Alice drives from City A to City B at 60 km/h. The trip takes 3 hours. She then drives back from City B to City A, but hits traffic and averages only 40 km/h on the return.\n\n"
            "What is her average speed for the entire round trip?"
        ),
        success_case="Computes both trip legs and reports the combined average speed.",
        failure_case="Averages 60 and 40 directly instead of using total distance over total time.",
        canonical_answer="ANSWER: avg_speed=48 km/h",
        checkpoints=["distance_one_way=180", "return_time=4.5", "total_distance=360", "total_time=7.5", "avg_speed=48"],
    ),
    ScenarioSpec(
        id="RM-08",
        title="Rate and Proportion Problem",
        category="C",
        description="Combines positive fill rates and a negative drain rate into one net-fill calculation.",
        user_message=(
            "A bathtub has two faucets. Faucet A alone fills the tub in 12 minutes. Faucet B alone fills it in 18 minutes. There's also a drain that empties the full tub in 36 minutes. If both faucets are open and the drain is open, how long does it take to fill the tub?\n\n"
            "Return the final line exactly in this format:\n"
            "ANSWER: fill_time=<number> minutes"
        ),
        success_case="Adds the rates correctly and inverts the combined rate for the time.",
        failure_case="Adds or subtracts times directly instead of rates.",
        canonical_answer="ANSWER: fill_time=9 minutes",
        checkpoints=["faucet a=1/12", "faucet b=1/18", "drain=-1/36", "net rate=1/9", "fill_time=9"],
    ),
    ScenarioSpec(
        id="RM-09",
        title="Age Problem with Temporal Reasoning",
        category="C",
        description="Temporal algebra with two age relationships anchored at different times.",
        user_message="Five years ago, Maria was 3 times as old as her son. In 5 years from now, Maria will be twice as old as her son. How old is Maria now?",
        success_case="Sets up the two time-shifted equations and solves them consistently.",
        failure_case="Confuses the time offsets or solves only one equation.",
        canonical_answer="ANSWER: maria=35",
        checkpoints=["m-5=3(s-5)", "m+5=2(s+5)", "s=15", "m=35"],
    ),
    ScenarioSpec(
        id="RM-10",
        title="The Classic Bat and Ball \u2014 Extended",
        category="D",
        description="The classic price trap with one more dependent step added for the glove.",
        user_message="A bat and a ball cost $1.10 together. The bat costs $1.00 more than the ball. A glove costs twice as much as the bat. How much does the glove cost?",
        success_case="Solves the original bat-and-ball relation correctly before doubling the bat price.",
        failure_case="Falls for the $0.10 ball trap and propagates the wrong price.",
        canonical_answer="ANSWER: glove=$2.10",
        checkpoints=["ball=0.05", "bat=1.05", "glove=2.10"],
    ),
    ScenarioSpec(
        id="RM-11",
        title="The Lily Pad Problem \u2014 Backward Reasoning",
        category="D",
        description="A classic backward-reasoning check on doubling growth.",
        user_message=(
            "A lake has lily pads growing on it. The area covered by lily pads doubles every day. On day 30, the entire lake is covered. On what day was the lake half covered?\n\n"
            "Return the final line exactly in this format:\n"
            "ANSWER: day=<integer>"
        ),
        success_case="Reasons backward one day from the fully covered state.",
        failure_case="Treats the growth as linear instead of doubling.",
        canonical_answer="ANSWER: day=29",
        checkpoints=["doubles every day", "day 30=full", "day 29=half"],
    ),
    ScenarioSpec(
        id="RM-12",
        title="Family-Relation Riddle",
        category="D",
        description="Relation-logic parsing with a self-reference trap.",
        user_message=(
            "A man points to a photograph and says, \"Brothers and sisters I have none, but that man's father is my father's son.\" Who is in the photograph?\n\n"
            "Use the relationship from the speaker's point of view.\n"
            "Return the final line exactly in this format:\n"
            "ANSWER: person=<relationship>"
        ),
        success_case="Resolves the self-reference and ends with his son as the answer.",
        failure_case="Misreads `my father's son` as someone other than the speaker.",
        canonical_answer="ANSWER: person=his son",
        accepted_answers=["ANSWER: person=my son", "ANSWER: person=the speaker's son", "ANSWER: person=son"],
        checkpoints=["my father's son=me", "that man's father=me", "person=son"],
    ),
    ScenarioSpec(
        id="RM-13",
        title="Compound Interest Calculation",
        category="E",
        description="Practical finance math with monthly compounding and a required total-interest output.",
        user_message=(
            "I invest $5,000 in a savings account that earns 4.5% annual interest, compounded monthly. How much will I have after 3 years? What is the total interest earned?\n\n"
            "Return the final line exactly in this format:\n"
            "ANSWER: amount=<money>; interest=<money>"
        ),
        success_case="Uses the compound-interest formula and reports both the balance and interest earned.",
        failure_case="Uses simple interest or omits one of the requested outputs.",
        canonical_answer="ANSWER: amount=$5721.24; interest=$721.24",
        accepted_answers=[
            "ANSWER: amount=5721.24; interest=721.24",
            "ANSWER: amount=$5,721.24; interest=$721.24",
            "ANSWER: amount=5,721.24; interest=721.24",
        ],
        checkpoints=["a=p(1+r/n)^(nt)", "p=5000", "r=0.045", "n=12", "t=3", "amount=5721.24", "interest=721.24"],
    ),
    ScenarioSpec(
        id="RM-14",
        title="Conversion with Multiple Systems",
        category="E",
        description="Temperature and time conversion across metric, imperial, and convection-oven adjustments.",
        user_message=(
            "A European recipe says to bake at 180\u00b0C for 45 minutes in a regular oven. I have a convection oven and I'm in the US. What temperature should I set in Fahrenheit, and for how long? Convection ovens should be set 25\u00b0F lower than the regular-oven equivalent, and baking time should be reduced by about 25%.\n\n"
            "Round the baking time to the nearest whole minute.\n\n"
            "Return the final line exactly in this format:\n"
            "ANSWER: temp_f=<integer>; time_min=<integer>"
        ),
        success_case="Converts to Fahrenheit, applies the convection adjustment, and shortens the baking time correctly.",
        failure_case="Misses one of the conversion steps or rounds the wrong value.",
        canonical_answer="ANSWER: temp_f=331; time_min=34",
        accepted_answers=["ANSWER: temp_f=330; time_min=34"],
        checkpoints=["180c=356f", "356-25=331", "45*0.75=33.75", "time=34"],
    ),
    ScenarioSpec(
        id="RM-15",
        title="Combinatorial Reasoning \u2014 PIN Possibilities",
        category="E",
        description="A counting problem that combines uniqueness, leading-digit, and strict-order constraints.",
        user_message=(
            "A website requires a 4-digit PIN, where each digit is 0\u20139. How many possible PINs are there if:\n"
            "1. All 4 digits must be different\n"
            "2. The PIN must start with a non-zero digit\n"
            "3. The digits must be in strictly increasing order\n\n"
            "Return the final line exactly in this format:\n"
            "ANSWER: count=<integer>"
        ),
        success_case="Recognizes that strict ordering collapses arrangements to combinations and excludes zero correctly.",
        failure_case="Counts permutations instead of the valid ordered digit sets.",
        canonical_answer="ANSWER: count=126",
        checkpoints=["one arrangement per set", "0 cannot be included", "choose 4 digits from 9", "126"],
    ),
]

CATEGORY_LABELS: dict[str, str] = {
    "A": "Everyday Arithmetic",
    "B": "Logic Puzzles",
    "C": "Multi-Step Word Problems",
    "D": "Trick Questions and Traps",
    "E": "Applied Reasoning",
}

CATEGORY_WEIGHTS: dict[str, int] = {
    "A": 15,
    "B": 25,
    "C": 20,
    "D": 25,
    "E": 15,
}

SCENARIO_DISPLAY_DETAILS: dict[str, dict[str, str]] = {
    spec.id: {"successCase": spec.success_case, "failureCase": spec.failure_case}
    for spec in SCENARIO_SPECS
}


# ---------------------------------------------------------------------------
# Evaluate a scenario
# ---------------------------------------------------------------------------


def evaluate_scenario(spec: ScenarioSpec, state: ScenarioState) -> ScenarioEvaluation:
    answer_axis = evaluate_answer_axis(spec, state.final_answer)
    trace_axis = evaluate_trace_axis(spec, state.final_answer)
    score = score_scenario(answer_axis["points"], trace_axis["points"])
    notes = [answer_axis.get("note"), trace_axis.get("note")]
    note = " ".join(n for n in notes if n).strip() or None

    return ScenarioEvaluation(
        status=status_for_score(score),
        score=score,
        summary=f"Answer axis {answer_axis['points']}/2, trace axis {trace_axis['points']}/2 ({score}%).",
        note=note,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_model_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    categories = ["A", "B", "C", "D", "E"]
    category_scores = []

    for category in categories:
        cat_results = [
            r for r in results
            if any(s.id == r["scenarioId"] and s.category == category for s in SCENARIO_SPECS)
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
        "maxScore": len(SCENARIO_SPECS) * 100,
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
) -> str:
    """Call an OpenAI-compatible chat/completions endpoint. Returns assistant text."""
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
    spec: ScenarioSpec,
    evaluation: dict[str, Any],
    trace_lines: list[str],
) -> str:
    lines = [
        f"model={model.model}",
        f"scenario={spec.id} {spec.title}",
        f"prompt={spec.user_message}",
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
    spec: ScenarioSpec,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Execute a single scenario against a model and return the result."""
    state = ScenarioState()
    messages = create_initial_messages(spec.user_message)
    trace_lines: list[str] = ["assistant=starting"]
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
        trace_lines.append(f"assistant_response={response}")

    except Exception as exc:
        summary = str(exc)
        trace_lines.append(f"error={summary}")
        eval_info = {"status": "fail", "summary": summary}
        return {
            "scenarioId": spec.id,
            "status": "fail",
            "score": 0,
            "summary": summary,
            "rawLog": format_scenario_trace(model, spec, eval_info, trace_lines),
        }

    trace_lines.append(f"final_answer={state.final_answer}")

    evaluation = evaluate_scenario(spec, state)

    eval_dict = {
        "status": evaluation.status,
        "summary": evaluation.summary,
        "note": evaluation.note,
    }

    return {
        "scenarioId": spec.id,
        "status": evaluation.status,
        "score": evaluation.score,
        "summary": evaluation.summary,
        "note": evaluation.note,
        "rawLog": format_scenario_trace(model, spec, eval_dict, trace_lines),
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
        description="ReasonMath-15 Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        dest="scenario_ids",
        help="Run a specific scenario (e.g. RM-01). Can be repeated.",
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


def resolve_scenarios(requested_ids: list[str]) -> list[ScenarioSpec]:
    if not requested_ids:
        return SCENARIO_SPECS
    selected = [s for s in SCENARIO_SPECS if s.id in requested_ids]
    if len(selected) != len(set(requested_ids)):
        found_ids = {s.id for s in selected}
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
    scenario_ids = list(dict.fromkeys(scenario_ids))

    # Merge model IDs from --model and --models
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

    for scenario_index, spec in enumerate(scenarios):
        sys.stdout.write(
            f"\n[{scenario_index + 1}/{len(scenarios)}] {spec.id} {spec.title}\n"
        )
        sys.stdout.flush()

        scenario_summary: dict[str, Any] = {
            "scenarioId": spec.id,
            "title": spec.title,
            "results": [],
        }

        for model in models:
            sys.stdout.write(
                f"  {model.id} {spec.id}: Calling model\n"
            )
            sys.stdout.flush()

            result = run_scenario_for_model(model, spec, generation_params)
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
            if result.get("rawLog"):
                print("  raw:")
                for line in result["rawLog"].split("\n"):
                    print(f"    {line}")

    ran_full_suite = len(scenarios) == len(SCENARIO_SPECS)

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
