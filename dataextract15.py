#!/usr/bin/env python3
"""
DataExtract-15 Benchmark - Python Port

A faithful port of the DataExtract-15 benchmark from TypeScript to Python.
Evaluates LLM data extraction ability across 15 scenarios in 5 categories.

Dependencies: requests (pip install requests)
"""

from __future__ import annotations

import argparse
import json
import math
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
    "You are a data extraction assistant. The user will provide you with unstructured text and a list of fields to extract.\n"
    "\n"
    "Rules:\n"
    "- Extract ONLY information that is explicitly stated in the source text.\n"
    "- For string fields, copy the exact value from the source text. You may return an exact subspan when the requested field is only one component of a longer phrase.\n"
    "- Do NOT paraphrase, summarize, translate, expand abbreviations, correct typos, normalize formatting, or rewrite values.\n"
    "- If a field spans multiple consecutive source lines, preserve those line breaks in the JSON string.\n"
    "- If a field's value cannot be determined from explicit source text, use null.\n"
    "- Do NOT infer, guess, or use background knowledge.\n"
    "- Output valid JSON with the exact field names specified.\n"
    "- Output ONLY the JSON object or JSON array. No explanations, no markdown fences.\n"
    "- For numeric fields, output a JSON number only when the number is explicitly stated in the source text and can be copied by deterministic parsing.\n"
    "- For boolean fields, output true or false only when the source text explicitly states the condition.\n"
    "- If the prompt explicitly instructs you to resolve conflicts, use the final / most recent stated value. Otherwise do not resolve by inference.\n"
    "- Preserve original capitalization, punctuation, wording, and language for string values."
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


# ---------------------------------------------------------------------------
# Category Config
# ---------------------------------------------------------------------------

CATEGORY_LABELS: dict[str, str] = {
    "A": "Clean Extraction",
    "B": "Noisy and Informal",
    "C": "Multi-Entity",
    "D": "Implicit and Missing",
    "E": "Complex Documents",
}

CATEGORY_WEIGHTS: dict[str, int] = {
    "A": 15,
    "B": 20,
    "C": 25,
    "D": 25,
    "E": 15,
}

ARRAY_OBJECT_ANCHORS: dict[str, str] = {
    "DE-02.items": "name",
    "DE-07.$root": "name",
    "DE-13.line_items": "description",
    "DE-13.discounts": "description",
}

# ---------------------------------------------------------------------------
# Evaluation Engine
# ---------------------------------------------------------------------------


def is_plain_object(value: Any) -> bool:
    return isinstance(value, dict)


def top_level_shape(value: Any) -> str:
    if isinstance(value, list):
        return "array"
    if is_plain_object(value):
        return "object"
    return "other"


def normalize_string(value: str) -> str:
    return value.strip()


def compare_scalar(expected: Any, actual: Any) -> dict[str, Any]:
    if expected is None:
        return {"correct": actual is None, "reason": None if actual is None else "expected null"}

    if isinstance(expected, str):
        return {
            "correct": isinstance(actual, str) and normalize_string(actual) == normalize_string(expected),
            "reason": None if isinstance(actual, str) else "expected string",
        }

    if isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if isinstance(actual, (int, float)) and not isinstance(actual, bool):
            return {
                "correct": math.isfinite(actual) and abs(actual - expected) <= 0.01,
                "reason": None,
            }
        return {"correct": False, "reason": "expected number"}

    if isinstance(expected, bool):
        return {
            "correct": actual is expected or (isinstance(actual, bool) and actual == expected),
            "reason": None if isinstance(actual, bool) else "expected boolean",
        }

    return {"correct": False, "reason": "unsupported scalar type"}


def compare_scalar_array(expected: list, actual: Any) -> dict[str, Any]:
    if not isinstance(actual, list):
        return {"correct": 0, "total": 1, "notes": ["expected array"]}

    if len(expected) != len(actual):
        return {
            "correct": 0,
            "total": 1,
            "notes": [f"expected {len(expected)} items but received {len(actual)}"],
        }

    remaining = list(actual)
    for expected_item in expected:
        match_index = -1
        for i, candidate in enumerate(remaining):
            if compare_scalar(expected_item, candidate)["correct"]:
                match_index = i
                break
        if match_index == -1:
            return {"correct": 0, "total": 1, "notes": ["array values did not match expected set"]}
        remaining.pop(match_index)

    return {"correct": 1, "total": 1, "notes": []}


def compare_object_array(
    expected: list[dict[str, Any]],
    actual: Any,
    scenario_id: str,
    path: str,
) -> dict[str, Any]:
    if not isinstance(actual, list):
        field_count = len(expected[0]) if expected and expected[0] else 1
        return {"correct": 0, "total": len(expected) * field_count, "notes": ["expected array"]}

    anchor_key = f"{scenario_id}.{path or '$root'}"
    anchor = ARRAY_OBJECT_ANCHORS.get(anchor_key)
    if not anchor:
        return {
            "correct": 0,
            "total": len(expected) or 1,
            "notes": [f"missing anchor key for {anchor_key}"],
        }

    actual_by_anchor: dict[str, dict[str, Any]] = {}
    for item in actual:
        if is_plain_object(item) and anchor in item and isinstance(item[anchor], str):
            actual_by_anchor[str(item[anchor])] = item

    correct = 0
    total = 0
    notes: list[str] = []

    for expected_item in expected:
        actual_item = actual_by_anchor.get(str(expected_item.get(anchor)))
        for key, expected_value in expected_item.items():
            total += 1
            if actual_item is None:
                notes.append(f"missing object with {anchor}={expected_item.get(anchor)}")
                continue
            result = compare_value(
                expected_value,
                actual_item.get(key),
                scenario_id,
                f"{path}.{key}" if path else key,
            )
            correct += result["correct"]
            if result["notes"]:
                notes.extend(result["notes"])

    return {"correct": correct, "total": total, "notes": notes}


def compare_object(
    expected: dict[str, Any],
    actual: Any,
    scenario_id: str,
    path: str = "",
) -> dict[str, Any]:
    if not is_plain_object(actual):
        return {"correct": 0, "total": len(expected), "notes": ["expected object"]}

    correct = 0
    total = 0
    notes: list[str] = []

    for key, expected_value in expected.items():
        nested_path = f"{path}.{key}" if path else key
        result = compare_value(expected_value, actual.get(key), scenario_id, nested_path)
        correct += result["correct"]
        total += result["total"]
        if result["notes"]:
            notes.extend(result["notes"])

    return {"correct": correct, "total": total, "notes": notes}


def compare_value(
    expected: Any,
    actual: Any,
    scenario_id: str,
    path: str,
) -> dict[str, Any]:
    if isinstance(expected, list):
        if all(is_plain_object(item) for item in expected):
            return compare_object_array(expected, actual, scenario_id, path)
        return compare_scalar_array(expected, actual)

    if is_plain_object(expected):
        return compare_object(expected, actual, scenario_id, path)

    scalar = compare_scalar(expected, actual)
    return {
        "correct": 1 if scalar["correct"] else 0,
        "total": 1,
        "notes": [] if scalar["correct"] else [f"{path}: {scalar.get('reason') or 'mismatch'}"],
    }


def evaluate_compliance(expected: Any, actual: Any) -> dict[str, Any]:
    notes: list[str] = []
    expected_shape = top_level_shape(expected)
    actual_shape = top_level_shape(actual)
    exact_top_level_shape = expected_shape == actual_shape
    if not exact_top_level_shape:
        notes.append(f"top-level shape mismatch: expected {expected_shape}, received {actual_shape}")

    requested_fields_only = True
    no_missing_expected_fields = True

    if is_plain_object(expected) and is_plain_object(actual):
        expected_keys = set(expected.keys())
        actual_keys = set(actual.keys())
        extra = [k for k in actual_keys if k not in expected_keys]
        missing = [k for k in expected_keys if k not in actual_keys]

        if extra:
            requested_fields_only = False
            notes.append(f"extra top-level fields: {', '.join(extra)}")
        if missing:
            no_missing_expected_fields = False
            notes.append(f"missing top-level fields: {', '.join(missing)}")

    return {
        "validJson": True,
        "exactTopLevelShape": exact_top_level_shape,
        "requestedFieldsOnly": requested_fields_only,
        "noMissingExpectedFields": no_missing_expected_fields,
        "notes": notes,
    }


def evaluate_scenario_output(
    scenario_id: str, expected: Any, final_answer: str
) -> ScenarioEvaluation:
    try:
        parsed = json.loads(final_answer)
    except (json.JSONDecodeError, TypeError) as e:
        message = str(e)
        return ScenarioEvaluation(
            status="fail",
            score=0,
            summary=f"Invalid JSON: {message}",
            note="Official score is 0 when the response is not valid JSON.",
        )

    compliance = evaluate_compliance(expected, parsed)
    comparison = compare_value(expected, parsed, scenario_id, "")
    score = 0 if comparison["total"] == 0 else round(comparison["correct"] / comparison["total"] * 100)

    compliance_flags = ", ".join([
        "shape ok" if compliance["exactTopLevelShape"] else "shape fail",
        "fields only" if compliance["requestedFieldsOnly"] else "extra fields",
        "no missing fields" if compliance["noMissingExpectedFields"] else "missing fields",
    ])
    note_parts = compliance["notes"] + comparison["notes"]

    return ScenarioEvaluation(
        status=status_for_score(score),
        score=score,
        summary=f"{comparison['correct']}/{comparison['total']} atomic fields correct ({score}%). {compliance_flags}.",
        note=" | ".join(note_parts) if note_parts else None,
    )


def status_for_score(score: int) -> Literal["pass", "partial", "fail"]:
    if score >= 85:
        return "pass"
    if score >= 60:
        return "partial"
    return "fail"


# ---------------------------------------------------------------------------
# Scenario Definitions
# ---------------------------------------------------------------------------

SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "DE-01",
        "title": "Business Card / Contact Info",
        "category": "A",
        "description": "Baseline extraction from well-structured input. Every field is clearly labeled and unambiguous.",
        "user_message": (
            "Extract the contact information from this text:\n\n"
            "Dr. Sarah Chen, Ph.D.\n"
            "Senior Research Scientist\n"
            "BioGen Therapeutics, Inc.\n"
            "1420 Harbor Blvd, Suite 300\n"
            "San Diego, CA 92101\n"
            "Tel: (858) 555-0147\n"
            "Email: s.chen@biogentherapeutics.com\n"
            "LinkedIn: linkedin.com/in/sarahchen-phd\n\n"
            "Fields: name, title, company, address, city, state, zip, phone, email, linkedin_url"
        ),
        "expected": {
            "name": "Dr. Sarah Chen, Ph.D.",
            "title": "Senior Research Scientist",
            "company": "BioGen Therapeutics, Inc.",
            "address": "1420 Harbor Blvd, Suite 300",
            "city": "San Diego",
            "state": "CA",
            "zip": "92101",
            "phone": "(858) 555-0147",
            "email": "s.chen@biogentherapeutics.com",
            "linkedin_url": "linkedin.com/in/sarahchen-phd",
        },
    },
    {
        "id": "DE-02",
        "title": "Receipt / Invoice",
        "category": "A",
        "description": 'Structured extraction from a receipt format. Prices must be numbers (not strings with "$"). The model must handle the items array and distinguish between subtotal, tax, and total.',
        "user_message": (
            "Extract the transaction details from this receipt:\n\n"
            "URBAN BEAN COFFEE\n"
            "847 Market Street\n"
            "San Francisco, CA 94103\n\n"
            "Date: 03/15/2026  Time: 08:42 AM\n"
            "Cashier: Mike  Register: #3\n\n"
            "Americano (L)          $4.75\n"
            "Oat Milk Latte (M)     $5.50\n"
            "Blueberry Muffin       $3.25\n\n"
            "Subtotal:             $13.50\n"
            "Tax (8.625%):          $1.16\n"
            "Total:                $14.66\n\n"
            "Visa ending 4821\n"
            "Auth: 772941\n\n"
            "Fields: store_name, store_address, date, time, items (array of {name, price}), subtotal, tax_rate, tax_amount, total, payment_method, card_last_four"
        ),
        "expected": {
            "store_name": "URBAN BEAN COFFEE",
            "store_address": "847 Market Street\nSan Francisco, CA 94103",
            "date": "03/15/2026",
            "time": "08:42 AM",
            "items": [
                {"name": "Americano (L)", "price": 4.75},
                {"name": "Oat Milk Latte (M)", "price": 5.50},
                {"name": "Blueberry Muffin", "price": 3.25},
            ],
            "subtotal": 13.50,
            "tax_rate": "8.625%",
            "tax_amount": 1.16,
            "total": 14.66,
            "payment_method": "Visa",
            "card_last_four": "4821",
        },
    },
    {
        "id": "DE-03",
        "title": "Job Posting",
        "category": "A",
        "description": 'Extracting from semi-structured text with multiple field types: strings, numbers, arrays. The model must parse salary range, distinguish required vs. preferred skills, and extract years of experience from prose ("3+ years").',
        "user_message": (
            "Extract the job details from this posting:\n\n"
            "\U0001f680 We're Hiring! Senior Frontend Engineer\n\n"
            "Location: Austin, TX (Hybrid - 3 days in office)\n"
            "Team: Product Engineering\n"
            "Reports to: VP of Engineering\n\n"
            "Compensation: $145,000 - $185,000 base + equity + benefits\n\n"
            "About the role:\n"
            "We're looking for a senior frontend engineer to lead our design system rebuild. You'll work with React, TypeScript, and our custom component library. 3+ years of experience required.\n\n"
            "Must have: React, TypeScript, CSS-in-JS, accessibility (WCAG 2.1)\n"
            "Nice to have: GraphQL, Storybook, animation (Framer Motion)\n\n"
            "Apply at careers.example.com/senior-fe or email jobs@example.com\n\n"
            "Fields: job_title, location, work_model, team, reports_to, salary_min, salary_max, required_skills (array), preferred_skills (array), experience_years_min, apply_url, apply_email"
        ),
        "expected": {
            "job_title": "Senior Frontend Engineer",
            "location": "Austin, TX",
            "work_model": "Hybrid",
            "team": "Product Engineering",
            "reports_to": "VP of Engineering",
            "salary_min": 145000,
            "salary_max": 185000,
            "required_skills": ["React", "TypeScript", "CSS-in-JS", "accessibility (WCAG 2.1)"],
            "preferred_skills": ["GraphQL", "Storybook", "animation (Framer Motion)"],
            "experience_years_min": 3,
            "apply_url": "careers.example.com/senior-fe",
            "apply_email": "jobs@example.com",
        },
    },
    {
        "id": "DE-04",
        "title": "Informal Email Thread",
        "category": "B",
        "description": "Extraction from conversational, informal text. The model must: read the thread chronologically (the latest email has the final decision), distinguish the sprint planning meeting from the retro meeting, and extract from casual language.",
        "user_message": (
            "Extract the meeting details from this email thread:\n\n"
            "---\n"
            "From: Lisa Park <lisa.p@company.com>\n"
            "To: Team-Alpha <team-alpha@company.com>\n"
            "Date: March 10, 2026, 3:47 PM\n"
            "Subject: Re: Re: sprint planning??\n\n"
            "ok so after talking to Jake, let's do Thursday instead of Wednesday. Same time tho - 10am. We can use the Maple room since Oak is booked. bring your laptops, we're gonna do live estimation this time\n\n"
            "btw the retro from last sprint is still on for Friday 2pm in Birch. don't confuse the two lol\n\n"
            "- Lisa\n"
            "---\n"
            "From: Jake Torres <j.torres@company.com>\n"
            "To: Team-Alpha <team-alpha@company.com>\n"
            "Date: March 10, 2026, 2:15 PM\n"
            "Subject: Re: sprint planning??\n\n"
            "Wednesday doesn't work for me, can we move it? Thursday or Friday works. Also the Oak room has a broken projector so maybe book a different one.\n"
            "---\n\n"
            "Fields: meeting_name, day, time, room, organizer_name, organizer_email, note"
        ),
        "expected": {
            "meeting_name": "sprint planning",
            "day": "Thursday",
            "time": "10am",
            "room": "Maple",
            "organizer_name": "Lisa Park",
            "organizer_email": "lisa.p@company.com",
            "note": "bring your laptops, we're gonna do live estimation this time",
        },
    },
    {
        "id": "DE-05",
        "title": "Product Review with Embedded Specs",
        "category": "B",
        "description": "Extracting from an opinionated, informal product review. The model must distinguish the reviewed product's specs from competitor specs, handle two different prices (sale vs. original), identify competitors mentioned in comparison, and copy complaint / recommendation text exactly rather than paraphrasing it.",
        "user_message": (
            "Extract the product details and reviewer's assessment from this review:\n\n"
            "\u2605\u2605\u2605\u2605\u2606\n"
            "Reviewed by: TechDad_42 on Feb 28, 2026\n"
            "Verified Purchase\n\n"
            "Bought the XR-7500 Pro noise-cancelling headphones for my daily commute. Was deciding between these and the Sony WH-1000XM5 ($348) but went with these since they were on sale for $279 (normally $329). Battery life is honestly amazing \u2014 I get about 38 hours on a single charge vs the 30 hours Sony claims. The ANC is good but not quite as good as my old Bose QC45s tbh. Sound quality is excellent for the price point. They fold flat which is great for my bag.\n\n"
            "One complaint: the app (v3.2.1) is buggy on Android 14. Crashes when trying to customize EQ. Not a dealbreaker since the default sound profile is great.\n\n"
            "Weight: 254g. Bluetooth 5.3. USB-C charging.\n\n"
            "Would I recommend? Yeah, especially at the sale price. Best value under $300 IMO.\n\n"
            "Fields: product_name, product_price_paid, product_price_original, rating_stars, reviewer_name, battery_life_hours, weight_grams, bluetooth_version, charging_type, competitor_1_name, competitor_1_price, competitor_2_name, complaint, recommendation"
        ),
        "expected": {
            "product_name": "XR-7500 Pro",
            "product_price_paid": 279,
            "product_price_original": 329,
            "rating_stars": 4,
            "reviewer_name": "TechDad_42",
            "battery_life_hours": 38,
            "weight_grams": 254,
            "bluetooth_version": "5.3",
            "charging_type": "USB-C",
            "competitor_1_name": "Sony WH-1000XM5",
            "competitor_1_price": 348,
            "competitor_2_name": "Bose QC45s",
            "complaint": "the app (v3.2.1) is buggy on Android 14. Crashes when trying to customize EQ.",
            "recommendation": "Yeah, especially at the sale price.",
        },
    },
    {
        "id": "DE-06",
        "title": "Handwritten-Style Notes (OCR Simulation)",
        "category": "B",
        "description": 'Medical abbreviation parsing (CC, BP, HR, h/o, r/t, Rx, Pt) and extracting structured data from abbreviated notes without expanding abbreviations in string fields. The `referral` field is explicitly "none at this time" \u2014 the correct extraction is `null` (no referral was made).',
        "user_message": (
            "Extract the patient visit details from these medical office notes (transcribed from handwritten):\n\n"
            "Pt: Margaret Liu    DOB: 5/12/1958\n"
            "Visit: 3-20-2026    Provider: Dr. Patel\n\n"
            "CC: persistent cough x 3 weeks, worse at night\n"
            "no fever, no weight loss\n"
            "h/o asthma - childhood, resolved\n\n"
            "BP 128/82  HR 76  Temp 98.4  SpO2 97%\n\n"
            "Assessment: likely post-nasal drip r/t seasonal allergies\n"
            "Plan: fluticasone nasal spray, follow up 2 wks if no improvement\n"
            "Referral: none at this time\n\n"
            "Rx: fluticasone propionate 50mcg, 2 sprays each nostril daily x 30 days\n\n"
            "Fields: patient_name, date_of_birth, visit_date, provider, chief_complaint, blood_pressure_systolic, blood_pressure_diastolic, heart_rate, temperature, oxygen_saturation, assessment, medication_name, medication_dose, medication_duration, referral"
        ),
        "expected": {
            "patient_name": "Margaret Liu",
            "date_of_birth": "5/12/1958",
            "visit_date": "3-20-2026",
            "provider": "Dr. Patel",
            "chief_complaint": "persistent cough x 3 weeks, worse at night",
            "blood_pressure_systolic": 128,
            "blood_pressure_diastolic": 82,
            "heart_rate": 76,
            "temperature": 98.4,
            "oxygen_saturation": 97,
            "assessment": "likely post-nasal drip r/t seasonal allergies",
            "medication_name": "fluticasone propionate",
            "medication_dose": "50mcg, 2 sprays each nostril daily",
            "medication_duration": "30 days",
            "referral": None,
        },
    },
    {
        "id": "DE-07",
        "title": "Multiple People in One Document",
        "category": "C",
        "description": "Multi-entity extraction where the model must correctly associate attributes with the right person. Marcus's phone is not Sarah's phone. Sarah's location is LA (her new location), not Chicago (where she's moving from). Priya's rate belongs only to Priya.",
        "user_message": (
            "Extract information about each person mentioned in this email:\n\n"
            "Hi team,\n\n"
            "Quick updates after today's meeting:\n\n"
            "\u2014 Marcus Washington (Senior Designer, NYC office) is taking over the Acme rebrand from Sarah. He'll be on-site in Chicago next week. His direct line is 212-555-0189. Contact him at m.washington@studio.com\n\n"
            "\u2014 Sarah Kim is transitioning to the Globex account effective April 1. She's relocating from Chicago to the LA office. No new phone yet \u2014 use her email sarah.k@studio.com for now.\n\n"
            "\u2014 The freelance illustrator we hired for Acme is Priya Desai. She charges $95/hour. Her portfolio is at priyadesai.com. She's based in Toronto but works US East Coast hours. Email: priya@priyadesai.com\n\n"
            "Please update your contacts accordingly.\n"
            "\u2014 Jordan\n\n"
            "Fields per person: name, role, location, email, phone, hourly_rate, note\n"
            "Extract as an array of person objects."
        ),
        "expected": [
            {
                "name": "Marcus Washington",
                "role": "Senior Designer",
                "location": "NYC",
                "email": "m.washington@studio.com",
                "phone": "212-555-0189",
                "hourly_rate": None,
                "note": "taking over the Acme rebrand from Sarah. He'll be on-site in Chicago next week",
            },
            {
                "name": "Sarah Kim",
                "role": None,
                "location": "LA",
                "email": "sarah.k@studio.com",
                "phone": None,
                "hourly_rate": None,
                "note": "transitioning to the Globex account effective April 1. She's relocating from Chicago to the LA office",
            },
            {
                "name": "Priya Desai",
                "role": "freelance illustrator",
                "location": "Toronto",
                "email": "priya@priyadesai.com",
                "phone": None,
                "hourly_rate": 95,
                "note": "works US East Coast hours",
            },
        ],
    },
    {
        "id": "DE-08",
        "title": "Conflicting Information",
        "category": "C",
        "description": "Temporal conflict resolution. The model must track which update supersedes which. The start time changed in the March 18 correction. The location changed in the March 12 update. The catering price changed in the March 12 update. The date did not change (explicitly 'Same date').",
        "user_message": (
            "Extract the event details from this text. If information conflicts, extract the MOST RECENT version.\n\n"
            "=== ORIGINAL INVITE (March 1) ===\n"
            "Annual Company Picnic\n"
            "Date: Saturday, June 14\n"
            "Time: 11:00 AM \u2013 4:00 PM\n"
            "Location: Riverside Park, Shelter B\n"
            "RSVP by May 15 to events@company.com\n"
            "Catering by Fresh Bites ($22/person)\n\n"
            "=== UPDATE (March 12) ===\n"
            "Hey all \u2014 slight change of plans. We're moving the picnic to Lincoln Park (Shelter A) because Riverside is under construction. Same date and time. Also the caterer changed their pricing to $25/person.\n\n"
            "=== CORRECTION (March 18) ===\n"
            "Sorry, one more update. The start time is now 11:30 AM (the park opens later than we thought). End time stays 4:00 PM. Everything else from the March 12 update still stands.\n\n"
            "Fields: event_name, date, start_time, end_time, location_park, location_shelter, rsvp_deadline, rsvp_email, catering_company, catering_price_per_person"
        ),
        "expected": {
            "event_name": "Annual Company Picnic",
            "date": "Saturday, June 14",
            "start_time": "11:30 AM",
            "end_time": "4:00 PM",
            "location_park": "Lincoln Park",
            "location_shelter": "Shelter A",
            "rsvp_deadline": "May 15",
            "rsvp_email": "events@company.com",
            "catering_company": "Fresh Bites",
            "catering_price_per_person": 25,
        },
    },
    {
        "id": "DE-09",
        "title": "Nested Quotes \u2014 Email Within Email",
        "category": "C",
        "description": "Context layering. The email thread contains three different room requests from three different people at three different times. The model must extract only Alice's current request (the outermost message), not Bob's cancelled request or Carol's suggestion.",
        "user_message": (
            "Extract ONLY the current (outermost) request details. Ignore quoted/forwarded content.\n\n"
            "From: Alice Wong <alice@company.com>\n"
            "To: Facilities <facilities@company.com>\n"
            "Date: March 20, 2026\n"
            "Subject: FW: Re: Conference Room Request\n\n"
            "Hi Facilities,\n\n"
            "Forwarding this thread for context, but here's what I actually need:\n\n"
            "Please book the Atlas room for 8 people on March 28, 2:00 PM\u20133:30 PM. We need a projector and whiteboard. No catering needed.\n\n"
            "Thanks,\nAlice\n\n"
            "-------- Forwarded Message --------\n"
            "From: Bob Chen <bob@company.com>\n"
            "Date: March 15, 2026\n\n"
            "Alice, I originally asked for the Summit room on March 22 for 12 people with lunch catering for the Q1 review. But that's been cancelled now. Can you see if Atlas is available for a smaller meeting instead?\n\n"
            "-------- Original Message --------\n"
            "From: Carol Davis <carol@company.com>\n"
            "Date: March 10, 2026\n\n"
            "Bob \u2014 the Phoenix room is booked for March 22. Try Summit or Atlas. We can do catering for up to 20 people.\n\n"
            "Fields: requester_name, requester_email, room, date, start_time, end_time, attendee_count, needs_projector, needs_whiteboard, needs_catering"
        ),
        "expected": {
            "requester_name": "Alice Wong",
            "requester_email": "alice@company.com",
            "room": "Atlas",
            "date": "March 28",
            "start_time": "2:00 PM",
            "end_time": "3:30 PM",
            "attendee_count": 8,
            "needs_projector": True,
            "needs_whiteboard": True,
            "needs_catering": False,
        },
    },
    {
        "id": "DE-10",
        "title": "The Over-Extraction Trap",
        "category": "D",
        "description": 'The model\'s ability to return null for fields that are genuinely unknowable. The reviewer explicitly says they are "not 100% sure" about the neighborhood, "didn\'t catch" the chef\'s name, has "no idea" what was paid, and can\'t pinpoint the location. There is no rating score, only positive sentiment.',
        "user_message": (
            "Extract the restaurant details from this review:\n\n"
            '"Had dinner at Sakura Sushi last night with my partner. The place is tucked away on a quiet street in what I think is the Nob Hill neighborhood \u2014 honestly not 100% sure. The omakase was incredible, maybe the best I\'ve had in the city. We spent about 2 hours there. The chef (didn\'t catch his name) was super friendly. No idea what we paid \u2014 my partner handled the bill. We took an Uber home so I know it\'s somewhere near California Street but I really couldn\'t point to it on a map. Definitely going back."\n\n'
            "Fields: restaurant_name, cuisine_type, neighborhood, street_address, chef_name, price_paid, visit_duration, rating_score, reservation_required, parking_available"
        ),
        "expected": {
            "restaurant_name": "Sakura Sushi",
            "cuisine_type": "Sushi",
            "neighborhood": None,
            "street_address": None,
            "chef_name": None,
            "price_paid": None,
            "visit_duration": "about 2 hours",
            "rating_score": None,
            "reservation_required": None,
            "parking_available": None,
        },
    },
    {
        "id": "DE-11",
        "title": "Negation and Correction in Text",
        "category": "D",
        "description": "Tracking corrections and self-corrections in stream-of-consciousness text. Every field has been revised at least once. The model must extract the final stated value, not the first.",
        "user_message": (
            "Extract the final confirmed details from this planning message:\n\n"
            '"OK so for the team offsite: we were thinking Aspen but actually that\'s too expensive so let\'s do Lake Tahoe instead. Budget per person is $500, wait no, I just checked and it\'s $450 after the flights went up. Dates are April 10-12. Actually hold on \u2014 April 10 is a Monday and people need to fly in Sunday night, so let\'s do April 11-13 (Tue-Thu). We\'ll need 8 rooms. No wait, Jessica and Tom are sharing, and Mike can\'t make it anymore, so 6 rooms. The hotel is Mountain View Lodge. Or was it Mountain View Inn? Let me check... it\'s Mountain View Lodge. Confirmed."\n\n'
            "Fields: destination, budget_per_person, start_date, end_date, num_rooms, hotel_name, total_attendees"
        ),
        "expected": {
            "destination": "Lake Tahoe",
            "budget_per_person": 450,
            "start_date": "April 11",
            "end_date": "April 13",
            "num_rooms": 6,
            "hotel_name": "Mountain View Lodge",
            "total_attendees": None,
        },
    },
    {
        "id": "DE-12",
        "title": "Decoy Entity \u2014 Don't Extract the Wrong Thing",
        "category": "D",
        "description": "Entity disambiguation. Three laptops are mentioned with their specs. The model must extract only the ZenBook's details. MacBook's 18-hour battery and $1,099 price, Dell's 1.2kg weight and $1,199 price are decoys.",
        "user_message": (
            "Extract details about the PRODUCT BEING REVIEWED, not any competitors mentioned.\n\n"
            '"Bought a ZenBook Pro 15 laptop for my daughter\'s college. $1,299 at Best Buy. She was also considering the MacBook Air M3 ($1,099) and the Dell XPS 13 ($1,199) but we went with ASUS because of the OLED display and 16GB RAM. Battery life is about 10 hours which is less than the MacBook\'s 18 hours but she\'ll mostly use it plugged in. It\'s heavier than the Dell at 1.8kg vs 1.2kg but the 15-inch screen was important to her.\n\n'
            "The i7-13700H processor handles everything she throws at it. 512GB SSD is fine for now \u2014 she keeps everything in Google Drive anyway. One thing that bugs me: it came with McAfee preinstalled and a bunch of other bloatware. The ASUS customer support was also pretty bad when I called about a driver issue.\n\n"
            'Overall: 4/5 stars. Great laptop for the price, just wish ASUS cleaned up the software experience."\n\n'
            "Fields: product_name, brand, price, store, display_type, display_size, ram_gb, processor, storage, battery_life_hours, weight_kg, rating, operating_system, complaint"
        ),
        "expected": {
            "product_name": "ZenBook Pro 15",
            "brand": "ASUS",
            "price": 1299,
            "store": "Best Buy",
            "display_type": "OLED",
            "display_size": "15-inch",
            "ram_gb": 16,
            "processor": "i7-13700H",
            "storage": "512GB SSD",
            "battery_life_hours": 10,
            "weight_kg": 1.8,
            "rating": 4,
            "operating_system": None,
            "complaint": "it came with McAfee preinstalled and a bunch of other bloatware. The ASUS customer support was also pretty bad when I called about a driver issue.",
        },
    },
    {
        "id": "DE-13",
        "title": "Multi-Section Invoice with Discounts",
        "category": "E",
        "description": "Complex structured extraction with nested arrays, multiple sections, and financial data requiring exact precision.",
        "user_message": (
            "Extract the invoice details:\n\n"
            "INVOICE #INV-2026-0847\n\n"
            "From: CloudTech Solutions, LLC\n"
            "123 Innovation Way, Seattle, WA 98101\n"
            "Tax ID: 91-7654321\n\n"
            "Bill To: Meridian Corp\n"
            "Attn: Accounts Payable\n"
            "500 Commerce Dr, Portland, OR 97201\n"
            "PO#: MC-2026-445\n\n"
            "Invoice Date: March 15, 2026\n"
            "Due Date: April 14, 2026\n"
            "Terms: Net 30\n\n"
            "\u250c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2510\n"
            "\u2502 Description                    \u2502 Qty \u2502 Unit Price \u2502 Amount         \u2502\n"
            "\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524\n"
            "\u2502 Cloud Hosting - Pro Plan       \u2502 12  \u2502 $149.00    \u2502 $1,788.00     \u2502\n"
            "\u2502 API Gateway - Standard         \u2502 1   \u2502 $299.00    \u2502 $299.00       \u2502\n"
            "\u2502 SSL Certificate - Wildcard     \u2502 2   \u2502 $89.00     \u2502 $178.00       \u2502\n"
            "\u2502 Professional Services (hrs)    \u2502 8   \u2502 $175.00    \u2502 $1,400.00     \u2502\n"
            "\u2502 Data Migration (one-time)      \u2502 1   \u2502 $500.00    \u2502 $500.00       \u2502\n"
            "\u251c\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2524\n"
            "\u2502 Subtotal                                           \u2502 $4,165.00     \u2502\n"
            "\u2502 Volume Discount (10%)                              \u2502 -$416.50      \u2502\n"
            "\u2502 Early Payment Credit                               \u2502 -$50.00       \u2502\n"
            "\u2502 \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500 \u2502\n"
            "\u2502 Total Due                                          \u2502 $3,698.50     \u2502\n"
            "\u2514\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2518\n\n"
            "Payment: Wire transfer to Account #****4521 (Chase Bank)\n"
            "For questions: billing@cloudtech.io\n\n"
            "Fields: invoice_number, vendor_name, vendor_address, vendor_tax_id, client_name, client_po, invoice_date, due_date, payment_terms, line_items (array of {description, qty, unit_price, amount}), subtotal, discounts (array of {description, amount}), total_due, payment_method, billing_contact_email"
        ),
        "expected": {
            "invoice_number": "INV-2026-0847",
            "vendor_name": "CloudTech Solutions, LLC",
            "vendor_address": "123 Innovation Way, Seattle, WA 98101",
            "vendor_tax_id": "91-7654321",
            "client_name": "Meridian Corp",
            "client_po": "MC-2026-445",
            "invoice_date": "March 15, 2026",
            "due_date": "April 14, 2026",
            "payment_terms": "Net 30",
            "line_items": [
                {"description": "Cloud Hosting - Pro Plan", "qty": 12, "unit_price": 149.00, "amount": 1788.00},
                {"description": "API Gateway - Standard", "qty": 1, "unit_price": 299.00, "amount": 299.00},
                {"description": "SSL Certificate - Wildcard", "qty": 2, "unit_price": 89.00, "amount": 178.00},
                {"description": "Professional Services (hrs)", "qty": 8, "unit_price": 175.00, "amount": 1400.00},
                {"description": "Data Migration (one-time)", "qty": 1, "unit_price": 500.00, "amount": 500.00},
            ],
            "subtotal": 4165.00,
            "discounts": [
                {"description": "Volume Discount (10%)", "amount": -416.50},
                {"description": "Early Payment Credit", "amount": -50.00},
            ],
            "total_due": 3698.50,
            "payment_method": "Wire transfer",
            "billing_contact_email": "billing@cloudtech.io",
        },
    },
    {
        "id": "DE-14",
        "title": "Multi-Language Document",
        "category": "E",
        "description": "Extraction from bilingual content (Japanese/English). The model must handle Japanese yen formatting, bilingual labels, CJK characters, and preserve source-language strings exactly when a field is returned as text.",
        "user_message": (
            "Extract the product details from this bilingual product listing:\n\n"
            "\u3010\u65b0\u767a\u58f2\u3011\u30ef\u30a4\u30e4\u30ec\u30b9\u30a4\u30e4\u30db\u30f3 AirPulse X3\n"
            "New Release: AirPulse X3 Wireless Earbuds\n\n"
            "\u4fa1\u683c / Price: \u00a512,980 (\u7a0e\u8fbc / tax included)\n"
            "\u8272 / Colors: \u30df\u30c3\u30c9\u30ca\u30a4\u30c8\u30d6\u30e9\u30c3\u30af (Midnight Black), \u30d1\u30fc\u30eb\u30db\u30ef\u30a4\u30c8 (Pearl White), \u30b5\u30af\u30e9\u30d4\u30f3\u30af (Sakura Pink)\n\n"
            "\u4ed5\u69d8 / Specifications:\n"
            "\u30c9\u30e9\u30a4\u30d0\u30fc / Driver: 10mm \u30c0\u30a4\u30ca\u30df\u30c3\u30af / 10mm Dynamic\n"
            "Bluetooth: 5.3\n"
            "\u30d0\u30c3\u30c6\u30ea\u30fc / Battery: \u30a4\u30e4\u30db\u30f3\u672c\u4f53 8\u6642\u9593 / Earbuds 8 hours, \u30b1\u30fc\u30b9\u8fbc\u307f 32\u6642\u9593 / With case 32 hours\n"
            "\u9632\u6c34 / Water resistance: IPX5\n"
            "\u91cd\u91cf / Weight: \u7247\u8033 5.2g / Per earbud 5.2g, \u30b1\u30fc\u30b9 42g / Case 42g\n"
            "\u30ce\u30a4\u30ad\u30e3\u30f3 / ANC: \u2713 \u30a2\u30c0\u30d7\u30c6\u30a3\u30d6 / Adaptive\n\n"
            "\u4ed8\u5c5e\u54c1 / Included: USB-C\u5145\u96fb\u30b1\u30fc\u30d6\u30eb, \u30a4\u30e4\u30fc\u30c1\u30c3\u30d7(S/M/L), \u53d6\u6271\u8aac\u660e\u66f8\n"
            "Included: USB-C charging cable, ear tips (S/M/L), user manual\n\n"
            "\u8ca9\u58f2\u5143 / Sold by: SoundWave Electronics Co., Ltd. (\u6771\u4eac\u90fd\u6e0b\u8c37\u533a)\n\n"
            "Fields: product_name, product_type, price_jpy, tax_included, colors (array), driver_size, bluetooth_version, battery_life_earbuds_hours, battery_life_with_case_hours, water_resistance_rating, earbud_weight_grams, case_weight_grams, has_anc, anc_type, included_items (array), seller_name, seller_location"
        ),
        "expected": {
            "product_name": "AirPulse X3",
            "product_type": "Wireless Earbuds",
            "price_jpy": 12980,
            "tax_included": True,
            "colors": ["Midnight Black", "Pearl White", "Sakura Pink"],
            "driver_size": "10mm",
            "bluetooth_version": "5.3",
            "battery_life_earbuds_hours": 8,
            "battery_life_with_case_hours": 32,
            "water_resistance_rating": "IPX5",
            "earbud_weight_grams": 5.2,
            "case_weight_grams": 42,
            "has_anc": True,
            "anc_type": "Adaptive",
            "included_items": ["USB-C charging cable", "ear tips (S/M/L)", "user manual"],
            "seller_name": "SoundWave Electronics Co., Ltd.",
            "seller_location": "\u6771\u4eac\u90fd\u6e0b\u8c37\u533a",
        },
    },
    {
        "id": "DE-15",
        "title": "The Empty Document Trap",
        "category": "E",
        "description": 'The model\'s ability to say "I don\'t know" when the text contains zero extractable information. This generic "About Us" page has no specific details whatsoever, so every field should be null.',
        "user_message": (
            'Extract the company details from this "About Us" page:\n\n'
            "Welcome to our website! We're passionate about what we do and we've been doing it for a long time. Our team of dedicated professionals works hard every day to deliver the best results for our clients. We believe in innovation, integrity, and excellence.\n\n"
            '"Great service!" \u2014 A happy customer\n'
            '"Would recommend!" \u2014 Another satisfied client\n\n'
            "Contact us today to learn how we can help your business grow!\n"
            "\u00a9 2026 All rights reserved.\n\n"
            "Fields: company_name, founding_year, ceo_name, employee_count, headquarters_city, industry, annual_revenue, phone_number, email, website_url"
        ),
        "expected": {
            "company_name": None,
            "founding_year": None,
            "ceo_name": None,
            "employee_count": None,
            "headquarters_city": None,
            "industry": None,
            "annual_revenue": None,
            "phone_number": None,
            "email": None,
            "website_url": None,
        },
    },
]

# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def score_model_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute category scores and final score from scenario results."""
    categories = ["A", "B", "C", "D", "E"]
    category_scores = []

    for category in categories:
        cat_results = [
            r for r in results
            if any(
                s["id"] == r["scenarioId"] and s["category"] == category
                for s in SCENARIOS
            )
        ]
        average_score = (
            0 if not cat_results
            else round(sum(r["score"] for r in cat_results) / len(cat_results))
        )
        category_scores.append({
            "category": category,
            "label": CATEGORY_LABELS[category],
            "weight": CATEGORY_WEIGHTS[category],
            "averageScore": average_score,
            "percent": average_score,
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


def strip_markdown_fences(text: str) -> str:
    """Strip markdown code fences if the model wraps its JSON in them."""
    stripped = text.strip()
    if stripped.startswith("```"):
        # Remove opening fence (possibly with language tag)
        first_newline = stripped.find("\n")
        if first_newline != -1:
            stripped = stripped[first_newline + 1:]
        # Remove closing fence
        if stripped.rstrip().endswith("```"):
            stripped = stripped.rstrip()[:-3].rstrip()
    return stripped


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
        state.final_answer = strip_markdown_fences(content)
        trace_lines.append(f"assistant_response={content[:200]}...")

    except Exception as exc:
        summary = str(exc)
        trace_lines.append(f"error={summary}")
        eval_info = {"status": "fail", "summary": summary}
        return {
            "scenarioId": scenario["id"],
            "status": "fail",
            "score": 0,
            "summary": summary,
            "rawLog": format_scenario_trace(model, scenario, eval_info, trace_lines),
        }

    trace_lines.append(f"final_answer={state.final_answer[:200]}...")

    evaluation = evaluate_scenario_output(
        scenario["id"], scenario["expected"], state.final_answer
    )

    eval_dict = {
        "status": evaluation.status,
        "summary": evaluation.summary,
        "note": evaluation.note,
    }

    return {
        "scenarioId": scenario["id"],
        "status": evaluation.status,
        "score": evaluation.score,
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
        description="DataExtract-15 Benchmark CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        dest="scenario_ids",
        help="Run a specific scenario (e.g. DE-01). Can be repeated.",
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
        scenario_ids.extend(
            s.strip() for s in args.scenario_list.split(",") if s.strip()
        )
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
