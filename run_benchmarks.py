#!/usr/bin/env python3
"""
BenchLocal Full Benchmark Runner

Runs all or selected benchmarks, collects results, and generates a unified report.

Usage:
  python run_benchmarks.py                          # Run all 7 benchmarks
  python run_benchmarks.py --bench toolcall15       # Run one benchmark
  python run_benchmarks.py --bench reasonmath15 --bench bugfind15  # Run selected
  python run_benchmarks.py --report-only results/   # Generate report from saved results
  python run_benchmarks.py --json                   # JSON output
  python run_benchmarks.py --model openrouter:openai/gpt-4.1  # Specific model

Dependencies: requests (pip install requests)
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent

BENCHMARKS: dict[str, dict[str, Any]] = {
    "toolcall15": {
        "script": "toolcall15.py",
        "label": "ToolCall-15",
        "description": "Tool calling ability across 15 scenarios",
        "scenarios": 15,
        "score_key": "finalScore",
        "max_per_scenario": 2,
        "score_scale": "points",
    },
    "bugfind15": {
        "script": "bugfind15.py",
        "label": "BugFind-15",
        "description": "Bug finding and fixing across 15 code scenarios",
        "scenarios": 15,
        "score_key": "finalScore",
        "max_per_scenario": 100,
        "score_scale": "percent",
    },
    "dataextract15": {
        "script": "dataextract15.py",
        "label": "DataExtract-15",
        "description": "Structured data extraction from 15 document types",
        "scenarios": 15,
        "score_key": "finalScore",
        "max_per_scenario": 100,
        "score_scale": "percent",
    },
    "instructfollow15": {
        "script": "instructfollow15.py",
        "label": "InstructFollow-15",
        "description": "Instruction following with 15 constraint sets",
        "scenarios": 15,
        "score_key": "finalScore",
        "max_per_scenario": 100,
        "score_scale": "percent",
    },
    "reasonmath15": {
        "script": "reasonmath15.py",
        "label": "ReasonMath-15",
        "description": "Mathematical reasoning across 15 problems",
        "scenarios": 15,
        "score_key": "finalScore",
        "max_per_scenario": 100,
        "score_scale": "percent",
    },
    "structoutput15": {
        "script": "structoutput15.py",
        "label": "StructOutput-15",
        "description": "Structured output generation in 15 formats",
        "scenarios": 15,
        "score_key": "finalScore",
        "max_per_scenario": 100,
        "score_scale": "percent",
    },
    "hermesagent20": {
        "script": "hermesagent20.py",
        "label": "HermesAgent-20",
        "description": "Agent capabilities across 20 scenarios",
        "scenarios": 20,
        "score_key": "finalScore",
        "max_per_scenario": 100,
        "score_scale": "percent",
    },
}

BENCHMARK_ORDER = [
    "toolcall15",
    "reasonmath15",
    "instructfollow15",
    "dataextract15",
    "bugfind15",
    "structoutput15",
    "hermesagent20",
]

# ---------------------------------------------------------------------------
# Data Classes
# ---------------------------------------------------------------------------


@dataclass
class BenchmarkResult:
    benchmark: str
    label: str
    model_scores: dict[str, dict[str, Any]] = field(default_factory=dict)
    raw_json: dict[str, Any] | None = None
    error: str | None = None
    duration_seconds: float = 0.0


@dataclass
class RunReport:
    timestamp: str
    benchmarks: list[BenchmarkResult] = field(default_factory=list)
    models: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_single_benchmark(
    bench_key: str,
    extra_args: list[str],
    timeout_minutes: int = 30,
) -> BenchmarkResult:
    """Run a single benchmark script and capture its JSON output."""
    info = BENCHMARKS[bench_key]
    script_path = SCRIPT_DIR / info["script"]

    if not script_path.exists():
        return BenchmarkResult(
            benchmark=bench_key,
            label=info["label"],
            error=f"Script not found: {script_path}",
        )

    cmd = [sys.executable, str(script_path), "--json"] + extra_args

    sys.stdout.write(f"\n{'='*60}\n")
    sys.stdout.write(f"  Running {info['label']} ({info['script']})\n")
    sys.stdout.write(f"  {info['description']}\n")
    sys.stdout.write(f"{'='*60}\n")
    sys.stdout.flush()

    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_minutes * 60,
            cwd=str(SCRIPT_DIR),
        )
    except subprocess.TimeoutExpired:
        return BenchmarkResult(
            benchmark=bench_key,
            label=info["label"],
            error=f"Timed out after {timeout_minutes} minutes",
            duration_seconds=time.time() - start,
        )
    except Exception as exc:
        return BenchmarkResult(
            benchmark=bench_key,
            label=info["label"],
            error=str(exc),
            duration_seconds=time.time() - start,
        )

    duration = time.time() - start

    # Print stderr (progress output) to console
    if proc.stderr:
        sys.stderr.write(proc.stderr)

    if proc.returncode != 0:
        error_msg = proc.stderr.strip() if proc.stderr else f"Exit code {proc.returncode}"
        return BenchmarkResult(
            benchmark=bench_key,
            label=info["label"],
            error=error_msg,
            duration_seconds=duration,
        )

    # Parse JSON output — scripts may print progress lines before the JSON blob
    stdout = proc.stdout
    json_start = stdout.find("{")
    if json_start == -1:
        return BenchmarkResult(
            benchmark=bench_key,
            label=info["label"],
            error="No JSON object found in output",
            duration_seconds=duration,
        )
    # Show progress lines that preceded the JSON
    if json_start > 0:
        sys.stdout.write(stdout[:json_start])
        sys.stdout.flush()
    try:
        raw = json.loads(stdout[json_start:])
    except json.JSONDecodeError as exc:
        return BenchmarkResult(
            benchmark=bench_key,
            label=info["label"],
            error=f"Invalid JSON output: {exc}",
            duration_seconds=duration,
        )

    # Extract per-model scores
    model_scores: dict[str, dict[str, Any]] = {}
    scores_data = raw.get("scores", {})
    for model_id, score_info in scores_data.items():
        model_scores[model_id] = {
            "finalScore": score_info.get("finalScore", 0),
            "rating": score_info.get("rating", ""),
            "categoryScores": score_info.get("categoryScores", []),
            "totalPoints": score_info.get("totalPoints", score_info.get("totalScore", 0)),
            "maxPoints": score_info.get("maxPoints", score_info.get("maxScore", 0)),
        }

    return BenchmarkResult(
        benchmark=bench_key,
        label=info["label"],
        model_scores=model_scores,
        raw_json=raw,
        duration_seconds=duration,
    )


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------


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


def normalize_score(result: BenchmarkResult, model_id: str) -> int:
    """Normalize all benchmark scores to 0-100 scale."""
    score_info = result.model_scores.get(model_id)
    if not score_info:
        return 0
    return score_info.get("finalScore", 0)


def generate_text_report(report: RunReport) -> str:
    """Generate a human-readable text report."""
    lines: list[str] = []
    lines.append("")
    lines.append("=" * 70)
    lines.append("  BenchLocal - Full Benchmark Report")
    lines.append(f"  Generated: {report.timestamp}")
    lines.append("=" * 70)

    for model_id in report.models:
        lines.append("")
        lines.append(f"  Model: {model_id}")
        lines.append("-" * 70)
        lines.append("")

        # Per-benchmark scores
        scores: list[tuple[str, int]] = []
        lines.append(f"  {'Benchmark':<22} {'Score':>7}  {'Rating':<20}")
        lines.append(f"  {'-'*22} {'-'*7}  {'-'*20}")

        for result in report.benchmarks:
            score = normalize_score(result, model_id)
            scores.append((result.label, score))

            if result.error:
                lines.append(f"  {result.label:<22} {'ERROR':>7}  {result.error}")
            else:
                rating = rating_for_score(score)
                lines.append(f"  {result.label:<22} {score:>6}/100  {rating}")

        # Overall score
        valid_scores = [s for _, s in scores if s > 0 or not any(r.error for r in report.benchmarks if r.label == _)]
        if valid_scores:
            overall = round(sum(valid_scores) / len(valid_scores))
            lines.append(f"  {'-'*22} {'-'*7}  {'-'*20}")
            lines.append(f"  {'OVERALL':<22} {overall:>6}/100  {rating_for_score(overall)}")

        lines.append("")

        # Category breakdown per benchmark
        lines.append("  Category Breakdown:")
        lines.append("")

        for result in report.benchmarks:
            if result.error:
                continue
            score_info = result.model_scores.get(model_id, {})
            cats = score_info.get("categoryScores", [])
            if not cats:
                continue

            lines.append(f"    {result.label}:")
            for cat in cats:
                label = cat.get("label", cat.get("category", "?"))
                pct = cat.get("percent", cat.get("averageScore", cat.get("score", 0)))
                bar_len = pct // 5
                bar = "\u2588" * bar_len + "\u2591" * (20 - bar_len)
                lines.append(f"      {label:<28} {bar} {pct:>3}%")
            lines.append("")

    # Timing summary
    lines.append("-" * 70)
    lines.append("  Timing:")
    total_time = 0.0
    for result in report.benchmarks:
        mins = result.duration_seconds / 60
        total_time += result.duration_seconds
        status = "OK" if not result.error else "FAIL"
        lines.append(f"    {result.label:<22} {mins:>6.1f}m  [{status}]")
    lines.append(f"    {'TOTAL':<22} {total_time / 60:>6.1f}m")
    lines.append("=" * 70)
    lines.append("")

    return "\n".join(lines)


def generate_json_report(report: RunReport) -> dict[str, Any]:
    """Generate a structured JSON report."""
    model_summaries: dict[str, dict[str, Any]] = {}

    for model_id in report.models:
        bench_scores: dict[str, Any] = {}
        score_values: list[int] = []

        for result in report.benchmarks:
            score = normalize_score(result, model_id)
            entry: dict[str, Any] = {
                "score": score,
                "rating": rating_for_score(score),
                "duration_seconds": round(result.duration_seconds, 1),
            }
            if result.error:
                entry["error"] = result.error
            else:
                score_values.append(score)
                score_info = result.model_scores.get(model_id, {})
                entry["categoryScores"] = score_info.get("categoryScores", [])
                entry["totalPoints"] = score_info.get("totalPoints", 0)
                entry["maxPoints"] = score_info.get("maxPoints", 0)

            bench_scores[result.benchmark] = entry

        overall = round(sum(score_values) / len(score_values)) if score_values else 0

        model_summaries[model_id] = {
            "overallScore": overall,
            "overallRating": rating_for_score(overall),
            "benchmarks": bench_scores,
        }

    return {
        "timestamp": report.timestamp,
        "models": report.models,
        "benchmarkOrder": [r.benchmark for r in report.benchmarks],
        "results": model_summaries,
    }


def generate_markdown_report(report: RunReport) -> str:
    """Generate a Markdown report suitable for sharing."""
    lines: list[str] = []
    lines.append("# BenchLocal Benchmark Report")
    lines.append("")
    lines.append(f"**Generated:** {report.timestamp}")
    lines.append("")

    for model_id in report.models:
        lines.append(f"## Model: `{model_id}`")
        lines.append("")

        # Summary table
        lines.append("| Benchmark | Score | Rating |")
        lines.append("|-----------|------:|--------|")

        score_values: list[int] = []
        for result in report.benchmarks:
            if result.error:
                lines.append(f"| {result.label} | ERROR | {result.error} |")
            else:
                score = normalize_score(result, model_id)
                score_values.append(score)
                rating = rating_for_score(score)
                lines.append(f"| {result.label} | {score}/100 | {rating} |")

        if score_values:
            overall = round(sum(score_values) / len(score_values))
            lines.append(f"| **OVERALL** | **{overall}/100** | **{rating_for_score(overall)}** |")

        lines.append("")

        # Category details
        lines.append("### Category Breakdown")
        lines.append("")

        for result in report.benchmarks:
            if result.error:
                continue
            score_info = result.model_scores.get(model_id, {})
            cats = score_info.get("categoryScores", [])
            if not cats:
                continue

            lines.append(f"**{result.label}**")
            lines.append("")
            lines.append("| Category | Score |")
            lines.append("|----------|------:|")
            for cat in cats:
                label = cat.get("label", cat.get("category", "?"))
                pct = cat.get("percent", cat.get("averageScore", cat.get("score", 0)))
                lines.append(f"| {label} | {pct}% |")
            lines.append("")

    # Timing
    lines.append("### Timing")
    lines.append("")
    lines.append("| Benchmark | Duration | Status |")
    lines.append("|-----------|----------|--------|")
    total_time = 0.0
    for result in report.benchmarks:
        mins = result.duration_seconds / 60
        total_time += result.duration_seconds
        status = "OK" if not result.error else "FAIL"
        lines.append(f"| {result.label} | {mins:.1f}m | {status} |")
    lines.append(f"| **Total** | **{total_time / 60:.1f}m** | |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Report-Only Mode
# ---------------------------------------------------------------------------


def load_saved_results(results_dir: str) -> RunReport:
    """Load previously saved JSON results from a directory."""
    rdir = Path(results_dir)
    if not rdir.is_dir():
        print(f"Error: {results_dir} is not a directory.", file=sys.stderr)
        sys.exit(1)

    report = RunReport(timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    all_models: set[str] = set()

    for bench_key in BENCHMARK_ORDER:
        info = BENCHMARKS[bench_key]
        json_file = rdir / f"{bench_key}.json"
        if not json_file.exists():
            continue

        try:
            raw = json.loads(json_file.read_text(encoding="utf-8"))
        except Exception as exc:
            report.benchmarks.append(BenchmarkResult(
                benchmark=bench_key,
                label=info["label"],
                error=f"Failed to load {json_file}: {exc}",
            ))
            continue

        model_scores: dict[str, dict[str, Any]] = {}
        for model_id, score_info in raw.get("scores", {}).items():
            all_models.add(model_id)
            model_scores[model_id] = {
                "finalScore": score_info.get("finalScore", 0),
                "rating": score_info.get("rating", ""),
                "categoryScores": score_info.get("categoryScores", []),
                "totalPoints": score_info.get("totalPoints", score_info.get("totalScore", 0)),
                "maxPoints": score_info.get("maxPoints", score_info.get("maxScore", 0)),
            }

        report.benchmarks.append(BenchmarkResult(
            benchmark=bench_key,
            label=info["label"],
            model_scores=model_scores,
            raw_json=raw,
        ))

    report.models = sorted(all_models)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="BenchLocal Full Benchmark Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available benchmarks:
  toolcall15       ToolCall-15        Tool calling (15 scenarios)
  reasonmath15     ReasonMath-15      Math reasoning (15 scenarios)
  instructfollow15 InstructFollow-15  Instruction following (15 scenarios)
  dataextract15    DataExtract-15     Data extraction (15 scenarios)
  bugfind15        BugFind-15         Bug finding (15 scenarios)
  structoutput15   StructOutput-15    Structured output (15 scenarios)
  hermesagent20    HermesAgent-20     Agent capabilities (20 scenarios)

Examples:
  python run_benchmarks.py
  python run_benchmarks.py --bench toolcall15 --bench reasonmath15
  python run_benchmarks.py --model openrouter:openai/gpt-4.1 --json
  python run_benchmarks.py --report-only results/ --markdown
""",
    )
    parser.add_argument(
        "--bench",
        action="append",
        default=[],
        dest="benchmarks",
        choices=list(BENCHMARKS.keys()),
        help="Run specific benchmark(s). Repeat for multiple. Default: all.",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        dest="model_ids",
        help="Run specific model(s). Passed through to each benchmark script.",
    )
    parser.add_argument(
        "--models",
        default="",
        help="Comma-separated model IDs.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        default=[],
        dest="scenario_ids",
        help="Run specific scenario(s). Passed through to each benchmark script.",
    )
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None, dest="top_p")
    parser.add_argument("--top-k", type=int, default=None, dest="top_k")
    parser.add_argument("--min-p", type=float, default=None, dest="min_p")
    parser.add_argument("--repetition-penalty", type=float, default=None, dest="repetition_penalty")
    parser.add_argument("--timeout", type=int, default=None, help="Per-request timeout in seconds.")
    parser.add_argument(
        "--bench-timeout",
        type=int,
        default=30,
        dest="bench_timeout",
        help="Timeout per benchmark in minutes (default: 30).",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        default=False,
        help="Output report as JSON.",
    )
    parser.add_argument(
        "--markdown",
        action="store_true",
        default=False,
        help="Output report as Markdown.",
    )
    parser.add_argument(
        "--save-to",
        default="",
        dest="save_to",
        help="Directory to save individual benchmark JSON results.",
    )
    parser.add_argument(
        "--report-only",
        default="",
        dest="report_only",
        help="Skip running; generate report from saved results in this directory.",
    )
    parser.add_argument(
        "--show-raw",
        action="store_true",
        default=False,
        dest="show_raw",
        help="Pass --show-raw to each benchmark script.",
    )
    return parser


def build_passthrough_args(args: argparse.Namespace) -> list[str]:
    """Build CLI args to pass through to each benchmark script."""
    extra: list[str] = []
    for mid in args.model_ids:
        extra.extend(["--model", mid])
    if args.models:
        extra.extend(["--models", args.models])
    for sid in args.scenario_ids:
        extra.extend(["--scenario", sid])
    if args.temperature is not None:
        extra.extend(["--temperature", str(args.temperature)])
    if args.top_p is not None:
        extra.extend(["--top-p", str(args.top_p)])
    if args.top_k is not None:
        extra.extend(["--top-k", str(args.top_k)])
    if args.min_p is not None:
        extra.extend(["--min-p", str(args.min_p)])
    if args.repetition_penalty is not None:
        extra.extend(["--repetition-penalty", str(args.repetition_penalty)])
    if args.timeout is not None:
        extra.extend(["--timeout", str(args.timeout)])
    if args.show_raw:
        extra.append("--show-raw")
    return extra


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    # Report-only mode
    if args.report_only:
        report = load_saved_results(args.report_only)
        if args.json:
            print(json.dumps(generate_json_report(report), indent=2, ensure_ascii=False))
        elif args.markdown:
            print(generate_markdown_report(report))
        else:
            print(generate_text_report(report))
        return

    # Determine which benchmarks to run
    bench_keys = args.benchmarks if args.benchmarks else BENCHMARK_ORDER
    extra_args = build_passthrough_args(args)

    # Save directory
    save_dir: Path | None = None
    if args.save_to:
        save_dir = Path(args.save_to)
        save_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    report = RunReport(timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    all_models: set[str] = set()

    total_start = time.time()

    for bench_key in bench_keys:
        result = run_single_benchmark(bench_key, extra_args, args.bench_timeout)
        report.benchmarks.append(result)

        for mid in result.model_scores:
            all_models.add(mid)

        # Save individual result
        if save_dir and result.raw_json:
            out_path = save_dir / f"{bench_key}.json"
            out_path.write_text(
                json.dumps(result.raw_json, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        # Progress summary
        if result.error:
            sys.stdout.write(f"\n  {result.label}: FAILED - {result.error}\n")
        else:
            for mid, score_info in result.model_scores.items():
                sys.stdout.write(
                    f"\n  {result.label} [{mid}]: "
                    f"{score_info['finalScore']}/100 {score_info.get('rating', '')}\n"
                )
        sys.stdout.flush()

    total_elapsed = time.time() - total_start
    report.models = sorted(all_models)

    # Generate report
    if args.json:
        payload = generate_json_report(report)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    elif args.markdown:
        print(generate_markdown_report(report))
    else:
        print(generate_text_report(report))
        sys.stdout.write(f"Total elapsed: {total_elapsed / 60:.1f} minutes\n")

    # Save full report
    if save_dir:
        report_json = generate_json_report(report)
        (save_dir / "report.json").write_text(
            json.dumps(report_json, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        (save_dir / "report.md").write_text(
            generate_markdown_report(report),
            encoding="utf-8",
        )
        sys.stdout.write(f"\nResults saved to {save_dir}/\n")


if __name__ == "__main__":
    main()
