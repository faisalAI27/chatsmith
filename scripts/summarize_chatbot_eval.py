#!/usr/bin/env python3
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any


FLAG_NAMES = [
    "missing_sources",
    "missing_url",
    "missing_markdown_link",
    "missing_phone",
    "missing_email",
    "missing_hours",
    "missing_bullets",
    "missing_city_grouping",
    "possible_hallucination",
    "non_rag_mode",
    "very_short_answer",
    "raw_unformatted_long_answer",
]


def load_result_rows(path: str | Path) -> list[dict[str, Any]]:
    """Load evaluation results from CSV or JSONL."""
    result_path = Path(path)
    if result_path.suffix.lower() == ".jsonl":
        rows = []
        with result_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if result_path.suffix.lower() == ".csv":
        with result_path.open("r", encoding="utf-8", newline="") as file:
            return list(csv.DictReader(file))
    raise ValueError("Only .csv and .jsonl evaluation result files are supported")


def summarize_results(path: str | Path) -> dict[str, Any]:
    rows = load_result_rows(path)
    categories: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        categories[str(row.get("category") or "uncategorized")].append(row)

    latencies = [_to_float(row.get("latency_ms")) for row in rows]
    latencies = [latency for latency in latencies if latency is not None]
    manual_scores = [_parse_manual_score(row.get("manual_score")) for row in rows]
    manual_scores = [score for score in manual_scores if isinstance(score, (int, float))]

    summary = {
        "total_questions": len(rows),
        "answer_count": sum(1 for row in rows if str(row.get("answer") or "").strip()),
        "error_count": sum(1 for row in rows if str(row.get("error") or "").strip()),
        "average_latency_ms": round(mean(latencies), 2) if latencies else 0,
        "questions_by_category": dict(sorted(Counter(str(row.get("category") or "uncategorized") for row in rows).items())),
        "flags": {flag: _count_flag(rows, flag) for flag in FLAG_NAMES},
        "answers_with_urls": sum(1 for row in rows if _truthy(row.get("has_raw_url")) or str(row.get("detected_url") or "").strip()),
        "answers_with_markdown_links": sum(1 for row in rows if _truthy(row.get("has_clickable_markdown_link"))),
        "answers_missing_sources": _count_flag(rows, "missing_sources"),
        "possible_hallucinations": _count_flag(rows, "possible_hallucination"),
        "by_category": {},
        "manual_scores": {
            "average": round(mean(manual_scores), 2) if manual_scores else None,
            "counts": dict(Counter(str(row.get("manual_score") or "").strip() for row in rows if str(row.get("manual_score") or "").strip())),
        },
    }

    for category, category_rows in sorted(categories.items()):
        category_scores = [_parse_manual_score(row.get("manual_score")) for row in category_rows]
        category_scores = [score for score in category_scores if isinstance(score, (int, float))]
        summary["by_category"][category] = {
            "total": len(category_rows),
            "answers": sum(1 for row in category_rows if str(row.get("answer") or "").strip()),
            "errors": sum(1 for row in category_rows if str(row.get("error") or "").strip()),
            "flags": {flag: _count_flag(category_rows, flag) for flag in FLAG_NAMES},
            "answers_with_urls": sum(1 for row in category_rows if _truthy(row.get("has_raw_url")) or str(row.get("detected_url") or "").strip()),
            "answers_with_markdown_links": sum(1 for row in category_rows if _truthy(row.get("has_clickable_markdown_link"))),
            "average_manual_score": round(mean(category_scores), 2) if category_scores else None,
        }
    return summary


def _count_flag(rows: list[dict[str, Any]], flag: str) -> int:
    return sum(1 for row in rows if flag in _row_flags(row))


def _row_flags(row: dict[str, Any]) -> set[str]:
    flags = row.get("flags", [])
    if isinstance(flags, list):
        return {str(flag) for flag in flags}
    if isinstance(flags, str):
        return {flag.strip() for flag in flags.split(";") if flag.strip()}
    return set()


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_manual_score(value: Any) -> float | str | None:
    text = str(value or "").strip()
    if not text:
        return None
    if text.upper() == "N/A":
        return "N/A"
    try:
        return float(text)
    except ValueError:
        return None


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in {"true", "1", "yes"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize ChatSMITH chatbot evaluation results.")
    parser.add_argument("result_file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print(json.dumps(summarize_results(args.result_file), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
