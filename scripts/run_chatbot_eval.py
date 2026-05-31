#!/usr/bin/env python3
import argparse
import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib import error, request


DEFAULT_QUESTION_FILE = "evaluation/questions/customer_service_smoke.json"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_OUTPUT_PREFIX = "evaluation/results/chatbot_eval"

URL_RE = re.compile(r"https?://[^\s<>)\"']+", re.IGNORECASE)
PHONE_RE = re.compile(r"(\+?\d[\d\s().-]{7,}\d)")
NOT_ENOUGH_RE = re.compile(
    r"(not enough information|does not provide enough information|do not have enough information|"
    r"doesn't provide enough information|not provided|not available|cannot determine)",
    re.IGNORECASE,
)

CSV_FIELDS = [
    "question_id",
    "category",
    "question",
    "answer",
    "mode",
    "source_count",
    "warnings",
    "flags",
    "latency_ms",
    "manual_score",
    "manual_notes",
    "issue_type",
]


def load_questions(
    question_file: str | Path,
    category: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load and optionally filter an evaluation question file."""
    path = Path(question_file)
    questions = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(questions, list):
        raise ValueError(f"Question file must contain a JSON list: {path}")

    filtered = [question for question in questions if isinstance(question, dict)]
    if category:
        filtered = [question for question in filtered if question.get("category") == category]
    if limit is not None:
        filtered = filtered[: max(0, int(limit))]
    return filtered


def build_chat_payload(website_id: str, question: str) -> dict[str, Any]:
    """Build the current /api/chat request shape."""
    return {
        "website_id": website_id,
        "question": question,
        "messages": [{"role": "user", "content": question}],
    }


def post_chat_question(
    base_url: str,
    website_id: str,
    question: str,
    timeout: int = 60,
) -> dict[str, Any]:
    """Send one question to the local ChatSMITH chat API."""
    endpoint = base_url.rstrip("/") + "/api/chat/"
    payload = json.dumps(build_chat_payload(website_id, question)).encode("utf-8")
    req = request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout) as response:
            body = response.read().decode("utf-8")
            return json.loads(body)
    except error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc
    except error.URLError as exc:
        raise RuntimeError(f"Request failed: {exc.reason}") from exc


def evaluate_questions(
    website_id: str,
    questions: list[dict[str, Any]],
    base_url: str = DEFAULT_BASE_URL,
    delay_seconds: float = 0.5,
    timeout: int = 60,
    include_debug: bool = False,
    chat_client=post_chat_question,
) -> list[dict[str, Any]]:
    """Run chatbot evaluation questions and return result records."""
    results = []
    for index, question_record in enumerate(questions):
        started = time.perf_counter()
        response: dict[str, Any] = {}
        error_message = ""
        question_text = str(question_record.get("question") or "")

        try:
            response = chat_client(
                base_url=base_url,
                website_id=website_id,
                question=question_text,
                timeout=timeout,
            )
        except Exception as exc:
            error_message = str(exc)

        latency_ms = int((time.perf_counter() - started) * 1000)
        result = build_result_record(
            website_id=website_id,
            question_record=question_record,
            response=response,
            latency_ms=latency_ms,
            error=error_message,
            include_debug=include_debug,
        )
        results.append(result)

        if delay_seconds and index < len(questions) - 1:
            time.sleep(max(0.0, float(delay_seconds)))

    return results


def build_result_record(
    website_id: str,
    question_record: dict[str, Any],
    response: dict[str, Any],
    latency_ms: int,
    error: str = "",
    include_debug: bool = False,
) -> dict[str, Any]:
    """Combine a question and API response into one reviewable result record."""
    answer = extract_answer(response)
    sources = response.get("sources") if isinstance(response.get("sources"), list) else []
    warnings = response.get("warnings") if isinstance(response.get("warnings"), list) else []
    heuristics = compute_heuristics(question_record, answer, response)

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "website_id": website_id,
        "question_id": question_record.get("id", ""),
        "category": question_record.get("category", ""),
        "intent": question_record.get("intent", ""),
        "question": question_record.get("question", ""),
        "expected_behavior": question_record.get("expected_behavior", ""),
        "answer": answer,
        "mode": response.get("mode", ""),
        "sources": sources,
        "warnings": warnings,
        "error": error,
        "latency_ms": latency_ms,
        "manual_score": "",
        "manual_notes": "",
        "issue_type": "",
        "requires_source": bool(question_record.get("requires_source", False)),
        "hallucination_sensitive": bool(question_record.get("hallucination_sensitive", False)),
        **heuristics,
    }
    if include_debug:
        record["retrieval_debug"] = response.get("retrieval_debug", {})
    return record


def extract_answer(response: dict[str, Any]) -> str:
    """Extract assistant answer text from current or legacy response shapes."""
    answer = response.get("answer")
    if isinstance(answer, str) and answer.strip():
        return answer.strip()
    message = response.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content.strip()
    return ""


def compute_heuristics(
    question_record: dict[str, Any],
    answer: str,
    response: dict[str, Any],
) -> dict[str, Any]:
    """Compute lightweight flags for manual review, not final judgment."""
    sources = response.get("sources") if isinstance(response.get("sources"), list) else []
    source_count = len(sources)
    category = str(question_record.get("category") or "")
    question = str(question_record.get("question") or "")
    question_lower = question.lower()
    answer_text = answer or ""
    mode = str(response.get("mode") or "")

    has_url = bool(URL_RE.search(answer_text))
    has_phone_like_pattern = bool(PHONE_RE.search(answer_text))
    says_not_enough_info = bool(NOT_ENOUGH_RE.search(answer_text))
    answer_length = len(answer_text.split())
    flags = []

    if question_record.get("requires_source") and source_count == 0:
        flags.append("missing_sources")
    if answer_length < 5:
        flags.append("short_answer")
    if question_record.get("hallucination_sensitive") and not says_not_enough_info and source_count == 0:
        flags.append("possible_hallucination")
    if category == "social_links" and "link" in question_lower and not has_url:
        flags.append("missing_url")
    if category == "contact_support" and "contact number" in question_lower and not has_phone_like_pattern:
        flags.append("missing_phone")
    if mode != "rag":
        flags.append("non_rag_mode")

    return {
        "source_count": source_count,
        "has_sources": source_count > 0,
        "says_not_enough_info": says_not_enough_info,
        "answer_length": answer_length,
        "has_url": has_url,
        "has_phone_like_pattern": has_phone_like_pattern,
        "possible_hallucination_flag": "possible_hallucination" in flags,
        "flags": flags,
    }


def write_results(results: list[dict[str, Any]], output_prefix: str | Path) -> dict[str, Path]:
    """Write JSONL, CSV, and Markdown reports."""
    prefix = Path(output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {
        "jsonl": prefix.with_suffix(".jsonl"),
        "csv": prefix.with_suffix(".csv"),
        "md": prefix.with_suffix(".md"),
    }
    write_jsonl(results, paths["jsonl"])
    write_csv(results, paths["csv"])
    write_markdown_report(results, paths["md"])
    return paths


def write_jsonl(results: list[dict[str, Any]], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as file:
        for result in results:
            file.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")


def write_csv(results: list[dict[str, Any]], path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {
                    "question_id": result.get("question_id", ""),
                    "category": result.get("category", ""),
                    "question": result.get("question", ""),
                    "answer": result.get("answer", ""),
                    "mode": result.get("mode", ""),
                    "source_count": result.get("source_count", 0),
                    "warnings": "; ".join(result.get("warnings") or []),
                    "flags": "; ".join(result.get("flags") or []),
                    "latency_ms": result.get("latency_ms", 0),
                    "manual_score": result.get("manual_score", ""),
                    "manual_notes": result.get("manual_notes", ""),
                    "issue_type": result.get("issue_type", ""),
                }
            )


def write_markdown_report(results: list[dict[str, Any]], path: str | Path) -> None:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(str(result.get("category") or "uncategorized"), []).append(result)

    lines = [
        "# Chatbot Evaluation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        f"Total questions: {len(results)}",
        "",
    ]
    for category, category_results in sorted(grouped.items()):
        lines.extend([f"## {category}", ""])
        for result in category_results:
            flags = ", ".join(result.get("flags") or []) or "none"
            lines.extend(
                [
                    f"### {result.get('question_id', '')}: {result.get('question', '')}",
                    "",
                    f"- Mode: {result.get('mode', '')}",
                    f"- Sources: {result.get('source_count', 0)}",
                    f"- Flags: {flags}",
                    f"- Latency: {result.get('latency_ms', 0)} ms",
                    "",
                    "**Answer**",
                    "",
                    str(result.get("answer") or result.get("error") or "").strip(),
                    "",
                    "**Manual review**",
                    "",
                    "- Score:",
                    "- Notes:",
                    "- Issue type:",
                    "",
                ]
            )
    Path(path).write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ChatSMITH chatbot evaluation questions.")
    parser.add_argument("--website-id", required=True)
    parser.add_argument("--question-file", default=DEFAULT_QUESTION_FILE)
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    parser.add_argument("--delay-seconds", type=float, default=0.5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--category")
    parser.add_argument("--include-debug", action="store_true")
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions = load_questions(args.question_file, category=args.category, limit=args.limit)
    results = evaluate_questions(
        website_id=args.website_id,
        questions=questions,
        base_url=args.base_url,
        delay_seconds=args.delay_seconds,
        timeout=args.timeout,
        include_debug=args.include_debug,
    )
    paths = write_results(results, args.output_prefix)
    print(f"Evaluated {len(results)} questions.")
    for label, path in paths.items():
        print(f"{label}: {path}")


if __name__ == "__main__":
    main()
