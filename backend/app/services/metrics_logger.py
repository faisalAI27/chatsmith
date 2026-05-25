import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Default path can be overridden via METRICS_LOG_PATH; caller should guard with ENABLE_METRICS_LOGGING
LOG_PATH = Path(os.getenv("METRICS_LOG_PATH", "metrics_logs/chat_answers.jsonl"))
JOB_LOG_PATH = Path(os.getenv("METRICS_JOB_LOG_PATH", "metrics_logs/job_runs.jsonl"))


def _metrics_enabled() -> bool:
    return (os.getenv("ENABLE_METRICS_LOGGING", "false") or "").strip().lower() == "true"


def metrics_enabled() -> bool:
    """Public helper to check if metrics are enabled."""
    return _metrics_enabled()


def log_chat_answer(
    question: str,
    answer: str,
    provenance: str,
    user: Optional[str] = None,
    log_path: Path = LOG_PATH,
) -> None:
    """
    Append a single chat Q/A record as JSONL for downstream accuracy sampling.
    """
    try:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "question": question,
            "answer": answer,
            "provenance": provenance,
            "user": user,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        # Best-effort logging; never break the request path
        print(f"⚠️ Metrics logging failed: {exc}")


def log_job_metrics(
    url: str,
    stats: dict,
    user_id: Optional[str] = None,
    log_path: Path = JOB_LOG_PATH,
) -> None:
    """Append a single job metrics record as JSONL."""
    try:
        log_path = Path(log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "url": url,
            "cache_hit": bool(stats.get("cache_hit", False)),
            "tcr_seconds": float(stats.get("tcr_seconds", 0.0) or 0.0),
            "searches_run": int(stats.get("searches_run", 0) or 0),
            "pages_scraped": int(stats.get("pages_scraped", 0) or 0),
            "gaps_found": int(stats.get("gaps_found", 0) or 0),
            "user_id": user_id,
        }
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        # Best-effort logging; never break the request path
        print(f"⚠️ Job metrics logging failed: {exc}")
