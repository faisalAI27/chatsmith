import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Default path can be overridden via METRICS_LOG_PATH; caller should guard with ENABLE_METRICS_LOGGING
LOG_PATH = Path(os.getenv("METRICS_LOG_PATH", "metrics_logs/chat_answers.jsonl"))


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
