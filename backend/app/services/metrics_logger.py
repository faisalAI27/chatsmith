import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from supabase import Client, create_client

# Default path can be overridden via METRICS_LOG_PATH; caller should guard with ENABLE_METRICS_LOGGING
LOG_PATH = Path(os.getenv("METRICS_LOG_PATH", "metrics_logs/chat_answers.jsonl"))

# Read Supabase settings (support common alt casing to avoid env mismatches)
SUPABASE_URL = (os.getenv("SUPABASE_URL") or os.getenv("supabase_url") or "").strip() or None
# Prefer service role key; fall back to anon for dev. Accept lowercase variants too.
SUPABASE_KEY = (
    os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    or os.getenv("supabase_service_role_key")
    or os.getenv("SUPABASE_ANON_KEY")
    or os.getenv("supabase_anon_key")
    or ""
).strip() or None
_supabase_client: Optional[Client] = None
_warned_no_supabase = False


def _metrics_enabled() -> bool:
    return (os.getenv("ENABLE_METRICS_LOGGING", "false") or "").strip().lower() == "true"


def metrics_enabled() -> bool:
    """Public helper to check if metrics are enabled."""
    return _metrics_enabled()


def get_supabase_client() -> Optional[Client]:
    """
    Return a Supabase client or None if not configured/initialization fails.
    """
    global _supabase_client
    global _warned_no_supabase
    if _supabase_client:
        return _supabase_client
    if not SUPABASE_URL or not SUPABASE_KEY:
        if _metrics_enabled() and not _warned_no_supabase:
            print("⚠️ Metrics Supabase not configured: missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY/ANON.")
            _warned_no_supabase = True
        return None
    try:
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as exc:
        print(f"⚠️ Metrics Supabase init failed: {exc}")
        _supabase_client = None
    return _supabase_client


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


def save_job_metrics_to_supabase(url: str, stats: dict, user_id: Optional[str] = None) -> None:
    """
    Insert a row into metrics_job_runs. Best-effort; quietly return on failure.
    """
    client = get_supabase_client()
    if not client:
        return
    try:
        payload = {
            "url": url,
            "cache_hit": bool(stats.get("cache_hit", False)),
            "tcr_seconds": float(stats.get("tcr_seconds", 0.0) or 0.0),
            "searches_run": int(stats.get("searches_run", 0) or 0),
            "pages_scraped": int(stats.get("pages_scraped", 0) or 0),
            "gaps_found": int(stats.get("gaps_found", 0) or 0),
            "user_id": user_id,
        }
        client.table("metrics_job_runs").insert(payload).execute()
    except Exception as exc:
        print(f"⚠️ Supabase job metrics insert failed: {exc}")


def save_chat_answer_to_supabase(
    question: str,
    answer: str,
    system_prompt: str,
    user_id: Optional[str] = None,
    url: Optional[str] = None,
) -> None:
    """
    Insert a chat answer row into metrics_chat_answers. Best-effort.
    """
    client = get_supabase_client()
    if not client:
        return
    try:
        provenance = "primary_plus_secondary" if "SECONDARY SOURCE" in (system_prompt or "") else "primary_only"
        payload = {
            "user_id": user_id,
            "url": url,
            "question": question,
            "answer": answer,
            "provenance": provenance,
        }
        client.table("metrics_chat_answers").insert(payload).execute()
    except Exception as exc:
        print(f"⚠️ Supabase chat metrics insert failed: {exc}")
