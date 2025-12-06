from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import asyncio

from ..models.jobs import JobCreate, JobStatus
from ..services import scrape_pipeline
from ..services.metrics_logger import save_job_metrics_to_supabase
from ..services.scrape_pipeline import ENABLE_METRICS

router = APIRouter()


@router.post("/run", response_model=JobStatus, summary="Run pipeline synchronously (dev only)")
async def run_job(body: JobCreate) -> JobStatus:
    """
    Temporary endpoint to exercise the pipeline synchronously.
    In production this should enqueue a background job.
    """
    try:
        status_text, system_prompt, name, history, msg_update, send_update, stats = await scrape_pipeline.run_full_research_new(
            str(body.url), force_refresh=body.force_refresh
        )
        stats_out: Dict[str, Any] = {
            "status_text": status_text,
            "name": name,
            "history": history,
            "system_prompt": system_prompt,
            "searches_run": stats.get("searches_run", 0),
            "pages_scraped": stats.get("pages_scraped", 0),
            "gaps_found": stats.get("gaps_found", 0),
            "tcr_seconds": stats.get("tcr_seconds", 0.0),
            "cache_hit": bool(stats.get("cache_hit", False)),
        }
        if ENABLE_METRICS:
            try:
                save_job_metrics_to_supabase(
                    url=str(body.url),
                    stats=stats,
                    user_id=None,  # user id not available in this dev endpoint
                )
            except Exception as exc:
                print(f"⚠️ Metrics logging skipped: {exc}")

        return JobStatus(
            job_id="dev-inline",
            status="completed",
            progress=100.0,
            stats=stats_out,
            errors=None,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
