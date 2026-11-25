from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Any, List


class JobCreate(BaseModel):
    url: HttpUrl
    force_refresh: bool = False
    # Optional crawl budget for this job; defaults to CRAWLER_MAX_PAGES.
    max_pages: Optional[int] = Field(default=None, ge=1, le=2000)


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    stats: dict[str, Any] = {}
    errors: Optional[List[str]] = None
