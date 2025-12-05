from pydantic import BaseModel, HttpUrl
from typing import Optional, Any, List


class JobCreate(BaseModel):
    url: HttpUrl
    force_refresh: bool = False


class JobStatus(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    stats: dict[str, Any] = {}
    errors: Optional[List[str]] = None
