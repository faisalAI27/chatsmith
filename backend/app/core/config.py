from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: Optional[str] = None
    cors_allow_origins: Optional[str] = None
    enable_metrics_logging: bool = False
    scraper_render_mode: str = "auto"
    playwright_timeout_ms: int = 15000
    playwright_wait_ms: int = 1000
    playwright_block_heavy_resources: bool = True
    redis_url: str = "redis://localhost:6379/0"
    environment: str = "dev"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
