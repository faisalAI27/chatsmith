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
    vector_db_provider: str = "chroma"
    chroma_db_dir: str = "backend/vector_store/chroma"
    chroma_collection_name: str = "website_chunks"
    embedding_provider: str = "openai"
    embedding_model: str = "text-embedding-3-small"
    embedding_batch_size: int = 64
    retrieval_top_k: int = 5
    hybrid_retrieval_enabled: bool = True
    hybrid_lexical_candidate_limit: int = 2000
    hybrid_vector_candidate_multiplier: int = 4
    redis_url: str = "redis://localhost:6379/0"
    environment: str = "dev"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
