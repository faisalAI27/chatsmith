from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    openai_api_key: Optional[str] = None
    cors_allow_origins: Optional[str] = None
    enable_metrics_logging: bool = False
    scraper_render_mode: str = "auto"
    playwright_timeout_ms: int = 30000
    playwright_wait_ms: int = 1000
    playwright_block_heavy_resources: bool = True
    playwright_scroll_steps: int = 4
    playwright_content_wait_ms: int = 10000
    playwright_max_concurrent_pages: int = 2

    # Crawler (see backend/app/services/crawler.py)
    crawler_max_pages: int = 100
    crawler_max_depth: int = 5
    crawler_concurrency: int = 4
    crawler_politeness_delay_ms: int = 250
    crawler_respect_robots: bool = True
    crawler_include_subdomains: bool = False
    crawler_include_documents: bool = True
    crawler_max_documents: int = 10
    crawler_specific_page_max_pages: int = 5
    crawler_near_duplicate_distance: int = 3
    crawler_boilerplate_min_pages: int = 4
    crawler_boilerplate_ratio: float = 0.5
    crawler_time_budget_seconds: int = 900
    crawler_allow_private_hosts: bool = False

    # Trafilatura cross-check of each page's main text (coverage metric + safety net)
    extractor_trafilatura_check: bool = True

    # Optional extras around the crawl
    enable_web_search_supplement: bool = False
    knowledge_export_enabled: bool = True
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
