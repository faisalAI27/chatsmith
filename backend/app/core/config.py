from functools import lru_cache
from typing import Optional

from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    supabase_url: Optional[AnyHttpUrl] = None
    supabase_service_role_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    redis_url: str = "redis://localhost:6379/0"
    environment: str = "dev"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
