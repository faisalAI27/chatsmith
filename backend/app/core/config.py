from functools import lru_cache
from pydantic import BaseSettings, AnyHttpUrl
from typing import Optional


class Settings(BaseSettings):
    supabase_url: Optional[AnyHttpUrl] = None
    supabase_service_role_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    redis_url: str = "redis://localhost:6379/0"
    environment: str = "dev"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
