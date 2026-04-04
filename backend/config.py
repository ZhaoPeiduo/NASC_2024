from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Literal

class Settings(BaseSettings):
    # LLM backend: "openrouter" or "local"
    llm_provider: Literal["openrouter", "local"] = "openrouter"

    # OpenRouter
    openrouter_api_key: str = ""
    openrouter_model: str = "openai/gpt-4o-mini"

    # Local HuggingFace model
    local_model_name: str = "stabilityai/japanese-stablelm-instruct-gamma-7b"

    # Auth
    secret_key: SecretStr = SecretStr("dev-secret-change-in-production-!!!")
    access_token_expire_minutes: int = 60 * 24 * 7  # 1 week

    # Database
    database_url: str = "sqlite+aiosqlite:///./jlpt_sensei.db"

    # YouTube
    youtube_api_key: str = ""

    # CORS
    allowed_origins: list[str] = ["http://localhost:5173"]

    model_config = SettingsConfigDict(env_file=".env")

settings = Settings()
