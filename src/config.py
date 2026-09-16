from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Typed, validated environment configuration.

    Failing fast here (missing/placeholder secrets raise at import time)
    beats discovering a misconfigured GEMINI_API_KEY or JWT_SECRET only
    when the first ticket is submitted or the first token is signed.
    """

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    gemini_api_key: str = Field(alias="GEMINI_API_KEY")
    jwt_secret: str = Field(alias="JWT_SECRET")
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = Field(default=60, alias="JWT_EXPIRE_MINUTES")
    database_url: str = Field(
        default=f"sqlite:///{BASE_DIR / 'app.db'}", alias="DATABASE_URL"
    )
    api_base_url: str = Field(default="http://127.0.0.1:8000", alias="API_BASE_URL")

    @field_validator("gemini_api_key")
    @classmethod
    def gemini_key_must_be_set(cls, v: str) -> str:
        if not v or v == "your-gemini-api-key-here":
            raise ValueError(
                "GEMINI_API_KEY is not set. Copy .env.example to .env and set a real key "
                "from https://aistudio.google.com/u/0/api-keys"
            )
        return v

    @field_validator("jwt_secret")
    @classmethod
    def jwt_secret_must_be_set(cls, v: str) -> str:
        if not v or v == "change-this-to-a-random-secret-string":
            raise ValueError(
                "JWT_SECRET is not set to a real value. Copy .env.example to .env and set "
                "a random secret."
            )
        return v


settings = Settings()

# Re-exported as module-level constants so existing `from src.config import X`
# call sites don't need to change.
GEMINI_API_KEY = settings.gemini_api_key
JWT_SECRET = settings.jwt_secret
JWT_ALGORITHM = settings.jwt_algorithm
JWT_EXPIRE_MINUTES = settings.jwt_expire_minutes
DATABASE_URL = settings.database_url
API_BASE_URL = settings.api_base_url

KNOWLEDGE_BASE_DIR = BASE_DIR / "knowledge_base"
RETRIEVAL_CACHE_DIR = BASE_DIR / "retrieval_cache"
