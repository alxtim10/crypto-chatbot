"""Application settings loaded from environment variables (and an optional `.env` file).

Secrets are typed as `SecretStr` so they never appear in `repr()`, `str()`, or logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class ConfigError(Exception):
    """Raised when required configuration is missing or invalid. Never contains secret values."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,
    )

    telegram_bot_token: SecretStr
    gemini_api_key: SecretStr
    gemini_model: str = Field(min_length=1)
    database_url: SecretStr
    allowed_telegram_user_ids: Annotated[tuple[int, ...], NoDecode] = Field(min_length=1)

    @field_validator("telegram_bot_token", "gemini_api_key")
    @classmethod
    def _secret_not_blank(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().strip():
            raise ValueError("must not be blank")
        return value

    @field_validator("database_url")
    @classmethod
    def _postgres_url(cls, value: SecretStr) -> SecretStr:
        if not value.get_secret_value().startswith("postgresql"):
            raise ValueError("must be a PostgreSQL URL (postgresql+psycopg://...)")
        return value

    @field_validator("allowed_telegram_user_ids", mode="before")
    @classmethod
    def _parse_user_ids(cls, value: object) -> object:
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            try:
                return tuple(int(part) for part in parts)
            except ValueError:
                raise ValueError("must be comma-separated integer Telegram user IDs") from None
        return value


def load_settings(env_file: Path | None = Path(".env")) -> Settings:
    """Load settings, converting validation errors into a `ConfigError` without input values."""
    try:
        return Settings(_env_file=env_file)
    except ValidationError as exc:
        problems = []
        for error in exc.errors(include_input=False, include_url=False):
            field = ".".join(str(part) for part in error["loc"])
            problems.append(f"{field.upper()}: {error['msg']}")
        raise ConfigError("Invalid configuration:\n  " + "\n  ".join(problems)) from None
