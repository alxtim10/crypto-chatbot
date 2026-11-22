"""Application settings loaded from environment variables (and an optional `.env` file).

Secrets are typed as `SecretStr` so they never appear in `repr()`, `str()`, or logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from crypto_chatbot.schemas.tradingview import TRADINGVIEW_PUBLISHED_WEBHOOK_IPS


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
    gemini_timeout_seconds: float = Field(default=15.0, gt=0)
    gemini_max_output_tokens: int = Field(default=1024, ge=64, le=4096)
    gemini_user_daily_cap: int = Field(default=30, ge=1)
    gemini_global_daily_cap: int = Field(default=200, ge=1)
    database_url: SecretStr
    allowed_telegram_user_ids: Annotated[tuple[int, ...], NoDecode] = Field(min_length=1)
    rate_limit_per_minute: int = Field(default=10, ge=1)
    tradingview_webhook_secret: SecretStr | None = None
    tradingview_webhook_port: int = Field(default=443, ge=1, le=65535)
    tradingview_allowed_ips: Annotated[tuple[str, ...], NoDecode] = (
        TRADINGVIEW_PUBLISHED_WEBHOOK_IPS
    )

    @field_validator("tradingview_webhook_secret", mode="before")
    @classmethod
    def _blank_optional_secret_to_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("tradingview_allowed_ips", mode="before")
    @classmethod
    def _parse_allowed_ips(cls, value: object) -> object:
        if isinstance(value, str):
            parts = tuple(part.strip() for part in value.split(",") if part.strip())
            return parts or TRADINGVIEW_PUBLISHED_WEBHOOK_IPS
        return value

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
