"""Application settings loaded from environment variables or optional `.env`.

Secrets are typed as `SecretStr` so they never appear in `repr()`, `str()`, or logs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

from pydantic import (
    Field,
    SecretStr,
    ValidationError,
    field_validator,
)
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

from crypto_chatbot.schemas.ml import MlInferenceMode
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

    allowed_telegram_user_ids: Annotated[tuple[int, ...], NoDecode] = Field(
        default=(),
        validate_default=True,
    )
    rate_limit_per_minute: int = Field(default=10, ge=1)

    tradingview_webhook_secret: SecretStr | None = None
    tradingview_webhook_port: int = Field(default=443, ge=1, le=65535)
    tradingview_allowed_ips: Annotated[tuple[str, ...], NoDecode] = (
        TRADINGVIEW_PUBLISHED_WEBHOOK_IPS
    )
    ml_inference_mode: MlInferenceMode = MlInferenceMode.OFF
    futures_exchange: str = Field(default="gate")
    futures_exchange_fallbacks: Annotated[tuple[str, ...], NoDecode] = Field(
        default=(),
        validate_default=True,
    )
    primary_venue: str = Field(default="gateio")
    venue_fallbacks: Annotated[tuple[str, ...], NoDecode] = Field(
        default=(),
        validate_default=True,
    )
    binance_enabled: bool = Field(default=False)
    venue_cooldown_min: int = Field(default=360, ge=1, le=10080)
    gate_direct_candles: bool = Field(default=True)
    gate_contract_stats_enabled: bool = Field(default=False)
    gate_contract_stats_interval_seconds: int = Field(default=300, ge=10, le=3600)
    gate_contract_stats_backfill_days: int = Field(default=179, ge=1, le=180)

    @field_validator("futures_exchange", mode="before")
    @classmethod
    def _validate_futures_exchange(cls, value: object) -> str:
        if value is None or (isinstance(value, str) and not value.strip()):
            return "gate"
        cleaned = str(value).strip().lower()
        if cleaned == "gateio":
            return "gate"
        if cleaned not in {"binance", "gate", "bingx"}:
            raise ValueError("must be one of: binance, gate, gateio, bingx")
        return cleaned

    @field_validator("primary_venue", mode="before")
    @classmethod
    def _validate_primary_venue(cls, value: object) -> str:
        if value is None or (isinstance(value, str) and not value.strip()):
            return "gateio"
        cleaned = str(value).strip().lower()
        if cleaned not in {"binance", "gate", "gateio", "bingx"}:
            raise ValueError("must be one of: gateio, gate, bingx, binance")
        return cleaned

    @field_validator("futures_exchange_fallbacks", "venue_fallbacks", mode="before")
    @classmethod
    def _validate_futures_exchange_fallbacks(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        items: list[str]
        if isinstance(value, str):
            items = [part.strip().lower() for part in value.split(",") if part.strip()]
        elif isinstance(value, (tuple, list)):
            items = [str(part).strip().lower() for part in value if str(part).strip()]
        else:
            raise ValueError("must be a comma-separated list of exchanges")

        allowed = {"binance", "gate", "gateio", "bingx"}
        normalized: list[str] = []
        for item in items:
            if item not in allowed:
                raise ValueError(
                    f"unsupported fallback exchange '{item}'; "
                    "must be one of: binance, gate, gateio, bingx"
                )
            mapped = "gate" if item == "gateio" else item
            if mapped not in normalized:
                normalized.append(mapped)
        return tuple(normalized)

    @property
    def futures_exchange_chain(self) -> tuple[str, ...]:
        """Ordered, deduplicated exchange chain honoring D-01 defaults and legacy flags."""
        if (
            "futures_exchange" in self.model_fields_set
            or "futures_exchange_fallbacks" in self.model_fields_set
        ):
            return tuple(dict.fromkeys((self.futures_exchange, *self.futures_exchange_fallbacks)))

        primary_norm = "gate" if self.primary_venue == "gateio" else self.primary_venue
        raw_chain = tuple(dict.fromkeys((primary_norm, *self.venue_fallbacks)))
        if not self.binance_enabled:
            filtered = tuple(ex for ex in raw_chain if ex != "binance")
            if not filtered:
                raise ValueError(
                    "No enabled venue remains when binance_enabled is false; "
                    "set PRIMARY_VENUE=gateio or BINANCE_ENABLED=true"
                )
            return filtered
        return raw_chain

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
        parsed: tuple[int, ...]
        if isinstance(value, str):
            parts = [part.strip() for part in value.split(",") if part.strip()]
            try:
                parsed = tuple(int(part) for part in parts)
            except ValueError:
                raise ValueError("must be comma-separated integer Telegram user IDs") from None
        elif isinstance(value, (tuple, list)):
            parsed = tuple(int(v) for v in value)
        else:
            parsed = ()

        if not parsed:
            raise ValueError("must contain at least one Telegram user ID")
        return parsed


def collect_secret_values(settings: Settings) -> tuple[str, ...]:
    """Return all configured secret values for log redaction filters."""
    secrets = [
        settings.telegram_bot_token.get_secret_value(),
        settings.gemini_api_key.get_secret_value(),
        settings.database_url.get_secret_value(),
    ]
    if settings.tradingview_webhook_secret is not None:
        secrets.append(settings.tradingview_webhook_secret.get_secret_value())
    return tuple(secrets)


def load_settings(env_file: Path | None = Path(".env")) -> Settings:
    """Load settings from env or optional `.env` file without leaking secrets."""
    try:
        return Settings(_env_file=env_file)
    except ValidationError as exc:
        problems = []
        for error in exc.errors(include_input=False, include_url=False):
            loc = ".".join(str(part) for part in error["loc"])
            label = loc.upper() if loc else "SETTINGS"
            problems.append(f"{label}: {error['msg']}")
        raise ConfigError("Invalid configuration:\n  " + "\n  ".join(problems)) from None
