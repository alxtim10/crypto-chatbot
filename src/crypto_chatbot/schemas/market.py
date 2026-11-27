"""Market data schemas, UTC time semantics, and quality enums."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, model_validator


def _require_aware_utc(value: datetime) -> datetime:
    """Reject timezone-naive datetimes and normalize timezone-aware datetimes to UTC."""
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware (naive datetimes are rejected)")
    return value.astimezone(UTC)


def _require_finite(value: float) -> float:
    if not math.isfinite(value):
        raise ValueError("value must be finite")
    return value


UtcDatetime = Annotated[datetime, AfterValidator(_require_aware_utc)]
FiniteFloat = Annotated[float, AfterValidator(_require_finite)]


class MarketType(StrEnum):
    SPOT = "spot"
    PERPETUAL = "perpetual"


class Timeframe(StrEnum):
    M5 = "5m"
    M15 = "15m"
    H1 = "1h"
    H4 = "4h"
    D1 = "1d"

    @property
    def duration(self) -> timedelta:
        return _TIMEFRAME_DURATIONS[self]


PRIMARY_TIMEFRAMES: frozenset[Timeframe] = frozenset(
    {Timeframe.M15, Timeframe.H1, Timeframe.H4, Timeframe.D1}
)

TRADING_STYLE_ALIASES: dict[str, Timeframe] = {
    "scalp": Timeframe.M15,
    "scalping": Timeframe.M15,
    "intraday": Timeframe.H1,
    "daytrade": Timeframe.H1,
    "hourly": Timeframe.H1,
    "swing": Timeframe.H4,
    "position": Timeframe.D1,
    "macro": Timeframe.D1,
    "daily": Timeframe.D1,
}

TRADING_STYLE_LABELS: dict[Timeframe, str] = {
    Timeframe.M5: "MICRO (5m)",
    Timeframe.M15: "SCALP / Quick Intraday (15m → 5m)",
    Timeframe.H1: "INTRADAY / Hourly (1h → 15m → 5m)",
    Timeframe.H4: "SWING / Day-Swing (1d → 4h → 1h → 15m → 5m)",
    Timeframe.D1: "POSITION / Macro Swing (1d → 4h → 15m)",
}

MULTI_STYLE_SCAN_TIMEFRAMES: tuple[Timeframe, ...] = (
    Timeframe.M15,
    Timeframe.H1,
    Timeframe.H4,
)

TRADING_STYLE_SHORT_LABELS: dict[Timeframe, str] = {
    Timeframe.M5: "MICRO (5m)",
    Timeframe.M15: "SCALP (15m)",
    Timeframe.H1: "INTRADAY (1h)",
    Timeframe.H4: "SWING (4h)",
    Timeframe.D1: "POSITION (1d)",
}


def resolve_primary_timeframe(raw: str) -> Timeframe | None:
    """Resolve a primary timeframe ('15m','1h','4h','1d') or style ('scalp','intraday','swing')."""
    cleaned = raw.strip().lower()
    alias_tf = TRADING_STYLE_ALIASES.get(cleaned)
    if alias_tf is not None:
        return alias_tf
    try:
        tf = Timeframe(cleaned)
    except ValueError:
        return None
    return tf if tf in PRIMARY_TIMEFRAMES else None


def format_trading_style_label(timeframe: Timeframe) -> str:
    """Return human-readable trading style mode label for a primary timeframe."""
    return TRADING_STYLE_LABELS.get(timeframe, timeframe.value)


def format_trading_style_short(timeframe: Timeframe) -> str:
    """Return compact trading style badge (e.g. 'SCALP (15m)', 'INTRADAY (1h)', 'SWING (4h)')."""
    return TRADING_STYLE_SHORT_LABELS.get(timeframe, timeframe.value)


_TIMEFRAME_DURATIONS: dict[Timeframe, timedelta] = {
    Timeframe.M5: timedelta(minutes=5),
    Timeframe.M15: timedelta(minutes=15),
    Timeframe.H1: timedelta(hours=1),
    Timeframe.H4: timedelta(hours=4),
    Timeframe.D1: timedelta(days=1),
}


class QualityFlag(StrEnum):
    STALE = "STALE"
    GAP = "GAP"
    DUPLICATE = "DUPLICATE"
    INVALID_OHLC = "INVALID_OHLC"
    SHORT_HISTORY = "SHORT_HISTORY"
    FLAT_PRICE = "FLAT_PRICE"
    ZERO_VOLUME = "ZERO_VOLUME"
    SYMBOL_MISMATCH = "SYMBOL_MISMATCH"
    PROVIDER_ERROR = "PROVIDER_ERROR"
    DERIVATIVES_UNAVAILABLE = "DERIVATIVES_UNAVAILABLE"
    ORDERBOOK_UNAVAILABLE = "ORDERBOOK_UNAVAILABLE"
    SEQUENCE_GAP = "SEQUENCE_GAP"
    PRICE_MISMATCH = "PRICE_MISMATCH"
    SEVERE_PRICE_MISMATCH = "SEVERE_PRICE_MISMATCH"
    UNIT_MISMATCH = "UNIT_MISMATCH"
    BACKFILL = "BACKFILL"
    QUOTE_VOL_ESTIMATED = "QUOTE_VOL_ESTIMATED"


class QualityStatus(StrEnum):
    READY = "READY"
    DEGRADED = "DEGRADED"
    INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


BLOCKING_QUALITY_FLAGS: frozenset[QualityFlag] = frozenset(
    {
        QualityFlag.STALE,
        QualityFlag.GAP,
        QualityFlag.INVALID_OHLC,
        QualityFlag.SHORT_HISTORY,
        QualityFlag.SYMBOL_MISMATCH,
        QualityFlag.PROVIDER_ERROR,
        QualityFlag.DERIVATIVES_UNAVAILABLE,
        QualityFlag.ORDERBOOK_UNAVAILABLE,
        QualityFlag.PRICE_MISMATCH,
        QualityFlag.SEVERE_PRICE_MISMATCH,
        QualityFlag.UNIT_MISMATCH,
    }
)


def validate_symbol_market_type(
    canonical_symbol: str,
    provider_symbol: str,
    market_type: MarketType,
) -> None:
    """Prevent spot and perpetual symbols from being silently mixed."""
    has_settle = ":" in canonical_symbol
    is_perp_provider = provider_symbol.upper().endswith(".P")

    if market_type is MarketType.SPOT:
        if has_settle:
            raise ValueError(
                f"spot market cannot use perpetual canonical symbol '{canonical_symbol}'"
            )
        if is_perp_provider:
            raise ValueError(
                f"spot market cannot use perpetual provider symbol '{provider_symbol}'"
            )
        if "/" not in canonical_symbol:
            raise ValueError(
                f"spot canonical symbol must have BASE/QUOTE format: '{canonical_symbol}'"
            )
    elif market_type is MarketType.PERPETUAL and (not has_settle or "/" not in canonical_symbol):
        raise ValueError(
            f"perpetual canonical symbol must have BASE/QUOTE:SETTLE format: '{canonical_symbol}'"
        )


class ProviderMetadata(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    market_type: MarketType
    base_unit: str = Field(min_length=1)
    quote_currency: str = Field(min_length=1)


class MarketCandle(BaseModel):
    """Normalized OHLCV candle obeying PROJECT_PLAN §5.1-5.2."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    canonical_symbol: str = Field(min_length=1)
    provider_symbol: str = Field(min_length=1)
    market_type: MarketType
    timeframe: Timeframe
    open_time: UtcDatetime
    close_time: UtcDatetime
    received_time: UtcDatetime
    available_time: UtcDatetime
    open: FiniteFloat = Field(gt=0)
    high: FiniteFloat = Field(gt=0)
    low: FiniteFloat = Field(gt=0)
    close: FiniteFloat = Field(gt=0)
    raw_contracts: FiniteFloat | None = Field(default=None, ge=0)
    base_volume: FiniteFloat = Field(ge=0)
    quote_volume: FiniteFloat | None = Field(default=None, ge=0)
    taker_buy_base_volume: FiniteFloat | None = Field(default=None, ge=0)
    base_unit: str = Field(min_length=1)
    quote_currency: str = Field(min_length=1)
    is_closed: bool
    quality_flags: tuple[QualityFlag, ...] = ()

    @model_validator(mode="after")
    def _validate_invariants(self) -> MarketCandle:
        validate_symbol_market_type(
            self.canonical_symbol,
            self.provider_symbol,
            self.market_type,
        )

        expected_close = self.open_time + self.timeframe.duration
        if self.close_time != expected_close:
            raise ValueError(
                f"close_time ({self.close_time.isoformat()}) must equal "
                f"open_time + timeframe ({expected_close.isoformat()})"
            )

        if self.is_closed and self.close_time > self.received_time:
            raise ValueError("candle cannot be marked is_closed when close_time > received_time")

        if self.available_time < self.close_time:
            raise ValueError("available_time cannot be earlier than close_time")

        if self.high < max(self.open, self.close):
            raise ValueError("high must be >= max(open, close)")

        if self.low > min(self.open, self.close):
            raise ValueError("low must be <= min(open, close)")

        if self.low > self.high:
            raise ValueError("low must be <= high")

        if self.taker_buy_base_volume is not None and self.taker_buy_base_volume > (
            self.base_volume + 1e-6
        ):
            raise ValueError("taker_buy_base_volume cannot exceed base_volume")

        return self
