"""Narrow market-data provider contract and batch result schema."""

from __future__ import annotations

from datetime import timedelta
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.schemas.market import (
    MarketCandle,
    ProviderMetadata,
    QualityFlag,
    Timeframe,
    UtcDatetime,
)

DEFAULT_CANDLE_LIMIT = 1000
MAX_CANDLE_LIMIT = 1000
MIN_FULL_HISTORY_CANDLES = 600
MIN_DEGRADED_HISTORY_CANDLES = 150
DEFAULT_FRESHNESS_GRACE = timedelta(minutes=5)
DEFAULT_BACKFILL_AVAILABILITY_LAG = timedelta(seconds=60)


class OhlcvBatch(BaseModel):
    """Normalized closed-candle batch with explicit provenance and quality flags."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: ProviderMetadata
    canonical_symbol: str = Field(min_length=1)
    provider_symbol: str = Field(min_length=1)
    timeframe: Timeframe
    received_time: UtcDatetime
    candles: tuple[MarketCandle, ...] = ()
    quality_flags: tuple[QualityFlag, ...] = ()
    gap_intervals: tuple[tuple[UtcDatetime, UtcDatetime], ...] = ()
    duplicate_count: int = Field(default=0, ge=0)
    invalid_row_count: int = Field(default=0, ge=0)
    excluded_open_count: int = Field(default=0, ge=0)
    error_message: str | None = None

    @property
    def has_provider_error(self) -> bool:
        return QualityFlag.PROVIDER_ERROR in self.quality_flags


class MarketDataProvider(Protocol):
    """Narrow async interface for fetching closed OHLCV candles."""

    async def get_closed_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = DEFAULT_CANDLE_LIMIT,
    ) -> OhlcvBatch: ...

    async def close(self) -> None: ...
