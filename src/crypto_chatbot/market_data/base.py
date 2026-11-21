"""Narrow market-data provider contract and batch result schema."""

from __future__ import annotations

from datetime import timedelta
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.schemas.derivatives import (
    FundingRateBatch,
    LiquidationBatch,
    OpenInterestBatch,
    OrderBookBatch,
)
from crypto_chatbot.schemas.market import (
    MarketCandle,
    MarketType,
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


class DerivativesLiquidityProvider(Protocol):
    """Separate provider interface for perpetual derivatives and L2 order-book data."""

    async def get_funding_rate(self, symbol: str) -> FundingRateBatch: ...

    async def get_open_interest(
        self,
        symbol: str,
        timeframe: Timeframe = Timeframe.H4,
        history_limit: int = 30,
        *,
        reference_price: float | None = None,
    ) -> OpenInterestBatch: ...

    async def get_order_book(
        self,
        symbol: str,
        *,
        market_type: MarketType = MarketType.SPOT,
        limit: int = 100,
    ) -> OrderBookBatch: ...

    async def get_liquidations(
        self,
        symbol: str,
        *,
        limit: int = 50,
    ) -> LiquidationBatch: ...

    async def close(self) -> None: ...
