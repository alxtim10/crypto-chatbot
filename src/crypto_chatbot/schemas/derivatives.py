"""Typed schemas for Phase 9 derivatives, order-book liquidity, liquidations, and cross-checks."""

from __future__ import annotations

import math
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crypto_chatbot.schemas.market import (
    FiniteFloat,
    MarketType,
    QualityFlag,
    Timeframe,
    UtcDatetime,
    validate_symbol_market_type,
)

FUNDING_COMPARISON_INTERVAL_HOURS = 8.0
HOURS_PER_YEAR = 8760.0


class FundingRateObservation(BaseModel):
    """Venue-specific perpetual funding-rate snapshot with interval normalization."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    canonical_symbol: str = Field(min_length=1)
    provider_symbol: str = Field(min_length=1)
    market_type: MarketType = MarketType.PERPETUAL
    observed_time: UtcDatetime
    next_funding_time: UtcDatetime | None = None
    received_time: UtcDatetime
    available_time: UtcDatetime
    funding_rate: FiniteFloat
    funding_interval_hours: FiniteFloat = Field(gt=0)
    rate_per_8h: FiniteFloat
    annualized_rate: FiniteFloat
    mark_price: FiniteFloat | None = Field(default=None, gt=0)
    index_price: FiniteFloat | None = Field(default=None, gt=0)
    quality_flags: tuple[QualityFlag, ...] = ()
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_invariants(self) -> FundingRateObservation:
        if self.market_type is not MarketType.PERPETUAL:
            raise ValueError("funding rate observation must have market_type=perpetual")
        validate_symbol_market_type(
            self.canonical_symbol,
            self.provider_symbol,
            self.market_type,
        )
        if self.available_time < self.observed_time:
            raise ValueError("available_time cannot be earlier than observed_time")
        expected_8h = self.funding_rate * (
            FUNDING_COMPARISON_INTERVAL_HOURS / self.funding_interval_hours
        )
        expected_ann = self.funding_rate * (HOURS_PER_YEAR / self.funding_interval_hours)
        if not math.isclose(self.rate_per_8h, expected_8h, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError("rate_per_8h must match funding_rate * (8 / funding_interval_hours)")
        if not math.isclose(self.annualized_rate, expected_ann, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                "annualized_rate must match funding_rate * (8760 / funding_interval_hours)"
            )
        return self


class OpenInterestObservation(BaseModel):
    """Venue-specific perpetual open-interest record with normalized base & quote units."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    canonical_symbol: str = Field(min_length=1)
    provider_symbol: str = Field(min_length=1)
    market_type: MarketType = MarketType.PERPETUAL
    period: str = Field(min_length=1)
    observed_time: UtcDatetime
    received_time: UtcDatetime
    available_time: UtcDatetime
    open_interest_contracts: FiniteFloat = Field(ge=0)
    raw_unit: str = Field(min_length=1)
    contract_size: FiniteFloat = Field(gt=0)
    open_interest_base: FiniteFloat | None = Field(default=None, ge=0)
    base_unit: str = Field(min_length=1)
    open_interest_quote: FiniteFloat | None = Field(default=None, ge=0)
    quote_currency: str = Field(min_length=1)
    quality_flags: tuple[QualityFlag, ...] = ()
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_invariants(self) -> OpenInterestObservation:
        if self.market_type is not MarketType.PERPETUAL:
            raise ValueError("open interest observation must have market_type=perpetual")
        validate_symbol_market_type(
            self.canonical_symbol,
            self.provider_symbol,
            self.market_type,
        )
        if self.available_time < self.observed_time:
            raise ValueError("available_time cannot be earlier than observed_time")
        return self


class OrderBookLevel(BaseModel):
    """Single price/quantity level on one side of an order book."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    price: FiniteFloat = Field(gt=0)
    quantity: FiniteFloat = Field(gt=0)


class OrderBookDepthBand(BaseModel):
    """Cumulative bid/ask liquidity and imbalance within ±`band_bps` of mid price."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    band_bps: int = Field(gt=0)
    bid_depth_base: FiniteFloat = Field(ge=0)
    ask_depth_base: FiniteFloat = Field(ge=0)
    bid_depth_quote: FiniteFloat = Field(ge=0)
    ask_depth_quote: FiniteFloat = Field(ge=0)
    imbalance: FiniteFloat | None = Field(default=None, ge=-1.0, le=1.0)


class OrderBookDiff(BaseModel):
    """Incremental order-book diff update used for sequence-gap detection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    first_update_id: int = Field(ge=1)
    final_update_id: int = Field(ge=1)
    prev_final_update_id: int | None = Field(default=None, ge=1)
    event_time: UtcDatetime
    bids: tuple[tuple[FiniteFloat, FiniteFloat], ...] = ()
    asks: tuple[tuple[FiniteFloat, FiniteFloat], ...] = ()

    @model_validator(mode="after")
    def _validate_diff(self) -> OrderBookDiff:
        if self.final_update_id < self.first_update_id:
            raise ValueError("final_update_id must be >= first_update_id")
        for side_name, levels in (("bids", self.bids), ("asks", self.asks)):
            for price, qty in levels:
                if price <= 0 or qty < 0:
                    raise ValueError(f"{side_name} diff levels require price > 0 and quantity >= 0")
        return self


class OrderBookSnapshot(BaseModel):
    """Normalized L2 order-book snapshot with spread and fixed basis-point depth bands."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    canonical_symbol: str = Field(min_length=1)
    provider_symbol: str = Field(min_length=1)
    market_type: MarketType
    observed_time: UtcDatetime
    received_time: UtcDatetime
    available_time: UtcDatetime
    sequence_id: int | None = Field(default=None, ge=1)
    bids: tuple[OrderBookLevel, ...] = Field(min_length=1)
    asks: tuple[OrderBookLevel, ...] = Field(min_length=1)
    best_bid: FiniteFloat = Field(gt=0)
    best_ask: FiniteFloat = Field(gt=0)
    mid_price: FiniteFloat = Field(gt=0)
    spread_abs: FiniteFloat = Field(gt=0)
    spread_bps: FiniteFloat = Field(gt=0)
    depth_bands: tuple[OrderBookDepthBand, ...] = ()
    base_unit: str = Field(min_length=1)
    quote_currency: str = Field(min_length=1)
    resync_count: int = Field(default=0, ge=0)
    quality_flags: tuple[QualityFlag, ...] = ()
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_book(self) -> OrderBookSnapshot:
        validate_symbol_market_type(
            self.canonical_symbol,
            self.provider_symbol,
            self.market_type,
        )
        if self.best_bid >= self.best_ask:
            raise ValueError("crossed or locked order book rejected (best_bid >= best_ask)")
        if self.available_time < self.observed_time:
            raise ValueError("available_time cannot be earlier than observed_time")
        return self


class LiquidationSide(StrEnum):
    LONG_LIQUIDATED = "long_liquidated"
    SHORT_LIQUIDATED = "short_liquidated"


class LiquidationEvent(BaseModel):
    """Single normalized forced-liquidation event from a reliable venue feed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    canonical_symbol: str = Field(min_length=1)
    provider_symbol: str = Field(min_length=1)
    market_type: MarketType = MarketType.PERPETUAL
    event_time: UtcDatetime
    received_time: UtcDatetime
    available_time: UtcDatetime
    side: LiquidationSide
    price: FiniteFloat = Field(gt=0)
    base_quantity: FiniteFloat = Field(gt=0)
    quote_notional: FiniteFloat = Field(gt=0)
    base_unit: str = Field(min_length=1)
    quote_currency: str = Field(min_length=1)
    event_key: str = Field(min_length=1)
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_event(self) -> LiquidationEvent:
        if self.market_type is not MarketType.PERPETUAL:
            raise ValueError("liquidation event must have market_type=perpetual")
        validate_symbol_market_type(
            self.canonical_symbol,
            self.provider_symbol,
            self.market_type,
        )
        if self.available_time < self.event_time:
            raise ValueError("available_time cannot be earlier than event_time")
        return self


class ReferencePrice(BaseModel):
    """Independent reference price used for cross-source divergence detection."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    provider: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    canonical_symbol: str = Field(min_length=1)
    market_type: MarketType
    price: FiniteFloat = Field(gt=0)
    observed_time: UtcDatetime
    received_time: UtcDatetime
    available_time: UtcDatetime


class CrossSourcePriceCheck(BaseModel):
    """Deterministic divergence check between two contemporaneous price sources."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source_a: str = Field(min_length=1)
    price_a: FiniteFloat = Field(gt=0)
    time_a: UtcDatetime
    source_b: str = Field(min_length=1)
    price_b: FiniteFloat = Field(gt=0)
    time_b: UtcDatetime
    time_diff_seconds: FiniteFloat = Field(ge=0)
    deviation_bps: FiniteFloat = Field(ge=0)
    status: Literal["ok", "degrade", "block", "non_comparable_time"]


class FundingRateBatch(BaseModel):
    """Provider batch wrapper for a funding-rate fetch."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_symbol: str = Field(min_length=1)
    provider_symbol: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    received_time: UtcDatetime
    observation: FundingRateObservation | None = None
    quality_flags: tuple[QualityFlag, ...] = ()
    error_message: str | None = None

    @property
    def has_provider_error(self) -> bool:
        return QualityFlag.PROVIDER_ERROR in self.quality_flags


class OpenInterestBatch(BaseModel):
    """Provider batch wrapper for live + historical open-interest fetches."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_symbol: str = Field(min_length=1)
    provider_symbol: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    timeframe: Timeframe
    received_time: UtcDatetime
    latest: OpenInterestObservation | None = None
    history: tuple[OpenInterestObservation, ...] = ()
    quality_flags: tuple[QualityFlag, ...] = ()
    error_message: str | None = None

    @property
    def has_provider_error(self) -> bool:
        return QualityFlag.PROVIDER_ERROR in self.quality_flags


class OrderBookBatch(BaseModel):
    """Provider batch wrapper for an order-book snapshot fetch (or diff resync)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_symbol: str = Field(min_length=1)
    provider_symbol: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    market_type: MarketType
    received_time: UtcDatetime
    snapshot: OrderBookSnapshot | None = None
    resync_count: int = Field(default=0, ge=0)
    quality_flags: tuple[QualityFlag, ...] = ()
    error_message: str | None = None

    @property
    def has_provider_error(self) -> bool:
        return QualityFlag.PROVIDER_ERROR in self.quality_flags


class LiquidationBatch(BaseModel):
    """Provider batch wrapper for forced-liquidation events (explicit source availability)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_symbol: str = Field(min_length=1)
    provider_symbol: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    received_time: UtcDatetime
    source_available: bool = False
    events: tuple[LiquidationEvent, ...] = ()
    quality_flags: tuple[QualityFlag, ...] = ()
    error_message: str | None = None

    @property
    def has_provider_error(self) -> bool:
        return QualityFlag.PROVIDER_ERROR in self.quality_flags


class DerivativesContext(BaseModel):
    """Optional Phase 9 market microstructure & derivatives inputs passed to `AnalysisService`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    funding: FundingRateBatch | None = None
    open_interest: OpenInterestBatch | None = None
    order_book: OrderBookBatch | None = None
    liquidations: LiquidationBatch | None = None
    reference_prices: tuple[ReferencePrice, ...] = ()


class OrderFlowMetrics(BaseModel):
    """Closed-bar order flow metrics calculated from taker buy aggression."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    taker_buy_ratio: FiniteFloat = Field(ge=0.0, le=1.0)
    cvd_20: FiniteFloat
    cvd_divergence: Literal["bullish", "bearish", "none"] = "none"
    has_order_flow: bool = True


class CrossVenueDerivativesSummary(BaseModel):
    """Comparative derivatives pricing and funding dispersion across venues."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    venues: tuple[str, ...]
    primary_venue: str
    funding_dispersion_bps: FiniteFloat | None = None
    mark_price_spread_bps: FiniteFloat | None = None
    coverage_status: Literal["full", "partial", "unavailable"] = "unavailable"
