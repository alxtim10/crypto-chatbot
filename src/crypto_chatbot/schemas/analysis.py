"""Deterministic analysis schemas: facts, technical metrics, events, and AnalysisSnapshot."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crypto_chatbot.schemas.market import (
    BLOCKING_QUALITY_FLAGS,
    FiniteFloat,
    MarketType,
    QualityFlag,
    QualityStatus,
    Timeframe,
    UtcDatetime,
    validate_symbol_market_type,
)
from crypto_chatbot.schemas.news import SentimentMetrics
from crypto_chatbot.schemas.tradingview import TradingViewMetrics

FACT_ID_PATTERN = r"^F[1-9][0-9]*$"


class Regime(StrEnum):
    HIGH_VOLATILITY = "high_volatility"
    TRENDING_UP = "trending_up"
    TRENDING_DOWN = "trending_down"
    RANGING = "ranging"
    TRANSITIONAL = "transitional"


class MarketStructure(StrEnum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    MIXED = "mixed"


class MtfRsiAlignment(StrEnum):
    ALIGNED_BULLISH = "aligned_bullish"
    ALIGNED_BEARISH = "aligned_bearish"
    MIXED = "mixed"


class FactorDirection(StrEnum):
    SUPPORTS_BULLISH = "supports_bullish"
    SUPPORTS_BEARISH = "supports_bearish"
    NEUTRAL = "neutral"


class StructureEventType(StrEnum):
    SWING_HIGH = "swing_high"
    SWING_LOW = "swing_low"
    FAILED_BREAKOUT_BULLISH = "failed_breakout_bullish"
    FAILED_BREAKOUT_BEARISH = "failed_breakout_bearish"
    BOS_BULLISH = "bos_bullish"
    BOS_BEARISH = "bos_bearish"
    CHOCH_BULLISH = "choch_bullish"
    CHOCH_BEARISH = "choch_bearish"
    LIQUIDITY_SWEEP_BULLISH = "liquidity_sweep_bullish"
    LIQUIDITY_SWEEP_BEARISH = "liquidity_sweep_bearish"


class DealingZone(StrEnum):
    DISCOUNT = "discount"
    EQUILIBRIUM = "equilibrium"
    PREMIUM = "premium"


class PriceZone(BaseModel):
    """Causal unmitigated FVG or Order Block price zone."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    zone_type: Literal["bullish_fvg", "bearish_fvg", "bullish_ob", "bearish_ob"]
    zone_low: FiniteFloat = Field(gt=0)
    zone_high: FiniteFloat = Field(gt=0)
    event_time: UtcDatetime
    confirmation_time: UtcDatetime
    mitigated: bool = False

    @model_validator(mode="after")
    def _validate_zone(self) -> PriceZone:
        if self.zone_high < self.zone_low:
            raise ValueError("zone_high must be >= zone_low")
        if self.confirmation_time < self.event_time:
            raise ValueError("confirmation_time cannot be earlier than event_time")
        return self


class SmcSummary(BaseModel):
    """Hybrid causal Smart Money Concepts measurements for crypto."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    latest_structure_break: StructureEvent | None = None
    latest_sweep: StructureEvent | None = None
    active_bullish_fvg: PriceZone | None = None
    active_bearish_fvg: PriceZone | None = None
    active_bullish_ob: PriceZone | None = None
    active_bearish_ob: PriceZone | None = None
    dealing_range_high: FiniteFloat | None = Field(default=None, gt=0)
    dealing_range_low: FiniteFloat | None = Field(default=None, gt=0)
    equilibrium: FiniteFloat | None = Field(default=None, gt=0)
    position_in_range: FiniteFloat | None = None
    dealing_zone: DealingZone | None = None


class TimeframeOverview(BaseModel):
    """Deterministic summary of structure, SMC, and S/R for one timeframe in the top-down chain."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeframe: Timeframe
    trend_label: str | None = None
    market_structure: MarketStructure | None = None
    structure_break: StructureEventType | None = None
    structure_break_level: FiniteFloat | None = Field(default=None, gt=0)
    sweep: StructureEventType | None = None
    dealing_zone: DealingZone | None = None
    support_level: FiniteFloat | None = Field(default=None, gt=0)
    resistance_level: FiniteFloat | None = Field(default=None, gt=0)
    active_poi: str | None = None
    rsi_14: FiniteFloat | None = Field(default=None, ge=0, le=100)


class SetupStatus(StrEnum):
    READY = "ready"
    WAIT = "wait"


class SetupBias(StrEnum):
    LONG = "long"
    SHORT = "short"
    NEUTRAL = "neutral"


class TradeSetup(BaseModel):
    """Deterministic conditional trade scenario (Entry, SL, TP 1-3) or explicit WAIT."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: SetupStatus
    bias: SetupBias
    is_probabilistic: Literal[False] = False
    entry_low: FiniteFloat | None = Field(default=None, gt=0)
    entry_high: FiniteFloat | None = Field(default=None, gt=0)
    entry_mid: FiniteFloat | None = Field(default=None, gt=0)
    entry_source: str | None = None
    entry_timeframe: Timeframe | None = None
    ltf_confirmation: str | None = None
    htf_support: FiniteFloat | None = Field(default=None, gt=0)
    htf_resistance: FiniteFloat | None = Field(default=None, gt=0)
    ltf_support: FiniteFloat | None = Field(default=None, gt=0)
    ltf_resistance: FiniteFloat | None = Field(default=None, gt=0)
    stop_loss: FiniteFloat | None = Field(default=None, gt=0)
    invalidation_reason: str | None = None
    tp1: FiniteFloat | None = Field(default=None, gt=0)
    tp1_label: str | None = None
    tp2: FiniteFloat | None = Field(default=None, gt=0)
    tp2_label: str | None = None
    tp3: FiniteFloat | None = Field(default=None, gt=0)
    tp3_label: str | None = None
    rr_tp1: FiniteFloat | None = Field(default=None, gt=0)
    rr_tp2: FiniteFloat | None = Field(default=None, gt=0)
    rr_tp3: FiniteFloat | None = Field(default=None, gt=0)
    wait_reason: str | None = None
    risk_notes: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_setup_levels(self) -> TradeSetup:
        if self.status is SetupStatus.WAIT:
            if not self.wait_reason or not self.wait_reason.strip():
                raise ValueError("wait_reason is required when setup status is WAIT")
            return self

        if self.bias not in (SetupBias.LONG, SetupBias.SHORT):
            raise ValueError("READY trade setup must have bias LONG or SHORT")

        if (
            self.entry_low is None
            or self.entry_high is None
            or self.entry_mid is None
            or self.stop_loss is None
            or self.tp1 is None
            or self.tp2 is None
            or self.tp3 is None
            or self.rr_tp1 is None
            or self.rr_tp2 is None
            or self.rr_tp3 is None
        ):
            raise ValueError("READY trade setup requires all entry, SL, TP 1-3, and R:R values")

        if not (self.entry_low <= self.entry_mid <= self.entry_high):
            raise ValueError("require entry_low <= entry_mid <= entry_high")

        if self.bias is SetupBias.LONG:
            if not (self.stop_loss < self.entry_low):
                raise ValueError("LONG setup requires stop_loss < entry_low")
            if not (self.entry_mid < self.tp1 < self.tp2 < self.tp3):
                raise ValueError("LONG setup requires entry_mid < tp1 < tp2 < tp3")
        else:
            if not (self.stop_loss > self.entry_high):
                raise ValueError("SHORT setup requires stop_loss > entry_high")
            if not (self.entry_mid > self.tp1 > self.tp2 > self.tp3):
                raise ValueError("SHORT setup requires entry_mid > tp1 > tp2 > tp3")
        return self


class Fact(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    fact_id: str = Field(pattern=FACT_ID_PATTERN)
    label: str = Field(min_length=1)
    value: FiniteFloat
    unit: str = Field(min_length=1)
    display_format: str = Field(min_length=1)
    source: str = Field(min_length=1)
    as_of: UtcDatetime


class InputRange(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    first_open_time: UtcDatetime
    last_open_time: UtcDatetime
    candle_count: int = Field(ge=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def _validate_order(self) -> InputRange:
        if self.last_open_time < self.first_open_time:
            raise ValueError("last_open_time cannot be earlier than first_open_time")
        return self


class StructureEvent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_type: StructureEventType
    event_time: UtcDatetime
    confirmation_time: UtcDatetime
    level: FiniteFloat = Field(gt=0)

    @model_validator(mode="after")
    def _validate_confirmation(self) -> StructureEvent:
        if self.confirmation_time < self.event_time:
            raise ValueError("confirmation_time cannot be earlier than event_time")
        return self


class FactorItem(BaseModel):
    """Non-probabilistic rule output referencing backend facts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str = Field(min_length=1)
    description: str = Field(min_length=1)
    direction: FactorDirection
    fact_ids: tuple[str, ...] = Field(min_length=1)
    is_probabilistic: Literal[False] = False


class TechnicalMetrics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    last_close: FiniteFloat = Field(gt=0)
    return_1: FiniteFloat | None = None
    return_20: FiniteFloat | None = None
    ema_20: FiniteFloat | None = Field(default=None, gt=0)
    ema_50: FiniteFloat | None = Field(default=None, gt=0)
    ema_200: FiniteFloat | None = Field(default=None, gt=0)
    rsi_14: FiniteFloat | None = Field(default=None, ge=0, le=100)
    macd_line: FiniteFloat | None = None
    macd_signal: FiniteFloat | None = None
    macd_histogram: FiniteFloat | None = None
    atr_14: FiniteFloat | None = Field(default=None, ge=0)
    atr_pct: FiniteFloat | None = Field(default=None, ge=0)
    adx_14: FiniteFloat | None = Field(default=None, ge=0, le=100)
    plus_di_14: FiniteFloat | None = Field(default=None, ge=0, le=100)
    minus_di_14: FiniteFloat | None = Field(default=None, ge=0, le=100)
    relative_volume_20: FiniteFloat | None = Field(default=None, ge=0)
    atr_percentile_100: FiniteFloat | None = Field(default=None, ge=0, le=100)
    mtf_rsi_alignment: MtfRsiAlignment | None = None
    prev_day_high: FiniteFloat | None = Field(default=None, gt=0)
    prev_day_low: FiniteFloat | None = Field(default=None, gt=0)
    prev_day_close: FiniteFloat | None = Field(default=None, gt=0)
    prev_week_high: FiniteFloat | None = Field(default=None, gt=0)
    prev_week_low: FiniteFloat | None = Field(default=None, gt=0)
    prev_week_close: FiniteFloat | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def _validate_period_levels(self) -> TechnicalMetrics:
        if (
            self.prev_day_high is not None
            and self.prev_day_low is not None
            and self.prev_day_high < self.prev_day_low
        ):
            raise ValueError("prev_day_high must be >= prev_day_low")
        if (
            self.prev_week_high is not None
            and self.prev_week_low is not None
            and self.prev_week_high < self.prev_week_low
        ):
            raise ValueError("prev_week_high must be >= prev_week_low")
        return self


class DerivativesMetrics(BaseModel):
    """Venue-attributed perpetual derivatives, order-book liquidity, and cross-source checks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    funding_venue: str | None = None
    funding_symbol: str | None = None
    funding_rate: FiniteFloat | None = None
    funding_interval_hours: FiniteFloat | None = Field(default=None, gt=0)
    funding_rate_per_8h: FiniteFloat | None = None
    funding_rate_annualized: FiniteFloat | None = None
    next_funding_time: UtcDatetime | None = None
    mark_price: FiniteFloat | None = Field(default=None, gt=0)
    index_price: FiniteFloat | None = Field(default=None, gt=0)
    mark_index_basis_bps: FiniteFloat | None = None

    open_interest_venue: str | None = None
    open_interest_symbol: str | None = None
    open_interest_base: FiniteFloat | None = Field(default=None, ge=0)
    open_interest_base_unit: str | None = None
    open_interest_quote: FiniteFloat | None = Field(default=None, ge=0)
    open_interest_quote_currency: str | None = None
    open_interest_change_pct: FiniteFloat | None = None
    open_interest_change_window: str | None = None

    orderbook_venue: str | None = None
    orderbook_symbol: str | None = None
    orderbook_market_type: MarketType | None = None
    best_bid: FiniteFloat | None = Field(default=None, gt=0)
    best_ask: FiniteFloat | None = Field(default=None, gt=0)
    mid_price: FiniteFloat | None = Field(default=None, gt=0)
    spread_abs: FiniteFloat | None = Field(default=None, gt=0)
    spread_bps: FiniteFloat | None = Field(default=None, gt=0)
    depth_10bps_bid_quote: FiniteFloat | None = Field(default=None, ge=0)
    depth_10bps_ask_quote: FiniteFloat | None = Field(default=None, ge=0)
    imbalance_10bps: FiniteFloat | None = Field(default=None, ge=-1.0, le=1.0)
    depth_50bps_bid_quote: FiniteFloat | None = Field(default=None, ge=0)
    depth_50bps_ask_quote: FiniteFloat | None = Field(default=None, ge=0)
    imbalance_50bps: FiniteFloat | None = Field(default=None, ge=-1.0, le=1.0)
    orderbook_resync_count: int = Field(default=0, ge=0)

    liquidation_source_available: bool = False
    liquidation_count: int | None = Field(default=None, ge=0)
    liquidation_long_quote: FiniteFloat | None = Field(default=None, ge=0)
    liquidation_short_quote: FiniteFloat | None = Field(default=None, ge=0)

    cross_source_deviation_bps: FiniteFloat | None = Field(default=None, ge=0)
    cross_source_pair: str | None = None


class ModelPredictionSummary(BaseModel):
    """User-facing experimental prediction attached only when ML release mode is PROMOTED."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prediction_id: str = Field(min_length=1)
    target_name: str = Field(min_length=1)
    horizon_hours: int = Field(ge=1)
    calibrated_probability: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    abstained: bool
    abstention_reason: str | None = None
    confidence_label: str = Field(min_length=1)
    data_quality_status: QualityStatus
    model_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    limitations: tuple[str, ...]
    non_guarantee_notice: str = Field(min_length=1)


class AnalysisSnapshot(BaseModel):
    """Complete deterministic snapshot passed to templates and Gemini (PROJECT_PLAN §5.8)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_id: str = Field(min_length=1)
    provider: str = Field(min_length=1)
    venue: str = Field(min_length=1)
    canonical_symbol: str = Field(min_length=1)
    provider_symbol: str = Field(min_length=1)
    market_type: MarketType
    timeframe: Timeframe
    as_of: UtcDatetime
    input_range: InputRange | None = None
    calc_version: str = Field(min_length=1)
    indicator_parameters: dict[str, int | float | str] = Field(default_factory=dict)
    technical: TechnicalMetrics | None = None
    derivatives: DerivativesMetrics | None = None
    smc: SmcSummary | None = None
    mtf_overview: tuple[TimeframeOverview, ...] = ()
    tradingview: TradingViewMetrics | None = None
    sentiment: SentimentMetrics | None = None
    trade_setup: TradeSetup | None = None
    ml_prediction: ModelPredictionSummary | None = None
    facts: tuple[Fact, ...] = ()
    regime: Regime | None = None
    market_structure: MarketStructure | None = None
    events: tuple[StructureEvent, ...] = ()
    factors: tuple[FactorItem, ...] = ()
    quality_status: QualityStatus
    quality_flags: tuple[QualityFlag, ...] = ()
    limitations: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _validate_snapshot(self) -> AnalysisSnapshot:
        validate_symbol_market_type(
            self.canonical_symbol,
            self.provider_symbol,
            self.market_type,
        )

        fact_ids = [fact.fact_id for fact in self.facts]
        if len(fact_ids) != len(set(fact_ids)):
            raise ValueError("facts must have unique fact_id values")
        known_fact_ids = set(fact_ids)

        for factor in self.factors:
            missing = [fid for fid in factor.fact_ids if fid not in known_fact_ids]
            if missing:
                raise ValueError(
                    f"factor '{factor.rule_id}' references unknown fact_ids: {missing}"
                )

        for event in self.events:
            if event.confirmation_time > self.as_of:
                raise ValueError("event confirmation_time cannot exceed snapshot as_of (lookahead)")

        if self.tradingview is not None and self.tradingview.candle_close_time > self.as_of:
            raise ValueError(
                "tradingview candle_close_time cannot exceed snapshot as_of (lookahead)"
            )

        if (
            self.sentiment is not None
            and self.sentiment.latest_available_time is not None
            and self.sentiment.latest_available_time > self.as_of
        ):
            raise ValueError(
                "sentiment latest_available_time cannot exceed snapshot as_of (lookahead)"
            )

        if self.smc is not None:
            for ev in (self.smc.latest_structure_break, self.smc.latest_sweep):
                if ev is not None and ev.confirmation_time > self.as_of:
                    raise ValueError(
                        "smc event confirmation_time cannot exceed snapshot as_of (lookahead)"
                    )
            for zone in (
                self.smc.active_bullish_fvg,
                self.smc.active_bearish_fvg,
                self.smc.active_bullish_ob,
                self.smc.active_bearish_ob,
            ):
                if zone is not None and zone.confirmation_time > self.as_of:
                    raise ValueError(
                        "smc zone confirmation_time cannot exceed snapshot as_of (lookahead)"
                    )

        if self.quality_status in (QualityStatus.READY, QualityStatus.DEGRADED):
            if self.input_range is None:
                raise ValueError("input_range is required when quality_status is READY or DEGRADED")
            if self.technical is None:
                raise ValueError("technical metrics are required when READY or DEGRADED")
            if self.regime is None or self.market_structure is None:
                raise ValueError("regime and market_structure are required when READY or DEGRADED")
            if not self.facts:
                raise ValueError("facts must be non-empty when READY or DEGRADED")

        if self.quality_status is QualityStatus.READY:
            blocking = [flag for flag in self.quality_flags if flag in BLOCKING_QUALITY_FLAGS]
            if blocking:
                raise ValueError(f"READY snapshot cannot contain blocking flags: {blocking}")

        return self
