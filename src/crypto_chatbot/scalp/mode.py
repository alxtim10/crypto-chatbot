"""SC-04: Deterministic 1h context mode selection for `scalp_v1` (SCALP-SPEC-v1 §3).

Uses the exact same deterministic indicator/regime/structure functions as `analyze`
(and accepts an `AnalysisSnapshot` directly when called from live `alerts run`),
guaranteeing live vs `analyze` parity at the same `as_of` timestamp.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.analysis.indicators import compute_technical_metrics
from crypto_chatbot.analysis.regimes import classify_regime
from crypto_chatbot.analysis.structure import analyze_market_structure
from crypto_chatbot.scalp.config import ScalpConfig
from crypto_chatbot.schemas.analysis import (
    AnalysisSnapshot,
    MarketStructure,
    Regime,
)
from crypto_chatbot.schemas.market import MarketCandle, QualityStatus


class ScalpMode(StrEnum):
    """Active scalp trading mode selected from the last closed 1h candle (§3)."""

    TREND_PULLBACK_LONG = "TREND_PULLBACK_LONG"
    TREND_PULLBACK_SHORT = "TREND_PULLBACK_SHORT"
    RANGE_FADE = "RANGE_FADE"
    NO_TRADE = "NO_TRADE"


class ModeInputs(BaseModel):
    """Auditable inputs recorded on every decision row (including NO_TRADE)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    regime_1h: Regime | None = None
    adx_14_1h: float | None = None
    plus_di_14_1h: float | None = None
    minus_di_14_1h: float | None = None
    close_1h: float | None = None
    ema_50_1h: float | None = None
    atr_14_1h: float | None = None
    structure_15m: MarketStructure | None = None
    data_quality_1h: QualityStatus | None = None
    has_valid_range: bool = False
    range_low: float | None = None
    range_high: float | None = None


class ModeSelectionResult(BaseModel):
    """Output of §3 mode selection with explicit reason code and recorded inputs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mode: ScalpMode
    reason_code: str
    reason_detail: str
    inputs: ModeInputs = Field(default_factory=ModeInputs)


def _filter_closed_pit(
    candles: Sequence[MarketCandle],
    as_of: datetime | None,
) -> list[MarketCandle]:
    if as_of is None:
        return [c for c in candles if c.is_closed]
    cutoff = as_of.astimezone(UTC)
    return [
        c
        for c in candles
        if c.is_closed
        and c.close_time.astimezone(UTC) <= cutoff
        and c.available_time.astimezone(UTC) <= cutoff
    ]


def select_scalp_mode_from_inputs(
    *,
    regime_1h: Regime | None,
    adx_14_1h: float | None,
    plus_di_14_1h: float | None,
    minus_di_14_1h: float | None,
    close_1h: float | None,
    ema_50_1h: float | None,
    atr_14_1h: float | None,
    structure_15m: MarketStructure | None,
    data_quality_1h: QualityStatus | None,
    has_valid_range: bool = False,
    range_low: float | None = None,
    range_high: float | None = None,
    config: ScalpConfig | None = None,
) -> ModeSelectionResult:
    """Pure mode selection predicate implementing SCALP-SPEC-v1 §3."""
    cfg = config or ScalpConfig()
    inputs = ModeInputs(
        regime_1h=regime_1h,
        adx_14_1h=adx_14_1h,
        plus_di_14_1h=plus_di_14_1h,
        minus_di_14_1h=minus_di_14_1h,
        close_1h=close_1h,
        ema_50_1h=ema_50_1h,
        atr_14_1h=atr_14_1h,
        structure_15m=structure_15m,
        data_quality_1h=data_quality_1h,
        has_valid_range=has_valid_range,
        range_low=range_low,
        range_high=range_high,
    )

    if data_quality_1h is not None and data_quality_1h is not QualityStatus.READY:
        return ModeSelectionResult(
            mode=ScalpMode.NO_TRADE,
            reason_code="DATA_QUALITY_NOT_READY",
            reason_detail=f"1h data quality is {data_quality_1h.value} (requires READY)",
            inputs=inputs,
        )

    if regime_1h is None or adx_14_1h is None:
        return ModeSelectionResult(
            mode=ScalpMode.NO_TRADE,
            reason_code="INSUFFICIENT_1H_CONTEXT",
            reason_detail="1h regime or ADX(14) unavailable from closed candles",
            inputs=inputs,
        )

    if regime_1h in (Regime.HIGH_VOLATILITY, Regime.TRANSITIONAL):
        return ModeSelectionResult(
            mode=ScalpMode.NO_TRADE,
            reason_code=f"REGIME_{regime_1h.value.upper()}",
            reason_detail=f"1h regime is {regime_1h.value}; no scalp mode active",
            inputs=inputs,
        )

    if regime_1h is Regime.TRENDING_UP:
        cond_trend = (
            adx_14_1h >= cfg.trend_adx_min
            and plus_di_14_1h is not None
            and minus_di_14_1h is not None
            and plus_di_14_1h > minus_di_14_1h
            and close_1h is not None
            and ema_50_1h is not None
            and close_1h > ema_50_1h
        )
        if not cond_trend:
            return ModeSelectionResult(
                mode=ScalpMode.NO_TRADE,
                reason_code="TREND_UP_CRITERIA_UNMET",
                reason_detail="1h TRENDING_UP failed ADX/+DI/EMA50 confirmation",
                inputs=inputs,
            )
        if structure_15m is MarketStructure.BEARISH:
            return ModeSelectionResult(
                mode=ScalpMode.NO_TRADE,
                reason_code="STRUCTURE_15M_CONFLICT",
                reason_detail="1h TRENDING_UP blocked because 15m structure is BEARISH",
                inputs=inputs,
            )
        return ModeSelectionResult(
            mode=ScalpMode.TREND_PULLBACK_LONG,
            reason_code="TREND_PULLBACK_LONG",
            reason_detail="1h TRENDING_UP (ADX>=25, +DI>-DI, close>EMA50) and 15m not bearish",
            inputs=inputs,
        )

    if regime_1h is Regime.TRENDING_DOWN:
        cond_trend = (
            adx_14_1h >= cfg.trend_adx_min
            and plus_di_14_1h is not None
            and minus_di_14_1h is not None
            and minus_di_14_1h > plus_di_14_1h
            and close_1h is not None
            and ema_50_1h is not None
            and close_1h < ema_50_1h
        )
        if not cond_trend:
            return ModeSelectionResult(
                mode=ScalpMode.NO_TRADE,
                reason_code="TREND_DOWN_CRITERIA_UNMET",
                reason_detail="1h TRENDING_DOWN failed ADX/-DI/EMA50 confirmation",
                inputs=inputs,
            )
        if structure_15m is MarketStructure.BULLISH:
            return ModeSelectionResult(
                mode=ScalpMode.NO_TRADE,
                reason_code="STRUCTURE_15M_CONFLICT",
                reason_detail="1h TRENDING_DOWN blocked because 15m structure is BULLISH",
                inputs=inputs,
            )
        return ModeSelectionResult(
            mode=ScalpMode.TREND_PULLBACK_SHORT,
            reason_code="TREND_PULLBACK_SHORT",
            reason_detail="1h TRENDING_DOWN (ADX>=25, -DI>+DI, close<EMA50) and 15m not bullish",
            inputs=inputs,
        )

    if regime_1h is Regime.RANGING:
        if adx_14_1h >= cfg.range_adx_max:
            return ModeSelectionResult(
                mode=ScalpMode.NO_TRADE,
                reason_code="RANGE_ADX_TOO_HIGH",
                reason_detail=f"1h ADX {adx_14_1h:.1f} >= {cfg.range_adx_max:.1f} for RANGE_FADE",
                inputs=inputs,
            )
        if not has_valid_range:
            return ModeSelectionResult(
                mode=ScalpMode.NO_TRADE,
                reason_code="NO_VALID_RANGE",
                reason_detail="1h RANGING but no valid dealing range passed width checks (§4.2)",
                inputs=inputs,
            )
        return ModeSelectionResult(
            mode=ScalpMode.RANGE_FADE,
            reason_code="RANGE_FADE",
            reason_detail="1h RANGING (ADX<20) with valid dealing range",
            inputs=inputs,
        )

    return ModeSelectionResult(
        mode=ScalpMode.NO_TRADE,
        reason_code="UNKNOWN_REGIME",
        reason_detail=f"Unhandled 1h regime: {regime_1h}",
        inputs=inputs,
    )


def select_scalp_mode_from_snapshot(
    snapshot_1h: AnalysisSnapshot,
    *,
    structure_15m: MarketStructure | None = None,
    has_valid_range: bool = False,
    range_low: float | None = None,
    range_high: float | None = None,
    config: ScalpConfig | None = None,
) -> ModeSelectionResult:
    """Select scalp mode directly from a 1h `AnalysisSnapshot` produced by `AnalysisService`."""
    tech = snapshot_1h.technical
    return select_scalp_mode_from_inputs(
        regime_1h=snapshot_1h.regime,
        adx_14_1h=tech.adx_14 if tech is not None else None,
        plus_di_14_1h=tech.plus_di_14 if tech is not None else None,
        minus_di_14_1h=tech.minus_di_14 if tech is not None else None,
        close_1h=tech.last_close if tech is not None else None,
        ema_50_1h=tech.ema_50 if tech is not None else None,
        atr_14_1h=tech.atr_14 if tech is not None else None,
        structure_15m=structure_15m,
        data_quality_1h=snapshot_1h.quality_status,
        has_valid_range=has_valid_range,
        range_low=range_low,
        range_high=range_high,
        config=config,
    )


def select_scalp_mode_from_candles(
    candles_1h: Sequence[MarketCandle],
    candles_15m: Sequence[MarketCandle],
    *,
    as_of: datetime | None = None,
    data_quality_1h: QualityStatus = QualityStatus.READY,
    has_valid_range: bool = False,
    range_low: float | None = None,
    range_high: float | None = None,
    config: ScalpConfig | None = None,
) -> ModeSelectionResult:
    """Shared candle-level mode selector (identical math to `AnalysisService.analyze` 1h)."""
    closed_1h = _filter_closed_pit(candles_1h, as_of)
    closed_15m = _filter_closed_pit(candles_15m, as_of)

    if not closed_1h:
        return select_scalp_mode_from_inputs(
            regime_1h=None,
            adx_14_1h=None,
            plus_di_14_1h=None,
            minus_di_14_1h=None,
            close_1h=None,
            ema_50_1h=None,
            atr_14_1h=None,
            structure_15m=None,
            data_quality_1h=data_quality_1h,
            has_valid_range=has_valid_range,
            range_low=range_low,
            range_high=range_high,
            config=config,
        )

    tech_1h = compute_technical_metrics(closed_1h).metrics
    regime_1h = classify_regime(tech_1h)
    structure_15m: MarketStructure | None = None
    if closed_15m:
        structure_15m = analyze_market_structure(closed_15m).market_structure

    return select_scalp_mode_from_inputs(
        regime_1h=regime_1h,
        adx_14_1h=tech_1h.adx_14,
        plus_di_14_1h=tech_1h.plus_di_14,
        minus_di_14_1h=tech_1h.minus_di_14,
        close_1h=tech_1h.last_close,
        ema_50_1h=tech_1h.ema_50,
        atr_14_1h=tech_1h.atr_14,
        structure_15m=structure_15m,
        data_quality_1h=data_quality_1h,
        has_valid_range=has_valid_range,
        range_low=range_low,
        range_high=range_high,
        config=config,
    )
