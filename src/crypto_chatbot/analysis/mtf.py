"""Shared deterministic multi-timeframe (MTF) context evaluator.

Single source of truth for computing higher-timeframe (and lower-timeframe) regime,
market structure, ADX, RSI, and SMC summaries across:
- `AnalysisService.analyze` / `analyze_multi_style` (top-down MTF overview)
- `alerts run` (15m MTF top-down bias gate)
- `scan_market_setups` (scanner)

Strict invariants:
1. Closed candles only: `c.is_closed` AND `c.close_time <= as_of`
   (and `c.available_time <= as_of` when `use_available_time=True`).
2. Single-venue integrity: if `expected_venue` is provided, every candle must match it.
3. Exact indicator/regime/structure parity with standalone `build_analysis_snapshot`.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from crypto_chatbot.analysis.indicators import (
    MIN_HISTORY_EMA_20,
    compute_technical_metrics,
)
from crypto_chatbot.analysis.regimes import classify_regime
from crypto_chatbot.analysis.smc import analyze_smc
from crypto_chatbot.analysis.structure import StructureAnalysis, analyze_market_structure
from crypto_chatbot.market_data.base import OhlcvBatch
from crypto_chatbot.schemas.analysis import (
    MarketStructure,
    Regime,
    SmcSummary,
    TechnicalMetrics,
    TimeframeOverview,
)
from crypto_chatbot.schemas.market import MarketCandle, QualityFlag, Timeframe

logger = logging.getLogger(__name__)

MTF_REGIME_LABELS: dict[Regime, str] = {
    Regime.HIGH_VOLATILITY: "Volatilitas Tinggi",
    Regime.TRENDING_UP: "Tren Naik",
    Regime.TRENDING_DOWN: "Tren Turun",
    Regime.RANGING: "Mendatar (Ranging)",
    Regime.TRANSITIONAL: "Transisi",
}

_BLOCKING_MTF_FLAGS: frozenset[QualityFlag] = frozenset(
    {
        QualityFlag.PROVIDER_ERROR,
        QualityFlag.STALE,
        QualityFlag.INVALID_OHLC,
        QualityFlag.GAP,
        QualityFlag.SYMBOL_MISMATCH,
    }
)


class MtfVenueMismatchError(ValueError):
    """Raised when an MTF candle series does not match the pinned venue."""


def filter_causal_venue_candles(
    candles: Sequence[MarketCandle],
    *,
    as_of: datetime,
    expected_venue: str | None = None,
    use_available_time: bool = False,
) -> tuple[MarketCandle, ...]:
    """Filter strictly to closed causal candles (`close_time <= as_of`) from one venue."""
    cutoff = as_of.astimezone(UTC)
    norm_expected = expected_venue.strip().lower() if expected_venue else None
    result: list[MarketCandle] = []
    for candle in candles:
        if not candle.is_closed:
            continue
        if candle.close_time.astimezone(UTC) > cutoff:
            continue
        if use_available_time and candle.available_time.astimezone(UTC) > cutoff:
            continue
        if norm_expected is not None and candle.venue.strip().lower() != norm_expected:
            raise MtfVenueMismatchError(
                f"MTF venue mismatch for {candle.canonical_symbol} {candle.timeframe.value}: "
                f"expected '{norm_expected}', got '{candle.venue}'"
            )
        result.append(candle)
    return tuple(result)


@dataclass(frozen=True)
class TimeframeContext:
    """Canonical deterministic evaluation for one timeframe at a causal `as_of` cutoff."""

    timeframe: Timeframe
    venue: str
    as_of: datetime
    candle_count: int
    regime: Regime | None
    trend_label: str | None
    market_structure: MarketStructure
    adx_14: float | None
    rsi_14: float | None
    metrics: TechnicalMetrics | None
    structure: StructureAnalysis
    smc: SmcSummary

    def to_overview(self) -> TimeframeOverview:
        sup = (
            self.smc.dealing_range_low
            if self.smc.dealing_range_low is not None
            else (
                self.structure.last_swing_lows[-1].level if self.structure.last_swing_lows else None
            )
        )
        res = (
            self.smc.dealing_range_high
            if self.smc.dealing_range_high is not None
            else (
                self.structure.last_swing_highs[-1].level
                if self.structure.last_swing_highs
                else None
            )
        )
        pois: list[str] = []
        if self.smc.active_bullish_fvg is not None:
            pois.append("Bull FVG")
        if self.smc.active_bullish_ob is not None:
            pois.append("Bull OB")
        if self.smc.active_bearish_fvg is not None:
            pois.append("Bear FVG")
        if self.smc.active_bearish_ob is not None:
            pois.append("Bear OB")

        return TimeframeOverview(
            timeframe=self.timeframe,
            regime=self.regime,
            trend_label=self.trend_label,
            market_structure=self.market_structure,
            structure_break=(
                self.smc.latest_structure_break.event_type
                if self.smc.latest_structure_break is not None
                else None
            ),
            structure_break_level=(
                self.smc.latest_structure_break.level
                if self.smc.latest_structure_break is not None
                else None
            ),
            sweep=(self.smc.latest_sweep.event_type if self.smc.latest_sweep is not None else None),
            dealing_zone=self.smc.dealing_zone,
            support_level=sup,
            resistance_level=res,
            active_poi=", ".join(pois) if pois else None,
            rsi_14=self.rsi_14,
            adx_14=self.adx_14,
        )


def evaluate_timeframe_context(
    candles: Sequence[MarketCandle],
    *,
    timeframe: Timeframe,
    as_of: datetime,
    expected_venue: str | None = None,
    use_available_time: bool = False,
    min_candles: int = 15,
) -> TimeframeContext | None:
    """Compute canonical regime, structure, ADX, RSI, and SMC for one timeframe.

    Uses the exact same indicator and regime pipeline as standalone `build_analysis_snapshot`,
    strictly on closed candles with `close_time <= as_of` from the same venue.
    """
    closed = filter_causal_venue_candles(
        candles,
        as_of=as_of,
        expected_venue=expected_venue,
        use_available_time=use_available_time,
    )
    if len(closed) < min_candles:
        return None

    venue = closed[-1].venue.strip().lower()
    last_close_time = closed[-1].close_time.astimezone(UTC)
    struct = analyze_market_structure(closed)
    smc_tf = analyze_smc(closed, structure=struct)

    metrics: TechnicalMetrics | None = None
    regime: Regime | None = None
    trend_label: str | None = None
    rsi_val: float | None = None
    adx_val: float | None = None

    if len(closed) >= MIN_HISTORY_EMA_20:
        ind_tf = compute_technical_metrics(closed)
        metrics = ind_tf.metrics
        rsi_val = metrics.rsi_14
        adx_val = metrics.adx_14
        regime = classify_regime(metrics)
        if regime is not None:
            mapped_trend = MTF_REGIME_LABELS.get(regime)
            if mapped_trend is None:
                logger.warning(
                    "Missing MTF regime label for %s; falling back to value",
                    regime.value,
                )
                trend_label = regime.value
            else:
                trend_label = mapped_trend
        else:
            ref_ema = metrics.ema_50 if metrics.ema_50 is not None else metrics.ema_20
            if ref_ema is not None:
                trend_label = "Tren Naik" if metrics.last_close > ref_ema else "Tren Turun"

    return TimeframeContext(
        timeframe=timeframe,
        venue=venue,
        as_of=last_close_time,
        candle_count=len(closed),
        regime=regime,
        trend_label=trend_label,
        market_structure=struct.market_structure,
        adx_14=adx_val,
        rsi_14=rsi_val,
        metrics=metrics,
        structure=struct,
        smc=smc_tf,
    )


def evaluate_mtf_overview(
    primary_batch: OhlcvBatch,
    *,
    mtf_batches: Mapping[Timeframe, OhlcvBatch] | None,
    as_of: datetime,
    use_available_time: bool = False,
) -> tuple[TimeframeOverview, ...]:
    """Build top-down `TimeframeOverview` tuple using the single shared evaluator."""
    if not mtf_batches:
        return ()

    expected_venue = primary_batch.metadata.venue.strip().lower() or None
    by_tf: dict[Timeframe, Sequence[MarketCandle]] = {
        primary_batch.timeframe: primary_batch.candles
    }
    for tf, mtf_b in mtf_batches.items():
        if tf is primary_batch.timeframe:
            continue
        if not mtf_b.candles or (set(mtf_b.quality_flags) & _BLOCKING_MTF_FLAGS):
            continue
        if expected_venue is not None and mtf_b.metadata.venue.strip().lower() != expected_venue:
            logger.warning(
                "Dropping MTF batch %s due to venue mismatch (expected=%s, got=%s)",
                tf.value,
                expected_venue,
                mtf_b.metadata.venue,
            )
            continue
        by_tf[tf] = mtf_b.candles

    if len(by_tf) < 2:
        return ()

    overviews: list[TimeframeOverview] = []
    for tf in (Timeframe.D1, Timeframe.H4, Timeframe.H1, Timeframe.M15, Timeframe.M5):
        seq = by_tf.get(tf)
        if not seq:
            continue
        try:
            ctx = evaluate_timeframe_context(
                seq,
                timeframe=tf,
                as_of=as_of,
                expected_venue=expected_venue,
                use_available_time=use_available_time,
                min_candles=15,
            )
        except MtfVenueMismatchError as exc:
            logger.warning("Skipping MTF %s: %s", tf.value, exc)
            continue
        if ctx is not None:
            overviews.append(ctx.to_overview())

    return tuple(overviews)
