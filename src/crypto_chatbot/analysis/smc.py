"""Causal Hybrid Smart Money Concepts (SMC) detectors for crypto (ADR 0003)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from crypto_chatbot.analysis.indicators import validate_closed_candles
from crypto_chatbot.analysis.structure import (
    DEFAULT_PIVOT_K,
    StructureAnalysis,
    analyze_market_structure,
)
from crypto_chatbot.schemas.analysis import (
    DealingZone,
    PriceZone,
    SmcSummary,
    StructureEvent,
    StructureEventType,
)
from crypto_chatbot.schemas.market import MarketCandle

SMC_LOOKBACK_BARS = 50
OB_LOOKBACK_BARS = 10
SWEEP_RECENT_BARS = 10
MIN_FVG_WIDTH_BPS = 10.0


def _detect_causal_breaks_and_sweeps(
    candles: Sequence[MarketCandle],
    *,
    pivots: Sequence[StructureEvent],
) -> tuple[list[tuple[int, StructureEvent]], list[tuple[int, StructureEvent]]]:
    by_confirm: dict[datetime, list[StructureEvent]] = {}
    for pivot in pivots:
        by_confirm.setdefault(pivot.confirmation_time, []).append(pivot)

    latest_sh: StructureEvent | None = None
    latest_sl: StructureEvent | None = None
    consumed_sh_time: datetime | None = None
    consumed_sl_time: datetime | None = None
    trend: str | None = None

    breaks: list[tuple[int, StructureEvent]] = []
    sweeps: list[tuple[int, StructureEvent]] = []

    for bar_idx, candle in enumerate(candles):
        for pivot in by_confirm.get(candle.close_time, ()):
            if pivot.event_type is StructureEventType.SWING_HIGH:
                latest_sh = pivot
            elif pivot.event_type is StructureEventType.SWING_LOW:
                latest_sl = pivot

        # Causal liquidity sweeps against already-confirmed swings (wick pierce + close inside)
        if (
            latest_sh is not None
            and candle.high > latest_sh.level
            and candle.close <= latest_sh.level
        ):
            sweeps.append(
                (
                    bar_idx,
                    StructureEvent(
                        event_type=StructureEventType.LIQUIDITY_SWEEP_BEARISH,
                        event_time=candle.close_time,
                        confirmation_time=candle.close_time,
                        level=latest_sh.level,
                    ),
                )
            )
        if (
            latest_sl is not None
            and candle.low < latest_sl.level
            and candle.close >= latest_sl.level
        ):
            sweeps.append(
                (
                    bar_idx,
                    StructureEvent(
                        event_type=StructureEventType.LIQUIDITY_SWEEP_BULLISH,
                        event_time=candle.close_time,
                        confirmation_time=candle.close_time,
                        level=latest_sl.level,
                    ),
                )
            )

        # Causal BOS / CHoCH on close beyond an unconsumed confirmed swing
        if (
            latest_sh is not None
            and latest_sh.event_time != consumed_sh_time
            and candle.close > latest_sh.level
        ):
            ev_type = (
                StructureEventType.CHOCH_BULLISH
                if trend == "bearish"
                else StructureEventType.BOS_BULLISH
            )
            trend = "bullish"
            consumed_sh_time = latest_sh.event_time
            breaks.append(
                (
                    bar_idx,
                    StructureEvent(
                        event_type=ev_type,
                        event_time=candle.close_time,
                        confirmation_time=candle.close_time,
                        level=latest_sh.level,
                    ),
                )
            )
        elif (
            latest_sl is not None
            and latest_sl.event_time != consumed_sl_time
            and candle.close < latest_sl.level
        ):
            ev_type = (
                StructureEventType.CHOCH_BEARISH
                if trend == "bullish"
                else StructureEventType.BOS_BEARISH
            )
            trend = "bearish"
            consumed_sl_time = latest_sl.event_time
            breaks.append(
                (
                    bar_idx,
                    StructureEvent(
                        event_type=ev_type,
                        event_time=candle.close_time,
                        confirmation_time=candle.close_time,
                        level=latest_sl.level,
                    ),
                )
            )

    return breaks, sweeps


def _find_active_fvgs(
    candles: Sequence[MarketCandle],
    *,
    last_close: float,
    lookback_bars: int,
) -> tuple[PriceZone | None, PriceZone | None]:
    n = len(candles)
    if n < 3:
        return None, None

    start = max(2, n - lookback_bars)
    bull_candidates: list[PriceZone] = []
    bear_candidates: list[PriceZone] = []

    for i in range(start, n):
        first = candles[i - 2]
        third = candles[i]

        # Bullish FVG: gap between candle[i-2].high and candle[i].low
        if third.low > first.high:
            width_bps = ((third.low - first.high) / third.close) * 10_000.0
            if width_bps >= MIN_FVG_WIDTH_BPS:
                mitigated = any(
                    candles[j].low <= first.high or candles[j].close < first.high
                    for j in range(i + 1, n)
                )
                if not mitigated and third.low <= last_close * 1.002:
                    bull_candidates.append(
                        PriceZone(
                            zone_type="bullish_fvg",
                            zone_low=first.high,
                            zone_high=third.low,
                            event_time=third.close_time,
                            confirmation_time=third.close_time,
                            mitigated=False,
                        )
                    )

        # Bearish FVG: gap between candle[i].high and candle[i-2].low
        if third.high < first.low:
            width_bps = ((first.low - third.high) / third.close) * 10_000.0
            if width_bps >= MIN_FVG_WIDTH_BPS:
                mitigated = any(
                    candles[j].high >= first.low or candles[j].close > first.low
                    for j in range(i + 1, n)
                )
                if not mitigated and third.high >= last_close * 0.998:
                    bear_candidates.append(
                        PriceZone(
                            zone_type="bearish_fvg",
                            zone_low=third.high,
                            zone_high=first.low,
                            event_time=third.close_time,
                            confirmation_time=third.close_time,
                            mitigated=False,
                        )
                    )

    active_bull = max(bull_candidates, key=lambda z: z.zone_high) if bull_candidates else None
    active_bear = min(bear_candidates, key=lambda z: z.zone_low) if bear_candidates else None
    return active_bull, active_bear


def _find_active_order_blocks(
    candles: Sequence[MarketCandle],
    *,
    breaks: Sequence[tuple[int, StructureEvent]],
    last_close: float,
    lookback_bars: int,
) -> tuple[PriceZone | None, PriceZone | None]:
    n = len(candles)
    min_bar = max(0, n - lookback_bars)
    bull_obs: list[PriceZone] = []
    bear_obs: list[PriceZone] = []

    for break_idx, ev in breaks:
        if break_idx < min_bar or break_idx <= 0:
            continue
        search_start = max(0, break_idx - OB_LOOKBACK_BARS)

        if ev.event_type in (StructureEventType.BOS_BULLISH, StructureEventType.CHOCH_BULLISH):
            for idx in range(break_idx - 1, search_start - 1, -1):
                c = candles[idx]
                if c.close < c.open:
                    mitigated = any(candles[j].close < c.low for j in range(break_idx + 1, n))
                    if not mitigated and c.high <= last_close * 1.005:
                        bull_obs.append(
                            PriceZone(
                                zone_type="bullish_ob",
                                zone_low=c.low,
                                zone_high=c.high,
                                event_time=c.close_time,
                                confirmation_time=ev.confirmation_time,
                                mitigated=False,
                            )
                        )
                    break

        elif ev.event_type in (StructureEventType.BOS_BEARISH, StructureEventType.CHOCH_BEARISH):
            for idx in range(break_idx - 1, search_start - 1, -1):
                c = candles[idx]
                if c.close > c.open:
                    mitigated = any(candles[j].close > c.high for j in range(break_idx + 1, n))
                    if not mitigated and c.low >= last_close * 0.995:
                        bear_obs.append(
                            PriceZone(
                                zone_type="bearish_ob",
                                zone_low=c.low,
                                zone_high=c.high,
                                event_time=c.close_time,
                                confirmation_time=ev.confirmation_time,
                                mitigated=False,
                            )
                        )
                    break

    active_bull = max(bull_obs, key=lambda z: z.zone_high) if bull_obs else None
    active_bear = min(bear_obs, key=lambda z: z.zone_low) if bear_obs else None
    return active_bull, active_bear


def analyze_smc(
    candles: Sequence[MarketCandle],
    *,
    structure: StructureAnalysis | None = None,
    pivot_k: int = DEFAULT_PIVOT_K,
    lookback_bars: int = SMC_LOOKBACK_BARS,
) -> SmcSummary:
    """Detect causal BOS/CHoCH, liquidity sweeps, active FVG/OB zones, and dealing range."""
    validate_closed_candles(candles)
    struct = (
        structure
        if structure is not None
        else analyze_market_structure(
            candles,
            k=pivot_k,
        )
    )

    last_close = candles[-1].close
    n = len(candles)

    breaks, sweeps = _detect_causal_breaks_and_sweeps(candles, pivots=struct.pivots)
    latest_break = breaks[-1][1] if breaks else None
    latest_sweep = sweeps[-1][1] if sweeps and (n - 1 - sweeps[-1][0]) < SWEEP_RECENT_BARS else None

    active_bull_fvg, active_bear_fvg = _find_active_fvgs(
        candles,
        last_close=last_close,
        lookback_bars=lookback_bars,
    )
    active_bull_ob, active_bear_ob = _find_active_order_blocks(
        candles,
        breaks=breaks,
        last_close=last_close,
        lookback_bars=lookback_bars,
    )

    dr_high: float | None = None
    dr_low: float | None = None
    eq: float | None = None
    pos_in_range: float | None = None
    dealing_zone: DealingZone | None = None

    if struct.last_swing_highs and struct.last_swing_lows:
        sh_level = struct.last_swing_highs[-1].level
        sl_level = struct.last_swing_lows[-1].level
        if sh_level > sl_level:
            dr_high = sh_level
            dr_low = sl_level
            eq = (dr_high + dr_low) / 2.0
            pos_in_range = (last_close - dr_low) / (dr_high - dr_low)
            if pos_in_range < 0.45:
                dealing_zone = DealingZone.DISCOUNT
            elif pos_in_range > 0.55:
                dealing_zone = DealingZone.PREMIUM
            else:
                dealing_zone = DealingZone.EQUILIBRIUM

    return SmcSummary(
        latest_structure_break=latest_break,
        latest_sweep=latest_sweep,
        active_bullish_fvg=active_bull_fvg,
        active_bearish_fvg=active_bear_fvg,
        active_bullish_ob=active_bull_ob,
        active_bearish_ob=active_bear_ob,
        dealing_range_high=dr_high,
        dealing_range_low=dr_low,
        equilibrium=eq,
        position_in_range=pos_in_range,
        dealing_zone=dealing_zone,
    )
