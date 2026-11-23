"""Causal failed-breakout detector around confirmed swing pivots (PROJECT_PLAN §5.6)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from crypto_chatbot.analysis.indicators import validate_closed_candles
from crypto_chatbot.analysis.structure import DEFAULT_PIVOT_K, detect_confirmed_swings
from crypto_chatbot.schemas.analysis import StructureEvent, StructureEventType
from crypto_chatbot.schemas.market import MarketCandle

DEFAULT_FAILED_BREAKOUT_WINDOW = 3


def detect_failed_breakouts(
    candles: Sequence[MarketCandle],
    *,
    pivot_k: int = DEFAULT_PIVOT_K,
    max_reentry_bars: int = DEFAULT_FAILED_BREAKOUT_WINDOW,
) -> tuple[StructureEvent, ...]:
    """Detect failed breakouts causally using only already-confirmed swing pivots.

    A bullish failure occurs when a bar closes above the most recent confirmed swing high
    and within `max_reentry_bars` a bar closes back below that level. Bearish failure is
    symmetric against the most recent confirmed swing low.
    """
    if pivot_k < 1:
        raise ValueError("pivot_k must be >= 1")
    if max_reentry_bars < 1:
        raise ValueError("max_reentry_bars must be >= 1")
    validate_closed_candles(candles)

    pivots = detect_confirmed_swings(candles, k=pivot_k)
    by_confirm: dict[datetime, list[StructureEvent]] = {}
    for pivot in pivots:
        by_confirm.setdefault(pivot.confirmation_time, []).append(pivot)

    latest_sh: StructureEvent | None = None
    latest_sl: StructureEvent | None = None
    pending_high: tuple[float, int] | None = None
    pending_low: tuple[float, int] | None = None
    events: list[StructureEvent] = []

    for bar_idx, candle in enumerate(candles):
        for pivot in by_confirm.get(candle.close_time, ()):
            if pivot.event_type is StructureEventType.SWING_HIGH:
                latest_sh = pivot
            elif pivot.event_type is StructureEventType.SWING_LOW:
                latest_sl = pivot

        prev_close = candles[bar_idx - 1].close if bar_idx > 0 else None

        if pending_high is not None:
            level, breakout_idx = pending_high
            elapsed = bar_idx - breakout_idx
            if candle.close < level and 1 <= elapsed <= max_reentry_bars:
                events.append(
                    StructureEvent(
                        event_type=StructureEventType.FAILED_BREAKOUT_BULLISH,
                        event_time=candle.close_time,
                        confirmation_time=candle.close_time,
                        level=level,
                    )
                )
                pending_high = None
            elif elapsed >= max_reentry_bars:
                pending_high = None
        elif (
            latest_sh is not None
            and candle.close > latest_sh.level
            and (prev_close is None or prev_close <= latest_sh.level)
        ):
            pending_high = (latest_sh.level, bar_idx)

        if pending_low is not None:
            level, breakout_idx = pending_low
            elapsed = bar_idx - breakout_idx
            if candle.close > level and 1 <= elapsed <= max_reentry_bars:
                events.append(
                    StructureEvent(
                        event_type=StructureEventType.FAILED_BREAKOUT_BEARISH,
                        event_time=candle.close_time,
                        confirmation_time=candle.close_time,
                        level=level,
                    )
                )
                pending_low = None
            elif elapsed >= max_reentry_bars:
                pending_low = None
        elif (
            latest_sl is not None
            and candle.close < latest_sl.level
            and (prev_close is None or prev_close >= latest_sl.level)
        ):
            pending_low = (latest_sl.level, bar_idx)

    return tuple(events)
