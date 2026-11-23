"""Confirmed swing pivot and market-structure classification (PROJECT_PLAN §5.6)."""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from crypto_chatbot.analysis.indicators import validate_closed_candles
from crypto_chatbot.schemas.analysis import (
    MarketStructure,
    StructureEvent,
    StructureEventType,
)
from crypto_chatbot.schemas.market import MarketCandle

DEFAULT_PIVOT_K = 3


class StructureAnalysis(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    pivots: tuple[StructureEvent, ...]
    market_structure: MarketStructure
    last_swing_highs: tuple[StructureEvent, ...]
    last_swing_lows: tuple[StructureEvent, ...]


def detect_confirmed_swings(
    candles: Sequence[MarketCandle],
    *,
    k: int = DEFAULT_PIVOT_K,
) -> tuple[StructureEvent, ...]:
    """Detect strict k-bar swing pivots with explicit confirmation timestamps."""
    if k < 1:
        raise ValueError("k must be >= 1")
    validate_closed_candles(candles)

    events: list[StructureEvent] = []
    n = len(candles)
    for i in range(k, n - k):
        center = candles[i]
        confirm_time = candles[i + k].close_time
        left_right_highs = [candles[j].high for j in range(i - k, i + k + 1) if j != i]
        left_right_lows = [candles[j].low for j in range(i - k, i + k + 1) if j != i]

        if center.high > max(left_right_highs):
            events.append(
                StructureEvent(
                    event_type=StructureEventType.SWING_HIGH,
                    event_time=center.close_time,
                    confirmation_time=confirm_time,
                    level=center.high,
                )
            )
        if center.low < min(left_right_lows):
            events.append(
                StructureEvent(
                    event_type=StructureEventType.SWING_LOW,
                    event_time=center.close_time,
                    confirmation_time=confirm_time,
                    level=center.low,
                )
            )

    return tuple(events)


def analyze_market_structure(
    candles: Sequence[MarketCandle],
    *,
    k: int = DEFAULT_PIVOT_K,
) -> StructureAnalysis:
    """Classify market structure (bullish/bearish/mixed) from confirmed pivots only."""
    pivots = detect_confirmed_swings(candles, k=k)
    highs = tuple(p for p in pivots if p.event_type is StructureEventType.SWING_HIGH)
    lows = tuple(p for p in pivots if p.event_type is StructureEventType.SWING_LOW)

    last_highs = highs[-2:]
    last_lows = lows[-2:]

    if len(last_highs) < 2 or len(last_lows) < 2:
        structure = MarketStructure.MIXED
    else:
        higher_high = last_highs[-1].level > last_highs[-2].level
        higher_low = last_lows[-1].level > last_lows[-2].level
        lower_high = last_highs[-1].level < last_highs[-2].level
        lower_low = last_lows[-1].level < last_lows[-2].level

        if higher_high and higher_low:
            structure = MarketStructure.BULLISH
        elif lower_high and lower_low:
            structure = MarketStructure.BEARISH
        else:
            structure = MarketStructure.MIXED

    return StructureAnalysis(
        pivots=pivots,
        market_structure=structure,
        last_swing_highs=last_highs,
        last_swing_lows=last_lows,
    )
