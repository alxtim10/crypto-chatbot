"""Closed-bar order-flow microstructure: taker buy ratio, CVD, and divergence."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal

from crypto_chatbot.schemas.derivatives import OrderFlowMetrics
from crypto_chatbot.schemas.market import MarketCandle


def compute_closed_candle_orderflow(
    candles: Sequence[MarketCandle],
    *,
    window: int = 20,
    divergence_window: int = 10,
) -> OrderFlowMetrics | None:
    """Compute deterministic order flow metrics strictly on closed candles.

    Requires `taker_buy_base_volume` on the latest closed candle.
    Returns None if taker buy data is not available.
    """
    if not candles:
        return None

    latest = candles[-1]
    if latest.taker_buy_base_volume is None:
        return None

    # Taker buy ratio on the latest closed candle
    if latest.base_volume > 0.0:
        taker_buy_ratio = min(1.0, max(0.0, latest.taker_buy_base_volume / latest.base_volume))
    else:
        taker_buy_ratio = 0.5

    # Rolling CVD across the past `window` candles
    sub_candles = candles[-window:]
    deltas: list[float] = []
    for c in sub_candles:
        if c.taker_buy_base_volume is not None:
            deltas.append(2.0 * c.taker_buy_base_volume - c.base_volume)
        else:
            deltas.append(0.0)

    cvd_20 = sum(deltas)

    # CVD divergence detection on recent swing window
    divergence: Literal["bullish", "bearish", "none"] = "none"
    if len(sub_candles) >= divergence_window:
        div_candles = sub_candles[-divergence_window:]
        div_deltas = deltas[-divergence_window:]

        # Running cumulative delta in divergence window
        running_cvd: list[float] = []
        acc = 0.0
        for d in div_deltas:
            acc += d
            running_cvd.append(acc)

        latest_high = div_candles[-1].high
        prev_highs = [c.high for c in div_candles[:-1]]
        latest_low = div_candles[-1].low
        prev_lows = [c.low for c in div_candles[:-1]]

        latest_cvd_val = running_cvd[-1]
        prev_cvds = running_cvd[:-1]

        # Bearish divergence: price made a new high or equal high,
        # but CVD failed to confirm (exhaustion)
        if prev_highs and latest_high >= max(prev_highs) and latest_cvd_val < max(prev_cvds):
            divergence = "bearish"
        # Bullish divergence: price made a new low or equal low,
        # but CVD made a higher low (absorption)
        elif prev_lows and latest_low <= min(prev_lows) and latest_cvd_val > min(prev_cvds):
            divergence = "bullish"

    return OrderFlowMetrics(
        taker_buy_ratio=taker_buy_ratio,
        cvd_20=cvd_20,
        cvd_divergence=divergence,
        has_order_flow=True,
    )
