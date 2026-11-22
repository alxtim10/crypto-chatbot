"""Deterministic market regime classification (PROJECT_PLAN §5.6)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from crypto_chatbot.analysis.statistics import realized_volatility_cc
from crypto_chatbot.schemas.analysis import Regime, TechnicalMetrics, VolatilityRegime
from crypto_chatbot.schemas.market import MarketCandle

HIGH_VOLATILITY_ATR_PERCENTILE = 80.0
TRENDING_ADX_MIN = 25.0
RANGING_ADX_MAX = 20.0

# Volatility-regime percentile thresholds (realized-vol rank within the trailing window).
VOL_REGIME_LOW_PERCENTILE = 30.0
VOL_REGIME_HIGH_PERCENTILE = 70.0
VOL_REGIME_WINDOW = 20
VOL_REGIME_LOOKBACK = 100


class VolatilityRegimeResult:
    """Deterministic volatility-regime classification with its driving percentile."""

    __slots__ = ("percentile", "realized_vol", "regime")

    def __init__(
        self,
        regime: VolatilityRegime | None,
        realized_vol: float | None,
        percentile: float | None,
    ) -> None:
        self.regime = regime
        self.realized_vol = realized_vol
        self.percentile = percentile


def classify_regime(metrics: TechnicalMetrics) -> Regime | None:
    """Classify current market regime from validated technical measurements.

    Returns `None` when required warm-up metrics are unavailable.
    """
    atr_pct_rank = metrics.atr_percentile_100
    adx = metrics.adx_14
    plus_di = metrics.plus_di_14
    minus_di = metrics.minus_di_14
    ema_50 = metrics.ema_50

    if atr_pct_rank is None or adx is None or plus_di is None or minus_di is None or ema_50 is None:
        return None

    if atr_pct_rank >= HIGH_VOLATILITY_ATR_PERCENTILE:
        return Regime.HIGH_VOLATILITY

    if adx >= TRENDING_ADX_MIN:
        if plus_di > minus_di and metrics.last_close > ema_50:
            return Regime.TRENDING_UP
        if minus_di > plus_di and metrics.last_close < ema_50:
            return Regime.TRENDING_DOWN

    if adx < RANGING_ADX_MAX:
        return Regime.RANGING

    return Regime.TRANSITIONAL


def classify_volatility_regime(
    candles: Sequence[MarketCandle],
    *,
    vol_window: int = VOL_REGIME_WINDOW,
    lookback: int = VOL_REGIME_LOOKBACK,
) -> VolatilityRegimeResult:
    """Classify the current volatility regime by percentile rank of realized volatility.

    Computes close-to-close realized volatility over a rolling `vol_window` for each bar in
    the trailing `lookback` window, then ranks the latest value. Low/high thresholds are
    fixed percentile cutoffs (30/70). This is a deterministic complement to the ATR-based
    regime and never expresses a probability. Returns a `None` regime when history is
    insufficient to compute a stable percentile.
    """
    if vol_window < 2:
        raise ValueError("vol_window must be >= 2")
    if lookback < 2:
        raise ValueError("lookback must be >= 2")

    # Need enough bars so the trailing window of realized-vol samples is well populated.
    min_required = vol_window + lookback
    closes = [c.close for c in candles]
    if len(closes) < min_required:
        return VolatilityRegimeResult(None, None, None)

    closes_arr = np.asarray(closes, dtype=np.float64)
    samples: list[float] = []
    for end in range(len(closes_arr) - lookback + 1, len(closes_arr) + 1):
        window_closes = closes_arr[:end]
        rv = realized_volatility_cc(window_closes, window=vol_window)
        if rv is not None:
            samples.append(rv)

    if len(samples) < lookback // 2:
        return VolatilityRegimeResult(None, None, None)

    current = samples[-1]
    arr = np.asarray(samples, dtype=np.float64)
    below = float(np.sum(arr < current))
    denom = float(len(arr) - 1)
    percentile = 100.0 * below / denom if denom > 0.0 else 50.0
    percentile = min(100.0, max(0.0, percentile))

    if percentile <= VOL_REGIME_LOW_PERCENTILE:
        regime = VolatilityRegime.LOW
    elif percentile >= VOL_REGIME_HIGH_PERCENTILE:
        regime = VolatilityRegime.HIGH
    else:
        regime = VolatilityRegime.NORMAL

    return VolatilityRegimeResult(regime, current, percentile)
