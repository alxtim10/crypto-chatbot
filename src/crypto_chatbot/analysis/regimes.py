"""Deterministic market regime classification (PROJECT_PLAN §5.6)."""

from __future__ import annotations

from crypto_chatbot.schemas.analysis import Regime, TechnicalMetrics

HIGH_VOLATILITY_ATR_PERCENTILE = 80.0
TRENDING_ADX_MIN = 25.0
RANGING_ADX_MAX = 20.0


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
