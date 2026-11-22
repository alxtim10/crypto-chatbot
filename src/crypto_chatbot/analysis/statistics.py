"""Deterministic, causal statistical-structure estimators for crypto price series.

These complement the technical indicators in `indicators.py` with model-free statistical
measurements inspired by standard quant methodology (realized volatility estimators,
the Hurst exponent, and the Ornstein-Uhlenbeck mean-reversion half-life). Every function is
causal (uses only closed candles up to the evaluation bar), returns `None` when history is
insufficient, and never labels its output as a success probability.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict

from crypto_chatbot.analysis.indicators import validate_closed_candles
from crypto_chatbot.schemas.market import MarketCandle

# Trading-period count used to annualize realized volatility of daily log returns.
ANNUALIZATION_TRADING_DAYS = 365

STATISTICS_PARAMETERS: dict[str, int | float | str] = {
    "realized_vol_window": 20,
    "realized_vol_annualization_days": ANNUALIZATION_TRADING_DAYS,
    "hurst_min_window": 64,
    "hurst_max_lag": 20,
    "ou_half_life_window": 120,
}

# Minimum closed-candle history per statistic.
MIN_HISTORY_REALIZED_VOL = 21
MIN_HISTORY_PARKINSON = 20
MIN_HISTORY_GARMAN_KLASS = 20
MIN_HISTORY_HURST = 64
MIN_HISTORY_OU_HALF_LIFE = 30

_EPS = 1e-12


class StatisticalMetrics(BaseModel):
    """Model-free statistical-structure measurements for a closed candle series."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    realized_vol_cc: float | None = None
    realized_vol_cc_annualized: float | None = None
    parkinson_vol: float | None = None
    garman_klass_vol: float | None = None
    hurst_exponent: float | None = None
    ou_half_life: float | None = None


def _as_1d_float64(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    return arr


def log_return_series(closes: ArrayLike) -> NDArray[np.float64]:
    """Natural-log returns `ln(close_t / close_{t-1})`; first element and non-positive are NaN."""
    arr = _as_1d_float64(closes, "closes")
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    for t in range(1, n):
        prev = arr[t - 1]
        curr = arr[t]
        if np.isfinite(prev) and np.isfinite(curr) and prev > 0.0 and curr > 0.0:
            out[t] = math.log(curr / prev)
    return out


def realized_volatility_cc(
    closes: ArrayLike,
    *,
    window: int = 20,
) -> float | None:
    """Close-to-close realized volatility: sample std of the last `window` log returns.

    Uses a sample (ddof=1) standard deviation over the most recent `window` log returns.
    Returns `None` if fewer than `window` finite log returns are available.
    """
    if window < 2:
        raise ValueError("window must be >= 2")
    log_rets = log_return_series(closes)
    finite = log_rets[np.isfinite(log_rets)]
    if len(finite) < window:
        return None
    recent = finite[-window:]
    vol = float(np.std(recent, ddof=1))
    return vol if math.isfinite(vol) else None


def annualize_volatility(
    per_bar_vol: float,
    *,
    bars_per_year: float,
) -> float:
    """Scale a per-bar volatility to an annualized figure by sqrt(bars_per_year)."""
    if bars_per_year <= 0.0:
        raise ValueError("bars_per_year must be > 0")
    return per_bar_vol * math.sqrt(bars_per_year)


def parkinson_volatility(
    highs: ArrayLike,
    lows: ArrayLike,
    *,
    window: int = 20,
) -> float | None:
    """Parkinson high-low range volatility estimator over the last `window` bars.

    `sigma = sqrt( mean( ln(high/low)^2 ) / (4 ln 2) )`. Returns `None` when the window
    contains a non-positive or non-finite high/low pair.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    h = _as_1d_float64(highs, "highs")
    low_arr = _as_1d_float64(lows, "lows")
    if len(h) != len(low_arr):
        raise ValueError("highs and lows must have equal length")
    if len(h) < window:
        return None
    h_win = h[-window:]
    low_win = low_arr[-window:]
    if not (np.all(np.isfinite(h_win)) and np.all(np.isfinite(low_win))):
        return None
    if np.any(h_win <= 0.0) or np.any(low_win <= 0.0) or np.any(h_win < low_win):
        return None
    log_hl = np.log(h_win / low_win)
    factor = 1.0 / (4.0 * math.log(2.0))
    var = factor * float(np.mean(log_hl**2))
    if var < 0.0 or not math.isfinite(var):
        return None
    return math.sqrt(var)


def garman_klass_volatility(
    opens: ArrayLike,
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    *,
    window: int = 20,
) -> float | None:
    """Garman-Klass OHLC volatility estimator over the last `window` bars.

    Per-bar term: `0.5 * ln(high/low)^2 - (2 ln 2 - 1) * ln(close/open)^2`.
    Returns `None` when the window contains invalid (non-positive/non-finite) OHLC values.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    o = _as_1d_float64(opens, "opens")
    h = _as_1d_float64(highs, "highs")
    low_arr = _as_1d_float64(lows, "lows")
    c = _as_1d_float64(closes, "closes")
    if not (len(o) == len(h) == len(low_arr) == len(c)):
        raise ValueError("opens, highs, lows, and closes must have equal length")
    if len(c) < window:
        return None
    o_w, h_w, low_w, c_w = o[-window:], h[-window:], low_arr[-window:], c[-window:]
    for arr in (o_w, h_w, low_w, c_w):
        if not np.all(np.isfinite(arr)) or np.any(arr <= 0.0):
            return None
    if np.any(h_w < low_w):
        return None
    log_hl = np.log(h_w / low_w)
    log_co = np.log(c_w / o_w)
    per_bar = 0.5 * log_hl**2 - (2.0 * math.log(2.0) - 1.0) * log_co**2
    var = float(np.mean(per_bar))
    if var < 0.0 or not math.isfinite(var):
        return None
    return math.sqrt(var)


def hurst_exponent(
    closes: ArrayLike,
    *,
    max_lag: int = 20,
    min_window: int = 64,
) -> float | None:
    """Hurst exponent via the rescaled log-variance of lagged differences on log prices.

    H ~ 0.5 is a random walk, H > 0.5 trending (persistent), H < 0.5 mean-reverting.
    Estimated by regressing `ln(std(diff_lag))` on `ln(lag)` for lags `2..max_lag`; the
    slope equals H. Returns `None` when history is shorter than `min_window` or the price
    path is degenerate (zero variance across all lags).
    """
    if max_lag < 2:
        raise ValueError("max_lag must be >= 2")
    if min_window < max_lag + 1:
        raise ValueError("min_window must exceed max_lag")
    arr = _as_1d_float64(closes, "closes")
    finite = arr[np.isfinite(arr)]
    if len(finite) < min_window:
        return None
    if np.any(finite <= 0.0):
        return None
    log_prices = np.log(finite)

    lags = np.arange(2, max_lag + 1)
    log_lags: list[float] = []
    log_tau: list[float] = []
    for lag in lags:
        diffs = log_prices[lag:] - log_prices[:-lag]
        if len(diffs) < 2:
            continue
        std = float(np.std(diffs, ddof=1))
        if std <= _EPS or not math.isfinite(std):
            continue
        log_lags.append(math.log(float(lag)))
        log_tau.append(math.log(std))

    if len(log_lags) < 2:
        return None

    slope, _intercept = np.polyfit(np.asarray(log_lags), np.asarray(log_tau), 1)
    h_val = float(slope)
    if not math.isfinite(h_val):
        return None
    return min(1.0, max(0.0, h_val))


def ou_half_life(
    closes: ArrayLike,
    *,
    window: int = 120,
) -> float | None:
    """Ornstein-Uhlenbeck mean-reversion half-life (in bars) on log prices.

    Fits `delta_y_t = alpha + beta * y_{t-1} + eps` by OLS over the last `window` log
    prices; half-life = `-ln(2) / ln(1 + beta)`. Returns `None` when the series is not
    mean-reverting (`beta >= 0`) or history/variance is insufficient.
    """
    if window < 10:
        raise ValueError("window must be >= 10")
    arr = _as_1d_float64(closes, "closes")
    finite = arr[np.isfinite(arr)]
    if len(finite) < window:
        return None
    tail = finite[-window:]
    if np.any(tail <= 0.0):
        return None
    y = np.log(tail)

    y_lag = y[:-1]
    delta = y[1:] - y_lag
    if len(y_lag) < 3:
        return None
    if float(np.var(y_lag)) <= _EPS:
        return None

    design = np.column_stack([np.ones_like(y_lag), y_lag])
    coeffs, *_ = np.linalg.lstsq(design, delta, rcond=None)
    beta = float(coeffs[1])
    if not math.isfinite(beta) or beta >= 0.0:
        return None
    one_plus_beta = 1.0 + beta
    if one_plus_beta <= 0.0 or one_plus_beta >= 1.0:
        return None
    half_life = -math.log(2.0) / math.log(one_plus_beta)
    if not math.isfinite(half_life) or half_life <= 0.0:
        return None
    return half_life


def compute_statistical_metrics(
    candles: Sequence[MarketCandle],
    *,
    realized_vol_window: int = 20,
    hurst_max_lag: int = 20,
    hurst_min_window: int = 64,
    ou_window: int = 120,
) -> StatisticalMetrics:
    """Compute all model-free statistical measurements for a closed candle series."""
    validate_closed_candles(candles)
    n = len(candles)

    opens = np.fromiter((c.open for c in candles), dtype=np.float64, count=n)
    highs = np.fromiter((c.high for c in candles), dtype=np.float64, count=n)
    lows = np.fromiter((c.low for c in candles), dtype=np.float64, count=n)
    closes = np.fromiter((c.close for c in candles), dtype=np.float64, count=n)

    rv_cc = (
        realized_volatility_cc(closes, window=realized_vol_window)
        if n >= MIN_HISTORY_REALIZED_VOL
        else None
    )
    rv_cc_annual = (
        annualize_volatility(rv_cc, bars_per_year=ANNUALIZATION_TRADING_DAYS)
        if rv_cc is not None
        else None
    )
    parkinson = (
        parkinson_volatility(highs, lows, window=realized_vol_window)
        if n >= MIN_HISTORY_PARKINSON
        else None
    )
    garman_klass = (
        garman_klass_volatility(opens, highs, lows, closes, window=realized_vol_window)
        if n >= MIN_HISTORY_GARMAN_KLASS
        else None
    )
    hurst = (
        hurst_exponent(closes, max_lag=hurst_max_lag, min_window=hurst_min_window)
        if n >= MIN_HISTORY_HURST
        else None
    )
    half_life = ou_half_life(closes, window=ou_window) if n >= MIN_HISTORY_OU_HALF_LIFE else None

    return StatisticalMetrics(
        realized_vol_cc=rv_cc,
        realized_vol_cc_annualized=rv_cc_annual,
        parkinson_vol=parkinson,
        garman_klass_vol=garman_klass,
        hurst_exponent=hurst,
        ou_half_life=half_life,
    )
