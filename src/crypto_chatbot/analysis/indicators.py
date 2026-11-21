"""Deterministic, causal technical indicator formulas (PROJECT_PLAN §5.4-5.6)."""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict

from crypto_chatbot.schemas.analysis import (
    MtfRsiAlignment,
    TechnicalMetrics,
)
from crypto_chatbot.schemas.market import (
    MarketCandle,
    QualityFlag,
    Timeframe,
    UtcDatetime,
)

CALC_VERSION = "1.0.0"

INDICATOR_PARAMETERS: dict[str, int | float | str] = {
    "return_short": 1,
    "return_long": 20,
    "ema_fast": 20,
    "ema_mid": 50,
    "ema_slow": 200,
    "rsi_period": 14,
    "macd_fast": 12,
    "macd_slow": 26,
    "macd_signal": 9,
    "atr_period": 14,
    "adx_period": 14,
    "relative_volume_period": 20,
    "atr_percentile_window": 100,
    "swing_pivot_k": 3,
    "failed_breakout_window": 3,
    "regime_atr_pct_high_vol": 80.0,
    "regime_adx_trend_min": 25.0,
    "regime_adx_range_max": 20.0,
}

# Minimum closed-candle history per indicator (3x longest period, §5.4).
MIN_HISTORY_RETURN_1 = 2
MIN_HISTORY_RETURN_20 = 21
MIN_HISTORY_EMA_20 = 60
MIN_HISTORY_EMA_50 = 150
MIN_HISTORY_EMA_200 = 600
MIN_HISTORY_RSI_14 = 43
MIN_HISTORY_MACD = 87
MIN_HISTORY_ATR_14 = 42
MIN_HISTORY_ADX_14 = 56
MIN_HISTORY_REL_VOL_20 = 21
MIN_HISTORY_ATR_PCT_100 = 113

_BOUND_EPS = 1e-12


class PreviousPeriodLevels(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    prev_day_high: float | None = None
    prev_day_low: float | None = None
    prev_day_close: float | None = None
    prev_week_high: float | None = None
    prev_week_low: float | None = None
    prev_week_close: float | None = None


class MtfRsiResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    alignment: MtfRsiAlignment | None
    rsi_by_timeframe: dict[Timeframe, float | None]


class IndicatorResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    as_of: UtcDatetime
    metrics: TechnicalMetrics
    quality_flags: tuple[QualityFlag, ...] = ()
    mtf_rsi_values: dict[Timeframe, float | None] = {}


def _require_utc(dt: datetime, label: str) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware UTC")
    return dt.astimezone(UTC)


def _clamp_pct_scalar(value: float) -> float:
    if -_BOUND_EPS <= value < 0.0:
        return 0.0
    if 100.0 < value <= 100.0 + _BOUND_EPS:
        return 100.0
    return value


def _clamp_pct_array(arr: NDArray[np.float64]) -> NDArray[np.float64]:
    out = arr.copy()
    neg_mask = np.isfinite(out) & (out < 0.0) & (out >= -_BOUND_EPS)
    pos_mask = np.isfinite(out) & (out > 100.0) & (out <= 100.0 + _BOUND_EPS)
    out[neg_mask] = 0.0
    out[pos_mask] = 100.0
    return out


def validate_closed_candles(candles: Sequence[MarketCandle]) -> None:
    """Ensure candles are non-empty, closed, uniform in symbol/timeframe, and ascending."""
    if not candles:
        raise ValueError("candles must not be empty")

    first = candles[0]
    for candle in candles:
        if not candle.is_closed:
            raise ValueError("closed-candle analysis received an open candle")
        if candle.canonical_symbol != first.canonical_symbol:
            raise ValueError("candles contain mixed canonical_symbol values")
        if candle.timeframe is not first.timeframe:
            raise ValueError("candles contain mixed timeframe values")

    for prev, curr in itertools.pairwise(candles):
        if prev.open_time >= curr.open_time:
            raise ValueError("candles must be strictly ascending by open_time")


def _as_1d_float64(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    return arr


def _smoothed_ma(values: ArrayLike, period: int, alpha: float) -> NDArray[np.float64]:
    """Causal exponential/Wilder MA seeded with the SMA of the first `period` finite values."""
    if period < 1:
        raise ValueError("period must be >= 1")
    arr = _as_1d_float64(values, "values")
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)

    finite_indices = np.flatnonzero(np.isfinite(arr))
    if len(finite_indices) == 0:
        return out
    start = int(finite_indices[0])
    if n - start < period:
        return out

    seed_slice = arr[start : start + period]
    if not np.all(np.isfinite(seed_slice)):
        return out

    out[start + period - 1] = float(np.mean(seed_slice))
    decay = 1.0 - alpha
    for i in range(start + period, n):
        if not np.isfinite(arr[i]) or not np.isfinite(out[i - 1]):
            out[i] = np.nan
        else:
            out[i] = alpha * float(arr[i]) + decay * float(out[i - 1])
    return out


def ema_series(values: ArrayLike, period: int) -> NDArray[np.float64]:
    """Exponential moving average with alpha=2/(period+1) and SMA seed (TradingView ta.ema)."""
    return _smoothed_ma(values, period, alpha=2.0 / (period + 1))


def rma_series(values: ArrayLike, period: int) -> NDArray[np.float64]:
    """Wilder moving average with alpha=1/period and SMA seed (TradingView ta.rma)."""
    return _smoothed_ma(values, period, alpha=1.0 / period)


def simple_return_series(closes: ArrayLike, periods: int) -> NDArray[np.float64]:
    """Simple bar-over-bar returns `close_t / close_{t-periods} - 1`."""
    if periods < 1:
        raise ValueError("periods must be >= 1")
    arr = _as_1d_float64(closes, "closes")
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)
    for t in range(periods, n):
        curr = arr[t]
        prev = arr[t - periods]
        if np.isfinite(curr) and np.isfinite(prev) and prev > 0.0:
            out[t] = curr / prev - 1.0
    return out


def rsi_series(
    closes: ArrayLike,
    period: int = 14,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Wilder RSI (TradingView ta.rsi semantics). Returns `(rsi, is_flat)`."""
    if period < 1:
        raise ValueError("period must be >= 1")
    arr = _as_1d_float64(closes, "closes")
    n = len(arr)
    rsi = np.full(n, np.nan, dtype=np.float64)
    is_flat = np.zeros(n, dtype=np.bool_)
    if n < 2:
        return rsi, is_flat

    deltas = np.full(n, np.nan, dtype=np.float64)
    deltas[1:] = arr[1:] - arr[:-1]

    finite_deltas = np.isfinite(deltas)
    gains = np.where(finite_deltas, np.maximum(deltas, 0.0), np.nan)
    losses = np.where(finite_deltas, np.maximum(-deltas, 0.0), np.nan)

    avg_gain = rma_series(gains, period)
    avg_loss = rma_series(losses, period)

    for i in range(n):
        ag = avg_gain[i]
        al = avg_loss[i]
        if not (np.isfinite(ag) and np.isfinite(al)):
            continue
        if ag == 0.0 and al == 0.0:
            is_flat[i] = True
            rsi[i] = 100.0
        elif al == 0.0:
            rsi[i] = 100.0
        elif ag == 0.0:
            rsi[i] = 0.0
        else:
            rs = ag / al
            rsi[i] = _clamp_pct_scalar(100.0 - 100.0 / (1.0 + rs))

    return rsi, is_flat


def macd_series(
    closes: ArrayLike,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """MACD line, signal line, and histogram (`EMA_fast - EMA_slow`, `EMA_signal(MACD)`)."""
    if fast < 1 or slow <= fast or signal < 1:
        raise ValueError("MACD requires 0 < fast < slow and signal >= 1")
    arr = _as_1d_float64(closes, "closes")
    macd_line = ema_series(arr, fast) - ema_series(arr, slow)
    signal_line = ema_series(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def true_range_series(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    *,
    first_bar_hl: bool = True,
) -> NDArray[np.float64]:
    """True Range. `first_bar_hl=True` matches `ta.atr`; `False` matches `ta.dmi`."""
    h = _as_1d_float64(highs, "highs")
    low_arr = _as_1d_float64(lows, "lows")
    c = _as_1d_float64(closes, "closes")
    if not (len(h) == len(low_arr) == len(c)):
        raise ValueError("highs, lows, and closes must have equal length")

    n = len(h)
    tr = np.full(n, np.nan, dtype=np.float64)
    if n == 0:
        return tr

    if first_bar_hl and np.isfinite(h[0]) and np.isfinite(low_arr[0]):
        tr[0] = float(h[0] - low_arr[0])

    for t in range(1, n):
        if np.isfinite(h[t]) and np.isfinite(low_arr[t]) and np.isfinite(c[t - 1]):
            tr[t] = max(
                float(h[t] - low_arr[t]),
                abs(float(h[t] - c[t - 1])),
                abs(float(low_arr[t] - c[t - 1])),
            )
    return tr


def atr_series(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    period: int = 14,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Wilder ATR and ATR% (`ATR / close`)."""
    h = _as_1d_float64(highs, "highs")
    low_arr = _as_1d_float64(lows, "lows")
    c = _as_1d_float64(closes, "closes")
    tr = true_range_series(h, low_arr, c, first_bar_hl=True)
    atr = rma_series(tr, period)
    atr_pct = np.where(np.isfinite(atr) & np.isfinite(c) & (c > 0.0), atr / c, np.nan)
    return atr, atr_pct


def adx_dmi_series(
    highs: ArrayLike,
    lows: ArrayLike,
    closes: ArrayLike,
    period: int = 14,
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    """Wilder ADX, +DI, and -DI (TradingView `ta.dmi` convention)."""
    if period < 1:
        raise ValueError("period must be >= 1")
    h = _as_1d_float64(highs, "highs")
    low_arr = _as_1d_float64(lows, "lows")
    c = _as_1d_float64(closes, "closes")
    if not (len(h) == len(low_arr) == len(c)):
        raise ValueError("highs, lows, and closes must have equal length")

    n = len(h)
    tr = true_range_series(h, low_arr, c, first_bar_hl=False)
    plus_dm = np.full(n, np.nan, dtype=np.float64)
    minus_dm = np.full(n, np.nan, dtype=np.float64)

    for t in range(1, n):
        if not (
            np.isfinite(h[t])
            and np.isfinite(h[t - 1])
            and np.isfinite(low_arr[t])
            and np.isfinite(low_arr[t - 1])
        ):
            continue
        up = float(h[t] - h[t - 1])
        down = float(low_arr[t - 1] - low_arr[t])
        plus_dm[t] = up if (up > down and up > 0.0) else 0.0
        minus_dm[t] = down if (down > up and down > 0.0) else 0.0

    smoothed_tr = rma_series(tr, period)
    smoothed_plus = rma_series(plus_dm, period)
    smoothed_minus = rma_series(minus_dm, period)

    plus_di = np.full(n, np.nan, dtype=np.float64)
    minus_di = np.full(n, np.nan, dtype=np.float64)
    dx = np.full(n, np.nan, dtype=np.float64)

    for t in range(n):
        str_val = smoothed_tr[t]
        sp_val = smoothed_plus[t]
        sm_val = smoothed_minus[t]
        if not (np.isfinite(str_val) and np.isfinite(sp_val) and np.isfinite(sm_val)):
            continue
        if str_val == 0.0:
            pdi = 0.0
            mdi = 0.0
        else:
            pdi = 100.0 * float(sp_val) / float(str_val)
            mdi = 100.0 * float(sm_val) / float(str_val)
        plus_di[t] = pdi
        minus_di[t] = mdi
        di_sum = pdi + mdi
        dx[t] = 0.0 if di_sum == 0.0 else 100.0 * abs(pdi - mdi) / di_sum

    adx = rma_series(dx, period)
    return _clamp_pct_array(adx), _clamp_pct_array(plus_di), _clamp_pct_array(minus_di)


def relative_volume_series(
    volumes: ArrayLike,
    period: int = 20,
) -> tuple[NDArray[np.float64], NDArray[np.bool_]]:
    """Relative volume: `volume_t / mean(previous period volumes, excluding t)`."""
    if period < 1:
        raise ValueError("period must be >= 1")
    arr = _as_1d_float64(volumes, "volumes")
    n = len(arr)
    rel_vol = np.full(n, np.nan, dtype=np.float64)
    zero_mean = np.zeros(n, dtype=np.bool_)

    for t in range(period, n):
        curr = arr[t]
        window = arr[t - period : t]
        if not np.isfinite(curr) or not np.all(np.isfinite(window)):
            continue
        mean_vol = float(np.mean(window))
        if mean_vol == 0.0:
            zero_mean[t] = True
        else:
            rel_vol[t] = float(curr) / mean_vol

    return rel_vol, zero_mean


def atr_percentile_series(
    atr_pct: ArrayLike,
    window: int = 100,
) -> NDArray[np.float64]:
    """Strict percentile rank (0-100) of current `atr_pct` within last `window` closed bars."""
    if window < 2:
        raise ValueError("window must be >= 2")
    arr = _as_1d_float64(atr_pct, "atr_pct")
    n = len(arr)
    out = np.full(n, np.nan, dtype=np.float64)

    for t in range(window - 1, n):
        slice_w = arr[t - window + 1 : t + 1]
        if not np.all(np.isfinite(slice_w)):
            continue
        below = float(np.sum(slice_w < slice_w[-1]))
        out[t] = _clamp_pct_scalar(100.0 * below / float(window - 1))

    return out


def _aggregate_complete_period(
    candles: Sequence[MarketCandle],
    period_start: datetime,
    period_end: datetime,
    timeframe: Timeframe,
) -> tuple[float, float, float] | None:
    expected_bars = round((period_end - period_start) / timeframe.duration)
    bars = [c for c in candles if c.open_time >= period_start and c.close_time <= period_end]
    if len(bars) != expected_bars:
        return None
    if bars[0].open_time != period_start or bars[-1].close_time != period_end:
        return None
    for prev, curr in itertools.pairwise(bars):
        if curr.open_time != prev.close_time:
            return None
    return (
        max(c.high for c in bars),
        min(c.low for c in bars),
        bars[-1].close,
    )


def compute_previous_period_levels(
    candles: Sequence[MarketCandle],
    *,
    as_of: datetime,
) -> PreviousPeriodLevels:
    """High/low/close of the previous complete UTC day and ISO week at `as_of`."""
    as_of_utc = _require_utc(as_of, "as_of")
    validate_closed_candles(candles)

    eligible = [c for c in candles if c.close_time <= as_of_utc]
    if not eligible:
        return PreviousPeriodLevels()

    tf = eligible[0].timeframe
    day_start = as_of_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    prev_day = _aggregate_complete_period(
        eligible,
        day_start - timedelta(days=1),
        day_start,
        tf,
    )

    week_start = day_start - timedelta(days=as_of_utc.weekday())
    prev_week = _aggregate_complete_period(
        eligible,
        week_start - timedelta(days=7),
        week_start,
        tf,
    )

    return PreviousPeriodLevels(
        prev_day_high=prev_day[0] if prev_day is not None else None,
        prev_day_low=prev_day[1] if prev_day is not None else None,
        prev_day_close=prev_day[2] if prev_day is not None else None,
        prev_week_high=prev_week[0] if prev_week is not None else None,
        prev_week_low=prev_week[1] if prev_week is not None else None,
        prev_week_close=prev_week[2] if prev_week is not None else None,
    )


def compute_mtf_rsi_alignment(
    candles_by_timeframe: Mapping[Timeframe, Sequence[MarketCandle]],
    *,
    as_of: datetime,
    use_available_time: bool = False,
) -> MtfRsiResult:
    """Multi-timeframe RSI(14) alignment across `1h`, `4h`, `1d` strictly at `as_of` (§5.5)."""
    as_of_utc = _require_utc(as_of, "as_of")
    rsi_by_tf: dict[Timeframe, float | None] = {}

    for tf in (Timeframe.H1, Timeframe.H4, Timeframe.D1):
        seq = candles_by_timeframe.get(tf)
        if not seq:
            rsi_by_tf[tf] = None
            continue

        validate_closed_candles(seq)
        if seq[0].timeframe is not tf:
            raise ValueError(
                f"candles_by_timeframe[{tf.value}] contains {seq[0].timeframe.value} candles"
            )

        filtered = [
            c
            for c in seq
            if (c.available_time if use_available_time else c.close_time) <= as_of_utc
        ]
        if len(filtered) < MIN_HISTORY_RSI_14:
            rsi_by_tf[tf] = None
            continue

        closes = np.fromiter((c.close for c in filtered), dtype=np.float64, count=len(filtered))
        rsi_vals, _ = rsi_series(closes, period=14)
        latest = float(rsi_vals[-1])
        rsi_by_tf[tf] = _clamp_pct_scalar(latest) if np.isfinite(latest) else None

    values = [rsi_by_tf[tf] for tf in (Timeframe.H1, Timeframe.H4, Timeframe.D1)]
    if any(v is None for v in values):
        alignment: MtfRsiAlignment | None = None
    else:
        concrete = [v for v in values if v is not None]
        if all(v > 50.0 for v in concrete):
            alignment = MtfRsiAlignment.ALIGNED_BULLISH
        elif all(v < 50.0 for v in concrete):
            alignment = MtfRsiAlignment.ALIGNED_BEARISH
        else:
            alignment = MtfRsiAlignment.MIXED

    return MtfRsiResult(alignment=alignment, rsi_by_timeframe=rsi_by_tf)


def _finite_last(arr: NDArray[np.float64], n: int, min_history: int) -> float | None:
    if n < min_history:
        return None
    val = float(arr[-1])
    return val if np.isfinite(val) else None


def compute_technical_metrics(
    candles: Sequence[MarketCandle],
    *,
    mtf_candles: Mapping[Timeframe, Sequence[MarketCandle]] | None = None,
    use_available_time: bool = False,
) -> IndicatorResult:
    """Compute all deterministic §5.6 metrics for a closed candle series."""
    validate_closed_candles(candles)
    n = len(candles)
    as_of = candles[-1].close_time

    highs = np.fromiter((c.high for c in candles), dtype=np.float64, count=n)
    lows = np.fromiter((c.low for c in candles), dtype=np.float64, count=n)
    closes = np.fromiter((c.close for c in candles), dtype=np.float64, count=n)
    volumes = np.fromiter((c.base_volume for c in candles), dtype=np.float64, count=n)

    ret_1 = simple_return_series(closes, 1)
    ret_20 = simple_return_series(closes, 20)
    ema_20 = ema_series(closes, 20)
    ema_50 = ema_series(closes, 50)
    ema_200 = ema_series(closes, 200)
    rsi_14, is_flat = rsi_series(closes, 14)
    macd_line, macd_signal, macd_hist = macd_series(closes, 12, 26, 9)
    atr_14, atr_pct = atr_series(highs, lows, closes, 14)
    adx_14, plus_di_14, minus_di_14 = adx_dmi_series(highs, lows, closes, 14)
    rel_vol_20, zero_vol_mask = relative_volume_series(volumes, 20)
    atr_pct_100 = atr_percentile_series(atr_pct, 100)

    flags: list[QualityFlag] = []
    if n < MIN_HISTORY_EMA_200:
        flags.append(QualityFlag.SHORT_HISTORY)

    rsi_val = _finite_last(rsi_14, n, MIN_HISTORY_RSI_14)
    if rsi_val is not None:
        rsi_val = _clamp_pct_scalar(rsi_val)
        if bool(is_flat[-1]):
            flags.append(QualityFlag.FLAT_PRICE)

    rel_vol_val = _finite_last(rel_vol_20, n, MIN_HISTORY_REL_VOL_20)
    if n >= MIN_HISTORY_REL_VOL_20 and bool(zero_vol_mask[-1]):
        flags.append(QualityFlag.ZERO_VOLUME)

    macd_ok = n >= MIN_HISTORY_MACD
    macd_l = _finite_last(macd_line, n, MIN_HISTORY_MACD) if macd_ok else None
    macd_s = _finite_last(macd_signal, n, MIN_HISTORY_MACD) if macd_ok else None
    macd_h = _finite_last(macd_hist, n, MIN_HISTORY_MACD) if macd_ok else None

    adx_val = _finite_last(adx_14, n, MIN_HISTORY_ADX_14)
    pdi_val = _finite_last(plus_di_14, n, MIN_HISTORY_ADX_14)
    mdi_val = _finite_last(minus_di_14, n, MIN_HISTORY_ADX_14)
    atr_pct_rank = _finite_last(atr_pct_100, n, MIN_HISTORY_ATR_PCT_100)

    mtf_result: MtfRsiResult | None = None
    if mtf_candles is not None:
        mtf_result = compute_mtf_rsi_alignment(
            mtf_candles,
            as_of=as_of,
            use_available_time=use_available_time,
        )

    levels = compute_previous_period_levels(candles, as_of=as_of)

    metrics = TechnicalMetrics(
        last_close=float(closes[-1]),
        return_1=_finite_last(ret_1, n, MIN_HISTORY_RETURN_1),
        return_20=_finite_last(ret_20, n, MIN_HISTORY_RETURN_20),
        ema_20=_finite_last(ema_20, n, MIN_HISTORY_EMA_20),
        ema_50=_finite_last(ema_50, n, MIN_HISTORY_EMA_50),
        ema_200=_finite_last(ema_200, n, MIN_HISTORY_EMA_200),
        rsi_14=rsi_val,
        macd_line=macd_l,
        macd_signal=macd_s,
        macd_histogram=macd_h,
        atr_14=_finite_last(atr_14, n, MIN_HISTORY_ATR_14),
        atr_pct=_finite_last(atr_pct, n, MIN_HISTORY_ATR_14),
        adx_14=_clamp_pct_scalar(adx_val) if adx_val is not None else None,
        plus_di_14=_clamp_pct_scalar(pdi_val) if pdi_val is not None else None,
        minus_di_14=_clamp_pct_scalar(mdi_val) if mdi_val is not None else None,
        relative_volume_20=rel_vol_val,
        atr_percentile_100=_clamp_pct_scalar(atr_pct_rank) if atr_pct_rank is not None else None,
        mtf_rsi_alignment=mtf_result.alignment if mtf_result is not None else None,
        prev_day_high=levels.prev_day_high,
        prev_day_low=levels.prev_day_low,
        prev_day_close=levels.prev_day_close,
        prev_week_high=levels.prev_week_high,
        prev_week_low=levels.prev_week_low,
        prev_week_close=levels.prev_week_close,
    )

    return IndicatorResult(
        as_of=as_of,
        metrics=metrics,
        quality_flags=tuple(flags),
        mtf_rsi_values=mtf_result.rsi_by_timeframe if mtf_result is not None else {},
    )
