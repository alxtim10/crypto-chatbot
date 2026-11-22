from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from crypto_chatbot.analysis.indicators import (
    bollinger_bands_series,
    donchian_channel_series,
    rolling_vwap_series,
)
from crypto_chatbot.analysis.statistics import (
    MIN_HISTORY_HURST,
    annualize_volatility,
    compute_statistical_metrics,
    garman_klass_volatility,
    hurst_exponent,
    log_return_series,
    ou_half_life,
    parkinson_volatility,
    realized_volatility_cc,
)
from crypto_chatbot.schemas.market import MarketCandle, MarketType, Timeframe


def _candle(
    open_time: datetime,
    *,
    open_p: float,
    high_p: float,
    low_p: float,
    close_p: float,
    volume: float = 100.0,
) -> MarketCandle:
    close_time = open_time + Timeframe.H1.duration
    return MarketCandle(
        provider="ccxt",
        venue="binance",
        canonical_symbol="BTC/USDT",
        provider_symbol="BINANCE:BTCUSDT",
        market_type=MarketType.SPOT,
        timeframe=Timeframe.H1,
        open_time=open_time,
        close_time=close_time,
        received_time=close_time + timedelta(seconds=1),
        available_time=close_time + timedelta(seconds=5),
        open=open_p,
        high=high_p,
        low=low_p,
        close=close_p,
        base_volume=volume,
        base_unit="BTC",
        quote_currency="USDT",
        is_closed=True,
    )


def _series_from_closes(closes: list[float], *, spread: float = 2.0) -> list[MarketCandle]:
    t0 = datetime(2026, 9, 1, tzinfo=UTC)
    out: list[MarketCandle] = []
    for i, close_p in enumerate(closes):
        open_p = closes[i - 1] if i > 0 else close_p
        high_p = max(open_p, close_p) + spread
        low_p = max(1.0, min(open_p, close_p) - spread)
        out.append(
            _candle(
                t0 + i * Timeframe.H1.duration,
                open_p=open_p,
                high_p=high_p,
                low_p=low_p,
                close_p=close_p,
                volume=100.0 + i,
            )
        )
    return out


# --- realized volatility ---------------------------------------------------


def test_log_return_series_and_realized_vol() -> None:
    closes = np.array([100.0, 110.0, 99.0, 108.9])
    lr = log_return_series(closes)
    assert np.isnan(lr[0])
    assert lr[1] == pytest.approx(np.log(110.0 / 100.0))

    # Constant-ratio series has zero dispersion of log returns.
    geo = np.array([100.0 * (1.01**i) for i in range(30)])
    assert realized_volatility_cc(geo, window=20) == pytest.approx(0.0, abs=1e-9)


def test_realized_vol_insufficient_history_returns_none() -> None:
    closes = np.array([100.0, 101.0, 102.0])
    assert realized_volatility_cc(closes, window=20) is None


def test_annualize_volatility_scales_by_sqrt() -> None:
    assert annualize_volatility(0.02, bars_per_year=365.0) == pytest.approx(0.02 * np.sqrt(365.0))


def test_parkinson_and_garman_klass_positive_and_handle_invalid() -> None:
    rng = np.random.default_rng(3)
    base = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 40)))
    highs = base + 1.0
    lows = base - 1.0
    opens = base
    closes = base
    pk = parkinson_volatility(highs, lows, window=20)
    gk = garman_klass_volatility(opens, highs, lows, closes, window=20)
    assert pk is not None
    assert pk > 0.0
    assert gk is not None
    assert gk > 0.0

    # high < low in the window -> None (invalid OHLC, never silently repaired)
    bad_high = highs.copy()
    bad_high[-1] = lows[-1] - 5.0
    assert parkinson_volatility(bad_high, lows, window=20) is None


# --- Hurst & OU half-life --------------------------------------------------


def test_hurst_random_walk_near_half() -> None:
    rng = np.random.default_rng(11)
    rw = 100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 600)))
    h = hurst_exponent(rw)
    assert h is not None
    assert 0.35 <= h <= 0.65


def test_hurst_trending_series_above_half() -> None:
    # Persistent (trending) path: drift plus small noise so lagged diffs have positive variance.
    # Positively autocorrelated (momentum) increments produce a persistent path (H > 0.5).
    rng = np.random.default_rng(21)
    noise = rng.normal(0.0, 0.01, 400)
    increments = np.zeros_like(noise)
    increments[0] = noise[0]
    phi = 0.6
    for i in range(1, len(noise)):
        increments[i] = phi * increments[i - 1] + noise[i]
    trend = 100.0 * np.exp(np.cumsum(increments))
    h = hurst_exponent(trend)
    assert h is not None
    assert h >= 0.55


def test_hurst_deterministic_line_is_none() -> None:
    # A perfectly log-linear path has zero dispersion of lagged diffs -> undefined (None).
    line = np.array([100.0 * (1.002**i) for i in range(200)])
    assert hurst_exponent(line) is None


def test_hurst_requires_min_window() -> None:
    closes = np.linspace(100.0, 110.0, MIN_HISTORY_HURST - 1)
    assert hurst_exponent(closes, min_window=MIN_HISTORY_HURST) is None


def test_ou_half_life_none_for_log_linear_trend() -> None:
    # Constant-growth-rate path has an exact unit root in log space (beta ~ 0) -> None.
    trend = np.array([100.0 * (1.003**i) for i in range(150)])
    assert ou_half_life(trend, window=120) is None


def test_ou_half_life_finite_for_mean_reverting() -> None:
    x = 100.0 + np.sin(np.linspace(0.0, 25.0, 200)) * 4.0
    hl = ou_half_life(x, window=120)
    assert hl is not None
    assert hl > 0.0


def test_compute_statistical_metrics_populates_fields() -> None:
    rng = np.random.default_rng(5)
    closes = list(100.0 * np.exp(np.cumsum(rng.normal(0.0, 0.01, 130))))
    candles = _series_from_closes(closes)
    metrics = compute_statistical_metrics(candles)
    assert metrics.realized_vol_cc is not None
    assert metrics.realized_vol_cc > 0.0
    assert metrics.realized_vol_cc_annualized is not None
    assert metrics.parkinson_vol is not None
    assert metrics.garman_klass_vol is not None
    assert metrics.hurst_exponent is not None


# --- Bollinger / Donchian / VWAP ------------------------------------------


def test_bollinger_bands_golden_flat_window() -> None:
    closes = np.full(25, 50.0)
    mid, upper, lower, pct_b, bw = bollinger_bands_series(closes, 20, 2.0)
    assert mid[-1] == pytest.approx(50.0)
    assert upper[-1] == pytest.approx(50.0)
    assert lower[-1] == pytest.approx(50.0)
    assert pct_b[-1] == pytest.approx(0.5)  # degenerate band -> midpoint
    assert bw[-1] == pytest.approx(0.0)


def test_bollinger_percent_b_bounds() -> None:
    rng = np.random.default_rng(2)
    closes = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 60))
    _mid, upper, lower, pct_b, bw = bollinger_bands_series(closes, 20, 2.0)
    assert upper[-1] > lower[-1]
    assert np.isfinite(pct_b[-1])
    assert bw[-1] >= 0.0


def test_bollinger_no_lookahead() -> None:
    rng = np.random.default_rng(9)
    closes = 100.0 + np.cumsum(rng.normal(0.0, 1.0, 80))
    full = bollinger_bands_series(closes, 20, 2.0)[0]
    truncated = bollinger_bands_series(closes[:60], 20, 2.0)[0]
    np.testing.assert_allclose(full[:60], truncated, equal_nan=True)


def test_donchian_channel_min_max() -> None:
    highs = np.array([float(i) for i in range(1, 31)])
    lows = highs - 2.0
    upper, lower, mid = donchian_channel_series(highs, lows, 20)
    assert upper[-1] == pytest.approx(30.0)
    assert lower[-1] == pytest.approx(9.0)  # low at bar index 10 (value 11-2) over last 20
    assert mid[-1] == pytest.approx((30.0 + 9.0) / 2.0)


def test_vwap_zero_volume_window_unavailable() -> None:
    highs = np.full(25, 101.0)
    lows = np.full(25, 99.0)
    closes = np.full(25, 100.0)
    volumes = np.zeros(25)
    vwap, _upper, _lower, z = rolling_vwap_series(highs, lows, closes, volumes, 20, 2.0)
    assert np.isnan(vwap[-1])
    assert np.isnan(z[-1])


def test_vwap_equals_typical_price_when_constant() -> None:
    highs = np.full(25, 102.0)
    lows = np.full(25, 98.0)
    closes = np.full(25, 100.0)
    volumes = np.full(25, 10.0)
    vwap, _upper, _lower, z = rolling_vwap_series(highs, lows, closes, volumes, 20, 2.0)
    # typical price = (102+98+100)/3 = 100
    assert vwap[-1] == pytest.approx(100.0)
    assert z[-1] == pytest.approx(0.0)
