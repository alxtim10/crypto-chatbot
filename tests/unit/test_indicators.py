from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import numpy as np
import pytest

from crypto_chatbot.analysis.indicators import (
    MIN_HISTORY_EMA_200,
    adx_dmi_series,
    atr_percentile_series,
    atr_series,
    compute_mtf_rsi_alignment,
    compute_previous_period_levels,
    compute_technical_metrics,
    ema_series,
    macd_series,
    relative_volume_series,
    rma_series,
    rsi_series,
    simple_return_series,
    validate_closed_candles,
)
from crypto_chatbot.schemas.analysis import MtfRsiAlignment
from crypto_chatbot.schemas.market import (
    MarketCandle,
    MarketType,
    QualityFlag,
    Timeframe,
)

WIB = timezone(timedelta(hours=7))


def _make_candle(
    open_time: datetime,
    *,
    timeframe: Timeframe = Timeframe.H1,
    symbol: str = "BTC/USDT",
    open_p: float = 100.0,
    high_p: float = 105.0,
    low_p: float = 95.0,
    close_p: float = 102.0,
    volume: float = 10.0,
    is_closed: bool = True,
    available_lag: timedelta = timedelta(seconds=5),
) -> MarketCandle:
    close_time = open_time + timeframe.duration
    received = close_time + timedelta(seconds=1) if is_closed else open_time + timedelta(minutes=1)
    return MarketCandle(
        provider="ccxt",
        venue="binance",
        canonical_symbol=symbol,
        provider_symbol="BINANCE:BTCUSDT",
        market_type=MarketType.SPOT,
        timeframe=timeframe,
        open_time=open_time,
        close_time=close_time,
        received_time=received,
        available_time=close_time + available_lag,
        open=open_p,
        high=high_p,
        low=low_p,
        close=close_p,
        base_volume=volume,
        base_unit="BTC",
        quote_currency="USDT",
        is_closed=is_closed,
    )


def _make_candle_series(
    count: int,
    *,
    start_open: datetime | None = None,
    timeframe: Timeframe = Timeframe.H1,
    closes: list[float] | None = None,
    spread: float = 2.0,
    volume: float = 100.0,
) -> list[MarketCandle]:
    t0 = start_open or datetime(2026, 9, 1, 0, 0, tzinfo=UTC)
    candles: list[MarketCandle] = []
    for i in range(count):
        close_p = closes[i] if closes is not None else 100.0 + i * 0.5 + (i % 5) * 0.2
        open_p = closes[i - 1] if (closes is not None and i > 0) else close_p
        high_p = max(open_p, close_p) + spread
        low_p = max(1.0, min(open_p, close_p) - spread)
        candles.append(
            _make_candle(
                t0 + i * timeframe.duration,
                timeframe=timeframe,
                open_p=open_p,
                high_p=high_p,
                low_p=low_p,
                close_p=close_p,
                volume=volume + (i % 7) * 5.0 if volume > 0.0 else 0.0,
            )
        )
    return candles


def test_ema_and_rma_golden_sma_seed() -> None:
    vals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    ema3 = ema_series(vals, 3)
    assert np.all(np.isnan(ema3[:2]))
    assert ema3[2] == pytest.approx(2.0)  # SMA(1, 2, 3)
    assert ema3[3] == pytest.approx(3.0)  # 0.5*4 + 0.5*2
    assert ema3[4] == pytest.approx(4.0)  # 0.5*5 + 0.5*3

    rma3 = rma_series(vals, 3)
    assert np.all(np.isnan(rma3[:2]))
    assert rma3[2] == pytest.approx(2.0)
    expected_3 = (1.0 / 3.0) * 4.0 + (2.0 / 3.0) * 2.0
    expected_4 = (1.0 / 3.0) * 5.0 + (2.0 / 3.0) * expected_3
    assert rma3[3] == pytest.approx(expected_3)
    assert rma3[4] == pytest.approx(expected_4)


def test_simple_returns_and_missing_values_propagate_honestly() -> None:
    closes = np.array([100.0, 110.0, 121.0, np.nan, 130.0])
    r1 = simple_return_series(closes, 1)
    assert np.isnan(r1[0])
    assert r1[1] == pytest.approx(0.10)
    assert r1[2] == pytest.approx(0.10)
    assert np.isnan(r1[3])
    assert np.isnan(r1[4])

    vals_with_gap = np.array([1.0, 2.0, 3.0, np.nan, 5.0, 6.0])
    ema3 = ema_series(vals_with_gap, 3)
    assert ema3[2] == pytest.approx(2.0)
    assert np.isnan(ema3[3])
    assert np.isnan(ema3[4])
    assert np.isnan(ema3[5])


def test_rsi_edge_cases_up_down_flat_and_mixed() -> None:
    up_only = np.arange(100.0, 160.0)
    rsi_up, flat_up = rsi_series(up_only, 14)
    assert rsi_up[-1] == pytest.approx(100.0)
    assert not bool(flat_up[-1])

    down_only = np.arange(160.0, 100.0, -1.0)
    rsi_down, flat_down = rsi_series(down_only, 14)
    assert rsi_down[-1] == pytest.approx(0.0)
    assert not bool(flat_down[-1])

    flat = np.full(60, 100.0)
    rsi_flat, is_flat = rsi_series(flat, 14)
    assert rsi_flat[-1] == pytest.approx(100.0)
    assert bool(is_flat[-1])

    # Golden hand calculation for period=2: [10, 12, 11, 13]
    # gains=[nan, 2, 0, 2], losses=[nan, 0, 1, 0]
    # seed idx=2: avg_gain=(2+0)/2=1.0, avg_loss=(0+1)/2=0.5 -> RS=2 -> RSI=200/3
    # idx=3: avg_gain=(2+1)/2=1.5, avg_loss=(0+0.5)/2=0.25 -> RS=6 -> RSI=100 - 100/7
    mixed = np.array([10.0, 12.0, 11.0, 13.0])
    rsi_m, flat_m = rsi_series(mixed, 2)
    assert np.all(np.isnan(rsi_m[:2]))
    assert rsi_m[2] == pytest.approx(200.0 / 3.0)
    assert rsi_m[3] == pytest.approx(100.0 - 100.0 / 7.0)
    assert not np.any(flat_m)


def test_macd_atr_adx_golden_and_flat_safety() -> None:
    closes = np.linspace(100.0, 150.0, 60)
    macd_line, signal_line, hist = macd_series(closes, 12, 26, 9)
    assert np.all(np.isnan(macd_line[:25]))
    assert np.all(np.isnan(signal_line[:33]))
    assert np.isfinite(signal_line[33])
    assert hist[-1] == pytest.approx(macd_line[-1] - signal_line[-1])

    # Linearly increasing closes -> EMA12 - EMA26 > 0
    assert macd_line[-1] > 0.0

    # Flat prices for ATR and ADX/DMI
    flat = np.full(60, 100.0)
    atr, atr_pct = atr_series(flat, flat, flat, 14)
    assert atr[-1] == pytest.approx(0.0)
    assert atr_pct[-1] == pytest.approx(0.0)

    adx, pdi, mdi = adx_dmi_series(flat, flat, flat, 14)
    assert np.all(np.isnan(adx[:27]))
    assert adx[-1] == pytest.approx(0.0)
    assert pdi[-1] == pytest.approx(0.0)
    assert mdi[-1] == pytest.approx(0.0)


def test_relative_volume_excludes_current_bar_and_handles_zero_volume() -> None:
    vols = np.array([10.0] * 20 + [30.0])
    rel, zero_mask = relative_volume_series(vols, 20)
    assert np.all(np.isnan(rel[:20]))
    assert rel[20] == pytest.approx(3.0)
    assert not bool(zero_mask[20])

    zero_prev = np.array([0.0] * 20 + [50.0])
    rel_z, zero_z = relative_volume_series(zero_prev, 20)
    assert np.isnan(rel_z[20])
    assert bool(zero_z[20])


def test_atr_percentile_ranks_strictly_and_flat_is_zero() -> None:
    flat_atr_pct = np.zeros(120)
    pct_flat = atr_percentile_series(flat_atr_pct, 100)
    assert pct_flat[-1] == pytest.approx(0.0)

    increasing = np.linspace(0.01, 1.0, 100)
    pct_inc = atr_percentile_series(increasing, 100)
    assert np.all(np.isnan(pct_inc[:99]))
    assert pct_inc[99] == pytest.approx(100.0)

    # Window [1..99, 50.5]: 50 elements strictly < 50.5 (values 1..50), denom = 99
    custom = np.concatenate([np.arange(1.0, 100.0), np.array([50.5])])
    pct_custom = atr_percentile_series(custom, 100)
    assert pct_custom[99] == pytest.approx(100.0 * 50.0 / 99.0)


def test_compute_technical_metrics_warmup_thresholds_and_flags() -> None:
    # N = 100: ema_50, ema_200, atr_percentile_100 are unavailable; SHORT_HISTORY set
    c100 = _make_candle_series(100)
    res100 = compute_technical_metrics(c100)
    assert res100.metrics.ema_20 is not None
    assert res100.metrics.ema_50 is None
    assert res100.metrics.ema_200 is None
    assert res100.metrics.atr_percentile_100 is None
    assert res100.metrics.macd_line is not None
    assert res100.metrics.adx_14 is not None
    assert QualityFlag.SHORT_HISTORY in res100.quality_flags

    # N = 200: ema_50 & atr_percentile_100 populated, ema_200 still None
    c200 = _make_candle_series(200)
    res200 = compute_technical_metrics(c200)
    assert res200.metrics.ema_50 is not None
    assert res200.metrics.atr_percentile_100 is not None
    assert res200.metrics.ema_200 is None
    assert QualityFlag.SHORT_HISTORY in res200.quality_flags

    # N = 600: all single-timeframe indicators populated, no SHORT_HISTORY
    c600 = _make_candle_series(MIN_HISTORY_EMA_200)
    res600 = compute_technical_metrics(c600)
    assert res600.metrics.ema_200 is not None
    assert res600.quality_flags == ()

    # Flat price + zero volume sets both FLAT_PRICE and ZERO_VOLUME flags
    flat_candles = _make_candle_series(
        120,
        closes=[100.0] * 120,
        spread=0.0,
        volume=0.0,
    )
    res_flat = compute_technical_metrics(flat_candles)
    assert res_flat.metrics.rsi_14 == pytest.approx(100.0)
    assert res_flat.metrics.adx_14 == pytest.approx(0.0)
    assert res_flat.metrics.atr_percentile_100 == pytest.approx(0.0)
    assert res_flat.metrics.relative_volume_20 is None
    assert QualityFlag.FLAT_PRICE in res_flat.quality_flags
    assert QualityFlag.ZERO_VOLUME in res_flat.quality_flags


def test_previous_period_levels_complete_vs_partial() -> None:
    # Start Monday 2026-10-26 00:00 UTC and produce 8 days of 4h candles (48 bars) + 1 bar Monday
    start = datetime(2026, 10, 26, 0, 0, tzinfo=UTC)
    candles: list[MarketCandle] = []
    for i in range(49):
        open_t = start + i * timedelta(hours=4)
        # Distinct high/low/close per day & week while keeping low <= open, close <= high
        day_idx = i // 6
        high_p = 220.0 - day_idx
        low_p = 60.0 + day_idx
        close_p = 100.0 + i
        candles.append(
            _make_candle(
                open_t,
                timeframe=Timeframe.H4,
                open_p=100.0,
                high_p=high_p,
                low_p=low_p,
                close_p=close_p,
            )
        )

    as_of = candles[-1].close_time
    nov2_04 = datetime(2026, 11, 2, 4, 0, tzinfo=UTC)
    levels = compute_previous_period_levels(candles, as_of=nov2_04)

    # Previous UTC day = 2026-11-01 (day_idx=6, bars 36..41)
    assert levels.prev_day_high == pytest.approx(214.0)
    assert levels.prev_day_low == pytest.approx(66.0)
    assert levels.prev_day_close == pytest.approx(141.0)

    # Previous ISO week = 2026-10-26..2026-11-02 (bars 0..41)
    assert levels.prev_week_high == pytest.approx(220.0)
    assert levels.prev_week_low == pytest.approx(60.0)
    assert levels.prev_week_close == pytest.approx(141.0)
    assert as_of > nov2_04

    # Partial day (missing one candle) -> prev_day_* and prev_week_* become None
    with_gap = [c for c in candles if c.open_time != datetime(2026, 11, 1, 8, 0, tzinfo=UTC)]
    partial_levels = compute_previous_period_levels(with_gap, as_of=nov2_04)
    assert partial_levels.prev_day_high is None
    assert partial_levels.prev_week_high is None


def test_mtf_rsi_alignment_respects_as_of_without_lookahead() -> None:
    # 1h as_of = 2026-11-01 05:00 UTC; 4h candle 04:00..08:00 must NEVER be used.
    as_of = datetime(2026, 11, 1, 5, 0, tzinfo=UTC)

    h1_start = as_of - timedelta(hours=60)
    h1_closes = [100.0 + i for i in range(60)]
    h1 = _make_candle_series(60, start_open=h1_start, timeframe=Timeframe.H1, closes=h1_closes)

    # 4h candles ending at 04:00 (50 bars) PLUS an extra closed 04:00..08:00 bar with huge crash
    h4_closed_end = datetime(2026, 11, 1, 4, 0, tzinfo=UTC)
    h4_start = h4_closed_end - timedelta(hours=4 * 50)
    h4_closes = [100.0 + i for i in range(50)] + [10.0]
    h4 = _make_candle_series(51, start_open=h4_start, timeframe=Timeframe.H4, closes=h4_closes)

    d1_closed_end = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    d1_start = d1_closed_end - timedelta(days=50)
    d1_closes = [100.0 + i for i in range(50)]
    d1 = _make_candle_series(50, start_open=d1_start, timeframe=Timeframe.D1, closes=d1_closes)

    res = compute_mtf_rsi_alignment(
        {Timeframe.H1: h1, Timeframe.H4: h4, Timeframe.D1: d1},
        as_of=as_of,
    )
    # All three valid series rise monotonically (RSI=100) -> ALIGNED_BULLISH despite crash at 08:00
    assert res.alignment is MtfRsiAlignment.ALIGNED_BULLISH
    assert res.rsi_by_timeframe[Timeframe.H4] == pytest.approx(100.0)

    # Compare without the future 04:00..08:00 4h bar -> bit-identical result
    res_no_future = compute_mtf_rsi_alignment(
        {Timeframe.H1: h1, Timeframe.H4: h4[:-1], Timeframe.D1: d1},
        as_of=as_of,
    )
    assert res == res_no_future

    # Bearish alignment
    h1_down = _make_candle_series(
        60,
        start_open=h1_start,
        timeframe=Timeframe.H1,
        closes=[200.0 - i for i in range(60)],
    )
    h4_down = _make_candle_series(
        50,
        start_open=h4_start,
        timeframe=Timeframe.H4,
        closes=[200.0 - i for i in range(50)],
    )
    d1_down = _make_candle_series(
        50,
        start_open=d1_start,
        timeframe=Timeframe.D1,
        closes=[200.0 - i for i in range(50)],
    )
    res_bear = compute_mtf_rsi_alignment(
        {Timeframe.H1: h1_down, Timeframe.H4: h4_down, Timeframe.D1: d1_down},
        as_of=as_of,
    )
    assert res_bear.alignment is MtfRsiAlignment.ALIGNED_BEARISH

    # Mixed alignment
    res_mixed = compute_mtf_rsi_alignment(
        {Timeframe.H1: h1, Timeframe.H4: h4_down, Timeframe.D1: d1},
        as_of=as_of,
    )
    assert res_mixed.alignment is MtfRsiAlignment.MIXED

    # Short HTF history -> None
    res_short = compute_mtf_rsi_alignment(
        {Timeframe.H1: h1, Timeframe.H4: h4[:30], Timeframe.D1: d1},
        as_of=as_of,
    )
    assert res_short.alignment is None
    assert res_short.rsi_by_timeframe[Timeframe.H4] is None

    # Available time check excludes candles whose available_time > as_of (even if close_time <=)
    h1_lagged = _make_candle_series(
        60,
        start_open=h1_start,
        timeframe=Timeframe.H1,
        closes=h1_closes,
    )
    # Replace last 20 candles with large availability lag (> as_of), leaving < 43 available
    lagged = [
        _make_candle(
            c.open_time,
            timeframe=c.timeframe,
            open_p=c.open,
            high_p=c.high,
            low_p=c.low,
            close_p=c.close,
            available_lag=timedelta(hours=100) if idx >= 40 else timedelta(seconds=5),
        )
        for idx, c in enumerate(h1_lagged)
    ]
    res_avail = compute_mtf_rsi_alignment(
        {Timeframe.H1: lagged, Timeframe.H4: h4, Timeframe.D1: d1},
        as_of=as_of,
        use_available_time=True,
    )
    assert res_avail.rsi_by_timeframe[Timeframe.H1] is None
    assert res_avail.alignment is None


def test_no_future_leakage_prefix_truncation() -> None:
    candles = _make_candle_series(650, timeframe=Timeframe.H4)
    highs = np.array([c.high for c in candles])
    lows = np.array([c.low for c in candles])
    closes = np.array([c.close for c in candles])
    vols = np.array([c.base_volume for c in candles])

    full_r20 = simple_return_series(closes, 20)
    full_ema200 = ema_series(closes, 200)
    full_rsi, _ = rsi_series(closes, 14)
    full_macd, full_sig, full_hist = macd_series(closes)
    full_atr, full_atr_pct = atr_series(highs, lows, closes, 14)
    full_adx, full_pdi, full_mdi = adx_dmi_series(highs, lows, closes, 14)
    full_rvol, _ = relative_volume_series(vols, 20)
    full_atrp = atr_percentile_series(full_atr_pct, 100)

    for cut in (160, 300, 500, 620):
        trunc_r20 = simple_return_series(closes[:cut], 20)
        trunc_ema200 = ema_series(closes[:cut], 200)
        trunc_rsi, _ = rsi_series(closes[:cut], 14)
        trunc_macd, trunc_sig, trunc_hist = macd_series(closes[:cut])
        trunc_atr, trunc_atr_pct = atr_series(highs[:cut], lows[:cut], closes[:cut], 14)
        trunc_adx, trunc_pdi, trunc_mdi = adx_dmi_series(highs[:cut], lows[:cut], closes[:cut], 14)
        trunc_rvol, _ = relative_volume_series(vols[:cut], 20)
        trunc_atrp = atr_percentile_series(trunc_atr_pct, 100)

        np.testing.assert_array_equal(trunc_r20, full_r20[:cut])
        np.testing.assert_array_equal(trunc_ema200, full_ema200[:cut])
        np.testing.assert_array_equal(trunc_rsi, full_rsi[:cut])
        np.testing.assert_array_equal(trunc_macd, full_macd[:cut])
        np.testing.assert_array_equal(trunc_sig, full_sig[:cut])
        np.testing.assert_array_equal(trunc_hist, full_hist[:cut])
        np.testing.assert_array_equal(trunc_atr, full_atr[:cut])
        np.testing.assert_array_equal(trunc_adx, full_adx[:cut])
        np.testing.assert_array_equal(trunc_pdi, full_pdi[:cut])
        np.testing.assert_array_equal(trunc_mdi, full_mdi[:cut])
        np.testing.assert_array_equal(trunc_rvol, full_rvol[:cut])
        np.testing.assert_array_equal(trunc_atrp, full_atrp[:cut])


def test_validate_closed_candles_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="empty"):
        validate_closed_candles([])

    t0 = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    open_c = _make_candle(t0, is_closed=False)
    with pytest.raises(ValueError, match="open candle"):
        validate_closed_candles([open_c])

    c1 = _make_candle(t0, symbol="BTC/USDT")
    c2 = _make_candle(t0 + timedelta(hours=1), symbol="ETH/USDT")
    with pytest.raises(ValueError, match="canonical_symbol"):
        validate_closed_candles([c1, c2])

    c_tf = _make_candle(t0 + timedelta(hours=1), timeframe=Timeframe.H4)
    with pytest.raises(ValueError, match="timeframe"):
        validate_closed_candles([c1, c_tf])

    c_next = _make_candle(t0 + timedelta(hours=1))
    with pytest.raises(ValueError, match="ascending"):
        validate_closed_candles([c_next, c1])
