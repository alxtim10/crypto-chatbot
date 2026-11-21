from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from crypto_chatbot.analysis.failed_breakout import detect_failed_breakouts
from crypto_chatbot.analysis.regimes import classify_regime
from crypto_chatbot.analysis.structure import analyze_market_structure, detect_confirmed_swings
from crypto_chatbot.schemas.analysis import (
    MarketStructure,
    Regime,
    StructureEventType,
    TechnicalMetrics,
)
from crypto_chatbot.schemas.market import MarketCandle, MarketType, Timeframe


def _candle_bar(
    index: int,
    *,
    high: float,
    low: float,
    close: float,
    open_p: float | None = None,
    timeframe: Timeframe = Timeframe.H1,
) -> MarketCandle:
    start = datetime(2026, 11, 1, 0, 0, tzinfo=UTC) + index * timeframe.duration
    close_t = start + timeframe.duration
    op = close if open_p is None else open_p
    return MarketCandle(
        provider="ccxt",
        venue="binance",
        canonical_symbol="BTC/USDT",
        provider_symbol="BINANCE:BTCUSDT",
        market_type=MarketType.SPOT,
        timeframe=timeframe,
        open_time=start,
        close_time=close_t,
        received_time=close_t + timedelta(seconds=1),
        available_time=close_t + timedelta(seconds=1),
        open=op,
        high=max(high, op, close),
        low=min(low, op, close),
        close=close,
        base_volume=100.0,
        base_unit="BTC",
        quote_currency="USDT",
        is_closed=True,
    )


def _bars_from_ohlc(specs: list[tuple[float, float, float]]) -> list[MarketCandle]:
    return [_candle_bar(idx, high=h, low=low_v, close=c) for idx, (h, low_v, c) in enumerate(specs)]


@pytest.mark.parametrize(
    ("metrics", "expected"),
    [
        (
            TechnicalMetrics(
                last_close=100.0,
                ema_50=95.0,
                adx_14=30.0,
                plus_di_14=28.0,
                minus_di_14=12.0,
            ),
            None,
        ),
        (
            TechnicalMetrics(
                last_close=110.0,
                ema_50=100.0,
                adx_14=35.0,
                plus_di_14=30.0,
                minus_di_14=10.0,
                atr_percentile_100=80.0,
            ),
            Regime.HIGH_VOLATILITY,
        ),
        (
            TechnicalMetrics(
                last_close=110.0,
                ema_50=100.0,
                adx_14=25.0,
                plus_di_14=26.0,
                minus_di_14=14.0,
                atr_percentile_100=79.9,
            ),
            Regime.TRENDING_UP,
        ),
        (
            TechnicalMetrics(
                last_close=90.0,
                ema_50=100.0,
                adx_14=25.0,
                plus_di_14=14.0,
                minus_di_14=26.0,
                atr_percentile_100=50.0,
            ),
            Regime.TRENDING_DOWN,
        ),
        (
            TechnicalMetrics(
                last_close=100.0,
                ema_50=100.0,
                adx_14=19.9,
                plus_di_14=15.0,
                minus_di_14=16.0,
                atr_percentile_100=40.0,
            ),
            Regime.RANGING,
        ),
        (
            TechnicalMetrics(
                last_close=110.0,
                ema_50=100.0,
                adx_14=20.0,
                plus_di_14=25.0,
                minus_di_14=15.0,
                atr_percentile_100=40.0,
            ),
            Regime.TRANSITIONAL,
        ),
        (
            # ADX >= 25 but price below EMA50 while +DI > -DI -> transitional
            TechnicalMetrics(
                last_close=98.0,
                ema_50=100.0,
                adx_14=30.0,
                plus_di_14=28.0,
                minus_di_14=12.0,
                atr_percentile_100=40.0,
            ),
            Regime.TRANSITIONAL,
        ),
    ],
)
def test_classify_regime_precedence_and_boundaries(
    metrics: TechnicalMetrics,
    expected: Regime | None,
) -> None:
    assert classify_regime(metrics) is expected


def test_confirmed_swings_require_k_bars_and_reject_equal_peaks() -> None:
    # Bar 3 is a strict high at 110; confirmed only at bar 6 (3 + 3)
    specs = [
        (101.0, 99.0, 100.0),
        (103.0, 99.0, 102.0),
        (106.0, 99.0, 105.0),
        (110.0, 99.0, 108.0),
        (107.0, 99.0, 104.0),
        (105.0, 99.0, 102.0),
        (104.0, 99.0, 101.0),
        # Equal peaks at bars 9 and 10 (115.0) should NOT become strict swing highs
        (105.0, 99.0, 101.0),
        (106.0, 99.0, 102.0),
        (115.0, 99.0, 112.0),
        (115.0, 99.0, 112.0),
        (107.0, 99.0, 104.0),
        (106.0, 99.0, 103.0),
        (105.0, 99.0, 102.0),
        # Trailing peak at bar 15 only has 2 bars after it (bars 16, 17) -> NOT confirmed yet
        (105.0, 99.0, 102.0),
        (130.0, 99.0, 125.0),
        (108.0, 99.0, 104.0),
        (107.0, 99.0, 103.0),
    ]
    candles = _bars_from_ohlc(specs)

    swings = detect_confirmed_swings(candles, k=3)
    highs = [e for e in swings if e.event_type is StructureEventType.SWING_HIGH]

    assert len(highs) == 1
    assert highs[0].level == 110.0
    assert highs[0].event_time == candles[3].close_time
    assert highs[0].confirmation_time == candles[6].close_time

    # Truncating at bar 5 (only 2 right bars after index 3) must NOT return the pivot
    assert [
        e
        for e in detect_confirmed_swings(candles[:6], k=3)
        if e.event_type is StructureEventType.SWING_HIGH
    ] == []


def test_market_structure_bullish_bearish_and_mixed() -> None:
    # Build clear HH + HL sequence (pivots at 3, 8, 13, 18 with padding)
    bull_specs = [
        (102.0, 98.0, 100.0),
        (103.0, 98.0, 101.0),
        (104.0, 98.0, 102.0),
        (110.0, 98.0, 108.0),  # SH1 = 110
        (105.0, 96.0, 100.0),
        (104.0, 94.0, 96.0),
        (103.0, 90.0, 92.0),  # SL1 = 90
        (104.0, 93.0, 97.0),
        (105.0, 95.0, 101.0),
        (106.0, 96.0, 104.0),
        (115.0, 97.0, 113.0),  # SH2 = 115 (HH)
        (108.0, 97.0, 103.0),
        (107.0, 96.0, 99.0),
        (106.0, 95.0, 97.0),  # SL2 = 95 (HL)
        (107.0, 96.0, 100.0),
        (108.0, 97.0, 102.0),
        (109.0, 98.0, 105.0),
    ]
    bull_candles = _bars_from_ohlc(bull_specs)
    bull_analysis = analyze_market_structure(bull_candles, k=3)
    assert bull_analysis.market_structure is MarketStructure.BULLISH

    # Invert high/low symmetrically for LH + LL bearish structure
    bear_specs = [(200.0 - lv, 200.0 - hv, 200.0 - cv) for (hv, lv, cv) in bull_specs]
    bear_candles = _bars_from_ohlc(bear_specs)
    bear_analysis = analyze_market_structure(bear_candles, k=3)
    assert bear_analysis.market_structure is MarketStructure.BEARISH

    # Too few pivots -> MIXED
    short_analysis = analyze_market_structure(bull_candles[:10], k=3)
    assert short_analysis.market_structure is MarketStructure.MIXED


def test_failed_breakout_window_timing_and_symmetry() -> None:
    # Base sequence with SH at index 3 (level=110.0), confirmed at index 6 (close=102)
    base = [
        (102.0, 99.0, 100.0),
        (104.0, 99.0, 102.0),
        (106.0, 99.0, 105.0),
        (110.0, 99.0, 108.0),
        (106.0, 99.0, 103.0),
        (105.0, 99.0, 102.0),
        (104.0, 99.0, 102.0),
    ]

    # Case 1: Re-entry on bar 1 (break at 7, fail at 8)
    c_1bar = _bars_from_ohlc([*base, (112.0, 101.0, 111.0), (112.0, 107.0, 109.0)])
    ev_1 = detect_failed_breakouts(c_1bar, pivot_k=3, max_reentry_bars=3)
    assert len(ev_1) == 1
    assert ev_1[0].event_type is StructureEventType.FAILED_BREAKOUT_BULLISH
    assert ev_1[0].level == 110.0
    assert ev_1[0].event_time == c_1bar[8].close_time
    assert ev_1[0].confirmation_time == c_1bar[8].close_time

    # Case 2: Re-entry on bar 3 (break at 7, stay above at 8 & 9, fail at 10)
    c_3bar = _bars_from_ohlc(
        [
            *base,
            (112.0, 101.0, 111.0),
            (113.0, 109.0, 112.0),
            (114.0, 110.5, 111.5),
            (112.0, 107.0, 109.5),
        ]
    )
    ev_3 = detect_failed_breakouts(c_3bar, pivot_k=3, max_reentry_bars=3)
    assert len(ev_3) == 1
    assert ev_3[0].event_time == c_3bar[10].close_time

    # Case 3: Re-entry on bar 4 (expired after 3 bars) -> no failed breakout
    c_4bar = _bars_from_ohlc(
        [
            *base,
            (112.0, 101.0, 111.0),
            (113.0, 109.0, 112.0),
            (114.0, 110.5, 111.5),
            (113.0, 110.5, 111.0),
            (112.0, 107.0, 109.0),
        ]
    )
    assert detect_failed_breakouts(c_4bar, pivot_k=3, max_reentry_bars=3) == ()

    # Case 4: Bearish failure (breakdown below SL=90.0 then close back above within 2 bars)
    bear_base = [(100.0, 98.0, 99.0), (100.0, 96.0, 97.0), (100.0, 93.0, 94.0), (100.0, 90.0, 92.0)]
    bear_base += [(100.0, 94.0, 96.0), (100.0, 95.0, 97.0), (100.0, 95.0, 96.0)]
    c_bear = _bars_from_ohlc(
        [*bear_base, (97.0, 88.0, 89.0), (96.0, 87.5, 88.5), (96.0, 88.0, 91.0)]
    )
    ev_bear = detect_failed_breakouts(c_bear, pivot_k=3, max_reentry_bars=3)
    assert len(ev_bear) == 1
    assert ev_bear[0].event_type is StructureEventType.FAILED_BREAKOUT_BEARISH
    assert ev_bear[0].level == 90.0
    assert ev_bear[0].event_time == c_bear[9].close_time


def test_structure_and_failed_breakouts_are_strictly_causal() -> None:
    specs = [
        (102.0, 99.0, 100.0),
        (104.0, 99.0, 102.0),
        (106.0, 99.0, 105.0),
        (110.0, 99.0, 108.0),
        (106.0, 99.0, 103.0),
        (105.0, 99.0, 102.0),
        (104.0, 99.0, 102.0),
        (112.0, 101.0, 111.0),
        (112.0, 107.0, 109.0),
        (114.0, 106.0, 113.0),
        (118.0, 106.0, 116.0),
        (115.0, 106.0, 110.0),
        (114.0, 106.0, 109.0),
        (113.0, 106.0, 108.0),
    ]
    candles = _bars_from_ohlc(specs)
    full_swings = detect_confirmed_swings(candles, k=3)
    full_failed = detect_failed_breakouts(candles, pivot_k=3, max_reentry_bars=3)

    for cut in (7, 9, 12):
        cutoff = candles[cut - 1].close_time
        expected_swings = tuple(e for e in full_swings if e.confirmation_time <= cutoff)
        expected_failed = tuple(e for e in full_failed if e.confirmation_time <= cutoff)

        assert detect_confirmed_swings(candles[:cut], k=3) == expected_swings
        assert detect_failed_breakouts(candles[:cut], pivot_k=3, max_reentry_bars=3) == (
            expected_failed
        )
