"""Unit tests for PRD v1.7 Wave 5 tasks (T-25 through T-32)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from crypto_chatbot.analysis.macro import (
    asof_join_macro_candles,
    build_macro_benchmark_context,
    correlation_critical_value,
)
from crypto_chatbot.analysis.momentum_screener import (
    MomentumScreenerReport,
    ScreenerUniverseSnapshot,
    capture_daily_universe_snapshot,
    render_momentum_screener_report,
)
from crypto_chatbot.analysis.setup import compute_leverage_ceiling
from crypto_chatbot.analysis.statistics import (
    adf_test_constant,
    hurst_exponent_with_se,
    ou_half_life_with_adf,
)
from crypto_chatbot.bot.access import AccessController
from crypto_chatbot.llm.guard import check_directional_consistency
from crypto_chatbot.schemas.analysis import SetupBias
from crypto_chatbot.schemas.market import MarketCandle, MarketType, Timeframe


def _make_candle(
    ts: datetime,
    *,
    close: float,
    symbol: str = "BTC/USDT",
    tf: Timeframe = Timeframe.D1,
) -> MarketCandle:
    return MarketCandle(
        canonical_symbol=symbol,
        provider="bingx",
        venue="bingx",
        provider_symbol=symbol,
        market_type=MarketType.SPOT,
        timeframe=tf,
        open_time=ts - tf.duration,
        close_time=ts,
        available_time=ts,
        received_time=ts,
        open=close,
        high=close * 1.01,
        low=close * 0.99,
        close=close,
        base_volume=100.0,
        base_unit="BTC",
        quote_currency="USDT",
        is_closed=True,
    )


# ---------------------------------------------------------------------------
# T-25: Leverage Ceiling with Maintenance Margin Rate (MMR)
# ---------------------------------------------------------------------------


def test_t25_mmr_leverage_reduces_to_v16_at_zero_and_is_monotonic() -> None:
    # MMR=0 reproduces v1.6 exactly
    for sl_pct in (0.01, 0.02, 0.035, 0.05, 0.10):
        for cost in (0.001, 0.003, 0.005):
            v16 = min(20, max(1, int(0.75 / (sl_pct + cost))))
            lev_0, assumed_0 = compute_leverage_ceiling(sl_pct, cost, maintenance_margin_rate=0.0)
            lev_1pct, assumed_1pct = compute_leverage_ceiling(
                sl_pct, cost, maintenance_margin_rate=0.01
            )
            lev_def, assumed_def = compute_leverage_ceiling(sl_pct, cost)
            assert lev_0 == v16
            assert assumed_0 is False
            assert assumed_1pct is False
            assert assumed_def is True
            assert lev_1pct <= lev_0
            assert lev_def == lev_1pct


# ---------------------------------------------------------------------------
# T-26: Macro As-Of Join Across Weekend & r_crit Significance
# ---------------------------------------------------------------------------


def test_t26_macro_asof_join_across_weekend_and_rcrit_neutral() -> None:
    # Friday DXY close available at 2026-01-02 21:00 UTC; crypto 1d closes Saturday & Sunday
    fri_dxy = datetime(2026, 1, 2, 21, 0, tzinfo=UTC)
    sat_btc = datetime(2026, 1, 4, 0, 0, tzinfo=UTC)
    sun_btc = datetime(2026, 1, 5, 0, 0, tzinfo=UTC)
    dxy_candles = [_make_candle(fri_dxy, close=105.0, symbol="DXY/USD", tf=Timeframe.D1)]
    btc_candles = [
        _make_candle(sat_btc, close=95000.0, tf=Timeframe.D1),
        _make_candle(sun_btc, close=96000.0, tf=Timeframe.D1),
    ]
    joined, stale = asof_join_macro_candles(btc_candles, dxy_candles)
    assert len(joined) == 2
    assert stale == 0

    # r_crit at n=20 is ~0.444 -> |r|=0.30 must yield DXY impact = neutral
    r_crit_20 = correlation_critical_value(20)
    assert 0.43 <= r_crit_20 <= 0.46
    r_crit_60 = correlation_critical_value(60)
    assert 0.24 <= r_crit_60 <= 0.27

    # Construct 60 daily candles with weak correlation and bullish DXY trend
    base = datetime(2026, 1, 1, tzinfo=UTC)
    rng = np.random.default_rng(7)
    dxy_series = [100.0 + 0.2 * i for i in range(60)]
    btc_series = [50000.0 * (1.0 + 0.005 * float(rng.normal())) for _ in range(60)]
    d_c = [
        _make_candle(base + timedelta(days=i), close=dxy_series[i], symbol="DXY/USD")
        for i in range(60)
    ]
    b_c = [_make_candle(base + timedelta(days=i), close=btc_series[i]) for i in range(60)]
    ctx = build_macro_benchmark_context(
        target_candles=b_c,
        target_canonical_symbol="BTC/USDT",
        as_of=base + timedelta(days=60),
        dxy_candles=d_c,
    )
    assert ctx.dxy is not None
    if ctx.dxy.rolling_corr_20 is not None and abs(ctx.dxy.rolling_corr_20) < r_crit_20:
        assert ctx.dxy.macro_impact == "neutral"


# ---------------------------------------------------------------------------
# T-27: ADF Stationarity Gate for OU Half-Life & Hurst SE
# ---------------------------------------------------------------------------


def test_t27_adf_gate_rejects_random_walk_and_estimates_ar1_halflife() -> None:
    rng = np.random.default_rng(42)
    rw_rejected = 0
    for _ in range(200):
        steps = rng.normal(0.0, 0.01, size=120)
        prices = 100.0 * np.exp(np.cumsum(steps))
        hl, not_stat = ou_half_life_with_adf(prices, window=120, require_stationary=True)
        if hl is None and not_stat:
            rw_rejected += 1
    assert rw_rejected / 200.0 >= 0.90

    # AR(1) with phi = 0.8 -> theoretical half-life = -ln(2) / ln(0.8) ~= 3.106 bars
    phi = 0.8
    theo_hl = -math.log(2.0) / math.log(phi)
    within_30pct = 0
    for _ in range(100):
        y = np.zeros(240, dtype=np.float64)
        eps = rng.normal(0.0, 0.01, size=240)
        for t in range(1, 240):
            y[t] = phi * y[t - 1] + eps[t]
        prices = 100.0 * np.exp(y)
        hl, _ns = ou_half_life_with_adf(prices, window=240, require_stationary=True)
        if hl is not None and abs(hl - theo_hl) / theo_hl <= 0.30:
            within_30pct += 1
    assert within_30pct / 100.0 >= 0.80

    # Hurst SE & HURST_UNRELIABLE check
    h_val, se_h, unreliable = hurst_exponent_with_se(
        100.0 * np.exp(np.cumsum(rng.normal(0, 0.01, 120)))
    )
    assert h_val is not None
    assert se_h is not None
    assert se_h >= 0.0
    assert isinstance(unreliable, bool)
    t_stat, crit, _is_stat = adf_test_constant(np.zeros(120) + rng.normal(0, 0.01, 120))
    assert crit == pytest.approx(-2.8859, abs=0.01)
    assert math.isfinite(t_stat)


# ---------------------------------------------------------------------------
# T-30: Screener Daily Universe Snapshot & UNVALIDATED Badge
# ---------------------------------------------------------------------------


def test_t30_screener_daily_snapshot_idempotent_and_unvalidated_badge() -> None:
    now = datetime(2026, 2, 10, 14, 0, tzinfo=UTC)
    report = MomentumScreenerReport(
        venue="BINGX",
        side_filter="all",
        min_volume_usd=5_000_000.0,
        total_pairs_seen=50,
        liquid_pairs_count=20,
        top_gainers=(),
        top_losers=(),
        long_signals=(),
        short_signals=(),
        watching_items=(),
        failed_symbols=(),
        as_of=now,
        risk_management_note="note",
        disclaimer="disclaimer",
    )

    store: dict[tuple[str, str], ScreenerUniverseSnapshot] = {}
    snap1, created1 = capture_daily_universe_snapshot(report, store=store)
    snap2, created2 = capture_daily_universe_snapshot(report, store=store)
    assert created1 is True
    assert created2 is False
    assert snap1.snapshot_date_utc == "2026-02-10"
    assert snap2 == snap1

    rendered_unval = render_momentum_screener_report(report, html_mode=False)
    assert "[UNVALIDATED]" in rendered_unval
    rendered_val = render_momentum_screener_report(report, html_mode=False, is_validated=True)
    assert "[UNVALIDATED]" not in rendered_val


# ---------------------------------------------------------------------------
# T-31: Directional Consistency Guard
# ---------------------------------------------------------------------------


def test_t31_directional_consistency_guard() -> None:
    assert check_directional_consistency("Momentum terlihat bullish kuat.", SetupBias.SHORT) != ()
    assert check_directional_consistency("Peluang short terlihat jelas.", SetupBias.LONG) != ()
    assert (
        check_directional_consistency("Harga bergerak dekat area referensi.", SetupBias.SHORT) == ()
    )
    assert (
        check_directional_consistency("Struktur bearish mendukung skenario short.", SetupBias.SHORT)
        == ()
    )
    assert (
        check_directional_consistency("Struktur bullish mendukung skenario long.", SetupBias.LONG)
        == ()
    )


# ---------------------------------------------------------------------------
# T-32: Per-User Rate Limit (11th request in 1 min rejected)
# ---------------------------------------------------------------------------


def test_t32_rate_limit_rejects_11th_request_within_one_minute() -> None:
    now = 1000.0
    ctrl = AccessController(
        allowed_user_ids=(42,),
        rate_limit_per_minute=10,
        window_seconds=60.0,
        clock=lambda: now,
    )
    for _ in range(10):
        dec = ctrl.check_access(42)
        assert dec.allowed is True
    eleventh = ctrl.check_access(42)
    assert eleventh.allowed is False
    assert eleventh.reason == "rate_limited"
