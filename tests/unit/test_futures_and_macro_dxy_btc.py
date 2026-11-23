"""Unit tests for Perpetual Futures primary engine and Macro DXY / BTC benchmark (ADR 0013)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from crypto_chatbot.analysis.macro import (
    build_macro_benchmark_context,
    classify_positioning_regime,
    compute_rolling_beta,
    compute_rolling_correlation,
)
from crypto_chatbot.bot.templates import (
    render_deterministic_snapshot,
)
from crypto_chatbot.market_data.ccxt_provider import CcxtMarketDataProvider
from crypto_chatbot.market_data.macro_provider import (
    is_dxy_market_closed,
)
from crypto_chatbot.market_data.symbols import resolve_default_perp_symbol
from crypto_chatbot.schemas.analysis import (
    AnalysisSnapshot,
    Fact,
    InputRange,
    MarketStructure,
    Regime,
    SetupBias,
    SetupStatus,
    TechnicalMetrics,
    TradeSetup,
)
from crypto_chatbot.schemas.macro import (
    BtcBenchmarkMetrics,
    DxySnapshotMetrics,
    MacroBenchmarkContext,
    PositioningRegime,
)
from crypto_chatbot.schemas.market import (
    MarketCandle,
    MarketType,
    QualityStatus,
    Timeframe,
)


def _make_candle(
    idx: int,
    price: float,
    *,
    base_time: datetime,
    timeframe: Timeframe = Timeframe.H1,
    symbol: str = "BTC/USDT:USDT",
    provider_symbol: str = "BTCUSDT.P",
    market_type: MarketType = MarketType.PERPETUAL,
    venue: str = "binance",
) -> MarketCandle:
    open_time = base_time + idx * timeframe.duration
    close_time = open_time + timeframe.duration
    return MarketCandle(
        provider="ccxt",
        venue=venue,
        canonical_symbol=symbol,
        provider_symbol=provider_symbol,
        market_type=market_type,
        timeframe=timeframe,
        open_time=open_time,
        close_time=close_time,
        received_time=close_time,
        available_time=close_time,
        open=price - 1.0,
        high=price + 2.0,
        low=price - 2.0,
        close=price,
        base_volume=100.0,
        quote_volume=100.0 * price,
        taker_buy_base_volume=55.0,
        base_unit=symbol.split("/")[0],
        quote_currency="USDT",
        is_closed=True,
    )


def test_resolve_default_perp_symbol() -> None:
    mapping_btc = resolve_default_perp_symbol("BTC")
    assert mapping_btc.canonical_symbol == "BTC/USDT:USDT"
    assert mapping_btc.market_type == MarketType.PERPETUAL
    assert mapping_btc.venue == "binance"
    assert mapping_btc.provider_symbol == "BINANCE:BTCUSDT.P"

    mapping_eth = resolve_default_perp_symbol("eth/usdt")
    assert mapping_eth.canonical_symbol == "ETH/USDT:USDT"
    assert mapping_eth.market_type == MarketType.PERPETUAL
    assert mapping_eth.provider_symbol == "BINANCE:ETHUSDT.P"

    mapping_sol = resolve_default_perp_symbol("SOLUSDT")
    assert mapping_sol.canonical_symbol == "SOL/USDT:USDT"
    assert mapping_sol.market_type == MarketType.PERPETUAL
    assert mapping_sol.provider_symbol == "BINANCE:SOLUSDT.P"


def test_ccxt_provider_defaults_to_perpetual() -> None:
    provider = CcxtMarketDataProvider()
    assert provider._market_type == MarketType.PERPETUAL


def test_classify_positioning_regime() -> None:
    # 1. Crowded squeeze risk
    regime = classify_positioning_regime(
        price_ret_pct=-1.5,
        oi_change_pct=2.0,
        cvd=10.0,
        funding_rate_8h=0.0006,  # positive funding while price falls
    )
    assert regime == PositioningRegime.CROWDED_SQUEEZE_RISK

    regime_short_squeeze = classify_positioning_regime(
        price_ret_pct=1.5,
        oi_change_pct=2.0,
        cvd=-10.0,
        funding_rate_8h=-0.0006,  # negative funding while price rises
    )
    assert regime_short_squeeze == PositioningRegime.CROWDED_SQUEEZE_RISK

    # 2. Long Buildup: Price UP, OI UP, CVD BUYER
    regime_long = classify_positioning_regime(
        price_ret_pct=2.0,
        oi_change_pct=5.0,
        cvd=150.0,
        funding_rate_8h=0.0001,
    )
    assert regime_long == PositioningRegime.LONG_BUILDUP

    # 3. Short Buildup: Price DOWN, OI UP, CVD SELLER
    regime_short = classify_positioning_regime(
        price_ret_pct=-2.0,
        oi_change_pct=5.0,
        cvd=-150.0,
        funding_rate_8h=0.0001,
    )
    assert regime_short == PositioningRegime.SHORT_BUILDUP

    # 4. Short Covering: Price UP, OI DOWN, CVD BUYER
    regime_covering = classify_positioning_regime(
        price_ret_pct=1.5,
        oi_change_pct=-4.0,
        cvd=80.0,
        funding_rate_8h=0.0001,
    )
    assert regime_covering == PositioningRegime.SHORT_COVERING

    # 5. Long Liquidation Flush: Price DOWN, OI DOWN, CVD SELLER
    regime_flush = classify_positioning_regime(
        price_ret_pct=-3.0,
        oi_change_pct=-6.0,
        cvd=-120.0,
        funding_rate_8h=0.0001,
    )
    assert regime_flush == PositioningRegime.LONG_LIQUIDATION_FLUSH

    # 6. Neutral Positioning
    regime_neutral = classify_positioning_regime(
        price_ret_pct=None,
        oi_change_pct=None,
        cvd=None,
        funding_rate_8h=None,
    )
    assert regime_neutral == PositioningRegime.NEUTRAL_POSITIONING


def test_rolling_correlation_and_beta() -> None:
    # Insufficient length
    assert compute_rolling_correlation([1.0, 2.0], [1.0, 2.0]) is None
    assert compute_rolling_beta([1.0, 2.0], [1.0, 2.0]) is None

    # Perfect correlation
    a = [0.01, 0.02, -0.01, 0.03, -0.02]
    b = [0.02, 0.04, -0.02, 0.06, -0.04]  # b = 2 * a
    r = compute_rolling_correlation(a, b)
    assert r is not None
    assert math.isclose(r, 1.0, rel_tol=1e-3)

    # Beta: Cov(b, a) / Var(a) = 2.0
    beta = compute_rolling_beta(b, a)
    assert beta is not None
    assert math.isclose(beta, 2.0, rel_tol=1e-3)

    # Inverse correlation
    neg_b = [-x for x in a]
    r_neg = compute_rolling_correlation(a, neg_b)
    assert r_neg is not None
    assert math.isclose(r_neg, -1.0, rel_tol=1e-3)


def test_dxy_weekend_market_closed() -> None:
    # Saturday full day
    sat = datetime(2026, 11, 21, 14, 0, tzinfo=UTC)
    assert is_dxy_market_closed(sat) is True

    # Friday before 22:00 UTC
    fri_open = datetime(2026, 11, 20, 18, 0, tzinfo=UTC)
    assert is_dxy_market_closed(fri_open) is False

    # Friday after 22:00 UTC
    fri_closed = datetime(2026, 11, 20, 22, 30, tzinfo=UTC)
    assert is_dxy_market_closed(fri_closed) is True

    # Sunday before 22:00 UTC
    sun_closed = datetime(2026, 11, 22, 10, 0, tzinfo=UTC)
    assert is_dxy_market_closed(sun_closed) is True

    # Monday normal trading
    mon_open = datetime(2026, 11, 23, 10, 0, tzinfo=UTC)
    assert is_dxy_market_closed(mon_open) is False


def test_build_macro_benchmark_context_btc_and_alt() -> None:
    base_time = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    # 30 target candles
    target_candles = [
        _make_candle(i, 100.0 + i * 0.5, base_time=base_time, symbol="SOL/USDT:USDT")
        for i in range(30)
    ]
    btc_candles = [
        _make_candle(i, 50000.0 + i * 100.0, base_time=base_time, symbol="BTC/USDT:USDT")
        for i in range(30)
    ]
    as_of = target_candles[-1].close_time

    ctx = build_macro_benchmark_context(
        target_candles=target_candles,
        target_canonical_symbol="SOL/USDT:USDT",
        as_of=as_of,
        btc_candles=btc_candles,
    )
    assert ctx.positioning_regime is not None
    assert ctx.btc_benchmark is not None
    assert ctx.btc_benchmark.canonical_symbol == "BTC/USDT:USDT"
    assert ctx.btc_benchmark.relative_strength_20_pct is not None
    assert ctx.btc_benchmark.rolling_corr_20 is not None
    assert ctx.btc_benchmark.rolling_beta_20 is not None

    # Self benchmark for BTC target
    ctx_btc = build_macro_benchmark_context(
        target_candles=btc_candles,
        target_canonical_symbol="BTC/USDT:USDT",
        as_of=as_of,
        btc_candles=btc_candles,
    )
    assert ctx_btc.btc_benchmark is not None
    assert ctx_btc.btc_benchmark.leader_alignment == "self"


def test_trade_setup_sl_distance_and_max_leverage() -> None:
    # Build a valid READY setup directly and check SL distance % and leverage
    setup = TradeSetup(
        status=SetupStatus.READY,
        bias=SetupBias.LONG,
        entry_low=100.0,
        entry_high=102.0,
        entry_mid=101.0,
        stop_loss=96.0,  # risk = 5.0, distance = 5 / 101 ~= 4.95%
        stop_loss_distance_pct=4.9505,
        max_theoretical_leverage=16,  # int(0.8 / 0.049505) = 16x
        tp1=106.0,
        tp2=111.0,
        tp3=116.0,
        rr_tp1=1.0,
        rr_tp2=2.0,
        rr_tp3=3.0,
        risk_notes=("Catatan risiko",),
    )
    assert setup.stop_loss_distance_pct == 4.9505
    assert setup.max_theoretical_leverage == 16


def test_macro_context_temporal_lookahead_validation() -> None:
    now = datetime(2026, 11, 23, 10, 0, tzinfo=UTC)
    future = now + timedelta(hours=1)

    macro_future = MacroBenchmarkContext(
        as_of=future,
        positioning_regime=PositioningRegime.NEUTRAL_POSITIONING,
    )

    with pytest.raises(ValueError, match="lookahead"):
        AnalysisSnapshot(
            snapshot_id="snap-test1",
            provider="ccxt",
            venue="binance",
            canonical_symbol="BTC/USDT:USDT",
            provider_symbol="BTCUSDT.P",
            market_type=MarketType.PERPETUAL,
            timeframe=Timeframe.H1,
            as_of=now,
            calc_version="1.5.0",
            macro_context=macro_future,
            quality_status=QualityStatus.INSUFFICIENT_DATA,
        )


def test_templates_rendering_with_macro_and_perpetual() -> None:
    now = datetime(2026, 11, 23, 10, 0, tzinfo=UTC)
    dxy_metric = DxySnapshotMetrics(
        symbol="DXY",
        as_of=now,
        last_close=104.50,
        trend="bullish",
        rolling_corr_20=-0.65,
        macro_impact="headwind",
        is_weekend_or_closed=False,
    )
    btc_metric = BtcBenchmarkMetrics(
        canonical_symbol="BTC/USDT:USDT",
        as_of=now,
        last_close=92500.0,
        relative_strength_20_pct=3.5,
        rolling_corr_20=0.85,
        rolling_beta_20=1.25,
        leader_alignment="aligned",
    )
    macro_ctx = MacroBenchmarkContext(
        as_of=now,
        dxy=dxy_metric,
        btc_benchmark=btc_metric,
        positioning_regime=PositioningRegime.LONG_BUILDUP,
    )
    setup = TradeSetup(
        status=SetupStatus.READY,
        bias=SetupBias.LONG,
        entry_low=100.0,
        entry_high=102.0,
        entry_mid=101.0,
        stop_loss=96.0,
        stop_loss_distance_pct=4.95,
        max_theoretical_leverage=16,
        tp1=106.0,
        tp2=111.0,
        tp3=116.0,
        rr_tp1=1.0,
        rr_tp2=2.0,
        rr_tp3=3.0,
    )
    tech = TechnicalMetrics(
        last_close=101.0,
        rsi_14=55.0,
        ema_20=100.0,
        ema_50=98.0,
    )
    input_rng = InputRange(
        first_open_time=now - timedelta(days=5),
        last_open_time=now,
        candle_count=100,
        sha256="a" * 64,
    )
    fact1 = Fact(
        fact_id="F1",
        label="Harga penutupan terakhir",
        value=101.0,
        unit="USDT",
        display_format="price",
        source="binance",
        as_of=now,
    )

    snapshot = AnalysisSnapshot(
        snapshot_id="snap-test2",
        provider="ccxt",
        venue="binance",
        canonical_symbol="SOL/USDT:USDT",
        provider_symbol="BINANCE:SOLUSDT.P",
        market_type=MarketType.PERPETUAL,
        timeframe=Timeframe.H1,
        as_of=now,
        calc_version="1.5.0",
        input_range=input_rng,
        technical=tech,
        regime=Regime.TRENDING_UP,
        market_structure=MarketStructure.BULLISH,
        facts=(fact1,),
        macro_context=macro_ctx,
        trade_setup=setup,
        quality_status=QualityStatus.DEGRADED,
    )

    rendered = render_deterministic_snapshot(snapshot)
    assert "PERPETUAL" in rendered
    assert "REZIM POSISI" in rendered.upper() or "LONG_BUILDUP" in rendered.upper()
    assert "US DOLLAR INDEX (DXY)" in rendered.upper()
    assert "BENCHMARK BTC" in rendered.upper()
    assert "Jarak Stop Loss: 4,95%" in rendered
    assert "16x" in rendered
