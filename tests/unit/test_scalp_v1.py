"""Comprehensive unit tests for `scalp_v1` strategy covering SC-01 through SC-14."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from crypto_chatbot.alerts.config import ScreenerConfig, load_screener_config, load_universe
from crypto_chatbot.alerts.deadman import AlertDeadManMonitor
from crypto_chatbot.alerts.worker import compute_active_params_sha256
from crypto_chatbot.analysis.forward_proof import freeze_variant_params
from crypto_chatbot.scalp.alerts import (
    build_scalp_digest_summary,
    build_scalp_inline_keyboard,
    compute_position_size,
    make_scalp_run_id,
    render_scalp_alert_message,
    render_scalp_digest,
    render_scalp_settlement_reply,
)
from crypto_chatbot.scalp.backtest import (
    check_1m_contiguity,
    compute_scalp_validation_report,
    render_scalp_validation_markdown,
    run_scalp_backtest_1m,
)
from crypto_chatbot.scalp.config import (
    MacroEventWindow,
    ScalpConfig,
    ScalpCostConfig,
    ScalpPositionConfig,
    load_event_calendar,
)
from crypto_chatbot.scalp.costs import CostFlag, estimate_scalp_trade_costs
from crypto_chatbot.scalp.features import extract_scalp_features_pit
from crypto_chatbot.scalp.filters import (
    FilterReasonCode,
    ScalpFilterInputs,
    evaluate_scalp_filters,
)
from crypto_chatbot.scalp.freeze import SCALP_ENGINE_VERSION, compute_scalp_params_bundle
from crypto_chatbot.scalp.mode import (
    ScalpMode,
    select_scalp_mode_from_candles,
    select_scalp_mode_from_inputs,
    select_scalp_mode_from_snapshot,
)
from crypto_chatbot.scalp.settlement import (
    ScalpOutcomeStatus,
    ScalpTradeSpec,
    settle_scalp_trade_1m,
)
from crypto_chatbot.scalp.setups import (
    ValidDealingRange,
    evaluate_range_fade_setup,
)
from crypto_chatbot.scalp.universe import (
    ScalpUniverseCandidate,
    evaluate_scalp_universe_eligibility,
    refresh_scalp_universe,
)
from crypto_chatbot.schemas.analysis import (
    AnalysisSnapshot,
    Fact,
    InputRange,
    MarketStructure,
    Regime,
)
from crypto_chatbot.schemas.derivatives import (
    ContractStatsObservation,
    OrderBookLevel,
    OrderBookSnapshot,
)
from crypto_chatbot.schemas.market import (
    MarketCandle,
    MarketType,
    QualityStatus,
    Timeframe,
)
from crypto_chatbot.schemas.news import FearGreedObservation
from crypto_chatbot.storage.repositories import ScalpDecisionRecord, TradeSetupRecord

BASE_T = datetime(2026, 11, 20, 10, 0, tzinfo=UTC)


def _make_candle(
    *,
    open_time: datetime,
    tf: Timeframe,
    o: float,
    h: float,
    lo: float,
    c: float,
    vol: float = 1000.0,
    symbol: str = "BTC/USDT:USDT",
) -> MarketCandle:
    close_time = open_time + tf.duration
    base_asset = symbol.split("/", 1)[0]
    return MarketCandle(
        provider="gate",
        venue="gateio",
        canonical_symbol=symbol,
        provider_symbol=f"{base_asset}_USDT",
        market_type=MarketType.PERPETUAL,
        timeframe=tf,
        open_time=open_time,
        close_time=close_time,
        received_time=close_time,
        available_time=close_time,
        open=o,
        high=h,
        low=lo,
        close=c,
        base_volume=vol,
        base_unit=base_asset,
        quote_currency="USDT",
        is_closed=True,
    )


def _make_btc_l2(mid: float = 100_000.0) -> OrderBookSnapshot:
    bid1 = mid - 2.0  # 0.2 bps half-spread -> 0.4 bps full spread
    ask1 = mid + 2.0
    return OrderBookSnapshot(
        provider="gate",
        venue="gateio",
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol="BTC_USDT",
        market_type=MarketType.PERPETUAL,
        observed_time=BASE_T,
        received_time=BASE_T,
        available_time=BASE_T,
        best_bid=bid1,
        best_ask=ask1,
        mid_price=mid,
        spread_abs=ask1 - bid1,
        spread_bps=((ask1 - bid1) / mid) * 10_000.0,
        base_unit="BTC",
        quote_currency="USDT",
        bids=(
            OrderBookLevel(price=bid1, quantity=0.001),  # ~100 USD
            OrderBookLevel(price=bid1 - 5.0, quantity=0.005),  # ~500 USD
        ),
        asks=(
            OrderBookLevel(price=ask1, quantity=0.001),
            OrderBookLevel(price=ask1 + 5.0, quantity=0.005),
        ),
    )


# ---------------------------------------------------------------------------
# SC-01: Strategy config switch & universe
# ---------------------------------------------------------------------------
def test_sc01_strategy_config_switch_and_scalp_universe() -> None:
    cfg = load_screener_config("config/screener.yaml")
    assert cfg.strategy == "scalp_v1"
    assert cfg.is_scalp_mode is True
    assert cfg.is_mtf_mode is False
    assert cfg.effective_universe_path == "config/universe_scalp.txt"

    scalp_uni = load_universe(cfg.effective_universe_path)
    assert scalp_uni == (
        "BTC/USDT:USDT",
        "ETH/USDT:USDT",
        "SOL/USDT:USDT",
        "XRP/USDT:USDT",
        "DOGE/USDT:USDT",
    )

    swing_cfg = ScreenerConfig(strategy="swing_mtf")
    assert swing_cfg.is_scalp_mode is False


# ---------------------------------------------------------------------------
# SC-02: Per-symbol, per-order-type cost model (golden L2 tests + legacy flat)
# ---------------------------------------------------------------------------
def test_sc02_btc_cost_model_golden_l2_and_fallbacks() -> None:
    l2 = _make_btc_l2(100_000.0)
    cost = estimate_scalp_trade_costs(
        canonical_symbol="BTC/USDT:USDT",
        direction="long",
        entry_price=100_000.0,
        stop_loss=99_500.0,  # 50 bps risk
        config=ScalpCostConfig(order_notional_usd=200.0),
        orderbook=l2,
        funding_rate_8h=0.0001,  # +1 bps
        decision_time_utc=datetime(2026, 11, 20, 7, 0, tzinfo=UTC),  # 08:00 UTC inside 90m
        max_hold_minutes=90,
    )
    assert cost.entry_leg.order_type.value == "maker"
    assert cost.entry_leg.total_bps == pytest.approx(2.0)
    assert cost.tp_exit_leg.total_bps == pytest.approx(2.0)
    assert cost.sl_exit_leg.fee_bps == pytest.approx(5.0)
    assert cost.sl_exit_leg.half_spread_bps == pytest.approx(0.2)
    assert cost.sl_exit_leg.slippage_bps > 0.0
    assert cost.funding_settlements_in_window == 1
    assert cost.funding_bps == pytest.approx(1.0)
    assert cost.flags == ()
    assert cost.passes_cost_gate is False  # ~8.45 bps / 50 bps = 16.9% > 15%

    # Fallback when L2 missing flags SPREAD_ESTIMATED + SLIPPAGE_ESTIMATED
    est = estimate_scalp_trade_costs(
        canonical_symbol="BTC/USDT:USDT",
        direction="short",
        entry_price=100_000.0,
        stop_loss=100_700.0,  # 70 bps risk
        orderbook=None,
        rolling_median_spread_bps=0.8,
    )
    assert CostFlag.SPREAD_ESTIMATED in est.flags
    assert CostFlag.SLIPPAGE_ESTIMATED in est.flags
    assert est.passes_cost_gate is True  # (2 + 5 + 0.4 + 1.5) = 8.9 / 70 = 12.7% <= 15%

    # Legacy flat model available as ablation trial
    flat = estimate_scalp_trade_costs(
        canonical_symbol="BTC/USDT:USDT",
        direction="long",
        entry_price=100_000.0,
        stop_loss=99_200.0,
        config=ScalpCostConfig(legacy_flat_mode=True, legacy_flat_round_trip_bps=10.0),
    )
    assert flat.flags == (CostFlag.LEGACY_FLAT_MODEL,)
    assert flat.expected_round_trip_bps == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# SC-03: Universe refresh with eligibility rules & PIT snapshot
# ---------------------------------------------------------------------------
def test_sc03_universe_eligibility_and_idempotent_file_write(tmp_path: Path) -> None:
    candidates = [
        ScalpUniverseCandidate(
            canonical_symbol="BTC/USDT:USDT",
            base_asset="BTC",
            quote_volume_24h_usd=500_000_000.0,
            median_spread_7d_bps=0.4,
            median_bid_depth_10bps_usd=1_000_000.0,
            median_ask_depth_10bps_usd=1_000_000.0,
            listing_age_days=1000,
        ),
        ScalpUniverseCandidate(
            canonical_symbol="WIDE/USDT:USDT",
            base_asset="WIDE",
            quote_volume_24h_usd=200_000_000.0,
            median_spread_7d_bps=2.5,  # > 1.0 bps
            median_bid_depth_10bps_usd=500_000.0,
            median_ask_depth_10bps_usd=500_000.0,
            listing_age_days=300,
        ),
        ScalpUniverseCandidate(
            canonical_symbol="THIN/USDT:USDT",
            base_asset="THIN",
            quote_volume_24h_usd=150_000_000.0,
            median_spread_7d_bps=0.8,
            median_bid_depth_10bps_usd=100_000.0,  # < 250k
            median_ask_depth_10bps_usd=400_000.0,
            listing_age_days=400,
        ),
        ScalpUniverseCandidate(
            canonical_symbol="USDC/USDT:USDT",
            base_asset="USDC",
            quote_volume_24h_usd=500_000_000.0,
            median_spread_7d_bps=0.2,
            median_bid_depth_10bps_usd=2_000_000.0,
            median_ask_depth_10bps_usd=2_000_000.0,
            listing_age_days=800,
        ),
        ScalpUniverseCandidate(
            canonical_symbol="XAU/USDT:USDT",
            base_asset="XAU",
            quote_volume_24h_usd=500_000_000.0,
            median_spread_7d_bps=0.5,
            median_bid_depth_10bps_usd=500_000.0,
            median_ask_depth_10bps_usd=500_000.0,
            listing_age_days=500,
        ),
    ]
    evals = evaluate_scalp_universe_eligibility(candidates)
    by_base = {e.base_asset: e for e in evals}
    assert by_base["BTC"].eligible is True
    assert any("WIDE_SPREAD" in r for r in by_base["WIDE"].rejection_reasons)
    assert any("LOW_BID_DEPTH" in r for r in by_base["THIN"].rejection_reasons)
    assert "EXCLUDED_STABLECOIN" in by_base["USDC"].rejection_reasons
    assert "EXCLUDED_TRADFI" in by_base["XAU"].rejection_reasons

    out_file = tmp_path / "universe_scalp.txt"
    res = refresh_scalp_universe(candidates, captured_at=BASE_T, output_path=out_file)
    assert res.eligible_symbols == ("BTC/USDT:USDT",)
    assert out_file.is_file()
    assert "BTC" in out_file.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# SC-04: Mode selection & parity vs AnalysisSnapshot
# ---------------------------------------------------------------------------
def test_sc04_mode_selection_rules_and_snapshot_parity() -> None:
    res_long = select_scalp_mode_from_inputs(
        regime_1h=Regime.TRENDING_UP,
        adx_14_1h=30.0,
        plus_di_14_1h=28.0,
        minus_di_14_1h=12.0,
        close_1h=150.0,
        ema_50_1h=145.0,
        atr_14_1h=2.0,
        structure_15m=MarketStructure.BULLISH,
        data_quality_1h=QualityStatus.READY,
    )
    assert res_long.mode is ScalpMode.TREND_PULLBACK_LONG

    # Blocked if 15m structure is bearish
    res_conflict = select_scalp_mode_from_inputs(
        regime_1h=Regime.TRENDING_UP,
        adx_14_1h=30.0,
        plus_di_14_1h=28.0,
        minus_di_14_1h=12.0,
        close_1h=150.0,
        ema_50_1h=145.0,
        atr_14_1h=2.0,
        structure_15m=MarketStructure.BEARISH,
        data_quality_1h=QualityStatus.READY,
    )
    assert res_conflict.mode is ScalpMode.NO_TRADE
    assert res_conflict.reason_code == "STRUCTURE_15M_CONFLICT"

    # Range fade requires ADX < 20 AND valid range
    res_rng = select_scalp_mode_from_inputs(
        regime_1h=Regime.RANGING,
        adx_14_1h=15.0,
        plus_di_14_1h=14.0,
        minus_di_14_1h=13.0,
        close_1h=150.0,
        ema_50_1h=150.0,
        atr_14_1h=2.0,
        structure_15m=None,
        data_quality_1h=QualityStatus.READY,
        has_valid_range=True,
        range_low=146.0,
        range_high=154.0,
    )
    assert res_rng.mode is ScalpMode.RANGE_FADE

    # Data quality != READY -> NO_TRADE
    res_dq = select_scalp_mode_from_inputs(
        regime_1h=Regime.TRENDING_UP,
        adx_14_1h=30.0,
        plus_di_14_1h=28.0,
        minus_di_14_1h=12.0,
        close_1h=150.0,
        ema_50_1h=145.0,
        atr_14_1h=2.0,
        structure_15m=MarketStructure.BULLISH,
        data_quality_1h=QualityStatus.DEGRADED,
    )
    assert res_dq.mode is ScalpMode.NO_TRADE
    assert res_dq.reason_code == "DATA_QUALITY_NOT_READY"

    # Parity test: candles path vs snapshot path at identical as_of
    candles_1h = [
        _make_candle(
            open_time=BASE_T - timedelta(hours=160 - i),
            tf=Timeframe.H1,
            o=100.0 + i * 0.5,
            h=101.0 + i * 0.5,
            lo=99.5 + i * 0.5,
            c=100.5 + i * 0.5,
        )
        for i in range(160)
    ]
    candles_15m = [
        _make_candle(
            open_time=BASE_T - timedelta(minutes=15 * (40 - i)),
            tf=Timeframe.M15,
            o=160.0 + i * 0.1,
            h=160.5 + i * 0.1,
            lo=159.8 + i * 0.1,
            c=160.3 + i * 0.1,
        )
        for i in range(40)
    ]
    as_of = candles_1h[-1].close_time
    from crypto_chatbot.analysis.indicators import compute_technical_metrics
    from crypto_chatbot.analysis.regimes import classify_regime
    from crypto_chatbot.analysis.structure import analyze_market_structure

    tech_1h = compute_technical_metrics(candles_1h).metrics
    reg_1h = classify_regime(tech_1h)
    struct_15m = analyze_market_structure(candles_15m).market_structure
    snap_1h = AnalysisSnapshot(
        snapshot_id="snap-parity",
        provider="gate",
        venue="gateio",
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol="BTC_USDT",
        market_type=MarketType.PERPETUAL,
        timeframe=Timeframe.H1,
        as_of=as_of,
        quality_status=QualityStatus.READY,
        quality_flags=(),
        calc_version="1.0",
        input_range=InputRange(
            first_open_time=candles_1h[0].open_time,
            last_open_time=candles_1h[-1].open_time,
            candle_count=len(candles_1h),
            sha256="a" * 64,
        ),
        technical=tech_1h,
        regime=reg_1h,
        market_structure=MarketStructure.BULLISH,
        facts=(
            Fact(
                fact_id="F1",
                label="Close",
                value=tech_1h.last_close,
                unit="USDT",
                display_format="price",
                source="gateio",
                as_of=as_of,
            ),
        ),
    )
    m_candles = select_scalp_mode_from_candles(candles_1h, candles_15m, as_of=as_of)
    m_snap = select_scalp_mode_from_snapshot(snap_1h, structure_15m=struct_15m)
    assert m_candles.mode == m_snap.mode
    assert m_candles.reason_code == m_snap.reason_code


# ---------------------------------------------------------------------------
# SC-05: Trend Pullback + Range Fade setups (common rules: min/max R, net RR)
# ---------------------------------------------------------------------------
def test_sc05_range_fade_and_trend_pullback_rules() -> None:
    # Build 50 5m candles (>= MIN_HISTORY_ATR_14=42) where last bar sweeps range_low=100.0
    candles_5m = [
        _make_candle(
            open_time=BASE_T - timedelta(minutes=5 * (50 - i)),
            tf=Timeframe.M5,
            o=100.2,
            h=100.4,
            lo=100.05,
            c=100.2,
        )
        for i in range(49)
    ]
    # Sweep low=99.50 (SL ~ 99.40 -> ~60 bps risk, cost ~13% of 1R), close back inside at 100.25
    candles_5m.append(
        _make_candle(
            open_time=BASE_T - timedelta(minutes=5),
            tf=Timeframe.M5,
            o=100.1,
            h=100.3,
            lo=99.50,
            c=100.25,
        )
    )
    dr = ValidDealingRange(
        range_low=100.0,
        range_high=103.0,
        width_price=3.0,
        width_bps=295.0,
        source_timeframe="1h",
    )
    res = evaluate_range_fade_setup(
        canonical_symbol="SOL/USDT:USDT",
        dealing_range=dr,
        candles_5m=candles_5m,
        decision_time_utc=BASE_T,
        config=ScalpConfig(
            costs=ScalpCostConfig(fallback_spread_bps=0.6, fallback_taker_slippage_bps=0.5)
        ),
    )
    assert res.status == "READY"
    assert res.direction == "long"
    assert res.risk_bps is not None
    assert 20.0 <= res.risk_bps <= 80.0
    assert res.net_rr_tp2 is not None
    assert res.net_rr_tp2 >= 1.3
    assert res.entry_valid_until_utc == BASE_T + timedelta(minutes=15)

    # Too tight stop -> RISK_TOO_TIGHT
    tight_5m = [
        *list(candles_5m[:-1]),
        _make_candle(
            open_time=BASE_T - timedelta(minutes=5),
            tf=Timeframe.M5,
            o=100.05,
            h=100.06,
            lo=99.99,
            c=100.02,
        ),
    ]
    res_tight = evaluate_range_fade_setup(
        canonical_symbol="SOL/USDT:USDT",
        dealing_range=dr,
        candles_5m=tight_5m,
        decision_time_utc=BASE_T,
    )
    assert res_tight.status == "WAIT"
    assert res_tight.reason_code == "RISK_TOO_TIGHT"

    # Too wide stop (> max_r_bps = 80) -> RISK_TOO_WIDE
    wide_5m = [
        *candles_5m[:-1],
        _make_candle(
            open_time=BASE_T - timedelta(minutes=5),
            tf=Timeframe.M5,
            o=100.1,
            h=100.3,
            lo=98.80,  # ~120 bps risk > 80 bps
            c=100.25,
        ),
    ]
    res_wide = evaluate_range_fade_setup(
        canonical_symbol="SOL/USDT:USDT",
        dealing_range=dr,
        candles_5m=wide_5m,
        decision_time_utc=BASE_T,
    )
    assert res_wide.status == "WAIT"
    assert res_wide.reason_code == "RISK_TOO_WIDE"

    # Cost > 15% of 1R -> COST_GATE_FAILED
    res_cost = evaluate_range_fade_setup(
        canonical_symbol="SOL/USDT:USDT",
        dealing_range=dr,
        candles_5m=candles_5m,
        decision_time_utc=BASE_T,
        config=ScalpConfig(
            costs=ScalpCostConfig(fallback_spread_bps=4.0, fallback_taker_slippage_bps=4.0)
        ),
    )
    assert res_cost.status == "WAIT"
    assert res_cost.reason_code == "COST_GATE_FAILED"

    # Net R:R to TP2 < 1.3 -> NET_RR_TP2_TOO_LOW
    narrow_dr = ValidDealingRange(
        range_low=100.0,
        range_high=101.0,
        width_price=1.0,
        width_bps=99.5,
        source_timeframe="1h",
    )
    res_rr = evaluate_range_fade_setup(
        canonical_symbol="SOL/USDT:USDT",
        dealing_range=narrow_dr,
        candles_5m=candles_5m,
        decision_time_utc=BASE_T,
        config=ScalpConfig(
            costs=ScalpCostConfig(fallback_spread_bps=0.6, fallback_taker_slippage_bps=0.5)
        ),
    )
    assert res_rr.status == "WAIT"
    assert res_rr.reason_code == "NET_RR_TP2_TOO_LOW"


# ---------------------------------------------------------------------------
# SC-06: All 8 blocking filters with reason codes
# ---------------------------------------------------------------------------
def test_sc06_all_eight_blocking_filters() -> None:
    # 1. Funding window (±10m around 08:00 UTC)
    f1 = evaluate_scalp_filters(
        ScalpFilterInputs(
            decision_time_utc=datetime(2026, 11, 20, 7, 55, tzinfo=UTC),
            canonical_symbol="BTC/USDT:USDT",
        )
    )
    assert f1.blocked
    assert FilterReasonCode.FUNDING_WINDOW.value in f1.reason_codes

    # 2. Macro event (-15m / +30m)
    ev = MacroEventWindow(name="US CPI", event_time_utc=datetime(2026, 11, 20, 13, 30, tzinfo=UTC))
    f2 = evaluate_scalp_filters(
        ScalpFilterInputs(
            decision_time_utc=datetime(2026, 11, 20, 13, 40, tzinfo=UTC),
            canonical_symbol="BTC/USDT:USDT",
            macro_events=(ev,),
        )
    )
    assert f2.blocked
    assert FilterReasonCode.MACRO_EVENT.value in f2.reason_codes

    # 3. Liquidation cascade
    f3 = evaluate_scalp_filters(
        ScalpFilterInputs(
            decision_time_utc=BASE_T,
            canonical_symbol="BTC/USDT:USDT",
            recent_5m_liq_usd=(10_000.0, 2_500_000.0, 20_000.0),
            liq_p99_30d_usd=2_000_000.0,
        )
    )
    assert f3.blocked
    assert FilterReasonCode.LIQUIDATION_CASCADE.value in f3.reason_codes

    # 4. Crowded funding (blocks crowded direction only)
    f4_long = evaluate_scalp_filters(
        ScalpFilterInputs(
            decision_time_utc=BASE_T,
            canonical_symbol="BTC/USDT:USDT",
            direction="long",
            funding_rate_8h=0.0006,
        )
    )
    assert f4_long.blocked
    assert FilterReasonCode.CROWDED_FUNDING.value in f4_long.reason_codes
    f4_short = evaluate_scalp_filters(
        ScalpFilterInputs(
            decision_time_utc=BASE_T,
            canonical_symbol="BTC/USDT:USDT",
            direction="short",
            funding_rate_8h=0.0006,
        )
    )
    assert not f4_short.blocked

    # 5. News burst
    f5 = evaluate_scalp_filters(
        ScalpFilterInputs(
            decision_time_utc=BASE_T,
            canonical_symbol="BTC/USDT:USDT",
            news_count_30m=8,
            news_p99_30d_threshold=6.0,
        )
    )
    assert f5.blocked
    assert FilterReasonCode.NEWS_BURST.value in f5.reason_codes

    # 6. Daily stop (<= -3R)
    f6 = evaluate_scalp_filters(
        ScalpFilterInputs(
            decision_time_utc=BASE_T,
            canonical_symbol="BTC/USDT:USDT",
            realized_r_today_utc=-3.2,
        )
    )
    assert f6.blocked
    assert FilterReasonCode.DAILY_STOP.value in f6.reason_codes

    # 7. Loss cooldown (30m)
    f7 = evaluate_scalp_filters(
        ScalpFilterInputs(
            decision_time_utc=BASE_T,
            canonical_symbol="BTC/USDT:USDT",
            last_sl_time_for_symbol_utc=BASE_T - timedelta(minutes=15),
        )
    )
    assert f7.blocked
    assert FilterReasonCode.LOSS_COOLDOWN.value in f7.reason_codes

    # 8. Max concurrency (>= 2 open taken trades)
    f8 = evaluate_scalp_filters(
        ScalpFilterInputs(
            decision_time_utc=BASE_T,
            canonical_symbol="BTC/USDT:USDT",
            open_taken_trades_count=2,
        )
    )
    assert f8.blocked
    assert FilterReasonCode.MAX_CONCURRENCY.value in f8.reason_codes

    # Config file loads cleanly
    cal = load_event_calendar("config/event_calendar.yaml")
    assert len(cal) >= 1


# ---------------------------------------------------------------------------
# SC-07: Point-in-time feature extraction (no future leakage)
# ---------------------------------------------------------------------------
def test_sc07_features_strict_pit_filtering() -> None:
    past_cs = ContractStatsObservation(
        venue="gateio",
        canonical_symbol="SOL/USDT:USDT",
        provider_symbol="SOL_USDT",
        interval="5m",
        observed_time=BASE_T - timedelta(minutes=5),
        received_time=BASE_T - timedelta(minutes=4),
        available_time=BASE_T - timedelta(minutes=4),
        lsr_taker=1.4,
        top_lsr_size=1.2,
        long_liq_usd=50_000.0,
        short_liq_usd=10_000.0,
        open_interest_usd=100_000_000.0,
        mark_price=142.0,
    )
    future_cs = past_cs.model_copy(
        update={
            "observed_time": BASE_T + timedelta(minutes=5),
            "available_time": BASE_T + timedelta(minutes=5),
            "lsr_taker": 9.9,
        }
    )
    past_fg = FearGreedObservation(
        source="alternative_me",
        observed_time=BASE_T - timedelta(hours=2),
        received_time=BASE_T - timedelta(hours=2),
        available_time=BASE_T - timedelta(hours=2),
        value=62,
        classification="Greed",
        normalized_score=0.24,
    )
    future_fg = past_fg.model_copy(
        update={"available_time": BASE_T + timedelta(minutes=1), "value": 99}
    )
    feats = extract_scalp_features_pit(
        decision_time_utc=BASE_T,
        canonical_symbol="SOL/USDT:USDT",
        contract_stats_5m=(past_cs, future_cs),
        fear_greed_observations=(past_fg, future_fg),
    )
    assert feats.taker_buy_sell_ratio_5m_3bar == pytest.approx(1.4)
    assert feats.fear_greed_value == 62


# ---------------------------------------------------------------------------
# SC-08: Alert format, position size calculator, [SHADOW] tag, keyboard
# ---------------------------------------------------------------------------
def test_sc08_alert_format_position_size_and_shadow_tag() -> None:
    pos = compute_position_size(
        canonical_symbol="SOL/USDT:USDT",
        entry_price=142.15,
        stop_loss=141.62,
        position_config=ScalpPositionConfig(equity_usd=1000.0, risk_per_trade_pct=0.5),
    )
    assert pos is not None
    assert pos.risk_usd == pytest.approx(5.0)
    assert pos.qty_base == pytest.approx(5.0 / 0.53, rel=1e-3)

    # Build a READY setup (>= MIN_HISTORY_ATR_14=42 bars) and verify rendered message
    candles_5m = [
        _make_candle(
            open_time=BASE_T - timedelta(minutes=5 * (50 - i)),
            tf=Timeframe.M5,
            o=142.3,
            h=142.5,
            lo=142.15,
            c=142.3,
            symbol="SOL/USDT:USDT",
        )
        for i in range(49)
    ] + [
        _make_candle(
            open_time=BASE_T - timedelta(minutes=5),
            tf=Timeframe.M5,
            o=142.15,
            h=142.4,
            lo=141.35,
            c=142.30,
            symbol="SOL/USDT:USDT",
        )
    ]
    dr = ValidDealingRange(
        range_low=142.10,
        range_high=145.50,
        width_price=3.40,
        width_bps=236.0,
        source_timeframe="1h",
    )
    setup = evaluate_range_fade_setup(
        canonical_symbol="SOL/USDT:USDT",
        dealing_range=dr,
        candles_5m=candles_5m,
        decision_time_utc=BASE_T,
        config=ScalpConfig(
            costs=ScalpCostConfig(fallback_spread_bps=0.5, fallback_taker_slippage_bps=0.5)
        ),
    )
    assert setup.status == "READY"
    feats = extract_scalp_features_pit(decision_time_utc=BASE_T, canonical_symbol="SOL/USDT:USDT")
    msg_shadow = render_scalp_alert_message(
        canonical_symbol="SOL/USDT:USDT",
        setup=setup,
        regime_1h_label="RANGING",
        structure_15m_label="ranging",
        features=feats,
        position_config=ScalpPositionConfig(equity_usd=1000.0, risk_per_trade_pct=0.5),
        phase="shadow",
        snapshot_id="scdec-1234",
        params_sha256="abcdef1234567890",
    )
    assert "[SHADOW]" in msg_shadow
    assert "SCALP LONG · SOL" in msg_shadow
    assert "LIMIT BUY" in msg_shadow
    assert "Time stop: 90 menit setelah terisi" in msg_shadow
    assert "Ukuran (risk 0.5% dari modal di config):" in msg_shadow

    msg_live = render_scalp_alert_message(
        canonical_symbol="SOL/USDT:USDT",
        setup=setup,
        regime_1h_label="RANGING",
        structure_15m_label="unknown",
        features=feats,
        position_config=ScalpPositionConfig(equity_usd=1000.0, risk_per_trade_pct=0.5),
        phase="live",
    )
    assert "[SHADOW]" not in msg_live

    # Missing equity_usd / risk_per_trade_pct -> "size: not configured"
    msg_unconfigured = render_scalp_alert_message(
        canonical_symbol="SOL/USDT:USDT",
        setup=setup,
        regime_1h_label="RANGING",
        structure_15m_label="unknown",
        features=feats,
        position_config=ScalpPositionConfig(),
        phase="shadow",
    )
    assert "size: not configured" in msg_unconfigured

    kb = build_scalp_inline_keyboard("al-abc")
    texts = [b["text"] for b in kb["inline_keyboard"][0]]
    assert texts == ["Pasang", "Skip", "Mute 1j", "Detail"]


# ---------------------------------------------------------------------------
# SC-09: Shared 1m settlement rules (fill-through, TP1-before-fill, SL-before-TP, gap, time stop)
# ---------------------------------------------------------------------------
def test_sc09_1m_settlement_rules() -> None:
    spec = ScalpTradeSpec(
        direction="long",
        entry_limit=100.0,
        stop_loss=99.5,  # R = 0.5
        tp1=100.5,
        tp2=101.0,
        signal_time_utc=BASE_T,
        human_delay_sec=0,
        entry_valid_minutes=15,
        max_hold_minutes=90,
        tick_size=0.01,
    )

    # 1. Touch alone (low == 100.0) must NOT fill
    touch_bar = _make_candle(open_time=BASE_T, tf=Timeframe.M1, o=100.2, h=100.3, lo=100.0, c=100.2)
    out_touch = settle_scalp_trade_1m(spec, [touch_bar])
    assert out_touch.filled is False
    assert out_touch.status is ScalpOutcomeStatus.PENDING

    # 2. TP1 touched before fill -> CANCELLED_TP1_BEFORE_FILL
    tp1_first = _make_candle(open_time=BASE_T, tf=Timeframe.M1, o=100.2, h=100.6, lo=100.1, c=100.5)
    out_cancel = settle_scalp_trade_1m(spec, [tp1_first])
    assert out_cancel.status is ScalpOutcomeStatus.CANCELLED_TP1_BEFORE_FILL
    assert out_cancel.filled is False

    # 3. Trade through by >= tick -> fills, then TP1 + TP2 -> TP2_HIT
    fill_bar = _make_candle(open_time=BASE_T, tf=Timeframe.M1, o=100.1, h=100.2, lo=99.98, c=100.15)
    tp1_bar = _make_candle(
        open_time=BASE_T + timedelta(minutes=1),
        tf=Timeframe.M1,
        o=100.15,
        h=100.55,
        lo=100.1,
        c=100.5,
    )
    tp2_bar = _make_candle(
        open_time=BASE_T + timedelta(minutes=2),
        tf=Timeframe.M1,
        o=100.5,
        h=101.1,
        lo=100.45,
        c=101.05,
    )
    out_tp2 = settle_scalp_trade_1m(spec, [fill_bar, tp1_bar, tp2_bar])
    assert out_tp2.status is ScalpOutcomeStatus.TP2_HIT
    assert out_tp2.gross_r == pytest.approx(0.6 * 1.0 + 0.4 * 2.0)
    assert out_tp2.net_r < out_tp2.gross_r
    reply = render_scalp_settlement_reply(
        canonical_symbol="BTC/USDT:USDT", snapshot_id="sc-1", outcome=out_tp2
    )
    assert "TP2 TERCAPAI" in reply

    # 4. Gap through SL fills at bar.open (worse than SL)
    gap_bar = _make_candle(
        open_time=BASE_T + timedelta(minutes=1),
        tf=Timeframe.M1,
        o=99.0,  # gapped below SL=99.5
        h=99.2,
        lo=98.9,
        c=99.1,
    )
    out_gap = settle_scalp_trade_1m(spec, [fill_bar, gap_bar])
    assert out_gap.status is ScalpOutcomeStatus.SL_HIT
    assert out_gap.exit_price == pytest.approx(99.0)
    assert out_gap.gross_r == pytest.approx(-2.0)

    # 5. Time stop at 90m (18 x 5m bars)
    flat_bars = [fill_bar] + [
        _make_candle(
            open_time=BASE_T + timedelta(minutes=i),
            tf=Timeframe.M1,
            o=100.1,
            h=100.2,
            lo=99.9,
            c=100.1,
        )
        for i in range(1, 95)
    ]
    out_ts = settle_scalp_trade_1m(spec, flat_bars)
    assert out_ts.status is ScalpOutcomeStatus.TIME_STOP
    assert out_ts.is_final is True

    # 6. Intrabar SL-before-TP: bar hitting BOTH SL (99.5) and TP1 (100.5) settles as SL_HIT
    both_bar = _make_candle(
        open_time=BASE_T + timedelta(minutes=1),
        tf=Timeframe.M1,
        o=100.1,
        h=101.2,
        lo=99.4,
        c=100.8,
    )
    out_sl_first = settle_scalp_trade_1m(spec, [fill_bar, both_bar])
    assert out_sl_first.status is ScalpOutcomeStatus.SL_HIT
    assert out_sl_first.exit_price == pytest.approx(99.5)

    # 7. TP1 (60%) -> SL moves to BE + costs -> BE hit on remaining 40%
    be_hit_bar = _make_candle(
        open_time=BASE_T + timedelta(minutes=2),
        tf=Timeframe.M1,
        o=100.4,
        h=100.45,
        lo=100.0,  # drops below BE+costs (>100.0)
        c=100.05,
    )
    out_tp1_be = settle_scalp_trade_1m(spec, [fill_bar, tp1_bar, be_hit_bar])
    assert out_tp1_be.status is ScalpOutcomeStatus.TP1_THEN_BE
    assert out_tp1_be.tp1_hit is True
    assert out_tp1_be.exit_price is not None
    assert out_tp1_be.exit_price > 100.0

    # 8. Entry validity 3 x 5m bars (15 min) -> EXPIRED_UNFILLED
    late_bar = _make_candle(
        open_time=BASE_T + timedelta(minutes=15),
        tf=Timeframe.M1,
        o=100.1,
        h=100.2,
        lo=99.9,
        c=100.1,
    )
    out_exp = settle_scalp_trade_1m(spec, [late_bar])
    assert out_exp.status is ScalpOutcomeStatus.EXPIRED_UNFILLED
    assert out_exp.filled is False

    # 9. Human delay 90s: bar closing at +60s cannot fill; bar closing at +120s fills
    spec_delay = spec.model_copy(update={"human_delay_sec": 90})
    bar_60s = _make_candle(
        open_time=BASE_T,
        tf=Timeframe.M1,
        o=100.1,
        h=100.2,
        lo=99.95,
        c=100.1,
    )
    bar_120s = _make_candle(
        open_time=BASE_T + timedelta(minutes=1),
        tf=Timeframe.M1,
        o=100.1,
        h=100.2,
        lo=99.95,
        c=100.1,
    )
    out_only_60s = settle_scalp_trade_1m(spec_delay, [bar_60s])
    assert out_only_60s.filled is False
    out_with_120s = settle_scalp_trade_1m(spec_delay, [bar_60s, bar_120s])
    assert out_with_120s.filled is True
    assert out_with_120s.fill_time_utc == BASE_T + timedelta(seconds=90)


# ---------------------------------------------------------------------------
# SC-10 & SC-11: Contiguity, deterministic 1m backtest, validation report
# ---------------------------------------------------------------------------
def test_sc10_sc11_contiguity_and_validation_report(tmp_path: Path) -> None:
    # Build 200 contiguous 1m bars, then introduce 1 gap
    bars = [
        _make_candle(
            open_time=BASE_T + timedelta(minutes=i),
            tf=Timeframe.M1,
            o=100.0,
            h=100.1,
            lo=99.9,
            c=100.0,
        )
        for i in range(10)
    ]
    ok_rep = check_1m_contiguity(bars, symbol="BTC/USDT:USDT")
    assert ok_rep.is_contiguous is True

    gapped = [
        *bars[:5],
        _make_candle(
            open_time=BASE_T + timedelta(minutes=8),
            tf=Timeframe.M1,
            o=100.0,
            h=100.1,
            lo=99.9,
            c=100.0,
        ),
    ]
    bad_rep = check_1m_contiguity(gapped, symbol="BTC/USDT:USDT")
    assert bad_rep.is_contiguous is False
    assert bad_rep.gap_count == 1
    assert bad_rep.missing_minutes == 3

    # Run backtest + report on empty/short fixture cleanly & deterministically
    bt_a = run_scalp_backtest_1m({"BTC/USDT:USDT": bars}, seed=42)
    bt_b = run_scalp_backtest_1m({"BTC/USDT:USDT": bars}, seed=42)
    assert bt_a.model_dump() == bt_b.model_dump()

    sha, _ = compute_scalp_params_bundle(scalp_config=ScalpConfig(), universe=("BTC/USDT:USDT",))
    val = compute_scalp_validation_report(
        base_result=bt_a,
        candles_1m_by_symbol={"BTC/USDT:USDT": bars},
        trial_id="scalp_v1-unit-001",
        params_sha256=sha,
        registered_trials_count=1,
        seed=42,
    )
    md = render_scalp_validation_markdown(val)
    assert "scalp_v1-unit-001" in md
    assert "Cost Sensitivity & DSR" in md
    assert "Random-Entry Benchmark" in md


# ---------------------------------------------------------------------------
# SC-12: Systemd timers & 15m dead-man threshold
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_sc12_systemd_and_15m_deadman() -> None:
    script = Path("deploy/gen-systemd.sh").read_text(encoding="utf-8")
    assert "*:00/5:20" in script
    assert "settle-setups --strategy scalp_v1" in script
    assert "*:01/5:00" in script
    assert "universe refresh --strategy scalp_v1" in script

    sent: list[str] = []

    class _FakeRepo:
        def get_latest_completed_run(self) -> Any:
            return None

    class _FakeSender:
        async def send_message(self, **kw: Any) -> None:
            sent.append(str(kw["text"]))

    mon = AlertDeadManMonitor(
        alert_repo=_FakeRepo(),  # type: ignore[arg-type]
        sender=_FakeSender(),  # type: ignore[arg-type]
        chat_id=123,
        max_silence_minutes=15.0,
        clock=lambda: BASE_T,
    )
    assert await mon.check_once() is True
    assert len(sent) == 1


# ---------------------------------------------------------------------------
# SC-13: Scalp daily digest (per mode, symbol, filters fired, fill rate)
# ---------------------------------------------------------------------------
def test_sc13_scalp_digest_fixture() -> None:
    decisions = [
        ScalpDecisionRecord(
            decision_id="d1",
            run_id=make_scalp_run_id(BASE_T),
            venue="gateio",
            canonical_symbol="BTC/USDT:USDT",
            decision_time=BASE_T,
            mode=ScalpMode.TREND_PULLBACK_LONG.value,
            decision_status="READY",
            direction="long",
            reason_codes=("SETUP_READY",),
            created_at=BASE_T,
        ),
        ScalpDecisionRecord(
            decision_id="d2",
            run_id=make_scalp_run_id(BASE_T),
            venue="gateio",
            canonical_symbol="SOL/USDT:USDT",
            decision_time=BASE_T,
            mode=ScalpMode.NO_TRADE.value,
            decision_status="NO_TRADE",
            reason_codes=(FilterReasonCode.FUNDING_WINDOW.value,),
            created_at=BASE_T,
        ),
    ]
    setups = [
        TradeSetupRecord(
            snapshot_id="d1",
            venue="gateio",
            canonical_symbol="BTC/USDT:USDT",
            timeframe=Timeframe.M5,
            as_of=BASE_T,
            status="ready",
            bias="long",
            reference_price=100_000.0,
            strategy="scalp_v1",
            outcome_status=ScalpOutcomeStatus.TP2_HIT.value,
            realized_r=1.35,
        ),
        TradeSetupRecord(
            snapshot_id="d3",
            venue="gateio",
            canonical_symbol="ETH/USDT:USDT",
            timeframe=Timeframe.M5,
            as_of=BASE_T,
            status="ready",
            bias="short",
            reference_price=3_500.0,
            strategy="scalp_v1",
            outcome_status=ScalpOutcomeStatus.EXPIRED_UNFILLED.value,
            realized_r=0.0,
        ),
    ]
    summary = build_scalp_digest_summary(decisions=decisions, setups=setups)
    assert summary.ready_count == 1
    assert summary.no_trade_count == 1
    assert summary.filters_fired[FilterReasonCode.FUNDING_WINDOW.value] == 1
    assert summary.fill_rate == pytest.approx(0.5)
    rendered = render_scalp_digest(summary)
    assert "Ringkasan Harian Scalp v1" in rendered
    assert "FILTER_FUNDING_WINDOW" in rendered
    assert "50.0%" in rendered


# ---------------------------------------------------------------------------
# SC-14: Freeze params hash == alerts run hash
# ---------------------------------------------------------------------------
def test_sc14_freeze_params_matches_alerts_run_hash(tmp_path: Path) -> None:
    cfg = load_screener_config("config/screener.yaml")
    universe = load_universe(cfg.effective_universe_path)
    art = freeze_variant_params(
        "scalp_v1",
        output_path=tmp_path / "frozen_params.json",
        screener_config=cfg,
        universe=universe,
    )
    run_sha = compute_active_params_sha256(screener_config=cfg, universe=universe)
    assert art.params_sha256 == run_sha
    assert art.mtf_engine_version == SCALP_ENGINE_VERSION

    # Changing cost config changes hash
    cfg_cost = cfg.model_copy(
        update={
            "scalp": cfg.scalp.model_copy(
                update={"costs": cfg.scalp.costs.model_copy(update={"maker_fee_bps": 1.5})}
            )
        }
    )
    assert compute_active_params_sha256(screener_config=cfg_cost, universe=universe) != run_sha


def test_sc08_position_config_reads_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SCALP_EQUITY_USD", "2500")
    monkeypatch.setenv("SCALP_RISK_PER_TRADE_PCT", "0.75")
    cfg = load_screener_config("config/screener.yaml")
    assert cfg.scalp.position.equity_usd == pytest.approx(2500.0)
    assert cfg.scalp.position.risk_per_trade_pct == pytest.approx(0.75)
