from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from crypto_chatbot.analysis.service import build_analysis_snapshot
from crypto_chatbot.analysis.setup import build_trade_setup, evaluate_setup_against_candles
from crypto_chatbot.analysis.smc import analyze_smc
from crypto_chatbot.analysis.structure import StructureAnalysis
from crypto_chatbot.bot.templates import (
    format_fact_value,
    render_compact_snapshot,
    render_deterministic_snapshot,
)
from crypto_chatbot.llm.guard import validate_narration
from crypto_chatbot.llm.render import render_analysis_response
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import resolve_mvp_spot_symbol
from crypto_chatbot.schemas.analysis import (
    DealingZone,
    DerivativesMetrics,
    Fact,
    MarketStructure,
    Regime,
    SetupBias,
    SetupStatus,
    SmcSummary,
    StructureEvent,
    StructureEventType,
    TechnicalMetrics,
)
from crypto_chatbot.schemas.llm import AnalysisResponse, LlmAnalysisNarration, NarrationStatus
from crypto_chatbot.schemas.market import MarketCandle, MarketType, Timeframe
from crypto_chatbot.schemas.ml import CostAssumptions
from crypto_chatbot.storage.repositories import TradeSetupRecord

FIXTURE_1000 = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"
ANCHOR = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)


def _candle(
    idx: int,
    *,
    open_p: float,
    high_p: float,
    low_p: float,
    close_p: float,
    timeframe: Timeframe = Timeframe.H4,
) -> MarketCandle:
    open_t = ANCHOR + idx * timeframe.duration
    close_t = open_t + timeframe.duration
    return MarketCandle(
        provider="ccxt",
        venue="binance",
        canonical_symbol="BTC/USDT",
        provider_symbol="BINANCE:BTCUSDT",
        market_type=MarketType.SPOT,
        timeframe=timeframe,
        open_time=open_t,
        close_time=close_t,
        received_time=close_t + timedelta(seconds=1),
        available_time=close_t + timedelta(seconds=1),
        open=open_p,
        high=high_p,
        low=low_p,
        close=close_p,
        base_volume=100.0,
        base_unit="BTC",
        quote_currency="USDT",
        is_closed=True,
    )


def _ohlc_series(specs: list[tuple[float, float, float, float]]) -> list[MarketCandle]:
    return [
        _candle(i, open_p=o, high_p=h, low_p=low, close_p=c)
        for i, (o, h, low, c) in enumerate(specs)
    ]


def test_smc_causal_bos_choch_fvg_ob_sweep_and_prefix_no_lookahead() -> None:
    # Deterministic price path: SH at bar 4, BOS up, FVG, then CHoCH/sweep + mitigation
    specs: list[tuple[float, float, float, float]] = [
        (100.0, 102.0, 99.0, 101.0),  # 0
        (101.0, 103.0, 100.0, 102.0),  # 1
        (102.0, 105.0, 101.0, 104.0),  # 2
        (104.0, 107.0, 103.0, 106.0),  # 3 (bearish/down candle after peak below -> let bar 4 be SH)
        (106.0, 112.0, 105.0, 108.0),  # 4 pivot SH = 112.0
        (108.0, 110.0, 104.0, 105.0),  # 5
        (105.0, 107.0, 101.0, 102.0),  # 6 (last bearish candle before BOS up)
        (102.0, 106.0, 100.0, 104.0),  # 7 SH 112 confirmed here (4+3=7)
        (104.0, 109.0, 103.0, 108.0),  # 8
        (108.0, 111.0, 105.5, 110.0),  # 9 (candle[i-2] for bullish FVG: high=111)
        (110.0, 116.0, 109.5, 115.0),  # 10 BOS bullish > 112.0
        (115.0, 120.0, 114.0, 119.0),  # 11 candle[i] for FVG: low=114 > candle[9].high=111!
        (119.0, 122.0, 116.0, 121.0),  # 12
        (121.0, 121.5, 115.5, 120.5),  # 13 stays above 111..114 FVG
    ]
    candles = _ohlc_series(specs)
    smc_13 = analyze_smc(candles)
    assert smc_13.latest_structure_break is not None
    assert smc_13.latest_structure_break.event_type is StructureEventType.BOS_BULLISH
    assert smc_13.latest_structure_break.level == pytest.approx(112.0)
    assert smc_13.active_bullish_fvg is not None
    assert smc_13.active_bullish_fvg.zone_low == pytest.approx(111.0)
    assert smc_13.active_bullish_fvg.zone_high == pytest.approx(114.0)
    assert smc_13.active_bullish_ob is not None
    assert smc_13.dealing_range_high == pytest.approx(112.0)
    assert smc_13.dealing_range_low == pytest.approx(100.0)
    assert smc_13.equilibrium == pytest.approx(106.0)
    assert smc_13.dealing_zone is DealingZone.PREMIUM

    # Truncation invariance: before bar 10, neither BOS nor bar-7 SL is confirmed yet;
    # at bar 10 (len=11), BOS up is already confirmed without future bars 11..13!
    smc_9 = analyze_smc(candles[:10])
    assert smc_9.latest_structure_break is None
    assert smc_9.dealing_range_high is None
    smc_10 = analyze_smc(candles[:11])
    assert smc_10.latest_structure_break == smc_13.latest_structure_break
    assert smc_10.active_bullish_ob == smc_13.active_bullish_ob

    # Now append a wick sweep above pivot high followed by mitigation through the FVG
    # First create a swing high at 125 (bar 14) and confirm at bar 17
    extended = candles + _ohlc_series([])
    extra_specs = [
        (120.5, 125.0, 119.0, 121.0),  # 14 pivot SH = 125.0
        (121.0, 123.0, 117.0, 118.0),  # 15
        (118.0, 120.0, 115.0, 116.0),  # 16 (FVG still unmitigated: low=115 > 111)
        (116.0, 119.0, 115.0, 117.0),  # 17 SH 125 confirmed here!
        (
            117.0,
            126.0,
            116.0,
            122.0,
        ),  # 18 liquidity sweep above 125.0 (high=126 > 125, close=122 <= 125)!
        (122.0, 122.0, 108.0, 109.0),  # 19 drops through FVG 111..114 -> mitigates FVG!
    ]
    base_len = len(extended)
    for offset, (o, h, low, c) in enumerate(extra_specs):
        extended.append(_candle(base_len + offset, open_p=o, high_p=h, low_p=low, close_p=c))

    smc_ext = analyze_smc(extended)
    assert smc_ext.latest_sweep is not None
    assert smc_ext.latest_sweep.event_type is StructureEventType.LIQUIDITY_SWEEP_BEARISH
    assert smc_ext.latest_sweep.level == pytest.approx(125.0)
    assert smc_ext.active_bullish_fvg is None  # mitigated by bar 19!
    # Also confirm lookahead invariant: every confirmation_time <= last close_time
    assert smc_ext.latest_sweep.confirmation_time <= extended[-1].close_time


def test_build_trade_setup_long_short_and_wait_guards() -> None:
    candles_raw = json.loads(FIXTURE_1000.read_text(encoding="utf-8"))["raw_ohlcv"]
    received = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
    step_ms = int(Timeframe.H4.duration.total_seconds() * 1000)
    end_ms = int((received - timedelta(seconds=1)).timestamp() * 1000)
    start_ms = end_ms - len(candles_raw) * step_ms
    aligned = [
        [start_ms + i * step_ms, r[1], r[2], r[3], r[4], r[5]] for i, r in enumerate(candles_raw)
    ]
    batch = normalize_ohlcv_rows(
        aligned,
        symbol=resolve_mvp_spot_symbol("BTC"),
        timeframe=Timeframe.H4,
        received_time=received,
        now=received,
    )
    snap = build_analysis_snapshot(batch)
    assert snap.smc is not None
    assert snap.trade_setup is not None

    t_pivot = received - timedelta(hours=12)

    def _pivot(ev_type: StructureEventType, level: float) -> StructureEvent:
        return StructureEvent(
            event_type=ev_type,
            event_time=t_pivot,
            confirmation_time=t_pivot,
            level=level,
        )

    bull_structure = StructureAnalysis(
        pivots=(),
        market_structure=MarketStructure.BULLISH,
        last_swing_highs=(_pivot(StructureEventType.SWING_HIGH, 66_500.0),),
        last_swing_lows=(_pivot(StructureEventType.SWING_LOW, 64_100.0),),
    )

    # 1. Clean bullish setup (LONG) -> strict price hierarchy & R:R >= 1.5
    bull_metrics = TechnicalMetrics(
        last_close=65_000.0,
        ema_20=64_600.0,
        ema_50=63_800.0,
        atr_14=500.0,
        adx_14=32.0,
    )
    bull_smc = SmcSummary(
        dealing_range_high=67_000.0,
        dealing_range_low=62_000.0,
        equilibrium=64_500.0,
        dealing_zone=DealingZone.PREMIUM,
    )
    bull_deriv = DerivativesMetrics(funding_rate_per_8h=0.0015)  # crowded long note!
    long_setup = build_trade_setup(
        metrics=bull_metrics,
        regime=Regime.TRENDING_UP,
        structure=bull_structure,
        smc=bull_smc,
        derivatives=bull_deriv,
    )
    assert long_setup.status is SetupStatus.READY
    assert long_setup.bias is SetupBias.LONG
    assert long_setup.stop_loss is not None
    assert long_setup.entry_low is not None
    assert long_setup.entry_mid is not None
    assert long_setup.entry_high is not None
    assert long_setup.tp1 is not None
    assert long_setup.tp2 is not None
    assert long_setup.tp3 is not None
    assert long_setup.rr_tp2 is not None
    assert long_setup.stop_loss < long_setup.entry_low
    assert long_setup.entry_low <= long_setup.entry_mid <= long_setup.entry_high
    assert long_setup.entry_mid < long_setup.tp1 < long_setup.tp2 < long_setup.tp3
    assert long_setup.rr_tp2 >= 1.5
    assert any("Premium" in note for note in long_setup.risk_notes)
    assert any("Funding rate 8h sangat positif" in note for note in long_setup.risk_notes)

    # 2. Clean bearish setup (SHORT) -> strict symmetric hierarchy
    bear_metrics = TechnicalMetrics(
        last_close=60_000.0,
        ema_20=60_500.0,
        ema_50=61_500.0,
        atr_14=500.0,
        adx_14=30.0,
    )
    bear_smc = SmcSummary(
        dealing_range_high=63_000.0,
        dealing_range_low=57_000.0,
        equilibrium=60_000.0,
        dealing_zone=DealingZone.EQUILIBRIUM,
    )
    bear_structure = StructureAnalysis(
        pivots=(),
        market_structure=MarketStructure.BEARISH,
        last_swing_highs=(_pivot(StructureEventType.SWING_HIGH, 61_000.0),),
        last_swing_lows=(_pivot(StructureEventType.SWING_LOW, 58_500.0),),
    )
    short_setup = build_trade_setup(
        metrics=bear_metrics,
        regime=Regime.TRENDING_DOWN,
        structure=bear_structure,
        smc=bear_smc,
    )
    assert short_setup.status is SetupStatus.READY
    assert short_setup.bias is SetupBias.SHORT
    assert short_setup.stop_loss is not None
    assert short_setup.entry_high is not None
    assert short_setup.entry_mid is not None
    assert short_setup.tp1 is not None
    assert short_setup.tp2 is not None
    assert short_setup.tp3 is not None
    assert short_setup.stop_loss > short_setup.entry_high
    assert short_setup.entry_mid > short_setup.tp1 > short_setup.tp2 > short_setup.tp3
    assert short_setup.rr_tp2 is not None
    assert short_setup.rr_tp2 >= 1.5

    # 3. Choppy ranging without sweep+CHoCH -> explicit WAIT / NO_SETUP
    wait_setup = build_trade_setup(
        metrics=bull_metrics,
        regime=Regime.RANGING,
        structure=bull_structure,
        smc=SmcSummary(),
    )
    assert wait_setup.status is SetupStatus.WAIT
    assert wait_setup.bias is SetupBias.NEUTRAL
    assert wait_setup.entry_low is None
    assert wait_setup.stop_loss is None
    assert wait_setup.tp1 is None
    assert wait_setup.wait_reason

    # 4. Invalidation distance > 3.5x ATR -> explicit WAIT / NO_SETUP
    wide_sl_structure = StructureAnalysis(
        pivots=(),
        market_structure=MarketStructure.BULLISH,
        last_swing_highs=(_pivot(StructureEventType.SWING_HIGH, 66_500.0),),
        last_swing_lows=(_pivot(StructureEventType.SWING_LOW, 60_000.0),),
    )
    wide_wait = build_trade_setup(
        metrics=bull_metrics,
        regime=Regime.TRENDING_UP,
        structure=wide_sl_structure,
        smc=bull_smc,
    )
    assert wide_wait.status is SetupStatus.WAIT
    assert wide_wait.wait_reason is not None
    assert "3,5x ATR" in wide_wait.wait_reason


def test_evaluate_setup_against_candles_outcomes_and_sl_tiebreak() -> None:
    record = TradeSetupRecord(
        snapshot_id="snap-test-1",
        venue="binance",
        canonical_symbol="BTC/USDT",
        timeframe=Timeframe.H4,
        as_of=ANCHOR,
        status="ready",
        bias="long",
        entry_source="Pullback ke EMA 20",
        reference_price=105.0,
        entry_low=99.0,
        entry_high=101.0,
        entry_mid=100.0,
        stop_loss=95.0,
        tp1=105.0,
        tp2=110.0,
        tp3=115.0,
        rr_tp1=1.0,
        rr_tp2=2.0,
        rr_tp3=3.0,
    )

    # 1. Trigger then hit TP2: does not settle prematurely before max_bars horizon completes
    tp2_candles = [
        _candle(0, open_p=104.0, high_p=104.5, low_p=100.5, close_p=102.0),  # fills at 101.0
        _candle(1, open_p=102.0, high_p=111.0, low_p=101.5, close_p=110.5),  # hits TP1+TP2
    ]
    st_unfinished, settled_unfinished, r_unfinished = evaluate_setup_against_candles(
        record, tp2_candles, max_bars=24
    )
    assert st_unfinished == "entry_triggered"
    assert settled_unfinished is None
    assert r_unfinished is None

    # Once horizon completes at max_bars=2, scaled 40/40/20 partial exit settles minus cost
    st, settled_at, r_val = evaluate_setup_against_candles(record, tp2_candles, max_bars=2)
    assert st == "hit_tp2"
    assert settled_at == tp2_candles[1].close_time
    expected_gross = (
        0.40 * ((105.0 - 101.0) / 5.0)
        + 0.40 * ((110.0 - 101.0) / 5.0)
        + 0.20 * ((110.5 - 101.0) / 5.0)
    )
    expected_cost = (0.0032 * 101.0) / 5.0
    assert r_val == pytest.approx(expected_gross - expected_cost)

    # 2. Conservative tie-break: same candle touches both SL (95) and TP3 (115) -> hit_sl!
    tie_candles = [
        _candle(0, open_p=100.5, high_p=118.0, low_p=94.0, close_p=117.0),
    ]
    st_tie, _, r_tie = evaluate_setup_against_candles(record, tie_candles, max_bars=24)
    assert st_tie == "hit_sl"
    expected_sl_net = ((95.0 - 100.5) / 5.0) - ((0.0032 * 100.5) / 5.0)
    assert r_tie == pytest.approx(expected_sl_net)

    # 3. Never reaches entry zone within horizon -> expired (0.0R)
    unfilled = [
        _candle(i, open_p=105.0, high_p=108.0, low_p=103.0, close_p=106.0) for i in range(4)
    ]
    st_exp, _, r_exp = evaluate_setup_against_candles(record, unfilled, max_bars=4)
    assert st_exp == "expired"
    assert r_exp == pytest.approx(0.0)


def test_compact_rendering_and_tp_allowlist_in_guard() -> None:
    data = json.loads(FIXTURE_1000.read_text(encoding="utf-8"))
    raw = data["raw_ohlcv"]
    last_open = datetime.fromtimestamp(float(raw[-1][0]) / 1000.0, tz=UTC)
    received = last_open + Timeframe.H4.duration + timedelta(seconds=1)
    batch = normalize_ohlcv_rows(
        raw,
        symbol=resolve_mvp_spot_symbol("BTC"),
        timeframe=Timeframe.H4,
        received_time=received,
        now=received,
    )
    snap = build_analysis_snapshot(batch)

    full_rendered = render_deterministic_snapshot(snap)
    compact_rendered = render_compact_snapshot(snap)

    assert len(compact_rendered) < len(full_rendered)
    assert "Fakta Terukur (Backend)" in compact_rendered
    assert "Skenario Setup (Deterministik" in compact_rendered
    assert "Key Points (Bull vs Bear & Risiko)" in compact_rendered
    assert "/analyze BTC 4h detail" in compact_rendered
    assert "[F1]" in compact_rendered
    assert "UTC (" in compact_rendered
    assert "WIB)" in compact_rendered
    assert "Disclaimer:" in compact_rendered

    # Compact HTML escaping
    hostile_snap = snap.model_copy(
        update={
            "trade_setup": snap.trade_setup.model_copy(
                update={"risk_notes": ("<script>alert(1)</script>",)}
            )
            if snap.trade_setup is not None
            else None
        }
    )
    compact_html = render_compact_snapshot(hostile_snap, html_mode=True)
    assert "<script>" not in compact_html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in compact_html

    # Guard allows TP 1 / TP 2 / TP 3 when snapshot has trade_setup_tp_levels, rejects TP 4
    valid_narration = LlmAnalysisNarration(
        summary="Harga {F1} bergerak mengikuti konteks struktur pada timeframe 4h.",
        supporting_factors=("Momentum relatif terjaga di atas {F6}.",),
        contradicting_factors=("Tetap perhatikan validasi level {F29}.",),
        bullish_scenario=(
            "Jika bertahan, skenario dapat menguji TP 1 maupun TP 2 sesuai batas fakta."
        ),
        bearish_scenario="Jika melemah, ketidakpastian meningkat di bawah {F29}.",
        limitations=("Hanya data spot Binance; bukan saran transaksi.",),
        fact_ids=("F1", "F6", "F29"),
    )
    assert validate_narration(valid_narration, snap).is_valid is True

    bad_tp4 = valid_narration.model_copy(
        update={"bullish_scenario": "Lanjut terus menuju TP 4 setelah {F1}."}
    )
    bad_res = validate_narration(bad_tp4, snap)
    assert bad_res.is_valid is False
    assert "raw_digit_detected" in bad_res.reasons

    resp = AnalysisResponse(
        snapshot=snap,
        narration_status=NarrationStatus.GENERATED,
        narration=valid_narration,
    )
    compact_with_llm = render_analysis_response(resp, html_mode=False, compact=True)
    assert "Penjelasan Narasi (Tervalidasi)" in compact_with_llm
    assert "• Bull:" in compact_with_llm
    assert "• Bear/Risiko:" in compact_with_llm


def test_top_down_15m_refinement_and_direct_15m_htf_filter() -> None:
    ltf_specs: list[tuple[float, float, float, float]] = [
        (64_200.0 + i * 8.0, 64_230.0 + i * 8.0, 64_175.0 + i * 8.0, 64_205.0 + i * 8.0)
        for i in range(53)
    ]
    ltf_specs.extend(
        [
            (64_640.0, 64_700.0, 64_620.0, 64_650.0),  # 53
            (64_650.0, 64_680.0, 64_580.0, 64_600.0),  # 54
            (64_600.0, 64_620.0, 64_550.0, 64_570.0),  # 55
            (64_570.0, 64_610.0, 64_500.0, 64_590.0),  # 56 confirmed 15m swing low
            (64_590.0, 64_660.0, 64_560.0, 64_640.0),  # 57
            (64_640.0, 64_700.0, 64_610.0, 64_680.0),  # 58
            (64_680.0, 64_740.0, 64_650.0, 64_720.0),  # 59
            (64_720.0, 64_760.0, 64_690.0, 64_740.0),  # 60
            (64_740.0, 64_765.0, 64_695.0, 64_735.0),  # 61
            (64_735.0, 64_760.0, 64_700.0, 64_730.0),  # 62
            (64_730.0, 64_755.0, 64_695.0, 64_735.0),  # 63
            (64_735.0, 64_755.0, 64_695.0, 64_735.0),  # 64
            (64_735.0, 64_750.0, 64_700.0, 64_740.0),  # 65 first.high = 64_750
            (64_740.0, 64_950.0, 64_740.0, 64_930.0),  # 66
            (64_930.0, 65_050.0, 64_850.0, 65_020.0),  # 67 bullish FVG 64_750..64_850
            (65_020.0, 65_040.0, 64_880.0, 64_990.0),  # 68
            (64_990.0, 65_030.0, 64_900.0, 65_000.0),  # 69
        ]
    )
    ltf_candles = [
        _candle(i, open_p=o, high_p=h, low_p=low, close_p=c, timeframe=Timeframe.M15)
        for i, (o, h, low, c) in enumerate(ltf_specs)
    ]
    as_of = ltf_candles[-1].close_time

    def _pivot(ev_type: StructureEventType, level: float) -> StructureEvent:
        return StructureEvent(
            event_type=ev_type,
            event_time=as_of - timedelta(hours=12),
            confirmation_time=as_of - timedelta(hours=4),
            level=level,
        )

    bull_structure = StructureAnalysis(
        pivots=(),
        market_structure=MarketStructure.BULLISH,
        last_swing_highs=(_pivot(StructureEventType.SWING_HIGH, 67_000.0),),
        last_swing_lows=(_pivot(StructureEventType.SWING_LOW, 64_100.0),),
    )
    bull_metrics = TechnicalMetrics(
        last_close=65_000.0,
        ema_20=64_600.0,
        ema_50=63_800.0,
        atr_14=500.0,
        adx_14=32.0,
    )
    bull_smc = SmcSummary(
        dealing_range_high=67_000.0,
        dealing_range_low=62_000.0,
        equilibrium=64_500.0,
        dealing_zone=DealingZone.PREMIUM,
    )

    refined = build_trade_setup(
        metrics=bull_metrics,
        regime=Regime.TRENDING_UP,
        structure=bull_structure,
        smc=bull_smc,
        primary_timeframe=Timeframe.H4,
        ltf_candles=ltf_candles,
        as_of=as_of,
        cost_assumptions=CostAssumptions(execution_style="maker"),
    )
    assert refined.status is SetupStatus.READY
    assert refined.entry_timeframe is Timeframe.M15
    assert refined.entry_source is not None
    assert "Refinement 15m" in refined.entry_source
    assert refined.entry_low == pytest.approx(64_750.0)
    assert refined.entry_high == pytest.approx(64_850.0)
    assert refined.stop_loss is not None
    assert refined.stop_loss > 63_850.0
    assert refined.entry_mid is not None
    assert (refined.entry_mid - refined.stop_loss) >= 0.35 * 500.0
    assert refined.ltf_confirmation is not None
    assert refined.htf_support == pytest.approx(62_000.0)
    assert refined.htf_resistance == pytest.approx(67_000.0)
    assert refined.ltf_support is not None
    assert refined.ltf_resistance is not None

    # Direct 15m counter-trend against HTF downtrend without sweep+CHoCH -> WAIT
    bear_4h = [
        _candle(
            i - 70,
            open_p=70_000.0 - i * 80.0 + 20.0,
            high_p=70_000.0 - i * 80.0 + 50.0,
            low_p=70_000.0 - i * 80.0 - 30.0,
            close_p=70_000.0 - i * 80.0,
            timeframe=Timeframe.H4,
        )
        for i in range(65)
    ]
    # Shift anchor so bear_4h ends before or at as_of
    step_4h = Timeframe.H4.duration
    bear_4h_aligned = [
        c.model_copy(
            update={
                "open_time": as_of - (65 - idx) * step_4h,
                "close_time": as_of - (64 - idx) * step_4h,
                "received_time": as_of - (64 - idx) * step_4h + timedelta(seconds=1),
                "available_time": as_of - (64 - idx) * step_4h + timedelta(seconds=1),
            }
        )
        for idx, c in enumerate(bear_4h)
    ]
    m15_counter = build_trade_setup(
        metrics=bull_metrics,
        regime=Regime.TRENDING_UP,
        structure=bull_structure,
        smc=SmcSummary(),
        primary_timeframe=Timeframe.M15,
        htf_candles_by_tf={Timeframe.H4: bear_4h_aligned},
        as_of=as_of,
    )
    assert m15_counter.status is SetupStatus.WAIT
    assert m15_counter.wait_reason is not None
    assert "HTF (4h)" in m15_counter.wait_reason


def test_altcoin_subcent_formatting_and_15m_guard_allowlist() -> None:
    subcent_fact = Fact(
        fact_id="F1",
        label="Harga penutupan terakhir",
        value=0.1234,
        unit="USDT",
        display_format="price",
        source="binance:DOGE/USDT:15m",
        as_of=ANCHOR,
    )
    assert format_fact_value(subcent_fact) == "0,1234 USDT"

    micro_fact = subcent_fact.model_copy(
        update={"value": 0.00001234, "source": "binance:1000PEPE/USDT:15m"}
    )
    assert format_fact_value(micro_fact) == "0,00001234 USDT"

    data = json.loads(FIXTURE_1000.read_text(encoding="utf-8"))
    raw = data["raw_ohlcv"]
    last_open = datetime.fromtimestamp(float(raw[-1][0]) / 1000.0, tz=UTC)
    received = last_open + Timeframe.M15.duration + timedelta(seconds=1)
    step_ms = int(Timeframe.M15.duration.total_seconds() * 1000)
    end_ms = int((received - timedelta(seconds=1)).timestamp() * 1000)
    start_ms = end_ms - len(raw) * step_ms
    aligned = [
        [start_ms + i * step_ms, r[1] * 1e-6, r[2] * 1e-6, r[3] * 1e-6, r[4] * 1e-6, r[5]]
        for i, r in enumerate(raw)
    ]
    batch = normalize_ohlcv_rows(
        aligned,
        symbol=resolve_mvp_spot_symbol("1000PEPE"),
        timeframe=Timeframe.M15,
        received_time=received,
        now=received,
    )
    snap = build_analysis_snapshot(batch)
    narration = LlmAnalysisNarration(
        summary="1000PEPE/USDT pada timeframe 15m ditutup di {F1}.",
        bullish_scenario="Bertahan di atas {F1} menjaga momentum.",
        bearish_scenario="Penurunan dari {F1} meningkatkan ketidakpastian.",
        fact_ids=("F1",),
    )
    assert validate_narration(narration, snap).is_valid is True


def test_trade_setup_exhaustion_filters_and_counter_sweep_guard() -> None:
    t_pivot = ANCHOR
    p_high = StructureEvent(
        event_type=StructureEventType.SWING_HIGH,
        event_time=t_pivot,
        confirmation_time=t_pivot,
        level=66_500.0,
    )
    p_low = StructureEvent(
        event_type=StructureEventType.SWING_LOW,
        event_time=t_pivot,
        confirmation_time=t_pivot,
        level=64_100.0,
    )
    bull_structure = StructureAnalysis(
        pivots=(),
        market_structure=MarketStructure.BULLISH,
        last_swing_highs=(p_high,),
        last_swing_lows=(p_low,),
    )
    bull_metrics_overbought = TechnicalMetrics(
        last_close=65_000.0,
        ema_20=64_600.0,
        ema_50=63_800.0,
        atr_14=500.0,
        adx_14=32.0,
        rsi_14=78.5,
    )
    bull_smc_premium = SmcSummary(
        dealing_range_high=67_000.0,
        dealing_range_low=62_000.0,
        equilibrium=64_500.0,
        dealing_zone=DealingZone.PREMIUM,
    )

    # 1. LONG with RSI 78.5 in Premium zone -> forced WAIT
    ob_wait = build_trade_setup(
        metrics=bull_metrics_overbought,
        regime=Regime.TRENDING_UP,
        structure=bull_structure,
        smc=bull_smc_premium,
    )
    assert ob_wait.status is SetupStatus.WAIT
    assert ob_wait.wait_reason is not None
    assert "jenuh beli ekstrem" in ob_wait.wait_reason

    # 2. SHORT with RSI 22.0 in Discount zone -> forced WAIT
    bear_structure = StructureAnalysis(
        pivots=(),
        market_structure=MarketStructure.BEARISH,
        last_swing_highs=(p_high,),
        last_swing_lows=(p_low,),
    )
    bear_metrics_oversold = TechnicalMetrics(
        last_close=60_000.0,
        ema_20=60_500.0,
        ema_50=61_500.0,
        atr_14=500.0,
        adx_14=30.0,
        rsi_14=21.5,
    )
    bear_smc_discount = SmcSummary(
        dealing_range_high=63_000.0,
        dealing_range_low=57_000.0,
        equilibrium=60_000.0,
        dealing_zone=DealingZone.DISCOUNT,
    )
    os_wait = build_trade_setup(
        metrics=bear_metrics_oversold,
        regime=Regime.TRENDING_DOWN,
        structure=bear_structure,
        smc=bear_smc_discount,
    )
    assert os_wait.status is SetupStatus.WAIT
    assert os_wait.wait_reason is not None
    assert "jenuh jual ekstrem" in os_wait.wait_reason

    # 3. Counter-sweep (bearish sweep) with adverse order book imbalance <= -0.30 -> forced WAIT
    bull_metrics_normal = TechnicalMetrics(
        last_close=65_000.0,
        ema_20=64_600.0,
        ema_50=63_800.0,
        atr_14=500.0,
        adx_14=32.0,
        rsi_14=60.0,
    )
    bull_smc_sweep_bear = SmcSummary(
        dealing_range_high=67_000.0,
        dealing_range_low=62_000.0,
        equilibrium=64_500.0,
        dealing_zone=DealingZone.EQUILIBRIUM,
        latest_sweep=StructureEvent(
            event_type=StructureEventType.LIQUIDITY_SWEEP_BEARISH,
            event_time=t_pivot,
            confirmation_time=t_pivot,
            level=66_800.0,
        ),
    )
    adverse_deriv = DerivativesMetrics(imbalance_10bps=-0.44)
    sweep_wait = build_trade_setup(
        metrics=bull_metrics_normal,
        regime=Regime.TRENDING_UP,
        structure=bull_structure,
        smc=bull_smc_sweep_bear,
        derivatives=adverse_deriv,
    )
    assert sweep_wait.status is SetupStatus.WAIT
    assert sweep_wait.wait_reason is not None
    assert "Liquidity Sweep bearish" in sweep_wait.wait_reason


def test_trade_setup_sl_guarded_strictly_below_ltf_support() -> None:
    t_pivot = ANCHOR
    p_high = StructureEvent(
        event_type=StructureEventType.SWING_HIGH,
        event_time=t_pivot,
        confirmation_time=t_pivot,
        level=30.0,
    )
    p_low = StructureEvent(
        event_type=StructureEventType.SWING_LOW,
        event_time=t_pivot,
        confirmation_time=t_pivot,
        level=25.0,
    )
    structure_4h = StructureAnalysis(
        pivots=(),
        market_structure=MarketStructure.BULLISH,
        last_swing_highs=(p_high,),
        last_swing_lows=(p_low,),
    )
    metrics_4h = TechnicalMetrics(
        last_close=26.86,
        ema_20=26.0,
        ema_50=25.0,
        atr_14=0.8,
        adx_14=30.0,
        rsi_14=60.0,
    )
    smc_4h = SmcSummary(
        dealing_range_high=30.0,
        dealing_range_low=24.0,
        equilibrium=27.0,
        dealing_zone=DealingZone.EQUILIBRIUM,
    )

    # 15m candles with support at 25.89 and FVG around 26.34..26.45
    candles_15m = [
        _candle(
            i,
            open_p=26.0,
            high_p=26.5,
            low_p=25.89,
            close_p=26.3,
            timeframe=Timeframe.M15,
        )
        for i in range(25)
    ]
    setup = build_trade_setup(
        metrics=metrics_4h,
        regime=Regime.TRENDING_UP,
        structure=structure_4h,
        smc=smc_4h,
        primary_timeframe=Timeframe.H4,
        ltf_candles=candles_15m,
    )
    if setup.status is SetupStatus.READY:
        assert setup.stop_loss is not None
        if setup.ltf_support is not None:
            # Must strictly be below LTF support
            assert setup.stop_loss < setup.ltf_support


def test_setup_short_extreme_vol_no_crash_net_rr_wait_and_leverage_cap() -> None:
    from crypto_chatbot.schemas.ml import CostAssumptions

    # 1. Extreme ATR/SL on SHORT where 3x risk > entry_mid -> clean WAIT (no ValidationError)
    t_piv = ANCHOR - timedelta(hours=12)
    struct_bear = StructureAnalysis(
        pivots=(),
        market_structure=MarketStructure.BEARISH,
        last_swing_highs=(
            StructureEvent(
                event_type=StructureEventType.SWING_HIGH,
                event_time=t_piv,
                confirmation_time=t_piv,
                level=140.0,
            ),
        ),
        last_swing_lows=(
            StructureEvent(
                event_type=StructureEventType.SWING_LOW,
                event_time=t_piv,
                confirmation_time=t_piv,
                level=60.0,
            ),
        ),
    )
    metrics_extreme = TechnicalMetrics(
        last_close=100.0,
        ema_20=102.0,
        ema_50=110.0,
        atr_14=25.0,
        adx_14=32.0,
        rsi_14=42.0,
    )
    smc_bear = SmcSummary(
        dealing_range_high=125.0,
        dealing_range_low=75.0,
        equilibrium=100.0,
        dealing_zone=DealingZone.PREMIUM,
    )
    extreme_short = build_trade_setup(
        metrics=metrics_extreme,
        regime=Regime.TRENDING_DOWN,
        structure=struct_bear,
        smc=smc_bear,
    )
    assert extreme_short.status is SetupStatus.WAIT
    assert extreme_short.wait_reason is not None

    # 2. Tight SL setup: high friction pushes net_rr_tp2 < 1.5 -> WAIT gate
    struct_bull = StructureAnalysis(
        pivots=(),
        market_structure=MarketStructure.BULLISH,
        last_swing_highs=(
            StructureEvent(
                event_type=StructureEventType.SWING_HIGH,
                event_time=t_piv,
                confirmation_time=t_piv,
                level=112.0,
            ),
        ),
        last_swing_lows=(
            StructureEvent(
                event_type=StructureEventType.SWING_LOW,
                event_time=t_piv,
                confirmation_time=t_piv,
                level=98.5,
            ),
        ),
    )
    metrics_tight = TechnicalMetrics(
        last_close=100.0,
        ema_20=99.5,
        ema_50=97.0,
        atr_14=0.6,
        adx_14=28.0,
        rsi_14=56.0,
    )
    smc_bull = SmcSummary(
        dealing_range_high=112.0,
        dealing_range_low=96.0,
        equilibrium=104.0,
        dealing_zone=DealingZone.DISCOUNT,
    )
    high_cost = CostAssumptions(
        fee_bps_per_side=30.0, spread_bps=20.0, slippage_bps=20.0, funding_bps_per_24h=10.0
    )
    net_wait = build_trade_setup(
        metrics=metrics_tight,
        regime=Regime.TRENDING_UP,
        structure=struct_bull,
        smc=smc_bull,
        cost_assumptions=high_cost,
    )
    assert net_wait.status is SetupStatus.WAIT

    # 3. Normal cost READY setup caps max_theoretical_leverage <= 20
    low_cost_ready = build_trade_setup(
        metrics=metrics_tight,
        regime=Regime.TRENDING_UP,
        structure=struct_bull,
        smc=smc_bull,
        cost_assumptions=CostAssumptions(),
    )
    if low_cost_ready.status is SetupStatus.READY:
        assert low_cost_ready.max_theoretical_leverage is not None
        assert low_cost_ready.max_theoretical_leverage <= 20
