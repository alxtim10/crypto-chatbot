from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

from crypto_chatbot.analysis.pine_export import _format_pine_price, generate_pine_script
from crypto_chatbot.schemas.analysis import (
    AnalysisSnapshot,
    DealingZone,
    Fact,
    FactorDirection,
    FactorItem,
    InputRange,
    MarketStructure,
    PriceZone,
    Regime,
    SetupBias,
    SetupStatus,
    SmcSummary,
    StructureEvent,
    StructureEventType,
    TechnicalMetrics,
    TradeSetup,
)
from crypto_chatbot.schemas.market import MarketType, QualityStatus, Timeframe

NOW = datetime(2026, 11, 23, 8, 0, tzinfo=UTC)
SHA256_SAMPLE = "a" * 64


def _make_snapshot(
    *,
    symbol: str = "BTC/USDT:USDT",
    provider_symbol: str = "GATE:BTCUSDT.P",
    setup: TradeSetup | None = None,
    smc: SmcSummary | None = None,
    last_close: float = 91000.0,
) -> AnalysisSnapshot:
    fact = Fact(
        fact_id="F1",
        label="Last close",
        value=last_close,
        unit="USDT",
        display_format="price",
        source="ccxt:gate",
        as_of=NOW,
    )
    return AnalysisSnapshot(
        snapshot_id="snap-pine-001",
        provider="ccxt",
        venue="gate",
        canonical_symbol=symbol,
        provider_symbol=provider_symbol,
        market_type=MarketType.PERPETUAL,
        timeframe=Timeframe.H4,
        as_of=NOW,
        input_range=InputRange(
            first_open_time=NOW - timedelta(hours=400),
            last_open_time=NOW - timedelta(hours=4),
            candle_count=100,
            sha256=SHA256_SAMPLE,
        ),
        calc_version="1.0.0",
        technical=TechnicalMetrics(last_close=last_close),
        smc=smc,
        trade_setup=setup,
        facts=(fact,),
        regime=Regime.TRENDING_UP,
        market_structure=MarketStructure.BULLISH,
        events=(
            StructureEvent(
                event_type=StructureEventType.SWING_LOW,
                event_time=NOW - timedelta(hours=12),
                confirmation_time=NOW - timedelta(hours=4),
                level=last_close * 0.95,
            ),
        ),
        factors=(
            FactorItem(
                rule_id="sample",
                description="Sample fact",
                direction=FactorDirection.SUPPORTS_BULLISH,
                fact_ids=("F1",),
            ),
        ),
        quality_status=QualityStatus.READY,
    )


def test_format_pine_price_never_uses_scientific_notation() -> None:
    assert _format_pine_price(0.00001234) == "0.00001234"
    assert "e-" not in _format_pine_price(0.00000005).lower()
    assert _format_pine_price(26.86) == "26.86"
    assert _format_pine_price(90000.0) == "90000.00"


def test_generate_pine_script_for_ready_long_and_short_setups() -> None:
    long_setup = TradeSetup(
        status=SetupStatus.READY,
        bias=SetupBias.LONG,
        entry_low=89500.0,
        entry_high=90000.0,
        entry_mid=89750.0,
        stop_loss=88500.0,
        tp1=91000.0,
        tp2=92250.0,
        tp3=93500.0,
        rr_tp1=1.0,
        rr_tp2=2.0,
        rr_tp3=3.0,
        htf_support=88000.0,
        htf_resistance=94000.0,
        ltf_support=89000.0,
        ltf_resistance=91500.0,
    )
    smc = SmcSummary(
        dealing_range_high=94000.0,
        dealing_range_low=88000.0,
        equilibrium=91000.0,
        dealing_zone=DealingZone.DISCOUNT,
        active_bullish_fvg=PriceZone(
            zone_type="bullish_fvg",
            zone_low=89400.0,
            zone_high=89900.0,
            event_time=NOW - timedelta(hours=8),
            confirmation_time=NOW - timedelta(hours=4),
        ),
    )
    snap_long = _make_snapshot(setup=long_setup, smc=smc)
    pine_long = generate_pine_script(snap_long)

    assert pine_long.startswith("//@version=6\n")
    assert "if barstate.islast" in pine_long
    assert "ENTRY LONG (89500.00 - 90000.00)" in pine_long
    assert "SL (88500.00)" in pine_long
    assert "TP1 (91000.00 | 1:1.00)" in pine_long
    assert "TP2 (92250.00 | 1:2.00)" in pine_long
    assert "TP3 (93500.00 | 1:3.00)" in pine_long
    assert "Bull FVG (89400.00-89900.00)" in pine_long
    assert re.search(r"\de[-+]\d", pine_long) is None

    short_setup = TradeSetup(
        status=SetupStatus.READY,
        bias=SetupBias.SHORT,
        entry_low=0.00001200,
        entry_high=0.00001250,
        entry_mid=0.00001225,
        stop_loss=0.00001300,
        tp1=0.00001150,
        tp2=0.00001075,
        tp3=0.00001000,
        rr_tp1=1.0,
        rr_tp2=2.0,
        rr_tp3=3.0,
    )
    snap_short = _make_snapshot(
        symbol="PEPE/USDT:USDT",
        provider_symbol="GATE:PEPEUSDT.P",
        setup=short_setup,
        last_close=0.00001225,
    )
    pine_short = generate_pine_script(snap_short)
    assert "ENTRY SHORT (0.000012 - 0.0000125)" in pine_short
    assert "SL (0.000013)" in pine_short
    assert re.search(r"\de[-+]\d", pine_short) is None


def test_generate_pine_script_for_wait_setup_renders_sr_and_smc() -> None:
    wait_setup = TradeSetup(
        status=SetupStatus.WAIT,
        bias=SetupBias.NEUTRAL,
        wait_reason="Menunggu konfirmasi.",
        htf_support=88000.0,
        htf_resistance=94000.0,
    )
    snap = _make_snapshot(setup=wait_setup)
    pine = generate_pine_script(snap)

    assert "// Setup    : WAIT | Bias: NEUTRAL" in pine
    assert "HTF Sup (88000.00)" in pine
    assert "HTF Res (94000.00)" in pine
    assert "ENTRY" not in pine
