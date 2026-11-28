from __future__ import annotations

import json
import logging
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from crypto_chatbot.analysis import momentum_screener as screener_mod
from crypto_chatbot.analysis.factors import DEALING_ZONE_FACTOR_DESCRIPTIONS, build_factors
from crypto_chatbot.analysis.momentum_screener import (
    MOMENTUM_PATTERN_LABELS,
    MomentumPattern,
    render_momentum_screener_report,
    screen_momentum_movers,
)
from crypto_chatbot.analysis.scanner import (
    render_market_scan_report,
    scan_market_setups,
)
from crypto_chatbot.analysis.service import MTF_REGIME_LABELS
from crypto_chatbot.analysis.structure import StructureAnalysis
from crypto_chatbot.bot.templates import (
    DEALING_ZONE_LABELS,
    MTF_LABELS,
    REGIME_LABELS,
    SENTIMENT_LABELS,
    SETUP_BIAS_LABELS,
    SETUP_STATUS_LABELS,
    STRUCTURE_LABELS,
    format_enum_label,
    render_compact_snapshot,
    render_deterministic_snapshot,
)
from crypto_chatbot.llm.prompts import build_narration_user_payload
from crypto_chatbot.market_data.base import FuturesTickerBatch, FuturesTickerSnapshot, OhlcvBatch
from crypto_chatbot.schemas.analysis import (
    AnalysisSnapshot,
    DealingZone,
    Fact,
    InputRange,
    MarketStructure,
    MtfRsiAlignment,
    Regime,
    SetupBias,
    SetupStatus,
    SmcSummary,
    StructureEvent,
    StructureEventType,
    TechnicalMetrics,
    TimeframeOverview,
    TradeSetup,
)
from crypto_chatbot.schemas.market import (
    PRIMARY_TIMEFRAMES,
    TRADING_STYLE_LABELS,
    TRADING_STYLE_SHORT_LABELS,
    MarketCandle,
    MarketType,
    ProviderMetadata,
    QualityStatus,
    Timeframe,
)
from crypto_chatbot.schemas.news import SentimentLabel

NOW = datetime(2026, 11, 28, 12, 0, tzinfo=UTC)
SHA = "c" * 64


def test_all_renderer_label_dicts_cover_every_enum_member() -> None:
    """Exhaustiveness guard: adding a new Enum member without a label fails CI immediately."""
    enum_mappings: tuple[tuple[str, type[Any], dict[Any, str]], ...] = (
        ("REGIME_LABELS", Regime, REGIME_LABELS),
        ("MTF_REGIME_LABELS", Regime, MTF_REGIME_LABELS),
        ("STRUCTURE_LABELS", MarketStructure, STRUCTURE_LABELS),
        ("MTF_LABELS", MtfRsiAlignment, MTF_LABELS),
        ("DEALING_ZONE_LABELS", DealingZone, DEALING_ZONE_LABELS),
        (
            "DEALING_ZONE_FACTOR_DESCRIPTIONS",
            DealingZone,
            DEALING_ZONE_FACTOR_DESCRIPTIONS,
        ),
        ("SENTIMENT_LABELS", SentimentLabel, SENTIMENT_LABELS),
        ("SETUP_STATUS_LABELS", SetupStatus, SETUP_STATUS_LABELS),
        ("SETUP_BIAS_LABELS", SetupBias, SETUP_BIAS_LABELS),
        ("MOMENTUM_PATTERN_LABELS", MomentumPattern, MOMENTUM_PATTERN_LABELS),
    )
    for dict_name, enum_cls, mapping in enum_mappings:
        missing = [member for member in enum_cls if member not in mapping]
        assert not missing, f"{dict_name} missing {enum_cls.__name__} members: {missing}"
        for member in enum_cls:
            assert isinstance(mapping[member], str)
            assert mapping[member].strip() != ""

    for tf in (*PRIMARY_TIMEFRAMES, Timeframe.M5):
        assert tf in TRADING_STYLE_LABELS
        assert tf in TRADING_STYLE_SHORT_LABELS


def test_format_enum_label_never_crashes_and_logs_warning(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with caplog.at_level(logging.WARNING):
        fallback = format_enum_label({}, DealingZone.BELOW_RANGE)
    assert fallback == DealingZone.BELOW_RANGE.value
    assert "BELOW_RANGE" in caplog.text

    # Empty out DEALING_ZONE_LABELS at runtime to prove renderers still do not crash.
    monkeypatch.setattr("crypto_chatbot.bot.templates.DEALING_ZONE_LABELS", {})
    snap = _make_dealing_zone_snapshot(DealingZone.ABOVE_RANGE)
    caplog.clear()
    with caplog.at_level(logging.WARNING):
        compact = render_compact_snapshot(snap, html_mode=False)
        det = render_deterministic_snapshot(snap, html_mode=False)
    assert "above_range" in compact
    assert "above_range" in det
    assert "ABOVE_RANGE" in caplog.text


def _make_dealing_zone_snapshot(
    zone: DealingZone,
    *,
    mtf_zone: DealingZone | None = None,
) -> AnalysisSnapshot:
    facts = (
        Fact(
            fact_id="F1",
            label="Harga penutupan terakhir",
            value=99_000.0,
            unit="USDT",
            display_format="price",
            source="gate:BTC/USDT:USDT:4h",
            as_of=NOW,
        ),
        Fact(
            fact_id="F2",
            label="Equilibrium dealing range",
            value=95_000.0,
            unit="USDT",
            display_format="price",
            source="gate:BTC/USDT:USDT:4h",
            as_of=NOW,
        ),
        Fact(
            fact_id="F3",
            label="RSI 14",
            value=54.0,
            unit="index",
            display_format="decimal_2",
            source="gate:BTC/USDT:USDT:4h",
            as_of=NOW,
        ),
    )
    smc = SmcSummary(
        dealing_range_high=97_000.0,
        dealing_range_low=93_000.0,
        equilibrium=95_000.0,
        dealing_zone=zone,
    )
    mtf_overview = (
        TimeframeOverview(
            timeframe=Timeframe.D1,
            trend_label="Tren Naik",
            market_structure=MarketStructure.BULLISH,
            dealing_zone=mtf_zone or zone,
        ),
        TimeframeOverview(
            timeframe=Timeframe.H4,
            trend_label="Tren Naik",
            market_structure=MarketStructure.BULLISH,
            dealing_zone=zone,
        ),
    )
    swing_high_ev = StructureEvent(
        event_type=StructureEventType.SWING_HIGH,
        level=97_000.0,
        event_time=NOW - timedelta(hours=8),
        confirmation_time=NOW,
    )
    factors = build_factors(
        metrics=TechnicalMetrics(
            last_close=99_000.0,
            rsi_14=54.0,
            mtf_rsi_alignment=MtfRsiAlignment.ALIGNED_BULLISH,
        ),
        regime=Regime.TRENDING_UP,
        structure=StructureAnalysis(
            market_structure=MarketStructure.BULLISH,
            pivots=(swing_high_ev,),
            last_swing_highs=(swing_high_ev,),
            last_swing_lows=(),
        ),
        failed_breakouts=(),
        recent_cutoff_time=NOW - timedelta(hours=24),
        fact_ids_by_key={
            "last_close": "F1",
            "smc_equilibrium": "F2",
            "rsi_14": "F3",
        },
        smc=smc,
    )
    setup = TradeSetup(
        status=SetupStatus.READY,
        bias=SetupBias.LONG,
        entry_low=94_500.0,
        entry_high=95_000.0,
        entry_mid=94_750.0,
        stop_loss=92_500.0,
        tp1=97_000.0,
        tp2=99_250.0,
        tp3=101_500.0,
        rr_tp1=1.0,
        rr_tp2=2.0,
        rr_tp3=3.0,
        stop_loss_distance_pct=2.37,
        max_theoretical_leverage=20,
        ltf_confirmation="15m BOS BULLISH",
    )
    return AnalysisSnapshot(
        snapshot_id="snap-test-dz",
        provider="gate_rest",
        venue="gate",
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol="GATE:BTCUSDT.P",
        market_type=MarketType.PERPETUAL,
        timeframe=Timeframe.H4,
        as_of=NOW,
        input_range=InputRange(
            first_open_time=NOW - timedelta(hours=40),
            last_open_time=NOW - timedelta(hours=4),
            candle_count=10,
            sha256=SHA,
        ),
        calc_version="1.7.0",
        technical=TechnicalMetrics(
            last_close=99_000.0,
            rsi_14=54.0,
            mtf_rsi_alignment=MtfRsiAlignment.ALIGNED_BULLISH,
        ),
        smc=smc,
        mtf_overview=mtf_overview,
        trade_setup=setup,
        facts=facts,
        regime=Regime.TRENDING_UP,
        market_structure=MarketStructure.BULLISH,
        events=(swing_high_ev,),
        factors=factors,
        quality_status=QualityStatus.READY,
    )


@pytest.mark.parametrize(
    ("zone", "expected_label", "expected_factor_fragment"),
    [
        (DealingZone.BELOW_RANGE, "Below Range", "di bawah rentang swing"),
        (DealingZone.ABOVE_RANGE, "Above Range", "di atas rentang swing"),
    ],
)
def test_below_range_and_above_range_render_across_all_paths(
    zone: DealingZone,
    expected_label: str,
    expected_factor_fragment: str,
) -> None:
    other = DealingZone.ABOVE_RANGE if zone is DealingZone.BELOW_RANGE else DealingZone.BELOW_RANGE
    snap = _make_dealing_zone_snapshot(zone, mtf_zone=other)

    compact = render_compact_snapshot(snap, html_mode=False)
    assert f"Range: {expected_label}" in compact
    assert f"• Posisi Dealing Range: {expected_label}" in compact
    assert f"Range: {DEALING_ZONE_LABELS[other]}" in compact

    det = render_deterministic_snapshot(snap, html_mode=False)
    assert f"- Posisi Dealing Range: {expected_label}" in det
    assert expected_factor_fragment in det

    prompt = build_narration_user_payload(snap)
    payload_str = prompt.split("Validated AnalysisSnapshot facts (source of truth):\n", 1)[1].split(
        "\n\n", 1
    )[0]
    payload = json.loads(payload_str)
    assert payload["smc_provenance"]["dealing_zone"] == zone.value
    assert payload["smc_provenance"]["dealing_zone_label"] == expected_label


@pytest.mark.asyncio
async def test_scanner_and_screener_isolate_per_symbol_errors_and_report_summary() -> None:
    snap_ok = _make_dealing_zone_snapshot(DealingZone.BELOW_RANGE)

    class _ScanSvc:
        async def analyze(self, sym: str, tf: Timeframe, **kw: Any) -> AnalysisSnapshot:
            del tf, kw
            if sym.startswith("ETH"):
                raise ValueError("simulated corrupt symbol failure")
            return snap_ok.model_copy(update={"canonical_symbol": sym})

    scan_rep = await scan_market_setups(_ScanSvc(), ["BTC", "ETH"])  # type: ignore[arg-type]
    assert len(scan_rep.ready_setups) == 1
    assert scan_rep.ready_setups[0].dealing_zone is DealingZone.BELOW_RANGE
    assert scan_rep.failed_symbols == ("ETH/USDT:USDT",)
    assert "ERROR" in scan_rep.symbol_errors["ETH/USDT:USDT"]
    scan_txt = render_market_scan_report(scan_rep, html_mode=False)
    assert "Dealing Range: Below Range" in scan_txt
    assert "Gagal/ERROR (1): ETH/USDT:USDT [ERROR: simulated corrupt symbol failure]" in scan_txt

    # Screener: verify mid-evaluation exception in evaluate_15m_momentum_setup does not crash run!
    tickers = FuturesTickerBatch(
        venue="gate",
        received_time=NOW,
        tickers=(
            FuturesTickerSnapshot(
                canonical_symbol="BTC/USDT:USDT",
                provider_symbol="GATE:BTCUSDT.P",
                venue="gate",
                last_price=90_000.0,
                change_24h_pct=6.5,
                quote_volume_24h_usd=50_000_000.0,
                observed_time=NOW,
            ),
            FuturesTickerSnapshot(
                canonical_symbol="BAD/USDT:USDT",
                provider_symbol="GATE:BADUSDT.P",
                venue="gate",
                last_price=10.0,
                change_24h_pct=8.0,
                quote_volume_24h_usd=20_000_000.0,
                observed_time=NOW,
            ),
        ),
    )

    sample_bars = tuple(
        MarketCandle(
            provider="gate_rest",
            venue="gate",
            canonical_symbol="BTC/USDT:USDT",
            provider_symbol="GATE:BTCUSDT.P",
            market_type=MarketType.PERPETUAL,
            timeframe=Timeframe.M15,
            open_time=NOW - timedelta(minutes=15 * (25 - i)),
            close_time=NOW - timedelta(minutes=15 * (24 - i)),
            received_time=NOW,
            available_time=NOW - timedelta(minutes=15 * (24 - i)),
            open=90_000.0,
            high=90_100.0,
            low=89_900.0,
            close=90_050.0,
            base_volume=10.0,
            base_unit="BTC",
            quote_currency="USDT",
            is_closed=True,
        )
        for i in range(25)
    )

    class _ScrSvc:
        async def get_futures_tickers(self) -> FuturesTickerBatch:
            return tickers

        async def get_closed_ohlcv(
            self, symbol: str, timeframe: Timeframe, limit: int = 200
        ) -> OhlcvBatch:
            del timeframe, limit
            if symbol == "BAD/USDT:USDT":
                raise RuntimeError("corrupt candle stream")
            return OhlcvBatch(
                metadata=ProviderMetadata(
                    provider="gate_rest",
                    venue="gate",
                    market_type=MarketType.PERPETUAL,
                    base_unit="BTC",
                    quote_currency="USDT",
                ),
                canonical_symbol=symbol,
                provider_symbol="GATE:BTCUSDT.P",
                timeframe=Timeframe.M15,
                received_time=NOW,
                candles=sample_bars,
            )

    scr_rep = await screen_momentum_movers(_ScrSvc())  # type: ignore[arg-type]
    assert scr_rep.failed_symbols == ("BAD/USDT:USDT",)
    assert "ERROR" in scr_rep.symbol_errors["BAD/USDT:USDT"]
    assert len(scr_rep.watching_items) == 1
    assert scr_rep.watching_items[0].symbol == "BTC/USDT:USDT"
    scr_txt = render_momentum_screener_report(scr_rep, html_mode=False)
    assert "Gagal/ERROR: 1" in scr_txt
    assert "BAD/USDT:USDT [ERROR: corrupt candle stream]" in scr_txt
    assert screener_mod.SCREENER_DISCLAIMER in scr_txt
