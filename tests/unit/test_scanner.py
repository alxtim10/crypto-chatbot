from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from crypto_chatbot.analysis.scanner import (
    render_market_scan_report,
    scan_market_setups,
)
from crypto_chatbot.schemas.analysis import (
    AnalysisSnapshot,
    Fact,
    FactorDirection,
    FactorItem,
    InputRange,
    MarketStructure,
    Regime,
    SetupBias,
    SetupStatus,
    StructureEvent,
    StructureEventType,
    TechnicalMetrics,
    TradeSetup,
)
from crypto_chatbot.schemas.market import (
    MULTI_STYLE_SCAN_TIMEFRAMES,
    MarketType,
    QualityStatus,
    Timeframe,
)

NOW = datetime(2026, 11, 23, 10, 0, tzinfo=UTC)
SHA = "b" * 64


def _make_scan_snapshot(
    symbol: str,
    *,
    timeframe: Timeframe = Timeframe.H4,
    status: SetupStatus,
    bias: SetupBias = SetupBias.LONG,
    rr_tp2: float = 2.0,
    wait_reason: str | None = None,
    quality: QualityStatus = QualityStatus.READY,
) -> AnalysisSnapshot:
    base = symbol.split("/")[0]
    fact = Fact(
        fact_id="F1",
        label="Close",
        value=100.0,
        unit="USDT",
        display_format="price",
        source=f"ccxt:{base}",
        as_of=NOW,
    )
    setup: TradeSetup | None = None
    if quality is not QualityStatus.INSUFFICIENT_DATA:
        if status is SetupStatus.READY:
            if bias is SetupBias.SHORT:
                setup = TradeSetup(
                    status=SetupStatus.READY,
                    bias=bias,
                    entry_low=101.0,
                    entry_high=102.0,
                    entry_mid=101.5,
                    stop_loss=103.5,
                    tp1=99.5,
                    tp2=101.5 - rr_tp2 * 2.0,
                    tp3=93.5,
                    rr_tp1=1.0,
                    rr_tp2=rr_tp2,
                    rr_tp3=4.0,
                    ltf_confirmation="15m CHoCH BEARISH | 5m struktur searah",
                )
            else:
                setup = TradeSetup(
                    status=SetupStatus.READY,
                    bias=bias,
                    entry_low=98.0,
                    entry_high=99.0,
                    entry_mid=98.5,
                    stop_loss=96.5,
                    tp1=100.5,
                    tp2=98.5 + rr_tp2 * 2.0,
                    tp3=106.5,
                    rr_tp1=1.0,
                    rr_tp2=rr_tp2,
                    rr_tp3=4.0,
                    ltf_confirmation="15m CHoCH BULLISH | 5m struktur searah",
                )
        else:
            setup = TradeSetup(
                status=SetupStatus.WAIT,
                bias=SetupBias.NEUTRAL,
                wait_reason=wait_reason or "RSI overbought di zona Premium.",
            )

    return AnalysisSnapshot(
        snapshot_id=f"snap-{base}",
        provider="ccxt",
        venue="gate",
        canonical_symbol=symbol,
        provider_symbol=f"GATE:{base}USDT.P",
        market_type=MarketType.PERPETUAL,
        timeframe=timeframe,
        as_of=NOW,
        input_range=(
            None
            if quality is QualityStatus.INSUFFICIENT_DATA
            else InputRange(
                first_open_time=NOW - 100 * timeframe.duration,
                last_open_time=NOW - timeframe.duration,
                candle_count=100,
                sha256=SHA,
            )
        ),
        calc_version="1.0.0",
        technical=(
            None
            if quality is QualityStatus.INSUFFICIENT_DATA
            else TechnicalMetrics(last_close=100.0)
        ),
        trade_setup=setup,
        facts=() if quality is QualityStatus.INSUFFICIENT_DATA else (fact,),
        regime=None if quality is QualityStatus.INSUFFICIENT_DATA else Regime.TRENDING_UP,
        market_structure=(
            None if quality is QualityStatus.INSUFFICIENT_DATA else MarketStructure.BULLISH
        ),
        events=(
            ()
            if quality is QualityStatus.INSUFFICIENT_DATA
            else (
                StructureEvent(
                    event_type=StructureEventType.SWING_LOW,
                    event_time=NOW - timedelta(hours=12),
                    confirmation_time=NOW - timedelta(hours=4),
                    level=95.0,
                ),
            )
        ),
        factors=(
            ()
            if quality is QualityStatus.INSUFFICIENT_DATA
            else (
                FactorItem(
                    rule_id="rule1",
                    description="Above EMA",
                    direction=FactorDirection.SUPPORTS_BULLISH,
                    fact_ids=("F1",),
                ),
            )
        ),
        quality_status=quality,
    )


class _StubScanAnalysisService:
    def __init__(
        self,
        responses: dict[
            str,
            AnalysisSnapshot | Exception | dict[Timeframe, AnalysisSnapshot | Exception],
        ],
    ) -> None:
        self._responses = responses

    async def analyze(
        self,
        symbol: str,
        timeframe: Timeframe,
        *,
        include_mtf: bool = True,
        include_derivatives: bool = True,
        persist: bool = True,
    ) -> AnalysisSnapshot:
        del include_mtf, include_derivatives, persist
        res = self._responses[symbol]
        if isinstance(res, dict):
            res = res[timeframe]
        if isinstance(res, Exception):
            raise res
        return res


@pytest.mark.asyncio
async def test_scan_market_setups_sorts_ready_by_rr_and_handles_failures() -> None:
    svc: Any = _StubScanAnalysisService(
        {
            "BTC/USDT:USDT": _make_scan_snapshot(
                "BTC/USDT:USDT", status=SetupStatus.READY, rr_tp2=1.8
            ),
            "ETH/USDT:USDT": _make_scan_snapshot(
                "ETH/USDT:USDT", status=SetupStatus.WAIT, wait_reason="Rezim ranging."
            ),
            "SOL/USDT:USDT": _make_scan_snapshot(
                "SOL/USDT:USDT", status=SetupStatus.READY, rr_tp2=2.7
            ),
            "DOGE/USDT:USDT": RuntimeError("exchange timeout"),
        }
    )

    report = await scan_market_setups(
        svc,
        ["BTC", "ETH", "SOL", "DOGE", "btc"],
        timeframe=Timeframe.H4,
        max_concurrency=2,
    )

    assert report.scanned_count == 4
    assert [item.symbol for item in report.ready_setups] == [
        "SOL/USDT:USDT",
        "BTC/USDT:USDT",
    ]
    assert len(report.wait_setups) == 1
    assert report.wait_setups[0].symbol == "ETH/USDT:USDT"
    assert report.failed_symbols == ("DOGE/USDT:USDT",)

    rendered = render_market_scan_report(report, html_mode=True)
    assert "Setup READY: 2" in rendered
    assert "SOL/USDT:USDT" in rendered
    assert "[SWING (4h)]" in rendered
    assert "Gaya Setup:" in rendered
    assert "1:2,70" in rendered
    assert "5m struktur searah" in rendered


@pytest.mark.asyncio
async def test_scan_market_setups_multi_style_labels_scalp_intraday_swing() -> None:
    svc: Any = _StubScanAnalysisService(
        {
            "SOL/USDT:USDT": {
                Timeframe.M15: _make_scan_snapshot(
                    "SOL/USDT:USDT",
                    timeframe=Timeframe.M15,
                    status=SetupStatus.READY,
                    rr_tp2=2.8,
                ),
                Timeframe.H1: _make_scan_snapshot(
                    "SOL/USDT:USDT",
                    timeframe=Timeframe.H1,
                    status=SetupStatus.WAIT,
                    wait_reason="Belum pullback.",
                ),
                Timeframe.H4: _make_scan_snapshot(
                    "SOL/USDT:USDT",
                    timeframe=Timeframe.H4,
                    status=SetupStatus.WAIT,
                    wait_reason="Rezim ranging.",
                ),
            },
            "BTC/USDT:USDT": {
                Timeframe.M15: _make_scan_snapshot(
                    "BTC/USDT:USDT",
                    timeframe=Timeframe.M15,
                    status=SetupStatus.WAIT,
                    wait_reason="Belum konfirmasi.",
                ),
                Timeframe.H1: _make_scan_snapshot(
                    "BTC/USDT:USDT",
                    timeframe=Timeframe.H1,
                    status=SetupStatus.READY,
                    bias=SetupBias.SHORT,
                    rr_tp2=2.1,
                ),
                Timeframe.H4: _make_scan_snapshot(
                    "BTC/USDT:USDT",
                    timeframe=Timeframe.H4,
                    status=SetupStatus.WAIT,
                    wait_reason="Rezim ranging.",
                ),
            },
        }
    )

    report = await scan_market_setups(
        svc,
        ["BTC", "SOL"],
        timeframes=MULTI_STYLE_SCAN_TIMEFRAMES,
    )

    assert report.scanned_timeframes == MULTI_STYLE_SCAN_TIMEFRAMES
    assert len(report.ready_setups) == 2
    assert len(report.wait_setups) == 4

    rendered = render_market_scan_report(report, html_mode=True)
    assert "Multi-Style" in rendered
    assert "[SCALP (15m)]" in rendered
    assert "SCALP / Quick Intraday (15m → 5m)" in rendered
    assert "[INTRADAY (1h)]" in rendered
    assert "INTRADAY / Hourly (1h → 15m → 5m)" in rendered


@pytest.mark.asyncio
async def test_scan_market_setups_reports_honest_empty_state_when_all_wait() -> None:
    svc: Any = _StubScanAnalysisService(
        {
            "BTC/USDT:USDT": _make_scan_snapshot(
                "BTC/USDT:USDT", status=SetupStatus.WAIT, wait_reason="Rezim ranging."
            ),
            "ETH/USDT:USDT": _make_scan_snapshot(
                "ETH/USDT:USDT", status=SetupStatus.WAIT, wait_reason="Rezim ranging."
            ),
        }
    )

    report = await scan_market_setups(svc, ["BTC", "ETH"])
    assert len(report.ready_setups) == 0
    assert len(report.wait_setups) == 2

    rendered = render_market_scan_report(report, html_mode=False)
    assert "Belum ada koin dari daftar pantau yang berstatus READY" in rendered
    assert "(2 koin) Rezim ranging." in rendered
    assert "bukan saran transaksi" in rendered.lower()


@pytest.mark.asyncio
async def test_analysis_service_ohlcv_ttl_cache_deduplicates_calls() -> None:
    from crypto_chatbot.analysis.service import AnalysisService
    from crypto_chatbot.market_data.base import OhlcvBatch
    from crypto_chatbot.schemas.market import MarketCandle, ProviderMetadata, QualityFlag

    t0 = NOW - timedelta(hours=8)
    candle = MarketCandle(
        provider="ccxt",
        venue="binance",
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol="BINANCE:BTCUSDT.P",
        market_type=MarketType.PERPETUAL,
        timeframe=Timeframe.H4,
        open_time=t0,
        close_time=t0 + timedelta(hours=4),
        received_time=NOW,
        available_time=NOW,
        open=90_000.0,
        high=90_500.0,
        low=89_500.0,
        close=90_200.0,
        base_volume=100.0,
        base_unit="BTC",
        quote_currency="USDT",
        is_closed=True,
    )
    meta = ProviderMetadata(
        provider="ccxt",
        venue="binance",
        market_type=MarketType.PERPETUAL,
        base_unit="BTC",
        quote_currency="USDT",
    )

    class _CountingProvider:
        def __init__(self) -> None:
            self.calls = 0
            self.fail = False

        async def get_closed_ohlcv(
            self, symbol: str, timeframe: Timeframe, limit: int = 100
        ) -> OhlcvBatch:
            del limit
            self.calls += 1
            if self.fail:
                return OhlcvBatch(
                    metadata=meta,
                    canonical_symbol=symbol,
                    provider_symbol="BINANCE:BTCUSDT.P",
                    timeframe=timeframe,
                    received_time=NOW,
                    candles=(),
                    quality_flags=(QualityFlag.PROVIDER_ERROR,),
                    error_message="err",
                )
            return OhlcvBatch(
                metadata=meta,
                canonical_symbol=symbol,
                provider_symbol="BINANCE:BTCUSDT.P",
                timeframe=timeframe,
                received_time=NOW,
                candles=(candle,),
                quality_flags=(),
            )

        async def close(self) -> None:
            return None

    prov = _CountingProvider()
    svc = AnalysisService(provider=prov, cache_ttl_seconds=45.0)
    await svc.get_closed_ohlcv("BTC/USDT:USDT", Timeframe.H4, limit=100)
    await svc.get_closed_ohlcv("BTC/USDT:USDT", Timeframe.H4, limit=100)
    assert prov.calls == 1

    # Error responses are never cached
    prov.fail = True
    await svc.get_closed_ohlcv("ETH/USDT:USDT", Timeframe.H4, limit=100)
    await svc.get_closed_ohlcv("ETH/USDT:USDT", Timeframe.H4, limit=100)
    assert prov.calls == 3
