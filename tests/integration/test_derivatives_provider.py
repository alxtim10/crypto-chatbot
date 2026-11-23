from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ccxt
import pytest

from crypto_chatbot.analysis.service import AnalysisService
from crypto_chatbot.market_data.ccxt_provider import CcxtMarketDataProvider
from crypto_chatbot.market_data.derivatives_provider import CcxtDerivativesLiquidityProvider
from crypto_chatbot.market_data.symbols import SymbolMapping
from crypto_chatbot.schemas.derivatives import OrderBookDiff
from crypto_chatbot.schemas.market import MarketType, QualityFlag, QualityStatus, Timeframe

CANDLE_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"
DERIV_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "binance_btc_derivatives_sample.json"
)


class FakeSpotClient:
    def __init__(
        self,
        *,
        order_books: list[dict[str, Any] | BaseException | float] | None = None,
        anchor_close: datetime = datetime(2026, 11, 21, 12, 0, tzinfo=UTC),
        client_id: str = "binance",
    ) -> None:
        self.id = client_id
        self._order_books = list(order_books or [])
        self._raw_candles = json.loads(CANDLE_FIXTURE.read_text(encoding="utf-8"))["raw_ohlcv"]
        self._anchor_close = anchor_close
        self.closed = False

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "4h",
        since: int | None = None,
        limit: int | None = None,
    ) -> list[list[Any]]:
        del symbol, timeframe, since
        step_ms = 14_400_000
        count = min(len(self._raw_candles), max(1, (limit or 1000) - 1))
        end_ms = int(self._anchor_close.timestamp() * 1000)
        start_ms = end_ms - count * step_ms
        return [
            [start_ms + i * step_ms, r[1], r[2], r[3], r[4], r[5]]
            for i, r in enumerate(self._raw_candles[-count:])
        ]

    async def fetch_order_book(
        self,
        symbol: str,
        limit: int | None = None,
    ) -> dict[str, Any]:
        del symbol, limit
        outcome = self._order_books.pop(0) if len(self._order_books) > 1 else self._order_books[0]
        if isinstance(outcome, float):
            await asyncio.sleep(outcome)
            return {}
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def close(self) -> None:
        self.closed = True


class FakeFuturesClient:
    def __init__(
        self,
        fixture: Mapping[str, Any],
        *,
        client_id: str = "binanceusdm",
        funding_outcome: BaseException | None = None,
        oi_outcome: BaseException | None = None,
    ) -> None:
        self.id = client_id
        self._fixture = fixture
        self._funding_outcome = funding_outcome
        self._oi_outcome = oi_outcome
        self.markets = {
            "BTC/USDT:USDT": {"contractSize": 1.0, "inverse": False},
        }
        self.closed = False

    async def fetch_funding_rate(self, symbol: str) -> dict[str, Any]:
        del symbol
        if self._funding_outcome is not None:
            raise self._funding_outcome
        return dict(self._fixture["funding_rate"])

    async def fetch_funding_intervals(
        self,
        symbols: Sequence[str] | None = None,
    ) -> dict[str, Any]:
        del symbols
        return dict(self._fixture["funding_intervals"])

    async def fetch_open_interest(self, symbol: str) -> dict[str, Any]:
        del symbol
        if self._oi_outcome is not None:
            raise self._oi_outcome
        return dict(self._fixture["open_interest"]["latest"])

    async def fetch_open_interest_history(
        self,
        symbol: str,
        timeframe: str = "4h",
        since: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        del symbol, timeframe, since, limit
        if self._oi_outcome is not None:
            raise self._oi_outcome
        return [dict(item) for item in self._fixture["open_interest"]["history"]]

    async def fetch_order_book(
        self,
        symbol: str,
        limit: int | None = None,
    ) -> dict[str, Any]:
        del symbol, limit
        return dict(self._fixture["order_book"])

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_ccxt_derivatives_provider_end_to_end_and_resync() -> None:
    fixture = json.loads(DERIV_FIXTURE.read_text(encoding="utf-8"))
    received = datetime.fromisoformat(fixture["received_time"])

    initial_book = dict(fixture["order_book"])
    resynced_book = {
        **initial_book,
        "nonce": 500200,
        "bids": [[65577.0, 4.0], *initial_book["bids"][1:]],
    }
    spot_client = FakeSpotClient(order_books=[initial_book, resynced_book])
    futures_client = FakeFuturesClient(fixture)

    deriv_provider = CcxtDerivativesLiquidityProvider(
        spot_client=spot_client,
        futures_client=futures_client,
        retry_initial_seconds=0.0,
        retry_max_seconds=0.0,
        clock=lambda: received,
    )

    funding = await deriv_provider.get_funding_rate("BTC")
    assert funding.observation is not None
    assert funding.observation.canonical_symbol == "BTC/USDT:USDT"
    assert funding.observation.rate_per_8h == pytest.approx(0.000125)

    oi = await deriv_provider.get_open_interest("BTC", Timeframe.H4)
    assert oi.latest is not None
    assert oi.latest.open_interest_base == pytest.approx(82400.0)
    assert len(oi.history) == 2

    # First diff sequence gap (500150 > 500100 + 1) forces resynchronization to nonce 500200
    gap_diff = OrderBookDiff(
        first_update_id=500150,
        final_update_id=500155,
        event_time=received,
        bids=((65576.0, 2.0),),
    )
    ob = await deriv_provider.get_order_book(
        "BTC",
        market_type=MarketType.SPOT,
        pending_diffs=[gap_diff],
    )
    assert ob.snapshot is not None
    assert ob.snapshot.sequence_id == 500200
    assert ob.resync_count == 1
    assert QualityFlag.SEQUENCE_GAP in ob.quality_flags

    # Binance public REST has no public liquidation endpoint -> explicit source_available=False
    liq = await deriv_provider.get_liquidations("BTC")
    assert liq.source_available is False
    assert liq.events == ()


@pytest.mark.asyncio
async def test_analysis_service_with_derivatives_and_failure_degradation() -> None:
    fixture = json.loads(DERIV_FIXTURE.read_text(encoding="utf-8"))
    received = datetime.fromisoformat(fixture["received_time"])

    spot_client = FakeSpotClient(order_books=[dict(fixture["order_book"])])
    candle_provider = CcxtMarketDataProvider(
        client=spot_client,
        retry_initial_seconds=0.0,
        retry_max_seconds=0.0,
        clock=lambda: received,
    )
    futures_client = FakeFuturesClient(
        fixture,
        funding_outcome=ccxt.NetworkError("futures funding down"),
    )

    async def _liq_fetcher(mapping: SymbolMapping, limit: int) -> list[dict[str, Any]]:
        del mapping, limit
        return list(fixture["liquidations"])

    deriv_provider = CcxtDerivativesLiquidityProvider(
        spot_client=spot_client,
        futures_client=futures_client,
        liquidation_fetcher=_liq_fetcher,
        max_attempts=1,
        retry_initial_seconds=0.0,
        retry_max_seconds=0.0,
        clock=lambda: received,
    )

    service = AnalysisService(candle_provider, derivatives_provider=deriv_provider)
    snapshot = await service.analyze("BTC", Timeframe.H4, include_mtf=False, persist=False)

    assert snapshot.quality_status is QualityStatus.DEGRADED
    assert QualityFlag.DERIVATIVES_UNAVAILABLE in snapshot.quality_flags
    assert snapshot.derivatives is not None
    assert snapshot.derivatives.funding_rate is None
    assert snapshot.derivatives.open_interest_base == pytest.approx(82400.0)
    assert snapshot.derivatives.open_interest_change_pct is not None
    assert snapshot.derivatives.mid_price is not None
    assert snapshot.derivatives.liquidation_source_available is True
    assert snapshot.derivatives.liquidation_count == 2


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_binance_derivatives_and_spot_orderbook_smoke() -> None:
    async with CcxtDerivativesLiquidityProvider(timeout_seconds=10.0, max_attempts=2) as provider:
        funding, oi, ob, liq = await asyncio.gather(
            provider.get_funding_rate("BTC"),
            provider.get_open_interest("BTC", Timeframe.H4, history_limit=5),
            provider.get_order_book("BTC", market_type=MarketType.SPOT, limit=20),
            provider.get_liquidations("BTC"),
        )
    assert funding.observation is not None
    assert oi.latest is not None
    assert ob.snapshot is not None
    assert ob.snapshot.spread_bps > 0
    assert liq.source_available is False
