from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import ccxt
import pytest

from crypto_chatbot.market_data.base import MAX_HISTORY_CANDLE_LIMIT
from crypto_chatbot.market_data.ccxt_provider import (
    DEFAULT_BINANCE_PUBLIC_API_URL,
    CcxtMarketDataProvider,
)
from crypto_chatbot.market_data.symbols import SymbolError
from crypto_chatbot.schemas.market import MarketType, QualityFlag, Timeframe

FIXTURE_PATH = Path(__file__).resolve().parents[1] / "fixtures" / "binance_btc_4h_sample.json"


class FakeCcxtClient:
    def __init__(
        self,
        outcomes: list[list[list[Any]] | BaseException | float],
        *,
        client_id: str = "binance",
    ) -> None:
        self.id = client_id
        self._outcomes = list(outcomes)
        self.calls: list[tuple[str, str, int | None, int | None]] = []
        self.closed = False

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
    ) -> list[list[Any]]:
        self.calls.append((symbol, timeframe, since, limit))
        outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
        if isinstance(outcome, float):
            await asyncio.sleep(outcome)
            return []
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_ccxt_provider_normalizes_recorded_fixture() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    received_time = datetime.fromisoformat(fixture["received_time"])
    fake_client = FakeCcxtClient([fixture["raw_ohlcv"]])

    provider = CcxtMarketDataProvider(
        client=fake_client,
        retry_initial_seconds=0.0,
        retry_max_seconds=0.0,
        clock=lambda: received_time,
    )

    batch = await provider.get_closed_ohlcv("BTC", Timeframe.H4, limit=10)

    assert fake_client.calls == [("BTC/USDT:USDT", "4h", None, 11)]
    assert batch.canonical_symbol == "BTC/USDT:USDT"
    assert batch.provider_symbol == "BINANCE:BTCUSDT.P"
    assert batch.metadata.base_unit == "BTC"
    assert batch.metadata.quote_currency == "USDT"
    assert len(batch.candles) == 10
    assert batch.excluded_open_count == 1
    assert not batch.has_provider_error
    assert QualityFlag.STALE not in batch.quality_flags
    assert QualityFlag.GAP not in batch.quality_flags
    assert all(c.is_closed and c.close_time <= received_time for c in batch.candles)


@pytest.mark.asyncio
async def test_ccxt_provider_retries_transient_errors_then_succeeds() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    received_time = datetime.fromisoformat(fixture["received_time"])
    fake_client = FakeCcxtClient(
        [
            ccxt.NetworkError("temporary network glitch"),
            TimeoutError(),
            fixture["raw_ohlcv"],
        ]
    )

    provider = CcxtMarketDataProvider(
        client=fake_client,
        max_attempts=3,
        retry_initial_seconds=0.0,
        retry_max_seconds=0.0,
        clock=lambda: received_time,
    )

    batch = await provider.get_closed_ohlcv("BTC", Timeframe.H4, limit=10)

    assert len(fake_client.calls) == 3
    assert len(batch.candles) == 10
    assert not batch.has_provider_error


@pytest.mark.asyncio
async def test_ccxt_provider_returns_controlled_error_on_timeout() -> None:
    received_time = datetime(2026, 11, 1, 17, 0, tzinfo=UTC)
    fake_client = FakeCcxtClient([0.2])

    provider = CcxtMarketDataProvider(
        client=fake_client,
        timeout_seconds=0.02,
        max_attempts=2,
        retry_initial_seconds=0.0,
        retry_max_seconds=0.0,
        clock=lambda: received_time,
    )

    batch = await provider.get_closed_ohlcv("ETH", Timeframe.H1, limit=10)

    assert len(fake_client.calls) == 2
    assert batch.has_provider_error is True
    assert batch.quality_flags == (QualityFlag.PROVIDER_ERROR,)
    assert batch.candles == ()
    assert batch.error_message is not None
    assert "timeout" in batch.error_message.lower()


@pytest.mark.asyncio
async def test_ccxt_provider_does_not_retry_non_transient_exchange_errors() -> None:
    received_time = datetime(2026, 11, 1, 17, 0, tzinfo=UTC)
    fake_client = FakeCcxtClient([ccxt.BadSymbol("unknown pair")])

    provider = CcxtMarketDataProvider(
        client=fake_client,
        max_attempts=3,
        retry_initial_seconds=0.0,
        retry_max_seconds=0.0,
        clock=lambda: received_time,
    )

    batch = await provider.get_closed_ohlcv("SOL", Timeframe.D1, limit=10)

    assert len(fake_client.calls) == 1
    assert batch.has_provider_error is True
    assert batch.quality_flags == (QualityFlag.PROVIDER_ERROR,)
    assert batch.error_message is not None
    assert "BadSymbol" in batch.error_message


@pytest.mark.asyncio
async def test_ccxt_provider_flags_venue_mismatch() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    received_time = datetime.fromisoformat(fixture["received_time"])
    fake_client = FakeCcxtClient([fixture["raw_ohlcv"]], client_id="kraken")

    provider = CcxtMarketDataProvider(
        client=fake_client,
        retry_initial_seconds=0.0,
        retry_max_seconds=0.0,
        clock=lambda: received_time,
    )

    batch = await provider.get_closed_ohlcv("BTC", Timeframe.H4, limit=10)

    assert QualityFlag.SYMBOL_MISMATCH in batch.quality_flags


@pytest.mark.asyncio
async def test_ccxt_provider_rejects_invalid_symbol_or_limit_before_network() -> None:
    fake_client = FakeCcxtClient([])
    provider = CcxtMarketDataProvider(client=fake_client)

    with pytest.raises(SymbolError):
        await provider.get_closed_ohlcv("BTC spot", Timeframe.H4, limit=10)

    with pytest.raises(ValueError, match="limit"):
        await provider.get_closed_ohlcv("BTC", Timeframe.H4, limit=1001)

    assert fake_client.calls == []


@pytest.mark.parametrize(
    "bad_url",
    [
        "http://data-api.binance.vision/api/v3",
        "https://data-api.binance.vision/api/v3 ",
        "https://data-api.binance.vision/api/v3\n",
        "",
    ],
)
def test_ccxt_provider_rejects_invalid_public_api_url(bad_url: str) -> None:
    fake_client = FakeCcxtClient([])
    with pytest.raises(ValueError, match="public_api_base_url"):
        CcxtMarketDataProvider(client=fake_client, public_api_base_url=bad_url)


@pytest.mark.asyncio
async def test_ccxt_provider_default_client_uses_public_spot_endpoint_without_secrets() -> None:
    async with CcxtMarketDataProvider() as provider:
        client: Any = provider._client
        assert client.options["defaultType"] == "swap"
        assert client.options["fetchMarkets"]["types"] == ["linear"]
        assert not getattr(client, "apiKey", "")
        assert not getattr(client, "secret", "")

    async with CcxtMarketDataProvider(market_type=MarketType.SPOT) as spot_provider:
        spot_client: Any = spot_provider._client
        assert spot_client.urls["api"]["public"] == DEFAULT_BINANCE_PUBLIC_API_URL
        assert spot_client.options["defaultType"] == "spot"
        assert spot_client.options["fetchMarkets"]["types"] == ["spot"]


@pytest.mark.asyncio
async def test_ccxt_provider_falls_back_to_1000_perp_contract_on_bad_symbol() -> None:
    fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    received_time = datetime.fromisoformat(fixture["received_time"])
    fake_client = FakeCcxtClient(
        [
            ccxt.BadSymbol("binanceusdm does not have market symbol PEPE/USDT:USDT"),
            fixture["raw_ohlcv"],
        ]
    )

    provider = CcxtMarketDataProvider(
        client=fake_client,
        retry_initial_seconds=0.0,
        retry_max_seconds=0.0,
        clock=lambda: received_time,
    )

    batch = await provider.get_closed_ohlcv("PEPE", Timeframe.H4, limit=10)

    assert [call[0] for call in fake_client.calls] == [
        "PEPE/USDT:USDT",
        "1000PEPE/USDT:USDT",
    ]
    assert not batch.has_provider_error
    assert batch.canonical_symbol == "1000PEPE/USDT:USDT"
    assert batch.provider_symbol == "BINANCE:1000PEPEUSDT.P"
    assert batch.metadata.base_unit == "1000PEPE"
    assert len(batch.candles) == 10


@pytest.mark.asyncio
async def test_ccxt_provider_paginated_history_and_deduplication() -> None:
    step_ms = int(Timeframe.H4.duration.total_seconds() * 1000)
    base_dt = datetime(2025, 1, 1, 0, 0, tzinfo=UTC)
    base_ms = int(base_dt.timestamp() * 1000)
    # 1500 closed candles + 1 overlapping boundary row + 1 open candle at end
    total_closed = 1500
    all_rows = [
        [
            base_ms + i * step_ms,
            50000.0 + i,
            50100.0 + i,
            49900.0 + i,
            50050.0 + i,
            100.0 + i,
        ]
        for i in range(total_closed + 1)
    ]
    # Received 1 minute into the 1501st (index 1500) candle -> index 0..1499 are closed.
    received_time = datetime.fromtimestamp((base_ms + total_closed * step_ms + 60_000) / 1000, UTC)

    # Page 1 returns 1000 rows (0..999), Page 2 returns rows with an overlap (999..1500)
    page1 = all_rows[:1000]
    page2 = all_rows[999:]
    fake_client = FakeCcxtClient([page1, page2])

    provider = CcxtMarketDataProvider(
        client=fake_client,
        retry_initial_seconds=0.0,
        retry_max_seconds=0.0,
        clock=lambda: received_time,
    )

    batch = await provider.get_closed_ohlcv_history("BTC", Timeframe.H4, limit=1500)

    assert not batch.has_provider_error, batch.error_message
    assert len(fake_client.calls) == 2
    assert fake_client.calls[0][2] is not None
    assert fake_client.calls[1][2] == page1[-1][0] + step_ms
    assert len(batch.candles) == 1500
    assert batch.excluded_open_count == 1
    assert batch.duplicate_count == 0
    assert QualityFlag.GAP not in batch.quality_flags
    assert all(c.is_closed for c in batch.candles)

    with pytest.raises(ValueError, match="between 1 and"):
        await provider.get_closed_ohlcv_history(
            "BTC",
            Timeframe.H4,
            limit=MAX_HISTORY_CANDLE_LIMIT + 1,
        )


@pytest.mark.live
@pytest.mark.asyncio
async def test_live_binance_btc_4h_smoke() -> None:
    async with CcxtMarketDataProvider(timeout_seconds=15.0) as provider:
        batch = await provider.get_closed_ohlcv("BTC", Timeframe.H4, limit=10)

    assert not batch.has_provider_error, batch.error_message
    assert len(batch.candles) == 10
    assert batch.metadata.base_unit == "BTC"
    assert batch.metadata.quote_currency == "USDT"
    assert QualityFlag.STALE not in batch.quality_flags
    assert QualityFlag.GAP not in batch.quality_flags
    assert QualityFlag.INVALID_OHLC not in batch.quality_flags
    assert all(c.is_closed and c.close_time <= batch.received_time for c in batch.candles)
    assert all(c.open_time.tzinfo is UTC for c in batch.candles)
