from __future__ import annotations

from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

import httpx
import pytest

from crypto_chatbot.market_data.ccxt_provider import CcxtMarketDataProvider
from crypto_chatbot.market_data.fallback_provider import build_futures_providers
from crypto_chatbot.market_data.gate_candles import (
    GateFuturesMarketDataProvider,
    GateFuturesRestClient,
    normalize_gate_candlesticks,
    parse_contract_multipliers,
)
from crypto_chatbot.market_data.symbols import resolve_default_perp_symbol
from crypto_chatbot.schemas.market import QualityFlag, Timeframe

# Recorded verbatim from live Gate.io REST verification (2026-11-27)
RECORDED_CONTRACTS_PAYLOAD: list[dict[str, Any]] = [
    {"name": "BTC_USDT", "type": "direct", "quanto_multiplier": "0.0001"},
    {"name": "ETH_USDT", "type": "direct", "quanto_multiplier": "0.01"},
    {"name": "SOL_USDT", "type": "direct", "quanto_multiplier": "1"},
]

RECORDED_BTC_1H_CANDLES: list[dict[str, Any]] = [
    {
        "o": "61623.2",
        "v": 47089285,
        "t": 1795766400,
        "c": "61696.2",
        "l": "61607.4",
        "h": "62200",
        "sum": "291265470.34431",
    },
    {
        "o": "61696.4",
        "v": 19774015,
        "t": 1795770000,
        "c": "61600",
        "l": "61561.9",
        "h": "61834",
        "sum": "121999225.03933",
    },
    {
        # Currently forming open 1h candle when as_of = 1795774800 (close=1795777200 > as_of)
        "o": "61600",
        "v": 4603636,
        "t": 1795773600,
        "c": "61615.1",
        "l": "61560",
        "h": "61696.1",
        "sum": "28370290.66036",
    },
]


def test_parse_contract_multipliers_verifies_quanto_multiplier() -> None:
    multipliers = parse_contract_multipliers(RECORDED_CONTRACTS_PAYLOAD)
    assert multipliers == {
        "BTC_USDT": Decimal("0.0001"),
        "ETH_USDT": Decimal("0.01"),
        "SOL_USDT": Decimal("1"),
    }


def test_normalize_gate_candlesticks_exact_contracts_multiplier_and_closed_filter() -> None:
    multipliers = parse_contract_multipliers(RECORDED_CONTRACTS_PAYLOAD)
    mult = multipliers["BTC_USDT"]
    symbol = resolve_default_perp_symbol("BTC")
    as_of = datetime.fromtimestamp(1795774800, tz=UTC)

    batch = normalize_gate_candlesticks(
        RECORDED_BTC_1H_CANDLES,
        symbol=symbol,
        provider_symbol="GATE:BTCUSDT.P",
        timeframe=Timeframe.H1,
        quanto_multiplier=mult,
        received_time=as_of,
        as_of=as_of,
    )

    # 3rd candle (open=1795773600, close=1795777200 > as_of) MUST be dropped by closed-candle rule.
    assert len(batch.candles) == 2
    assert QualityFlag.QUOTE_VOL_ESTIMATED not in batch.quality_flags

    c0, c1 = batch.candles
    assert c0.raw_contracts == pytest.approx(47089285.0)
    assert c0.base_volume == float(Decimal("47089285") * Decimal("0.0001"))
    assert c0.base_volume == pytest.approx(4708.9285)
    assert c0.quote_volume == pytest.approx(291265470.34431)
    assert QualityFlag.QUOTE_VOL_ESTIMATED not in c0.quality_flags

    assert c1.raw_contracts == pytest.approx(19774015.0)
    assert c1.base_volume == float(Decimal("19774015") * Decimal("0.0001"))
    assert c1.base_volume == pytest.approx(1977.4015)
    assert c1.quote_volume == pytest.approx(121999225.03933)


def test_normalize_gate_candlesticks_estimates_quote_volume_when_sum_missing() -> None:
    symbol = resolve_default_perp_symbol("ETH")
    as_of = datetime.fromtimestamp(1795774800, tz=UTC)
    row_no_sum = {
        "t": 1795770000,
        "o": "2500.0",
        "h": "2550.0",
        "l": "2490.0",
        "c": "2520.0",
        "v": 12345,
    }

    batch = normalize_gate_candlesticks(
        [row_no_sum],
        symbol=symbol,
        provider_symbol="GATE:ETHUSDT.P",
        timeframe=Timeframe.H1,
        quanto_multiplier=Decimal("0.01"),
        received_time=as_of,
        as_of=as_of,
    )
    assert len(batch.candles) == 1
    candle = batch.candles[0]
    assert candle.raw_contracts == 12345.0
    assert candle.base_volume == 123.45
    assert candle.quote_volume == pytest.approx(123.45 * 2520.0)
    assert QualityFlag.QUOTE_VOL_ESTIMATED in candle.quality_flags
    assert QualityFlag.QUOTE_VOL_ESTIMATED in batch.quality_flags


@pytest.mark.asyncio
async def test_gate_provider_fetches_contracts_cache_and_candlesticks() -> None:
    now = datetime.fromtimestamp(1795774800, tz=UTC)
    contract_calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal contract_calls
        if request.url.path.endswith("/futures/usdt/contracts"):
            contract_calls += 1
            return httpx.Response(200, json=RECORDED_CONTRACTS_PAYLOAD)
        if request.url.path.endswith("/futures/usdt/candlesticks"):
            assert request.url.params["contract"] == "BTC_USDT"
            assert request.url.params["interval"] == "1h"
            return httpx.Response(200, json=RECORDED_BTC_1H_CANDLES)
        if request.url.path.endswith("/futures/usdt/tickers"):
            return httpx.Response(
                200,
                json=[
                    {
                        "contract": "BTC_USDT",
                        "last": "61615.1",
                        "change_percentage": "1.25",
                        "volume_24h_settle": "500000000",
                    }
                ],
            )
        raise AssertionError(f"unexpected URL {request.url}")

    client = GateFuturesRestClient(transport=httpx.MockTransport(handler))
    provider = GateFuturesMarketDataProvider(client=client, clock=lambda: now)
    try:
        b1 = await provider.get_closed_ohlcv("BTC", Timeframe.H1, limit=2)
        b2 = await provider.get_closed_ohlcv("BTC", Timeframe.H1, limit=2)
        tickers = await provider.get_futures_tickers()
    finally:
        await client.close()

    assert contract_calls == 1
    assert len(b1.candles) == 2
    assert b1.candles[0].base_volume == pytest.approx(4708.9285)
    assert b1.candles[0].raw_contracts == pytest.approx(47089285.0)
    assert b2.candles[-1].open_time == datetime.fromtimestamp(1795770000, tz=UTC)
    assert len(tickers.tickers) == 1
    assert tickers.tickers[0].canonical_symbol == "BTC/USDT:USDT"


@pytest.mark.asyncio
async def test_build_futures_providers_gate_direct_candles_flag() -> None:
    p_direct, d_direct = build_futures_providers(["gate"], gate_direct_candles=True)
    assert isinstance(p_direct, GateFuturesMarketDataProvider)
    await p_direct.close()
    await d_direct.close()

    p_legacy, d_legacy = build_futures_providers(["gate"], gate_direct_candles=False)
    assert isinstance(p_legacy, CcxtMarketDataProvider)
    await p_legacy.close()
    await d_legacy.close()
