from __future__ import annotations

import pytest

from crypto_chatbot.market_data.symbols import (
    SymbolError,
    SymbolMapping,
    resolve_mvp_spot_symbol,
    resolve_symbol,
)
from crypto_chatbot.schemas.market import MarketType


@pytest.mark.parametrize(
    ("raw", "expected_canonical", "expected_provider"),
    [
        ("BTC", "BTC/USDT", "BINANCE:BTCUSDT"),
        ("btc", "BTC/USDT", "BINANCE:BTCUSDT"),
        ("  BTC/USDT  ", "BTC/USDT", "BINANCE:BTCUSDT"),
        ("BTCUSDT", "BTC/USDT", "BINANCE:BTCUSDT"),
        ("BINANCE:BTCUSDT", "BTC/USDT", "BINANCE:BTCUSDT"),
        ("BTC spot", "BTC/USDT", "BINANCE:BTCUSDT"),
        ("ETH", "ETH/USDT", "BINANCE:ETHUSDT"),
        ("SOL/USDT", "SOL/USDT", "BINANCE:SOLUSDT"),
        ("DOGE", "DOGE/USDT", "BINANCE:DOGEUSDT"),
        ("DOGE/USDT", "DOGE/USDT", "BINANCE:DOGEUSDT"),
        ("sui", "SUI/USDT", "BINANCE:SUIUSDT"),
        ("1000PEPE/USDT", "1000PEPE/USDT", "BINANCE:1000PEPEUSDT"),
    ],
)
def test_resolve_spot_symbols(
    raw: str,
    expected_canonical: str,
    expected_provider: str,
) -> None:
    mapping = resolve_mvp_spot_symbol(raw)

    assert mapping.market_type is MarketType.SPOT
    assert mapping.canonical_symbol == expected_canonical
    assert mapping.ccxt_symbol == expected_canonical
    assert mapping.provider_symbol == expected_provider
    assert mapping.venue == "binance"


@pytest.mark.parametrize(
    ("raw", "expected_canonical", "expected_provider"),
    [
        ("BTC", "BTC/USDT:USDT", "BINANCE:BTCUSDT.P"),
        ("BTC perpetual", "BTC/USDT:USDT", "BINANCE:BTCUSDT.P"),
        ("btc-perp", "BTC/USDT:USDT", "BINANCE:BTCUSDT.P"),
        ("BTC/USDT:USDT", "BTC/USDT:USDT", "BINANCE:BTCUSDT.P"),
        ("BINANCE:BTCUSDT.P", "BTC/USDT:USDT", "BINANCE:BTCUSDT.P"),
        ("ETH perpetual", "ETH/USDT:USDT", "BINANCE:ETHUSDT.P"),
        ("SOLUSDT.P", "SOL/USDT:USDT", "BINANCE:SOLUSDT.P"),
        ("DOGE perpetual", "DOGE/USDT:USDT", "BINANCE:DOGEUSDT.P"),
    ],
)
def test_resolve_perpetual_symbols(
    raw: str,
    expected_canonical: str,
    expected_provider: str,
) -> None:
    mapping = resolve_symbol(raw, market_type=MarketType.PERPETUAL)

    assert mapping.market_type is MarketType.PERPETUAL
    assert mapping.canonical_symbol == expected_canonical
    assert mapping.ccxt_symbol == expected_canonical
    assert mapping.provider_symbol == expected_provider


@pytest.mark.parametrize(
    "raw",
    [
        "BTC perpetual",
        "BTC-PERP",
        "BTC/USDT:USDT",
        "BINANCE:BTCUSDT.P",
        "ETHUSDT.P",
    ],
)
def test_perpetual_symbols_rejected_when_resolving_spot(raw: str) -> None:
    with pytest.raises(SymbolError, match="perpetual"):
        resolve_mvp_spot_symbol(raw)


@pytest.mark.parametrize(
    "raw",
    [
        "BTC/USDT",
        "BTCUSDT",
        "BINANCE:BTCUSDT",
        "BTC spot",
    ],
)
def test_explicit_spot_symbols_rejected_when_resolving_perpetual(raw: str) -> None:
    with pytest.raises(SymbolError, match="spot"):
        resolve_symbol(raw, market_type=MarketType.PERPETUAL)


@pytest.mark.parametrize(
    "raw",
    [
        "",
        "   ",
        "USDT",
        "12345",
        "BTC/EUR",
        "BTC/USDT/ETH",
        "BINANCE:BTCBUSD",
        "$$$",
    ],
)
def test_invalid_or_unsupported_symbols_rejected(raw: str) -> None:
    with pytest.raises(SymbolError):
        resolve_mvp_spot_symbol(raw)


def test_symbol_mapping_round_trip() -> None:
    mapping = resolve_mvp_spot_symbol("BTC")
    restored = SymbolMapping.model_validate_json(mapping.model_dump_json())
    assert restored == mapping
