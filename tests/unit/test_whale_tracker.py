"""Unit tests for perpetual whale tracking, on-chain flows, and bot/CLI integration."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from crypto_chatbot.analysis.whale_tracker import (
    HEDGING_DISCLAIMER,
    WhaleWatchlistManager,
    aggregate_whale_positions,
    format_whale_overview_telegram,
)
from crypto_chatbot.bot.commands import (
    WhaleIntent,
    parse_whale_args,
    resolve_followup_intent,
)
from crypto_chatbot.cli import main
from crypto_chatbot.market_data.whale_provider import (
    WhaleDataProvider,
    detect_network_type,
)
from crypto_chatbot.schemas.commands import WhaleAction
from crypto_chatbot.schemas.whale import (
    NetworkType,
    WhaleFlowType,
    WhalePosition,
    WhalePositionSide,
)


class MockWhaleTransport:
    """Mocked HTTP transport returning predefined JSON responses."""

    def __init__(
        self,
        *,
        post_response: Any = None,
        get_responses: dict[str, Any] | None = None,
    ) -> None:
        self.post_response = post_response
        self.get_responses = get_responses or {}
        self.post_calls: list[tuple[str, dict[str, Any]]] = []
        self.get_calls: list[str] = []

    async def post_json(
        self,
        url: str,
        json_payload: dict[str, Any],
        *,
        timeout_seconds: float = 8.0,
    ) -> Any:
        self.post_calls.append((url, json_payload))
        return self.post_response

    async def get_json(
        self,
        url: str,
        *,
        timeout_seconds: float = 8.0,
    ) -> Any:
        self.get_calls.append(url)
        for pattern, resp in self.get_responses.items():
            if pattern in url:
                return resp
        return None


def test_detect_network_type() -> None:
    # EVM
    assert detect_network_type("0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be") == NetworkType.EVM
    assert detect_network_type("0x0000000000000000000000000000000000000000") == NetworkType.EVM

    # Bitcoin
    assert detect_network_type("1ndy2qek11yxh6p2fum7kkd8gyqwwx1v5") == NetworkType.BITCOIN
    assert detect_network_type("3J98t1WpEZ73CNmQviecrnyiWrnqRhWNLy") == NetworkType.BITCOIN
    assert detect_network_type("bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h") == NetworkType.BITCOIN

    # Solana
    assert detect_network_type("So11111111111111111111111111111111111111112") == NetworkType.SOLANA

    # Invalid
    assert detect_network_type("invalid_address_123") is None
    assert detect_network_type("") is None


@pytest.mark.asyncio
async def test_hyperliquid_position_parsing() -> None:
    mock_payload = {
        "assetPositions": [
            {
                "position": {
                    "coin": "BTC",
                    "szi": "10.0",
                    "entryPx": "60000.0",
                    "liquidationPx": "45000.0",
                    "leverage": {"value": 10},
                    "unrealizedPnl": "50000.0",
                }
            },
            {
                "position": {
                    "coin": "ETH",
                    "szi": "-100.0",
                    "entryPx": "3000.0",
                    "liquidationPx": "3800.0",
                    "leverage": 5,
                    "unrealizedPnl": "-10000.0",
                }
            },
            {
                "position": {
                    "coin": "SOL",
                    "szi": "0.0",
                    "entryPx": "150.0",
                }
            },
        ]
    }
    transport = MockWhaleTransport(post_response=mock_payload)
    provider = WhaleDataProvider(transport=transport)

    positions = await provider.get_positions_for_address(
        "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be",
        label="Fund Alpha",
    )
    assert len(positions) == 2

    btc_pos = next(p for p in positions if p.symbol == "BTC")
    assert btc_pos.side == WhalePositionSide.LONG
    assert btc_pos.size_usd == 600000.0
    assert btc_pos.entry_price == 60000.0
    assert btc_pos.liquidation_price == 45000.0
    assert btc_pos.leverage == 10.0
    assert btc_pos.unrealized_pnl_usd == 50000.0
    assert btc_pos.wallet_label == "Fund Alpha"

    eth_pos = next(p for p in positions if p.symbol == "ETH")
    assert eth_pos.side == WhalePositionSide.SHORT
    assert eth_pos.size_usd == 300000.0
    assert eth_pos.entry_price == 3000.0
    assert eth_pos.leverage == 5.0
    assert eth_pos.unrealized_pnl_usd == -10000.0


@pytest.mark.asyncio
async def test_bitcoin_flow_detection() -> None:
    stats_resp = {
        "chain_stats": {
            "funded_txo_sum": 500_000_000,
            "spent_txo_sum": 200_000_000,
        }
    }
    txs_resp = [
        {
            "txid": "abcdef1234567890abcdef1234567890",
            "status": {"block_time": 1700000000},
            "vin": [
                {
                    "prevout": {
                        "scriptpubkey_address": "1ndy2qek11yxh6p2fum7kkd8gyqwwx1v5",
                    }
                }
            ],
            "vout": [
                {
                    "scriptpubkey_address": "bc1qtestdestinationaddress123456789",
                    "value": 150_000_000,
                }
            ],
        },
        {
            "txid": "9999991234567890abcdef1234567890",
            "status": {"block_time": 1700001000},
            "vin": [{"prevout": {"scriptpubkey_address": "bc1qsourceuseraddr"}}],
            "vout": [
                {
                    "scriptpubkey_address": "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h",
                    "value": 200_000_000,
                }
            ],
        },
    ]

    test_btc_addr = "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq"
    transport = MockWhaleTransport(
        get_responses={
            f"/address/{test_btc_addr}/txs": txs_resp,
            f"/address/{test_btc_addr}": stats_resp,
        }
    )
    provider = WhaleDataProvider(transport=transport)
    profile = await provider.get_wallet_profile(test_btc_addr)

    assert profile.network == NetworkType.BITCOIN
    assert profile.balance_native == 3.0  # 300M satoshis = 3.0 BTC
    assert len(profile.recent_flows) == 2

    # Flow 1: input was Binance Cold -> CEX Outflow
    assert profile.recent_flows[0].flow_type == WhaleFlowType.CEX_OUTFLOW
    assert profile.recent_flows[0].counterpart_label == "Binance Cold BTC"

    # Flow 2: output was Binance BTC Deposit -> CEX Inflow
    assert profile.recent_flows[1].flow_type == WhaleFlowType.CEX_INFLOW
    assert profile.recent_flows[1].counterpart_label == "Binance BTC Deposit"


def test_aggregate_whale_positions_and_bias() -> None:
    now = datetime(2026, 11, 25, 12, 0, tzinfo=UTC)
    positions = [
        WhalePosition(
            wallet_address="0x1111111111111111111111111111111111111111",
            symbol="BTC",
            side=WhalePositionSide.LONG,
            size_tokens=10.0,
            size_usd=800_000.0,
            entry_price=80_000.0,
        ),
        WhalePosition(
            wallet_address="0x2222222222222222222222222222222222222222",
            symbol="BTC",
            side=WhalePositionSide.SHORT,
            size_tokens=2.0,
            size_usd=200_000.0,
            entry_price=100_000.0,
        ),
    ]

    overview = aggregate_whale_positions("BTC", positions, as_of=now)
    assert overview.symbol == "BTC"
    assert overview.total_long_usd == 800_000.0
    assert overview.total_short_usd == 200_000.0
    assert overview.long_percentage == 80.0
    assert overview.short_percentage == 20.0
    assert overview.net_bias == "HEAVY_LONG"
    assert overview.active_whale_count == 2
    assert len(overview.top_positions) == 2
    assert HEDGING_DISCLAIMER in overview.notes[1]

    # Test telegram formatters
    rendered_overview = format_whale_overview_telegram(overview)
    assert "WHALE PERP TRACKER — BTC" in rendered_overview
    assert "HEAVY_LONG" in overview.net_bias or "BULLISH KUAT" in rendered_overview
    assert "$800.0K" in rendered_overview


def test_whale_watchlist_manager() -> None:
    manager = WhaleWatchlistManager()
    entries = manager.list_entries()
    assert len(entries) > 0  # starts with curated institutional defaults

    # Add new valid EVM address
    new_addr = "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    entry = manager.add_entry(new_addr, "Whale VIP")
    assert entry.address == new_addr.lower()
    assert entry.label == "Whale VIP"

    user_entries = manager.list_entries()
    assert any(e.address == new_addr.lower() for e in user_entries)

    # Remove entry
    removed = manager.remove_entry(new_addr)
    assert removed is True
    assert not any(e.address == new_addr.lower() for e in manager.list_entries())

    # Invalid address raises ValueError
    with pytest.raises(ValueError, match="Format alamat wallet tidak valid"):
        manager.add_entry("not_an_address", "Invalid")


def test_whale_html_escaping_empty_watchlist_and_negative_usd() -> None:
    from crypto_chatbot.analysis.whale_tracker import (
        MAX_WHALE_WATCHLIST_PER_USER,
        format_currency_usd,
        format_whale_wallet_telegram,
        format_whale_watchlist_telegram,
    )
    from crypto_chatbot.schemas.whale import WhaleWalletProfile, WhaleWatchlistEntry

    assert format_currency_usd(-150_000.0) == "-$150.0K"
    assert format_currency_usd(-25.5) == "-$25.50"

    empty_html = format_whale_watchlist_telegram([], html_mode=True)
    assert "<b>DAFTAR WATCHLIST WHALE</b>" in empty_html
    assert "<address>" not in empty_html
    assert "&lt;address&gt;" in empty_html

    empty_cli = format_whale_watchlist_telegram([], html_mode=False)
    assert "<address>" in empty_cli
    assert "<b>" not in empty_cli

    xss_entry = WhaleWatchlistEntry(
        address="0x1111111111111111111111111111111111111111",
        network=NetworkType.EVM,
        label="<script>alert(1)</script>",
        added_at=datetime(2026, 11, 25, 12, 0, tzinfo=UTC),
    )
    rendered_list = format_whale_watchlist_telegram([xss_entry], html_mode=True)
    assert "<script>" not in rendered_list
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered_list

    profile = WhaleWalletProfile(
        address="0x1111111111111111111111111111111111111111",
        network=NetworkType.EVM,
        label="Trader <Bad>",
    )
    rendered_wallet = format_whale_wallet_telegram(profile, html_mode=True)
    assert "<Bad>" not in rendered_wallet
    assert "&lt;Bad&gt;" in rendered_wallet

    # Removing all user wallets keeps watchlist empty instead of resurrecting defaults
    manager = WhaleWatchlistManager()
    uid = 9001
    for entry in manager.list_entries(uid):
        assert manager.remove_entry(entry.address, user_id=uid) is True
    assert manager.list_entries(uid) == ()

    # Enforcing MAX_WHALE_WATCHLIST_PER_USER cap
    for idx in range(MAX_WHALE_WATCHLIST_PER_USER):
        addr = f"0x{idx + 1:040x}"
        manager.add_entry(addr, f"Wallet {idx}", user_id=uid)
    with pytest.raises(ValueError, match="maksimal"):
        manager.add_entry(f"0x{999:040x}", "Overflow", user_id=uid)


@pytest.mark.asyncio
async def test_whale_provider_position_value_and_precise_btc_flow_and_multi_out_cex() -> None:
    # 1. Hyperliquid positionValue is preferred over entry_px * size_tokens
    hl_payload = {
        "assetPositions": [
            {
                "position": {
                    "coin": "ETH",
                    "szi": "10.0",
                    "entryPx": "2000.0",
                    "positionValue": "35000.0",
                    "unrealizedPnl": "15000.0",
                }
            }
        ]
    }
    hl_transport = MockWhaleTransport(post_response=hl_payload)
    hl_provider = WhaleDataProvider(transport=hl_transport)
    positions = await hl_provider.get_positions_for_address(
        "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be"
    )
    assert len(positions) == 1
    assert positions[0].size_usd == 35_000.0

    # 2. Multi-output Bitcoin tx: CEX address at index > 0 is detected and UTXO change excluded
    tracked = "bc1qar0srrr7xfkvy5l643lydnw9re59gtzzwf5mdq"
    cex_deposit = "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h"  # Binance BTC Deposit
    cex_cold = "1ndy2qek11yxh6p2fum7kkd8gyqwwx1v5"  # Binance Cold BTC
    txs_resp = [
        {
            "txid": "11111111111111111111111111111111",
            "status": {"block_time": 1700002000},
            "vin": [{"prevout": {"scriptpubkey_address": tracked}}],
            "vout": [
                {"scriptpubkey_address": tracked, "value": 900_000_000},  # 9 BTC self change!
                {
                    "scriptpubkey_address": cex_deposit,
                    "value": 50_000_000,
                },  # 0.5 BTC deposit at idx 1
            ],
        },
        {
            "txid": "22222222222222222222222222222222",
            "status": {"block_time": 1700003000},
            "vin": [{"prevout": {"scriptpubkey_address": cex_cold}}],
            "vout": [
                {"scriptpubkey_address": "bc1qsomeotherchange999", "value": 800_000_000},
                {
                    "scriptpubkey_address": tracked,
                    "value": 120_000_000,
                },  # 1.2 BTC withdrawn to whale
            ],
        },
    ]
    btc_transport = MockWhaleTransport(
        get_responses={
            f"/address/{tracked}/txs": txs_resp,
            f"/address/{tracked}": {
                "chain_stats": {"funded_txo_sum": 1_000_000_000, "spent_txo_sum": 0}
            },
        }
    )
    btc_provider = WhaleDataProvider(transport=btc_transport)
    profile = await btc_provider.get_wallet_profile(tracked)
    assert len(profile.recent_flows) == 2
    assert profile.recent_flows[0].flow_type == WhaleFlowType.CEX_INFLOW
    assert profile.recent_flows[0].amount == pytest.approx(0.5)
    assert profile.recent_flows[1].flow_type == WhaleFlowType.CEX_OUTFLOW
    assert profile.recent_flows[1].amount == pytest.approx(1.2)


def test_parse_whale_args() -> None:
    # Default without args
    args1 = parse_whale_args([])
    assert args1.action == WhaleAction.OVERVIEW
    assert args1.symbol == "BTC"

    # Specific symbol
    args2 = parse_whale_args(["SOL"])
    assert args2.action == WhaleAction.OVERVIEW
    assert args2.symbol == "SOL"

    # Address
    evm_addr = "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be"
    args3 = parse_whale_args([evm_addr])
    assert args3.action == WhaleAction.WALLET
    assert args3.address == evm_addr

    # List action
    args4 = parse_whale_args(["list"])
    assert args4.action == WhaleAction.LIST

    # Add action
    args5 = parse_whale_args(["add", evm_addr, "Smart", "Money", "Fund"])
    assert args5.action == WhaleAction.ADD
    assert args5.address == evm_addr
    assert args5.label == "Smart Money Fund"

    # Remove action
    args6 = parse_whale_args(["remove", evm_addr])
    assert args6.action == WhaleAction.REMOVE
    assert args6.address == evm_addr


def test_resolve_whale_followup_intents() -> None:
    # Natural language whale query for symbol
    intent1 = resolve_followup_intent("posisi whale di SOL sekarang gimana?", recent_messages=[])
    assert isinstance(intent1, WhaleIntent)
    assert intent1.action == WhaleAction.OVERVIEW
    assert intent1.symbol is not None
    assert "SOL" in intent1.symbol

    # Natural language whale query for address
    intent2 = resolve_followup_intent(
        "cek dompet 0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be dong",
        recent_messages=[],
    )
    assert isinstance(intent2, WhaleIntent)
    assert intent2.action == WhaleAction.WALLET
    assert intent2.address == "0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be"

    # Natural language whale list query
    intent3 = resolve_followup_intent("tampilkan daftar watchlist whale", recent_messages=[])
    assert isinstance(intent3, WhaleIntent)
    assert intent3.action == WhaleAction.LIST


def test_cli_whale_subcommand(capsys: pytest.CaptureFixture[str]) -> None:
    # 1. Test CLI whale --list
    exit_code1 = main(["whale", "--list"])
    assert exit_code1 == 0
    out1 = capsys.readouterr().out
    assert "DAFTAR WATCHLIST WHALE" in out1

    # 2. Test CLI whale --symbol BTC --json
    exit_code2 = main(["whale", "--symbol", "BTC", "--json"])
    assert exit_code2 == 0
    out2 = capsys.readouterr().out
    parsed2 = json.loads(out2)
    assert parsed2["symbol"] == "BTC"
    assert "total_long_usd" in parsed2
