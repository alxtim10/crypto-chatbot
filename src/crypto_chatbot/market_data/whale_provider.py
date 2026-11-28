"""Market data provider for perpetual whale positions (Hyperliquid) and on-chain wallet tracking."""

from __future__ import annotations

import asyncio
import http.client
import json
import logging
import re
import urllib.parse
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

from crypto_chatbot.schemas.whale import (
    NetworkType,
    WhaleFlowEvent,
    WhaleFlowType,
    WhalePosition,
    WhalePositionSide,
    WhaleWalletProfile,
)

logger = logging.getLogger(__name__)

EVM_ADDRESS_RE = re.compile(r"^0x[a-fA-F0-9]{40}$")
BITCOIN_ADDRESS_RE = re.compile(
    r"^(1[a-km-zA-HJ-NP-Z1-9]{25,34}|3[a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-zA-HJ-NP-Z0-9]{39,59})$",
    re.IGNORECASE,
)
SOLANA_ADDRESS_RE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,44}$")

# Curated CEX hot wallet addresses (lowercased)
KNOWN_CEX_WALLETS: dict[str, str] = {
    # Binance
    "0x28c6c06298d514db089934071355e5743bf21d60": "Binance 14",
    "0x21a31ee1afc51d94c2efccaa2092ad1028285549": "Binance 8",
    "0xdfd5293d8e347dfe59e90efd55b2956a1343963d": "Binance Hot",
    "1ndy2qek11yxh6p2fum7kkd8gyqwwx1v5": "Binance Cold BTC",
    "bc1qm34lsc65zpw79lxes69zkqmk6ee3ewf0j77s3h": "Binance BTC Deposit",
    # Bybit
    "0xf977814e90da44bfa03b6295a0616a897441acec": "Bybit Hot Wallet",
    "0x1db93e1ea66a3d1326442c754d9c7d1e8ecfb4fd": "Bybit Cold 2",
    # OKX
    "0x6cc5f688a315f3dc28a7781717a9a798a59fda7b": "OKX Hot Wallet",
    "0xa7efae728d2936e78bda97dc267687568dd593f3": "OKX Cold",
    # Coinbase
    "0x503828976d22510aad0201ac7ec88293211d23da": "Coinbase 2",
    "0xddfabcdc4d8ffc6d5beaf154f18b778f892a0740": "Coinbase 3",
    "bc1qgdjqv0av3q56jvd82tkdjpy7gdp9ut8tlqmgrpmv24sq90ecnvqqjwvw97": "Coinbase BTC Hot",
    # Kraken
    "0x2910543af39aba0cd09dbb2d50200b3e800a63d2": "Kraken Hot 1",
    "0x0a869d79a7052c7f1b55a8ebabbea3420f0d1e13": "Kraken Hot 2",
}

# Curated prominent institutional / vault addresses on Hyperliquid
# (Users can add/remove any EVM Hyperliquid wallet via `/whale add <address> [label]`)
CURATED_HYPERLIQUID_WHALES: tuple[tuple[str, str], ...] = (
    ("0xdfc24b077bc1425ad1dea75bcb6f8158e10df303", "Hyperliquid HLP Vault"),
    ("0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be", "Whale Alpha (Fund)"),
    ("0x5a52e96bacdabb82fd05763e25335261b270efcb", "High-Stakes Perp Trader"),
    ("0x47ac0fb4f2d84898e4d9e7b4dab3c24507a6d503", "Institutional Perp LP"),
    ("0x8e870d67f660d95d5be530380d0ec0bd388289e1", "Top Leaderboard Whale"),
)


def detect_network_type(address: str) -> NetworkType | None:
    """Detect network type from wallet address string."""
    addr = address.strip()
    if EVM_ADDRESS_RE.match(addr):
        return NetworkType.EVM
    if BITCOIN_ADDRESS_RE.match(addr):
        return NetworkType.BITCOIN
    if SOLANA_ADDRESS_RE.match(addr):
        return NetworkType.SOLANA
    return None


class WhaleHttpTransport(Protocol):
    """Protocol for fetching JSON from remote endpoints."""

    async def post_json(
        self,
        url: str,
        json_payload: dict[str, Any],
        *,
        timeout_seconds: float = 8.0,
    ) -> Any: ...

    async def get_json(
        self,
        url: str,
        *,
        timeout_seconds: float = 8.0,
    ) -> Any: ...


def _sync_http_request(
    method: str,
    url: str,
    body_data: bytes | None,
    headers: dict[str, str],
    timeout_seconds: float,
) -> Any:
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme.lower()
    host = parsed.hostname
    if not host or scheme not in ("http", "https"):
        raise ValueError(f"Invalid URL for HTTP request: {url}")

    port = parsed.port or (443 if scheme == "https" else 80)
    target_path = parsed.path or "/"
    if parsed.query:
        target_path = f"{target_path}?{parsed.query}"

    conn: http.client.HTTPConnection = (
        http.client.HTTPSConnection(host, port=port, timeout=timeout_seconds)
        if scheme == "https"
        else http.client.HTTPConnection(host, port=port, timeout=timeout_seconds)
    )
    try:
        conn.request(method, target_path, body=body_data, headers=headers)
        resp = conn.getresponse()
        raw_body = resp.read()
        if resp.status >= 400:
            logger.warning(
                "HTTP %s %s returned status %d: %s",
                method,
                url,
                resp.status,
                raw_body[:200].decode("utf-8", errors="replace"),
            )
            return None
        if not raw_body:
            return None
        return json.loads(raw_body.decode("utf-8"))
    finally:
        conn.close()


class DefaultWhaleHttpTransport:
    """Standard HTTP transport using connection pooling via sync thread execution."""

    async def post_json(
        self,
        url: str,
        json_payload: dict[str, Any],
        *,
        timeout_seconds: float = 8.0,
    ) -> Any:
        body = json.dumps(json_payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "crypto-chatbot/0.1.0",
        }
        return await asyncio.to_thread(
            _sync_http_request,
            "POST",
            url,
            body,
            headers,
            timeout_seconds,
        )

    async def get_json(
        self,
        url: str,
        *,
        timeout_seconds: float = 8.0,
    ) -> Any:
        headers = {
            "Accept": "application/json",
            "User-Agent": "crypto-chatbot/0.1.0",
        }
        return await asyncio.to_thread(
            _sync_http_request,
            "GET",
            url,
            None,
            headers,
            timeout_seconds,
        )


class WhaleDataProvider:
    """Read-only deterministic data provider for whale positions and on-chain metrics."""

    def __init__(
        self,
        *,
        transport: WhaleHttpTransport | None = None,
        hyperliquid_api_url: str = "https://api.hyperliquid.xyz/info",
        mempool_api_url: str = "https://mempool.space/api",
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._transport: WhaleHttpTransport = transport or DefaultWhaleHttpTransport()
        self._hl_url = hyperliquid_api_url.rstrip("/")
        self._mempool_url = mempool_api_url.rstrip("/")
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))

    async def get_hyperliquid_clearinghouse_state(
        self,
        address: str,
        timeout_seconds: float = 6.0,
    ) -> dict[str, Any] | None:
        """Fetch Hyperliquid clearinghouse state for an EVM address."""
        payload = {"type": "clearinghouseState", "user": address.lower()}
        try:
            res = await self._transport.post_json(
                self._hl_url,
                payload,
                timeout_seconds=timeout_seconds,
            )
            if isinstance(res, dict):
                return res
            return None
        except Exception as exc:
            logger.debug("Failed to fetch Hyperliquid state for %s: %s", address, exc)
            return None

    async def get_positions_for_address(
        self,
        address: str,
        label: str | None = None,
    ) -> tuple[WhalePosition, ...]:
        """Fetch and parse perpetual positions held by a specific address on Hyperliquid."""
        network = detect_network_type(address)
        if network is not NetworkType.EVM:
            return ()

        state = await self.get_hyperliquid_clearinghouse_state(address)
        if not state or not isinstance(state, dict):
            return ()

        asset_positions = state.get("assetPositions")
        if not isinstance(asset_positions, list):
            return ()

        positions: list[WhalePosition] = []
        for item in asset_positions:
            if not isinstance(item, dict):
                continue
            pos_dict = item.get("position")
            if not isinstance(pos_dict, dict):
                continue

            coin = str(pos_dict.get("coin", "")).upper()
            if not coin:
                continue

            try:
                szi = float(pos_dict.get("szi", 0.0))
                if abs(szi) < 1e-6:
                    continue
                entry_px = float(pos_dict.get("entryPx", 0.0))
                liq_px_raw = pos_dict.get("liquidationPx")
                liq_px = float(liq_px_raw) if liq_px_raw is not None else None
                upnl = float(pos_dict.get("unrealizedPnl", 0.0))

                # Leverage object can be {'value': 10, 'type': 'cross'} or integer
                lev_data = pos_dict.get("leverage")
                leverage: float | None = None
                if isinstance(lev_data, dict):
                    lev_val = lev_data.get("value")
                    if lev_val is not None:
                        leverage = float(lev_val)
                elif isinstance(lev_data, (int, float)):
                    leverage = float(lev_data)

                size_tokens = abs(szi)
                pos_val_raw = pos_dict.get("positionValue")
                if pos_val_raw is not None:
                    pos_val = abs(float(pos_val_raw))
                    size_usd = pos_val if pos_val > 0.0 else (size_tokens * entry_px)
                else:
                    size_usd = size_tokens * entry_px
                side = WhalePositionSide.LONG if szi > 0 else WhalePositionSide.SHORT

                upnl_percent = (upnl / size_usd * 100.0) if size_usd > 0 else None

                positions.append(
                    WhalePosition(
                        wallet_address=address.lower(),
                        wallet_label=label,
                        symbol=coin,
                        side=side,
                        size_tokens=size_tokens,
                        size_usd=size_usd,
                        entry_price=entry_px,
                        liquidation_price=liq_px,
                        leverage=leverage,
                        unrealized_pnl_usd=upnl,
                        unrealized_pnl_percent=upnl_percent,
                    )
                )
            except (ValueError, TypeError) as exc:
                logger.debug("Failed parsing position entry for %s: %s", coin, exc)
                continue

        return tuple(positions)

    async def get_positions_for_symbol(
        self,
        symbol: str,
        wallets: Sequence[tuple[str, str]] | None = None,
    ) -> tuple[WhalePosition, ...]:
        """Fetch perpetual positions for a given symbol across a list of whale wallets."""
        clean_sym = symbol.strip().upper()
        if "/" in clean_sym:
            clean_sym = clean_sym.split("/")[0]
        elif clean_sym.endswith("USDT") and len(clean_sym) > 4:
            clean_sym = clean_sym[:-4]

        targets = wallets or CURATED_HYPERLIQUID_WHALES
        sem = asyncio.Semaphore(5)

        async def _fetch_addr(addr: str, lbl: str) -> tuple[WhalePosition, ...]:
            async with sem:
                return await self.get_positions_for_address(addr, label=lbl)

        results = await asyncio.gather(
            *(_fetch_addr(addr, label) for addr, label in targets),
            return_exceptions=True,
        )

        matched_positions: list[WhalePosition] = []
        for res in results:
            if isinstance(res, tuple):
                for pos in res:
                    if pos.symbol == clean_sym:
                        matched_positions.append(pos)

        # Sort positions descending by size_usd
        matched_positions.sort(key=lambda p: p.size_usd, reverse=True)
        return tuple(matched_positions)

    async def get_bitcoin_address_profile(
        self,
        address: str,
        label: str | None = None,
        timeout_seconds: float = 6.0,
    ) -> WhaleWalletProfile:
        """Fetch Bitcoin on-chain balance and recent transaction flows from Mempool.space."""
        url = f"{self._mempool_url}/address/{address}"
        txs_url = f"{self._mempool_url}/address/{address}/txs"

        balance_btc = 0.0
        flows: list[WhaleFlowEvent] = []

        try:
            stats = await self._transport.get_json(url, timeout_seconds=timeout_seconds)
            if isinstance(stats, dict):
                chain_stats = stats.get("chain_stats", {})
                if isinstance(chain_stats, dict):
                    funded = float(chain_stats.get("funded_txo_sum", 0))
                    spent = float(chain_stats.get("spent_txo_sum", 0))
                    balance_sat = funded - spent
                    balance_btc = max(0.0, balance_sat / 100_000_000.0)
        except Exception as exc:
            logger.debug("Mempool stats error for %s: %s", address, exc)

        try:
            raw_txs = await self._transport.get_json(txs_url, timeout_seconds=timeout_seconds)
            if isinstance(raw_txs, list):
                for tx in raw_txs[:10]:
                    if not isinstance(tx, dict):
                        continue
                    txid = str(tx.get("txid", ""))
                    status = tx.get("status", {})
                    block_time = 0
                    if isinstance(status, dict):
                        block_time = int(status.get("block_time", 0))
                    dt = (
                        datetime.fromtimestamp(block_time, tz=UTC)
                        if block_time > 0
                        else self._clock()
                    )

                    vin = tx.get("vin", [])
                    vout = tx.get("vout", [])

                    # Detect if inputs or outputs match known CEX
                    input_addrs = [
                        v.get("prevout", {}).get("scriptpubkey_address", "")
                        for v in vin
                        if isinstance(v, dict) and isinstance(v.get("prevout"), dict)
                    ]
                    output_addrs = [
                        vo.get("scriptpubkey_address", "") for vo in vout if isinstance(vo, dict)
                    ]

                    # Check for CEX inflow (outputs contain CEX) or outflow (inputs contain CEX)
                    matched_cex_out = next(
                        (
                            KNOWN_CEX_WALLETS[oa.lower()]
                            for oa in output_addrs
                            if oa and oa.lower() in KNOWN_CEX_WALLETS
                        ),
                        None,
                    )
                    matched_cex_in = next(
                        (
                            KNOWN_CEX_WALLETS[ia.lower()]
                            for ia in input_addrs
                            if ia and ia.lower() in KNOWN_CEX_WALLETS
                        ),
                        None,
                    )

                    flow_type = WhaleFlowType.TRANSFER
                    counterpart = ""
                    c_label = None
                    addr_lower = address.lower()

                    if matched_cex_out:
                        flow_type = WhaleFlowType.CEX_INFLOW
                        c_label = matched_cex_out
                        counterpart = next(
                            oa for oa in output_addrs if oa and oa.lower() in KNOWN_CEX_WALLETS
                        )
                        total_sats = sum(
                            float(vo.get("value", 0))
                            for vo in vout
                            if isinstance(vo, dict)
                            and str(vo.get("scriptpubkey_address", "")).lower() in KNOWN_CEX_WALLETS
                        )
                    elif matched_cex_in:
                        flow_type = WhaleFlowType.CEX_OUTFLOW
                        c_label = matched_cex_in
                        counterpart = next(
                            ia for ia in input_addrs if ia and ia.lower() in KNOWN_CEX_WALLETS
                        )
                        has_direct_recipient = any(
                            isinstance(vo, dict)
                            and str(vo.get("scriptpubkey_address", "")).lower() == addr_lower
                            for vo in vout
                        )
                        if has_direct_recipient:
                            total_sats = sum(
                                float(vo.get("value", 0))
                                for vo in vout
                                if isinstance(vo, dict)
                                and str(vo.get("scriptpubkey_address", "")).lower() == addr_lower
                            )
                        else:
                            total_sats = sum(
                                float(vo.get("value", 0))
                                for vo in vout
                                if isinstance(vo, dict)
                                and str(vo.get("scriptpubkey_address", "")).lower()
                                not in KNOWN_CEX_WALLETS
                            )
                    else:
                        counterpart = output_addrs[0] if output_addrs else "unknown"
                        non_self = [
                            float(vo.get("value", 0))
                            for vo in vout
                            if isinstance(vo, dict)
                            and str(vo.get("scriptpubkey_address", "")).lower() != addr_lower
                        ]
                        total_sats = (
                            sum(non_self)
                            if non_self
                            else sum(
                                float(vo.get("value", 0)) for vo in vout if isinstance(vo, dict)
                            )
                        )

                    amount_btc = total_sats / 100_000_000.0

                    flows.append(
                        WhaleFlowEvent(
                            tx_hash=txid[:16] + "...",
                            timestamp=dt,
                            flow_type=flow_type,
                            amount=amount_btc,
                            unit="BTC",
                            counterpart_address=counterpart,
                            counterpart_label=c_label,
                        )
                    )
        except Exception as exc:
            logger.debug("Mempool txs error for %s: %s", address, exc)

        return WhaleWalletProfile(
            address=address,
            network=NetworkType.BITCOIN,
            label=label,
            balance_native=balance_btc,
            positions=(),
            recent_flows=tuple(flows),
        )

    async def get_wallet_profile(
        self,
        address: str,
        label: str | None = None,
    ) -> WhaleWalletProfile:
        """Inspect wallet profile across perps positions (EVM) or on-chain flows."""
        net = detect_network_type(address)
        if net is NetworkType.BITCOIN:
            return await self.get_bitcoin_address_profile(address, label=label)

        if net is NetworkType.EVM:
            positions = await self.get_positions_for_address(address, label=label)
            # Check if address is in known CEX list
            cex_name = KNOWN_CEX_WALLETS.get(address.lower())
            resolved_label = label or cex_name
            return WhaleWalletProfile(
                address=address.lower(),
                network=NetworkType.EVM,
                label=resolved_label,
                balance_native=0.0,
                positions=positions,
                recent_flows=(),
            )

        # Default fallback for Solana or unsupported
        return WhaleWalletProfile(
            address=address,
            network=net or NetworkType.EVM,
            label=label,
            balance_native=0.0,
            positions=(),
            recent_flows=(),
        )
