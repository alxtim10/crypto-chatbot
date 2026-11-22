"""Read-only Bybit public derivatives adapter for cross-venue comparison (ADR 0010)."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Protocol

import ccxt
import ccxt.async_support as ccxt_async

from crypto_chatbot.market_data.derivatives_quality import normalize_funding_rate_payload
from crypto_chatbot.market_data.symbols import SymbolMapping, resolve_symbol
from crypto_chatbot.schemas.derivatives import FundingRateBatch
from crypto_chatbot.schemas.market import MarketType, QualityFlag

DEFAULT_BYBIT_TIMEOUT_SECONDS = 3.0


class BybitClientProtocol(Protocol):
    """Protocol for Bybit CCXT client."""

    id: str

    async def fetch_funding_rate(self, symbol: str) -> dict[str, Any]: ...

    async def close(self) -> None: ...


class BybitDerivativesProvider:
    """Read-only Bybit public derivatives provider with strict timeout and graceful degradation."""

    def __init__(
        self,
        *,
        client: BybitClientProtocol | None = None,
        timeout_seconds: float = DEFAULT_BYBIT_TIMEOUT_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client: BybitClientProtocol = (
            client
            if client is not None
            else ccxt_async.bybit(
                {
                    "enableRateLimit": True,
                    "timeout": int(timeout_seconds * 1000),
                    "options": {"defaultType": "linear"},
                }
            )
        )
        self._timeout_seconds = timeout_seconds
        self._clock = clock or (lambda: datetime.now(UTC))

    async def get_funding_rate(self, symbol: str) -> FundingRateBatch:
        """Fetch latest Bybit linear perpetual funding rate without credentials."""
        # Bybit symbols in CCXT linear perpetual: BTC/USDT:USDT
        canonical = symbol if ":" in symbol else f"{symbol}:USDT"
        mapping = resolve_symbol(canonical, market_type=MarketType.PERPETUAL)
        received_time = self._clock()

        try:
            raw_funding = await asyncio.wait_for(
                self._client.fetch_funding_rate(mapping.ccxt_symbol),
                timeout=self._timeout_seconds,
            )
            bybit_mapping = SymbolMapping(
                canonical_symbol=mapping.canonical_symbol,
                provider_symbol=mapping.provider_symbol,
                ccxt_symbol=mapping.ccxt_symbol,
                venue="bybit",
                market_type=MarketType.PERPETUAL,
                base=mapping.base,
                quote=mapping.quote,
            )
            return normalize_funding_rate_payload(
                raw_funding,
                symbol=bybit_mapping,
                provider="bybit",
                received_time=received_time,
                funding_interval_hours=8.0,
            )
        except (TimeoutError, ccxt.BaseError, Exception) as exc:
            return FundingRateBatch(
                canonical_symbol=mapping.canonical_symbol,
                provider_symbol=mapping.provider_symbol,
                venue="bybit",
                received_time=received_time,
                observation=None,
                quality_flags=(QualityFlag.DERIVATIVES_UNAVAILABLE,),
                error_message=str(exc),
            )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()
