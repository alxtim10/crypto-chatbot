"""Audit per-venue historical taker-buy volume availability via CCXT OHLCV (PRD v1.7 T-16)."""

from __future__ import annotations

import asyncio
import json
import sys

from crypto_chatbot.market_data.ccxt_provider import (
    SUPPORTED_PERP_EXCHANGES,
    CcxtMarketDataProvider,
)
from crypto_chatbot.schemas.market import Timeframe


async def audit_venues(symbol: str = "BTC", limit: int = 20) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for exchange_id in sorted(SUPPORTED_PERP_EXCHANGES):
        async with CcxtMarketDataProvider(
            exchange_id=exchange_id, timeout_seconds=10.0
        ) as provider:
            batch = await provider.get_closed_ohlcv(symbol, Timeframe.H4, limit=limit)
            if batch.has_provider_error or not batch.candles:
                results[exchange_id] = {
                    "reachable": False,
                    "candles": 0,
                    "taker_buy_available": False,
                    "error": batch.error_message,
                }
            else:
                with_tb = sum(1 for c in batch.candles if c.taker_buy_base_volume is not None)
                results[exchange_id] = {
                    "reachable": True,
                    "candles": len(batch.candles),
                    "taker_buy_count": with_tb,
                    "taker_buy_available": with_tb > 0,
                }
    return results


def main() -> int:
    report = asyncio.run(audit_venues())
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
