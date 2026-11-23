"""Read-only Polymarket public connector with liquidity filtering (ADR 0012)."""

from __future__ import annotations

import asyncio
import contextlib
import json
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any, Protocol

import httpx

from crypto_chatbot.schemas.market import QualityFlag
from crypto_chatbot.schemas.prediction_market import (
    PredictionMarketCategory,
    PredictionMarketObservation,
    PredictionMarketSummary,
)

DEFAULT_TIMEOUT_SECONDS = 3.0
DEFAULT_MIN_VOLUME_USD = 1000.0
DEFAULT_MAX_SPREAD_BPS = 500.0


class PredictionMarketClientProtocol(Protocol):
    """Protocol for fetching public prediction market events."""

    async def get_events(self) -> list[dict[str, Any]]: ...

    async def close(self) -> None: ...


class HttpxGammaClient:
    """Default httpx implementation for Polymarket Gamma public API."""

    def __init__(self, timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS) -> None:
        self._client = httpx.AsyncClient(
            base_url="https://gamma-api.polymarket.com",
            timeout=timeout_seconds,
            headers={"User-Agent": "crypto-chatbot-research/1.0"},
        )

    async def get_events(self) -> list[dict[str, Any]]:
        resp = await self._client.get(
            "/events",
            params={
                "closed": "false",
                "limit": 30,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data if isinstance(data, list) else []

    async def close(self) -> None:
        await self._client.aclose()


def _classify_category(question: str) -> tuple[PredictionMarketCategory, str | None]:
    q_lower = question.lower()
    symbol: str | None = None
    if "bitcoin" in q_lower or "btc" in q_lower:
        symbol = "BTC/USDT"
    elif "ethereum" in q_lower or "eth" in q_lower:
        symbol = "ETH/USDT"
    elif "solana" in q_lower or "sol" in q_lower:
        symbol = "SOL/USDT"

    if any(k in q_lower for k in ("fed", "fomc", "rate cut", "interest rate", "cpi", "inflation")):
        return "macro_rate", symbol
    if any(k in q_lower for k in ("etf", "sec", "regulatory")):
        return "regulatory_etf", symbol
    if symbol is not None or any(k in q_lower for k in ("price", "hit", "reach", "dip", "above")):
        return "crypto_price", symbol
    return "other", symbol


def normalize_polymarket_market(
    raw: dict[str, Any],
    *,
    received_time: datetime,
    min_volume_usd: float = DEFAULT_MIN_VOLUME_USD,
    max_spread_bps: float = DEFAULT_MAX_SPREAD_BPS,
) -> PredictionMarketObservation | None:
    """Normalize a raw Polymarket market dictionary with strict validation."""
    market_id = str(raw.get("id") or raw.get("market_id") or "")
    question = str(raw.get("question") or raw.get("title") or "").strip()
    if not market_id or not question:
        return None

    # Outcome prices in Polymarket gamma API are typically in outcomePrices
    outcome_prices = raw.get("outcomePrices")
    yes_prob: float | None = None
    if isinstance(outcome_prices, str):
        with contextlib.suppress(ValueError, TypeError):
            parsed = json.loads(outcome_prices)
            if isinstance(parsed, list) and parsed:
                yes_prob = float(parsed[0])
    elif isinstance(outcome_prices, Sequence) and outcome_prices:
        with contextlib.suppress(ValueError, TypeError):
            yes_prob = float(outcome_prices[0])

    if yes_prob is None:
        raw_price = raw.get("lastPrice") or raw.get("price") or raw.get("implied_yes_prob")
        if raw_price is not None:
            with contextlib.suppress(ValueError, TypeError):
                yes_prob = float(raw_price)

    if yes_prob is None or not (0.0 <= yes_prob <= 1.0):
        return None

    vol_24h = float(raw.get("volume24hr") or raw.get("volume_24h_usd") or 0.0)
    liquidity = float(raw.get("liquidity") or raw.get("liquidity_usd") or 0.0)
    spread_bps = (
        float(raw["bid_ask_spread_bps"]) if raw.get("bid_ask_spread_bps") is not None else None
    )

    quality_flags: list[QualityFlag] = []
    if vol_24h < min_volume_usd or liquidity < min_volume_usd:
        quality_flags.append(QualityFlag.ZERO_VOLUME)
    if spread_bps is not None and spread_bps > max_spread_bps:
        quality_flags.append(QualityFlag.PRICE_MISMATCH)

    cat, sym = _classify_category(question)

    return PredictionMarketObservation(
        venue="polymarket",
        market_id=market_id,
        question=question,
        category=cat,
        canonical_symbol=sym,
        implied_yes_prob=min(1.0, max(0.0, yes_prob)),
        bid_ask_spread_bps=spread_bps,
        volume_24h_usd=vol_24h,
        liquidity_usd=liquidity,
        observed_time=received_time,
        received_time=received_time,
        available_time=received_time,
        url=f"https://polymarket.com/event/{raw.get('slug', market_id)}",
        quality_flags=tuple(quality_flags),
    )


class PolymarketConnector:
    """Read-only adapter for Polymarket public market data with fallback and timeout."""

    def __init__(
        self,
        *,
        client: PredictionMarketClientProtocol | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._owns_client = client is None
        self._client: PredictionMarketClientProtocol = (
            client if client is not None else HttpxGammaClient(timeout_seconds=timeout_seconds)
        )
        self._timeout_seconds = timeout_seconds
        self._clock = clock or (lambda: datetime.now(UTC))

    async def get_summary(
        self,
        *,
        as_of: datetime | None = None,
        target_symbol: str | None = None,
    ) -> PredictionMarketSummary:
        """Fetch and aggregate liquid prediction markets causal up to `as_of`."""
        ref_time = as_of or self._clock()
        try:
            raw_events = await asyncio.wait_for(
                self._client.get_events(),
                timeout=self._timeout_seconds,
            )
        except Exception:
            return PredictionMarketSummary(
                as_of=ref_time,
                observations=(),
                has_liquid_markets=False,
            )

        observations: list[PredictionMarketObservation] = []
        for event in raw_events:
            # Events can contain nested markets
            nested_markets = event.get("markets")
            markets_to_process = (
                nested_markets if isinstance(nested_markets, list) and nested_markets else [event]
            )
            for mkt in markets_to_process:
                obs = normalize_polymarket_market(mkt, received_time=ref_time)
                if obs is not None and obs.available_time <= ref_time:
                    # Filter by target symbol if matching crypto market
                    if (
                        target_symbol
                        and obs.category == "crypto_price"
                        and obs.canonical_symbol
                        and obs.canonical_symbol != target_symbol
                    ):
                        continue
                    observations.append(obs)

        # Aggregate key macro/catalyst benchmarks
        fomc_obs = [
            o.implied_yes_prob
            for o in observations
            if o.category == "macro_rate" and "cut" in o.question.lower()
        ]
        macro_fomc = float(fomc_obs[0]) if fomc_obs else None

        crypto_obs = [
            o.implied_yes_prob
            for o in observations
            if o.category == "crypto_price"
            and (not target_symbol or o.canonical_symbol == target_symbol)
        ]
        crypto_upside = float(crypto_obs[0]) if crypto_obs else None

        liquid = [o for o in observations if QualityFlag.ZERO_VOLUME not in o.quality_flags]

        return PredictionMarketSummary(
            as_of=ref_time,
            observations=tuple(observations),
            macro_fomc_cut_prob=macro_fomc,
            crypto_upside_prob=crypto_upside,
            has_liquid_markets=bool(liquid),
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()
