"""Gate.io perpetual funding rate history client, normalizer, and causal lookup (D-07).

Empirically verified Gate `/futures/usdt/funding_rate` behavior:
- Returns newest-first JSON list `[{"r": "0.0000379", "t": 1795766400}, ...]`.
- Capped at 90 entries per page (30 days at 8h settlement cadence); paginates backward via `to`.
- REST retention limit: 180 days (older history available via D-04 `funding_applies` CSV).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from crypto_chatbot.market_data.gate_contract_stats import (
    DEFAULT_GATE_API_URL,
    to_gate_contract_symbol,
)
from crypto_chatbot.market_data.symbols import SymbolMapping
from crypto_chatbot.schemas.derivatives import FundingRateObservation
from crypto_chatbot.schemas.market import MarketType, QualityFlag
from crypto_chatbot.storage.repositories import DerivativesRepository

GATE_FUNDING_REST_MAX_DAYS = 180
GATE_FUNDING_PAGE_SIZE = 90
DEFAULT_FUNDING_INTERVAL_HOURS = 8.0


class _RetryableFundingHttpError(RuntimeError):
    """Transient HTTP error eligible for bounded retry."""


def normalize_gate_funding_rate_payload(
    raw_items: Sequence[dict[str, Any]],
    *,
    symbol: SymbolMapping,
    provider_symbol: str,
    received_time: datetime,
    venue: str = "gate",
    provider: str = "gate_rest",
    funding_interval_hours: float = DEFAULT_FUNDING_INTERVAL_HOURS,
    as_of: datetime | None = None,
    backfill: bool = False,
) -> tuple[FundingRateObservation, ...]:
    """Normalize Gate `/futures/usdt/funding_rate` items sorted oldest -> newest with UTC safety."""
    if received_time.tzinfo is None:
        raise ValueError("received_time must be timezone-aware")
    if funding_interval_hours <= 0:
        raise ValueError("funding_interval_hours must be positive")
    received_utc = received_time.astimezone(UTC)
    cutoff_utc = (as_of or received_time).astimezone(UTC)
    scale_8h = 8.0 / funding_interval_hours
    flags = (QualityFlag.BACKFILL,) if backfill else ()

    by_time: dict[datetime, FundingRateObservation] = {}
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        raw_t = item.get("t")
        raw_r = item.get("r")
        if raw_t is None or raw_r is None:
            continue
        observed_time = datetime.fromtimestamp(int(raw_t), tz=UTC)
        if observed_time > cutoff_utc:
            continue
        rate = float(raw_r)
        if not math.isfinite(rate):
            continue
        rate_8h = rate * scale_8h
        by_time[observed_time] = FundingRateObservation(
            provider=provider,
            venue=venue,
            canonical_symbol=symbol.canonical_symbol,
            provider_symbol=provider_symbol,
            market_type=MarketType.PERPETUAL,
            observed_time=observed_time,
            next_funding_time=None,
            received_time=received_utc,
            available_time=observed_time,
            funding_rate=rate,
            funding_interval_hours=funding_interval_hours,
            rate_per_8h=rate_8h,
            annualized_rate=rate_8h * 3.0 * 365.0,
            quality_flags=flags,
            raw_payload=dict(item),
        )
    return tuple(by_time[ts] for ts in sorted(by_time))


class GateFundingHistoryClient:
    """Async public REST client paginating Gate `/futures/usdt/funding_rate` backward in time."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_GATE_API_URL,
        timeout_seconds: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.4, min=0.2, max=2.0),
        retry=retry_if_exception_type((httpx.TransportError, _RetryableFundingHttpError)),
    )
    async def fetch_funding_page(
        self,
        contract: str,
        *,
        limit: int = 100,
        from_time: int | None = None,
        to_time: int | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "contract": contract,
            "limit": min(max(1, int(limit)), 1000),
        }
        if from_time is not None:
            params["from"] = int(from_time)
        if to_time is not None:
            params["to"] = int(to_time)
        response = await self._client.get("/futures/usdt/funding_rate", params=params)
        if response.status_code in {429, 500, 502, 503, 504}:
            raise _RetryableFundingHttpError(f"Gate HTTP {response.status_code}")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise ValueError("unexpected Gate /funding_rate payload type")
        return [item for item in payload if isinstance(item, dict)]

    async def fetch_funding_history(
        self,
        contract: str,
        *,
        days: int = 90,
        end_time: datetime | None = None,
        funding_interval_hours: float = DEFAULT_FUNDING_INTERVAL_HOURS,
    ) -> list[dict[str, Any]]:
        """Paginate backward using `to` until `days` (clamped at 180d REST limit) is covered."""
        if days < 1 or days > GATE_FUNDING_REST_MAX_DAYS:
            raise ValueError(f"days must be within 1..{GATE_FUNDING_REST_MAX_DAYS} for REST")
        now_utc = (end_time or datetime.now(UTC)).astimezone(UTC)
        target_start_s = int((now_utc - timedelta(days=days)).timestamp())
        # Clamp earliest query slightly inside Gate's 180d hard reject boundary
        min_allowed_s = int((now_utc - timedelta(days=179, hours=20)).timestamp())
        target_start_s = max(target_start_s, min_allowed_s)

        by_t: dict[int, dict[str, Any]] = {}
        cursor_to = int(now_utc.timestamp())
        max_pages = max(2, math.ceil((days * 24.0 / funding_interval_hours) / 90.0) + 2)

        for _ in range(max_pages):
            if cursor_to < target_start_s:
                break
            page = await self.fetch_funding_page(contract, limit=100, to_time=cursor_to)
            if not page:
                break
            oldest_t: int | None = None
            for item in page:
                raw_t = item.get("t")
                if raw_t is None:
                    continue
                t_val = int(raw_t)
                if t_val >= target_start_s:
                    by_t[t_val] = item
                oldest_t = t_val if oldest_t is None else min(oldest_t, t_val)
            if oldest_t is None or oldest_t <= target_start_s or oldest_t - 1 >= cursor_to:
                break
            cursor_to = oldest_t - 1

        return [by_t[k] for k in sorted(by_t)]


@dataclass(frozen=True, slots=True)
class CausalFundingFeatures:
    """Strictly causal historical funding summary at a given `as_of` instant."""

    as_of: datetime
    latest_observation: FundingRateObservation | None
    latest_rate_8h: float | None
    mean_7d_rate_8h: float | None
    mean_30d_rate_8h: float | None
    zscore_30d: float | None
    observation_count: int


def lookup_causal_funding_at(
    observations: Sequence[FundingRateObservation],
    *,
    as_of: datetime,
    use_available_time: bool = True,
) -> CausalFundingFeatures:
    """Compute causal point-in-time funding stats using ONLY settlements <= `as_of`."""
    if as_of.tzinfo is None:
        raise ValueError("as_of must be timezone-aware")
    cutoff = as_of.astimezone(UTC)

    eligible = [
        obs
        for obs in observations
        if (obs.available_time if use_available_time else obs.observed_time) <= cutoff
    ]
    eligible.sort(key=lambda o: o.observed_time)
    if not eligible:
        return CausalFundingFeatures(
            as_of=cutoff,
            latest_observation=None,
            latest_rate_8h=None,
            mean_7d_rate_8h=None,
            mean_30d_rate_8h=None,
            zscore_30d=None,
            observation_count=0,
        )

    latest = eligible[-1]
    cut_7d = cutoff - timedelta(days=7)
    cut_30d = cutoff - timedelta(days=30)
    rates_7d = [obs.rate_per_8h for obs in eligible if obs.observed_time >= cut_7d]
    rates_30d = [obs.rate_per_8h for obs in eligible if obs.observed_time >= cut_30d]

    mean_7d = (sum(rates_7d) / len(rates_7d)) if rates_7d else None
    mean_30d = (sum(rates_30d) / len(rates_30d)) if rates_30d else None

    zscore_30d: float | None = None
    if len(rates_30d) >= 3 and mean_30d is not None:
        var = sum((r - mean_30d) ** 2 for r in rates_30d) / len(rates_30d)
        std = math.sqrt(var)
        if std > 1e-12:
            zscore_30d = (latest.rate_per_8h - mean_30d) / std

    return CausalFundingFeatures(
        as_of=cutoff,
        latest_observation=latest,
        latest_rate_8h=latest.rate_per_8h,
        mean_7d_rate_8h=mean_7d,
        mean_30d_rate_8h=mean_30d,
        zscore_30d=zscore_30d,
        observation_count=len(eligible),
    )


def to_backtest_funding_tuples(
    observations: Sequence[FundingRateObservation],
    *,
    as_of: datetime | None = None,
    use_available_time: bool = True,
) -> tuple[tuple[datetime, float], ...]:
    """Convert causal `FundingRateObservation` sequence into `(timestamp, rate)` pairs."""
    cutoff = as_of.astimezone(UTC) if as_of is not None else None
    pairs: list[tuple[datetime, float]] = []
    for obs in observations:
        ts = obs.available_time if use_available_time else obs.observed_time
        if cutoff is not None and ts > cutoff:
            continue
        pairs.append((obs.observed_time, obs.funding_rate))
    pairs.sort(key=lambda item: item[0])
    return tuple(pairs)


async def sync_gate_funding_history(
    symbols: Sequence[str],
    *,
    client: GateFundingHistoryClient,
    repository: DerivativesRepository,
    days: int = 90,
    as_of: datetime | None = None,
) -> dict[str, int]:
    """Fetch paginated Gate funding history (>= 90d) and persist idempotently in `funding_rates`."""
    now = (as_of or datetime.now(UTC)).astimezone(UTC)
    counts: dict[str, int] = {}
    for raw_sym in symbols:
        mapping, contract = to_gate_contract_symbol(raw_sym)
        raw_items = await client.fetch_funding_history(
            contract,
            days=days,
            end_time=now,
        )
        observations = normalize_gate_funding_rate_payload(
            raw_items,
            symbol=mapping,
            provider_symbol=f"GATE:{mapping.base}{mapping.quote}.P",
            received_time=now,
            as_of=now,
            backfill=True,
        )
        inserted = repository.save_funding_observations(observations)
        counts[mapping.canonical_symbol] = inserted
    return counts
