"""Alternative.me Crypto Fear & Greed Index client, normalizer, and recorder (S-02)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from crypto_chatbot.schemas.news import FearGreedObservation
from crypto_chatbot.storage.repositories import FearGreedRepository

DEFAULT_FNG_API_URL = "https://api.alternative.me/fng/"

# Assumption for backfilled historical rows: Alternative.me publishes its daily
# Fear & Greed update at 00:00 UTC (the value's UTC midnight timestamp).
# Backfilled rows set available_time = observed_time + FNG_DAILY_PUBLICATION_OFFSET
# and is_backfilled = True; live fetches (backfill=False) keep available_time = fetch time.
FNG_DAILY_PUBLICATION_OFFSET = timedelta(0)


class _RetryableFngHttpError(RuntimeError):
    """Transient HTTP error eligible for bounded retry."""


def normalize_fear_greed_payload(
    payload: dict[str, Any],
    *,
    received_time: datetime,
    source: str = "alternative.me",
    as_of: datetime | None = None,
    backfill: bool = False,
) -> tuple[FearGreedObservation, ...]:
    """Normalize Alternative.me `/fng/` JSON payload into sorted UTC `FearGreedObservation`."""
    if received_time.tzinfo is None:
        raise ValueError("received_time must be timezone-aware")
    meta = payload.get("metadata")
    if isinstance(meta, dict) and meta.get("error"):
        raise ValueError(f"alternative.me error: {meta['error']}")
    data = payload.get("data")
    if not isinstance(data, list):
        raise ValueError("alternative.me payload missing 'data' list")

    received_utc = received_time.astimezone(UTC)
    cutoff_utc = (as_of or received_time).astimezone(UTC)
    by_time: dict[datetime, FearGreedObservation] = {}

    for item in data:
        if not isinstance(item, dict):
            continue
        raw_ts = item.get("timestamp")
        raw_val = item.get("value")
        raw_cls = item.get("value_classification")
        if raw_ts is None or raw_val is None or not raw_cls:
            continue
        observed_time = datetime.fromtimestamp(int(raw_ts), tz=UTC)
        if observed_time > cutoff_utc:
            continue
        val_int = int(raw_val)
        if val_int < 0 or val_int > 100:
            continue
        normalized = round((val_int - 50.0) / 50.0, 4)
        available_time = (
            observed_time + FNG_DAILY_PUBLICATION_OFFSET
            if backfill
            else max(observed_time, received_utc)
        )
        by_time[observed_time] = FearGreedObservation(
            source=source,
            observed_time=observed_time,
            received_time=received_utc,
            available_time=available_time,
            value=val_int,
            classification=str(raw_cls).strip(),
            normalized_score=normalized,
            is_backfilled=backfill,
            raw_payload=dict(item),
        )
    return tuple(by_time[ts] for ts in sorted(by_time))


class FearGreedClient:
    """Async public client for `https://api.alternative.me/fng/` (no API key required)."""

    def __init__(
        self,
        *,
        api_url: str = DEFAULT_FNG_API_URL,
        timeout_seconds: float = 15.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._api_url = api_url
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    @retry(
        reraise=True,
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=0.4, min=0.2, max=2.0),
        retry=retry_if_exception_type((httpx.TransportError, _RetryableFngHttpError)),
    )
    async def fetch_fng_payload(self, *, limit: int = 1) -> dict[str, Any]:
        """Fetch latest `limit` daily observations (`limit=0` retrieves full history)."""
        if limit < 0:
            raise ValueError("limit must be >= 0")
        response = await self._client.get(
            self._api_url,
            params={"limit": limit, "format": "json"},
        )
        if response.status_code in {429, 500, 502, 503, 504}:
            raise _RetryableFngHttpError(f"alternative.me HTTP {response.status_code}")
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("unexpected alternative.me JSON root type")
        return payload


@dataclass(frozen=True, slots=True)
class FearGreedSyncResult:
    """Summary of one Fear & Greed sync/backfill pass."""

    fetched_rows: int
    inserted_rows: int
    skipped_duplicates: int
    latest: FearGreedObservation | None


async def sync_fear_greed(
    *,
    client: FearGreedClient,
    repository: FearGreedRepository,
    limit: int = 0,
    as_of: datetime | None = None,
) -> FearGreedSyncResult:
    """Fetch and idempotently persist Fear & Greed daily observations (`limit=0` = backfill)."""
    now = (as_of or datetime.now(UTC)).astimezone(UTC)
    payload = await client.fetch_fng_payload(limit=limit)
    observations = normalize_fear_greed_payload(
        payload,
        received_time=now,
        as_of=now,
        backfill=(limit == 0 or limit > 1),
    )
    inserted = repository.save_observations(observations)
    latest = observations[-1] if observations else None
    return FearGreedSyncResult(
        fetched_rows=len(observations),
        inserted_rows=inserted,
        skipped_duplicates=max(0, len(observations) - inserted),
        latest=latest,
    )
