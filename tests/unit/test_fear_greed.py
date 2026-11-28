from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert as pg_insert

from crypto_chatbot import cli
from crypto_chatbot.news.fear_greed import (
    FearGreedClient,
    normalize_fear_greed_payload,
    sync_fear_greed,
)
from crypto_chatbot.schemas.news import FearGreedObservation
from crypto_chatbot.storage.models import FearGreedRow

# Verbatim recorded sample from https://api.alternative.me/fng/?limit=5&format=json
RECORDED_FNG_PAYLOAD: dict[str, Any] = {
    "name": "Fear and Greed Index",
    "data": [
        {
            "value": "12",
            "value_classification": "Extreme Fear",
            "timestamp": "1795737600",
            "time_until_update": "49170",
        },
        {
            "value": "11",
            "value_classification": "Extreme Fear",
            "timestamp": "1795651200",
        },
        {
            "value": "13",
            "value_classification": "Extreme Fear",
            "timestamp": "1795564800",
        },
    ],
    "metadata": {"error": None},
}


class _MemFearGreedRepo:
    def __init__(self) -> None:
        self.rows: dict[tuple[str, datetime], FearGreedObservation] = {}

    def save_observations(self, observations: Sequence[FearGreedObservation]) -> int:
        inserted = 0
        for obs in observations:
            key = (obs.source, obs.observed_time)
            if key not in self.rows:
                self.rows[key] = obs
                inserted += 1
        return inserted

    def load_observations(
        self,
        *,
        source: str = "alternative.me",
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[FearGreedObservation, ...]:
        matched = [
            obs
            for obs in self.rows.values()
            if obs.source == source
            and (end_available_time is None or obs.available_time <= end_available_time)
        ]
        matched.sort(key=lambda o: o.observed_time)
        if limit is not None:
            matched = matched[-limit:]
        return tuple(matched)


def test_normalize_fear_greed_payload_and_causal_cutoff() -> None:
    as_of = datetime.fromtimestamp(1795651200, tz=UTC)
    obs = normalize_fear_greed_payload(
        RECORDED_FNG_PAYLOAD,
        received_time=as_of,
        as_of=as_of,
        backfill=True,
    )
    assert len(obs) == 2  # 1795737600 strictly excluded at 1795651200
    assert obs[0].value == 13
    assert obs[-1].value == 11
    assert obs[-1].classification == "Extreme Fear"
    assert obs[-1].normalized_score == pytest.approx((11.0 - 50.0) / 50.0)
    assert obs[-1].is_backfilled is True
    # Backfilled rows use publication time (00:00 UTC value date), NOT fetch time
    assert obs[0].available_time == obs[0].observed_time
    assert obs[-1].available_time == obs[-1].observed_time

    # Live fetch (backfill=False) keeps available_time = fetch time (received_time)
    fetch_time = as_of + timedelta(hours=3, minutes=15)
    live_obs = normalize_fear_greed_payload(
        RECORDED_FNG_PAYLOAD,
        received_time=fetch_time,
        as_of=as_of,
        backfill=False,
    )
    assert live_obs[-1].is_backfilled is False
    assert live_obs[-1].available_time == fetch_time


@pytest.mark.asyncio
async def test_sync_fear_greed_backfill_is_idempotent() -> None:
    anchor = datetime.fromtimestamp(1795800000, tz=UTC)

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.params["limit"] == "0"
        return httpx.Response(200, json=RECORDED_FNG_PAYLOAD)

    client = FearGreedClient(transport=httpx.MockTransport(handler))
    repo = _MemFearGreedRepo()
    try:
        first = await sync_fear_greed(client=client, repository=repo, limit=0, as_of=anchor)
        second = await sync_fear_greed(client=client, repository=repo, limit=0, as_of=anchor)
    finally:
        await client.close()

    assert first.fetched_rows == 3
    assert first.inserted_rows == 3
    assert first.skipped_duplicates == 0
    assert first.latest is not None
    assert first.latest.value == 12

    assert second.fetched_rows == 3
    assert second.inserted_rows == 0
    assert second.skipped_duplicates == 3


def test_fear_greed_pg_insert_compiles() -> None:
    stmt = (
        pg_insert(FearGreedRow)
        .values(source="alternative.me", observed_time=datetime.now(UTC))
        .on_conflict_do_nothing(constraint="uq_fear_greed_source_time")
        .returning(FearGreedRow.id)
    )
    compiled = str(stmt.compile(dialect=postgresql.dialect()))  # type: ignore[no-untyped-call]
    assert "ON CONFLICT ON CONSTRAINT uq_fear_greed_source_time DO NOTHING" in compiled


def test_cli_sync_fear_greed_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _DummyEngine:
        def dispose(self) -> None:
            return None

    class _FakeClient:
        async def fetch_fng_payload(self, *, limit: int = 1) -> dict[str, Any]:
            del limit
            return RECORDED_FNG_PAYLOAD

        async def close(self) -> None:
            return None

    repo = _MemFearGreedRepo()
    monkeypatch.setattr(cli, "create_db_engine", lambda url: _DummyEngine())
    monkeypatch.setattr(cli, "init_db", lambda engine: None)
    monkeypatch.setattr(cli, "PostgresFearGreedRepository", lambda engine: repo)

    import crypto_chatbot.news.fear_greed as fg

    monkeypatch.setattr(fg, "FearGreedClient", _FakeClient)

    rc = cli.main(
        [
            "sync-fear-greed",
            "--limit",
            "0",
            "--database-url",
            "postgresql+psycopg://u:p@localhost/db",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["inserted_rows"] == 3
