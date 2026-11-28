from __future__ import annotations

import asyncio
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import httpx
import pytest
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.postgresql.base import PGDialect

from crypto_chatbot import cli
from crypto_chatbot.market_data.gate_contract_stats import (
    ContractStatsRecorder,
    ContractStatsScheduler,
    GateContractStatsClient,
    compute_symbol_gap_report,
    normalize_contract_stats_payload,
    to_gate_contract_symbol,
)
from crypto_chatbot.market_data.symbols import resolve_default_perp_symbol
from crypto_chatbot.schemas.derivatives import ContractStatsObservation
from crypto_chatbot.schemas.market import QualityFlag
from crypto_chatbot.storage.models import ContractStatsRow

LIVE_SAMPLE_ITEM: dict[str, Any] = {
    "time": 1795774500,
    "lsr_taker": 1.5311786953396056,
    "lsr_account": 1.4225679685620338,
    "long_liq_size": 0,
    "short_liq_size": 0,
    "open_interest": 637089156,
    "short_liq_usd": 0,
    "mark_price": 61637.84,
    "top_lsr_size": 0.9664237335561044,
    "top_long_size": 34459111,
    "top_short_size": 35656317,
    "short_liq_amount": 0,
    "long_liq_amount": 0,
    "short_liq_usd_new": 0,
    "long_liq_usd_new": 0,
    "open_interest_usd": 3926879946.326304,
    "top_lsr_account": 0.4302325581395349,
    "top_long_account": 37,
    "top_short_account": 86,
    "long_liq_usd": 0,
    "long_taker_size": 48987,
    "short_taker_size": 31993,
    "long_users": 17738,
    "short_users": 12469,
    "last_funding_rate": "0.0000234",
}


class InMemoryContractStatsRepository:
    """In-memory unique-constrained store mirroring Postgres ON CONFLICT DO NOTHING."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, str, str, datetime], ContractStatsObservation] = {}

    def save_contract_stats(self, observations: Sequence[ContractStatsObservation]) -> int:
        inserted = 0
        for obs in observations:
            key = (obs.venue, obs.canonical_symbol, obs.interval, obs.observed_time)
            if key not in self.rows:
                self.rows[key] = obs
                inserted += 1
        return inserted

    def load_contract_stats(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        interval: str = "5m",
        start_time: datetime | None = None,
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[ContractStatsObservation, ...]:
        matched = [
            obs
            for obs in self.rows.values()
            if obs.venue == venue
            and obs.canonical_symbol == canonical_symbol
            and obs.interval == interval
            and (start_time is None or obs.observed_time >= start_time)
            and (end_available_time is None or obs.available_time <= end_available_time)
        ]
        matched.sort(key=lambda item: item.observed_time)
        if limit is not None:
            matched = matched[-limit:]
        return tuple(matched)

    def list_contract_stats_times(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        interval: str = "5m",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> tuple[datetime, ...]:
        times = [
            obs.observed_time
            for obs in self.rows.values()
            if obs.venue == venue
            and obs.canonical_symbol == canonical_symbol
            and obs.interval == interval
            and (start_time is None or obs.observed_time >= start_time)
            and (end_time is None or obs.observed_time <= end_time)
        ]
        times.sort()
        return tuple(times)


def test_normalize_contract_stats_enforces_closed_bucket_and_extracts_all_fields() -> None:
    symbol = resolve_default_perp_symbol("BTC")
    open_bar = dict(LIVE_SAMPLE_ITEM)
    open_bar["time"] = 1795774800  # closes at 1795775100
    closed_as_of = datetime.fromtimestamp(1795774800, tz=UTC)

    normalized = normalize_contract_stats_payload(
        [LIVE_SAMPLE_ITEM, open_bar],
        symbol=symbol,
        provider_symbol="BTC_USDT",
        interval="5m",
        received_time=closed_as_of,
        as_of=closed_as_of,
        closed_only=True,
        backfill=True,
    )
    assert len(normalized) == 1
    obs = normalized[0]
    assert obs.observed_time == datetime.fromtimestamp(1795774500, tz=UTC)
    assert obs.available_time == datetime.fromtimestamp(1795774800, tz=UTC)
    assert obs.lsr_taker == pytest.approx(1.5311786953396056)
    assert obs.lsr_account == pytest.approx(1.4225679685620338)
    assert obs.top_lsr_account == pytest.approx(0.4302325581395349)
    assert obs.top_lsr_size == pytest.approx(0.9664237335561044)
    assert obs.open_interest == pytest.approx(637089156.0)
    assert obs.open_interest_usd == pytest.approx(3926879946.326304)
    assert obs.long_liq_usd == 0.0
    assert obs.short_liq_usd == 0.0
    assert QualityFlag.BACKFILL in obs.quality_flags
    assert obs.raw_payload == LIVE_SAMPLE_ITEM

    # Live poll (backfill=False) records available_time = fetch time (>= bar close) without BACKFILL
    late_fetch = closed_as_of + timedelta(minutes=2)
    live_norm = normalize_contract_stats_payload(
        [LIVE_SAMPLE_ITEM],
        symbol=symbol,
        provider_symbol="BTC_USDT",
        interval="5m",
        received_time=late_fetch,
        as_of=late_fetch,
        closed_only=True,
        backfill=False,
    )
    assert len(live_norm) == 1
    assert live_norm[0].available_time == late_fetch
    assert QualityFlag.BACKFILL not in live_norm[0].quality_flags


@pytest.mark.asyncio
async def test_repeatable_backfill_inserts_zero_duplicates() -> None:
    anchor = datetime(2026, 11, 27, 10, 30, tzinfo=UTC)
    start_epoch = int((anchor - timedelta(hours=2)).timestamp())
    requested_froms: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        params = dict(request.url.params)
        contract = params["contract"]
        from_ts = int(params["from"])
        requested_froms.append(from_ts)
        # Return 12 closed bars per symbol split into two pages of 6 when page_limit=6
        all_bars = [
            {
                **LIVE_SAMPLE_ITEM,
                "time": start_epoch + i * 300,
                "open_interest": 1000 + i if contract == "BTC_USDT" else 500 + i,
            }
            for i in range(12)
        ]
        page = [item for item in all_bars if item["time"] >= from_ts][: int(params["limit"])]
        return httpx.Response(200, json=page)

    client = GateContractStatsClient(transport=httpx.MockTransport(handler))
    repo = InMemoryContractStatsRepository()
    recorder = ContractStatsRecorder(
        client=client,
        repository=repo,
        symbols=("BTC", "ETH"),
        interval="5m",
    )
    try:
        first = await recorder.backfill_history(lookback_days=1, as_of=anchor, page_limit=6)
        second = await recorder.backfill_history(lookback_days=1, as_of=anchor, page_limit=6)
    finally:
        await client.close()

    assert not first.errors
    assert first.symbols_processed == 2
    assert first.inserted_rows == 24
    assert first.skipped_duplicates == 0

    assert not second.errors
    assert second.fetched_rows == 24
    assert second.inserted_rows == 0
    assert second.skipped_duplicates == 24
    # Gate rejects requests > 180d ago; ensure clamping keeps every query within window
    assert all(ts >= int((anchor - timedelta(days=180)).timestamp()) for ts in requested_froms)


def test_gap_report_identifies_missing_bars_accurately() -> None:
    t0 = datetime(2026, 11, 27, 9, 0, tzinfo=UTC)
    window_end = datetime(2026, 11, 27, 9, 30, tzinfo=UTC)
    # 6 expected 5m bars: 09:00, 09:05, 09:10, 09:15, 09:20, 09:25
    # Drop 09:10 and 09:15 (contiguous 2-bar gap)
    observed = [
        t0,
        t0 + timedelta(minutes=5),
        t0 + timedelta(minutes=20),
        t0 + timedelta(minutes=25),
    ]
    report = compute_symbol_gap_report(
        observed,
        venue="gateio",
        canonical_symbol="BTC/USDT:USDT",
        interval="5m",
        window_start=t0,
        window_end=window_end,
    )
    assert report.expected_bars == 6
    assert report.observed_bars == 4
    assert report.missing_bars == 2
    assert report.coverage_pct == pytest.approx(66.67)
    assert len(report.gaps) == 1
    assert report.gaps[0].start_time == t0 + timedelta(minutes=10)
    assert report.gaps[0].end_time == t0 + timedelta(minutes=15)
    assert report.gaps[0].missing_bars == 2


@pytest.mark.asyncio
async def test_scheduler_runs_startup_backfill_poll_and_stops_cleanly() -> None:
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(str(request.url))
        # Use timestamp safely in the past so closed_only accepts it
        past_epoch = int((datetime.now(UTC) - timedelta(minutes=15)).timestamp() // 300) * 300
        return httpx.Response(200, json=[{**LIVE_SAMPLE_ITEM, "time": past_epoch}])

    client = GateContractStatsClient(transport=httpx.MockTransport(handler))
    repo = InMemoryContractStatsRepository()
    recorder = ContractStatsRecorder(client=client, repository=repo, symbols=("BTC",))
    scheduler = ContractStatsScheduler(
        recorder,
        interval_seconds=60,
        backfill_days=1,
        run_backfill_on_start=True,
    )
    try:
        scheduler.start()
        for _ in range(30):
            if len(calls) >= 2:
                break
            await asyncio.sleep(0.02)
        await scheduler.stop()
    finally:
        await client.close()

    assert len(calls) >= 2
    assert len(repo.rows) == 1


def test_postgres_insert_on_conflict_stmt_compiles() -> None:
    stmt = (
        pg_insert(ContractStatsRow)
        .values([{"venue": "gateio", "canonical_symbol": "BTC/USDT:USDT"}])
        .on_conflict_do_nothing(constraint="uq_contract_stats_identity")
        .returning(ContractStatsRow.id)
    )
    compiled = str(stmt.compile(dialect=PGDialect()))  # type: ignore[no-untyped-call]
    assert "ON CONFLICT ON CONSTRAINT uq_contract_stats_identity DO NOTHING" in compiled
    assert "RETURNING contract_stats_observations.id" in compiled


def test_to_gate_contract_symbol_normalizes_formats() -> None:
    for raw in ("BTC", "btc/usdt", "BTC/USDT:USDT", "BTC_USDT"):
        mapping, contract_id = to_gate_contract_symbol(raw)
        assert mapping.canonical_symbol == "BTC/USDT:USDT"
        assert contract_id == "BTC_USDT"


def test_cli_contract_stats_gap_report_json(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repo = InMemoryContractStatsRepository()
    mapping, contract = to_gate_contract_symbol("BTC")
    now = datetime.now(UTC).replace(second=0, microsecond=0)
    bar_ts = now - timedelta(minutes=(now.minute % 5) + 10)
    repo.save_contract_stats(
        normalize_contract_stats_payload(
            [{**LIVE_SAMPLE_ITEM, "time": int(bar_ts.timestamp())}],
            symbol=mapping,
            provider_symbol=contract,
            received_time=now,
            as_of=now,
        )
    )

    class _DummyEngine:
        def dispose(self) -> None:
            return None

    monkeypatch.setattr(cli, "create_db_engine", lambda url: _DummyEngine())
    monkeypatch.setattr(cli, "init_db", lambda engine: None)
    monkeypatch.setattr(cli, "PostgresContractStatsRepository", lambda engine: repo)

    rc = cli.main(
        [
            "contract-stats",
            "gap-report",
            "--symbols",
            "BTC",
            "--days",
            "1",
            "--database-url",
            "postgresql+psycopg://user:pass@localhost/testdb",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["venue"] == "gateio"
    assert out["interval"] == "5m"
    assert len(out["symbols"]) == 1
    assert out["symbols"][0]["observed_bars"] >= 1
