from __future__ import annotations

import gzip
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import httpx
import pytest
from sqlalchemy.dialects import postgresql
from sqlalchemy.dialects.postgresql import insert as pg_insert

from crypto_chatbot import cli
from crypto_chatbot.market_data.gate_history_ingest import (
    GateHistoryIngestor,
    iter_gatedata_trades_csv,
    iter_months,
    parse_gatedata_candlesticks_csv,
    parse_gatedata_funding_applies_csv,
    validate_candle_contiguity,
    validate_dataset_type,
)
from crypto_chatbot.market_data.symbols import resolve_default_perp_symbol
from crypto_chatbot.schemas.analysis import AnalysisSnapshot
from crypto_chatbot.schemas.derivatives import (
    FundingRateObservation,
    LiquidationEvent,
    OpenInterestObservation,
    OrderBookSnapshot,
)
from crypto_chatbot.schemas.market import MarketCandle, MarketType, QualityFlag, Timeframe
from crypto_chatbot.storage.models import GateIngestLedgerRow

# Verbatim recorded CSV lines from download.gatedata.org/futures_usdt (BTC_USDT 202610)
RECORDED_CANDLESTICKS_1H_CSV = [
    "1790812800,13530341,83472,83618.3,83381.2,83578.7",
    "1790816400,11514440,83479.9,83658.5,83441.6,83472",
    "1790820000,12013019,83445.6,83630.4,83361.6,83479.8",
    "1790823600,12452215,83726.5,83756,83390.6,83445",
]

RECORDED_FUNDING_APPLIES_CSV = [
    "1790812800,0.000088",
    "1790841600,0.000007",
    "1790870400,0.000014",
    "1790899200,-0.000005",
]

RECORDED_TRADES_CSV = [
    "1790812800.013589,844545515,83578.7,1",
    "1790812800.013603,844545516,83578.7,31",
    "1790812800.577279,844545519,83578.6,-3",
]


class _MemMarketRepo:
    def __init__(self) -> None:
        self.candles: dict[tuple[str, str, MarketType, Timeframe, datetime], MarketCandle] = {}

    def save_candles(self, candles: Sequence[MarketCandle]) -> int:
        inserted = 0
        for c in candles:
            key = (c.venue, c.canonical_symbol, c.market_type, c.timeframe, c.open_time)
            if key not in self.candles:
                self.candles[key] = c
                inserted += 1
        return inserted

    def load_candles(self, **kwargs: Any) -> tuple[MarketCandle, ...]:
        del kwargs
        return tuple(sorted(self.candles.values(), key=lambda c: c.open_time))

    def save_snapshot(self, snapshot: AnalysisSnapshot) -> None:
        del snapshot

    def get_snapshot(self, snapshot_id: str) -> AnalysisSnapshot | None:
        del snapshot_id
        return None


class _MemDerivRepo:
    def __init__(self) -> None:
        self.funding: dict[tuple[str, str, datetime], FundingRateObservation] = {}

    def save_funding_observations(self, observations: Sequence[FundingRateObservation]) -> int:
        inserted = 0
        for obs in observations:
            key = (obs.venue, obs.canonical_symbol, obs.observed_time)
            if key not in self.funding:
                self.funding[key] = obs
                inserted += 1
        return inserted

    def load_funding_observations(self, **kwargs: Any) -> tuple[FundingRateObservation, ...]:
        del kwargs
        return tuple(self.funding.values())

    def save_open_interest_observations(
        self, observations: Sequence[OpenInterestObservation]
    ) -> int:
        del observations
        return 0

    def load_open_interest_observations(self, **kwargs: Any) -> tuple[OpenInterestObservation, ...]:
        del kwargs
        return ()

    def save_order_book_snapshot(self, snapshot: OrderBookSnapshot) -> bool:
        del snapshot
        return False

    def load_order_book_snapshots(self, **kwargs: Any) -> tuple[OrderBookSnapshot, ...]:
        del kwargs
        return ()

    def save_liquidation_events(self, events: Sequence[LiquidationEvent]) -> int:
        del events
        return 0

    def load_liquidation_events(self, **kwargs: Any) -> tuple[LiquidationEvent, ...]:
        del kwargs
        return ()


class _MemLedgerRepo:
    def __init__(self) -> None:
        self.entries: dict[tuple[str, str, str], str] = {}

    def get_ingested_sha256(self, *, dataset: str, contract: str, yyyymm: str) -> str | None:
        return self.entries.get((dataset, contract, yyyymm))

    def record_ingested_file(
        self,
        *,
        dataset: str,
        contract: str,
        yyyymm: str,
        sha256: str,
        byte_size: int,
        local_path: str,
        row_count: int,
        inserted_rows: int,
        ingested_at: datetime,
    ) -> None:
        del byte_size, local_path, row_count, inserted_rows, ingested_at
        self.entries[(dataset, contract, yyyymm)] = sha256


def test_parse_gatedata_candlesticks_csv_matches_verified_column_order() -> None:
    symbol = resolve_default_perp_symbol("BTC")
    as_of = datetime.fromtimestamp(1790830000, tz=UTC)
    candles = parse_gatedata_candlesticks_csv(
        RECORDED_CANDLESTICKS_1H_CSV,
        symbol=symbol,
        contract_id="BTC_USDT",
        timeframe=Timeframe.H1,
        quanto_multiplier=Decimal("0.0001"),
        received_time=as_of,
    )
    assert len(candles) == 4
    first = candles[0]
    # Col 5 is OPEN (83578.7), Col 2 is CLOSE (83472) - empirically verified against Gate REST
    assert first.open == pytest.approx(83578.7)
    assert first.high == pytest.approx(83618.3)
    assert first.low == pytest.approx(83381.2)
    assert first.close == pytest.approx(83472.0)
    assert first.raw_contracts == pytest.approx(13530341.0)
    assert first.base_volume == pytest.approx(1353.0341)
    assert first.available_time == first.close_time
    assert QualityFlag.BACKFILL in first.quality_flags
    assert QualityFlag.QUOTE_VOL_ESTIMATED in first.quality_flags

    contig = validate_candle_contiguity(candles, timeframe=Timeframe.H1)
    assert contig.is_contiguous is True
    assert contig.available_eq_close_ok is True
    assert contig.backfill_flagged_ok is True


def test_validate_dataset_type_rejects_unavailable_404_intervals() -> None:
    assert validate_dataset_type("candlesticks_1h") == "candlesticks_1h"
    assert validate_dataset_type("trades") == "trades"
    with pytest.raises(ValueError, match="404"):
        validate_dataset_type("candlesticks_15m")
    with pytest.raises(ValueError, match="404"):
        validate_dataset_type("candlesticks_30m")


def test_iter_months_funding_and_trades_parsers() -> None:
    assert iter_months("202611", "202702") == ("202611", "202612", "202701", "202702")
    symbol = resolve_default_perp_symbol("BTC")
    as_of = datetime.fromtimestamp(1791000000, tz=UTC)
    funding = parse_gatedata_funding_applies_csv(
        RECORDED_FUNDING_APPLIES_CSV,
        symbol=symbol,
        contract_id="BTC_USDT",
        received_time=as_of,
    )
    assert len(funding) == 4
    assert funding[0].funding_rate == pytest.approx(0.000088)
    assert QualityFlag.BACKFILL in funding[0].quality_flags

    trades = list(iter_gatedata_trades_csv(RECORDED_TRADES_CSV))
    assert len(trades) == 3
    assert trades[0].is_taker_buy is True
    assert trades[0].abs_contracts == 1.0
    assert trades[2].is_taker_buy is False
    assert trades[2].abs_contracts == 3.0


@pytest.mark.asyncio
async def test_ingest_gate_history_idempotent_repeatable_and_zero_dupes(tmp_path: Path) -> None:
    downloads = 0
    candlesticks_gz = gzip.compress(("\n".join(RECORDED_CANDLESTICKS_1H_CSV) + "\n").encode())
    funding_gz = gzip.compress(("\n".join(RECORDED_FUNDING_APPLIES_CSV) + "\n").encode())
    trades_gz = gzip.compress(("\n".join(RECORDED_TRADES_CSV) + "\n").encode())

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal downloads
        downloads += 1
        url = str(request.url)
        if "/candlesticks_1h/202610/BTC_USDT-202610.csv.gz" in url:
            return httpx.Response(200, content=candlesticks_gz)
        if "/funding_applies/202610/BTC_USDT-202610.csv.gz" in url:
            return httpx.Response(200, content=funding_gz)
        if "/trades/202610/BTC_USDT-202610.csv.gz" in url:
            return httpx.Response(200, content=trades_gz)
        return httpx.Response(404)

    market_repo = _MemMarketRepo()
    deriv_repo = _MemDerivRepo()
    ledger_repo = _MemLedgerRepo()

    ingestor = GateHistoryIngestor(
        data_dir=tmp_path / "raw" / "gate",
        market_repository=market_repo,
        derivatives_repository=deriv_repo,
        ledger_repository=ledger_repo,
        multipliers={"BTC_USDT": Decimal("0.0001")},
        transport=httpx.MockTransport(handler),
    )
    as_of = datetime.fromtimestamp(1791000000, tz=UTC)
    try:
        first = await ingestor.ingest_range(
            symbols=("BTC",),
            from_ym="202610",
            to_ym="202610",
            datasets=("candlesticks_1h", "funding_applies", "trades"),
            as_of=as_of,
        )
        assert first.downloaded_files == 3
        assert first.inserted_rows == 8  # 4 candles + 4 funding rows
        assert len(market_repo.candles) == 4
        assert len(deriv_repo.funding) == 4
        assert all(r.contiguity_ok is not False for r in first.results)

        # Second run across exact same range -> 0 downloads, 0 inserted DB rows.
        second = await ingestor.ingest_range(
            symbols=("BTC",),
            from_ym="202610",
            to_ym="202610",
            datasets=("candlesticks_1h", "funding_applies", "trades"),
            as_of=as_of,
        )
        assert second.downloaded_files == 0
        assert second.skipped_files == 3
        assert second.inserted_rows == 0
        assert downloads == 3

        # Even if ledger were cleared, ON CONFLICT DO NOTHING semantics insert 0 new rows.
        ledger_repo.entries.clear()
        third = await ingestor.ingest_range(
            symbols=("BTC",),
            from_ym="202610",
            to_ym="202610",
            datasets=("candlesticks_1h", "funding_applies", "trades"),
            as_of=as_of,
        )
        assert third.downloaded_files == 0
        assert third.inserted_rows == 0
    finally:
        await ingestor.close()


def test_ledger_upsert_statement_compiles() -> None:
    stmt = (
        pg_insert(GateIngestLedgerRow)
        .values(dataset="candlesticks_1h", contract="BTC_USDT", yyyymm="202610")
        .on_conflict_do_nothing(constraint="uq_gate_ingest_file")
    )
    compiled = str(stmt.compile(dialect=postgresql.dialect()))  # type: ignore[no-untyped-call]
    assert "ON CONFLICT ON CONSTRAINT uq_gate_ingest_file DO NOTHING" in compiled


def test_cli_ingest_gate_history_dispatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    class _DummyEngine:
        def dispose(self) -> None:
            return None

    class _DummyContracts:
        async def fetch_contracts(self) -> dict[str, Decimal]:
            return {"BTC_USDT": Decimal("0.0001")}

        async def close(self) -> None:
            return None

    class _DummyIngestor:
        def __init__(self, **kwargs: Any) -> None:
            del kwargs

        async def ingest_range(self, **kwargs: Any) -> Any:
            del kwargs
            from crypto_chatbot.market_data.gate_history_ingest import BulkIngestSummary

            return BulkIngestSummary(
                files_processed=1,
                downloaded_files=1,
                skipped_files=0,
                inserted_rows=4,
                missing_files=0,
                results=(),
            )

        async def close(self) -> None:
            return None

    monkeypatch.setattr(cli, "create_db_engine", lambda url: _DummyEngine())
    monkeypatch.setattr(cli, "init_db", lambda engine: None)
    monkeypatch.setattr(cli, "PostgresMarketRepository", lambda engine: _MemMarketRepo())
    monkeypatch.setattr(cli, "PostgresDerivativesRepository", lambda engine: _MemDerivRepo())
    monkeypatch.setattr(cli, "PostgresGateIngestRepository", lambda engine: _MemLedgerRepo())

    import crypto_chatbot.market_data.gate_candles as gc
    import crypto_chatbot.market_data.gate_history_ingest as ghi

    monkeypatch.setattr(gc, "GateFuturesRestClient", lambda **kw: _DummyContracts())
    monkeypatch.setattr(ghi, "GateHistoryIngestor", _DummyIngestor)

    rc = cli.main(
        [
            "ingest-gate-history",
            "--symbols",
            "BTC",
            "--from",
            "202610",
            "--to",
            "202610",
            "--types",
            "candlesticks_1h",
            "--data-dir",
            str(tmp_path),
            "--database-url",
            "postgresql+psycopg://u:p@localhost/db",
            "--json",
        ]
    )
    assert rc == 0
    out = json.loads(capsys.readouterr().out)
    assert out["inserted_rows"] == 4
