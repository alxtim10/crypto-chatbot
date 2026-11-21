from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from crypto_chatbot.analysis.service import AnalysisService
from crypto_chatbot.market_data.base import DEFAULT_CANDLE_LIMIT, OhlcvBatch
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import resolve_mvp_spot_symbol
from crypto_chatbot.schemas.market import MarketType, Timeframe
from crypto_chatbot.storage.database import create_db_engine, init_db
from crypto_chatbot.storage.repositories import PostgresMarketRepository

FIXTURE_1000 = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"


def _run(cmd: list[str], *, env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, text=True, capture_output=True, env=env)  # noqa: S603


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def pg_repository(tmp_path: Path) -> Iterator[PostgresMarketRepository]:
    configured_url = os.environ.get("TEST_DATABASE_URL", "").strip()
    if configured_url:
        engine = create_db_engine(configured_url)
        init_db(engine)
        yield PostgresMarketRepository(engine)
        engine.dispose()
        return

    initdb_bin = shutil.which("initdb")
    pg_ctl_bin = shutil.which("pg_ctl")
    psql_bin = shutil.which("psql")
    if not initdb_bin or not pg_ctl_bin or not psql_bin:
        pytest.skip("PostgreSQL binaries (initdb/pg_ctl/psql) are not available")

    data_dir = tmp_path / "pgdata"
    # macOS limits Unix socket paths to ~103 bytes, so keep sock_dir short under /tmp.
    sock_dir = Path(tempfile.mkdtemp(prefix="ccpg_", dir="/tmp"))
    log_file = tmp_path / "postgres.log"
    port = _free_port()

    try:
        init_res = _run(
            [
                initdb_bin,
                "-D",
                str(data_dir),
                "-U",
                "postgres",
                "-A",
                "trust",
                "--encoding=UTF8",
                "--locale=C",
            ]
        )
        if init_res.returncode != 0:
            pytest.skip(f"initdb failed in current environment: {init_res.stderr.strip()}")

        start_res = _run(
            [
                pg_ctl_bin,
                "-D",
                str(data_dir),
                "-l",
                str(log_file),
                "-o",
                f"-k {sock_dir} -h '' -p {port}",
                "-w",
                "start",
            ]
        )
        if start_res.returncode != 0:
            pg_log = log_file.read_text(encoding="utf-8", errors="replace").strip()
            pytest.skip(f"pg_ctl start failed: {start_res.stderr.strip()} | {pg_log}")

        try:
            db_res = _run(
                [
                    psql_bin,
                    "-h",
                    str(sock_dir),
                    "-p",
                    str(port),
                    "-U",
                    "postgres",
                    "-d",
                    "postgres",
                    "-c",
                    "CREATE DATABASE crypto_chatbot_test;",
                ]
            )
            if db_res.returncode != 0:
                pytest.skip(f"CREATE DATABASE failed: {db_res.stderr.strip()}")

            url = (
                f"postgresql+psycopg://postgres@localhost:{port}/crypto_chatbot_test"
                f"?host={sock_dir}"
            )
            engine = create_db_engine(url)
            init_db(engine)
            try:
                yield PostgresMarketRepository(engine)
            finally:
                engine.dispose()
        finally:
            _run([pg_ctl_bin, "-D", str(data_dir), "-m", "immediate", "-w", "stop"])
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)


class _StaticProvider:
    def __init__(self, batch: OhlcvBatch) -> None:
        self._batch = batch

    async def get_closed_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = DEFAULT_CANDLE_LIMIT,
    ) -> OhlcvBatch:
        return self._batch

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_postgres_repository_round_trip_and_recomputation(
    pg_repository: PostgresMarketRepository,
) -> None:
    data = json.loads(FIXTURE_1000.read_text(encoding="utf-8"))
    raw = data["raw_ohlcv"]
    last_open = datetime.fromtimestamp(float(raw[-1][0]) / 1000.0, tz=UTC)
    received = last_open + Timeframe.H4.duration + timedelta(seconds=2)
    batch = normalize_ohlcv_rows(
        raw,
        symbol=resolve_mvp_spot_symbol("BTC"),
        timeframe=Timeframe.H4,
        received_time=received,
        now=received,
    )

    inserted_first = pg_repository.save_candles(batch.candles)
    inserted_duplicate = pg_repository.save_candles(batch.candles[:50])
    assert inserted_first == 1000
    assert inserted_duplicate == 0

    loaded_tail = pg_repository.load_candles(
        venue="binance",
        canonical_symbol="BTC/USDT",
        market_type=MarketType.SPOT,
        timeframe=Timeframe.H4,
        limit=10,
    )
    assert len(loaded_tail) == 10
    assert [c.open_time for c in loaded_tail] == [c.open_time for c in batch.candles[-10:]]

    service = AnalysisService(provider=_StaticProvider(batch), repository=pg_repository)
    original = await service.analyze("BTC", Timeframe.H4, include_mtf=False, persist=True)
    # Re-saving same snapshot is idempotent
    pg_repository.save_snapshot(original)

    fetched = pg_repository.get_snapshot(original.snapshot_id)
    assert fetched == original
    assert pg_repository.get_snapshot("snap-nonexistent") is None

    recomputed = service.recompute_from_stored(original.snapshot_id)
    assert recomputed.input_range == original.input_range
    assert recomputed.facts == original.facts
    assert recomputed.technical == original.technical
    assert recomputed.snapshot_id == original.snapshot_id
