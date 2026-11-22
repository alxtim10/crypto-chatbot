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

from crypto_chatbot.analysis.service import AnalysisService, build_analysis_snapshot
from crypto_chatbot.analysis.setup import evaluate_setup_against_candles
from crypto_chatbot.market_data.base import DEFAULT_CANDLE_LIMIT, OhlcvBatch
from crypto_chatbot.market_data.derivatives_quality import (
    normalize_funding_rate_payload,
    normalize_liquidation_events,
    normalize_open_interest_payload,
)
from crypto_chatbot.market_data.orderbook import normalize_order_book_payload
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import resolve_mvp_spot_symbol, resolve_symbol
from crypto_chatbot.market_data.tradingview import parse_ai_agent_lens_packet
from crypto_chatbot.ml.experiment import build_model_artifact, run_offline_experiment
from crypto_chatbot.ml.shadow import (
    ShadowInferenceEngine,
    compute_shadow_monitoring_report,
    settle_shadow_prediction,
)
from crypto_chatbot.news.pipeline import build_sentiment_observations, deduplicate_articles
from crypto_chatbot.schemas.market import MarketType, Timeframe
from crypto_chatbot.schemas.ml import MlExperimentConfig, MlInferenceMode
from crypto_chatbot.schemas.news import RawNewsItem
from crypto_chatbot.storage.database import create_db_engine, init_db
from crypto_chatbot.storage.models import Base
from crypto_chatbot.storage.repositories import (
    LlmInteractionRecord,
    PostgresDerivativesRepository,
    PostgresMarketRepository,
    PostgresMlExperimentRepository,
    PostgresModelPredictionRepository,
    PostgresNewsRepository,
    PostgresSetupLogRepository,
    PostgresTradingViewRepository,
    PostgresUserMemoryRepository,
    UserPreferences,
)

FIXTURE_1000 = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"
DERIV_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "binance_btc_derivatives_sample.json"
)


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
        Base.metadata.drop_all(engine)
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


def test_postgres_user_memory_isolation_forget_and_retention(
    pg_repository: PostgresMarketRepository,
) -> None:
    mem_repo = PostgresUserMemoryRepository(pg_repository._engine)
    now = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)

    # Large 64-bit Telegram IDs supported via BigInteger
    user_a = 9_123_456_789
    user_b = 9_987_654_321

    mem_repo.save_preferences(
        user_a,
        UserPreferences(watchlist=("BTC/USDT", "SOL/USDT"), default_timeframe=Timeframe.H4),
    )
    mem_repo.save_preferences(
        user_b,
        UserPreferences(watchlist=("ETH/USDT",), default_timeframe=Timeframe.D1),
    )

    assert mem_repo.get_preferences(user_a).watchlist == ("BTC/USDT", "SOL/USDT")
    assert mem_repo.get_preferences(user_b).watchlist == ("ETH/USDT",)

    # Append 25 messages for user_a -> capped strictly at 20
    for idx in range(25):
        mem_repo.append_message(
            user_a,
            role="user" if idx % 2 == 0 else "assistant",
            content=f"msg-{idx}",
            snapshot_id="snap-a" if idx % 2 == 1 else None,
            max_messages=20,
            created_at=now - timedelta(minutes=25 - idx),
        )
    recent_a = mem_repo.get_recent_messages(user_a, limit=20)
    assert len(recent_a) == 20
    assert recent_a[0].content == "msg-5"
    assert recent_a[-1].content == "msg-24"

    # Old expired message + interaction for user_b (45 days old)
    mem_repo.append_message(
        user_b,
        role="user",
        content="expired-msg",
        created_at=now - timedelta(days=45),
    )
    mem_repo.append_message(
        user_b,
        role="user",
        content="fresh-msg",
        created_at=now - timedelta(days=1),
    )
    mem_repo.record_llm_interaction(
        LlmInteractionRecord(
            telegram_user_id=user_a,
            request_id="req-a",
            snapshot_id="snap-a",
            model_id="gemini-flash",
            prompt_version="v1.0.0",
            validation_passed=True,
            latency_ms=100,
            created_at=now,
        )
    )
    mem_repo.record_llm_interaction(
        LlmInteractionRecord(
            telegram_user_id=user_b,
            request_id="req-b-old",
            snapshot_id="snap-b",
            model_id="gemini-flash",
            prompt_version="v1.0.0",
            validation_passed=True,
            latency_ms=100,
            created_at=now - timedelta(days=45),
        )
    )

    # /forget for user_a deletes user_a data only, leaving user_b intact
    forget_res = mem_repo.forget_user_data(user_a)
    assert forget_res.deleted_messages == 20
    assert forget_res.deleted_conversations == 1
    assert forget_res.deleted_preferences == 1
    assert forget_res.deleted_interactions == 1
    assert mem_repo.get_recent_messages(user_a) == ()
    assert mem_repo.get_preferences(user_a).watchlist == ()
    assert len(mem_repo.get_recent_messages(user_b)) == 2
    assert mem_repo.get_preferences(user_b).watchlist == ("ETH/USDT",)

    # Purge expired removes >30d message and interaction, preserving fresh message & prefs
    purge_res = mem_repo.purge_expired(now=now)
    assert purge_res.deleted_messages == 1
    assert purge_res.deleted_interactions == 1
    remaining_b = mem_repo.get_recent_messages(user_b)
    assert [m.content for m in remaining_b] == ["fresh-msg"]
    assert mem_repo.get_preferences(user_b).watchlist == ("ETH/USDT",)


def test_postgres_derivatives_repository_persistence_and_point_in_time(
    pg_repository: PostgresMarketRepository,
) -> None:
    repo = PostgresDerivativesRepository(pg_repository._engine)
    raw = json.loads(DERIV_FIXTURE.read_text(encoding="utf-8"))
    received = datetime.fromisoformat(raw["received_time"])
    perp = resolve_symbol("BTC", market_type=MarketType.PERPETUAL)
    spot = resolve_symbol("BTC", market_type=MarketType.SPOT)

    funding_batch = normalize_funding_rate_payload(
        raw["funding_rate"],
        symbol=perp,
        received_time=received,
        funding_interval_hours=raw["funding_interval_hours"],
        now=received,
    )
    assert funding_batch.observation is not None
    assert repo.save_funding_observations([funding_batch.observation]) == 1
    assert repo.save_funding_observations([funding_batch.observation]) == 0
    loaded_funding = repo.load_funding_observations(
        venue="binance",
        canonical_symbol=perp.canonical_symbol,
        end_available_time=received,
    )
    assert len(loaded_funding) == 1
    assert loaded_funding[0].rate_per_8h == funding_batch.observation.rate_per_8h
    assert (
        repo.load_funding_observations(
            venue="binance",
            canonical_symbol=perp.canonical_symbol,
            end_available_time=received - timedelta(minutes=5),
        )
        == ()
    )

    oi_batch = normalize_open_interest_payload(
        raw["open_interest"]["latest"],
        raw["open_interest"]["history"],
        symbol=perp,
        timeframe=Timeframe.H4,
        received_time=received,
    )
    assert oi_batch.latest is not None
    oi_all = [oi_batch.latest, *oi_batch.history]
    assert repo.save_open_interest_observations(oi_all) == len(oi_all)
    assert repo.save_open_interest_observations(oi_all) == 0
    loaded_oi_hist = repo.load_open_interest_observations(
        venue="binance",
        canonical_symbol=perp.canonical_symbol,
        period="4h",
    )
    assert len(loaded_oi_hist) == 2

    ob_batch = normalize_order_book_payload(
        raw["order_book"],
        symbol=spot,
        received_time=received,
    )
    assert ob_batch.snapshot is not None
    assert repo.save_order_book_snapshot(ob_batch.snapshot) is True
    assert repo.save_order_book_snapshot(ob_batch.snapshot) is False
    loaded_books = repo.load_order_book_snapshots(
        venue="binance",
        canonical_symbol=spot.canonical_symbol,
        market_type=MarketType.SPOT,
    )
    assert len(loaded_books) == 1
    assert loaded_books[0].sequence_id == 500100

    liq_batch = normalize_liquidation_events(
        raw["liquidations"],
        symbol=perp,
        received_time=received,
        source_available=True,
    )
    assert repo.save_liquidation_events(liq_batch.events) == 2
    assert repo.save_liquidation_events(liq_batch.events) == 0
    loaded_liqs = repo.load_liquidation_events(
        venue="binance",
        canonical_symbol=perp.canonical_symbol,
    )
    assert len(loaded_liqs) == 2


def test_postgres_setup_log_repository_lifecycle(
    pg_repository: PostgresMarketRepository,
) -> None:
    setup_repo = PostgresSetupLogRepository(pg_repository._engine)
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
    snap = build_analysis_snapshot(batch)
    assert snap.trade_setup is not None

    assert setup_repo.save_setup(snap, snap.trade_setup) is True
    assert setup_repo.save_setup(snap, snap.trade_setup) is False

    loaded = setup_repo.get_setup(snap.snapshot_id)
    assert loaded is not None
    assert loaded.snapshot_id == snap.snapshot_id

    recent = setup_repo.list_recent_setups(canonical_symbol="BTC/USDT", limit=5)
    assert len(recent) == 1

    # Verify update_settlement transitions outcome cleanly
    assert (
        setup_repo.update_settlement(
            snap.snapshot_id,
            outcome_status="hit_tp1",
            settled_at=received + timedelta(hours=4),
            realized_r=1.0,
        )
        is True
    )
    updated = setup_repo.get_setup(snap.snapshot_id)
    assert updated is not None
    assert updated.outcome_status == "hit_tp1"
    assert updated.realized_r == 1.0
    assert setup_repo.list_pending_setups() == ()
    assert evaluate_setup_against_candles(updated, ()) == ("pending", None, None) or (
        updated.status != "ready"
    )


def test_postgres_tradingview_repository_persistence_and_dedup(
    pg_repository: PostgresMarketRepository,
) -> None:
    fixture_path = (
        Path(__file__).resolve().parents[1] / "fixtures" / "tradingview_ai_agent_lens_sample.json"
    )
    raw = json.loads(fixture_path.read_text(encoding="utf-8"))
    raw["secret"] = "tv-secret-never-store"
    received = datetime(2026, 10, 14, 16, 0, 5, tzinfo=UTC)
    packet = parse_ai_agent_lens_packet(raw, received_time=received, now=received)

    tv_repo = PostgresTradingViewRepository(pg_repository._engine)
    assert tv_repo.save_packet(packet) is True
    assert tv_repo.save_packet(packet) is False

    # Point-in-time query excludes packets not yet available.
    before = tv_repo.load_latest_packet(
        venue="binance",
        canonical_symbol="BTC/USDT",
        timeframe=packet.timeframe,
        end_available_time=packet.available_time - timedelta(seconds=1),
    )
    assert before is None

    loaded = tv_repo.load_latest_packet(
        venue="binance",
        canonical_symbol="BTC/USDT",
        timeframe=packet.timeframe,
        end_available_time=packet.available_time,
    )
    assert loaded is not None
    assert loaded.idempotency_key == packet.idempotency_key
    assert loaded.is_probabilistic is False
    assert "secret" not in loaded.sanitized_payload
    assert "tv-secret-never-store" not in loaded.model_dump_json()

    packets = tv_repo.load_packets(
        venue="binance",
        canonical_symbol="BTC/USDT",
        timeframe=packet.timeframe,
        end_available_time=packet.available_time,
    )
    assert len(packets) == 1
    assert packets[0].idempotency_key == packet.idempotency_key


def test_postgres_news_sentiment_repository_dedup_and_point_in_time(
    pg_repository: PostgresMarketRepository,
) -> None:
    fixture_path = Path(__file__).resolve().parents[1] / "fixtures" / "news_sentiment_sample.json"
    raw_items = [
        RawNewsItem.model_validate(item)
        for item in json.loads(fixture_path.read_text(encoding="utf-8"))
    ]
    articles, dropped = deduplicate_articles(raw_items)
    assert len(articles) == 5
    assert dropped == 2

    observations = build_sentiment_observations(articles)
    news_repo = PostgresNewsRepository(pg_repository._engine)

    assert news_repo.save_articles(articles) == 5
    assert news_repo.save_articles(articles) == 0

    inserted_obs = news_repo.save_observations(observations)
    assert inserted_obs == len(observations)
    assert news_repo.save_observations(observations) == 0

    anchor = datetime(2026, 10, 14, 16, 0, 0, tzinfo=UTC)
    causal_btc = news_repo.load_observations(
        canonical_symbol="BTC/USDT",
        start_available_time=anchor - timedelta(hours=24),
        end_available_time=anchor,
    )
    assert len(causal_btc) == 3
    assert all(obs.available_time <= anchor for obs in causal_btc)
    assert all(obs.is_probabilistic is False for obs in causal_btc)

    later_btc = news_repo.load_observations(
        canonical_symbol="BTC/USDT",
        start_available_time=anchor - timedelta(hours=24),
        end_available_time=anchor + timedelta(minutes=10),
    )
    assert len(later_btc) == 4
    assert news_repo.load_observations(canonical_symbol="SOL/USDT") == ()


def test_postgres_ml_experiment_repository_round_trip(
    pg_repository: PostgresMarketRepository,
) -> None:
    raw = json.loads(FIXTURE_1000.read_text(encoding="utf-8"))
    received = datetime(2026, 10, 14, 16, 0, 1, tzinfo=UTC)
    batch = normalize_ohlcv_rows(
        raw["raw_ohlcv"],
        symbol=resolve_mvp_spot_symbol("BTC"),
        timeframe=Timeframe.H4,
        received_time=received,
        now=received,
    )
    config = MlExperimentConfig(canonical_symbol="BTC/USDT", timeframe=Timeframe.H4)
    report = run_offline_experiment(batch.candles, config=config)

    ml_repo = PostgresMlExperimentRepository(pg_repository._engine)
    assert ml_repo.save_experiment(report) is True
    assert ml_repo.save_experiment(report) is False

    loaded = ml_repo.get_experiment(report.experiment_id)
    assert loaded == report
    listed = ml_repo.list_experiments(canonical_symbol="BTC/USDT", timeframe=Timeframe.H4)
    assert len(listed) == 1
    assert listed[0].experiment_id == report.experiment_id


@pytest.mark.asyncio
async def test_postgres_model_prediction_repository_lifecycle(
    pg_repository: PostgresMarketRepository,
) -> None:
    raw = json.loads(FIXTURE_1000.read_text(encoding="utf-8"))
    received = datetime(2026, 10, 14, 16, 0, 1, tzinfo=UTC)
    batch = normalize_ohlcv_rows(
        raw["raw_ohlcv"],
        symbol=resolve_mvp_spot_symbol("BTC"),
        timeframe=Timeframe.H4,
        received_time=received,
        now=received,
    )
    config = MlExperimentConfig(canonical_symbol="BTC/USDT", timeframe=Timeframe.H4)
    artifact, _ = build_model_artifact(batch.candles, config=config)
    promoted_artifact = artifact.model_copy(update={"accepted": True})

    horizon_bars = config.horizon_hours // 4
    prefix_candles = batch.candles[:-horizon_bars]
    prefix_batch = batch.model_copy(update={"candles": prefix_candles})

    class _FixtureProvider:
        async def get_closed_ohlcv(
            self,
            symbol: str,
            timeframe: Timeframe,
            limit: int = DEFAULT_CANDLE_LIMIT,
        ) -> OhlcvBatch:
            _ = symbol, limit
            if timeframe is Timeframe.H4:
                return prefix_batch
            return prefix_batch.model_copy(update={"timeframe": timeframe, "candles": ()})

        async def close(self) -> None:
            return None

    pred_repo = PostgresModelPredictionRepository(pg_repository._engine)
    shadow_engine = ShadowInferenceEngine(promoted_artifact, mode=MlInferenceMode.SHADOW)
    service = AnalysisService(
        provider=_FixtureProvider(),
        repository=pg_repository,
        shadow_engine=shadow_engine,
        prediction_repository=pred_repo,
    )

    snapshot = await service.analyze("BTC", Timeframe.H4, include_mtf=False, persist=True)
    assert snapshot.ml_prediction is None

    saved_list = pred_repo.list_predictions(canonical_symbol="BTC/USDT", timeframe=Timeframe.H4)
    assert len(saved_list) == 1
    rec = saved_list[0]
    assert rec.exposed_to_user is False
    assert rec.snapshot_id == snapshot.snapshot_id
    assert pred_repo.save_prediction(rec) is False

    unsettled = pred_repo.list_unsettled_predictions(canonical_symbol="BTC/USDT")
    assert len(unsettled) == 1
    assert unsettled[0].prediction_id == rec.prediction_id

    fwd_ret, label, settled_at = settle_shadow_prediction(
        rec,
        batch.candles,
        cost_assumptions=config.cost_assumptions,
        noise_buffer_bps=config.noise_buffer_bps,
    )
    assert fwd_ret is not None
    assert label in (0, 1)
    assert settled_at is not None

    assert (
        pred_repo.update_realized_outcome(
            rec.prediction_id,
            realized_return=fwd_ret,
            realized_label=label,
            settled_at=settled_at,
        )
        is True
    )
    # Second settlement must be idempotent no-op.
    assert (
        pred_repo.update_realized_outcome(
            rec.prediction_id,
            realized_return=fwd_ret,
            realized_label=label,
            settled_at=settled_at,
        )
        is False
    )

    assert pred_repo.list_unsettled_predictions(canonical_symbol="BTC/USDT") == ()
    settled_rec = pred_repo.get_prediction(rec.prediction_id)
    assert settled_rec is not None
    assert settled_rec.realized_label == label
    assert settled_rec.realized_return == pytest.approx(fwd_ret)

    report = compute_shadow_monitoring_report([settled_rec], min_settled=1)
    assert report.total_predictions == 1
    assert report.settled_predictions == 1
