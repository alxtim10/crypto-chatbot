"""Command-line entry point (Milestone A deterministic CLI and config check)."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from crypto_chatbot.analysis.service import AnalysisService, build_analysis_snapshot
from crypto_chatbot.analysis.setup import evaluate_setup_against_candles
from crypto_chatbot.app import run_bot
from crypto_chatbot.bot.templates import render_compact_snapshot, render_deterministic_snapshot
from crypto_chatbot.config import ConfigError, load_settings
from crypto_chatbot.market_data.ccxt_provider import CcxtMarketDataProvider
from crypto_chatbot.market_data.derivatives_provider import CcxtDerivativesLiquidityProvider
from crypto_chatbot.market_data.derivatives_quality import (
    normalize_funding_rate_payload,
    normalize_liquidation_events,
    normalize_open_interest_payload,
)
from crypto_chatbot.market_data.orderbook import normalize_order_book_payload
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import (
    SymbolError,
    SymbolMapping,
    resolve_mvp_spot_symbol,
    resolve_symbol,
)
from crypto_chatbot.market_data.tradingview import parse_ai_agent_lens_packet
from crypto_chatbot.ml.experiment import run_offline_experiment
from crypto_chatbot.ml.shadow import compute_shadow_monitoring_report
from crypto_chatbot.news.pipeline import build_sentiment_observations, deduplicate_articles
from crypto_chatbot.ops import (
    BackupError,
    check_database_health,
    dump_postgres_database,
    restore_postgres_database,
)
from crypto_chatbot.schemas.analysis import AnalysisSnapshot
from crypto_chatbot.schemas.derivatives import DerivativesContext
from crypto_chatbot.schemas.market import MarketType, Timeframe
from crypto_chatbot.schemas.ml import MlExperimentConfig
from crypto_chatbot.schemas.news import RawNewsItem, SentimentObservation
from crypto_chatbot.schemas.tradingview import TradingViewPacket
from crypto_chatbot.storage.database import create_db_engine, init_db
from crypto_chatbot.storage.repositories import (
    DerivativesRepository,
    MarketRepository,
    NewsRepository,
    PostgresDerivativesRepository,
    PostgresMarketRepository,
    PostgresMlExperimentRepository,
    PostgresModelPredictionRepository,
    PostgresNewsRepository,
    PostgresSetupLogRepository,
    PostgresTradingViewRepository,
    PostgresUserMemoryRepository,
    SetupLogRepository,
    TradingViewRepository,
)


def _check_config() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    print("Configuration OK")
    print(f"  gemini_model: {settings.gemini_model}")
    print(f"  allowed_telegram_user_ids: {len(settings.allowed_telegram_user_ids)} configured")
    return 0


def _build_repositories_from_env() -> tuple[
    MarketRepository,
    DerivativesRepository,
    SetupLogRepository,
    TradingViewRepository,
    NewsRepository,
    Any,
]:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required when --persist is set")
    engine = create_db_engine(database_url)
    init_db(engine)
    return (
        PostgresMarketRepository(engine),
        PostgresDerivativesRepository(engine),
        PostgresSetupLogRepository(engine),
        PostgresTradingViewRepository(engine),
        PostgresNewsRepository(engine),
        engine,
    )


def _load_derivatives_fixture(
    *,
    spot_mapping: SymbolMapping,
    timeframe: Timeframe,
    derivatives_fixture_path: Path,
    default_received_time: datetime,
    derivatives_repository: DerivativesRepository | None,
) -> DerivativesContext:
    if not derivatives_fixture_path.is_file():
        raise ValueError(f"derivatives fixture file not found: {derivatives_fixture_path}")
    raw = json.loads(derivatives_fixture_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("derivatives fixture must be a JSON object")

    received_raw = raw.get("received_time")
    received_time = (
        datetime.fromisoformat(received_raw)
        if isinstance(received_raw, str) and received_raw.strip()
        else default_received_time
    )
    perp_mapping = resolve_symbol(spot_mapping.base, market_type=MarketType.PERPETUAL)

    funding_batch = None
    if isinstance(raw.get("funding_rate"), dict):
        funding_batch = normalize_funding_rate_payload(
            raw["funding_rate"],
            symbol=perp_mapping,
            received_time=received_time,
            funding_interval_hours=raw.get("funding_interval_hours"),
            now=received_time,
        )

    oi_batch = None
    if isinstance(raw.get("open_interest"), dict):
        oi_data = raw["open_interest"]
        oi_batch = normalize_open_interest_payload(
            oi_data.get("latest"),
            oi_data.get("history") or [],
            symbol=perp_mapping,
            timeframe=timeframe,
            received_time=received_time,
            contract_size=float(oi_data.get("contract_size", 1.0)),
            raw_unit=oi_data.get("raw_unit"),
            is_inverse=bool(oi_data.get("is_inverse", False)),
        )

    ob_batch = None
    if isinstance(raw.get("order_book"), dict):
        ob_batch = normalize_order_book_payload(
            raw["order_book"],
            symbol=spot_mapping,
            received_time=received_time,
        )

    liq_batch = None
    if "liquidations" in raw:
        raw_liqs = raw.get("liquidations")
        liq_batch = normalize_liquidation_events(
            raw_liqs or [],
            symbol=perp_mapping,
            received_time=received_time,
            source_available=isinstance(raw_liqs, list),
        )

    context = DerivativesContext(
        funding=funding_batch,
        open_interest=oi_batch,
        order_book=ob_batch,
        liquidations=liq_batch,
    )
    if derivatives_repository is not None:
        if funding_batch is not None and funding_batch.observation is not None:
            derivatives_repository.save_funding_observations([funding_batch.observation])
        if oi_batch is not None:
            oi_rows = [
                *([oi_batch.latest] if oi_batch.latest is not None else []),
                *oi_batch.history,
            ]
            if oi_rows:
                derivatives_repository.save_open_interest_observations(oi_rows)
        if ob_batch is not None and ob_batch.snapshot is not None:
            derivatives_repository.save_order_book_snapshot(ob_batch.snapshot)
        if liq_batch is not None and liq_batch.source_available and liq_batch.events:
            derivatives_repository.save_liquidation_events(liq_batch.events)

    return context


def _analyze_from_fixture(
    *,
    symbol: str,
    timeframe: Timeframe,
    fixture_path: Path,
    derivatives_fixture_path: Path | None,
    tradingview_fixture_path: Path | None,
    news_fixture_path: Path | None,
    repository: MarketRepository | None,
    derivatives_repository: DerivativesRepository | None,
    setup_repository: SetupLogRepository | None = None,
    tradingview_repository: TradingViewRepository | None = None,
    news_repository: NewsRepository | None = None,
) -> AnalysisSnapshot:
    mapping = resolve_mvp_spot_symbol(symbol)
    if not fixture_path.is_file():
        raise ValueError(f"fixture file not found: {fixture_path}")

    raw_data = json.loads(fixture_path.read_text(encoding="utf-8"))
    if isinstance(raw_data, dict):
        raw_rows = raw_data.get("raw_ohlcv")
        received_str = raw_data.get("received_time")
    elif isinstance(raw_data, list):
        raw_rows = raw_data
        received_str = None
    else:
        raise ValueError("fixture must be a JSON object with 'raw_ohlcv' or a list of OHLCV rows")

    if not isinstance(raw_rows, list) or not raw_rows:
        raise ValueError("fixture contains no OHLCV rows")

    if isinstance(received_str, str) and received_str.strip():
        received_time = datetime.fromisoformat(received_str)
    else:
        last_ts_ms = max(float(row[0]) for row in raw_rows if isinstance(row, list) and row)
        last_open = datetime.fromtimestamp(last_ts_ms / 1000.0, tz=UTC)
        received_time = last_open + timeframe.duration + timedelta(seconds=1)

    batch = normalize_ohlcv_rows(
        raw_rows,
        symbol=mapping,
        timeframe=timeframe,
        received_time=received_time,
        now=received_time,
    )
    derivatives_context = (
        _load_derivatives_fixture(
            spot_mapping=mapping,
            timeframe=timeframe,
            derivatives_fixture_path=derivatives_fixture_path,
            default_received_time=received_time,
            derivatives_repository=derivatives_repository,
        )
        if derivatives_fixture_path is not None
        else None
    )
    tv_packet: TradingViewPacket | None = None
    if tradingview_fixture_path is not None:
        if not tradingview_fixture_path.is_file():
            raise ValueError(f"tradingview fixture file not found: {tradingview_fixture_path}")
        tv_raw = json.loads(tradingview_fixture_path.read_text(encoding="utf-8"))
        if not isinstance(tv_raw, dict):
            raise ValueError("tradingview fixture must be a JSON object")
        tv_received_raw = tv_raw.get("received_time")
        tv_received = (
            datetime.fromisoformat(tv_received_raw)
            if isinstance(tv_received_raw, str) and tv_received_raw.strip()
            else received_time
        )
        tv_packet = parse_ai_agent_lens_packet(
            tv_raw,
            received_time=tv_received,
            now=tv_received,
        )
    news_observations: tuple[SentimentObservation, ...] | None = None
    if news_fixture_path is not None:
        if not news_fixture_path.is_file():
            raise ValueError(f"news fixture file not found: {news_fixture_path}")
        raw_news_json = json.loads(news_fixture_path.read_text(encoding="utf-8"))
        raw_items_list = (
            raw_news_json.get("articles") if isinstance(raw_news_json, dict) else raw_news_json
        )
        if not isinstance(raw_items_list, list):
            raise ValueError("news fixture must be a JSON array or object containing 'articles'")
        raw_items = [RawNewsItem.model_validate(item) for item in raw_items_list]
        articles, _ = deduplicate_articles(raw_items)
        news_observations = build_sentiment_observations(articles)
        if news_repository is not None:
            if articles:
                news_repository.save_articles(articles)
            if news_observations:
                news_repository.save_observations(news_observations)

    snapshot = build_analysis_snapshot(
        batch,
        derivatives_context=derivatives_context,
        tradingview_packet=tv_packet,
        news_observations=news_observations,
    )
    if repository is not None:
        if batch.candles:
            repository.save_candles(batch.candles)
        repository.save_snapshot(snapshot)
    if tradingview_repository is not None and tv_packet is not None:
        tradingview_repository.save_packet(tv_packet)
    if setup_repository is not None and snapshot.trade_setup is not None:
        setup_repository.save_setup(snapshot, snapshot.trade_setup)
    return snapshot


async def _analyze_live(
    *,
    symbol: str,
    timeframe: Timeframe,
    include_mtf: bool,
    include_derivatives: bool,
    repository: MarketRepository | None,
    derivatives_repository: DerivativesRepository | None,
    setup_repository: SetupLogRepository | None = None,
    tradingview_repository: TradingViewRepository | None = None,
    news_repository: NewsRepository | None = None,
) -> AnalysisSnapshot:
    async with CcxtMarketDataProvider() as provider:
        if include_derivatives:
            async with CcxtDerivativesLiquidityProvider() as deriv_provider:
                service = AnalysisService(
                    provider=provider,
                    repository=repository,
                    derivatives_provider=deriv_provider,
                    derivatives_repository=derivatives_repository,
                    setup_repository=setup_repository,
                    tradingview_repository=tradingview_repository,
                    news_repository=news_repository,
                )
                return await service.analyze(
                    symbol,
                    timeframe,
                    include_mtf=include_mtf,
                    include_derivatives=True,
                    persist=repository is not None,
                )
        service = AnalysisService(
            provider=provider,
            repository=repository,
            setup_repository=setup_repository,
            tradingview_repository=tradingview_repository,
            news_repository=news_repository,
        )
        return await service.analyze(
            symbol,
            timeframe,
            include_mtf=include_mtf,
            include_derivatives=False,
            persist=repository is not None,
        )


def _run_analyze(
    *,
    symbol_raw: str,
    timeframe_raw: str,
    fixture_path: Path | None,
    derivatives_fixture_path: Path | None,
    tradingview_fixture_path: Path | None,
    news_fixture_path: Path | None,
    as_json: bool,
    detail: bool,
    persist: bool,
    include_mtf: bool,
    include_derivatives: bool,
) -> int:
    engine: Any = None
    try:
        mapping = resolve_mvp_spot_symbol(symbol_raw)
        try:
            timeframe = Timeframe(timeframe_raw.strip().lower())
        except ValueError:
            supported = ", ".join(tf.value for tf in Timeframe)
            raise ValueError(
                f"unsupported timeframe '{timeframe_raw}' (supported: {supported})"
            ) from None

        repository: MarketRepository | None = None
        derivatives_repository: DerivativesRepository | None = None
        setup_repository: SetupLogRepository | None = None
        tradingview_repository: TradingViewRepository | None = None
        news_repository: NewsRepository | None = None
        if persist:
            (
                repository,
                derivatives_repository,
                setup_repository,
                tradingview_repository,
                news_repository,
                engine,
            ) = _build_repositories_from_env()

        if fixture_path is not None:
            snapshot = _analyze_from_fixture(
                symbol=mapping.canonical_symbol,
                timeframe=timeframe,
                fixture_path=fixture_path,
                derivatives_fixture_path=derivatives_fixture_path,
                tradingview_fixture_path=tradingview_fixture_path,
                news_fixture_path=news_fixture_path,
                repository=repository,
                derivatives_repository=derivatives_repository,
                setup_repository=setup_repository,
                tradingview_repository=tradingview_repository,
                news_repository=news_repository,
            )
        else:
            snapshot = asyncio.run(
                _analyze_live(
                    symbol=mapping.canonical_symbol,
                    timeframe=timeframe,
                    include_mtf=include_mtf,
                    include_derivatives=include_derivatives,
                    repository=repository,
                    derivatives_repository=derivatives_repository,
                    setup_repository=setup_repository,
                    tradingview_repository=tradingview_repository,
                    news_repository=news_repository,
                )
            )
    except (SymbolError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()

    if as_json:
        print(snapshot.model_dump_json(indent=2))
    elif detail:
        print(render_deterministic_snapshot(snapshot, html_mode=False))
    else:
        print(render_compact_snapshot(snapshot, html_mode=False))
    return 0


def _run_settle_setups(max_bars: int = 24) -> int:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        print("Error: DATABASE_URL environment variable is required", file=sys.stderr)
        return 1
    try:
        engine = create_db_engine(database_url)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    updated = 0
    try:
        init_db(engine)
        market_repo = PostgresMarketRepository(engine)
        setup_repo = PostgresSetupLogRepository(engine)
        pending = setup_repo.list_pending_setups(limit=200)
        for rec in pending:
            future_candles = market_repo.load_candles(
                venue=rec.venue,
                canonical_symbol=rec.canonical_symbol,
                market_type=MarketType.SPOT,
                timeframe=rec.timeframe,
                start_open_time=rec.as_of,
            )
            outcome_status, settled_at, realized_r = evaluate_setup_against_candles(
                rec,
                future_candles,
                max_bars=max_bars,
            )
            if outcome_status != rec.outcome_status or settled_at != rec.settled_at:
                setup_repo.update_settlement(
                    rec.snapshot_id,
                    outcome_status=outcome_status,
                    settled_at=settled_at,
                    realized_r=realized_r,
                )
                updated += 1
    finally:
        engine.dispose()

    print(f"Settle setups completed: checked={len(pending)}, updated={updated}")
    return 0


def _run_purge_expired() -> int:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        print("Error: DATABASE_URL environment variable is required", file=sys.stderr)
        return 1
    try:
        engine = create_db_engine(database_url)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    try:
        init_db(engine)
        repo = PostgresUserMemoryRepository(engine)
        result = repo.purge_expired(now=datetime.now(UTC))
    finally:
        engine.dispose()

    print(
        "Purge completed: "
        f"messages={result.deleted_messages}, "
        f"conversations={result.deleted_conversations}, "
        f"interactions={result.deleted_interactions}, "
        f"snapshots={result.deleted_snapshots}"
    )
    return 0


def _resolve_database_url(explicit_url: str | None = None) -> str:
    if explicit_url is not None and explicit_url.strip():
        return explicit_url.strip()
    env_url = os.environ.get("DATABASE_URL", "").strip()
    if env_url:
        return env_url
    try:
        return load_settings().database_url.get_secret_value()
    except ConfigError as exc:
        raise ValueError("DATABASE_URL (or configured Settings database_url) is required") from exc


def _run_bot() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    run_bot(settings)
    return 0


def _run_backup_db(*, output_path: Path, database_url: str | None) -> int:
    try:
        resolved_url = _resolve_database_url(database_url)
        res = dump_postgres_database(resolved_url, output_path)
    except (ValueError, BackupError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(
        f"Backup completed: path={res.output_path}, bytes={res.size_bytes}, "
        f"duration_ms={res.duration_ms}"
    )
    return 0


def _run_restore_db(*, backup_path: Path, database_url: str | None) -> int:
    try:
        resolved_url = _resolve_database_url(database_url)
        res = restore_postgres_database(resolved_url, backup_path)
    except (ValueError, BackupError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    print(f"Restore completed: path={res.backup_path}, duration_ms={res.duration_ms}")
    return 0


def _run_health_check(*, database_url: str | None) -> int:
    config_ok = True
    try:
        settings = load_settings()
    except ConfigError as exc:
        config_ok = False
        print(exc, file=sys.stderr)
        settings = None

    engine: Any = None
    db_ok = False
    try:
        if database_url is not None and database_url.strip():
            resolved_url = database_url.strip()
        elif settings is not None:
            resolved_url = settings.database_url.get_secret_value()
        else:
            resolved_url = os.environ.get("DATABASE_URL", "").strip()
        if resolved_url:
            engine = create_db_engine(resolved_url)
            db_ok = check_database_health(engine)
    except ValueError:
        db_ok = False
    finally:
        if engine is not None:
            engine.dispose()

    payload = {
        "config_ok": config_ok,
        "db_healthy": db_ok,
    }
    print(json.dumps(payload, indent=2))
    return 0 if (config_ok and db_ok) else 1


def _load_news_observations_from_fixture(
    news_fixture_path: Path | None,
) -> tuple[SentimentObservation, ...] | None:
    if news_fixture_path is None:
        return None
    if not news_fixture_path.is_file():
        raise ValueError(f"news fixture file not found: {news_fixture_path}")
    raw_news_json = json.loads(news_fixture_path.read_text(encoding="utf-8"))
    raw_items_list = (
        raw_news_json.get("articles") if isinstance(raw_news_json, dict) else raw_news_json
    )
    if not isinstance(raw_items_list, list):
        raise ValueError("news fixture must be a JSON array or object containing 'articles'")
    raw_items = [RawNewsItem.model_validate(item) for item in raw_items_list]
    articles, _ = deduplicate_articles(raw_items)
    return build_sentiment_observations(articles)


def _run_ml_experiment(
    *,
    symbol_raw: str,
    timeframe_raw: str,
    fixture_path: Path | None,
    news_fixture_path: Path | None,
    as_json: bool,
    persist: bool,
) -> int:
    engine: Any = None
    try:
        mapping = resolve_mvp_spot_symbol(symbol_raw)
        try:
            timeframe = Timeframe(timeframe_raw.strip().lower())
        except ValueError:
            supported = ", ".join(tf.value for tf in Timeframe)
            raise ValueError(
                f"unsupported timeframe '{timeframe_raw}' (supported: {supported})"
            ) from None

        if fixture_path is None:
            raise ValueError("--fixture is required for offline ml-experiment")
        if not fixture_path.is_file():
            raise ValueError(f"fixture file not found: {fixture_path}")

        raw_data = json.loads(fixture_path.read_text(encoding="utf-8"))
        raw_rows = raw_data.get("raw_ohlcv") if isinstance(raw_data, dict) else raw_data
        received_str = raw_data.get("received_time") if isinstance(raw_data, dict) else None
        if not isinstance(raw_rows, list) or not raw_rows:
            raise ValueError("fixture contains no OHLCV rows")

        if isinstance(received_str, str) and received_str.strip():
            received_time = datetime.fromisoformat(received_str)
        else:
            last_ts_ms = max(float(row[0]) for row in raw_rows if isinstance(row, list) and row)
            last_open = datetime.fromtimestamp(last_ts_ms / 1000.0, tz=UTC)
            received_time = last_open + timeframe.duration + timedelta(seconds=1)

        batch = normalize_ohlcv_rows(
            raw_rows,
            symbol=mapping,
            timeframe=timeframe,
            received_time=received_time,
            now=received_time,
        )
        news_obs = _load_news_observations_from_fixture(news_fixture_path)
        config = MlExperimentConfig(
            canonical_symbol=mapping.canonical_symbol,
            timeframe=timeframe,
        )
        report = run_offline_experiment(
            batch.candles,
            config=config,
            sentiment_observations=news_obs,
        )

        if persist:
            database_url = os.environ.get("DATABASE_URL", "").strip()
            if not database_url:
                raise ValueError(
                    "DATABASE_URL environment variable is required when --persist is set"
                )
            engine = create_db_engine(database_url)
            init_db(engine)
            PostgresMlExperimentRepository(engine).save_experiment(report)
    except (SymbolError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        if engine is not None:
            engine.dispose()

    if as_json:
        print(report.model_dump_json(indent=2))
    else:
        status_lbl = "ACCEPTED" if report.accepted else "REJECTED"
        print(
            f"ML Experiment {report.experiment_id} ({report.config.canonical_symbol} "
            f"{report.config.timeframe.value}) [{status_lbl}]"
        )
        print(
            f"  Target: {report.config.target_name} | Rows: {report.total_rows} "
            f"(dev={report.dev_rows}, holdout={report.holdout_rows})"
        )
        print(
            f"  OOS Brier: {report.oof_summary.model_brier:.4f} "
            f"(naive {report.oof_summary.naive_brier:.4f}) | "
            f"OOS Net After Costs: {report.oof_summary.net_return_after_costs * 100.0:.2f}%"
        )
        print(
            f"  Holdout Brier: {report.holdout_summary.model_brier:.4f} "
            f"(naive {report.holdout_summary.naive_brier:.4f}) | "
            f"Holdout Net After Costs: {report.holdout_summary.net_return_after_costs * 100.0:.2f}%"
        )
        for reason in report.rejection_reasons:
            print(f"  Rejected because: {reason}")
    return 0


def _run_setup_stats(
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    as_json: bool = False,
    database_url: str | None = None,
) -> int:
    try:
        resolved_url = _resolve_database_url(database_url)
        engine = create_db_engine(resolved_url)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    tf_enum: Timeframe | None = None
    if timeframe:
        try:
            tf_enum = Timeframe(timeframe)
        except ValueError:
            print(f"Error: Invalid timeframe '{timeframe}'", file=sys.stderr)
            return 1

    canonical: str | None = None
    if symbol:
        try:
            canonical = resolve_mvp_spot_symbol(symbol).canonical_symbol
        except Exception:
            canonical = symbol if "/" in symbol else f"{symbol}/USDT"

    try:
        init_db(engine)
        setup_repo = PostgresSetupLogRepository(engine)
        setups = setup_repo.list_recent_setups(
            canonical_symbol=canonical,
            timeframe=tf_enum,
            limit=500,
        )
    finally:
        engine.dispose()

    status_counts: dict[str, int] = {}
    realized_rs: list[float] = []

    for s in setups:
        st = s.outcome_status if s.outcome_status else "pending"
        status_counts[st] = status_counts.get(st, 0) + 1
        if s.realized_r is not None:
            realized_rs.append(s.realized_r)

    total = len(setups)
    tp_count = sum(status_counts.get(k, 0) for k in ("tp1", "tp2", "tp3"))
    sl_count = status_counts.get("sl", 0)
    closed_count = tp_count + sl_count
    hit_rate = (tp_count / closed_count) if closed_count > 0 else None
    mean_r = float(np.mean(realized_rs)) if realized_rs else None
    median_r = float(np.median(realized_rs)) if realized_rs else None

    report = {
        "canonical_symbol": canonical or "all",
        "timeframe": tf_enum.value if tf_enum else "all",
        "total_setups": total,
        "status_counts": status_counts,
        "hit_rate_tp_vs_sl": round(hit_rate, 4) if hit_rate is not None else None,
        "mean_realized_r": round(mean_r, 4) if mean_r is not None else None,
        "median_realized_r": round(median_r, 4) if median_r is not None else None,
        "realized_trade_count": len(realized_rs),
    }

    if as_json:
        print(json.dumps(report, indent=2))
        return 0

    print("=====================================================================")
    print("        AUDIT LEDGER: TRADE SETUP HISTORICAL OUTCOMES               ")
    print("=====================================================================")
    print(
        f"Filter Simbol: {canonical or 'Semua'} | "
        f"Timeframe: {tf_enum.value if tf_enum else 'Semua'}"
    )
    print(f"Total Setup Terdata: {total}")
    print(
        f"- TP (Hit Target 1/2/3): {tp_count} "
        f"(TP1: {status_counts.get('tp1', 0)}, "
        f"TP2: {status_counts.get('tp2', 0)}, "
        f"TP3: {status_counts.get('tp3', 0)})"
    )
    print(f"- SL (Stop Loss): {sl_count}")
    print(f"- Pending: {status_counts.get('pending', 0)}")
    print(f"- Expired: {status_counts.get('expired', 0)}")
    print(f"- Invalidated: {status_counts.get('invalidated', 0)}")
    hr_str = f"{hit_rate * 100.0:.1f}%" if hit_rate is not None else "N/A"
    print(f"Hit Rate Selesai (TP / (TP + SL)): {hr_str}")
    mr_str = f"{mean_r:+.2f}R" if mean_r is not None else "N/A"
    med_str = f"{median_r:+.2f}R" if median_r is not None else "N/A"
    print(f"Rata-rata Realized R: {mr_str} | Median: {med_str} (Sampel: {len(realized_rs)})")
    print("---------------------------------------------------------------------")
    print("Catatan Integritas & Kepatuhan:")
    print("Statistik di atas merupakan pengukuran matematis retrospektif dari setup")
    print("yang tersimpan di database dan BUKAN jaminan keberhasilan di masa depan.")
    print("=====================================================================")
    return 0


def _run_shadow_report(
    *,
    symbol: str | None = None,
    timeframe: str | None = None,
    as_json: bool = False,
    database_url: str | None = None,
) -> int:
    try:
        resolved_url = _resolve_database_url(database_url)
        engine = create_db_engine(resolved_url)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    tf_enum: Timeframe | None = None
    if timeframe:
        try:
            tf_enum = Timeframe(timeframe)
        except ValueError:
            print(f"Error: Invalid timeframe '{timeframe}'", file=sys.stderr)
            return 1

    canonical: str | None = None
    if symbol:
        try:
            canonical = resolve_mvp_spot_symbol(symbol).canonical_symbol
        except Exception:
            canonical = symbol if "/" in symbol else f"{symbol}/USDT"

    try:
        init_db(engine)
        pred_repo = PostgresModelPredictionRepository(engine)
        records = pred_repo.list_predictions(
            canonical_symbol=canonical,
            timeframe=tf_enum,
            limit=500,
        )
    finally:
        engine.dispose()

    report = compute_shadow_monitoring_report(records)

    if as_json:
        print(report.model_dump_json(indent=2))
        return 0

    print("=====================================================================")
    print("           AUDIT LEDGER: SHADOW MODEL MONITORING REPORT              ")
    print("=====================================================================")
    print(
        f"Filter Simbol: {canonical or 'Semua'} | "
        f"Timeframe: {tf_enum.value if tf_enum else 'Semua'}"
    )
    status_str = "SEHAT (HEALTHY)" if report.is_healthy else "PERINGATAN (ALERTS ACTIVE)"
    print(f"Status Model: {status_str}")
    print(f"Total Prediksi Shadow: {report.total_predictions}")
    print(f"Prediksi Selesai (Settled): {report.settled_predictions}")
    ab_rate = report.abstention_rate * 100.0
    print(f"Tingkat Abstain: {ab_rate:.1f}% ({report.abstained_count} abstain)")
    brier_str = f"{report.realized_brier:.4f}" if report.realized_brier is not None else "N/A"
    bss_str = (
        f"{report.realized_brier_skill_score:.4f}"
        if report.realized_brier_skill_score is not None
        else "N/A"
    )
    ece_str = f"{report.realized_ece:.4f}" if report.realized_ece is not None else "N/A"
    print(f"Realized Brier Score: {brier_str} | Brier Skill Score (BSS): {bss_str}")
    print(f"Realized ECE (Kalibrasi): {ece_str}")
    print(f"Alert Deteksi Drift Fitur: {report.drift_alert_count} alert")
    if report.alerts:
        print(f"Daftar Alert Aktif: {', '.join(report.alerts)}")

    if report.calibration_bins:
        print("\nTabel Keandalan Kalibrasi (Reliability Bins):")
        print(
            f"{'Bin Range':<15} | {'Count':<6} | {'Mean Prob':<10} | {'Empirical':<10} | {'Gap':<8}"
        )
        print("-" * 60)
        for b in report.calibration_bins:
            rng = f"[{b.bin_low:.2f}, {b.bin_high:.2f}]"
            mp = f"{b.mean_predicted_prob:.3f}" if b.mean_predicted_prob is not None else "-"
            ep = (
                f"{b.empirical_positive_rate:.3f}" if b.empirical_positive_rate is not None else "-"
            )
            gp = f"{b.calibration_gap:.3f}" if b.calibration_gap is not None else "-"
            print(f"{rng:<15} | {b.sample_count:<6} | {mp:<10} | {ep:<10} | {gp:<8}")

    print("=====================================================================")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crypto-chatbot")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("check-config", help="Validate configuration without printing secrets.")

    analyze_parser = subcommands.add_parser(
        "analyze",
        help="Run deterministic technical analysis from a fixture or live CCXT data.",
    )
    analyze_parser.add_argument(
        "symbol",
        help="Target Binance /USDT spot symbol (e.g. BTC, ETH, SOL, DOGE, PEPE, SUI/USDT).",
    )
    analyze_parser.add_argument(
        "timeframe",
        nargs="?",
        default="4h",
        help="Primary timeframe (default: 4h with automatic 1d -> 4h -> 1h -> 15m top-down).",
    )
    analyze_parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Path to recorded CCXT JSON fixture (runs offline with no network access).",
    )
    analyze_parser.add_argument(
        "--derivatives-fixture",
        type=Path,
        default=None,
        help="Optional path to recorded Phase 9 derivatives/order-book JSON fixture.",
    )
    analyze_parser.add_argument(
        "--tradingview-fixture",
        type=Path,
        default=None,
        help="Optional path to recorded TradingView ai-agent-lens/v1 JSON fixture.",
    )
    analyze_parser.add_argument(
        "--news-fixture",
        type=Path,
        default=None,
        help="Optional path to recorded Phase 11 news JSON fixture.",
    )
    analyze_parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw AnalysisSnapshot JSON instead of formatted Indonesian report.",
    )
    analyze_parser.add_argument(
        "--detail",
        action="store_true",
        help="Print full [F1..Fn] audit dump instead of compact key points.",
    )
    analyze_parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist candles and snapshot to PostgreSQL via DATABASE_URL.",
    )
    analyze_parser.add_argument(
        "--no-mtf",
        action="store_true",
        help="Skip multi-timeframe RSI alignment fetch in live mode.",
    )
    analyze_parser.add_argument(
        "--no-derivatives",
        action="store_true",
        help="Skip Phase 9 derivatives and order-book fetch in live mode.",
    )
    subcommands.add_parser(
        "run-bot",
        help="Run closed-beta Telegram bot via long polling.",
    )
    subcommands.add_parser(
        "purge-expired",
        help="Delete expired messages, interactions, and snapshots per retention policy.",
    )
    settle_parser = subcommands.add_parser(
        "settle-setups",
        help="Evaluate pending logged trade setups objectively against stored closed candles.",
    )
    settle_parser.add_argument(
        "--max-bars",
        type=int,
        default=24,
        help="Maximum closed candles horizon after setup as_of (default: 24).",
    )
    ml_parser = subcommands.add_parser(
        "ml-experiment",
        help="Run offline Phase 12 ML validation experiment on recorded candle fixtures.",
    )
    ml_parser.add_argument("symbol", help="Target spot symbol (e.g. BTC, ETH).")
    ml_parser.add_argument(
        "timeframe", nargs="?", default="4h", help="Candle timeframe (default 4h)."
    )
    ml_parser.add_argument(
        "--fixture", type=Path, required=True, help="Recorded OHLCV JSON fixture."
    )
    ml_parser.add_argument(
        "--news-fixture",
        type=Path,
        default=None,
        help="Optional Phase 11 news JSON fixture.",
    )
    ml_parser.add_argument(
        "--json", action="store_true", help="Print full MlExperimentReport JSON."
    )
    ml_parser.add_argument(
        "--persist",
        action="store_true",
        help="Persist experiment report to PostgreSQL ml_experiments via DATABASE_URL.",
    )

    backup_parser = subcommands.add_parser(
        "backup-db",
        help="Create a custom-format PostgreSQL backup using pg_dump.",
    )
    backup_parser.add_argument("output_path", type=Path, help="Destination dump file path.")
    backup_parser.add_argument(
        "--database-url",
        default=None,
        help="Optional PostgreSQL URL override (falls back to DATABASE_URL / Settings).",
    )

    restore_parser = subcommands.add_parser(
        "restore-db",
        help="Restore a PostgreSQL custom-format backup using pg_restore.",
    )
    restore_parser.add_argument("backup_path", type=Path, help="Source dump file path.")
    restore_parser.add_argument(
        "--database-url",
        default=None,
        help="Optional PostgreSQL URL override (falls back to DATABASE_URL / Settings).",
    )

    health_parser = subcommands.add_parser(
        "health-check",
        help="Run configuration + PostgreSQL connectivity probe and evaluate operational alerts.",
    )
    health_parser.add_argument(
        "--database-url",
        default=None,
        help="Optional PostgreSQL URL override for DB health probe.",
    )

    setup_stats_parser = subcommands.add_parser(
        "setup-stats",
        help="Summarize settled trade setups objectively from PostgreSQL audit ledger.",
    )
    setup_stats_parser.add_argument(
        "--symbol",
        default=None,
        help="Optional filter by symbol (e.g. BTC, ETH, SOL).",
    )
    setup_stats_parser.add_argument(
        "--timeframe",
        default=None,
        help="Optional filter by timeframe (e.g. 15m, 1h, 4h).",
    )
    setup_stats_parser.add_argument(
        "--json",
        action="store_true",
        help="Print summary in JSON format.",
    )
    setup_stats_parser.add_argument(
        "--database-url",
        default=None,
        help="Optional PostgreSQL URL override.",
    )

    shadow_report_parser = subcommands.add_parser(
        "shadow-report",
        help="Audit live drift, abstention, and calibration across shadow ML prediction logs.",
    )
    shadow_report_parser.add_argument(
        "--symbol",
        default=None,
        help="Optional filter by symbol (e.g. BTC, ETH, SOL).",
    )
    shadow_report_parser.add_argument(
        "--timeframe",
        default=None,
        help="Optional filter by timeframe (e.g. 15m, 1h, 4h).",
    )
    shadow_report_parser.add_argument(
        "--json",
        action="store_true",
        help="Print report in JSON format.",
    )
    shadow_report_parser.add_argument(
        "--database-url",
        default=None,
        help="Optional PostgreSQL URL override.",
    )

    args = parser.parse_args(argv)
    if args.command == "check-config":
        return _check_config()
    if args.command == "analyze":
        return _run_analyze(
            symbol_raw=args.symbol,
            timeframe_raw=args.timeframe,
            fixture_path=args.fixture,
            derivatives_fixture_path=args.derivatives_fixture,
            tradingview_fixture_path=args.tradingview_fixture,
            news_fixture_path=args.news_fixture,
            as_json=args.json,
            detail=args.detail,
            persist=args.persist,
            include_mtf=not args.no_mtf,
            include_derivatives=not args.no_derivatives,
        )
    if args.command == "setup-stats":
        return _run_setup_stats(
            symbol=args.symbol,
            timeframe=args.timeframe,
            as_json=args.json,
            database_url=args.database_url,
        )
    if args.command == "shadow-report":
        return _run_shadow_report(
            symbol=args.symbol,
            timeframe=args.timeframe,
            as_json=args.json,
            database_url=args.database_url,
        )
    if args.command == "run-bot":
        return _run_bot()
    if args.command == "purge-expired":
        return _run_purge_expired()
    if args.command == "settle-setups":
        return _run_settle_setups(max_bars=args.max_bars)
    if args.command == "ml-experiment":
        return _run_ml_experiment(
            symbol_raw=args.symbol,
            timeframe_raw=args.timeframe,
            fixture_path=args.fixture,
            news_fixture_path=args.news_fixture,
            as_json=args.json,
            persist=args.persist,
        )
    if args.command == "backup-db":
        return _run_backup_db(
            output_path=args.output_path,
            database_url=args.database_url,
        )
    if args.command == "restore-db":
        return _run_restore_db(
            backup_path=args.backup_path,
            database_url=args.database_url,
        )
    if args.command == "health-check":
        return _run_health_check(database_url=args.database_url)
    parser.error(f"unknown command: {args.command}")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
