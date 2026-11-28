"""Command-line entry point (Milestone A deterministic CLI and config check)."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
from sqlalchemy import Engine

from crypto_chatbot.analysis.backtest import (
    BacktestConfig,
    RoundBoundaryMode,
    render_backtest_report,
    render_dual_book_report,
    run_dual_paper_book_backtest,
    run_setup_backtest,
    send_round_events_webhook,
    write_decision_log_jsonl,
)
from crypto_chatbot.analysis.forward_proof import freeze_variant_params
from crypto_chatbot.analysis.momentum_screener import (
    DEFAULT_MIN_VOLUME_24H_USD,
    DEFAULT_TOP_PER_SIDE,
    render_momentum_screener_report,
    screen_momentum_movers,
)
from crypto_chatbot.analysis.pine_export import generate_pine_script
from crypto_chatbot.analysis.scanner import (
    DEFAULT_SCAN_BASKET,
    render_market_scan_report,
    scan_market_setups,
)
from crypto_chatbot.analysis.service import AnalysisService, build_analysis_snapshot
from crypto_chatbot.analysis.setup import evaluate_setup_against_candles
from crypto_chatbot.analysis.whale_tracker import (
    WhaleWatchlistManager,
    aggregate_whale_positions,
    format_whale_overview_telegram,
    format_whale_wallet_telegram,
    format_whale_watchlist_telegram,
)
from crypto_chatbot.app import run_bot
from crypto_chatbot.bot.templates import render_compact_snapshot, render_deterministic_snapshot
from crypto_chatbot.config import ConfigError, load_settings
from crypto_chatbot.market_data.base import HistoryMarketDataProvider, OhlcvBatch
from crypto_chatbot.market_data.bybit_derivatives import BybitDerivativesProvider
from crypto_chatbot.market_data.derivatives_quality import (
    normalize_funding_rate_payload,
    normalize_liquidation_events,
    normalize_open_interest_payload,
)
from crypto_chatbot.market_data.fallback_provider import (
    build_futures_providers,
    normalize_exchange_chain,
)
from crypto_chatbot.market_data.gate_risk_tiers import GateRiskTiersClient
from crypto_chatbot.market_data.macro_provider import DxyMarketDataProvider
from crypto_chatbot.market_data.orderbook import normalize_order_book_payload
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import (
    SymbolError,
    SymbolMapping,
    resolve_default_perp_symbol,
    resolve_mvp_spot_symbol,
    resolve_symbol,
)
from crypto_chatbot.market_data.tradingview import parse_ai_agent_lens_packet
from crypto_chatbot.market_data.whale_provider import (
    WhaleDataProvider,
)
from crypto_chatbot.ml.experiment import run_offline_experiment
from crypto_chatbot.ml.meta_labels import (
    build_meta_label_dataset,
    run_meta_labeling_experiment,
)
from crypto_chatbot.ml.shadow import compute_shadow_monitoring_report
from crypto_chatbot.ml.trials import (
    JsonlTrialRegistry,
    PostgresTrialRegistry,
    TrialRecord,
    TrialRegistry,
)
from crypto_chatbot.news.pipeline import build_sentiment_observations, deduplicate_articles
from crypto_chatbot.ops import (
    BackupError,
    check_database_health,
    dump_postgres_database,
    restore_postgres_database,
)
from crypto_chatbot.schemas.analysis import AnalysisSnapshot
from crypto_chatbot.schemas.derivatives import DerivativesContext
from crypto_chatbot.schemas.market import (
    MULTI_STYLE_SCAN_TIMEFRAMES,
    PRIMARY_TIMEFRAMES,
    MarketType,
    QualityStatus,
    Timeframe,
    resolve_primary_timeframe,
)
from crypto_chatbot.schemas.ml import MlExperimentConfig
from crypto_chatbot.schemas.news import RawNewsItem, SentimentObservation
from crypto_chatbot.schemas.tradingview import TradingViewPacket
from crypto_chatbot.schemas.whale import NetworkType
from crypto_chatbot.storage.database import create_db_engine, init_db
from crypto_chatbot.storage.repositories import (
    ContractStatsRepository,
    DerivativesRepository,
    FearGreedRepository,
    MarketRepository,
    NewsRepository,
    PostgresContractStatsRepository,
    PostgresDerivativesRepository,
    PostgresFearGreedRepository,
    PostgresGateIngestRepository,
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


def _resolve_optional_database_url(database_url: str | None = None) -> str | None:
    if database_url and database_url.strip():
        return database_url.strip()
    env_url = os.environ.get("DATABASE_URL", "").strip()
    if env_url:
        return env_url
    try:
        return load_settings().database_url.get_secret_value().strip() or None
    except Exception:
        return None


def _build_repositories_from_env() -> tuple[
    MarketRepository,
    DerivativesRepository,
    SetupLogRepository,
    TradingViewRepository,
    NewsRepository,
    ContractStatsRepository,
    FearGreedRepository,
    Any,
]:
    database_url = _resolve_optional_database_url()
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
        PostgresContractStatsRepository(engine),
        PostgresFearGreedRepository(engine),
        engine,
    )


def _build_sentiment_read_repositories() -> tuple[
    NewsRepository | None,
    ContractStatsRepository | None,
    FearGreedRepository | None,
    Any,
]:
    """Best-effort read-only sentiment DB repos for CLI `analyze` when DATABASE_URL is set."""
    database_url = _resolve_optional_database_url()
    if not database_url:
        return None, None, None, None
    try:
        engine = create_db_engine(database_url)
        return (
            PostgresNewsRepository(engine),
            PostgresContractStatsRepository(engine),
            PostgresFearGreedRepository(engine),
            engine,
        )
    except Exception:
        return None, None, None, None


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


def _load_fixture_batch(
    *,
    mapping: SymbolMapping,
    timeframe: Timeframe,
    fixture_path: Path,
) -> tuple[OhlcvBatch, datetime]:
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
    return batch, received_time


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
    batch, received_time = _load_fixture_batch(
        mapping=mapping,
        timeframe=timeframe,
        fixture_path=fixture_path,
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


def _resolve_exchange_chain_from_env() -> tuple[str, ...]:
    legacy_primary = os.environ.get("FUTURES_EXCHANGE", "").strip().lower()
    legacy_fallbacks = os.environ.get("FUTURES_EXCHANGE_FALLBACKS", "").strip()
    if legacy_primary or legacy_fallbacks:
        primary = legacy_primary or "gate"
        fallbacks = [part.strip().lower() for part in legacy_fallbacks.split(",") if part.strip()]
        return normalize_exchange_chain((primary, *fallbacks), default=("gate",))

    primary_venue = os.environ.get("PRIMARY_VENUE", "gateio").strip().lower() or "gateio"
    raw_fallbacks = os.environ.get("VENUE_FALLBACKS", "")
    fallbacks = [part.strip().lower() for part in raw_fallbacks.split(",") if part.strip()]
    binance_enabled = os.environ.get("BINANCE_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    chain = normalize_exchange_chain((primary_venue, *fallbacks), default=("gate",))
    if not binance_enabled:
        chain = tuple(ex for ex in chain if ex != "binance")
    return chain or ("gate",)


def _resolve_venue_cooldown_from_env() -> int:
    raw = os.environ.get("VENUE_COOLDOWN_MIN", "360").strip()
    try:
        parsed = int(raw)
    except ValueError:
        return 360
    return max(1, parsed)


def _resolve_gate_direct_candles_from_env() -> bool:
    raw = os.environ.get("GATE_DIRECT_CANDLES", "true").strip().lower()
    return raw not in {"0", "false", "no", "off"}


def _resolve_resample_from_env() -> Timeframe | None:
    raw = os.environ.get("RESAMPLE_FROM", "").strip().lower()
    if not raw:
        return None
    if raw not in {"1m", "5m", "15m", "1h"}:
        raise ConfigError(f"RESAMPLE_FROM must be one of 1m, 5m, 15m, 1h (got {raw!r})")
    return Timeframe(raw)


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
    contract_stats_repository: ContractStatsRepository | None = None,
    fear_greed_repository: FearGreedRepository | None = None,
) -> AnalysisSnapshot:
    chain = _resolve_exchange_chain_from_env()
    provider, deriv_provider = build_futures_providers(
        chain,
        cooldown_minutes=_resolve_venue_cooldown_from_env(),
        gate_direct_candles=_resolve_gate_direct_candles_from_env(),
    )
    bybit_provider = BybitDerivativesProvider() if "binance" in chain else None
    dxy_provider = DxyMarketDataProvider() if include_derivatives else None
    risk_tiers_client = GateRiskTiersClient() if "gate" in chain else None
    try:
        service = AnalysisService(
            provider=provider,
            repository=repository,
            derivatives_provider=deriv_provider if include_derivatives else None,
            bybit_provider=bybit_provider if include_derivatives else None,
            derivatives_repository=derivatives_repository,
            setup_repository=setup_repository,
            tradingview_repository=tradingview_repository,
            news_repository=news_repository,
            contract_stats_repository=contract_stats_repository,
            fear_greed_repository=fear_greed_repository,
            dxy_provider=dxy_provider,
            risk_tiers_client=risk_tiers_client,
            resample_from=_resolve_resample_from_env(),
        )
        return await service.analyze(
            symbol,
            timeframe,
            include_mtf=include_mtf,
            include_derivatives=include_derivatives,
            persist=repository is not None,
        )
    finally:
        await provider.close()
        await deriv_provider.close()
        if risk_tiers_client is not None:
            await risk_tiers_client.close()
        if bybit_provider is not None:
            await bybit_provider.close()
        if dxy_provider is not None:
            await dxy_provider.close()


def _parse_primary_timeframe(timeframe_raw: str) -> Timeframe:
    resolved = resolve_primary_timeframe(timeframe_raw)
    if resolved is not None:
        return resolved
    supported = ", ".join(
        tf.value for tf in sorted(PRIMARY_TIMEFRAMES, key=lambda item: item.duration)
    )
    raise ValueError(
        f"unsupported timeframe '{timeframe_raw}' (supported: {supported}, scalp, intraday, swing)"
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
        mapping = (
            resolve_mvp_spot_symbol(symbol_raw)
            if fixture_path is not None
            else resolve_default_perp_symbol(symbol_raw)
        )
        timeframe = _parse_primary_timeframe(timeframe_raw)

        repository: MarketRepository | None = None
        derivatives_repository: DerivativesRepository | None = None
        setup_repository: SetupLogRepository | None = None
        tradingview_repository: TradingViewRepository | None = None
        news_repository: NewsRepository | None = None
        contract_stats_repository: ContractStatsRepository | None = None
        fear_greed_repository: FearGreedRepository | None = None
        if persist:
            (
                repository,
                derivatives_repository,
                setup_repository,
                tradingview_repository,
                news_repository,
                contract_stats_repository,
                fear_greed_repository,
                engine,
            ) = _build_repositories_from_env()
        elif fixture_path is None:
            (
                news_repository,
                contract_stats_repository,
                fear_greed_repository,
                engine,
            ) = _build_sentiment_read_repositories()

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
                    contract_stats_repository=contract_stats_repository,
                    fear_greed_repository=fear_greed_repository,
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


def _resolve_alert_bot_token() -> str | None:
    env_tok = os.environ.get("TELEGRAM_BOT_TOKEN", "").strip()
    if env_tok:
        return env_tok
    try:
        return load_settings().telegram_bot_token.get_secret_value().strip() or None
    except Exception:
        return None


def _resolve_alert_chat_id(explicit_chat_id: int | None = None) -> int | None:
    if explicit_chat_id is not None:
        return explicit_chat_id
    for env_key in ("ALERT_CHAT_ID", "CHAT_ID"):
        raw = os.environ.get(env_key, "").strip()
        if raw:
            try:
                return int(raw)
            except ValueError:
                pass
    try:
        settings = load_settings()
        if settings.alert_chat_id is not None:
            return settings.alert_chat_id
        if settings.allowed_telegram_user_ids:
            return settings.allowed_telegram_user_ids[0]
    except Exception:
        return None
    return None


def _run_settle_setups(max_bars: int = 24) -> int:
    from crypto_chatbot.alerts.sender import TelegramOutboxSender
    from crypto_chatbot.alerts.worker import render_settlement_reply
    from crypto_chatbot.storage.repositories import PostgresAlertRepository

    try:
        database_url = _resolve_database_url(None)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    try:
        engine = create_db_engine(database_url)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    updated = 0
    replies_to_send: list[tuple[int, int, str]] = []
    try:
        init_db(engine)
        market_repo = PostgresMarketRepository(engine)
        setup_repo = PostgresSetupLogRepository(engine)
        alert_repo = PostgresAlertRepository(engine)
        pending = setup_repo.list_pending_setups(limit=200)
        for rec in pending:
            if rec.market_type in (MarketType.SPOT.value, MarketType.PERPETUAL.value):
                resolved_market_type = MarketType(rec.market_type)
            else:
                resolved_market_type = (
                    MarketType.PERPETUAL if ":" in rec.canonical_symbol else MarketType.SPOT
                )
            future_candles = market_repo.load_candles(
                venue=rec.venue,
                canonical_symbol=rec.canonical_symbol,
                market_type=resolved_market_type,
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
                chat_id = rec.telegram_chat_id
                msg_id = rec.telegram_message_id
                if chat_id is None or msg_id is None:
                    linked = alert_repo.get_alert_by_snapshot_id(rec.snapshot_id)
                    if linked is not None:
                        chat_id = chat_id or linked.chat_id
                        msg_id = msg_id or linked.telegram_message_id
                if chat_id is not None and msg_id is not None:
                    reply_html = render_settlement_reply(
                        rec,
                        outcome_status=outcome_status,
                        realized_r=realized_r,
                        settled_at=settled_at,
                    )
                    replies_to_send.append((chat_id, msg_id, reply_html))
    finally:
        engine.dispose()

    replied = 0
    bot_token = _resolve_alert_bot_token()
    if replies_to_send and bot_token:

        async def _send_replies() -> int:
            sender = TelegramOutboxSender(bot_token=bot_token)
            sent_cnt = 0
            try:
                for target_chat, reply_to_id, text_html in replies_to_send:
                    try:
                        await sender.send_message(
                            chat_id=target_chat,
                            text=text_html,
                            parse_mode="HTML",
                            reply_to_message_id=reply_to_id,
                        )
                        sent_cnt += 1
                    except Exception as exc:
                        logging.getLogger(__name__).warning(
                            "Failed to send settlement thread reply: %s", exc
                        )
            finally:
                await sender.close()
            return sent_cnt

        replied = asyncio.run(_send_replies())

    print(f"Settle setups completed: checked={len(pending)}, updated={updated}, replied={replied}")
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


logger = logging.getLogger(__name__)


def _record_cli_trial(
    trial: TrialRecord,
    *,
    engine: Engine | None = None,
) -> None:
    """Best-effort Trial Registry recording (PostgreSQL when configured, else local JSONL)."""
    registry: TrialRegistry
    if engine is not None:
        registry = PostgresTrialRegistry(engine)
    else:
        trials_path_raw = os.environ.get("TRIALS_JSONL_PATH", "").strip()
        registry = (
            JsonlTrialRegistry(Path(trials_path_raw)) if trials_path_raw else JsonlTrialRegistry()
        )
    try:
        registry.record_trial(trial)
    except Exception:
        logger.debug("Trial registry recording skipped due to storage error", exc_info=True)


def _run_ml_experiment(
    *,
    symbol_raw: str,
    timeframe_raw: str,
    fixture_path: Path | None,
    news_fixture_path: Path | None,
    as_json: bool,
    persist: bool,
    target: str = "legacy",
) -> int:
    engine: Any = None
    try:
        mapping = (
            resolve_mvp_spot_symbol(symbol_raw)
            if fixture_path is not None
            else resolve_default_perp_symbol(symbol_raw)
        )
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
        if target == "meta":
            meta_ds = build_meta_label_dataset(
                {mapping.canonical_symbol: batch.candles},
                timeframe=timeframe,
                sentiment_by_symbol=(
                    {mapping.canonical_symbol: news_obs} if news_obs is not None else None
                ),
            )
            meta_res = run_meta_labeling_experiment(meta_ds)
            if as_json:
                print(meta_res.model_dump_json(indent=2))
            else:
                print(
                    f"Meta-Labeling {meta_res.timeframe.value} ({', '.join(meta_res.symbols)}) "
                    f"| status={meta_res.status} | model={meta_res.model_version} "
                    f"| filled={meta_res.filled_events_count} "
                    f"| no_fill={meta_res.no_fill_excluded_count} "
                    f"| promoted={meta_res.promoted}"
                )
            return 0
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

        _record_cli_trial(
            TrialRecord.create(
                family=f"ml:{config.target_name}",
                variant_params=config.model_dump(mode="json"),
                data_window=(
                    f"{batch.candles[0].open_time.strftime('%Y-%m-%d')}.."
                    f"{batch.candles[-1].close_time.strftime('%Y-%m-%d')}"
                ),
                symbols=(config.canonical_symbol,),
                timeframe=config.timeframe.value,
                obs_frequency=config.timeframe.value,
                sr_per_obs=None,
                n_obs=report.oof_summary.sample_count,
            ),
            engine=engine,
        )
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
            canonical = resolve_default_perp_symbol(symbol).canonical_symbol
        except Exception:
            canonical = symbol if ":" in symbol else f"{symbol}/USDT:USDT"

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
            canonical = resolve_default_perp_symbol(symbol).canonical_symbol
        except Exception:
            canonical = symbol if ":" in symbol else f"{symbol}/USDT:USDT"

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


async def _fetch_live_ohlcv_for_backtest(
    *,
    symbol: str,
    timeframe: Timeframe,
    limit: int,
) -> OhlcvBatch:
    chain = _resolve_exchange_chain_from_env()
    provider, deriv_provider = build_futures_providers(
        chain,
        cooldown_minutes=_resolve_venue_cooldown_from_env(),
        gate_direct_candles=_resolve_gate_direct_candles_from_env(),
    )
    try:
        if isinstance(provider, HistoryMarketDataProvider):
            return await provider.get_closed_ohlcv_history(symbol, timeframe, limit=limit)
        return await provider.get_closed_ohlcv(symbol, timeframe, limit=limit)
    finally:
        await provider.close()
        await deriv_provider.close()


def _load_causal_funding_for_backtest(
    *,
    venue: str,
    canonical_symbol: str,
    as_of: datetime,
) -> tuple[tuple[datetime, float], ...] | None:
    db_url = os.environ.get("DATABASE_URL")
    if not db_url:
        return None
    engine = create_db_engine(db_url)
    try:
        repo = PostgresDerivativesRepository(engine)
        obs = repo.load_funding_observations(
            venue=venue,
            canonical_symbol=canonical_symbol,
            end_available_time=as_of,
        )
        if not obs:
            return None
        return tuple((o.observed_time, o.funding_rate) for o in obs)
    except Exception:
        return None
    finally:
        engine.dispose()


def _run_backtest(
    *,
    symbol_raw: str,
    timeframe_raw: str,
    limit: int,
    warmup: int,
    max_hold: int,
    step: int,
    fixture_path: Path | None,
    as_json: bool,
    dual_book: bool = False,
    boundary: str = "daily_utc",
    log_jsonl: Path | None = None,
    webhook_url: str | None = None,
) -> int:
    try:
        resolved_tf = resolve_primary_timeframe(timeframe_raw)
        if resolved_tf is not None:
            timeframe = resolved_tf
        else:
            try:
                timeframe = Timeframe(timeframe_raw.strip().lower())
            except ValueError:
                supported = ", ".join(tf.value for tf in Timeframe)
                raise ValueError(
                    f"unsupported timeframe '{timeframe_raw}' (supported: {supported})"
                ) from None

        if fixture_path is not None:
            mapping = resolve_mvp_spot_symbol(symbol_raw)
            batch, _ = _load_fixture_batch(
                mapping=mapping,
                timeframe=timeframe,
                fixture_path=fixture_path,
            )
        else:
            mapping = resolve_default_perp_symbol(symbol_raw)
            batch = asyncio.run(
                _fetch_live_ohlcv_for_backtest(
                    symbol=mapping.canonical_symbol,
                    timeframe=timeframe,
                    limit=limit,
                )
            )

        if batch.has_provider_error or not batch.candles:
            msg = batch.error_message or "failed to load OHLCV candles for backtest"
            raise ValueError(msg)

        config = BacktestConfig(
            warmup_bars=warmup,
            max_hold_bars=max_hold,
            step_bars=step,
        )
        funding_pairs = (
            _load_causal_funding_for_backtest(
                venue=batch.metadata.venue,
                canonical_symbol=batch.canonical_symbol,
                as_of=batch.candles[-1].close_time,
            )
            if fixture_path is None
            else None
        )
        if dual_book:
            dual_report = run_dual_paper_book_backtest(
                batch.candles,
                config=config,
                boundary_mode=RoundBoundaryMode(boundary),
                funding_history=funding_pairs,
            )
            events = (
                *dual_report.always_on.decision_log,
                *dual_report.flat_at_boundary.decision_log,
            )
            if log_jsonl is not None:
                write_decision_log_jsonl(events, log_jsonl)
            if webhook_url:
                send_round_events_webhook(
                    webhook_url,
                    {
                        "mode": "dual_book",
                        "symbol": dual_report.always_on.symbol,
                        "timeframe": dual_report.always_on.timeframe.value,
                        "boundary_mode": dual_report.boundary_mode.value,
                        "summary": dual_report.summary_note,
                        "events": [ev.model_dump(mode="json") for ev in events],
                    },
                )
            if as_json:
                print(dual_report.model_dump_json(indent=2))
            else:
                print(render_dual_book_report(dual_report))
            return 0

        report = run_setup_backtest(
            batch.candles,
            config=config,
            funding_history=funding_pairs,
        )
        _record_cli_trial(
            TrialRecord.create(
                family="setup_backtest",
                variant_params={
                    "symbol": report.symbol,
                    "timeframe": report.timeframe.value,
                    "warmup_bars": config.warmup_bars,
                    "max_hold_bars": config.max_hold_bars,
                    "step_bars": config.step_bars,
                    "require_ema200": config.require_ema200,
                    "cost_bps": config.cost_assumptions.round_trip_cost_bps,
                },
                data_window=(
                    f"{report.start_time.strftime('%Y-%m-%d')}.."
                    f"{report.end_time.strftime('%Y-%m-%d')}"
                ),
                symbols=(report.symbol,),
                timeframe=report.timeframe.value,
                obs_frequency="per_trade",
                sr_per_obs=report.sharpe_per_trade,
                n_obs=report.executed_trades,
            ),
        )
        if log_jsonl is not None:
            write_decision_log_jsonl(report.decision_log, log_jsonl)
        if webhook_url:
            send_round_events_webhook(
                webhook_url,
                {
                    "mode": "single_book",
                    "symbol": report.symbol,
                    "timeframe": report.timeframe.value,
                    "book_name": report.book_name,
                    "executed_trades": report.executed_trades,
                    "total_net_r": report.total_net_r,
                    "events": [ev.model_dump(mode="json") for ev in report.decision_log],
                },
            )
    except (SymbolError, ValueError, OSError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(report.model_dump_json(indent=2))
    else:
        print(render_backtest_report(report))
    return 0


def _run_pine(
    *,
    symbol_raw: str,
    timeframe_raw: str,
    fixture_path: Path | None,
    out_path: Path | None,
) -> int:
    try:
        mapping = (
            resolve_mvp_spot_symbol(symbol_raw)
            if fixture_path is not None
            else resolve_default_perp_symbol(symbol_raw)
        )
        timeframe = _parse_primary_timeframe(timeframe_raw)

        if fixture_path is not None:
            snapshot = _analyze_from_fixture(
                symbol=mapping.canonical_symbol,
                timeframe=timeframe,
                fixture_path=fixture_path,
                derivatives_fixture_path=None,
                tradingview_fixture_path=None,
                news_fixture_path=None,
                repository=None,
                derivatives_repository=None,
            )
        else:
            snapshot = asyncio.run(
                _analyze_live(
                    symbol=mapping.canonical_symbol,
                    timeframe=timeframe,
                    include_mtf=True,
                    include_derivatives=True,
                    repository=None,
                    derivatives_repository=None,
                )
            )
    except (SymbolError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if snapshot.quality_status is QualityStatus.INSUFFICIENT_DATA:
        print(
            f"Error: INSUFFICIENT_DATA for {snapshot.canonical_symbol} {snapshot.timeframe.value}",
            file=sys.stderr,
        )
        return 1

    script = generate_pine_script(snapshot)
    if out_path is not None:
        out_path.write_text(script, encoding="utf-8")
        print(f"Pine Script written to {out_path}")
    else:
        print(script, end="")
    return 0


async def _run_scan_live(
    *,
    symbols: Sequence[str],
    timeframe: Timeframe,
    timeframes: Sequence[Timeframe] | None = None,
) -> Any:
    chain = _resolve_exchange_chain_from_env()
    provider, deriv_provider = build_futures_providers(
        chain,
        cooldown_minutes=_resolve_venue_cooldown_from_env(),
        gate_direct_candles=_resolve_gate_direct_candles_from_env(),
    )
    bybit_provider = BybitDerivativesProvider() if "binance" in chain else None
    dxy_provider = DxyMarketDataProvider()
    try:
        service = AnalysisService(
            provider=provider,
            derivatives_provider=deriv_provider,
            bybit_provider=bybit_provider,
            dxy_provider=dxy_provider,
        )
        return await scan_market_setups(
            service,
            symbols,
            timeframe=timeframe,
            timeframes=timeframes,
        )
    finally:
        await provider.close()
        await deriv_provider.close()
        await dxy_provider.close()
        if bybit_provider is not None:
            await bybit_provider.close()


def _run_scan(
    *,
    symbols_raw: str | None,
    timeframe_raw: str,
    as_json: bool,
) -> int:
    try:
        tf_norm = timeframe_raw.strip().lower()
        if tf_norm in ("all", "multi"):
            timeframe = Timeframe.H4
            scan_tfs: tuple[Timeframe, ...] | None = MULTI_STYLE_SCAN_TIMEFRAMES
        else:
            timeframe = _parse_primary_timeframe(timeframe_raw)
            scan_tfs = None
        if symbols_raw and symbols_raw.strip():
            targets = [part.strip() for part in symbols_raw.split(",") if part.strip()]
        else:
            from crypto_chatbot.alerts.config import load_universe

            targets = list(load_universe(fallback=DEFAULT_SCAN_BASKET))
        report = asyncio.run(
            _run_scan_live(
                symbols=targets,
                timeframe=timeframe,
                timeframes=scan_tfs,
            )
        )
    except (SymbolError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(report.model_dump_json(indent=2))
    else:
        print(render_market_scan_report(report, html_mode=False))
    return 0


async def _run_screen_live(
    *,
    side: Any,
    min_vol: float,
    top_per_side: int,
) -> Any:
    chain = _resolve_exchange_chain_from_env()
    provider, deriv_provider = build_futures_providers(
        chain,
        cooldown_minutes=_resolve_venue_cooldown_from_env(),
        gate_direct_candles=_resolve_gate_direct_candles_from_env(),
    )
    bybit_provider = BybitDerivativesProvider() if "binance" in chain else None
    dxy_provider = DxyMarketDataProvider()
    try:
        service = AnalysisService(
            provider=provider,
            derivatives_provider=deriv_provider,
            bybit_provider=bybit_provider,
            dxy_provider=dxy_provider,
        )
        return await screen_momentum_movers(
            service,
            side=side,
            min_volume_usd=min_vol,
            top_per_side=top_per_side,
        )
    finally:
        await provider.close()
        await deriv_provider.close()
        await dxy_provider.close()
        if bybit_provider is not None:
            await bybit_provider.close()


def _run_screen(
    *,
    side: Any,
    min_vol: float,
    top_per_side: int,
    as_json: bool,
) -> int:
    try:
        report = asyncio.run(
            _run_screen_live(
                side=side,
                min_vol=min_vol,
                top_per_side=top_per_side,
            )
        )
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(report.model_dump_json(indent=2))
    else:
        print(render_momentum_screener_report(report, html_mode=False))
    return 0


async def _run_whale_live(
    *,
    symbol: str | None,
    wallet: str | None,
    list_wallets: bool,
) -> tuple[str, dict[str, Any]]:
    manager = WhaleWatchlistManager()
    if list_wallets:
        entries = manager.list_entries()
        text = format_whale_watchlist_telegram(entries, html_mode=False)
        json_data = {"wallets": [e.model_dump(mode="json") for e in entries]}
        return text, json_data

    provider = WhaleDataProvider()
    if wallet:
        profile = await provider.get_wallet_profile(wallet)
        text = format_whale_wallet_telegram(profile, html_mode=False)
        json_data = profile.model_dump(mode="json")
        return text, json_data

    sym = symbol or "BTC"
    entries = manager.list_entries()
    targets = [(e.address, e.label) for e in entries if e.network == NetworkType.EVM]
    positions = await provider.get_positions_for_symbol(sym, wallets=targets)
    overview = aggregate_whale_positions(sym, positions)
    text = format_whale_overview_telegram(overview, html_mode=False)
    json_data = overview.model_dump(mode="json")
    return text, json_data


def _run_whale(
    *,
    symbol: str | None,
    wallet: str | None,
    list_wallets: bool,
    as_json: bool,
) -> int:
    try:
        text, json_data = asyncio.run(
            _run_whale_live(
                symbol=symbol,
                wallet=wallet,
                list_wallets=list_wallets,
            )
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(json.dumps(json_data, indent=2, default=str))
    else:
        print(text)
    return 0


def _run_contract_stats(
    *,
    action: str,
    symbols_raw: str | None,
    interval: str,
    days: int,
    limit: int,
    interval_seconds: int,
    as_json: bool,
    database_url: str | None,
) -> int:
    from crypto_chatbot.alerts.config import load_universe
    from crypto_chatbot.analysis.scanner import DEFAULT_SCAN_BASKET
    from crypto_chatbot.market_data.gate_contract_stats import (
        ContractStatsRecorder,
        ContractStatsScheduler,
        GateContractStatsClient,
    )

    resolved_url = database_url or os.environ.get("DATABASE_URL")
    if not resolved_url:
        try:
            settings = load_settings()
            resolved_url = settings.database_url.get_secret_value()
        except Exception:
            resolved_url = None
    if not resolved_url:
        print("Error: DATABASE_URL is required for contract-stats.", file=sys.stderr)
        return 1

    if interval not in {"5m", "1h"}:
        print(f"Error: unsupported interval '{interval}' (use 5m or 1h).", file=sys.stderr)
        return 1
    interval_lit: Any = interval
    symbols = (
        tuple(item.strip() for item in symbols_raw.split(",") if item.strip())
        if symbols_raw
        else load_universe(fallback=DEFAULT_SCAN_BASKET)
    )

    engine = create_db_engine(resolved_url)
    init_db(engine)
    repo = PostgresContractStatsRepository(engine)
    client = GateContractStatsClient()
    recorder = ContractStatsRecorder(
        client=client,
        repository=repo,
        symbols=symbols,
        interval=interval_lit,
    )

    try:
        if action == "gap-report":
            report = recorder.generate_gap_report(lookback_days=days)
            if as_json:
                print(report.model_dump_json(indent=2))
            else:
                print(
                    f"Gate contract_stats gap report ({report.venue}, interval={report.interval}, "
                    f"lookback={report.lookback_days}d)"
                )
                for item in report.symbols:
                    print(
                        f"- {item.canonical_symbol}: observed={item.observed_bars}/"
                        f"{item.expected_bars} ({item.coverage_pct:.2f}%), "
                        f"missing={item.missing_bars}, gaps={len(item.gaps)}"
                    )
            return 0

        async def _run_async() -> int:
            try:
                if action == "poll":
                    summary = await recorder.poll_once(limit=limit)
                elif action == "backfill":
                    summary = await recorder.backfill_history(lookback_days=days)
                elif action == "schedule":
                    scheduler = ContractStatsScheduler(
                        recorder,
                        interval_seconds=interval_seconds,
                        backfill_days=days,
                        run_backfill_on_start=True,
                    )
                    await scheduler.run_loop()
                    return 0
                else:
                    raise ValueError(f"unknown action: {action}")
            finally:
                await client.close()

            payload = {
                "action": action,
                "venue": summary.venue,
                "interval": summary.interval,
                "symbols_processed": summary.symbols_processed,
                "fetched_rows": summary.fetched_rows,
                "inserted_rows": summary.inserted_rows,
                "skipped_duplicates": summary.skipped_duplicates,
                "errors": list(summary.errors),
            }
            if as_json:
                print(json.dumps(payload, indent=2))
            else:
                print(
                    f"contract-stats {action}: symbols={summary.symbols_processed}, "
                    f"fetched={summary.fetched_rows}, inserted={summary.inserted_rows}, "
                    f"dupes={summary.skipped_duplicates}, errors={len(summary.errors)}"
                )
            return 0 if not summary.errors else 1

        return asyncio.run(_run_async())
    finally:
        engine.dispose()


def _run_ingest_gate_history(
    *,
    symbols_raw: str | None,
    from_ym: str,
    to_ym: str,
    types_raw: str,
    data_dir: Path,
    validate_contiguity: bool,
    as_json: bool,
    database_url: str | None,
) -> int:
    from crypto_chatbot.market_data.gate_candles import GateFuturesRestClient
    from crypto_chatbot.market_data.gate_contract_stats import DEFAULT_CONTRACT_STATS_UNIVERSE
    from crypto_chatbot.market_data.gate_history_ingest import (
        VERIFIED_GATEDATA_CANDLESTICK_TYPES,
        GateHistoryIngestor,
        validate_dataset_type,
    )

    symbols = (
        tuple(item.strip() for item in symbols_raw.split(",") if item.strip())
        if symbols_raw
        else DEFAULT_CONTRACT_STATS_UNIVERSE
    )
    datasets = tuple(item.strip() for item in types_raw.split(",") if item.strip())
    if not datasets:
        print("Error: --types must specify at least one dataset.", file=sys.stderr)
        return 1
    try:
        validated_datasets = tuple(validate_dataset_type(d) for d in datasets)
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    resolved_url = database_url or os.environ.get("DATABASE_URL")
    if not resolved_url:
        try:
            settings = load_settings()
            resolved_url = settings.database_url.get_secret_value()
        except Exception:
            resolved_url = None
    if not resolved_url:
        print("Error: DATABASE_URL is required for ingest-gate-history.", file=sys.stderr)
        return 1

    engine = create_db_engine(resolved_url)
    init_db(engine)
    market_repo = PostgresMarketRepository(engine)
    deriv_repo = PostgresDerivativesRepository(engine)
    ledger_repo = PostgresGateIngestRepository(engine)

    async def _run_async() -> dict[str, Any]:
        need_multipliers = any(
            d in VERIFIED_GATEDATA_CANDLESTICK_TYPES and VERIFIED_GATEDATA_CANDLESTICK_TYPES[d]
            for d in validated_datasets
        )
        multipliers = {}
        if need_multipliers:
            spec_client = GateFuturesRestClient()
            try:
                multipliers = await spec_client.fetch_contracts()
            finally:
                await spec_client.close()

        ingestor = GateHistoryIngestor(
            data_dir=data_dir,
            market_repository=market_repo,
            derivatives_repository=deriv_repo,
            ledger_repository=ledger_repo,
            multipliers=multipliers,
        )
        try:
            summary = await ingestor.ingest_range(
                symbols=symbols,
                from_ym=from_ym,
                to_ym=to_ym,
                datasets=validated_datasets,
                validate_contiguity=validate_contiguity,
            )
        finally:
            await ingestor.close()

        return {
            "files_processed": summary.files_processed,
            "downloaded_files": summary.downloaded_files,
            "skipped_files": summary.skipped_files,
            "inserted_rows": summary.inserted_rows,
            "missing_files": summary.missing_files,
            "results": [
                {
                    "dataset": r.dataset,
                    "contract": r.contract,
                    "yyyymm": r.yyyymm,
                    "status": r.status,
                    "sha256": r.sha256,
                    "row_count": r.row_count,
                    "inserted_rows": r.inserted_rows,
                    "contiguity_ok": r.contiguity_ok,
                    "local_path": r.local_path,
                }
                for r in summary.results
            ],
        }

    try:
        payload = asyncio.run(_run_async())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"ingest-gate-history: files={payload['files_processed']}, "
            f"downloaded={payload['downloaded_files']}, skipped={payload['skipped_files']}, "
            f"inserted={payload['inserted_rows']}, missing={payload['missing_files']}"
        )
    return 0


def _run_sync_gate_funding(
    *,
    symbols_raw: str | None,
    days: int,
    as_json: bool,
    database_url: str | None,
) -> int:
    from crypto_chatbot.market_data.gate_contract_stats import DEFAULT_CONTRACT_STATS_UNIVERSE
    from crypto_chatbot.market_data.gate_funding import (
        GateFundingHistoryClient,
        sync_gate_funding_history,
    )

    resolved_url = database_url or os.environ.get("DATABASE_URL")
    if not resolved_url:
        try:
            settings = load_settings()
            resolved_url = settings.database_url.get_secret_value()
        except Exception:
            resolved_url = None
    if not resolved_url:
        print("Error: DATABASE_URL is required for sync-gate-funding.", file=sys.stderr)
        return 1

    symbols = (
        tuple(item.strip() for item in symbols_raw.split(",") if item.strip())
        if symbols_raw
        else DEFAULT_CONTRACT_STATS_UNIVERSE
    )
    engine = create_db_engine(resolved_url)
    init_db(engine)
    deriv_repo = PostgresDerivativesRepository(engine)

    async def _run_async() -> dict[str, int]:
        client = GateFundingHistoryClient()
        try:
            return await sync_gate_funding_history(
                symbols,
                client=client,
                repository=deriv_repo,
                days=days,
            )
        finally:
            await client.close()

    try:
        counts = asyncio.run(_run_async())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    if as_json:
        print(json.dumps({"days": days, "inserted": counts}, indent=2))
    else:
        total = sum(counts.values())
        print(f"sync-gate-funding ({days}d): symbols={len(counts)} total_inserted={total}")
        for sym, count in counts.items():
            print(f"- {sym}: {count}")
    return 0


def _run_sync_fear_greed(
    *,
    limit: int,
    as_json: bool,
    database_url: str | None,
) -> int:
    from crypto_chatbot.news.fear_greed import FearGreedClient, sync_fear_greed

    resolved_url = database_url or os.environ.get("DATABASE_URL")
    if not resolved_url:
        try:
            settings = load_settings()
            resolved_url = settings.database_url.get_secret_value()
        except Exception:
            resolved_url = None
    if not resolved_url:
        print("Error: DATABASE_URL is required for sync-fear-greed.", file=sys.stderr)
        return 1

    engine = create_db_engine(resolved_url)
    init_db(engine)
    repo = PostgresFearGreedRepository(engine)

    async def _run_async() -> Any:
        client = FearGreedClient()
        try:
            return await sync_fear_greed(client=client, repository=repo, limit=limit)
        finally:
            await client.close()

    try:
        res = asyncio.run(_run_async())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    payload = {
        "fetched_rows": res.fetched_rows,
        "inserted_rows": res.inserted_rows,
        "skipped_duplicates": res.skipped_duplicates,
        "latest_value": res.latest.value if res.latest else None,
        "latest_classification": res.latest.classification if res.latest else None,
    }
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"sync-fear-greed: fetched={res.fetched_rows}, inserted={res.inserted_rows}, "
            f"dupes={res.skipped_duplicates}"
        )
    return 0


def _run_resample_parity(
    *,
    symbol_raw: str,
    bars: int,
    out_path: str | None,
    as_json: bool,
) -> int:
    from crypto_chatbot.market_data.gate_candles import GateFuturesMarketDataProvider
    from crypto_chatbot.market_data.resampling import (
        ResampleParityReport,
        compare_resampled_with_native,
        resample_candles,
    )

    try:
        mapping = resolve_default_perp_symbol(symbol_raw)
    except SymbolError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    target_bars = max(1, bars)

    async def _run_async() -> ResampleParityReport:
        provider = GateFuturesMarketDataProvider()
        try:
            needed_1h = target_bars * 24 + 24
            base_batch, native_4h, native_1d = await asyncio.gather(
                provider.get_closed_ohlcv_history(
                    mapping.canonical_symbol, Timeframe.H1, limit=needed_1h
                ),
                provider.get_closed_ohlcv_history(
                    mapping.canonical_symbol, Timeframe.H4, limit=target_bars + 2
                ),
                provider.get_closed_ohlcv_history(
                    mapping.canonical_symbol, Timeframe.D1, limit=target_bars + 2
                ),
            )
            if not base_batch.candles:
                raise RuntimeError(base_batch.error_message or "No closed 1h Gate candles received")
            as_of = base_batch.candles[-1].close_time
            res_4h = resample_candles(
                base_batch.candles, target_timeframe=Timeframe.H4, as_of=as_of
            )
            res_1d = resample_candles(
                base_batch.candles, target_timeframe=Timeframe.D1, as_of=as_of
            )
            summary_4h = compare_resampled_with_native(
                res_4h.candles,
                native_4h.candles,
                base_timeframe=Timeframe.H1,
                target_timeframe=Timeframe.H4,
                max_bars=target_bars,
            )
            summary_1d = compare_resampled_with_native(
                res_1d.candles,
                native_1d.candles,
                base_timeframe=Timeframe.H1,
                target_timeframe=Timeframe.D1,
                max_bars=target_bars,
            )
            return ResampleParityReport(
                generated_at=datetime.now(UTC),
                venue="gate",
                canonical_symbol=mapping.canonical_symbol,
                base_timeframe=Timeframe.H1,
                targets=(summary_4h, summary_1d),
            )
        finally:
            await provider.close()

    try:
        report = asyncio.run(_run_async())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    rendered = report.model_dump_json(indent=2)
    if out_path:
        dest = Path(out_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(rendered + "\n", encoding="utf-8")

    if as_json:
        print(rendered)
    else:
        for t in report.targets:
            print(
                f"{report.canonical_symbol} {t.base_timeframe.value}->{t.target_timeframe.value}: "
                f"matched={t.matched_bars}, max_diff_bps={t.max_close_diff_bps:.4f}, "
                f"mean_diff_bps={t.mean_close_diff_bps:.4f}, mismatch={t.mismatch_flagged}"
            )
    return 0


def _run_sync_news(
    *,
    source: str,
    as_json: bool,
    database_url: str | None,
) -> int:
    from crypto_chatbot.news.ingestor import PublicRssNewsClient, sync_public_rss_news
    from crypto_chatbot.storage.database import create_db_engine, init_db
    from crypto_chatbot.storage.repositories import PostgresNewsRepository

    src = source.strip().lower()
    if src == "cryptopanic":
        print(
            "Error: CryptoPanic API live endpoints return HTTP 403 Cloudflare challenge "
            "without a verified CRYPTOPANIC_API_KEY; use --source rss instead.",
            file=sys.stderr,
        )
        return 1
    if src != "rss":
        print(f"Error: unsupported news source '{source}' (expected: rss)", file=sys.stderr)
        return 1

    try:
        db_url = _resolve_database_url(database_url)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    engine = create_db_engine(db_url)
    init_db(engine)
    repo = PostgresNewsRepository(engine)

    async def _run_async() -> tuple[int, int, int, int, int, int]:
        client = PublicRssNewsClient()
        try:
            summary, _arts, _obs = await sync_public_rss_news(client=client, repository=repo)
            return (
                summary.feeds_attempted,
                summary.feeds_succeeded,
                summary.dedup_articles,
                summary.inserted_articles,
                summary.inserted_observations,
                summary.inserted_snapshots,
            )
        finally:
            await client.close()

    try:
        attempted, ok, dedup, ins_art, ins_obs, ins_snap = asyncio.run(_run_async())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    payload = {
        "source": "rss",
        "feeds_attempted": attempted,
        "feeds_succeeded": ok,
        "dedup_articles": dedup,
        "inserted_articles": ins_art,
        "inserted_observations": ins_obs,
        "inserted_snapshots": ins_snap,
    }
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(
            f"sync-news (rss): feeds={ok}/{attempted}, dedup={dedup}, "
            f"inserted_articles={ins_art}, inserted_observations={ins_obs}, "
            f"inserted_snapshots={ins_snap}"
        )
    return 0


def _run_export_sentiment_eval(
    *,
    count: int,
    out_path: Path,
    as_json: bool,
) -> int:
    from crypto_chatbot.news.eval_dataset import export_unlabeled_headlines_csv
    from crypto_chatbot.news.ingestor import PublicRssNewsClient
    from crypto_chatbot.news.pipeline import deduplicate_articles

    async def _fetch() -> int:
        client = PublicRssNewsClient()
        try:
            raw_items, _ok = await client.fetch_all()
            dedup_articles, _dropped = deduplicate_articles(raw_items)
            return export_unlabeled_headlines_csv(
                dedup_articles,
                output_path=out_path,
                count=count,
            )
        finally:
            await client.close()

    try:
        exported = asyncio.run(_fetch())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    payload = {
        "output_path": str(out_path),
        "exported_rows": exported,
        "manual_label_prefilled": False,
    }
    if as_json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"export-sentiment-eval: wrote {exported} unlabeled rows -> {out_path}")
    return 0


def _run_evaluate_sentiment_csv(
    *,
    input_path: Path,
    as_json: bool,
) -> int:
    from crypto_chatbot.news.eval_dataset import evaluate_labeled_headlines_csv

    try:
        report = evaluate_labeled_headlines_csv(input_path)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(report.model_dump_json(indent=2))
    else:
        print(
            f"evaluate-sentiment-csv: labeled={report.labeled_rows}/{report.total_rows}, "
            f"accuracy={report.accuracy:.4f}, matrix={report.confusion_matrix}"
        )
    return 0


def _run_alerts(
    *,
    action: str,
    config_path: Path | None,
    universe_path: Path | None,
    chat_id: int | None,
    database_url: str | None,
    dry_run: bool,
    as_json: bool,
) -> int:
    from crypto_chatbot.alerts.config import load_screener_config, load_universe
    from crypto_chatbot.alerts.sender import TelegramOutboxSender
    from crypto_chatbot.alerts.worker import run_alert_cycle, run_alert_digest, run_alert_warmup
    from crypto_chatbot.storage.repositories import PostgresAlertRepository

    try:
        db_url = _resolve_database_url(database_url)
    except ConfigError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    screener_cfg = load_screener_config(config_path)
    resolved_chat_id = _resolve_alert_chat_id(chat_id)
    bot_token = _resolve_alert_bot_token()

    engine = create_db_engine(db_url)
    init_db(engine)
    alert_repo = PostgresAlertRepository(engine)
    setup_repo = PostgresSetupLogRepository(engine)
    market_repo = PostgresMarketRepository(engine)

    if action == "digest":

        async def _do_digest() -> Any:
            sender = (
                TelegramOutboxSender(bot_token=bot_token)
                if (not dry_run and bot_token and resolved_chat_id is not None)
                else None
            )
            try:
                return await run_alert_digest(
                    alert_repo=alert_repo,
                    sender=sender,
                    chat_id=resolved_chat_id,
                    dry_run=dry_run,
                )
            finally:
                if sender is not None:
                    await sender.close()

        try:
            digest = asyncio.run(_do_digest())
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        finally:
            engine.dispose()

        if as_json:
            print(digest.model_dump_json(indent=2))
        else:
            print(
                f"alerts digest: recent_24h={digest.recent_alerts_24h}, "
                f"held={digest.held_alerts_included}, sent={digest.sent}"
            )
        return 0

    symbols = load_universe(universe_path or screener_cfg.universe_path)

    if action == "warmup":

        async def _do_warmup() -> Any:
            chain = _resolve_exchange_chain_from_env()
            provider, deriv_provider = build_futures_providers(
                chain,
                cooldown_minutes=_resolve_venue_cooldown_from_env(),
                gate_direct_candles=_resolve_gate_direct_candles_from_env(),
            )
            try:
                service = AnalysisService(
                    provider=provider,
                    repository=market_repo,
                    resample_from=_resolve_resample_from_env(),
                )
                return await run_alert_warmup(
                    analysis_service=service,
                    screener_config=screener_cfg,
                    symbols=symbols,
                )
            finally:
                await provider.close()
                await deriv_provider.close()

        try:
            warmup = asyncio.run(_do_warmup())
        except Exception as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        finally:
            engine.dispose()

        if as_json:
            print(warmup.model_dump_json(indent=2))
        else:
            tf_list = ",".join(tf.value for tf in warmup.timeframes)
            print(
                f"alerts warmup [{tf_list}]: symbols={warmup.symbols_total}, "
                f"ok={len(warmup.counts_by_symbol)}, errors={len(warmup.failed_symbols)}"
            )
        return 0

    async def _do_run() -> Any:
        chain = _resolve_exchange_chain_from_env()
        provider, deriv_provider = build_futures_providers(
            chain,
            cooldown_minutes=_resolve_venue_cooldown_from_env(),
            gate_direct_candles=_resolve_gate_direct_candles_from_env(),
        )
        bybit_provider = BybitDerivativesProvider() if "binance" in chain else None
        dxy_provider = DxyMarketDataProvider()
        risk_tiers_client = GateRiskTiersClient() if "gate" in chain else None
        sender = (
            TelegramOutboxSender(bot_token=bot_token)
            if (not dry_run and bot_token and resolved_chat_id is not None)
            else None
        )
        try:
            service = AnalysisService(
                provider=provider,
                repository=market_repo,
                derivatives_provider=deriv_provider,
                bybit_provider=bybit_provider,
                setup_repository=setup_repo,
                dxy_provider=dxy_provider,
                risk_tiers_client=risk_tiers_client,
                resample_from=_resolve_resample_from_env(),
            )
            return await run_alert_cycle(
                analysis_service=service,
                alert_repo=alert_repo,
                setup_repo=setup_repo,
                sender=sender,
                screener_config=screener_cfg,
                symbols=symbols,
                chat_id=resolved_chat_id,
                dry_run=dry_run,
            )
        finally:
            if sender is not None:
                await sender.close()
            await provider.close()
            await deriv_provider.close()
            await dxy_provider.close()
            if risk_tiers_client is not None:
                await risk_tiers_client.close()
            if bybit_provider is not None:
                await bybit_provider.close()

    try:
        summary = asyncio.run(_do_run())
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    if as_json:
        print(summary.model_dump_json(indent=2))
    else:
        if summary.locked_out:
            print("alerts run: skipped (advisory lock busy)")
        else:
            for r in summary.results:
                if r.skipped_reason:
                    print(
                        f"alerts run [{r.timeframe.value}] {r.run_id}: skipped ({r.skipped_reason})"
                    )
                else:
                    print(
                        f"alerts run [{r.timeframe.value}] {r.run_id}: scanned={r.scanned_count}, "
                        f"ready={r.ready_count}, armed={r.armed_count}, expired={r.expired_count}, "
                        f"sent={r.sent_alerts}, held={r.held_digest_alerts}, "
                        f"errors={len(r.failed_symbols)}"
                    )
    return 0


def _run_debug_sentiment(
    *,
    symbol_raw: str | None = None,
    as_of_raw: str | None = None,
    database_url: str | None = None,
    as_json: bool = False,
) -> int:
    """Inspect min/max timestamps, symbol keys, and future-dated counts in sentiment tables."""
    from sqlalchemy import text as sa_text

    resolved_url = _resolve_optional_database_url(database_url)
    if not resolved_url:
        print("Error: DATABASE_URL is required for debug-sentiment.", file=sys.stderr)
        return 1

    if as_of_raw:
        as_of_dt = datetime.fromisoformat(as_of_raw.replace("Z", "+00:00"))
        if as_of_dt.tzinfo is None:
            as_of_dt = as_of_dt.replace(tzinfo=UTC)
        as_of_utc = as_of_dt.astimezone(UTC)
    else:
        as_of_utc = datetime.now(UTC)

    def _iso(val: Any) -> str | None:
        if val is None:
            return None
        if isinstance(val, datetime):
            dt = val.astimezone(UTC) if val.tzinfo is not None else val.replace(tzinfo=UTC)
            return dt.isoformat()
        return str(val)

    engine = create_db_engine(resolved_url)
    report: dict[str, Any] = {
        "as_of_utc": as_of_utc.isoformat(),
        "symbol_filter": symbol_raw,
        "tables": {},
    }
    try:
        with engine.connect() as conn:
            # 1. fear_greed_observations
            fg_row = conn.execute(
                sa_text(
                    "SELECT COUNT(*), MIN(observed_time), MAX(observed_time), "
                    "MIN(available_time), MAX(available_time), "
                    "SUM(CASE WHEN available_time > :as_of THEN 1 ELSE 0 END) "
                    "FROM fear_greed_observations"
                ),
                {"as_of": as_of_utc},
            ).one()
            report["tables"]["fear_greed_observations"] = {
                "rows": int(fg_row[0] or 0),
                "min_observed_time": _iso(fg_row[1]),
                "max_observed_time": _iso(fg_row[2]),
                "min_available_time": _iso(fg_row[3]),
                "max_available_time": _iso(fg_row[4]),
                "future_available_rows": int(fg_row[5] or 0),
            }

            # 2. news_sentiment
            ns_row = conn.execute(
                sa_text(
                    "SELECT COUNT(*), MIN(published_time), MAX(published_time), "
                    "MIN(available_time), MAX(available_time), "
                    "MIN(scored_at), MAX(scored_at), "
                    "SUM(CASE WHEN available_time > :as_of THEN 1 ELSE 0 END) "
                    "FROM news_sentiment"
                ),
                {"as_of": as_of_utc},
            ).one()
            ns_syms = [
                str(r[0])
                for r in conn.execute(
                    sa_text(
                        "SELECT DISTINCT canonical_symbol FROM news_sentiment "
                        "ORDER BY canonical_symbol LIMIT 20"
                    )
                ).all()
            ]
            report["tables"]["news_sentiment"] = {
                "rows": int(ns_row[0] or 0),
                "min_published_time": _iso(ns_row[1]),
                "max_published_time": _iso(ns_row[2]),
                "min_available_time": _iso(ns_row[3]),
                "max_available_time": _iso(ns_row[4]),
                "min_scored_at": _iso(ns_row[5]),
                "max_scored_at": _iso(ns_row[6]),
                "future_available_rows": int(ns_row[7] or 0),
                "sample_symbols": ns_syms,
            }

            # 3. news_vote_snapshots
            nv_row = conn.execute(
                sa_text(
                    "SELECT COUNT(*), MIN(snapshot_time), MAX(snapshot_time), "
                    "SUM(CASE WHEN snapshot_time > :as_of THEN 1 ELSE 0 END) "
                    "FROM news_vote_snapshots"
                ),
                {"as_of": as_of_utc},
            ).one()
            report["tables"]["news_vote_snapshots"] = {
                "rows": int(nv_row[0] or 0),
                "min_snapshot_time": _iso(nv_row[1]),
                "max_snapshot_time": _iso(nv_row[2]),
                "future_snapshot_rows": int(nv_row[3] or 0),
            }

            # 4. contract_stats
            cs_row = conn.execute(
                sa_text(
                    "SELECT COUNT(*), MIN(observed_time), MAX(observed_time), "
                    "MIN(available_time), MAX(available_time), "
                    "SUM(CASE WHEN available_time > :as_of THEN 1 ELSE 0 END) "
                    "FROM contract_stats"
                ),
                {"as_of": as_of_utc},
            ).one()
            cs_venues = [
                str(r[0])
                for r in conn.execute(
                    sa_text("SELECT DISTINCT venue FROM contract_stats ORDER BY venue")
                ).all()
            ]
            cs_syms = [
                str(r[0])
                for r in conn.execute(
                    sa_text(
                        "SELECT DISTINCT canonical_symbol FROM contract_stats "
                        "ORDER BY canonical_symbol LIMIT 25"
                    )
                ).all()
            ]
            report["tables"]["contract_stats"] = {
                "rows": int(cs_row[0] or 0),
                "min_observed_time": _iso(cs_row[1]),
                "max_observed_time": _iso(cs_row[2]),
                "min_available_time": _iso(cs_row[3]),
                "max_available_time": _iso(cs_row[4]),
                "future_available_rows": int(cs_row[5] or 0),
                "venues": cs_venues,
                "sample_symbols": cs_syms,
            }
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()

    if as_json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Sentiment DB Diagnostics (as_of={report['as_of_utc']})")
        for tbl, info in report["tables"].items():
            print(f"- {tbl}: rows={info['rows']}")
            for k, v in info.items():
                if k != "rows":
                    print(f"    {k}: {v}")
    return 0


def _run_init_db(*, database_url: str | None = None) -> int:
    """Idempotently create all database tables and apply backwards-compatible column migrations."""
    db_url = database_url
    if not db_url:
        try:
            settings = load_settings()
            db_url = settings.database_url.get_secret_value()
        except Exception as exc:
            print(f"Configuration error: {exc}", file=sys.stderr)
            return 1
    try:
        engine = create_db_engine(db_url)
    except Exception as exc:
        print(f"Database engine error: {exc}", file=sys.stderr)
        return 1
    try:
        init_db(engine)
    except Exception as exc:
        print(f"Database initialization failed: {exc}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()
    print("OK: database tables and migrations initialized.")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crypto-chatbot")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("check-config", help="Validate configuration without printing secrets.")
    init_db_parser = subcommands.add_parser(
        "init-db",
        help="Idempotently create database tables and apply schema migrations.",
    )
    init_db_parser.add_argument(
        "--database-url",
        default=None,
        help="Optional PostgreSQL URL override (defaults to DATABASE_URL from .env).",
    )
    debug_sent_parser = subcommands.add_parser(
        "debug-sentiment",
        help="Print min/max timestamps, symbol keys, and future counts for sentiment tables.",
    )
    debug_sent_parser.add_argument(
        "--symbol",
        default=None,
        help="Optional symbol label for context.",
    )
    debug_sent_parser.add_argument(
        "--as-of",
        default=None,
        help="Optional ISO-8601 UTC cutoff timestamp (defaults to now UTC).",
    )
    debug_sent_parser.add_argument(
        "--database-url",
        default=None,
        help="Optional PostgreSQL URL override (defaults to DATABASE_URL from .env).",
    )
    debug_sent_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )

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
        "--target",
        choices=("legacy", "meta"),
        default="legacy",
        help="ML target mode: 'meta' (PRD v1.7 meta-labeling) or 'legacy' (shadow-only 24h).",
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

    backtest_parser = subcommands.add_parser(
        "backtest",
        help="Run deterministic point-in-time trade setup replay backtest.",
    )
    backtest_parser.add_argument("symbol", help="Target symbol (e.g. BTC, ETH, SOL).")
    backtest_parser.add_argument(
        "timeframe",
        nargs="?",
        default="4h",
        help="Candle timeframe (default: 4h).",
    )
    backtest_parser.add_argument(
        "--limit",
        type=int,
        default=1000,
        help="Maximum closed candles to fetch in live mode (default: 1000).",
    )
    backtest_parser.add_argument(
        "--warmup",
        type=int,
        default=600,
        help="Minimum warmup bars before first signal evaluation (default: 600).",
    )
    backtest_parser.add_argument(
        "--max-hold",
        type=int,
        default=24,
        help="Maximum holding bars per setup before expiration (default: 24).",
    )
    backtest_parser.add_argument(
        "--step",
        type=int,
        default=1,
        help="Bar step interval between evaluations when idle (default: 1).",
    )
    backtest_parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Optional recorded CCXT JSON fixture for offline replay.",
    )
    backtest_parser.add_argument(
        "--json",
        action="store_true",
        help="Print full BacktestReport JSON instead of Indonesian text report.",
    )
    backtest_parser.add_argument(
        "--dual-book",
        action="store_true",
        help="Run always-on vs flat-at-boundary dual paper books on identical candles.",
    )
    backtest_parser.add_argument(
        "--boundary",
        choices=("daily_utc", "funding_8h"),
        default="daily_utc",
        help="Session boundary mode for the second paper book (default: daily_utc).",
    )
    backtest_parser.add_argument(
        "--log-jsonl",
        type=Path,
        default=None,
        help="Optional output .jsonl path for plain-English HOLD/FILL/FEE/EXIT/FLAT log.",
    )
    backtest_parser.add_argument(
        "--webhook-url",
        default=None,
        help="Optional HTTPS (or localhost HTTP) webhook URL to POST round log summary.",
    )

    pine_parser = subcommands.add_parser(
        "pine",
        help="Generate TradingView Pine Script v6 overlay from deterministic analysis.",
    )
    pine_parser.add_argument("symbol", help="Target symbol (e.g. BTC, ETH, SOL).")
    pine_parser.add_argument(
        "timeframe",
        nargs="?",
        default="4h",
        help="Primary timeframe (default: 4h).",
    )
    pine_parser.add_argument(
        "--fixture",
        type=Path,
        default=None,
        help="Optional recorded CCXT JSON fixture for offline generation.",
    )
    pine_parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Optional output .pine file path (prints to stdout if omitted).",
    )

    scan_parser = subcommands.add_parser(
        "scan",
        help="Scan multiple futures pairs concurrently to surface READY unified MTF setups.",
    )
    scan_parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols (default: liquid perpetual basket).",
    )
    scan_parser.add_argument(
        "--timeframe",
        default="4h",
        help="Primary timeframe, style, or 'all' for multi-style (default: 4h).",
    )
    scan_parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw MarketScanReport JSON instead of Indonesian text report.",
    )

    screen_parser = subcommands.add_parser(
        "screen",
        help="Screen Top Gainers/Losers (>= $5M vol) for 15m Engulfing/Marubozu momentum setups.",
    )
    screen_parser.add_argument(
        "--side",
        choices=("all", "long", "short"),
        default="all",
        help="Filter side: all (gainers + losers), long (gainers), or short (losers).",
    )
    screen_parser.add_argument(
        "--min-vol",
        type=float,
        default=DEFAULT_MIN_VOLUME_24H_USD,
        help="Minimum 24h quote volume in USD (default: 5000000).",
    )
    screen_parser.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP_PER_SIDE,
        help="Maximum top movers per side to inspect on 15m (default: 8).",
    )
    screen_parser.add_argument(
        "--json",
        action="store_true",
        help="Print raw MomentumScreenerReport JSON instead of Indonesian text report.",
    )

    whale_parser = subcommands.add_parser(
        "whale",
        help="Track perpetual futures whale positioning (Hyperliquid) & on-chain wallet flows.",
    )
    whale_parser.add_argument(
        "--symbol",
        type=str,
        default=None,
        help="Futures symbol to aggregate whale positioning for (default: BTC).",
    )
    whale_parser.add_argument(
        "--wallet",
        type=str,
        default=None,
        help="Wallet address to inspect (EVM 0x... or Bitcoin bc1.../1.../3...).",
    )
    whale_parser.add_argument(
        "--list",
        action="store_true",
        help="List all tracked whale and institutional addresses.",
    )
    whale_parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text report.",
    )

    freeze_parser = subcommands.add_parser(
        "freeze-params",
        help="Freeze strategy variant parameters to frozen_params.json with params_sha256.",
    )
    freeze_parser.add_argument(
        "variant",
        type=str,
        help="Strategy variant name to freeze (e.g. v3, v0).",
    )
    freeze_parser.add_argument(
        "--output",
        type=Path,
        default=Path("frozen_params.json"),
        help="Output JSON path (default: frozen_params.json).",
    )
    freeze_parser.add_argument(
        "--json",
        action="store_true",
        help="Print frozen artifact JSON to stdout.",
    )

    stats_parser = subcommands.add_parser(
        "contract-stats",
        help="Poll, backfill, schedule, or audit gaps for Gate.io futures contract_stats.",
    )
    stats_parser.add_argument(
        "action",
        choices=("poll", "backfill", "gap-report", "schedule"),
        help="Action to execute: poll once, backfill history, print gap report, or run scheduler.",
    )
    stats_parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols (default: liquid perpetual basket).",
    )
    stats_parser.add_argument(
        "--interval",
        choices=("5m", "1h"),
        default="5m",
        help="contract_stats bucket interval (default: 5m).",
    )
    stats_parser.add_argument(
        "--days",
        type=int,
        default=179,
        help="Lookback window in days for backfill or gap-report (default: 179).",
    )
    stats_parser.add_argument(
        "--limit",
        type=int,
        default=12,
        help="Bars per symbol to fetch on `poll` (default: 12).",
    )
    stats_parser.add_argument(
        "--interval-seconds",
        type=int,
        default=300,
        help="Polling cadence in seconds for `schedule` (default: 300).",
    )
    stats_parser.add_argument(
        "--database-url",
        default=None,
        help="Optional PostgreSQL URL override.",
    )
    stats_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )

    ingest_parser = subcommands.add_parser(
        "ingest-gate-history",
        help="Idempotently download & ingest bulk archives from download.gatedata.org.",
    )
    ingest_parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols (default: liquid perpetual basket).",
    )
    ingest_parser.add_argument(
        "--from",
        dest="from_month",
        required=True,
        help="Start month inclusive (YYYYMM).",
    )
    ingest_parser.add_argument(
        "--to",
        dest="to_month",
        required=True,
        help="End month inclusive (YYYYMM).",
    )
    ingest_parser.add_argument(
        "--types",
        required=True,
        help=(
            "Comma-separated dataset types (candlesticks_1h, trades, mark_prices, funding_applies)."
        ),
    )
    ingest_parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/gate"),
        help="Root directory for compressed archive storage (default: data/raw/gate).",
    )
    ingest_parser.add_argument(
        "--no-contiguity-check",
        action="store_true",
        help="Skip contiguity validation on loaded candle series.",
    )
    ingest_parser.add_argument(
        "--database-url",
        default=None,
        help="Optional PostgreSQL URL override.",
    )
    ingest_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )

    funding_parser = subcommands.add_parser(
        "sync-gate-funding",
        help="Fetch paginated Gate /futures/usdt/funding_rate history into PostgreSQL.",
    )
    funding_parser.add_argument(
        "--symbols",
        default=None,
        help="Comma-separated symbols (default: liquid perpetual basket).",
    )
    funding_parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Lookback window in days (1..180, default: 90).",
    )
    funding_parser.add_argument(
        "--database-url",
        default=None,
        help="Optional PostgreSQL URL override.",
    )
    funding_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )

    fng_parser = subcommands.add_parser(
        "sync-fear-greed",
        help="Fetch Alternative.me Crypto Fear & Greed daily observations into PostgreSQL.",
    )
    fng_parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Days to fetch (0 = full historical backfill since 2018-02-01, default: 0).",
    )
    fng_parser.add_argument(
        "--database-url",
        default=None,
        help="Optional PostgreSQL URL override.",
    )
    fng_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )

    parity_parser = subcommands.add_parser(
        "resample-parity",
        help="Compare UTC-resampled 1h->4h/1d candles against native Gate candlesticks.",
    )
    parity_parser.add_argument(
        "--symbol",
        default="BTC",
        help="Target symbol (default: BTC).",
    )
    parity_parser.add_argument(
        "--bars",
        type=int,
        default=200,
        help="Number of overlapping bars to compare per target timeframe (default: 200).",
    )
    parity_parser.add_argument(
        "--out",
        default=None,
        help="Optional JSON report output file path.",
    )
    parity_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )

    sync_news_parser = subcommands.add_parser(
        "sync-news",
        help="Ingest verified public crypto RSS news feeds and append vote snapshots.",
    )
    sync_news_parser.add_argument(
        "--source",
        default="rss",
        choices=("rss", "cryptopanic"),
        help="News source (default: rss).",
    )
    sync_news_parser.add_argument(
        "--database-url",
        default=None,
        help="Optional PostgreSQL URL override.",
    )
    sync_news_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )

    export_eval_parser = subcommands.add_parser(
        "export-sentiment-eval",
        help="Export recent unlabeled headlines CSV for manual bullish/bearish/neutral labeling.",
    )
    export_eval_parser.add_argument(
        "--count",
        type=int,
        default=300,
        help="Maximum headlines to export (default: 300).",
    )
    export_eval_parser.add_argument(
        "--out",
        type=Path,
        default=Path("tests/fixtures/sentiment_eval_unlabeled.csv"),
        help="Output CSV path (default: tests/fixtures/sentiment_eval_unlabeled.csv).",
    )
    export_eval_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )

    eval_csv_parser = subcommands.add_parser(
        "evaluate-sentiment-csv",
        help="Score manually labeled headlines CSV and print lexicon accuracy/confusion matrix.",
    )
    eval_csv_parser.add_argument(
        "--input",
        type=Path,
        default=Path("tests/fixtures/sentiment_eval_unlabeled.csv"),
        help="Input labeled CSV path.",
    )
    eval_csv_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
    )

    alerts_parser = subcommands.add_parser(
        "alerts",
        help="One-shot MTF setup alert worker (`run`), daily summary (`digest`), or `warmup`.",
    )
    alerts_parser.add_argument(
        "action",
        choices=("run", "digest", "warmup"),
        help="Sub-action: 'run' (MTF scan + outbox), 'digest' (daily summary), or 'warmup'.",
    )
    alerts_parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to screener YAML config (default: config/screener.yaml).",
    )
    alerts_parser.add_argument(
        "--universe",
        type=Path,
        default=None,
        help="Path to universe text file (default: config/universe.txt).",
    )
    alerts_parser.add_argument(
        "--chat-id",
        type=int,
        default=None,
        help="Override target Telegram CHAT_ID.",
    )
    alerts_parser.add_argument(
        "--database-url",
        default=None,
        help="Optional PostgreSQL URL override.",
    )
    alerts_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write outbox/digest without sending Telegram API requests.",
    )
    alerts_parser.add_argument(
        "--json",
        action="store_true",
        help="Print JSON output.",
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
    if args.command == "backtest":
        return _run_backtest(
            symbol_raw=args.symbol,
            timeframe_raw=args.timeframe,
            limit=args.limit,
            warmup=args.warmup,
            max_hold=args.max_hold,
            step=args.step,
            fixture_path=args.fixture,
            as_json=args.json,
            dual_book=args.dual_book,
            boundary=args.boundary,
            log_jsonl=args.log_jsonl,
            webhook_url=args.webhook_url,
        )
    if args.command == "pine":
        return _run_pine(
            symbol_raw=args.symbol,
            timeframe_raw=args.timeframe,
            fixture_path=args.fixture,
            out_path=args.out,
        )
    if args.command == "scan":
        return _run_scan(
            symbols_raw=args.symbols,
            timeframe_raw=args.timeframe,
            as_json=args.json,
        )
    if args.command == "screen":
        return _run_screen(
            side=args.side,
            min_vol=args.min_vol,
            top_per_side=args.top,
            as_json=args.json,
        )
    if args.command == "whale":
        return _run_whale(
            symbol=args.symbol,
            wallet=args.wallet,
            list_wallets=args.list,
            as_json=args.json,
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
            target=args.target,
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
    if args.command == "freeze-params":
        try:
            artifact = freeze_variant_params(args.variant, output_path=args.output)
        except ValueError as exc:
            print(f"Error: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(artifact.model_dump_json(indent=2))
        else:
            print(
                f"Frozen variant={artifact.variant} params_sha256={artifact.params_sha256} "
                f"frozen_at={artifact.frozen_at.isoformat()} -> {args.output}"
            )
        return 0
    if args.command == "contract-stats":
        return _run_contract_stats(
            action=args.action,
            symbols_raw=args.symbols,
            interval=args.interval,
            days=args.days,
            limit=args.limit,
            interval_seconds=args.interval_seconds,
            as_json=args.json,
            database_url=args.database_url,
        )
    if args.command == "ingest-gate-history":
        return _run_ingest_gate_history(
            symbols_raw=args.symbols,
            from_ym=args.from_month,
            to_ym=args.to_month,
            types_raw=args.types,
            data_dir=args.data_dir,
            validate_contiguity=not args.no_contiguity_check,
            as_json=args.json,
            database_url=args.database_url,
        )
    if args.command == "sync-gate-funding":
        return _run_sync_gate_funding(
            symbols_raw=args.symbols,
            days=args.days,
            as_json=args.json,
            database_url=args.database_url,
        )
    if args.command == "sync-fear-greed":
        return _run_sync_fear_greed(
            limit=args.limit,
            as_json=args.json,
            database_url=args.database_url,
        )
    if args.command == "resample-parity":
        return _run_resample_parity(
            symbol_raw=args.symbol,
            bars=args.bars,
            out_path=args.out,
            as_json=args.json,
        )
    if args.command == "sync-news":
        return _run_sync_news(
            source=args.source,
            as_json=args.json,
            database_url=args.database_url,
        )
    if args.command == "export-sentiment-eval":
        return _run_export_sentiment_eval(
            count=args.count,
            out_path=args.out,
            as_json=args.json,
        )
    if args.command == "evaluate-sentiment-csv":
        return _run_evaluate_sentiment_csv(
            input_path=args.input,
            as_json=args.json,
        )
    if args.command == "alerts":
        return _run_alerts(
            action=args.action,
            config_path=args.config,
            universe_path=args.universe,
            chat_id=args.chat_id,
            database_url=args.database_url,
            dry_run=args.dry_run,
            as_json=args.json,
        )
    if args.command == "init-db":
        return _run_init_db(database_url=args.database_url)
    if args.command == "debug-sentiment":
        return _run_debug_sentiment(
            symbol_raw=args.symbol,
            as_of_raw=args.as_of,
            database_url=args.database_url,
            as_json=args.json,
        )
    parser.error(f"unknown command: {args.command}")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
