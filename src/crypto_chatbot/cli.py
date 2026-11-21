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

from crypto_chatbot.analysis.service import AnalysisService, build_analysis_snapshot
from crypto_chatbot.app import run_polling
from crypto_chatbot.bot.templates import render_deterministic_snapshot
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
from crypto_chatbot.schemas.analysis import AnalysisSnapshot
from crypto_chatbot.schemas.derivatives import DerivativesContext
from crypto_chatbot.schemas.market import MarketType, Timeframe
from crypto_chatbot.storage.database import create_db_engine, init_db
from crypto_chatbot.storage.repositories import (
    DerivativesRepository,
    MarketRepository,
    PostgresDerivativesRepository,
    PostgresMarketRepository,
    PostgresUserMemoryRepository,
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


def _build_repositories_from_env() -> tuple[MarketRepository, DerivativesRepository, Any]:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required when --persist is set")
    engine = create_db_engine(database_url)
    init_db(engine)
    return PostgresMarketRepository(engine), PostgresDerivativesRepository(engine), engine


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
    repository: MarketRepository | None,
    derivatives_repository: DerivativesRepository | None,
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
    snapshot = build_analysis_snapshot(batch, derivatives_context=derivatives_context)
    if repository is not None:
        if batch.candles:
            repository.save_candles(batch.candles)
        repository.save_snapshot(snapshot)
    return snapshot


async def _analyze_live(
    *,
    symbol: str,
    timeframe: Timeframe,
    include_mtf: bool,
    include_derivatives: bool,
    repository: MarketRepository | None,
    derivatives_repository: DerivativesRepository | None,
) -> AnalysisSnapshot:
    async with CcxtMarketDataProvider() as provider:
        if include_derivatives:
            async with CcxtDerivativesLiquidityProvider() as deriv_provider:
                service = AnalysisService(
                    provider=provider,
                    repository=repository,
                    derivatives_provider=deriv_provider,
                    derivatives_repository=derivatives_repository,
                )
                return await service.analyze(
                    symbol,
                    timeframe,
                    include_mtf=include_mtf,
                    include_derivatives=True,
                    persist=repository is not None,
                )
        service = AnalysisService(provider=provider, repository=repository)
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
    as_json: bool,
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
        if persist:
            repository, derivatives_repository, engine = _build_repositories_from_env()

        if fixture_path is not None:
            snapshot = _analyze_from_fixture(
                symbol=mapping.canonical_symbol,
                timeframe=timeframe,
                fixture_path=fixture_path,
                derivatives_fixture_path=derivatives_fixture_path,
                repository=repository,
                derivatives_repository=derivatives_repository,
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
    else:
        print(render_deterministic_snapshot(snapshot, html_mode=False))
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


def _run_bot() -> int:
    try:
        settings = load_settings()
    except ConfigError as exc:
        print(exc, file=sys.stderr)
        return 1
    run_polling(settings)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="crypto-chatbot")
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser("check-config", help="Validate configuration without printing secrets.")

    analyze_parser = subcommands.add_parser(
        "analyze",
        help="Run deterministic technical analysis from a fixture or live CCXT data.",
    )
    analyze_parser.add_argument("symbol", help="Target symbol (e.g. BTC, BTC/USDT, ETH, SOL).")
    analyze_parser.add_argument("timeframe", help="Timeframe (1h, 4h, 1d).")
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
        "--json",
        action="store_true",
        help="Print raw AnalysisSnapshot JSON instead of formatted Indonesian report.",
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

    args = parser.parse_args(argv)
    if args.command == "check-config":
        return _check_config()
    if args.command == "analyze":
        return _run_analyze(
            symbol_raw=args.symbol,
            timeframe_raw=args.timeframe,
            fixture_path=args.fixture,
            derivatives_fixture_path=args.derivatives_fixture,
            as_json=args.json,
            persist=args.persist,
            include_mtf=not args.no_mtf,
            include_derivatives=not args.no_derivatives,
        )
    if args.command == "run-bot":
        return _run_bot()
    if args.command == "purge-expired":
        return _run_purge_expired()
    parser.error(f"unknown command: {args.command}")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
