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

from crypto_chatbot.analysis.service import AnalysisService, build_analysis_snapshot
from crypto_chatbot.bot.templates import render_deterministic_snapshot
from crypto_chatbot.config import ConfigError, load_settings
from crypto_chatbot.market_data.ccxt_provider import CcxtMarketDataProvider
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import SymbolError, resolve_mvp_spot_symbol
from crypto_chatbot.schemas.analysis import AnalysisSnapshot
from crypto_chatbot.schemas.market import Timeframe
from crypto_chatbot.storage.database import create_db_engine, init_db
from crypto_chatbot.storage.repositories import MarketRepository, PostgresMarketRepository


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


def _build_repository_from_env() -> MarketRepository:
    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        raise ValueError("DATABASE_URL environment variable is required when --persist is set")
    engine = create_db_engine(database_url)
    init_db(engine)
    return PostgresMarketRepository(engine)


def _analyze_from_fixture(
    *,
    symbol: str,
    timeframe: Timeframe,
    fixture_path: Path,
    repository: MarketRepository | None,
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
    snapshot = build_analysis_snapshot(batch)
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
    repository: MarketRepository | None,
) -> AnalysisSnapshot:
    async with CcxtMarketDataProvider() as provider:
        service = AnalysisService(provider=provider, repository=repository)
        return await service.analyze(
            symbol,
            timeframe,
            include_mtf=include_mtf,
            persist=repository is not None,
        )


def _run_analyze(
    *,
    symbol_raw: str,
    timeframe_raw: str,
    fixture_path: Path | None,
    as_json: bool,
    persist: bool,
    include_mtf: bool,
) -> int:
    try:
        mapping = resolve_mvp_spot_symbol(symbol_raw)
        try:
            timeframe = Timeframe(timeframe_raw.strip().lower())
        except ValueError:
            supported = ", ".join(tf.value for tf in Timeframe)
            raise ValueError(
                f"unsupported timeframe '{timeframe_raw}' (supported: {supported})"
            ) from None

        repository = _build_repository_from_env() if persist else None
        if fixture_path is not None:
            snapshot = _analyze_from_fixture(
                symbol=mapping.canonical_symbol,
                timeframe=timeframe,
                fixture_path=fixture_path,
                repository=repository,
            )
        else:
            snapshot = asyncio.run(
                _analyze_live(
                    symbol=mapping.canonical_symbol,
                    timeframe=timeframe,
                    include_mtf=include_mtf,
                    repository=repository,
                )
            )
    except (SymbolError, ValueError, json.JSONDecodeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    if as_json:
        print(snapshot.model_dump_json(indent=2))
    else:
        print(render_deterministic_snapshot(snapshot, html_mode=False))
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

    args = parser.parse_args(argv)
    if args.command == "check-config":
        return _check_config()
    if args.command == "analyze":
        return _run_analyze(
            symbol_raw=args.symbol,
            timeframe_raw=args.timeframe,
            fixture_path=args.fixture,
            as_json=args.json,
            persist=args.persist,
            include_mtf=not args.no_mtf,
        )
    parser.error(f"unknown command: {args.command}")  # pragma: no cover


if __name__ == "__main__":
    raise SystemExit(main())
