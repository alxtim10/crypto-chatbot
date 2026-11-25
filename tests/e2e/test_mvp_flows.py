from __future__ import annotations

import io
import json
import logging
import os
import re
import shutil
import socket
import subprocess
import tempfile
from collections.abc import Iterator, Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import ccxt
import pytest
from sqlalchemy.exc import OperationalError
from telegram.constants import ParseMode
from telegram.ext import ApplicationBuilder

from crypto_chatbot.analysis.service import AnalysisService
from crypto_chatbot.bot.access import AccessController
from crypto_chatbot.bot.handlers import (
    BotService,
    SecretRedactingFilter,
    register_handlers,
)
from crypto_chatbot.bot.templates import TELEGRAM_MAX_MESSAGE_LENGTH
from crypto_chatbot.llm.gemini import (
    DailyQuotaTracker,
    GeminiNarrator,
    RawGeminiOutput,
)
from crypto_chatbot.market_data.ccxt_provider import CcxtMarketDataProvider
from crypto_chatbot.memory.service import MemoryService
from crypto_chatbot.storage.database import create_db_engine, init_db
from crypto_chatbot.storage.models import Base
from crypto_chatbot.storage.repositories import (
    ForgetResult,
    LlmInteractionRecord,
    PostgresMarketRepository,
    PostgresUserMemoryRepository,
    PurgeResult,
    StoredMessage,
    UserMemoryRepository,
    UserPreferences,
)

FIXTURE_1000 = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"
ANCHOR_NOW = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
_TF_DURATION_MS = {
    "15m": 900_000,
    "1h": 3_600_000,
    "4h": 14_400_000,
    "1d": 86_400_000,
}


class _E2ECcxtClient:
    """Offline fake CCXT exchange returning deterministic closed candles (plus 1 open bar)."""

    id = "binance"

    def __init__(
        self,
        *,
        candle_count: int = 1000,
        anchor_close: datetime = ANCHOR_NOW,
        error: BaseException | None = None,
    ) -> None:
        self._raw = json.loads(FIXTURE_1000.read_text(encoding="utf-8"))["raw_ohlcv"]
        self._candle_count = candle_count
        self._anchor_close = anchor_close
        self._error = error
        self.calls: list[tuple[str, str, int | None]] = []

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
    ) -> list[list[Any]]:
        del since
        self.calls.append((symbol, timeframe, limit))
        if self._error is not None:
            raise self._error

        step_ms = _TF_DURATION_MS[timeframe]
        clean_sym = symbol.split(":")[0]
        scale = 1.0 if clean_sym == "BTC/USDT" else 0.05 if clean_sym == "ETH/USDT" else 0.0025
        closed_count = min(self._candle_count, max(1, (limit or 1000) - 1))
        rows_slice = self._raw[-closed_count:]
        end_ms = int(self._anchor_close.timestamp() * 1000)
        start_ms = end_ms - closed_count * step_ms

        out: list[list[Any]] = []
        for idx, row in enumerate(rows_slice):
            open_ms = start_ms + idx * step_ms
            out.append(
                [
                    open_ms,
                    round(float(row[1]) * scale, 4),
                    round(float(row[2]) * scale, 4),
                    round(float(row[3]) * scale, 4),
                    round(float(row[4]) * scale, 4),
                    float(row[5]),
                ]
            )
        # Append currently open candle that must be excluded by CcxtMarketDataProvider
        last = out[-1]
        out.append([end_ms, last[4], last[4] * 1.005, last[4] * 0.995, last[4], 42.0])
        return out

    async def close(self) -> None:
        return None


class _E2EGeminiTransport:
    def __init__(self, outcomes: Sequence[RawGeminiOutput | BaseException | float] = ()) -> None:
        self._outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    async def generate_json(
        self,
        *,
        model: str,
        system_instruction: str,
        contents: str,
        response_json_schema: dict[str, Any],
        max_output_tokens: int,
        temperature: float,
    ) -> RawGeminiOutput:
        self.calls.append({"model": model, "contents": contents})
        del system_instruction, response_json_schema, max_output_tokens, temperature
        if self._outcomes:
            outcome = self._outcomes.pop(0) if len(self._outcomes) > 1 else self._outcomes[0]
            if isinstance(outcome, float):
                import asyncio

                await asyncio.sleep(outcome)
                return RawGeminiOutput(text="{}")
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

        valid = json.dumps(
            {
                "summary": "Pada timeframe 4h, harga penutupan {F1} berada di atas EMA 200 ({F6}).",
                "supporting_factors": ["EMA 20 ({F4}) berada di atas EMA 50 ({F5})."],
                "contradicting_factors": ["Arah jangka pendek tetap tidak pasti."],
                "bullish_scenario": "Selama bertahan di atas {F6}, struktur tetap terjaga.",
                "bearish_scenario": "Penurunan menembus {F6} melemahkan struktur.",
                "limitations": ["Data spot Binance saja."],
                "fact_ids": ["F1", "F4", "F5", "F6"],
            }
        )
        return RawGeminiOutput(
            text=valid,
            model_version="gemini-flash-e2e",
            prompt_tokens=350,
            output_tokens=120,
        )


class _InMemoryMemoryRepo:
    def __init__(self) -> None:
        self.users: set[int] = set()
        self.preferences: dict[int, UserPreferences] = {}
        self.messages: dict[int, list[StoredMessage]] = {}
        self.interactions: list[LlmInteractionRecord] = []

    def ensure_user(self, telegram_user_id: int) -> None:
        self.users.add(telegram_user_id)

    def get_preferences(self, telegram_user_id: int) -> UserPreferences:
        return self.preferences.get(telegram_user_id, UserPreferences())

    def save_preferences(self, telegram_user_id: int, prefs: UserPreferences) -> None:
        self.users.add(telegram_user_id)
        self.preferences[telegram_user_id] = prefs

    def append_message(
        self,
        telegram_user_id: int,
        *,
        role: Literal["user", "assistant"],
        content: str,
        snapshot_id: str | None = None,
        max_messages: int = 20,
        created_at: datetime | None = None,
    ) -> StoredMessage:
        self.users.add(telegram_user_id)
        msg = StoredMessage(
            role=role,
            content=content,
            snapshot_id=snapshot_id,
            created_at=created_at or ANCHOR_NOW,
        )
        bucket = self.messages.setdefault(telegram_user_id, [])
        bucket.append(msg)
        if len(bucket) > max_messages:
            del bucket[: len(bucket) - max_messages]
        return msg

    def get_recent_messages(
        self,
        telegram_user_id: int,
        *,
        limit: int = 20,
    ) -> tuple[StoredMessage, ...]:
        return tuple(self.messages.get(telegram_user_id, [])[-limit:])

    def record_llm_interaction(self, record: LlmInteractionRecord) -> None:
        self.interactions.append(record)

    def forget_user_data(self, telegram_user_id: int) -> ForgetResult:
        msgs = len(self.messages.pop(telegram_user_id, []))
        prefs = 1 if self.preferences.pop(telegram_user_id, None) is not None else 0
        before = len(self.interactions)
        self.interactions = [r for r in self.interactions if r.telegram_user_id != telegram_user_id]
        return ForgetResult(
            deleted_messages=msgs,
            deleted_conversations=1 if msgs else 0,
            deleted_preferences=prefs,
            deleted_interactions=before - len(self.interactions),
        )

    def purge_expired(
        self,
        *,
        now: datetime,
        message_retention_days: int = 30,
        interaction_retention_days: int = 30,
        snapshot_retention_days: int = 90,
    ) -> PurgeResult:
        del now, message_retention_days, interaction_retention_days, snapshot_retention_days
        return PurgeResult(
            deleted_messages=0,
            deleted_conversations=0,
            deleted_interactions=0,
            deleted_snapshots=0,
        )


class _PtbHarness:
    """Dispatches updates through real PTB `CommandHandler` and `MessageHandler` wiring."""

    def __init__(self, bot_service: BotService) -> None:
        self.bot_service = bot_service
        self.app = (
            ApplicationBuilder().token("123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi").build()
        )
        register_handlers(self.app, bot_service)
        self._handlers = self.app.handlers[0]

    async def send(
        self,
        text: str,
        *,
        user_id: int = 111,
    ) -> list[str]:
        replies: list[str] = []

        class _Msg:
            def __init__(self, raw_text: str) -> None:
                self.text = raw_text

            async def reply_text(
                self,
                reply: str,
                parse_mode: str | None = None,
                disable_web_page_preview: bool | None = None,
            ) -> None:
                del disable_web_page_preview
                assert parse_mode == ParseMode.HTML
                assert len(reply) <= TELEGRAM_MAX_MESSAGE_LENGTH
                replies.append(reply)

        update: Any = SimpleNamespace(
            effective_user=SimpleNamespace(id=user_id),
            effective_message=_Msg(text),
        )
        stripped = text.strip()
        if stripped.startswith("/"):
            parts = stripped.split()
            cmd = parts[0][1:].split("@")[0].lower()
            args = parts[1:]
            cmd_map = {
                "start": 0,
                "help": 1,
                "analyze": 2,
                "compare": 3,
                "pine": 4,
                "scan": 5,
                "screen": 6,
                "watchlist": 7,
                "memory": 8,
                "forget": 9,
            }
            idx = cmd_map.get(cmd, 10)
            ctx: Any = SimpleNamespace(args=args)
        else:
            idx = 11
            ctx = SimpleNamespace(args=[])

        await self._handlers[idx].callback(update, ctx)
        return replies


def _build_stack(
    *,
    ccxt_client: _E2ECcxtClient | None = None,
    clock_now: datetime = ANCHOR_NOW + timedelta(seconds=5),
    gemini_outcomes: Sequence[RawGeminiOutput | BaseException | float] = (),
    gemini_timeout_seconds: float = 5.0,
    user_daily_cap: int = 30,
    global_daily_cap: int = 200,
    rate_limit_per_minute: int = 50,
    allowed_ids: Sequence[int] = (111, 222),
    memory_repo: UserMemoryRepository | None = None,
) -> tuple[_PtbHarness, _E2ECcxtClient, _E2EGeminiTransport, UserMemoryRepository]:
    client = ccxt_client or _E2ECcxtClient()
    provider = CcxtMarketDataProvider(
        client=client,
        max_attempts=2,
        retry_initial_seconds=0.0,
        retry_max_seconds=0.0,
        clock=lambda: clock_now,
    )
    analysis_service = AnalysisService(provider=provider)
    transport = _E2EGeminiTransport(gemini_outcomes)
    quota = DailyQuotaTracker(
        per_user_daily_cap=user_daily_cap,
        global_daily_cap=global_daily_cap,
        clock=lambda: ANCHOR_NOW,
    )
    narrator = GeminiNarrator(
        model="gemini-flash-e2e",
        transport=transport,
        timeout_seconds=gemini_timeout_seconds,
        quota_tracker=quota,
    )
    repo: UserMemoryRepository = memory_repo or _InMemoryMemoryRepo()
    memory = MemoryService(repo)
    access = AccessController(allowed_ids, rate_limit_per_minute=rate_limit_per_minute)
    bot_service = BotService(
        access=access,
        analysis_service=analysis_service,
        narrator=narrator,
        memory=memory,
    )
    return _PtbHarness(bot_service), client, transport, repo


@pytest.mark.asyncio
async def test_e2e_complete_mvp_user_journey() -> None:
    harness, _, transport, repo = _build_stack()

    start_out = "\n".join(await harness.send("/start"))
    assert "Selamat datang" in start_out
    assert "Disclaimer:" in start_out

    analyze_out = "\n".join(await harness.send("/analyze BTC 4h"))
    assert "BTC/USDT:USDT (4h)" in analyze_out
    assert "Status kualitas: READY" in analyze_out
    assert "ccxt:BINANCE:BTCUSDT.P" in analyze_out
    assert "UTC (" in analyze_out
    assert "WIB)" in analyze_out
    assert "[F1]" in analyze_out
    assert "Bukan Probabilitas" in analyze_out
    assert "Penjelasan Narasi (Tervalidasi)" in analyze_out
    assert "65.577,38 USDT" in analyze_out

    followup_out = "\n".join(await harness.send("bandingkan momentumnya dengan ETH"))
    assert "Perbandingan Deskriptif (BTC/USDT:USDT, ETH/USDT:USDT - 4h)" in followup_out
    assert "ccxt:BINANCE:BTCUSDT.P" in followup_out
    assert "ccxt:BINANCE:ETHUSDT.P" in followup_out

    compare_out = "\n".join(await harness.send("/compare BTC ETH 1d"))
    assert "Perbandingan Deskriptif (BTC/USDT:USDT, ETH/USDT:USDT - 1d)" in compare_out
    assert "BTC/USDT:USDT (1d)" in compare_out
    assert "ETH/USDT:USDT (1d)" in compare_out

    memory_out = "\n".join(await harness.send("/memory"))
    assert "Pesan tersimpan (maks 20): 6 pesan" in memory_out
    recent = repo.get_recent_messages(111)
    assert len(recent) == 6
    assert all("65.577" not in msg.content for msg in recent if msg.role == "assistant")
    assert all(msg.snapshot_id is not None for msg in recent if msg.role == "assistant")

    forget_out = "\n".join(await harness.send("/forget"))
    assert "Penghapusan Memori Selesai" in forget_out
    assert "Pesan percakapan: 6" in forget_out

    after_forget = "\n".join(await harness.send("/memory"))
    assert "Pesan tersimpan (maks 20): 0 pesan" in after_forget
    assert len(transport.calls) == 5


@pytest.mark.asyncio
async def test_e2e_failure_exchange_unavailable() -> None:
    client = _E2ECcxtClient(error=ccxt.NetworkError("exchange unreachable"))
    harness, _, transport, _ = _build_stack(ccxt_client=client)

    reply = "\n".join(await harness.send("/analyze BTC 4h"))
    assert "INSUFFICIENT_DATA" in reply
    assert "PROVIDER_ERROR" in reply
    assert "UTC (" in reply
    assert "WIB)" in reply
    assert transport.calls == []


@pytest.mark.asyncio
async def test_e2e_failure_stale_candles() -> None:
    # _E2ECcxtClient appends the 12:00..16:00 bar; checking at 20:10 UTC exceeds 4h + 5m grace.
    stale_clock = ANCHOR_NOW + timedelta(hours=8, minutes=10)
    harness, _, transport, _ = _build_stack(clock_now=stale_clock)

    reply = "\n".join(await harness.send("/analyze BTC 4h"))
    assert "INSUFFICIENT_DATA" in reply
    assert "STALE" in reply
    assert "Fakta Terukur" not in reply
    assert "65.577" not in reply
    assert transport.calls == []


@pytest.mark.asyncio
async def test_e2e_failure_insufficient_warmup_and_degraded() -> None:
    short_harness, _, transport_short, _ = _build_stack(ccxt_client=_E2ECcxtClient(candle_count=50))
    short_reply = "\n".join(await short_harness.send("/analyze BTC 4h"))
    assert "INSUFFICIENT_DATA" in short_reply
    assert "SHORT_HISTORY" in short_reply
    assert transport_short.calls == []

    degraded_harness, _, _, _ = _build_stack(ccxt_client=_E2ECcxtClient(candle_count=200))
    degraded_reply = "\n".join(await degraded_harness.send("/analyze BTC 4h"))
    assert "Status kualitas: DEGRADED" in degraded_reply
    assert "SHORT_HISTORY" in degraded_reply
    assert "EMA 200:" not in degraded_reply


@pytest.mark.asyncio
async def test_e2e_failure_gemini_unavailable_over_cap_and_hallucination() -> None:
    # 1. Gemini timeout -> deterministic fallback
    timeout_harness, _, _, _ = _build_stack(
        gemini_outcomes=[0.1],
        gemini_timeout_seconds=0.02,
    )
    timeout_reply = "\n".join(await timeout_harness.send("/analyze BTC 4h"))
    assert "Fakta Terukur (Backend)" in timeout_reply
    assert "65.577,38 USDT" in timeout_reply
    assert "Penjelasan Narasi" not in timeout_reply

    # 2. Gemini hallucination (unsupported price + certainty probability) -> rejected
    hallucination = json.dumps(
        {
            "summary": "Harga {F1} pasti naik menuju target 99999 dengan peluang 80%.",
            "bullish_scenario": "Naik ke 105000.",
            "bearish_scenario": "Turun dari {F6}.",
            "fact_ids": ["F1", "F6"],
        }
    )
    bad_harness, _, _, _ = _build_stack(gemini_outcomes=[RawGeminiOutput(text=hallucination)])
    bad_reply = "\n".join(await bad_harness.send("/analyze BTC 4h"))
    assert "99999" not in bad_reply
    assert "105000" not in bad_reply
    assert "pasti naik" not in bad_reply.lower()
    assert "80%" not in bad_reply
    assert "Fakta Terukur (Backend)" in bad_reply

    # 3. Daily cap exceeded -> second call skips Gemini and returns deterministic output
    cap_harness, _, cap_transport, _ = _build_stack(user_daily_cap=1)
    first_reply = "\n".join(await cap_harness.send("/analyze BTC 4h"))
    second_reply = "\n".join(await cap_harness.send("/analyze ETH 4h"))
    assert "Penjelasan Narasi (Tervalidasi)" in first_reply
    assert "Penjelasan Narasi" not in second_reply
    assert "Fakta Terukur (Backend)" in second_reply
    assert len(cap_transport.calls) == 1


@pytest.mark.asyncio
async def test_e2e_failure_malformed_symbol_and_hostile_html() -> None:
    harness, client, transport, _ = _build_stack()

    for cmd in (
        "/analyze BTC/EUR 4h",
        "/analyze BTC-SPOT 4h",
        "/analyze <script>alert()</script> 4h",
        "/<evil_cmd>",
    ):
        reply = "\n".join(await harness.send(cmd))
        assert "<script>" not in reply
        assert "<evil_cmd>" not in reply
        assert reply.strip() != ""

    assert client.calls == []
    assert transport.calls == []


@pytest.mark.asyncio
async def test_e2e_failure_request_flood_and_non_allowlisted_user() -> None:
    harness, client, transport, repo = _build_stack(
        allowed_ids=(111,),
        rate_limit_per_minute=2,
    )

    unauth_reply = "\n".join(await harness.send("/analyze BTC 4h", user_id=999))
    assert "dibatasi" in unauth_reply.lower()
    assert client.calls == []
    assert transport.calls == []
    assert repo.get_recent_messages(999) == ()

    # Allowlisted user flood: 2 allowed, 3rd blocked by rate limiter
    await harness.send("/start", user_id=111)
    await harness.send("/help", user_id=111)
    flood_reply = "\n".join(await harness.send("/analyze BTC 4h", user_id=111))
    assert "batas frekuensi" in flood_reply.lower()
    assert client.calls == []
    assert transport.calls == []


@pytest.mark.asyncio
async def test_e2e_failure_database_unavailable_and_log_redaction() -> None:
    db_secret = "TopSecretDbPw99"
    tg_token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
    gemini_key = "AIzaSySecretKeyForRedactionTest123456"

    class _FailingMemoryRepo(_InMemoryMemoryRepo):
        def _boom(self) -> OperationalError:
            return OperationalError(
                "SELECT 1",
                {},
                RuntimeError(f"postgresql+psycopg://u:{db_secret}@db:5432/crypto_chatbot"),
            )

        def ensure_user(self, telegram_user_id: int) -> None:
            del telegram_user_id
            raise self._boom()

        def get_preferences(self, telegram_user_id: int) -> UserPreferences:
            del telegram_user_id
            raise self._boom()

        def get_recent_messages(
            self,
            telegram_user_id: int,
            *,
            limit: int = 20,
        ) -> tuple[StoredMessage, ...]:
            del telegram_user_id, limit
            raise self._boom()

        def forget_user_data(self, telegram_user_id: int) -> ForgetResult:
            del telegram_user_id
            raise self._boom()

    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    handler.addFilter(SecretRedactingFilter(secrets=(db_secret, tg_token, gemini_key)))
    handler_logger = logging.getLogger("crypto_chatbot.bot.handlers")
    prev_level = handler_logger.level
    handler_logger.setLevel(logging.INFO)
    handler_logger.addHandler(handler)

    try:
        harness, _, _, _ = _build_stack(memory_repo=_FailingMemoryRepo())
        for cmd in ("/start", "/analyze BTC 4h", "/memory", "/forget"):
            reply = "\n".join(await harness.send(cmd))
            assert "database" in reply.lower()
            assert db_secret not in reply
    finally:
        handler_logger.removeHandler(handler)
        handler_logger.setLevel(prev_level)

    logs = log_buf.getvalue()
    assert "[REDACTED]" in logs
    assert db_secret not in logs
    assert tg_token not in logs
    assert gemini_key not in logs


def _run_cmd(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, check=False, text=True, capture_output=True)  # noqa: S603


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@pytest.fixture
def pg_engine_url(tmp_path: Path) -> Iterator[str]:
    configured = os.environ.get("TEST_DATABASE_URL", "").strip()
    if configured:
        engine = create_db_engine(configured)
        Base.metadata.drop_all(engine)
        engine.dispose()
        yield configured
        return

    initdb_bin = shutil.which("initdb")
    pg_ctl_bin = shutil.which("pg_ctl")
    psql_bin = shutil.which("psql")
    if not initdb_bin or not pg_ctl_bin or not psql_bin:
        pytest.skip("PostgreSQL binaries (initdb/pg_ctl/psql) are not available")

    data_dir = tmp_path / "pgdata"
    sock_dir = Path(tempfile.mkdtemp(prefix="ccpg_", dir="/tmp"))
    log_file = tmp_path / "postgres.log"
    port = _free_port()

    try:
        init_res = _run_cmd(
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

        start_res = _run_cmd(
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
            pytest.skip(f"pg_ctl start failed: {start_res.stderr.strip()}")

        try:
            db_res = _run_cmd(
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
                    "CREATE DATABASE crypto_chatbot_e2e;",
                ]
            )
            if db_res.returncode != 0:
                pytest.skip(f"CREATE DATABASE failed: {db_res.stderr.strip()}")
            yield (
                f"postgresql+psycopg://postgres@localhost:{port}/crypto_chatbot_e2e?host={sock_dir}"
            )
        finally:
            _run_cmd([pg_ctl_bin, "-D", str(data_dir), "-m", "immediate", "-w", "stop"])
    finally:
        shutil.rmtree(sock_dir, ignore_errors=True)


@pytest.mark.asyncio
async def test_e2e_mvp_flow_with_real_postgres(pg_engine_url: str) -> None:
    engine = create_db_engine(pg_engine_url)
    init_db(engine)

    try:
        market_repo = PostgresMarketRepository(engine)
        memory_repo = PostgresUserMemoryRepository(engine)
        provider = CcxtMarketDataProvider(
            client=_E2ECcxtClient(),
            max_attempts=1,
            retry_initial_seconds=0.0,
            retry_max_seconds=0.0,
            clock=lambda: ANCHOR_NOW + timedelta(seconds=5),
        )
        analysis_service = AnalysisService(provider=provider, repository=market_repo)
        narrator = GeminiNarrator(model="gemini-flash-e2e", transport=_E2EGeminiTransport())
        bot = BotService(
            access=AccessController((111,)),
            analysis_service=analysis_service,
            narrator=narrator,
            memory=MemoryService(memory_repo),
        )
        harness = _PtbHarness(bot)

        await harness.send("/start")
        wl_reply = "\n".join(await harness.send("/watchlist add BTC"))
        assert "BTC/USDT ditambahkan" in wl_reply

        btc_reply = "\n".join(await harness.send("/analyze BTC 4h"))
        assert "BTC/USDT (4h)" in btc_reply
        snap_match = re.search(r"Snapshot:\s*(snap-[0-9a-f]+)", btc_reply)
        assert snap_match is not None
        btc_snap_id = snap_match.group(1)

        followup_reply = "\n".join(await harness.send("bandingkan momentumnya dengan ETH"))
        assert "Perbandingan Deskriptif (BTC/USDT, ETH/USDT - 4h)" in followup_reply

        cmp_reply = "\n".join(await harness.send("/compare BTC ETH 1d"))
        assert "Perbandingan Deskriptif (BTC/USDT, ETH/USDT - 1d)" in cmp_reply

        mem_reply = "\n".join(await harness.send("/memory"))
        assert "Watchlist: BTC/USDT" in mem_reply
        assert "Pesan tersimpan (maks 20): 6 pesan" in mem_reply

        # Snapshot persisted in PostgreSQL recomputes identically from stored candles
        recomputed = analysis_service.recompute_from_stored(btc_snap_id)
        stored = market_repo.get_snapshot(btc_snap_id)
        assert stored is not None
        assert recomputed.input_range == stored.input_range
        assert recomputed.facts == stored.facts

        # /forget removes user data cleanly while preserving public snapshots & candles
        forget_reply = "\n".join(await harness.send("/forget"))
        assert "Pesan percakapan: 6" in forget_reply
        assert "Preferensi pengguna: 1" in forget_reply
        assert memory_repo.get_recent_messages(111) == ()
        assert memory_repo.get_preferences(111).watchlist == ()
        assert market_repo.get_snapshot(btc_snap_id) is not None

        # Retention purge runs cleanly
        purge_res = memory_repo.purge_expired(now=datetime.now(UTC) + timedelta(days=91))
        assert purge_res.deleted_snapshots >= 1
    finally:
        engine.dispose()
