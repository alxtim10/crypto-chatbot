from __future__ import annotations

import json
import logging
import re
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
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
from crypto_chatbot.llm.gemini import GeminiNarrator, RawGeminiOutput
from crypto_chatbot.market_data.base import DEFAULT_CANDLE_LIMIT, OhlcvBatch
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import (
    resolve_default_perp_symbol,
)
from crypto_chatbot.memory.service import MemoryService
from crypto_chatbot.schemas.market import Timeframe
from crypto_chatbot.storage.repositories import (
    ForgetResult,
    LlmInteractionRecord,
    PurgeResult,
    StoredMessage,
    UserPreferences,
)

FIXTURE_1000 = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"


class InMemoryUserMemoryRepository:
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
            created_at=created_at or datetime.now(UTC),
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
        bucket = self.messages.get(telegram_user_id, [])
        return tuple(bucket[-limit:])

    def record_llm_interaction(self, record: LlmInteractionRecord) -> None:
        self.interactions.append(record)

    def forget_user_data(self, telegram_user_id: int) -> ForgetResult:
        msgs = len(self.messages.pop(telegram_user_id, []))
        prefs = 1 if self.preferences.pop(telegram_user_id, None) is not None else 0
        before = len(self.interactions)
        self.interactions = [
            item for item in self.interactions if item.telegram_user_id != telegram_user_id
        ]
        ints = before - len(self.interactions)
        return ForgetResult(
            deleted_messages=msgs,
            deleted_conversations=1 if msgs else 0,
            deleted_preferences=prefs,
            deleted_interactions=ints,
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


class _FixtureProvider:
    def __init__(self) -> None:
        self._raw = json.loads(FIXTURE_1000.read_text(encoding="utf-8"))["raw_ohlcv"]

    async def get_closed_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = DEFAULT_CANDLE_LIMIT,
    ) -> OhlcvBatch:
        mapping = resolve_default_perp_symbol(symbol)
        # Re-stamp timestamps onto clean grid for the requested timeframe so MTF is valid.
        now_anchor = datetime(2026, 11, 21, 12, 0, tzinfo=UTC)
        rows = self._raw[-limit:]
        start_open = now_anchor - len(rows) * timeframe.duration
        reindexed = [
            [
                (start_open + idx * timeframe.duration).timestamp() * 1000.0,
                row[1],
                row[2],
                row[3],
                row[4],
                row[5],
            ]
            for idx, row in enumerate(rows)
        ]
        return normalize_ohlcv_rows(
            reindexed,
            symbol=mapping,
            timeframe=timeframe,
            received_time=now_anchor + timedelta(seconds=1),
            now=now_anchor + timedelta(seconds=1),
        )

    async def close(self) -> None:
        return None


class _FakeGeminiTransport:
    def __init__(self, texts: list[str] | None = None) -> None:
        self._texts = list(texts) if texts else []
        self.calls: list[str] = []

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
        del model, system_instruction, response_json_schema, max_output_tokens, temperature
        self.calls.append(contents)
        payload = (
            self._texts.pop(0)
            if self._texts
            else json.dumps(
                {
                    "summary": "Penutupan {F1} berada di atas EMA 200 ({F6}) pada timeframe 4h.",
                    "supporting_factors": ["EMA 20 ({F4}) lebih tinggi dari EMA 50 ({F5})."],
                    "contradicting_factors": ["Arah jangka pendek tetap tidak pasti."],
                    "bullish_scenario": "Ketahanan di atas {F6} menjaga struktur saat ini.",
                    "bearish_scenario": "Penembusan ke bawah {F6} melemahkan struktur.",
                    "limitations": ["Hanya data spot Binance."],
                    "fact_ids": ["F1", "F4", "F5", "F6"],
                }
            )
        )
        return RawGeminiOutput(
            text=payload,
            model_version="gemini-flash-test",
            prompt_tokens=300,
            output_tokens=120,
        )


def _build_bot_stack(
    *,
    allowed_ids: Sequence[int] = (111, 222),
    gemini_texts: list[str] | None = None,
) -> tuple[BotService, InMemoryUserMemoryRepository, _FakeGeminiTransport]:
    repo = InMemoryUserMemoryRepository()
    provider = _FixtureProvider()
    analysis_service = AnalysisService(provider=provider)
    transport = _FakeGeminiTransport(gemini_texts)
    narrator = GeminiNarrator(model="gemini-flash-test", transport=transport)
    memory = MemoryService(repo)
    access = AccessController(allowed_ids, rate_limit_per_minute=50)
    bot = BotService(
        access=access,
        analysis_service=analysis_service,
        narrator=narrator,
        memory=memory,
    )
    return bot, repo, transport


@pytest.mark.asyncio
async def test_non_allowlisted_user_refused_without_persisting_anything() -> None:
    bot, repo, transport = _build_bot_stack(allowed_ids=(111,))

    for chunks in (
        await bot.handle_start(999),
        await bot.handle_analyze(999, ["BTC", "4h"]),
        await bot.handle_watchlist(999, ["add", "BTC"]),
        await bot.handle_text_message(999, "analisis BTC 4h"),
    ):
        assert len(chunks) == 1
        assert "dibatasi" in chunks[0].lower()

    assert 999 not in repo.users
    assert repo.messages == {}
    assert repo.preferences == {}
    assert repo.interactions == []
    assert transport.calls == []


@pytest.mark.asyncio
async def test_user_isolation_and_forget_scope() -> None:
    bot, repo, _ = _build_bot_stack(allowed_ids=(111, 222))

    await bot.handle_watchlist(111, ["add", "BTC"])
    await bot.handle_watchlist(222, ["add", "ETH"])
    await bot.handle_analyze(111, ["BTC", "4h"])
    await bot.handle_analyze(222, ["ETH", "1d"])

    mem_111 = "\n".join(await bot.handle_memory(111))
    mem_222 = "\n".join(await bot.handle_memory(222))
    assert "BTC/USDT" in mem_111
    assert "ETH/USDT" not in mem_111
    assert "ETH/USDT" in mem_222
    assert "BTC/USDT" not in mem_222

    forget_msg = "\n".join(await bot.handle_forget(111))
    assert "Penghapusan Memori Selesai" in forget_msg
    assert repo.get_recent_messages(111) == ()
    assert repo.get_preferences(111).watchlist == ()
    assert [i for i in repo.interactions if i.telegram_user_id == 111] == []

    # User 222 remains completely untouched
    assert len(repo.get_recent_messages(222)) == 2
    assert repo.get_preferences(222).watchlist == ("ETH/USDT:USDT",)
    assert len([i for i in repo.interactions if i.telegram_user_id == 222]) == 1


@pytest.mark.asyncio
async def test_memory_stays_bounded_at_20_and_stores_placeholders_not_numbers() -> None:
    bot, repo, _ = _build_bot_stack(allowed_ids=(111,))

    # 13 analyze turns = 26 user+assistant messages -> bounded to 20
    for _ in range(13):
        await bot.handle_analyze(111, ["BTC", "4h"])

    recent = repo.get_recent_messages(111)
    assert len(recent) == 20
    assistant_msgs = [m for m in recent if m.role == "assistant"]
    assert assistant_msgs
    for msg in assistant_msgs:
        assert msg.snapshot_id is not None
        assert "{F1}" in msg.content
        assert "65.577" not in msg.content
        assert "62.923" not in msg.content


@pytest.mark.asyncio
async def test_old_numbers_from_memory_cannot_pass_output_guard() -> None:
    # Even if Gemini repeats an old rendered price number from context, output guard rejects it.
    poisoned_reply = json.dumps(
        {
            "summary": "Seperti sebelumnya, harga masih 65.577,38 USDT di atas {F1}.",
            "bullish_scenario": "Bertahan di {F6}.",
            "bearish_scenario": "Turun dari {F6}.",
            "fact_ids": ["F1", "F6"],
        }
    )
    bot, repo, _ = _build_bot_stack(allowed_ids=(111,), gemini_texts=[poisoned_reply])

    chunks = await bot.handle_analyze(111, ["BTC", "4h"])
    full = "\n".join(chunks)
    assert "Fakta Terukur (Backend)" in full
    assert "Penjelasan Narasi" not in full
    assert repo.interactions[-1].validation_passed is False
    assert "raw_digit_detected" in repo.interactions[-1].rejection_reasons


@pytest.mark.asyncio
async def test_ptb_command_and_followup_handlers_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bot, _, _ = _build_bot_stack(allowed_ids=(111,))
    app = ApplicationBuilder().token("123456:ABCdefGHIjklMNOpqrsTUVwxyz0123456789").build()
    register_handlers(app, bot)

    handlers = app.handlers[0]
    assert len(handlers) == 13

    replies: list[tuple[str, Any]] = []

    class _FakeMsg:
        def __init__(self, text: str) -> None:
            self.text = text

        async def reply_text(
            self,
            text: str,
            parse_mode: str | None = None,
            disable_web_page_preview: bool | None = None,
        ) -> None:
            del disable_web_page_preview
            replies.append((text, parse_mode))

    async def invoke(idx: int, text: str, args: list[str] | None = None) -> list[str]:
        replies.clear()
        update: Any = SimpleNamespace(
            effective_user=SimpleNamespace(id=111),
            effective_message=_FakeMsg(text),
        )
        ctx: Any = SimpleNamespace(args=args or [])
        await handlers[idx].callback(update, ctx)
        assert all(pm == ParseMode.HTML for _, pm in replies)
        assert all(len(chunk) <= TELEGRAM_MAX_MESSAGE_LENGTH for chunk, _ in replies)
        return [chunk for chunk, _ in replies]

    # 0=start..4=pine, 5=scan, 6=screen, 7=whale, 8=watchlist..11=unknown, 12=text
    start_out = "\n".join(await invoke(0, "/start"))
    assert "Selamat datang" in start_out

    analyze_out = "\n".join(await invoke(2, "/analyze BTC 4h", ["BTC", "4h"]))
    assert "BTC/USDT:USDT (4h)" in analyze_out
    assert "65.577,38 USDT" in analyze_out

    pine_out = "\n".join(await invoke(4, "/pine BTC 4h", ["BTC", "4h"]))
    assert "TradingView Pine Script v6" in pine_out
    assert "//@version=6" in pine_out

    scan_out = "\n".join(await invoke(5, "/scan BTC", ["BTC"]))
    assert "Pemindai Setup Futures" in scan_out

    whale_out = "\n".join(await invoke(7, "/whale BTC", ["BTC"]))
    assert "WHALE PERP TRACKER" in whale_out

    followup_out = "\n".join(await invoke(12, "bandingkan momentumnya dengan ETH"))
    assert "Perbandingan Deskriptif (BTC/USDT:USDT, ETH/USDT:USDT - 4h)" in followup_out
    assert "BTC/USDT:USDT (4h)" in followup_out
    assert "ETH/USDT:USDT (4h)" in followup_out

    alt_m15_out = "\n".join(await invoke(2, "/analyze DOGE 15m", ["DOGE", "15m"]))
    assert "DOGE/USDT:USDT (15m)" in alt_m15_out

    bad_cmd_out = "\n".join(await invoke(2, "/analyze BTC 5m", ["BTC", "5m"]))
    assert "tidak didukung" in bad_cmd_out.lower()

    unknown_out = "\n".join(await invoke(11, "/tidakada"))
    assert "tidak dikenali" in unknown_out.lower()

    # /screen nan / inf rejected cleanly instead of NaN propagation
    nan_out = "\n".join(await invoke(6, "/screen nan", ["nan"]))
    assert "valid" in nan_out.lower()
    inf_out = "\n".join(await invoke(6, "/screen inf", ["inf"]))
    assert "valid" in inf_out.lower()

    # Natural-language screen ValueError is caught cleanly instead of crashing text handler
    import crypto_chatbot.bot.handlers as handlers_mod

    async def _boom_screener(*args: Any, **kwargs: Any) -> Any:
        del args, kwargs
        raise ValueError("Ticker screener sementara gagal & timeout")

    monkeypatch.setattr(handlers_mod, "screen_momentum_movers", _boom_screener)
    nl_screen_out = "\n".join(await invoke(12, "tolong screening top gainers"))
    assert "gagal" in nl_screen_out.lower()
    assert "&amp;" in nl_screen_out


def test_secret_redacting_filter_strips_tokens_keys_and_db_passwords() -> None:
    tg_token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
    gemini_key = "AIzaSyTestSecretKey1234567890abcdef"
    db_pass = "SuperSecretDbPass99!"
    flt = SecretRedactingFilter(secrets=[tg_token, gemini_key, db_pass])

    rec1 = logging.LogRecord(
        name="httpx",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="HTTP Request: POST https://api.telegram.org/bot%s/sendMessage %s",
        args=(tg_token, f"postgresql+psycopg://user:{db_pass}@db:5432/app"),
        exc_info=None,
    )
    assert flt.filter(rec1) is True
    rendered1 = rec1.getMessage()
    assert tg_token not in rendered1
    assert db_pass not in rendered1
    assert "[REDACTED]" in rendered1

    rec2 = logging.LogRecord(
        name="app",
        level=logging.ERROR,
        pathname=__file__,
        lineno=1,
        msg="Key leak check: %s",
        args=(gemini_key,),
        exc_info=None,
    )
    assert flt.filter(rec2) is True
    rendered2 = rec2.getMessage()
    assert gemini_key not in rendered2
    assert re.search(r"AIza[0-9A-Za-z_-]{10,}", rendered2) is None
