"""Telegram command/message service, PTB handlers, and secret-redacting log filter."""

from __future__ import annotations

import html
import logging
import uuid
from collections.abc import Sequence
from typing import Any

from sqlalchemy.exc import SQLAlchemyError
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from crypto_chatbot.analysis.service import AnalysisService
from crypto_chatbot.bot.access import AccessController
from crypto_chatbot.bot.commands import (
    AnalyzeIntent,
    CommandParseError,
    CompareIntent,
    RefusalIntent,
    parse_analyze_args,
    parse_compare_args,
    parse_watchlist_args,
    resolve_followup_intent,
)
from crypto_chatbot.bot.templates import (
    TELEGRAM_MAX_MESSAGE_LENGTH,
    render_forget_message,
    render_help_message,
    render_memory_message,
    render_start_message,
    render_watchlist_message,
    split_telegram_message,
)
from crypto_chatbot.llm.gemini import GeminiNarrator
from crypto_chatbot.llm.render import render_analysis_response, render_comparison_response
from crypto_chatbot.memory.service import MemoryService
from crypto_chatbot.ops.logging import SecretRedactingFilter as SecretRedactingFilter
from crypto_chatbot.schemas.commands import WatchlistAction
from crypto_chatbot.schemas.llm import AnalysisResponse
from crypto_chatbot.schemas.market import Timeframe

logger = logging.getLogger(__name__)


def _deny_chunks(reason: str | None) -> tuple[str, ...]:
    if reason == "rate_limited":
        return (
            "Permintaan Anda melebihi batas frekuensi sementara. "
            "Silakan tunggu sebentar sebelum mengirim perintah lagi.",
        )
    return ("Akses bot dibatasi untuk pengguna beta terdaftar.",)


class BotService:
    """Stateless-per-request orchestration backing Telegram commands and follow-ups."""

    def __init__(
        self,
        *,
        access: AccessController,
        analysis_service: AnalysisService,
        narrator: GeminiNarrator,
        memory: MemoryService,
    ) -> None:
        self._access = access
        self._analysis_service = analysis_service
        self._narrator = narrator
        self._memory = memory

    def _authorize(
        self, user_id: int | None, request_id: str, action: str
    ) -> tuple[str, ...] | None:
        decision = self._access.check_access(user_id)
        if not decision.allowed:
            logger.info(
                "request_id=%s user_id=%s action=%s access_denied=%s",
                request_id,
                user_id,
                action,
                decision.reason,
            )
            return _deny_chunks(decision.reason)
        return None

    def _safe_default_timeframe(self, user_id: int) -> Timeframe:
        prefs = self._memory.get_preferences(user_id)
        return prefs.default_timeframe

    async def handle_start(self, user_id: int | None) -> tuple[str, ...]:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        denied = self._authorize(user_id, request_id, "start")
        if denied is not None or user_id is None:
            return denied or _deny_chunks("not_allowlisted")
        try:
            self._memory.ensure_user(user_id)
        except SQLAlchemyError:
            logger.exception("request_id=%s user_id=%s action=start db_error", request_id, user_id)
            return ("Layanan database sedang bermasalah. Coba beberapa saat lagi.",)
        logger.info("request_id=%s user_id=%s action=start", request_id, user_id)
        return split_telegram_message(render_start_message(), TELEGRAM_MAX_MESSAGE_LENGTH)

    async def handle_help(self, user_id: int | None) -> tuple[str, ...]:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        denied = self._authorize(user_id, request_id, "help")
        if denied is not None or user_id is None:
            return denied or _deny_chunks("not_allowlisted")
        logger.info("request_id=%s user_id=%s action=help", request_id, user_id)
        return split_telegram_message(render_help_message(), TELEGRAM_MAX_MESSAGE_LENGTH)

    async def handle_analyze(
        self,
        user_id: int | None,
        args: Sequence[str],
    ) -> tuple[str, ...]:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        denied = self._authorize(user_id, request_id, "analyze")
        if denied is not None or user_id is None:
            return denied or _deny_chunks("not_allowlisted")

        try:
            default_tf = self._safe_default_timeframe(user_id)
            parsed = parse_analyze_args(args, default_timeframe=default_tf)
            user_text = "/analyze " + " ".join(args)
            return await self._execute_single_analysis(
                user_id=user_id,
                request_id=request_id,
                symbol=parsed.symbol,
                timeframe=parsed.timeframe,
                user_text=user_text,
                compact=not parsed.detail,
            )
        except CommandParseError as exc:
            return split_telegram_message(
                html.escape(str(exc), quote=False),
                TELEGRAM_MAX_MESSAGE_LENGTH,
            )
        except SQLAlchemyError:
            logger.exception(
                "request_id=%s user_id=%s action=analyze db_error", request_id, user_id
            )
            return ("Gagal mengakses database. Analisis tidak dapat diselesaikan saat ini.",)

    async def handle_compare(
        self,
        user_id: int | None,
        args: Sequence[str],
    ) -> tuple[str, ...]:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        denied = self._authorize(user_id, request_id, "compare")
        if denied is not None or user_id is None:
            return denied or _deny_chunks("not_allowlisted")

        try:
            default_tf = self._safe_default_timeframe(user_id)
            parsed = parse_compare_args(args, default_timeframe=default_tf)
            user_text = "/compare " + " ".join(args)
            return await self._execute_comparison(
                user_id=user_id,
                request_id=request_id,
                symbols=parsed.symbols,
                timeframe=parsed.timeframe,
                user_text=user_text,
            )
        except CommandParseError as exc:
            return split_telegram_message(
                html.escape(str(exc), quote=False),
                TELEGRAM_MAX_MESSAGE_LENGTH,
            )
        except SQLAlchemyError:
            logger.exception(
                "request_id=%s user_id=%s action=compare db_error", request_id, user_id
            )
            return ("Gagal mengakses database. Perbandingan tidak dapat diselesaikan saat ini.",)

    async def handle_watchlist(
        self,
        user_id: int | None,
        args: Sequence[str],
    ) -> tuple[str, ...]:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        denied = self._authorize(user_id, request_id, "watchlist")
        if denied is not None or user_id is None:
            return denied or _deny_chunks("not_allowlisted")

        try:
            parsed = parse_watchlist_args(args)
            note: str | None = None
            if parsed.action is WatchlistAction.LIST:
                prefs = self._memory.get_preferences(user_id)
            elif parsed.action is WatchlistAction.ADD and parsed.symbol is not None:
                prefs, added = self._memory.add_to_watchlist(user_id, parsed.symbol)
                note = (
                    f"{parsed.symbol} ditambahkan ke daftar pantau."
                    if added
                    else f"{parsed.symbol} sudah ada di daftar pantau."
                )
            elif parsed.symbol is not None:
                prefs, removed = self._memory.remove_from_watchlist(user_id, parsed.symbol)
                note = (
                    f"{parsed.symbol} dihapus dari daftar pantau."
                    if removed
                    else f"{parsed.symbol} tidak ditemukan di daftar pantau."
                )
            else:
                prefs = self._memory.get_preferences(user_id)
            rendered = render_watchlist_message(prefs.watchlist, note=note, html_mode=True)
            return split_telegram_message(rendered, TELEGRAM_MAX_MESSAGE_LENGTH)
        except CommandParseError as exc:
            return split_telegram_message(
                html.escape(str(exc), quote=False),
                TELEGRAM_MAX_MESSAGE_LENGTH,
            )
        except SQLAlchemyError:
            logger.exception(
                "request_id=%s user_id=%s action=watchlist db_error", request_id, user_id
            )
            return ("Gagal mengakses preferensi watchlist di database.",)

    async def handle_memory(self, user_id: int | None) -> tuple[str, ...]:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        denied = self._authorize(user_id, request_id, "memory")
        if denied is not None or user_id is None:
            return denied or _deny_chunks("not_allowlisted")

        try:
            prefs = self._memory.get_preferences(user_id)
            recent = self._memory.get_recent_messages(user_id)
            logger.info("request_id=%s user_id=%s action=memory", request_id, user_id)
            return split_telegram_message(
                render_memory_message(prefs, recent, html_mode=True),
                TELEGRAM_MAX_MESSAGE_LENGTH,
            )
        except SQLAlchemyError:
            logger.exception("request_id=%s user_id=%s action=memory db_error", request_id, user_id)
            return ("Gagal mengakses memori di database.",)

    async def handle_forget(self, user_id: int | None) -> tuple[str, ...]:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        denied = self._authorize(user_id, request_id, "forget")
        if denied is not None or user_id is None:
            return denied or _deny_chunks("not_allowlisted")

        try:
            result = self._memory.forget_user(user_id)
            logger.info("request_id=%s user_id=%s action=forget completed", request_id, user_id)
            return split_telegram_message(
                render_forget_message(result, html_mode=True),
                TELEGRAM_MAX_MESSAGE_LENGTH,
            )
        except SQLAlchemyError:
            logger.exception("request_id=%s user_id=%s action=forget db_error", request_id, user_id)
            return ("Gagal menghapus memori di database. Coba beberapa saat lagi.",)

    async def handle_unknown_command(
        self,
        user_id: int | None,
        command_text: str,
    ) -> tuple[str, ...]:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        denied = self._authorize(user_id, request_id, "unknown_command")
        if denied is not None or user_id is None:
            return denied or _deny_chunks("not_allowlisted")
        cmd = command_text.strip().split()[0] if command_text.strip() else "/"
        escaped_cmd = html.escape(cmd, quote=False)
        return (
            f"Perintah '{escaped_cmd}' tidak dikenali. "
            "Gunakan /help untuk melihat daftar perintah.",
        )

    async def handle_text_message(
        self,
        user_id: int | None,
        text: str,
    ) -> tuple[str, ...]:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        denied = self._authorize(user_id, request_id, "followup")
        if denied is not None or user_id is None:
            return denied or _deny_chunks("not_allowlisted")

        try:
            prefs = self._memory.get_preferences(user_id)
            recent = self._memory.get_recent_messages(user_id)
            intent = resolve_followup_intent(
                text,
                recent_messages=recent,
                default_timeframe=prefs.default_timeframe,
            )
            if isinstance(intent, RefusalIntent):
                return (html.escape(intent.message, quote=False),)
            if isinstance(intent, AnalyzeIntent):
                return await self._execute_single_analysis(
                    user_id=user_id,
                    request_id=request_id,
                    symbol=intent.symbol,
                    timeframe=intent.timeframe,
                    user_text=text,
                    compact=True,
                )
            if isinstance(intent, CompareIntent):
                return await self._execute_comparison(
                    user_id=user_id,
                    request_id=request_id,
                    symbols=intent.symbols,
                    timeframe=intent.timeframe,
                    user_text=text,
                )
            return ("Permintaan tidak dapat diproses. Gunakan /help.",)  # pragma: no cover
        except SQLAlchemyError:
            logger.exception(
                "request_id=%s user_id=%s action=followup db_error", request_id, user_id
            )
            return ("Layanan database sedang bermasalah. Coba beberapa saat lagi.",)

    async def _execute_single_analysis(
        self,
        *,
        user_id: int,
        request_id: str,
        symbol: str,
        timeframe: Timeframe,
        user_text: str,
        compact: bool = True,
    ) -> tuple[str, ...]:
        context = self._memory.build_context_for_llm(user_id, current_question=user_text)
        snapshot = await self._analysis_service.analyze(symbol, timeframe)
        outcome = await self._narrator.narrate_snapshot(
            snapshot,
            user_id=user_id,
            user_context=context,
        )
        self._memory.record_user_message(user_id, user_text)
        self._memory.record_assistant_turn(
            user_id,
            snapshot=snapshot,
            narration=outcome.response.narration,
        )
        self._memory.record_llm_outcome(user_id=user_id, request_id=request_id, outcome=outcome)
        logger.info(
            "request_id=%s user_id=%s action=analyze snapshot_id=%s quality=%s narration=%s",
            request_id,
            user_id,
            snapshot.snapshot_id,
            snapshot.quality_status.value,
            outcome.response.narration_status.value,
        )
        rendered = render_analysis_response(outcome.response, html_mode=True, compact=compact)
        return split_telegram_message(rendered, TELEGRAM_MAX_MESSAGE_LENGTH)

    async def _execute_comparison(
        self,
        *,
        user_id: int,
        request_id: str,
        symbols: Sequence[str],
        timeframe: Timeframe,
        user_text: str,
    ) -> tuple[str, ...]:
        context = self._memory.build_context_for_llm(user_id, current_question=user_text)
        responses: list[AnalysisResponse] = []
        snapshots = []
        narrations = []
        for symbol in symbols:
            snap = await self._analysis_service.analyze(symbol, timeframe)
            outcome = await self._narrator.narrate_snapshot(
                snap,
                user_id=user_id,
                user_context=context,
            )
            self._memory.record_llm_outcome(user_id=user_id, request_id=request_id, outcome=outcome)
            logger.info(
                "request_id=%s user_id=%s action=compare snapshot_id=%s quality=%s narration=%s",
                request_id,
                user_id,
                snap.snapshot_id,
                snap.quality_status.value,
                outcome.response.narration_status.value,
            )
            responses.append(outcome.response)
            snapshots.append(snap)
            narrations.append(outcome.response.narration)

        self._memory.record_user_message(user_id, user_text)
        self._memory.record_comparison_turn(
            user_id,
            snapshots=snapshots,
            narrations=narrations,
        )
        rendered = render_comparison_response(responses, html_mode=True, compact=True)
        return split_telegram_message(rendered, TELEGRAM_MAX_MESSAGE_LENGTH)


async def _reply_chunks(update: Update, chunks: Sequence[str]) -> None:
    message = update.effective_message
    if message is None:
        return
    for chunk in chunks:
        await message.reply_text(
            chunk,
            parse_mode=ParseMode.HTML,
            disable_web_page_preview=True,
        )


def register_handlers(
    application: Application[Any, Any, Any, Any, Any, Any],
    bot_service: BotService,
) -> None:
    """Bind all MVP Telegram commands and message handlers to a PTB `Application`."""

    async def start_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        uid = update.effective_user.id if update.effective_user else None
        await _reply_chunks(update, await bot_service.handle_start(uid))

    async def help_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        uid = update.effective_user.id if update.effective_user else None
        await _reply_chunks(update, await bot_service.handle_help(uid))

    async def analyze_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id if update.effective_user else None
        await _reply_chunks(update, await bot_service.handle_analyze(uid, context.args or []))

    async def compare_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id if update.effective_user else None
        await _reply_chunks(update, await bot_service.handle_compare(uid, context.args or []))

    async def watchlist_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id if update.effective_user else None
        await _reply_chunks(update, await bot_service.handle_watchlist(uid, context.args or []))

    async def memory_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        uid = update.effective_user.id if update.effective_user else None
        await _reply_chunks(update, await bot_service.handle_memory(uid))

    async def forget_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        uid = update.effective_user.id if update.effective_user else None
        await _reply_chunks(update, await bot_service.handle_forget(uid))

    async def unknown_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        uid = update.effective_user.id if update.effective_user else None
        text = (
            update.effective_message.text
            if update.effective_message and update.effective_message.text
            else "/"
        )
        await _reply_chunks(update, await bot_service.handle_unknown_command(uid, text))

    async def text_msg(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        del context
        uid = update.effective_user.id if update.effective_user else None
        text = (
            update.effective_message.text
            if update.effective_message and update.effective_message.text
            else ""
        )
        await _reply_chunks(update, await bot_service.handle_text_message(uid, text))

    async def global_error_handler(update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        logger.error(
            "Unhandled exception processing update: %s",
            context.error,
            exc_info=context.error,
        )
        if isinstance(update, Update) and update.effective_message:
            try:
                await update.effective_message.reply_text(
                    "⚠️ Terjadi kendala saat memproses permintaan Anda. "
                    "Silakan coba beberapa saat lagi."
                )
            except Exception as err:
                logger.error("Failed to deliver error response to user: %s", err)

    application.add_handler(CommandHandler("start", start_cmd))
    application.add_handler(CommandHandler("help", help_cmd))
    application.add_handler(CommandHandler("analyze", analyze_cmd))
    application.add_handler(CommandHandler("compare", compare_cmd))
    application.add_handler(CommandHandler("watchlist", watchlist_cmd))
    application.add_handler(CommandHandler("memory", memory_cmd))
    application.add_handler(CommandHandler("forget", forget_cmd))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_cmd))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_msg))
    application.add_error_handler(global_error_handler)
