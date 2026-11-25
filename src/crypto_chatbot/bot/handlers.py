"""Telegram command/message service, PTB handlers, and secret-redacting log filter."""

from __future__ import annotations

import html
import logging
import uuid
from collections.abc import Sequence
from typing import Any, Literal

from sqlalchemy.exc import SQLAlchemyError
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from crypto_chatbot.analysis.momentum_screener import (
    DEFAULT_MIN_VOLUME_24H_USD,
    render_momentum_screener_report,
    screen_momentum_movers,
)
from crypto_chatbot.analysis.pine_export import generate_pine_script
from crypto_chatbot.analysis.scanner import (
    DEFAULT_SCAN_BASKET,
    render_market_scan_report,
    scan_market_setups,
)
from crypto_chatbot.analysis.service import AnalysisService
from crypto_chatbot.bot.access import AccessController
from crypto_chatbot.bot.commands import (
    AnalyzeIntent,
    ChatQaIntent,
    CommandParseError,
    CompareIntent,
    PineIntent,
    RefusalIntent,
    ScanIntent,
    ScreenIntent,
    _extract_recent_context,
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
from crypto_chatbot.market_data.symbols import resolve_default_perp_symbol
from crypto_chatbot.memory.service import MemoryService
from crypto_chatbot.ops.logging import SecretRedactingFilter as SecretRedactingFilter
from crypto_chatbot.schemas.commands import WatchlistAction
from crypto_chatbot.schemas.llm import AnalysisResponse
from crypto_chatbot.schemas.market import (
    MULTI_STYLE_SCAN_TIMEFRAMES,
    QualityStatus,
    Timeframe,
    format_trading_style_short,
    resolve_primary_timeframe,
)

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

    async def handle_pine(
        self,
        user_id: int | None,
        args: Sequence[str],
    ) -> tuple[str, ...]:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        denied = self._authorize(user_id, request_id, "pine")
        if denied is not None or user_id is None:
            return denied or _deny_chunks("not_allowlisted")

        try:
            default_tf = self._safe_default_timeframe(user_id)
            parsed = parse_analyze_args(args, default_timeframe=default_tf)
            return await self._execute_pine(
                symbol=parsed.symbol,
                timeframe=parsed.timeframe,
            )
        except CommandParseError as exc:
            return split_telegram_message(
                html.escape(str(exc), quote=False),
                TELEGRAM_MAX_MESSAGE_LENGTH,
            )
        except SQLAlchemyError:
            logger.exception("request_id=%s user_id=%s action=pine db_error", request_id, user_id)
            return ("Gagal mengakses database. Script Pine tidak dapat dibuat saat ini.",)

    async def handle_scan(
        self,
        user_id: int | None,
        args: Sequence[str],
    ) -> tuple[str, ...]:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        denied = self._authorize(user_id, request_id, "scan")
        if denied is not None or user_id is None:
            return denied or _deny_chunks("not_allowlisted")

        try:
            prefs = self._memory.get_preferences(user_id)
            timeframe = prefs.default_timeframe
            explicit_tf_provided = False
            explicit_symbols: list[str] = []
            for raw_arg in args:
                cleaned_arg = raw_arg.strip()
                if not cleaned_arg:
                    continue
                tf_cand = resolve_primary_timeframe(cleaned_arg)
                if tf_cand is not None:
                    timeframe = tf_cand
                    explicit_tf_provided = True
                    continue
                try:
                    explicit_symbols.append(
                        resolve_default_perp_symbol(cleaned_arg).canonical_symbol
                    )
                except ValueError:
                    esc_arg = html.escape(cleaned_arg, quote=False)
                    return (
                        f"Argumen '{esc_arg}' tidak valid untuk /scan. "
                        "Gunakan: /scan [scalp|intraday|swing|15m|1h|4h|1d] atau /scan BTC ETH SOL",
                    )

            return await self._execute_scan(
                user_id=user_id,
                request_id=request_id,
                timeframe=timeframe,
                timeframes=None if explicit_tf_provided else MULTI_STYLE_SCAN_TIMEFRAMES,
                explicit_symbols=explicit_symbols or None,
                watchlist=prefs.watchlist,
                user_text="/scan " + " ".join(args),
            )
        except SQLAlchemyError:
            logger.exception("request_id=%s user_id=%s action=scan db_error", request_id, user_id)
            return ("Gagal mengakses database. Pemindaian tidak dapat diselesaikan saat ini.",)

    async def handle_screen(
        self,
        user_id: int | None,
        args: Sequence[str],
    ) -> tuple[str, ...]:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        denied = self._authorize(user_id, request_id, "screen")
        if denied is not None or user_id is None:
            return denied or _deny_chunks("not_allowlisted")

        side: Literal["all", "long", "short"] = "all"
        min_vol_usd = DEFAULT_MIN_VOLUME_24H_USD

        for raw_arg in args:
            tok = raw_arg.strip().lower()
            if not tok:
                continue
            if tok in ("long", "buy", "gainer", "gainers"):
                side = "long"
            elif tok in ("short", "sell", "loser", "losers"):
                side = "short"
            elif tok == "all":
                side = "all"
            else:
                try:
                    val = float(tok)
                except ValueError:
                    esc = html.escape(raw_arg, quote=False)
                    return (
                        f"Argumen '{esc}' tidak valid untuk /screen. "
                        "Gunakan: /screen [long|short|all] [min_volume_juta_usd] "
                        "(contoh: /screen long 10)",
                    )
                if val <= 0:
                    return ("Minimum volume harus lebih besar dari 0.",)
                min_vol_usd = val * 1_000_000.0 if val < 10_000.0 else val

        try:
            return await self._execute_screen(
                user_id=user_id,
                request_id=request_id,
                side=side,
                min_volume_usd=min_vol_usd,
                user_text="/screen " + " ".join(args),
            )
        except ValueError as exc:
            return (html.escape(str(exc), quote=False),)
        except SQLAlchemyError:
            logger.exception("request_id=%s user_id=%s action=screen db_error", request_id, user_id)
            return ("Gagal mengakses database. Screening tidak dapat diselesaikan saat ini.",)

    async def _execute_screen(
        self,
        *,
        user_id: int,
        request_id: str,
        side: Literal["all", "long", "short"],
        min_volume_usd: float,
        user_text: str,
    ) -> tuple[str, ...]:
        report = await screen_momentum_movers(
            self._analysis_service,
            side=side,
            min_volume_usd=min_volume_usd,
        )
        rendered = render_momentum_screener_report(report, html_mode=True)
        self._memory.record_user_message(user_id, user_text)
        ready_syms = (
            ", ".join(s.symbol for s in (*report.long_signals, *report.short_signals))
            or "tidak ada"
        )
        self._memory.record_assistant_text(
            user_id,
            f"[SCREEN 15m] Sisi={side}; READY: {ready_syms}.",
        )
        logger.info(
            "request_id=%s user_id=%s action=screen side=%s long=%d short=%d",
            request_id,
            user_id,
            side,
            len(report.long_signals),
            len(report.short_signals),
        )
        return split_telegram_message(rendered, TELEGRAM_MAX_MESSAGE_LENGTH)

    async def _execute_pine(
        self,
        *,
        symbol: str,
        timeframe: Timeframe,
    ) -> tuple[str, ...]:
        snapshot = await self._analysis_service.analyze(symbol, timeframe)
        if snapshot.quality_status is QualityStatus.INSUFFICIENT_DATA:
            return (
                f"Data tidak mencukupi (INSUFFICIENT_DATA) untuk membuat Pine Script "
                f"{html.escape(snapshot.canonical_symbol)} ({snapshot.timeframe.value}).",
            )

        pine_code = generate_pine_script(snapshot)
        header = (
            f"<b>TradingView Pine Script v6 — {html.escape(snapshot.canonical_symbol)} "
            f"({snapshot.timeframe.value})</b>\n"
            "Salin kode berikut ke TradingView Pine Editor → Add to chart:"
        )
        msg = f"{header}\n<pre>{html.escape(pine_code, quote=False)}</pre>"
        if len(msg) <= TELEGRAM_MAX_MESSAGE_LENGTH:
            return (msg,)

        code_chunks = split_telegram_message(pine_code, TELEGRAM_MAX_MESSAGE_LENGTH - 64)
        chunks = [f"{header}\n<pre>{html.escape(code_chunks[0], quote=False)}</pre>"]
        for extra in code_chunks[1:]:
            chunks.append(f"<pre>{html.escape(extra, quote=False)}</pre>")
        return tuple(chunks)

    async def _execute_scan(
        self,
        *,
        user_id: int,
        request_id: str,
        timeframe: Timeframe,
        timeframes: Sequence[Timeframe] | None = None,
        explicit_symbols: Sequence[str] | None,
        watchlist: Sequence[str],
        user_text: str,
    ) -> tuple[str, ...]:
        if explicit_symbols:
            targets = list(dict.fromkeys(explicit_symbols))
        else:
            targets = list(dict.fromkeys((*watchlist, *DEFAULT_SCAN_BASKET)))[:10]

        report = await scan_market_setups(
            self._analysis_service,
            targets,
            timeframe=timeframe,
            timeframes=timeframes,
        )
        rendered = render_market_scan_report(report, html_mode=True)
        self._memory.record_user_message(user_id, user_text)
        ready_syms = (
            ", ".join(
                f"{item.symbol} [{format_trading_style_short(item.timeframe)}]"
                for item in report.ready_setups
            )
            or "tidak ada"
        )
        self._memory.record_assistant_text(
            user_id,
            f"[SCAN {timeframe.value}] Dipindai {report.scanned_count} koin; "
            f"READY ({len(report.ready_setups)}): {ready_syms}.",
        )
        logger.info(
            "request_id=%s user_id=%s action=scan tf=%s scanned=%d ready=%d",
            request_id,
            user_id,
            timeframe.value,
            report.scanned_count,
            len(report.ready_setups),
        )
        return split_telegram_message(rendered, TELEGRAM_MAX_MESSAGE_LENGTH)

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
            if isinstance(intent, RefusalIntent) and not intent.is_hard_refusal:
                intent = await self._resolve_via_llm_fallback(
                    text=text,
                    user_id=user_id,
                    recent_messages=recent,
                    default_timeframe=prefs.default_timeframe,
                    fallback_refusal=intent,
                )

            if isinstance(intent, RefusalIntent):
                return (html.escape(intent.message, quote=False),)
            if isinstance(intent, ScreenIntent):
                return await self._execute_screen(
                    user_id=user_id,
                    request_id=request_id,
                    side=intent.side,
                    min_volume_usd=intent.min_volume_usd,
                    user_text=text,
                )
            if isinstance(intent, ScanIntent):
                return await self._execute_scan(
                    user_id=user_id,
                    request_id=request_id,
                    timeframe=intent.timeframe,
                    timeframes=intent.timeframes,
                    explicit_symbols=intent.symbols,
                    watchlist=prefs.watchlist,
                    user_text=text,
                )
            if isinstance(intent, PineIntent):
                return await self._execute_pine(
                    symbol=intent.symbol,
                    timeframe=intent.timeframe,
                )
            if isinstance(intent, AnalyzeIntent):
                return await self._execute_single_analysis(
                    user_id=user_id,
                    request_id=request_id,
                    symbol=intent.symbol,
                    timeframe=intent.timeframe,
                    user_text=text,
                    compact=not intent.detail,
                )
            if isinstance(intent, CompareIntent):
                return await self._execute_comparison(
                    user_id=user_id,
                    request_id=request_id,
                    symbols=intent.symbols,
                    timeframe=intent.timeframe,
                    user_text=text,
                )
            if isinstance(intent, ChatQaIntent):
                return await self._execute_chat_qa(
                    user_id=user_id,
                    request_id=request_id,
                    question=intent.question,
                )
            return ("Permintaan tidak dapat diproses. Gunakan /help.",)  # pragma: no cover
        except SQLAlchemyError:
            logger.exception(
                "request_id=%s user_id=%s action=followup db_error", request_id, user_id
            )
            return ("Layanan database sedang bermasalah. Coba beberapa saat lagi.",)

    async def _resolve_via_llm_fallback(
        self,
        *,
        text: str,
        user_id: int,
        recent_messages: Sequence[Any],
        default_timeframe: Timeframe,
        fallback_refusal: RefusalIntent,
    ) -> (
        AnalyzeIntent
        | CompareIntent
        | PineIntent
        | ScanIntent
        | ScreenIntent
        | ChatQaIntent
        | RefusalIntent
    ):
        recent_syms, recent_tf = _extract_recent_context(recent_messages, default_timeframe)
        classified = await self._narrator.classify_intent(
            text,
            user_id=user_id,
            recent_symbols=recent_syms,
        )
        if classified is None or classified.intent == "unsupported":
            return fallback_refusal

        if classified.intent == "screen":
            return ScreenIntent(side=classified.screen_side or "all")

        resolved_cls_tf = (
            resolve_primary_timeframe(classified.timeframe)
            if classified.timeframe is not None
            else None
        )
        target_tf = resolved_cls_tf if resolved_cls_tf is not None else default_timeframe
        resolved_syms: list[str] = []
        for raw_sym in classified.symbols:
            try:
                canon = resolve_default_perp_symbol(raw_sym).canonical_symbol
            except ValueError:
                continue
            if canon not in resolved_syms:
                resolved_syms.append(canon)

        if classified.intent == "scan":
            return ScanIntent(
                timeframe=target_tf,
                timeframes=(None if resolved_cls_tf is not None else MULTI_STYLE_SCAN_TIMEFRAMES),
                symbols=tuple(resolved_syms) if resolved_syms else None,
            )
        if classified.intent == "pine":
            target_sym = (
                resolved_syms[0] if resolved_syms else (recent_syms[0] if recent_syms else None)
            )
            tf = target_tf if classified.timeframe is not None or resolved_syms else recent_tf
            if target_sym is not None:
                return PineIntent(symbol=target_sym, timeframe=tf)
            return fallback_refusal
        if classified.intent == "compare":
            if len(resolved_syms) >= 2:
                return CompareIntent(symbols=tuple(resolved_syms[:3]), timeframe=target_tf)
            if len(resolved_syms) == 1 and recent_syms:
                prior = next((s for s in recent_syms if s != resolved_syms[0]), None)
                if prior is not None:
                    return CompareIntent(symbols=(prior, resolved_syms[0]), timeframe=target_tf)
            return fallback_refusal
        if classified.intent in ("analyze", "analyze_detail"):
            is_detail = classified.intent == "analyze_detail"
            if resolved_syms:
                return AnalyzeIntent(symbol=resolved_syms[0], timeframe=target_tf, detail=is_detail)
            if recent_syms:
                tf = target_tf if classified.timeframe is not None else recent_tf
                return AnalyzeIntent(symbol=recent_syms[0], timeframe=tf, detail=is_detail)
            return fallback_refusal
        if classified.intent == "chat_qa":
            return ChatQaIntent(question=text.strip())
        return fallback_refusal

    async def _execute_chat_qa(
        self,
        *,
        user_id: int,
        request_id: str,
        question: str,
    ) -> tuple[str, ...]:
        context = self._memory.build_context_for_llm(user_id, current_question=question)
        answer = await self._narrator.answer_research_question(
            question,
            user_id=user_id,
            user_context=context,
        )
        if answer is None:
            return (
                "Maaf, layanan diskusi AI sedang mencapai batas kuota atau sibuk saat ini. "
                "Anda tetap bisa menyebut nama koin (mis. 'analisa BTC') atau /scan.",
            )
        self._memory.record_user_message(user_id, question)
        self._memory.record_assistant_text(user_id, answer)
        logger.info("request_id=%s user_id=%s action=chat_qa", request_id, user_id)
        return split_telegram_message(
            html.escape(answer, quote=False),
            TELEGRAM_MAX_MESSAGE_LENGTH,
        )

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

    async def pine_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id if update.effective_user else None
        await _reply_chunks(update, await bot_service.handle_pine(uid, context.args or []))

    async def scan_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id if update.effective_user else None
        await _reply_chunks(update, await bot_service.handle_scan(uid, context.args or []))

    async def screen_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id if update.effective_user else None
        await _reply_chunks(update, await bot_service.handle_screen(uid, context.args or []))

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
    application.add_handler(CommandHandler("pine", pine_cmd))
    application.add_handler(CommandHandler("scan", scan_cmd))
    application.add_handler(CommandHandler("screen", screen_cmd))
    application.add_handler(CommandHandler("watchlist", watchlist_cmd))
    application.add_handler(CommandHandler("memory", memory_cmd))
    application.add_handler(CommandHandler("forget", forget_cmd))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_cmd))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_msg))
    application.add_error_handler(global_error_handler)
