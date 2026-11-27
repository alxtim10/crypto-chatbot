"""Telegram command/message service, PTB handlers, and secret-redacting log filter."""

from __future__ import annotations

import asyncio
import html
import logging
import math
import time
import uuid
from collections.abc import Callable, Collection, Sequence
from typing import Any, Literal

from sqlalchemy.exc import SQLAlchemyError
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from crypto_chatbot.analysis.forward_proof import ForwardProofEvaluation
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
from crypto_chatbot.analysis.whale_tracker import (
    WhaleWatchlistManager,
    aggregate_whale_positions,
    format_whale_overview_telegram,
    format_whale_wallet_telegram,
    format_whale_watchlist_telegram,
)
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
    WhaleIntent,
    _extract_recent_context,
    parse_analyze_args,
    parse_compare_args,
    parse_watchlist_args,
    parse_whale_args,
    resolve_followup_intent_with_trace,
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
from crypto_chatbot.llm.gemini import GeminiNarrator, IntentClassificationOutcome
from crypto_chatbot.llm.guard import validate_conversational_answer
from crypto_chatbot.llm.render import render_analysis_response, render_comparison_response
from crypto_chatbot.market_data.symbols import (
    DEFAULT_PERP_UNIVERSE,
    resolve_default_perp_symbol,
)
from crypto_chatbot.market_data.whale_provider import WhaleDataProvider
from crypto_chatbot.memory.service import MemoryService
from crypto_chatbot.ops.logging import SecretRedactingFilter as SecretRedactingFilter
from crypto_chatbot.schemas.commands import WatchlistAction, WhaleAction
from crypto_chatbot.schemas.llm import AnalysisResponse, RoutingTrace
from crypto_chatbot.schemas.market import (
    MULTI_STYLE_SCAN_TIMEFRAMES,
    QualityStatus,
    Timeframe,
    format_trading_style_short,
    resolve_primary_timeframe,
)
from crypto_chatbot.schemas.whale import NetworkType

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
        whale_provider: WhaleDataProvider | None = None,
        whale_watchlist: WhaleWatchlistManager | None = None,
        forward_track_provider: Callable[[], Sequence[ForwardProofEvaluation]] | None = None,
    ) -> None:
        self._access = access
        self._analysis_service = analysis_service
        self._narrator = narrator
        self._memory = memory
        self._whale_provider = whale_provider or WhaleDataProvider()
        self._whale_watchlist = whale_watchlist or WhaleWatchlistManager()
        self._forward_track_provider = forward_track_provider
        self._perp_universe_cache: tuple[float, frozenset[str]] | None = None

    _PERP_UNIVERSE_CACHE_TTL_SECONDS = 600.0

    async def _get_perp_universe(self) -> frozenset[str]:
        now_mono = time.monotonic()
        cached = self._perp_universe_cache
        if cached is not None and (now_mono - cached[0]) < self._PERP_UNIVERSE_CACHE_TTL_SECONDS:
            return cached[1]
        get_tickers = getattr(self._analysis_service, "get_futures_tickers", None)
        if callable(get_tickers):
            try:
                batch = await get_tickers()
                if not batch.has_provider_error and batch.tickers:
                    bases: set[str] = set()
                    for ticker in batch.tickers:
                        base = ticker.canonical_symbol.split("/")[0].upper()
                        bases.add(base)
                        if base.startswith("1000") and len(base) > 4:
                            bases.add(base[4:])
                    frozen = frozenset(bases)
                    self._perp_universe_cache = (now_mono, frozen)
                    return frozen
            except Exception as exc:
                logger.debug("Failed to refresh perp universe cache: %s", exc)
        return DEFAULT_PERP_UNIVERSE

    async def handle_track(
        self,
        user_id: int | None,
        args: Sequence[str] = (),
    ) -> tuple[str, ...]:
        """Render forward-ledger evaluation summary per timeframe (PRD v1.7 §2.1, T-32)."""
        del args
        request_id = uuid.uuid4().hex[:12]
        denied = self._authorize(user_id, request_id, "track")
        if denied is not None:
            return denied
        if self._forward_track_provider is None:
            return (
                "<b>Forward Evaluation Ledger (/track)</b>\n"
                "Belum ada data forward ledger pasca-freeze yang terhubung. "
                "Jalankan <code>crypto-chatbot freeze-params v3</code> lalu "
                "<code>settle-setups</code> secara berkala untuk mengumpulkan evaluasi forward.",
            )
        evals = self._forward_track_provider()
        if not evals:
            return (
                "<b>Forward Evaluation Ledger (/track)</b>\n"
                "Belum ada setup settled pasca-freeze yang tercatat.",
            )
        lines: list[str] = ["<b>Forward Evaluation Ledger (/track)</b>"]
        for ev in evals:
            exp_str = f"{ev.expectancy_r_1x:+.4f}R" if ev.expectancy_r_1x is not None else "N/A"
            ci_str = (
                f"[{ev.block_ci_low:+.4f}R, {ev.block_ci_high:+.4f}R]"
                if ev.block_ci_low is not None and ev.block_ci_high is not None
                else "N/A"
            )
            lines.append(
                f"• <b>{ev.timeframe.value}</b> | status={ev.status} | "
                f"settled={ev.eligible_settled_filled} ({ev.duration_days:.1f}d) | "
                f"E[R]={exp_str} | 95% Block CI={ci_str}"
            )
        return split_telegram_message("\n".join(lines), max_length=TELEGRAM_MAX_MESSAGE_LENGTH)

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

    async def _safe_default_timeframe(self, user_id: int) -> Timeframe:
        prefs = await asyncio.to_thread(self._memory.get_preferences, user_id)
        return prefs.default_timeframe

    async def handle_start(self, user_id: int | None) -> tuple[str, ...]:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        denied = self._authorize(user_id, request_id, "start")
        if denied is not None or user_id is None:
            return denied or _deny_chunks("not_allowlisted")
        try:
            await asyncio.to_thread(self._memory.ensure_user, user_id)
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
            default_tf = await self._safe_default_timeframe(user_id)
            parsed = parse_analyze_args(args, default_timeframe=default_tf)
            user_text = "/analyze " + " ".join(args)
            return await self._execute_single_analysis(
                user_id=user_id,
                request_id=request_id,
                symbol=parsed.symbol,
                timeframe=parsed.timeframe,
                user_text=user_text,
                compact=not parsed.detail,
                routing_trace=RoutingTrace(
                    layer="command",
                    detected_symbol=parsed.symbol,
                    intent="analyze_detail" if parsed.detail else "analyze",
                    confidence=1.0,
                ),
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
            default_tf = await self._safe_default_timeframe(user_id)
            parsed = parse_compare_args(args, default_timeframe=default_tf)
            user_text = "/compare " + " ".join(args)
            return await self._execute_comparison(
                user_id=user_id,
                request_id=request_id,
                symbols=parsed.symbols,
                timeframe=parsed.timeframe,
                user_text=user_text,
                routing_trace=RoutingTrace(
                    layer="command",
                    detected_symbol=",".join(parsed.symbols),
                    intent="compare",
                    confidence=1.0,
                ),
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
            default_tf = await self._safe_default_timeframe(user_id)
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
            prefs = await asyncio.to_thread(self._memory.get_preferences, user_id)
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
                if not math.isfinite(val) or val <= 0:
                    return ("Minimum volume harus berupa angka valid lebih besar dari 0.",)
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
        await asyncio.to_thread(self._memory.record_user_message, user_id, user_text)
        ready_syms = (
            ", ".join(s.symbol for s in (*report.long_signals, *report.short_signals))
            or "tidak ada"
        )
        await asyncio.to_thread(
            self._memory.record_assistant_text,
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
        user_id: int | None = None,
        request_id: str | None = None,
        routing_trace: RoutingTrace | None = None,
    ) -> tuple[str, ...]:
        snapshot = await self._analysis_service.analyze(symbol, timeframe)
        if snapshot.quality_status is QualityStatus.INSUFFICIENT_DATA:
            if user_id is not None and request_id is not None and routing_trace is not None:
                enriched = routing_trace.model_copy(
                    update={"fallback_reason": "snapshot_insufficient_data"}
                )
                await asyncio.to_thread(
                    self._memory.record_routing_trace,
                    user_id=user_id,
                    request_id=request_id,
                    trace=enriched,
                    validation_passed=False,
                    rejection_reasons=("snapshot_insufficient_data",),
                    snapshot_id=snapshot.snapshot_id,
                )
            return (
                f"Data tidak mencukupi (INSUFFICIENT_DATA) untuk membuat Pine Script "
                f"{html.escape(snapshot.canonical_symbol)} ({snapshot.timeframe.value}).",
            )
        if user_id is not None and request_id is not None and routing_trace is not None:
            await asyncio.to_thread(
                self._memory.record_routing_trace,
                user_id=user_id,
                request_id=request_id,
                trace=routing_trace,
                snapshot_id=snapshot.snapshot_id,
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
        await asyncio.to_thread(self._memory.record_user_message, user_id, user_text)
        ready_syms = (
            ", ".join(
                f"{item.symbol} [{format_trading_style_short(item.timeframe)}]"
                for item in report.ready_setups
            )
            or "tidak ada"
        )
        await asyncio.to_thread(
            self._memory.record_assistant_text,
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
                prefs = await asyncio.to_thread(self._memory.get_preferences, user_id)
            elif parsed.action is WatchlistAction.ADD and parsed.symbol is not None:
                prefs, added = await asyncio.to_thread(
                    self._memory.add_to_watchlist, user_id, parsed.symbol
                )
                note = (
                    f"{parsed.symbol} ditambahkan ke daftar pantau."
                    if added
                    else f"{parsed.symbol} sudah ada di daftar pantau."
                )
            elif parsed.symbol is not None:
                prefs, removed = await asyncio.to_thread(
                    self._memory.remove_from_watchlist, user_id, parsed.symbol
                )
                note = (
                    f"{parsed.symbol} dihapus dari daftar pantau."
                    if removed
                    else f"{parsed.symbol} tidak ditemukan di daftar pantau."
                )
            else:
                prefs = await asyncio.to_thread(self._memory.get_preferences, user_id)
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

    async def handle_whale(
        self,
        user_id: int | None,
        args: Sequence[str],
    ) -> tuple[str, ...]:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        denied = self._authorize(user_id, request_id, "whale")
        if denied is not None or user_id is None:
            return denied or _deny_chunks("not_allowlisted")

        try:
            parsed = parse_whale_args(args)
            intent = WhaleIntent(
                action=parsed.action,
                symbol=parsed.symbol,
                address=parsed.address,
                label=parsed.label,
            )
            return await self._execute_whale(
                user_id=user_id,
                request_id=request_id,
                intent=intent,
            )
        except CommandParseError as exc:
            return split_telegram_message(
                html.escape(str(exc), quote=False),
                TELEGRAM_MAX_MESSAGE_LENGTH,
            )

    async def _execute_whale(
        self,
        *,
        user_id: int | None,
        request_id: str,
        intent: WhaleIntent,
    ) -> tuple[str, ...]:
        if intent.action == WhaleAction.LIST:
            entries = await asyncio.to_thread(self._whale_watchlist.list_entries, user_id)
            rendered = format_whale_watchlist_telegram(entries, html_mode=True)
            return split_telegram_message(rendered, TELEGRAM_MAX_MESSAGE_LENGTH)

        if intent.action == WhaleAction.ADD:
            if not intent.address:
                return ("Format salah: alamat dompet diperlukan.",)
            try:
                entry = await asyncio.to_thread(
                    self._whale_watchlist.add_entry,
                    intent.address,
                    intent.label or "Whale",
                    user_id=user_id,
                )
                safe_addr = html.escape(entry.address, quote=False)
                safe_label = html.escape(entry.label, quote=False)
                msg = (
                    f"✅ Alamat <code>{safe_addr}</code> berhasil ditambahkan ke watchlist whale "
                    f"dengan label <b>{safe_label}</b>."
                )
                return (msg,)
            except ValueError as exc:
                return (f"❌ {html.escape(str(exc), quote=False)}",)

        if intent.action == WhaleAction.REMOVE:
            if not intent.address:
                return ("Format salah: alamat dompet diperlukan.",)
            removed = await asyncio.to_thread(
                self._whale_watchlist.remove_entry, intent.address, user_id=user_id
            )
            safe_addr = html.escape(intent.address.strip(), quote=False)
            if removed:
                return (
                    f"✅ Alamat <code>{safe_addr}</code> berhasil dihapus dari watchlist whale.",
                )
            return (f"[Info] Alamat <code>{safe_addr}</code> tidak ditemukan di watchlist.",)

        if intent.action == WhaleAction.WALLET:
            if not intent.address:
                return ("Alamat wallet diperlukan.",)
            try:
                profile = await self._whale_provider.get_wallet_profile(intent.address)
            except Exception as exc:
                logger.warning(
                    "request_id=%s user_id=%s action=whale_wallet provider_error=%s",
                    request_id,
                    user_id,
                    exc,
                )
                return (
                    "Maaf, data dompet whale dari provider eksternal sedang tidak dapat diakses "
                    "saat ini. Coba lagi beberapa saat.",
                )
            rendered = format_whale_wallet_telegram(profile, html_mode=True)
            return split_telegram_message(rendered, TELEGRAM_MAX_MESSAGE_LENGTH)

        symbol = intent.symbol or "BTC"
        clean_sym = symbol.upper()
        if "/" in clean_sym:
            clean_sym = clean_sym.split("/")[0]
        elif clean_sym.endswith("USDT") and len(clean_sym) > 4:
            clean_sym = clean_sym[:-4]

        tracked_entries = await asyncio.to_thread(self._whale_watchlist.list_entries, user_id)
        targets = [(e.address, e.label) for e in tracked_entries if e.network == NetworkType.EVM]
        try:
            positions = await self._whale_provider.get_positions_for_symbol(
                clean_sym, wallets=targets
            )
        except Exception as exc:
            logger.warning(
                "request_id=%s user_id=%s action=whale_overview provider_error=%s",
                request_id,
                user_id,
                exc,
            )
            return (
                "Maaf, data posisi whale dari provider eksternal sedang tidak dapat diakses "
                "saat ini. Coba lagi beberapa saat.",
            )
        overview = aggregate_whale_positions(clean_sym, positions)
        rendered = format_whale_overview_telegram(overview, html_mode=True)
        return split_telegram_message(rendered, TELEGRAM_MAX_MESSAGE_LENGTH)

    async def handle_memory(self, user_id: int | None) -> tuple[str, ...]:
        request_id = f"req-{uuid.uuid4().hex[:12]}"
        denied = self._authorize(user_id, request_id, "memory")
        if denied is not None or user_id is None:
            return denied or _deny_chunks("not_allowlisted")

        try:
            prefs = await asyncio.to_thread(self._memory.get_preferences, user_id)
            recent = await asyncio.to_thread(self._memory.get_recent_messages, user_id)
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
            result = await asyncio.to_thread(self._memory.forget_user, user_id)
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

    @staticmethod
    def _log_routing_trace(request_id: str, trace: RoutingTrace) -> None:
        logger.info(
            "routing_trace request_id=%s layer=%s intent=%s detected_symbol=%s "
            "confidence=%s fallback_reason=%s guard_violations=%s",
            request_id,
            trace.layer,
            trace.intent,
            trace.detected_symbol,
            trace.confidence,
            trace.fallback_reason,
            ",".join(trace.guard_violations) if trace.guard_violations else "none",
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
            prefs = await asyncio.to_thread(self._memory.get_preferences, user_id)
            recent = await asyncio.to_thread(self._memory.get_recent_messages, user_id)
            perp_universe = await self._get_perp_universe()
            intent, trace = resolve_followup_intent_with_trace(
                text,
                recent_messages=recent,
                default_timeframe=prefs.default_timeframe,
                perp_universe=perp_universe,
            )
            cls_outcome: IntentClassificationOutcome | None = None
            if isinstance(intent, RefusalIntent) and not intent.is_hard_refusal:
                intent, trace, cls_outcome = await self._resolve_via_llm_fallback_with_trace(
                    text=text,
                    user_id=user_id,
                    recent_messages=recent,
                    default_timeframe=prefs.default_timeframe,
                    fallback_refusal=intent,
                    perp_universe=perp_universe,
                )

            if isinstance(intent, RefusalIntent):
                self._log_routing_trace(request_id, trace)
                await asyncio.to_thread(
                    self._memory.record_routing_trace,
                    user_id=user_id,
                    request_id=request_id,
                    trace=trace,
                    model_id=cls_outcome.model_id if cls_outcome is not None else "deterministic",
                    validation_passed=False,
                    rejection_reasons=(trace.fallback_reason,) if trace.fallback_reason else (),
                    prompt_tokens=cls_outcome.prompt_tokens if cls_outcome is not None else None,
                    output_tokens=cls_outcome.output_tokens if cls_outcome is not None else None,
                    latency_ms=cls_outcome.latency_ms if cls_outcome is not None else 0,
                )
                return (html.escape(intent.message, quote=False),)
            if isinstance(intent, WhaleIntent):
                self._log_routing_trace(request_id, trace)
                await asyncio.to_thread(
                    self._memory.record_routing_trace,
                    user_id=user_id,
                    request_id=request_id,
                    trace=trace,
                    model_id=cls_outcome.model_id if cls_outcome is not None else "deterministic",
                    prompt_tokens=cls_outcome.prompt_tokens if cls_outcome is not None else None,
                    output_tokens=cls_outcome.output_tokens if cls_outcome is not None else None,
                    latency_ms=cls_outcome.latency_ms if cls_outcome is not None else 0,
                )
                return await self._execute_whale(
                    user_id=user_id,
                    request_id=request_id,
                    intent=intent,
                )
            if isinstance(intent, ScreenIntent):
                self._log_routing_trace(request_id, trace)
                await asyncio.to_thread(
                    self._memory.record_routing_trace,
                    user_id=user_id,
                    request_id=request_id,
                    trace=trace,
                    model_id=cls_outcome.model_id if cls_outcome is not None else "deterministic",
                    prompt_tokens=cls_outcome.prompt_tokens if cls_outcome is not None else None,
                    output_tokens=cls_outcome.output_tokens if cls_outcome is not None else None,
                    latency_ms=cls_outcome.latency_ms if cls_outcome is not None else 0,
                )
                try:
                    return await self._execute_screen(
                        user_id=user_id,
                        request_id=request_id,
                        side=intent.side,
                        min_volume_usd=intent.min_volume_usd,
                        user_text=text,
                    )
                except ValueError as exc:
                    return (html.escape(str(exc), quote=False),)
            if isinstance(intent, ScanIntent):
                self._log_routing_trace(request_id, trace)
                await asyncio.to_thread(
                    self._memory.record_routing_trace,
                    user_id=user_id,
                    request_id=request_id,
                    trace=trace,
                    model_id=cls_outcome.model_id if cls_outcome is not None else "deterministic",
                    prompt_tokens=cls_outcome.prompt_tokens if cls_outcome is not None else None,
                    output_tokens=cls_outcome.output_tokens if cls_outcome is not None else None,
                    latency_ms=cls_outcome.latency_ms if cls_outcome is not None else 0,
                )
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
                self._log_routing_trace(request_id, trace)
                return await self._execute_pine(
                    symbol=intent.symbol,
                    timeframe=intent.timeframe,
                    user_id=user_id,
                    request_id=request_id,
                    routing_trace=trace,
                )
            if isinstance(intent, AnalyzeIntent):
                return await self._execute_single_analysis(
                    user_id=user_id,
                    request_id=request_id,
                    symbol=intent.symbol,
                    timeframe=intent.timeframe,
                    user_text=text,
                    compact=not intent.detail,
                    routing_trace=trace,
                )
            if isinstance(intent, CompareIntent):
                return await self._execute_comparison(
                    user_id=user_id,
                    request_id=request_id,
                    symbols=intent.symbols,
                    timeframe=intent.timeframe,
                    user_text=text,
                    routing_trace=trace,
                )
            if isinstance(intent, ChatQaIntent):
                return await self._execute_chat_qa(
                    user_id=user_id,
                    request_id=request_id,
                    question=intent.question,
                    routing_trace=trace,
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
        | WhaleIntent
        | ChatQaIntent
        | RefusalIntent
    ):
        intent, _, _ = await self._resolve_via_llm_fallback_with_trace(
            text=text,
            user_id=user_id,
            recent_messages=recent_messages,
            default_timeframe=default_timeframe,
            fallback_refusal=fallback_refusal,
        )
        return intent

    async def _resolve_via_llm_fallback_with_trace(
        self,
        *,
        text: str,
        user_id: int,
        recent_messages: Sequence[Any],
        default_timeframe: Timeframe,
        fallback_refusal: RefusalIntent,
        perp_universe: Collection[str] | None = None,
    ) -> tuple[
        AnalyzeIntent
        | CompareIntent
        | PineIntent
        | ScanIntent
        | ScreenIntent
        | WhaleIntent
        | ChatQaIntent
        | RefusalIntent,
        RoutingTrace,
        IntentClassificationOutcome,
    ]:
        del fallback_refusal
        recent_syms, recent_tf = _extract_recent_context(
            recent_messages,
            default_timeframe,
            perp_universe=perp_universe,
        )
        cls_outcome = await self._narrator.classify_intent_with_diagnostics(
            text,
            user_id=user_id,
            recent_symbols=recent_syms,
        )
        classified = cls_outcome.classification
        if classified is None:
            return (
                ChatQaIntent(question=text.strip()),
                RoutingTrace(
                    layer="layer2_fallback",
                    intent="chat_qa",
                    confidence=0.0,
                    fallback_reason=f"classifier_{cls_outcome.failure_reason or 'failed'}",
                ),
                cls_outcome,
            )
        if classified.intent == "unsupported":
            return (
                RefusalIntent(
                    message=(
                        "Bot ini adalah asisten riset dan edukasi kripto. Eksekusi order langsung, "
                        "perintah beli/jual wajib, jaminan profit, atau alokasi modal pribadi "
                        "berada di luar cakupan layanan."
                    ),
                    is_hard_refusal=True,
                ),
                RoutingTrace(
                    layer="layer2",
                    intent="unsupported",
                    fallback_reason="classifier_unsupported",
                ),
                cls_outcome,
            )

        if classified.intent == "whale":
            w_addr = classified.whale_address
            if not w_addr and classified.symbols:
                return (
                    WhaleIntent(action=WhaleAction.OVERVIEW, symbol=classified.symbols[0]),
                    RoutingTrace(
                        layer="layer2",
                        detected_symbol=classified.symbols[0],
                        intent="whale",
                    ),
                    cls_outcome,
                )
            if w_addr:
                return (
                    WhaleIntent(action=WhaleAction.WALLET, address=w_addr),
                    RoutingTrace(layer="layer2", intent="whale"),
                    cls_outcome,
                )
            target_whale_sym = classified.symbols[0] if classified.symbols else "BTC"
            return (
                WhaleIntent(
                    action=WhaleAction(classified.whale_action or "overview"),
                    symbol=target_whale_sym,
                ),
                RoutingTrace(
                    layer="layer2",
                    detected_symbol=target_whale_sym,
                    intent="whale",
                ),
                cls_outcome,
            )

        if classified.intent == "screen":
            return (
                ScreenIntent(side=classified.screen_side or "all"),
                RoutingTrace(layer="layer2", intent="screen"),
                cls_outcome,
            )

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
            return (
                ScanIntent(
                    timeframe=target_tf,
                    timeframes=(
                        None if resolved_cls_tf is not None else MULTI_STYLE_SCAN_TIMEFRAMES
                    ),
                    symbols=tuple(resolved_syms) if resolved_syms else None,
                ),
                RoutingTrace(
                    layer="layer2",
                    detected_symbol=",".join(resolved_syms) if resolved_syms else None,
                    intent="scan",
                ),
                cls_outcome,
            )
        if classified.intent == "pine":
            target_sym = (
                resolved_syms[0] if resolved_syms else (recent_syms[0] if recent_syms else None)
            )
            tf = target_tf if classified.timeframe is not None or resolved_syms else recent_tf
            if target_sym is not None:
                return (
                    PineIntent(symbol=target_sym, timeframe=tf),
                    RoutingTrace(
                        layer="layer2",
                        detected_symbol=target_sym,
                        intent="pine",
                    ),
                    cls_outcome,
                )
            return (
                ChatQaIntent(question=text.strip()),
                RoutingTrace(
                    layer="layer2",
                    intent="chat_qa",
                    fallback_reason="classifier_missing_symbol_fallback_qa",
                ),
                cls_outcome,
            )
        if classified.intent == "compare":
            if len(resolved_syms) >= 2:
                syms = tuple(resolved_syms[:3])
                return (
                    CompareIntent(symbols=syms, timeframe=target_tf),
                    RoutingTrace(
                        layer="layer2",
                        detected_symbol=",".join(syms),
                        intent="compare",
                    ),
                    cls_outcome,
                )
            if len(resolved_syms) == 1 and recent_syms:
                prior = next((s for s in recent_syms if s != resolved_syms[0]), None)
                if prior is not None:
                    syms = (prior, resolved_syms[0])
                    return (
                        CompareIntent(symbols=syms, timeframe=target_tf),
                        RoutingTrace(
                            layer="layer2",
                            detected_symbol=",".join(syms),
                            intent="compare",
                        ),
                        cls_outcome,
                    )
            return (
                ChatQaIntent(question=text.strip()),
                RoutingTrace(
                    layer="layer2",
                    detected_symbol=",".join(resolved_syms) if resolved_syms else None,
                    intent="chat_qa",
                    fallback_reason="classifier_missing_symbol_fallback_qa",
                ),
                cls_outcome,
            )
        if classified.intent in ("analyze", "analyze_detail"):
            is_detail = classified.intent == "analyze_detail"
            if resolved_syms:
                return (
                    AnalyzeIntent(symbol=resolved_syms[0], timeframe=target_tf, detail=is_detail),
                    RoutingTrace(
                        layer="layer2",
                        detected_symbol=resolved_syms[0],
                        intent=classified.intent,
                    ),
                    cls_outcome,
                )
            if recent_syms:
                tf = target_tf if classified.timeframe is not None else recent_tf
                return (
                    AnalyzeIntent(symbol=recent_syms[0], timeframe=tf, detail=is_detail),
                    RoutingTrace(
                        layer="layer2",
                        detected_symbol=recent_syms[0],
                        intent=classified.intent,
                    ),
                    cls_outcome,
                )
            return (
                ChatQaIntent(question=text.strip()),
                RoutingTrace(
                    layer="layer2",
                    intent="chat_qa",
                    fallback_reason="classifier_missing_symbol_fallback_qa",
                ),
                cls_outcome,
            )
        return (
            ChatQaIntent(question=text.strip()),
            RoutingTrace(layer="layer2", intent="chat_qa"),
            cls_outcome,
        )

    async def _execute_chat_qa(
        self,
        *,
        user_id: int,
        request_id: str,
        question: str,
        routing_trace: RoutingTrace | None = None,
    ) -> tuple[str, ...]:
        context = await asyncio.to_thread(
            self._memory.build_context_for_llm, user_id, current_question=question
        )
        qa_outcome = await self._narrator.answer_research_question_with_diagnostics(
            question,
            user_id=user_id,
            user_context=context,
        )
        base_trace = routing_trace or RoutingTrace(layer="layer2", intent="chat_qa")
        if qa_outcome.answer is None:
            qa_reason = qa_outcome.failure_reason or "failed"
            enriched_trace = base_trace.model_copy(
                update={
                    "fallback_reason": (
                        f"qa_{qa_reason}"
                        if not base_trace.fallback_reason
                        else f"{base_trace.fallback_reason}|qa_{qa_reason}"
                    ),
                    "guard_violations": qa_outcome.guard_violations,
                }
            )
            self._log_routing_trace(request_id, enriched_trace)
            await asyncio.to_thread(
                self._memory.record_routing_trace,
                user_id=user_id,
                request_id=request_id,
                trace=enriched_trace,
                model_id=qa_outcome.model_id,
                validation_passed=False,
                rejection_reasons=qa_outcome.guard_violations or (f"qa_{qa_reason}",),
                prompt_tokens=qa_outcome.prompt_tokens,
                output_tokens=qa_outcome.output_tokens,
                latency_ms=qa_outcome.latency_ms,
            )
            return (
                "Maaf, layanan diskusi AI sedang mencapai batas kuota atau sibuk saat ini. "
                "Anda tetap bisa menyebut nama koin (mis. 'analisa BTC') atau /scan.",
            )
        answer = qa_outcome.answer
        guard = validate_conversational_answer(answer)
        if not guard.is_valid:
            enriched_trace = base_trace.model_copy(
                update={
                    "fallback_reason": "qa_guard_rejected",
                    "guard_violations": guard.reasons,
                }
            )
            self._log_routing_trace(request_id, enriched_trace)
            await asyncio.to_thread(
                self._memory.record_routing_trace,
                user_id=user_id,
                request_id=request_id,
                trace=enriched_trace,
                model_id=qa_outcome.model_id,
                validation_passed=False,
                rejection_reasons=guard.reasons,
                prompt_tokens=qa_outcome.prompt_tokens,
                output_tokens=qa_outcome.output_tokens,
                latency_ms=qa_outcome.latency_ms,
            )
            logger.warning(
                "request_id=%s user_id=%s action=chat_qa guard_rejected=%s",
                request_id,
                user_id,
                ",".join(guard.reasons),
            )
            return (
                "Maaf, jawaban tidak lolos filter keamanan riset (mengandung klaim kepastian, "
                "instruksi eksekusi langsung, atau tautan terlarang). Silakan ajukan pertanyaan "
                "edukasi lain atau gunakan /analyze BTC.",
            )
        self._log_routing_trace(request_id, base_trace)
        await asyncio.to_thread(
            self._memory.record_routing_trace,
            user_id=user_id,
            request_id=request_id,
            trace=base_trace,
            model_id=qa_outcome.model_id,
            validation_passed=True,
            prompt_tokens=qa_outcome.prompt_tokens,
            output_tokens=qa_outcome.output_tokens,
            latency_ms=qa_outcome.latency_ms,
        )
        await asyncio.to_thread(self._memory.record_user_message, user_id, question)
        await asyncio.to_thread(self._memory.record_assistant_text, user_id, answer)
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
        routing_trace: RoutingTrace | None = None,
    ) -> tuple[str, ...]:
        context = await asyncio.to_thread(
            self._memory.build_context_for_llm, user_id, current_question=user_text
        )
        snapshot = await self._analysis_service.analyze(symbol, timeframe)
        outcome = await self._narrator.narrate_snapshot(
            snapshot,
            user_id=user_id,
            user_context=context,
        )
        trace_to_save = routing_trace
        if trace_to_save is not None:
            if snapshot.quality_status is QualityStatus.INSUFFICIENT_DATA:
                trace_to_save = trace_to_save.model_copy(
                    update={"fallback_reason": "snapshot_insufficient_data"}
                )
            elif not outcome.validation_passed:
                trace_to_save = trace_to_save.model_copy(
                    update={
                        "fallback_reason": (
                            outcome.rejection_reasons[0]
                            if outcome.rejection_reasons
                            else "narration_fallback"
                        ),
                        "guard_violations": outcome.rejection_reasons,
                    }
                )
            self._log_routing_trace(request_id, trace_to_save)
        await asyncio.to_thread(self._memory.record_user_message, user_id, user_text)
        await asyncio.to_thread(
            self._memory.record_assistant_turn,
            user_id,
            snapshot=snapshot,
            narration=outcome.response.narration,
        )
        await asyncio.to_thread(
            self._memory.record_llm_outcome,
            user_id=user_id,
            request_id=request_id,
            outcome=outcome,
            routing_trace=trace_to_save,
        )
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
        routing_trace: RoutingTrace | None = None,
    ) -> tuple[str, ...]:
        context = await asyncio.to_thread(
            self._memory.build_context_for_llm, user_id, current_question=user_text
        )
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
            trace_to_save = routing_trace
            if trace_to_save is not None:
                if snap.quality_status is QualityStatus.INSUFFICIENT_DATA:
                    trace_to_save = trace_to_save.model_copy(
                        update={"fallback_reason": "snapshot_insufficient_data"}
                    )
                elif not outcome.validation_passed:
                    trace_to_save = trace_to_save.model_copy(
                        update={
                            "fallback_reason": (
                                outcome.rejection_reasons[0]
                                if outcome.rejection_reasons
                                else "narration_fallback"
                            ),
                            "guard_violations": outcome.rejection_reasons,
                        }
                    )
                self._log_routing_trace(request_id, trace_to_save)
            await asyncio.to_thread(
                self._memory.record_llm_outcome,
                user_id=user_id,
                request_id=request_id,
                outcome=outcome,
                routing_trace=trace_to_save,
            )
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

        await asyncio.to_thread(self._memory.record_user_message, user_id, user_text)
        await asyncio.to_thread(
            self._memory.record_comparison_turn,
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

    async def track_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id if update.effective_user else None
        await _reply_chunks(update, await bot_service.handle_track(uid, context.args or []))

    async def whale_cmd(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        uid = update.effective_user.id if update.effective_user else None
        await _reply_chunks(update, await bot_service.handle_whale(uid, context.args or []))

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
    application.add_handler(CommandHandler("whale", whale_cmd))
    application.add_handler(CommandHandler("watchlist", watchlist_cmd))
    application.add_handler(CommandHandler("memory", memory_cmd))
    application.add_handler(CommandHandler("forget", forget_cmd))
    application.add_handler(CommandHandler("screener", screen_cmd))
    application.add_handler(CommandHandler("track", track_cmd))
    application.add_handler(MessageHandler(filters.COMMAND, unknown_cmd))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_msg))
    application.add_error_handler(global_error_handler)
