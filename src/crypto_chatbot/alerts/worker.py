"""Deterministic hourly alert worker, outbox flusher, and daily digest (no LLM calls)."""

from __future__ import annotations

import hashlib
import html
import logging
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.alerts.config import WIB, ScreenerConfig, load_universe
from crypto_chatbot.alerts.mtf import (
    ArmedSetupSnapshot,
    MtfSymbolDecision,
    build_armed_setup_dict,
    evaluate_symbol_mtf,
    make_mtf_run_id,
)
from crypto_chatbot.alerts.sender import TelegramOutboxSender, TelegramSendError
from crypto_chatbot.analysis.forward_proof import DEFAULT_VARIANT_PARAMS, compute_params_sha256
from crypto_chatbot.analysis.scanner import ScannedSetupItem, scan_market_setups
from crypto_chatbot.analysis.service import AnalysisService
from crypto_chatbot.bot.templates import (
    DEALING_ZONE_LABELS,
    format_enum_label,
    format_indonesian_number,
    format_timestamp_utc_wib,
)
from crypto_chatbot.schemas.analysis import AnalysisSnapshot, SetupStatus
from crypto_chatbot.schemas.market import Timeframe, format_trading_style_short
from crypto_chatbot.storage.repositories import (
    AlertRecord,
    AlertRepository,
    AlertRunRecord,
    AlertSymbolState,
    SetupLogRepository,
    TradeSetupRecord,
)

logger = logging.getLogger(__name__)

ALERT_ADVISORY_LOCK_KEY = 710_202_601


def compute_active_params_sha256() -> str:
    """Compute canonical SHA-256 of the active deterministic setup parameters (`v3`)."""
    return compute_params_sha256(DEFAULT_VARIANT_PARAMS["v3"])


def _fmt_price(value: float | None) -> str:
    if value is None:
        return "-"
    abs_v = abs(value)
    if abs_v == 0.0 or abs_v >= 1.0:
        decimals = 2
    elif abs_v >= 0.01:
        decimals = 4
    elif abs_v >= 0.0001:
        decimals = 6
    else:
        decimals = 8
    return format_indonesian_number(value, decimals)


def build_alert_inline_keyboard(alert_id: str) -> dict[str, Any]:
    """Build Telegram InlineKeyboardMarkup JSON for alert action buttons."""
    return {
        "inline_keyboard": [
            [
                {"text": "Detail", "callback_data": f"al:dt:{alert_id}"},
                {"text": "Jelaskan", "callback_data": f"al:ex:{alert_id}"},
            ],
            [
                {"text": "Ambil", "callback_data": f"al:tk:{alert_id}"},
                {"text": "Skip", "callback_data": f"al:sk:{alert_id}"},
                {"text": "Mute 24j", "callback_data": f"al:mt:{alert_id}"},
            ],
        ]
    }


def render_alert_message(
    item: ScannedSetupItem,
    *,
    snapshot_id: str,
    params_sha256: str,
) -> str:
    """Render deterministic Telegram HTML alert message (zero LLM calls)."""
    bias_str = item.bias.value.upper()
    style_badge = format_trading_style_short(item.timeframe)
    e_lo = _fmt_price(item.entry_low)
    e_hi = _fmt_price(item.entry_high)
    sl_p = _fmt_price(item.stop_loss)
    tp1_p = _fmt_price(item.tp1)
    tp2_p = _fmt_price(item.tp2)
    tp3_p = _fmt_price(item.tp3)
    rr_str = f"1:{format_indonesian_number(item.rr_tp2, 2)}" if item.rr_tp2 is not None else "-"
    close_str = _fmt_price(item.last_close)

    lines: list[str] = [
        f"<b>SETUP READY · {html.escape(item.symbol)} ({html.escape(style_badge)})</b>",
        f"Venue: {html.escape(item.venue)} | Bias: <b>{html.escape(bias_str)}</b> "
        f"| Close: {html.escape(close_str)}",
    ]
    if item.as_of is not None:
        lines.append(f"Candle tutup: {html.escape(format_timestamp_utc_wib(item.as_of))}")
    if item.dealing_zone is not None:
        dz_lbl = format_enum_label(DEALING_ZONE_LABELS, item.dealing_zone)
        lines.append(f"Dealing Range: {html.escape(dz_lbl)}")
    lines.append(f"• Entry: {html.escape(e_lo)} - {html.escape(e_hi)}")
    lines.append(f"• SL: {html.escape(sl_p)}")
    lines.append(
        f"• TP1: {html.escape(tp1_p)} | TP2: {html.escape(tp2_p)} "
        f"(R:R {html.escape(rr_str)}) | TP3: {html.escape(tp3_p)}"
    )
    if item.ltf_confirmation:
        lines.append(f"• Konfirmasi: {html.escape(item.ltf_confirmation)}")
    lines.append(
        f"<i>Ledger: {html.escape(snapshot_id)} | params={html.escape(params_sha256[:12])}</i>"
    )
    lines.append(
        "<i>Pengukuran deterministik untuk riset/edukasi, "
        "bukan saran transaksi atau jaminan hasil.</i>"
    )
    return "\n".join(lines)


def render_settlement_reply(
    rec: TradeSetupRecord,
    *,
    outcome_status: str,
    realized_r: float | None,
    settled_at: datetime | None,
) -> str:
    """Render deterministic thread reply when `settle-setups` updates a setup outcome."""
    label_map = {
        "entry_triggered": "ENTRY TERISI (FILLED)",
        "tp1_hit": "TP1 TERCAPAI",
        "tp2_hit": "TP2 TERCAPAI",
        "tp3_hit": "TP3 TERCAPAI",
        "sl_hit": "STOP LOSS TERKENA",
        "expired": "EXPIRED (BELUM TERISI)",
    }
    status_lbl = label_map.get(outcome_status, outcome_status.upper())
    r_str = (
        f" | Realized R: {format_indonesian_number(realized_r, 2, signed=True)}R"
        if realized_r is not None
        else ""
    )
    ts_str = (
        f"\nWaktu: {html.escape(format_timestamp_utc_wib(settled_at))}"
        if settled_at is not None
        else ""
    )
    return (
        f"<b>Update Ledger Setup · {html.escape(rec.canonical_symbol)} "
        f"({html.escape(rec.timeframe.value)})</b>\n"
        f"Status: <b>{html.escape(status_lbl)}</b>{html.escape(r_str)}{ts_str}\n"
        f"<i>Snapshot: {html.escape(rec.snapshot_id)}</i>"
    )


def _floor_candle_close(now_utc: datetime, timeframe: Timeframe) -> datetime:
    step_s = int(timeframe.duration.total_seconds())
    floored = (int(now_utc.timestamp()) // step_s) * step_s
    return datetime.fromtimestamp(floored, tz=UTC)


def make_run_id(timeframe: Timeframe, candle_close_time: datetime) -> str:
    """Build deterministic idempotent run_id from timeframe + last closed candle close_time."""
    close_utc = candle_close_time.astimezone(UTC)
    return f"{timeframe.value}:{close_utc.strftime('%Y-%m-%dT%H:%M:%SZ')}"


def _make_alert_id(run_id: str, canonical_symbol: str, direction: str) -> str:
    digest = hashlib.sha256(f"{run_id}|{canonical_symbol}|{direction}".encode()).hexdigest()[:12]
    return f"al-{digest}"


class TimeframeCycleResult(BaseModel):
    """Per-timeframe (or MTF) execution summary for one `alerts run` invocation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeframe: Timeframe
    run_id: str | None = None
    skipped_reason: str | None = None
    scanned_count: int = 0
    ready_count: int = 0
    armed_count: int = 0
    expired_count: int = 0
    queued_alerts: int = 0
    sent_alerts: int = 0
    held_digest_alerts: int = 0
    failed_deliveries: int = 0
    failed_symbols: tuple[str, ...] = ()
    symbol_errors: dict[str, str] = Field(default_factory=dict)
    stage_failures: dict[str, dict[str, str]] = Field(default_factory=dict)


class AlertCycleSummary(BaseModel):
    """Aggregate summary returned by `alerts run`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    locked_out: bool = False
    params_sha256: str
    results: tuple[TimeframeCycleResult, ...] = ()


async def flush_pending_outbox(
    *,
    alert_repo: AlertRepository,
    setup_repo: SetupLogRepository | None,
    sender: TelegramOutboxSender | None,
    default_chat_id: int | None,
    clock: Callable[[], datetime],
    dry_run: bool = False,
) -> tuple[int, int]:
    """Deliver all `pending` alerts from the outbox table; returns `(sent, failed)`."""
    if dry_run or sender is None:
        return 0, 0
    pending = alert_repo.list_pending_alerts(limit=100)
    sent_count = 0
    failed_count = 0
    for alert in pending:
        target_chat = alert.chat_id if alert.chat_id is not None else default_chat_id
        if target_chat is None:
            alert_repo.mark_alert_failed(
                alert.alert_id,
                error="missing target CHAT_ID",
                attempts=alert.send_attempts + 1,
            )
            failed_count += 1
            continue
        try:
            msg_id, attempts = await sender.send_message(
                chat_id=target_chat,
                text=alert.message_text,
                parse_mode="HTML",
                reply_markup=alert.reply_markup_json or None,
            )
            sent_at = clock().astimezone(UTC)
            alert_repo.mark_alert_sent(
                alert.alert_id,
                telegram_message_id=msg_id,
                sent_at=sent_at,
                attempts=alert.send_attempts + attempts,
            )
            if setup_repo is not None and hasattr(setup_repo, "link_telegram_message"):
                setup_repo.link_telegram_message(
                    alert.snapshot_id,
                    chat_id=target_chat,
                    message_id=msg_id,
                )
            sent_count += 1
        except TelegramSendError as exc:
            logger.warning("Alert outbox delivery failed for %s: %s", alert.alert_id, exc)
            alert_repo.mark_alert_failed(
                alert.alert_id,
                error=str(exc),
                attempts=alert.send_attempts + exc.attempts,
            )
            failed_count += 1
    return sent_count, failed_count


def _save_mtf_setup_to_ledger(
    setup_repo: SetupLogRepository,
    *,
    armed: ArmedSetupSnapshot,
    trade_setup: Any,
    params_sha: str,
) -> None:
    """Persist an MTF READY setup into `trade_setups` linked by `snapshot_id` & `params_sha256`."""
    from crypto_chatbot.schemas.analysis import (
        DealingZone,
        Fact,
        InputRange,
        MarketStructure,
        Regime,
        SetupBias,
        SmcSummary,
        TechnicalMetrics,
        TradeSetup,
    )
    from crypto_chatbot.schemas.market import MarketType, QualityStatus

    ts_obj: TradeSetup
    if isinstance(trade_setup, TradeSetup):
        ts_obj = trade_setup
    else:
        ts_obj = TradeSetup(
            status=SetupStatus.READY,
            bias=SetupBias.LONG if armed.direction == "long" else SetupBias.SHORT,
            entry_timeframe=armed.timeframe,
            entry_low=armed.entry_low,
            entry_high=armed.entry_high,
            entry_mid=(armed.entry_low + armed.entry_high) / 2.0,
            entry_source=armed.entry_source,
            stop_loss=armed.stop_loss,
            tp1=armed.tp1,
            tp2=armed.tp2,
            tp3=armed.tp3,
            rr_tp1=armed.rr_tp1,
            rr_tp2=armed.rr_tp2,
            rr_tp3=armed.rr_tp3,
            ltf_confirmation=armed.htf_bias_summary,
        )
    dz: DealingZone | None = None
    if armed.dealing_zone:
        try:
            dz = DealingZone(armed.dealing_zone)
        except ValueError:
            dz = None
    open_t = armed.as_of - armed.timeframe.duration
    range_sha = hashlib.sha256(f"{armed.snapshot_id}|{open_t.isoformat()}".encode()).hexdigest()
    snap = AnalysisSnapshot(
        snapshot_id=armed.snapshot_id,
        as_of=armed.as_of,
        provider="mtf_alert_worker",
        venue=armed.venue,
        canonical_symbol=armed.canonical_symbol,
        provider_symbol=armed.canonical_symbol,
        market_type=MarketType.PERPETUAL,
        timeframe=armed.timeframe,
        calc_version="mtf-15m-v1",
        quality_status=QualityStatus.READY,
        input_range=InputRange(
            first_open_time=open_t,
            last_open_time=open_t,
            candle_count=1,
            sha256=range_sha,
        ),
        regime=Regime.TRENDING_UP if armed.direction == "long" else Regime.TRENDING_DOWN,
        market_structure=(
            MarketStructure.BULLISH if armed.direction == "long" else MarketStructure.BEARISH
        ),
        technical=TechnicalMetrics(last_close=armed.last_close),
        smc=SmcSummary(dealing_zone=dz),
        trade_setup=ts_obj,
        facts=(
            Fact(
                fact_id="F1",
                label="Harga penutupan 15m",
                value=armed.last_close,
                unit="USDT",
                display_format="price",
                as_of=armed.as_of,
                source=f"{armed.venue}:{armed.canonical_symbol}",
            ),
        ),
    )
    setup_repo.save_setup(snap, ts_obj, params_sha256=params_sha)


async def _run_mtf_alert_cycle(
    *,
    analysis_service: AnalysisService,
    alert_repo: AlertRepository,
    setup_repo: SetupLogRepository | None,
    sender: TelegramOutboxSender | None,
    screener_config: ScreenerConfig,
    targets: tuple[str, ...],
    chat_id: int | None,
    dry_run: bool,
    params_sha: str,
    now_fn: Callable[[], datetime],
) -> TimeframeCycleResult:
    """Execute one 15m multi-timeframe top-down alert cycle (4h/1h -> 15m -> 5m)."""
    mtf_cfg = screener_config.mtf_timeframes
    if mtf_cfg is None:
        raise ValueError("mtf_timeframes must be configured for MTF alert cycle")
    setup_tf = mtf_cfg.setup
    started_at = now_fn().astimezone(UTC)

    # Fast idempotency check on floored 15m close_time before hitting exchange APIs.
    floored_close = _floor_candle_close(started_at, setup_tf)
    prelim_run_id = make_mtf_run_id(floored_close)
    if alert_repo.has_completed_run(prelim_run_id):
        logger.info("MTF alert run %s already completed; skipping", prelim_run_id)
        return TimeframeCycleResult(
            timeframe=setup_tf,
            run_id=prelim_run_id,
            skipped_reason="already_processed",
        )

    tf_limits: dict[Timeframe, int] = {}
    for btf in mtf_cfg.bias:
        tf_limits[btf] = 120
    tf_limits[setup_tf] = 200
    tf_limits[mtf_cfg.confirm] = 60

    decisions: list[MtfSymbolDecision] = []
    failed_symbols: list[str] = []
    symbol_errors: dict[str, str] = {}
    stage_failures: dict[str, dict[str, str]] = {}

    for sym in targets:
        try:
            batches, _venue = await analysis_service.fetch_mtf_candle_bundle(
                sym,
                tf_limits,
                as_of=started_at,
                incremental=True,
            )
            prev = alert_repo.get_symbol_state(sym, setup_tf)
            decision = evaluate_symbol_mtf(
                canonical_symbol=sym,
                batches=batches,
                mtf_config=mtf_cfg,
                prev_state=prev,
                now_utc=now_fn().astimezone(UTC),
            )
            decisions.append(decision)
        except Exception as exc:
            err_msg = f"{type(exc).__name__}: {exc}"
            logger.warning("MTF alert scan failed for %s: %s", sym, err_msg)
            failed_symbols.append(sym)
            symbol_errors[sym] = err_msg
            stage_failures[sym] = {"stage": "error", "reason": err_msg[:240]}

    close_times = [d.setup_close_time.astimezone(UTC) for d in decisions]
    candle_close_time = max(close_times) if close_times else floored_close
    run_id = make_mtf_run_id(candle_close_time)

    if alert_repo.has_completed_run(run_id):
        logger.info("MTF alert run %s already completed; skipping", run_id)
        return TimeframeCycleResult(
            timeframe=setup_tf,
            run_id=run_id,
            skipped_reason="already_processed",
            scanned_count=len(decisions),
            ready_count=sum(1 for d in decisions if d.outcome == "ready"),
            failed_symbols=tuple(failed_symbols),
            symbol_errors=symbol_errors,
            stage_failures=stage_failures,
        )

    alert_repo.record_run_start(
        AlertRunRecord(
            run_id=run_id,
            timeframe=setup_tf,
            candle_close_time=candle_close_time,
            started_at=started_at,
            status="running",
            scanned_count=len(decisions),
            failed_symbols=tuple(failed_symbols),
            symbol_errors=symbol_errors,
            stage_failures=stage_failures,
        )
    )

    now_utc = now_fn().astimezone(UTC)
    is_quiet = screener_config.quiet_hours_wib.is_quiet(now_utc)
    cooldown = screener_config.cooldown_for(setup_tf)
    expiry_delta = timedelta(minutes=screener_config.setup_expiry_minutes)

    ready_count = 0
    armed_count = 0
    expired_count = 0
    queued_live = 0
    held_digest = 0

    for decision in decisions:
        sym = decision.canonical_symbol
        prev = alert_repo.get_symbol_state(sym, setup_tf)

        if decision.failed_stage is not None:
            stage_failures[sym] = {
                "stage": decision.failed_stage,
                "reason": (decision.failed_reason or "")[:240],
            }

        if decision.outcome == "armed" and decision.armed_setup is not None:
            armed_count += 1
            keep_existing_expiry = (
                prev is not None
                and prev.last_status == "armed"
                and prev.armed_direction == decision.direction
                and prev.armed_expires_at is not None
                and now_utc < prev.armed_expires_at.astimezone(UTC)
            )
            armed_at = prev.armed_at if keep_existing_expiry and prev and prev.armed_at else now_utc
            armed_exp = (
                prev.armed_expires_at
                if keep_existing_expiry and prev and prev.armed_expires_at
                else (now_utc + expiry_delta)
            )
            alert_repo.upsert_symbol_state(
                AlertSymbolState(
                    canonical_symbol=sym,
                    timeframe=setup_tf,
                    last_status="armed",
                    last_bias=decision.direction,
                    last_run_id=run_id,
                    last_alert_time_long=prev.last_alert_time_long if prev else None,
                    last_alert_time_short=prev.last_alert_time_short if prev else None,
                    muted_until=prev.muted_until if prev else None,
                    armed_direction=decision.direction,
                    armed_at=armed_at,
                    armed_expires_at=armed_exp,
                    armed_setup_json=build_armed_setup_dict(decision.armed_setup),
                    last_stage_failed=decision.failed_stage,
                    last_stage_reason=decision.failed_reason,
                    updated_at=now_utc,
                )
            )
            continue

        if decision.outcome in {"wait", "expired"}:
            if decision.outcome == "expired":
                expired_count += 1
            alert_repo.upsert_symbol_state(
                AlertSymbolState(
                    canonical_symbol=sym,
                    timeframe=setup_tf,
                    last_status=SetupStatus.WAIT.value,
                    last_bias=None,
                    last_run_id=run_id,
                    last_alert_time_long=prev.last_alert_time_long if prev else None,
                    last_alert_time_short=prev.last_alert_time_short if prev else None,
                    muted_until=prev.muted_until if prev else None,
                    armed_direction=None,
                    armed_at=None,
                    armed_expires_at=None,
                    armed_setup_json=None,
                    last_stage_failed=decision.failed_stage,
                    last_stage_reason=decision.failed_reason,
                    updated_at=now_utc,
                )
            )
            continue

        # Outcome == "ready"
        ready_count += 1
        if decision.item is None or decision.direction is None:
            continue
        direction = decision.direction
        ready_item = decision.item

        # 1. State-change check (WAIT/ARMED -> READY, or bias flip).
        is_state_change = (
            prev is None
            or prev.last_status != SetupStatus.READY.value
            or prev.last_bias != direction
        )
        if not is_state_change:
            alert_repo.upsert_symbol_state(
                AlertSymbolState(
                    canonical_symbol=sym,
                    timeframe=setup_tf,
                    last_status=SetupStatus.READY.value,
                    last_bias=direction,
                    last_run_id=run_id,
                    last_alert_time_long=prev.last_alert_time_long if prev else None,
                    last_alert_time_short=prev.last_alert_time_short if prev else None,
                    muted_until=prev.muted_until if prev else None,
                    updated_at=now_utc,
                )
            )
            continue

        # 2. Mute check.
        if prev is not None and prev.muted_until is not None and now_utc < prev.muted_until:
            alert_repo.upsert_symbol_state(
                AlertSymbolState(
                    canonical_symbol=sym,
                    timeframe=setup_tf,
                    last_status=SetupStatus.READY.value,
                    last_bias=direction,
                    last_run_id=run_id,
                    last_alert_time_long=prev.last_alert_time_long,
                    last_alert_time_short=prev.last_alert_time_short,
                    muted_until=prev.muted_until,
                    updated_at=now_utc,
                )
            )
            continue

        # 3. Cooldown check per (symbol, direction).
        last_dir_alert = (
            (prev.last_alert_time_long if direction == "long" else prev.last_alert_time_short)
            if prev is not None
            else None
        )
        if last_dir_alert is not None and (now_utc - last_dir_alert) < cooldown:
            alert_repo.upsert_symbol_state(
                AlertSymbolState(
                    canonical_symbol=sym,
                    timeframe=setup_tf,
                    last_status=SetupStatus.READY.value,
                    last_bias=direction,
                    last_run_id=run_id,
                    last_alert_time_long=prev.last_alert_time_long if prev else None,
                    last_alert_time_short=prev.last_alert_time_short if prev else None,
                    muted_until=prev.muted_until if prev else None,
                    updated_at=now_utc,
                )
            )
            continue

        # 4. Link to setup ledger (`trade_setups`) with params_sha256.
        snap_id = ready_item.snapshot_id or f"snap-{_make_alert_id(run_id, sym, direction)}"
        if setup_repo is not None and decision.armed_setup is not None:
            _save_mtf_setup_to_ledger(
                setup_repo,
                armed=decision.armed_setup,
                trade_setup=decision.trade_setup,
                params_sha=params_sha,
            )

        # 5. Write to `alerts` outbox first (`pending` vs `held_digest`).
        alert_id = _make_alert_id(run_id, sym, direction)
        msg_html = render_alert_message(
            ready_item,
            snapshot_id=snap_id,
            params_sha256=params_sha,
        )
        keyboard = build_alert_inline_keyboard(alert_id)

        if is_quiet:
            out_status = "held_digest"
            hold_reason: str | None = "quiet_hours"
            held_digest += 1
        elif queued_live >= screener_config.max_alerts_per_run:
            out_status = "held_digest"
            hold_reason = "max_per_run"
            held_digest += 1
        else:
            out_status = "pending"
            hold_reason = None
            queued_live += 1

        alert_repo.create_alert(
            AlertRecord(
                alert_id=alert_id,
                run_id=run_id,
                snapshot_id=snap_id,
                params_sha256=params_sha,
                canonical_symbol=sym,
                timeframe=setup_tf,
                direction=direction,
                venue=ready_item.venue,
                chat_id=chat_id,
                status=out_status,
                hold_reason=hold_reason,
                message_text=msg_html,
                reply_markup_json=keyboard,
                created_at=now_utc,
            )
        )

        new_long_t = (
            now_utc if direction == "long" else (prev.last_alert_time_long if prev else None)
        )
        new_short_t = (
            now_utc if direction == "short" else (prev.last_alert_time_short if prev else None)
        )
        alert_repo.upsert_symbol_state(
            AlertSymbolState(
                canonical_symbol=sym,
                timeframe=setup_tf,
                last_status=SetupStatus.READY.value,
                last_bias=direction,
                last_run_id=run_id,
                last_alert_time_long=new_long_t,
                last_alert_time_short=new_short_t,
                muted_until=prev.muted_until if prev else None,
                armed_direction=None,
                armed_at=None,
                armed_expires_at=None,
                armed_setup_json=None,
                last_stage_failed=None,
                last_stage_reason=None,
                updated_at=now_utc,
            )
        )

    sent_now, failed_now = await flush_pending_outbox(
        alert_repo=alert_repo,
        setup_repo=setup_repo,
        sender=sender,
        default_chat_id=chat_id,
        clock=now_fn,
        dry_run=dry_run,
    )

    alert_repo.record_run_finish(
        run_id,
        status="completed",
        finished_at=now_fn().astimezone(UTC),
        scanned_count=len(decisions),
        ready_count=ready_count,
        armed_count=armed_count,
        expired_count=expired_count,
        alerted_count=sent_now,
        held_digest_count=held_digest,
        failed_symbols=tuple(failed_symbols),
        symbol_errors=symbol_errors,
        stage_failures=stage_failures,
    )
    return TimeframeCycleResult(
        timeframe=setup_tf,
        run_id=run_id,
        scanned_count=len(decisions),
        ready_count=ready_count,
        armed_count=armed_count,
        expired_count=expired_count,
        queued_alerts=queued_live,
        sent_alerts=sent_now,
        held_digest_alerts=held_digest,
        failed_deliveries=failed_now,
        failed_symbols=tuple(failed_symbols),
        symbol_errors=symbol_errors,
        stage_failures=stage_failures,
    )


async def run_alert_cycle(
    *,
    analysis_service: AnalysisService,
    alert_repo: AlertRepository,
    setup_repo: SetupLogRepository | None = None,
    sender: TelegramOutboxSender | None = None,
    screener_config: ScreenerConfig,
    symbols: Sequence[str] | None = None,
    chat_id: int | None = None,
    dry_run: bool = False,
    lock_key: int = ALERT_ADVISORY_LOCK_KEY,
    clock: Callable[[], datetime] | None = None,
) -> AlertCycleSummary:
    """Execute one idempotent, advisory-locked alert worker cycle (MTF 15m or legacy single-TF)."""
    now_fn = clock or (lambda: datetime.now(UTC))
    params_sha = compute_active_params_sha256()

    if not alert_repo.try_acquire_advisory_lock(lock_key):
        logger.info("Alert run skipped: advisory lock %s is already held", lock_key)
        return AlertCycleSummary(locked_out=True, params_sha256=params_sha)

    targets = (
        tuple(symbols)
        if symbols is not None and len(symbols) > 0
        else load_universe(screener_config.universe_path)
    )
    results: list[TimeframeCycleResult] = []
    try:
        if screener_config.is_mtf_mode:
            mtf_res = await _run_mtf_alert_cycle(
                analysis_service=analysis_service,
                alert_repo=alert_repo,
                setup_repo=setup_repo,
                sender=sender,
                screener_config=screener_config,
                targets=targets,
                chat_id=chat_id,
                dry_run=dry_run,
                params_sha=params_sha,
                now_fn=now_fn,
            )
            results.append(mtf_res)
            return AlertCycleSummary(
                locked_out=False,
                params_sha256=params_sha,
                results=tuple(results),
            )
        for tf in screener_config.timeframes:
            started_at = now_fn().astimezone(UTC)
            captured_snapshots: list[AnalysisSnapshot] = []
            report = await scan_market_setups(
                analysis_service,
                targets,
                timeframe=tf,
                snapshots_out=captured_snapshots,
            )
            snaps_by_id = {s.snapshot_id: s for s in captured_snapshots}

            close_times = [s.as_of.astimezone(UTC) for s in captured_snapshots]
            candle_close_time = (
                max(close_times) if close_times else _floor_candle_close(started_at, tf)
            )
            run_id = make_run_id(tf, candle_close_time)

            if alert_repo.has_completed_run(run_id):
                logger.info("Alert run %s already completed; skipping", run_id)
                results.append(
                    TimeframeCycleResult(
                        timeframe=tf,
                        run_id=run_id,
                        skipped_reason="already_processed",
                        scanned_count=report.scanned_count,
                        ready_count=len(report.ready_setups),
                        failed_symbols=report.failed_symbols,
                        symbol_errors=dict(report.symbol_errors),
                    )
                )
                continue

            alert_repo.record_run_start(
                AlertRunRecord(
                    run_id=run_id,
                    timeframe=tf,
                    candle_close_time=candle_close_time,
                    started_at=started_at,
                    status="running",
                    scanned_count=report.scanned_count,
                    ready_count=len(report.ready_setups),
                    failed_symbols=report.failed_symbols,
                    symbol_errors=dict(report.symbol_errors),
                )
            )

            now_utc = now_fn().astimezone(UTC)
            is_quiet = screener_config.quiet_hours_wib.is_quiet(now_utc)
            cooldown = screener_config.cooldown_for(tf)

            # Update WAIT states so subsequent transitions WAIT -> READY trigger alerts.
            for wait_item in report.wait_setups:
                prev = alert_repo.get_symbol_state(wait_item.symbol, tf)
                alert_repo.upsert_symbol_state(
                    AlertSymbolState(
                        canonical_symbol=wait_item.symbol,
                        timeframe=tf,
                        last_status=SetupStatus.WAIT.value,
                        last_bias=wait_item.bias.value,
                        last_run_id=run_id,
                        last_alert_time_long=prev.last_alert_time_long if prev else None,
                        last_alert_time_short=prev.last_alert_time_short if prev else None,
                        muted_until=prev.muted_until if prev else None,
                        updated_at=now_utc,
                    )
                )

            queued_live = 0
            held_digest = 0

            for ready_item in report.ready_setups:
                direction = ready_item.bias.value
                prev = alert_repo.get_symbol_state(ready_item.symbol, tf)

                # 1. State-change check: alert only on transition to READY (or bias flip).
                is_state_change = (
                    prev is None
                    or prev.last_status != SetupStatus.READY.value
                    or prev.last_bias != direction
                )
                if not is_state_change:
                    alert_repo.upsert_symbol_state(
                        AlertSymbolState(
                            canonical_symbol=ready_item.symbol,
                            timeframe=tf,
                            last_status=SetupStatus.READY.value,
                            last_bias=direction,
                            last_run_id=run_id,
                            last_alert_time_long=prev.last_alert_time_long if prev else None,
                            last_alert_time_short=prev.last_alert_time_short if prev else None,
                            muted_until=prev.muted_until if prev else None,
                            updated_at=now_utc,
                        )
                    )
                    continue

                # 2. Mute check (from [Mute 24j] inline button).
                if prev is not None and prev.muted_until is not None and now_utc < prev.muted_until:
                    alert_repo.upsert_symbol_state(
                        AlertSymbolState(
                            canonical_symbol=ready_item.symbol,
                            timeframe=tf,
                            last_status=SetupStatus.READY.value,
                            last_bias=direction,
                            last_run_id=run_id,
                            last_alert_time_long=prev.last_alert_time_long,
                            last_alert_time_short=prev.last_alert_time_short,
                            muted_until=prev.muted_until,
                            updated_at=now_utc,
                        )
                    )
                    continue

                # 3. Cooldown check per (symbol, direction).
                last_dir_alert = (
                    (
                        prev.last_alert_time_long
                        if direction == "long"
                        else prev.last_alert_time_short
                    )
                    if prev is not None
                    else None
                )
                if last_dir_alert is not None and (now_utc - last_dir_alert) < cooldown:
                    alert_repo.upsert_symbol_state(
                        AlertSymbolState(
                            canonical_symbol=ready_item.symbol,
                            timeframe=tf,
                            last_status=SetupStatus.READY.value,
                            last_bias=direction,
                            last_run_id=run_id,
                            last_alert_time_long=prev.last_alert_time_long if prev else None,
                            last_alert_time_short=prev.last_alert_time_short if prev else None,
                            muted_until=prev.muted_until if prev else None,
                            updated_at=now_utc,
                        )
                    )
                    continue

                # 4. Link to setup ledger (`trade_setups`) with params_sha256.
                snap_id = (
                    ready_item.snapshot_id
                    or f"snap-{_make_alert_id(run_id, ready_item.symbol, direction)}"
                )
                snap_obj = snaps_by_id.get(snap_id)
                if (
                    setup_repo is not None
                    and snap_obj is not None
                    and snap_obj.trade_setup is not None
                ):
                    setup_repo.save_setup(
                        snap_obj,
                        snap_obj.trade_setup,
                        params_sha256=params_sha,
                    )

                # 5. Write to `alerts` outbox first (`pending` vs `held_digest`).
                alert_id = _make_alert_id(run_id, ready_item.symbol, direction)
                msg_html = render_alert_message(
                    ready_item,
                    snapshot_id=snap_id,
                    params_sha256=params_sha,
                )
                keyboard = build_alert_inline_keyboard(alert_id)

                if is_quiet:
                    out_status = "held_digest"
                    hold_reason: str | None = "quiet_hours"
                    held_digest += 1
                elif queued_live >= screener_config.max_alerts_per_run:
                    out_status = "held_digest"
                    hold_reason = "max_per_run"
                    held_digest += 1
                else:
                    out_status = "pending"
                    hold_reason = None
                    queued_live += 1

                alert_repo.create_alert(
                    AlertRecord(
                        alert_id=alert_id,
                        run_id=run_id,
                        snapshot_id=snap_id,
                        params_sha256=params_sha,
                        canonical_symbol=ready_item.symbol,
                        timeframe=tf,
                        direction=direction,
                        venue=ready_item.venue,
                        chat_id=chat_id,
                        status=out_status,
                        hold_reason=hold_reason,
                        message_text=msg_html,
                        reply_markup_json=keyboard,
                        created_at=now_utc,
                    )
                )

                new_long_t = (
                    now_utc
                    if direction == "long"
                    else (prev.last_alert_time_long if prev else None)
                )
                new_short_t = (
                    now_utc
                    if direction == "short"
                    else (prev.last_alert_time_short if prev else None)
                )
                alert_repo.upsert_symbol_state(
                    AlertSymbolState(
                        canonical_symbol=ready_item.symbol,
                        timeframe=tf,
                        last_status=SetupStatus.READY.value,
                        last_bias=direction,
                        last_run_id=run_id,
                        last_alert_time_long=new_long_t,
                        last_alert_time_short=new_short_t,
                        muted_until=prev.muted_until if prev else None,
                        updated_at=now_utc,
                    )
                )

            sent_now, failed_now = await flush_pending_outbox(
                alert_repo=alert_repo,
                setup_repo=setup_repo,
                sender=sender,
                default_chat_id=chat_id,
                clock=now_fn,
                dry_run=dry_run,
            )

            alert_repo.record_run_finish(
                run_id,
                status="completed",
                finished_at=now_fn().astimezone(UTC),
                scanned_count=report.scanned_count,
                ready_count=len(report.ready_setups),
                alerted_count=sent_now,
                held_digest_count=held_digest,
                failed_symbols=report.failed_symbols,
                symbol_errors=dict(report.symbol_errors),
            )
            results.append(
                TimeframeCycleResult(
                    timeframe=tf,
                    run_id=run_id,
                    scanned_count=report.scanned_count,
                    ready_count=len(report.ready_setups),
                    queued_alerts=queued_live,
                    sent_alerts=sent_now,
                    held_digest_alerts=held_digest,
                    failed_deliveries=failed_now,
                    failed_symbols=report.failed_symbols,
                    symbol_errors=dict(report.symbol_errors),
                )
            )
    finally:
        alert_repo.release_advisory_lock(lock_key)

    return AlertCycleSummary(
        locked_out=False,
        params_sha256=params_sha,
        results=tuple(results),
    )


class DigestResult(BaseModel):
    """Summary returned by `alerts digest`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    generated_at: datetime
    recent_alerts_24h: int
    held_alerts_included: int
    sent: bool
    telegram_message_id: int | None = None
    message_text: str


async def run_alert_digest(
    *,
    alert_repo: AlertRepository,
    sender: TelegramOutboxSender | None = None,
    chat_id: int | None = None,
    dry_run: bool = False,
    clock: Callable[[], datetime] | None = None,
) -> DigestResult:
    """Build and send daily digest including alerts held during quiet hours or max_per_run cap."""
    now_utc = (clock or (lambda: datetime.now(UTC)))().astimezone(UTC)
    since = now_utc - timedelta(hours=24)
    recent = alert_repo.list_recent_alerts(since=since)
    held = alert_repo.list_held_digest_alerts(since=since)
    recent_runs = (
        alert_repo.list_recent_runs(since=since) if hasattr(alert_repo, "list_recent_runs") else ()
    )

    stage_counts: dict[str, int] = {}
    latest_stage_by_symbol: dict[str, tuple[str, str]] = {}
    for run in recent_runs:
        for sym, info in run.stage_failures.items():
            st = str(info.get("stage", "unknown"))
            stage_counts[st] = stage_counts.get(st, 0) + 1
            if sym not in latest_stage_by_symbol:
                latest_stage_by_symbol[sym] = (st, str(info.get("reason", "")))

    sent_24h = [a for a in recent if a.status == "sent"]
    quiet_held = [a for a in held if a.hold_reason == "quiet_hours"]
    cap_held = [a for a in held if a.hold_reason == "max_per_run"]

    wib_date = now_utc.astimezone(WIB).strftime("%Y-%m-%d %H:%M WIB")
    lines: list[str] = [
        f"<b>Ringkasan Harian Alert Setup ({html.escape(wib_date)})</b>",
        f"• Alert terkirim (24j): {len(sent_24h)}",
        f"• Ditahan saat quiet hours: {len(quiet_held)}",
        f"• Ekstra melebihi batas per run: {len(cap_held)}",
    ]

    if stage_counts:
        stage_labels = {
            "bias": "Bias HTF (4h/1h)",
            "setup": "Setup 15m",
            "confirm": "Konfirmasi 5m",
            "cost_gate": "Cost Gate",
            "error": "Error Data/Venue",
        }
        parts = [
            f"{stage_labels.get(st, st)}={cnt}"
            for st, cnt in sorted(stage_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ]
        lines.append(f"• Penyaringan 24j ({len(recent_runs)} run): {html.escape(', '.join(parts))}")

    if held:
        lines.append("")
        lines.append("<b>Setup Ditahan untuk Digest:</b>")
        for idx, al in enumerate(held, start=1):
            reason_lbl = "Quiet Hours" if al.hold_reason == "quiet_hours" else "Batas/Run"
            lines.append(
                f"{idx}. <b>{html.escape(al.canonical_symbol)}</b> "
                f"({html.escape(al.timeframe.value)} · {html.escape(al.direction.upper())}) "
                f"[{html.escape(reason_lbl)}] — Ledger: {html.escape(al.snapshot_id)}"
            )
    elif sent_24h:
        lines.append("")
        lines.append("<b>Alert Terkirim 24 Jam Terakhir:</b>")
        for idx, al in enumerate(sent_24h[:10], start=1):
            lines.append(
                f"{idx}. <b>{html.escape(al.canonical_symbol)}</b> "
                f"({html.escape(al.timeframe.value)} · {html.escape(al.direction.upper())}) "
                f"— Ledger: {html.escape(al.snapshot_id)}"
            )
    else:
        lines.append("")
        lines.append("Belum ada alert setup READY dalam 24 jam terakhir.")
        if latest_stage_by_symbol:
            lines.append("<b>Alasan Terakhir per Simbol:</b>")
            for sym, (st, rsn) in list(latest_stage_by_symbol.items())[:8]:
                short_rsn = rsn[:90] if rsn else "-"
                lines.append(f"• {html.escape(sym)}: [{html.escape(st)}] {html.escape(short_rsn)}")

    lines.append("")
    lines.append(
        "<i>Ringkasan deterministik riset/edukasi, bukan saran transaksi atau jaminan hasil.</i>"
    )
    text = "\n".join(lines)

    msg_id: int | None = None
    delivered = False
    if not dry_run and sender is not None and chat_id is not None:
        msg_id, _attempts = await sender.send_message(
            chat_id=chat_id,
            text=text,
            parse_mode="HTML",
        )
        delivered = True
        if held:
            alert_repo.mark_alerts_digest_sent([a.alert_id for a in held], sent_at=now_utc)

    return DigestResult(
        generated_at=now_utc,
        recent_alerts_24h=len(recent),
        held_alerts_included=len(held),
        sent=delivered,
        telegram_message_id=msg_id,
        message_text=text,
    )
