"""Deterministic multi-coin perpetual futures setup scanner."""

from __future__ import annotations

import asyncio
import html
import logging
from collections import Counter
from collections.abc import Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.alerts.config import load_universe
from crypto_chatbot.analysis.service import AnalysisService
from crypto_chatbot.bot.templates import (
    DEALING_ZONE_LABELS,
    format_enum_label,
    format_indonesian_number,
)
from crypto_chatbot.market_data.symbols import resolve_default_perp_symbol
from crypto_chatbot.schemas.analysis import (
    AnalysisSnapshot,
    DealingZone,
    Regime,
    SetupBias,
    SetupStatus,
)
from crypto_chatbot.schemas.market import (
    QualityStatus,
    Timeframe,
    format_trading_style_label,
    format_trading_style_short,
)

logger = logging.getLogger(__name__)

DEFAULT_SCAN_BASKET: tuple[str, ...] = (
    "BTC/USDT:USDT",
    "ETH/USDT:USDT",
    "SOL/USDT:USDT",
    "BNB/USDT:USDT",
    "XRP/USDT:USDT",
    "SUI/USDT:USDT",
    "DOGE/USDT:USDT",
    "AVAX/USDT:USDT",
    "LINK/USDT:USDT",
    "ADA/USDT:USDT",
)


class ScannedSetupItem(BaseModel):
    """Summary of one scanned perpetual futures contract's deterministic setup."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    timeframe: Timeframe
    venue: str
    status: SetupStatus
    bias: SetupBias
    last_close: float
    entry_low: float | None = None
    entry_high: float | None = None
    stop_loss: float | None = None
    tp1: float | None = None
    tp2: float | None = None
    tp3: float | None = None
    rr_tp2: float | None = None
    ltf_confirmation: str | None = None
    wait_reason: str | None = None
    regime: Regime | None = None
    dealing_zone: DealingZone | None = None
    snapshot_id: str | None = None
    as_of: datetime | None = None


class MarketScanReport(BaseModel):
    """Aggregate report of a multi-coin deterministic futures setup scan."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeframe: Timeframe
    scanned_timeframes: tuple[Timeframe, ...] = ()
    scanned_count: int
    ready_setups: tuple[ScannedSetupItem, ...]
    wait_setups: tuple[ScannedSetupItem, ...]
    failed_symbols: tuple[str, ...]
    symbol_errors: dict[str, str] = Field(default_factory=dict)
    disclaimer: str


def _fmt_price(value: float) -> str:
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


async def scan_market_setups(
    analysis_service: AnalysisService,
    symbols: Sequence[str],
    *,
    timeframe: Timeframe = Timeframe.H4,
    timeframes: Sequence[Timeframe] | None = None,
    max_concurrency: int = 3,
    snapshots_out: list[AnalysisSnapshot] | None = None,
) -> MarketScanReport:
    """Concurrently scan perpetual futures coins and surface deterministic READY setups."""
    normalized: list[str] = []
    for raw in symbols:
        try:
            canon = resolve_default_perp_symbol(raw).canonical_symbol
        except ValueError:
            continue
        if canon not in normalized:
            normalized.append(canon)
    targets = tuple(normalized) if normalized else load_universe(fallback=DEFAULT_SCAN_BASKET)
    scan_tfs: tuple[Timeframe, ...] = (
        tuple(dict.fromkeys(timeframes)) if timeframes else (timeframe,)
    )

    sem = asyncio.Semaphore(max(1, max_concurrency))
    ready_items: list[ScannedSetupItem] = []
    wait_items: list[ScannedSetupItem] = []
    failed: list[str] = []
    symbol_errors: dict[str, str] = {}

    async def _scan_one(sym: str) -> None:
        async with sem:
            snaps: list[AnalysisSnapshot] = []
            try:
                if len(scan_tfs) > 1 and hasattr(analysis_service, "analyze_multi_style"):
                    snap_map = await analysis_service.analyze_multi_style(
                        sym,
                        scan_tfs,
                        include_derivatives=True,
                    )
                    snaps.extend(snap_map[tf] for tf in scan_tfs if tf in snap_map)
                else:
                    for tf in scan_tfs:
                        snaps.append(
                            await analysis_service.analyze(
                                sym,
                                tf,
                                include_mtf=True,
                                include_derivatives=True,
                                persist=False,
                            )
                        )

                had_valid_snapshot = False
                last_canonical = sym
                for snap in snaps:
                    last_canonical = snap.canonical_symbol
                    if (
                        snap.quality_status is QualityStatus.INSUFFICIENT_DATA
                        or snap.technical is None
                        or snap.trade_setup is None
                    ):
                        continue

                    had_valid_snapshot = True
                    if snapshots_out is not None:
                        snapshots_out.append(snap)
                    setup = snap.trade_setup
                    item = ScannedSetupItem(
                        symbol=snap.canonical_symbol,
                        timeframe=snap.timeframe,
                        venue=snap.venue.upper(),
                        status=setup.status,
                        bias=setup.bias,
                        last_close=snap.technical.last_close,
                        entry_low=setup.entry_low,
                        entry_high=setup.entry_high,
                        stop_loss=setup.stop_loss,
                        tp1=setup.tp1,
                        tp2=setup.tp2,
                        tp3=setup.tp3,
                        rr_tp2=setup.rr_tp2,
                        ltf_confirmation=setup.ltf_confirmation,
                        wait_reason=setup.wait_reason,
                        regime=snap.regime,
                        dealing_zone=snap.smc.dealing_zone if snap.smc is not None else None,
                        snapshot_id=snap.snapshot_id,
                        as_of=snap.as_of,
                    )
                    if setup.status is SetupStatus.READY:
                        ready_items.append(item)
                    else:
                        wait_items.append(item)

                if not had_valid_snapshot:
                    failed.append(last_canonical)
                    symbol_errors[last_canonical] = (
                        "ERROR: data tidak mencukupi (INSUFFICIENT_DATA)"
                    )
            except Exception as exc:
                logger.warning("Market scan failed for %s: %s", sym, exc)
                failed.append(sym)
                symbol_errors[sym] = f"ERROR: {exc}"

    await asyncio.gather(*(_scan_one(sym) for sym in targets))

    ready_sorted = tuple(
        sorted(
            ready_items,
            key=lambda item: (-(item.rr_tp2 or 0.0), item.symbol, item.timeframe.value),
        )
    )
    wait_sorted = tuple(sorted(wait_items, key=lambda item: (item.symbol, item.timeframe.value)))

    return MarketScanReport(
        timeframe=scan_tfs[0] if len(scan_tfs) == 1 else timeframe,
        scanned_timeframes=scan_tfs,
        scanned_count=len(targets),
        ready_setups=ready_sorted,
        wait_setups=wait_sorted,
        failed_symbols=tuple(failed),
        symbol_errors=symbol_errors,
        disclaimer=(
            "Hasil pemindaian merupakan pengukuran deterministik multi-timeframe dan bukan "
            "saran transaksi atau jaminan hasil."
        ),
    )


def render_market_scan_report(report: MarketScanReport, *, html_mode: bool = True) -> str:
    """Render deterministic Indonesian scan report for Telegram HTML or CLI text."""

    def _b(text: str) -> str:
        esc = html.escape(text, quote=False) if html_mode else text
        return f"<b>{esc}</b>" if html_mode else esc

    def _e(text: str) -> str:
        return html.escape(text, quote=False) if html_mode else text

    is_multi = len(report.scanned_timeframes) > 1
    if is_multi:
        styles_hdr = " · ".join(format_trading_style_short(tf) for tf in report.scanned_timeframes)
        header_title = f"Pemindai Setup Futures Multi-Style ({styles_hdr})"
    else:
        style_lbl = format_trading_style_label(report.timeframe)
        header_title = (
            f"Pemindai Setup Futures Unified MTF ({report.timeframe.value} · {style_lbl})"
        )

    lines: list[str] = [
        _b(header_title),
        _e(
            f"Dipindai: {report.scanned_count} koin | "
            f"Setup READY: {len(report.ready_setups)} | "
            f"WAIT: {len(report.wait_setups)}"
            + (f" | Gagal/Skip: {len(report.failed_symbols)}" if report.failed_symbols else "")
        ),
    ]

    if is_multi and report.ready_setups:
        style_counts = Counter(item.timeframe for item in report.ready_setups)
        breakdown = " | ".join(
            f"{format_trading_style_short(tf)}: {style_counts[tf]}"
            for tf in report.scanned_timeframes
            if style_counts[tf] > 0
        )
        if breakdown:
            lines.append(_e(f"Sebaran Gaya READY: {breakdown}"))

    if report.ready_setups:
        lines.append("")
        lines.append(_b("Koin dengan Setup READY (Terurut berdasar R:R TP2):"))
        for idx, item in enumerate(report.ready_setups, start=1):
            e_lo = _fmt_price(item.entry_low) if item.entry_low is not None else "-"
            e_hi = _fmt_price(item.entry_high) if item.entry_high is not None else "-"
            sl_p = _fmt_price(item.stop_loss) if item.stop_loss is not None else "-"
            tp1_p = _fmt_price(item.tp1) if item.tp1 is not None else "-"
            tp2_p = _fmt_price(item.tp2) if item.tp2 is not None else "-"
            tp3_p = _fmt_price(item.tp3) if item.tp3 is not None else "-"
            rr_str = (
                f"1:{format_indonesian_number(item.rr_tp2, 2)}" if item.rr_tp2 is not None else "-"
            )
            bias_lbl = item.bias.value.upper()
            style_short = format_trading_style_short(item.timeframe)
            style_full = format_trading_style_label(item.timeframe)
            lines.append("")
            lines.append(
                f"{idx}. {_b(item.symbol)} [{_e(item.venue)}] · "
                f"{_b(bias_lbl)} · {_b(f'[{style_short}]')} "
                f"(Harga: {_e(_fmt_price(item.last_close))})"
            )
            lines.append(_e(f"   • Gaya Setup: {style_full}"))
            if item.dealing_zone is not None:
                dz_lbl = format_enum_label(DEALING_ZONE_LABELS, item.dealing_zone)
                lines.append(_e(f"   • Dealing Range: {dz_lbl}"))
            lines.append(_e(f"   • Entry: {e_lo} - {e_hi} | SL: {sl_p}"))
            lines.append(_e(f"   • TP1: {tp1_p} | TP2: {tp2_p} (R:R {rr_str}) | TP3: {tp3_p}"))
            if item.ltf_confirmation:
                lines.append(_e(f"   • Konfirmasi: {item.ltf_confirmation}"))

        lines.append("")
        lines.append(
            _e(
                "Tip: Balas langsung nama koin (misal 'bedah SUI' atau 'bikinin pine script BTC') "
                "untuk analisa lengkap."
            )
        )
    else:
        lines.append("")
        checked_suffix = ""
        if is_multi:
            checked_lbl = ", ".join(
                format_trading_style_short(tf) for tf in report.scanned_timeframes
            )
            checked_suffix = f" (mode dicek: {checked_lbl})"
        lines.append(
            _e(
                "Belum ada koin dari daftar pantau yang berstatus READY saat ini "
                "(seluruhnya masih WAIT agar terhindar dari entry berisiko tinggi)"
                f"{checked_suffix}."
            )
        )
        reasons = [item.wait_reason for item in report.wait_setups if item.wait_reason]
        if reasons:
            top_reasons = Counter(reasons).most_common(3)
            lines.append(_b("Alasan Dominan Kondisi WAIT:"))
            for reason, count in top_reasons:
                lines.append(_e(f"• ({count} koin) {reason}"))

    if report.failed_symbols:
        lines.append("")
        err_items = [
            f"{sym} [{report.symbol_errors.get(sym, 'ERROR')}]" for sym in report.failed_symbols
        ]
        lines.append(_e(f"Gagal/ERROR ({len(report.failed_symbols)}): {', '.join(err_items)}"))

    lines.append("")
    lines.append(_e(f"Disclaimer: {report.disclaimer}"))
    return "\n".join(lines)
