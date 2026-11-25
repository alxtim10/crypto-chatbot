"""Top Gainers/Losers Volume Screener & 15m Momentum Continuation Engine."""

from __future__ import annotations

import asyncio
import html
import logging
from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Literal

from pydantic import BaseModel, ConfigDict

from crypto_chatbot.analysis.indicators import (
    MIN_HISTORY_EMA_20,
    compute_technical_metrics,
)
from crypto_chatbot.bot.templates import format_indonesian_number
from crypto_chatbot.market_data.base import FuturesTickerSnapshot
from crypto_chatbot.schemas.analysis import SetupStatus
from crypto_chatbot.schemas.market import MarketCandle, Timeframe, UtcDatetime

if TYPE_CHECKING:
    from crypto_chatbot.analysis.service import AnalysisService

logger = logging.getLogger(__name__)

DEFAULT_MIN_VOLUME_24H_USD = 5_000_000.0
DEFAULT_TOP_PER_SIDE = 8
MARUBOZU_MIN_BODY_RATIO = 0.72
MARUBOZU_MAX_OPP_WICK_RATIO = 0.15
MARUBOZU_MAX_LEAD_WICK_RATIO = 0.22
MIN_MOMENTUM_BODY_ATR_RATIO = 0.55
SL_BUFFER_ATR_RATIO = 0.15
SL_MIN_BUFFER_PCT = 0.001

RISK_MANAGEMENT_EDUCATIONAL_NOTE = (
    "Referensi Risk: Konservatif 1% per trade (tahan 100x loss beruntun). "
    "Mode agresif max 3%-5% per trade agar tetap bertahan 20-30x trading; "
    "hindari risiko >=20% per trade (5x loss habis)."
)

SCREENER_DISCLAIMER = (
    "Screening momentum 15m bersifat deterministik berdasarkan data historis & volume 24j; "
    "bukan saran transaksi, ajakan beli/jual, atau jaminan keuntungan."
)


class MomentumPattern(StrEnum):
    BULLISH_ENGULFING = "bullish_engulfing"
    BULLISH_MARUBOZU = "bullish_marubozu"
    BEARISH_ENGULFING = "bearish_engulfing"
    BEARISH_MARUBOZU = "bearish_marubozu"
    NONE = "none"


class MomentumSetupResult(BaseModel):
    """Single coin evaluation under the 15m Momentum Continuation rules."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    venue: str
    side: Literal["long", "short"]
    status: SetupStatus
    pattern: MomentumPattern
    change_24h_pct: float
    quote_volume_24h_usd: float
    ema_20: float | None = None
    ema_50: float | None = None
    trend_aligned: bool
    entry_price: float | None = None
    stop_loss: float | None = None
    sl_anchor_note: str | None = None
    tp1_1r: float | None = None
    tp2_2r: float | None = None
    tp3_3r: float | None = None
    sl_distance_pct: float | None = None
    wait_reason: str | None = None
    as_of: UtcDatetime


class MomentumScreenerReport(BaseModel):
    """Full Top Gainers/Losers screener output filtered by volume >= $5M and 15m momentum."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    venue: str
    side_filter: Literal["all", "long", "short"]
    min_volume_usd: float
    total_pairs_seen: int
    liquid_pairs_count: int
    top_gainers: tuple[FuturesTickerSnapshot, ...]
    top_losers: tuple[FuturesTickerSnapshot, ...]
    long_signals: tuple[MomentumSetupResult, ...]
    short_signals: tuple[MomentumSetupResult, ...]
    watching_items: tuple[MomentumSetupResult, ...]
    failed_symbols: tuple[str, ...]
    as_of: UtcDatetime
    risk_management_note: str
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


def _fmt_usd_millions(value: float) -> str:
    if value >= 1_000_000_000.0:
        return f"${format_indonesian_number(value / 1_000_000_000.0, 2)}B"
    return f"${format_indonesian_number(value / 1_000_000.0, 1)}M"


def _pattern_label(pattern: MomentumPattern) -> str:
    mapping = {
        MomentumPattern.BULLISH_ENGULFING: "Bullish Engulfing (15m)",
        MomentumPattern.BULLISH_MARUBOZU: "Bullish Marubozu (15m)",
        MomentumPattern.BEARISH_ENGULFING: "Bearish Engulfing (15m)",
        MomentumPattern.BEARISH_MARUBOZU: "Bearish Marubozu (15m)",
        MomentumPattern.NONE: "Belum ada Engulfing/Marubozu",
    }
    return mapping[pattern]


def evaluate_15m_momentum_setup(
    candles: Sequence[MarketCandle],
    *,
    side: Literal["long", "short"],
    change_24h_pct: float,
    quote_volume_24h_usd: float,
) -> MomentumSetupResult:
    """Evaluate closed 15m candles for Engulfing/Marubozu continuation aligned with EMA trend."""
    closed = sorted((c for c in candles if c.is_closed), key=lambda c: c.open_time)
    if not closed:
        raise ValueError("candles must contain at least one closed candle")

    latest = closed[-1]
    symbol = latest.canonical_symbol
    venue = latest.venue.upper()
    as_of = latest.close_time

    def _wait(
        wait_reason: str,
        *,
        pattern: MomentumPattern = MomentumPattern.NONE,
        ema_20: float | None = None,
        ema_50: float | None = None,
        trend_aligned: bool = False,
    ) -> MomentumSetupResult:
        return MomentumSetupResult(
            symbol=symbol,
            venue=venue,
            side=side,
            status=SetupStatus.WAIT,
            pattern=pattern,
            change_24h_pct=change_24h_pct,
            quote_volume_24h_usd=quote_volume_24h_usd,
            ema_20=ema_20,
            ema_50=ema_50,
            trend_aligned=trend_aligned,
            wait_reason=wait_reason,
            as_of=as_of,
        )

    def _ready(
        *,
        pattern: MomentumPattern,
        ema_20: float | None,
        ema_50: float | None,
        entry_price: float,
        stop_loss: float,
        sl_anchor_note: str | None,
        tp1: float,
        tp2: float,
        tp3: float,
        risk: float,
    ) -> MomentumSetupResult:
        return MomentumSetupResult(
            symbol=symbol,
            venue=venue,
            side=side,
            status=SetupStatus.READY,
            pattern=pattern,
            change_24h_pct=change_24h_pct,
            quote_volume_24h_usd=quote_volume_24h_usd,
            ema_20=ema_20,
            ema_50=ema_50,
            trend_aligned=True,
            entry_price=entry_price,
            stop_loss=stop_loss,
            sl_anchor_note=sl_anchor_note,
            tp1_1r=tp1,
            tp2_2r=tp2,
            tp3_3r=tp3,
            sl_distance_pct=round((risk / entry_price) * 100.0, 2),
            as_of=as_of,
        )

    if len(closed) < MIN_HISTORY_EMA_20:
        return _wait(
            f"History 15m belum mencukupi untuk EMA 20 "
            f"({len(closed)}/{MIN_HISTORY_EMA_20} candle tertutup)."
        )

    ind = compute_technical_metrics(closed)
    ema_20 = ind.metrics.ema_20
    ema_50 = ind.metrics.ema_50
    atr_14 = ind.metrics.atr_14 or max(latest.high - latest.low, latest.close * 0.002)

    prev = closed[-2]
    curr = closed[-1]
    candle_range = curr.high - curr.low
    body = abs(curr.close - curr.open)
    upper_wick = curr.high - max(curr.open, curr.close)
    lower_wick = min(curr.open, curr.close) - curr.low
    min_body = MIN_MOMENTUM_BODY_ATR_RATIO * atr_14

    if side == "long":
        trend_aligned = (
            ema_20 is not None and curr.close > ema_20 and (ema_50 is None or ema_20 >= ema_50)
        )
        is_engulfing = (
            prev.close < prev.open
            and curr.close > curr.open
            and curr.open <= prev.close * 1.001
            and curr.close > prev.open
            and body >= min_body
        )
        is_marubozu = (
            curr.close > curr.open
            and candle_range > 0.0
            and (body / candle_range) >= MARUBOZU_MIN_BODY_RATIO
            and (lower_wick / candle_range) <= MARUBOZU_MAX_OPP_WICK_RATIO
            and (upper_wick / candle_range) <= MARUBOZU_MAX_LEAD_WICK_RATIO
            and body >= min_body
        )

        pattern = MomentumPattern.NONE
        sl_anchor: float | None = None
        sl_note: str | None = None
        if is_engulfing:
            pattern = MomentumPattern.BULLISH_ENGULFING
            sl_anchor = min(prev.low, curr.low)
            sl_note = "Di bawah low terendah 2 candle engulfing + buffer tipis"
        elif is_marubozu:
            pattern = MomentumPattern.BULLISH_MARUBOZU
            sl_anchor = curr.low
            sl_note = "Di bawah low candle marubozu 15m + buffer tipis"

        if pattern is MomentumPattern.NONE or sl_anchor is None:
            return _wait(
                "Candle 15m terakhir belum membentuk Bullish Engulfing/Marubozu.",
                ema_20=ema_20,
                ema_50=ema_50,
                trend_aligned=trend_aligned,
            )
        if not trend_aligned:
            return _wait(
                f"{_pattern_label(pattern)} terbentuk, namun harga belum di atas tren "
                "EMA 20 >= EMA 50 (15m).",
                pattern=pattern,
                ema_20=ema_20,
                ema_50=ema_50,
                trend_aligned=False,
            )

        buffer = max(SL_BUFFER_ATR_RATIO * atr_14, SL_MIN_BUFFER_PCT * sl_anchor)
        stop_loss = sl_anchor - buffer
        entry_price = curr.close
        risk = entry_price - stop_loss
        if stop_loss <= 0.0 or risk <= 0.0:
            return _wait(
                "Jarak stop loss tidak valid.",
                pattern=pattern,
                ema_20=ema_20,
                ema_50=ema_50,
                trend_aligned=True,
            )

        return _ready(
            pattern=pattern,
            ema_20=ema_20,
            ema_50=ema_50,
            entry_price=entry_price,
            stop_loss=stop_loss,
            sl_anchor_note=sl_note,
            tp1=entry_price + 1.0 * risk,
            tp2=entry_price + 2.0 * risk,
            tp3=entry_price + 3.0 * risk,
            risk=risk,
        )

    # SHORT (from Top Losers)
    trend_aligned = (
        ema_20 is not None and curr.close < ema_20 and (ema_50 is None or ema_20 <= ema_50)
    )
    is_engulfing = (
        prev.close > prev.open
        and curr.close < curr.open
        and curr.open >= prev.close * 0.999
        and curr.close < prev.open
        and body >= min_body
    )
    is_marubozu = (
        curr.close < curr.open
        and candle_range > 0.0
        and (body / candle_range) >= MARUBOZU_MIN_BODY_RATIO
        and (upper_wick / candle_range) <= MARUBOZU_MAX_OPP_WICK_RATIO
        and (lower_wick / candle_range) <= MARUBOZU_MAX_LEAD_WICK_RATIO
        and body >= min_body
    )

    pattern = MomentumPattern.NONE
    sl_anchor = None
    sl_note = None
    if is_engulfing:
        pattern = MomentumPattern.BEARISH_ENGULFING
        sl_anchor = max(prev.high, curr.high)
        sl_note = "Di atas high tertinggi 2 candle engulfing + buffer tipis"
    elif is_marubozu:
        pattern = MomentumPattern.BEARISH_MARUBOZU
        sl_anchor = curr.high
        sl_note = "Di atas high candle marubozu 15m + buffer tipis"

    if pattern is MomentumPattern.NONE or sl_anchor is None:
        return _wait(
            "Candle 15m terakhir belum membentuk Bearish Engulfing/Marubozu.",
            ema_20=ema_20,
            ema_50=ema_50,
            trend_aligned=trend_aligned,
        )
    if not trend_aligned:
        return _wait(
            f"{_pattern_label(pattern)} terbentuk, namun harga belum di bawah tren "
            "EMA 20 <= EMA 50 (15m).",
            pattern=pattern,
            ema_20=ema_20,
            ema_50=ema_50,
            trend_aligned=False,
        )

    buffer = max(SL_BUFFER_ATR_RATIO * atr_14, SL_MIN_BUFFER_PCT * sl_anchor)
    stop_loss = sl_anchor + buffer
    entry_price = curr.close
    risk = stop_loss - entry_price
    if risk <= 0.0:
        return _wait(
            "Jarak stop loss tidak valid.",
            pattern=pattern,
            ema_20=ema_20,
            ema_50=ema_50,
            trend_aligned=True,
        )

    tp1 = entry_price - 1.0 * risk
    tp2 = entry_price - 2.0 * risk
    tp3 = entry_price - 3.0 * risk
    if tp1 <= 0.0 or tp2 <= 0.0:
        return _wait(
            "Target 2R melampaui batas harga positif.",
            pattern=pattern,
            ema_20=ema_20,
            ema_50=ema_50,
            trend_aligned=True,
        )

    return _ready(
        pattern=pattern,
        ema_20=ema_20,
        ema_50=ema_50,
        entry_price=entry_price,
        stop_loss=stop_loss,
        sl_anchor_note=sl_note,
        tp1=tp1,
        tp2=tp2,
        tp3=max(1e-8, tp3),
        risk=risk,
    )


async def screen_momentum_movers(
    analysis_service: AnalysisService,
    *,
    side: Literal["all", "long", "short"] = "all",
    min_volume_usd: float = DEFAULT_MIN_VOLUME_24H_USD,
    top_per_side: int = DEFAULT_TOP_PER_SIDE,
    max_concurrency: int = 4,
) -> MomentumScreenerReport:
    """Screen exchange futures Top Gainers/Losers with vol >= $5M for 15m momentum setups."""
    ticker_batch = await analysis_service.get_futures_tickers()
    if ticker_batch.has_provider_error or not ticker_batch.tickers:
        msg = ticker_batch.error_message or "Gagal mengambil daftar ticker futures."
        raise ValueError(msg)

    liquid = [t for t in ticker_batch.tickers if t.quote_volume_24h_usd >= min_volume_usd]

    gainers: list[FuturesTickerSnapshot] = []
    losers: list[FuturesTickerSnapshot] = []
    if side in ("all", "long"):
        gainers = sorted(
            (t for t in liquid if t.change_24h_pct > 0.0),
            key=lambda t: t.change_24h_pct,
            reverse=True,
        )[:top_per_side]
    if side in ("all", "short"):
        losers = sorted(
            (t for t in liquid if t.change_24h_pct < 0.0),
            key=lambda t: t.change_24h_pct,
        )[:top_per_side]

    tasks_spec: list[tuple[FuturesTickerSnapshot, Literal["long", "short"]]] = [
        *((t, "long") for t in gainers),
        *((t, "short") for t in losers),
    ]

    sem = asyncio.Semaphore(max(1, max_concurrency))
    long_ready: list[MomentumSetupResult] = []
    short_ready: list[MomentumSetupResult] = []
    watching: list[MomentumSetupResult] = []
    failed: list[str] = []

    async def _eval_one(
        snap: FuturesTickerSnapshot,
        target_side: Literal["long", "short"],
    ) -> None:
        async with sem:
            try:
                batch = await analysis_service.get_closed_ohlcv(
                    snap.canonical_symbol,
                    Timeframe.M15,
                    limit=200,
                )
            except Exception as exc:
                logger.warning("Screener 15m OHLCV failed for %s: %s", snap.canonical_symbol, exc)
                failed.append(snap.canonical_symbol)
                return

            if batch.has_provider_error or not batch.candles:
                failed.append(snap.canonical_symbol)
                return

            result = evaluate_15m_momentum_setup(
                batch.candles,
                side=target_side,
                change_24h_pct=snap.change_24h_pct,
                quote_volume_24h_usd=snap.quote_volume_24h_usd,
            )
            if result.status is SetupStatus.READY:
                if target_side == "long":
                    long_ready.append(result)
                else:
                    short_ready.append(result)
            else:
                watching.append(result)

    if tasks_spec:
        await asyncio.gather(*(_eval_one(t, s) for t, s in tasks_spec))

    long_ready.sort(key=lambda item: item.change_24h_pct, reverse=True)
    short_ready.sort(key=lambda item: item.change_24h_pct)
    watching.sort(key=lambda item: abs(item.change_24h_pct), reverse=True)

    return MomentumScreenerReport(
        venue=ticker_batch.venue.upper(),
        side_filter=side,
        min_volume_usd=min_volume_usd,
        total_pairs_seen=len(ticker_batch.tickers),
        liquid_pairs_count=len(liquid),
        top_gainers=tuple(gainers),
        top_losers=tuple(losers),
        long_signals=tuple(long_ready),
        short_signals=tuple(short_ready),
        watching_items=tuple(watching),
        failed_symbols=tuple(failed),
        as_of=ticker_batch.received_time,
        risk_management_note=RISK_MANAGEMENT_EDUCATIONAL_NOTE,
        disclaimer=SCREENER_DISCLAIMER,
    )


def render_momentum_screener_report(
    report: MomentumScreenerReport,
    *,
    html_mode: bool = True,
) -> str:
    """Render Top Gainers/Losers + 15m Momentum screener report in Indonesian."""

    def _b(text: str) -> str:
        esc = html.escape(text, quote=False) if html_mode else text
        return f"<b>{esc}</b>" if html_mode else esc

    def _e(text: str) -> str:
        return html.escape(text, quote=False) if html_mode else text

    min_vol_str = _fmt_usd_millions(report.min_volume_usd)
    side_label = {
        "all": "Top Gainers & Top Losers",
        "long": "Top Gainers (LONG)",
        "short": "Top Losers (SHORT)",
    }[report.side_filter]

    lines: list[str] = [
        _b(f"Screener Momentum 15m Futures · {report.venue} ({side_label})"),
        _e(
            f"Filter Vol 24j >= {min_vol_str} | "
            f"Pasangan Likuid: {report.liquid_pairs_count}/{report.total_pairs_seen} | "
            f"Sinyal READY: LONG ({len(report.long_signals)}), SHORT ({len(report.short_signals)})"
        ),
    ]

    if report.side_filter in ("all", "long"):
        lines.append("")
        lines.append(_b("Watchlist Top Gainers (Kandidat LONG):"))
        if report.top_gainers:
            g_str = ", ".join(
                f"{t.canonical_symbol.split('/')[0]} "
                f"({format_indonesian_number(t.change_24h_pct, 2, signed=True)}% | "
                f"Vol {_fmt_usd_millions(t.quote_volume_24h_usd)})"
                for t in report.top_gainers[:5]
            )
            lines.append(_e(f"• {g_str}"))
        else:
            lines.append(_e("• Tidak ada top gainers dengan volume di atas batas minimum."))

    if report.side_filter in ("all", "short"):
        lines.append("")
        lines.append(_b("Watchlist Top Losers (Kandidat SHORT):"))
        if report.top_losers:
            l_str = ", ".join(
                f"{t.canonical_symbol.split('/')[0]} "
                f"({format_indonesian_number(t.change_24h_pct, 2, signed=True)}% | "
                f"Vol {_fmt_usd_millions(t.quote_volume_24h_usd)})"
                for t in report.top_losers[:5]
            )
            lines.append(_e(f"• {l_str}"))
        else:
            lines.append(_e("• Tidak ada top losers dengan volume di atas batas minimum."))

    def _append_signals(title: str, items: Sequence[MomentumSetupResult]) -> None:
        if not items:
            return
        lines.append("")
        lines.append(_b(title))
        for idx, item in enumerate(items, start=1):
            chg = format_indonesian_number(item.change_24h_pct, 2, signed=True)
            vol_s = _fmt_usd_millions(item.quote_volume_24h_usd)
            ent = _fmt_price(item.entry_price) if item.entry_price is not None else "-"
            sl = _fmt_price(item.stop_loss) if item.stop_loss is not None else "-"
            tp1 = _fmt_price(item.tp1_1r) if item.tp1_1r is not None else "-"
            tp2 = _fmt_price(item.tp2_2r) if item.tp2_2r is not None else "-"
            tp3 = _fmt_price(item.tp3_3r) if item.tp3_3r is not None else "-"
            sl_pct = (
                f"{format_indonesian_number(item.sl_distance_pct, 2)}%"
                if item.sl_distance_pct is not None
                else "-"
            )
            lines.append("")
            lines.append(
                f"{idx}. {_b(item.symbol)} ({_e(chg)}% | Vol {_e(vol_s)}) · "
                f"{_b(item.side.upper())} [{_e(_pattern_label(item.pattern))}]"
            )
            lines.append(_e("   • Gaya Setup: SCALP / Quick Intraday (15m Momentum Continuation)"))
            lines.append(
                _e("   • Gaya Setup: SCALP / Quick Intraday (15m Momentum Continuation)")
            )
            lines.append(_e(f"   • Entry Close 15m: {ent} | Tren EMA20/50 Selaras"))
            lines.append(_e(f"   • Stop Loss: {sl} ({sl_pct}) - {item.sl_anchor_note or ''}"))
            lines.append(
                _e(f"   • TP1 (1R): {tp1} | TP2 (2R Target Standar): {tp2} | TP3 (3R): {tp3}")
            )

    _append_signals("Sinyal Momentum 15m READY — LONG (Top Gainers):", report.long_signals)
    _append_signals("Sinyal Momentum 15m READY — SHORT (Top Losers):", report.short_signals)

    if not report.long_signals and not report.short_signals:
        lines.append("")
        lines.append(
            _e(
                "Belum ada koin dari Top Gainers/Losers likuid yang membentuk candle momentum "
                "Engulfing/Marubozu 15m terkonfirmasi selaras tren EMA saat ini."
            )
        )
        if report.watching_items:
            lines.append(_b("Pantauan Watchlist Terdekat (WAIT):"))
            for item in report.watching_items[:5]:
                chg = format_indonesian_number(item.change_24h_pct, 2, signed=True)
                lines.append(
                    _e(f"• {item.symbol} ({item.side.upper()}, {chg}%): {item.wait_reason}")
                )

    lines.append("")
    lines.append(_e(report.risk_management_note))
    lines.append(_e(f"Disclaimer: {report.disclaimer}"))
    return "\n".join(lines)
