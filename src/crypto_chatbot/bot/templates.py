"""Deterministic Indonesian formatting and Telegram HTML rendering."""

from __future__ import annotations

import html
import logging
import re
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta, timezone
from enum import Enum
from typing import Literal

from crypto_chatbot.schemas.analysis import (
    AnalysisSnapshot,
    DealingZone,
    Fact,
    FactorDirection,
    MarketStructure,
    MtfRsiAlignment,
    Regime,
    SetupBias,
    SetupStatus,
)
from crypto_chatbot.schemas.market import (
    QualityStatus,
    Timeframe,
    format_trading_style_label,
)
from crypto_chatbot.schemas.news import SentimentLabel
from crypto_chatbot.storage.repositories import (
    ForgetResult,
    StoredMessage,
    UserPreferences,
)

logger = logging.getLogger(__name__)

WIB = timezone(timedelta(hours=7))
TELEGRAM_MAX_MESSAGE_LENGTH = 4096

REGIME_LABELS: dict[Regime, str] = {
    Regime.HIGH_VOLATILITY: "Volatilitas Tinggi",
    Regime.TRENDING_UP: "Tren Naik",
    Regime.TRENDING_DOWN: "Tren Turun",
    Regime.RANGING: "Mendatar (Ranging)",
    Regime.TRANSITIONAL: "Transisi",
}
_REGIME_LABELS = REGIME_LABELS

STRUCTURE_LABELS: dict[MarketStructure, str] = {
    MarketStructure.BULLISH: "Bullish (HH + HL terkonfirmasi)",
    MarketStructure.BEARISH: "Bearish (LH + LL terkonfirmasi)",
    MarketStructure.MIXED: "Bercampur (Mixed)",
}
_STRUCTURE_LABELS = STRUCTURE_LABELS

MTF_LABELS: dict[MtfRsiAlignment, str] = {
    MtfRsiAlignment.ALIGNED_BULLISH: "Selaras di atas 50 (1h, 4h, 1d)",
    MtfRsiAlignment.ALIGNED_BEARISH: "Selaras di bawah 50 (1h, 4h, 1d)",
    MtfRsiAlignment.MIXED: "Tercampur antar timeframe",
}
_MTF_LABELS = MTF_LABELS

DEALING_ZONE_LABELS: dict[DealingZone, str] = {
    DealingZone.BELOW_RANGE: "Below Range",
    DealingZone.DISCOUNT: "Discount",
    DealingZone.EQUILIBRIUM: "Equilibrium",
    DealingZone.PREMIUM: "Premium",
    DealingZone.ABOVE_RANGE: "Above Range",
}

SENTIMENT_LABELS: dict[SentimentLabel, str] = {
    SentimentLabel.POSITIVE: "Positif",
    SentimentLabel.NEGATIVE: "Negatif",
    SentimentLabel.NEUTRAL: "Netral",
}

SETUP_STATUS_LABELS: dict[SetupStatus, str] = {
    SetupStatus.READY: "READY",
    SetupStatus.WAIT: "WAIT / NO_SETUP",
}

SETUP_BIAS_LABELS: dict[SetupBias, str] = {
    SetupBias.LONG: "LONG (Pullback)",
    SetupBias.SHORT: "SHORT (Reli Pullback)",
    SetupBias.NEUTRAL: "NEUTRAL",
}


def format_enum_label[E: Enum](mapping: Mapping[E, str], value: E) -> str:
    """Resolve an enum display label, falling back cleanly to `value.value` with a warning."""
    label = mapping.get(value)
    if label is not None:
        return label
    logger.warning(
        "Missing display label mapping for %s.%s; falling back to %r",
        type(value).__name__,
        value.name,
        value.value,
    )
    return str(value.value)


def format_timestamp_utc_wib(dt: datetime) -> str:
    """Format a timezone-aware datetime showing both UTC and WIB."""
    utc_dt = dt.astimezone(UTC)
    wib_dt = utc_dt.astimezone(WIB)
    return f"{utc_dt.strftime('%Y-%m-%d %H:%M UTC')} ({wib_dt.strftime('%H:%M WIB')})"


def format_indonesian_number(value: float, decimals: int, *, signed: bool = False) -> str:
    """Format a float with `.` thousands separator and `,` decimal separator."""
    if decimals < 0:
        raise ValueError("decimals must be >= 0")
    sign = ""
    if value < 0:
        sign = "-"
    elif signed and value > 0:
        sign = "+"

    abs_str = f"{abs(value):,.{decimals}f}"
    int_part, _, frac_part = abs_str.partition(".")
    int_id = int_part.replace(",", ".")
    if decimals == 0:
        return f"{sign}{int_id}"
    return f"{sign}{int_id},{frac_part}"


def format_fact_value(fact: Fact) -> str:
    """Render a `Fact` value deterministically according to its `display_format`."""
    fmt = fact.display_format
    val = fact.value
    if fmt == "price":
        abs_v = abs(val)
        if abs_v == 0.0 or abs_v >= 1.0:
            decimals = 2
        elif abs_v >= 0.01:
            decimals = 4
        elif abs_v >= 0.0001:
            decimals = 6
        else:
            decimals = 8
        return f"{format_indonesian_number(val, decimals)} {fact.unit}"
    if fmt == "signed_percent":
        return f"{format_indonesian_number(val, 2, signed=True)}%"
    if fmt == "funding_percent":
        return f"{format_indonesian_number(val, 4, signed=True)}%"
    if fmt == "percent":
        return f"{format_indonesian_number(val, 2)}%"
    if fmt == "bps":
        return f"{format_indonesian_number(val, 2)} bps"
    if fmt == "signed_bps":
        return f"{format_indonesian_number(val, 2, signed=True)} bps"
    if fmt in {"notional_quote", "quantity"}:
        return f"{format_indonesian_number(val, 2)} {fact.unit}"
    if fmt == "signed_decimal_2":
        return format_indonesian_number(val, 2, signed=True)
    if fmt == "rr_ratio":
        return f"1:{format_indonesian_number(val, 2)}"
    if fmt == "decimal_1":
        return format_indonesian_number(val, 1)
    if fmt == "integer":
        return format_indonesian_number(val, 0)
    if fmt == "ratio_2":
        return f"{format_indonesian_number(val, 2)}x"
    return format_indonesian_number(val, 2)


def _esc(text: str, html_mode: bool) -> str:
    return html.escape(text, quote=False) if html_mode else text


def _bold(text: str, html_mode: bool) -> str:
    escaped = _esc(text, html_mode)
    return f"<b>{escaped}</b>" if html_mode else escaped


def _append_sentiment_lines(
    lines: list[str],
    snapshot: AnalysisSnapshot,
    *,
    html_mode: bool,
    max_sources: int = 3,
) -> None:
    sent = snapshot.sentiment
    if sent is None:
        return
    dom = format_enum_label(SENTIMENT_LABELS, sent.dominant_label)
    score_fact = next(
        (f for f in snapshot.facts if f.label == "Skor sentimen berita 24h (bukan probabilitas)"),
        None,
    )
    conf_fact = next(
        (f for f in snapshot.facts if f.label == "Confidence sentimen berita 24h"),
        None,
    )
    score_tok = _fmt_fact(score_fact) or format_indonesian_number(
        sent.net_sentiment_score, 2, signed=True
    )
    conf_tok = _fmt_fact(conf_fact) or (
        f"{format_indonesian_number(sent.mean_confidence * 100.0, 1)}%"
    )
    lines.append("")
    lines.append(
        _bold(
            f"Sentimen Berita {sent.window_hours}h (Bukan Probabilitas — {sent.model_version})",
            html_mode,
        )
    )
    lines.append(
        _esc(
            f"- Ringkasan: {dom} | skor: {score_tok} | confidence: {conf_tok} "
            f"| artikel unik: {sent.article_count} "
            f"(+{sent.positive_count}/-{sent.negative_count}/={sent.neutral_count})",
            html_mode,
        )
    )
    for src in sent.sources[:max_sources]:
        pub_esc = _esc(src.publisher, html_mode)
        title_esc = _esc(src.title, html_mode)
        if html_mode:
            href = html.escape(src.canonical_url, quote=True)
            lines.append(
                f'- Sumber: <a href="{href}">{pub_esc}</a> — {title_esc} '
                f"({src.sentiment_label.value})"
            )
        else:
            lines.append(f"- Sumber: {src.publisher} — {src.title} ({src.canonical_url})")


def _append_ml_prediction_lines(
    lines: list[str],
    snapshot: AnalysisSnapshot,
    *,
    html_mode: bool,
) -> None:
    pred = snapshot.ml_prediction
    if pred is None:
        return
    lines.append("")
    lines.append(_bold("Estimasi Model ML (Eksperimental — Terpisah dari Fakta)", html_mode))
    lines.append(
        _esc(
            f"- Horizon & target: {pred.horizon_hours}h ({pred.target_name})",
            html_mode,
        )
    )
    if pred.abstained or pred.calibrated_probability is None:
        reason = pred.abstention_reason or "insufficient_confidence"
        lines.append(_esc(f"- Status probabilitas: ABSTAIN ({reason})", html_mode))
    else:
        pct = format_indonesian_number(pred.calibrated_probability * 100.0, 1)
        lines.append(_esc(f"- Probabilitas terkalibrasi: {pct}%", html_mode))
    lines.append(
        _esc(
            f"- Confidence & kualitas data: confidence={pred.confidence_label} "
            f"| kualitas={pred.data_quality_status.value}",
            html_mode,
        )
    )
    lines.append(
        _esc(
            f"- Versi model & fitur: {pred.model_version} / {pred.feature_version}",
            html_mode,
        )
    )
    for note in pred.limitations:
        lines.append(f"- Keterbatasan model: {_esc(note, html_mode)}")
    lines.append(f"- Peringatan: {_esc(pred.non_guarantee_notice, html_mode)}")


def _append_positioning_lines(
    lines: list[str],
    snapshot: AnalysisSnapshot,
    *,
    html_mode: bool,
) -> None:
    pos = snapshot.positioning
    if pos is None:
        return
    lines.append("")
    lines.append(
        _bold(
            f"Panel Positioning Derivatif ({pos.parity_mode} — {pos.venue.upper()})",
            html_mode,
        )
    )
    if pos.status == "UNAVAILABLE":
        lines.append(
            _esc(
                f"- Status: UNAVAILABLE ({pos.note or 'belum ada histori contract_stats'})",
                html_mode,
            )
        )
        return
    parts: list[str] = [f"Status: {pos.status}"]
    if pos.top_lsr_account is not None:
        top_txt = f"Top L/S: {format_indonesian_number(pos.top_lsr_account, 2)}"
        if pos.top_lsr_account_pct_30d is not None and pos.top_lsr_account_z_30d is not None:
            pct = format_indonesian_number(pos.top_lsr_account_pct_30d, 1)
            z = format_indonesian_number(pos.top_lsr_account_z_30d, 2, signed=True)
            top_txt += f" (P{pct} / z={z})"
        parts.append(top_txt)
    if pos.lsr_taker is not None:
        tkr_txt = f"Taker L/S: {format_indonesian_number(pos.lsr_taker, 2)}"
        if pos.lsr_taker_pct_30d is not None and pos.lsr_taker_z_30d is not None:
            pct = format_indonesian_number(pos.lsr_taker_pct_30d, 1)
            z = format_indonesian_number(pos.lsr_taker_z_30d, 2, signed=True)
            tkr_txt += f" (P{pct} / z={z})"
        parts.append(tkr_txt)
    if pos.oi_24h_delta_pct is not None:
        parts.append(f"OI 24h: {format_indonesian_number(pos.oi_24h_delta_pct, 2, signed=True)}%")
    if pos.liq_skew_24h is not None:
        parts.append(f"Liq skew: {format_indonesian_number(pos.liq_skew_24h, 2, signed=True)}")
    lines.append(_esc("- " + " | ".join(parts), html_mode))
    if pos.note:
        lines.append(_esc(f"- Catatan: {pos.note}", html_mode))


def _append_composite_sentiment_panel(
    lines: list[str],
    snapshot: AnalysisSnapshot,
    *,
    html_mode: bool,
) -> None:
    """Render unified S-04 Sentimen panel with BACKTESTED / CONTEXT_ONLY / FORWARD_ONLY tags."""
    has_funding = snapshot.derivatives is not None and snapshot.derivatives.funding_rate is not None
    has_pos = snapshot.positioning is not None
    has_fg = snapshot.fear_greed is not None
    has_news = snapshot.sentiment is not None
    if not (has_funding or has_pos or has_fg or has_news):
        return

    lines.append("")
    lines.append(_bold("Panel Sentimen (Komposit Multi-Sumber — Bukan Probabilitas)", html_mode))

    diag = snapshot.sentiment_diagnostics

    # 1. Positioning: BACKTESTED (funding 8h D-07) + FORWARD_ONLY (contract_stats S-01)
    if (
        has_funding
        and snapshot.derivatives is not None
        and snapshot.derivatives.funding_rate is not None
    ):
        fr_pct = format_indonesian_number(snapshot.derivatives.funding_rate * 100.0, 4, signed=True)
        lines.append(_esc(f"- Positioning Funding 8h [BACKTESTED]: {fr_pct}%", html_mode))
    if snapshot.positioning is not None:
        pos = snapshot.positioning
        pos_tag = pos.parity_mode
        if pos.status == "UNAVAILABLE":
            pos_reason = (
                pos.reason_code
                or (diag.positioning.reason if diag is not None else None)
                or "no_data"
            )
            lines.append(
                _esc(
                    f"- Positioning Derivatif [{pos_tag}]: UNAVAILABLE (reason={pos_reason})",
                    html_mode,
                )
            )
        else:
            if pos.status == "WARMING_UP":
                status_str = f"insufficient_history ({pos.observation_count} bars)"
            else:
                status_str = f"status={pos.status}"
            p_parts = [status_str]
            if pos.top_lsr_account is not None:
                p_parts.append(f"Top L/S={format_indonesian_number(pos.top_lsr_account, 2)}")
            if pos.lsr_taker is not None:
                p_parts.append(f"Taker L/S={format_indonesian_number(pos.lsr_taker, 2)}")
            if pos.oi_24h_delta_pct is not None:
                p_parts.append(
                    f"OI 24h={format_indonesian_number(pos.oi_24h_delta_pct, 2, signed=True)}%"
                )
            lines.append(
                _esc(f"- Positioning Derivatif [{pos_tag}]: " + " | ".join(p_parts), html_mode)
            )
    else:
        pos_reason = (diag.positioning.reason if diag is not None else None) or "not_wired"
        lines.append(
            _esc(
                f"- Positioning Derivatif [FORWARD_ONLY]: UNAVAILABLE (reason={pos_reason})",
                html_mode,
            )
        )

    # 2. Fear & Greed Index: CONTEXT_ONLY + Alternative.me attribution (S-02 / S-04)
    if snapshot.fear_greed is not None:
        fg = snapshot.fear_greed
        lines.append(
            _esc(
                f"- Fear & Greed [CONTEXT_ONLY]: {fg.value}/100 ({fg.classification}) — "
                "Sumber: Alternative.me Crypto Fear & Greed Index "
                "(konteks rezim saja, bukan pemicu entry)",
                html_mode,
            )
        )
    else:
        fg_reason = (diag.fear_greed.reason if diag is not None else None) or "not_wired"
        lines.append(
            _esc(f"- Fear & Greed [CONTEXT_ONLY]: UNAVAILABLE (reason={fg_reason})", html_mode)
        )

    # 3. News tone & count: FORWARD_ONLY (S-03 / S-04)
    if snapshot.sentiment is not None:
        sent = snapshot.sentiment
        dom = format_enum_label(SENTIMENT_LABELS, sent.dominant_label)
        sc = format_indonesian_number(sent.net_sentiment_score, 2, signed=True)
        lines.append(
            _esc(
                f"- Berita {sent.window_hours}h [FORWARD_ONLY]: {dom} | skor={sc} "
                f"| artikel={sent.article_count} "
                f"(+{sent.positive_count}/-{sent.negative_count}/={sent.neutral_count})",
                html_mode,
            )
        )
    else:
        news_reason = (diag.news.reason if diag is not None else None) or "not_wired"
        lines.append(
            _esc(f"- Berita 24h [FORWARD_ONLY]: UNAVAILABLE (reason={news_reason})", html_mode)
        )


def render_deterministic_snapshot(
    snapshot: AnalysisSnapshot,
    *,
    html_mode: bool = False,
) -> str:
    """Render an `AnalysisSnapshot` deterministically without any LLM involvement."""
    lines: list[str] = []
    header = (
        f"Analisis {snapshot.canonical_symbol} ({snapshot.timeframe.value}) - "
        f"{snapshot.venue.upper()} {snapshot.market_type.value.upper()}"
    )
    lines.append(_bold(header, html_mode))
    lines.append(
        f"Waktu candle tutup (as_of): {_esc(format_timestamp_utc_wib(snapshot.as_of), html_mode)}"
    )
    lines.append(
        f"Sumber: {_esc(snapshot.provider, html_mode)}:"
        f"{_esc(snapshot.provider_symbol, html_mode)} "
        f"| Snapshot: {_esc(snapshot.snapshot_id, html_mode)} "
        f"| Versi: {_esc(snapshot.calc_version, html_mode)}"
    )

    flags_text = (
        ", ".join(flag.value for flag in snapshot.quality_flags)
        if snapshot.quality_flags
        else "tidak ada"
    )
    lines.append(
        f"Status kualitas: {_esc(snapshot.quality_status.value, html_mode)} "
        f"(flag: {_esc(flags_text, html_mode)})"
    )

    if snapshot.quality_status is QualityStatus.INSUFFICIENT_DATA or snapshot.technical is None:
        lines.append("")
        lines.append(_bold("Ringkasan Status", html_mode))
        lines.append(
            _esc(
                "Data tidak mencukupi, tidak segar, atau penyedia bermasalah sehingga "
                "kondisi pasar terkini tidak dapat disajikan.",
                html_mode,
            )
        )
    else:
        lines.append("")
        lines.append(_bold("Fakta Terukur (Backend)", html_mode))
        for fact in snapshot.facts:
            lines.append(
                f"- [{_esc(fact.fact_id, html_mode)}] {_esc(fact.label, html_mode)}: "
                f"{_esc(format_fact_value(fact), html_mode)}"
            )

        lines.append("")
        lines.append(_bold("Kondisi & Struktur", html_mode))
        if snapshot.regime is not None:
            lines.append(
                f"- Rezim: {_esc(format_enum_label(REGIME_LABELS, snapshot.regime), html_mode)}"
            )
        if snapshot.market_structure is not None:
            struct_lbl = format_enum_label(STRUCTURE_LABELS, snapshot.market_structure)
            lines.append(f"- Struktur pivot: {_esc(struct_lbl, html_mode)}")
        if snapshot.technical.mtf_rsi_alignment is not None:
            mtf_label = _esc(
                format_enum_label(MTF_LABELS, snapshot.technical.mtf_rsi_alignment),
                html_mode,
            )
            lines.append(f"- Penyelarasan RSI MTF: {mtf_label}")

        if snapshot.events:
            lines.append("- Event terkonfirmasi terbaru:")
            fact_by_event = {
                (fact.as_of, fact.value): fact.fact_id
                for fact in snapshot.facts
                if fact.display_format == "price"
            }
            for event in snapshot.events:
                fid = fact_by_event.get((event.confirmation_time, event.level), "-")
                confirm_text = _esc(format_timestamp_utc_wib(event.confirmation_time), html_mode)
                lines.append(
                    f"  * [{_esc(fid, html_mode)}] {_esc(event.event_type.value, html_mode)} | "
                    f"konfirmasi: {confirm_text}"
                )

        if snapshot.derivatives is not None:
            deriv = snapshot.derivatives
            lines.append("")
            lines.append(_bold("Derivatif & Likuiditas (Venue Tunggal)", html_mode))
            if (
                deriv.funding_venue
                and deriv.funding_symbol
                and deriv.funding_rate_per_8h is not None
            ):
                lines.append(
                    f"- Sumber funding rate: {_esc(deriv.funding_venue.upper(), html_mode)} "
                    f"{_esc(deriv.funding_symbol, html_mode)}"
                )
            else:
                lines.append("- Sumber funding rate: tidak tersedia (nilai tidak direkayasa)")
            if (
                deriv.open_interest_venue
                and deriv.open_interest_symbol
                and deriv.open_interest_base is not None
            ):
                lines.append(
                    f"- Sumber open interest: {_esc(deriv.open_interest_venue.upper(), html_mode)} "
                    f"{_esc(deriv.open_interest_symbol, html_mode)}"
                )
            else:
                lines.append("- Sumber open interest: tidak tersedia (nilai tidak direkayasa)")
            if deriv.orderbook_venue and deriv.orderbook_symbol and deriv.mid_price is not None:
                ob_mkt = (
                    deriv.orderbook_market_type.value if deriv.orderbook_market_type else "spot"
                )
                resync_note = (
                    " (telah resinkronisasi setelah sequence gap)"
                    if deriv.orderbook_resync_count > 0
                    else ""
                )
                lines.append(
                    f"- Sumber order book: {_esc(deriv.orderbook_venue.upper(), html_mode)} "
                    f"{_esc(deriv.orderbook_symbol, html_mode)} ({_esc(ob_mkt, html_mode)})"
                    f"{_esc(resync_note, html_mode)}"
                )
            else:
                lines.append("- Sumber order book: tidak tersedia (nilai tidak direkayasa)")
            if deriv.liquidation_source_available:
                lines.append("- Status data likuidasi: tersedia dari sumber terkonfigurasi")
            else:
                lines.append(
                    "- Status data likuidasi: tidak tersedia pada endpoint REST publik "
                    "(tidak direkayasa)"
                )
            if deriv.cross_source_pair:
                lines.append(
                    f"- Pemeriksaan harga lintas sumber: {_esc(deriv.cross_source_pair, html_mode)}"
                )

        if snapshot.order_flow is not None and snapshot.order_flow.has_order_flow:
            of = snapshot.order_flow
            lines.append("")
            lines.append(_bold("Mikrostruktur Order Flow (Candle Tutup)", html_mode))
            tbr_str = (
                f"{format_indonesian_number(of.taker_buy_ratio * 100.0, 1)}%"
                if of.taker_buy_ratio is not None
                else "tidak tersedia"
            )
            cvd_str = (
                format_indonesian_number(of.cvd_20, 2, signed=True)
                if of.cvd_20 is not None
                else "tidak tersedia"
            )
            div_map = {
                "bullish": "Divergensi Bullish (harga melemah, delta menguat)",
                "bearish": "Divergensi Bearish (harga menguat, delta melemah)",
                "none": "Tidak ada divergensi",
            }
            div_str = div_map.get(of.cvd_divergence or "none", of.cvd_divergence or "none")
            lines.append(f"- Rasio taker buy: {_esc(tbr_str, html_mode)}")
            lines.append(f"- CVD (20 candle): {_esc(cvd_str, html_mode)}")
            lines.append(f"- Divergensi CVD: {_esc(div_str, html_mode)}")

        if snapshot.cross_venue_derivatives is not None:
            cv = snapshot.cross_venue_derivatives
            lines.append("")
            lines.append(_bold("Derivatif Lintas Venue (Multi-Venue)", html_mode))
            v_list = ", ".join(v.upper() for v in cv.venues)
            lines.append(
                f"- Venue teramati: {_esc(v_list, html_mode)} "
                f"(status: {_esc(cv.coverage_status, html_mode)})"
            )
            f_disp = (
                f"{format_indonesian_number(cv.funding_dispersion_bps, 2)} bps"
                if cv.funding_dispersion_bps is not None
                else "tidak tersedia"
            )
            p_spr = (
                f"{format_indonesian_number(cv.mark_price_spread_bps, 2)} bps"
                if cv.mark_price_spread_bps is not None
                else "tidak tersedia"
            )
            lines.append(f"- Dispersi funding: {_esc(f_disp, html_mode)}")
            lines.append(f"- Spread mark price: {_esc(p_spr, html_mode)}")

        if snapshot.macro_context is not None:
            mc = snapshot.macro_context
            lines.append("")
            lines.append(_bold("Konteks Makro & Derivatif (DXY & BTC Benchmark)", html_mode))
            lines.append(f"- Rezim posisi: {_esc(mc.positioning_regime.value.upper(), html_mode)}")
            if mc.dxy is not None:
                dxy = mc.dxy
                dxy_closed_tag = " [FX Weekend/Tutup]" if dxy.is_weekend_or_closed else ""
                corr_str = (
                    format_indonesian_number(dxy.rolling_corr_20, 2, signed=True)
                    if dxy.rolling_corr_20 is not None
                    else "tidak tersedia"
                )
                lines.append(
                    f"- US Dollar Index (DXY): {format_indonesian_number(dxy.last_close, 2)} "
                    f"| tren: {_esc(dxy.trend.upper(), html_mode)} "
                    f"| korelasi (20b): {_esc(corr_str, html_mode)} "
                    f"| dampak: {_esc(dxy.macro_impact.upper(), html_mode)}"
                    f"{_esc(dxy_closed_tag, html_mode)}"
                )
            if mc.btc_benchmark is not None and mc.btc_benchmark.leader_alignment != "self":
                btc = mc.btc_benchmark
                rs_str = (
                    f"{format_indonesian_number(btc.relative_strength_20_pct, 2, signed=True)}%"
                    if btc.relative_strength_20_pct is not None
                    else "tidak tersedia"
                )
                beta_str = (
                    format_indonesian_number(btc.rolling_beta_20, 2)
                    if btc.rolling_beta_20 is not None
                    else "tidak tersedia"
                )
                corr_btc_str = (
                    format_indonesian_number(btc.rolling_corr_20, 2, signed=True)
                    if btc.rolling_corr_20 is not None
                    else "tidak tersedia"
                )
                lines.append(
                    f"- Benchmark BTC: RS (20b): {_esc(rs_str, html_mode)} "
                    f"| beta: {_esc(beta_str, html_mode)} "
                    f"| korelasi: {_esc(corr_btc_str, html_mode)} "
                    f"| keselarasan: {_esc(btc.leader_alignment.upper(), html_mode)}"
                )

        if snapshot.tradingview is not None:
            tv = snapshot.tradingview
            lines.append("")
            lines.append(
                _bold(
                    "TradingView AI Agent Lens (ai-agent-lens/v1 — Bukan Probabilitas)", html_mode
                )
            )
            lines.append(
                f"- Sumber: TradingView {_esc(tv.provider_symbol, html_mode)} "
                f"({tv.timeframe.value}) | event: {_esc(tv.event_type, html_mode)} "
                f"| key: {_esc(tv.idempotency_key, html_mode)}"
            )
            if tv.rule_label or tv.rule_score is not None:
                lbl = _esc(tv.rule_label or "-", html_mode)
                lines.append(
                    f"- Label aturan: {lbl} (commit {_esc(tv.attribution_commit[:12], html_mode)})"
                )

        if snapshot.sentiment is not None:
            _append_sentiment_lines(lines, snapshot, html_mode=html_mode, max_sources=3)

        if snapshot.positioning is not None:
            _append_positioning_lines(lines, snapshot, html_mode=html_mode)

        _append_composite_sentiment_panel(lines, snapshot, html_mode=html_mode)

        if snapshot.prediction_markets is not None and snapshot.prediction_markets.observations:
            pm = snapshot.prediction_markets
            lines.append("")
            lines.append(
                _bold(
                    "Konteks Pasar Prediksi (Harga Kontrak Tersirat — "
                    "Bukan Probabilitas Menang Absolut)",
                    html_mode,
                )
            )
            for obs in pm.observations[:3]:
                pct = format_indonesian_number(obs.implied_yes_prob * 100.0, 1)
                vol_str = f"${format_indonesian_number(obs.volume_24h_usd, 0)}"
                q_text = _esc(obs.question, html_mode)
                v_name = obs.venue.upper()
                if html_mode and obs.url:
                    href = html.escape(obs.url, quote=True)
                    lines.append(
                        f'- <a href="{href}">{v_name}</a>: {q_text} → YES: {pct}% '
                        f"(vol 24h: {vol_str})"
                    )
                else:
                    lines.append(f"- [{v_name}] {q_text} → YES: {pct}% (vol 24h: {vol_str})")
            lines.append(
                _esc(
                    "Peringatan: Harga kontrak mencerminkan dinamika taruhan & likuiditas pasar, "
                    "bukan probabilitas kepastian statistik.",
                    html_mode,
                )
            )

        if snapshot.smc is not None or snapshot.trade_setup is not None:
            lines.append("")
            lines.append(
                _bold("SMC & Skenario Setup (Deterministik — Bukan Probabilitas)", html_mode)
            )
            if snapshot.smc is not None:
                smc = snapshot.smc
                if smc.latest_structure_break is not None:
                    brk_ev = smc.latest_structure_break.event_type.value
                    lines.append(f"- Break SMC terbaru: {_esc(brk_ev, html_mode)}")
                if smc.dealing_zone is not None:
                    dz_lbl = format_enum_label(DEALING_ZONE_LABELS, smc.dealing_zone)
                    lines.append(f"- Posisi Dealing Range: {_esc(dz_lbl, html_mode)}")
            if snapshot.trade_setup is not None:
                setup = snapshot.trade_setup
                status_lbl = format_enum_label(SETUP_STATUS_LABELS, setup.status)
                lines.append(
                    f"- Status setup: {_esc(status_lbl, html_mode)} "
                    f"(bias: {_esc(setup.bias.value.upper(), html_mode)})"
                )
                if setup.entry_timeframe is not None:
                    lines.append(
                        f"- Timeframe acuan entry: {_esc(setup.entry_timeframe.value, html_mode)}"
                    )
                if setup.ltf_confirmation:
                    lines.append(f"- Konfirmasi MTF/15m: {_esc(setup.ltf_confirmation, html_mode)}")
                if setup.wait_reason:
                    lines.append(f"- Alasan WAIT: {_esc(setup.wait_reason, html_mode)}")
                if setup.entry_source:
                    lines.append(f"- Sumber zona entry: {_esc(setup.entry_source, html_mode)}")
                if setup.stop_loss_distance_pct is not None:
                    sl_dist_str = format_indonesian_number(setup.stop_loss_distance_pct, 2)
                    lines.append(f"- Jarak Stop Loss: {sl_dist_str}%")
                if setup.max_theoretical_leverage is not None:
                    mmr_tag = " [MMR_ASSUMED]" if setup.mmr_assumed else ""
                    lines.append(
                        f"- Batas leverage teoretis referensi: {setup.max_theoretical_leverage}x"
                        f"{mmr_tag} (berdasarkan jarak SL + biaya + MMR)"
                    )
                if setup.net_rr_blend is not None:
                    blend_str = format_indonesian_number(setup.net_rr_blend, 2)
                    lines.append(f"- Net R:R Blend (40/40/20, report-only): 1:{blend_str}")

        if snapshot.factors:
            lines.append("")
            lines.append(_bold("Daftar Faktor Aturan (Bukan Probabilitas)", html_mode))
            groups = (
                (FactorDirection.SUPPORTS_BULLISH, "Mendukung naik"),
                (FactorDirection.SUPPORTS_BEARISH, "Mendukung turun"),
                (FactorDirection.NEUTRAL, "Netral / konteks"),
            )
            for direction, group_label in groups:
                items = [f for f in snapshot.factors if f.direction is direction]
                if not items:
                    continue
                lines.append(f"{_esc(group_label, html_mode)}:")
                for item in items:
                    refs = ", ".join(item.fact_ids)
                    lines.append(f"- {_esc(item.description, html_mode)} [{_esc(refs, html_mode)}]")

    _append_ml_prediction_lines(lines, snapshot, html_mode=html_mode)

    if snapshot.limitations:
        lines.append("")
        lines.append(_bold("Keterbatasan & Catatan", html_mode))
        for note in snapshot.limitations:
            lines.append(f"- {_esc(note, html_mode)}")

    lines.append("")
    lines.append(
        _esc(
            "Disclaimer: Informasi ini bersifat riset dan edukasi berdasarkan pengukuran historis, "
            "bukan saran transaksi, jaminan hasil, atau eksekusi order.",
            html_mode,
        )
    )
    return "\n".join(lines)


def _fmt_fact(fact: Fact | None) -> str | None:
    if fact is None:
        return None
    return f"{format_fact_value(fact)} [{fact.fact_id}]"


def render_compact_snapshot(
    snapshot: AnalysisSnapshot,
    *,
    html_mode: bool = False,
) -> str:
    """Render concise mobile-friendly key-points + Entry/SL/TP1-3 brief."""
    lines: list[str] = []
    header = (
        f"{snapshot.canonical_symbol} ({snapshot.timeframe.value}) · "
        f"{snapshot.venue.upper()} {snapshot.market_type.value}"
    )
    lines.append(_bold(header, html_mode))
    lines.append(f"Waktu (as_of): {_esc(format_timestamp_utc_wib(snapshot.as_of), html_mode)}")
    flags_text = (
        ", ".join(flag.value for flag in snapshot.quality_flags)
        if snapshot.quality_flags
        else "tidak ada"
    )
    prov = _esc(snapshot.provider, html_mode)
    psym = _esc(snapshot.provider_symbol, html_mode)
    lines.append(
        f"Status kualitas: {_esc(snapshot.quality_status.value, html_mode)} "
        f"(flag: {_esc(flags_text, html_mode)}) | "
        f"Sumber: {prov}:{psym} | "
        f"Snapshot: {_esc(snapshot.snapshot_id, html_mode)}"
    )

    if snapshot.quality_status is QualityStatus.INSUFFICIENT_DATA or snapshot.technical is None:
        lines.append("")
        lines.append(_bold("Status Analisis", html_mode))
        lines.append(
            _esc(
                "Data tidak mencukupi, tidak segar, atau deviasi harga berat sehingga "
                "kondisi terkini tidak dapat disajikan.",
                html_mode,
            )
        )
        for note in snapshot.limitations[:3]:
            lines.append(f"• {_esc(note, html_mode)}")
        lines.append("")
        lines.append(
            _esc(
                "Disclaimer: Informasi ini bersifat riset dan edukasi historis, "
                "bukan saran transaksi atau eksekusi order.",
                html_mode,
            )
        )
        return "\n".join(lines)

    by_label: dict[str, Fact] = {fact.label: fact for fact in snapshot.facts}
    price_tok = _fmt_fact(by_label.get("Harga penutupan terakhir"))
    ret1_tok = _fmt_fact(by_label.get("Return 1 bar"))
    ema200_tok = _fmt_fact(by_label.get("EMA 200"))
    rsi_tok = _fmt_fact(by_label.get("RSI 14"))
    adx_tok = _fmt_fact(by_label.get("ADX 14"))

    lines.append("")
    lines.append(_bold("Fakta Terukur (Backend)", html_mode))
    line1_parts = [f"Harga: {price_tok or '-'}"]
    if ret1_tok:
        line1_parts.append(f"1-bar: {ret1_tok}")
    if ema200_tok:
        line1_parts.append(f"EMA200: {ema200_tok}")
    lines.append(_esc(" | ".join(line1_parts), html_mode))

    regime_str = format_enum_label(REGIME_LABELS, snapshot.regime) if snapshot.regime else "-"
    struct_str = snapshot.market_structure.value if snapshot.market_structure else "-"
    line2_parts = [f"Rezim: {regime_str}", f"Struktur: {struct_str}"]
    if snapshot.smc is not None:
        if snapshot.smc.latest_structure_break is not None:
            brk = snapshot.smc.latest_structure_break.event_type.value.upper().replace("_", " ")
            brk_tok = _fmt_fact(by_label.get("Level BOS/CHoCH terbaru"))
            line2_parts.append(f"SMC: {brk} ({brk_tok})" if brk_tok else f"SMC: {brk}")
        if snapshot.smc.dealing_zone is not None:
            line2_parts.append(
                f"Range: {format_enum_label(DEALING_ZONE_LABELS, snapshot.smc.dealing_zone)}"
            )
    if rsi_tok:
        line2_parts.append(f"RSI14: {rsi_tok}")
    if adx_tok:
        line2_parts.append(f"ADX14: {adx_tok}")
    lines.append(_esc(" | ".join(line2_parts), html_mode))

    if snapshot.derivatives is not None:
        deriv = snapshot.derivatives
        deriv_parts: list[str] = []
        fr_tok = _fmt_fact(by_label.get("Funding rate ternormalisasi (8h)"))
        oi_chg_fact = next(
            (f for f in snapshot.facts if f.label.startswith("Perubahan open interest")),
            None,
        )
        oi_chg_tok = _fmt_fact(oi_chg_fact)
        spr_tok = _fmt_fact(by_label.get("Spread bid-ask order book"))
        imb_tok = _fmt_fact(by_label.get("Imbalance order book (10 bps)"))
        if fr_tok:
            deriv_parts.append(f"Funding 8h: {fr_tok}")
        if oi_chg_tok:
            deriv_parts.append(f"OI chg: {oi_chg_tok}")
        if spr_tok:
            deriv_parts.append(f"Spread: {spr_tok}")
        if imb_tok:
            deriv_parts.append(f"Imb 10bps: {imb_tok}")
        if deriv_parts:
            venue_tag = deriv.funding_venue or deriv.orderbook_venue or snapshot.venue
            lines.append(
                _esc(f"Derivatif ({venue_tag.upper()}): " + " | ".join(deriv_parts), html_mode)
            )

    if snapshot.order_flow is not None and snapshot.order_flow.has_order_flow:
        of = snapshot.order_flow
        of_parts: list[str] = []
        if of.taker_buy_ratio is not None:
            of_parts.append(
                f"Taker buy: {format_indonesian_number(of.taker_buy_ratio * 100.0, 1)}%"
            )
        if of.cvd_20 is not None:
            of_parts.append(f"CVD20: {format_indonesian_number(of.cvd_20, 2, signed=True)}")
        if of.cvd_divergence and of.cvd_divergence != "none":
            of_parts.append(f"Div: {of.cvd_divergence.upper()}")
        if of_parts:
            lines.append(_esc("Order Flow (Closed): " + " | ".join(of_parts), html_mode))

    if (
        snapshot.cross_venue_derivatives is not None
        and snapshot.cross_venue_derivatives.coverage_status == "full"
    ):
        cv = snapshot.cross_venue_derivatives
        cv_parts: list[str] = []
        if cv.funding_dispersion_bps is not None:
            cv_parts.append(
                f"Disp funding: {format_indonesian_number(cv.funding_dispersion_bps, 2)} bps"
            )
        if cv.mark_price_spread_bps is not None:
            cv_parts.append(
                f"Spread mark: {format_indonesian_number(cv.mark_price_spread_bps, 2)} bps"
            )
        if cv_parts:
            v_tags = "/".join(v.upper() for v in cv.venues)
            lines.append(_esc(f"Lintas Venue ({v_tags}): " + " | ".join(cv_parts), html_mode))

    if snapshot.tradingview is not None:
        tv = snapshot.tradingview
        tv_parts = [f"Event: {tv.event_type}"]
        score_tok = _fmt_fact(by_label.get("Skor aturan TradingView (bukan probabilitas)"))
        dev_tok = _fmt_fact(by_label.get("Deviasi harga TradingView vs bursa"))
        if score_tok:
            lbl = f" ({tv.rule_label})" if tv.rule_label else ""
            tv_parts.append(f"Skor aturan: {score_tok}{lbl}")
        if dev_tok:
            tv_parts.append(f"Deviasi: {dev_tok}")
        lines.append(
            _esc("TradingView Lens (Bukan Probabilitas): " + " | ".join(tv_parts), html_mode)
        )

    if snapshot.sentiment is not None:
        _append_sentiment_lines(lines, snapshot, html_mode=html_mode, max_sources=2)

    if snapshot.prediction_markets is not None and snapshot.prediction_markets.observations:
        pm = snapshot.prediction_markets
        pm_parts: list[str] = []
        if pm.macro_fomc_cut_prob is not None:
            pm_parts.append(
                f"FOMC Cut: {format_indonesian_number(pm.macro_fomc_cut_prob * 100.0, 0)}%"
            )
        if pm.crypto_upside_prob is not None:
            pm_parts.append(
                f"Target YES: {format_indonesian_number(pm.crypto_upside_prob * 100.0, 0)}%"
            )
        elif pm.observations:
            top_obs = pm.observations[0]
            pct_val = format_indonesian_number(top_obs.implied_yes_prob * 100.0, 0)
            pm_parts.append(f"{top_obs.question[:20]}… {pct_val}%")
        if pm_parts:
            lines.append(_esc("Pasar Prediksi (Tersirat): " + " | ".join(pm_parts), html_mode))

    if snapshot.macro_context is not None:
        mc = snapshot.macro_context
        macro_parts: list[str] = [f"Posisi: {mc.positioning_regime.value.upper()}"]
        if mc.dxy is not None:
            dxy_info = f"DXY: {mc.dxy.macro_impact.upper()}"
            if mc.dxy.rolling_corr_20 is not None:
                dxy_info += (
                    f" (r={format_indonesian_number(mc.dxy.rolling_corr_20, 2, signed=True)})"
                )
            macro_parts.append(dxy_info)
        if mc.btc_benchmark is not None and mc.btc_benchmark.leader_alignment != "self":
            btc_info = "vs BTC: "
            if mc.btc_benchmark.relative_strength_20_pct is not None:
                rs_val = format_indonesian_number(
                    mc.btc_benchmark.relative_strength_20_pct, 2, signed=True
                )
                btc_info += f"{rs_val}%"
            if mc.btc_benchmark.rolling_beta_20 is not None:
                btc_info += (
                    f" (beta={format_indonesian_number(mc.btc_benchmark.rolling_beta_20, 2)})"
                )
            macro_parts.append(btc_info)
        lines.append(_esc("Konteks Makro & Posisi: " + " | ".join(macro_parts), html_mode))

    if snapshot.positioning is not None:
        pos = snapshot.positioning
        pos_parts: list[str] = [f"Status: {pos.status}"]
        if pos.top_lsr_account is not None:
            top_tok = format_indonesian_number(pos.top_lsr_account, 2)
            if pos.top_lsr_account_pct_30d is not None:
                top_tok += f" (P{format_indonesian_number(pos.top_lsr_account_pct_30d, 0)})"
            pos_parts.append(f"Top L/S: {top_tok}")
        if pos.lsr_taker is not None:
            pos_parts.append(f"Taker: {format_indonesian_number(pos.lsr_taker, 2)}")
        if pos.oi_24h_delta_pct is not None:
            pos_parts.append(
                f"OI 24h: {format_indonesian_number(pos.oi_24h_delta_pct, 1, signed=True)}%"
            )
        lines.append(
            _esc(
                f"Positioning ({pos.parity_mode}): " + " | ".join(pos_parts),
                html_mode,
            )
        )

    _append_composite_sentiment_panel(lines, snapshot, html_mode=html_mode)

    if len(snapshot.mtf_overview) >= 2:
        tf_chain = " → ".join(ov.timeframe.value for ov in snapshot.mtf_overview)
        lines.append("")
        lines.append(_bold(f"Alur Top-Down Multi-Timeframe ({tf_chain})", html_mode))
        for ov in snapshot.mtf_overview:
            parts: list[str] = []
            if ov.trend_label:
                parts.append(ov.trend_label)
            if ov.market_structure is not None:
                parts.append(f"Struktur: {ov.market_structure.value}")
            if ov.structure_break is not None:
                b_lbl = ov.structure_break.value.upper().replace("_", " ")
                parts.append(f"SMC: {b_lbl}")
            if ov.sweep is not None:
                s_lbl = "Sweep Bull" if "bullish" in ov.sweep.value else "Sweep Bear"
                parts.append(s_lbl)
            if ov.dealing_zone is not None:
                parts.append(f"Range: {format_enum_label(DEALING_ZONE_LABELS, ov.dealing_zone)}")
            if ov.active_poi:
                parts.append(f"POI: {ov.active_poi}")
            if ov.timeframe is Timeframe.M15:
                s15 = _fmt_fact(by_label.get("Support struktur 15m"))
                r15 = _fmt_fact(by_label.get("Resistance struktur 15m"))
                if s15 or r15:
                    parts.append(f"S/R 15m: {s15 or '-'} / {r15 or '-'}")
            row_body = " | ".join(parts) if parts else "data terbatas"
            lines.append(_esc(f"• {ov.timeframe.value}: {row_body}", html_mode))

    # Skenario Setup (Deterministik)
    lines.append("")
    lines.append(_bold("Skenario Setup (Deterministik — Bukan Probabilitas)", html_mode))
    lines.append(
        _esc(
            f"• Gaya / Mode: {format_trading_style_label(snapshot.timeframe)}",
            html_mode,
        )
    )
    if snapshot.smc is not None and snapshot.smc.dealing_zone is not None:
        dz_setup_lbl = format_enum_label(DEALING_ZONE_LABELS, snapshot.smc.dealing_zone)
        lines.append(_esc(f"• Posisi Dealing Range: {dz_setup_lbl}", html_mode))
    setup = snapshot.trade_setup
    sh_htf = _fmt_fact(by_label.get("High dealing range")) or _fmt_fact(
        by_label.get("Swing high terkonfirmasi")
    )
    sl_htf = _fmt_fact(by_label.get("Low dealing range")) or _fmt_fact(
        by_label.get("Swing low terkonfirmasi")
    )
    s_ltf = _fmt_fact(by_label.get("Support struktur 15m"))
    r_ltf = _fmt_fact(by_label.get("Resistance struktur 15m"))
    if setup is None:
        lines.append("• Status: belum dihitung")
    elif setup.status is SetupStatus.WAIT:
        lines.append(
            f"• Status: <b>WAIT / NO_SETUP</b> — {_esc(setup.wait_reason or '-', html_mode)}"
            if html_mode
            else f"• Status: WAIT / NO_SETUP — {setup.wait_reason or '-'}"
        )
        if sl_htf or sh_htf:
            lines.append(
                _esc(
                    f"• Level pantau ({snapshot.timeframe.value}): "
                    f"Support {sl_htf or '-'} | Resistance {sh_htf or '-'}",
                    html_mode,
                )
            )
        if s_ltf or r_ltf:
            lines.append(
                _esc(
                    f"• Level pantau (15m): Support {s_ltf or '-'} | Resistance {r_ltf or '-'}",
                    html_mode,
                )
            )
    else:
        bias_label = format_enum_label(SETUP_BIAS_LABELS, setup.bias)
        e_low = _fmt_fact(by_label.get("Zona entry (bawah)")) or "-"
        e_high = _fmt_fact(by_label.get("Zona entry (atas)")) or "-"
        sl_tok = _fmt_fact(by_label.get("Stop loss (invalidasi)")) or "-"
        tp1_tok = _fmt_fact(by_label.get("Target TP 1")) or "-"
        tp2_tok = _fmt_fact(by_label.get("Target TP 2")) or "-"
        tp3_tok = _fmt_fact(by_label.get("Target TP 3")) or "-"
        rr1_tok = _fmt_fact(by_label.get("Rasio R:R TP 1")) or "-"
        rr2_tok = _fmt_fact(by_label.get("Rasio R:R TP 2")) or "-"
        rr3_tok = _fmt_fact(by_label.get("Rasio R:R TP 3")) or "-"

        lines.append(_esc(f"• Bias ({snapshot.timeframe.value}): {bias_label}", html_mode))
        if sl_htf or sh_htf:
            lines.append(
                _esc(
                    f"• S/R HTF ({snapshot.timeframe.value}): "
                    f"Support {sl_htf or '-'} | Resistance {sh_htf or '-'}",
                    html_mode,
                )
            )
        if s_ltf or r_ltf:
            lines.append(
                _esc(
                    f"• S/R LTF (15m): Support {s_ltf or '-'} | Resistance {r_ltf or '-'}",
                    html_mode,
                )
            )
        src_note = f" ({setup.entry_source})" if setup.entry_source else ""
        lines.append(_esc(f"• Zona Entry: {e_low} s/d {e_high}{src_note}", html_mode))
        inv_note = f" ({setup.invalidation_reason})" if setup.invalidation_reason else ""
        lines.append(_esc(f"• Stop Loss: {sl_tok}{inv_note}", html_mode))
        t1_note = f" · {setup.tp1_label}" if setup.tp1_label else ""
        t2_note = f" · {setup.tp2_label}" if setup.tp2_label else ""
        t3_note = f" · {setup.tp3_label}" if setup.tp3_label else ""
        lines.append(_esc(f"• TP 1: {tp1_tok} (R:R {rr1_tok}{t1_note})", html_mode))
        lines.append(_esc(f"• TP 2: {tp2_tok} (R:R {rr2_tok}{t2_note})", html_mode))
        lines.append(_esc(f"• TP 3: {tp3_tok} (R:R {rr3_tok}{t3_note})", html_mode))
        if setup.stop_loss_distance_pct is not None:
            lev_str = (
                f" | Max Ref Lev: {setup.max_theoretical_leverage}x"
                if setup.max_theoretical_leverage
                else ""
            )
            sl_pct_str = format_indonesian_number(setup.stop_loss_distance_pct, 2)
            lines.append(_esc(f"• Jarak SL: {sl_pct_str}%{lev_str}", html_mode))
        if setup.ltf_confirmation:
            lines.append(_esc(f"• Konfirmasi 15m/MTF: {setup.ltf_confirmation}", html_mode))

    # Key points Bull vs Bear & Risiko
    bull_items = [f for f in snapshot.factors if f.direction is FactorDirection.SUPPORTS_BULLISH][
        :2
    ]
    bear_items = [f for f in snapshot.factors if f.direction is FactorDirection.SUPPORTS_BEARISH][
        :2
    ]
    lines.append("")
    lines.append(_bold("Key Points (Bull vs Bear & Risiko)", html_mode))
    if bull_items:
        for item in bull_items:
            refs = ", ".join(item.fact_ids)
            lines.append(f"• Bull: {_esc(item.description, html_mode)} [{_esc(refs, html_mode)}]")
    else:
        lines.append("• Bull: belum ada faktor penguat dominan.")
    if bear_items:
        for item in bear_items:
            refs = ", ".join(item.fact_ids)
            lines.append(f"• Bear: {_esc(item.description, html_mode)} [{_esc(refs, html_mode)}]")
    else:
        lines.append("• Bear: belum ada faktor pelemah dominan.")
    if setup is not None and setup.risk_notes:
        lines.append(f"• Risiko: {_esc(setup.risk_notes[0], html_mode)}")
    elif snapshot.derivatives is not None:
        deriv_venue = (
            snapshot.derivatives.funding_venue
            or snapshot.derivatives.open_interest_venue
            or snapshot.derivatives.orderbook_venue
            or snapshot.venue
        ).capitalize()
        lines.append(
            "• Risiko: Data derivatif hanya berasal dari satu venue "
            f"({_esc(deriv_venue, html_mode)})."
        )

    _append_ml_prediction_lines(lines, snapshot, html_mode=html_mode)

    base_coin = snapshot.canonical_symbol.split("/")[0]
    lines.append("")
    lines.append(
        _esc(
            f"Detail lengkap: /analyze {base_coin} {snapshot.timeframe.value} detail",
            html_mode,
        )
    )
    lines.append(
        _esc(
            "Disclaimer: Pengukuran historis untuk riset & edukasi, bukan saran transaksi atau "
            "jaminan hasil.",
            html_mode,
        )
    )
    return "\n".join(lines)


def split_telegram_message(
    text: str,
    max_length: int = TELEGRAM_MAX_MESSAGE_LENGTH,
) -> tuple[str, ...]:
    """Split rendered text into chunks strictly <= `max_length` at line boundaries."""
    if max_length < 64:
        raise ValueError("max_length must be >= 64")
    if len(text) <= max_length:
        return (text,)

    chunks: list[str] = []
    current_lines: list[str] = []
    current_len = 0

    for line in text.splitlines():
        while len(line) > max_length:
            if current_lines:
                chunks.append("\n".join(current_lines))
                current_lines = []
                current_len = 0
            chunks.append(line[:max_length])
            line = line[max_length:]

        added_len = len(line) if not current_lines else len(line) + 1
        if current_len + added_len > max_length:
            chunks.append("\n".join(current_lines))
            current_lines = [line]
            current_len = len(line)
        else:
            current_lines.append(line)
            current_len += added_len

    if current_lines:
        chunks.append("\n".join(current_lines))

    return tuple(chunks)


def render_start_message() -> str:
    """Render Indonesian onboarding message and research disclaimer."""
    lines = [
        "<b>Selamat datang di Crypto Research Assistant</b>",
        (
            "Bot ini adalah asisten riset dan edukasi pasar kripto berbasis data candle "
            "tertutup futures perpetual &amp; spot serta perhitungan indikator deterministik."
        ),
        "",
        "<b>Cakupan Analisis:</b>",
        "- Simbol futures perpetual /USDT:USDT (mis. BTC, ETH, SOL, DOGE, SUI, PEPE, VVV, dll.)",
        "- Unified Multi-Timeframe (1d → 4h → 1h → 15m → 5m) berjalan otomatis dalam 1 setup",
        "",
        "<b>Cara Pakai (Chat Natural atau Perintah):</b>",
        '- Ketik santai: <i>"coba analisa btc sekarang"</i> (default: mode Swing 4h)',
        (
            "- Pilih gaya trading: sebut <b>scalp</b> (15m), <b>intraday</b> (1h), "
            'atau <b>swing</b> (4h) — misal <i>"analisa sol buat scalp"</i>'
        ),
        (
            '- Pindai koin: <i>"ada token yang bisa disetup buat scalp gak?"</i> '
            "atau /scan [scalp|intraday|swing]"
        ),
        (
            '- Screener 15m: <i>"screening top gainers dan top losers dong"</i> '
            "atau /screen [long|short]"
        ),
        (
            "- /analyze BTC [scalp|intraday|swing|15m|1h|4h|1d] — "
            "ringkasan key points + Entry/SL/TP1-3"
        ),
        "- /pine BTC [scalp|intraday|swing] — export TradingView Pine Script v6",
        "- /compare BTC ETH 1d — perbandingan deskriptif antar koin",
        "- /watchlist — lihat atau kelola daftar pantau eksplisit",
        "- /memory — lihat preferensi dan batas memori percakapan",
        "- /forget — hapus preferensi dan riwayat percakapan Anda",
        "- /help — panduan lengkap penggunaan",
        "",
        (
            "Disclaimer: Bot ini hanya menyajikan pengukuran historis untuk riset dan edukasi; "
            "bukan saran transaksi, jaminan hasil, alokasi portofolio, atau eksekusi order."
        ),
    ]
    return "\n".join(lines)


def render_help_message() -> str:
    """Render Indonesian help message covering all supported MVP commands."""
    lines = [
        "<b>Panduan Perintah</b>",
        (
            "- /analyze &lt;SIMBOL&gt; [scalp|intraday|swing|15m|1h|4h|1d] [detail] — contoh: "
            "/analyze DOGE swing atau /analyze SUI scalp"
        ),
        (
            "- /compare &lt;SIMBOL1&gt; &lt;SIMBOL2&gt; [SIMBOL3] "
            "[scalp|intraday|swing|15m|1h|4h|1d] — contoh: /compare BTC ETH swing"
        ),
        (
            "- /pine &lt;SIMBOL&gt; [scalp|intraday|swing|15m|1h|4h|1d] — "
            "export TradingView Pine Script v6 untuk Entry, SL, TP, S/R, dan zona SMC"
        ),
        (
            "- /scan [scalp|intraday|swing|15m|1h|4h|1d] — "
            "pindai watchlist &amp; basket futures untuk cari setup READY"
        ),
        (
            "- /screen [long|short|all] [min_vol_juta] — "
            "screener Top Gainers/Losers (Vol 24j >= $5M) + momentum Engulfing/Marubozu 15m"
        ),
        (
            "- /whale [SIMBOL|alamat|add|remove|list] — "
            "pantau posisi perpetual whale (Hyperliquid) &amp; pelacak dompet on-chain / aliran CEX"
        ),
        "- /watchlist — tampilkan daftar pantau",
        "- /watchlist add &lt;SIMBOL&gt; — tambah pair futures /USDT:USDT ke daftar pantau",
        "- /watchlist remove &lt;SIMBOL&gt; — hapus dari daftar pantau",
        "- /memory — periksa preferensi eksplisit dan pesan tersimpan (maksimal 20 pesan)",
        "- /forget — hapus seluruh preferensi dan memori percakapan Anda",
        "",
        (
            "Pertanyaan lanjutan berbahasa alami seperti "
            '<i>"bandingkan momentumnya dengan ETH"</i> atau <i>"cek koin doge 15m"</i> '
            "akan memicu analisis backend segar."
        ),
    ]
    return "\n".join(lines)


def render_watchlist_message(
    watchlist: tuple[str, ...] | list[str],
    *,
    note: str | None = None,
    html_mode: bool = True,
) -> str:
    """Render user's explicit watchlist safely."""
    lines = [_bold("Daftar Pantau (Watchlist)", html_mode)]
    if note:
        lines.append(_esc(note, html_mode))
    if watchlist:
        for sym in watchlist:
            lines.append(f"- {_esc(sym, html_mode)}")
    else:
        lines.append("Daftar pantau masih kosong. Tambah dengan /watchlist add BTC.")
    return "\n".join(lines)


def render_memory_message(
    prefs: UserPreferences,
    messages: tuple[StoredMessage, ...] | list[StoredMessage],
    *,
    html_mode: bool = True,
) -> str:
    """Render stored user preferences and bounded recent message count."""
    wl = ", ".join(prefs.watchlist) if prefs.watchlist else "kosong"
    lines = [
        _bold("Memori & Preferensi Tersimpan", html_mode),
        f"- Bahasa: {_esc(prefs.language, html_mode)}",
        f"- Timeframe default: {_esc(prefs.default_timeframe.value, html_mode)}",
        f"- Mata uang tampilan: {_esc(prefs.display_currency, html_mode)}",
        f"- Watchlist: {_esc(wl, html_mode)}",
        f"- Pesan tersimpan (maks 20): {len(messages)} pesan (retensi 30 hari)",
    ]
    if messages:
        lines.append("")
        lines.append(_bold("Riwayat Terbaru (Konteks Topik)", html_mode))
        for msg in messages[-5:]:
            ts = _esc(format_timestamp_utc_wib(msg.created_at), html_mode)
            snippet = msg.content.replace("\n", " ")[:120]
            lines.append(f"- [{_esc(msg.role, html_mode)} | {ts}] {_esc(snippet, html_mode)}")
    lines.append("")
    lines.append("Gunakan /forget kapan saja untuk menghapus preferensi dan riwayat percakapan.")
    return "\n".join(lines)


def render_forget_message(result: ForgetResult, *, html_mode: bool = True) -> str:
    """Render confirmation of deleted user-scoped memory records."""
    lines = [
        _bold("Penghapusan Memori Selesai", html_mode),
        "Data pribadi berikut untuk akun Anda telah dihapus permanen:",
        f"- Pesan percakapan: {result.deleted_messages}",
        f"- Utas percakapan: {result.deleted_conversations}",
        f"- Preferensi pengguna: {result.deleted_preferences}",
        f"- Log interaksi LLM: {result.deleted_interactions}",
        "",
        "Data pasar publik dan snapshot anonim tidak terpengaruh.",
    ]
    return "\n".join(lines)


def render_smalltalk_reply(subtype: Literal["greeting", "thanks", "ack"]) -> str:
    """Render concise deterministic Indonesian reply for greetings, thanks, and acks."""
    if subtype == "thanks":
        return (
            "Sama-sama! Jika butuh cek koin lain, tanya istilah trading, atau pindai setup, "
            "kabari saja."
        )
    if subtype == "ack":
        return "Siap! Kabari saja kalau ingin cek koin atau diskusi konsep trading."
    return (
        "Halo! Saya asisten riset & edukasi futures kripto. "
        "Tanya konsep trading (mis. 'apa itu RSI 70?'), cek koin (mis. 'analisa BTC 4h'), "
        "atau ketik /help."
    )


_EDU_GLOSSARY: tuple[tuple[re.Pattern[str], str, str], ...] = (
    (
        re.compile(r"\brsi\b", re.IGNORECASE),
        "RSI (Relative Strength Index)",
        "Mengukur kecepatan momentum harga (skala 0-100). Di atas 70 sering dianggap jenuh beli "
        "(overbought) dan di bawah 30 jenuh jual (oversold), namun pada tren kuat RSI bisa "
        "bertahan di area ekstrem cukup lama.",
    ),
    (
        re.compile(r"\bema\b", re.IGNORECASE),
        "EMA (Exponential Moving Average)",
        "Rata-rata bergerak dengan bobot lebih besar pada penutupan terbaru. EMA 20 lebih "
        "responsif terhadap momentum pendek, sedangkan EMA 200 membantu membaca arah tren makro.",
    ),
    (
        re.compile(r"\bmacd\b", re.IGNORECASE),
        "MACD",
        "Mengukur selisih dua EMA beserta garis sinyal dan histogramnya untuk membaca perubahan "
        "momentum serta potensi divergensi.",
    ),
    (
        re.compile(r"\bfunding(?:\s+rate)?\b", re.IGNORECASE),
        "Funding Rate (Perpetual)",
        "Biaya periodik (umumnya tiap 8 jam) antar pemegang posisi long dan short agar harga "
        "kontrak perpetual tetap dekat dengan harga indeks spot.",
    ),
    (
        re.compile(r"\bspot\b", re.IGNORECASE),
        "Spot vs Perpetual Futures",
        "Pasar spot memperdagangkan aset dasar secara langsung tanpa leverage, sedangkan perpetual "
        "futures memperdagangkan kontrak derivatif dengan margin, leverage, dan funding rate.",
    ),
    (
        re.compile(r"\b(?:bos|choch)\b", re.IGNORECASE),
        "BOS & CHoCH (Market Structure)",
        "BOS (Break of Structure) menandakan kelanjutan struktur tren saat swing utama tertembus; "
        "CHoCH (Change of Character) memberi indikasi awal potensi perubahan arah tren.",
    ),
    (
        re.compile(r"\blong\b.*\bshort\b|\bshort\b.*\blong\b", re.IGNORECASE),
        "Posisi Long vs Short",
        "Long bertujuan memperoleh selisih saat harga naik, sedangkan Short bertujuan memperoleh "
        "selisih saat harga turun pada kontrak derivatif/futures.",
    ),
    (
        re.compile(r"\brisk[\s/]*reward\b|\br:r\b", re.IGNORECASE),
        "Risk / Reward (R:R)",
        "Perbandingan jarak risiko ke Stop Loss terhadap jarak potensi target Take Profit "
        "(mis. 1:2 berarti potensi target 2x jarak risiko).",
    ),
    (
        re.compile(r"\ball[\s-]+in\b", re.IGNORECASE),
        "Risiko All-In",
        "Menaruh seluruh modal pada satu posisi memperbesar risiko likuidasi atau drawdown tajam. "
        "Pembatasan ukuran risiko per transaksi membantu ketahanan modal.",
    ),
    (
        re.compile(r"\bleverage\b", re.IGNORECASE),
        "Leverage",
        "Pengganda ukuran posisi terhadap margin jaminan. Leverage yang lebih tinggi mempersempit "
        "jarak ke harga likuidasi.",
    ),
    (
        re.compile(r"\b(?:isolated|cross)\s*margin\b", re.IGNORECASE),
        "Isolated vs Cross Margin",
        "Isolated membatasi risiko kerugian pada margin posisi tertentu saja, sedangkan Cross "
        "menggunakan seluruh saldo margin futures yang tersedia.",
    ),
    (
        re.compile(r"\bfvg\b", re.IGNORECASE),
        "FVG (Fair Value Gap)",
        "Celah ketidakseimbangan harga pada rangkaian 3 candle yang sering dipantau sebagai zona "
        "reaksi pullback.",
    ),
    (
        re.compile(r"\border\s*block\b", re.IGNORECASE),
        "Order Block",
        "Area candle berlawanan terakhir sebelum pergerakan impulsif kuat yang kerap dijadikan "
        "referensi zona minat harga historis.",
    ),
    (
        re.compile(r"\bdxy\b", re.IGNORECASE),
        "US Dollar Index (DXY)",
        "Indeks kekuatan dolar AS terhadap sekeranjang mata uang utama. Penguatan DXY sering "
        "menjadi konteks tekanan bagi aset berisiko seperti kripto.",
    ),
    (
        re.compile(r"\bliquidation\b|\blikuidasi\b", re.IGNORECASE),
        "Liquidation Price",
        "Level harga saat margin tidak lagi cukup menahan kerugian posisi sehingga bursa menutup "
        "posisi secara paksa.",
    ),
    (
        re.compile(r"\bdca\b", re.IGNORECASE),
        "DCA (Dollar-Cost Averaging)",
        "Metode masuk bertahap secara berkala dengan nominal terencana untuk meratakan harga "
        "akuisisi.",
    ),
    (
        re.compile(r"\b(?:fomo|fud)\b", re.IGNORECASE),
        "FOMO & FUD",
        "FOMO adalah dorongan masuk terburu-buru karena takut ketinggalan gerak harga; FUD adalah "
        "narasi ketakutan/ketidakpastian yang memicu keputusan emosional.",
    ),
    (
        re.compile(r"\bpsikologi\b|\bloss\b", re.IGNORECASE),
        "Psikologi & Kontrol Loss",
        "Pengendalian emosi saat loss bertumpu pada ukuran risiko terukur, kepatuhan pada Stop "
        "Loss, dan jeda sejenak untuk menghindari revenge trading.",
    ),
    (
        re.compile(r"\b(?:ath|atl)\b", re.IGNORECASE),
        "ATH & ATL",
        "ATH (All-Time High) adalah rekor harga tertinggi sepanjang masa; ATL (All-Time Low) "
        "adalah rekor harga terendah sepanjang masa.",
    ),
    (
        re.compile(r"\bsupport\b|\bresistance\b", re.IGNORECASE),
        "Support & Resistance",
        "Support adalah area minat beli historis yang menahan penurunan; Resistance adalah area "
        "minat jual historis yang menahan kenaikan.",
    ),
    (
        re.compile(r"\bcandle\b|\b5m\b|\b15m\b", re.IGNORECASE),
        "Candle & Timeframe",
        "Setiap candle merangkum harga Open, High, Low, Close pada satu rentang waktu "
        "(mis. 5m mikro, 15m scalp, 1h intraday, 4h swing).",
    ),
)


def render_context_fallback(
    category: str,
    *,
    question: str = "",
    active_symbol: str | None = None,
) -> str:
    """Render context-specific Indonesian fallback guidance for each non-analysis category."""
    if category == "smalltalk":
        return (
            "Halo! Lagi santai memantau pasar nih. Kalau mau tanya istilah trading, cek "
            "kondisi market, atau analisa koin tertentu, langsung ketik saja."
        )
    if category == "edu_qa":
        matches: list[tuple[str, str]] = []
        for pattern, title, desc in _EDU_GLOSSARY:
            if pattern.search(question):
                matches.append((title, desc))
            if len(matches) >= 2:
                break
        if matches:
            lines = ["<b>Ringkasan Glosarium (Mode Deterministik)</b>"]
            for title, desc in matches:
                lines.append(
                    f"• <b>{html.escape(title, quote=False)}</b>: {html.escape(desc, quote=False)}"
                )
            lines.append("")
            lines.append(
                "<i>Catatan: Layanan penjelasan AI interaktif sedang sibuk/mencapai batas kuota "
                "saat ini.</i>"
            )
            return "\n".join(lines)
        return (
            "Layanan penjelasan AI untuk pertanyaan edukasi sedang sibuk atau mencapai batas "
            "kuota saat ini. Silakan ulangi beberapa saat lagi atau ketik /help."
        )
    if category == "followup_expired":
        return (
            "Konteks koin sebelumnya sudah berakhir (lebih dari 15 menit atau topik sudah "
            "berpindah). Sebutkan kembali nama koin yang ingin dicek (mis. 'analisa BTC 1h') "
            "untuk data terbaru."
        )
    if category == "followup_active" and active_symbol:
        base = active_symbol.split("/")[0]
        return (
            f"Layanan penjelasan lanjutan AI sedang sibuk saat ini. Untuk pembaruan data "
            f"{base}, ketik 'cek {base}' atau '/analyze {base} detail'."
        )
    if category == "out_of_scope_non_crypto":
        return (
            "Bot ini khusus untuk riset dan edukasi pasar perpetual futures kripto (/USDT:USDT), "
            "sehingga tidak melayani topik di luar kripto seperti saham atau pertanyaan umum "
            "lainnya. Ketik /help untuk melihat daftar fitur."
        )
    return (
        "Pesan belum dapat dipahami. Anda bisa mencoba contoh berikut:\n"
        "• 'analisa BTC 4h' — analisis teknikal & setup koin\n"
        "• 'apa itu RSI 70?' — pertanyaan edukasi trading\n"
        "• 'ada setup apa hari ini?' — pindai peluang futures\n"
        "Atau ketik /help untuk panduan lengkap."
    )
