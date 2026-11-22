"""Deterministic Indonesian formatting and Telegram HTML rendering."""

from __future__ import annotations

import html
from datetime import UTC, datetime, timedelta, timezone

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
from crypto_chatbot.schemas.market import QualityStatus, Timeframe
from crypto_chatbot.storage.repositories import (
    ForgetResult,
    StoredMessage,
    UserPreferences,
)

WIB = timezone(timedelta(hours=7))
TELEGRAM_MAX_MESSAGE_LENGTH = 4096

_REGIME_LABELS: dict[Regime, str] = {
    Regime.HIGH_VOLATILITY: "Volatilitas Tinggi",
    Regime.TRENDING_UP: "Tren Naik",
    Regime.TRENDING_DOWN: "Tren Turun",
    Regime.RANGING: "Mendatar (Ranging)",
    Regime.TRANSITIONAL: "Transisi",
}

_STRUCTURE_LABELS: dict[MarketStructure, str] = {
    MarketStructure.BULLISH: "Bullish (HH + HL terkonfirmasi)",
    MarketStructure.BEARISH: "Bearish (LH + LL terkonfirmasi)",
    MarketStructure.MIXED: "Bercampur (Mixed)",
}

_MTF_LABELS: dict[MtfRsiAlignment, str] = {
    MtfRsiAlignment.ALIGNED_BULLISH: "Selaras di atas 50 (1h, 4h, 1d)",
    MtfRsiAlignment.ALIGNED_BEARISH: "Selaras di bawah 50 (1h, 4h, 1d)",
    MtfRsiAlignment.MIXED: "Tercampur antar timeframe",
}


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
    if fmt == "ratio_2":
        return f"{format_indonesian_number(val, 2)}x"
    return format_indonesian_number(val, 2)


def _esc(text: str, html_mode: bool) -> str:
    return html.escape(text, quote=False) if html_mode else text


def _bold(text: str, html_mode: bool) -> str:
    escaped = _esc(text, html_mode)
    return f"<b>{escaped}</b>" if html_mode else escaped


def render_deterministic_snapshot(
    snapshot: AnalysisSnapshot,
    *,
    html_mode: bool = False,
) -> str:
    """Render an `AnalysisSnapshot` deterministically without any LLM involvement."""
    lines: list[str] = []
    header = (
        f"Analisis {snapshot.canonical_symbol} ({snapshot.timeframe.value}) - "
        f"{snapshot.venue.upper()} {snapshot.market_type.value}"
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
            lines.append(f"- Rezim: {_esc(_REGIME_LABELS[snapshot.regime], html_mode)}")
        if snapshot.market_structure is not None:
            lines.append(
                f"- Struktur pivot: {_esc(_STRUCTURE_LABELS[snapshot.market_structure], html_mode)}"
            )
        if snapshot.technical.mtf_rsi_alignment is not None:
            mtf_label = _esc(_MTF_LABELS[snapshot.technical.mtf_rsi_alignment], html_mode)
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
                    lines.append(
                        f"- Posisi Dealing Range: {_esc(smc.dealing_zone.value, html_mode)}"
                    )
            if snapshot.trade_setup is not None:
                setup = snapshot.trade_setup
                lines.append(
                    f"- Status setup: {_esc(setup.status.value.upper(), html_mode)} "
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

    regime_str = _REGIME_LABELS[snapshot.regime] if snapshot.regime else "-"
    struct_str = snapshot.market_structure.value if snapshot.market_structure else "-"
    line2_parts = [f"Rezim: {regime_str}", f"Struktur: {struct_str}"]
    if snapshot.smc is not None:
        if snapshot.smc.latest_structure_break is not None:
            brk = snapshot.smc.latest_structure_break.event_type.value.upper().replace("_", " ")
            brk_tok = _fmt_fact(by_label.get("Level BOS/CHoCH terbaru"))
            line2_parts.append(f"SMC: {brk} ({brk_tok})" if brk_tok else f"SMC: {brk}")
        if snapshot.smc.dealing_zone is not None:
            dz_labels = {
                DealingZone.DISCOUNT: "Discount",
                DealingZone.EQUILIBRIUM: "Equilibrium",
                DealingZone.PREMIUM: "Premium",
            }
            line2_parts.append(f"Range: {dz_labels[snapshot.smc.dealing_zone]}")
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

    if len(snapshot.mtf_overview) >= 2:
        lines.append("")
        lines.append(_bold("Alur Top-Down Multi-Timeframe (1d → 4h → 1h → 15m)", html_mode))
        dz_map = {
            DealingZone.DISCOUNT: "Discount",
            DealingZone.EQUILIBRIUM: "Equilibrium",
            DealingZone.PREMIUM: "Premium",
        }
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
                parts.append(f"Range: {dz_map[ov.dealing_zone]}")
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
        bias_label = "LONG (Pullback)" if setup.bias is SetupBias.LONG else "SHORT (Reli Pullback)"
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
        lines.append("• Risiko: Data derivatif hanya berasal dari satu venue (Binance).")

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
            "tertutup Binance spot dan perhitungan indikator deterministik."
        ),
        "",
        "<b>Cakupan Analisis:</b>",
        "- Simbol spot Binance /USDT (mis. BTC, ETH, SOL, DOGE, SUI, PEPE, dll.)",
        "- Timeframe: 15m, 1h, 4h, 1d (dengan refinement entry 15m untuk analisis 1h/4h/1d)",
        "",
        "<b>Perintah Utama:</b>",
        "- /analyze BTC 4h — ringkasan key points + Entry/SL/TP1-3 kondisional",
        "- /analyze SUI 15m — analisis langsung timeframe 15 menit (dengan filter HTF)",
        "- /analyze BTC 4h detail — rincian seluruh metrik dan fakta backend",
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
            "- /analyze &lt;SIMBOL&gt; [15m|1h|4h|1d] [detail] — contoh: "
            "/analyze DOGE 4h atau /analyze SUI 15m"
        ),
        (
            "- /compare &lt;SIMBOL1&gt; &lt;SIMBOL2&gt; [SIMBOL3] [15m|1h|4h|1d] "
            "— contoh: /compare BTC ETH 1d"
        ),
        "- /watchlist — tampilkan daftar pantau",
        "- /watchlist add &lt;SIMBOL&gt; — tambah pair spot /USDT ke daftar pantau",
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
