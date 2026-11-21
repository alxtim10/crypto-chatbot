"""Deterministic Indonesian formatting and Telegram HTML rendering."""

from __future__ import annotations

import html
from datetime import UTC, datetime, timedelta, timezone

from crypto_chatbot.schemas.analysis import (
    AnalysisSnapshot,
    Fact,
    FactorDirection,
    MarketStructure,
    MtfRsiAlignment,
    Regime,
)
from crypto_chatbot.schemas.market import QualityStatus
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
        decimals = 4 if abs(val) < 1.0 else 2
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
        "<b>Cakupan MVP:</b>",
        "- Simbol spot: BTC/USDT, ETH/USDT, SOL/USDT",
        "- Timeframe: 1h, 4h, 1d (hanya candle yang sudah tutup)",
        "",
        "<b>Perintah Utama:</b>",
        "- /analyze BTC 4h — analisis teknikal deterministik + penjelasan tervalidasi",
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
        "- /analyze &lt;SIMBOL&gt; [1h|4h|1d] — contoh: /analyze BTC 4h",
        (
            "- /compare &lt;SIMBOL1&gt; &lt;SIMBOL2&gt; [SIMBOL3] [1h|4h|1d] "
            "— contoh: /compare BTC ETH 1d"
        ),
        "- /watchlist — tampilkan daftar pantau",
        "- /watchlist add &lt;BTC|ETH|SOL&gt; — tambah ke daftar pantau",
        "- /watchlist remove &lt;BTC|ETH|SOL&gt; — hapus dari daftar pantau",
        "- /memory — periksa preferensi eksplisit dan pesan tersimpan (maksimal 20 pesan)",
        "- /forget — hapus seluruh preferensi dan memori percakapan Anda",
        "",
        (
            "Pertanyaan lanjutan berbahasa alami seperti "
            '<i>"bandingkan momentumnya dengan ETH"</i> akan memicu analisis backend segar.'
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
