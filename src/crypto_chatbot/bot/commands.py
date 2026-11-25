"""Command argument parsers and deterministic follow-up intent resolution."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from crypto_chatbot.market_data.symbols import resolve_default_perp_symbol
from crypto_chatbot.schemas.commands import (
    AnalyzeCommandArgs,
    CompareCommandArgs,
    WatchlistAction,
    WatchlistCommandArgs,
)
from crypto_chatbot.schemas.market import (
    MULTI_STYLE_SCAN_TIMEFRAMES,
    PRIMARY_TIMEFRAMES,
    Timeframe,
    resolve_primary_timeframe,
)
from crypto_chatbot.storage.repositories import StoredMessage

MAX_USER_MESSAGE_CHARS = 1000

_SUPPORTED_TFS = frozenset(tf.value for tf in PRIMARY_TIMEFRAMES)
_TIMEFRAME_LIKE_RE = re.compile(r"^\d+[mhdwM]$")
_BAD_TIMEFRAME_TEXT_RE = re.compile(
    r"\b(?:1m|3m|5m|30m|2h|6h|8h|12h|3d|1w|1M)\b",
)
_SUPPORTED_TIMEFRAME_TEXT_RE = re.compile(r"\b(15m|1h|4h|1d)\b", re.IGNORECASE)
_TRADING_STYLE_TEXT_PATTERNS: tuple[tuple[re.Pattern[str], Timeframe], ...] = (
    (re.compile(r"\b(?:scalp|scalping)\b", re.IGNORECASE), Timeframe.M15),
    (
        re.compile(
            r"(?:\bintraday\b|\bintra\s+day\b|\bday\s*trade\b|\bday\s*trading\b|\bper\s+jam\b|\bhourly\b)",
            re.IGNORECASE,
        ),
        Timeframe.H1,
    ),
    (re.compile(r"\b(?:swing)\b", re.IGNORECASE), Timeframe.H4),
    (
        re.compile(r"(?:\bposition\s+trade\b|\bmacro\s+swing\b)", re.IGNORECASE),
        Timeframe.D1,
    ),
)


def _extract_timeframe_or_style_from_text(text: str) -> Timeframe | None:
    """Extract explicit primary timeframe or trading style ('scalp', 'intraday', 'swing')."""
    tf_match = _SUPPORTED_TIMEFRAME_TEXT_RE.search(text)
    if tf_match is not None:
        return Timeframe(tf_match.group(1).lower())
    for pattern, mapped_tf in _TRADING_STYLE_TEXT_PATTERNS:
        if pattern.search(text) is not None:
            return mapped_tf
    return None


_SYMBOL_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(r"\b(?:BTC/USDT:USDT|BTC/USDT|BTCUSDT|BTC|BITCOIN)\b", re.IGNORECASE),
        "BTC/USDT:USDT",
    ),
    (
        re.compile(r"\b(?:ETH/USDT:USDT|ETH/USDT|ETHUSDT|ETH|ETHEREUM)\b", re.IGNORECASE),
        "ETH/USDT:USDT",
    ),
    (
        re.compile(r"\b(?:SOL/USDT:USDT|SOL/USDT|SOLUSDT|SOL|SOLANA)\b", re.IGNORECASE),
        "SOL/USDT:USDT",
    ),
)

_USDT_PAIR_TEXT_RE = re.compile(r"\b([A-Z0-9]{2,12})/USDT\b", re.IGNORECASE)
_USDT_SUFFIX_TEXT_RE = re.compile(r"\b([A-Z0-9]{2,12})USDT\b", re.IGNORECASE)
_KOIN_CUE_TEXT_RE = re.compile(
    r"\b(?:koin|coin|token|pair|simbol)\s+([A-Za-z0-9]{2,12})\b",
    re.IGNORECASE,
)
_ACTION_CUE_TEXT_RE = re.compile(
    r"\b(?:analisa|analisis|analyze|cek|lihat|entry|setup|smc|plan)\s+"
    r"(?:koin\s+|coin\s+|token\s+|pair\s+|simbol\s+)?([A-Za-z0-9]{2,12})\b",
    re.IGNORECASE,
)
_UPPER_TICKER_RE = re.compile(r"\b((?=[A-Z0-9]*[A-Z])[A-Z0-9]{2,12})\b")
_NON_TICKER_ACRONYMS: frozenset[str] = frozenset(
    {
        "RSI",
        "EMA",
        "SMA",
        "MACD",
        "ATR",
        "ADX",
        "DMI",
        "SMC",
        "FVG",
        "OB",
        "BOS",
        "CHOCH",
        "SL",
        "TP",
        "TP1",
        "TP2",
        "TP3",
        "RR",
        "HTF",
        "LTF",
        "MTF",
        "OI",
        "USD",
        "USDT",
        "BUSD",
        "FDUSD",
        "USDC",
        "TUSD",
        "DAI",
        "EUR",
        "GBP",
        "JPY",
        "IDR",
        "UTC",
        "WIB",
        "READY",
        "DEGRADED",
        "INSUFFICIENT",
        "DATA",
        "WAIT",
        "SETUP",
        "LONG",
        "SHORT",
        "BUY",
        "SELL",
        "BULL",
        "BEAR",
        "MVP",
        "API",
        "HTML",
        "JSON",
        "SPOT",
        "PERP",
        "PERPETUAL",
        "FUTURES",
        "ALL",
        "NO",
        "INFO",
        "HELP",
        "OK",
        "YA",
        "DAN",
        "DI",
        "KE",
        "15M",
        "1H",
        "4H",
        "1D",
        "MOMENTUM",
        "MOMENTUMNYA",
        "KONDISI",
        "KONDISINYA",
        "STRUKTUR",
        "STRUKTURNYA",
        "HARGA",
        "HARGANYA",
        "PASAR",
        "ARAH",
        "TREN",
        "TREND",
        "TRENNYA",
        "RISIKO",
        "RISIKONYA",
        "LEVEL",
        "LEVELNYA",
        "POSISI",
        "AREA",
        "ZONA",
        "TARGET",
        "DETAIL",
        "FULL",
        "LAGI",
        "DULU",
        "DONG",
        "SEKARANG",
        "HARI",
        "MALAM",
        "PAGI",
        "SORE",
        "INI",
        "ITU",
        "NYA",
        "YANG",
        "UNTUK",
        "PADA",
        "DARI",
        "DENGAN",
        "ATAU",
        "JIKA",
        "KALAU",
        "KALO",
        "BISA",
        "GAK",
        "TIDAK",
        "BELUM",
        "SUDAH",
        "MASIH",
        "KENAPA",
        "MENGAPA",
        "BAGAIMANA",
        "GIMANA",
        "JELASKAN",
        "LANJUT",
        "KESIMPULAN",
        "BANDINGKAN",
        "PERBANDINGAN",
        "SUPPORT",
        "RESISTANCE",
        "LIKUIDITAS",
        "LIQUIDITY",
        "FUNDING",
        "VOLUME",
        "VOLATILITAS",
        "ORDER",
        "BLOCK",
        "SWING",
        "RANGE",
        "DEALING",
        "PREMIUM",
        "DISCOUNT",
        "EQUILIBRIUM",
        "KOIN",
        "COIN",
        "TOKEN",
        "PAIR",
        "SIMBOL",
        "BAGUS",
        "LAYAK",
        "SIAP",
        "JELEK",
        "OKE",
        "COBA",
        "BIKIN",
        "BIKININ",
        "BUAT",
        "BUATKAN",
        "CARI",
        "CARIIN",
        "ADA",
        "ENAK",
        "SIH",
        "NIH",
        "DEH",
        "KAH",
        "LAH",
        "PUNYA",
        "SCRIPT",
        "PINESCRIPT",
        "PINE",
        "SCAN",
        "SCANNER",
        "MARKET",
        "WATCHLIST",
        "RINCIAN",
        "LENGKAP",
        "PENJELASAN",
        "BEDANYA",
        "SAMA",
        "APA",
        "EFEK",
        "ARTINYA",
        "SAAT",
        "PERLU",
        "BANTU",
        "KASIH",
        "TOLONG",
        "DISETUP",
        "BARUSAN",
        "KEMARIN",
        "TERAKHIR",
        "BESAR",
        "KECIL",
        "NAIK",
        "TURUN",
        "BULLISH",
        "BEARISH",
        "TRADINGVIEW",
        "REKOMENDASI",
        "SARAN",
        "PROSPEK",
        "SCALP",
        "SCALPING",
        "INTRADAY",
        "INTRA",
        "DAY",
        "DAYTRADE",
        "HOURLY",
        "POSITION",
        "MACRO",
        "DAILY",
        "HARIAN",
        "JAM",
        "CEPAT",
        "PENDEK",
        "MENENGAH",
        "PANJANG",
        "MODE",
        "GAYA",
        "STYLE",
        "OPSI",
        "SCREEN",
        "SCREENING",
        "GAINER",
        "GAINERS",
        "LOSER",
        "LOSERS",
        "TOP",
        "ENGULFING",
        "MARUBOZU",
        "JUTA",
        "SINYAL",
        "PELUANG",
        "LIST",
        "DAFTAR",
    }
)
_NON_USDT_PAIR_RE = re.compile(r"\b([A-Z0-9]{2,12}/(?!USDT\b)[A-Z]{2,10})\b", re.IGNORECASE)
_SPOT_HINT_RE = re.compile(r"\b(?:spot)\b", re.IGNORECASE)
_COMPARE_WORD_RE = re.compile(
    r"\b(?:bandingkan|dibanding|compare|versus|vs|perbandingan)\b",
    re.IGNORECASE,
)
_PINE_INTENT_RE = re.compile(r"(?:\bpine\s*script\b|\bpinescript\b|\bpine\b)", re.IGNORECASE)
_DETAIL_INTENT_RE = re.compile(r"\b(?:detail|rincian|lengkap|full)\b", re.IGNORECASE)
_SCREEN_INTENT_RE = re.compile(
    r"(?:\bscreen(?:ing)?\b|\btop\s+gainers?\b|\btop\s+losers?\b|"
    r"\bengulfing\b|\bmarubozu\b|\bmomentum\s+15m\b|"
    r"\bvolume\s*(?:di\s*atas|>)\s*5\s*(?:juta|m)\b)",
    re.IGNORECASE,
)
_SCREEN_LONG_HINT_RE = re.compile(r"\b(?:gainer|gainers|long|buy|naik)\b", re.IGNORECASE)
_SCREEN_SHORT_HINT_RE = re.compile(r"\b(?:loser|losers|short|sell|turun)\b", re.IGNORECASE)
_SCAN_INTENT_RE = re.compile(
    r"(?:\b(?:ada\s+(?:token|koin|coin|setup|sinyal|peluang)|"
    r"(?:setup|sinyal)\s+apa|"
    r"cari(?:in)?\s+(?:token|koin|coin|setup|sinyal)|"
    r"token\s+apa|koin\s+apa|rekomendasi\s+(?:koin|token|setup)|"
    r"bisa\s+di\s*setup|siap\s+(?:di\s*setup|entry)|"
    r"scan(?:\s+market|\s+watchlist|\s+koin|\s+token)?)\b)",
    re.IGNORECASE,
)
_CONTEXT_TOPIC_RE = re.compile(
    r"\b(?:momentum|tren|trend|rsi|ema|macd|atr|adx|dmi|volume|volatilitas|"
    r"support|resistance|level|struktur|analisis|analisa|kondisi|bagaimana|"
    r"gimana|kenapa|mengapa|jelaskan|lanjut|kesimpulan|risiko|"
    r"entry|sl|tp|smc|fvg|order\s*block|setup|plan|likuiditas|liquidity|funding|oi|dxy|benchmark)\b",
    re.IGNORECASE,
)
_ORDER_OR_ADVICE_RE = re.compile(
    r"(?:\bpasang\s+order\b|\bexecute\s+order\b|\border\s+sekarang\b|"
    r"\bbeli\s+sekarang\b|\bjual\s+sekarang\b|\bbelikan\b|\bjualkan\b|"
    r"\balokasi\s+portofolio\b|\bberapa\s+persen\s+modal\b|\bwajib\s+beli\b|"
    r"\bwajib\s+jual\b|\ball[\s-]+in\b)",
    re.IGNORECASE,
)
_HEADER_BRACKET_RE = re.compile(r"^\[([^\]]+)\s+(15m|1h|4h|1d)\]", re.IGNORECASE)
_HEADER_HTML_RE = re.compile(r"^<b>([^ (]+(?:\s*,\s*[^ (]+)*)\s*\((15m|1h|4h|1d)\)", re.IGNORECASE)
_HEADER_RE = _HEADER_BRACKET_RE


class CommandParseError(ValueError):
    """Raised when command syntax or arguments are invalid, with actionable guidance."""


class AnalyzeIntent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["analyze"] = "analyze"
    symbol: str
    timeframe: Timeframe
    detail: bool = False


class CompareIntent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["compare"] = "compare"
    symbols: tuple[str, ...]
    timeframe: Timeframe


class PineIntent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["pine"] = "pine"
    symbol: str
    timeframe: Timeframe


class ScanIntent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["scan"] = "scan"
    timeframe: Timeframe = Timeframe.H4
    timeframes: tuple[Timeframe, ...] | None = None
    symbols: tuple[str, ...] | None = None


class ScreenIntent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["screen"] = "screen"
    side: Literal["all", "long", "short"] = "all"
    min_volume_usd: float = 5_000_000.0


class ChatQaIntent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["chat_qa"] = "chat_qa"
    question: str


class RefusalIntent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["refusal"] = "refusal"
    message: str
    is_hard_refusal: bool = False


FollowupIntent = (
    AnalyzeIntent
    | CompareIntent
    | PineIntent
    | ScanIntent
    | ScreenIntent
    | ChatQaIntent
    | RefusalIntent
)


def _clean_symbol_error(raw: str, detail: str) -> str:
    return (
        f"Simbol '{raw}' tidak valid atau belum didukung ({detail}). "
        "Gunakan simbol perpetual Binance /USDT:USDT (contoh: BTC, ETH, SOL, DOGE, SUI)."
    )


def parse_analyze_args(
    args: Sequence[str],
    *,
    default_timeframe: Timeframe = Timeframe.H4,
) -> AnalyzeCommandArgs:
    """Parse `/analyze <symbol> [timeframe] [detail]` arguments."""
    if not args or len(args) > 3:
        raise CommandParseError(
            "Format salah. Gunakan: /analyze <SIMBOL> [15m|1h|4h|1d] [detail] "
            "(contoh: /analyze BTC 4h)"
        )

    raw_symbol = args[0]
    raw_tf = default_timeframe.value
    detail = False

    if len(args) == 2:
        second = args[1].strip().lower()
        if second in ("detail", "full"):
            detail = True
        else:
            raw_tf = second
    elif len(args) == 3:
        second = args[1].strip().lower()
        third = args[2].strip().lower()
        if third in ("detail", "full"):
            detail = True
            raw_tf = second
        else:
            raise CommandParseError(
                "Format salah. Gunakan: /analyze <SIMBOL> [15m|1h|4h|1d] [detail] "
                "(contoh: /analyze BTC 4h)"
            )

    resolved_tf = resolve_primary_timeframe(raw_tf)
    if resolved_tf is None:
        raise CommandParseError(
            f"Timeframe/gaya '{raw_tf}' tidak didukung. Pilihan yang tersedia: "
            "15m (scalp), 1h (intraday), 4h (swing), 1d."
        )

    try:
        return AnalyzeCommandArgs(
            symbol=raw_symbol,
            timeframe=resolved_tf,
            detail=detail,
        )
    except ValidationError as exc:
        raise CommandParseError(_clean_symbol_error(raw_symbol, str(exc))) from None


def parse_compare_args(
    args: Sequence[str],
    *,
    default_timeframe: Timeframe = Timeframe.H4,
) -> CompareCommandArgs:
    """Parse `/compare <sym1> <sym2> [sym3] [timeframe|style]` arguments."""
    if len(args) < 2 or len(args) > 4:
        raise CommandParseError(
            "Format salah. Gunakan: /compare <SIMBOL1> <SIMBOL2> [SIMBOL3] "
            "[15m|1h|4h|1d|scalp|intraday|swing] (contoh: /compare BTC ETH 1d)"
        )

    last_lower = args[-1].strip().lower()
    resolved_last_tf = resolve_primary_timeframe(last_lower)
    if resolved_last_tf is not None:
        symbol_tokens = list(args[:-1])
        tf_enum = resolved_last_tf
    elif _TIMEFRAME_LIKE_RE.match(last_lower):
        raise CommandParseError(
            f"Timeframe '{args[-1]}' tidak didukung. Pilihan timeframe yang tersedia: "
            "15m (scalp), 1h (intraday), 4h (swing), 1d."
        )
    else:
        symbol_tokens = list(args)
        tf_enum = default_timeframe

    if len(symbol_tokens) < 2 or len(symbol_tokens) > 3:
        raise CommandParseError(
            "Perbandingan membutuhkan 2 atau 3 simbol yang berbeda (contoh: /compare BTC ETH 1d)."
        )

    try:
        return CompareCommandArgs(
            symbols=tuple(symbol_tokens),
            timeframe=tf_enum,
        )
    except ValidationError as exc:
        raise CommandParseError(
            "Argumen simbol tidak valid atau duplikat. "
            f"Gunakan 2-3 simbol spot /USDT yang unik ({exc})."
        ) from None


def parse_watchlist_args(args: Sequence[str]) -> WatchlistCommandArgs:
    """Parse `/watchlist [list | add <symbol> | remove <symbol>]`."""
    if not args:
        return WatchlistCommandArgs(action=WatchlistAction.LIST, symbol=None)

    action_raw = args[0].strip().lower()
    if action_raw == "list" and len(args) == 1:
        return WatchlistCommandArgs(action=WatchlistAction.LIST, symbol=None)

    if action_raw in ("add", "remove") and len(args) == 2:
        try:
            return WatchlistCommandArgs(
                action=WatchlistAction(action_raw),
                symbol=args[1],
            )
        except ValidationError as exc:
            raise CommandParseError(_clean_symbol_error(args[1], str(exc))) from None

    raise CommandParseError(
        "Format salah. Gunakan:\n"
        "- /watchlist\n"
        "- /watchlist add <SIMBOL>\n"
        "- /watchlist remove <SIMBOL>"
    )


def _extract_mvp_symbols(text: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for pattern, canonical in _SYMBOL_ALIASES:
        for match in pattern.finditer(text):
            found.append((match.start(), canonical))

    for pair_re in (
        _USDT_PAIR_TEXT_RE,
        _USDT_SUFFIX_TEXT_RE,
        _KOIN_CUE_TEXT_RE,
        _ACTION_CUE_TEXT_RE,
    ):
        for match in pair_re.finditer(text):
            token = match.group(1).upper()
            if token in _NON_TICKER_ACRONYMS:
                continue
            try:
                found.append((match.start(), resolve_default_perp_symbol(token).canonical_symbol))
            except ValueError:
                continue

    for match in _UPPER_TICKER_RE.finditer(text):
        token = match.group(1)
        if token in _NON_TICKER_ACRONYMS:
            continue
        try:
            found.append((match.start(), resolve_default_perp_symbol(token).canonical_symbol))
        except ValueError:
            continue

    found.sort(key=lambda item: item[0])
    ordered: list[str] = []
    for _, sym in found:
        if sym not in ordered:
            ordered.append(sym)
    return ordered


def _extract_recent_context(
    recent_messages: Sequence[StoredMessage],
    default_timeframe: Timeframe,
) -> tuple[list[str], Timeframe]:
    recent_symbols: list[str] = []
    recent_tf = default_timeframe
    tf_found = False

    for msg in reversed(recent_messages):
        content_stripped = msg.content.strip()
        header_match = _HEADER_BRACKET_RE.match(content_stripped) or _HEADER_HTML_RE.match(
            content_stripped
        )
        if header_match is not None:
            extracted_any = False
            for raw_part in header_match.group(1).split(","):
                part_clean = raw_part.strip()
                if part_clean.upper() in _NON_TICKER_ACRONYMS:
                    continue
                try:
                    canonical = resolve_default_perp_symbol(part_clean).canonical_symbol
                except ValueError:
                    continue
                extracted_any = True
                if canonical not in recent_symbols:
                    recent_symbols.append(canonical)
            if extracted_any and not tf_found:
                recent_tf = Timeframe(header_match.group(2).lower())
                tf_found = True
            if recent_symbols:
                break
            continue

        msg_syms = _extract_mvp_symbols(msg.content)
        if msg_syms:
            for sym in msg_syms:
                if sym not in recent_symbols:
                    recent_symbols.append(sym)
            if not tf_found:
                extracted_tf = _extract_timeframe_or_style_from_text(msg.content)
                if extracted_tf is not None:
                    recent_tf = extracted_tf
            break

    return recent_symbols, recent_tf


def resolve_followup_intent(
    text: str,
    *,
    recent_messages: Sequence[StoredMessage],
    default_timeframe: Timeframe = Timeframe.H4,
) -> FollowupIntent:
    """Resolve a bounded natural-language message into an explicit backend analysis intent."""
    cleaned = text.strip()
    if not cleaned:
        return RefusalIntent(
            message="Pesan kosong. Gunakan /analyze <SIMBOL> [15m|1h|4h|1d] atau /help.",
            is_hard_refusal=True,
        )
    if len(cleaned) > MAX_USER_MESSAGE_CHARS:
        return RefusalIntent(
            message=(
                f"Pesan terlalu panjang (maksimal {MAX_USER_MESSAGE_CHARS} karakter). "
                "Ringkas pertanyaan Anda atau gunakan /analyze /compare."
            ),
            is_hard_refusal=True,
        )

    if _ORDER_OR_ADVICE_RE.search(cleaned):
        return RefusalIntent(
            message=(
                "Bot ini hanya asisten riset dan edukasi. Eksekusi order, instruksi beli/jual "
                "wajib, dan alokasi portofolio pribadi berada di luar cakupan layanan."
            ),
            is_hard_refusal=True,
        )

    if _SPOT_HINT_RE.search(cleaned):
        return RefusalIntent(
            message=(
                "Analisis difokuskan pada kontrak Perpetual Futures Binance (BASE/USDT:USDT) "
                "dan tidak melayani pasar spot atau eksekusi order langsung."
            ),
            is_hard_refusal=True,
        )

    non_usdt = _NON_USDT_PAIR_RE.findall(cleaned)
    if non_usdt:
        return RefusalIntent(
            message=(
                f"Pasangan '{non_usdt[0]}' belum didukung. Saat ini analisis difokuskan pada "
                "kontrak perpetual Binance /USDT:USDT (contoh: BTC, DOGE, SUI)."
            ),
            is_hard_refusal=True,
        )

    if _BAD_TIMEFRAME_TEXT_RE.search(cleaned) and not _SUPPORTED_TIMEFRAME_TEXT_RE.search(cleaned):
        return RefusalIntent(
            message=("Timeframe tersebut belum didukung. Gunakan timeframe 15m, 1h, 4h, atau 1d."),
            is_hard_refusal=True,
        )

    recent_symbols, recent_tf = _extract_recent_context(recent_messages, default_timeframe)
    explicit_tf = _extract_timeframe_or_style_from_text(cleaned)
    new_query_tf = explicit_tf if explicit_tf is not None else default_timeframe
    followup_tf = explicit_tf if explicit_tf is not None else recent_tf

    mentioned = _extract_mvp_symbols(cleaned)
    is_compare = _COMPARE_WORD_RE.search(cleaned) is not None
    is_pine = _PINE_INTENT_RE.search(cleaned) is not None
    is_detail = _DETAIL_INTENT_RE.search(cleaned) is not None

    if not mentioned and _SCREEN_INTENT_RE.search(cleaned):
        has_long = _SCREEN_LONG_HINT_RE.search(cleaned) is not None
        has_short = _SCREEN_SHORT_HINT_RE.search(cleaned) is not None
        screen_side: Literal["all", "long", "short"] = "all"
        if has_long and not has_short:
            screen_side = "long"
        elif has_short and not has_long:
            screen_side = "short"
        return ScreenIntent(side=screen_side)

    if not mentioned and _SCAN_INTENT_RE.search(cleaned):
        if explicit_tf is not None:
            return ScanIntent(timeframe=explicit_tf)
        return ScanIntent(
            timeframe=new_query_tf,
            timeframes=MULTI_STYLE_SCAN_TIMEFRAMES,
        )

    if is_pine:
        if mentioned:
            return PineIntent(symbol=mentioned[0], timeframe=new_query_tf)
        if recent_symbols:
            return PineIntent(symbol=recent_symbols[0], timeframe=followup_tf)

    if len(mentioned) >= 2:
        return CompareIntent(symbols=tuple(mentioned[:3]), timeframe=new_query_tf)

    if len(mentioned) == 1:
        if is_compare and recent_symbols:
            prior = next((sym for sym in recent_symbols if sym != mentioned[0]), None)
            if prior is not None:
                return CompareIntent(symbols=(prior, mentioned[0]), timeframe=followup_tf)
        return AnalyzeIntent(symbol=mentioned[0], timeframe=new_query_tf, detail=is_detail)

    if recent_symbols and (_CONTEXT_TOPIC_RE.search(cleaned) or is_compare or is_detail):
        if is_compare and len(recent_symbols) >= 2:
            return CompareIntent(symbols=tuple(recent_symbols[:2]), timeframe=followup_tf)
        return AnalyzeIntent(symbol=recent_symbols[0], timeframe=followup_tf, detail=is_detail)

    return RefusalIntent(
        message=(
            "Saya belum dapat menentukan simbol atau perintah yang dimaksud. "
            "Cukup sebutkan koinnya (mis. 'analisa BTC sekarang'), tanya "
            "'ada token yang bisa disetup gak?', atau ketik /help."
        ),
        is_hard_refusal=False,
    )
