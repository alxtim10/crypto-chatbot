"""Command argument parsers and deterministic follow-up intent resolution."""

from __future__ import annotations

import re
from collections.abc import Collection, Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from crypto_chatbot.market_data.symbols import (
    AMBIGUOUS_PERP_TICKERS,
    DEFAULT_PERP_UNIVERSE,
    resolve_default_perp_symbol,
)
from crypto_chatbot.market_data.whale_provider import detect_network_type
from crypto_chatbot.schemas.commands import (
    AnalyzeCommandArgs,
    CompareCommandArgs,
    WatchlistAction,
    WatchlistCommandArgs,
    WhaleAction,
    WhaleCommandArgs,
)
from crypto_chatbot.schemas.llm import RoutingTrace
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
_CASHTAG_TEXT_RE = re.compile(r"(?<!\w)\$([A-Za-z][A-Za-z0-9]{1,11})\b")
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
        "KAPAN",
        "AJA",
        "SAJA",
        "YUK",
        "BRO",
        "MIN",
        "BOT",
        "BANG",
        "GAN",
        "HALO",
        "HAI",
        "SIANG",
        "THANKS",
        "MAKASIH",
        "TERIMA",
        "MANTAP",
        "KEREN",
        "WADUH",
        "TAPI",
        "SOALNYA",
        "MENURUT",
        "KAMU",
        "ANDA",
        "SAYA",
        "AKU",
        "GUE",
        "CRYPTO",
        "KRIPTO",
        "TRADING",
        "TRADE",
        "CHART",
        "CANDLE",
        "TIMEFRAME",
        "STRATEGI",
        "MINDSET",
        "PSIKOLOGI",
        "MODAL",
        "LOSS",
        "PROFIT",
        "CUAN",
        "NYANGKUT",
        "LEVERAGE",
        "MARGIN",
        "FUTURE",
        "BESOK",
        "NANTI",
        "TADI",
        "BARU",
        "NGGAK",
        "ENGGAK",
        "JUGA",
        "BGT",
        "BANGET",
        "KAN",
        "YAH",
        "MANA",
        "SIAPA",
        "BERAPA",
        "PAKAI",
        "PAKE",
        "BINGUNG",
        "SUSAH",
        "GAMPANG",
        "ENAKNYA",
        "BAIKNYA",
        "COCOK",
    }
)
_NON_USDT_PAIR_RE = re.compile(r"\b([A-Z0-9]{2,12})/(?!USDT\b)([A-Z]{2,10})\b", re.IGNORECASE)
_KNOWN_NON_USDT_QUOTES: frozenset[str] = frozenset(
    {
        "AUD",
        "BNB",
        "BRL",
        "BTC",
        "BUSD",
        "DAI",
        "ETH",
        "EUR",
        "FDUSD",
        "GBP",
        "IDR",
        "JPY",
        "KRW",
        "NGN",
        "PLN",
        "RON",
        "RUB",
        "TRY",
        "TUSD",
        "UAH",
        "USD",
        "USDC",
        "ZAR",
    }
)
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
_WHALE_INTENT_RE = re.compile(
    r"(?:\bwhale\b|\bpaus\b|\bsmart\s*money\b|\bdompet\b|\bwallet\b)",
    re.IGNORECASE,
)
_WALLET_ADDRESS_RE = re.compile(
    r"\b(0x[a-fA-F0-9]{40}|1[a-km-zA-HJ-NP-Z1-9]{25,34}|3[a-km-zA-HJ-NP-Z1-9]{25,34}|bc1[a-zA-HJ-NP-Z0-9]{39,59})\b",
    re.IGNORECASE,
)
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
_CONCEPTUAL_QA_RE = re.compile(
    r"(?:\b(?:apa\s+itu|apa\s+arti|apa\s+bedanya|artinya(?:\s+apa)?|perbedaan|bedanya|"
    r"maksudnya|gimana\s+cara|bagaimana\s+cara|cara\s+atur|cara\s+pakai|"
    r"cara\s+hitung|cara\s+kerja|kenapa|mengapa|jelaskan|"
    r"tips\s+trading|psikologi\s+trading)\b|"
    r"\bfunding(?:\s+rate)?\b.*\b8h\b|\b8h\b.*\bfunding(?:\s+rate)?\b)",
    re.IGNORECASE,
)


def _is_conceptual_question(text: str) -> bool:
    return _CONCEPTUAL_QA_RE.search(text) is not None


def _find_non_usdt_pairs(text: str, universe: frozenset[str]) -> list[str]:
    matches: list[str] = []
    for match in _NON_USDT_PAIR_RE.finditer(text):
        base = match.group(1).upper()
        quote = match.group(2).upper()
        if quote in _KNOWN_NON_USDT_QUOTES and _in_perp_universe(base, universe):
            matches.append(f"{match.group(1)}/{match.group(2)}")
    return matches


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


class WhaleIntent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["whale"] = "whale"
    action: WhaleAction = WhaleAction.OVERVIEW
    symbol: str | None = None
    address: str | None = None
    label: str | None = None


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
    | WhaleIntent
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


def parse_whale_args(args: Sequence[str]) -> WhaleCommandArgs:
    """Parse arguments for the /whale command."""
    if not args:
        return WhaleCommandArgs(action=WhaleAction.OVERVIEW, symbol="BTC")

    first = args[0].strip()
    first_lower = first.lower()

    if first_lower == "list":
        return WhaleCommandArgs(action=WhaleAction.LIST)

    if first_lower == "add":
        if len(args) < 2:
            raise CommandParseError("Penggunaan: /whale add <alamat_wallet> [label]")
        addr = args[1].strip()
        net = detect_network_type(addr)
        if net is None:
            raise CommandParseError(
                f"Alamat wallet '{addr}' tidak valid. "
                "Harus format EVM (0x...), Bitcoin, atau Solana."
            )
        label = " ".join(args[2:]).strip() if len(args) > 2 else f"Whale-{addr[:6]}"
        return WhaleCommandArgs(action=WhaleAction.ADD, address=addr, label=label)

    if first_lower == "remove":
        if len(args) < 2:
            raise CommandParseError("Penggunaan: /whale remove <alamat_wallet>")
        addr = args[1].strip()
        return WhaleCommandArgs(action=WhaleAction.REMOVE, address=addr)

    # Check if first argument is a wallet address
    net = detect_network_type(first)
    if net is not None:
        return WhaleCommandArgs(action=WhaleAction.WALLET, address=first)

    # Otherwise treat as ticker symbol
    clean_sym = first.upper()
    return WhaleCommandArgs(action=WhaleAction.OVERVIEW, symbol=clean_sym)


def _in_perp_universe(base: str, universe: frozenset[str]) -> bool:
    upper = base.upper()
    return upper in universe or f"1000{upper}" in universe


def _extract_mvp_symbols(
    text: str,
    *,
    perp_universe: Collection[str] | None = None,
) -> list[str]:
    universe = (
        frozenset(s.upper() for s in perp_universe)
        if perp_universe is not None
        else DEFAULT_PERP_UNIVERSE
    )
    has_tf = _extract_timeframe_or_style_from_text(text) is not None
    found: list[tuple[int, str]] = []

    for pattern, canonical in _SYMBOL_ALIASES:
        base = canonical.split("/")[0]
        if not _in_perp_universe(base, universe):
            continue
        for match in pattern.finditer(text):
            found.append((match.start(), canonical))

    # Strong explicit cues: $TICKER, TICKER/USDT, TICKERUSDT
    for strong_re in (_CASHTAG_TEXT_RE, _USDT_PAIR_TEXT_RE, _USDT_SUFFIX_TEXT_RE):
        for match in strong_re.finditer(text):
            token = match.group(1).upper()
            if token in _NON_TICKER_ACRONYMS or not _in_perp_universe(token, universe):
                continue
            try:
                found.append((match.start(), resolve_default_perp_symbol(token).canonical_symbol))
            except ValueError:
                continue

    # Cue verbs / nouns: require universe membership; ambiguous tickers must be uppercase.
    for cue_re in (_KOIN_CUE_TEXT_RE, _ACTION_CUE_TEXT_RE):
        for match in cue_re.finditer(text):
            raw_token = match.group(1)
            token = raw_token.upper()
            if token in _NON_TICKER_ACRONYMS or not _in_perp_universe(token, universe):
                continue
            if token in AMBIGUOUS_PERP_TICKERS and not raw_token.isupper():
                continue
            try:
                found.append((match.start(), resolve_default_perp_symbol(token).canonical_symbol))
            except ValueError:
                continue

    # Bare uppercase tickers: must be in universe; ambiguous tickers also need a timeframe cue.
    for match in _UPPER_TICKER_RE.finditer(text):
        token = match.group(1)
        if token in _NON_TICKER_ACRONYMS or not _in_perp_universe(token, universe):
            continue
        if token in AMBIGUOUS_PERP_TICKERS and not has_tf:
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
    *,
    perp_universe: Collection[str] | None = None,
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

        msg_syms = _extract_mvp_symbols(msg.content, perp_universe=perp_universe)
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


def resolve_followup_intent_with_trace(
    text: str,
    *,
    recent_messages: Sequence[StoredMessage],
    default_timeframe: Timeframe = Timeframe.H4,
    perp_universe: Collection[str] | None = None,
) -> tuple[FollowupIntent, RoutingTrace]:
    """Resolve a bounded natural-language message and return PII-free routing telemetry."""
    cleaned = text.strip()
    if not cleaned:
        return (
            RefusalIntent(
                message="Pesan kosong. Gunakan /analyze <SIMBOL> [15m|1h|4h|1d] atau /help.",
                is_hard_refusal=True,
            ),
            RoutingTrace(
                layer="layer1",
                intent="refusal_hard",
                confidence=1.0,
                fallback_reason="empty_message",
            ),
        )
    if len(cleaned) > MAX_USER_MESSAGE_CHARS:
        return (
            RefusalIntent(
                message=(
                    f"Pesan terlalu panjang (maksimal {MAX_USER_MESSAGE_CHARS} karakter). "
                    "Ringkas pertanyaan Anda atau gunakan /analyze /compare."
                ),
                is_hard_refusal=True,
            ),
            RoutingTrace(
                layer="layer1",
                intent="refusal_hard",
                confidence=1.0,
                fallback_reason="message_too_long",
            ),
        )

    universe = (
        frozenset(s.upper() for s in perp_universe)
        if perp_universe is not None
        else DEFAULT_PERP_UNIVERSE
    )
    is_conceptual = _is_conceptual_question(cleaned)

    if not is_conceptual and _ORDER_OR_ADVICE_RE.search(cleaned):
        return (
            RefusalIntent(
                message=(
                    "Bot ini hanya asisten riset dan edukasi. Eksekusi order, instruksi beli/jual "
                    "wajib, dan alokasi portofolio pribadi berada di luar cakupan layanan."
                ),
                is_hard_refusal=True,
            ),
            RoutingTrace(
                layer="layer1",
                intent="refusal_hard",
                confidence=1.0,
                fallback_reason="order_or_advice_policy",
            ),
        )

    if not is_conceptual and _SPOT_HINT_RE.search(cleaned):
        return (
            RefusalIntent(
                message=(
                    "Analisis difokuskan pada kontrak Perpetual Futures Binance (BASE/USDT:USDT) "
                    "dan tidak melayani pasar spot atau eksekusi order langsung."
                ),
                is_hard_refusal=True,
            ),
            RoutingTrace(
                layer="layer1",
                intent="refusal_hard",
                confidence=1.0,
                fallback_reason="spot_market_unsupported",
            ),
        )

    non_usdt = _find_non_usdt_pairs(cleaned, universe)
    if non_usdt:
        return (
            RefusalIntent(
                message=(
                    f"Pasangan '{non_usdt[0]}' belum didukung. Saat ini analisis difokuskan pada "
                    "kontrak perpetual Binance /USDT:USDT (contoh: BTC, DOGE, SUI)."
                ),
                is_hard_refusal=True,
            ),
            RoutingTrace(
                layer="layer1",
                intent="refusal_hard",
                confidence=1.0,
                fallback_reason="non_usdt_pair_unsupported",
            ),
        )

    if (
        not is_conceptual
        and _BAD_TIMEFRAME_TEXT_RE.search(cleaned)
        and not _SUPPORTED_TIMEFRAME_TEXT_RE.search(cleaned)
    ):
        return (
            RefusalIntent(
                message=(
                    "Timeframe tersebut belum didukung. Gunakan timeframe 15m, 1h, 4h, atau 1d."
                ),
                is_hard_refusal=True,
            ),
            RoutingTrace(
                layer="layer1",
                intent="refusal_hard",
                confidence=1.0,
                fallback_reason="unsupported_timeframe",
            ),
        )

    recent_symbols, recent_tf = _extract_recent_context(
        recent_messages,
        default_timeframe,
        perp_universe=perp_universe,
    )
    explicit_tf = _extract_timeframe_or_style_from_text(cleaned)
    new_query_tf = explicit_tf if explicit_tf is not None else default_timeframe
    followup_tf = explicit_tf if explicit_tf is not None else recent_tf

    mentioned = _extract_mvp_symbols(cleaned, perp_universe=perp_universe)
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
        return (
            ScreenIntent(side=screen_side),
            RoutingTrace(layer="layer1", intent="screen", confidence=1.0),
        )

    if not mentioned and _SCAN_INTENT_RE.search(cleaned):
        if explicit_tf is not None:
            return (
                ScanIntent(timeframe=explicit_tf),
                RoutingTrace(layer="layer1", intent="scan", confidence=1.0),
            )
        return (
            ScanIntent(
                timeframe=new_query_tf,
                timeframes=MULTI_STYLE_SCAN_TIMEFRAMES,
            ),
            RoutingTrace(layer="layer1", intent="scan", confidence=1.0),
        )

    addr_match = _WALLET_ADDRESS_RE.search(cleaned)
    if addr_match is not None:
        extracted_addr = addr_match.group(1)
        if re.search(r"\b(?:tambah|add)\b", cleaned, re.IGNORECASE):
            return (
                WhaleIntent(action=WhaleAction.ADD, address=extracted_addr, label="User Tracked"),
                RoutingTrace(layer="layer1", intent="whale", confidence=1.0),
            )
        if re.search(r"\b(?:hapus|remove|delete)\b", cleaned, re.IGNORECASE):
            return (
                WhaleIntent(action=WhaleAction.REMOVE, address=extracted_addr),
                RoutingTrace(layer="layer1", intent="whale", confidence=1.0),
            )
        return (
            WhaleIntent(action=WhaleAction.WALLET, address=extracted_addr),
            RoutingTrace(layer="layer1", intent="whale", confidence=1.0),
        )

    if _WHALE_INTENT_RE.search(cleaned):
        if re.search(r"\b(?:list|daftar|watchlist)\b", cleaned, re.IGNORECASE):
            return (
                WhaleIntent(action=WhaleAction.LIST),
                RoutingTrace(layer="layer1", intent="whale", confidence=1.0),
            )
        target_sym = mentioned[0] if mentioned else (recent_symbols[0] if recent_symbols else "BTC")
        return (
            WhaleIntent(action=WhaleAction.OVERVIEW, symbol=target_sym),
            RoutingTrace(
                layer="layer1",
                detected_symbol=target_sym,
                intent="whale",
                confidence=1.0,
            ),
        )

    if is_pine:
        if mentioned:
            return (
                PineIntent(symbol=mentioned[0], timeframe=new_query_tf),
                RoutingTrace(
                    layer="layer1",
                    detected_symbol=mentioned[0],
                    intent="pine",
                    confidence=1.0,
                ),
            )
        if recent_symbols:
            return (
                PineIntent(symbol=recent_symbols[0], timeframe=followup_tf),
                RoutingTrace(
                    layer="layer1",
                    detected_symbol=recent_symbols[0],
                    intent="pine",
                    confidence=1.0,
                ),
            )

    if len(mentioned) >= 2:
        syms = tuple(mentioned[:3])
        return (
            CompareIntent(symbols=syms, timeframe=new_query_tf),
            RoutingTrace(
                layer="layer1",
                detected_symbol=",".join(syms),
                intent="compare",
                confidence=1.0,
            ),
        )

    if len(mentioned) == 1:
        if is_compare and recent_symbols:
            prior = next((sym for sym in recent_symbols if sym != mentioned[0]), None)
            if prior is not None:
                syms = (prior, mentioned[0])
                return (
                    CompareIntent(symbols=syms, timeframe=followup_tf),
                    RoutingTrace(
                        layer="layer1",
                        detected_symbol=",".join(syms),
                        intent="compare",
                        confidence=1.0,
                    ),
                )
        return (
            AnalyzeIntent(symbol=mentioned[0], timeframe=new_query_tf, detail=is_detail),
            RoutingTrace(
                layer="layer1",
                detected_symbol=mentioned[0],
                intent="analyze_detail" if is_detail else "analyze",
                confidence=1.0,
            ),
        )

    if (
        recent_symbols
        and (_CONTEXT_TOPIC_RE.search(cleaned) or is_compare or is_detail)
        and _CONCEPTUAL_QA_RE.search(cleaned) is None
    ):
        if is_compare and len(recent_symbols) >= 2:
            syms = tuple(recent_symbols[:2])
            return (
                CompareIntent(symbols=syms, timeframe=followup_tf),
                RoutingTrace(
                    layer="layer1",
                    detected_symbol=",".join(syms),
                    intent="compare",
                    confidence=1.0,
                ),
            )
        return (
            AnalyzeIntent(symbol=recent_symbols[0], timeframe=followup_tf, detail=is_detail),
            RoutingTrace(
                layer="layer1",
                detected_symbol=recent_symbols[0],
                intent="analyze_detail" if is_detail else "analyze",
                confidence=1.0,
            ),
        )

    return (
        RefusalIntent(
            message=(
                "Saya belum dapat menentukan simbol atau perintah yang dimaksud. "
                "Cukup sebutkan koinnya (mis. 'analisa BTC sekarang'), tanya "
                "'ada token yang bisa disetup gak?', atau ketik /help."
            ),
            is_hard_refusal=False,
        ),
        RoutingTrace(
            layer="layer1",
            intent="refusal_soft",
            confidence=0.0,
            fallback_reason="layer1_unmatched",
        ),
    )


def resolve_followup_intent(
    text: str,
    *,
    recent_messages: Sequence[StoredMessage],
    default_timeframe: Timeframe = Timeframe.H4,
    perp_universe: Collection[str] | None = None,
) -> FollowupIntent:
    """Resolve a bounded natural-language message into an explicit backend analysis intent."""
    intent, _ = resolve_followup_intent_with_trace(
        text,
        recent_messages=recent_messages,
        default_timeframe=default_timeframe,
        perp_universe=perp_universe,
    )
    return intent
