"""Command argument parsers and deterministic follow-up intent resolution."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError

from crypto_chatbot.market_data.symbols import resolve_mvp_spot_symbol
from crypto_chatbot.schemas.commands import (
    AnalyzeCommandArgs,
    CompareCommandArgs,
    WatchlistAction,
    WatchlistCommandArgs,
)
from crypto_chatbot.schemas.market import Timeframe
from crypto_chatbot.storage.repositories import StoredMessage

MAX_USER_MESSAGE_CHARS = 1000

_SUPPORTED_TFS = frozenset(tf.value for tf in Timeframe)
_TIMEFRAME_LIKE_RE = re.compile(r"^\d+[mhdwM]$")
_BAD_TIMEFRAME_TEXT_RE = re.compile(
    r"\b(?:1m|3m|5m|15m|30m|2h|6h|8h|12h|3d|1w|1M)\b",
)
_SUPPORTED_TIMEFRAME_TEXT_RE = re.compile(r"\b(1h|4h|1d)\b", re.IGNORECASE)

_SYMBOL_ALIASES: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\b(?:BTC/USDT|BTCUSDT|BTC|BITCOIN)\b", re.IGNORECASE), "BTC/USDT"),
    (re.compile(r"\b(?:ETH/USDT|ETHUSDT|ETH|ETHEREUM)\b", re.IGNORECASE), "ETH/USDT"),
    (re.compile(r"\b(?:SOL/USDT|SOLUSDT|SOL|SOLANA)\b", re.IGNORECASE), "SOL/USDT"),
)

_UNSUPPORTED_PAIR_RE = re.compile(r"\b([A-Z]{2,10}/USDT(?::USDT)?)\b", re.IGNORECASE)
_PERPETUAL_HINT_RE = re.compile(r"\b(?:perp|perpetual|futures)\b|\.P\b", re.IGNORECASE)
_COMPARE_WORD_RE = re.compile(
    r"\b(?:bandingkan|dibanding|compare|versus|vs|perbandingan)\b",
    re.IGNORECASE,
)
_CONTEXT_TOPIC_RE = re.compile(
    r"\b(?:momentum|tren|trend|rsi|ema|macd|atr|adx|dmi|volume|volatilitas|"
    r"support|resistance|level|struktur|analisis|analisa|kondisi|bagaimana|"
    r"gimana|kenapa|mengapa|jelaskan|lanjut|kesimpulan|risiko)\b",
    re.IGNORECASE,
)
_ORDER_OR_ADVICE_RE = re.compile(
    r"(?:\bpasang\s+order\b|\bexecute\s+order\b|\border\s+sekarang\b|"
    r"\bbeli\s+sekarang\b|\bjual\s+sekarang\b|\bbelikan\b|\bjualkan\b|"
    r"\balokasi\s+portofolio\b|\bberapa\s+persen\s+modal\b|\bwajib\s+beli\b|"
    r"\bwajib\s+jual\b|\ball[\s-]+in\b)",
    re.IGNORECASE,
)
_HEADER_RE = re.compile(r"^\[([^\]]+)\s+(1h|4h|1d)\]", re.IGNORECASE)


class CommandParseError(ValueError):
    """Raised when command syntax or arguments are invalid, with actionable guidance."""


class AnalyzeIntent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["analyze"] = "analyze"
    symbol: str
    timeframe: Timeframe


class CompareIntent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["compare"] = "compare"
    symbols: tuple[str, ...]
    timeframe: Timeframe


class RefusalIntent(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: Literal["refusal"] = "refusal"
    message: str


FollowupIntent = AnalyzeIntent | CompareIntent | RefusalIntent


def _clean_symbol_error(raw: str, detail: str) -> str:
    return (
        f"Simbol '{raw}' tidak valid atau belum didukung ({detail}). "
        "Simbol yang didukung saat ini hanya spot: BTC, ETH, SOL."
    )


def parse_analyze_args(
    args: Sequence[str],
    *,
    default_timeframe: Timeframe = Timeframe.H4,
) -> AnalyzeCommandArgs:
    """Parse `/analyze <symbol> [timeframe]` arguments."""
    if not args or len(args) > 2:
        raise CommandParseError(
            "Format salah. Gunakan: /analyze <SIMBOL> [1h|4h|1d] (contoh: /analyze BTC 4h)"
        )

    raw_symbol = args[0]
    raw_tf = args[1] if len(args) == 2 else default_timeframe.value

    if raw_tf.strip().lower() not in _SUPPORTED_TFS:
        raise CommandParseError(
            f"Timeframe '{raw_tf}' tidak didukung. Pilihan timeframe yang tersedia: 1h, 4h, 1d."
        )

    try:
        return AnalyzeCommandArgs(symbol=raw_symbol, timeframe=Timeframe(raw_tf.strip().lower()))
    except ValidationError as exc:
        raise CommandParseError(_clean_symbol_error(raw_symbol, str(exc))) from None


def parse_compare_args(
    args: Sequence[str],
    *,
    default_timeframe: Timeframe = Timeframe.H4,
) -> CompareCommandArgs:
    """Parse `/compare <sym1> <sym2> [sym3] [timeframe]` arguments."""
    if len(args) < 2 or len(args) > 4:
        raise CommandParseError(
            "Format salah. Gunakan: /compare <SIMBOL1> <SIMBOL2> [SIMBOL3] [1h|4h|1d] "
            "(contoh: /compare BTC ETH 1d)"
        )

    last_lower = args[-1].strip().lower()
    if last_lower in _SUPPORTED_TFS:
        symbol_tokens = list(args[:-1])
        tf_value = last_lower
    elif _TIMEFRAME_LIKE_RE.match(last_lower):
        raise CommandParseError(
            f"Timeframe '{args[-1]}' tidak didukung. Pilihan timeframe yang tersedia: 1h, 4h, 1d."
        )
    else:
        symbol_tokens = list(args)
        tf_value = default_timeframe.value

    if len(symbol_tokens) < 2 or len(symbol_tokens) > 3:
        raise CommandParseError(
            "Perbandingan membutuhkan 2 atau 3 simbol yang berbeda (contoh: /compare BTC ETH 1d)."
        )

    try:
        return CompareCommandArgs(
            symbols=tuple(symbol_tokens),
            timeframe=Timeframe(tf_value),
        )
    except ValidationError as exc:
        raise CommandParseError(
            "Argumen simbol tidak valid atau duplikat. "
            f"Gunakan 2-3 simbol spot unik dari BTC, ETH, SOL ({exc})."
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
        "- /watchlist add <BTC|ETH|SOL>\n"
        "- /watchlist remove <BTC|ETH|SOL>"
    )


def _extract_mvp_symbols(text: str) -> list[str]:
    found: list[tuple[int, str]] = []
    for pattern, canonical in _SYMBOL_ALIASES:
        for match in pattern.finditer(text):
            found.append((match.start(), canonical))
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
        header_match = _HEADER_RE.match(msg.content.strip())
        if header_match is not None:
            if not tf_found:
                recent_tf = Timeframe(header_match.group(2).lower())
                tf_found = True
            for raw_part in header_match.group(1).split(","):
                try:
                    canonical = resolve_mvp_spot_symbol(raw_part.strip()).canonical_symbol
                except ValueError:
                    continue
                if canonical not in recent_symbols:
                    recent_symbols.append(canonical)
            if recent_symbols:
                break

        msg_syms = _extract_mvp_symbols(msg.content)
        if msg_syms:
            for sym in msg_syms:
                if sym not in recent_symbols:
                    recent_symbols.append(sym)
            if not tf_found:
                tf_match = _SUPPORTED_TIMEFRAME_TEXT_RE.search(msg.content)
                if tf_match is not None:
                    recent_tf = Timeframe(tf_match.group(1).lower())
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
            message="Pesan kosong. Gunakan /analyze <SIMBOL> [1h|4h|1d] atau /help."
        )
    if len(cleaned) > MAX_USER_MESSAGE_CHARS:
        return RefusalIntent(
            message=(
                f"Pesan terlalu panjang (maksimal {MAX_USER_MESSAGE_CHARS} karakter). "
                "Ringkas pertanyaan Anda atau gunakan /analyze /compare."
            )
        )

    if _ORDER_OR_ADVICE_RE.search(cleaned):
        return RefusalIntent(
            message=(
                "Bot ini hanya asisten riset dan edukasi. Eksekusi order, instruksi beli/jual "
                "wajib, dan alokasi portofolio pribadi berada di luar cakupan layanan."
            )
        )

    if _PERPETUAL_HINT_RE.search(cleaned):
        return RefusalIntent(
            message=(
                "Versi MVP saat ini hanya mendukung pasar spot (BTC/USDT, ETH/USDT, SOL/USDT) "
                "dan belum mendukung instrumen perpetual/futures."
            )
        )

    for pair in _UNSUPPORTED_PAIR_RE.findall(cleaned):
        upper_pair = pair.upper()
        if upper_pair not in ("BTC/USDT", "ETH/USDT", "SOL/USDT"):
            return RefusalIntent(
                message=(
                    f"Pasangan '{pair}' belum didukung. Saat ini hanya mendukung spot "
                    "BTC/USDT, ETH/USDT, dan SOL/USDT."
                )
            )

    if _BAD_TIMEFRAME_TEXT_RE.search(cleaned) and not _SUPPORTED_TIMEFRAME_TEXT_RE.search(cleaned):
        return RefusalIntent(
            message="Timeframe tersebut belum didukung. Gunakan timeframe 1h, 4h, atau 1d."
        )

    recent_symbols, recent_tf = _extract_recent_context(recent_messages, default_timeframe)
    tf_match = _SUPPORTED_TIMEFRAME_TEXT_RE.search(cleaned)
    target_tf = Timeframe(tf_match.group(1).lower()) if tf_match is not None else recent_tf

    mentioned = _extract_mvp_symbols(cleaned)
    is_compare = _COMPARE_WORD_RE.search(cleaned) is not None

    if len(mentioned) >= 2:
        return CompareIntent(symbols=tuple(mentioned[:3]), timeframe=target_tf)

    if len(mentioned) == 1:
        if is_compare and recent_symbols:
            prior = next((sym for sym in recent_symbols if sym != mentioned[0]), None)
            if prior is not None:
                return CompareIntent(symbols=(prior, mentioned[0]), timeframe=target_tf)
        return AnalyzeIntent(symbol=mentioned[0], timeframe=target_tf)

    if recent_symbols and (_CONTEXT_TOPIC_RE.search(cleaned) or is_compare):
        if is_compare and len(recent_symbols) >= 2:
            return CompareIntent(symbols=tuple(recent_symbols[:2]), timeframe=target_tf)
        return AnalyzeIntent(symbol=recent_symbols[0], timeframe=target_tf)

    return RefusalIntent(
        message=(
            "Saya belum dapat menentukan simbol atau perintah yang dimaksud. "
            "Gunakan /analyze BTC 4h, /compare BTC ETH 1d, atau /help untuk daftar perintah."
        )
    )
