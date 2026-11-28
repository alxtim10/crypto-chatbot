"""Explicit symbol normalization and spot/perpetual isolation."""

from __future__ import annotations

import re

from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.schemas.market import MarketType


class SymbolError(ValueError):
    """Raised when a symbol is unknown, malformed, or mixes spot and perpetual markets."""


class SymbolMapping(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    base: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    market_type: MarketType
    canonical_symbol: str = Field(min_length=1)
    ccxt_symbol: str = Field(min_length=1)
    provider_symbol: str = Field(min_length=1)
    venue: str = Field(default="binance", min_length=1)


SUPPORTED_MVP_BASES: tuple[str, ...] = ("BTC", "ETH", "SOL")

DEFAULT_PERP_UNIVERSE: frozenset[str] = frozenset(
    {
        "1000BONK",
        "1000FLOKI",
        "1000PEPE",
        "1000SHIB",
        "AAVE",
        "ACE",
        "ACT",
        "ADA",
        "AI",
        "ALGO",
        "ALT",
        "APE",
        "APT",
        "AR",
        "ARB",
        "ARC",
        "ATOM",
        "AVAX",
        "AXS",
        "BB",
        "BCH",
        "BIO",
        "BLUR",
        "BNB",
        "BOME",
        "BONK",
        "BTC",
        "CAKE",
        "CAT",
        "CFX",
        "CHZ",
        "COMP",
        "COOKIE",
        "CRV",
        "DAR",
        "DASH",
        "DATA",
        "DOG",
        "DOGE",
        "DOT",
        "DYDX",
        "EIGEN",
        "ENA",
        "ENS",
        "EOS",
        "ETC",
        "ETH",
        "FARTCOIN",
        "FET",
        "FIL",
        "FLOKI",
        "FLOW",
        "FUN",
        "GALA",
        "GAS",
        "GMX",
        "HBAR",
        "HIGH",
        "HOOK",
        "HOT",
        "HYPE",
        "ICP",
        "ID",
        "IMX",
        "INJ",
        "IO",
        "JASMY",
        "JUP",
        "KAS",
        "KAVA",
        "KEY",
        "KSM",
        "LDO",
        "LINK",
        "LISTA",
        "LIT",
        "LTC",
        "MAGIC",
        "MANA",
        "MASK",
        "ME",
        "MEME",
        "MEW",
        "MKR",
        "MOVE",
        "NEAR",
        "NEIRO",
        "NOT",
        "OMG",
        "ONDO",
        "ONE",
        "OP",
        "ORDI",
        "PENDLE",
        "PENGU",
        "PEOPLE",
        "PEPE",
        "PIXEL",
        "PNUT",
        "POPCAT",
        "PORTAL",
        "PYTH",
        "RAY",
        "REN",
        "RENDER",
        "RSR",
        "RUNE",
        "SAND",
        "SC",
        "SEI",
        "SHIB",
        "SNX",
        "SOL",
        "SSV",
        "STRK",
        "STX",
        "SUI",
        "SUN",
        "SUPER",
        "SUSHI",
        "TAO",
        "THETA",
        "TIA",
        "TON",
        "TRB",
        "TRUMP",
        "TRX",
        "TURBO",
        "UNI",
        "VET",
        "VIRTUAL",
        "VVV",
        "WIF",
        "WIN",
        "WLD",
        "XLM",
        "XMR",
        "XRP",
        "XTZ",
        "ZEC",
        "ZK",
        "ZRO",
    }
)

# Universe tickers that collide with common Indonesian/English words, slang, or acronyms.
AMBIGUOUS_PERP_TICKERS: frozenset[str] = frozenset(
    {
        "ACE",
        "ACT",
        "ADA",
        "AI",
        "ALL",
        "ALT",
        "APE",
        "ARC",
        "BB",
        "BIO",
        "CAKE",
        "CAT",
        "COMP",
        "COOKIE",
        "DAR",
        "DATA",
        "DOG",
        "FLOW",
        "FUN",
        "GALA",
        "GAS",
        "HIGH",
        "HOOK",
        "HOT",
        "ID",
        "IO",
        "KEY",
        "LINK",
        "LISTA",
        "LIT",
        "MAGIC",
        "MANA",
        "MASK",
        "ME",
        "MEME",
        "MOVE",
        "NEAR",
        "NOT",
        "OMG",
        "ONE",
        "OP",
        "PEOPLE",
        "PIXEL",
        "PORTAL",
        "RAY",
        "REN",
        "RUNE",
        "SAND",
        "SC",
        "SUN",
        "SUPER",
        "SUSHI",
        "TON",
        "TRUMP",
        "TURBO",
        "UNI",
        "US",
        "WIN",
    }
)

_SPOT_MAPPINGS: dict[str, SymbolMapping] = {
    base: SymbolMapping(
        base=base,
        quote="USDT",
        market_type=MarketType.SPOT,
        canonical_symbol=f"{base}/USDT",
        ccxt_symbol=f"{base}/USDT",
        provider_symbol=f"BINANCE:{base}USDT",
    )
    for base in SUPPORTED_MVP_BASES
}

_PERPETUAL_MAPPINGS: dict[str, SymbolMapping] = {
    base: SymbolMapping(
        base=base,
        quote="USDT",
        market_type=MarketType.PERPETUAL,
        canonical_symbol=f"{base}/USDT:USDT",
        ccxt_symbol=f"{base}/USDT:USDT",
        provider_symbol=f"BINANCE:{base}USDT.P",
    )
    for base in SUPPORTED_MVP_BASES
}

_PERP_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^BINANCE:([A-Z0-9]+)USDT\.P$"),
    re.compile(r"^([A-Z0-9]+)/USDT:USDT$"),
    re.compile(r"^([A-Z0-9]+)(?:[-_ ]?(?:PERP|PERPETUAL))$"),
    re.compile(r"^([A-Z0-9]+)USDT\.P$"),
)

_EXPLICIT_SPOT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"^BINANCE:([A-Z0-9]+)USDT$"),
    re.compile(r"^([A-Z0-9]+)/USDT$"),
    re.compile(r"^([A-Z0-9]+)USDT$"),
    re.compile(r"^([A-Z0-9]+)(?:[-_ ]?SPOT)$"),
)

_DISALLOWED_BASES: frozenset[str] = frozenset(
    {
        "USDT",
        "USD",
        "BUSD",
        "FDUSD",
        "USDC",
        "TUSD",
        "DAI",
        "EUR",
        "GBP",
        "JPY",
        "IDR",
        "SPOT",
        "PERP",
        "PERPETUAL",
        "FUTURES",
    }
)
_VALID_BASE_RE = re.compile(r"^(?=[A-Z0-9]*[A-Z])[A-Z0-9]{2,12}$")

_BARE_BASE_PATTERN = re.compile(r"^([A-Z0-9]{2,12})$")


def _lookup(base: str, market_type: MarketType) -> SymbolMapping:
    if base in _DISALLOWED_BASES or _VALID_BASE_RE.match(base) is None:
        raise SymbolError(f"unsupported or invalid symbol '{base}'")

    table = _SPOT_MAPPINGS if market_type is MarketType.SPOT else _PERPETUAL_MAPPINGS
    mapping = table.get(base)
    if mapping is not None:
        return mapping

    if market_type is MarketType.SPOT:
        return SymbolMapping(
            base=base,
            quote="USDT",
            market_type=MarketType.SPOT,
            canonical_symbol=f"{base}/USDT",
            ccxt_symbol=f"{base}/USDT",
            provider_symbol=f"BINANCE:{base}USDT",
        )
    return SymbolMapping(
        base=base,
        quote="USDT",
        market_type=MarketType.PERPETUAL,
        canonical_symbol=f"{base}/USDT:USDT",
        ccxt_symbol=f"{base}/USDT:USDT",
        provider_symbol=f"BINANCE:{base}USDT.P",
    )


def resolve_symbol(raw: str, *, market_type: MarketType = MarketType.SPOT) -> SymbolMapping:
    """Resolve user or provider symbol input without silently mixing spot and perpetual markets."""
    cleaned = " ".join(raw.strip().upper().split())
    if not cleaned:
        raise SymbolError("symbol must not be empty")

    for pattern in _PERP_PATTERNS:
        match = pattern.match(cleaned)
        if match is not None:
            if market_type is not MarketType.PERPETUAL:
                raise SymbolError(
                    f"perpetual symbol '{raw}' cannot be resolved as {market_type.value}"
                )
            return _lookup(match.group(1), MarketType.PERPETUAL)

    for pattern in _EXPLICIT_SPOT_PATTERNS:
        match = pattern.match(cleaned)
        if match is not None:
            if market_type is not MarketType.SPOT:
                raise SymbolError(f"spot symbol '{raw}' cannot be resolved as {market_type.value}")
            return _lookup(match.group(1), MarketType.SPOT)

    bare_match = _BARE_BASE_PATTERN.match(cleaned)
    if bare_match is not None:
        return _lookup(bare_match.group(1), market_type)

    raise SymbolError(f"invalid symbol format: '{raw}'")


def resolve_mvp_spot_symbol(raw: str) -> SymbolMapping:
    """Resolve a symbol strictly for the legacy MVP spot universe."""
    return resolve_symbol(raw, market_type=MarketType.SPOT)


def resolve_default_perp_symbol(raw: str) -> SymbolMapping:
    """Resolve user input directly to perpetual futures contract (e.g. BTC -> BTC/USDT:USDT)."""
    cleaned = raw.strip().upper()
    if "/" in cleaned and ":" not in cleaned:
        base, _, quote = cleaned.partition("/")
        cleaned = f"{base}/{quote}:{quote}"
    elif (
        cleaned.endswith("USDT")
        and "/" not in cleaned
        and ":" not in cleaned
        and not cleaned.endswith(".P")
    ):
        base = cleaned[:-4]
        cleaned = f"{base}/USDT:USDT"
    return resolve_symbol(cleaned, market_type=MarketType.PERPETUAL)
