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

_BARE_BASE_PATTERN = re.compile(r"^([A-Z0-9]{2,10})$")


def _lookup(base: str, market_type: MarketType) -> SymbolMapping:
    table = _SPOT_MAPPINGS if market_type is MarketType.SPOT else _PERPETUAL_MAPPINGS
    mapping = table.get(base)
    if mapping is None:
        supported = ", ".join(SUPPORTED_MVP_BASES)
        raise SymbolError(f"unsupported symbol '{base}' (supported bases: {supported})")
    return mapping


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
    """Resolve a symbol strictly for the MVP spot universe."""
    return resolve_symbol(raw, market_type=MarketType.SPOT)
