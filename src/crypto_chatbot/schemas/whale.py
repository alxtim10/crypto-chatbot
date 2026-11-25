"""Typed schemas for whale tracking, perpetual positions, and on-chain wallet flows."""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class NetworkType(StrEnum):
    EVM = "evm"
    BITCOIN = "bitcoin"
    SOLANA = "solana"


class WhalePositionSide(StrEnum):
    LONG = "long"
    SHORT = "short"


class WhaleFlowType(StrEnum):
    CEX_INFLOW = "cex_inflow"
    CEX_OUTFLOW = "cex_outflow"
    TRANSFER = "transfer"


class WhalePosition(BaseModel):
    """A perpetual futures position held by a tracked whale wallet."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    wallet_address: str
    wallet_label: str | None = None
    symbol: str
    side: WhalePositionSide
    size_tokens: float = Field(ge=0.0)
    size_usd: float = Field(ge=0.0)
    entry_price: float = Field(ge=0.0)
    liquidation_price: float | None = Field(default=None, ge=0.0)
    leverage: float | None = Field(default=None, ge=0.0)
    unrealized_pnl_usd: float = 0.0
    unrealized_pnl_percent: float | None = None


class WhaleFlowEvent(BaseModel):
    """An on-chain transfer event involving a whale wallet."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tx_hash: str
    timestamp: datetime
    flow_type: WhaleFlowType
    amount: float = Field(ge=0.0)
    unit: str
    amount_usd: float | None = Field(default=None, ge=0.0)
    counterpart_address: str
    counterpart_label: str | None = None

    @field_validator("timestamp")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)


class WhaleWalletProfile(BaseModel):
    """Detailed profile of a tracked whale address (Perps + On-Chain)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    address: str
    network: NetworkType
    label: str | None = None
    balance_native: float = Field(default=0.0, ge=0.0)
    balance_usd: float | None = Field(default=None, ge=0.0)
    positions: tuple[WhalePosition, ...] = ()
    recent_flows: tuple[WhaleFlowEvent, ...] = ()
    is_tracked: bool = False


class WhaleMarketOverview(BaseModel):
    """Aggregated perpetual futures whale positioning for a specific symbol."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    as_of: datetime
    total_long_usd: float = Field(ge=0.0)
    total_short_usd: float = Field(ge=0.0)
    long_percentage: float = Field(ge=0.0, le=100.0)
    short_percentage: float = Field(ge=0.0, le=100.0)
    net_bias: Literal["HEAVY_LONG", "LEAN_LONG", "NEUTRAL", "LEAN_SHORT", "HEAVY_SHORT"]
    active_whale_count: int = Field(ge=0)
    top_positions: tuple[WhalePosition, ...] = ()
    notes: tuple[str, ...] = ()

    @field_validator("as_of")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)


class WhaleWatchlistEntry(BaseModel):
    """User-saved or system-curated whale address entry."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    address: str
    network: NetworkType
    label: str
    added_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @field_validator("added_at")
    @classmethod
    def _ensure_utc(cls, v: datetime) -> datetime:
        if v.tzinfo is None:
            return v.replace(tzinfo=UTC)
        return v.astimezone(UTC)
