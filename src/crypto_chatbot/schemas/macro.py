"""Typed schemas for DXY, BTC benchmark, and futures positioning regime (ADR 0013)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.schemas.analysis import MarketStructure, Regime
from crypto_chatbot.schemas.market import FiniteFloat, QualityFlag, UtcDatetime


class PositioningRegime(StrEnum):
    """Classification of futures open interest, price change, CVD, and funding mechanics."""

    LONG_BUILDUP = "long_buildup"
    SHORT_BUILDUP = "short_buildup"
    SHORT_COVERING = "short_covering"
    LONG_LIQUIDATION_FLUSH = "long_liquidation_flush"
    CROWDED_SQUEEZE_RISK = "crowded_squeeze_risk"
    NEUTRAL_POSITIONING = "neutral_positioning"


class DxySnapshotMetrics(BaseModel):
    """Closed-bar US Dollar Index (DXY) observations and correlation with target asset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = "DXY"
    as_of: UtcDatetime
    last_close: FiniteFloat = Field(gt=0)
    change_1_bar_pct: FiniteFloat | None = None
    change_20_bar_pct: FiniteFloat | None = None
    ema_20: FiniteFloat | None = None
    ema_50: FiniteFloat | None = None
    rsi_14: FiniteFloat | None = None
    trend: Literal["bullish", "bearish", "neutral"] = "neutral"
    rolling_corr_20: FiniteFloat | None = Field(default=None, ge=-1.0, le=1.0)
    macro_impact: Literal["headwind", "tailwind", "neutral"] = "neutral"
    is_weekend_or_closed: bool = False
    quality_flags: tuple[QualityFlag, ...] = ()


class BtcBenchmarkMetrics(BaseModel):
    """Perpetual BTC benchmark metrics for cross-asset relative strength and beta alignment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_symbol: str = "BTC/USDT:USDT"
    as_of: UtcDatetime
    last_close: FiniteFloat = Field(gt=0)
    change_20_bar_pct: FiniteFloat | None = None
    regime: Regime | None = None
    market_structure: MarketStructure | None = None
    relative_strength_20_pct: FiniteFloat | None = None
    rolling_corr_20: FiniteFloat | None = Field(default=None, ge=-1.0, le=1.0)
    rolling_beta_20: FiniteFloat | None = None
    leader_alignment: Literal["aligned", "divergent_warning", "self"] = "self"


class MacroBenchmarkContext(BaseModel):
    """Unified macro context (DXY + BTC benchmark + positioning regime) attached to snapshot."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    as_of: UtcDatetime
    dxy: DxySnapshotMetrics | None = None
    btc_benchmark: BtcBenchmarkMetrics | None = None
    positioning_regime: PositioningRegime = PositioningRegime.NEUTRAL_POSITIONING
