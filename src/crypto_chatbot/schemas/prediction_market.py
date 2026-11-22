"""Typed schemas for read-only prediction market context (ADR 0012)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crypto_chatbot.schemas.market import FiniteFloat, QualityFlag, UtcDatetime

PredictionMarketVenue = Literal["polymarket", "kalshi"]
PredictionMarketCategory = Literal["crypto_price", "macro_rate", "regulatory_etf", "other"]


class PredictionMarketObservation(BaseModel):
    """Normalized observation of a single binary prediction market contract."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    venue: PredictionMarketVenue
    market_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    category: PredictionMarketCategory = "other"
    canonical_symbol: str | None = None
    implied_yes_prob: FiniteFloat = Field(ge=0.0, le=1.0)
    bid_ask_spread_bps: FiniteFloat | None = Field(default=None, ge=0.0)
    volume_24h_usd: FiniteFloat = Field(default=0.0, ge=0.0)
    liquidity_usd: FiniteFloat = Field(default=0.0, ge=0.0)
    resolution_time: UtcDatetime | None = None
    observed_time: UtcDatetime
    received_time: UtcDatetime
    available_time: UtcDatetime
    url: str | None = None
    quality_flags: tuple[QualityFlag, ...] = ()

    @model_validator(mode="after")
    def _validate_temporal_order(self) -> PredictionMarketObservation:
        if self.available_time < self.observed_time:
            raise ValueError("available_time cannot be earlier than observed_time")
        return self


class PredictionMarketSummary(BaseModel):
    """Aggregated prediction market context attached to an `AnalysisSnapshot`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    as_of: UtcDatetime
    observations: tuple[PredictionMarketObservation, ...] = ()
    macro_fomc_cut_prob: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    crypto_upside_prob: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    has_liquid_markets: bool = True
