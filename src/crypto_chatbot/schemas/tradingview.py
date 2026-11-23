"""Validated schemas for TradingView AI Agent Lens (`ai-agent-lens/v1`) packets."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crypto_chatbot.schemas.market import (
    FiniteFloat,
    MarketType,
    QualityFlag,
    Timeframe,
    UtcDatetime,
    validate_symbol_market_type,
)

AI_AGENT_LENS_SCHEMA_VERSION: Literal["ai-agent-lens/v1"] = "ai-agent-lens/v1"
AI_AGENT_LENS_PINNED_COMMIT = "7f3c9a2e1d8b4056c2a91e4f7b8d3c1a5e6f9012"
AI_AGENT_LENS_ATTRIBUTION = "MIT-licensed AI Agent Lens (ai-agent-lens/v1)"

TRADINGVIEW_PUBLISHED_WEBHOOK_IPS: tuple[str, ...] = (
    "52.89.214.238",
    "34.212.75.30",
    "54.218.53.128",
    "52.32.178.7",
)

FORBIDDEN_SECRET_PAYLOAD_KEYS: frozenset[str] = frozenset(
    {
        "secret",
        "webhook_secret",
        "token",
        "api_key",
        "password",
    }
)


def _assert_no_forbidden_keys(mapping: Mapping[str, Any]) -> None:
    for key, val in mapping.items():
        if str(key).strip().lower() in FORBIDDEN_SECRET_PAYLOAD_KEYS:
            raise ValueError(f"sanitized_payload cannot contain forbidden key '{key}'")
        if isinstance(val, Mapping):
            _assert_no_forbidden_keys(val)
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, Mapping):
                    _assert_no_forbidden_keys(item)


class TradingViewEventType(StrEnum):
    BAR_CLOSE = "bar_close"
    STRUCTURE_BREAK = "structure_break"
    FAILED_BREAKOUT = "failed_breakout"
    MTF_RSI_SHIFT = "mtf_rsi_shift"
    REGIME_SHIFT = "regime_shift"
    CUSTOM_SIGNAL = "custom_signal"


class TradingViewPacket(BaseModel):
    """Normalized closed-candle packet from TradingView `ai-agent-lens/v1`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    idempotency_key: str = Field(min_length=16, max_length=64)
    schema_version: Literal["ai-agent-lens/v1"] = AI_AGENT_LENS_SCHEMA_VERSION
    attribution_commit: str = Field(default=AI_AGENT_LENS_PINNED_COMMIT, min_length=7)
    provider: str = Field(default="tradingview", min_length=1)
    venue: str = Field(min_length=1)
    canonical_symbol: str = Field(min_length=1)
    provider_symbol: str = Field(min_length=1)
    market_type: MarketType
    timeframe: Timeframe
    event_type: TradingViewEventType
    candle_open_time: UtcDatetime
    candle_close_time: UtcDatetime
    received_time: UtcDatetime
    available_time: UtcDatetime
    open: FiniteFloat = Field(gt=0)
    high: FiniteFloat = Field(gt=0)
    low: FiniteFloat = Field(gt=0)
    close: FiniteFloat = Field(gt=0)
    volume: FiniteFloat | None = Field(default=None, ge=0)
    rule_score: FiniteFloat | None = Field(default=None, ge=-100.0, le=100.0)
    rule_label: str | None = None
    is_probabilistic: Literal[False] = False
    quality_flags: tuple[QualityFlag, ...] = ()
    sanitized_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_packet(self) -> TradingViewPacket:
        validate_symbol_market_type(
            self.canonical_symbol,
            self.provider_symbol,
            self.market_type,
        )
        expected_close = self.candle_open_time + self.timeframe.duration
        if self.candle_close_time != expected_close:
            raise ValueError("candle_close_time must equal candle_open_time + timeframe.duration")
        if self.candle_close_time > self.received_time:
            raise ValueError(
                "tradingview packet candle_close_time cannot exceed received_time (unclosed candle)"
            )
        if self.available_time < self.candle_close_time:
            raise ValueError("available_time cannot be earlier than candle_close_time")
        if self.high < max(self.open, self.close, self.low) or self.low > min(
            self.open, self.close, self.high
        ):
            raise ValueError("invalid OHLC bounds in TradingViewPacket")
        _assert_no_forbidden_keys(self.sanitized_payload)
        return self


class TradingViewExchangeComparison(BaseModel):
    """Cross-source comparison of a TradingView packet against closed exchange candles."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    idempotency_key: str
    canonical_symbol: str
    timeframe: Timeframe
    candle_open_time: UtcDatetime
    candle_close_time: UtcDatetime
    tv_close: FiniteFloat = Field(gt=0)
    exchange_close: FiniteFloat | None = Field(default=None, gt=0)
    close_deviation_bps: FiniteFloat | None = Field(default=None, ge=0)
    high_deviation_bps: FiniteFloat | None = Field(default=None, ge=0)
    low_deviation_bps: FiniteFloat | None = Field(default=None, ge=0)
    max_deviation_bps: FiniteFloat | None = Field(default=None, ge=0)
    exchange_candle_found: bool = False
    quality_flags: tuple[QualityFlag, ...] = ()
    limitations: tuple[str, ...] = ()


class TradingViewMetrics(BaseModel):
    """Snapshot-attached TradingView AI Agent Lens metrics and cross-source checks."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = AI_AGENT_LENS_SCHEMA_VERSION
    attribution_commit: str = AI_AGENT_LENS_PINNED_COMMIT
    idempotency_key: str
    venue: str
    canonical_symbol: str
    provider_symbol: str
    market_type: MarketType
    timeframe: Timeframe
    event_type: str
    candle_close_time: UtcDatetime
    tv_close: FiniteFloat = Field(gt=0)
    exchange_close: FiniteFloat | None = Field(default=None, gt=0)
    max_deviation_bps: FiniteFloat | None = Field(default=None, ge=0)
    rule_score: FiniteFloat | None = None
    rule_label: str | None = None
    is_probabilistic: Literal[False] = False
