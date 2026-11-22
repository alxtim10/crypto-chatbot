"""SQLAlchemy ORM models for market data, snapshots, and bounded user memory."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    DateTime,
    Double,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    """Declarative base for all PostgreSQL tables."""


class MarketCandleRow(Base):
    """Normalized public OHLCV candle storage (`market_candles`)."""

    __tablename__ = "market_candles"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "venue",
            "canonical_symbol",
            "market_type",
            "timeframe",
            "open_time",
            name="uq_market_candles_identity",
        ),
        Index(
            "ix_market_candles_lookup",
            "venue",
            "canonical_symbol",
            "market_type",
            "timeframe",
            "open_time",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    venue: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    close_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    available_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[float] = mapped_column(Double, nullable=False)
    high: Mapped[float] = mapped_column(Double, nullable=False)
    low: Mapped[float] = mapped_column(Double, nullable=False)
    close: Mapped[float] = mapped_column(Double, nullable=False)
    base_volume: Mapped[float] = mapped_column(Double, nullable=False)
    quote_volume: Mapped[float | None] = mapped_column(Double, nullable=True)
    base_unit: Mapped[str] = mapped_column(String(32), nullable=False)
    quote_currency: Mapped[str] = mapped_column(String(32), nullable=False)
    is_closed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    quality_flags: Mapped[list[str]] = mapped_column(JSONB, nullable=False)


class AnalysisSnapshotRow(Base):
    """Persisted deterministic analysis snapshot (`analysis_snapshots`)."""

    __tablename__ = "analysis_snapshots"

    snapshot_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    venue: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    calc_version: Mapped[str] = mapped_column(String(32), nullable=False)
    quality_status: Mapped[str] = mapped_column(String(32), nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    first_open_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    last_open_time: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    candle_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    input_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    quality_flags: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    snapshot_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)


class UserRow(Base):
    """Minimal user identity (`users`): numeric Telegram ID only, never PII."""

    __tablename__ = "users"

    telegram_user_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    is_allowlisted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )


class UserPreferenceRow(Base):
    """Explicit user preferences (`user_preferences`)."""

    __tablename__ = "user_preferences"

    telegram_user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.telegram_user_id", ondelete="CASCADE"),
        primary_key=True,
    )
    watchlist: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    language: Mapped[str] = mapped_column(String(16), nullable=False, default="id")
    default_timeframe: Mapped[str] = mapped_column(String(16), nullable=False, default="4h")
    display_currency: Mapped[str] = mapped_column(String(16), nullable=False, default="USDT")
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )


class ConversationRow(Base):
    """Bounded conversation thread per user (`conversations`)."""

    __tablename__ = "conversations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.telegram_user_id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )


class MessageRow(Base):
    """Bounded recent chat turns (`messages`); assistant turns store placeholders only."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    conversation_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    role: Mapped[str] = mapped_column(String(16), nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    snapshot_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )


class LlmInteractionRow(Base):
    """Audit trail of Gemini narration calls (`llm_interactions`); never contains secrets."""

    __tablename__ = "llm_interactions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    request_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    snapshot_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    validation_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rejection_reasons: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
        index=True,
    )


class FundingRateRow(Base):
    """Raw venue-specific perpetual funding-rate observations (`funding_rate_observations`)."""

    __tablename__ = "funding_rate_observations"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "venue",
            "canonical_symbol",
            "market_type",
            "observed_time",
            name="uq_funding_rate_identity",
        ),
        Index(
            "ix_funding_rate_lookup",
            "venue",
            "canonical_symbol",
            "available_time",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    venue: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    observed_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    next_funding_time: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True), nullable=True
    )
    received_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    available_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    funding_rate: Mapped[float] = mapped_column(Double, nullable=False)
    funding_interval_hours: Mapped[float] = mapped_column(Double, nullable=False)
    rate_per_8h: Mapped[float] = mapped_column(Double, nullable=False)
    annualized_rate: Mapped[float] = mapped_column(Double, nullable=False)
    mark_price: Mapped[float | None] = mapped_column(Double, nullable=True)
    index_price: Mapped[float | None] = mapped_column(Double, nullable=True)
    quality_flags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)


class OpenInterestRow(Base):
    """Raw venue-specific perpetual open-interest observations (`open_interest_observations`)."""

    __tablename__ = "open_interest_observations"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "venue",
            "canonical_symbol",
            "market_type",
            "period",
            "observed_time",
            name="uq_open_interest_identity",
        ),
        Index(
            "ix_open_interest_lookup",
            "venue",
            "canonical_symbol",
            "period",
            "available_time",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    venue: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    period: Mapped[str] = mapped_column(String(16), nullable=False)
    observed_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    available_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open_interest_contracts: Mapped[float] = mapped_column(Double, nullable=False)
    raw_unit: Mapped[str] = mapped_column(String(32), nullable=False)
    contract_size: Mapped[float] = mapped_column(Double, nullable=False)
    open_interest_base: Mapped[float | None] = mapped_column(Double, nullable=True)
    base_unit: Mapped[str] = mapped_column(String(32), nullable=False)
    open_interest_quote: Mapped[float | None] = mapped_column(Double, nullable=True)
    quote_currency: Mapped[str] = mapped_column(String(32), nullable=False)
    quality_flags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)


class OrderBookSnapshotRow(Base):
    """Venue-specific L2 order-book snapshots (`order_book_snapshots`)."""

    __tablename__ = "order_book_snapshots"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "venue",
            "canonical_symbol",
            "market_type",
            "observed_time",
            "sequence_key",
            name="uq_order_book_snapshot_identity",
        ),
        Index(
            "ix_order_book_lookup",
            "venue",
            "canonical_symbol",
            "market_type",
            "available_time",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    venue: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    observed_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    available_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    sequence_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    sequence_key: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    best_bid: Mapped[float] = mapped_column(Double, nullable=False)
    best_ask: Mapped[float] = mapped_column(Double, nullable=False)
    mid_price: Mapped[float] = mapped_column(Double, nullable=False)
    spread_abs: Mapped[float] = mapped_column(Double, nullable=False)
    spread_bps: Mapped[float] = mapped_column(Double, nullable=False)
    base_unit: Mapped[str] = mapped_column(String(32), nullable=False)
    quote_currency: Mapped[str] = mapped_column(String(32), nullable=False)
    resync_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    depth_bands: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, nullable=False, default=list)
    quality_flags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)


class LiquidationEventRow(Base):
    """Normalized forced-liquidation events (`liquidation_events`)."""

    __tablename__ = "liquidation_events"
    __table_args__ = (
        UniqueConstraint(
            "provider",
            "venue",
            "canonical_symbol",
            "event_key",
            name="uq_liquidation_event_identity",
        ),
        Index(
            "ix_liquidation_lookup",
            "venue",
            "canonical_symbol",
            "available_time",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    venue: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    event_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    available_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    side: Mapped[str] = mapped_column(String(32), nullable=False)
    price: Mapped[float] = mapped_column(Double, nullable=False)
    base_quantity: Mapped[float] = mapped_column(Double, nullable=False)
    quote_notional: Mapped[float] = mapped_column(Double, nullable=False)
    base_unit: Mapped[str] = mapped_column(String(32), nullable=False)
    quote_currency: Mapped[str] = mapped_column(String(32), nullable=False)
    event_key: Mapped[str] = mapped_column(String(64), nullable=False)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)


class TradeSetupRow(Base):
    """Deterministic setup log (`trade_setups`) with TradingAgents-style outcome settlement."""

    __tablename__ = "trade_setups"
    __table_args__ = (
        Index(
            "ix_trade_setups_lookup",
            "venue",
            "canonical_symbol",
            "timeframe",
            "as_of",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    venue: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    as_of: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        default=lambda: datetime.now(UTC),
    )
    status: Mapped[str] = mapped_column(String(16), nullable=False)
    bias: Mapped[str] = mapped_column(String(16), nullable=False)
    entry_source: Mapped[str | None] = mapped_column(String(128), nullable=True)
    wait_reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    reference_price: Mapped[float] = mapped_column(Double, nullable=False)
    entry_low: Mapped[float | None] = mapped_column(Double, nullable=True)
    entry_high: Mapped[float | None] = mapped_column(Double, nullable=True)
    entry_mid: Mapped[float | None] = mapped_column(Double, nullable=True)
    stop_loss: Mapped[float | None] = mapped_column(Double, nullable=True)
    tp1: Mapped[float | None] = mapped_column(Double, nullable=True)
    tp2: Mapped[float | None] = mapped_column(Double, nullable=True)
    tp3: Mapped[float | None] = mapped_column(Double, nullable=True)
    rr_tp1: Mapped[float | None] = mapped_column(Double, nullable=True)
    rr_tp2: Mapped[float | None] = mapped_column(Double, nullable=True)
    rr_tp3: Mapped[float | None] = mapped_column(Double, nullable=True)
    outcome_status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    realized_r: Mapped[float | None] = mapped_column(Double, nullable=True)
