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
    raw_contracts: Mapped[float | None] = mapped_column(Double, nullable=True)
    base_volume: Mapped[float] = mapped_column(Double, nullable=False)
    quote_volume: Mapped[float | None] = mapped_column(Double, nullable=True)
    taker_buy_base_volume: Mapped[float | None] = mapped_column(Double, nullable=True)
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
    snapshot_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    model_id: Mapped[str] = mapped_column(String(128), nullable=False)
    prompt_version: Mapped[str] = mapped_column(String(32), nullable=False)
    validation_passed: Mapped[bool] = mapped_column(Boolean, nullable=False)
    rejection_reasons: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    prompt_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    output_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    layer: Mapped[str | None] = mapped_column(String(32), nullable=True)
    detected_symbol: Mapped[str | None] = mapped_column(String(128), nullable=True)
    intent: Mapped[str | None] = mapped_column(String(64), nullable=True)
    confidence: Mapped[float | None] = mapped_column(Double, nullable=True)
    fallback_reason: Mapped[str | None] = mapped_column(String(128), nullable=True)
    guard_violations: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
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
    market_type: Mapped[str] = mapped_column(String(16), nullable=False, default="spot")
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
    params_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    forward_only_guards_fired: Mapped[list[str]] = mapped_column(
        JSONB, nullable=False, default=list
    )


class TradingViewPacketRow(Base):
    """Sanitized TradingView AI Agent Lens packets (`tradingview_packets`); never stores secrets."""

    __tablename__ = "tradingview_packets"
    __table_args__ = (
        UniqueConstraint("idempotency_key", name="uq_tradingview_packet_idempotency"),
        Index(
            "ix_tradingview_packets_lookup",
            "venue",
            "canonical_symbol",
            "market_type",
            "timeframe",
            "available_time",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    idempotency_key: Mapped[str] = mapped_column(String(64), nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False)
    attribution_commit: Mapped[str] = mapped_column(String(64), nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False)
    venue: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    market_type: Mapped[str] = mapped_column(String(32), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    candle_open_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    candle_close_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    available_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    open: Mapped[float] = mapped_column(Double, nullable=False)
    high: Mapped[float] = mapped_column(Double, nullable=False)
    low: Mapped[float] = mapped_column(Double, nullable=False)
    close: Mapped[float] = mapped_column(Double, nullable=False)
    volume: Mapped[float | None] = mapped_column(Double, nullable=True)
    rule_score: Mapped[float | None] = mapped_column(Double, nullable=True)
    rule_label: Mapped[str | None] = mapped_column(String(64), nullable=True)
    is_probabilistic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    quality_flags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    sanitized_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)


class NewsArticleRow(Base):
    """Deduplicated canonical news article metadata (`news_articles`); no secrets."""

    __tablename__ = "news_articles"
    __table_args__ = (
        UniqueConstraint("article_id", name="uq_news_articles_article_id"),
        UniqueConstraint("canonical_url", name="uq_news_articles_canonical_url"),
        UniqueConstraint("content_fingerprint", name="uq_news_articles_fingerprint"),
        Index("ix_news_articles_available", "available_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    article_id: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_url: Mapped[str] = mapped_column(Text, nullable=False)
    content_fingerprint: Mapped[str] = mapped_column(String(64), nullable=False)
    publisher: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False, default="")
    published_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    first_seen_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    available_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    source_provider: Mapped[str] = mapped_column(String(64), nullable=False)
    affected_entities: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)


class SentimentObservationRow(Base):
    """Point-in-time non-probabilistic sentiment observations (`sentiment_observations`)."""

    __tablename__ = "sentiment_observations"
    __table_args__ = (
        UniqueConstraint("observation_id", name="uq_sentiment_obs_id"),
        UniqueConstraint(
            "article_id",
            "canonical_symbol",
            "model_version",
            name="uq_sentiment_obs_identity",
        ),
        Index(
            "ix_sentiment_obs_lookup",
            "canonical_symbol",
            "available_time",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    observation_id: Mapped[str] = mapped_column(String(64), nullable=False)
    article_id: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_url: Mapped[str] = mapped_column(Text, nullable=False)
    publisher: Mapped[str] = mapped_column(String(128), nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False)
    canonical_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    base_asset: Mapped[str] = mapped_column(String(32), nullable=False)
    entity_confidence: Mapped[float] = mapped_column(Double, nullable=False)
    sentiment_label: Mapped[str] = mapped_column(String(16), nullable=False)
    sentiment_score: Mapped[float] = mapped_column(Double, nullable=False)
    sentiment_confidence: Mapped[float] = mapped_column(Double, nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    is_probabilistic: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    published_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    available_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    scored_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    scorer_id: Mapped[str] = mapped_column(String(64), nullable=False, default="crypto-lexicon")
    scorer_version: Mapped[str] = mapped_column(
        String(64), nullable=False, default="crypto-lexicon-v1.0.0"
    )
    scorer_class: Mapped[str] = mapped_column(
        String(32), nullable=False, default="lexicon_deterministic"
    )
    is_backfilled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)


class MlExperimentRow(Base):
    """Offline ML experiment tracking (`ml_experiments`); no user data or secrets."""

    __tablename__ = "ml_experiments"
    __table_args__ = (
        UniqueConstraint("experiment_id", name="uq_ml_experiments_id"),
        Index(
            "ix_ml_experiments_lookup",
            "canonical_symbol",
            "timeframe",
            "created_at",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    venue: Mapped[str] = mapped_column(String(64), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    target_name: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(64), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    config_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    dataset_sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    accepted: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    oof_brier: Mapped[float] = mapped_column(Double, nullable=False)
    holdout_brier: Mapped[float] = mapped_column(Double, nullable=False)
    oof_net_return: Mapped[float] = mapped_column(Double, nullable=False)
    holdout_net_return: Mapped[float] = mapped_column(Double, nullable=False)
    report_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)


class ModelPredictionRow(Base):
    """Phase 13 shadow & released ML prediction audit log (`model_predictions`); no secrets."""

    __tablename__ = "model_predictions"
    __table_args__ = (
        UniqueConstraint("prediction_id", name="uq_model_predictions_id"),
        Index(
            "ix_model_predictions_lookup",
            "canonical_symbol",
            "timeframe",
            "prediction_time",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[str] = mapped_column(String(64), nullable=False)
    snapshot_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    experiment_id: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    target_name: Mapped[str] = mapped_column(String(64), nullable=False)
    horizon_hours: Mapped[int] = mapped_column(Integer, nullable=False)
    prediction_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    model_version: Mapped[str] = mapped_column(String(64), nullable=False)
    feature_version: Mapped[str] = mapped_column(String(64), nullable=False)
    quality_status: Mapped[str] = mapped_column(String(32), nullable=False)
    regime: Mapped[str | None] = mapped_column(String(32), nullable=True)
    calibrated_probability: Mapped[float | None] = mapped_column(Double, nullable=True)
    confidence_label: Mapped[str] = mapped_column(String(32), nullable=False)
    abstained: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    abstention_reason: Mapped[str | None] = mapped_column(String(64), nullable=True)
    max_feature_zscore: Mapped[float | None] = mapped_column(Double, nullable=True)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    exposed_to_user: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    realized_return: Mapped[float | None] = mapped_column(Double, nullable=True)
    realized_label: Mapped[int | None] = mapped_column(Integer, nullable=True)
    settled_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class WhaleWatchlistRow(Base):
    """Per-user whale wallet watchlist (`whale_watchlists`)."""

    __tablename__ = "whale_watchlists"
    __table_args__ = (
        UniqueConstraint("telegram_user_id", "address", name="uq_whale_watchlist_user_addr"),
        Index("ix_whale_watchlists_user", "telegram_user_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    telegram_user_id: Mapped[int] = mapped_column(
        BigInteger,
        ForeignKey("users.telegram_user_id", ondelete="CASCADE"),
        nullable=False,
    )
    address: Mapped[str] = mapped_column(String(128), nullable=False)
    network: Mapped[str] = mapped_column(String(16), nullable=False)
    label: Mapped[str] = mapped_column(String(128), nullable=False)
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class TrialRow(Base):
    """Registered historical backtest / ML experiment trial (`trials`, PRD v1.7 §8.2)."""

    __tablename__ = "trials"
    __table_args__ = (Index("ix_trials_family_freq", "family", "obs_frequency"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    trial_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    family: Mapped[str] = mapped_column(String(64), nullable=False)
    variant_params_json: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    params_sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    data_window: Mapped[str] = mapped_column(String(128), nullable=False)
    symbols: Mapped[list[str]] = mapped_column(JSONB, nullable=False)
    timeframe: Mapped[str] = mapped_column(String(16), nullable=False)
    obs_frequency: Mapped[str] = mapped_column(String(32), nullable=False)
    sr_per_obs: Mapped[float | None] = mapped_column(Double, nullable=True)
    n_obs: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    git_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)


class ScreenerUniverseSnapshotRow(Base):
    """Daily point-in-time screener universe snapshot (`screener_universe_snapshots`, T-30)."""

    __tablename__ = "screener_universe_snapshots"
    __table_args__ = (
        UniqueConstraint("snapshot_date_utc", "venue", name="uq_screener_snapshot_day_venue"),
        Index("ix_screener_snapshot_lookup", "venue", "snapshot_date_utc"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    snapshot_date_utc: Mapped[str] = mapped_column(String(16), nullable=False)
    venue: Mapped[str] = mapped_column(String(64), nullable=False)
    captured_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    min_volume_usd: Mapped[float] = mapped_column(Double, nullable=False)
    total_pairs_seen: Mapped[int] = mapped_column(Integer, nullable=False)
    liquid_pairs_count: Mapped[int] = mapped_column(Integer, nullable=False)
    top_gainers_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )
    top_losers_json: Mapped[list[dict[str, Any]]] = mapped_column(
        JSONB, nullable=False, default=list
    )


class ContractStatsRow(Base):
    """Append-only Gate.io perpetual contract_stats history (`contract_stats_observations`)."""

    __tablename__ = "contract_stats_observations"
    __table_args__ = (
        UniqueConstraint(
            "venue",
            "canonical_symbol",
            "interval",
            "observed_time",
            name="uq_contract_stats_identity",
        ),
        Index(
            "ix_contract_stats_lookup",
            "venue",
            "canonical_symbol",
            "interval",
            "available_time",
        ),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    venue: Mapped[str] = mapped_column(String(64), nullable=False)
    canonical_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    provider_symbol: Mapped[str] = mapped_column(String(64), nullable=False)
    interval: Mapped[str] = mapped_column(String(16), nullable=False)
    observed_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    available_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    lsr_taker: Mapped[float | None] = mapped_column(Double, nullable=True)
    lsr_account: Mapped[float | None] = mapped_column(Double, nullable=True)
    top_lsr_account: Mapped[float | None] = mapped_column(Double, nullable=True)
    top_lsr_size: Mapped[float | None] = mapped_column(Double, nullable=True)
    long_liq_usd: Mapped[float | None] = mapped_column(Double, nullable=True)
    short_liq_usd: Mapped[float | None] = mapped_column(Double, nullable=True)
    open_interest: Mapped[float | None] = mapped_column(Double, nullable=True)
    open_interest_usd: Mapped[float | None] = mapped_column(Double, nullable=True)
    mark_price: Mapped[float | None] = mapped_column(Double, nullable=True)
    last_funding_rate: Mapped[float | None] = mapped_column(Double, nullable=True)
    long_users: Mapped[int | None] = mapped_column(Integer, nullable=True)
    short_users: Mapped[int | None] = mapped_column(Integer, nullable=True)
    quality_flags: Mapped[list[str]] = mapped_column(JSONB, nullable=False, default=list)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)


class GateIngestLedgerRow(Base):
    """Idempotent checksum ledger for `download.gatedata.org` bulk archive files (D-04)."""

    __tablename__ = "gate_ingest_ledger"
    __table_args__ = (
        UniqueConstraint("dataset", "contract", "yyyymm", name="uq_gate_ingest_file"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    dataset: Mapped[str] = mapped_column(String(64), nullable=False)
    contract: Mapped[str] = mapped_column(String(64), nullable=False)
    yyyymm: Mapped[str] = mapped_column(String(6), nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    byte_size: Mapped[int] = mapped_column(BigInteger, nullable=False)
    local_path: Mapped[str] = mapped_column(Text, nullable=False)
    row_count: Mapped[int] = mapped_column(Integer, nullable=False)
    inserted_rows: Mapped[int] = mapped_column(Integer, nullable=False)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)


class FearGreedRow(Base):
    """Idempotent daily Crypto Fear & Greed Index observations (`fear_greed_observations`, S-02)."""

    __tablename__ = "fear_greed_observations"
    __table_args__ = (
        UniqueConstraint("source", "observed_time", name="uq_fear_greed_source_time"),
        Index("ix_fear_greed_available", "source", "available_time"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    source: Mapped[str] = mapped_column(String(64), nullable=False)
    observed_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    received_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    available_time: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    value: Mapped[int] = mapped_column(Integer, nullable=False)
    classification: Mapped[str] = mapped_column(String(64), nullable=False)
    normalized_score: Mapped[float] = mapped_column(Double, nullable=False)
    is_backfilled: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    raw_payload: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False, default=dict)
