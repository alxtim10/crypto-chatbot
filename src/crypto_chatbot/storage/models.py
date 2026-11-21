"""SQLAlchemy ORM models for Phase 5 market data and analysis snapshot tables."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Double,
    Index,
    Integer,
    String,
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
