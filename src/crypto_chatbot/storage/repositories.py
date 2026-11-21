"""Parameterized PostgreSQL repositories for market candles and analysis snapshots."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol

from sqlalchemy import Engine, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from crypto_chatbot.schemas.analysis import AnalysisSnapshot
from crypto_chatbot.schemas.market import (
    MarketCandle,
    MarketType,
    QualityFlag,
    Timeframe,
)
from crypto_chatbot.storage.models import AnalysisSnapshotRow, MarketCandleRow


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class MarketRepository(Protocol):
    """Repository interface for immutable candles and deterministic snapshots."""

    def save_candles(self, candles: Sequence[MarketCandle]) -> int: ...

    def load_candles(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        market_type: MarketType,
        timeframe: Timeframe,
        start_open_time: datetime | None = None,
        end_open_time: datetime | None = None,
        end_close_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[MarketCandle, ...]: ...

    def save_snapshot(self, snapshot: AnalysisSnapshot) -> None: ...

    def get_snapshot(self, snapshot_id: str) -> AnalysisSnapshot | None: ...


class PostgresMarketRepository:
    """Synchronous PostgreSQL implementation using `psycopg` 3."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save_candles(self, candles: Sequence[MarketCandle]) -> int:
        """Persist closed candles idempotently; existing rows stay immutable."""
        if not candles:
            return 0

        rows = [
            {
                "provider": candle.provider,
                "venue": candle.venue,
                "canonical_symbol": candle.canonical_symbol,
                "provider_symbol": candle.provider_symbol,
                "market_type": candle.market_type.value,
                "timeframe": candle.timeframe.value,
                "open_time": candle.open_time,
                "close_time": candle.close_time,
                "received_time": candle.received_time,
                "available_time": candle.available_time,
                "open": candle.open,
                "high": candle.high,
                "low": candle.low,
                "close": candle.close,
                "base_volume": candle.base_volume,
                "quote_volume": candle.quote_volume,
                "base_unit": candle.base_unit,
                "quote_currency": candle.quote_currency,
                "is_closed": candle.is_closed,
                "quality_flags": [flag.value for flag in candle.quality_flags],
            }
            for candle in candles
        ]

        stmt = (
            insert(MarketCandleRow)
            .values(rows)
            .on_conflict_do_nothing(constraint="uq_market_candles_identity")
            .returning(MarketCandleRow.id)
        )
        with Session(self._engine) as session, session.begin():
            inserted_ids = session.scalars(stmt).all()
            return len(inserted_ids)

    def load_candles(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        market_type: MarketType,
        timeframe: Timeframe,
        start_open_time: datetime | None = None,
        end_open_time: datetime | None = None,
        end_close_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[MarketCandle, ...]:
        """Load stored candles ordered ascending by `open_time`."""
        stmt = select(MarketCandleRow).where(
            MarketCandleRow.venue == venue,
            MarketCandleRow.canonical_symbol == canonical_symbol,
            MarketCandleRow.market_type == market_type.value,
            MarketCandleRow.timeframe == timeframe.value,
        )
        if start_open_time is not None:
            stmt = stmt.where(MarketCandleRow.open_time >= _to_utc(start_open_time))
        if end_open_time is not None:
            stmt = stmt.where(MarketCandleRow.open_time <= _to_utc(end_open_time))
        if end_close_time is not None:
            stmt = stmt.where(MarketCandleRow.close_time <= _to_utc(end_close_time))

        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be >= 1")
            stmt = stmt.order_by(MarketCandleRow.open_time.desc()).limit(limit)
        else:
            stmt = stmt.order_by(MarketCandleRow.open_time.asc())

        with Session(self._engine) as session:
            db_rows = list(session.scalars(stmt))

        if limit is not None:
            db_rows.reverse()

        return tuple(
            MarketCandle(
                provider=row.provider,
                venue=row.venue,
                canonical_symbol=row.canonical_symbol,
                provider_symbol=row.provider_symbol,
                market_type=MarketType(row.market_type),
                timeframe=Timeframe(row.timeframe),
                open_time=_to_utc(row.open_time),
                close_time=_to_utc(row.close_time),
                received_time=_to_utc(row.received_time),
                available_time=_to_utc(row.available_time),
                open=row.open,
                high=row.high,
                low=row.low,
                close=row.close,
                base_volume=row.base_volume,
                quote_volume=row.quote_volume,
                base_unit=row.base_unit,
                quote_currency=row.quote_currency,
                is_closed=row.is_closed,
                quality_flags=tuple(QualityFlag(flag) for flag in row.quality_flags),
            )
            for row in db_rows
        )

    def save_snapshot(self, snapshot: AnalysisSnapshot) -> None:
        """Persist an `AnalysisSnapshot` idempotently (contains no user ID or secrets)."""
        payload = {
            "snapshot_id": snapshot.snapshot_id,
            "provider": snapshot.provider,
            "venue": snapshot.venue,
            "canonical_symbol": snapshot.canonical_symbol,
            "provider_symbol": snapshot.provider_symbol,
            "market_type": snapshot.market_type.value,
            "timeframe": snapshot.timeframe.value,
            "calc_version": snapshot.calc_version,
            "quality_status": snapshot.quality_status.value,
            "as_of": snapshot.as_of,
            "first_open_time": (
                snapshot.input_range.first_open_time if snapshot.input_range else None
            ),
            "last_open_time": (
                snapshot.input_range.last_open_time if snapshot.input_range else None
            ),
            "candle_count": snapshot.input_range.candle_count if snapshot.input_range else None,
            "input_sha256": snapshot.input_range.sha256 if snapshot.input_range else None,
            "quality_flags": [flag.value for flag in snapshot.quality_flags],
            "snapshot_json": snapshot.model_dump(mode="json"),
        }
        stmt = (
            insert(AnalysisSnapshotRow)
            .values(payload)
            .on_conflict_do_nothing(index_elements=["snapshot_id"])
        )
        with Session(self._engine) as session, session.begin():
            session.execute(stmt)

    def get_snapshot(self, snapshot_id: str) -> AnalysisSnapshot | None:
        """Fetch an `AnalysisSnapshot` by ID, or return `None` if absent."""
        stmt = select(AnalysisSnapshotRow).where(AnalysisSnapshotRow.snapshot_id == snapshot_id)
        with Session(self._engine) as session:
            row = session.scalar(stmt)
        if row is None:
            return None
        return AnalysisSnapshot.model_validate(row.snapshot_json)
