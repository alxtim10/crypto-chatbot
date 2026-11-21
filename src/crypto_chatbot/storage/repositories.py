"""Parameterized PostgreSQL repositories for market data and bounded user memory."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, delete, select
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from crypto_chatbot.schemas.analysis import AnalysisSnapshot
from crypto_chatbot.schemas.derivatives import (
    FundingRateObservation,
    LiquidationEvent,
    LiquidationSide,
    OpenInterestObservation,
    OrderBookDepthBand,
    OrderBookLevel,
    OrderBookSnapshot,
)
from crypto_chatbot.schemas.market import (
    MarketCandle,
    MarketType,
    QualityFlag,
    Timeframe,
    UtcDatetime,
)
from crypto_chatbot.storage.models import (
    AnalysisSnapshotRow,
    ConversationRow,
    FundingRateRow,
    LiquidationEventRow,
    LlmInteractionRow,
    MarketCandleRow,
    MessageRow,
    OpenInterestRow,
    OrderBookSnapshotRow,
    UserPreferenceRow,
    UserRow,
)


def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None or dt.utcoffset() is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _safe_rowcount(result: Any) -> int:
    count = getattr(result, "rowcount", 0)
    return int(count) if count is not None and count > 0 else 0


class UserPreferences(BaseModel):
    """Explicit user preferences (never inferred financial circumstances)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    watchlist: tuple[str, ...] = ()
    language: str = "id"
    default_timeframe: Timeframe = Timeframe.H4
    display_currency: str = "USDT"


class StoredMessage(BaseModel):
    """Persisted recent conversation message."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: Literal["user", "assistant"]
    content: str
    snapshot_id: str | None = None
    created_at: UtcDatetime


class LlmInteractionRecord(BaseModel):
    """Audit record for a single Gemini narration attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    telegram_user_id: int
    request_id: str
    snapshot_id: str
    model_id: str
    prompt_version: str
    validation_passed: bool
    rejection_reasons: tuple[str, ...] = ()
    prompt_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int = Field(ge=0)
    created_at: UtcDatetime | None = None


class ForgetResult(BaseModel):
    """Counts of deleted records when a user executes `/forget`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    deleted_messages: int = Field(ge=0)
    deleted_conversations: int = Field(ge=0)
    deleted_preferences: int = Field(ge=0)
    deleted_interactions: int = Field(ge=0)


class PurgeResult(BaseModel):
    """Counts of expired records deleted by retention enforcement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    deleted_messages: int = Field(ge=0)
    deleted_conversations: int = Field(ge=0)
    deleted_interactions: int = Field(ge=0)
    deleted_snapshots: int = Field(ge=0)


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


class DerivativesRepository(Protocol):
    """Repository interface for raw venue-specific derivatives & order-book observations."""

    def save_funding_observations(self, observations: Sequence[FundingRateObservation]) -> int: ...

    def load_funding_observations(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[FundingRateObservation, ...]: ...

    def save_open_interest_observations(
        self, observations: Sequence[OpenInterestObservation]
    ) -> int: ...

    def load_open_interest_observations(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        period: str | None = None,
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[OpenInterestObservation, ...]: ...

    def save_order_book_snapshot(self, snapshot: OrderBookSnapshot) -> bool: ...

    def load_order_book_snapshots(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        market_type: MarketType,
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[OrderBookSnapshot, ...]: ...

    def save_liquidation_events(self, events: Sequence[LiquidationEvent]) -> int: ...

    def load_liquidation_events(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[LiquidationEvent, ...]: ...


class UserMemoryRepository(Protocol):
    """Strictly user-scoped persistence for preferences, bounded messages, and audit logs."""

    def ensure_user(self, telegram_user_id: int) -> None: ...

    def get_preferences(self, telegram_user_id: int) -> UserPreferences: ...

    def save_preferences(self, telegram_user_id: int, prefs: UserPreferences) -> None: ...

    def append_message(
        self,
        telegram_user_id: int,
        *,
        role: Literal["user", "assistant"],
        content: str,
        snapshot_id: str | None = None,
        max_messages: int = 20,
        created_at: datetime | None = None,
    ) -> StoredMessage: ...

    def get_recent_messages(
        self,
        telegram_user_id: int,
        *,
        limit: int = 20,
    ) -> tuple[StoredMessage, ...]: ...

    def record_llm_interaction(self, record: LlmInteractionRecord) -> None: ...

    def forget_user_data(self, telegram_user_id: int) -> ForgetResult: ...

    def purge_expired(
        self,
        *,
        now: datetime,
        message_retention_days: int = 30,
        interaction_retention_days: int = 30,
        snapshot_retention_days: int = 90,
    ) -> PurgeResult: ...


class PostgresMarketRepository:
    """Synchronous PostgreSQL implementation for market candles and snapshots."""

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


class PostgresUserMemoryRepository:
    """Synchronous PostgreSQL implementation for user preferences, messages, and LLM logs."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    @staticmethod
    def _ensure_user_in_session(session: Session, telegram_user_id: int) -> None:
        stmt = (
            insert(UserRow)
            .values(
                telegram_user_id=telegram_user_id,
                is_allowlisted=True,
                created_at=datetime.now(UTC),
            )
            .on_conflict_do_nothing(index_elements=["telegram_user_id"])
        )
        session.execute(stmt)

    def ensure_user(self, telegram_user_id: int) -> None:
        with Session(self._engine) as session, session.begin():
            self._ensure_user_in_session(session, telegram_user_id)

    def get_preferences(self, telegram_user_id: int) -> UserPreferences:
        stmt = select(UserPreferenceRow).where(
            UserPreferenceRow.telegram_user_id == telegram_user_id
        )
        with Session(self._engine) as session:
            row = session.scalar(stmt)
        if row is None:
            return UserPreferences()
        return UserPreferences(
            watchlist=tuple(row.watchlist),
            language=row.language,
            default_timeframe=Timeframe(row.default_timeframe),
            display_currency=row.display_currency,
        )

    def save_preferences(self, telegram_user_id: int, prefs: UserPreferences) -> None:
        now = datetime.now(UTC)
        with Session(self._engine) as session, session.begin():
            self._ensure_user_in_session(session, telegram_user_id)
            stmt = (
                insert(UserPreferenceRow)
                .values(
                    telegram_user_id=telegram_user_id,
                    watchlist=list(prefs.watchlist),
                    language=prefs.language,
                    default_timeframe=prefs.default_timeframe.value,
                    display_currency=prefs.display_currency,
                    updated_at=now,
                )
                .on_conflict_do_update(
                    index_elements=["telegram_user_id"],
                    set_={
                        "watchlist": list(prefs.watchlist),
                        "language": prefs.language,
                        "default_timeframe": prefs.default_timeframe.value,
                        "display_currency": prefs.display_currency,
                        "updated_at": now,
                    },
                )
            )
            session.execute(stmt)

    def append_message(
        self,
        telegram_user_id: int,
        *,
        role: Literal["user", "assistant"],
        content: str,
        snapshot_id: str | None = None,
        max_messages: int = 20,
        created_at: datetime | None = None,
    ) -> StoredMessage:
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        ts = _to_utc(created_at) if created_at is not None else datetime.now(UTC)

        with Session(self._engine) as session, session.begin():
            self._ensure_user_in_session(session, telegram_user_id)
            conv_stmt = (
                select(ConversationRow)
                .where(ConversationRow.telegram_user_id == telegram_user_id)
                .order_by(ConversationRow.id.desc())
                .limit(1)
            )
            conv = session.scalar(conv_stmt)
            if conv is None:
                conv = ConversationRow(
                    telegram_user_id=telegram_user_id,
                    created_at=ts,
                    updated_at=ts,
                )
                session.add(conv)
                session.flush()
            else:
                conv.updated_at = ts

            msg = MessageRow(
                conversation_id=conv.id,
                telegram_user_id=telegram_user_id,
                role=role,
                content=content,
                snapshot_id=snapshot_id,
                created_at=ts,
            )
            session.add(msg)
            session.flush()

            excess_ids = list(
                session.scalars(
                    select(MessageRow.id)
                    .where(MessageRow.telegram_user_id == telegram_user_id)
                    .order_by(MessageRow.created_at.desc(), MessageRow.id.desc())
                    .offset(max_messages)
                )
            )
            if excess_ids:
                session.execute(
                    delete(MessageRow).where(
                        MessageRow.telegram_user_id == telegram_user_id,
                        MessageRow.id.in_(excess_ids),
                    )
                )

            return StoredMessage(
                role=role,
                content=content,
                snapshot_id=snapshot_id,
                created_at=ts,
            )

    def get_recent_messages(
        self,
        telegram_user_id: int,
        *,
        limit: int = 20,
    ) -> tuple[StoredMessage, ...]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        stmt = (
            select(MessageRow)
            .where(MessageRow.telegram_user_id == telegram_user_id)
            .order_by(MessageRow.created_at.desc(), MessageRow.id.desc())
            .limit(limit)
        )
        with Session(self._engine) as session:
            rows = list(session.scalars(stmt))
        rows.reverse()
        return tuple(
            StoredMessage(
                role="user" if row.role == "user" else "assistant",
                content=row.content,
                snapshot_id=row.snapshot_id,
                created_at=_to_utc(row.created_at),
            )
            for row in rows
        )

    def record_llm_interaction(self, record: LlmInteractionRecord) -> None:
        ts = _to_utc(record.created_at) if record.created_at is not None else datetime.now(UTC)
        with Session(self._engine) as session, session.begin():
            row = LlmInteractionRow(
                telegram_user_id=record.telegram_user_id,
                request_id=record.request_id,
                snapshot_id=record.snapshot_id,
                model_id=record.model_id,
                prompt_version=record.prompt_version,
                validation_passed=record.validation_passed,
                rejection_reasons=list(record.rejection_reasons),
                prompt_tokens=record.prompt_tokens,
                output_tokens=record.output_tokens,
                latency_ms=record.latency_ms,
                created_at=ts,
            )
            session.add(row)

    def forget_user_data(self, telegram_user_id: int) -> ForgetResult:
        with Session(self._engine) as session, session.begin():
            deleted_msgs = _safe_rowcount(
                session.execute(
                    delete(MessageRow).where(MessageRow.telegram_user_id == telegram_user_id)
                )
            )
            deleted_convs = _safe_rowcount(
                session.execute(
                    delete(ConversationRow).where(
                        ConversationRow.telegram_user_id == telegram_user_id
                    )
                )
            )
            deleted_prefs = _safe_rowcount(
                session.execute(
                    delete(UserPreferenceRow).where(
                        UserPreferenceRow.telegram_user_id == telegram_user_id
                    )
                )
            )
            deleted_ints = _safe_rowcount(
                session.execute(
                    delete(LlmInteractionRow).where(
                        LlmInteractionRow.telegram_user_id == telegram_user_id
                    )
                )
            )
        return ForgetResult(
            deleted_messages=deleted_msgs,
            deleted_conversations=deleted_convs,
            deleted_preferences=deleted_prefs,
            deleted_interactions=deleted_ints,
        )

    def purge_expired(
        self,
        *,
        now: datetime,
        message_retention_days: int = 30,
        interaction_retention_days: int = 30,
        snapshot_retention_days: int = 90,
    ) -> PurgeResult:
        now_utc = _to_utc(now)
        msg_cutoff = now_utc - timedelta(days=message_retention_days)
        int_cutoff = now_utc - timedelta(days=interaction_retention_days)
        snap_cutoff = now_utc - timedelta(days=snapshot_retention_days)

        with Session(self._engine) as session, session.begin():
            deleted_msgs = _safe_rowcount(
                session.execute(delete(MessageRow).where(MessageRow.created_at < msg_cutoff))
            )
            deleted_convs = _safe_rowcount(
                session.execute(
                    delete(ConversationRow).where(ConversationRow.updated_at < msg_cutoff)
                )
            )
            deleted_ints = _safe_rowcount(
                session.execute(
                    delete(LlmInteractionRow).where(LlmInteractionRow.created_at < int_cutoff)
                )
            )
            deleted_snaps = _safe_rowcount(
                session.execute(
                    delete(AnalysisSnapshotRow).where(AnalysisSnapshotRow.created_at < snap_cutoff)
                )
            )

        return PurgeResult(
            deleted_messages=deleted_msgs,
            deleted_conversations=deleted_convs,
            deleted_interactions=deleted_ints,
            deleted_snapshots=deleted_snaps,
        )


class PostgresDerivativesRepository:
    """Synchronous PostgreSQL repository for raw venue-specific derivatives & order books."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save_funding_observations(self, observations: Sequence[FundingRateObservation]) -> int:
        if not observations:
            return 0
        rows = [
            {
                "provider": obs.provider,
                "venue": obs.venue,
                "canonical_symbol": obs.canonical_symbol,
                "provider_symbol": obs.provider_symbol,
                "market_type": obs.market_type.value,
                "observed_time": obs.observed_time,
                "next_funding_time": obs.next_funding_time,
                "received_time": obs.received_time,
                "available_time": obs.available_time,
                "funding_rate": obs.funding_rate,
                "funding_interval_hours": obs.funding_interval_hours,
                "rate_per_8h": obs.rate_per_8h,
                "annualized_rate": obs.annualized_rate,
                "mark_price": obs.mark_price,
                "index_price": obs.index_price,
                "quality_flags": [flag.value for flag in obs.quality_flags],
                "raw_payload": obs.raw_payload,
            }
            for obs in observations
        ]
        stmt = (
            insert(FundingRateRow)
            .values(rows)
            .on_conflict_do_nothing(constraint="uq_funding_rate_identity")
            .returning(FundingRateRow.id)
        )
        with Session(self._engine) as session, session.begin():
            return len(session.scalars(stmt).all())

    def load_funding_observations(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[FundingRateObservation, ...]:
        stmt = select(FundingRateRow).where(
            FundingRateRow.venue == venue,
            FundingRateRow.canonical_symbol == canonical_symbol,
        )
        if end_available_time is not None:
            stmt = stmt.where(FundingRateRow.available_time <= _to_utc(end_available_time))
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be >= 1")
            stmt = stmt.order_by(FundingRateRow.observed_time.desc()).limit(limit)
        else:
            stmt = stmt.order_by(FundingRateRow.observed_time.asc())
        with Session(self._engine) as session:
            db_rows = list(session.scalars(stmt))
        if limit is not None:
            db_rows.reverse()
        return tuple(
            FundingRateObservation(
                provider=row.provider,
                venue=row.venue,
                canonical_symbol=row.canonical_symbol,
                provider_symbol=row.provider_symbol,
                market_type=MarketType(row.market_type),
                observed_time=_to_utc(row.observed_time),
                next_funding_time=_to_utc(row.next_funding_time) if row.next_funding_time else None,
                received_time=_to_utc(row.received_time),
                available_time=_to_utc(row.available_time),
                funding_rate=row.funding_rate,
                funding_interval_hours=row.funding_interval_hours,
                rate_per_8h=row.rate_per_8h,
                annualized_rate=row.annualized_rate,
                mark_price=row.mark_price,
                index_price=row.index_price,
                quality_flags=tuple(QualityFlag(flag) for flag in row.quality_flags),
                raw_payload=dict(row.raw_payload),
            )
            for row in db_rows
        )

    def save_open_interest_observations(
        self,
        observations: Sequence[OpenInterestObservation],
    ) -> int:
        if not observations:
            return 0
        rows = [
            {
                "provider": obs.provider,
                "venue": obs.venue,
                "canonical_symbol": obs.canonical_symbol,
                "provider_symbol": obs.provider_symbol,
                "market_type": obs.market_type.value,
                "period": obs.period,
                "observed_time": obs.observed_time,
                "received_time": obs.received_time,
                "available_time": obs.available_time,
                "open_interest_contracts": obs.open_interest_contracts,
                "raw_unit": obs.raw_unit,
                "contract_size": obs.contract_size,
                "open_interest_base": obs.open_interest_base,
                "base_unit": obs.base_unit,
                "open_interest_quote": obs.open_interest_quote,
                "quote_currency": obs.quote_currency,
                "quality_flags": [flag.value for flag in obs.quality_flags],
                "raw_payload": obs.raw_payload,
            }
            for obs in observations
        ]
        stmt = (
            insert(OpenInterestRow)
            .values(rows)
            .on_conflict_do_nothing(constraint="uq_open_interest_identity")
            .returning(OpenInterestRow.id)
        )
        with Session(self._engine) as session, session.begin():
            return len(session.scalars(stmt).all())

    def load_open_interest_observations(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        period: str | None = None,
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[OpenInterestObservation, ...]:
        stmt = select(OpenInterestRow).where(
            OpenInterestRow.venue == venue,
            OpenInterestRow.canonical_symbol == canonical_symbol,
        )
        if period is not None:
            stmt = stmt.where(OpenInterestRow.period == period)
        if end_available_time is not None:
            stmt = stmt.where(OpenInterestRow.available_time <= _to_utc(end_available_time))
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be >= 1")
            stmt = stmt.order_by(OpenInterestRow.observed_time.desc()).limit(limit)
        else:
            stmt = stmt.order_by(OpenInterestRow.observed_time.asc())
        with Session(self._engine) as session:
            db_rows = list(session.scalars(stmt))
        if limit is not None:
            db_rows.reverse()
        return tuple(
            OpenInterestObservation(
                provider=row.provider,
                venue=row.venue,
                canonical_symbol=row.canonical_symbol,
                provider_symbol=row.provider_symbol,
                market_type=MarketType(row.market_type),
                period=row.period,
                observed_time=_to_utc(row.observed_time),
                received_time=_to_utc(row.received_time),
                available_time=_to_utc(row.available_time),
                open_interest_contracts=row.open_interest_contracts,
                raw_unit=row.raw_unit,
                contract_size=row.contract_size,
                open_interest_base=row.open_interest_base,
                base_unit=row.base_unit,
                open_interest_quote=row.open_interest_quote,
                quote_currency=row.quote_currency,
                quality_flags=tuple(QualityFlag(flag) for flag in row.quality_flags),
                raw_payload=dict(row.raw_payload),
            )
            for row in db_rows
        )

    def save_order_book_snapshot(self, snapshot: OrderBookSnapshot) -> bool:
        sequence_key = str(snapshot.sequence_id) if snapshot.sequence_id is not None else ""
        stored_payload = dict(snapshot.raw_payload)
        stored_payload.setdefault("bids", [[lvl.price, lvl.quantity] for lvl in snapshot.bids[:25]])
        stored_payload.setdefault("asks", [[lvl.price, lvl.quantity] for lvl in snapshot.asks[:25]])
        payload = {
            "provider": snapshot.provider,
            "venue": snapshot.venue,
            "canonical_symbol": snapshot.canonical_symbol,
            "provider_symbol": snapshot.provider_symbol,
            "market_type": snapshot.market_type.value,
            "observed_time": snapshot.observed_time,
            "received_time": snapshot.received_time,
            "available_time": snapshot.available_time,
            "sequence_id": snapshot.sequence_id,
            "sequence_key": sequence_key,
            "best_bid": snapshot.best_bid,
            "best_ask": snapshot.best_ask,
            "mid_price": snapshot.mid_price,
            "spread_abs": snapshot.spread_abs,
            "spread_bps": snapshot.spread_bps,
            "base_unit": snapshot.base_unit,
            "quote_currency": snapshot.quote_currency,
            "resync_count": snapshot.resync_count,
            "depth_bands": [band.model_dump(mode="json") for band in snapshot.depth_bands],
            "quality_flags": [flag.value for flag in snapshot.quality_flags],
            "raw_payload": stored_payload,
        }
        stmt = (
            insert(OrderBookSnapshotRow)
            .values(payload)
            .on_conflict_do_nothing(constraint="uq_order_book_snapshot_identity")
            .returning(OrderBookSnapshotRow.id)
        )
        with Session(self._engine) as session, session.begin():
            inserted_ids = session.scalars(stmt).all()
            return bool(inserted_ids)

    def load_order_book_snapshots(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        market_type: MarketType,
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[OrderBookSnapshot, ...]:
        stmt = select(OrderBookSnapshotRow).where(
            OrderBookSnapshotRow.venue == venue,
            OrderBookSnapshotRow.canonical_symbol == canonical_symbol,
            OrderBookSnapshotRow.market_type == market_type.value,
        )
        if end_available_time is not None:
            stmt = stmt.where(OrderBookSnapshotRow.available_time <= _to_utc(end_available_time))
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be >= 1")
            stmt = stmt.order_by(OrderBookSnapshotRow.observed_time.desc()).limit(limit)
        else:
            stmt = stmt.order_by(OrderBookSnapshotRow.observed_time.asc())
        with Session(self._engine) as session:
            db_rows = list(session.scalars(stmt))
        if limit is not None:
            db_rows.reverse()

        results: list[OrderBookSnapshot] = []
        for row in db_rows:
            raw_bids = row.raw_payload.get("bids") or [[row.best_bid, 1.0]]
            raw_asks = row.raw_payload.get("asks") or [[row.best_ask, 1.0]]
            bids = tuple(
                OrderBookLevel(price=float(lvl[0]), quantity=float(lvl[1])) for lvl in raw_bids
            )
            asks = tuple(
                OrderBookLevel(price=float(lvl[0]), quantity=float(lvl[1])) for lvl in raw_asks
            )
            bands = tuple(OrderBookDepthBand.model_validate(b) for b in row.depth_bands)
            results.append(
                OrderBookSnapshot(
                    provider=row.provider,
                    venue=row.venue,
                    canonical_symbol=row.canonical_symbol,
                    provider_symbol=row.provider_symbol,
                    market_type=MarketType(row.market_type),
                    observed_time=_to_utc(row.observed_time),
                    received_time=_to_utc(row.received_time),
                    available_time=_to_utc(row.available_time),
                    sequence_id=row.sequence_id,
                    bids=bids,
                    asks=asks,
                    best_bid=row.best_bid,
                    best_ask=row.best_ask,
                    mid_price=row.mid_price,
                    spread_abs=row.spread_abs,
                    spread_bps=row.spread_bps,
                    depth_bands=bands,
                    base_unit=row.base_unit,
                    quote_currency=row.quote_currency,
                    resync_count=row.resync_count,
                    quality_flags=tuple(QualityFlag(flag) for flag in row.quality_flags),
                    raw_payload=dict(row.raw_payload),
                )
            )
        return tuple(results)

    def save_liquidation_events(self, events: Sequence[LiquidationEvent]) -> int:
        if not events:
            return 0
        rows = [
            {
                "provider": ev.provider,
                "venue": ev.venue,
                "canonical_symbol": ev.canonical_symbol,
                "provider_symbol": ev.provider_symbol,
                "market_type": ev.market_type.value,
                "event_time": ev.event_time,
                "received_time": ev.received_time,
                "available_time": ev.available_time,
                "side": ev.side.value,
                "price": ev.price,
                "base_quantity": ev.base_quantity,
                "quote_notional": ev.quote_notional,
                "base_unit": ev.base_unit,
                "quote_currency": ev.quote_currency,
                "event_key": ev.event_key,
                "raw_payload": ev.raw_payload,
            }
            for ev in events
        ]
        stmt = (
            insert(LiquidationEventRow)
            .values(rows)
            .on_conflict_do_nothing(constraint="uq_liquidation_event_identity")
            .returning(LiquidationEventRow.id)
        )
        with Session(self._engine) as session, session.begin():
            return len(session.scalars(stmt).all())

    def load_liquidation_events(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[LiquidationEvent, ...]:
        stmt = select(LiquidationEventRow).where(
            LiquidationEventRow.venue == venue,
            LiquidationEventRow.canonical_symbol == canonical_symbol,
        )
        if end_available_time is not None:
            stmt = stmt.where(LiquidationEventRow.available_time <= _to_utc(end_available_time))
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be >= 1")
            stmt = stmt.order_by(LiquidationEventRow.event_time.desc()).limit(limit)
        else:
            stmt = stmt.order_by(LiquidationEventRow.event_time.asc())
        with Session(self._engine) as session:
            db_rows = list(session.scalars(stmt))
        if limit is not None:
            db_rows.reverse()
        return tuple(
            LiquidationEvent(
                provider=row.provider,
                venue=row.venue,
                canonical_symbol=row.canonical_symbol,
                provider_symbol=row.provider_symbol,
                market_type=MarketType(row.market_type),
                event_time=_to_utc(row.event_time),
                received_time=_to_utc(row.received_time),
                available_time=_to_utc(row.available_time),
                side=LiquidationSide(row.side),
                price=row.price,
                base_quantity=row.base_quantity,
                quote_notional=row.quote_notional,
                base_unit=row.base_unit,
                quote_currency=row.quote_currency,
                event_key=row.event_key,
                raw_payload=dict(row.raw_payload),
            )
            for row in db_rows
        )
