"""Parameterized PostgreSQL repositories for market data and bounded user memory."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, delete, select, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.orm import Session

from crypto_chatbot.schemas.analysis import AnalysisSnapshot, TradeSetup
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
    QualityStatus,
    Timeframe,
    UtcDatetime,
)
from crypto_chatbot.schemas.ml import MlExperimentReport, ShadowPredictionRecord
from crypto_chatbot.schemas.news import (
    MIN_ENTITY_MATCH_CONFIDENCE,
    NormalizedNewsArticle,
    SentimentLabel,
    SentimentObservation,
)
from crypto_chatbot.schemas.tradingview import (
    FORBIDDEN_SECRET_PAYLOAD_KEYS,
    TradingViewEventType,
    TradingViewPacket,
)
from crypto_chatbot.schemas.whale import NetworkType, WhaleWatchlistEntry
from crypto_chatbot.storage.models import (
    AnalysisSnapshotRow,
    ConversationRow,
    FundingRateRow,
    LiquidationEventRow,
    LlmInteractionRow,
    MarketCandleRow,
    MessageRow,
    MlExperimentRow,
    ModelPredictionRow,
    NewsArticleRow,
    OpenInterestRow,
    OrderBookSnapshotRow,
    SentimentObservationRow,
    TradeSetupRow,
    TradingViewPacketRow,
    UserPreferenceRow,
    UserRow,
    WhaleWatchlistRow,
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
    """Audit record for a single Gemini narration or routing trace event."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    telegram_user_id: int
    request_id: str
    snapshot_id: str | None = None
    model_id: str
    prompt_version: str
    validation_passed: bool
    rejection_reasons: tuple[str, ...] = ()
    prompt_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)
    latency_ms: int = Field(ge=0)
    layer: str | None = None
    detected_symbol: str | None = None
    intent: str | None = None
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    fallback_reason: str | None = None
    guard_violations: tuple[str, ...] = ()
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


class TradeSetupRecord(BaseModel):
    """Persisted conditional trade setup and objective settlement state (`trade_setups`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot_id: str
    venue: str
    canonical_symbol: str
    market_type: str | None = None
    timeframe: Timeframe
    as_of: UtcDatetime
    status: str
    bias: str
    entry_source: str | None = None
    wait_reason: str | None = None
    reference_price: float = Field(gt=0)
    entry_low: float | None = Field(default=None, gt=0)
    entry_high: float | None = Field(default=None, gt=0)
    entry_mid: float | None = Field(default=None, gt=0)
    stop_loss: float | None = Field(default=None, gt=0)
    tp1: float | None = Field(default=None, gt=0)
    tp2: float | None = Field(default=None, gt=0)
    tp3: float | None = Field(default=None, gt=0)
    rr_tp1: float | None = Field(default=None, gt=0)
    rr_tp2: float | None = Field(default=None, gt=0)
    rr_tp3: float | None = Field(default=None, gt=0)
    outcome_status: str = "pending"
    settled_at: UtcDatetime | None = None
    realized_r: float | None = None


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


class SetupLogRepository(Protocol):
    """Repository interface for persistent deterministic trade setups & outcome settlement."""

    def save_setup(
        self,
        snapshot: AnalysisSnapshot,
        setup: TradeSetup,
    ) -> bool: ...

    def get_setup(self, snapshot_id: str) -> TradeSetupRecord | None: ...

    def list_pending_setups(self, *, limit: int = 100) -> tuple[TradeSetupRecord, ...]: ...

    def list_recent_setups(
        self,
        *,
        canonical_symbol: str | None = None,
        timeframe: Timeframe | None = None,
        limit: int = 20,
    ) -> tuple[TradeSetupRecord, ...]: ...

    def update_settlement(
        self,
        snapshot_id: str,
        *,
        outcome_status: str,
        settled_at: datetime | None,
        realized_r: float | None,
    ) -> bool: ...


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
                "taker_buy_base_volume": candle.taker_buy_base_volume,
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
                taker_buy_base_volume=row.taker_buy_base_volume,
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

    def update_watchlist(
        self,
        telegram_user_id: int,
        symbol: str,
        *,
        action: Literal["add", "remove"],
    ) -> tuple[UserPreferences, bool]:
        """Atomically add or remove a watchlist symbol under a row lock (`FOR UPDATE`)."""
        now = datetime.now(UTC)
        with Session(self._engine) as session, session.begin():
            self._ensure_user_in_session(session, telegram_user_id)
            row = session.scalar(
                select(UserPreferenceRow)
                .where(UserPreferenceRow.telegram_user_id == telegram_user_id)
                .with_for_update()
            )
            if row is None:
                defaults = UserPreferences()
                watchlist = list(defaults.watchlist)
                if action == "add":
                    if symbol in watchlist:
                        return defaults, False
                    watchlist.append(symbol)
                    changed = True
                else:
                    if symbol not in watchlist:
                        return defaults, False
                    watchlist = [s for s in watchlist if s != symbol]
                    changed = True
                session.add(
                    UserPreferenceRow(
                        telegram_user_id=telegram_user_id,
                        watchlist=watchlist,
                        language=defaults.language,
                        default_timeframe=defaults.default_timeframe.value,
                        display_currency=defaults.display_currency,
                        updated_at=now,
                    )
                )
                return defaults.model_copy(update={"watchlist": tuple(watchlist)}), changed

            watchlist = list(row.watchlist)
            if action == "add":
                if symbol in watchlist:
                    changed = False
                else:
                    watchlist.append(symbol)
                    row.watchlist = watchlist
                    row.updated_at = now
                    changed = True
            else:
                if symbol not in watchlist:
                    changed = False
                else:
                    watchlist = [s for s in watchlist if s != symbol]
                    row.watchlist = watchlist
                    row.updated_at = now
                    changed = True
            return (
                UserPreferences(
                    watchlist=tuple(watchlist),
                    language=row.language,
                    default_timeframe=Timeframe(row.default_timeframe),
                    display_currency=row.display_currency,
                ),
                changed,
            )

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
                layer=record.layer,
                detected_symbol=record.detected_symbol,
                intent=record.intent,
                confidence=record.confidence,
                fallback_reason=record.fallback_reason,
                guard_violations=list(record.guard_violations),
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


def _row_to_setup_record(row: TradeSetupRow) -> TradeSetupRecord:
    return TradeSetupRecord(
        snapshot_id=row.snapshot_id,
        venue=row.venue,
        canonical_symbol=row.canonical_symbol,
        market_type=getattr(row, "market_type", None),
        timeframe=Timeframe(row.timeframe),
        as_of=_to_utc(row.as_of),
        status=row.status,
        bias=row.bias,
        entry_source=row.entry_source,
        wait_reason=row.wait_reason,
        reference_price=row.reference_price,
        entry_low=row.entry_low,
        entry_high=row.entry_high,
        entry_mid=row.entry_mid,
        stop_loss=row.stop_loss,
        tp1=row.tp1,
        tp2=row.tp2,
        tp3=row.tp3,
        rr_tp1=row.rr_tp1,
        rr_tp2=row.rr_tp2,
        rr_tp3=row.rr_tp3,
        outcome_status=row.outcome_status,
        settled_at=_to_utc(row.settled_at) if row.settled_at else None,
        realized_r=row.realized_r,
    )


class PostgresSetupLogRepository:
    """Synchronous PostgreSQL setup log & settlement repository (`trade_setups`)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save_setup(
        self,
        snapshot: AnalysisSnapshot,
        setup: TradeSetup,
    ) -> bool:
        if snapshot.technical is None:
            return False
        payload = {
            "snapshot_id": snapshot.snapshot_id,
            "venue": snapshot.venue,
            "canonical_symbol": snapshot.canonical_symbol,
            "market_type": snapshot.market_type.value,
            "timeframe": snapshot.timeframe.value,
            "as_of": snapshot.as_of,
            "created_at": datetime.now(UTC),
            "status": setup.status.value,
            "bias": setup.bias.value,
            "entry_source": setup.entry_source,
            "wait_reason": setup.wait_reason,
            "reference_price": snapshot.technical.last_close,
            "entry_low": setup.entry_low,
            "entry_high": setup.entry_high,
            "entry_mid": setup.entry_mid,
            "stop_loss": setup.stop_loss,
            "tp1": setup.tp1,
            "tp2": setup.tp2,
            "tp3": setup.tp3,
            "rr_tp1": setup.rr_tp1,
            "rr_tp2": setup.rr_tp2,
            "rr_tp3": setup.rr_tp3,
            "outcome_status": "skipped_wait" if setup.status.value == "wait" else "pending",
            "settled_at": None,
            "realized_r": None,
        }
        stmt = (
            insert(TradeSetupRow)
            .values(payload)
            .on_conflict_do_nothing(index_elements=["snapshot_id"])
            .returning(TradeSetupRow.id)
        )
        with Session(self._engine) as session, session.begin():
            return bool(session.scalars(stmt).all())

    def get_setup(self, snapshot_id: str) -> TradeSetupRecord | None:
        stmt = select(TradeSetupRow).where(TradeSetupRow.snapshot_id == snapshot_id)
        with Session(self._engine) as session:
            row = session.scalar(stmt)
        return _row_to_setup_record(row) if row is not None else None

    def list_pending_setups(self, *, limit: int = 100) -> tuple[TradeSetupRecord, ...]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        stmt = (
            select(TradeSetupRow)
            .where(TradeSetupRow.outcome_status.in_(("pending", "entry_triggered")))
            .order_by(TradeSetupRow.as_of.asc())
            .limit(limit)
        )
        with Session(self._engine) as session:
            rows = list(session.scalars(stmt))
        return tuple(_row_to_setup_record(row) for row in rows)

    def list_recent_setups(
        self,
        *,
        canonical_symbol: str | None = None,
        timeframe: Timeframe | None = None,
        limit: int = 20,
    ) -> tuple[TradeSetupRecord, ...]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        stmt = select(TradeSetupRow)
        if canonical_symbol is not None:
            stmt = stmt.where(TradeSetupRow.canonical_symbol == canonical_symbol)
        if timeframe is not None:
            stmt = stmt.where(TradeSetupRow.timeframe == timeframe.value)
        stmt = stmt.order_by(TradeSetupRow.as_of.desc(), TradeSetupRow.id.desc()).limit(limit)
        with Session(self._engine) as session:
            rows = list(session.scalars(stmt))
        return tuple(_row_to_setup_record(row) for row in rows)

    def update_settlement(
        self,
        snapshot_id: str,
        *,
        outcome_status: str,
        settled_at: datetime | None,
        realized_r: float | None,
    ) -> bool:
        with Session(self._engine) as session, session.begin():
            row = session.scalar(
                select(TradeSetupRow).where(TradeSetupRow.snapshot_id == snapshot_id)
            )
            if row is None:
                return False
            row.outcome_status = outcome_status
            row.settled_at = _to_utc(settled_at) if settled_at is not None else None
            row.realized_r = realized_r
            return True


class TradingViewRepository(Protocol):
    """Repository interface for sanitized `ai-agent-lens/v1` TradingView webhook packets."""

    def save_packet(self, packet: TradingViewPacket) -> bool: ...

    def load_latest_packet(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        market_type: MarketType = MarketType.SPOT,
        timeframe: Timeframe,
        end_available_time: datetime | None = None,
    ) -> TradingViewPacket | None: ...

    def load_packets(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        market_type: MarketType | None = None,
        timeframe: Timeframe | None = None,
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[TradingViewPacket, ...]: ...


def _strip_forbidden_secrets(payload: Any) -> Any:
    if isinstance(payload, dict):
        return {
            str(k): _strip_forbidden_secrets(v)
            for k, v in payload.items()
            if str(k).strip().lower() not in FORBIDDEN_SECRET_PAYLOAD_KEYS
        }
    if isinstance(payload, list):
        return [_strip_forbidden_secrets(item) for item in payload]
    return payload


def _row_to_tv_packet(row: TradingViewPacketRow) -> TradingViewPacket:
    return TradingViewPacket(
        idempotency_key=row.idempotency_key,
        schema_version="ai-agent-lens/v1",
        attribution_commit=row.attribution_commit,
        provider=row.provider,
        venue=row.venue,
        canonical_symbol=row.canonical_symbol,
        provider_symbol=row.provider_symbol,
        market_type=MarketType(row.market_type),
        timeframe=Timeframe(row.timeframe),
        event_type=TradingViewEventType(row.event_type),
        candle_open_time=_to_utc(row.candle_open_time),
        candle_close_time=_to_utc(row.candle_close_time),
        received_time=_to_utc(row.received_time),
        available_time=_to_utc(row.available_time),
        open=row.open,
        high=row.high,
        low=row.low,
        close=row.close,
        volume=row.volume,
        rule_score=row.rule_score,
        rule_label=row.rule_label,
        quality_flags=tuple(QualityFlag(flag) for flag in row.quality_flags),
        sanitized_payload=_strip_forbidden_secrets(dict(row.sanitized_payload)),
    )


class PostgresTradingViewRepository:
    """Synchronous PostgreSQL persistence and point-in-time queries for `tradingview_packets`."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save_packet(self, packet: TradingViewPacket) -> bool:
        clean_payload = _strip_forbidden_secrets(packet.sanitized_payload)
        payload = {
            "idempotency_key": packet.idempotency_key,
            "schema_version": packet.schema_version,
            "attribution_commit": packet.attribution_commit,
            "provider": packet.provider,
            "venue": packet.venue,
            "canonical_symbol": packet.canonical_symbol,
            "provider_symbol": packet.provider_symbol,
            "market_type": packet.market_type.value,
            "timeframe": packet.timeframe.value,
            "event_type": packet.event_type.value,
            "candle_open_time": packet.candle_open_time,
            "candle_close_time": packet.candle_close_time,
            "received_time": packet.received_time,
            "available_time": packet.available_time,
            "open": packet.open,
            "high": packet.high,
            "low": packet.low,
            "close": packet.close,
            "volume": packet.volume,
            "rule_score": packet.rule_score,
            "rule_label": packet.rule_label,
            "is_probabilistic": False,
            "quality_flags": [flag.value for flag in packet.quality_flags],
            "sanitized_payload": clean_payload,
        }
        stmt = (
            insert(TradingViewPacketRow)
            .values(payload)
            .on_conflict_do_nothing(constraint="uq_tradingview_packet_idempotency")
            .returning(TradingViewPacketRow.id)
        )
        with Session(self._engine) as session, session.begin():
            return bool(session.scalars(stmt).all())

    def load_latest_packet(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        market_type: MarketType = MarketType.SPOT,
        timeframe: Timeframe,
        end_available_time: datetime | None = None,
    ) -> TradingViewPacket | None:
        packets = self.load_packets(
            venue=venue,
            canonical_symbol=canonical_symbol,
            market_type=market_type,
            timeframe=timeframe,
            end_available_time=end_available_time,
            limit=1,
        )
        return packets[-1] if packets else None

    def load_packets(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        market_type: MarketType | None = None,
        timeframe: Timeframe | None = None,
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[TradingViewPacket, ...]:
        stmt = select(TradingViewPacketRow).where(
            TradingViewPacketRow.venue == venue,
            TradingViewPacketRow.canonical_symbol == canonical_symbol,
        )
        if market_type is not None:
            stmt = stmt.where(TradingViewPacketRow.market_type == market_type.value)
        if timeframe is not None:
            stmt = stmt.where(TradingViewPacketRow.timeframe == timeframe.value)
        if end_available_time is not None:
            stmt = stmt.where(TradingViewPacketRow.available_time <= _to_utc(end_available_time))
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be >= 1")
            stmt = stmt.order_by(TradingViewPacketRow.candle_close_time.desc()).limit(limit)
        else:
            stmt = stmt.order_by(TradingViewPacketRow.candle_close_time.asc())
        with Session(self._engine) as session:
            db_rows = list(session.scalars(stmt))
        if limit is not None:
            db_rows.reverse()
        return tuple(_row_to_tv_packet(row) for row in db_rows)


class NewsRepository(Protocol):
    """Repository interface for canonical news articles and point-in-time sentiment."""

    def save_articles(self, articles: Sequence[NormalizedNewsArticle]) -> int: ...

    def save_observations(self, observations: Sequence[SentimentObservation]) -> int: ...

    def load_observations(
        self,
        *,
        canonical_symbol: str,
        start_available_time: datetime | None = None,
        end_available_time: datetime | None = None,
        min_entity_confidence: float = MIN_ENTITY_MATCH_CONFIDENCE,
        limit: int | None = None,
    ) -> tuple[SentimentObservation, ...]: ...


def _row_to_sentiment_observation(row: SentimentObservationRow) -> SentimentObservation:
    return SentimentObservation(
        observation_id=row.observation_id,
        article_id=row.article_id,
        canonical_url=row.canonical_url,
        publisher=row.publisher,
        title=row.title,
        canonical_symbol=row.canonical_symbol,
        base_asset=row.base_asset,
        entity_confidence=row.entity_confidence,
        sentiment_label=SentimentLabel(row.sentiment_label),
        sentiment_score=row.sentiment_score,
        sentiment_confidence=row.sentiment_confidence,
        model_version=row.model_version,
        is_probabilistic=False,
        published_time=_to_utc(row.published_time),
        available_time=_to_utc(row.available_time),
    )


class PostgresNewsRepository:
    """Synchronous PostgreSQL persistence and causal reads for news & sentiment."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save_articles(self, articles: Sequence[NormalizedNewsArticle]) -> int:
        items = list(articles)
        if not items:
            return 0
        payloads = [
            {
                "article_id": art.article_id,
                "canonical_url": art.canonical_url,
                "content_fingerprint": art.content_fingerprint,
                "publisher": art.publisher,
                "title": art.title,
                "summary": art.summary,
                "published_time": art.published_time,
                "first_seen_time": art.first_seen_time,
                "available_time": art.available_time,
                "source_provider": art.source_provider,
                "affected_entities": [e.model_dump(mode="json") for e in art.affected_entities],
                "raw_payload": dict(art.raw_payload),
            }
            for art in items
        ]
        stmt = (
            insert(NewsArticleRow)
            .values(payloads)
            .on_conflict_do_nothing()
            .returning(NewsArticleRow.id)
        )
        with Session(self._engine) as session, session.begin():
            inserted_ids = session.scalars(stmt).all()
            return len(inserted_ids)

    def save_observations(self, observations: Sequence[SentimentObservation]) -> int:
        items = list(observations)
        if not items:
            return 0
        payloads = [
            {
                "observation_id": obs.observation_id,
                "article_id": obs.article_id,
                "canonical_url": obs.canonical_url,
                "publisher": obs.publisher,
                "title": obs.title,
                "canonical_symbol": obs.canonical_symbol,
                "base_asset": obs.base_asset,
                "entity_confidence": obs.entity_confidence,
                "sentiment_label": obs.sentiment_label.value,
                "sentiment_score": obs.sentiment_score,
                "sentiment_confidence": obs.sentiment_confidence,
                "model_version": obs.model_version,
                "is_probabilistic": False,
                "published_time": obs.published_time,
                "available_time": obs.available_time,
            }
            for obs in items
        ]
        stmt = (
            insert(SentimentObservationRow)
            .values(payloads)
            .on_conflict_do_nothing()
            .returning(SentimentObservationRow.id)
        )
        with Session(self._engine) as session, session.begin():
            inserted_ids = session.scalars(stmt).all()
            return len(inserted_ids)

    def load_observations(
        self,
        *,
        canonical_symbol: str,
        start_available_time: datetime | None = None,
        end_available_time: datetime | None = None,
        min_entity_confidence: float = MIN_ENTITY_MATCH_CONFIDENCE,
        limit: int | None = None,
    ) -> tuple[SentimentObservation, ...]:
        stmt = select(SentimentObservationRow).where(
            SentimentObservationRow.canonical_symbol == canonical_symbol,
            SentimentObservationRow.entity_confidence >= min_entity_confidence,
        )
        if start_available_time is not None:
            stmt = stmt.where(
                SentimentObservationRow.available_time >= _to_utc(start_available_time)
            )
        if end_available_time is not None:
            stmt = stmt.where(SentimentObservationRow.available_time <= _to_utc(end_available_time))
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be >= 1")
            stmt = stmt.order_by(
                SentimentObservationRow.available_time.desc(),
                SentimentObservationRow.id.desc(),
            ).limit(limit)
        else:
            stmt = stmt.order_by(
                SentimentObservationRow.available_time.asc(),
                SentimentObservationRow.id.asc(),
            )
        with Session(self._engine) as session:
            db_rows = list(session.scalars(stmt))
        if limit is not None:
            db_rows.reverse()
        return tuple(_row_to_sentiment_observation(row) for row in db_rows)


class MlExperimentRepository(Protocol):
    """Repository interface for persisted offline ML experiment reports."""

    def save_experiment(self, report: MlExperimentReport) -> bool: ...

    def get_experiment(self, experiment_id: str) -> MlExperimentReport | None: ...

    def list_experiments(
        self,
        *,
        canonical_symbol: str | None = None,
        timeframe: Timeframe | None = None,
        limit: int = 20,
    ) -> tuple[MlExperimentReport, ...]: ...


class PostgresMlExperimentRepository:
    """Synchronous PostgreSQL repository for `ml_experiments`."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save_experiment(self, report: MlExperimentReport) -> bool:
        payload = {
            "experiment_id": report.experiment_id,
            "canonical_symbol": report.config.canonical_symbol,
            "venue": report.config.venue,
            "timeframe": report.config.timeframe.value,
            "target_name": report.config.target_name,
            "feature_version": report.config.feature_version,
            "model_version": report.config.model_version,
            "config_sha256": report.config_sha256,
            "dataset_sha256": report.dataset_sha256,
            "created_at": report.created_at,
            "accepted": report.accepted,
            "oof_brier": report.oof_summary.model_brier,
            "holdout_brier": report.holdout_summary.model_brier,
            "oof_net_return": report.oof_summary.net_return_after_costs,
            "holdout_net_return": report.holdout_summary.net_return_after_costs,
            "report_payload": report.model_dump(mode="json"),
        }
        stmt = (
            insert(MlExperimentRow)
            .values(payload)
            .on_conflict_do_nothing(constraint="uq_ml_experiments_id")
            .returning(MlExperimentRow.id)
        )
        with Session(self._engine) as session, session.begin():
            return bool(session.scalars(stmt).all())

    def get_experiment(self, experiment_id: str) -> MlExperimentReport | None:
        stmt = select(MlExperimentRow).where(MlExperimentRow.experiment_id == experiment_id)
        with Session(self._engine) as session:
            row = session.scalar(stmt)
        if row is None:
            return None
        return MlExperimentReport.model_validate(row.report_payload)

    def list_experiments(
        self,
        *,
        canonical_symbol: str | None = None,
        timeframe: Timeframe | None = None,
        limit: int = 20,
    ) -> tuple[MlExperimentReport, ...]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        stmt = select(MlExperimentRow)
        if canonical_symbol is not None:
            stmt = stmt.where(MlExperimentRow.canonical_symbol == canonical_symbol)
        if timeframe is not None:
            stmt = stmt.where(MlExperimentRow.timeframe == timeframe.value)
        stmt = stmt.order_by(MlExperimentRow.created_at.desc(), MlExperimentRow.id.desc()).limit(
            limit
        )
        with Session(self._engine) as session:
            rows = list(session.scalars(stmt))
        return tuple(MlExperimentReport.model_validate(r.report_payload) for r in rows)


class ModelPredictionRepository(Protocol):
    """Storage contract for shadow/promoted ML prediction logs and settlement updates."""

    def save_prediction(self, record: ShadowPredictionRecord) -> bool: ...

    def get_prediction(self, prediction_id: str) -> ShadowPredictionRecord | None: ...

    def list_predictions(
        self,
        *,
        canonical_symbol: str | None = None,
        timeframe: Timeframe | None = None,
        limit: int = 100,
    ) -> tuple[ShadowPredictionRecord, ...]: ...

    def list_unsettled_predictions(
        self,
        *,
        canonical_symbol: str | None = None,
        timeframe: Timeframe | None = None,
        limit: int = 100,
    ) -> tuple[ShadowPredictionRecord, ...]: ...

    def update_realized_outcome(
        self,
        prediction_id: str,
        *,
        realized_return: float,
        realized_label: int,
        settled_at: datetime,
    ) -> bool: ...


def _row_to_shadow_prediction(row: ModelPredictionRow) -> ShadowPredictionRecord:
    return ShadowPredictionRecord(
        prediction_id=row.prediction_id,
        snapshot_id=row.snapshot_id,
        experiment_id=row.experiment_id,
        canonical_symbol=row.canonical_symbol,
        timeframe=Timeframe(row.timeframe),
        target_name=row.target_name,
        horizon_hours=row.horizon_hours,
        prediction_time=_to_utc(row.prediction_time),
        model_version=row.model_version,
        feature_version=row.feature_version,
        quality_status=QualityStatus(row.quality_status),
        regime=row.regime,
        calibrated_probability=row.calibrated_probability,
        confidence_label=row.confidence_label,
        abstained=row.abstained,
        abstention_reason=row.abstention_reason,
        max_feature_zscore=row.max_feature_zscore,
        latency_ms=row.latency_ms,
        exposed_to_user=row.exposed_to_user,
        realized_return=row.realized_return,
        realized_label=row.realized_label,
        settled_at=_to_utc(row.settled_at) if row.settled_at is not None else None,
    )


class PostgresModelPredictionRepository:
    """PostgreSQL storage for Phase 13 shadow & released ML predictions (`model_predictions`)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save_prediction(self, record: ShadowPredictionRecord) -> bool:
        payload = {
            "prediction_id": record.prediction_id,
            "snapshot_id": record.snapshot_id,
            "experiment_id": record.experiment_id,
            "canonical_symbol": record.canonical_symbol,
            "timeframe": record.timeframe.value,
            "target_name": record.target_name,
            "horizon_hours": record.horizon_hours,
            "prediction_time": _to_utc(record.prediction_time),
            "model_version": record.model_version,
            "feature_version": record.feature_version,
            "quality_status": record.quality_status.value,
            "regime": record.regime,
            "calibrated_probability": record.calibrated_probability,
            "confidence_label": record.confidence_label,
            "abstained": record.abstained,
            "abstention_reason": record.abstention_reason,
            "max_feature_zscore": record.max_feature_zscore,
            "latency_ms": record.latency_ms,
            "exposed_to_user": record.exposed_to_user,
            "realized_return": record.realized_return,
            "realized_label": record.realized_label,
            "settled_at": _to_utc(record.settled_at) if record.settled_at is not None else None,
        }
        stmt = (
            insert(ModelPredictionRow)
            .values(payload)
            .on_conflict_do_nothing(constraint="uq_model_predictions_id")
            .returning(ModelPredictionRow.id)
        )
        with Session(self._engine) as session, session.begin():
            return bool(session.scalars(stmt).all())

    def get_prediction(self, prediction_id: str) -> ShadowPredictionRecord | None:
        stmt = select(ModelPredictionRow).where(ModelPredictionRow.prediction_id == prediction_id)
        with Session(self._engine) as session:
            row = session.scalar(stmt)
        if row is None:
            return None
        return _row_to_shadow_prediction(row)

    def list_predictions(
        self,
        *,
        canonical_symbol: str | None = None,
        timeframe: Timeframe | None = None,
        limit: int = 100,
    ) -> tuple[ShadowPredictionRecord, ...]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        stmt = select(ModelPredictionRow)
        if canonical_symbol is not None:
            stmt = stmt.where(ModelPredictionRow.canonical_symbol == canonical_symbol)
        if timeframe is not None:
            stmt = stmt.where(ModelPredictionRow.timeframe == timeframe.value)
        stmt = stmt.order_by(
            ModelPredictionRow.prediction_time.desc(), ModelPredictionRow.id.desc()
        ).limit(limit)
        with Session(self._engine) as session:
            rows = list(session.scalars(stmt))
        return tuple(_row_to_shadow_prediction(r) for r in rows)

    def list_unsettled_predictions(
        self,
        *,
        canonical_symbol: str | None = None,
        timeframe: Timeframe | None = None,
        limit: int = 100,
    ) -> tuple[ShadowPredictionRecord, ...]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        stmt = select(ModelPredictionRow).where(ModelPredictionRow.settled_at.is_(None))
        if canonical_symbol is not None:
            stmt = stmt.where(ModelPredictionRow.canonical_symbol == canonical_symbol)
        if timeframe is not None:
            stmt = stmt.where(ModelPredictionRow.timeframe == timeframe.value)
        stmt = stmt.order_by(
            ModelPredictionRow.prediction_time.asc(), ModelPredictionRow.id.asc()
        ).limit(limit)
        with Session(self._engine) as session:
            rows = list(session.scalars(stmt))
        return tuple(_row_to_shadow_prediction(r) for r in rows)

    def update_realized_outcome(
        self,
        prediction_id: str,
        *,
        realized_return: float,
        realized_label: int,
        settled_at: datetime,
    ) -> bool:
        if realized_label not in (0, 1):
            raise ValueError("realized_label must be 0 or 1")
        stmt = (
            update(ModelPredictionRow)
            .where(
                ModelPredictionRow.prediction_id == prediction_id,
                ModelPredictionRow.settled_at.is_(None),
            )
            .values(
                realized_return=realized_return,
                realized_label=realized_label,
                settled_at=_to_utc(settled_at),
            )
            .returning(ModelPredictionRow.id)
        )
        with Session(self._engine) as session, session.begin():
            return bool(session.scalars(stmt).all())


_EMPTY_WHALE_SENTINEL = "__EMPTY_WATCHLIST__"


class PostgresWhaleWatchlistRepository:
    """Synchronous PostgreSQL persistence for user whale watchlists (`whale_watchlists`)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def get_user_entries(self, user_id: int) -> tuple[WhaleWatchlistEntry, ...] | None:
        """Return saved entries for user_id, or None if the user has never customized defaults."""
        stmt = (
            select(WhaleWatchlistRow)
            .where(WhaleWatchlistRow.telegram_user_id == user_id)
            .order_by(WhaleWatchlistRow.id.asc())
        )
        with Session(self._engine) as session:
            rows = list(session.scalars(stmt))
        if not rows:
            return None
        return tuple(
            WhaleWatchlistEntry(
                address=row.address,
                network=NetworkType(row.network),
                label=row.label,
                added_at=_to_utc(row.added_at),
            )
            for row in rows
            if row.active and row.address != _EMPTY_WHALE_SENTINEL
        )

    def replace_user_entries(
        self,
        user_id: int,
        entries: Sequence[WhaleWatchlistEntry],
    ) -> None:
        """Persist the full ordered watchlist snapshot for user_id atomically."""
        now = datetime.now(UTC)
        with Session(self._engine) as session, session.begin():
            PostgresUserMemoryRepository._ensure_user_in_session(session, user_id)
            session.execute(
                update(WhaleWatchlistRow)
                .where(WhaleWatchlistRow.telegram_user_id == user_id)
                .values(active=False)
            )
            if not entries:
                existing_count = len(
                    session.scalars(
                        select(WhaleWatchlistRow.id).where(
                            WhaleWatchlistRow.telegram_user_id == user_id
                        )
                    ).all()
                )
                if existing_count == 0:
                    sentinel = (
                        insert(WhaleWatchlistRow)
                        .values(
                            telegram_user_id=user_id,
                            address=_EMPTY_WHALE_SENTINEL,
                            network=NetworkType.EVM.value,
                            label="",
                            added_at=now,
                            active=False,
                        )
                        .on_conflict_do_nothing(index_elements=["telegram_user_id", "address"])
                    )
                    session.execute(sentinel)
                return

            for entry in entries:
                stmt = (
                    insert(WhaleWatchlistRow)
                    .values(
                        telegram_user_id=user_id,
                        address=entry.address,
                        network=entry.network.value,
                        label=entry.label,
                        added_at=_to_utc(entry.added_at),
                        active=True,
                    )
                    .on_conflict_do_update(
                        index_elements=["telegram_user_id", "address"],
                        set_={
                            "network": entry.network.value,
                            "label": entry.label,
                            "added_at": _to_utc(entry.added_at),
                            "active": True,
                        },
                    )
                )
                session.execute(stmt)
