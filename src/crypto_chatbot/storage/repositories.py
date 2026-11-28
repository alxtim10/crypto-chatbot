"""Parameterized PostgreSQL repositories for market data and bounded user memory."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, delete, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine import Connection
from sqlalchemy.orm import Session

from crypto_chatbot.schemas.analysis import AnalysisSnapshot, TradeSetup
from crypto_chatbot.schemas.derivatives import (
    ContractStatsObservation,
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
    FearGreedObservation,
    NewsVoteSnapshot,
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
    AlertRow,
    AlertRunRow,
    AlertSymbolStateRow,
    AnalysisSnapshotRow,
    ContractStatsRow,
    ConversationRow,
    FearGreedRow,
    FundingRateRow,
    GateIngestLedgerRow,
    LiquidationEventRow,
    LlmInteractionRow,
    MarketCandleRow,
    MessageRow,
    MlExperimentRow,
    ModelPredictionRow,
    NewsArticleRow,
    NewsVoteSnapshotRow,
    OpenInterestRow,
    OrderBookSnapshotRow,
    ScalpDecisionRow,
    ScalpUniverseSnapshotRow,
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
    params_sha256: str | None = None
    strategy: str = "swing_mtf"
    mode: str | None = None
    cost_bps: float | None = None
    cost_pct_r: float | None = None
    features_json: dict[str, Any] | None = None
    user_action: str | None = None
    user_action_at: UtcDatetime | None = None
    telegram_chat_id: int | None = None
    telegram_message_id: int | None = None


class AlertRunRecord(BaseModel):
    """Persisted record of one hourly alert worker run (`alert_runs`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str
    strategy: str = "swing_mtf"
    timeframe: Timeframe
    candle_close_time: UtcDatetime
    started_at: UtcDatetime
    finished_at: UtcDatetime | None = None
    status: str = "running"
    scanned_count: int = 0
    ready_count: int = 0
    armed_count: int = 0
    expired_count: int = 0
    alerted_count: int = 0
    held_digest_count: int = 0
    failed_symbols: tuple[str, ...] = ()
    symbol_errors: dict[str, str] = Field(default_factory=dict)
    stage_failures: dict[str, dict[str, str]] = Field(default_factory=dict)


class AlertRecord(BaseModel):
    """Outbox alert record persisted before Telegram delivery (`alerts`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    alert_id: str
    run_id: str
    strategy: str = "swing_mtf"
    snapshot_id: str
    params_sha256: str | None = None
    canonical_symbol: str
    timeframe: Timeframe
    direction: str
    venue: str
    chat_id: int | None = None
    status: str = "pending"
    hold_reason: str | None = None
    message_text: str
    reply_markup_json: dict[str, Any] = Field(default_factory=dict)
    telegram_message_id: int | None = None
    send_attempts: int = 0
    last_error: str | None = None
    created_at: UtcDatetime
    sent_at: UtcDatetime | None = None


class AlertSymbolState(BaseModel):
    """Per-(symbol, timeframe) state-change, armed-setup, cooldown, and mute tracker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_symbol: str
    timeframe: Timeframe
    strategy: str = "swing_mtf"
    last_status: str
    last_bias: str | None = None
    last_run_id: str | None = None
    last_alert_time_long: UtcDatetime | None = None
    last_alert_time_short: UtcDatetime | None = None
    muted_until: UtcDatetime | None = None
    armed_direction: str | None = None
    armed_at: UtcDatetime | None = None
    armed_expires_at: UtcDatetime | None = None
    armed_setup_json: dict[str, Any] | None = None
    last_stage_failed: str | None = None
    last_stage_reason: str | None = None
    updated_at: UtcDatetime


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


class ContractStatsRepository(Protocol):
    """Repository interface for Gate.io perpetual contract_stats observations."""

    def save_contract_stats(self, observations: Sequence[ContractStatsObservation]) -> int: ...

    def load_contract_stats(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        interval: str = "5m",
        start_time: datetime | None = None,
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[ContractStatsObservation, ...]: ...

    def list_contract_stats_times(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        interval: str = "5m",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> tuple[datetime, ...]: ...


class GateIngestRepository(Protocol):
    """Repository protocol for checksum-tracked bulk Gate.io archive ingestion (D-04)."""

    def get_ingested_sha256(
        self,
        *,
        dataset: str,
        contract: str,
        yyyymm: str,
    ) -> str | None: ...

    def record_ingested_file(
        self,
        *,
        dataset: str,
        contract: str,
        yyyymm: str,
        sha256: str,
        byte_size: int,
        local_path: str,
        row_count: int,
        inserted_rows: int,
        ingested_at: datetime,
    ) -> None: ...


class FearGreedRepository(Protocol):
    """Repository protocol for daily Fear & Greed Index snapshots (S-02)."""

    def save_observations(self, observations: Sequence[FearGreedObservation]) -> int: ...

    def load_observations(
        self,
        *,
        source: str = "alternative.me",
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[FearGreedObservation, ...]: ...


class SetupLogRepository(Protocol):
    """Repository interface for persistent deterministic trade setups & outcome settlement."""

    def save_setup(
        self,
        snapshot: AnalysisSnapshot,
        setup: TradeSetup,
        *,
        params_sha256: str | None = None,
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


@runtime_checkable
class AlertRepository(Protocol):
    """Repository protocol for alert runs, outbox alerts, symbol states, and advisory locks."""

    def try_acquire_advisory_lock(self, lock_key: int) -> bool: ...

    def release_advisory_lock(self, lock_key: int) -> None: ...

    def has_completed_run(self, run_id: str) -> bool: ...

    def record_run_start(self, run: AlertRunRecord) -> bool: ...

    def record_run_finish(
        self,
        run_id: str,
        *,
        status: str,
        finished_at: datetime,
        scanned_count: int,
        ready_count: int,
        alerted_count: int,
        held_digest_count: int,
        failed_symbols: Sequence[str],
        symbol_errors: dict[str, str],
        armed_count: int = 0,
        expired_count: int = 0,
        stage_failures: dict[str, dict[str, str]] | None = None,
    ) -> None: ...

    def get_latest_completed_run(self) -> AlertRunRecord | None: ...

    def list_recent_runs(self, *, since: datetime) -> tuple[AlertRunRecord, ...]: ...

    def get_symbol_state(
        self, canonical_symbol: str, timeframe: Timeframe
    ) -> AlertSymbolState | None: ...

    def upsert_symbol_state(self, state: AlertSymbolState) -> None: ...

    def mute_symbol(
        self,
        canonical_symbol: str,
        timeframe: Timeframe,
        muted_until: datetime,
        *,
        now: datetime,
    ) -> None: ...

    def create_alert(self, alert: AlertRecord) -> bool: ...

    def list_pending_alerts(self, *, limit: int = 50) -> tuple[AlertRecord, ...]: ...

    def list_held_digest_alerts(
        self, *, since: datetime | None = None
    ) -> tuple[AlertRecord, ...]: ...

    def list_recent_alerts(self, *, since: datetime) -> tuple[AlertRecord, ...]: ...

    def mark_alert_sent(
        self,
        alert_id: str,
        *,
        telegram_message_id: int,
        sent_at: datetime,
        attempts: int,
    ) -> None: ...

    def mark_alert_failed(self, alert_id: str, *, error: str, attempts: int) -> None: ...

    def mark_alerts_digest_sent(self, alert_ids: Sequence[str], *, sent_at: datetime) -> int: ...

    def get_alert_by_id(self, alert_id: str) -> AlertRecord | None: ...

    def get_alert_by_snapshot_id(self, snapshot_id: str) -> AlertRecord | None: ...


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
                "raw_contracts": candle.raw_contracts,
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

        inserted_total = 0
        batch_size = 500
        with Session(self._engine) as session, session.begin():
            for idx in range(0, len(rows), batch_size):
                chunk = rows[idx : idx + batch_size]
                stmt = (
                    insert(MarketCandleRow)
                    .values(chunk)
                    .on_conflict_do_nothing(constraint="uq_market_candles_identity")
                    .returning(MarketCandleRow.id)
                )
                inserted_total += len(session.scalars(stmt).all())
        return inserted_total

    def upsert_taker_buy_volumes(self, candles: Sequence[MarketCandle]) -> int:
        """Update `taker_buy_base_volume` on existing closed candles (D-05)."""
        eligible = [c for c in candles if c.taker_buy_base_volume is not None]
        if not eligible:
            return 0
        updated = 0
        with Session(self._engine) as session, session.begin():
            for candle in eligible:
                stmt = (
                    update(MarketCandleRow)
                    .where(
                        MarketCandleRow.provider == candle.provider,
                        MarketCandleRow.venue == candle.venue,
                        MarketCandleRow.canonical_symbol == candle.canonical_symbol,
                        MarketCandleRow.market_type == candle.market_type.value,
                        MarketCandleRow.timeframe == candle.timeframe.value,
                        MarketCandleRow.open_time == _to_utc(candle.open_time),
                    )
                    .values(taker_buy_base_volume=candle.taker_buy_base_volume)
                )
                result = session.execute(stmt)
                updated += int(getattr(result, "rowcount", 0) or 0)
        return updated

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
                raw_contracts=row.raw_contracts,
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
    raw_features = getattr(row, "features_json", None)
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
        params_sha256=getattr(row, "params_sha256", None),
        strategy=getattr(row, "strategy", "swing_mtf") or "swing_mtf",
        mode=getattr(row, "mode", None),
        cost_bps=getattr(row, "cost_bps", None),
        cost_pct_r=getattr(row, "cost_pct_r", None),
        features_json=dict(raw_features) if isinstance(raw_features, dict) else None,
        user_action=getattr(row, "user_action", None),
        user_action_at=(_to_utc(row.user_action_at) if row.user_action_at is not None else None),
        telegram_chat_id=getattr(row, "telegram_chat_id", None),
        telegram_message_id=getattr(row, "telegram_message_id", None),
    )


class PostgresSetupLogRepository:
    """Synchronous PostgreSQL setup log & settlement repository (`trade_setups`)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save_setup(
        self,
        snapshot: AnalysisSnapshot,
        setup: TradeSetup,
        *,
        params_sha256: str | None = None,
        strategy: str = "swing_mtf",
        mode: str | None = None,
        cost_bps: float | None = None,
        cost_pct_r: float | None = None,
        features_json: dict[str, Any] | None = None,
        user_action: str | None = None,
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
            "params_sha256": params_sha256,
            "strategy": strategy,
            "mode": mode,
            "cost_bps": cost_bps,
            "cost_pct_r": cost_pct_r,
            "features_json": dict(features_json) if features_json is not None else None,
            "user_action": user_action,
        }
        stmt = (
            insert(TradeSetupRow)
            .values(payload)
            .on_conflict_do_nothing(index_elements=["snapshot_id"])
            .returning(TradeSetupRow.id)
        )
        with Session(self._engine) as session, session.begin():
            return bool(session.scalars(stmt).all())

    def record_user_action(
        self,
        snapshot_id: str,
        *,
        action: Literal["ambil", "pasang", "skip", "shadow", "shadow_expired"],
        acted_at: datetime | None = None,
    ) -> bool:
        stmt = (
            update(TradeSetupRow)
            .where(TradeSetupRow.snapshot_id == snapshot_id)
            .values(
                user_action=action,
                user_action_at=_to_utc(acted_at or datetime.now(UTC)),
            )
        )
        with Session(self._engine) as session, session.begin():
            result = session.execute(stmt)
            return _safe_rowcount(result) > 0

    def link_telegram_message(
        self,
        snapshot_id: str,
        *,
        chat_id: int,
        message_id: int,
    ) -> bool:
        stmt = (
            update(TradeSetupRow)
            .where(TradeSetupRow.snapshot_id == snapshot_id)
            .values(
                telegram_chat_id=chat_id,
                telegram_message_id=message_id,
            )
        )
        with Session(self._engine) as session, session.begin():
            result = session.execute(stmt)
            return _safe_rowcount(result) > 0

    def get_setup(self, snapshot_id: str) -> TradeSetupRecord | None:
        stmt = select(TradeSetupRow).where(TradeSetupRow.snapshot_id == snapshot_id)
        with Session(self._engine) as session:
            row = session.scalar(stmt)
        return _row_to_setup_record(row) if row is not None else None

    def list_pending_setups(
        self,
        *,
        limit: int = 100,
        strategy: str | None = None,
    ) -> tuple[TradeSetupRecord, ...]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        stmt = select(TradeSetupRow).where(
            TradeSetupRow.outcome_status.in_(("pending", "entry_triggered"))
        )
        if strategy is not None:
            stmt = stmt.where(TradeSetupRow.strategy == strategy)
        stmt = stmt.order_by(TradeSetupRow.as_of.asc()).limit(limit)
        with Session(self._engine) as session:
            rows = list(session.scalars(stmt))
        return tuple(_row_to_setup_record(row) for row in rows)

    def list_recent_setups(
        self,
        *,
        canonical_symbol: str | None = None,
        timeframe: Timeframe | None = None,
        strategy: str | None = None,
        limit: int = 20,
    ) -> tuple[TradeSetupRecord, ...]:
        if limit < 1:
            raise ValueError("limit must be >= 1")
        stmt = select(TradeSetupRow)
        if canonical_symbol is not None:
            stmt = stmt.where(TradeSetupRow.canonical_symbol == canonical_symbol)
        if timeframe is not None:
            stmt = stmt.where(TradeSetupRow.timeframe == timeframe.value)
        if strategy is not None:
            stmt = stmt.where(TradeSetupRow.strategy == strategy)
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


@runtime_checkable
class NewsVoteSnapshotRepository(Protocol):
    """Append-only persistence & causal queries for `news_vote_snapshots` (S-03)."""

    def save_vote_snapshots(self, snapshots: Sequence[NewsVoteSnapshot]) -> int: ...

    def load_vote_snapshots(
        self,
        *,
        article_id: str,
        end_snapshot_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[NewsVoteSnapshot, ...]: ...


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


def _symbol_lookup_keys(symbol: str) -> tuple[str, ...]:
    """Return normalized canonical/spot/perp/venue symbol keys for cross-table lookups."""
    raw = symbol.strip().upper()
    if not raw:
        return ()
    if "/" in raw:
        base = raw.split("/", 1)[0]
    elif "_" in raw:
        base = raw.split("_", 1)[0]
    elif ":" in raw:
        base = raw.split(":", 1)[0]
    elif raw.endswith("USDT") and len(raw) > 4:
        base = raw[:-4]
    else:
        base = raw
    candidates = (
        symbol.strip(),
        raw,
        f"{base}/USDT:USDT",
        f"{base}/USDT",
        f"{base}_USDT",
        base,
    )
    return tuple(dict.fromkeys(c for c in candidates if c))


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
                "published_time": _to_utc(obs.published_time),
                "available_time": _to_utc(obs.available_time),
                "scored_at": _to_utc(obs.scored_at) if obs.scored_at is not None else None,
                "scorer_id": obs.scorer_id,
                "scorer_version": obs.scorer_version,
                "scorer_class": obs.scorer_class,
                "is_backfilled": obs.is_backfilled,
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
        keys = _symbol_lookup_keys(canonical_symbol)
        stmt = select(SentimentObservationRow).where(
            SentimentObservationRow.canonical_symbol.in_(keys),
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

    def save_vote_snapshots(self, snapshots: Sequence[NewsVoteSnapshot]) -> int:
        items = list(snapshots)
        if not items:
            return 0
        payloads = [
            {
                "snapshot_id": snap.snapshot_id,
                "article_id": snap.article_id,
                "source_provider": snap.source_provider,
                "snapshot_time": _to_utc(snap.snapshot_time),
                "positive_votes": snap.positive_votes,
                "negative_votes": snap.negative_votes,
                "important_votes": snap.important_votes,
                "liked_votes": snap.liked_votes,
                "disliked_votes": snap.disliked_votes,
                "lol_votes": snap.lol_votes,
                "toxic_votes": snap.toxic_votes,
                "saved_votes": snap.saved_votes,
                "comments_count": snap.comments_count,
                "attribution_text": snap.attribution_text,
                "raw_payload": dict(snap.raw_payload),
            }
            for snap in items
        ]
        stmt = (
            insert(NewsVoteSnapshotRow)
            .values(payloads)
            .on_conflict_do_nothing()
            .returning(NewsVoteSnapshotRow.id)
        )
        with Session(self._engine) as session, session.begin():
            inserted_ids = session.scalars(stmt).all()
            return len(inserted_ids)

    def load_vote_snapshots(
        self,
        *,
        article_id: str,
        end_snapshot_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[NewsVoteSnapshot, ...]:
        stmt = select(NewsVoteSnapshotRow).where(NewsVoteSnapshotRow.article_id == article_id)
        if end_snapshot_time is not None:
            stmt = stmt.where(NewsVoteSnapshotRow.snapshot_time <= _to_utc(end_snapshot_time))
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be >= 1")
            stmt = stmt.order_by(
                NewsVoteSnapshotRow.snapshot_time.desc(),
                NewsVoteSnapshotRow.id.desc(),
            ).limit(limit)
        else:
            stmt = stmt.order_by(
                NewsVoteSnapshotRow.snapshot_time.asc(),
                NewsVoteSnapshotRow.id.asc(),
            )
        with Session(self._engine) as session:
            db_rows = list(session.scalars(stmt))
        if limit is not None:
            db_rows.reverse()
        return tuple(
            NewsVoteSnapshot(
                snapshot_id=row.snapshot_id,
                article_id=row.article_id,
                source_provider=row.source_provider,
                snapshot_time=_to_utc(row.snapshot_time),
                positive_votes=row.positive_votes,
                negative_votes=row.negative_votes,
                important_votes=row.important_votes,
                liked_votes=row.liked_votes,
                disliked_votes=row.disliked_votes,
                lol_votes=row.lol_votes,
                toxic_votes=row.toxic_votes,
                saved_votes=row.saved_votes,
                comments_count=row.comments_count,
                attribution_text=row.attribution_text,
                raw_payload=dict(row.raw_payload),
            )
            for row in db_rows
        )


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


class PostgresContractStatsRepository:
    """Synchronous PostgreSQL append-only repository for Gate contract_stats snapshots."""

    _INSERT_BATCH_SIZE = 500

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save_contract_stats(self, observations: Sequence[ContractStatsObservation]) -> int:
        if not observations:
            return 0
        rows = [
            {
                "venue": obs.venue,
                "canonical_symbol": obs.canonical_symbol,
                "provider_symbol": obs.provider_symbol,
                "interval": obs.interval,
                "observed_time": _to_utc(obs.observed_time),
                "received_time": _to_utc(obs.received_time),
                "available_time": _to_utc(obs.available_time),
                "lsr_taker": obs.lsr_taker,
                "lsr_account": obs.lsr_account,
                "top_lsr_account": obs.top_lsr_account,
                "top_lsr_size": obs.top_lsr_size,
                "long_liq_usd": obs.long_liq_usd,
                "short_liq_usd": obs.short_liq_usd,
                "open_interest": obs.open_interest,
                "open_interest_usd": obs.open_interest_usd,
                "mark_price": obs.mark_price,
                "last_funding_rate": obs.last_funding_rate,
                "long_users": obs.long_users,
                "short_users": obs.short_users,
                "quality_flags": [flag.value for flag in obs.quality_flags],
                "raw_payload": obs.raw_payload,
            }
            for obs in observations
        ]
        inserted_total = 0
        with Session(self._engine) as session, session.begin():
            for idx in range(0, len(rows), self._INSERT_BATCH_SIZE):
                chunk = rows[idx : idx + self._INSERT_BATCH_SIZE]
                stmt = (
                    insert(ContractStatsRow)
                    .values(chunk)
                    .on_conflict_do_nothing(constraint="uq_contract_stats_identity")
                    .returning(ContractStatsRow.id)
                )
                inserted_total += len(session.scalars(stmt).all())
        return inserted_total

    def upsert_taker_buy_volumes(self, candles: Sequence[MarketCandle]) -> int:
        """Update `taker_buy_base_volume` on existing closed candles (D-05)."""
        eligible = [c for c in candles if c.taker_buy_base_volume is not None]
        if not eligible:
            return 0
        updated = 0
        with Session(self._engine) as session, session.begin():
            for candle in eligible:
                stmt = (
                    update(MarketCandleRow)
                    .where(
                        MarketCandleRow.provider == candle.provider,
                        MarketCandleRow.venue == candle.venue,
                        MarketCandleRow.canonical_symbol == candle.canonical_symbol,
                        MarketCandleRow.market_type == candle.market_type.value,
                        MarketCandleRow.timeframe == candle.timeframe.value,
                        MarketCandleRow.open_time == _to_utc(candle.open_time),
                    )
                    .values(taker_buy_base_volume=candle.taker_buy_base_volume)
                )
                result = session.execute(stmt)
                updated += int(getattr(result, "rowcount", 0) or 0)
        return updated

    def load_contract_stats(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        interval: str = "5m",
        start_time: datetime | None = None,
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[ContractStatsObservation, ...]:
        keys = _symbol_lookup_keys(canonical_symbol)
        stmt = select(ContractStatsRow).where(
            ContractStatsRow.venue == venue,
            ContractStatsRow.canonical_symbol.in_(keys),
            ContractStatsRow.interval == interval,
        )
        if start_time is not None:
            stmt = stmt.where(ContractStatsRow.observed_time >= _to_utc(start_time))
        if end_available_time is not None:
            stmt = stmt.where(ContractStatsRow.available_time <= _to_utc(end_available_time))
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be >= 1")
            stmt = stmt.order_by(ContractStatsRow.observed_time.desc()).limit(limit)
        else:
            stmt = stmt.order_by(ContractStatsRow.observed_time.asc())
        with Session(self._engine) as session:
            db_rows = list(session.scalars(stmt))
        if limit is not None:
            db_rows.reverse()
        return tuple(
            ContractStatsObservation(
                venue=row.venue,
                canonical_symbol=row.canonical_symbol,
                provider_symbol=row.provider_symbol,
                interval=row.interval,  # type: ignore[arg-type]
                observed_time=_to_utc(row.observed_time),
                received_time=_to_utc(row.received_time),
                available_time=_to_utc(row.available_time),
                lsr_taker=row.lsr_taker,
                lsr_account=row.lsr_account,
                top_lsr_account=row.top_lsr_account,
                top_lsr_size=row.top_lsr_size,
                long_liq_usd=row.long_liq_usd,
                short_liq_usd=row.short_liq_usd,
                open_interest=row.open_interest,
                open_interest_usd=row.open_interest_usd,
                mark_price=row.mark_price,
                last_funding_rate=row.last_funding_rate,
                long_users=row.long_users,
                short_users=row.short_users,
                quality_flags=tuple(QualityFlag(flag) for flag in row.quality_flags),
                raw_payload=dict(row.raw_payload),
            )
            for row in db_rows
        )

    def list_contract_stats_times(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        interval: str = "5m",
        start_time: datetime | None = None,
        end_time: datetime | None = None,
    ) -> tuple[datetime, ...]:
        stmt = (
            select(ContractStatsRow.observed_time)
            .where(
                ContractStatsRow.venue == venue,
                ContractStatsRow.canonical_symbol == canonical_symbol,
                ContractStatsRow.interval == interval,
            )
            .order_by(ContractStatsRow.observed_time.asc())
        )
        if start_time is not None:
            stmt = stmt.where(ContractStatsRow.observed_time >= _to_utc(start_time))
        if end_time is not None:
            stmt = stmt.where(ContractStatsRow.observed_time <= _to_utc(end_time))
        with Session(self._engine) as session:
            return tuple(_to_utc(ts) for ts in session.scalars(stmt).all())


class PostgresGateIngestRepository:
    """PostgreSQL checksum ledger repository for idempotent bulk archive ingestion."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def get_ingested_sha256(
        self,
        *,
        dataset: str,
        contract: str,
        yyyymm: str,
    ) -> str | None:
        stmt = select(GateIngestLedgerRow.sha256).where(
            GateIngestLedgerRow.dataset == dataset,
            GateIngestLedgerRow.contract == contract,
            GateIngestLedgerRow.yyyymm == yyyymm,
        )
        with Session(self._engine) as session:
            return session.scalar(stmt)

    def record_ingested_file(
        self,
        *,
        dataset: str,
        contract: str,
        yyyymm: str,
        sha256: str,
        byte_size: int,
        local_path: str,
        row_count: int,
        inserted_rows: int,
        ingested_at: datetime,
    ) -> None:
        stmt = (
            insert(GateIngestLedgerRow)
            .values(
                dataset=dataset,
                contract=contract,
                yyyymm=yyyymm,
                sha256=sha256,
                byte_size=byte_size,
                local_path=local_path,
                row_count=row_count,
                inserted_rows=inserted_rows,
                ingested_at=_to_utc(ingested_at),
            )
            .on_conflict_do_update(
                constraint="uq_gate_ingest_file",
                set_={
                    "sha256": sha256,
                    "byte_size": byte_size,
                    "local_path": local_path,
                    "row_count": row_count,
                    "inserted_rows": inserted_rows,
                    "ingested_at": _to_utc(ingested_at),
                },
            )
        )
        with Session(self._engine) as session, session.begin():
            session.execute(stmt)


class PostgresFearGreedRepository:
    """Synchronous PostgreSQL repository for Alternative.me Fear & Greed observations."""

    _INSERT_BATCH_SIZE = 500

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save_observations(self, observations: Sequence[FearGreedObservation]) -> int:
        if not observations:
            return 0
        rows = [
            {
                "source": obs.source,
                "observed_time": _to_utc(obs.observed_time),
                "received_time": _to_utc(obs.received_time),
                "available_time": _to_utc(obs.available_time),
                "value": obs.value,
                "classification": obs.classification,
                "normalized_score": obs.normalized_score,
                "is_backfilled": obs.is_backfilled,
                "raw_payload": obs.raw_payload,
            }
            for obs in observations
        ]
        inserted_total = 0
        with Session(self._engine) as session, session.begin():
            for idx in range(0, len(rows), self._INSERT_BATCH_SIZE):
                chunk = rows[idx : idx + self._INSERT_BATCH_SIZE]
                stmt = (
                    insert(FearGreedRow)
                    .values(chunk)
                    .on_conflict_do_nothing(constraint="uq_fear_greed_source_time")
                    .returning(FearGreedRow.id)
                )
                inserted_total += len(session.scalars(stmt).all())
        return inserted_total

    def load_observations(
        self,
        *,
        source: str = "alternative.me",
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[FearGreedObservation, ...]:
        stmt = select(FearGreedRow).where(FearGreedRow.source == source)
        if end_available_time is not None:
            stmt = stmt.where(FearGreedRow.available_time <= _to_utc(end_available_time))
        if limit is not None:
            if limit < 1:
                raise ValueError("limit must be >= 1")
            stmt = stmt.order_by(FearGreedRow.observed_time.desc()).limit(limit)
        else:
            stmt = stmt.order_by(FearGreedRow.observed_time.asc())
        with Session(self._engine) as session:
            db_rows = list(session.scalars(stmt))
        if limit is not None:
            db_rows.reverse()
        return tuple(
            FearGreedObservation(
                source=row.source,
                observed_time=_to_utc(row.observed_time),
                received_time=_to_utc(row.received_time),
                available_time=_to_utc(row.available_time),
                value=row.value,
                classification=row.classification,
                normalized_score=row.normalized_score,
                is_backfilled=row.is_backfilled,
                raw_payload=dict(row.raw_payload),
            )
            for row in db_rows
        )


def _row_to_alert_run(row: AlertRunRow) -> AlertRunRecord:
    return AlertRunRecord(
        run_id=row.run_id,
        strategy=getattr(row, "strategy", "swing_mtf") or "swing_mtf",
        timeframe=Timeframe(row.timeframe),
        candle_close_time=_to_utc(row.candle_close_time),
        started_at=_to_utc(row.started_at),
        finished_at=_to_utc(row.finished_at) if row.finished_at else None,
        status=row.status,
        scanned_count=row.scanned_count,
        ready_count=row.ready_count,
        armed_count=row.armed_count,
        expired_count=row.expired_count,
        alerted_count=row.alerted_count,
        held_digest_count=row.held_digest_count,
        failed_symbols=tuple(row.failed_symbols or ()),
        symbol_errors=dict(row.symbol_errors or {}),
        stage_failures={k: dict(v) for k, v in (row.stage_failures or {}).items()},
    )


def _row_to_alert(row: AlertRow) -> AlertRecord:
    return AlertRecord(
        alert_id=row.alert_id,
        run_id=row.run_id,
        strategy=getattr(row, "strategy", "swing_mtf") or "swing_mtf",
        snapshot_id=row.snapshot_id,
        params_sha256=row.params_sha256,
        canonical_symbol=row.canonical_symbol,
        timeframe=Timeframe(row.timeframe),
        direction=row.direction,
        venue=row.venue,
        chat_id=row.chat_id,
        status=row.status,
        hold_reason=row.hold_reason,
        message_text=row.message_text,
        reply_markup_json=dict(row.reply_markup_json or {}),
        telegram_message_id=row.telegram_message_id,
        send_attempts=row.send_attempts,
        last_error=row.last_error,
        created_at=_to_utc(row.created_at),
        sent_at=_to_utc(row.sent_at) if row.sent_at else None,
    )


def _row_to_alert_symbol_state(row: AlertSymbolStateRow) -> AlertSymbolState:
    return AlertSymbolState(
        canonical_symbol=row.canonical_symbol,
        timeframe=Timeframe(row.timeframe),
        strategy=getattr(row, "strategy", "swing_mtf") or "swing_mtf",
        last_status=row.last_status,
        last_bias=row.last_bias,
        last_run_id=row.last_run_id,
        last_alert_time_long=_to_utc(row.last_alert_time_long)
        if row.last_alert_time_long
        else None,
        last_alert_time_short=(
            _to_utc(row.last_alert_time_short) if row.last_alert_time_short else None
        ),
        muted_until=_to_utc(row.muted_until) if row.muted_until else None,
        armed_direction=row.armed_direction,
        armed_at=_to_utc(row.armed_at) if row.armed_at else None,
        armed_expires_at=_to_utc(row.armed_expires_at) if row.armed_expires_at else None,
        armed_setup_json=dict(row.armed_setup_json) if row.armed_setup_json else None,
        last_stage_failed=row.last_stage_failed,
        last_stage_reason=row.last_stage_reason,
        updated_at=_to_utc(row.updated_at),
    )


class PostgresAlertRepository:
    """PostgreSQL implementation of `AlertRepository` with session advisory locks."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine
        self._lock_connections: dict[int, Connection] = {}

    def try_acquire_advisory_lock(self, lock_key: int) -> bool:
        if lock_key in self._lock_connections:
            return True
        conn = self._engine.connect()
        try:
            acquired = bool(
                conn.execute(
                    text("SELECT pg_try_advisory_lock(:lock_key)"),
                    {"lock_key": lock_key},
                ).scalar()
            )
            conn.commit()
        except Exception:
            conn.close()
            raise
        if not acquired:
            conn.close()
            return False
        self._lock_connections[lock_key] = conn
        return True

    def release_advisory_lock(self, lock_key: int) -> None:
        conn = self._lock_connections.pop(lock_key, None)
        if conn is None:
            return
        try:
            conn.execute(
                text("SELECT pg_advisory_unlock(:lock_key)"),
                {"lock_key": lock_key},
            )
            conn.commit()
        finally:
            conn.close()

    def has_completed_run(self, run_id: str) -> bool:
        stmt = select(AlertRunRow.id).where(
            AlertRunRow.run_id == run_id,
            AlertRunRow.status == "completed",
        )
        with Session(self._engine) as session:
            return session.scalar(stmt) is not None

    def record_run_start(self, run: AlertRunRecord) -> bool:
        stmt = (
            insert(AlertRunRow)
            .values(
                run_id=run.run_id,
                strategy=run.strategy,
                timeframe=run.timeframe.value,
                candle_close_time=_to_utc(run.candle_close_time),
                started_at=_to_utc(run.started_at),
                finished_at=None,
                status="running",
                scanned_count=run.scanned_count,
                ready_count=run.ready_count,
                armed_count=run.armed_count,
                expired_count=run.expired_count,
                alerted_count=run.alerted_count,
                held_digest_count=run.held_digest_count,
                failed_symbols=list(run.failed_symbols),
                symbol_errors=dict(run.symbol_errors),
                stage_failures={k: dict(v) for k, v in run.stage_failures.items()},
            )
            .on_conflict_do_nothing(constraint="uq_alert_runs_run_id")
            .returning(AlertRunRow.id)
        )
        with Session(self._engine) as session, session.begin():
            return bool(session.scalars(stmt).all())

    def record_run_finish(
        self,
        run_id: str,
        *,
        status: str,
        finished_at: datetime,
        scanned_count: int,
        ready_count: int,
        alerted_count: int,
        held_digest_count: int,
        failed_symbols: Sequence[str],
        symbol_errors: dict[str, str],
        armed_count: int = 0,
        expired_count: int = 0,
        stage_failures: dict[str, dict[str, str]] | None = None,
    ) -> None:
        stmt = (
            update(AlertRunRow)
            .where(AlertRunRow.run_id == run_id)
            .values(
                status=status,
                finished_at=_to_utc(finished_at),
                scanned_count=scanned_count,
                ready_count=ready_count,
                armed_count=armed_count,
                expired_count=expired_count,
                alerted_count=alerted_count,
                held_digest_count=held_digest_count,
                failed_symbols=list(failed_symbols),
                symbol_errors=dict(symbol_errors),
                stage_failures={k: dict(v) for k, v in (stage_failures or {}).items()},
            )
        )
        with Session(self._engine) as session, session.begin():
            session.execute(stmt)

    def get_latest_completed_run(self) -> AlertRunRecord | None:
        stmt = (
            select(AlertRunRow)
            .where(AlertRunRow.status == "completed")
            .order_by(AlertRunRow.finished_at.desc(), AlertRunRow.id.desc())
            .limit(1)
        )
        with Session(self._engine) as session:
            row = session.scalar(stmt)
        return _row_to_alert_run(row) if row is not None else None

    def list_recent_runs(self, *, since: datetime) -> tuple[AlertRunRecord, ...]:
        stmt = (
            select(AlertRunRow)
            .where(AlertRunRow.started_at >= _to_utc(since))
            .order_by(AlertRunRow.started_at.desc(), AlertRunRow.id.desc())
        )
        with Session(self._engine) as session:
            rows = list(session.scalars(stmt))
        return tuple(_row_to_alert_run(r) for r in rows)

    def get_symbol_state(
        self, canonical_symbol: str, timeframe: Timeframe
    ) -> AlertSymbolState | None:
        stmt = select(AlertSymbolStateRow).where(
            AlertSymbolStateRow.canonical_symbol == canonical_symbol,
            AlertSymbolStateRow.timeframe == timeframe.value,
        )
        with Session(self._engine) as session:
            row = session.scalar(stmt)
        return _row_to_alert_symbol_state(row) if row is not None else None

    def upsert_symbol_state(self, state: AlertSymbolState) -> None:
        values = {
            "canonical_symbol": state.canonical_symbol,
            "timeframe": state.timeframe.value,
            "last_status": state.last_status,
            "last_bias": state.last_bias,
            "last_run_id": state.last_run_id,
            "last_alert_time_long": (
                _to_utc(state.last_alert_time_long) if state.last_alert_time_long else None
            ),
            "last_alert_time_short": (
                _to_utc(state.last_alert_time_short) if state.last_alert_time_short else None
            ),
            "muted_until": _to_utc(state.muted_until) if state.muted_until else None,
            "armed_direction": state.armed_direction,
            "armed_at": _to_utc(state.armed_at) if state.armed_at else None,
            "armed_expires_at": (
                _to_utc(state.armed_expires_at) if state.armed_expires_at else None
            ),
            "armed_setup_json": dict(state.armed_setup_json) if state.armed_setup_json else None,
            "last_stage_failed": state.last_stage_failed,
            "last_stage_reason": state.last_stage_reason,
            "updated_at": _to_utc(state.updated_at),
        }
        stmt = (
            insert(AlertSymbolStateRow)
            .values(**values)
            .on_conflict_do_update(
                constraint="uq_alert_symbol_state_key",
                set_={
                    k: v for k, v in values.items() if k not in {"canonical_symbol", "timeframe"}
                },
            )
        )
        with Session(self._engine) as session, session.begin():
            session.execute(stmt)

    def mute_symbol(
        self,
        canonical_symbol: str,
        timeframe: Timeframe,
        muted_until: datetime,
        *,
        now: datetime,
    ) -> None:
        existing = self.get_symbol_state(canonical_symbol, timeframe)
        state = AlertSymbolState(
            canonical_symbol=canonical_symbol,
            timeframe=timeframe,
            last_status=existing.last_status if existing else "wait",
            last_bias=existing.last_bias if existing else None,
            last_run_id=existing.last_run_id if existing else None,
            last_alert_time_long=existing.last_alert_time_long if existing else None,
            last_alert_time_short=existing.last_alert_time_short if existing else None,
            muted_until=_to_utc(muted_until),
            armed_direction=existing.armed_direction if existing else None,
            armed_at=existing.armed_at if existing else None,
            armed_expires_at=existing.armed_expires_at if existing else None,
            armed_setup_json=existing.armed_setup_json if existing else None,
            last_stage_failed=existing.last_stage_failed if existing else None,
            last_stage_reason=existing.last_stage_reason if existing else None,
            updated_at=_to_utc(now),
        )
        self.upsert_symbol_state(state)

    def create_alert(self, alert: AlertRecord) -> bool:
        stmt = (
            insert(AlertRow)
            .values(
                alert_id=alert.alert_id,
                run_id=alert.run_id,
                strategy=alert.strategy,
                snapshot_id=alert.snapshot_id,
                params_sha256=alert.params_sha256,
                canonical_symbol=alert.canonical_symbol,
                timeframe=alert.timeframe.value,
                direction=alert.direction,
                venue=alert.venue,
                chat_id=alert.chat_id,
                status=alert.status,
                hold_reason=alert.hold_reason,
                message_text=alert.message_text,
                reply_markup_json=dict(alert.reply_markup_json),
                telegram_message_id=alert.telegram_message_id,
                send_attempts=alert.send_attempts,
                last_error=alert.last_error,
                created_at=_to_utc(alert.created_at),
                sent_at=_to_utc(alert.sent_at) if alert.sent_at else None,
            )
            .on_conflict_do_nothing(constraint="uq_alerts_alert_id")
            .returning(AlertRow.id)
        )
        with Session(self._engine) as session, session.begin():
            return bool(session.scalars(stmt).all())

    def list_pending_alerts(self, *, limit: int = 50) -> tuple[AlertRecord, ...]:
        stmt = (
            select(AlertRow)
            .where(AlertRow.status == "pending")
            .order_by(AlertRow.created_at.asc(), AlertRow.id.asc())
            .limit(max(1, limit))
        )
        with Session(self._engine) as session:
            rows = list(session.scalars(stmt))
        return tuple(_row_to_alert(r) for r in rows)

    def list_held_digest_alerts(self, *, since: datetime | None = None) -> tuple[AlertRecord, ...]:
        stmt = select(AlertRow).where(AlertRow.status == "held_digest")
        if since is not None:
            stmt = stmt.where(AlertRow.created_at >= _to_utc(since))
        stmt = stmt.order_by(AlertRow.created_at.asc(), AlertRow.id.asc())
        with Session(self._engine) as session:
            rows = list(session.scalars(stmt))
        return tuple(_row_to_alert(r) for r in rows)

    def list_recent_alerts(self, *, since: datetime) -> tuple[AlertRecord, ...]:
        stmt = (
            select(AlertRow)
            .where(AlertRow.created_at >= _to_utc(since))
            .order_by(AlertRow.created_at.desc(), AlertRow.id.desc())
        )
        with Session(self._engine) as session:
            rows = list(session.scalars(stmt))
        return tuple(_row_to_alert(r) for r in rows)

    def mark_alert_sent(
        self,
        alert_id: str,
        *,
        telegram_message_id: int,
        sent_at: datetime,
        attempts: int,
    ) -> None:
        stmt = (
            update(AlertRow)
            .where(AlertRow.alert_id == alert_id)
            .values(
                status="sent",
                telegram_message_id=telegram_message_id,
                sent_at=_to_utc(sent_at),
                send_attempts=attempts,
                last_error=None,
            )
        )
        with Session(self._engine) as session, session.begin():
            session.execute(stmt)

    def mark_alert_failed(self, alert_id: str, *, error: str, attempts: int) -> None:
        stmt = (
            update(AlertRow)
            .where(AlertRow.alert_id == alert_id)
            .values(
                status="failed",
                last_error=error[:500],
                send_attempts=attempts,
            )
        )
        with Session(self._engine) as session, session.begin():
            session.execute(stmt)

    def mark_alerts_digest_sent(self, alert_ids: Sequence[str], *, sent_at: datetime) -> int:
        ids = list(alert_ids)
        if not ids:
            return 0
        stmt = (
            update(AlertRow)
            .where(AlertRow.alert_id.in_(ids))
            .values(status="digest_sent", sent_at=_to_utc(sent_at))
        )
        with Session(self._engine) as session, session.begin():
            result = session.execute(stmt)
            return _safe_rowcount(result)

    def get_alert_by_id(self, alert_id: str) -> AlertRecord | None:
        stmt = select(AlertRow).where(AlertRow.alert_id == alert_id)
        with Session(self._engine) as session:
            row = session.scalar(stmt)
        return _row_to_alert(row) if row is not None else None

    def get_alert_by_snapshot_id(self, snapshot_id: str) -> AlertRecord | None:
        stmt = (
            select(AlertRow)
            .where(AlertRow.snapshot_id == snapshot_id)
            .order_by(AlertRow.created_at.desc(), AlertRow.id.desc())
            .limit(1)
        )
        with Session(self._engine) as session:
            row = session.scalar(stmt)
        return _row_to_alert(row) if row is not None else None


class ScalpDecisionRecord(BaseModel):
    """Persisted point-in-time decision & feature snapshot (`scalp_decisions`, SC-07)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    decision_id: str
    run_id: str
    strategy: str = "scalp_v1"
    venue: str
    canonical_symbol: str
    decision_time: UtcDatetime
    mode: str
    decision_status: Literal["READY", "WAIT", "NO_TRADE"]
    direction: Literal["long", "short"] | None = None
    reason_codes: tuple[str, ...] = ()
    mode_inputs_json: dict[str, Any] = Field(default_factory=dict)
    features_json: dict[str, Any] = Field(default_factory=dict)
    cost_breakdown_json: dict[str, Any] | None = None
    setup_json: dict[str, Any] | None = None
    params_sha256: str | None = None
    created_at: UtcDatetime


class ScalpUniverseSnapshotRecord(BaseModel):
    """Weekly point-in-time scalp universe eligibility snapshot (`scalp_universe_snapshots`)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: str = "scalp_v1"
    snapshot_week_utc: str
    venue: str
    captured_at: UtcDatetime
    eligible_symbols: tuple[str, ...] = ()
    evaluations_json: tuple[dict[str, Any], ...] = ()
    rules_json: dict[str, Any] = Field(default_factory=dict)


class PostgresScalpDecisionRepository:
    """PostgreSQL persistence for point-in-time `scalp_decisions` rows (SC-07)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save_decision(self, decision: ScalpDecisionRecord) -> bool:
        stmt = (
            insert(ScalpDecisionRow)
            .values(
                decision_id=decision.decision_id,
                run_id=decision.run_id,
                strategy=decision.strategy,
                venue=decision.venue,
                canonical_symbol=decision.canonical_symbol,
                decision_time=_to_utc(decision.decision_time),
                mode=decision.mode,
                decision_status=decision.decision_status,
                direction=decision.direction,
                reason_codes=list(decision.reason_codes),
                mode_inputs_json=dict(decision.mode_inputs_json),
                features_json=dict(decision.features_json),
                cost_breakdown_json=(
                    dict(decision.cost_breakdown_json)
                    if decision.cost_breakdown_json is not None
                    else None
                ),
                setup_json=(dict(decision.setup_json) if decision.setup_json is not None else None),
                params_sha256=decision.params_sha256,
                created_at=_to_utc(decision.created_at),
            )
            .on_conflict_do_nothing(constraint="uq_scalp_decisions_decision_id")
            .returning(ScalpDecisionRow.id)
        )
        with Session(self._engine) as session, session.begin():
            return bool(session.scalars(stmt).all())

    def list_decisions(
        self,
        *,
        strategy: str = "scalp_v1",
        since: datetime | None = None,
        canonical_symbol: str | None = None,
        limit: int = 500,
    ) -> tuple[ScalpDecisionRecord, ...]:
        stmt = select(ScalpDecisionRow).where(ScalpDecisionRow.strategy == strategy)
        if since is not None:
            stmt = stmt.where(ScalpDecisionRow.decision_time >= _to_utc(since))
        if canonical_symbol is not None:
            stmt = stmt.where(ScalpDecisionRow.canonical_symbol == canonical_symbol)
        stmt = stmt.order_by(
            ScalpDecisionRow.decision_time.desc(), ScalpDecisionRow.id.desc()
        ).limit(max(1, limit))
        with Session(self._engine) as session:
            rows = list(session.scalars(stmt))
        return tuple(
            ScalpDecisionRecord(
                decision_id=r.decision_id,
                run_id=r.run_id,
                strategy=r.strategy,
                venue=r.venue,
                canonical_symbol=r.canonical_symbol,
                decision_time=_to_utc(r.decision_time),
                mode=r.mode,
                decision_status=r.decision_status,  # type: ignore[arg-type]
                direction=r.direction,  # type: ignore[arg-type]
                reason_codes=tuple(r.reason_codes or ()),
                mode_inputs_json=dict(r.mode_inputs_json or {}),
                features_json=dict(r.features_json or {}),
                cost_breakdown_json=(
                    dict(r.cost_breakdown_json) if r.cost_breakdown_json is not None else None
                ),
                setup_json=dict(r.setup_json) if r.setup_json is not None else None,
                params_sha256=r.params_sha256,
                created_at=_to_utc(r.created_at),
            )
            for r in rows
        )


class PostgresScalpUniverseRepository:
    """PostgreSQL persistence for weekly PIT `scalp_universe_snapshots` (SC-03)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def save_snapshot(self, snap: ScalpUniverseSnapshotRecord) -> bool:
        stmt = (
            insert(ScalpUniverseSnapshotRow)
            .values(
                strategy=snap.strategy,
                snapshot_week_utc=snap.snapshot_week_utc,
                venue=snap.venue,
                captured_at=_to_utc(snap.captured_at),
                eligible_symbols=list(snap.eligible_symbols),
                evaluations_json=[dict(e) for e in snap.evaluations_json],
                rules_json=dict(snap.rules_json),
            )
            .on_conflict_do_nothing(constraint="uq_scalp_universe_snapshot_week")
            .returning(ScalpUniverseSnapshotRow.id)
        )
        with Session(self._engine) as session, session.begin():
            return bool(session.scalars(stmt).all())

    def get_latest_snapshot(
        self, *, strategy: str = "scalp_v1", venue: str | None = None
    ) -> ScalpUniverseSnapshotRecord | None:
        stmt = select(ScalpUniverseSnapshotRow).where(ScalpUniverseSnapshotRow.strategy == strategy)
        if venue is not None:
            stmt = stmt.where(ScalpUniverseSnapshotRow.venue == venue)
        stmt = stmt.order_by(
            ScalpUniverseSnapshotRow.captured_at.desc(),
            ScalpUniverseSnapshotRow.id.desc(),
        ).limit(1)
        with Session(self._engine) as session:
            row = session.scalar(stmt)
        if row is None:
            return None
        return ScalpUniverseSnapshotRecord(
            strategy=row.strategy,
            snapshot_week_utc=row.snapshot_week_utc,
            venue=row.venue,
            captured_at=_to_utc(row.captured_at),
            eligible_symbols=tuple(row.eligible_symbols or ()),
            evaluations_json=tuple(dict(e) for e in (row.evaluations_json or ())),
            rules_json=dict(row.rules_json or {}),
        )
