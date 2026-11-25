"""Bounded conversation memory and explicit user preference service (ADR 0001)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Protocol, runtime_checkable

from crypto_chatbot.llm.gemini import NarrationOutcome
from crypto_chatbot.llm.guard import validate_narration
from crypto_chatbot.market_data.symbols import resolve_default_perp_symbol
from crypto_chatbot.schemas.analysis import AnalysisSnapshot
from crypto_chatbot.schemas.llm import LlmAnalysisNarration
from crypto_chatbot.storage.repositories import (
    ForgetResult,
    LlmInteractionRecord,
    StoredMessage,
    UserMemoryRepository,
    UserPreferences,
)

MAX_RECENT_MESSAGES = 20
MAX_STORED_USER_MESSAGE_CHARS = 1000


@runtime_checkable
class _AtomicWatchlistRepo(Protocol):
    def update_watchlist(
        self,
        telegram_user_id: int,
        symbol: str,
        *,
        action: Literal["add", "remove"],
    ) -> tuple[UserPreferences, bool]: ...


class MemoryService:
    """Manages strictly user-scoped recent messages and explicit preferences.

    Never generates LLM conversation summaries in MVP and never stores inferred
    financial circumstances or rendered market numbers in assistant memory.
    """

    def __init__(
        self,
        repository: UserMemoryRepository,
        *,
        max_messages: int = MAX_RECENT_MESSAGES,
    ) -> None:
        if max_messages < 1:
            raise ValueError("max_messages must be >= 1")
        self._repository = repository
        self._max_messages = max_messages

    def ensure_user(self, user_id: int) -> None:
        self._repository.ensure_user(user_id)

    def get_preferences(self, user_id: int) -> UserPreferences:
        return self._repository.get_preferences(user_id)

    def add_to_watchlist(self, user_id: int, raw_symbol: str) -> tuple[UserPreferences, bool]:
        """Add an MVP perpetual symbol to user's explicit watchlist idempotently."""
        canonical = resolve_default_perp_symbol(raw_symbol).canonical_symbol
        if isinstance(self._repository, _AtomicWatchlistRepo):
            return self._repository.update_watchlist(user_id, canonical, action="add")
        prefs = self._repository.get_preferences(user_id)
        if canonical in prefs.watchlist:
            return prefs, False
        updated = prefs.model_copy(update={"watchlist": (*prefs.watchlist, canonical)})
        self._repository.save_preferences(user_id, updated)
        return updated, True

    def remove_from_watchlist(self, user_id: int, raw_symbol: str) -> tuple[UserPreferences, bool]:
        """Remove an MVP perpetual symbol from user's watchlist if present."""
        canonical = resolve_default_perp_symbol(raw_symbol).canonical_symbol
        if isinstance(self._repository, _AtomicWatchlistRepo):
            return self._repository.update_watchlist(user_id, canonical, action="remove")
        prefs = self._repository.get_preferences(user_id)
        if canonical not in prefs.watchlist:
            return prefs, False
        updated = prefs.model_copy(
            update={"watchlist": tuple(sym for sym in prefs.watchlist if sym != canonical)}
        )
        self._repository.save_preferences(user_id, updated)
        return updated, True

    def record_user_message(self, user_id: int, text: str) -> StoredMessage:
        """Persist a bounded user message."""
        trimmed = text.strip()[:MAX_STORED_USER_MESSAGE_CHARS]
        return self._repository.append_message(
            user_id,
            role="user",
            content=trimmed,
            snapshot_id=None,
            max_messages=self._max_messages,
        )

    def record_assistant_text(self, user_id: int, text: str) -> StoredMessage:
        """Persist a bounded assistant Q&A/scan summary turn without snapshot attachment."""
        trimmed = text.strip()[:MAX_STORED_USER_MESSAGE_CHARS]
        return self._repository.append_message(
            user_id,
            role="assistant",
            content=trimmed,
            snapshot_id=None,
            max_messages=self._max_messages,
        )

    def record_assistant_turn(
        self,
        user_id: int,
        *,
        snapshot: AnalysisSnapshot,
        narration: LlmAnalysisNarration | None,
    ) -> StoredMessage:
        """Persist assistant turn using `{F<n>}` placeholders and `snapshot_id` only."""
        header = f"[{snapshot.canonical_symbol} {snapshot.timeframe.value}]"
        if narration is not None and validate_narration(narration, snapshot).is_valid:
            content = f"{header} {narration.summary}"
        else:
            content = f"{header} Ringkasan deterministik status {snapshot.quality_status.value}."

        return self._repository.append_message(
            user_id,
            role="assistant",
            content=content,
            snapshot_id=snapshot.snapshot_id,
            max_messages=self._max_messages,
        )

    def record_comparison_turn(
        self,
        user_id: int,
        *,
        snapshots: Sequence[AnalysisSnapshot],
        narrations: Sequence[LlmAnalysisNarration | None],
    ) -> StoredMessage:
        """Persist side-by-side comparison turn with placeholder narration and primary ID."""
        if not snapshots:
            raise ValueError("snapshots must not be empty")
        symbols = ", ".join(snap.canonical_symbol for snap in snapshots)
        tf = snapshots[0].timeframe.value
        summaries: list[str] = []
        for snap, narration in zip(snapshots, narrations, strict=True):
            if narration is not None and validate_narration(narration, snap).is_valid:
                summaries.append(f"{snap.canonical_symbol}: {narration.summary}")
            else:
                summaries.append(f"{snap.canonical_symbol}: status {snap.quality_status.value}.")
        content = f"[{symbols} {tf}] " + " | ".join(summaries)
        return self._repository.append_message(
            user_id,
            role="assistant",
            content=content,
            snapshot_id=snapshots[0].snapshot_id,
            max_messages=self._max_messages,
        )

    def get_recent_messages(self, user_id: int) -> tuple[StoredMessage, ...]:
        return self._repository.get_recent_messages(user_id, limit=self._max_messages)

    def build_context_for_llm(
        self,
        user_id: int,
        *,
        current_question: str | None = None,
    ) -> str | None:
        """Format bounded history as non-authoritative context for Gemini."""
        recent = self.get_recent_messages(user_id)
        parts: list[str] = []
        if recent:
            parts.append(
                "Riwayat percakapan terbatas (hanya konteks topik; BUKAN fakta pasar saat ini):"
            )
            for msg in recent:
                parts.append(f"- {msg.role}: {msg.content}")
        if current_question is not None and current_question.strip():
            parts.append(f"Pertanyaan pengguna saat ini: {current_question.strip()}")
        return "\n".join(parts) if parts else None

    def record_llm_outcome(
        self,
        *,
        user_id: int,
        request_id: str,
        outcome: NarrationOutcome,
    ) -> None:
        self._repository.record_llm_interaction(
            LlmInteractionRecord(
                telegram_user_id=user_id,
                request_id=request_id,
                snapshot_id=outcome.response.snapshot.snapshot_id,
                model_id=outcome.model_id,
                prompt_version=outcome.prompt_version,
                validation_passed=outcome.validation_passed,
                rejection_reasons=outcome.rejection_reasons,
                prompt_tokens=outcome.prompt_tokens,
                output_tokens=outcome.output_tokens,
                latency_ms=outcome.latency_ms,
            )
        )

    def forget_user(self, user_id: int) -> ForgetResult:
        """Delete user's preferences, conversations, messages, and LLM interactions."""
        return self._repository.forget_user_data(user_id)
