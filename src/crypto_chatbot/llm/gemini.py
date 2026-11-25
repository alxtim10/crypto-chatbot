"""Gemini narration wrapper with structured output, quota caps, and safe fallback."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Sequence
from datetime import UTC, date, datetime
from typing import Any, Literal, Protocol

from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, Field, SecretStr, ValidationError

from crypto_chatbot.llm.guard import validate_narration
from crypto_chatbot.llm.prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    CONVERSATIONAL_QA_SYSTEM_PROMPT,
    INTENT_CLASSIFIER_SYSTEM_PROMPT,
    PROMPT_VERSION,
    build_narration_user_payload,
    sanitize_untrusted_text,
)
from crypto_chatbot.schemas.analysis import AnalysisSnapshot
from crypto_chatbot.schemas.llm import (
    AnalysisResponse,
    LlmAnalysisNarration,
    NarrationStatus,
)
from crypto_chatbot.schemas.market import QualityStatus


class RawGeminiOutput(BaseModel):
    """Raw structured response returned by a Gemini transport."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str | None = None
    model_version: str | None = None
    prompt_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class GeminiTransport(Protocol):
    """Transport abstraction enabling mocked offline Gemini tests."""

    async def generate_json(
        self,
        *,
        model: str,
        system_instruction: str,
        contents: str,
        response_json_schema: dict[str, Any],
        max_output_tokens: int,
        temperature: float,
    ) -> RawGeminiOutput: ...


class GoogleGenAiTransport:
    """Live Gemini transport backed by the official `google-genai` async SDK."""

    def __init__(self, *, api_key: SecretStr, timeout_seconds: float = 15.0) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        raw_key = api_key.get_secret_value().strip()
        if not raw_key:
            raise ValueError("api_key must not be blank")
        self._timeout_seconds = timeout_seconds
        self._client = genai.Client(
            api_key=raw_key,
            http_options=types.HttpOptions(timeout=int(timeout_seconds * 1000)),
        )

    def __repr__(self) -> str:
        return f"GoogleGenAiTransport(timeout_seconds={self._timeout_seconds})"

    async def generate_json(
        self,
        *,
        model: str,
        system_instruction: str,
        contents: str,
        response_json_schema: dict[str, Any],
        max_output_tokens: int,
        temperature: float,
    ) -> RawGeminiOutput:
        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            response_mime_type="application/json",
            response_json_schema=response_json_schema,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )
        response = await self._client.aio.models.generate_content(
            model=model,
            contents=contents,
            config=config,
        )
        usage = response.usage_metadata
        prompt_tokens = usage.prompt_token_count if usage is not None else None
        output_tokens = usage.candidates_token_count if usage is not None else None
        return RawGeminiOutput(
            text=response.text,
            model_version=response.model_version,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )


class DailyQuotaTracker:
    """UTC-day in-memory per-user and global Gemini call quota limiter."""

    def __init__(
        self,
        *,
        per_user_daily_cap: int = 30,
        global_daily_cap: int = 200,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if per_user_daily_cap < 1 or global_daily_cap < 1:
            raise ValueError("daily caps must be >= 1")
        self._per_user_cap = per_user_daily_cap
        self._global_cap = global_daily_cap
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._current_day: date = self._today_utc()
        self._global_count = 0
        self._user_counts: dict[str, int] = {}

    def _today_utc(self) -> date:
        now = self._clock()
        if now.tzinfo is None or now.utcoffset() is None:
            raise ValueError("quota clock must return timezone-aware UTC datetime")
        return now.astimezone(UTC).date()

    def _roll_day_if_needed(self) -> None:
        today = self._today_utc()
        if today != self._current_day:
            self._current_day = today
            self._global_count = 0
            self._user_counts.clear()

    def try_consume(self, user_id: int | str) -> tuple[bool, str | None]:
        """Attempt to reserve one Gemini call for `user_id` under UTC daily limits."""
        self._roll_day_if_needed()
        if self._global_count >= self._global_cap:
            return False, "global_daily_cap_exceeded"
        key = str(user_id)
        if self._user_counts.get(key, 0) >= self._per_user_cap:
            return False, "per_user_daily_cap_exceeded"

        self._global_count += 1
        self._user_counts[key] = self._user_counts.get(key, 0) + 1
        return True, None


class NlpIntentClassification(BaseModel):
    """Structured Layer-2 intent classification extracted from casual chat."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    intent: Literal[
        "analyze",
        "analyze_detail",
        "compare",
        "pine",
        "scan",
        "screen",
        "chat_qa",
        "unsupported",
    ]
    symbols: tuple[str, ...] = ()
    timeframe: Literal["15m", "1h", "4h", "1d", "scalp", "scalping", "intraday", "swing"] | None = (
        None
    )
    screen_side: Literal["all", "long", "short"] | None = None


class ConversationalAnswer(BaseModel):
    """Structured answer for educational or contextual research Q&A."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    answer: str = Field(min_length=1, max_length=2500)


class NarrationOutcome(BaseModel):
    """Full audit outcome of narrating an `AnalysisSnapshot` with Gemini."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    response: AnalysisResponse
    model_id: str
    prompt_version: str
    validation_passed: bool
    rejection_reasons: tuple[str, ...] = ()
    latency_ms: int = Field(ge=0)
    prompt_tokens: int | None = Field(default=None, ge=0)
    output_tokens: int | None = Field(default=None, ge=0)


class GeminiNarrator:
    """Orchestrates structured Gemini narration behind strict schema, quota, and output guards."""

    def __init__(
        self,
        *,
        model: str,
        transport: GeminiTransport,
        timeout_seconds: float = 15.0,
        max_output_tokens: int = 1024,
        temperature: float = 0.2,
        quota_tracker: DailyQuotaTracker | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_output_tokens < 64:
            raise ValueError("max_output_tokens must be >= 64")
        self._model = model.strip()
        self._transport = transport
        self._timeout_seconds = timeout_seconds
        self._max_output_tokens = max_output_tokens
        self._temperature = temperature
        self._quota_tracker = quota_tracker
        self._response_schema = LlmAnalysisNarration.model_json_schema()

    def _fallback(
        self,
        snapshot: AnalysisSnapshot,
        *,
        status: NarrationStatus,
        reasons: tuple[str, ...],
        start_monotonic: float,
        model_id: str | None = None,
        prompt_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> NarrationOutcome:
        elapsed_ms = max(0, int((time.monotonic() - start_monotonic) * 1000))
        return NarrationOutcome(
            response=AnalysisResponse(
                snapshot=snapshot,
                narration_status=status,
                narration=None,
            ),
            model_id=model_id or self._model,
            prompt_version=PROMPT_VERSION,
            validation_passed=False,
            rejection_reasons=reasons,
            latency_ms=elapsed_ms,
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
        )

    async def narrate_snapshot(
        self,
        snapshot: AnalysisSnapshot,
        *,
        user_id: int | str = "anonymous",
        user_context: str | None = None,
        external_data: Sequence[str] = (),
    ) -> NarrationOutcome:
        """Generate and validate Indonesian narration, falling back deterministically on failure."""
        started = time.monotonic()

        if snapshot.quality_status is QualityStatus.INSUFFICIENT_DATA or not snapshot.facts:
            return self._fallback(
                snapshot,
                status=NarrationStatus.SKIPPED,
                reasons=("insufficient_data_skipped",),
                start_monotonic=started,
            )

        if self._quota_tracker is not None:
            allowed, cap_reason = self._quota_tracker.try_consume(user_id)
            if not allowed:
                return self._fallback(
                    snapshot,
                    status=NarrationStatus.FALLBACK_DETERMINISTIC,
                    reasons=(cap_reason or "quota_exceeded",),
                    start_monotonic=started,
                )

        payload = build_narration_user_payload(
            snapshot,
            user_context=user_context,
            external_data=external_data,
        )

        try:
            raw = await asyncio.wait_for(
                self._transport.generate_json(
                    model=self._model,
                    system_instruction=ANALYSIS_SYSTEM_PROMPT,
                    contents=payload,
                    response_json_schema=self._response_schema,
                    max_output_tokens=self._max_output_tokens,
                    temperature=self._temperature,
                ),
                timeout=self._timeout_seconds,
            )
        except TimeoutError:
            return self._fallback(
                snapshot,
                status=NarrationStatus.FALLBACK_DETERMINISTIC,
                reasons=("gemini_timeout",),
                start_monotonic=started,
            )
        except Exception as exc:
            return self._fallback(
                snapshot,
                status=NarrationStatus.FALLBACK_DETERMINISTIC,
                reasons=(f"gemini_error:{type(exc).__name__}",),
                start_monotonic=started,
            )

        resolved_model = raw.model_version or self._model
        if not raw.text or not raw.text.strip():
            return self._fallback(
                snapshot,
                status=NarrationStatus.FALLBACK_DETERMINISTIC,
                reasons=("malformed_or_invalid_schema",),
                start_monotonic=started,
                model_id=resolved_model,
                prompt_tokens=raw.prompt_tokens,
                output_tokens=raw.output_tokens,
            )

        try:
            narration = LlmAnalysisNarration.model_validate_json(raw.text)
        except (ValidationError, ValueError):
            return self._fallback(
                snapshot,
                status=NarrationStatus.FALLBACK_DETERMINISTIC,
                reasons=("malformed_or_invalid_schema",),
                start_monotonic=started,
                model_id=resolved_model,
                prompt_tokens=raw.prompt_tokens,
                output_tokens=raw.output_tokens,
            )

        guard_result = validate_narration(narration, snapshot)
        if not guard_result.is_valid:
            return self._fallback(
                snapshot,
                status=NarrationStatus.FALLBACK_DETERMINISTIC,
                reasons=guard_result.reasons,
                start_monotonic=started,
                model_id=resolved_model,
                prompt_tokens=raw.prompt_tokens,
                output_tokens=raw.output_tokens,
            )

        elapsed_ms = max(0, int((time.monotonic() - started) * 1000))
        return NarrationOutcome(
            response=AnalysisResponse(
                snapshot=snapshot,
                narration_status=NarrationStatus.GENERATED,
                narration=narration,
            ),
            model_id=resolved_model,
            prompt_version=PROMPT_VERSION,
            validation_passed=True,
            rejection_reasons=(),
            latency_ms=elapsed_ms,
            prompt_tokens=raw.prompt_tokens,
            output_tokens=raw.output_tokens,
        )

    async def classify_intent(
        self,
        text: str,
        *,
        user_id: int | str = "anonymous",
        recent_symbols: Sequence[str] = (),
    ) -> NlpIntentClassification | None:
        """Classify natural-language chat intent into structured backend routing metadata."""
        if not text.strip():
            return None
        if self._quota_tracker is not None:
            allowed, _ = self._quota_tracker.try_consume(user_id)
            if not allowed:
                return None

        payload = json.dumps(
            {
                "recent_symbols": list(recent_symbols[:3]),
                "user_message": f"<untrusted_data>{sanitize_untrusted_text(text)}</untrusted_data>",
            },
            ensure_ascii=False,
        )
        try:
            raw = await asyncio.wait_for(
                self._transport.generate_json(
                    model=self._model,
                    system_instruction=INTENT_CLASSIFIER_SYSTEM_PROMPT,
                    contents=payload,
                    response_json_schema=NlpIntentClassification.model_json_schema(),
                    max_output_tokens=200,
                    temperature=0.0,
                ),
                timeout=self._timeout_seconds,
            )
            if not raw.text or not raw.text.strip():
                return None
            return NlpIntentClassification.model_validate_json(raw.text)
        except Exception:
            return None

    async def answer_research_question(
        self,
        question: str,
        *,
        user_id: int | str = "anonymous",
        user_context: str | None = None,
    ) -> str | None:
        """Answer an educational or contextual follow-up question safely."""
        if not question.strip():
            return None
        if self._quota_tracker is not None:
            allowed, _ = self._quota_tracker.try_consume(user_id)
            if not allowed:
                return None

        context_block = (
            f"<untrusted_data>\n{sanitize_untrusted_text(user_context)}\n</untrusted_data>"
            if user_context and user_context.strip()
            else "(belum ada riwayat analisa)"
        )
        safe_q = sanitize_untrusted_text(question)
        payload = (
            f"Konteks percakapan sebelumnya:\n{context_block}\n\n"
            f"Pertanyaan pengguna:\n<untrusted_data>{safe_q}</untrusted_data>"
        )
        try:
            raw = await asyncio.wait_for(
                self._transport.generate_json(
                    model=self._model,
                    system_instruction=CONVERSATIONAL_QA_SYSTEM_PROMPT,
                    contents=payload,
                    response_json_schema=ConversationalAnswer.model_json_schema(),
                    max_output_tokens=self._max_output_tokens,
                    temperature=0.3,
                ),
                timeout=self._timeout_seconds,
            )
            if not raw.text or not raw.text.strip():
                return None
            parsed = ConversationalAnswer.model_validate_json(raw.text)
            return parsed.answer.strip()
        except Exception:
            return None
