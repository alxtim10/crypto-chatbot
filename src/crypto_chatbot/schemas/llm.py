"""LLM structured output and combined analysis response schemas (PROJECT_PLAN §5.9)."""

from __future__ import annotations

import re
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from crypto_chatbot.schemas.analysis import AnalysisSnapshot

_FACT_ID_RE = re.compile(r"^F[1-9][0-9]*$")
_PLACEHOLDER_RE = re.compile(r"\{(F[1-9][0-9]*)\}")


class NarrationStatus(StrEnum):
    GENERATED = "generated"
    FALLBACK_DETERMINISTIC = "fallback_deterministic"
    SKIPPED = "skipped"


class LlmAnalysisNarration(BaseModel):
    """Structured Gemini output. Regime/prices come only from backend; numbers via `{F<n>}`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    summary: str = Field(min_length=1)
    supporting_factors: tuple[str, ...] = ()
    contradicting_factors: tuple[str, ...] = ()
    bullish_scenario: str = Field(min_length=1)
    bearish_scenario: str = Field(min_length=1)
    limitations: tuple[str, ...] = ()
    fact_ids: tuple[str, ...] = ()

    @field_validator(
        "summary",
        "bullish_scenario",
        "bearish_scenario",
    )
    @classmethod
    def _not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text field must not be blank")
        return value

    @field_validator("supporting_factors", "contradicting_factors", "limitations")
    @classmethod
    def _items_not_blank(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for item in values:
            if not item.strip():
                raise ValueError("list items must not be blank")
        return values

    @field_validator("fact_ids")
    @classmethod
    def _valid_fact_ids(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        for fid in values:
            if not _FACT_ID_RE.match(fid):
                raise ValueError(f"invalid fact_id format: '{fid}'")
        if len(values) != len(set(values)):
            raise ValueError("fact_ids must not contain duplicates")
        return values

    @model_validator(mode="after")
    def _validate_placeholder_coverage(self) -> LlmAnalysisNarration:
        text_parts = [
            self.summary,
            *self.supporting_factors,
            *self.contradicting_factors,
            self.bullish_scenario,
            self.bearish_scenario,
            *self.limitations,
        ]
        referenced = set()
        for part in text_parts:
            referenced.update(_PLACEHOLDER_RE.findall(part))

        declared = set(self.fact_ids)
        undeclared = sorted(referenced - declared)
        if undeclared:
            raise ValueError(
                f"placeholders referenced in text must be listed in fact_ids: {undeclared}"
            )
        return self


class AnalysisResponse(BaseModel):
    """Combined deterministic snapshot and optional validated LLM narration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    snapshot: AnalysisSnapshot
    narration_status: NarrationStatus
    narration: LlmAnalysisNarration | None = None

    @model_validator(mode="after")
    def _validate_narration_consistency(self) -> AnalysisResponse:
        if self.narration_status is NarrationStatus.GENERATED and self.narration is None:
            raise ValueError("narration is required when narration_status is GENERATED")
        if self.narration_status is not NarrationStatus.GENERATED and self.narration is not None:
            raise ValueError("narration must be None unless narration_status is GENERATED")
        if self.narration is not None:
            known = {fact.fact_id for fact in self.snapshot.facts}
            unknown = sorted(set(self.narration.fact_ids) - known)
            if unknown:
                raise ValueError(f"narration references unknown snapshot fact_ids: {unknown}")
        return self
