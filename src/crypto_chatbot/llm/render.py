"""Placeholder substitution and combined deterministic + narration rendering."""

from __future__ import annotations

import html
import re

from pydantic import BaseModel, ConfigDict

from crypto_chatbot.bot.templates import (
    format_fact_value,
    render_deterministic_snapshot,
)
from crypto_chatbot.llm.guard import validate_narration
from crypto_chatbot.schemas.analysis import AnalysisSnapshot
from crypto_chatbot.schemas.llm import (
    AnalysisResponse,
    LlmAnalysisNarration,
    NarrationStatus,
)

_PLACEHOLDER_RE = re.compile(r"\{(F[1-9][0-9]*)\}")


class RenderedNarration(BaseModel):
    """LLM narration with `{F<n>}` placeholders resolved to backend values."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    summary: str
    supporting_factors: tuple[str, ...]
    contradicting_factors: tuple[str, ...]
    bullish_scenario: str
    bearish_scenario: str
    limitations: tuple[str, ...]
    fact_ids: tuple[str, ...]


def render_narration(
    narration: LlmAnalysisNarration,
    snapshot: AnalysisSnapshot,
) -> RenderedNarration:
    """Validate and substitute all `{F<n>}` placeholders with deterministic formatted values."""
    guard = validate_narration(narration, snapshot)
    if not guard.is_valid:
        raise ValueError(f"narration failed output guard: {', '.join(guard.reasons)}")

    rendered_facts = {fact.fact_id: format_fact_value(fact) for fact in snapshot.facts}

    def _subst(text: str) -> str:
        return _PLACEHOLDER_RE.sub(lambda m: rendered_facts[m.group(1)], text)

    return RenderedNarration(
        summary=_subst(narration.summary),
        supporting_factors=tuple(_subst(item) for item in narration.supporting_factors),
        contradicting_factors=tuple(_subst(item) for item in narration.contradicting_factors),
        bullish_scenario=_subst(narration.bullish_scenario),
        bearish_scenario=_subst(narration.bearish_scenario),
        limitations=tuple(_subst(item) for item in narration.limitations),
        fact_ids=narration.fact_ids,
    )


def _esc(text: str, html_mode: bool) -> str:
    return html.escape(text, quote=False) if html_mode else text


def _bold(text: str, html_mode: bool) -> str:
    escaped = _esc(text, html_mode)
    return f"<b>{escaped}</b>" if html_mode else escaped


def render_analysis_response(
    response: AnalysisResponse,
    *,
    html_mode: bool = False,
) -> str:
    """Render backend facts first, plus validated Gemini narration if generated."""
    base = render_deterministic_snapshot(response.snapshot, html_mode=html_mode)
    if response.narration_status is not NarrationStatus.GENERATED or response.narration is None:
        return base

    rendered = render_narration(response.narration, response.snapshot)
    lines: list[str] = [base, "", _bold("Penjelasan Narasi (Tervalidasi)", html_mode)]
    lines.append(f"Ringkasan: {_esc(rendered.summary, html_mode)}")

    if rendered.supporting_factors:
        lines.append("Faktor pendukung:")
        for item in rendered.supporting_factors:
            lines.append(f"- {_esc(item, html_mode)}")

    if rendered.contradicting_factors:
        lines.append("Faktor penyeimbang:")
        for item in rendered.contradicting_factors:
            lines.append(f"- {_esc(item, html_mode)}")

    lines.append(f"Skenario naik: {_esc(rendered.bullish_scenario, html_mode)}")
    lines.append(f"Skenario turun: {_esc(rendered.bearish_scenario, html_mode)}")

    if rendered.limitations:
        lines.append("Catatan tambahan:")
        for note in rendered.limitations:
            lines.append(f"- {_esc(note, html_mode)}")

    return "\n".join(lines)
