"""Gemini tool-bound orchestration, prompt isolation, output guard, and rendering."""

from crypto_chatbot.llm.gemini import (
    DailyQuotaTracker,
    GeminiNarrator,
    GeminiTransport,
    GoogleGenAiTransport,
    NarrationOutcome,
    RawGeminiOutput,
)
from crypto_chatbot.llm.guard import (
    OVERCONFIDENCE_PATTERNS,
    GuardResult,
    validate_narration,
)
from crypto_chatbot.llm.prompts import (
    ANALYSIS_SYSTEM_PROMPT,
    MAX_UNTRUSTED_TEXT_CHARS,
    PROMPT_VERSION,
    build_narration_user_payload,
    sanitize_untrusted_text,
)
from crypto_chatbot.llm.render import (
    RenderedNarration,
    render_analysis_response,
    render_comparison_response,
    render_narration,
)
from crypto_chatbot.llm.tools import (
    ALLOWED_TOOL_NAMES,
    ToolError,
    ToolExecutionResult,
    ToolRegistry,
    build_gemini_tool_declarations,
)

__all__ = [
    "ALLOWED_TOOL_NAMES",
    "ANALYSIS_SYSTEM_PROMPT",
    "MAX_UNTRUSTED_TEXT_CHARS",
    "OVERCONFIDENCE_PATTERNS",
    "PROMPT_VERSION",
    "DailyQuotaTracker",
    "GeminiNarrator",
    "GeminiTransport",
    "GoogleGenAiTransport",
    "GuardResult",
    "NarrationOutcome",
    "RawGeminiOutput",
    "RenderedNarration",
    "ToolError",
    "ToolExecutionResult",
    "ToolRegistry",
    "build_gemini_tool_declarations",
    "build_narration_user_payload",
    "render_analysis_response",
    "render_comparison_response",
    "render_narration",
    "sanitize_untrusted_text",
    "validate_narration",
]
