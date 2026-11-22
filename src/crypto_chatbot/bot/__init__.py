"""Telegram bot access control, command parsing, and deterministic templates.

Import `crypto_chatbot.bot.handlers` directly for PTB handler registration so
downstream modules using `crypto_chatbot.bot.templates` avoid importing `llm`.
"""

from crypto_chatbot.bot.access import AccessController, AccessDecision
from crypto_chatbot.bot.commands import (
    MAX_USER_MESSAGE_CHARS,
    AnalyzeIntent,
    CommandParseError,
    CompareIntent,
    FollowupIntent,
    RefusalIntent,
    parse_analyze_args,
    parse_compare_args,
    parse_watchlist_args,
    resolve_followup_intent,
)
from crypto_chatbot.bot.templates import (
    TELEGRAM_MAX_MESSAGE_LENGTH,
    WIB,
    format_fact_value,
    format_indonesian_number,
    format_timestamp_utc_wib,
    render_compact_snapshot,
    render_deterministic_snapshot,
    render_forget_message,
    render_help_message,
    render_memory_message,
    render_start_message,
    render_watchlist_message,
    split_telegram_message,
)

__all__ = [
    "MAX_USER_MESSAGE_CHARS",
    "TELEGRAM_MAX_MESSAGE_LENGTH",
    "WIB",
    "AccessController",
    "AccessDecision",
    "AnalyzeIntent",
    "CommandParseError",
    "CompareIntent",
    "FollowupIntent",
    "RefusalIntent",
    "format_fact_value",
    "format_indonesian_number",
    "format_timestamp_utc_wib",
    "parse_analyze_args",
    "parse_compare_args",
    "parse_watchlist_args",
    "render_compact_snapshot",
    "render_deterministic_snapshot",
    "render_forget_message",
    "render_help_message",
    "render_memory_message",
    "render_start_message",
    "render_watchlist_message",
    "resolve_followup_intent",
    "split_telegram_message",
]
