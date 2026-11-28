"""Scalp v1 strategy package (`scalp_v1`, docs/SCALP-SPEC-v1.md)."""

from __future__ import annotations

from crypto_chatbot.scalp.config import (
    DEFAULT_EVENT_CALENDAR_PATH,
    DEFAULT_SCALP_UNIVERSE,
    DEFAULT_SCALP_UNIVERSE_PATH,
    MacroEventWindow,
    ScalpConfig,
    ScalpCostConfig,
    ScalpFilterConfig,
    ScalpPositionConfig,
    ScalpUniverseConfig,
    load_event_calendar,
)

__all__ = [
    "DEFAULT_EVENT_CALENDAR_PATH",
    "DEFAULT_SCALP_UNIVERSE",
    "DEFAULT_SCALP_UNIVERSE_PATH",
    "MacroEventWindow",
    "ScalpConfig",
    "ScalpCostConfig",
    "ScalpFilterConfig",
    "ScalpPositionConfig",
    "ScalpUniverseConfig",
    "load_event_calendar",
]
