"""Venue pinning, exchange alias normalization, and geo-block health cooldown cache (D-01)."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

DEFAULT_PRIMARY_VENUE = "gateio"
DEFAULT_VENUE_COOLDOWN_MINUTES = 360

_VENUE_ALIASES: dict[str, str] = {
    "gateio": "gate",
    "gate": "gate",
    "bingx": "bingx",
    "binance": "binance",
}

_VENUE_BLOCK_SUBSTRINGS: tuple[str, ...] = (
    "403",
    "451",
    "forbidden",
    "unavailable for legal reasons",
    "restricted location",
    "from a restricted location",
    "service unavailable from a restricted",
    "geo-block",
    "geoblock",
    "connection refused",
    "econnrefused",
    "failed to establish a new connection",
    "operation not permitted",
)


def normalize_venue_id(raw: str) -> str:
    """Map config/user exchange aliases (e.g. `gateio` -> `gate`) to lowercase provider ID."""
    cleaned = raw.strip().lower()
    return _VENUE_ALIASES.get(cleaned, cleaned)


def is_venue_block_error(error: Exception | str | None) -> bool:
    """Return True when error indicates HTTP 403/451 geo-block or connection refusal."""
    if error is None:
        return False
    if isinstance(error, ConnectionRefusedError):
        return True
    text = str(error).lower()
    return any(marker in text for marker in _VENUE_BLOCK_SUBSTRINGS)


@dataclass(frozen=True, slots=True)
class VenueCooldownEntry:
    """Cooldown state for a temporarily blocked or unreachable venue."""

    venue: str
    blocked_until: float
    reason: str


class VenueHealthCache:
    """In-memory per-venue health cooldown cache to skip blocked exchanges immediately."""

    def __init__(
        self,
        *,
        default_cooldown_minutes: int = DEFAULT_VENUE_COOLDOWN_MINUTES,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if default_cooldown_minutes < 1:
            raise ValueError("default_cooldown_minutes must be >= 1")
        self._default_cooldown_seconds = float(default_cooldown_minutes * 60)
        self._clock: Callable[[], float] = clock or time.monotonic
        self._cooldowns: dict[str, VenueCooldownEntry] = {}

    @property
    def default_cooldown_minutes(self) -> int:
        return int(self._default_cooldown_seconds // 60)

    def is_available(self, venue: str) -> bool:
        key = normalize_venue_id(venue)
        entry = self._cooldowns.get(key)
        if entry is None:
            return True
        if self._clock() >= entry.blocked_until:
            self._cooldowns.pop(key, None)
            return True
        return False

    def cooldown_remaining_seconds(self, venue: str) -> float:
        key = normalize_venue_id(venue)
        entry = self._cooldowns.get(key)
        if entry is None:
            return 0.0
        remaining = entry.blocked_until - self._clock()
        if remaining <= 0.0:
            self._cooldowns.pop(key, None)
            return 0.0
        return remaining

    def blocked_reason(self, venue: str) -> str | None:
        key = normalize_venue_id(venue)
        if self.is_available(key):
            return None
        entry = self._cooldowns.get(key)
        return entry.reason if entry is not None else None

    def mark_unavailable(
        self,
        venue: str,
        *,
        reason: str,
        cooldown_minutes: int | None = None,
    ) -> None:
        key = normalize_venue_id(venue)
        seconds = (
            float(cooldown_minutes * 60)
            if cooldown_minutes is not None
            else self._default_cooldown_seconds
        )
        self._cooldowns[key] = VenueCooldownEntry(
            venue=key,
            blocked_until=self._clock() + max(1.0, seconds),
            reason=reason,
        )

    def clear(self, venue: str | None = None) -> None:
        if venue is None:
            self._cooldowns.clear()
        else:
            self._cooldowns.pop(normalize_venue_id(venue), None)

    def snapshot_status(self) -> dict[str, str]:
        now = self._clock()
        active: dict[str, str] = {}
        expired: list[str] = []
        for key, entry in self._cooldowns.items():
            remaining = entry.blocked_until - now
            if remaining <= 0:
                expired.append(key)
            else:
                active[key] = f"blocked ({int(remaining)}s remaining): {entry.reason}"
        for key in expired:
            self._cooldowns.pop(key, None)
        return active
