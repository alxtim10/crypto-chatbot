"""Closed-beta allowlist and sliding-window per-user rate limit."""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable, Iterable
from typing import Literal

from pydantic import BaseModel, ConfigDict


class AccessDecision(BaseModel):
    """Result of an authorization and rate-limit check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: bool
    reason: Literal["not_allowlisted", "rate_limited"] | None = None


class AccessController:
    """Enforces closed allowlist and per-user sliding-window rate limit."""

    def __init__(
        self,
        allowed_user_ids: Iterable[int],
        *,
        rate_limit_per_minute: int = 10,
        window_seconds: float = 60.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        ids = frozenset(allowed_user_ids)
        if not ids:
            raise ValueError("allowed_user_ids must not be empty")
        if rate_limit_per_minute < 1:
            raise ValueError("rate_limit_per_minute must be >= 1")
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

        self._allowed_ids = ids
        self._limit = rate_limit_per_minute
        self._window = window_seconds
        self._clock: Callable[[], float] = clock or time.monotonic
        self._buckets: dict[int, deque[float]] = {}

    def check_access(self, user_id: int | None) -> AccessDecision:
        """Check allowlist and sliding-window rate limit."""
        if user_id is None or user_id not in self._allowed_ids:
            return AccessDecision(allowed=False, reason="not_allowlisted")

        now = self._clock()
        cutoff = now - self._window
        bucket = self._buckets.setdefault(user_id, deque())
        while bucket and bucket[0] <= cutoff:
            bucket.popleft()

        if len(bucket) >= self._limit:
            return AccessDecision(allowed=False, reason="rate_limited")

        bucket.append(now)
        return AccessDecision(allowed=True, reason=None)
