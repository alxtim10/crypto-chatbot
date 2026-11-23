"""Standard secret redaction for application and bot logging."""

from __future__ import annotations

import logging
import re
import traceback
from collections.abc import Iterable

_SECRET_LOG_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bbot\d{6,12}:[A-Za-z0-9_-]{25,}\b"), "bot[REDACTED]"),
    (re.compile(r"\b\d{8,12}:[A-Za-z0-9_-]{35}\b"), "[REDACTED_TOKEN]"),
    (re.compile(r"AIza[0-9A-Za-z_-]{20,}"), "[REDACTED_KEY]"),
    (re.compile(r"(postgresql[^\s]*://[^:\s]+:)([^@\s]+)(@)"), r"\1[REDACTED]\3"),
    (
        re.compile(r"(Authorization\s*[:=]\s*Bearer\s+)([^\s,;\"']+)", re.IGNORECASE),
        r"\1[REDACTED]",
    ),
    (
        re.compile(
            r'(["\']?(?:secret|secret_token|webhook_secret|token|api_key|password)["\']?\s*[:=]\s*["\']?)([^"\'\s,}]+)(["\']?)',
            re.IGNORECASE,
        ),
        r"\1[REDACTED]\3",
    ),
    (
        re.compile(
            r"([?&](?:secret|secret_token|webhook_secret|token|api_key|password)=)([^&\s\"']+)",
            re.IGNORECASE,
        ),
        r"\1[REDACTED]",
    ),
)


class SecretRedactingFilter(logging.Filter):
    """Redacts known secrets and credential-shaped tokens from all log records."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        super().__init__()
        self._literals = tuple(
            sorted({s for s in secrets if len(s.strip()) >= 4}, key=len, reverse=True)
        )

    def redact_text(self, text: str) -> str:
        masked = text
        for literal in self._literals:
            if literal in masked:
                masked = masked.replace(literal, "[REDACTED]")
        for pattern, replacement in _SECRET_LOG_PATTERNS:
            masked = pattern.sub(replacement, masked)
        return masked

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            formatted = record.getMessage()
        except Exception:
            formatted = str(record.msg)
        redacted = self.redact_text(formatted)
        if redacted != formatted or record.args:
            record.msg = redacted
            record.args = ()
        if record.exc_info:
            formatted_exc = "".join(traceback.format_exception(*record.exc_info))
            record.exc_text = self.redact_text(formatted_exc).rstrip("\n")
            record.exc_info = None
        elif record.exc_text:
            record.exc_text = self.redact_text(record.exc_text)
        return True
