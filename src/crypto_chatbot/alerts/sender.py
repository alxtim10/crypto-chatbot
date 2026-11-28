"""Telegram Bot API outbox sender with HTTP 429 retry_after handling (no secrets logged)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

logger = logging.getLogger(__name__)

DEFAULT_TELEGRAM_API_BASE = "https://api.telegram.org"


class TelegramSendError(RuntimeError):
    """Raised when Telegram Bot API delivery fails after bounded retries."""

    def __init__(self, message: str, *, attempts: int = 1) -> None:
        super().__init__(message)
        self.attempts = attempts


class TelegramOutboxSender:
    """Async Telegram `sendMessage` client honoring HTTP 429 `retry_after`."""

    def __init__(
        self,
        *,
        bot_token: str,
        api_base_url: str = DEFAULT_TELEGRAM_API_BASE,
        timeout_seconds: float = 15.0,
        max_retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ) -> None:
        token = bot_token.strip()
        if not token:
            raise ValueError("bot_token must not be blank")
        self._endpoint = f"{api_base_url.rstrip('/')}/bot{token}/sendMessage"
        self._max_retries = max(1, int(max_retries))
        self._sleep = sleep_fn or asyncio.sleep
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            transport=transport,
        )

    async def close(self) -> None:
        await self._client.aclose()

    async def send_message(
        self,
        *,
        chat_id: int,
        text: str,
        parse_mode: str = "HTML",
        reply_markup: dict[str, Any] | None = None,
        reply_to_message_id: int | None = None,
        disable_web_page_preview: bool = True,
    ) -> tuple[int, int]:
        """Deliver one Telegram message; returns `(telegram_message_id, attempts)`."""
        payload: dict[str, Any] = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": parse_mode,
            "disable_web_page_preview": disable_web_page_preview,
        }
        if reply_markup:
            payload["reply_markup"] = reply_markup
        if reply_to_message_id is not None:
            payload["reply_to_message_id"] = reply_to_message_id
            payload["allow_sending_without_reply"] = True

        last_err = "unknown error"
        for attempt in range(1, self._max_retries + 1):
            try:
                resp = await self._client.post(self._endpoint, json=payload)
            except httpx.TransportError as exc:
                last_err = f"transport error: {type(exc).__name__}"
                if attempt < self._max_retries:
                    await self._sleep(float(attempt))
                    continue
                raise TelegramSendError(last_err, attempts=attempt) from exc

            if resp.status_code == 429:
                retry_after = 1.0
                try:
                    body = resp.json()
                    raw_ra = body.get("parameters", {}).get("retry_after")
                    if raw_ra is not None:
                        retry_after = max(0.1, float(raw_ra))
                except Exception:
                    hdr = resp.headers.get("Retry-After")
                    if hdr:
                        try:
                            retry_after = max(0.1, float(hdr))
                        except ValueError:
                            retry_after = 1.0
                last_err = f"HTTP 429 rate limited (retry_after={retry_after}s)"
                logger.warning(
                    "Telegram sendMessage rate limited on attempt %d/%d; sleeping %.1fs",
                    attempt,
                    self._max_retries,
                    retry_after,
                )
                if attempt < self._max_retries:
                    await self._sleep(retry_after)
                    continue
                raise TelegramSendError(last_err, attempts=attempt)

            if resp.status_code in {500, 502, 503, 504}:
                last_err = f"Telegram HTTP {resp.status_code}"
                if attempt < self._max_retries:
                    await self._sleep(float(attempt))
                    continue
                raise TelegramSendError(last_err, attempts=attempt)

            try:
                body = resp.json()
            except ValueError as exc:
                raise TelegramSendError(
                    f"Telegram HTTP {resp.status_code}: invalid JSON response",
                    attempts=attempt,
                ) from exc

            if resp.status_code != 200 or not body.get("ok"):
                desc = str(body.get("description") or f"HTTP {resp.status_code}")
                raise TelegramSendError(desc, attempts=attempt)

            result = body.get("result") or {}
            msg_id = result.get("message_id")
            if not isinstance(msg_id, int):
                raise TelegramSendError(
                    "Telegram sendMessage succeeded without integer message_id",
                    attempts=attempt,
                )
            return msg_id, attempt

        raise TelegramSendError(last_err, attempts=self._max_retries)
