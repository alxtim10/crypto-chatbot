"""Dead-man monitor for `run-bot` that alerts if `alerts run` stops succeeding."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Callable
from datetime import UTC, datetime, timedelta

from crypto_chatbot.alerts.sender import TelegramOutboxSender
from crypto_chatbot.storage.repositories import AlertRepository

logger = logging.getLogger(__name__)


class AlertDeadManMonitor:
    """Periodic check that sends a 🔧 warning if no successful `alerts run` occurred recently."""

    def __init__(
        self,
        *,
        alert_repo: AlertRepository,
        sender: TelegramOutboxSender,
        chat_id: int,
        max_silence_hours: float = 1.0,
        check_interval_seconds: float = 600.0,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._alert_repo = alert_repo
        self._sender = sender
        self._chat_id = chat_id
        self._max_silence = timedelta(hours=max(0.1, float(max_silence_hours)))
        self._check_interval = max(10.0, float(check_interval_seconds))
        self._clock = clock or (lambda: datetime.now(UTC))
        self._last_notified_at: datetime | None = None
        self._stop_event = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    async def check_once(self) -> bool:
        """Return True if a 🔧 dead-man warning was sent during this check."""
        now = self._clock().astimezone(UTC)
        latest = self._alert_repo.get_latest_completed_run()
        if latest is not None and latest.finished_at is not None:
            age = now - latest.finished_at.astimezone(UTC)
            if age <= self._max_silence:
                return False
            finished_str = latest.finished_at.astimezone(UTC).strftime("%Y-%m-%d %H:%M UTC")
            detail = (
                f"run terakhir selesai {finished_str} ({age.total_seconds() / 3600.0:.1f} jam lalu)"
            )
        else:
            detail = "belum ada catatan `alerts run` yang selesai sukses di database"

        if (
            self._last_notified_at is not None
            and (now - self._last_notified_at) < self._max_silence
        ):
            return False

        msg = (
            "🔧 <b>Dead-Man Alert: Alert Worker Tidak Aktif</b>\n"
            f"Tidak ada eksekusi sukses <code>alerts run</code> dalam "
            f"{self._max_silence.total_seconds() / 3600.0:.1f} jam terakhir ({detail}).\n"
            "Periksa systemd timer / log VPS."
        )
        await self._sender.send_message(chat_id=self._chat_id, text=msg, parse_mode="HTML")
        self._last_notified_at = now
        return True

    async def _loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self.check_once()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.warning("AlertDeadManMonitor check failed: %s", exc)
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(self._stop_event.wait(), timeout=self._check_interval)

    def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._stop_event.clear()
        self._task = asyncio.create_task(self._loop(), name="alert-deadman-monitor")

    async def stop(self) -> None:
        self._stop_event.set()
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None
