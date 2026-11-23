"""Unit tests for personal operations, secret redaction, health checks, and backup helpers."""

from __future__ import annotations

import json
import logging
from unittest.mock import MagicMock

import pytest

from crypto_chatbot import cli
from crypto_chatbot.bot.access import AccessController
from crypto_chatbot.ops.backup import to_libpq_url
from crypto_chatbot.ops.logging import SecretRedactingFilter
from crypto_chatbot.ops.metrics import check_database_health


def test_secret_redacting_filter_redacts_tokens_keys_and_passwords() -> None:
    tg_token = "123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghi"
    gemini_key = "AIzaSyTestSecretKey1234567890abcdef"
    db_pass = "MySecretDbPass123"

    flt = SecretRedactingFilter(secrets=[tg_token, gemini_key, db_pass])

    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=10,
        msg="Connecting to %s using token %s and key %s",
        args=(f"postgresql+psycopg://user:{db_pass}@localhost/db", tg_token, gemini_key),
        exc_info=None,
    )
    assert flt.filter(record) is True
    rendered = record.getMessage()

    assert tg_token not in rendered
    assert gemini_key not in rendered
    assert db_pass not in rendered
    assert "[REDACTED]" in rendered


def test_secret_redacting_filter_redacts_auth_headers_and_patterns() -> None:
    flt = SecretRedactingFilter()
    sample_text = (
        "Authorization: Bearer secret-bearer-token-val\n"
        "X-Telegram-Bot-Api-Secret-Token: secret-telegram-token\n"
        "https://example.com/api?api_key=secret-in-query\n"
        '{"password": "secret-json-password"}'
    )
    redacted = flt.redact_text(sample_text)
    assert "secret-bearer-token-val" not in redacted
    assert "secret-in-query" not in redacted
    assert "secret-json-password" not in redacted
    assert "[REDACTED]" in redacted


def test_to_libpq_url_converts_psycopg_driver_cleanly() -> None:
    sa_url = "postgresql+psycopg://app_user:pass%40word@127.0.0.1:5432/crypto_db"
    libpq = to_libpq_url(sa_url)
    assert libpq.startswith("postgresql://")
    assert "+psycopg" not in libpq
    assert "app_user" in libpq


def test_check_database_health_handles_success_and_failure() -> None:
    mock_engine = MagicMock()
    # Success case
    assert check_database_health(mock_engine) is True

    # Error case
    from sqlalchemy.exc import OperationalError

    mock_engine.connect.side_effect = OperationalError("conn failed", None, Exception())
    assert check_database_health(mock_engine) is False


def test_access_controller_allowlist_and_rate_limiting() -> None:
    now = [100.0]
    controller = AccessController([111, 222], rate_limit_per_minute=2, clock=lambda: now[0])

    assert controller.check_access(None).allowed is False
    assert controller.check_access(None).reason == "not_allowlisted"

    assert controller.check_access(999).allowed is False
    assert controller.check_access(999).reason == "not_allowlisted"

    # User 111 within limit
    assert controller.check_access(111).allowed is True
    assert controller.check_access(111).allowed is True

    # User 111 exceeds limit
    denied = controller.check_access(111)
    assert denied.allowed is False
    assert denied.reason == "rate_limited"

    # User 222 still allowed
    assert controller.check_access(222).allowed is True

    # Window slides after 60s
    now[0] = 161.0
    assert controller.check_access(111).allowed is True


def test_cli_health_check_output(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "12345678:valid_bot_token_value_abc")
    monkeypatch.setenv("GEMINI_API_KEY", "AIzaTestGeminiKey123456")
    monkeypatch.setenv("GEMINI_MODEL", "gemini-2.5-flash")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://user:pass@localhost:5432/db")
    monkeypatch.setenv("ALLOWED_TELEGRAM_USER_IDS", "111")

    # Mock DB health check so we don't need real DB in this unit test
    monkeypatch.setattr("crypto_chatbot.cli.check_database_health", lambda engine: True)

    rc = cli.main(["health-check"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["config_ok"] is True
    assert payload["db_healthy"] is True
