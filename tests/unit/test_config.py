from __future__ import annotations

from pathlib import Path

import pytest

from crypto_chatbot.config import ConfigError, load_settings

ENV_VARS = (
    "TELEGRAM_BOT_TOKEN",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "DATABASE_URL",
    "ALLOWED_TELEGRAM_USER_IDS",
)

VALID_ENV = {
    "TELEGRAM_BOT_TOKEN": "test-telegram-token-value",
    "GEMINI_API_KEY": "test-gemini-key-value",
    "GEMINI_MODEL": "test-model",
    "DATABASE_URL": "postgresql+psycopg://user:test-db-password@localhost/db",
    "ALLOWED_TELEGRAM_USER_IDS": "111, 222",
}


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def _set_env(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    for name, value in {**VALID_ENV, **overrides}.items():
        monkeypatch.setenv(name, value)


def test_loads_valid_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)

    settings = load_settings(env_file=None)

    assert settings.telegram_bot_token.get_secret_value() == VALID_ENV["TELEGRAM_BOT_TOKEN"]
    assert settings.gemini_model == "test-model"
    assert settings.allowed_telegram_user_ids == (111, 222)


def test_missing_secrets_fail_with_clear_field_names() -> None:
    with pytest.raises(ConfigError) as exc_info:
        load_settings(env_file=None)

    message = str(exc_info.value)
    for name in ENV_VARS:
        assert name in message


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("TELEGRAM_BOT_TOKEN", "   "),
        ("GEMINI_API_KEY", ""),
        ("DATABASE_URL", "sqlite:///local.db"),
        ("ALLOWED_TELEGRAM_USER_IDS", ""),
        ("ALLOWED_TELEGRAM_USER_IDS", "111,abc"),
    ],
)
def test_invalid_values_are_rejected(
    monkeypatch: pytest.MonkeyPatch, name: str, value: str
) -> None:
    _set_env(monkeypatch, **{name: value})

    with pytest.raises(ConfigError, match=name):
        load_settings(env_file=None)


def test_secrets_never_appear_in_repr_or_str(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch)

    settings = load_settings(env_file=None)
    rendered = repr(settings) + str(settings) + str(settings.model_dump())

    assert VALID_ENV["TELEGRAM_BOT_TOKEN"] not in rendered
    assert VALID_ENV["GEMINI_API_KEY"] not in rendered
    assert "test-db-password" not in rendered


def test_error_message_does_not_leak_secret_values(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_env(monkeypatch, DATABASE_URL="mysql://user:leaky-password@host/db")

    with pytest.raises(ConfigError) as exc_info:
        load_settings(env_file=None)

    assert "leaky-password" not in str(exc_info.value)


def test_reads_env_file(tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("\n".join(f"{k}={v}" for k, v in VALID_ENV.items()), encoding="utf-8")

    settings = load_settings(env_file=env_file)

    assert settings.allowed_telegram_user_ids == (111, 222)
