from __future__ import annotations

from pathlib import Path

import pytest

from crypto_chatbot import cli
from crypto_chatbot.schemas.analysis import AnalysisSnapshot
from crypto_chatbot.schemas.market import QualityStatus

FIXTURE_1000 = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"
FIXTURE_SAMPLE = Path(__file__).resolve().parents[1] / "fixtures" / "binance_btc_4h_sample.json"
DERIV_FIXTURE = (
    Path(__file__).resolve().parents[1] / "fixtures" / "binance_btc_derivatives_sample.json"
)
SECRET_TOKEN = "cli-test-telegram-token"


@pytest.fixture
def isolated_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Run in an empty directory so a developer's real `.env` is never read."""
    monkeypatch.chdir(tmp_path)
    for name in (
        "TELEGRAM_BOT_TOKEN",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "DATABASE_URL",
        "ALLOWED_TELEGRAM_USER_IDS",
    ):
        monkeypatch.delenv(name, raising=False)
    return tmp_path


def test_check_config_fails_when_secrets_missing(
    isolated_cwd: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["check-config"]) == 1
    assert "TELEGRAM_BOT_TOKEN" in capsys.readouterr().err


def test_check_config_succeeds_without_printing_secrets(
    isolated_cwd: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", SECRET_TOKEN)
    monkeypatch.setenv("GEMINI_API_KEY", "cli-test-gemini-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:cli-test-pw@localhost/db")
    monkeypatch.setenv("ALLOWED_TELEGRAM_USER_IDS", "111")

    assert cli.main(["check-config"]) == 0

    captured = capsys.readouterr()
    output = captured.out + captured.err
    assert "Configuration OK" in output
    for secret in (SECRET_TOKEN, "cli-test-gemini-key", "cli-test-pw"):
        assert secret not in output


def test_cli_analyze_fixture_ready_without_secrets(
    isolated_cwd: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["analyze", "BTC", "4h", "--fixture", str(FIXTURE_1000)])
    assert rc == 0

    out = capsys.readouterr().out
    assert "BTC/USDT (4h)" in out
    assert "BINANCE spot" in out
    assert "Status kualitas: READY" in out
    assert "UTC (" in out
    assert "WIB)" in out
    assert "[F1]" in out
    assert "Disclaimer:" in out


def test_cli_analyze_fixture_json_is_deterministic(
    isolated_cwd: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["analyze", "BTC", "4h", "--fixture", str(FIXTURE_1000), "--json"]) == 0
    first_json = capsys.readouterr().out

    assert cli.main(["analyze", "BTC", "4h", "--fixture", str(FIXTURE_1000), "--json"]) == 0
    second_json = capsys.readouterr().out

    assert first_json == second_json
    snap = AnalysisSnapshot.model_validate_json(first_json)
    assert snap.quality_status is QualityStatus.READY


def test_cli_analyze_short_fixture_returns_insufficient_data(
    isolated_cwd: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = cli.main(["analyze", "BTC", "4h", "--fixture", str(FIXTURE_SAMPLE)])
    assert rc == 0
    assert "INSUFFICIENT_DATA" in capsys.readouterr().out


@pytest.mark.parametrize(
    "argv",
    [
        ["analyze", "DOGE", "4h", "--fixture", str(FIXTURE_1000)],
        ["analyze", "BTC", "15m", "--fixture", str(FIXTURE_1000)],
        ["analyze", "BTC", "4h", "--fixture", "does-not-exist.json"],
    ],
)
def test_cli_analyze_rejects_invalid_arguments(
    isolated_cwd: Path,
    capsys: pytest.CaptureFixture[str],
    argv: list[str],
) -> None:
    assert cli.main(argv) == 1
    assert "Error:" in capsys.readouterr().err


def test_cli_analyze_with_derivatives_fixture(
    isolated_cwd: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = cli.main(
        [
            "analyze",
            "BTC",
            "4h",
            "--fixture",
            str(FIXTURE_1000),
            "--derivatives-fixture",
            str(DERIV_FIXTURE),
            "--json",
        ]
    )
    assert rc == 0
    snap = AnalysisSnapshot.model_validate_json(capsys.readouterr().out)
    assert snap.derivatives is not None
    assert snap.derivatives.funding_symbol == "BTC/USDT:USDT"
    assert snap.derivatives.orderbook_symbol == "BTC/USDT"
    assert snap.derivatives.liquidation_source_available is True
    assert snap.derivatives.liquidation_count == 2
