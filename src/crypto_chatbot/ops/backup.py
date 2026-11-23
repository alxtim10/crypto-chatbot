"""Parameterized PostgreSQL backup and restore drill helpers (`pg_dump` / `pg_restore`)."""

from __future__ import annotations

import re
import shutil
import subprocess
import time
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

_URL_CREDS_RE = re.compile(r"(postgresql[^\s]*://[^:\s]+:)([^@\s]+)(@)")


class BackupError(RuntimeError):
    """Raised when `pg_dump` or `pg_restore` fails; never contains DB credentials."""


class BackupResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    output_path: Path
    size_bytes: int = Field(ge=0)
    duration_ms: int = Field(ge=0)


class RestoreResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    backup_path: Path
    duration_ms: int = Field(ge=0)


def _scrub_url(text: str, database_url: str) -> str:
    cleaned = text.replace(database_url, "[REDACTED_DATABASE_URL]")
    libpq = to_libpq_url(database_url)
    cleaned = cleaned.replace(libpq, "[REDACTED_DATABASE_URL]")
    return _URL_CREDS_RE.sub(r"\1[REDACTED]\3", cleaned)


def to_libpq_url(database_url: str) -> str:
    """Convert SQLAlchemy PostgreSQL URLs to libpq URLs accepted by `pg_dump`/`pg_restore`."""
    stripped = database_url.strip()
    if not stripped.startswith("postgresql"):
        raise BackupError("database_url must start with 'postgresql'")
    if stripped.startswith("postgresql+psycopg2://"):
        return "postgresql://" + stripped.removeprefix("postgresql+psycopg2://")
    if stripped.startswith("postgresql+psycopg://"):
        return "postgresql://" + stripped.removeprefix("postgresql+psycopg://")
    return stripped


def _require_tool(binary_name: str) -> str:
    resolved = shutil.which(binary_name)
    if resolved is None:
        raise BackupError(f"Required PostgreSQL tool '{binary_name}' was not found on PATH")
    return resolved


def dump_postgres_database(
    database_url: str,
    output_path: Path,
    *,
    timeout_seconds: float = 120.0,
) -> BackupResult:
    """Create a custom-format PostgreSQL backup using `pg_dump` with `shell=False`."""
    libpq_url = to_libpq_url(database_url)
    pg_dump_bin = _require_tool("pg_dump")
    resolved_output = Path(output_path)
    resolved_output.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        pg_dump_bin,
        "--format=custom",
        "--no-owner",
        "--no-privileges",
        "--file",
        str(resolved_output),
        "--dbname",
        libpq_url,
    ]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BackupError("pg_dump timed out") from exc
    elapsed_ms = max(0, int((time.monotonic() - t0) * 1000))

    if proc.returncode != 0 or not resolved_output.is_file():
        stderr = _scrub_url((proc.stderr or "").strip(), database_url)
        raise BackupError(f"pg_dump failed (exit {proc.returncode}): {stderr}")

    return BackupResult(
        output_path=resolved_output,
        size_bytes=resolved_output.stat().st_size,
        duration_ms=elapsed_ms,
    )


def restore_postgres_database(
    database_url: str,
    backup_path: Path,
    *,
    timeout_seconds: float = 120.0,
) -> RestoreResult:
    """Restore a custom-format PostgreSQL dump using `pg_restore` with `shell=False`."""
    resolved_backup = Path(backup_path)
    if not resolved_backup.is_file():
        raise BackupError(f"Backup file not found: {resolved_backup}")
    libpq_url = to_libpq_url(database_url)
    pg_restore_bin = _require_tool("pg_restore")

    cmd = [
        pg_restore_bin,
        "--clean",
        "--if-exists",
        "--no-owner",
        "--no-privileges",
        "--exit-on-error",
        "--dbname",
        libpq_url,
        str(resolved_backup),
    ]
    t0 = time.monotonic()
    try:
        proc = subprocess.run(  # noqa: S603
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise BackupError("pg_restore timed out") from exc
    elapsed_ms = max(0, int((time.monotonic() - t0) * 1000))

    if proc.returncode != 0:
        stderr = _scrub_url((proc.stderr or "").strip(), database_url)
        raise BackupError(f"pg_restore failed (exit {proc.returncode}): {stderr}")

    return RestoreResult(backup_path=resolved_backup, duration_ms=elapsed_ms)
