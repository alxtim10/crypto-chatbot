"""Trial Registry for backtest and ML experiment governance (PRD v1.7 §8.2, T-04)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Engine, select
from sqlalchemy.orm import Session

from crypto_chatbot.schemas.market import UtcDatetime
from crypto_chatbot.storage.models import TrialRow

DEFAULT_TRIALS_JSONL_PATH = Path(".trials") / "trials.jsonl"


def compute_params_sha256(params: Mapping[str, Any]) -> str:
    """Compute a deterministic SHA-256 digest over canonical JSON variant parameters."""
    encoded = json.dumps(params, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def current_git_sha() -> str | None:
    """Return current short HEAD git commit SHA if available, else None."""
    git_bin = shutil.which("git")
    if git_bin is None:
        return None
    try:
        out = subprocess.check_output(  # noqa: S603
            [git_bin, "rev-parse", "--short=12", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=2.0,
        ).strip()
        return out or None
    except Exception:
        return None


class TrialRecord(BaseModel):
    """Immutable record of one registered historical backtest or ML experiment run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    trial_id: str = Field(min_length=1, max_length=64)
    family: str = Field(min_length=1, max_length=64)
    variant_params_json: dict[str, Any]
    params_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    data_window: str = Field(min_length=1, max_length=128)
    symbols: tuple[str, ...] = Field(min_length=1)
    timeframe: str = Field(min_length=1, max_length=16)
    obs_frequency: str = Field(min_length=1, max_length=32)
    sr_per_obs: float | None = None
    n_obs: int = Field(ge=0)
    created_at: UtcDatetime
    git_sha: str | None = Field(default=None, max_length=64)

    @classmethod
    def create(
        cls,
        *,
        family: str,
        variant_params: Mapping[str, Any],
        data_window: str,
        symbols: Sequence[str],
        timeframe: str,
        obs_frequency: str,
        sr_per_obs: float | None,
        n_obs: int,
        created_at: datetime | None = None,
        git_sha: str | None = None,
    ) -> TrialRecord:
        params_dict = dict(variant_params)
        return cls(
            trial_id=f"trl-{uuid.uuid4().hex[:16]}",
            family=family.strip(),
            variant_params_json=params_dict,
            params_sha256=compute_params_sha256(params_dict),
            data_window=data_window.strip(),
            symbols=tuple(symbols),
            timeframe=timeframe.strip(),
            obs_frequency=obs_frequency.strip(),
            sr_per_obs=sr_per_obs,
            n_obs=n_obs,
            created_at=(created_at or datetime.now(UTC)).astimezone(UTC),
            git_sha=git_sha if git_sha is not None else current_git_sha(),
        )


class TrialRegistry(Protocol):
    """Persistence interface for registered historical trials."""

    def record_trial(self, trial: TrialRecord) -> TrialRecord: ...

    def list_trials(
        self,
        *,
        family: str,
        obs_frequency: str | None = None,
    ) -> tuple[TrialRecord, ...]: ...


class PostgresTrialRegistry:
    """PostgreSQL-backed Trial Registry (`trials` table)."""

    def __init__(self, engine: Engine) -> None:
        self._engine = engine

    def record_trial(self, trial: TrialRecord) -> TrialRecord:
        row = TrialRow(
            trial_id=trial.trial_id,
            family=trial.family,
            variant_params_json=trial.variant_params_json,
            params_sha256=trial.params_sha256,
            data_window=trial.data_window,
            symbols=list(trial.symbols),
            timeframe=trial.timeframe,
            obs_frequency=trial.obs_frequency,
            sr_per_obs=trial.sr_per_obs,
            n_obs=trial.n_obs,
            created_at=trial.created_at,
            git_sha=trial.git_sha,
        )
        with Session(self._engine) as session:
            session.add(row)
            session.commit()
        return trial

    def list_trials(
        self,
        *,
        family: str,
        obs_frequency: str | None = None,
    ) -> tuple[TrialRecord, ...]:
        stmt = select(TrialRow).where(TrialRow.family == family).order_by(TrialRow.created_at.asc())
        if obs_frequency is not None:
            stmt = stmt.where(TrialRow.obs_frequency == obs_frequency)
        with Session(self._engine) as session:
            rows = session.scalars(stmt).all()
        return tuple(
            TrialRecord(
                trial_id=r.trial_id,
                family=r.family,
                variant_params_json=dict(r.variant_params_json),
                params_sha256=r.params_sha256,
                data_window=r.data_window,
                symbols=tuple(r.symbols),
                timeframe=r.timeframe,
                obs_frequency=r.obs_frequency,
                sr_per_obs=r.sr_per_obs,
                n_obs=r.n_obs,
                created_at=r.created_at.astimezone(UTC),
                git_sha=r.git_sha,
            )
            for r in rows
        )


class JsonlTrialRegistry:
    """Append-only local JSONL Trial Registry fallback when PostgreSQL is not configured."""

    def __init__(self, path: Path = DEFAULT_TRIALS_JSONL_PATH) -> None:
        self._path = path

    @property
    def path(self) -> Path:
        return self._path

    def record_trial(self, trial: TrialRecord) -> TrialRecord:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        with self._path.open("a", encoding="utf-8") as fh:
            fh.write(trial.model_dump_json() + "\n")
        return trial

    def list_trials(
        self,
        *,
        family: str,
        obs_frequency: str | None = None,
    ) -> tuple[TrialRecord, ...]:
        if not self._path.exists():
            return ()
        out: list[TrialRecord] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            rec = TrialRecord.model_validate_json(stripped)
            if rec.family != family:
                continue
            if obs_frequency is not None and rec.obs_frequency != obs_frequency:
                continue
            out.append(rec)
        return tuple(out)


def get_family_trial_sharpes(
    registry: TrialRegistry,
    *,
    family: str,
    obs_frequency: str,
) -> tuple[int, tuple[float, ...]]:
    """Return `(total_registered_trials, finite_sharpes)` for a family at one `obs_frequency`."""
    records = registry.list_trials(family=family, obs_frequency=obs_frequency)
    sharpes = tuple(r.sr_per_obs for r in records if r.sr_per_obs is not None)
    return len(records), sharpes
