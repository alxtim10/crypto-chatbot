"""Parameter freeze and per-timeframe forward-proof evaluator (PRD v1.7 §7.6, §8.2, §8.3, T-22)."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from crypto_chatbot.analysis.bootstrap import stationary_block_bootstrap_ci
from crypto_chatbot.schemas.market import Timeframe, UtcDatetime

DEFAULT_VARIANT_PARAMS: dict[str, dict[str, Any]] = {
    "v0": {
        "variant": "v0",
        "ema200_gate": False,
        "max_exec_cost_r": None,
        "min_net_rr_tp2": 1.5,
        "entry_expiry_bars": 6,
        "max_hold_bars": 24,
        "warmup_bars": 600,
    },
    "v3": {
        "variant": "v3",
        "ema200_gate": True,
        "max_exec_cost_r": 0.15,
        "min_net_rr_tp2": 1.5,
        "entry_expiry_bars": 6,
        "max_hold_bars": 24,
        "warmup_bars": 600,
    },
}

# (min_settled_filled_setups, min_duration_days) per PRD v1.7 §8.3
FORWARD_PROOF_THRESHOLDS: dict[Timeframe, tuple[int, int]] = {
    Timeframe.H4: (100, 60),
    Timeframe.D1: (40, 180),
    Timeframe.M15: (200, 30),
    Timeframe.H1: (100, 60),
}


MTF_ENGINE_VERSION = "mtf-15m-v2"


def compute_params_sha256(params: dict[str, Any]) -> str:
    """Canonical SHA-256 digest of a parameter dictionary."""
    encoded = json.dumps(params, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def compute_runtime_params_bundle(
    *,
    variant: str = "v3",
    params_override: dict[str, Any] | None = None,
    screener_config: Any | None = None,
    universe: Sequence[str] | None = None,
    mtf_engine_version: str = MTF_ENGINE_VERSION,
) -> tuple[str, dict[str, Any]]:
    """Single shared SHA-256 hash function used by BOTH `freeze-params` and `alerts run`.

    Combines:
    - setup variant parameters (`setup_params_sha256`)
    - MTF alert engine version (`mtf_engine_version`)
    - normalized `screener.yaml` config (`screener_config_sha256`)
    - normalized `universe.txt` symbol list (`universe_sha256`)
    """
    key = variant.strip().lower()
    if params_override is not None:
        setup_params = dict(params_override)
    elif key in DEFAULT_VARIANT_PARAMS:
        setup_params = dict(DEFAULT_VARIANT_PARAMS[key])
    else:
        raise ValueError(
            f"Unknown variant {variant!r}; expected one of {sorted(DEFAULT_VARIANT_PARAMS)}"
        )

    setup_sha = compute_params_sha256(setup_params)
    screener_sha: str | None = None
    if screener_config is not None:
        cfg_dict = (
            screener_config.model_dump(mode="json")
            if hasattr(screener_config, "model_dump")
            else dict(screener_config)
        )
        screener_sha = compute_params_sha256(cfg_dict)

    universe_sha: str | None = None
    if universe is not None:
        universe_sha = compute_params_sha256({"universe": list(universe)})

    canonical_payload: dict[str, Any] = {
        "setup_params_sha256": setup_sha,
        "mtf_engine_version": mtf_engine_version,
        "screener_config_sha256": screener_sha,
        "universe_sha256": universe_sha,
    }
    digest = compute_params_sha256(canonical_payload)
    components: dict[str, Any] = {
        "setup_params": setup_params,
        "setup_params_sha256": setup_sha,
        "mtf_engine_version": mtf_engine_version,
        "screener_config_sha256": screener_sha,
        "universe_sha256": universe_sha,
    }
    return digest, components


def _detect_git_sha() -> str | None:
    git_bin = shutil.which("git")
    if git_bin is None:
        return None
    try:
        proc = subprocess.run(  # noqa: S603
            [git_bin, "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=3,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = proc.stdout.strip()
    return sha if proc.returncode == 0 and sha else None


class FrozenParamsArtifact(BaseModel):
    """Frozen strategy parameter snapshot written by `freeze-params` (PRD v1.7 §8.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    variant: str = Field(min_length=1)
    params: dict[str, Any]
    params_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    setup_params_sha256: str | None = None
    mtf_engine_version: str | None = None
    screener_config_sha256: str | None = None
    universe_sha256: str | None = None
    frozen_at: UtcDatetime
    git_sha: str | None = None


def freeze_variant_params(
    variant: str,
    *,
    output_path: Path | None = None,
    params_override: dict[str, Any] | None = None,
    screener_config: Any | None = None,
    universe: Sequence[str] | None = None,
    screener_config_path: Path | str | None = None,
    universe_path: Path | str | None = None,
    frozen_at: datetime | None = None,
) -> FrozenParamsArtifact:
    """Freeze a strategy variant + MTF config/universe using the shared `params_sha256` hash."""
    from crypto_chatbot.alerts.config import load_screener_config, load_universe

    key = variant.strip().lower()
    resolved_cfg = (
        screener_config
        if screener_config is not None
        else load_screener_config(screener_config_path)
    )
    resolved_universe = (
        tuple(universe)
        if universe is not None
        else load_universe(universe_path or resolved_cfg.universe_path)
    )

    digest, components = compute_runtime_params_bundle(
        variant=key,
        params_override=params_override,
        screener_config=resolved_cfg,
        universe=resolved_universe,
    )
    ts = (frozen_at or datetime.now(UTC)).astimezone(UTC)
    artifact = FrozenParamsArtifact(
        variant=key,
        params=components["setup_params"],
        params_sha256=digest,
        setup_params_sha256=components["setup_params_sha256"],
        mtf_engine_version=components["mtf_engine_version"],
        screener_config_sha256=components["screener_config_sha256"],
        universe_sha256=components["universe_sha256"],
        frozen_at=ts,
        git_sha=_detect_git_sha(),
    )
    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(artifact.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return artifact


def load_frozen_params(path: Path) -> FrozenParamsArtifact:
    """Load and validate a `frozen_params.json` artifact."""
    return FrozenParamsArtifact.model_validate_json(path.read_text(encoding="utf-8"))


class ForwardLedgerEntry(BaseModel):
    """Settled or pending forward setup record for out-of-sample evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    setup_id: str = Field(min_length=1)
    timeframe: Timeframe
    emitted_at: UtcDatetime
    settled_at: UtcDatetime | None = None
    filled: bool = True
    net_r_1x: float | None = None
    exec_cost_r: float = Field(default=0.0, ge=0.0)
    params_sha256: str | None = None
    forward_only_guards_fired: tuple[str, ...] = ()


class ForwardProofEvaluation(BaseModel):
    """Per-timeframe forward evaluation summary against PRD v1.7 §8.3 criteria."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    timeframe: Timeframe
    status: Literal["MET", "NOT_MET"]
    reasons: tuple[str, ...] = ()
    frozen_params_sha256: str
    frozen_at: UtcDatetime
    total_entries: int = Field(ge=0)
    excluded_hash_mismatch: int = Field(ge=0)
    excluded_before_freeze: int = Field(ge=0)
    eligible_settled_filled: int = Field(ge=0)
    duration_days: float = Field(ge=0.0)
    expectancy_r_1x: float | None = None
    expectancy_r_15x: float | None = None
    block_ci_low: float | None = None
    block_ci_high: float | None = None
    with_forward_only_expectancy_r: float | None = None
    without_forward_only_expectancy_r: float | None = None


def evaluate_forward_proof(
    entries: Sequence[ForwardLedgerEntry],
    *,
    frozen: FrozenParamsArtifact,
    timeframe: Timeframe,
    bootstrap_resamples: int = 2000,
    bootstrap_mean_block_length: float = 5.0,
    bootstrap_seed: int = 42,
) -> ForwardProofEvaluation:
    """Evaluate settled forward setups strictly matching `frozen.params_sha256` and `frozen_at`."""
    min_setups, min_days = FORWARD_PROOF_THRESHOLDS.get(timeframe, (100, 60))
    tf_entries = [e for e in entries if e.timeframe is timeframe]

    excluded_hash = 0
    excluded_pre_freeze = 0
    eligible: list[ForwardLedgerEntry] = []

    for entry in tf_entries:
        if entry.params_sha256 != frozen.params_sha256:
            excluded_hash += 1
            continue
        if entry.emitted_at.astimezone(UTC) < frozen.frozen_at.astimezone(UTC):
            excluded_pre_freeze += 1
            continue
        if entry.filled and entry.settled_at is not None and entry.net_r_1x is not None:
            eligible.append(entry)

    n_elig = len(eligible)
    if eligible:
        latest_settled = max(e.settled_at.astimezone(UTC) for e in eligible if e.settled_at)
        duration_days = max(
            0.0, (latest_settled - frozen.frozen_at.astimezone(UTC)).total_seconds() / 86400.0
        )
        r_1x = [float(e.net_r_1x) for e in eligible if e.net_r_1x is not None]
        r_15x = [
            float(e.net_r_1x) - 0.5 * float(e.exec_cost_r)
            for e in eligible
            if e.net_r_1x is not None
        ]
        exp_1x = float(np.mean(r_1x))
        exp_15x = float(np.mean(r_15x))

        # Split performance with vs without FORWARD_ONLY guards (PRD §7.6)
        with_fo = [
            float(e.net_r_1x)
            for e in eligible
            if e.net_r_1x is not None and len(e.forward_only_guards_fired) > 0
        ]
        without_fo = [
            float(e.net_r_1x)
            for e in eligible
            if e.net_r_1x is not None and len(e.forward_only_guards_fired) == 0
        ]
        with_fo_exp = float(np.mean(with_fo)) if with_fo else None
        without_fo_exp = float(np.mean(without_fo)) if without_fo else None

        # Bucket by UTC exit day for stationary block bootstrap
        by_day: dict[date, float] = defaultdict(float)
        for e in eligible:
            if e.settled_at is not None and e.net_r_1x is not None:
                by_day[e.settled_at.astimezone(UTC).date()] += float(e.net_r_1x)
        daily_vals = [by_day[d] for d in sorted(by_day)]
        series_for_ci = daily_vals if len(daily_vals) >= 5 else r_1x
        ci_low, ci_high, _ = stationary_block_bootstrap_ci(
            series_for_ci,
            resamples=bootstrap_resamples,
            mean_block_length=bootstrap_mean_block_length,
            seed=bootstrap_seed,
        )
    else:
        duration_days = 0.0
        exp_1x = None
        exp_15x = None
        ci_low = None
        ci_high = None
        with_fo_exp = None
        without_fo_exp = None

    reasons: list[str] = []
    if n_elig < min_setups:
        reasons.append(
            f"Settled filled setups ({n_elig}) below minimum ({min_setups}) for {timeframe.value}."
        )
    if duration_days < float(min_days):
        reasons.append(
            f"Forward duration ({duration_days:.1f}d) below minimum ({min_days}d) "
            f"for {timeframe.value}."
        )
    if ci_low is None or ci_low <= 0.0:
        reasons.append(
            f"Block-bootstrap 95% CI lower bound ({ci_low}) is not strictly positive at 1.0x costs."
        )
    if exp_15x is None or exp_15x < 0.0:
        reasons.append(f"Expectancy at 1.5x costs ({exp_15x}) is negative.")

    status: Literal["MET", "NOT_MET"] = "MET" if not reasons else "NOT_MET"
    return ForwardProofEvaluation(
        timeframe=timeframe,
        status=status,
        reasons=tuple(reasons),
        frozen_params_sha256=frozen.params_sha256,
        frozen_at=frozen.frozen_at,
        total_entries=len(tf_entries),
        excluded_hash_mismatch=excluded_hash,
        excluded_before_freeze=excluded_pre_freeze,
        eligible_settled_filled=n_elig,
        duration_days=round(duration_days, 2),
        expectancy_r_1x=round(exp_1x, 4) if exp_1x is not None else None,
        expectancy_r_15x=round(exp_15x, 4) if exp_15x is not None else None,
        block_ci_low=round(ci_low, 4) if ci_low is not None else None,
        block_ci_high=round(ci_high, 4) if ci_high is not None else None,
        with_forward_only_expectancy_r=(round(with_fo_exp, 4) if with_fo_exp is not None else None),
        without_forward_only_expectancy_r=(
            round(without_fo_exp, 4) if without_fo_exp is not None else None
        ),
    )
