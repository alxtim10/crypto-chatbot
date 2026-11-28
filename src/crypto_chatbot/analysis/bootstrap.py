"""Stationary block bootstrap (Politis-Romano) & daily portfolio metrics (PRD v1.7 §7.5, T-19)."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from datetime import date, datetime
from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field


class _TradeLike(Protocol):
    @property
    def signal_time(self) -> datetime: ...

    @property
    def exit_time(self) -> datetime | None: ...

    @property
    def net_r(self) -> float: ...


class BlockBootstrapSummary(BaseModel):
    """Stationary block bootstrap (Politis-Romano) 95% CIs on daily portfolio R series."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resamples: int = Field(ge=100)
    mean_block_length: float = Field(gt=0.0)
    seed: int
    n_days: int = Field(ge=1)
    daily_mean_r_ci_low: float
    daily_mean_r_ci_high: float
    trade_expectancy_r_ci_low: float
    trade_expectancy_r_ci_high: float
    positive_daily_mean_rate: float = Field(ge=0.0, le=1.0)
    max_drawdown_r_p95: float = Field(ge=0.0)


class PortfolioSummary(BaseModel):
    """Portfolio-level aggregation across daily UTC buckets and concurrent open trades."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_days: int = Field(ge=0)
    total_net_r: float
    portfolio_max_drawdown_r: float = Field(ge=0.0)
    max_concurrent_positions: int = Field(ge=0)
    block_bootstrap: BlockBootstrapSummary | None = None


def _stationary_bootstrap_indices(
    n: int,
    *,
    resamples: int,
    mean_block_length: float,
    rng: np.random.Generator,
) -> NDArray[np.int64]:
    """Generate `(resamples, n)` index matrix using Politis-Romano stationary block bootstrap."""
    if n < 1:
        raise ValueError("n must be >= 1")
    if mean_block_length <= 0.0:
        raise ValueError("mean_block_length must be > 0")
    p_new = min(1.0, 1.0 / mean_block_length)
    indices = np.empty((resamples, n), dtype=np.int64)
    starts = rng.integers(0, n, size=resamples)
    indices[:, 0] = starts
    u = rng.random(size=(resamples, n - 1))
    jumps = rng.integers(0, n, size=(resamples, n - 1))
    for col in range(1, n):
        start_new = u[:, col - 1] < p_new
        prev_next = (indices[:, col - 1] + 1) % n
        indices[:, col] = np.where(start_new, jumps[:, col - 1], prev_next)
    return indices


def stationary_block_bootstrap_ci(
    values: ArrayLike,
    *,
    resamples: int = 2000,
    mean_block_length: float = 5.0,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Return `(ci_low, ci_high, positive_rate)` via stationary block bootstrap."""
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("values must be a non-empty 1D array")
    rng = np.random.default_rng(seed)
    idx_mat = _stationary_bootstrap_indices(
        len(arr),
        resamples=resamples,
        mean_block_length=mean_block_length,
        rng=rng,
    )
    sampled = arr[idx_mat]
    means = sampled.mean(axis=1)
    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_low, ci_high = np.percentile(means, [lo_pct, hi_pct])
    pos_rate = float(np.mean(means > 0.0))
    return float(ci_low), float(ci_high), min(1.0, max(0.0, pos_rate))


def compute_max_concurrent_positions(trades: Sequence[_TradeLike]) -> int:
    """Sweep-line maximum number of simultaneously open trades."""
    events: list[tuple[datetime, int]] = []
    for t in trades:
        end = t.exit_time or t.signal_time
        events.append((t.signal_time, 1))
        events.append((end, -1))
    # Sort chronologically; close (-1) before open (+1) at identical timestamps
    events.sort(key=lambda item: (item[0], item[1]))
    curr = 0
    peak = 0
    for _ts, delta in events:
        curr += delta
        if curr > peak:
            peak = curr
    return peak


def compute_portfolio_and_block_bootstrap(
    trades: Sequence[_TradeLike],
    *,
    resamples: int = 2000,
    mean_block_length: float = 5.0,
    seed: int = 42,
) -> PortfolioSummary:
    """Bucket trades by UTC exit date and run stationary block bootstrap (Politis-Romano)."""
    if not trades:
        return PortfolioSummary(
            n_days=0,
            total_net_r=0.0,
            portfolio_max_drawdown_r=0.0,
            max_concurrent_positions=0,
            block_bootstrap=None,
        )

    daily_buckets: dict[date, list[float]] = defaultdict(list)
    for t in trades:
        exit_dt = t.exit_time or t.signal_time
        daily_buckets[exit_dt.date()].append(float(t.net_r))

    sorted_days = sorted(daily_buckets.keys())
    daily_rs = np.array([sum(daily_buckets[d]) for d in sorted_days], dtype=np.float64)
    daily_counts = np.array([len(daily_buckets[d]) for d in sorted_days], dtype=np.float64)
    n_days = len(sorted_days)

    equity = np.cumsum(daily_rs)
    peaks = np.maximum(0.0, np.maximum.accumulate(equity))
    port_max_dd = float(np.max(peaks - equity)) if n_days > 0 else 0.0
    max_conc = compute_max_concurrent_positions(trades)
    total_r = float(np.sum(daily_rs))

    bs_summary: BlockBootstrapSummary | None = None
    if n_days >= 5:
        rng = np.random.default_rng(seed)
        idx_mat = _stationary_bootstrap_indices(
            n_days,
            resamples=resamples,
            mean_block_length=mean_block_length,
            rng=rng,
        )
        sampled_daily = daily_rs[idx_mat]
        sampled_counts = daily_counts[idx_mat]

        daily_means = sampled_daily.mean(axis=1)
        d_low, d_high = np.percentile(daily_means, [2.5, 97.5])
        pos_rate = float(np.mean(daily_means > 0.0))

        total_counts = np.maximum(1.0, sampled_counts.sum(axis=1))
        trade_exps = sampled_daily.sum(axis=1) / total_counts
        t_low, t_high = np.percentile(trade_exps, [2.5, 97.5])

        eq_mat = np.cumsum(sampled_daily, axis=1)
        peak_mat = np.maximum(0.0, np.maximum.accumulate(eq_mat, axis=1))
        dds = np.max(peak_mat - eq_mat, axis=1)
        dd_p95 = float(np.percentile(dds, 95.0))

        bs_summary = BlockBootstrapSummary(
            resamples=resamples,
            mean_block_length=mean_block_length,
            seed=seed,
            n_days=n_days,
            daily_mean_r_ci_low=float(d_low),
            daily_mean_r_ci_high=float(d_high),
            trade_expectancy_r_ci_low=float(t_low),
            trade_expectancy_r_ci_high=float(t_high),
            positive_daily_mean_rate=min(1.0, max(0.0, pos_rate)),
            max_drawdown_r_p95=max(0.0, dd_p95),
        )

    return PortfolioSummary(
        n_days=n_days,
        total_net_r=round(total_r, 4),
        portfolio_max_drawdown_r=round(port_max_dd, 4),
        max_concurrent_positions=max_conc,
        block_bootstrap=bs_summary,
    )
