"""Chronological purged walk-forward CV with embargo and untouched holdout split."""

from __future__ import annotations

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

from crypto_chatbot.schemas.ml import FeatureRow


@dataclass(frozen=True, slots=True)
class PurgedSplit:
    """Single purged & embargoed training/validation split."""

    fold_index: int
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CpcvPath:
    """One combinatorial purged cross-validation path (a set of test groups)."""

    path_index: int
    test_group_indices: tuple[int, ...]
    train_indices: tuple[int, ...]
    test_indices: tuple[int, ...]


def split_dev_and_holdout(
    rows: Sequence[FeatureRow],
    *,
    holdout_fraction: float = 0.20,
    embargo_bars: int = 3,
) -> tuple[tuple[FeatureRow, ...], tuple[FeatureRow, ...]]:
    """Reserve a chronological untouched holdout tail with label purge + embargo."""
    if not rows or len(rows) < 40:
        raise ValueError("at least 40 feature rows are required to split dev and holdout")
    if not (0.0 < holdout_fraction < 0.5):
        raise ValueError("holdout_fraction must lie in (0.0, 0.5)")
    if embargo_bars < 0:
        raise ValueError("embargo_bars must be >= 0")

    holdout_count = max(10, int(len(rows) * holdout_fraction))
    holdout_start = len(rows) - holdout_count
    if holdout_start <= embargo_bars + 15:
        raise ValueError("insufficient development rows before holdout")

    holdout_rows = tuple(rows[holdout_start:])
    holdout_start_time = holdout_rows[0].prediction_time
    purge_idx_limit = holdout_start - embargo_bars
    dev_rows = tuple(
        row for row in rows[:purge_idx_limit] if row.label_end_time <= holdout_start_time
    )
    if len(dev_rows) < 25:
        raise ValueError("insufficient non-overlapping dev rows after holdout purge/embargo")
    return dev_rows, holdout_rows


def generate_purged_walk_forward_splits(
    rows: Sequence[FeatureRow],
    *,
    n_splits: int = 4,
    embargo_bars: int = 3,
) -> tuple[PurgedSplit, ...]:
    """Build expanding-window walk-forward folds enforcing zero temporal/label overlap."""
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    if embargo_bars < 0:
        raise ValueError("embargo_bars must be >= 0")

    n = len(rows)
    min_initial_train = max(20, n // (n_splits + 1))
    remaining = n - min_initial_train
    if remaining < n_splits * 4:
        raise ValueError("insufficient development rows for requested number of walk-forward folds")

    val_size = remaining // n_splits
    splits: list[PurgedSplit] = []

    for fold_idx in range(n_splits):
        val_start = min_initial_train + fold_idx * val_size
        val_end = n if fold_idx == n_splits - 1 else val_start + val_size
        val_start_time = rows[val_start].prediction_time
        train_idx_limit = val_start - embargo_bars
        train_indices = tuple(
            i for i in range(max(0, train_idx_limit)) if rows[i].label_end_time <= val_start_time
        )
        if not train_indices:
            raise ValueError(
                f"fold {fold_idx} has zero eligible training samples after purge and embargo"
            )
        val_indices = tuple(range(val_start, val_end))
        splits.append(
            PurgedSplit(
                fold_index=fold_idx,
                train_indices=train_indices,
                val_indices=val_indices,
            )
        )

    assert_no_fold_leakage(rows, splits)
    return tuple(splits)


def assert_no_fold_leakage(
    rows: Sequence[FeatureRow],
    splits: Sequence[PurgedSplit],
) -> None:
    """Assert strict temporal ordering, purge, and zero overlap across every fold."""
    for row in rows:
        if row.available_time > row.prediction_time:
            raise ValueError("row available_time exceeds prediction_time (causal violation)")
        if row.label_end_time <= row.prediction_time:
            raise ValueError("row label_end_time does not follow prediction_time")

    for split in splits:
        if not split.train_indices or not split.val_indices:
            raise ValueError(f"fold {split.fold_index} must have non-empty train and val sets")
        train_set = set(split.train_indices)
        val_set = set(split.val_indices)
        if train_set & val_set:
            raise ValueError(f"fold {split.fold_index} shares indices across train and val")

        min_val_idx = min(split.val_indices)
        max_train_idx = max(split.train_indices)
        if max_train_idx >= min_val_idx:
            raise ValueError(f"fold {split.fold_index} breaks chronological train < val order")

        val_start_time = rows[min_val_idx].prediction_time
        for tr_idx in split.train_indices:
            tr_row = rows[tr_idx]
            if tr_row.prediction_time >= val_start_time:
                raise ValueError(f"fold {split.fold_index} train prediction_time >= val_start_time")
            if tr_row.label_end_time > val_start_time:
                raise ValueError(
                    f"fold {split.fold_index} train label_end_time overlaps validation start"
                )


def generate_cpcv_paths(
    rows: Sequence[FeatureRow],
    *,
    n_groups: int = 6,
    test_groups_per_path: int = 2,
    embargo_bars: int = 3,
) -> tuple[CpcvPath, ...]:
    """Build Combinatorial Purged CV paths (Lopez de Prado).

    Splits rows into `n_groups` contiguous chronological groups and, for each combination of
    `test_groups_per_path` test groups, trains on the remaining groups. Training rows are
    purged so their label window never overlaps a test group and an `embargo_bars` buffer is
    removed immediately after each test group. Produces `C(n_groups, test_groups_per_path)`
    paths, giving many out-of-sample sequences rather than a single walk-forward path.
    """
    if n_groups < 3:
        raise ValueError("n_groups must be >= 3")
    if not (1 <= test_groups_per_path < n_groups):
        raise ValueError("test_groups_per_path must satisfy 1 <= k < n_groups")
    if embargo_bars < 0:
        raise ValueError("embargo_bars must be >= 0")

    n = len(rows)
    if n < n_groups * 4:
        raise ValueError("insufficient rows for the requested number of CPCV groups")

    assert_no_fold_leakage(rows, ())

    # Contiguous group index boundaries.
    bounds: list[tuple[int, int]] = []
    base = n // n_groups
    remainder = n % n_groups
    start = 0
    for g in range(n_groups):
        size = base + (1 if g < remainder else 0)
        bounds.append((start, start + size))
        start += size

    paths: list[CpcvPath] = []
    combos = list(itertools.combinations(range(n_groups), test_groups_per_path))
    for path_idx, test_groups in enumerate(combos):
        test_indices: list[int] = []
        test_spans: list[tuple[int, int]] = []
        for g in test_groups:
            lo, hi = bounds[g]
            test_indices.extend(range(lo, hi))
            test_spans.append((lo, hi))

        test_set = set(test_indices)
        test_time_spans = [
            (
                min(rows[i].prediction_time for i in range(lo, hi)),
                max(rows[i].label_end_time for i in range(lo, hi)),
            )
            for lo, hi in test_spans
        ]
        embargo_blocked: set[int] = set()
        for _lo, hi in test_spans:
            embargo_blocked.update(range(hi, min(n, hi + embargo_bars)))

        train_indices = tuple(
            i
            for i in range(n)
            if i not in test_set
            and i not in embargo_blocked
            and not any(
                rows[i].prediction_time < span_end and rows[i].label_end_time > span_start
                for span_start, span_end in test_time_spans
            )
        )
        if not train_indices:
            continue
        paths.append(
            CpcvPath(
                path_index=path_idx,
                test_group_indices=tuple(test_groups),
                train_indices=train_indices,
                test_indices=tuple(sorted(test_set)),
            )
        )

    if not paths:
        raise ValueError("no CPCV path produced non-empty purged training data")
    return tuple(paths)
