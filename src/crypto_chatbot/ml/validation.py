"""Chronological purged walk-forward CV with embargo and untouched holdout split."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from crypto_chatbot.schemas.ml import FeatureRow


@dataclass(frozen=True, slots=True)
class PurgedSplit:
    """Single purged & embargoed training/validation split."""

    fold_index: int
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]


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
