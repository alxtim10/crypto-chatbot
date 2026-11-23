"""Probabilistic calibration, post-cost strategy metrics, and strict model rejection."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
from numpy.typing import ArrayLike, NDArray

from crypto_chatbot.schemas.ml import (
    CalibrationBinMetrics,
    CostAssumptions,
    EvaluationSummary,
    FeatureRow,
    FoldMetrics,
    MlExperimentConfig,
    RegimeSliceMetrics,
)


def compute_brier_skill_score(
    brier: float,
    naive_brier: float,
) -> float:
    """Compute Brier Skill Score: 1 - brier / naive_brier.

    Returns 0.0 when naive_brier is zero and brier is zero.
    Returns -1.0 when naive_brier is zero but brier > 0.
    """
    if naive_brier <= 1e-9:
        return 0.0 if brier <= 1e-9 else -1.0
    return float(1.0 - (brier / naive_brier))


def compute_calibration_bins(
    y_true: NDArray[np.float64],
    p_pred: NDArray[np.float64],
    *,
    net_returns: NDArray[np.float64] | None = None,
    n_bins: int = 5,
) -> tuple[CalibrationBinMetrics, ...]:
    """Compute discrete calibration bin metrics across the probability interval [0, 1]."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[CalibrationBinMetrics] = []
    for b_idx in range(n_bins):
        lo, hi = float(edges[b_idx]), float(edges[b_idx + 1])
        if b_idx == n_bins - 1:
            mask = (p_pred >= lo) & (p_pred <= hi)
        else:
            mask = (p_pred >= lo) & (p_pred < hi)
        count = int(np.sum(mask))
        if count == 0:
            bins.append(
                CalibrationBinMetrics(
                    bin_index=b_idx,
                    bin_low=lo,
                    bin_high=hi,
                    sample_count=0,
                    mean_predicted_prob=None,
                    empirical_positive_rate=None,
                    calibration_gap=None,
                    mean_net_return=None,
                )
            )
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(p_pred[mask]))
        gap = abs(acc - conf)
        mean_ret = float(np.mean(net_returns[mask])) if net_returns is not None else None
        bins.append(
            CalibrationBinMetrics(
                bin_index=b_idx,
                bin_low=lo,
                bin_high=hi,
                sample_count=count,
                mean_predicted_prob=conf,
                empirical_positive_rate=acc,
                calibration_gap=gap,
                mean_net_return=mean_ret,
            )
        )
    return tuple(bins)


def compute_ece(
    y_true: NDArray[np.float64],
    p_pred: NDArray[np.float64],
    *,
    n_bins: int = 5,
) -> float:
    """Expected Calibration Error (ECE) over equal-width probability bins."""
    n = len(y_true)
    if n == 0:
        return 0.0
    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for b_idx in range(n_bins):
        lo, hi = float(edges[b_idx]), float(edges[b_idx + 1])
        if b_idx == n_bins - 1:
            mask = (p_pred >= lo) & (p_pred <= hi)
        else:
            mask = (p_pred >= lo) & (p_pred < hi)
        count = int(np.sum(mask))
        if count == 0:
            continue
        acc = float(np.mean(y_true[mask]))
        conf = float(np.mean(p_pred[mask]))
        ece += (count / n) * abs(acc - conf)
    return float(min(1.0, max(0.0, ece)))


def compute_evaluation_summary(
    y_true_in: ArrayLike,
    p_model_in: ArrayLike,
    p_naive_in: ArrayLike,
    forward_returns_in: ArrayLike,
    *,
    cost_assumptions: CostAssumptions,
    prob_threshold: float,
    prediction_times: Sequence[datetime] | None = None,
    label_end_times: Sequence[datetime] | None = None,
) -> EvaluationSummary:
    """Compute Brier score, ECE, balanced accuracy, precision/recall, and post-cost returns."""
    y_true = np.asarray(y_true_in, dtype=np.float64)
    p_model = np.asarray(p_model_in, dtype=np.float64)
    p_naive = np.asarray(p_naive_in, dtype=np.float64)
    fwd_ret = np.asarray(forward_returns_in, dtype=np.float64)
    n = len(y_true)
    if not (n > 0 and len(p_model) == n and len(p_naive) == n and len(fwd_ret) == n):
        raise ValueError("evaluation arrays must be non-empty and share identical lengths")

    naive_brier = float(np.mean((p_naive - y_true) ** 2))
    model_brier = float(np.mean((p_model - y_true) ** 2))
    skill = 1.0 - (model_brier / naive_brier) if naive_brier > 0.0 else 0.0
    bss = compute_brier_skill_score(model_brier, naive_brier)
    ece = compute_ece(y_true, p_model, n_bins=5)
    cal_bins = compute_calibration_bins(y_true, p_model, net_returns=fwd_ret, n_bins=5)

    y_pred = p_model >= prob_threshold
    y_pos = y_true >= 0.5

    tp = int(np.sum(y_pred & y_pos))
    fp = int(np.sum(y_pred & ~y_pos))
    tn = int(np.sum(~y_pred & ~y_pos))
    fn = int(np.sum(~y_pred & y_pos))

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    bal_acc = 0.5 * (tpr + tnr)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tpr

    round_trip_cost = cost_assumptions.round_trip_cost_rate
    maker_cost_rate = cost_assumptions.model_copy(
        update={"execution_style": "maker"}
    ).round_trip_cost_rate
    taker_cost_rate = cost_assumptions.model_copy(
        update={"execution_style": "taker"}
    ).round_trip_cost_rate

    equity = 1.0
    equity_maker = 1.0
    equity_taker = 1.0
    peak = 1.0
    max_dd = 0.0
    trade_returns: list[float] = []
    next_eligible_time: datetime | None = None

    for idx in range(n):
        if (
            prediction_times is not None
            and label_end_times is not None
            and next_eligible_time is not None
            and prediction_times[idx] < next_eligible_time
        ):
            continue
        if bool(y_pred[idx]):
            net_r = max(-0.99, float(fwd_ret[idx]) - round_trip_cost)
            net_r_m = max(-0.99, float(fwd_ret[idx]) - maker_cost_rate)
            net_r_t = max(-0.99, float(fwd_ret[idx]) - taker_cost_rate)
            trade_returns.append(net_r)
            equity *= 1.0 + net_r
            equity_maker *= 1.0 + net_r_m
            equity_taker *= 1.0 + net_r_t
            if equity > peak:
                peak = equity
            if peak > 0.0:
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
            if prediction_times is not None and label_end_times is not None:
                next_eligible_time = label_end_times[idx]

    trade_count = len(trade_returns)
    net_ret = equity - 1.0
    mean_ret = float(np.mean(trade_returns)) if trade_returns else 0.0

    return EvaluationSummary(
        sample_count=n,
        positive_rate=float(np.mean(y_true)),
        naive_brier=naive_brier,
        model_brier=model_brier,
        brier_skill_vs_naive=skill,
        brier_skill_score=bss,
        calibration_ece=ece,
        balanced_accuracy=bal_acc,
        precision=precision,
        recall=recall,
        trade_count=trade_count,
        net_return_after_costs=net_ret,
        mean_return_per_trade_after_costs=mean_ret,
        max_drawdown=min(1.0, max(0.0, max_dd)),
        calibration_bins=cal_bins,
        maker_net_return_after_costs=equity_maker - 1.0,
        taker_net_return_after_costs=equity_taker - 1.0,
    )


def compute_regime_stability(
    rows: Sequence[FeatureRow],
    p_model: NDArray[np.float64],
    p_naive: NDArray[np.float64],
    *,
    cost_assumptions: CostAssumptions,
    prob_threshold: float,
) -> tuple[RegimeSliceMetrics, ...]:
    """Slice OOS evaluation across recorded market regimes to expose regime fragility."""
    by_regime: dict[str, list[int]] = {}
    for idx, row in enumerate(rows):
        key = row.regime.value if row.regime is not None else "unknown"
        by_regime.setdefault(key, []).append(idx)

    slices: list[RegimeSliceMetrics] = []
    for regime_name, indices in sorted(by_regime.items()):
        y_slice = np.array([rows[i].label for i in indices], dtype=np.float64)
        pm_slice = p_model[indices]
        pn_slice = p_naive[indices]
        fwd_slice = np.array([rows[i].forward_return for i in indices], dtype=np.float64)
        summary = compute_evaluation_summary(
            y_slice,
            pm_slice,
            pn_slice,
            fwd_slice,
            cost_assumptions=cost_assumptions,
            prob_threshold=prob_threshold,
            prediction_times=[rows[i].prediction_time for i in indices],
            label_end_times=[rows[i].label_end_time for i in indices],
        )
        slices.append(
            RegimeSliceMetrics(
                regime=regime_name,
                sample_count=len(indices),
                naive_brier=summary.naive_brier,
                model_brier=summary.model_brier,
                net_return_after_costs=summary.net_return_after_costs,
                trade_count=summary.trade_count,
            )
        )
    return tuple(slices)


def decide_promotion(
    oof_summary: EvaluationSummary,
    holdout_summary: EvaluationSummary,
    fold_metrics: Sequence[FoldMetrics],
    *,
    config: MlExperimentConfig,
) -> tuple[bool, tuple[str, ...]]:
    """Reject unless OOS & untouched holdout improve consistently after explicit trading costs."""
    reasons: list[str] = []

    oof_brier_gain = oof_summary.naive_brier - oof_summary.model_brier
    if oof_brier_gain < config.min_brier_improvement:
        reasons.append(
            f"OOS Brier improvement ({oof_brier_gain:.4f}) below threshold "
            f"({config.min_brier_improvement:.4f})."
        )

    if fold_metrics:
        winning_folds = sum(
            1 for f in fold_metrics if f.brier_improvement > 0.0 and f.net_return_after_costs > 0.0
        )
        fold_win_rate = winning_folds / len(fold_metrics)
        if fold_win_rate < config.min_fold_win_rate:
            reasons.append(
                f"Walk-forward fold consistency ({fold_win_rate:.2f}) below minimum "
                f"({config.min_fold_win_rate:.2f})."
            )

    if oof_summary.trade_count < config.min_trades:
        reasons.append(
            f"OOS trade count ({oof_summary.trade_count}) below minimum ({config.min_trades})."
        )
    if oof_summary.net_return_after_costs <= 0.0:
        reasons.append(
            "OOS net return after costs is non-positive "
            f"({oof_summary.net_return_after_costs:.4f})."
        )

    holdout_gain = holdout_summary.naive_brier - holdout_summary.model_brier
    if holdout_gain <= 0.0:
        reasons.append(
            f"Untouched holdout Brier fails to beat naive baseline ({holdout_gain:.4f})."
        )
    if holdout_summary.trade_count < 1 or holdout_summary.net_return_after_costs <= 0.0:
        reasons.append(
            "Untouched holdout net return after costs is non-positive or executed zero trades."
        )

    return (not reasons), tuple(reasons)
