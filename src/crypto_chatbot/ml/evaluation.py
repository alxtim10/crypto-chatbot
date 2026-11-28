"""Probabilistic calibration, post-cost strategy metrics, and strict model rejection."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

import numpy as np
from numpy.typing import ArrayLike, NDArray

from crypto_chatbot.analysis.bootstrap import stationary_block_bootstrap_ci
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
    sample_weight: NDArray[np.float64] | None = None,
) -> float:
    """Expected Calibration Error (ECE) over equal-width probability bins (optionally weighted)."""
    n = len(y_true)
    if n == 0:
        return 0.0
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=np.float64)
        total_w = float(np.sum(weights))
        if total_w <= 0.0:
            weights = np.ones(n, dtype=np.float64)
            total_w = float(n)
    else:
        weights = np.ones(n, dtype=np.float64)
        total_w = float(n)

    ece = 0.0
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    for b_idx in range(n_bins):
        lo, hi = float(edges[b_idx]), float(edges[b_idx + 1])
        if b_idx == n_bins - 1:
            mask = (p_pred >= lo) & (p_pred <= hi)
        else:
            mask = (p_pred >= lo) & (p_pred < hi)
        if not np.any(mask):
            continue
        bin_w = weights[mask]
        bin_w_sum = float(np.sum(bin_w))
        if bin_w_sum <= 0.0:
            continue
        acc = float(np.sum(bin_w * y_true[mask]) / bin_w_sum)
        conf = float(np.sum(bin_w * p_pred[mask]) / bin_w_sum)
        ece += (bin_w_sum / total_w) * abs(acc - conf)
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
    sample_weight: ArrayLike | None = None,
    base_rate: float | None = None,
    base_rate_delta: float | None = None,
) -> EvaluationSummary:
    """Compute Brier score, ECE, balanced accuracy, precision/recall, and post-cost returns."""
    y_true = np.asarray(y_true_in, dtype=np.float64)
    p_model = np.asarray(p_model_in, dtype=np.float64)
    p_naive = np.asarray(p_naive_in, dtype=np.float64)
    fwd_ret = np.asarray(forward_returns_in, dtype=np.float64)
    n = len(y_true)
    if not (n > 0 and len(p_model) == n and len(p_naive) == n and len(fwd_ret) == n):
        raise ValueError("evaluation arrays must be non-empty and share identical lengths")

    if sample_weight is not None:
        sw = np.asarray(sample_weight, dtype=np.float64)
        if len(sw) != n or float(np.sum(sw)) <= 0.0 or np.any(sw < 0.0):
            raise ValueError("sample_weight must match length and have positive sum")
        norm_w = sw / float(np.sum(sw))
    else:
        sw = None
        norm_w = np.full(n, 1.0 / float(n), dtype=np.float64)

    round_trip_cost = cost_assumptions.round_trip_cost_rate
    naive_brier = float(np.sum(norm_w * (p_naive - y_true) ** 2))
    model_brier = float(np.sum(norm_w * (p_model - y_true) ** 2))
    skill = 1.0 - (model_brier / naive_brier) if naive_brier > 0.0 else 0.0
    bss = compute_brier_skill_score(model_brier, naive_brier)
    ece = compute_ece(y_true, p_model, n_bins=5, sample_weight=sw)
    cal_bins = compute_calibration_bins(
        y_true, p_model, net_returns=fwd_ret - round_trip_cost, n_bins=5
    )

    effective_threshold = (
        base_rate + base_rate_delta
        if base_rate is not None and base_rate_delta is not None
        else prob_threshold
    )
    y_pred = p_model >= effective_threshold
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


@dataclass(frozen=True)
class PromotionGateResult:
    """Single promotion gate evaluation outcome (PRD v1.7 §6.6)."""

    gate_id: str
    passed: bool
    value: float | None
    threshold: str
    detail: str


@dataclass(frozen=True)
class PromotionV2Config:
    """Configurable thresholds for Promotion Gates v2 (G1-G8)."""

    min_oos_bss: float = 0.02
    min_fold_win_count: int = 3
    min_oos_selected_trades: int = 50
    min_holdout_selected_trades: int = 30
    min_dsr: float = 0.95
    min_cpcv_profitable_fraction: float = 0.60
    max_holdout_ece: float = 0.10
    bootstrap_resamples: int = 2000
    bootstrap_mean_block_length: float = 5.0
    bootstrap_seed: int = 42


@dataclass(frozen=True)
class PromotionV2Decision:
    """Full Promotion Gates v2 decision with per-gate diagnostics."""

    promoted: bool
    gates: tuple[PromotionGateResult, ...]
    rejection_reasons: tuple[str, ...]


def decide_promotion_v2(
    *,
    oof_y_true: ArrayLike,
    oof_p_model: ArrayLike,
    oof_p_naive: ArrayLike,
    oof_selected_r_1x: Sequence[float],
    oof_selected_r_15x: Sequence[float],
    oof_all_setup_expectancy: float,
    fold_bss_and_expectancy: Sequence[tuple[float, float]],
    holdout_selected_trades: int,
    holdout_bss: float,
    holdout_selected_expectancy: float,
    holdout_weighted_ece: float,
    dsr: float | None,
    cpcv_profitable_fraction: float | None,
    oof_sample_weight: ArrayLike | None = None,
    config: PromotionV2Config | None = None,
) -> PromotionV2Decision:
    """Evaluate Promotion Gates v2 (G1-G8); reject if any single gate fails."""
    cfg = config or PromotionV2Config()
    y = np.asarray(oof_y_true, dtype=np.float64)
    pm = np.asarray(oof_p_model, dtype=np.float64)
    pn = np.asarray(oof_p_naive, dtype=np.float64)
    n = len(y)
    if n == 0 or len(pm) != n or len(pn) != n:
        raise ValueError("OOS arrays must be non-empty and share identical lengths")

    if oof_sample_weight is not None:
        sw = np.asarray(oof_sample_weight, dtype=np.float64)
        if len(sw) != n or float(np.sum(sw)) <= 0.0 or np.any(sw < 0.0):
            raise ValueError("oof_sample_weight must match length and have positive sum")
        norm_w = sw / float(np.sum(sw))
    else:
        norm_w = np.full(n, 1.0 / float(n), dtype=np.float64)

    naive_bs_vec = (pn - y) ** 2
    model_bs_vec = (pm - y) ** 2
    naive_bs = float(np.sum(norm_w * naive_bs_vec))
    model_bs = float(np.sum(norm_w * model_bs_vec))
    oos_bss = compute_brier_skill_score(model_bs, naive_bs)

    # Elementwise scaled Brier improvement so the mean equals weighted (BS_naive - BS_model)
    brier_diff_series = (norm_w * float(n)) * (naive_bs_vec - model_bs_vec)
    ci_low, _ci_high, _pos_rate = stationary_block_bootstrap_ci(
        brier_diff_series,
        resamples=cfg.bootstrap_resamples,
        mean_block_length=cfg.bootstrap_mean_block_length,
        seed=cfg.bootstrap_seed,
    )
    g1_pass = ci_low > 0.0 and oos_bss >= cfg.min_oos_bss
    g1 = PromotionGateResult(
        gate_id="G1",
        passed=g1_pass,
        value=oos_bss,
        threshold=f"CI_low > 0 and BSS >= {cfg.min_oos_bss:.2f}",
        detail=(
            f"G1: OOS block-CI lower bound={ci_low:.5f}, OOS BSS={oos_bss:.4f} "
            f"(need CI_low > 0 and BSS >= {cfg.min_oos_bss:.2f})"
        ),
    )

    winning_folds = sum(
        1 for bss_f, exp_f in fold_bss_and_expectancy if bss_f > 0.0 and exp_f > 0.0
    )
    total_folds = len(fold_bss_and_expectancy)
    g2_pass = winning_folds >= cfg.min_fold_win_count
    g2 = PromotionGateResult(
        gate_id="G2",
        passed=g2_pass,
        value=float(winning_folds),
        threshold=f">= {cfg.min_fold_win_count} folds",
        detail=(
            f"G2: {winning_folds}/{total_folds} folds had BSS > 0 and selected expectancy > 0 "
            f"(need >= {cfg.min_fold_win_count})"
        ),
    )

    oos_sel_count = len(oof_selected_r_1x)
    g3_pass = oos_sel_count >= cfg.min_oos_selected_trades
    g3 = PromotionGateResult(
        gate_id="G3",
        passed=g3_pass,
        value=float(oos_sel_count),
        threshold=f">= {cfg.min_oos_selected_trades} trades",
        detail=f"G3: OOS selected trades={oos_sel_count} (need >= {cfg.min_oos_selected_trades})",
    )

    sel_exp_1x = float(np.mean(oof_selected_r_1x)) if oof_selected_r_1x else float("-inf")
    sel_exp_15x = float(np.mean(oof_selected_r_15x)) if oof_selected_r_15x else float("-inf")
    lift = sel_exp_1x - oof_all_setup_expectancy
    g4_pass = (
        bool(oof_selected_r_1x)
        and bool(oof_selected_r_15x)
        and lift > 0.0
        and sel_exp_1x > 0.0
        and sel_exp_15x >= 0.0
    )
    g4 = PromotionGateResult(
        gate_id="G4",
        passed=g4_pass,
        value=sel_exp_1x if np.isfinite(sel_exp_1x) else None,
        threshold="lift > 0, E[R]@1.0x > 0, E[R]@1.5x >= 0",
        detail=(
            f"G4: selected E[R]@1.0x={sel_exp_1x:.4f} (all={oof_all_setup_expectancy:.4f}, "
            f"lift={lift:.4f}), E[R]@1.5x={sel_exp_15x:.4f}"
        ),
    )

    g5_pass = (
        holdout_selected_trades >= cfg.min_holdout_selected_trades
        and holdout_bss > 0.0
        and holdout_selected_expectancy > 0.0
    )
    g5 = PromotionGateResult(
        gate_id="G5",
        passed=g5_pass,
        value=holdout_selected_expectancy,
        threshold=f"trades >= {cfg.min_holdout_selected_trades}, BSS > 0, E[R] > 0",
        detail=(
            f"G5: holdout trades={holdout_selected_trades} "
            f"(need >= {cfg.min_holdout_selected_trades}), "
            f"BSS={holdout_bss:.4f}, E[R]={holdout_selected_expectancy:.4f}"
        ),
    )

    g6_pass = dsr is not None and dsr >= cfg.min_dsr
    g6 = PromotionGateResult(
        gate_id="G6",
        passed=g6_pass,
        value=dsr,
        threshold=f">= {cfg.min_dsr:.2f}",
        detail=f"G6: DSR={dsr if dsr is not None else 'N/A'} (need >= {cfg.min_dsr:.2f})",
    )

    g7_pass = (
        cpcv_profitable_fraction is not None
        and cpcv_profitable_fraction >= cfg.min_cpcv_profitable_fraction
    )
    g7 = PromotionGateResult(
        gate_id="G7",
        passed=g7_pass,
        value=cpcv_profitable_fraction,
        threshold=f">= {cfg.min_cpcv_profitable_fraction:.2f}",
        detail=(
            f"G7: CPCV profitable path fraction="
            f"{cpcv_profitable_fraction if cpcv_profitable_fraction is not None else 'N/A'} "
            f"(need >= {cfg.min_cpcv_profitable_fraction:.2f})"
        ),
    )

    g8_pass = holdout_weighted_ece <= cfg.max_holdout_ece
    g8 = PromotionGateResult(
        gate_id="G8",
        passed=g8_pass,
        value=holdout_weighted_ece,
        threshold=f"<= {cfg.max_holdout_ece:.2f}",
        detail=(
            f"G8: holdout weighted 5-bin ECE={holdout_weighted_ece:.4f} "
            f"(need <= {cfg.max_holdout_ece:.2f})"
        ),
    )

    gates = (g1, g2, g3, g4, g5, g6, g7, g8)
    reasons = tuple(g.detail for g in gates if not g.passed)
    return PromotionV2Decision(
        promoted=not reasons,
        gates=gates,
        rejection_reasons=reasons,
    )
