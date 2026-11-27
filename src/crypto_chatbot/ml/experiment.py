"""Reproducible Phase 12 offline ML experiment runner, ablations, and holdout gate."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from datetime import UTC, datetime

import numpy as np
from numpy.typing import NDArray

from crypto_chatbot.ml.evaluation import (
    compute_evaluation_summary,
    compute_regime_stability,
    decide_promotion,
)
from crypto_chatbot.ml.features import (
    ALL_FEATURE_NAMES,
    FEATURE_FAMILIES,
    build_point_in_time_dataset,
    select_available_feature_families,
)
from crypto_chatbot.ml.models import CalibratedLogisticPipeline, NaiveBaseRateModel
from crypto_chatbot.ml.validation import (
    PurgedSplit,
    assert_no_fold_leakage,
    generate_purged_walk_forward_splits,
    split_dev_and_holdout,
)
from crypto_chatbot.schemas.market import MarketCandle
from crypto_chatbot.schemas.ml import (
    AblationResult,
    EvaluationSummary,
    FeatureFamily,
    FeatureRow,
    FoldMetrics,
    MlExperimentConfig,
    MlExperimentReport,
    MlModelArtifact,
)
from crypto_chatbot.schemas.news import SentimentObservation


def _extract_matrix(
    rows: Sequence[FeatureRow],
    feature_names: Sequence[str],
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    x = np.array(
        [
            [
                np.nan if row.features.get(name) is None else float(row.features[name] or 0.0)
                for name in feature_names
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    y = np.array([row.label for row in rows], dtype=np.float64)
    fwd = np.array([row.forward_return for row in rows], dtype=np.float64)
    return x, y, fwd


def _evaluate_walk_forward(
    dev_rows: Sequence[FeatureRow],
    splits: Sequence[PurgedSplit],
    feature_names: Sequence[str],
    *,
    config: MlExperimentConfig,
) -> tuple[
    EvaluationSummary,
    tuple[FoldMetrics, ...],
    tuple[FeatureRow, ...],
    NDArray[np.float64],
    NDArray[np.float64],
]:
    assert_no_fold_leakage(dev_rows, splits)
    x_all, y_all, fwd_all = _extract_matrix(dev_rows, feature_names)

    oof_rows: list[FeatureRow] = []
    oof_p_model: list[float] = []
    oof_p_naive: list[float] = []
    oof_y: list[float] = []
    oof_fwd: list[float] = []
    fold_results: list[FoldMetrics] = []

    for split in splits:
        tr_idx = list(split.train_indices)
        va_idx = list(split.val_indices)

        x_tr, y_tr = x_all[tr_idx], y_all[tr_idx]
        x_va, y_va = x_all[va_idx], y_all[va_idx]
        fwd_va = fwd_all[va_idx]

        naive = NaiveBaseRateModel().fit(y_tr)
        p_naive_va = naive.predict_proba(len(va_idx))

        pipeline = CalibratedLogisticPipeline(
            feature_names=feature_names,
            l2_reg=config.l2_reg,
        ).fit(
            x_tr,
            y_tr,
            pred_times_train=[dev_rows[i].prediction_time for i in tr_idx],
            label_end_times_train=[dev_rows[i].label_end_time for i in tr_idx],
        )
        p_model_va = pipeline.predict_proba(x_va)

        val_slice_rows = [dev_rows[i] for i in va_idx]
        fold_eval = compute_evaluation_summary(
            y_va,
            p_model_va,
            p_naive_va,
            fwd_va,
            cost_assumptions=config.cost_assumptions,
            prob_threshold=config.prob_threshold,
            prediction_times=[r.prediction_time for r in val_slice_rows],
            label_end_times=[r.label_end_time for r in val_slice_rows],
        )
        fold_results.append(
            FoldMetrics(
                fold_index=split.fold_index,
                train_size=len(tr_idx),
                val_size=len(va_idx),
                train_end_time=dev_rows[tr_idx[-1]].prediction_time,
                val_start_time=dev_rows[va_idx[0]].prediction_time,
                val_end_time=dev_rows[va_idx[-1]].prediction_time,
                naive_brier=fold_eval.naive_brier,
                model_brier=fold_eval.model_brier,
                brier_improvement=fold_eval.naive_brier - fold_eval.model_brier,
                net_return_after_costs=fold_eval.net_return_after_costs,
                trade_count=fold_eval.trade_count,
            )
        )

        oof_rows.extend(val_slice_rows)
        oof_p_model.extend(float(v) for v in p_model_va)
        oof_p_naive.extend(float(v) for v in p_naive_va)
        oof_y.extend(float(v) for v in y_va)
        oof_fwd.extend(float(v) for v in fwd_va)

    p_model_arr = np.asarray(oof_p_model, dtype=np.float64)
    p_naive_arr = np.asarray(oof_p_naive, dtype=np.float64)
    summary = compute_evaluation_summary(
        np.asarray(oof_y, dtype=np.float64),
        p_model_arr,
        p_naive_arr,
        np.asarray(oof_fwd, dtype=np.float64),
        cost_assumptions=config.cost_assumptions,
        prob_threshold=config.prob_threshold,
        prediction_times=[r.prediction_time for r in oof_rows],
        label_end_times=[r.label_end_time for r in oof_rows],
    )
    return summary, tuple(fold_results), tuple(oof_rows), p_model_arr, p_naive_arr


def _resolve_active_ml_features(
    dev_rows: Sequence[FeatureRow],
    *,
    has_sentiment: bool,
) -> tuple[tuple[str, ...], bool]:
    """Select active feature names cleanly without silently retaining all-missing families."""
    candidates: list[FeatureFamily] = [
        FeatureFamily.TECHNICAL,
        FeatureFamily.STRUCTURE_PIVOT,
        FeatureFamily.ORDERFLOW,
    ]
    if has_sentiment:
        candidates.append(FeatureFamily.SENTIMENT)
    selected, excluded = select_available_feature_families(dev_rows, candidates)
    has_orderflow = FeatureFamily.ORDERFLOW not in excluded
    return selected, has_orderflow


def _run_ablations(
    dev_rows: Sequence[FeatureRow],
    splits: Sequence[PurgedSplit],
    *,
    config: MlExperimentConfig,
    has_sentiment: bool,
    has_orderflow: bool = False,
) -> tuple[AblationResult, ...]:
    ablation_specs: list[tuple[str, tuple[str, ...]]] = [
        ("technical", FEATURE_FAMILIES[FeatureFamily.TECHNICAL]),
        (
            "technical+structure_pivot",
            (
                *FEATURE_FAMILIES[FeatureFamily.TECHNICAL],
                *FEATURE_FAMILIES[FeatureFamily.STRUCTURE_PIVOT],
            ),
        ),
    ]
    if has_orderflow:
        ablation_specs.append(
            (
                "technical+structure_pivot+orderflow",
                (
                    *FEATURE_FAMILIES[FeatureFamily.TECHNICAL],
                    *FEATURE_FAMILIES[FeatureFamily.STRUCTURE_PIVOT],
                    *FEATURE_FAMILIES[FeatureFamily.ORDERFLOW],
                ),
            )
        )
    if has_sentiment:
        sentiment_combo = (
            ALL_FEATURE_NAMES
            if has_orderflow
            else (
                *FEATURE_FAMILIES[FeatureFamily.TECHNICAL],
                *FEATURE_FAMILIES[FeatureFamily.STRUCTURE_PIVOT],
                *FEATURE_FAMILIES[FeatureFamily.SENTIMENT],
            )
        )
        ablation_specs.append(("all_families", sentiment_combo))

    results: list[AblationResult] = []
    for name, feat_names in ablation_specs:
        summary, folds, _, _, _ = _evaluate_walk_forward(
            dev_rows,
            splits,
            feat_names,
            config=config,
        )
        results.append(
            AblationResult(
                family_name=name,
                feature_names=feat_names,
                oof_summary=summary,
                fold_metrics=folds,
            )
        )
    return tuple(results)


def evaluate_feature_rows_experiment(
    rows: Sequence[FeatureRow],
    *,
    dataset_sha256: str,
    config: MlExperimentConfig,
    has_sentiment: bool = False,
    created_at: datetime | None = None,
) -> MlExperimentReport:
    """Run full purged walk-forward CV, ablations, untouched holdout, and promotion gate."""
    dev_rows, holdout_rows = split_dev_and_holdout(
        rows,
        holdout_fraction=config.holdout_fraction,
        embargo_bars=config.embargo_bars,
    )
    splits = generate_purged_walk_forward_splits(
        dev_rows,
        n_splits=config.n_splits,
        embargo_bars=config.embargo_bars,
    )

    active_features, has_orderflow = _resolve_active_ml_features(
        dev_rows,
        has_sentiment=has_sentiment,
    )
    oof_summary, fold_metrics, oof_rows, oof_p_model, oof_p_naive = _evaluate_walk_forward(
        dev_rows,
        splits,
        active_features,
        config=config,
    )
    regime_stability = compute_regime_stability(
        oof_rows,
        oof_p_model,
        oof_p_naive,
        cost_assumptions=config.cost_assumptions,
        prob_threshold=config.prob_threshold,
    )
    ablations = _run_ablations(
        dev_rows,
        splits,
        config=config,
        has_sentiment=has_sentiment,
        has_orderflow=has_orderflow,
    )

    # Final model fit strictly on `dev_rows`, evaluated once on untouched `holdout_rows`.
    x_dev, y_dev, _ = _extract_matrix(dev_rows, active_features)
    x_hold, y_hold, fwd_hold = _extract_matrix(holdout_rows, active_features)

    naive_final = NaiveBaseRateModel().fit(y_dev)
    p_naive_hold = naive_final.predict_proba(len(holdout_rows))

    final_pipeline = CalibratedLogisticPipeline(
        feature_names=active_features,
        l2_reg=config.l2_reg,
    ).fit(
        x_dev,
        y_dev,
        pred_times_train=[r.prediction_time for r in dev_rows],
        label_end_times_train=[r.label_end_time for r in dev_rows],
    )
    p_model_hold = final_pipeline.predict_proba(x_hold)

    holdout_summary = compute_evaluation_summary(
        y_hold,
        p_model_hold,
        p_naive_hold,
        fwd_hold,
        cost_assumptions=config.cost_assumptions,
        prob_threshold=config.prob_threshold,
        prediction_times=[r.prediction_time for r in holdout_rows],
        label_end_times=[r.label_end_time for r in holdout_rows],
    )

    accepted, rejection_reasons = decide_promotion(
        oof_summary,
        holdout_summary,
        fold_metrics,
        config=config,
    )

    config_json = json.dumps(
        config.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    config_sha256 = hashlib.sha256(config_json.encode()).hexdigest()
    exp_digest = hashlib.sha256(f"{config_sha256}|{dataset_sha256}".encode()).hexdigest()
    experiment_id = f"exp-{exp_digest[:16]}"
    stamp = (created_at or rows[-1].prediction_time).astimezone(UTC)

    return MlExperimentReport(
        experiment_id=experiment_id,
        created_at=stamp,
        config=config,
        config_sha256=config_sha256,
        dataset_sha256=dataset_sha256,
        total_rows=len(rows),
        dev_rows=len(dev_rows),
        holdout_rows=len(holdout_rows),
        oof_summary=oof_summary,
        holdout_summary=holdout_summary,
        fold_metrics=fold_metrics,
        regime_stability=regime_stability,
        ablations=ablations,
        accepted=accepted,
        rejection_reasons=rejection_reasons,
    )


def run_offline_experiment(
    candles: Sequence[MarketCandle],
    *,
    config: MlExperimentConfig,
    sentiment_observations: Sequence[SentimentObservation] | None = None,
    use_available_time: bool = False,
    created_at: datetime | None = None,
) -> MlExperimentReport:
    """Build point-in-time dataset from closed candles and run full offline evaluation."""
    rows, dataset_sha256 = build_point_in_time_dataset(
        candles,
        config=config,
        sentiment_observations=sentiment_observations,
        use_available_time=use_available_time,
    )
    has_sentiment = bool(sentiment_observations) and any(
        r.features.get("sentiment_missing") == 0.0 for r in rows
    )
    return evaluate_feature_rows_experiment(
        rows,
        dataset_sha256=dataset_sha256,
        config=config,
        has_sentiment=has_sentiment,
        created_at=created_at,
    )


def build_model_artifact(
    candles: Sequence[MarketCandle],
    *,
    config: MlExperimentConfig,
    sentiment_observations: Sequence[SentimentObservation] | None = None,
    use_available_time: bool = False,
    created_at: datetime | None = None,
) -> tuple[MlModelArtifact, MlExperimentReport]:
    """Fit calibrated pipeline on dev split only and export immutable model artifact + report."""
    rows, dataset_sha256 = build_point_in_time_dataset(
        candles,
        config=config,
        sentiment_observations=sentiment_observations,
        use_available_time=use_available_time,
    )
    has_sentiment = bool(sentiment_observations) and any(
        r.features.get("sentiment_missing") == 0.0 for r in rows
    )
    report = evaluate_feature_rows_experiment(
        rows,
        dataset_sha256=dataset_sha256,
        config=config,
        has_sentiment=has_sentiment,
        created_at=created_at,
    )
    dev_rows, _ = split_dev_and_holdout(
        rows,
        holdout_fraction=config.holdout_fraction,
        embargo_bars=config.embargo_bars,
    )
    active_features, _ = _resolve_active_ml_features(
        dev_rows,
        has_sentiment=has_sentiment,
    )
    x_dev, y_dev, _ = _extract_matrix(dev_rows, active_features)
    final_pipeline = CalibratedLogisticPipeline(
        feature_names=active_features,
        l2_reg=config.l2_reg,
    ).fit(
        x_dev,
        y_dev,
        pred_times_train=[r.prediction_time for r in dev_rows],
        label_end_times_train=[r.label_end_time for r in dev_rows],
    )
    medians = final_pipeline.preprocessor.medians_
    means = final_pipeline.preprocessor.means_
    stds = final_pipeline.preprocessor.stds_
    if medians is None or means is None or stds is None:
        raise RuntimeError("CalibratedLogisticPipeline preprocessor was not fitted")

    seen_regimes = tuple(sorted({r.regime.value for r in dev_rows if r.regime is not None}))
    art_raw = (
        f"{report.experiment_id}|{config.model_version}|{config.feature_version}|"
        f"{','.join(active_features)}"
    )
    artifact_id = "art-" + hashlib.sha256(art_raw.encode()).hexdigest()[:20]
    artifact = MlModelArtifact(
        artifact_id=artifact_id,
        experiment_id=report.experiment_id,
        canonical_symbol=config.canonical_symbol,
        timeframe=config.timeframe,
        target_name=config.target_name,
        horizon_hours=config.horizon_hours,
        noise_buffer_bps=config.noise_buffer_bps,
        cost_assumptions=config.cost_assumptions,
        feature_version=config.feature_version,
        model_version=config.model_version,
        feature_names=tuple(active_features),
        medians=tuple(float(v) for v in medians),
        means=tuple(float(v) for v in means),
        stds=tuple(float(v) for v in stds),
        weights=tuple(float(v) for v in final_pipeline.model.weights_),
        bias=float(final_pipeline.model.bias_),
        platt_a=float(final_pipeline.calibrator.a),
        platt_b=float(final_pipeline.calibrator.b),
        base_rate=float(final_pipeline.base_rate_),
        seen_regimes=seen_regimes,
        accepted=report.accepted,
        created_at=report.created_at,
    )
    return artifact, report
