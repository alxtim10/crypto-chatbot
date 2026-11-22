"""Phase 12 offline ML research pipeline (leakage-controlled & strictly non-live)."""

from crypto_chatbot.ml.evaluation import (
    compute_ece,
    compute_evaluation_summary,
    compute_regime_stability,
    decide_promotion,
)
from crypto_chatbot.ml.experiment import (
    build_model_artifact,
    evaluate_feature_rows_experiment,
    run_offline_experiment,
)
from crypto_chatbot.ml.features import (
    ALL_FEATURE_NAMES,
    FEATURE_FAMILIES,
    build_point_in_time_dataset,
    extract_latest_production_features,
)
from crypto_chatbot.ml.models import (
    CalibratedLogisticPipeline,
    FeaturePreprocessor,
    L2LogisticRegression,
    NaiveBaseRateModel,
    PlattCalibrator,
)
from crypto_chatbot.ml.shadow import (
    ShadowInferenceEngine,
    compute_shadow_monitoring_report,
    settle_shadow_prediction,
)
from crypto_chatbot.ml.validation import (
    PurgedSplit,
    assert_no_fold_leakage,
    generate_purged_walk_forward_splits,
    split_dev_and_holdout,
)

__all__ = [
    "ALL_FEATURE_NAMES",
    "FEATURE_FAMILIES",
    "CalibratedLogisticPipeline",
    "FeaturePreprocessor",
    "L2LogisticRegression",
    "NaiveBaseRateModel",
    "PlattCalibrator",
    "PurgedSplit",
    "ShadowInferenceEngine",
    "assert_no_fold_leakage",
    "build_model_artifact",
    "build_point_in_time_dataset",
    "compute_ece",
    "compute_evaluation_summary",
    "compute_regime_stability",
    "compute_shadow_monitoring_report",
    "decide_promotion",
    "evaluate_feature_rows_experiment",
    "extract_latest_production_features",
    "generate_purged_walk_forward_splits",
    "run_offline_experiment",
    "settle_shadow_prediction",
    "split_dev_and_holdout",
]
