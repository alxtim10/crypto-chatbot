"""Typed Pydantic schemas for Phase 12 offline ML experiments and validation."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crypto_chatbot.schemas.analysis import (
    ModelPredictionSummary as ModelPredictionSummary,
)
from crypto_chatbot.schemas.analysis import (
    Regime,
)
from crypto_chatbot.schemas.market import FiniteFloat, QualityStatus, Timeframe, UtcDatetime

FEATURE_VERSION = "ml-feat-v1.2.0"
MODEL_VERSION = "l2-logreg-platt-v1.0.1"
DEFAULT_TARGET_NAME = "next_24h_return_exceeds_cost"


class FeatureFamily(StrEnum):
    TECHNICAL = "technical"
    STRUCTURE_PIVOT = "structure_pivot"
    SENTIMENT = "sentiment"
    ORDERFLOW = "orderflow"


class CostAssumptions(BaseModel):
    """Explicit execution friction assumptions applied to target hurdle and strategy sim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fee_bps_per_side: FiniteFloat = Field(default=10.0, ge=0.0)
    maker_fee_bps_per_side: FiniteFloat = Field(default=2.0, ge=0.0)
    spread_bps: FiniteFloat = Field(default=5.0, ge=0.0)
    slippage_bps: FiniteFloat = Field(default=5.0, ge=0.0)
    funding_bps_per_24h: FiniteFloat = Field(default=2.0, ge=0.0)
    execution_style: Literal["taker", "maker"] = "taker"

    @property
    def round_trip_cost_bps(self) -> float:
        if self.execution_style == "maker":
            return (
                2.0 * self.maker_fee_bps_per_side
                + (0.5 * self.spread_bps)
                + self.funding_bps_per_24h
            )
        return (
            2.0 * self.fee_bps_per_side
            + self.spread_bps
            + self.slippage_bps
            + self.funding_bps_per_24h
        )

    @property
    def round_trip_cost_rate(self) -> float:
        return self.round_trip_cost_bps / 10_000.0


class CalibrationBinMetrics(BaseModel):
    """Reliability bin metrics to audit calibration across probability intervals."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bin_index: int = Field(ge=0)
    bin_low: FiniteFloat = Field(ge=0.0, le=1.0)
    bin_high: FiniteFloat = Field(ge=0.0, le=1.0)
    sample_count: int = Field(ge=0)
    mean_predicted_prob: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    empirical_positive_rate: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    calibration_gap: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    mean_net_return: FiniteFloat | None = None


class FeatureRow(BaseModel):
    """Single point-in-time feature & label observation with strict causal bounds."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prediction_time: UtcDatetime
    available_time: UtcDatetime
    label_end_time: UtcDatetime
    regime: Regime | None = None
    features: Mapping[str, FiniteFloat | None]
    forward_return: FiniteFloat
    label: int = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _validate_temporal_order(self) -> FeatureRow:
        if self.available_time > self.prediction_time:
            raise ValueError("feature available_time cannot exceed prediction_time (lookahead)")
        if self.label_end_time <= self.prediction_time:
            raise ValueError("label_end_time must be strictly after prediction_time")
        return self


class MlExperimentConfig(BaseModel):
    """Versioned configuration for reproducible offline ML experiments."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_symbol: str = Field(min_length=1)
    venue: str = Field(default="binance", min_length=1)
    timeframe: Timeframe = Timeframe.H4
    target_name: str = Field(default=DEFAULT_TARGET_NAME, min_length=1)
    horizon_hours: int = Field(default=24, ge=1)
    noise_buffer_bps: FiniteFloat = Field(default=28.0, ge=0.0)
    cost_assumptions: CostAssumptions = Field(default_factory=CostAssumptions)
    feature_version: str = Field(default=FEATURE_VERSION, min_length=1)
    model_version: str = Field(default=MODEL_VERSION, min_length=1)
    l2_reg: FiniteFloat = Field(default=1.0, gt=0.0)
    prob_threshold: FiniteFloat = Field(default=0.55, gt=0.0, lt=1.0)
    n_splits: int = Field(default=4, ge=2, le=10)
    holdout_fraction: FiniteFloat = Field(default=0.20, gt=0.05, lt=0.50)
    embargo_bars: int = Field(default=3, ge=0)
    min_brier_improvement: FiniteFloat = Field(default=0.005, ge=0.0)
    min_trades: int = Field(default=5, ge=1)
    min_fold_win_rate: FiniteFloat = Field(default=0.60, gt=0.0, le=1.0)


class FoldMetrics(BaseModel):
    """Out-of-sample metrics for one purged walk-forward validation fold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    fold_index: int = Field(ge=0)
    train_size: int = Field(ge=1)
    val_size: int = Field(ge=1)
    train_end_time: UtcDatetime
    val_start_time: UtcDatetime
    val_end_time: UtcDatetime
    naive_brier: FiniteFloat = Field(ge=0.0, le=1.0)
    model_brier: FiniteFloat = Field(ge=0.0, le=1.0)
    brier_improvement: FiniteFloat
    net_return_after_costs: FiniteFloat
    trade_count: int = Field(ge=0)


class RegimeSliceMetrics(BaseModel):
    """Stability breakdown of out-of-sample model performance across market regimes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    regime: str = Field(min_length=1)
    sample_count: int = Field(ge=1)
    naive_brier: FiniteFloat = Field(ge=0.0, le=1.0)
    model_brier: FiniteFloat = Field(ge=0.0, le=1.0)
    net_return_after_costs: FiniteFloat
    trade_count: int = Field(ge=0)


class EvaluationSummary(BaseModel):
    """Probabilistic, classification, calibration, and post-cost strategy metrics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_count: int = Field(ge=1)
    positive_rate: FiniteFloat = Field(ge=0.0, le=1.0)
    naive_brier: FiniteFloat = Field(ge=0.0, le=1.0)
    model_brier: FiniteFloat = Field(ge=0.0, le=1.0)
    brier_skill_vs_naive: FiniteFloat
    brier_skill_score: FiniteFloat = Field(default=0.0)
    calibration_ece: FiniteFloat = Field(ge=0.0, le=1.0)
    balanced_accuracy: FiniteFloat = Field(ge=0.0, le=1.0)
    precision: FiniteFloat = Field(ge=0.0, le=1.0)
    recall: FiniteFloat = Field(ge=0.0, le=1.0)
    trade_count: int = Field(ge=0)
    net_return_after_costs: FiniteFloat
    mean_return_per_trade_after_costs: FiniteFloat
    max_drawdown: FiniteFloat = Field(ge=0.0, le=1.0)
    calibration_bins: tuple[CalibrationBinMetrics, ...] = ()
    maker_net_return_after_costs: FiniteFloat | None = None
    taker_net_return_after_costs: FiniteFloat | None = None


class AblationResult(BaseModel):
    """Walk-forward out-of-sample evaluation for one feature-family subset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    family_name: str = Field(min_length=1)
    feature_names: tuple[str, ...] = Field(min_length=1)
    oof_summary: EvaluationSummary
    fold_metrics: tuple[FoldMetrics, ...]


class MlExperimentReport(BaseModel):
    """Complete reproducible Phase 12 experiment report with explicit acceptance gate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(min_length=8, max_length=64)
    created_at: UtcDatetime
    config: MlExperimentConfig
    config_sha256: str = Field(min_length=64, max_length=64)
    dataset_sha256: str = Field(min_length=64, max_length=64)
    total_rows: int = Field(ge=1)
    dev_rows: int = Field(ge=1)
    holdout_rows: int = Field(ge=1)
    oof_summary: EvaluationSummary
    holdout_summary: EvaluationSummary
    fold_metrics: tuple[FoldMetrics, ...]
    regime_stability: tuple[RegimeSliceMetrics, ...]
    ablations: tuple[AblationResult, ...]
    accepted: bool
    rejection_reasons: tuple[str, ...] = ()


class MlInferenceMode(StrEnum):
    OFF = "off"
    SHADOW = "shadow"
    PROMOTED = "promoted"


class MlModelArtifact(BaseModel):
    """Serializable calibrated model artifact exported from an accepted offline experiment."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    artifact_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    canonical_symbol: str = Field(min_length=1)
    timeframe: Timeframe
    target_name: str = DEFAULT_TARGET_NAME
    horizon_hours: int = Field(default=24, ge=1)
    noise_buffer_bps: FiniteFloat = Field(default=15.0, ge=0.0)
    cost_assumptions: CostAssumptions = Field(default_factory=CostAssumptions)
    feature_version: str = FEATURE_VERSION
    model_version: str = MODEL_VERSION
    feature_names: tuple[str, ...]
    medians: tuple[FiniteFloat, ...]
    means: tuple[FiniteFloat, ...]
    stds: tuple[FiniteFloat, ...]
    weights: tuple[FiniteFloat, ...]
    bias: FiniteFloat
    platt_a: FiniteFloat
    platt_b: FiniteFloat
    base_rate: FiniteFloat = Field(default=0.5, gt=0.0, lt=1.0)
    seen_regimes: tuple[str, ...]
    accepted: bool
    created_at: UtcDatetime

    @model_validator(mode="after")
    def _validate_shapes(self) -> MlModelArtifact:
        n = len(self.feature_names)
        if n == 0:
            raise ValueError("feature_names must not be empty")
        if any(len(arr) != n for arr in (self.medians, self.means, self.stds, self.weights)):
            raise ValueError("preprocessing/model parameter lengths must match feature_names")
        if any(s <= 0.0 for s in self.stds):
            raise ValueError("all feature stds must be strictly positive")
        return self


class ShadowPredictionRecord(BaseModel):
    """Audit record for shadow/promoted inference, drift check, and realized settlement."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prediction_id: str = Field(min_length=1)
    snapshot_id: str = Field(min_length=1)
    experiment_id: str = Field(min_length=1)
    canonical_symbol: str = Field(min_length=1)
    timeframe: Timeframe
    target_name: str = Field(min_length=1)
    horizon_hours: int = Field(ge=1)
    prediction_time: UtcDatetime
    model_version: str = Field(min_length=1)
    feature_version: str = Field(min_length=1)
    quality_status: QualityStatus
    regime: str | None = None
    calibrated_probability: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    confidence_label: str = Field(min_length=1)
    abstained: bool
    abstention_reason: str | None = None
    max_feature_zscore: FiniteFloat | None = Field(default=None, ge=0.0)
    latency_ms: int = Field(ge=0)
    exposed_to_user: bool = False
    realized_return: FiniteFloat | None = None
    realized_label: int | None = Field(default=None, ge=0, le=1)
    settled_at: UtcDatetime | None = None


class ShadowMonitoringReport(BaseModel):
    """Live drift, abstention, and calibration audit across persisted shadow predictions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_predictions: int = Field(ge=0)
    settled_predictions: int = Field(ge=0)
    abstained_count: int = Field(ge=0)
    abstention_rate: FiniteFloat = Field(ge=0.0, le=1.0)
    realized_brier: FiniteFloat | None = Field(default=None, ge=0.0)
    realized_brier_skill_score: FiniteFloat | None = None
    realized_ece: FiniteFloat | None = Field(default=None, ge=0.0, le=1.0)
    mean_max_zscore: FiniteFloat | None = Field(default=None, ge=0.0)
    drift_alert_count: int = Field(ge=0)
    is_healthy: bool
    alerts: tuple[str, ...] = ()
    calibration_bins: tuple[CalibrationBinMetrics, ...] = ()
