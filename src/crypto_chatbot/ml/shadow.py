"""Phase 13 shadow ML inference, drift/calibration monitoring, and outcome settlement."""

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta

import numpy as np

from crypto_chatbot.ml.evaluation import (
    compute_brier_skill_score,
    compute_calibration_bins,
    compute_ece,
)
from crypto_chatbot.ml.features import extract_latest_production_features
from crypto_chatbot.schemas.analysis import AnalysisSnapshot, ModelPredictionSummary
from crypto_chatbot.schemas.market import MarketCandle, QualityStatus
from crypto_chatbot.schemas.ml import (
    CalibrationBinMetrics,
    CostAssumptions,
    MlExperimentConfig,
    MlInferenceMode,
    MlModelArtifact,
    ShadowMonitoringReport,
    ShadowPredictionRecord,
)
from crypto_chatbot.schemas.news import SentimentObservation

_NON_GUARANTEE_NOTICE = (
    "Estimasi probabilistik bersifat riset historis & eksperimental; bukan jaminan hasil, "
    "rekomendasi transaksi, atau pengganti analisis deterministik."
)


def _confidence_label(
    prob: float | None,
    *,
    abstained: bool,
    base_rate: float = 0.5,
) -> str:
    if abstained or prob is None:
        return "abstained"
    edge = abs(prob - base_rate)
    if edge >= 0.15:
        return "high"
    if edge >= 0.08:
        return "moderate"
    return "low"


class ShadowInferenceEngine:
    """Runs calibrated offline ML artifacts in off, shadow, or promoted release modes."""

    def __init__(
        self,
        artifact: MlModelArtifact,
        *,
        mode: MlInferenceMode = MlInferenceMode.SHADOW,
        min_prob_margin: float = 0.05,
        max_feature_zscore: float = 5.0,
    ) -> None:
        if min_prob_margin < 0.0 or min_prob_margin >= 0.5:
            raise ValueError("min_prob_margin must lie in [0.0, 0.5)")
        if max_feature_zscore <= 0.0:
            raise ValueError("max_feature_zscore must be positive")
        self._artifact = artifact
        self._mode = mode
        self._min_prob_margin = min_prob_margin
        self._max_feature_zscore = max_feature_zscore

    @property
    def mode(self) -> MlInferenceMode:
        return self._mode

    @property
    def artifact(self) -> MlModelArtifact:
        return self._artifact

    def _predict_from_features(
        self, feature_map: Mapping[str, float | None]
    ) -> tuple[float, float]:
        n = len(self._artifact.feature_names)
        x = np.empty(n, dtype=np.float64)
        for j, name in enumerate(self._artifact.feature_names):
            raw = feature_map.get(name)
            x[j] = (
                float(raw) if raw is not None and math.isfinite(raw) else self._artifact.medians[j]
            )

        means = np.asarray(self._artifact.means, dtype=np.float64)
        stds = np.asarray(self._artifact.stds, dtype=np.float64)
        weights = np.asarray(self._artifact.weights, dtype=np.float64)

        raw_z = np.abs((x - means) / stds)
        max_z = float(np.max(raw_z)) if n > 0 else 0.0
        scaled = np.clip((x - means) / stds, -10.0, 10.0)

        logit = float(self._artifact.bias + float(np.dot(scaled, weights)))
        clipped_logit = max(-30.0, min(30.0, logit))
        cal_logit = self._artifact.platt_a * clipped_logit + self._artifact.platt_b
        cal_clipped = max(-30.0, min(30.0, cal_logit))
        prob = 1.0 / (1.0 + math.exp(-cal_clipped))
        return max(1e-6, min(1.0 - 1e-6, prob)), max_z

    def predict_for_snapshot(
        self,
        snapshot: AnalysisSnapshot,
        candles: Sequence[MarketCandle],
        *,
        sentiment_observations: Sequence[SentimentObservation] | None = None,
        use_available_time: bool = False,
    ) -> tuple[ShadowPredictionRecord | None, ModelPredictionSummary | None]:
        """Run shadow/promoted prediction cleanly using identical causal feature definitions."""
        if self._mode is MlInferenceMode.OFF:
            return None, None
        if (
            snapshot.canonical_symbol != self._artifact.canonical_symbol
            or snapshot.timeframe is not self._artifact.timeframe
        ):
            return None, None

        t0 = time.monotonic()
        regime_name = snapshot.regime.value if snapshot.regime is not None else None
        cal_prob: float | None = None
        max_z: float | None = None
        abstained = False
        abstention_reason: str | None = None

        if not self._artifact.accepted:
            abstained = True
            abstention_reason = "model_not_accepted"
        elif snapshot.quality_status is not QualityStatus.READY:
            abstained = True
            abstention_reason = "poor_data_quality"
        elif regime_name is None or regime_name not in self._artifact.seen_regimes:
            abstained = True
            abstention_reason = "unseen_regime"
        else:
            config = MlExperimentConfig(
                canonical_symbol=self._artifact.canonical_symbol,
                timeframe=self._artifact.timeframe,
                target_name=self._artifact.target_name,
                horizon_hours=self._artifact.horizon_hours,
                noise_buffer_bps=self._artifact.noise_buffer_bps,
                cost_assumptions=self._artifact.cost_assumptions,
                feature_version=self._artifact.feature_version,
                model_version=self._artifact.model_version,
            )
            try:
                latest_row = extract_latest_production_features(
                    candles,
                    config=config,
                    sentiment_observations=sentiment_observations,
                    use_available_time=use_available_time,
                )
            except ValueError:
                abstained = True
                abstention_reason = "feature_extraction_failed"
            else:
                prob, raw_max_z = self._predict_from_features(latest_row.features)
                max_z = raw_max_z
                edge = abs(prob - self._artifact.base_rate)
                if raw_max_z > self._max_feature_zscore:
                    abstained = True
                    abstention_reason = "feature_drift_high"
                elif edge < self._min_prob_margin:
                    cal_prob = prob
                    abstained = True
                    abstention_reason = "low_confidence"
                else:
                    cal_prob = prob

        conf_lbl = _confidence_label(
            cal_prob,
            abstained=abstained,
            base_rate=self._artifact.base_rate,
        )
        latency_ms = max(0, int((time.monotonic() - t0) * 1000))
        pred_raw = (
            f"{snapshot.snapshot_id}|{self._artifact.artifact_id}|{snapshot.as_of.isoformat()}"
        )
        prediction_id = "pred-" + hashlib.sha256(pred_raw.encode()).hexdigest()[:24]
        exposed = self._mode is MlInferenceMode.PROMOTED

        record = ShadowPredictionRecord(
            prediction_id=prediction_id,
            snapshot_id=snapshot.snapshot_id,
            experiment_id=self._artifact.experiment_id,
            canonical_symbol=snapshot.canonical_symbol,
            timeframe=snapshot.timeframe,
            target_name=self._artifact.target_name,
            horizon_hours=self._artifact.horizon_hours,
            prediction_time=snapshot.as_of,
            model_version=self._artifact.model_version,
            feature_version=self._artifact.feature_version,
            quality_status=snapshot.quality_status,
            regime=regime_name,
            calibrated_probability=cal_prob,
            confidence_label=conf_lbl,
            abstained=abstained,
            abstention_reason=abstention_reason,
            max_feature_zscore=max_z,
            latency_ms=latency_ms,
            exposed_to_user=exposed,
        )

        if not exposed:
            return record, None

        limitations = [
            "Model ML dilatih dari data historis dengan asumsi biaya round-trip.",
            f"Kondisi kualitas input saat prediksi: {snapshot.quality_status.value}.",
        ]
        if abstained and abstention_reason:
            limitations.append(f"Model memilih abstain ({abstention_reason}).")
        summary = ModelPredictionSummary(
            prediction_id=prediction_id,
            target_name=self._artifact.target_name,
            horizon_hours=self._artifact.horizon_hours,
            calibrated_probability=cal_prob,
            abstained=abstained,
            abstention_reason=abstention_reason,
            confidence_label=conf_lbl,
            data_quality_status=snapshot.quality_status,
            model_version=self._artifact.model_version,
            feature_version=self._artifact.feature_version,
            limitations=tuple(limitations),
            non_guarantee_notice=_NON_GUARANTEE_NOTICE,
        )
        return record, summary


def settle_shadow_prediction(
    record: ShadowPredictionRecord,
    candles: Sequence[MarketCandle],
    *,
    cost_assumptions: CostAssumptions,
    noise_buffer_bps: float,
) -> tuple[float | None, int | None, datetime | None]:
    """Settle one shadow prediction once the target close candle is available."""
    pred_utc = record.prediction_time.astimezone(UTC)
    target_time = pred_utc + timedelta(hours=record.horizon_hours)

    ref_candle = next(
        (
            c
            for c in candles
            if c.is_closed
            and c.canonical_symbol == record.canonical_symbol
            and c.timeframe is record.timeframe
            and c.close_time == pred_utc
        ),
        None,
    )
    target_candle = next(
        (
            c
            for c in candles
            if c.is_closed
            and c.canonical_symbol == record.canonical_symbol
            and c.timeframe is record.timeframe
            and c.close_time == target_time
        ),
        None,
    )
    if ref_candle is None or target_candle is None or ref_candle.close <= 0.0:
        return None, None, None

    fwd_ret = (target_candle.close / ref_candle.close) - 1.0
    threshold = (cost_assumptions.round_trip_cost_bps + noise_buffer_bps) / 10_000.0
    label = 1 if fwd_ret > threshold else 0
    return fwd_ret, label, target_candle.close_time


def compute_shadow_monitoring_report(
    records: Sequence[ShadowPredictionRecord],
    *,
    max_brier: float | None = None,
    max_ece: float = 0.12,
    drift_zscore_threshold: float = 4.0,
    max_drift_rate: float = 0.20,
    min_settled: int = 30,
    min_settled_ece: int = 50,
    reference_base_rate: float | None = None,
) -> ShadowMonitoringReport:
    """Audit feature drift, abstention rates, and live calibration across shadow logs."""
    total = len(records)
    if total == 0:
        return ShadowMonitoringReport(
            total_predictions=0,
            settled_predictions=0,
            abstained_count=0,
            abstention_rate=0.0,
            realized_brier=None,
            realized_ece=None,
            mean_max_zscore=None,
            drift_alert_count=0,
            is_healthy=False,
            alerts=("no_shadow_predictions",),
        )

    abstained_count = sum(1 for r in records if r.abstained)
    abstention_rate = abstained_count / total

    zscores = [r.max_feature_zscore for r in records if r.max_feature_zscore is not None]
    mean_z = float(np.mean(zscores)) if zscores else None
    drift_alerts = sum(1 for z in zscores if z > drift_zscore_threshold)
    drift_rate = (drift_alerts / len(zscores)) if zscores else 0.0

    settled_pairs = [
        (float(r.realized_label), float(r.calibrated_probability))
        for r in records
        if not r.abstained and r.calibrated_probability is not None and r.realized_label is not None
    ]
    realized_brier: float | None = None
    realized_bss: float | None = None
    realized_ece: float | None = None
    cal_bins: tuple[CalibrationBinMetrics, ...] = ()
    if settled_pairs:
        y = np.array([pair[0] for pair in settled_pairs], dtype=np.float64)
        p = np.array([pair[1] for pair in settled_pairs], dtype=np.float64)
        realized_brier = float(np.mean((p - y) ** 2))
        naive_p = reference_base_rate if reference_base_rate is not None else float(np.mean(y))
        naive_brier = float(np.mean((naive_p - y) ** 2))
        realized_bss = compute_brier_skill_score(realized_brier, naive_brier)
        realized_ece = compute_ece(y, p, n_bins=5)
        cal_bins = compute_calibration_bins(y, p, n_bins=5)

    alerts: list[str] = []
    if len(settled_pairs) < min_settled:
        alerts.append("insufficient_settled_samples")
    else:
        if realized_bss is not None and realized_bss < 0.0:
            alerts.append("bss_below_zero")
    if max_brier is not None and realized_brier is not None and realized_brier > max_brier:
        alerts.append("brier_above_threshold")
    effective_ece_min = min(min_settled, min_settled_ece) if min_settled < 30 else min_settled_ece
    if (
        len(settled_pairs) >= effective_ece_min
        and realized_ece is not None
        and realized_ece > max_ece
    ):
        alerts.append("calibration_ece_above_threshold")
    if drift_rate > max_drift_rate:
        alerts.append("feature_drift_rate_high")

    return ShadowMonitoringReport(
        total_predictions=total,
        settled_predictions=len(settled_pairs),
        abstained_count=abstained_count,
        abstention_rate=abstention_rate,
        realized_brier=realized_brier,
        realized_brier_skill_score=realized_bss,
        realized_ece=realized_ece,
        mean_max_zscore=mean_z,
        drift_alert_count=drift_alerts,
        is_healthy=(not alerts),
        alerts=tuple(alerts),
        calibration_bins=cal_bins,
    )
