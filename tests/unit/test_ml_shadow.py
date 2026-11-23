"""Phase 13 shadow inference, feature parity, abstention, settlement, and promotion tests."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from crypto_chatbot.analysis.service import AnalysisService, build_analysis_snapshot
from crypto_chatbot.bot.templates import render_compact_snapshot, render_deterministic_snapshot
from crypto_chatbot.config import load_settings
from crypto_chatbot.market_data.base import DEFAULT_CANDLE_LIMIT, OhlcvBatch
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import resolve_mvp_spot_symbol
from crypto_chatbot.ml import (
    CalibratedLogisticPipeline,
    ShadowInferenceEngine,
    build_model_artifact,
    build_point_in_time_dataset,
    compute_shadow_monitoring_report,
    extract_latest_production_features,
    settle_shadow_prediction,
    split_dev_and_holdout,
)
from crypto_chatbot.schemas.market import MarketCandle, QualityStatus, Timeframe
from crypto_chatbot.schemas.ml import (
    CostAssumptions,
    MlExperimentConfig,
    MlInferenceMode,
    ShadowPredictionRecord,
)

CANDLE_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"
ANCHOR_NOW = datetime(2026, 10, 14, 16, 0, 1, tzinfo=UTC)


def _load_btc_batch() -> OhlcvBatch:
    raw = json.loads(CANDLE_FIXTURE.read_text(encoding="utf-8"))["raw_ohlcv"]
    return normalize_ohlcv_rows(
        raw,
        symbol=resolve_mvp_spot_symbol("BTC"),
        timeframe=Timeframe.H4,
        received_time=ANCHOR_NOW,
        now=ANCHOR_NOW,
    )


def _load_btc_candles() -> tuple[MarketCandle, ...]:
    return _load_btc_batch().candles


def _make_batch(candles: tuple[MarketCandle, ...]) -> OhlcvBatch:
    return _load_btc_batch().model_copy(update={"candles": candles})


class _InMemoryPredictionRepo:
    def __init__(self) -> None:
        self.records: dict[str, ShadowPredictionRecord] = {}

    def save_prediction(self, record: ShadowPredictionRecord) -> bool:
        if record.prediction_id in self.records:
            return False
        self.records[record.prediction_id] = record
        return True

    def get_prediction(self, prediction_id: str) -> ShadowPredictionRecord | None:
        return self.records.get(prediction_id)

    def list_predictions(
        self,
        *,
        canonical_symbol: str | None = None,
        timeframe: Timeframe | None = None,
        limit: int = 100,
    ) -> tuple[ShadowPredictionRecord, ...]:
        items = list(self.records.values())
        if canonical_symbol is not None:
            items = [r for r in items if r.canonical_symbol == canonical_symbol]
        if timeframe is not None:
            items = [r for r in items if r.timeframe is timeframe]
        items.sort(key=lambda r: r.prediction_time, reverse=True)
        return tuple(items[:limit])

    def list_unsettled_predictions(
        self,
        *,
        canonical_symbol: str | None = None,
        timeframe: Timeframe | None = None,
        limit: int = 100,
    ) -> tuple[ShadowPredictionRecord, ...]:
        items = [r for r in self.records.values() if r.settled_at is None]
        if canonical_symbol is not None:
            items = [r for r in items if r.canonical_symbol == canonical_symbol]
        if timeframe is not None:
            items = [r for r in items if r.timeframe is timeframe]
        return tuple(items[:limit])

    def update_realized_outcome(
        self,
        prediction_id: str,
        *,
        realized_return: float,
        realized_label: int,
        settled_at: datetime,
    ) -> bool:
        rec = self.records.get(prediction_id)
        if rec is None or rec.settled_at is not None:
            return False
        self.records[prediction_id] = rec.model_copy(
            update={
                "realized_return": realized_return,
                "realized_label": realized_label,
                "settled_at": settled_at,
            }
        )
        return True


class _StubProvider:
    def __init__(self, primary_batch: OhlcvBatch) -> None:
        self._primary_batch = primary_batch

    async def get_closed_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = DEFAULT_CANDLE_LIMIT,
    ) -> OhlcvBatch:
        _ = symbol, limit
        if timeframe is Timeframe.H4:
            return self._primary_batch
        return self._primary_batch.model_copy(update={"timeframe": timeframe, "candles": ()})

    async def close(self) -> None:
        return None


def test_production_and_training_feature_parity() -> None:
    candles = _load_btc_candles()
    config = MlExperimentConfig(canonical_symbol="BTC/USDT", timeframe=Timeframe.H4)
    rows, _ = build_point_in_time_dataset(candles, config=config)

    # Cut off the candle sequence right at `rows[-1].prediction_time` to test exact parity.
    target_row = rows[-1]
    prefix_candles = tuple(c for c in candles if c.close_time <= target_row.prediction_time)
    prod_row = extract_latest_production_features(prefix_candles, config=config)

    assert prod_row.prediction_time == target_row.prediction_time
    assert prod_row.regime == target_row.regime
    assert prod_row.features.keys() == target_row.features.keys()
    for fname, train_val in target_row.features.items():
        assert abs(prod_row.features[fname] - train_val) <= 1e-12


def test_artifact_and_shadow_engine_probability_parity() -> None:
    candles = _load_btc_candles()
    config = MlExperimentConfig(canonical_symbol="BTC/USDT", timeframe=Timeframe.H4)
    artifact, report = build_model_artifact(candles, config=config)
    assert artifact.experiment_id == report.experiment_id
    assert artifact.accepted == report.accepted

    rows, _ = build_point_in_time_dataset(candles, config=config)
    dev_rows, _ = split_dev_and_holdout(
        rows,
        holdout_fraction=config.holdout_fraction,
        embargo_bars=config.embargo_bars,
    )
    import numpy as np

    x_dev = np.array(
        [[r.features[name] for name in artifact.feature_names] for r in dev_rows],
        dtype=np.float64,
    )
    y_dev = np.array([float(r.label) for r in dev_rows], dtype=np.float64)
    pipeline = CalibratedLogisticPipeline(
        feature_names=artifact.feature_names,
        l2_reg=config.l2_reg,
    ).fit(
        x_dev,
        y_dev,
        pred_times_train=[r.prediction_time for r in dev_rows],
        label_end_times_train=[r.label_end_time for r in dev_rows],
    )

    last_row = extract_latest_production_features(candles, config=config)
    expected_prob = float(
        pipeline.predict_proba(
            np.array([[last_row.features[n] for n in artifact.feature_names]], dtype=np.float64)
        )[0]
    )

    engine = ShadowInferenceEngine(
        artifact.model_copy(update={"accepted": True}),
        mode=MlInferenceMode.PROMOTED,
        min_prob_margin=0.0,
        max_feature_zscore=100.0,
    )
    snapshot = build_analysis_snapshot(_make_batch(candles))
    record, summary = engine.predict_for_snapshot(snapshot, candles)
    assert record is not None
    assert summary is not None
    assert record.abstained is False
    assert record.calibrated_probability == pytest.approx(expected_prob, abs=1e-9)
    assert summary.calibrated_probability == pytest.approx(expected_prob, abs=1e-9)


@pytest.mark.asyncio
async def test_shadow_mode_logs_only_and_off_rollback_stays_silent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    monkeypatch.setenv("DATABASE_URL", "postgresql+psycopg://u:p@localhost/db")
    monkeypatch.setenv("ALLOWED_TELEGRAM_USER_IDS", "1")
    monkeypatch.delenv("ML_INFERENCE_MODE", raising=False)
    assert load_settings(env_file=None).ml_inference_mode is MlInferenceMode.OFF

    monkeypatch.setenv("ML_INFERENCE_MODE", "shadow")
    assert load_settings(env_file=None).ml_inference_mode is MlInferenceMode.SHADOW
    monkeypatch.setenv("ML_INFERENCE_MODE", "off")
    assert load_settings(env_file=None).ml_inference_mode is MlInferenceMode.OFF

    candles = _load_btc_candles()
    config = MlExperimentConfig(canonical_symbol="BTC/USDT", timeframe=Timeframe.H4)
    artifact, _ = build_model_artifact(candles, config=config)
    accepted_artifact = artifact.model_copy(update={"accepted": True})

    batch = _make_batch(candles)
    repo = _InMemoryPredictionRepo()
    shadow_engine = ShadowInferenceEngine(
        accepted_artifact,
        mode=MlInferenceMode.SHADOW,
        min_prob_margin=0.0,
    )
    service = AnalysisService(
        provider=_StubProvider(batch),
        shadow_engine=shadow_engine,
        prediction_repository=repo,
    )
    snapshot = await service.analyze("BTC", Timeframe.H4, include_mtf=False, persist=True)
    assert snapshot.ml_prediction is None
    assert "Estimasi Model ML" not in render_deterministic_snapshot(snapshot)
    assert "Estimasi Model ML" not in render_compact_snapshot(snapshot)

    saved = repo.list_predictions()
    assert len(saved) == 1
    assert saved[0].exposed_to_user is False
    assert saved[0].snapshot_id == snapshot.snapshot_id

    # Instant rollback to OFF leaves storage and snapshot untouched.
    repo_off = _InMemoryPredictionRepo()
    off_service = AnalysisService(
        provider=_StubProvider(batch),
        shadow_engine=ShadowInferenceEngine(accepted_artifact, mode=MlInferenceMode.OFF),
        prediction_repository=repo_off,
    )
    off_snapshot = await off_service.analyze("BTC", Timeframe.H4, include_mtf=False, persist=True)
    assert off_snapshot.ml_prediction is None
    assert repo_off.list_predictions() == ()


def test_abstention_reasons_cover_all_safety_gates() -> None:
    candles = _load_btc_candles()
    config = MlExperimentConfig(canonical_symbol="BTC/USDT", timeframe=Timeframe.H4)
    raw_artifact, _ = build_model_artifact(candles, config=config)
    snapshot = build_analysis_snapshot(_make_batch(candles))
    assert snapshot.quality_status is QualityStatus.READY
    assert snapshot.regime is not None

    # 1. Model rejected by offline validation gate.
    rec_unaccepted, _ = ShadowInferenceEngine(
        raw_artifact.model_copy(update={"accepted": False}),
        mode=MlInferenceMode.SHADOW,
    ).predict_for_snapshot(snapshot, candles)
    assert rec_unaccepted is not None
    assert rec_unaccepted.abstained is True
    assert rec_unaccepted.abstention_reason == "model_not_accepted"
    assert rec_unaccepted.calibrated_probability is None

    accepted = raw_artifact.model_copy(
        update={"accepted": True, "seen_regimes": (snapshot.regime.value,)}
    )

    # 2. Poor data quality input.
    degraded_snap = snapshot.model_copy(update={"quality_status": QualityStatus.DEGRADED})
    rec_quality, _ = ShadowInferenceEngine(
        accepted, mode=MlInferenceMode.SHADOW
    ).predict_for_snapshot(degraded_snap, candles)
    assert rec_quality is not None
    assert rec_quality.abstained is True
    assert rec_quality.abstention_reason == "poor_data_quality"

    # 3. Unseen market regime.
    rec_regime, _ = ShadowInferenceEngine(
        accepted.model_copy(update={"seen_regimes": ()}),
        mode=MlInferenceMode.SHADOW,
    ).predict_for_snapshot(snapshot, candles)
    assert rec_regime is not None
    assert rec_regime.abstained is True
    assert rec_regime.abstention_reason == "unseen_regime"

    # 4. Feature extraction failure (insufficient history passed for inference).
    rec_feat_err, _ = ShadowInferenceEngine(
        accepted, mode=MlInferenceMode.SHADOW
    ).predict_for_snapshot(snapshot, candles[:10])
    assert rec_feat_err is not None
    assert rec_feat_err.abstained is True
    assert rec_feat_err.abstention_reason == "feature_extraction_failed"

    # 5. Severe feature drift exceeding max z-score threshold.
    rec_drift, _ = ShadowInferenceEngine(
        accepted,
        mode=MlInferenceMode.SHADOW,
        max_feature_zscore=0.01,
    ).predict_for_snapshot(snapshot, candles)
    assert rec_drift is not None
    assert rec_drift.abstained is True
    assert rec_drift.abstention_reason == "feature_drift_high"
    assert rec_drift.max_feature_zscore is not None
    assert rec_drift.max_feature_zscore > 0.01

    # 6. Low calibrated probability margin near 0.50.
    rec_low_conf, _ = ShadowInferenceEngine(
        accepted,
        mode=MlInferenceMode.SHADOW,
        min_prob_margin=0.499,
        max_feature_zscore=100.0,
    ).predict_for_snapshot(snapshot, candles)
    assert rec_low_conf is not None
    assert rec_low_conf.abstained is True
    assert rec_low_conf.abstention_reason == "low_confidence"
    assert rec_low_conf.confidence_label == "abstained"


def test_promoted_template_renders_all_required_disclosures() -> None:
    candles = _load_btc_candles()
    config = MlExperimentConfig(canonical_symbol="BTC/USDT", timeframe=Timeframe.H4)
    artifact, _ = build_model_artifact(candles, config=config)
    snapshot = build_analysis_snapshot(_make_batch(candles))
    assert snapshot.regime is not None
    accepted = artifact.model_copy(
        update={"accepted": True, "seen_regimes": (snapshot.regime.value,)}
    )

    engine = ShadowInferenceEngine(
        accepted,
        mode=MlInferenceMode.PROMOTED,
        min_prob_margin=0.0,
        max_feature_zscore=100.0,
    )
    record, summary = engine.predict_for_snapshot(snapshot, candles)
    assert record is not None
    assert record.exposed_to_user is True
    assert summary is not None

    abstained_engine = ShadowInferenceEngine(
        accepted.model_copy(update={"seen_regimes": ()}),
        mode=MlInferenceMode.PROMOTED,
    )
    _, abstained_summary = abstained_engine.predict_for_snapshot(snapshot, candles)
    assert abstained_summary is not None
    assert "ABSTAIN (unseen_regime)" in render_deterministic_snapshot(
        snapshot.model_copy(update={"ml_prediction": abstained_summary})
    )

    promoted_snapshot = snapshot.model_copy(update={"ml_prediction": summary})
    for rendered in (
        render_deterministic_snapshot(promoted_snapshot, html_mode=False),
        render_deterministic_snapshot(promoted_snapshot, html_mode=True),
        render_compact_snapshot(promoted_snapshot, html_mode=False),
        render_compact_snapshot(promoted_snapshot, html_mode=True),
    ):
        assert "Estimasi Model ML" in rendered
        assert "24h" in rendered
        assert config.target_name in rendered
        assert "Probabilitas terkalibrasi:" in rendered
        assert "confidence=" in rendered
        assert "kualitas=READY" in rendered
        assert config.model_version in rendered
        assert config.feature_version in rendered
        assert "Keterbatasan model:" in rendered
        assert "bukan jaminan hasil" in rendered


def test_settlement_and_monitoring_alerts() -> None:
    candles = _load_btc_candles()
    config = MlExperimentConfig(canonical_symbol="BTC/USDT", timeframe=Timeframe.H4)
    horizon_bars = config.horizon_hours // 4
    ref_candle = candles[-horizon_bars - 1]
    costs = CostAssumptions()

    base_rec = ShadowPredictionRecord(
        prediction_id="pred-settle-1",
        snapshot_id="snap-1",
        experiment_id="exp-1",
        canonical_symbol="BTC/USDT",
        timeframe=Timeframe.H4,
        target_name=config.target_name,
        horizon_hours=config.horizon_hours,
        prediction_time=ref_candle.close_time,
        model_version=config.model_version,
        feature_version=config.feature_version,
        quality_status=QualityStatus.READY,
        regime="range",
        calibrated_probability=0.65,
        confidence_label="high",
        abstained=False,
        max_feature_zscore=1.2,
        latency_ms=4,
    )

    # Too early before target horizon candle closes -> still unsettled.
    assert settle_shadow_prediction(
        base_rec,
        candles[:-horizon_bars],
        cost_assumptions=costs,
        noise_buffer_bps=config.noise_buffer_bps,
    ) == (None, None, None)

    fwd_ret, label, settled_at = settle_shadow_prediction(
        base_rec,
        candles,
        cost_assumptions=costs,
        noise_buffer_bps=config.noise_buffer_bps,
    )
    assert fwd_ret is not None
    assert label in (0, 1)
    assert settled_at == ref_candle.close_time + timedelta(hours=config.horizon_hours)

    empty_report = compute_shadow_monitoring_report(())
    assert empty_report.is_healthy is False
    assert "no_shadow_predictions" in empty_report.alerts

    bad_records = [
        base_rec.model_copy(
            update={
                "prediction_id": f"pred-bad-{i}",
                "calibrated_probability": 0.90,
                "max_feature_zscore": 5.5,
                "realized_label": 0,
                "realized_return": -0.02,
                "settled_at": settled_at,
            }
        )
        for i in range(6)
    ]
    bad_report = compute_shadow_monitoring_report(bad_records, min_settled=5)
    assert bad_report.is_healthy is False
    assert "brier_above_threshold" in bad_report.alerts
    assert "calibration_ece_above_threshold" in bad_report.alerts
    assert "feature_drift_rate_high" in bad_report.alerts

    good_records = [
        base_rec.model_copy(
            update={
                "prediction_id": f"pred-good-{i}",
                "calibrated_probability": 0.92,
                "max_feature_zscore": 1.0,
                "realized_label": 1,
                "realized_return": 0.03,
                "settled_at": settled_at,
            }
        )
        for i in range(6)
    ]
    good_report = compute_shadow_monitoring_report(good_records, min_settled=5)
    assert good_report.is_healthy is True
    assert good_report.alerts == ()
