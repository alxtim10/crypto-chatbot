# ADR 0007: Shadow Inference, Monitoring, Abstention & Instant Rollback

- Status: Accepted
- Date: 2026-11-22
- Supersedes: none (extends ADR 0001–0006)

## Context

Phase 13 moves accepted Phase 12 offline ML artifacts into shadow operation (`PROJECT_PLAN.md` §6 Phase 13 & §8 `model_predictions`), validating feature parity, drift, calibration, and safe abstention before any user-facing release, with zero-downtime rollback to deterministic-only mode.

## Decisions

| Area | Decision | Rationale |
|---|---|---|
| Inference modes | Define `MlInferenceMode`: `off` (default, deterministic-only), `shadow` (persist predictions to `model_predictions` with `exposed_to_user=False` and `AnalysisSnapshot.ml_prediction=None`), and `promoted` (attach `ModelPredictionSummary` to `AnalysisSnapshot` and render required disclosures). | Enforces safe rollout and instant rollback by flipping `ml_inference_mode` to `off` or `shadow`. |
| Production/training feature parity | Extract one shared internal causal pipeline helper used identically by offline `build_point_in_time_dataset` and live `extract_latest_production_features`. | Guarantees zero skew between training and production feature definitions. |
| Abstention triggers | Abstain unconditionally when `not artifact.accepted`, `snapshot.quality_status != READY`, `regime` was unseen during training, `max(|z|) > 5.0` (excessive feature drift), or calibrated probability margin `|p - 0.5| < 0.05` (low confidence). Report confidence and data quality separately from calibrated probability. | Prevents silent degraded predictions or forced guesses in out-of-distribution markets. |
| Monitoring & settlement | Track every prediction (`prediction_time`, `model_version`, `feature_version`, `snapshot_id`, `calibrated_probability`, `abstention_reason`, `quality_status`, `latency_ms`) in `model_predictions`; settle realized 24h returns/labels via `settle_shadow_prediction` and compute `ShadowMonitoringReport` (realized Brier, ECE, drift rate, alert flags). | Verifies ongoing calibration and detects regime/feature drift before promotion. |
| User disclosure (when promoted) | Deterministic template renders 6 required fields alongside `ModelPredictionSummary`: prediction horizon, calibrated probability (or abstention reason), confidence + data quality separately, model + feature versions, limitations, and explicit non-guarantee. Never alters existing `Fact` IDs (`F1..Fn`). | Preserves deterministic auditability and financial-product boundaries. |

## Consequences

- Default runtime configuration (`ml_inference_mode = MlInferenceMode.OFF`) leaves Telegram & CLI analysis outputs unchanged until shadow logs are explicitly enabled and monitored.
- All shadow records persist safely in `model_predictions` without user PII or secrets.
