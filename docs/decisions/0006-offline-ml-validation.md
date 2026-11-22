# ADR 0006: Offline ML Target, Purged Walk-Forward Validation & Promotion Gate

- Status: Accepted
- Date: 2026-11-22
- Supersedes: none (extends ADR 0001–0005)

## Context

Phase 12 introduces offline machine-learning experimentation (`PROJECT_PLAN.md` §6 Phase 12, §7 ML validation tests, §8 `ml_experiments`) before any shadow inference (Phase 13) or user exposure.

## Decisions

| Area | Decision | Rationale |
|---|---|---|
| Initial target | Binary classification target `next_24h_return_exceeds_cost`: `1` iff `(close[t + h] / close[t] - 1) > (round_trip_cost_bps + noise_buffer_bps) / 10_000`, where `h = round(24h / timeframe.duration)` (6 bars on `4h`). | Focuses prediction on economically meaningful moves net of friction rather than raw noise around zero. |
| Point-in-time causality | Every `FeatureRow` satisfies `available_time <= prediction_time < label_end_time`. Technical indicators use only closed bars `0..t`; pivot and failed-breakout features filter strictly by `confirmation_time <= prediction_time`; sentiment features use `available_time <= prediction_time`. | Eliminates forward lookahead from indicators, delayed swing pivots, and news headlines. |
| Validation & embargo | Chronologically reserve the final 20% slice as an untouched holdout set (purged + embargoed against `dev`). Run expanding-window purged walk-forward CV across `dev`, excluding any training sample whose `label_end_time > val_start_time` or within `embargo_bars` of validation start. | Prevents multi-bar label overlap and autocorrelation leakage across fold and holdout boundaries. |
| Preprocessing & models | Implement pure-NumPy `NaiveBaseRateModel`, `FeaturePreprocessor` (median imputation + z-score standardization fit strictly on training fold indices), `L2LogisticRegression` (IRLS/Newton-Raphson), and inner-slice `PlattCalibrator`. Zero external ML dependencies added to `pyproject.toml`. | Guarantees deterministic reproduction, zero lookahead during scaling/calibration, and minimal dependency footprint. |
| Cost & rejection rules | Apply explicit round-trip costs (`fee_bps_per_side=10`, `spread_bps=5`, `slippage_bps=5`, `funding_bps_per_24h=2`, total `32 bps`). Simulate sequential non-overlapping trades when timestamps are supplied. Reject models (`accepted = False`) unless OOS Brier improves cleanly on naive, win rate across folds meets `min_fold_win_rate`, OOS trade count meets `min_trades`, and post-cost net returns are positive on both OOS CV and untouched holdout. | Prevents overfitting or economically negligible models from advancing to Phase 13 shadow mode. |

## Consequences

- Phase 12 remains strictly offline via `MlExperimentConfig` / `MlExperimentReport` and CLI `ml-experiment`; live Telegram `/analyze` outputs do not surface unpromoted ML probabilities.
- Experiment artifacts and rejection reasons persist cleanly in PostgreSQL table `ml_experiments`.
