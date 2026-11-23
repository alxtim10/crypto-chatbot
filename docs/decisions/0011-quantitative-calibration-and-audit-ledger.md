# ADR 0011: Quantitative Calibration Audit, Maker/Taker Cost Split, and Audit Ledger CLI

## Status

Accepted

## Context

In financial machine learning and systematic trade setups, reporting only headline accuracy or win rates is dangerous and misleading due to base-rate shifts, favorite-longshot bias, and fee frictions. Furthermore, a personal-use research system must provide transparent, deterministic CLI inspection tools into both deterministic setup records and experimental ML model predictions without requiring ad-hoc manual SQL queries.

Prior to this decision:
- `ml/evaluation.py` and `ml/shadow.py` computed Brier score, log-loss, expected calibration error (ECE), and net return, but lacked Brier Skill Score (BSS) relative to the empirical base-rate climatology and lacked discrete calibration bin metrics across the probability spectrum.
- `CostAssumptions` lumped fee, slippage, and spread into a single flat basis-point cost without distinguishing maker execution (rebate/lower fee, adverse selection/fill uncertainty) from taker execution (higher fee, immediate crossing of spread and slippage).
- Setup lifecycle outcomes (`tp1`, `tp2`, `tp3`, `sl`, `expired`, `pending`) stored in PostgreSQL `trade_setups` lacked a dedicated summary CLI command, and shadow prediction performance lacked a consolidated audit inspection command.

## Decision

1. **Brier Skill Score (BSS) & Calibration Bins**:
   - Implement `brier_skill_score = 1.0 - (brier_score / naive_brier)` where $naive\_brier = \bar{y}(1 - \bar{y})$ representing a climatological constant prediction of the base rate. If $naive\_brier \le 10^{-9}$, $BSS = 0.0$ if $brier\_score \le 10^{-9}$ else $-1.0$.
   - Implement `CalibrationBinMetrics` with discrete probability bins (e.g. 5–10 bins between 0.0 and 1.0) tracking count, mean forecast probability, empirical positive rate, calibration gap, and mean net return to identify tail overconfidence.

2. **Maker / Taker Cost Modeling**:
   - Extend `CostAssumptions` to allow evaluating both taker execution (default: 5.0 bps fee + full spread + slippage) and maker execution (default: 2.0 bps fee + half spread + 0 slippage), computing net returns across both execution styles to expose liquidity cost drag.

3. **Audit Ledger CLI Commands**:
   - Add CLI command `crypto-chatbot setup-stats [--symbol <SYM>] [--timeframe <TF>] [--json]` to aggregate historical settled setups from PostgreSQL into objective counts, hit rates, mean/median realized R, and regime breakdowns.
   - Add CLI command `crypto-chatbot shadow-report [--symbol <SYM>] [--timeframe <TF>] [--json]` to aggregate out-of-sample shadow model predictions, computing realized Brier score, BSS, calibration bins, and alert status.
   - Both CLI commands strictly report historical measurements and historical outcomes; they never output future guarantee claims or success probabilities.

## Consequences

- **Positive**: Complete transparency into whether ML models outperform a naive base rate (BSS > 0); detailed bin-by-bin calibration inspection; realistic maker vs taker friction visibility; streamlined CLI workflow for personal auditing of setups and shadow inference.
- **Negative**: Extra computation during evaluation; slightly expanded `EvaluationSummary` and `ShadowMonitoringReport` schemas.
- **Invariants**: Strictly deterministic calculations; zero trading/order execution capability; no secrets logged; strict UTC timestamps.
