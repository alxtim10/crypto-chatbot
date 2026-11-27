# ADR 0010: Meta-Labeling of Deterministic Trade Setups

- **Status**: Accepted
- **Date**: 2026-11-27
- **Supersedes / Amends**: ADR 0006 (Offline ML Validation), ADR 0007 (Shadow Inference & Release)
- **Related**: PRD v1.7.0 §6.1, Task `T-17`, Finding `F-14`

## Context

Prior offline ML experiments (ADR 0006) predicted a fixed-horizon directional binary target (`next_24h_return_exceeds_cost`) on every closed bar. That target suffered from two structural defects:

1. **Product mismatch**: The chatbot's primary output is a deterministic conditional trade setup (entry zone, SL, TP1–TP3, breakeven move, expiry) rather than a 24-hour market-order directional bet. A 24h directional probability cannot be faithfully mapped to the realized R-multiple of a limit-filled setup.
2. **Severe label overlap**: Sampling every 4h bar with a 24h forward horizon creates 6× overlapping windows, inflating effective sample size and diluting out-of-sample power.

## Decision

1. **Primary ML Target = Meta-Labeling (`meta_setup_net_positive`)**:
   - The deterministic setup engine (frozen via `params_sha256`, PRD §8.2) acts as the primary model.
   - Every `READY` setup emitted under `BACKTEST_PROFILE` is simulated by the exact §7 backtest engine (`entry_expiry_bars = 6`, `max_hold_bars = 24`, strict intrabar order, signed historical funding).
   - `NO_FILL_TP1_FIRST` and `NO_FILL_EXPIRED` setups are excluded from training labels and counted explicitly in reports.
   - Filled events receive label $y_i = \mathbb{I}(r^{\text{net}}_i > \tau_R)$ (default $\tau_R = 0$) with `prediction_time = setup_close_time` and `label_end_time = actual_exit_time`.
2. **Direction-Aligned & Setup Features**:
   - Signed base features (`ret_*`, `close_vs_ema20`, `ema20_vs_ema50`, `rsi_14_norm`, `macd_hist_pct`, `di_diff_norm`, `cvd_ratio_20`, `taker_buy_ratio - 0.5`, `sentiment_net_score`, `funding_8h`) are multiplied by setup direction $d_i \in \{+1, -1\}$.
   - Setup geometry features (`score_aligned`, `risk_atr`, `sl_pct`, `exec_cost_r`, `net_rr_tp2`, `entry_src_*`, `regime_*`, `structure_aligned`) are appended.
3. **Sample Size & Leakage Controls**:
   - Fewer than `300` filled events in the development set returns `INSUFFICIENT_DATA`.
   - Validation uses `embargo_bars = max(3, entry_expiry_bars + max_hold_bars)`, Lopez de Prado uniqueness weights, inner purged walk-forward L2 grid selection, and Promotion Gates v2 (`G1`–`G8`).
   - Model version is bumped to `meta-l2logreg-platt-v2.0.0`.
4. **Legacy Target Retention**:
   - `next_24h_return_exceeds_cost` is retained as `direction-24h-legacy` for shadow diagnostics only and is permanently **non-promotable**.
5. **Product Boundary**:
   - In `PROMOTED` mode, the meta-model displays $P(\text{setup net-positive})$ alongside the deterministic setup card. It never alters, creates, or suppresses deterministic setups in live output.

## Consequences

- ML probabilities now directly measure the historical conditional reliability of the exact setups shown to users.
- Pooling setups across multiple symbols per timeframe is required to meet the 300-event minimum.
- All promotion decisions flow through Promotion Gates v2 (`G1`–`G8`) with block-bootstrap CIs, cost stress at 1.5×, DSR, and CPCV.
