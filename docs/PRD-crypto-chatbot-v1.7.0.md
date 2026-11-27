# Product Requirements Document (PRD): Quantitative Analysis, ML & Backtesting Engine

**Product**: `crypto-chatbot` (Telegram Deterministic Crypto Market Research Assistant)
**Document Version**: `1.7.0`
**Target Versions**: `CALC_VERSION = "1.7.0"`, `FEATURE_VERSION = "ml-feat-v1.2.0"`, `MODEL_VERSION = "meta-l2logreg-platt-v2.0.0"` (legacy `l2-logreg-platt-v1.0.0` retained in shadow only)
**Supersedes**: v1.6.0. **Normative rule**: every formula, constant, and invariant in v1.6.0 remains in force unless explicitly amended, replaced, or deprecated in this document.
**Binding References**: `PROJECT_PLAN.md`, `AGENTS.md`, `docs/decisions/0001`–`0009`, new ADR `0010-meta-labeling.md` (to be written in T-17).

---

## 0. Document Control

### 0.1 Changelog v1.6.0 → v1.7.0
| Area | Change | Findings |
|---|---|---|
| Statistics | Corrected PSR, MinTRL, autocorrelation adjustment, DSR; replaced "non-normal adjusted Sharpe" with SR standard error | F-01..F-04 |
| Features | `cvd_zscore_20` → `cvd_ratio_20`; explicit missing flags; sentiment point-in-time scoring rule | F-05, F-20, F-21 |
| Setup engine | TP2/TP3 fallback monotonicity; Premium/Discount and DX zero guards; leverage ceiling includes maintenance margin | F-06, F-07, F-30, F-31 |
| ML | Target changed to **meta-labeling** of deterministic setups; embargo tied to label horizon; uniqueness weights; mean-loss L2; calibration/model mismatch fixed; base-rate-centered abstention | F-14..F-19 |
| Promotion | Gates rebuilt with minimum sample sizes, DSR, CPCV, CI-based Brier test, cost stress | F-15 |
| Backtest | EMA200 availability unified with live; entry expiry and cancellation; fixed intrabar event order; signed historical funding; cost sensitivity; block bootstrap; portfolio aggregation; live/backtest parity matrix | F-08, F-22..F-27 |
| Governance | Trial registry; parameter freeze hash; v1.6 §7.4 reclassified as "Research Results (Unconfirmed)"; forward-proof targets per timeframe | F-22, F-28 |
| Product | New Telegram UX, Numeric Output Guard, fact ledger, and non-functional requirements sections | F-35, F-36 |

### 0.2 Findings Register (source of tasks in §11)
| ID | Finding | Severity |
|---|---|---|
| F-01 | PSR denominator uses excess kurtosis `κ/4`; correct term is `(γ₄−1)/4` | Critical |
| F-02 | `√(1+2ρ₁)` undefined for ρ₁ < −0.5; clip at −0.999 does not prevent it | High |
| F-03 | DSR default `σ_trials = 1.0` is unit-mismatched (DSR ≈ 0 always); DSR not part of promotion gates | Critical |
| F-04 | MinTRL formula oversimplified; "years" unit incorrect | Medium |
| F-05 | `cvd_zscore_20` is a bounded ratio in [−1, 1], not a z-score | Low |
| F-06 | TP3 fallback (3R) can violate `TP3 ≥ TP2 + 0.5R` | Medium |
| F-07 | Premium/Discount denominator can be ≤ 0 | Medium |
| F-08 | Backtest `warmup_bars = 200` conflicts with `MIN_HISTORY_EMA_200 = 600` | High |
| F-09 | Confluence score double-counts EMA50 (already inside `TRENDING_*`) | Medium |
| F-10 | Execution-cost gate is effectively `SL% ≥ 2%` and is undocumented as such | Low |
| F-11 | Net R:R gate uses TP2 only while realized payoff is 40/40/20 | Low |
| F-14 | ML target (long-only 24h return) misaligned with product decision (bidirectional SL/TP setup) | Critical |
| F-15 | Promotion gates statistically meaningless (≥5 OOS trades, ≥1 holdout trade) | Critical |
| F-16 | `embargo_bars = 3` shorter than label horizon; no overlapping-label correction | High |
| F-17 | Abstention/confidence/monitoring thresholds centered at 0.5 instead of base rate | High |
| F-18 | Fixed λ with sum-loss means regularization vanishes as n grows | Medium |
| F-19 | Platt calibrator fit on inner model, but final model is refit | High |
| F-20 | Silent imputation (`rvol=1.0`, `TBR=0.5`, medians) violates Invariant #3 | High |
| F-21 | Sentiment may be scored retroactively by an LLM → lookahead | Critical (if applicable) |
| F-22 | Benchmark claimed as "proof" while 4h CI crosses 0 and V3 was selected on the same OOS | Critical |
| F-23 | Flat, unsigned funding cost | Medium |
| F-24 | Edge thin relative to costs; no cost sensitivity | High |
| F-25 | IID bootstrap on correlated assets; no portfolio-level drawdown | High |
| F-26 | Live-only guards (order book, whale, sentiment, LTF) not replayable; parity undocumented | High |
| F-27 | No entry expiry; no cancel-if-TP1-before-fill; intrabar order underspecified | High |
| F-28 | Forward proof of 100 setups in 30–60 days unrealistic for 1d | Medium |
| F-29 | Momentum screener has no backtest and no point-in-time universe | Medium |
| F-30 | DX undefined when +DI + −DI = 0 | Low |
| F-31 | Leverage ceiling ignores maintenance margin | Medium |
| F-32 | DXY correlation threshold 0.2 at n = 20 (5% critical value ≈ 0.44) | Medium |
| F-33 | OU half-life without stationarity test; Hurst without confidence interval | Medium |
| F-34 | DXY inner join on exact `close_time` can be empty for 1d | Medium |
| F-35 | Numeric Output Guard and `{F<n>}` ledger have no specification | High |
| F-36 | No product requirements (Telegram UX, metrics, NFRs) | Medium |
| F-37 | "3× longest period" heading does not match the warm-up table | Low |

---

## 1. Product Overview

### 1.1 Vision (unchanged in principle)
`crypto-chatbot` is a Telegram research and educational assistant for crypto perpetuals and spot. Deterministic Python computes 100% of facts. The LLM (Gemini) is a tool-bound narrator that may reference numbers **only** through `{F<n>}` placeholders resolved server-side (§2.3).

### 1.2 Invariants
The five invariants of v1.6 §1.2 remain. Added:

6. **Parity Disclosure**: every live output component is tagged `BACKTESTED`, `FORWARD_ONLY`, or `CONTEXT_ONLY` (§7.6). Only `BACKTESTED` components may be cited with historical performance.
7. **No Unregistered Trials**: every backtest variant or ML experiment run on historical data is recorded in the Trial Registry (§8.2). DSR and related statistics count all registered trials.
8. **Claims Discipline**: no user-facing or documentation claim of "proven", "verified", or "validated" edge unless the Forward Proof Standard (§8.3) is met.

### 1.3 Users & Goals `[CONFIRM]`
- **Primary user**: retail or semi-professional crypto trader on Telegram who wants an explainable, honest analysis per symbol and timeframe.
- **Proposed success metrics**:
  - Numeric guard violations reaching users: 0.
  - p95 latency of `/analyze`: ≤ 10 s.
  - Share of WAIT/INSUFFICIENT responses with an explicit reason: 100%.
  - Forward-ledger settlement coverage of emitted setups: ≥ 99%.

### 1.4 Non-Goals
Automated execution, portfolio allocation, LLM-generated targets, personalized advice, guaranteed-probability language.

---

## 2. Telegram Product Requirements `[CONFIRM command names against current bot]`

### 2.1 Commands
| Command | Behavior |
|---|---|
| `/analyze <SYMBOL> [TF=4h]` | Deterministic analysis + setup (READY/WAIT) + ML panel (if PROMOTED), narrated |
| `/setup <SYMBOL> [TF]` | Setup card only: entry zone, SL, TP1–3, Net R:R, leverage ceiling, parity tags |
| `/screener [long\|short]` | 15m momentum screener; results carry an `UNVALIDATED` badge until §7.8 is met |
| `/track` | Forward-ledger summary: settled count, expectancy with block-bootstrap CI, per timeframe |
| `/help`, `/disclaimer` | Static content |

### 2.2 Response States (deterministic templates)
| State | User sees |
|---|---|
| `READY` | Full setup card + narration + parity tags + disclaimer |
| `WAIT` | Explicit reason code mapped to plain language (e.g. `WAIT_EMA200_GATE` → "trend filter not aligned") |
| `DEGRADED` | Analysis with warning banner listing degrading flags; no setup if any setup input is degraded |
| `INSUFFICIENT_DATA` | No analysis; reason + required history length |
| ML abstained | "ML panel: abstained (<reason>)" — never silently hidden when mode is `PROMOTED` |

### 2.3 Numeric Output Guard & Fact Ledger (new, normative)
- **AnalysisSnapshot** is an immutable ledger of facts. Each fact has: `F<n>` id, `name`, `value`, `unit`, `precision`, `source_module`, `calc_version`, `as_of`, `status ∈ {OK, DEGRADED, NONE}`.
- **LLM input**: fact ids with names, units, and qualitative labels. **LLM output**: prose containing `{F<n>}` placeholders only.
- **Guard pipeline**, applied after generation and before rendering:
  1. Parse placeholders. An unknown id, or an id whose fact has status `NONE`, is a violation.
  2. Strip placeholders. Any remaining digit `[0-9]` is a violation, except tokens in a server-built allowlist (timeframe tokens from the snapshot such as `4h`, symbol tickers such as `1000PEPE`).
  3. **Directional consistency** (P2): bias words (`bullish/bearish/long/short`) that contradict `snapshot.bias` are a violation.
  4. On violation: retry once with the violation list as feedback. On a second violation, fall back to deterministic template narration.
  5. Render: the server substitutes formatted values. Every violation is logged with `violation_type` and `model_version`.

### 2.4 Non-Functional Requirements `[CONFIRM targets]`
- Cache OHLCV keyed by `(venue, symbol, tf, close_time)`. Provider backoff with jitter. Per-user rate limit (proposed: 10 requests/min).
- Scheduled jobs:
  - `settle-setups` hourly.
  - Shadow ML logging at each closed bar per tracked symbol.
  - Daily screener universe snapshot (§7.8).
- All timestamps UTC. Structured JSON logs. Versioned PostgreSQL migrations.

---

## 3. Data Integrity, Temporal Causality & Quality Gating (amends v1.6 §2)

### 3.1 Warm-Up Table
Rename the heading "Minimum Warm-Up History Requirements (`3x` Longest Period Rule)" to **"Minimum Warm-Up History Requirements (per-indicator constants)"** (F-37). Table values are unchanged.

### 3.2 EMA200 Availability (F-08)
A single predicate, used identically in the live engine and the backtester:

$$\texttt{ema200\_available}(t) \iff \text{closed\_count}(t) \ge \texttt{MIN\_HISTORY\_EMA\_200} = 600$$

The V3 EMA200 gate applies iff `ema200_available`. If it is not available, the setup returns `WAIT_EMA200_UNAVAILABLE`. Silently bypassing the gate is no longer allowed.

### 3.3 As-Of Joins for Macro Series (F-34)
Replace the exact `close_time` inner join with a **backward as-of join**. For each crypto bar *t*, take the latest DXY observation with `available_time ≤ close_time_t` and staleness `≤ max_staleness`:
- intraday: `max_staleness = 2·Δt`
- 1d: `max_staleness = 72 h` (covers weekends)

Rows beyond the staleness limit are excluded from correlation and beta, and flagged `MACRO_STALE`.

### 3.4 Funding History
`get_funding_history(symbol, start, end)` returns settlement timestamps and rates, point-in-time. If unavailable, the backtest falls back to the v1.6 flat 2 bps/24h and flags the trade `FUNDING_ASSUMED`.

### 3.5 Data Availability Audit (F-20)
Record per venue whether historical taker-buy volume is available. The CCXT unified OHLCV is `[t, o, h, l, c, v]` only, so availability must be verified per venue. If it is unavailable, the ORDERFLOW feature family is `UNAVAILABLE` for that venue and is excluded from training, not imputed.

---

## 4. Deterministic Analysis Amendments (amends v1.6 §3–§4)

### 4.1 DX Zero Guard (F-30)
If $+\text{DI}_t + -\text{DI}_t = 0$ then $\text{DX}_t = 0$.

### 4.2 Premium/Discount Guard (F-07)
Let $D = SH_{\text{latest}} - SL_{\text{latest}}$.
- If $D \le 0.1\,\text{ATR}_{14}$: Zone = `NONE`, flag `RANGE_INVALID`.
- Otherwise compute $P_{\text{pos}}$ as in v1.6, and add two zones:
  - `ABOVE_RANGE` if $P_{\text{pos}} > 1$
  - `BELOW_RANGE` if $P_{\text{pos}} < 0$

### 4.3 Statistical Reliability Gates (F-32, F-33)
- **Correlation significance**: report `DXY Impact` other than `neutral` only if $|r| \ge r_{\text{crit}}(n, 0.05)$, where
$$r_{\text{crit}} = \frac{t_{0.975,\,n-2}}{\sqrt{n - 2 + t_{0.975,\,n-2}^2}}$$
  This gives ≈ 0.444 at n = 20. An optional window `DXY_CORR_WINDOW = 60` gives ≈ 0.254.
- **OU half-life**: report it only if the ADF t-statistic of β (constant, no trend, lag 0) is below the 5% MacKinnon critical value for n (≈ −2.89 at n = 120). Otherwise return `None` with flag `NOT_STATIONARY`.
- **Hurst**: also report the OLS slope standard error $SE_H$. Flag `HURST_UNRELIABLE` if $2 \cdot 1.96\, SE_H > 0.30$.

---

## 5. Setup Engine Amendments (amends v1.6 §5)

### 5.1 Target Monotonicity (F-06)
All distances are in R from $E_{\text{mid}}$:
- **TP2 fallback** $= \max(2.0R,\; d_{TP1} + 0.3R)$
- **TP3 fallback** $= \min\!\big(5.0R,\; \max(3.0R,\; d_{TP2} + 0.5R)\big)$
- **Invariant**: $d_{TP1} < d_{TP2} < d_{TP3}$. If violated, return `WAIT_TARGET_INVALID`.

### 5.2 Cost Gate Documentation (F-10)
`MAX_EXEC_COST_R = 0.15` is equivalent to a minimum stop distance:
$$\text{SL\%} \ge \frac{c_{\text{exec}}}{0.15} = 2.0\% \;(\text{taker default})$$
Expose this as the derived constant `MIN_SL_PCT_IMPLIED` in setup output and documentation. No behavior change.

### 5.3 Blended Payoff Report (F-11, report-only)
$$\text{NetRR}_{\text{blend}} = 0.4\,\text{NetRR}(TP_1) + 0.4\,\text{NetRR}(TP_2) + 0.2\,\text{NetRR}(TP_3)$$
Shown on the setup card. **Not a gate** in v1.7.

### 5.4 Leverage Ceiling with Maintenance Margin (F-31)
MMR is the maintenance margin rate for the position tier, taken from venue metadata (fallback 1%, flag `MMR_ASSUMED`):
$$\text{Lev}_{\max} = \min\!\left(20,\; \max\!\left(1,\; \left\lfloor \frac{1}{\frac{\text{SL\%} + c_{\text{rt}}}{0.75} + \text{MMR}} \right\rfloor\right)\right)$$
With MMR = 0 this reduces to v1.6.

### 5.5 Score Double-Count — V4 Candidate (F-09, experiment only)
Variant `V4`: the $\pm1$ term for $C \gtrless \text{EMA}_{50}$ counts only when the regime is not in {`TRENDING_UP`, `TRENDING_DOWN`}.
- V4 ships **only** if it passes §8 against frozen V3 under the corrected engine.
- V4 must be registered as a trial.
- Default remains V3.

### 5.6 Pending Setup Lifecycle (F-27)
- `entry_expiry_bars` (default 6, configurable) is separate from `max_hold_bars`. The hold clock starts at fill.
- **Cancellation before fill**:
  - TP1 touched before fill → `NO_FILL_TP1_FIRST`
  - expiry reached → `NO_FILL_EXPIRED`
- These rules apply identically to live ledger settlement and to the backtest.
- Legacy mode `entry_expiry_bars = None` reproduces v1.6 behavior and is retained for regression comparison only.

---

## 6. Machine Learning (replaces v1.6 §6.1, amends §6.2–§6.7)

### 6.1 Target: Meta-Labeling of Deterministic Setups (F-14) — ADR 0010
- **Primary model**: the deterministic setup engine, frozen by parameter hash (§8.2).
- **Event set**: every `READY` setup emitted under `BACKTEST_PROFILE` (§7.6), pooled across symbols. One model per timeframe.
- **Outcome**: simulated by the exact §7 engine.
  - `NO_FILL_*` events are excluded from training labels but counted in reports.
  - Filled events produce $r^{\text{net}}_i$, the R-multiple at 1.0× costs.
- **Label**: $y_i = \mathbb{I}(r^{\text{net}}_i > \tau_R)$, default $\tau_R = 0$.
- **Times**:
  - `prediction_time` = setup emission `close_time`
  - `label_end_time` = actual exit time
  - Features require `available_time ≤ prediction_time`.
- **Direction-agnostic features**: each signed feature $x$ becomes $x \cdot d_i$, with $d_i = +1$ for LONG and $-1$ for SHORT. Applies to `ret_*`, `close_vs_ema20`, `ema20_vs_ema50`, `rsi_14_norm`, `macd_hist_pct`, `di_diff_norm`, `cvd_ratio_20`, `taker_buy_ratio−0.5`, `sentiment_net_score`, and funding.
- **New setup features**:

| Feature | Definition |
|---|---|
| `score_aligned` | $S \cdot d / 9$ |
| `risk_atr` | $R / \text{ATR}_{14}$ |
| `sl_pct` | $R / E_{\text{mid}}$ |
| `exec_cost_r` | as in v1.6 §5.1 C |
| `net_rr_tp2` | as in v1.6 §5.1 C |
| `entry_src_{fvg,ob,ema20,atr}` | one-hot of entry zone source |
| `regime_*` | one-hot of 5-state regime |
| `structure_aligned` | 1 if market structure agrees with direction |
| `funding_8h_aligned` | $r_{8h} \cdot d$ (point-in-time history only) |

- **Minimum data**: fewer than 300 filled events in the dev set → the experiment returns `INSUFFICIENT_DATA`.
- **Legacy target** `next_24h_return_exceeds_cost`: retained as experiment `direction-24h-legacy`, shadow only, **not promotable**.
- **Product usage**: in `PROMOTED` mode, show $P(\text{setup net-positive})$ next to the deterministic setup. ML never alters, creates, or suppresses a setup in live output. Filtering by ML is allowed only in research backtests.

### 6.2 Feature Set `ml-feat-v1.2.0` (F-05, F-20, F-21)
- **Rename** `cvd_zscore_20` → `cvd_ratio_20`:
$$\text{cvd\_ratio\_20}_t = \frac{\sum_{j=t-19}^{t}(2V^{\text{buy}}_j - V_j)}{\sum_{j=t-19}^{t} V_j} \in [-1, 1]$$
  No clip.
- **Add** `cvd_z_100`:
$$\text{cvd\_z\_100}_t = \operatorname{clip}_{[-3,3]}\big((\text{cvd\_ratio\_20}_t - \mu_{100}) / \sigma_{100}\big)$$
  computed over the trailing 100 values of `cvd_ratio_20` ending at *t*.
- **Missing flags**: `rvol_missing`, `orderflow_missing`, `sentiment_missing` ∈ {0, 1}.
  - Missing values are imputed with the training-fold median, and the flag is set.
  - The fixed constants (`1.0`, `0.5`) are removed.
- **Family auto-exclusion**: if a family's missing rate in the training fold exceeds 50%, the family is dropped for that experiment and the drop is logged.
- **Sentiment point-in-time rule**:
  - Each score stores `scored_at`, `scorer_id`, `scorer_version`.
  - A score is eligible iff `article.available_time ≤ prediction_time` **and** `scored_at ≤ prediction_time`.
  - Exception: scorers declared `scorer_class = "lexicon_deterministic"` (no world knowledge) may backfill, with flag `SENTIMENT_BACKFILLED`.
  - LLM-based scores are never backfilled.

### 6.3 Validation (amends v1.6 §6.3) (F-16)
- **Embargo**: `embargo_bars = max(3, H_label)`.
  - Meta-labels: `H_label = entry_expiry_bars + max_hold_bars`.
  - Legacy target: `H_label = H`.
- **Uniqueness weights**: let $c_t$ be the number of events whose interval $(\text{pred}_i, \text{end}_i]$ contains bar *t*:
$$u_i = \frac{1}{|T_i|}\sum_{t \in T_i} \frac{1}{c_t}, \qquad w_i = \frac{u_i}{\bar{u}}$$
  The weights are passed to model fitting and to Brier/ECE/BSS computation.
- **Minimum sizes**:
  - holdout ≥ max(20%, 60 filled events)
  - Platt calibration slice ≥ max(20% of fold, 50 events)
  - Below these, return `INSUFFICIENT_DATA`.

### 6.4 Anti-Overfitting Statistics — Corrected (replaces v1.6 §6.4 items 2–6) (F-01..F-04)
Notation: $\widehat{SR}$ = per-observation Sharpe, $n$ = number of observations, $\gamma_3$ = skewness, $\gamma_4$ = **raw** kurtosis (= 3 under normality).

1. **SR standard error** (replaces "Non-Normal Adjusted Sharpe"; field `sr_nn` deprecated):
$$\widehat{\sigma}_{SR} = \sqrt{\frac{1 - \gamma_3 \widehat{SR} + \frac{\gamma_4 - 1}{4}\widehat{SR}^2}{n - 1}}$$
2. **Probabilistic Sharpe Ratio**:
$$\text{PSR}(SR^*) = \Phi\!\left(\frac{(\widehat{SR} - SR^*)\sqrt{n - 1}}{\sqrt{1 - \gamma_3 \widehat{SR} + \frac{\gamma_4 - 1}{4}\widehat{SR}^2}}\right)$$
3. **Autocorrelation adjustment** (Bartlett long-run variance, default `L = 1`):
$$\eta = 1 + 2\sum_{k=1}^{L}\left(1 - \frac{k}{L+1}\right)\hat{\rho}_k, \qquad SR_{\text{AC}} = \begin{cases} \widehat{SR}/\sqrt{\eta} & \eta > 0 \\ \texttt{None} & \eta \le 0 \;(\text{flag } \texttt{AC\_UNDEFINED}) \end{cases}$$
4. **Deflated Sharpe Ratio**:
   - $M$ = number of trials in the Trial Registry for the experiment family.
   - $\sigma_{\text{trials}}$ = sample std of the registered trials' $\widehat{SR}$ at the **same observation frequency**. Mixed frequencies are rejected.
   - If $M = 1$: $E[\max SR_0] = 0$, so DSR = PSR(0).
   - Otherwise use the v1.6 $E[\max SR_0]$ formula.
   - The constant default `σ_trials = 1.0` is removed.
5. **Minimum Track Record Length**, in observations (returns `None` if $\widehat{SR} \le SR^*$):
$$n^* = 1 + \left[1 - \gamma_3 \widehat{SR} + \frac{\gamma_4 - 1}{4}\widehat{SR}^2\right]\left(\frac{z_{1-\alpha}}{\widehat{SR} - SR^*}\right)^2, \qquad \text{years} = \frac{n^*}{\text{obs\_per\_year}}$$
6. DoF-adjusted SR and the CPCV profitable path fraction are unchanged.

### 6.5 Model & Calibration (amends v1.6 §6.5) (F-18, F-19)
- **Loss** (mean, weighted):
$$\mathcal{L}(\theta) = -\frac{1}{\sum w_i}\sum_i w_i\left[y_i \ln p_i + (1 - y_i)\ln(1 - p_i)\right] + \frac{\lambda}{2}\|\theta_{1:d}\|^2$$
- **λ selection**: λ is chosen from `{0.001, 0.01, 0.1, 1.0}` by purged inner walk-forward on the training fold only (weighted Brier). Every grid point counts toward the trial count.
- **Calibration**: the **final predictor is the inner model** (trained on the purged inner-train slice) composed with the Platt calibrator fit on its own out-of-sample calibration logits. There is **no refit** after calibration. The artifact stores both, plus the training base rate $b$.

### 6.6 Evaluation & Promotion Gates v2 (replaces v1.6 §6.6 gates) (F-15)
- **Trade rule in evaluation**: a trade is taken iff $p_i \ge b + \delta$, with default $\delta = 0.05$. The fixed `0.55` threshold is removed.
- For the meta-model, the "selected set" is all events with $p_i \ge b + \delta$.

The model is **REJECTED** if any gate fails:

| # | Gate | Default |
|---|---|---|
| G1 | OOS block-bootstrap 95% CI lower bound of $(BS_{\text{naive}} - BS_{\text{model}})$ > 0 **and** OOS BSS ≥ 0.02 | — |
| G2 | Walk-forward folds where BSS > 0 **and** selected expectancy > 0 | ≥ 3 of 4 |
| G3 | OOS selected trades | ≥ 50 |
| G4 | OOS selected expectancy > all-setup expectancy (lift > 0) **and** > 0 at 1.0× costs **and** ≥ 0 at 1.5× costs | — |
| G5 | Holdout: selected trades ≥ 30, BSS > 0, selected expectancy > 0 | — |
| G6 | DSR on the selected per-trade R series | ≥ 0.95 |
| G7 | CPCV profitable path fraction | ≥ 0.60 |
| G8 | Holdout weighted 5-bin ECE | ≤ 0.10 |

All thresholds are config values recorded in `config_sha256`.

### 6.7 Shadow, Abstention & Monitoring (amends v1.6 §6.7) (F-17)
- **Edge**: $e = p_{\text{cal}} - b$.
  - `low_confidence` if $|e| < 0.05$.
  - Labels: `high` if $|e| \ge 0.15$; `moderate` if $|e| \ge 0.08$; otherwise `low`.
- **Monitoring** on settled shadow logs:
  - `insufficient_settled_samples` if settled < 30.
  - `bss_below_zero` if live $BSS = 1 - BS_{\text{model}}/BS_{\text{naive}(b)} < 0$ with settled ≥ 30.
  - `calibration_ece_above_threshold` if ECE > 0.12 with settled ≥ 50.
  - Drift rule unchanged.
- The fixed `Brier > 0.25` rule is removed.

---

## 7. Backtesting Engine Amendments (amends v1.6 §7)

### 7.1 Defaults
- `warmup_bars = 600` (§3.2).
- Every report shows `eval_start`, `eval_end`, and `eval_bars` explicitly.
- `entry_expiry_bars = 6`.

### 7.2 Intrabar Event Order (normative) (F-27)
On each bar, in this order:
1. **If the setup is pending**:
   - Check cancel-if-TP1-touched (bar high for LONG, bar low for SHORT).
   - Then check fill (v1.6 fill rule).
   - If TP1 and the fill zone are both touched on the same bar, the order is unknowable. Conservative assumption: the fill happened first, and TP1 is **not** credited on that bar.
2. **If filled**: check `current_sl` **as of bar open**, with the gap-through rule. SL is always checked before TP.
3. Then TP1 → TP2 → TP3, sequentially. Multiple TPs on one bar are allowed.
4. The breakeven move after TP1 takes effect **from the next bar**.
5. Expiry is handled as in v1.6 (mark-to-market at close).

Note: breakeven is gross, so a breakeven exit is net-negative by cost_R. Reports must show the "BE exits" count.

### 7.3 Signed Historical Funding (F-23)
For each funding settlement $s$ with $t_{\text{fill}} \le s < t_{\text{exit}}$, rate $f_s$, direction $d$, and remaining position fraction $q_s$:
$$\text{funding}_R = \sum_s \frac{-\,d \cdot f_s \cdot q_s \cdot P_{\text{fill}}}{R}$$
- Longs pay positive funding.
- The result is added to $r_{\text{net}}$.
- If funding history is missing, use the v1.6 flat assumption and flag `FUNDING_ASSUMED`.
- The report shows the fraction of trades using assumed funding.

### 7.4 Cost Sensitivity (F-24)
Every backtest report includes, for cost multipliers `{1.0, 1.25, 1.5, 2.0}` applied to $c_{\text{exec}}$:
- expectancy
- total net R
- profit factor
- max drawdown
- the break-even cost multiplier (the multiplier at which expectancy = 0)

### 7.5 Bootstrap & Portfolio (F-25)
- **Portfolio series**: bucket all trades by exit date (UTC day); daily R = sum of trade R. Report portfolio max drawdown and max concurrent positions.
- **Stationary block bootstrap** (Politis–Romano) on the daily series: $B = 2{,}000$, mean block length 5 days, `seed = 42`. Report:
  - 95% CI of mean daily R
  - 95% CI of per-trade expectancy (resample days, take the trades in them)
  - P95 max drawdown
- The IID trade bootstrap is retained as a secondary diagnostic only.

### 7.6 Live/Backtest Parity Matrix (F-26)
| Component | Tag | In `BACKTEST_PROFILE` |
|---|---|---|
| Indicators, regimes, structure, SMC, failed breakout | `BACKTESTED` | Yes |
| EMA200 gate, cost gate, Net R:R gate | `BACKTESTED` | Yes |
| Funding (historical) | `BACKTESTED` | Yes |
| 15m LTF refinement / HTF alignment | `BACKTESTED` iff 15m history is replayed point-in-time `[CONFIRM]` | Config |
| L2 order book imbalance guard | `FORWARD_ONLY` | No |
| Whale positioning, CVD from live trades | `FORWARD_ONLY` | No |
| Sentiment (LLM-scored) | `FORWARD_ONLY` | No |
| DXY / macro context | `CONTEXT_ONLY` (not a gate) | No |

- The live engine records in the ledger which `FORWARD_ONLY` guards fired on each setup.
- The forward ledger reports performance both **with and without** the setups that `FORWARD_ONLY` guards altered.

### 7.7 Research Results (reclassifies v1.6 §7.4) (F-22)
- Rename the v1.6 §7.4 table to **"Research Results (Unconfirmed, pre-v1.7 engine)"**, with these notes:
  1. V3 was selected after observing the OOS period, so the OOS is contaminated.
  2. The 4h CI includes 0.
  3. EMA200 availability was inconsistent (F-08).
- Regenerate the table under the v1.7 engine (T-21), including DSR with registered M, cost sensitivity, block-bootstrap CI, and portfolio drawdown.
- **Language rule**: the words "proof", "verified", and "deployed winner" are not allowed in documentation or user output until §8.3 passes.

### 7.8 Momentum Screener Validation (F-29)
- Persist a daily point-in-time universe snapshot: top gainers/losers, 24h quote volume, listing status.
- The screener stays `UNVALIDATED` until both:
  - ≥ 180 days of snapshots exist, and
  - it passes a §7 backtest on that universe, reported per §7.4/§7.5.

---

## 8. Governance & Forward Validation (amends v1.6 §8)

### 8.1 Acceptance Gates
The v1.6 static and test gates remain. Added:
- Property tests for the new invariants (TP monotonicity, η guard, PSR normal-case identity).
- Golden-value tests for the corrected statistics.
- A regression test showing legacy mode reproduces v1.6 backtest outputs bit-for-bit, except for the intended fixes.

### 8.2 Trial Registry & Parameter Freeze
- **Trial Registry** (PostgreSQL table `trials`): `trial_id`, `family`, `variant_params_json`, `params_sha256`, `data_window`, `symbols`, `timeframe`, `obs_frequency`, `sr_per_obs`, `n_obs`, `created_at`, `git_sha`. Every backtest or ML run inserts one row automatically; this is not optional.
- **Freeze**: `crypto-chatbot freeze-params <variant>` writes `frozen_params.json` with `params_sha256` and `frozen_at`.
  - Live setup emission records `params_sha256`.
  - Only data after `frozen_at` counts as true out-of-sample.

### 8.3 Forward Proof Standard (replaces the v1.6 §8.1 bullet) (F-28)
| Timeframe | Minimum settled filled setups | Minimum duration | Statistical condition |
|---|---|---|---|
| 4h | 100 | 60 days | Block-bootstrap 95% CI lower bound of expectancy > 0 at 1.0× costs **and** expectancy ≥ 0 at 1.5× |
| 1d | 40 | 180 days | Same |
| 15m / screener | 200 | 30 days | Same |

Only setups emitted after `frozen_at` with a matching `params_sha256` count.

---

## 9. Open Questions `[CONFIRM]`
1. How is sentiment scored today (LLM or lexicon)? Has historical sentiment been backfilled?
2. Is the live venue the same as the backtest venue (BingX)? Is historical taker-buy volume available there?
3. Which guards were active when the v1.6 §7.4 numbers were produced (order book guard, 15m LTF)?
4. What are the current Telegram command names and response formats?
5. What are the target p95 latency and the expected number of concurrent users?

---

## 10. Definition of Done (applies to every task)
- `ruff check .`, `ruff format --check .`, `mypy src tests`, and `pytest` all pass.
- New or changed formulas have golden-value unit tests. The reference value is computed independently inside the test.
- Any change that alters setup output or backtest results sits behind a config flag defaulting to legacy behavior **until** T-21 regenerates the benchmark.
- Version constants are bumped only in the task that introduces the change.
- One PR per task. `CHANGELOG.md` entry references the task ID and finding ID. Docs are updated if the change is user-visible.

---

## 11. Mitigation Task Backlog (for agent executor)

### 11.0 Execution Rules for the Agent
- Work on one task per branch: `task/T-XX-short-slug`. Open one PR per task. Do not merge tasks together.
- Respect `Depends on`. Do not start a task until its dependencies are merged.
- Do not change default trading behavior (setup output, backtest results) outside a config flag, unless the task explicitly says so.
- If a task touches a `[CONFIRM]` item, stop and report the question instead of guessing.
- In each PR description, list: findings addressed, files changed, tests added, and before/after numbers where relevant.

Each task below lists **ID · Priority · Depends on · Findings**, followed by Files, Change, and Acceptance.

### Phase P0-A — Statistical correctness (no trading-logic change)

**T-01 · P0 · — · F-01**
- **Files**: `src/crypto_chatbot/ml/statistics.py`, `tests/ml/test_statistics.py`
- **Change**: PSR denominator becomes `1 − γ₃·SR + (γ₄−1)/4·SR²`, with γ₄ = raw kurtosis. Audit every kurtosis call site for raw vs. excess (scipy/numpy conventions differ).
- **Acceptance**:
  - For γ₃=0, γ₄=3, SR=0.1, n=1000, SR*=0: PSR = Φ(0.1·√999 / √1.005) within 1e-9.
  - The `kurtosis()` helper returns 3.0 ± 0.05 on 1e6 seeded normal samples.

**T-02 · P0 · T-01 · F-01**
- **Files**: same as T-01, plus report schemas.
- **Change**: replace `sr_nn` with `sr_std_error` (§6.4.1). Keep `sr_nn` in the schema as deprecated (`None`) for one minor version.
- **Acceptance**: normal case gives `sr_std_error ≈ √((1+SR²/2)/(n−1))`. Schema tests are updated.

**T-03 · P0 · — · F-02**
- **Files**: `ml/statistics.py`, `analysis/backtest.py` (per-trade Sharpe).
- **Change**: Bartlett η (§6.4.3), with configurable `L` (default 1). Return `None` and flag `AC_UNDEFINED` when η ≤ 0.
- **Acceptance**:
  - A synthetic series with ρ₁ ≈ −0.6 returns `None`, not NaN and not an exception.
  - ρ₁ = 0 returns the unadjusted SR.

**T-04 · P0 · — · F-03 (governance)**
- **Files**: new `src/crypto_chatbot/ml/trials.py`, a migration for the `trials` table, hooks in the `backtest` and `run-ml-experiment` CLIs.
- **Change**: Trial Registry (§8.2) with automatic insert on every run.
- **Acceptance**:
  - Running `backtest` twice inserts 2 rows with identical `params_sha256` and distinct `trial_id`.
  - `obs_frequency` is mandatory.

**T-05 · P0 · T-01, T-04 · F-03**
- **Files**: `ml/statistics.py`
- **Change**: DSR uses M and σ_trials from the registry, same frequency only. M=1 → DSR = PSR(0). Remove the constant default 1.0. Mixed frequencies raise `ValueError`.
- **Acceptance**:
  - M=1 gives DSR == PSR(0).
  - Golden test: M=100, σ_trials=0.05, SR=0.1, n=1000, normal moments; reference value computed in the test from the formula.
  - Mixed frequencies raise.

**T-06 · P0 · T-01 · F-04**
- **Files**: `ml/statistics.py`
- **Change**: MinTRL per §6.4.5, returning `n_obs` and `years` (given `obs_per_year`), or `None` if SR ≤ SR*.
- **Acceptance**: golden test against the formula; the SR ≤ SR* case returns `None`.

**T-07 · P0 · — · F-19**
- **Files**: `ml/models.py`, `ml/experiment.py`, artifact schema.
- **Change**: final predictor = inner model + calibrator; remove the refit. The artifact stores `base_rate`. Bump the `MODEL_VERSION` patch for the legacy model.
- **Acceptance**: a test asserts that the calibrator's inputs equal the stored model's logits on the calibration slice (exact equality).

### Phase P0-B — Spec bugs in the deterministic engine (behind flags where outputs change)

**T-08 · P0 · — · F-06**
- **Files**: `analysis/setup.py`
- **Change**: TP fallbacks per §5.1, plus the monotonicity invariant → `WAIT_TARGET_INVALID`.
- **Acceptance**: Hypothesis property test: for random candidate sets, emitted TPs are strictly increasing in distance, or the result is a WAIT.

**T-09 · P0 · — · F-07, F-30**
- **Files**: `analysis/smc.py` (dealing range), `analysis/indicators.py` (DX)
- **Change**: guards per §4.1 and §4.2.
- **Acceptance**:
  - D ≤ 0.1·ATR → `NONE` plus flag.
  - Above-range and below-range zones are emitted correctly.
  - +DI = −DI = 0 → DX = 0, with no warning and no NaN.

**T-10 · P0 · — · F-08**
- **Files**: `analysis/setup.py`, `analysis/backtest.py`, config.
- **Change**: single `ema200_available` predicate (§3.2); `WAIT_EMA200_UNAVAILABLE`; backtest default `warmup_bars = 600`; reports print the evaluation window.
- **Acceptance**:
  - With 300 closed bars, both live and backtest return `WAIT_EMA200_UNAVAILABLE`.
  - The report contains `eval_start/eval_end/eval_bars`.

**T-11 · P0 · — · F-37, F-10**
- **Files**: docs, `analysis/setup.py` output.
- **Change**: rename the warm-up heading; add derived `MIN_SL_PCT_IMPLIED` to setup output.
- **Acceptance**: the field equals `c_exec / MAX_EXEC_COST_R`; docs are updated.

### Phase P0-C — Governance gates

**T-12 · P0 · T-04, T-05, T-19 · F-15**
- **Files**: `ml/experiment.py` (`decide_promotion`), config.
- **Change**: gates G1–G8 (§6.6) with config thresholds. G1 uses the block bootstrap from T-19. Each gate result is persisted.
- **Acceptance**:
  - Synthetic fixtures in which exactly one gate fails → `REJECTED`, naming that gate.
  - The all-pass fixture → `ACCEPTED`.

**T-13 · P0 · T-07 · F-17**
- **Files**: `ml/shadow.py`, `ml/evaluation.py`
- **Change**: base-rate-centered edge, thresholds, and labels; trade rule `p ≥ b+δ`; monitoring per §6.7 (BSS vs. naive, settled ≥ 30/50).
- **Acceptance**:
  - b = 0.40, p = 0.50 → label `moderate`, not abstained.
  - A monitoring fixture with BSS < 0 raises `bss_below_zero`.

### Phase P1 — Methodology

**T-14 · P1 · — · F-16**
- **Files**: `ml/validation.py`, `ml/models.py`, `ml/evaluation.py`
- **Change**: `embargo_bars = max(3, H_label)`; uniqueness weights (§6.3) passed to IRLS and to the metrics; minimum sizes → `INSUFFICIENT_DATA`.
- **Acceptance**:
  - `assert_no_fold_leakage` additionally checks embargo ≥ H_label.
  - Weight test: two fully overlapping events each get w = 0.5 relative to an isolated event (before normalization).

**T-15 · P1 · T-14 · F-18**
- **Files**: `ml/models.py`, `ml/experiment.py`
- **Change**: weighted mean loss; λ selected by inner grid; each grid point registered as a trial (T-04).
- **Acceptance**: duplicating the same data 2× yields the same θ within 1e-6 (mean-loss invariance).

**T-16 · P1 · — · F-20, F-05**
- **Files**: `ml/features.py`, `market_data/ccxt_provider.py`, new `scripts/audit_data_availability.py`
- **Change**:
  - Feature set v1.2: `cvd_ratio_20`, `cvd_z_100`, missing flags, in-fold median imputation, family auto-exclusion.
  - Audit script reporting per venue whether historical taker-buy volume is available.
  - `FEATURE_VERSION = "ml-feat-v1.2.0"`.
- **Acceptance**:
  - No constant-fill code paths remain (grep test for `1.0`/`0.5` fallbacks in `features.py`).
  - Prefix-invariance tests still pass.
  - The dataset SHA changes and is recorded.

**T-17 · P1 · T-08, T-10, T-14, T-16, T-18 · F-14**
- **Files**: new `ml/meta_labels.py`, `ml/features.py` (setup and aligned features), `ml/experiment.py`, `docs/decisions/0010-meta-labeling.md`
- **Change**:
  - Build the meta-label dataset from backtest events (§6.1), with direction-aligned features.
  - Experiment CLI: `run-ml-experiment --target meta <TF> --symbols ...`.
  - Legacy target kept as non-promotable.
  - `MODEL_VERSION = "meta-l2logreg-platt-v2.0.0"`.
- **Acceptance**:
  - Every event's `label_end_time` equals its simulated exit time.
  - NO_FILL events are excluded and counted.
  - Fewer than 300 events → `INSUFFICIENT_DATA`.
  - The leakage test passes.
  - ADR 0010 is committed.

**T-18 · P1 · T-10 · F-27**
- **Files**: `analysis/backtest.py`, `analysis/setup.py` (`evaluate_setup_against_candles`), config.
- **Change**: pending-setup lifecycle (§5.6), intrabar order (§7.2), BE-exit counting. Legacy mode `entry_expiry_bars=None`.
- **Acceptance** (scenario tests):
  - TP1 touched before fill → `NO_FILL_TP1_FIRST`.
  - Fill and TP1 on the same bar → no TP1 credit.
  - TP1 on bar k → breakeven SL applies from bar k+1.
  - Legacy mode reproduces the v1.6 golden backtest output.

**T-19 · P1 · — · F-25**
- **Files**: new `analysis/bootstrap.py`, `analysis/backtest.py` reports.
- **Change**: stationary block bootstrap; daily portfolio series; portfolio max drawdown; max concurrent positions; 95% CIs.
- **Acceptance**:
  - Output is deterministic with `seed=42`.
  - On an IID synthetic series, the block-bootstrap CI ≈ the IID CI (±10%).
  - On an AR(1) series with ρ=0.5, the block CI is wider.

**T-20 · P1 · — · F-23, F-24**
- **Files**: `market_data/ccxt_provider.py` (`get_funding_history`), `analysis/backtest.py`
- **Change**: signed historical funding (§7.3) with `FUNDING_ASSUMED` fallback; cost-multiplier sweep and break-even multiplier (§7.4).
- **Acceptance**:
  - A long held through +0.01% funding loses `0.0001·P/R` R; the same trade as a short gains it.
  - The sweep table is present in both JSON and CLI output.

**T-21 · P1 · T-04, T-10, T-18, T-19, T-20 · F-22, F-26**
- **Files**: `analysis/backtest.py` (`BACKTEST_PROFILE`), live setup ledger, docs.
- **Change**:
  - Implement the parity profile (§7.6) and ledger fields for fired `FORWARD_ONLY` guards.
  - Regenerate the V0 vs. V3 benchmark on the v1.7 engine (same symbols and windows), with DSR (registered M), cost sweep, block CI, and portfolio drawdown.
  - Replace the §7.4 doc table with the new "Research Results" table and its notes.
  - Remove the words "proof/verified/winner" from docs and user output.
- **Acceptance**:
  - A new report `docs/research/benchmark-v1.7.md` is committed, with `trial_id`s referenced.
  - A grep test finds no banned words in `docs/` or `src/**/templates`.

**T-22 · P1 · T-04 · F-22, F-28**
- **Files**: CLI `freeze-params`, setup ledger, `calibration-report`.
- **Change**: parameter freeze (§8.2); ledger stores `params_sha256`; per-timeframe forward-proof evaluator (§8.3); data source for `/track`.
- **Acceptance**:
  - Setups with a mismatched hash are excluded from the forward proof.
  - The evaluator returns `NOT_MET` with a reason until the thresholds pass.

**T-23 · P1 · — · F-21**
- **Files**: sentiment ingestion module, schema migration, `ml/features.py`
- **Change**: store `scored_at/scorer_id/scorer_version/scorer_class`; apply the eligibility rule (§6.2); block LLM backfill.
- **Acceptance**:
  - A training row cannot see a score with `scored_at > prediction_time` (leakage test).
  - Attempting a backfill with an LLM scorer raises.

**T-24 · P1 · — · F-35**
- **Files**: narrator module, new `narration/guard.py`, tests.
- **Change**: guard pipeline per §2.3 — placeholder validation, digit scan with allowlist, one retry, template fallback, violation logging.
- **Acceptance**:
  - Fuzz tests: LLM outputs containing raw numbers, an unknown `{F99}`, or a fact with status `NONE` never reach rendering.
  - The fallback is used after 2 violations.
  - `1000PEPE` and `4h` pass the guard.

### Phase P2 — Improvements & experiments

**T-25 · P2 · — · F-31**
- **Files**: `analysis/setup.py`, venue metadata loader.
- **Change**: leverage with MMR (§5.4); `MMR_ASSUMED` fallback.
- **Acceptance**:
  - MMR=0 reproduces v1.6.
  - Property test: MMR=0.01 yields leverage ≤ the MMR=0 result for all inputs.

**T-26 · P2 · — · F-32, F-34**
- **Files**: `analysis/macro.py`, data join utilities.
- **Change**: as-of join with staleness (§3.3); r_crit significance; optional 60-bar window.
- **Acceptance**:
  - A 1d BTC–DXY join across a weekend returns rows, not an empty result.
  - |r| = 0.3 at n = 20 → DXY Impact `neutral`.

**T-27 · P2 · — · F-33**
- **Files**: `analysis/statistics.py`
- **Change**: ADF gate for OU (constant-only, MacKinnon 5% critical value for n, implemented in NumPy); Hurst SE and `HURST_UNRELIABLE` flag.
- **Acceptance**:
  - Random walk, length 120: OU returns `None` with `NOT_STATIONARY` in ≥ 90% of 200 seeded runs.
  - AR(1) with φ=0.8: half-life within ±30% of the theoretical value in ≥ 80% of runs.

**T-28 · P2 · T-21 · F-09**
- **Files**: `analysis/setup.py` (variant flag), backtest.
- **Change**: implement V4 (§5.5) behind `SETUP_VARIANT=v4`; run the full v1.7 benchmark; register the trial.
- **Acceptance**:
  - A V3 vs. V4 comparison report exists.
  - The default stays V3 unless V4's block-CI lower bound beats V3's **and** DSR ≥ 0.95. The decision is recorded in an ADR.

**T-29 · P2 · — · F-11**
- **Files**: `analysis/setup.py`, setup card template.
- **Change**: report-only `NetRR_blend`.
- **Acceptance**: the field appears on the card; no gate logic changes (golden test: setup output otherwise identical).

**T-30 · P2 · — · F-29**
- **Files**: screener module, scheduler, new table `screener_universe_snapshots`.
- **Change**: daily point-in-time universe snapshot; `UNVALIDATED` badge in `/screener` output.
- **Acceptance**:
  - The snapshot job is idempotent per UTC day.
  - The badge renders until the validation flag is set manually after §7.8 is met.

**T-31 · P2 · T-24 · F-35**
- **Files**: `narration/guard.py`
- **Change**: directional consistency check (§2.3 step 3).
- **Acceptance**:
  - The text "bullish" with `snapshot.bias = SHORT` triggers a violation.
  - Neutral text with no bias words passes.
  - A text whose bias words match `snapshot.bias` passes.

**T-32 · P2 · — · F-36**
- **Files**: Telegram bot handlers, response templates, docs.
- **Change**: align commands and response states with §2.1/§2.2 (after the `[CONFIRM]` items are resolved); add a per-user rate limit and the `/track` command (using the data source from T-22).
- **Acceptance**:
  - Each state in §2.2 has a template test with a snapshot fixture.
  - Rate limit test: the 11th request within one minute is rejected with a friendly message.

### 11.1 Suggested Execution Order
| Wave | Tasks | Notes |
|---|---|---|
| 1 | T-01, T-03, T-04, T-07, T-08, T-09, T-10, T-11 | Independent; can run in parallel |
| 2 | T-02, T-05, T-06, T-13, T-14, T-16, T-18, T-19, T-20, T-23, T-24 | Needs wave 1 |
| 3 | T-12, T-15, T-21, T-22 | Gates + benchmark regeneration |
| 4 | T-17 | Meta-labeling (largest task; after the engine is corrected and benchmarked) |
| 5 | T-25 … T-32 | Improvements and experiments |

### 11.2 Traceability (Finding → Task)
| Finding | Task(s) |
|---|---|
| F-01 | T-01, T-02 |
| F-02 | T-03 |
| F-03 | T-04, T-05 |
| F-04 | T-06 |
| F-05 | T-16 |
| F-06 | T-08 |
| F-07 | T-09 |
| F-08 | T-10 |
| F-09 | T-28 |
| F-10 | T-11 |
| F-11 | T-29 |
| F-14 | T-17 |
| F-15 | T-12 |
| F-16 | T-14 |
| F-17 | T-13 |
| F-18 | T-15 |
| F-19 | T-07 |
| F-20 | T-16 |
| F-21 | T-23 |
| F-22 | T-21, T-22 |
| F-23 | T-20 |
| F-24 | T-20 |
| F-25 | T-19 |
| F-26 | T-21 |
| F-27 | T-18 |
| F-28 | T-22 |
| F-29 | T-30 |
| F-30 | T-09 |
| F-31 | T-25 |
| F-32 | T-26 |
| F-33 | T-27 |
| F-34 | T-26 |
| F-35 | T-24, T-31 |
| F-36 | T-32 |
| F-37 | T-11 |
