# Plan: Personal-Mode Simplification & Market Intelligence Expansion (Post-Phase 14)

- Status: Saved / Pending Execution
- Created: 2026-11-22
- Scope:
  - **Track 0**: Prune public-launch-only Phase 14 features unused in single-owner personal mode while keeping personal operational safeguards.
  - **Tracks 1–3**: Research & education enhancements inspired by read-only concepts from `Jon-Becker/prediction-market-analysis` (MIT) and `recogardtech/AutoPilotPM` (MIT).
- Out of Scope (Non-Negotiable): Order execution, exchange trading credentials, wallet/DeFi signing, autonomous LLM trading decisions, copy-trading, or guaranteed win probabilities (`AGENTS.md` #6, ADR 0001).

---

## Track 0: Simplify Phase 14 to Personal-Only Scope

**Goal**: Remove public-deployment overhead (webhook server, public access controls, abuse cooldown, managed `SECRETS_DIR` rotation, public privacy/terms/DSAR export) that is unnecessary for a private single-user bot, while preserving lightweight personal safeguards (`SecretRedactingFilter`, `health-check`, `backup-db`/`restore-db`, and container hardening).

### Planned Removals & Simplifications
1. **ADR Superseding & Plan Status**:
   - Create `docs/decisions/0009-personal-mode-simplification.md` (`Status: Accepted`, supersedes `0008-production-hardening.md`) recording the decision to defer public-launch hardening and operate strictly as a personal allowlist + long-polling bot.
   - Update `PROJECT_PLAN.md` §0 row 14 to reflect personal-only scope (`Done — simplified to personal-only scope per ADR 0009; public-launch hardening deferred`).
2. **Remove Telegram Webhook Transport & Public Access Controls**:
   - Delete `src/crypto_chatbot/bot/webhook.py` and remove webhook exports from `src/crypto_chatbot/bot/__init__.py`.
   - In `src/crypto_chatbot/config.py` and `.env.example`: remove `TelegramMode`, `AccessMode`, `LogFormat`, `SECRETS_DIR` (`_secrets_dir`) handling, `telegram_mode`, `telegram_webhook_*`, `access_mode`, `blocked_telegram_user_ids`, `global_rate_limit_per_minute`, `abuse_strike_threshold`, `abuse_cooldown_seconds`, and `log_format`. Keep `allowed_telegram_user_ids` strictly required (`min_length=1`) and `ml_inference_mode` (Phase 13 rollback).
   - Simplify `src/crypto_chatbot/bot/access.py` back to closed-beta/personal allowlist + per-user sliding-window `rate_limit_per_minute` (`reason: Literal["not_allowlisted", "rate_limited"] | None`).
   - In `src/crypto_chatbot/app.py` and `src/crypto_chatbot/cli.py`: remove `run_webhook` / `run_bot` webhook dispatch and keep `run_polling` directly.
3. **Remove Public Terms/Privacy Command, DSAR Export & Unneeded Ops Complexity**:
   - Delete `docs/PRIVACY_AND_TERMS.md`; trim `docs/OPERATIONS_RUNBOOK.md` to concise personal backup/restore, `health-check`, `purge-expired`, and outage recovery procedures.
   - Remove `render_privacy_terms_message`, `BotService.handle_privacy`, `CommandHandler(["privacy", "terms"], ...)`, and public-denial messages (`blocked`, `global_rate_limited`, `abuse_cooldown`) from `bot/templates.py`, `bot/handlers.py`, and `bot/__init__.py`.
   - Remove `UserDataExport`, `UserDataExportRepository`, `PostgresUserMemoryRepository.export_user_data`, and CLI `export-user` from `storage/repositories.py`, `storage/__init__.py`, and `cli.py` (keep `/forget` and `purge-expired`).
   - Simplify `src/crypto_chatbot/ops/`: remove `JsonRedactingFormatter`, `MetricsRegistry`, and `evaluate_operational_alerts` (keep `SecretRedactingFilter` in `ops/logging.py`, `check_database_health` in `ops/health.py` or `ops/metrics.py`, and `dump_postgres_database` / `restore_postgres_database` in `ops/backup.py`).
4. **Update Tests**:
   - Revert PTB handler count/indices in `tests/integration/test_telegram_handlers.py` (back to 9 handlers) and `tests/e2e/test_mvp_flows.py`.
   - Replace `tests/unit/test_production_hardening.py` with focused `tests/unit/test_personal_ops.py` covering `SecretRedactingFilter`, `to_libpq_url` / credential scrubbing in `ops/backup.py`, and `health-check`.
   - In `tests/integration/test_postgres_repository.py`: keep `test_postgres_backup_and_restore_drill` (and existing `test_postgres_user_memory_isolation_forget_and_retention`), remove `test_postgres_user_data_export_and_end_to_end_deletion`.

### What Is Kept (Useful for Personal Mode)
- `SecretRedactingFilter` (redacts bot tokens, API keys, DB passwords, webhook secrets in logs).
- `crypto-chatbot health-check` (`config_ok` + PostgreSQL `SELECT 1`).
- `crypto-chatbot backup-db` & `restore-db` (`pg_dump` / `pg_restore` to protect local candles, setups, and ML logs).
- Hardened `Dockerfile` (`HEALTHCHECK`, non-root user) and `compose.yaml` (`read_only`, `tmpfs`, `no-new-privileges`).
- Existing `/forget` and `purge-expired` commands.

### Verification Criteria
- `ruff format --check .`, `ruff check .`, `mypy src tests`, `pytest`, and `crypto-chatbot check-config` / `health-check` all pass with 0 errors.
- PostgreSQL integration suite including `test_postgres_backup_and_restore_drill` passes against temp DB.

---

## Track 1: Closed-Bar Order-Flow Microstructure & Multi-Venue Derivatives

**Goal**: Strengthen `15m` entry refinement, `4h` breakout/sweep validation, and derivatives context beyond single-venue Binance data.

### Planned Changes
1. **ADR 0010 (`docs/decisions/0010-orderflow-and-multi-venue-derivatives.md`)**:
   - Define causal closed-bar order-flow formulas (`taker_buy_ratio`, rolling `cvd_20`, `cvd_divergence` flag) sourced strictly from closed Binance klines (`taker_buy_base_asset_volume` -> `close_time <= received_time`).
   - Define second read-only public derivatives venue (**Hyperliquid** or **Bybit** public API via CCXT/HTTP, zero credentials) and cross-venue spread/dispersion contracts (`funding_dispersion_bps`, `mark_price_spread_bps`, `aggregate_oi_change_pct`).
2. **Schemas & Data Adapters**:
   - Extend `MarketCandle` optionally with `taker_buy_base_volume: FiniteFloat | None = None` (backward-compatible with existing stored rows/fixtures).
   - Add `OrderFlowMetrics` and `CrossVenueDerivativesSummary` to `AnalysisSnapshot` (`Fact` IDs deterministic; bump `CALC_VERSION` and `FEATURE_VERSION`).
   - Implement read-only second-venue derivatives client with strict timeouts, capped transient retries, and graceful degradation (missing venue flags partial coverage without failing spot analysis).
3. **Deterministic Analysis, Setup Refinement & ML Features**:
   - Incorporate closed-bar taker buy ratio & CVD into `15m` LTF entry confirmation and failed-breakout / sweep descriptive factors (measurements only, never win probabilities).
   - Add `FeatureFamily.ORDERFLOW` to `ml/features.py` for offline ablation against existing technical + structure families.
4. **Storage & Rendering**:
   - Persist optional `taker_buy_base_volume` on `market_candles` (or separate migration-safe column) and venue-tagged derivatives rows (`venue="hyperliquid"` / `"bybit"`).
   - Update compact and detailed Indonesian templates to display cross-venue funding dispersion and closed-bar order flow clearly with provenance.

### Verification Criteria
- Unit tests: closed-bar taker buy ratio & CVD calculations, lookahead rejection, graceful single-venue fallback when second venue times out.
- Parity & ML leakage tests: future-shock invariance on new `ORDERFLOW` features; zero lookahead on cross-venue `available_time`.
- Integration tests: multi-venue derivatives persistence & point-in-time loading in PostgreSQL.

---

## Track 2: Quantitative Calibration Audit, Maker/Taker Cost Split & Ledger CLI

**Goal**: Deepen statistical diagnostics for both deterministic trade setups and offline/shadow ML models.

### Planned Changes
1. **ML Calibration & Skill Diagnostics (`ml/evaluation.py`, `ml/shadow.py`, `schemas/ml.py`)**:
   - Add `brier_skill_score` (`1 - model_brier / naive_brier`) to `EvaluationSummary` and `ShadowMonitoringReport`.
   - Add `CalibrationBinMetrics` (`bin_low`, `bin_high`, `count`, `mean_predicted_prob`, `empirical_positive_rate`, `calibration_gap`, `mean_net_return`) to expose tail overconfidence / favorite-longshot bias.
   - Extend `CostAssumptions` with explicit `execution_style: Literal["taker", "maker"]` (separate maker fee/fill-haircut vs taker fee + full spread + slippage) and report maker vs taker sensitivity in `MlExperimentReport`.
2. **Audit Ledger CLI Commands (`cli.py`, `storage/repositories.py`)**:
   - Add `crypto-chatbot setup-stats [--symbol BTC] [--timeframe 4h] [--json]` summarizing settled `trade_setups` objectively: counts (`tp1`/`tp2`/`tp3`/`sl`/`expired`/`pending`), mean & median `realized_r`, broken down by `regime` and `timeframe` (explicitly labeled historical statistics, never future win probability).
   - Add `crypto-chatbot shadow-report [--symbol BTC] [--timeframe 4h] [--json]` querying `model_predictions`, running `compute_shadow_monitoring_report`, and displaying health status, abstention breakdown, realized Brier/BSS, calibration bins, and active alerts.

### Verification Criteria
- Unit tests: `brier_skill_score` math (exact edge cases when `naive_brier == 0`), per-bin reliability table sums to sample count, maker vs taker cost impact monotonicity.
- Integration tests: `setup-stats` and `shadow-report` against seeded PostgreSQL `trade_setups` and `model_predictions` rows.

---

## Track 3: Read-Only Prediction Market Context (Polymarket / Kalshi)

**Goal**: Surface real-money implied event odds for relevant crypto/macro catalysts alongside technicals and news sentiment, with strict liquidity and point-in-time safeguards.

### Planned Changes
1. **ADR 0011 (`docs/decisions/0011-prediction-market-context.md`)**:
   - Define read-only ingestion contract for Polymarket (Gamma/CLOB public APIs) and/or Kalshi public market APIs (no API trading keys).
   - Require strict point-in-time metadata (`venue`, `market_id`, `question`, `implied_yes_prob`, `bid_ask_spread_bps`, `volume_24h_usd`, `liquidity_usd`, `resolution_time`, `received_time`, `available_time`).
   - Enforce liquidity & calibration-bias gates: filter or flag thin markets (`volume_24h_usd < min_threshold` or wide spread), explicitly label as *crowd-implied contract prices* (not ground-truth probability), and guard against Polymarket block-timestamp noise.
2. **Schemas, Storage & Pipeline (`src/crypto_chatbot/prediction_markets/`)**:
   - Immutable raw snapshot ingestion, deterministic symbol/topic mapping (BTC/ETH/SOL price/ETF markets + macro FOMC/CPI markets), and strict `available_time <= snapshot.as_of` filtering.
   - Add `prediction_market_snapshots` table and `PostgresPredictionMarketRepository`.
3. **Snapshot & Template Integration**:
   - Attach optional `prediction_markets: PredictionMarketMetrics | None` to `AnalysisSnapshot` and render in a clearly separated section (*Konteks Pasar Prediksi — Harga Kontrak Tersirat, Bukan Probabilitas Absolut*) with source links, spread/volume caveats, and UTC/WIB freshness.
   - Optionally expose as a separate ablated feature family in offline ML research once sufficient point-in-time snapshots accumulate.

### Verification Criteria
- Unit tests: fixture parsing, liquidity/spread quality flagging, symbol/topic matching, strict `available_time` anti-lookahead filter, and template disclosure checks.
- Integration tests: idempotent PostgreSQL persistence & causal query windowing.

---

## Recommended Execution Order

0. **Track 0** (Simplify Phase 14 to Personal-Only Scope) — prune unused public-launch code first to keep the codebase lean.
1. **Track 1** (Order-Flow + Multi-Venue Derivatives) — immediate uplift to `15m`/`4h` market analysis.
2. **Track 2** (Calibration Bins, BSS, Maker/Taker Costs, `setup-stats` / `shadow-report`) — immediate visibility into setup & ML quality.
3. **Track 3** (Read-Only Prediction Markets) — adds macro/event catalyst context once Tracks 1–2 are complete.
