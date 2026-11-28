# Personal Operations, Backup & Recovery Runbook

This runbook documents personal operations for running the crypto research chatbot locally or on a private server (ADR 0009).

---

## 1. Environment & Secret Hygiene

- Secrets (`TELEGRAM_BOT_TOKEN`, `GEMINI_API_KEY`, `DATABASE_URL`) are read from environment variables or `.env`.
- Secrets are typed as `SecretStr` across all schemas and automatically masked in logs by `SecretRedactingFilter`.
- Never commit `.env` or paste real tokens in bug reports or public repos.
- Verify config health without printing secrets:
  ```bash
  .venv/bin/crypto-chatbot check-config
  .venv/bin/crypto-chatbot health-check
  ```

---

## 2. PostgreSQL Backup, Restore Drill & Retention Jobs

- **Scheduled retention cleanup**:
  ```bash
  # Purges expired conversation messages (>30d) and raw snapshots (>90d)
  crypto-chatbot purge-expired
  ```
- **Logical PostgreSQL backup (custom compressed format)**:
  ```bash
  crypto-chatbot backup-db /var/backups/crypto_chatbot_$(date -u +%Y%m%dT%H%M%SZ).dump
  ```
- **Restore drill (into an isolated staging or scratch database)**:
  ```bash
  crypto-chatbot restore-db /var/backups/latest.dump --database-url "postgresql+psycopg://.../crypto_chatbot_restore_check"
  ```
- **Reset personal memory**:
  Use `/forget` in Telegram to delete your preferences, conversation history, and LLM logs.

---

## 3. Incident & Provider-Outage Procedures

| Scenario | Symptom | Action |
|---|---|---|
| **Binance / CCXT outage** | Warnings in log, fallback messages on `/analyze` | Confirm Binance spot status. Bot safely marks quality flags without fabricating numbers. |
| **Gemini outage or quota exhaustion** | Gemini errors in log | Bot falls back automatically to deterministic formatted templates (`render_deterministic_snapshot` / `render_compact_snapshot`). |
| **PostgreSQL unavailable** | Bot logs redacted DB error; `health-check` fails | Restart PostgreSQL or restore from latest `.dump` via `restore-db`. |
| **ML drift or model rollback** | Setup anomalies observed | Set `ML_INFERENCE_MODE=off` in `.env` and restart bot. Deterministic analysis continues normally. |

---

## 4. Local Deployment & Execution

Run directly using Python virtual environment (`.venv`):

```bash
# 1. Health check & configuration
.venv/bin/crypto-chatbot health-check

# 2. Run bot in foreground
.venv/bin/crypto-chatbot run-bot

# Or run in background (daemon / nohup)
nohup .venv/bin/crypto-chatbot run-bot > bot.log 2>&1 &
```


## 15m Multi-Timeframe Alert Worker & Database Initialization

### Database Initialization (`init-db`)

`init_db(engine)` runs automatically at startup of `run-bot` and before database-backed CLI commands (`alerts run`, `alerts digest`, `settle-setups`, `scan`). For explicit, idempotent schema creation and column migrations during VPS deploys (before restarting services), run:

```bash
.venv/bin/crypto-chatbot init-db
```

### 15m Top-Down MTF Alert Flow (`alerts run` / `alerts digest`)

Configured in `config/screener.yaml` (all timeframes fetched incrementally from **one** venue, closed candles only, zero LLM calls):

1. **Cadence & Idempotency**: One run per closed `15m` candle with `run_id = "mtf:<15m close_time>"` and PostgreSQL `pg_try_advisory_lock`.
2. **Stage 1 — Bias (`4h` & `1h`)**: Both `4h` and `1h` must agree on regime and market structure (`TRENDING_UP + BULLISH` for LONG, `TRENDING_DOWN + BEARISH` for SHORT); otherwise `WAIT` (`stage=bias`).
3. **Stage 2 — Setup & Cost Gate (`15m`)**: Computes entry zone, SL, TP1–3, and execution cost / net R:R gate (`stage=cost_gate` or `stage=setup`).
4. **Stage 3 — 5m Confirmation Rule (`confirm: 5m`, last CLOSED 5m candle)**:
   - Uses strictly closed 5m candles (`close_time <= 15m_close_time` and `available_time <= cutoff`); requires at least 2 closed 5m bars (`prev`, `last`).
   - **Invalidation**: `last.close <= stop_loss` (LONG) or `last.close >= stop_loss` (SHORT) disarms the setup immediately.
   - **LONG confirmation**: `last.close > last.open` AND (`last.close > prev.high` OR (`last.low <= entry_high` AND `last.close >= entry_low`)).
   - **SHORT confirmation**: `last.close < last.open` AND (`last.close < prev.low` OR (`last.high >= entry_low` AND `last.close <= entry_high`)).
   - **ARMED → READY**: Without 5m confirmation, the 15m setup is stored as `ARMED` (no Telegram alert) with expiry `setup_expiry_minutes` (default `90`). If a subsequent 15m run sees 5m confirmation before expiry (and HTF bias still agrees), it promotes `ARMED → READY` and delivers the alert.
   - **Shadow Expiry (`shadow_expired`)**: If an `ARMED` setup expires without 5m confirmation, it is persisted to `trade_setups` with `user_action = 'shadow_expired'` (and `telegram_message_id = NULL`, not alerted). `settle-setups` settles both confirmed alerts and `shadow_expired` entries so confirmed vs unconfirmed forward performance can be compared directly.
5. **Cooldown Precedence**: Per-`(symbol, direction)` cooldown resolves in order `cooldown_hours[<timeframe>]` > legacy `cooldown_minutes` (if specified) > `default_cooldown_hours` (`2h`).
6. **Dead-Man Monitor**: `run-bot` sends a `🔧` alert if no `alerts run` completes within `dead_man_hours` (default `1h` for the 15m cadence).


### Point-in-Time `available_time` Semantics for Backfills vs Live Fetches

To keep historical datasets usable in backtests/ML without lookahead while preserving strict live arrival timestamps:

- **`sync-fear-greed` (Alternative.me `/fng/`)**:
  - Backfill (`limit=0` or `limit > 1`): `available_time = observed_time + 00:00` (assumption: Alternative.me publishes its daily index at `00:00 UTC` on the value date) with `is_backfilled = True`.
  - Live fetch (`limit=1`, `backfill=False`): `available_time = max(observed_time, received_time)` (actual fetch time) with `is_backfilled = False`.
- **`contract-stats` (Gate.io `/futures/usdt/contract_stats`)**:
  - Backfill (`backfill=True`): `available_time = observed_time + interval` (bucket close time, e.g. `+5m` or `+1h`) with `QualityFlag.BACKFILL`.
  - Live poll (`backfill=False`): `available_time = max(observed_time + interval, received_time)` (actual fetch time) with no `BACKFILL` flag.
- **`sync-gate-funding` (Gate.io `/futures/usdt/funding_rate`)**:
  - Backfill (`backfill=True`): `available_time = observed_time` (exact 8h settlement timestamp) with `QualityFlag.BACKFILL`.
  - Live fetch (`backfill=False`): `available_time = max(observed_time, received_time)` (actual fetch time) with no `BACKFILL` flag.
