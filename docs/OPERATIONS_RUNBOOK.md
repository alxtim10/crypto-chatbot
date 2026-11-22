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

## 4. Container Deployment

Run locally or on a private server via Docker Compose:

```bash
docker compose up -d
docker compose logs -f bot
```

The container runs as a non-root user (`appuser`), with a read-only root filesystem, `no-new-privileges: true`, and an automated `HEALTHCHECK` running `crypto-chatbot health-check`.
