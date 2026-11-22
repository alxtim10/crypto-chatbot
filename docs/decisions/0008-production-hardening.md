# ADR 0008: Production Hardening, Webhook Security, Public Access Controls & Recovery

- Status: Accepted
- Date: 2026-11-22
- Supersedes: none (extends ADR 0001–0007)

## Context

Phase 14 (`PROJECT_PLAN.md` §6 lines 726–750) prepares the crypto research and education chatbot for production operation beyond closed beta: webhook delivery, public-access and abuse safeguards, managed secret storage and rotation, structured redacted observability, PostgreSQL backup/restore verification, user-facing terms/privacy disclosures, and verified deletion and export procedures.

## Decisions

| Area | Decision | Rationale |
|---|---|---|
| Telegram transport mode | Add `TELEGRAM_MODE=polling\|webhook` (default `polling` to preserve closed-beta behavior). In `webhook` mode, require HTTPS `TELEGRAM_WEBHOOK_URL` and a valid 16–256 char `TELEGRAM_WEBHOOK_SECRET_TOKEN` (`[A-Za-z0-9_-]+`) passed to `setWebhook`. Verify `X-Telegram-Bot-Api-Secret-Token` on every update in constant time via `hmac.compare_digest`. Support zero-downtime dual-secret rotation via `TELEGRAM_WEBHOOK_SECRET_TOKEN_PREVIOUS`. | Keeps polling safe locally while ensuring authenticated, timing-safe webhook delivery in production. |
| Public-access & abuse controls | Add `ACCESS_MODE=allowlist\|public` (default `allowlist`), `BLOCKED_TELEGRAM_USER_IDS`, `GLOBAL_RATE_LIMIT_PER_MINUTE`, `ABUSE_STRIKE_THRESHOLD`, and `ABUSE_COOLDOWN_SECONDS`. Enforce blocklists before allowlist/public routing, sliding-window per-user and global limits, and automatic cooldown after repeated rate-limit violations. | Allows controlled transition from closed beta to public traffic without exposing Gemini/CCXT quotas to abuse. |
| Managed secret store & rotation | Support `SECRETS_DIR` mounted secret files (`pydantic-settings` `_secrets_dir`) via `load_settings`. Keep all credentials typed as `SecretStr`, extend `SecretRedactingFilter` to redact `X-Telegram-Bot-Api-Secret-Token` and `Authorization: Bearer` headers, and support zero-downtime secret rotation without printing secret values. | Decouples runtime secrets from `.env` files on disk and supports Docker/cloud secret mounts safely. |
| Observability & health | Add `LOG_FORMAT=text\|json` with single-line JSON structured logging (`JsonRedactingFormatter`) filtered by `SecretRedactingFilter`, `MetricsRegistry` tracking latency percentiles and failure counters, alert evaluation (`evaluate_operational_alerts`), `/healthz` HTTP probe, and CLI `health-check`. | Enables operator alerting on DB health, Gemini daily caps, exchange errors, and webhook auth spikes without exposing secrets or message payloads. |
| Backup, restore, deletion & export | Provide parameterized `pg_dump` / `pg_restore` wrappers (`ops/backup.py`) plus `crypto-chatbot backup-db` / `restore-db`, scoped `PostgresUserMemoryRepository.export_user_data` (`crypto-chatbot export-user`), `/forget`, `/privacy`, and `purge-expired`. Harden `Dockerfile` and `compose.yaml` (`HEALTHCHECK`, `read_only`, `no-new-privileges`, `tmpfs`). | Verifies end-to-end recoverability, user privacy rights, and least-privilege container runtime. |
| Legal & external release gate | Jurisdictional legal review, paid-tier Gemini billing verification, and external container/dependency CVE scans remain mandatory operational prerequisites before flipping `ACCESS_MODE=public`. | Code controls cannot substitute for jurisdiction-specific legal sign-off or CI/registry CVE scanning. |

## Consequences

- Default settings remain unchanged for existing closed-beta users (`TELEGRAM_MODE=polling`, `ACCESS_MODE=allowlist`, `ML_INFERENCE_MODE=off`, `LOG_FORMAT=text`).
- No new runtime Python dependencies are added in `pyproject.toml`; PostgreSQL client binaries (`pg_dump`, `pg_restore`) and external vulnerability scanners (`pip-audit`, `trivy`) are used as host/CI tools.
