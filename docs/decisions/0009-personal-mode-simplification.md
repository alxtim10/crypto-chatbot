# ADR 0009: Personal-Mode Simplification (Post-Phase 14 Overhead Reduction)

- Status: Accepted
- Date: 2026-11-22
- Supersedes: ADR 0008 (Production Hardening, Webhook Security, Public Access Controls & Recovery)

## Context

Phase 14 implemented production hardening targeted at multi-tenant public Telegram deployment: webhook HTTPS endpoints with secret rotation, public access routing, abusive user blocklists, automated abuse cooldowns, directory-based secret mounts (`SECRETS_DIR`), user-facing terms/privacy commands (`/privacy`, `/terms`), DSAR export utilities (`export-user`), and structured JSON metrics/alerting.

Since this chatbot is deployed and operated strictly for personal use by the project owner (single-user or closed personal allowlist) without external public distribution:
1. Public attack surface defenses (webhook servers, IP/rate-limit abuse strike counters, blocklists) and regulatory compliance features (DSAR data export, public terms of service disclosures) introduce operational overhead and maintenance burden without providing security value in a single-user private environment.
2. Long-polling is simpler, zero-infrastructure (no public HTTPS reverse proxy, domain name, or open inbound ports required), and naturally secure behind local firewalls/NAT.
3. Essential operational safeguards that benefit personal deployment must be preserved: secret log redaction (`SecretRedactingFilter`), automated backup/restore verification (`backup-db`, `restore-db`), fast health checks (`health-check`), container hardening (`HEALTHCHECK`, non-root, read-only rootfs), `/forget`, and retention cleanup (`purge-expired`).

## Decisions

| Area | Decision | Rationale |
|---|---|---|
| Telegram Transport | Operate exclusively via long-polling (`Application.run_polling`); remove custom webhook server and webhook CLI dispatch. | Eliminates inbound network exposure, SSL certificate management, and webhook secret rotation overhead for a personal bot. |
| Access & Abuse Controls | Keep strict allowlist (`allowed_telegram_user_ids` required, `min_length=1`) with sliding-window rate limit; remove public access mode, IP blocklists, abuse strike thresholds, and cooldown logic. | Single owner needs allowlist protection against unauthorized Telegram users, but does not need multi-tenant abuse penalties. |
| Configuration & Secrets | Load settings strictly from environment variables and `.env`; remove `SECRETS_DIR` file-walking; keep all secrets typed as `SecretStr`. | Simplifies configuration while retaining full protection against secret leaks in strings, `repr()`, and logs. |
| Disclosures & Data Export | Remove `/privacy`, `/terms`, and CLI `export-user`; retain self-service `/forget` and `purge-expired` retention cleanup. | Public terms and DSAR exports are not needed for a private tool; user memory purging and reset remain fully functional. |
| Observability & Logging | Standard text logging filtered through `SecretRedactingFilter`; remove `JsonRedactingFormatter`, `MetricsRegistry`, and alerting engine. Health check verifies `load_settings()` and DB `SELECT 1`. | Keeps log output clean and readable during personal runs while guaranteeing zero token/credential leaks. |
| Database Recovery & Container Security | Preserve `ops/backup.py` (`dump_postgres_database`, `restore_postgres_database`, `to_libpq_url`), CLI `backup-db` and `restore-db`, and hardened container specs. | Protects historical trade setups, candles, and ML logs against local data loss and enforces least-privilege container runtime. |

## Consequences

- Codebase footprint is substantially reduced and easier to maintain.
- Zero public exposure: the bot initiates outbound HTTPS connections to Telegram API and Binance only; no inbound ports needed.
- `AGENTS.md` and ADR 0001 invariants remain strictly enforced: no trading credentials, no order execution, no LLM math, deterministic measurements only.
