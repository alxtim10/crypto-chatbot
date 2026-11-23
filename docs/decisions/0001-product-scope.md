# ADR 0001: Product scope and Phase 0 decisions

- Status: Accepted
- Date: 2026-11-21
- Supersedes: none

## Context

`PROJECT_PLAN.md` Phase 0 requires product choices to be recorded before application code is written. A review of the plan found several underspecified areas (staleness rule, Telegram transport, chat memory, Gemini tier, retention, data availability semantics). This ADR records the accepted choices.

## Decisions

| Area | Decision | Rationale |
|---|---|---|
| Users | Personal / closed beta. Access restricted to an allowlist of Telegram numeric user IDs (`ALLOWED_TELEGRAM_USER_IDS`). | Telegram bots are publicly discoverable; closed beta requires explicit access control. |
| Telegram transport | Long polling locally **and** in closed-beta production. Webhook only at public launch (Phase 14). | Polling needs no public endpoint, domain, or TLS; smaller attack surface. |
| Telegram framework | `python-telegram-bot` 22.x. | Already installed in `.venv`; mature async API. |
| Hosting | Docker on a small VPS. VPS region must be verified to reach the Binance public API before Phase 8. | Binance blocks some jurisdictions (e.g. HTTP 451 for US IPs). |
| Runtime | Python 3.14 locally and in Docker; `requires-python = ">=3.12"`. | Matches existing `.venv`. |
| Exchange / market | Binance, spot, public endpoints via CCXT. No exchange credentials. | Public OHLCV is sufficient for MVP. |
| Pairs | `BTC/USDT`, `ETH/USDT`, `SOL/USDT`. | Liquid, long history (> 600 daily candles). |
| Timeframes | `1h`, `4h`, `1d`, closed candles only. | Matches MVP analysis scope. |
| Staleness | Latest closed candle is stale if `now − close_time > 1 × timeframe + 5 min grace`. | A `2×` rule would let one whole missing candle go undetected. |
| Database | PostgreSQL (also in integration tests). | Avoid SQLite/PostgreSQL dialect drift. |
| Chat memory | Last 20 messages + explicit preferences. **No LLM-generated summary in MVP.** | LLM summaries can persist hallucinated facts. |
| Stored assistant turns | Narration stored with `{F<n>}` placeholders + `snapshot_id`, not rendered numbers. | Prevents stale numbers from re-entering Gemini context. |
| User profile data | Telegram numeric user ID and explicit preferences only. No name, username, phone, or inferred financial circumstances. | Data minimization. |
| Retention | Messages 30 days; LLM interactions 30 days; analysis snapshots 90 days; market candles retained (public data); preferences until `/forget`. | Bounded personal data; reproducible analyses. |
| Answer language | Indonesian. Timestamps shown in UTC and WIB (UTC+7). | Target users are Indonesian; UTC keeps source semantics explicit. |
| LLM | Gemini only. Flash-class model; exact model ID pinned in config (`GEMINI_MODEL`), selected from current Google docs at Phase 6. | Cost and latency; pinning keeps behavior reproducible. |
| Gemini tier | Paid tier before any user other than the owner sends messages. Per-user and global daily call caps. | Free-tier content may be used by Google to improve products; cost control. |
| LLM numeric facts | Gemini outputs numbers only via `{F<n>}` placeholders; backend substitutes values. | Robust fact validation without parsing Indonesian number formats. |
| Aggregate confluence score | Not in MVP. Snapshot exposes a non-probabilistic factor list instead. | Avoid undocumented scores being read as probabilities. |
| ML | Offline only, after MVP (Phase 12+). | Data and deterministic pipeline must be trusted first. |
| TradingView | After MVP is stable (Phase 10). | Optional input source. |
| Order execution | Out of scope. | Research and education product. |
| Telegram rendering | HTML parse mode with escaping; messages split below the 4096-character limit. | MarkdownV2 escaping is error-prone. |

## Consequences

- No public HTTP endpoint exists until Phase 10 (TradingView) or Phase 14 (webhook).
- Follow-up questions that need market facts must re-run an analysis tool; old numbers are never reused from memory.
- `uvloop` and `aiosqlite` currently in `.venv` are not part of the plan and are removed in Phase 1 unless a concrete need is documented.
- Retention cleanup runs as a CLI command (`purge-expired`) triggered by host cron/systemd timer; no scheduler dependency.

## Open actions

| Action | Owner phase |
|---|---|
| Verify the chosen VPS region reaches Binance public API (live smoke test). If blocked, evaluate `data-api.binance.vision` or another region. | Phase 3, before Phase 8 |
| Select and pin the exact Gemini model ID; enable paid billing. | Phase 6 |
| Fill `ALLOWED_TELEGRAM_USER_IDS` for beta users. | Phase 7 |

## Revisit triggers

Public or paid launch, adding perpetual markets, adding a second exchange, or user count beyond closed beta.
