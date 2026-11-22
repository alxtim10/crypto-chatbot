# ADR 0004: TradingView AI Agent Lens Webhook & Cross-Source Integration

- Status: Accepted
- Date: 2026-11-22
- Supersedes: none (extends ADR 0001, ADR 0002, and ADR 0003)

## Context

Phase 10 integrates the MIT-licensed TradingView **AI Agent Lens** packet format (`ai-agent-lens/v1`) as an optional deterministic signal/candle context source, with strict webhook security, idempotency, cross-source exchange candle verification, and zero confusion between deterministic TradingView rule scores and future ML probabilities.

## Decisions

| Area | Decision | Rationale |
|---|---|---|
| Attribution & IP boundary | Pin `AI_AGENT_LENS_SCHEMA_VERSION = "ai-agent-lens/v1"` and `AI_AGENT_LENS_PINNED_COMMIT = "7f3c9a2e1d8b4056c2a91e4f7b8d3c1a5e6f9012"` for the MIT-licensed packet contract. Never copy unlicensed Heavy91 code; multi-timeframe RSI and failed breakout stay our independent deterministic implementations. | Clear OSS license provenance and strict schema boundaries. |
| Transport & Auth | Default HTTPS webhook listener on port `443` (`DEFAULT_WEBHOOK_PORT = 443`, TLS 1.2+). Because TradingView alerts cannot attach custom HTTP headers, authenticate with a JSON-body shared secret (`secret` or `webhook_secret`) compared via `hmac.compare_digest`, paired with source IP allowlisting defaulting to TradingView's published webhook IPs (`52.89.214.238`, `34.212.75.30`, `54.218.53.128`, `52.32.178.7`), payload byte cap, and per-IP sliding rate limit. Reject secrets in URL query parameters. | Defense-in-depth authentication and abuse prevention for unheadered webhooks. |
| Secret redaction | Strip all credential-shaped keys (`secret`, `webhook_secret`, `token`, `api_key`, `password`) before constructing `TradingViewPacket` or persisting `tradingview_packets`. Extend `SecretRedactingFilter` to redact JSON body secrets and query-string secrets in logs. | Prevents secret persistence or accidental log leakage. |
| Idempotency & verification | Compute deterministic idempotency key `tv-<sha256(canonical_symbol\|market_type\|timeframe\|event_type\|candle_close_time)[:32]>` and ignore duplicates safely. Compare packet close/high/low against closed exchange candles (`> 50 bps` -> `PRICE_MISMATCH` / `DEGRADED`, `> 200 bps` -> `SEVERE_PRICE_MISMATCH` / `INSUFFICIENT_DATA`). | Avoids duplicate alert storms and guards against vendor/exchange feed divergence. |
| Rule score separation | TradingView `rule_score` carries `is_probabilistic = False` and is labeled `"Skor aturan TradingView (bukan probabilitas)"`. Payloads claiming `ml_probability` or `win_probability` are rejected at ingestion. | Keeps rule-based TradingView overlays strictly separate from calibrated ML predictions (Phase 12+). |

## Consequences

- Stored TradingView packets live in PostgreSQL table `tradingview_packets` without secrets (`sanitized_payload` only) and are loaded causally via `available_time <= as_of`.
- Existing spot/SMC/setup `Fact` IDs remain unchanged whenever no TradingView packet is attached; when attached, TradingView facts are appended after `setup_ltf_resistance`.
