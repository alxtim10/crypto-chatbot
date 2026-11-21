# ADR 0002: Derivatives and liquidity data scope and integrity rules

- Status: Accepted
- Date: 2026-11-21
- Supersedes: none (extends ADR 0001 for Phase 9)

## Context

ADR 0001 scoped the MVP strictly to Binance spot public OHLCV (`BTC/USDT`, `ETH/USDT`, `SOL/USDT`) and listed "adding perpetual markets" as an explicit revisit trigger. Phase 9 adds derivatives and liquidity measurements (funding rate, open interest and change, bid-ask spread, order-book depth and imbalance, and liquidation events when reliably available) through separate provider methods without breaking spot/perpetual isolation.

## Decisions

| Area | Decision | Rationale |
|---|---|---|
| Spot order-book venue | Binance spot public REST API (`BTC/USDT`, `ETH/USDT`, `SOL/USDT`, `BINANCE:<BASE>USDT`) via `data-api.binance.vision/api/v3` (no credentials). | Matches existing spot market-data source and avoids US-IP / geo blocks seen on `api.binance.com`. |
| Perpetual derivatives venue | Binance USD-M Futures public endpoints (`BTC/USDT:USDT`, `ETH/USDT:USDT`, `SOL/USDT:USDT`, `BINANCE:<BASE>USDT.P`) via CCXT `binanceusdm` (`fapi.binance.com`, zero credentials). | Keeps spot and perpetual instruments explicitly isolated while exposing public funding rate and open interest. |
| Single-venue attribution | Store raw venue-specific measurements and always display/log the originating venue and contract symbol. Never claim one exchange represents the entire crypto market. | Prevents false market-wide generalizations from single-venue order books or OI. |
| Funding comparability | Normalize every raw funding observation using its explicit `funding_interval_hours > 0` into `rate_per_8h = funding_rate * (8.0 / funding_interval_hours)` and `annualized_rate = funding_rate * (8760.0 / funding_interval_hours)`. Reject missing or non-positive intervals. | Exchange contracts settle at 1h, 4h, or 8h intervals; raw per-period funding rates cannot be compared directly across intervals. |
| Open-interest unit normalization | Normalize `open_interest_base` (base asset) and `open_interest_quote` (quote currency) across linear (`contracts * contract_size`), quote-only (`quote / reference_price`), and inverse (`contracts * contract_size` quote converted by `reference_price`) contracts. Flag `UNIT_MISMATCH` if reported base × reference_price diverges from reported quote by > 10%. | Venues report OI inconsistently across contracts, base coins, and USD/USDT notional values. |
| Open-interest change | Compute `open_interest_change_pct` strictly from causal stored/historical OI (`<= as_of`); keep `None` if prior OI is unavailable or `<= 0`. | Never fabricate `0%` change when historical OI is absent. |
| Order-book bands & imbalance | Compute spread and bid/ask quote & base depth at fixed `(10, 50)` bps bands around `mid = (best_bid + best_ask) / 2` (`bid >= mid * (1 - bps/10000)`, `ask <= mid * (1 + bps/10000)`). Imbalance = `(bid_quote - ask_quote) / (bid_quote + ask_quote)`, or `None` if total depth is zero. Reject crossed/locked books (`best_bid >= best_ask`). | Standardizes depth across price scales and treats imbalance as transient liquidity context, not a directional guarantee. |
| Order-book sequence gaps | Detect sequence gaps (`U > last_update_id + 1` on spot diffs, `pu != last_update_id` on futures diffs, or snapshot sequence regressions), attach `SEQUENCE_GAP`, and perform bounded full-snapshot resynchronization. Failed resync emits `SEQUENCE_GAP` + `PROVIDER_ERROR` + `ORDERBOOK_UNAVAILABLE`. | Incremental L2 updates without continuity guarantees silently corrupt local book state. |
| Liquidation events | Provide a separate `get_liquidations` method and typed `LiquidationEvent` storage/normalization. Because Binance public REST API offers no public liquidation endpoint (`fetchLiquidations=False`), default REST calls return `source_available=False` with `None` counts/notionals unless a reliable source is configured. | Missing liquidation sources must never be disguised as zero liquidations. |
| Cross-source price check | Compare contemporaneous prices (`|time_a - time_b| <= 120 s`), such as spot order-book `mid_price` vs perpetual `index_price` or explicit reference prices: `< 50 bps` ok; `50 to < 200 bps` sets `PRICE_MISMATCH` (`DEGRADED`); `>= 200 bps` sets `SEVERE_PRICE_MISMATCH` (`INSUFFICIENT_DATA`). Non-contemporaneous comparisons (> 120 s apart) report `non_comparable_time`. | Surfaces venue/feed dislocations without raising false alarms across asynchronous candle closes and live order books. |
| Persistence | Persist raw venue-specific records in PostgreSQL tables (`funding_rate_observations`, `open_interest_observations`, `order_book_snapshots`, `liquidation_events`), retained as public market data alongside `market_candles`. | Enables reproducible analysis, causal ML backfills (Phase 12+), and auditability of raw units/payloads. |

## Consequences

- `AnalysisService(..., derivatives_provider=None)` continues to produce spot-only `AnalysisSnapshot`s for backward compatibility with Phase 1–8 fixtures and unit tests.
- `CALC_VERSION` bumps to `1.1.0` and `PROMPT_VERSION` bumps to `v1.1.0`.
- Revisit triggers: adding WebSocket streaming order books/liquidations, adding a second derivatives venue, or adding portfolio/margin risk calculations.
