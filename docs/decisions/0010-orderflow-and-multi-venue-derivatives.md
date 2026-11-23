# ADR 0010: Closed-Bar Order-Flow Microstructure & Multi-Venue Derivatives

- Status: Accepted
- Date: 2026-11-22
- Supersedes: none (extends ADR 0001, ADR 0002, ADR 0003, ADR 0009)

## Context

Phase 4 through Phase 10 established deterministic technical analysis, derivatives context (funding rates, open interest, liquidations, order book depth), and causal Smart Money Concepts (SMC) based strictly on Binance closed klines.
However:
1. Entry refinement on `15m` and confirmation of HTF breakout/sweep events (`4h`/`1d`) benefit significantly from closed-bar buyer vs. seller aggressor volume (order flow microstructure). Binance provides `taker_buy_base_asset_volume` on all closed klines.
2. Relying on a single venue (Binance) for derivatives funding rates and mark prices can create venue-specific bias or blind spots during exchange-specific liquidations. Incorporating a second major public derivatives venue (**Bybit** via public read-only CCXT/REST endpoints with zero credentials) enables cross-venue dispersion measurement (`funding_dispersion_bps`, `mark_price_spread_bps`).

## Decisions

| Area | Decision | Rationale |
|---|---|---|
| Order-Flow Microstructure | Parse `taker_buy_base_asset_volume` from closed Binance klines (index 9 in raw kline arrays). Define `taker_buy_ratio = taker_buy_base_volume / base_volume` and 20-bar rolling `cvd_20 = sum(2 * taker_buy_base - base_volume)`. | Operates strictly on closed bars (`close_time <= received_time`), causal and lookahead-free, providing clear volume aggression metrics without requiring streaming tick infrastructure. |
| CVD Divergence Detection | Flag `cvd_divergence` as `bullish` (price makes lower low while CVD-20 makes higher low — absorption) or `bearish` (price makes higher high while CVD-20 makes lower high — exhaustion), or `none`. | Provides an objective measurement factor for SMC liquidity sweeps and failed breakouts (Fact measurements only, never success probabilities). |
| Multi-Venue Derivatives | Add a read-only `BybitDerivativesProvider` fetching public ticker, funding rate, and open interest for matched perpetual pairs (`BTC/USDT:USDT`, `ETH/USDT:USDT`, `SOL/USDT:USDT`) without trading keys. | Surfaces cross-venue dispersion: `funding_dispersion_bps = abs(binance_rate - bybit_rate) * 10000` and `mark_price_spread_bps`. |
| Point-in-Time & Graceful Degradation | Require strict `available_time <= as_of` on cross-venue data. If Bybit API times out or fails (capped at 3.0s timeout, zero retries during interactive requests), mark `coverage_status="partial"` without failing spot analysis. | Protects analysis availability: secondary venue failure never blocks primary technical reporting. |
| ML Integration | Add `FeatureFamily.ORDERFLOW` (`feat_taker_buy_ratio`, `feat_cvd_zscore_20`, `feat_funding_dispersion_bps`) in `ml/features.py` for offline feature ablation. | Allows quantitative validation of order-flow features against deterministic trade setup outcomes. |
| Security Invariants | Zero trading execution, zero exchange trading API keys, zero wallet/signing credentials. All metrics are descriptive measurements, never predictive guarantees. | Preserves ADR 0001 and `AGENTS.md` boundaries completely. |

## Consequences

- `MarketCandle` includes optional `taker_buy_base_volume: FiniteFloat | None = None` (backward-compatible with existing 6-element fixtures).
- `AnalysisSnapshot` is extended with `order_flow: OrderFlowMetrics | None` and `cross_venue_derivatives: CrossVenueDerivativesSummary | None`.
- `CALC_VERSION` is bumped to `1.2.0` and `FEATURE_VERSION` is bumped to `1.2.0`.
- Templates present closed-bar order flow and multi-venue dispersion when available.
