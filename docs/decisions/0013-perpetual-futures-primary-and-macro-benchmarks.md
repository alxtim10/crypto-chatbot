# ADR 0013: Perpetual Futures Primary Engine and Cross-Asset Macro Benchmarks (`DXY` + `BTC`)

## Status

Accepted

## Context

The initial MVP of `crypto-chatbot` used Spot market data as its primary baseline (`MarketType.SPOT`, `BASE/USDT`), layering Perpetual Futures as a secondary context. However, active crypto derivatives analysis requires:
1. Primary price, OHLCV candles, and order flow metrics to reflect the actual Perpetual Futures contract (`BINANCE:BASEUSDT.P`, `MarketType.PERPETUAL`) where volume, wicks, and liquidations occur.
2. Direct derivation of Taker Buy Base Volume and Cumulative Volume Delta (CVD) natively from the futures exchange kline streams (`/fapi/v1/klines`).
3. Explicit analysis of futures positioning mechanics (Price + Open Interest + Funding + CVD) to distinguish Organic Trend Build-up, Short Covering, Long Flushes, and Crowded Squeeze Risks.
4. Macro and benchmark context: The US Dollar Index (`DXY`) heavily influences global liquidity and crypto asset valuation, while Bitcoin (`BTC/USDT:USDT`) serves as the dominant market benchmark for altcoin relative strength, beta, and leader directional alignment.

## Decision

1. **100% Perpetual Futures Primary Engine**:
   - Make `MarketType.PERPETUAL` the default market type for symbol resolution (`resolve_default_symbol`), bot commands (`/analyze`, `/ask`), and CLI operations.
   - Configure `CcxtMarketDataProvider` to connect by default to Binance USD-M Futures (`ccxt.async_support.binanceusdm`) for raw OHLCV and closed-bar order flow.
   - Configure L2 Order Book depth snapshots to default to Perpetual Futures.

2. **Futures Positioning Regime Matrix**:
   - Define deterministic `PositioningRegime` evaluating changes in Price, Open Interest (`oi_change_pct`), rolling 20-bar CVD (`cvd_20`), and 8-hour Funding Rate into discrete states: `LONG_BUILDUP`, `SHORT_BUILDUP`, `SHORT_COVERING`, `LONG_LIQUIDATION_FLUSH`, `CROWDED_SQUEEZE_RISK`, and `NEUTRAL_POSITIONING`.

3. **Futures Long/Short Trade Setup Mechanics**:
   - Explicitly label conditional setups as **`LONG`** (`SetupBias.BULLISH`) or **`SHORT`** (`SetupBias.BEARISH`).
   - Calculate `stop_loss_distance_pct` and mathematical `max_theoretical_leverage` based strictly on distance to liquidation point (as an educational risk reference, never financial advice).

4. **Macro Variables (`DXY` & `BTC` Benchmark)**:
   - Ingest closed-candle US Dollar Index (`DXY`) data via public Yahoo Finance API (`DX-Y.NYB`) or TradingView fallback with 3.0-second timeout and graceful degradation.
   - For weekends when traditional FX markets close, automatically carry forward the last closed Friday candle with explicit disclosure `is_weekend_or_closed=True`.
   - Concurrently fetch `BTC/USDT:USDT` perpetual benchmark for altcoin analysis to compute 20-bar rolling correlation, beta, relative strength vs BTC, and flag leader-conflict risks when an altcoin setup conflicts with BTC directional momentum.

## Consequences

- **Positive**: Direct alignment with crypto futures traders; native closed-bar CVD on all timeframes; rich macro DXY correlation and BTC beta context; explicit Long/Short setup framing.
- **Negative**: Adds Yahoo Finance public endpoint dependency for DXY (mitigated by strict timeout, caching, and fallback so analysis never fails if DXY is unreachable).
- **Invariants**: Strictly read-only; zero exchange private keys or trade execution; strictly closed candles with anti-lookahead checks (`available_time <= as_of`).
