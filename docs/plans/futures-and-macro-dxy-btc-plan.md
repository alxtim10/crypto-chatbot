# Plan: Full Perpetual Futures Engine & Macro/Benchmark Intelligence (`DXY` + `BTC`)

## Goal

1. Transition the chatbot from Spot-primary to **100% Perpetual Futures (`MarketType.PERPETUAL`)** across OHLCV candles, closed-bar order flow (Taker Buy Ratio & CVD), L2 order book depth, funding rates, open interest, and trade setups.
2. Add **US Dollar Index (`DXY`)** and **Bitcoin (`BTC/USDT:USDT`)** as deterministic cross-asset analysis variables (trend, relative strength, rolling correlation, beta, and macro headwind/tailwind regime) with strict point-in-time causality (`available_time <= as_of`) and weekend/offline graceful handling.

---

## Core Guardrails (`AGENTS.md`)

- **No Order Execution**: Purely observational and educational futures research. Zero exchange API keys, zero order placement, zero wallet/signing logic.
- **Deterministic Measurements Only**: All DXY, BTC correlation, OI regime, and setup metrics are computed deterministically in Python and exposed via `{F<n>}` facts. Gemini never invents or calculates numbers.
- **Closed Candles & Anti-Lookahead**: Only closed candles (`close_time <= received_time`, `available_time <= as_of`) are used for target futures, BTC benchmark, and DXY series.
- **Weekend & Outage Resilience**: Traditional DXY markets close on weekends. When DXY is closed, the engine uses the last closed Friday bar with an explicit `dxy_market_closed` informational note rather than failing or fabricating bars.

---

## Track 1: 100% Perpetual Futures Primary Engine

### Planned Changes
1. **ADR 0013 (`docs/decisions/0013-perpetual-futures-primary-and-macro-benchmarks.md`)**:
   - Supersede the MVP spot-only default. Set `MarketType.PERPETUAL` (`BASE/USDT:USDT`, `BINANCE:BASEUSDT.P`) as the primary market type for all user commands (`/analyze`, `/ask`, CLI `analyze`, `settle-setups`, `ml-experiment`).
2. **Symbol Resolution (`src/crypto_chatbot/market_data/symbols.py`)**:
   - Add `resolve_default_perp_symbol(raw: str) -> SymbolMapping` resolving bare inputs (`BTC`, `ETH`, `SOL`, `DOGE`, `SUI`, `PEPE`, `BTCUSDT`) directly to `MarketType.PERPETUAL` (`BASE/USDT:USDT`, `BINANCE:BASEUSDT.P`).
3. **USD-M Futures OHLCV & Native Taker Buy Volume (`src/crypto_chatbot/market_data/ccxt_provider.py`)**:
   - Switch `CcxtMarketDataProvider` default client to `ccxt_async.binanceusdm` (`options={"defaultType": "swap"}`).
   - Extract `taker_buy_base_volume` directly from Binance USD-M raw kline responses (index 9 of `/fapi/v1/klines`) so closed-bar Order Flow (`taker_buy_ratio`, `cvd_20`, `cvd_divergence`) is populated live on every timeframe (`1d`, `4h`, `1h`, `15m`).
4. **Perpetual Order Book Default (`src/crypto_chatbot/market_data/derivatives_provider.py`)**:
   - Default L2 order book depth (`depth_10bps`, `depth_50bps`, `imbalance`, `spread_bps`) to `MarketType.PERPETUAL` on Binance USD-M instead of spot.

---

## Track 2: Futures Positioning Matrix & Long/Short Setup Mechanics

### Planned Changes
1. **Derivatives Positioning Regime (`src/crypto_chatbot/analysis/derivatives.py` & `schemas/analysis.py`)**:
   - Add `PositioningRegime(StrEnum)` classifying the causal interaction of Price Change, Open Interest Change (`oi_change_pct`), Closed-Bar CVD (`cvd_20`), and 8h Funding Rate:
     - `LONG_BUILDUP`: Price rising + OI rising + positive/neutral CVD.
     - `SHORT_BUILDUP`: Price falling + OI rising + negative/neutral CVD.
     - `SHORT_COVERING`: Price rising + OI falling (short squeeze / covering rally).
     - `LONG_LIQUIDATION_FLUSH`: Price falling + OI dropping sharply (long flush / capitulation).
     - `CROWDED_LONG_SQUEEZE_RISK` / `CROWDED_SHORT_SQUEEZE_RISK`: Extreme funding rate (> +3 bps or < -3 bps per 8h) combined with elevated OI.
     - `NEUTRAL_POSITIONING`: Balanced or insufficient OI delta.
2. **Futures-Specific Setup Enhancements (`src/crypto_chatbot/analysis/setup.py`)**:
   - Explicitly frame setups as **`LONG`** (`SetupBias.BULLISH`) or **`SHORT`** (`SetupBias.BEARISH`).
   - Add `stop_loss_distance_pct` (percentage distance from entry mid to invalidation SL) and `max_theoretical_leverage` ($\lfloor 100 / \text{stop\_loss\_distance\_pct} \rfloor$ capped with a safety margin, presented strictly as a mathematical liquidation-distance reference, never personalized sizing advice).
   - Add Mark-to-Last basis awareness: if `abs(basis_bps)` is elevated, include an explicit warning factor on stop-loss buffer sensitivity.

---

## Track 3: `DXY` (US Dollar Index) & `BTC` Benchmark Variables

### Planned Changes
1. **Macro & Benchmark Schemas (`src/crypto_chatbot/schemas/macro.py`)**:
   - `DxySnapshotMetrics`:
     - `symbol`: `"DXY"`
     - `as_of`: last closed DXY candle timestamp (UTC)
     - `is_weekend_or_closed`: `bool` (True when traditional FX market is paused and Friday close is carried forward)
     - `last_close`: `FiniteFloat`
     - `change_1_bar_pct`: `FiniteFloat | None`
     - `change_20_bar_pct`: `FiniteFloat | None`
     - `ema_20`: `FiniteFloat | None`
     - `ema_50`: `FiniteFloat | None`
     - `rsi_14`: `FiniteFloat | None`
     - `trend`: `Literal["bullish", "bearish", "neutral"]`
     - `rolling_corr_20`: `FiniteFloat | None` (20-bar Pearson correlation between target returns and temporally aligned DXY returns, bounded $[-1, 1]$)
     - `macro_impact`: `Literal["headwind", "tailwind", "neutral"]` (e.g., rising DXY with negative correlation = `"headwind"` for crypto futures; falling DXY = `"tailwind"`)
   - `BtcBenchmarkMetrics`:
     - `canonical_symbol`: `"BTC/USDT:USDT"`
     - `as_of`: `UtcDatetime`
     - `last_close`: `FiniteFloat`
     - `change_20_bar_pct`: `FiniteFloat | None`
     - `regime`: `Regime | None`
     - `market_structure`: `MarketStructure | None`
     - `relative_strength_20_pct`: `FiniteFloat | None` (`target_ret_20 - btc_ret_20` when target is an altcoin; `0.0` for BTC)
     - `rolling_corr_20`: `FiniteFloat | None` (20-bar return correlation with BTC)
     - `rolling_beta_20`: `FiniteFloat | None` ($\text{Cov}(r_{\text{alt}}, r_{\text{btc}}) / \text{Var}(r_{\text{btc}})$)
     - `leader_alignment`: `Literal["aligned", "divergent_warning", "self"]`
   - `MacroBenchmarkContext`: Combines `dxy: DxySnapshotMetrics | None` and `btc_benchmark: BtcBenchmarkMetrics | None`.
2. **Read-Only DXY & BTC Benchmark Provider (`src/crypto_chatbot/market_data/macro_provider.py`)**:
   - **DXY Adapter**:
     - Primary public source: Yahoo Finance v8 public chart endpoint (`DX-Y.NYB`) via `httpx.AsyncClient` with strict 3.0s timeout, closed-bar filtering (`close_time <= received_time`), and zero credentials.
     - Secondary fallback: Cached/persisted TradingView `TVC:DXY` / `DXY` packet if available in PostgreSQL.
     - Graceful degradation: If DXY endpoint times out or is unreachable, `dxy` is set to `None` with an informational limitation note—never blocking futures analysis.
   - **BTC Benchmark Adapter**:
     - Reuses `CcxtMarketDataProvider` (`BTC/USDT:USDT` perpetual) on the target timeframe. When the user analyzes `BTC`, no duplicate fetch is needed (reuses the primary batch); when the user analyzes an altcoin (`ETH`, `SOL`, `SUI`, etc.), fetches `BTC/USDT:USDT` concurrently via `asyncio.gather`.
3. **Deterministic Cross-Asset Analysis (`src/crypto_chatbot/analysis/macro.py`)**:
   - Causal timestamp alignment: aligns target closed candles with the latest closed DXY and BTC candles where `candle.close_time <= target_candle.close_time` (zero lookahead).
   - Computes 20-bar Pearson correlation, rolling beta, relative strength, DXY headwind/tailwind classification, and rule factors (`dxy_headwind`, `dxy_tailwind`, `btc_leader_conflict`, `alt_outperforming_btc`).

---

## Track 4: Facts, Templates, CLI & Verification

### Planned Changes
1. **Fact Wiring (`src/crypto_chatbot/analysis/service.py`)**:
   - Register deterministic `{F<n>}` facts for:
     - `dxy_last_close`, `dxy_change_20_bar_pct`, `dxy_rsi_14`, `dxy_rolling_corr_20`
     - `btc_benchmark_close`, `alt_relative_strength_vs_btc_20`, `alt_beta_vs_btc_20`, `alt_corr_vs_btc_20`
     - `setup_stop_loss_distance_pct`
   - Bump `CALC_VERSION = "1.5.0"`.
2. **Indonesian Telegram & CLI Templates (`src/crypto_chatbot/bot/templates.py`)**:
   - Update header to explicitly show **`PERPETUAL FUTURES`** (`BINANCE BTC/USDT:USDT`).
   - Render **`Rezim Posisi Derivatif`** (e.g., *Long Build-up*, *Short Covering*, *Long Liquidation Flush*, *Risiko Squeeze*).
   - Render **`Variabel Makro & Benchmark (DXY & BTC)`**:
     - For `BTC`: DXY price, DXY 20-bar trend & RSI, 20-bar BTC-DXY correlation, and Headwind/Tailwind status (plus weekend closed notice if applicable).
     - For Altcoins (`ETH`, `SOL`, etc.): All DXY metrics + BTC Perpetual regime/trend, Relative Strength vs BTC (`+X,XX% Outperform` / `-X,XX% Underperform`), Correlation & Beta vs BTC, and any leader-conflict warning on the setup.
3. **Tests**:
   - `tests/unit/test_futures_and_macro_dxy_btc.py`:
     - Perpetual-first symbol resolution for `BTC`, `ETH`, `SOL`, `DOGE`, `BTC/USDT:USDT`.
     - Derivatives positioning regime matrix classification across all 6 cases.
     - DXY normalization, causal timestamp alignment, weekend carry-forward without lookahead, and Pearson correlation / Beta math.
     - Altcoin vs BTC relative strength and leader-conflict warning on trade setups.
     - Template rendering verification for both BTC and Altcoin perpetual snapshots.
