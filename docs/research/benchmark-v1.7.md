# Research Benchmark Report — v1.7 Backtesting Engine

> **Status**: Unconfirmed research benchmark (pre-forward-evaluation). Per PRD v1.7 §1.2 and §7.7, historical backtest numbers are descriptive research measurements only and do not constitute confirmation of live edge until the Forward Evaluation Standard (§8.3) is met on post-freeze setups.

## 1. Engine & Parity Configuration
- **Engine**: PRD v1.7 (`warmup_bars = 600`, `entry_expiry_bars = 6`, strict intrabar event order, same-bar fill+TP1 conservative rule, breakeven SL active from bar `k+1`, stationary Politis-Romano block bootstrap `B = 2,000`, `mean_block_length = 5.0`, `seed = 42`).
- **Dataset**: Deterministic offline parity fixture (`tests/fixtures/parity_1000_candles.json`, `BTC/USDT`, `4h`, 1,000 closed candles).
- **Evaluation Window**: `2026-08-09 00:00:00+00:00` to `2026-10-14 16:00:00+00:00` (after 600-bar warm-up).
- **Trial Registry**: `docs/research/trials-v1.7.jsonl` (`family = "setup_backtest_v17"`, registered trials `M = 2`, `obs_frequency = "per_trade"`).
- **Parity Profile (`BACKTEST_PROFILE`)**: Only `BACKTESTED` components are active (`indicators`, `regimes`, `structure`, `smc`, `failed_breakout`, `ema200_gate`, `cost_gate`, `net_rr_gate`, `historical_funding`). All `FORWARD_ONLY` guards (`l2_orderbook_imbalance_guard`, `whale_positioning`, `live_trade_cvd`, `llm_sentiment`) and `CONTEXT_ONLY` inputs (`dxy_macro_context`) are excluded.

## 2. Research Results Summary

| Variant | `trial_id` | Eval Bars | Trades | Win Rate | Net Exp (R) | Total Net R | Profit Factor | Max DD (R) | BE Exits | No-Fill (TP1 / Exp) | SR / trade | DSR (M=2) | Break-Even Cost Mult |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`v3_v17_engine`** (`entry_expiry=6`, `require_ema200=True`) | `trl-acf21a07801d43af` | 194 | 11 | 72.73% | +0.0308 R | +0.3393 R | 1.0922 | 2.0424 R | 3 | 12 / 1 | +0.0333 | 0.3317 (`False`) | 1.4721x |
| **`v3_legacy_mode`** (`entry_expiry=None`, `require_ema200=False`) | `trl-1e7e53d8d9c04b12` | 201 | 13 | 53.85% | -0.3463 R | -4.5022 R | 0.3317 | 4.8480 R | 4 | 0 / 0 | -0.4324 | 0.0176 (`False`) | -3.9007x |

## 3. Cost Sensitivity Sweep (`v3_v17_engine`, `trl-acf21a07801d43af`)

| Cost Multiplier | Net Expectancy (R) | Total Net R | Profit Factor | Max Drawdown (R) |
| :---: | :---: | :---: | :---: | :---: |
| **1.00x** | +0.0309 R | +0.3397 R | 1.0923 | 2.0423 R |
| **1.25x** | +0.0145 R | +0.1598 R | 1.0428 | 2.0777 R |
| **1.50x** | -0.0018 R | -0.0201 R | 0.9947 | 2.1131 R |
| **2.00x** | -0.0345 R | -0.3798 R | 0.9027 | 2.1839 R |

## 4. Portfolio & Stationary Block-Bootstrap Diagnostics (`B = 2,000`, `L = 5d`, `seed = 42`)

| Variant | Active UTC Days | Portfolio Max DD (R) | Max Concurrent Positions | 95% CI Daily Mean R | 95% CI Trade Expectancy R | Positive Resample Rate | P95 Max DD (R) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`v3_v17_engine`** | 11 | 2.0424 R | 1 | `[-0.2613, +0.3302]` | `[-0.2613, +0.3302]` | 66.05% | 3.4540 R |
| **`v3_legacy_mode`** | 13 | 4.8480 R | 1 | `[-0.5209, -0.1575]` | `[-0.5209, -0.1575]` | 0.15% | 6.6813 R |

## 5. Research Notes
1. Neither trial reaches statistical significance (`DSR >= 0.95`) on this 1,000-bar single-asset sample (`n = 11` and `n = 13` trades); the 95% block-bootstrap CI for `v3_v17_engine` spans zero (`[-0.2613, +0.3302]`).
2. `v3_v17_engine` turns negative at 1.50x execution costs (`break-even cost multiplier = 1.4721x`), highlighting sensitivity to slippage and taker fees.
3. Multi-asset live exchange regeneration (e.g. BingX `4h` / `1d` across the full basket) and post-freeze forward ledger tracking (`crypto-chatbot freeze-params v3`) are required before drawing deployment conclusions.
