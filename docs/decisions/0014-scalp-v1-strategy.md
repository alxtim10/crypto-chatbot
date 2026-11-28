# ADR 0014: Pivot Active Alert Strategy to `scalp_v1`

- **Status**: Accepted
- **Date**: 2026-11-28
- **Supersedes**: Swing MTF active default in `config/screener.yaml` (keeps swing MTF code intact and toggleable via `strategy: swing_mtf`)
- **Reference**: `docs/SCALP-SPEC-v1.md`

## Context

The owner pivoted the active alert strategy of `crypto-chatbot` from swing MTF (`4h/1h -> 15m -> 5m`) to a deterministic manual-execution scalping strategy (`scalp_v1`) on Gate.io USDT-settled perpetuals, with holding times from minutes to 90 minutes and strict pre-registered backtest/shadow gates before live trading.

## Decision

1. **Strategy switch (`SC-01`)**: `config/screener.yaml` sets `strategy: scalp_v1`. Legacy swing MTF code remains untouched and tested behind `strategy: swing_mtf`. All ledger and alert tables (`trade_setups`, `alert_runs`, `alerts`, `alert_symbol_states`) carry a `strategy` column so swing and scalp forward clocks are never mixed.
2. **Per-symbol, per-order-type cost model (`SC-02`)**: Entry and TP legs use maker fees; SL and time-stop legs use taker fees + L2 half-spread + L2 depth slippage (with explicit `SPREAD_ESTIMATED`/`SLIPPAGE_ESTIMATED` fallback flags), plus signed 8h funding when a settlement falls inside the holding window.
3. **Universe & cadence (`SC-03`, `SC-12`)**: Weekly eligibility refresh (`config/universe_scalp.txt` + `scalp_universe_snapshots`) restricted to deep Gate USDT perpetuals (initial: BTC, ETH, SOL, XRP, DOGE). Cadence is 5m (`*:00/5:20 UTC`), 1m settlement (`settle-setups --strategy scalp_v1`), and 15-minute dead-man threshold.
4. **Deterministic strategy & PIT dataset (`SC-04`..`SC-07`)**: 1h mode selection (`TREND_PULLBACK_LONG`, `TREND_PULLBACK_SHORT`, `RANGE_FADE`, `NO_TRADE`) via shared `analyze` math, Trend Pullback + Range Fade setups, 8 blocking filters with explicit reason codes, and full point-in-time feature logging (`scalp_decisions`) on every decision (`READY`, `WAIT`, `NO_TRADE`).
5. **Shared 1m settlement & validation (`SC-09`..`SC-11`, `SC-14`)**: `settle_scalp_trade_1m` is shared identically by live settlement and the event-driven 1m backtest engine (with 90s human delay, fill-through-by-1-tick maker rule, SL-before-TP intrabar order, gap-at-open SL, and 90m time stop). Validation enforces random-entry benchmark, daily block-bootstrap 95% CI, cost sensitivity sweep (1.0/1.5/2.0x), DSR, and pre-registered kill criteria. Parameter freeze (`freeze-params scalp_v1`) uses the exact same SHA-256 bundle as `alerts run`.

## Consequences

- All PRD v1.7 invariants remain binding (closed candles only, strict PIT `available_time`, zero LLM calculations, no automated order execution).
- Alerts start in `phase: shadow` with `[SHADOW]` tags until backtest and 2–4 week shadow criteria are satisfied.
