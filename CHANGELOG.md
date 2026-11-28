# Changelog

All notable changes to `crypto-chatbot` are documented in this file.

## [Unreleased]

### Added — `scalp_v1` Strategy Implementation (`docs/SCALP-SPEC-v1.md`, ADR 0014)
- **SC-01**: Added `strategy: scalp_v1 | swing_mtf` switch in `config/screener.yaml` and `ScreenerConfig`; added `strategy` column across `trade_setups`, `alert_runs`, `alerts`, and `alert_symbol_states`; routed `alerts run` by strategy while keeping swing MTF intact and disabled by config.
- **SC-02**: Added per-symbol, per-order-type execution cost model (`src/crypto_chatbot/scalp/costs.py`) with Gate VIP0 maker/taker fees, L2 half-spread & depth slippage, signed 8h funding, fallback flags (`SPREAD_ESTIMATED`, `SLIPPAGE_ESTIMATED`), and legacy flat-model toggle.
- **SC-03**: Added weekly scalp universe eligibility engine (`src/crypto_chatbot/scalp/universe.py`), `scalp_universe_snapshots` PIT table, `config/universe_scalp.txt`, and `crypto-chatbot universe refresh --strategy scalp_v1`.
- **SC-04**: Added 1h context mode selector (`src/crypto_chatbot/scalp/mode.py`) parity-tested against `AnalysisService.analyze` 1h (`TREND_PULLBACK_LONG`, `TREND_PULLBACK_SHORT`, `RANGE_FADE`, `NO_TRADE`).
- **SC-05**: Added Trend Pullback and Range Fade deterministic setup engines (`src/crypto_chatbot/scalp/setups.py`) with min/max `R_bps`, 15% cost gate, net R:R >= 1.3, and 15m limit-order validity.
- **SC-06**: Added all 8 blocking filters (`src/crypto_chatbot/scalp/filters.py`) and `config/event_calendar.yaml` with explicit `FilterReasonCode` values.
- **SC-07**: Added point-in-time feature extraction (`src/crypto_chatbot/scalp/features.py`) and `scalp_decisions` table recording features on every `READY`, `WAIT`, and `NO_TRADE` decision.
- **SC-08**: Added `scalp_v1` Telegram alert renderer, owner-config position size calculator, `[SHADOW]` phase badge, `[Pasang] [Skip] [Mute 1j] [Detail]` inline buttons (`al:ps:`, `al:m1:`), and settlement thread replies (`src/crypto_chatbot/scalp/alerts.py`).
- **SC-09**: Added shared 1m settlement engine (`src/crypto_chatbot/scalp/settlement.py`) used identically by `settle-setups --strategy scalp_v1` and the 1m backtest engine.
- **SC-10 & SC-11**: Added 1m contiguity audit, event-driven 1m backtest engine with 90s human delay, random-entry benchmark, daily block-bootstrap 95% CI, DSR, cost sensitivity sweep (1.0/1.5/2.0x), kill-criteria checks, and `crypto-chatbot scalp-backtest` report generator (`src/crypto_chatbot/scalp/backtest.py`).
- **SC-12**: Updated `deploy/gen-systemd.sh` to the 5m scalp cadence (`*:00/5:20`), 1m settlement, `*:01/5:00` contract-stats, weekly universe refresh, and 15-minute dead-man monitor threshold.
- **SC-13**: Added scalp daily digest aggregation and HTML renderer per mode, per symbol, filters fired, stage failures, and fill rate.
- **SC-14**: Unified `scalp_v1` parameter freeze hash (`src/crypto_chatbot/scalp/freeze.py`) across `freeze-params scalp_v1` and `alerts run`.
