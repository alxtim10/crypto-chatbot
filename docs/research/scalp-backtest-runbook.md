# `scalp_v1` Laptop Backtest Runbook (Gate Bulk Data -> `scalp-backtest`)

This runbook produces the 1m candle fixture and runs the pre-registered §9.3 validation report on your laptop using Gate.io perpetual futures bulk archives (`download.gatedata.org/futures_usdt`).

## 1. Prerequisites

Ensure `.env` has a reachable `DATABASE_URL` (local PostgreSQL) and `.venv` is active.

## 2. Ingest Gate Bulk Archives (`ingest-gate-history`)

Use `candlesticks_1m`, `mark_prices`, and `funding_applies` for at least 12 months across the initial `scalp_v1` universe (`BTC,ETH,SOL,XRP,DOGE`). Note that `candlesticks_15m` is unavailable on `download.gatedata.org` (404) — `scalp-backtest` resamples `5m`, `15m`, and `1h` deterministically from `1m`.

```bash
.venv/bin/crypto-chatbot ingest-gate-history \
  --symbols BTC,ETH,SOL,XRP,DOGE \
  --from 202511 \
  --to 202611 \
  --types candlesticks_1m,mark_prices,funding_applies
```

This command:
1. Downloads compressed `.csv.gz` archives into `data/raw/gate/`.
2. Verifies SHA-256 idempotently in `gate_ingest_ledger`.
3. Checks 1m candle contiguity and loads closed `1m` `MarketCandle` rows into PostgreSQL (`market_candles`).

## 3. Export the 1m Fixture JSON (`export-scalp-fixture`)

Export the ingested `1m` perpetual candles from PostgreSQL into the JSON fixture expected by `scalp-backtest`:

```bash
.venv/bin/crypto-chatbot export-scalp-fixture \
  --symbols BTC,ETH,SOL,XRP,DOGE \
  --from 2025-11-01 \
  --to 2026-11-28 \
  --venue gateio \
  --output data/scalp_1m_fixture.json
```

## 4. Run Event-Driven 1m Backtest & Validation Report (`scalp-backtest`)

```bash
.venv/bin/crypto-chatbot scalp-backtest \
  --fixture data/scalp_1m_fixture.json \
  --trial-id scalp_v1-trial-001 \
  --seed 42 \
  --report docs/research/scalp_v1-backtest.md
```

## 5. Freeze Parameters Before Shadow Deploy (`freeze-params`)

Only if `docs/research/scalp_v1-backtest.md` passes §9.3 (`passes_shadow_gate: YES`) and triggers zero §9.4 kill flags:

```bash
.venv/bin/crypto-chatbot freeze-params scalp_v1 --output frozen_params.json
```

Set `SCALP_EQUITY_USD` and `SCALP_RISK_PER_TRADE_PCT` in `.env` (never commit them) for Telegram position-size calculation.
