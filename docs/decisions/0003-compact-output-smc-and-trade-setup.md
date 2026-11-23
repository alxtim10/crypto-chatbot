# ADR 0003: Compact default output, Hybrid Crypto + Causal SMC, and deterministic Trade Setup

- Status: Accepted
- Date: 2026-11-21
- Supersedes: none (extends ADR 0001 and ADR 0002)

## Context

After completing Phase 9 (spot + derivatives + order-book liquidity), live output testing highlighted two practical UX and research gaps:
1. Rendering all `[F1..Fn]` backend facts in Telegram produces an overly long mobile message; the user needs a concise key-points brief by default with full detail opt-in.
2. Market analysis is more actionable for crypto research when structured around directional Entry Zone, invalidation Stop Loss, TP1/TP2/TP3, and R:R—or an explicit `WAIT / NO_SETUP` state—using crypto-appropriate Hybrid Causal SMC (liquidity sweeps, BOS/CHoCH, FVG, Order Blocks, Premium/Discount) combined with derivatives/order-book context, inspired by `TauricResearch/TradingAgents` synthesis & settlement logs without slow/quota-heavy multi-agent loops.

## Decisions

| Area | Decision | Rationale |
|---|---|---|
| Output format | Default Telegram `/analyze` (and natural follow-ups) and CLI `analyze` render a concise key-points brief (~15–22 lines). `/analyze <SYMBOL> [TF] detail` and CLI `--detail` render the full `[F1..Fn]` audit dump. | Scannable on mobile while keeping full backend transparency on demand. |
| SMC scope | Add strictly causal, non-repainting Hybrid Crypto + SMC analysis: `BOS`/`CHoCH` on confirmed `k=3` swing pivots, `Liquidity Sweep` (wick pierce + close back inside), unmitigated `FVG` (`>= 10 bps`), causal `Order Block` (origin candle confirmed only on BOS/CHoCH close), and `Dealing Range` (`Discount` < 45%, `Equilibrium` 45–55%, `Premium` > 55%). Exclude FX-specific ICT session killzones. | Crypto trades 24/7 with frequent leverage stop-hunts; strict `confirmation_time` guarantees zero lookahead. |
| Trade setup & levels | Python deterministically computes 1 dominant conditional scenario (`LONG` or `SHORT`: `entry_low`/`entry_high`/`entry_mid`, structural SL + `0.5 × ATR` buffer bounded in `[0.5, 3.5] × ATR`, monotonic `TP1`/`TP2`/`TP3`, and `R:R` ratios with `>= 1.5` to TP2) or returns `WAIT / NO_SETUP` when `RANGING` (without sweep+CHoCH), `HIGH_VOLATILITY + MIXED`, conflicting bias, excessive SL distance, or insufficient R:R. | Keeps all numeric targets 100% deterministic (`{F<n>}`), avoids forced setups in chop, and preserves non-probabilistic research boundaries. |
| LLM synthesis pattern | Adopt `TradingAgents`-style Bull vs Bear + Risk framing inside a **single** structured Gemini call (`gemini-3.5-flash-lite` or pinned Flash model). Do not adopt multi-call LangGraph debate loops. | Keeps latency low (~1–2 s) and preserves free-tier RPD/RPM quota. |
| Setup tracking & settlement | Persist deterministic setups in PostgreSQL (`trade_setups`) and provide closed-candle settlement (`settle-setups`) recording `pending`, `entry_triggered`, `hit_tp1`/`tp2`/`tp3`, `hit_sl`, `expired`, or `skipped_wait` using conservative SL-before-TP intra-bar ordering. | Enables objective historical accuracy auditing and future offline ML validation (Phase 12+). |
| Guard & prompt versioning | Bump `CALC_VERSION` to `1.2.0` and `PROMPT_VERSION` to `v1.2.0`. Output guard allows `TP 1`/`TP 2`/`TP 3` tokens only when `trade_setup_tp_levels` is present in `indicator_parameters`. | Preserves strict digit allowlist against LLM-invented prices. |
| Dynamic Binance `/USDT` symbols (Opsi B) | Accept any valid alphanumeric base ticker (`2–12` chars with at least one ASCII letter `A-Z`, excluding quote/reserved tokens like `USDT`, `USD`, `BUSD`, `EUR`) and normalize to `<BASE>/USDT` spot (`BINANCE:<BASE>USDT`) or `<BASE>/USDT:USDT` perpetual (`BINANCE:<BASE>USDT.P`). | Expands research coverage across Binance `/USDT` altcoins while keeping strict spot/perpetual isolation and graceful `DERIVATIVES_UNAVAILABLE` fallback when no perpetual contract exists. |
| `15m` & Top-Down HTF + LTF `15m` refinement (Mode 2) | Add `Timeframe.M15 = "15m"` for direct `15m` analysis and causal top-down `15m` Entry/SL refinement on `1h`/`4h`/`1d` (`CALC_VERSION = "1.3.0"`). HTF sets bias and macro targets; causal `15m` FVG/OB/EMA20 and `15m` swing pivots tighten Entry/SL bounded in `[0.35, 3.5] × ATR(HTF)`. Direct `15m` checks `4h`/`1h` as causal HTF filter. | Improves entry/SL precision on LTF structure without losing HTF trend discipline or violating zero-lookahead rules. |

## Consequences

- Gemini continues to calculate zero prices, zero SL/TP values, and zero R:R ratios.
- Programmatic callers of `render_deterministic_snapshot(snapshot)` and `render_analysis_response(resp)` retain exact full-detail audit output unless `compact=True` / `render_compact_snapshot` is explicitly used (as Telegram/CLI defaults now do).
- Natural-language follow-up parser recognizes setup/SMC keywords (`entry`, `sl`, `tp`, `smc`, `fvg`, `order block`, `setup`, `funding`, `oi`, `likuiditas`) as well as `/USDT` altcoin tickers and `15m` timeframe references.
- Dynamic `/USDT` symbols degrade gracefully (`PROVIDER_ERROR` / `DERIVATIVES_UNAVAILABLE`) when an exchange pair or perpetual contract is unlisted on Binance.
