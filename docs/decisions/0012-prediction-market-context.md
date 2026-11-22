# ADR 0012: Read-Only Prediction Market Context (Polymarket / Kalshi)

## Status

Accepted

## Context

Financial market research often benefits from understanding consensus probabilities of discrete external catalysts (e.g., central bank policy rate decisions, regulatory ETF approvals, monthly price milestones). Prediction markets (such as Polymarket and Kalshi) provide real-money order books trading binary outcome shares.

However, incorporating prediction market data into a financial research chatbot carries severe risks:
1. **False Equivalence with Physical Truth**: Contract prices reflect supply, demand, risk premiums, and liquidity constraints; they are NOT objective ground-truth success probabilities.
2. **Illiquidity and Favorite-Longshot Bias**: Longshot contracts often suffer from persistent overpricing due to retail speculation and wide bid-ask spreads.
3. **Lookahead Data Leakage**: Market contract updates must respect point-in-time causal ordering (`available_time <= snapshot.as_of`).
4. **No Execution Policy (`AGENTS.md` Rule 6)**: The system must remain strictly observational; absolutely no wallet credentials, signing keys, order submission, or bet placement may exist.

## Decision

1. **Read-Only Public Ingestion**:
   - Implement read-only public REST adapter (`PolymarketConnector`) targeting public market APIs (e.g. Polymarket Gamma endpoint). Zero API keys or private wallet keys.
   - Enforce short timeouts (3.0 seconds) and graceful degradation: if prediction market endpoints fail or time out, spot and technical analysis proceed without failure.

2. **Strict Point-in-Time Causality & Liquidity Gates**:
   - Enforce `available_time = max(observed_time, received_time)`.
   - Historical snapshot alignment strictly checks `available_time <= as_of`.
   - Markets with 24h volume below \$1,000 USD or bid-ask spread exceeding 500 bps are flagged as low-liquidity or filtered out.

3. **Semantic Topic Mapping**:
   - Categorize markets into discrete tags: `crypto_price`, `macro_rate` (FOMC/CPI), and `regulatory_etf`.
   - Compute macro benchmark indicators (e.g., FOMC rate cut implied odds) without claiming certainty.

4. **Transparent Template Disclosures**:
   - Both compact and detailed reports render prediction market context under an explicit header:
     *"Konteks Pasar Prediksi (Harga Kontrak Tersirat — Bukan Probabilitas Menang Absolut)"*.
   - Output lists source venue, contract question, implied probability (`implied_yes_prob`), volume, and freshness.

## Consequences

- **Positive**: Rich external catalyst perspective combining real-money market expectations with deterministic technical indicators.
- **Negative**: Additional external public REST dependency (mitigated by strict timeout, circuit breaker, and optional toggle).
- **Invariants**: Strictly read-only; zero trade execution; closed-candle temporal integrity preserved.
