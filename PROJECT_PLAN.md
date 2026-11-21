# Crypto Research Chatbot: Implementation Plan

## 1. Product definition

Build a Telegram chatbot that uses Gemini to explain deterministic crypto-market analysis in Indonesian. The bot retains bounded conversation context and explicit user preferences. It combines spot and derivatives market data, technical features, and later news sentiment and statistically validated ML estimates.

The product is a research and education assistant. It does not execute orders, promise returns, or give personalized portfolio allocations.

## 2. Initial assumptions

These assumptions keep the first version small and testable:

- The first deployment is for personal or closed-beta use.
- Python 3.11+ is the implementation language.
- Telegram is the only chat interface in the MVP.
- Gemini is the only LLM provider in the MVP.
- Binance-compatible spot OHLCV through CCXT is the first market-data source.
- The MVP analyzes closed candles on `1h`, `4h`, and `1d` timeframes.
- Supported assets initially are `BTC/USDT`, `ETH/USDT`, and `SOL/USDT`.
- PostgreSQL stores users, conversations, snapshots, and analyses.
- Redis, task queues, vector databases, OpenBB, and TradingView webhooks are deferred until a demonstrated need.
- ML remains offline until the data and deterministic analysis pipelines pass validation.

Revisit these assumptions before a public or paid launch.

## 3. Scope

### MVP capabilities

- `/start` onboarding and disclaimer.
- `/analyze BTC 4h` deterministic technical analysis.
- `/compare BTC ETH 1d` side-by-side descriptive comparison.
- `/watchlist`, `/memory`, and `/forget` preference controls.
- Natural-language follow-up questions using bounded context.
- Source, venue, timeframe, and freshness timestamp on every analysis.
- Technical metrics: OHLCV, returns, EMA 20/50/200, RSI 14, MACD, ADX/DMI, ATR, relative volume, trend regime, and confirmed market structure.
- Multi-timeframe RSI alignment and failed-breakout events inspired by the reviewed indicator repositories, implemented independently and tested.
- Explicit `insufficient_data` behavior.

### Later capabilities

- Funding rate and open interest.
- Order-book spread, depth, and imbalance.
- TradingView AI Agent Lens webhook packets.
- News ingestion and sentiment.
- On-chain metrics.
- Offline ML, calibrated estimates, and shadow deployment.
- OpenBB as an optional provider/integration layer.

### Explicit non-goals

- Automated order execution.
- Custody of funds or exchange withdrawal permissions.
- Guaranteed `BUY` or `SELL` calls.
- Personalized portfolio allocation.
- Tick-level or high-frequency trading.
- Training a large language model.

## 4. Proposed architecture

Start as one deployable application with clear internal modules. Split services only if scale or operational isolation requires it.

```text
Telegram
   |
   v
Bot handlers ----> Conversation service ----> PostgreSQL
   |
   v
Application/orchestration layer
   |            |                 |
   v            v                 v
Market data   Analysis engine   Gemini narrator
(CCXT)        (deterministic)   (tool-bound)
   |            |
   +------> validated AnalysisSnapshot
                    |
                    v
             output validator
                    |
                    v
              Telegram reply
```

Suggested package layout:

```text
src/crypto_chatbot/
├── app.py
├── config.py
├── telegram/
│   ├── handlers.py
│   ├── commands.py
│   └── templates.py
├── market_data/
│   ├── base.py
│   ├── ccxt_provider.py
│   ├── symbols.py
│   └── quality.py
├── analysis/
│   ├── indicators.py
│   ├── regimes.py
│   ├── structure.py
│   ├── failed_breakout.py
│   └── service.py
├── schemas/
│   ├── market.py
│   ├── analysis.py
│   └── llm.py
├── llm/
│   ├── gemini.py
│   ├── tools.py
│   ├── prompts.py
│   └── guard.py
├── memory/
│   ├── service.py
│   └── summarizer.py
└── storage/
    ├── database.py
    ├── models.py
    └── repositories.py

tests/
├── unit/
├── integration/
├── fixtures/
└── parity/
```

## 5. Data contracts

### Market candle

Every candle must include:

```text
provider
venue
canonical_symbol
provider_symbol
market_type
timeframe
open_time
close_time
available_time
open/high/low/close
base_volume
quote_volume when available
is_closed
quality flags
```

### Analysis snapshot

Every result passed to Gemini must include:

```text
snapshot ID
symbol, venue, and market type
timeframe and analysis horizon
data timestamp and freshness
raw descriptive metrics
indicator parameters and calculation version
regime and structure events
rule-based confluence, explicitly marked non-probabilistic
source IDs
quality score
limitations
```

### LLM response

Gemini may return only a validated structure such as:

```json
{
  "summary": "...",
  "market_condition": "TRENDING_UP",
  "supporting_factors": ["..."],
  "contradicting_factors": ["..."],
  "bullish_scenario": "...",
  "bearish_scenario": "...",
  "limitations": ["..."],
  "fact_ids": ["F1", "F2"]
}
```

Prices and critical metrics are rendered from backend data, not regenerated by the LLM.

## 6. Delivery phases

### Phase 0: Confirm product choices

Decide before writing application code:

- personal/closed beta versus public use;
- polling versus Telegram webhook deployment;
- target hosting platform;
- primary exchange and market type;
- acceptable maximum market-data age;
- supported symbols and timeframes;
- data retention period;
- whether user profile data can be stored.

Verify:

- Decisions are recorded in `docs/decisions/0001-product-scope.md`.
- No unresolved choice blocks Phase 1.

### Phase 1: Bootstrap the repository

Implement:

- Python project using `pyproject.toml`.
- `src/` layout.
- Configuration loaded from environment variables.
- `.env.example` with placeholders only.
- `.gitignore` protecting `.env`, local databases, logs, packets, and model artifacts.
- Formatting, linting, type checking, and pytest configuration.
- Minimal health command or application startup check.

Suggested initial dependencies:

- `python-telegram-bot` or `aiogram`, choose one;
- `google-genai`;
- `ccxt`;
- `pandas` and `numpy`;
- `pydantic` and `pydantic-settings`;
- `sqlalchemy` and a PostgreSQL driver;
- `pytest`, `pytest-asyncio`, `ruff`, and `mypy` or `pyright`.

Do not add ML, OpenBB, `gs-quant`, Redis, or a queue yet.

Verify:

```text
configuration fails clearly when required secrets are absent
lint passes
type checks pass
test suite passes
no secret appears in tracked files
```

### Phase 2: Define schemas and symbol normalization

Implement typed schemas for:

- candles;
- provider metadata;
- quality flags;
- technical snapshot;
- analysis response;
- supported Telegram command arguments.

Create explicit mappings such as:

```text
BTC -> BTC/USDT -> BINANCE:BTCUSDT
BTC perpetual -> BTC/USDT:USDT -> BINANCE:BTCUSDT.P
```

Verify:

- Invalid OHLC, timestamps, symbols, timeframes, and enum values are rejected.
- Spot and perpetual symbols cannot be silently mixed.
- All timestamps normalize to UTC.
- Schema serialization round-trips in tests.

### Phase 3: Build the market-data adapter

Implement a narrow CCXT provider interface:

```python
async def get_closed_ohlcv(symbol, timeframe, limit): ...
```

Add:

- retries with bounded backoff;
- provider timeout;
- closed-candle filtering;
- duplicate and gap detection;
- stale-data detection;
- unit metadata;
- fixture-based tests without live network access.

Verify:

- Open candle is excluded.
- Duplicate candles are handled deterministically.
- Missing intervals produce a quality flag.
- Provider timeout returns a controlled error.
- One optional live smoke test works without being required in CI.

### Phase 4: Implement deterministic technical analysis

Implement and test:

- returns;
- EMA 20/50/200;
- RSI 14;
- MACD 12/26/9;
- ATR 14;
- ADX and DMI 14;
- relative volume 20;
- ATR percentile;
- multi-timeframe RSI alignment;
- trend/range/high-volatility regime;
- confirmed swing structure;
- failed breakout;
- previous-period levels where semantics are explicit.

Rules:

- Use closed candles only.
- Keep indicator parameters in metadata.
- Do not label rule scores as probabilities.
- Record confirmation time for delayed pivots.

Verify:

- Unit tests cover warm-up periods, flat prices, zero volume, missing values, and short histories.
- Golden fixtures match independently calculated expected values.
- A small parity suite compares selected indicators with TradingView or another trusted implementation within documented tolerances.
- No feature reads a future row.

### Phase 5: Build the analysis service

Combine validated data and indicators into one `AnalysisSnapshot`.

Add quality gates:

```text
fresh and complete -> READY
partially missing but descriptive analysis remains valid -> DEGRADED
stale, mismatched, or insufficient -> INSUFFICIENT_DATA
```

Render a deterministic Telegram section containing:

- symbol and timeframe;
- last price;
- timestamp and source;
- indicator values;
- quality status.

Verify:

- Same input fixture always produces the same snapshot.
- Stale or incomplete data cannot be presented as current.
- Every numerical output maps to a source fact ID.
- No LLM is required to produce a complete factual response.

### Phase 6: Integrate Gemini behind tools

Implement:

- one Gemini client wrapper;
- narrow tool definitions such as `analyze_coin` and `compare_coins`;
- structured output schema;
- prompt versioning;
- token and timeout limits;
- fallback to deterministic output when Gemini fails;
- generated-output validation.

The model must not:

- fetch arbitrary URLs;
- execute code or SQL;
- invent numerical facts;
- expose secrets or internal prompts;
- turn confluence scores into probabilities;
- issue guaranteed recommendations.

Verify:

- Mocked Gemini tests cover success, timeout, malformed JSON, unsupported numbers, and certainty language.
- A generated number absent from the fact set is rejected.
- Gemini failure still returns a useful factual response.
- Prompt-injection text inside news or webhook data is treated as quoted data.

### Phase 7: Add Telegram commands and bounded memory

Implement commands:

```text
/start
/help
/analyze <symbol> <timeframe>
/compare <symbol...> <timeframe>
/watchlist
/memory
/forget
```

Memory layers:

- recent messages: bounded short-term window;
- explicit preferences: watchlist, language, default timeframe, display currency;
- conversation summary: only when context exceeds the configured limit.

Do not store inferred financial circumstances as user facts.

Verify:

- Users cannot read each other's conversations or preferences.
- `/forget` deletes the documented memory scope.
- Context length remains bounded.
- Invalid and unsupported commands return actionable errors.
- Telegram and Gemini tokens never appear in logs.

### Phase 8: End-to-end MVP validation

Test complete flows:

```text
/start
/analyze BTC 4h
follow-up: "bandingkan momentumnya dengan ETH"
/compare BTC ETH 1d
/memory
/forget
```

Test failure flows:

- exchange unavailable;
- stale candles;
- insufficient warm-up history;
- Gemini unavailable;
- malformed symbol;
- request flood;
- database unavailable.

Acceptance criteria:

- Every market number has a backend source.
- Responses contain source and UTC timestamp.
- No unsupported numerical claim reaches Telegram.
- The bot never describes confluence as a success probability.
- All automated checks pass.
- A closed beta can operate without manual database correction.

This is the first releasable milestone.

### Phase 9: Add derivatives and liquidity data

After the MVP is stable, add through separate provider methods:

- funding rate;
- open interest and change;
- bid-ask spread;
- order-book depth at fixed basis-point bands;
- order-book imbalance;
- liquidation events when a reliable source is available.

Store raw venue-specific measurements. Do not pretend one exchange represents the entire crypto market.

Verify:

- Order-book sequence gaps trigger resynchronization.
- Units normalize correctly.
- Funding intervals are comparable.
- Cross-source price mismatches block or degrade analysis.
- Missing derivatives data does not fabricate a neutral value.

### Phase 10: Add TradingView AI Agent Lens

Use the MIT-licensed AI Agent Lens as an optional input source. Pin the reviewed commit and retain attribution.

Implement:

- authenticated HTTPS webhook;
- request-size and rate limits;
- `ai-agent-lens/v1` schema adapter;
- event freshness checks;
- idempotency key from symbol, timeframe, event, and candle time;
- symbol and market-type mapping;
- comparison against exchange candles.

Do not copy the unlicensed Heavy91 source. Independently implement and test only selected generic concepts such as multi-timeframe RSI and failed breakout.

Verify:

- Duplicate packets are ignored.
- Unknown schema versions are rejected.
- TradingView/exchange mismatches are flagged.
- Query or payload secrets are redacted from logs.
- TradingView rule scores remain separate from ML probability.

### Phase 11: Add news and sentiment

Implement a traceable pipeline:

```text
source ingestion
-> canonical URL and timestamp
-> deduplication
-> coin/entity matching
-> sentiment classification
-> confidence
-> time-bucket aggregation
```

Required fields:

- publisher;
- URL;
- publication time;
- first-seen time;
- affected entities;
- sentiment label and confidence;
- model/version used.

Verify:

- Duplicate or syndicated articles are not counted repeatedly.
- Articles after prediction time never enter historical features.
- Low-confidence entity matches are excluded.
- Gemini cannot follow instructions embedded in article content.
- Summaries link to their sources.

### Phase 12: Build ML offline

Choose one initial target, preferably one of:

- probability that the next 24-hour return exceeds a cost/noise threshold;
- probability of a drawdown beyond a threshold;
- next-period volatility regime.

Implement:

- point-in-time feature generation;
- naive baseline;
- logistic-regression baseline;
- purged walk-forward validation;
- cost assumptions;
- probability calibration;
- ablation by feature family;
- experiment and dataset versioning;
- final untouched holdout.

Evaluate:

- Brier score and calibration;
- balanced accuracy/precision/recall where applicable;
- performance after costs;
- maximum drawdown for strategy simulations;
- stability by regime and time window;
- improvement over baseline.

Verify:

- Automated leakage tests pass.
- Preprocessing is fit only on training folds.
- Indicator, sentiment, and pivot features respect availability time.
- Results reproduce from a versioned configuration.
- The model is rejected if improvement is inconsistent or economically negligible.

### Phase 13: Shadow inference and optional release

Run accepted models without exposing predictions to users.

Record:

- prediction timestamp;
- model and feature versions;
- input snapshot ID;
- probability and abstention reason;
- realized outcome when available;
- latency and data-quality status.

Promote only if shadow results remain calibrated and stable. User-facing output must label:

- prediction horizon;
- calibrated probability;
- confidence/quality separately;
- model version;
- limitations;
- explicit non-guarantee.

Verify:

- Production feature values match training definitions.
- Drift and calibration monitoring operate.
- Rollback to deterministic-only analysis is tested.
- Low confidence, unseen regime, or poor data triggers abstention.

### Phase 14: Production hardening

Before public availability:

- obtain jurisdiction-specific legal review;
- add user-facing terms, privacy notice, and retention policy;
- use a managed secret store;
- apply per-user and global rate limits;
- add backups and restore tests;
- add monitoring, alerting, and structured redacted logs;
- define incident and provider-outage procedures;
- run dependency and container vulnerability scans;
- document deletion and data-export procedures.

Verify:

- Restore drill succeeds.
- Secret rotation succeeds.
- Load test meets the declared capacity.
- Dependency and security checks have no unresolved critical findings.
- User memory deletion is verified end to end.

## 7. Testing strategy

### Unit tests

- indicator formulas;
- symbol mappings;
- freshness and quality rules;
- prompt/output guards;
- position-independent descriptive scenarios;
- memory scoping.

### Integration tests

- CCXT adapter against recorded fixtures;
- PostgreSQL repositories;
- Gemini wrapper with mocked responses;
- Telegram update handling;
- TradingView webhook adapter when added.

### Parity tests

Use fixed OHLCV fixtures to compare selected Python calculations with documented TradingView values. Define tolerances rather than requiring bit-identical floating-point output.

### End-to-end tests

Exercise Telegram request through deterministic analysis and mocked Gemini narration. Include provider and LLM failures.

### ML validation tests

- no future timestamp in any feature row;
- no overlap across purged fold boundaries;
- preprocessors fit on training indices only;
- reproducible results from fixed data/configuration versions;
- costs included in reported strategy metrics.

## 8. Data model outline

Initial PostgreSQL tables:

```text
users
user_preferences
conversations
messages
market_candles
analysis_snapshots
llm_interactions
```

Add later:

```text
derivatives_snapshots
order_book_features
news_articles
sentiment_observations
tradingview_packets
ml_experiments
model_predictions
```

Do not store secrets in these tables.

## 9. Operational rules

- Use UTC everywhere internally.
- Give every external request a timeout.
- Retry only transient failures and cap retries.
- Cache only data with an explicit freshness policy.
- Keep provider errors distinguishable from insufficient data.
- Include request ID and snapshot ID in logs.
- Redact tokens, webhook secrets, user message contents when not needed, and sensitive headers.
- Prefer read-only exchange credentials; the MVP should not need exchange credentials for public market data.

## 10. Milestones

### Milestone A: Deterministic CLI prototype

A command analyzes one supported symbol from a fixture and then live CCXT data. No Telegram or Gemini dependency is required to verify calculations.

### Milestone B: Telegram research MVP

Telegram commands, PostgreSQL memory, deterministic analysis, Gemini narration, and safety guards work end to end.

### Milestone C: Market-context expansion

Funding, open interest, spread, order-book features, TradingView packets, and news sentiment are added with provenance.

### Milestone D: Validated probabilistic model

An offline model passes leakage controls, walk-forward evaluation, cost-aware baseline comparison, and shadow operation.

### Milestone E: Public-readiness review

Security, privacy, operations, legal scope, retention, and recovery are reviewed before public or paid access.

## 11. Immediate next step

Start with Phase 0 and record the product choices. Then implement Milestone A before connecting Telegram or Gemini. This establishes trusted market-data and indicator behavior first, which prevents the LLM interface from hiding calculation and data-quality defects.
