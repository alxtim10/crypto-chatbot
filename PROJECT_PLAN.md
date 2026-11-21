# Crypto Research Chatbot: Implementation Plan

Related documents:

- `docs/decisions/0001-product-scope.md`: accepted Phase 0 product decisions (binding).
- `AGENTS.md`: always-on agent rules and project commands.
- `.agents/skills/crypto-research-engineer/SKILL.md`: engineering rules.
- `.agents/skills/socratic-tutor/SKILL.md`: opt-in tutoring mode.

## 0. Phase status

Update this table in the completion report of every phase. Status values: `Not started`, `In progress`, `Done`, `Blocked`.

| Phase | Name | Status |
|---|---|---|
| 0 | Confirm product choices | Done (ADR 0001; open actions tracked there) |
| 1 | Bootstrap the repository | Done |
| 2 | Schemas and symbol normalization | Done |
| 3 | Market-data adapter | Done |
| 4 | Deterministic technical analysis | Done |
| 5 | Analysis service and CLI (Milestone A) | Done |
| 6 | Gemini behind tools | Done |
| 7 | Telegram commands and bounded memory | Done |
| 8 | End-to-end MVP validation (Milestone B) | Done (VPS live check pending per ADR 0001) |
| 9 | Derivatives and liquidity data | Not started |
| 10 | TradingView AI Agent Lens | Not started |
| 11 | News and sentiment | Not started |
| 12 | Offline ML | Not started |
| 13 | Shadow inference and optional release | Not started |
| 14 | Production hardening | Not started |

## 1. Product definition

Build a Telegram chatbot that uses Gemini to explain deterministic crypto-market analysis in Indonesian. The bot retains bounded conversation context and explicit user preferences. It combines spot and derivatives market data, technical features, and later news sentiment and statistically validated ML estimates.

The product is a research and education assistant. It does not execute orders, promise returns, or give personalized portfolio allocations.

## 2. Accepted assumptions

Recorded in ADR 0001. Summary:

- Personal / closed-beta use; access limited to an allowlist of Telegram numeric user IDs.
- Python 3.14 locally and in Docker; `requires-python = ">=3.12"`.
- Telegram is the only chat interface, via `python-telegram-bot` 22.x, using long polling locally and in closed-beta production. Webhook only at public launch.
- Gemini is the only LLM provider: a Flash-class model with the exact model ID pinned in config, on the paid tier before any non-owner user.
- Binance spot public OHLCV through CCXT is the first market-data source; no exchange credentials.
- Closed candles on `1h`, `4h`, and `1d`; assets `BTC/USDT`, `ETH/USDT`, `SOL/USDT`.
- PostgreSQL stores users, preferences, conversations, candles, snapshots, and LLM interactions.
- Memory: last 20 messages + explicit preferences; no LLM summary in MVP.
- Answers in Indonesian; timestamps shown in UTC and WIB.
- Docker on a small VPS in a region verified to reach Binance.
- Redis, task queues, vector databases, OpenBB, and TradingView webhooks are deferred until a demonstrated need.
- ML remains offline until the data and deterministic analysis pipelines pass validation.

Revisit these assumptions before a public or paid launch.

## 3. Scope

### MVP capabilities

- `/start` onboarding and disclaimer; `/help`.
- `/analyze BTC 4h` deterministic technical analysis.
- `/compare BTC ETH 1d` side-by-side descriptive comparison.
- `/watchlist`, `/memory`, and `/forget` preference controls.
- Natural-language follow-up questions using bounded context. A follow-up that needs market facts re-runs an analysis tool and gets a fresh snapshot.
- Source, venue, timeframe, and freshness timestamp (UTC and WIB) on every analysis.
- Technical metrics: OHLCV, returns, EMA 20/50/200, RSI 14, MACD, ADX/DMI, ATR, relative volume, ATR percentile, trend regime, and confirmed market structure.
- Multi-timeframe RSI alignment and failed-breakout events inspired by the reviewed indicator repositories, implemented independently and tested.
- Non-probabilistic factor list (supporting / contradicting rules). No aggregate score.
- Explicit `insufficient_data` behavior.
- User allowlist, per-user rate limits, and daily Gemini call caps.

### Later capabilities

- Funding rate and open interest.
- Order-book spread, depth, and imbalance.
- TradingView AI Agent Lens webhook packets.
- News ingestion and sentiment.
- On-chain metrics.
- Offline ML, calibrated estimates, and shadow deployment.
- OpenBB as an optional provider/integration layer.
- Aggregate confluence score, only with a documented formula and ablation evidence.
- Conversation summary, only if summaries are fact-free and tested.
- Telegram webhook transport (public launch).

### Explicit non-goals

- Automated order execution.
- Custody of funds or exchange withdrawal permissions.
- Guaranteed `BUY` or `SELL` calls.
- Personalized portfolio allocation.
- LLM-generated price targets or levels.
- Tick-level or high-frequency trading.
- Training a large language model.

## 4. Proposed architecture

Start as one deployable application with clear internal modules. Split services only if scale or operational isolation requires it.

```text
Telegram (long polling)
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
   +------> validated AnalysisSnapshot (facts F1..Fn)
                    |
                    v
             output validator + placeholder renderer
                    |
                    v
              Telegram reply (HTML)
```

Suggested package layout:

```text
src/crypto_chatbot/
├── app.py
├── cli.py                 # Milestone A: analyze from fixture or live data; purge-expired
├── config.py
├── bot/                   # named bot/ to avoid confusion with the `telegram` library
│   ├── handlers.py
│   ├── commands.py
│   ├── access.py          # allowlist and rate limits
│   └── templates.py       # HTML rendering, escaping, 4096-char splitting
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
│   ├── factors.py
│   └── service.py
├── schemas/
│   ├── market.py
│   ├── analysis.py
│   └── llm.py
├── llm/
│   ├── gemini.py
│   ├── tools.py
│   ├── prompts.py
│   ├── guard.py           # digit allowlist, overconfidence lexicon, fact-id checks
│   └── render.py          # {F<n>} placeholder substitution
├── memory/
│   └── service.py
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

### 5.1 Time semantics

- All internal timestamps are timezone-aware UTC. The host clock must be NTP-synchronized.
- `open_time`: candle start, inclusive.
- `close_time = open_time + timeframe duration`, exclusive end. Binance's own close time (`open + duration − 1 ms`) is not used.
- `received_time`: when the system received the data.
- `is_closed` is true only if `close_time <= received_time`. CCXT returns the currently open candle as the last row; it must be excluded.
- `available_time`: earliest time the observation may be used.
  - Live data: `max(close_time, received_time)`.
  - Historical backfill: `close_time + BACKFILL_AVAILABILITY_LAG` (default 60 s), so backtests never assume data was usable at the instant the candle closed.
- User-facing timestamps show UTC and WIB (UTC+7). Binance `1d` candles close at 00:00 UTC = 07:00 WIB.

### 5.2 Market candle

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
received_time
available_time
open/high/low/close
base_volume
quote_volume when available
is_closed
quality flags
```

### 5.3 Freshness

The latest closed candle is stale if:

```text
now − close_time > timeframe duration + FRESHNESS_GRACE   (default 5 minutes)
```

Thresholds: `1h` → 65 min, `4h` → 4 h 5 min, `1d` → 24 h 5 min. Stale data yields `INSUFFICIENT_DATA` for current-state analysis.

### 5.4 History and warm-up

- Fetch `limit = 1000` closed candles (one Binance request; no pagination in MVP).
- Per-indicator minimum history is 3× its longest period, so EMA 200 needs 600 candles.
- `>= 600` closed candles: all indicators available.
- `150–599`: EMA 200 and anything depending on it is reported unavailable; snapshot is `DEGRADED`.
- `< 150`: `INSUFFICIENT_DATA`.
- Parity tests use the full 1000-candle window to minimize seed effects.

### 5.5 Multi-timeframe alignment

- Analysis `as_of` = `close_time` of the latest closed candle of the requested timeframe.
- Live analysis: a candle from another timeframe is used only if its `close_time <= as_of`.
- Historical / ML: a candle is used only if its `available_time <= decision_time`.
- Example: a `1h` analysis with `as_of = 05:00 UTC` uses the `4h` candle 00:00–04:00, not 04:00–08:00.

### 5.6 Calculation conventions

`CALC_VERSION` is a string recorded in every snapshot. Bump it whenever any formula, parameter, or threshold below changes.

| Metric | Definition |
|---|---|
| Returns | Simple returns `close_t / close_{t−n} − 1` for `n = 1` and `n = 20` bars. |
| EMA n | `α = 2/(n+1)`, seeded with the SMA of the first `n` values (TradingView `ta.ema`). |
| RMA n (Wilder) | `α = 1/n`, seeded with the SMA of the first `n` values (TradingView `ta.rma`). |
| RSI 14 | `RS = RMA14(gains) / RMA14(losses)`; `RSI = 100 − 100/(1+RS)`. If average loss = 0 → 100; else if average gain = 0 → 0 (TradingView semantics). Flat prices also set flag `FLAT_PRICE`. |
| MACD 12/26/9 | `MACD = EMA12 − EMA26`; `signal = EMA9(MACD)`; `histogram = MACD − signal`. |
| ATR 14 | `TR = max(H−L, |H−C_prev|, |L−C_prev|)`; `ATR = RMA14(TR)`; `ATR% = ATR / close`. |
| ADX / DMI 14 | Wilder `+DM`/`−DM`; `+DI = 100·RMA14(+DM)/ATR14`, `−DI` likewise; `DX = 100·|+DI − −DI| / (+DI + −DI)` (0 if the sum is 0); `ADX = RMA14(DX)`. |
| Relative volume 20 | `base_volume_t / mean(base_volume of the previous 20 closed candles, excluding t)`. Unavailable if the mean is 0 (flag `ZERO_VOLUME`). |
| ATR percentile | Percentile rank (0–100) of current `ATR%` within the last 100 closed candles, inclusive. |
| MTF RSI alignment | RSI 14 on `1h`, `4h`, `1d` at `as_of`. `aligned_bullish` if all > 50, `aligned_bearish` if all < 50, otherwise `mixed`. |
| Regime | `high_volatility` if ATR percentile ≥ 80; else `trending_up` if ADX ≥ 25, `+DI > −DI`, and `close > EMA50`; `trending_down` symmetric; else `ranging` if ADX < 20; else `transitional`. Thresholds are fixed initial values, not optimized on data. |
| Swing pivot | Pivot high at bar `i` if `high_i` is strictly greater than the highs of `k = 3` bars on each side; pivot low symmetric. Confirmed at bar `i+3`; `confirmation_time = close_time` of bar `i+3`. Never used before `confirmation_time`. |
| Market structure | From the last two confirmed pivot highs and lows: `HH`/`LH` and `HL`/`LL`. `bullish` = HH + HL, `bearish` = LH + LL, otherwise `mixed`. |
| Failed breakout | Bullish failure: a bar closes above the most recent confirmed swing high, then within the next 3 bars a bar closes back below that level. Bearish failure is symmetric with the swing low. Event time = `close_time` of the re-entry bar. |
| Previous-period levels | Previous closed UTC day and ISO week high/low/close (Binance-aligned boundaries). |

### 5.7 Quality

Quality is a status plus flags, never a numeric score.

- Flags: `STALE`, `GAP`, `DUPLICATE`, `INVALID_OHLC`, `SHORT_HISTORY`, `FLAT_PRICE`, `ZERO_VOLUME`, `SYMBOL_MISMATCH`, `PROVIDER_ERROR`.
- Status:
  - `READY`: fresh, complete, `>= 600` candles, no blocking flags.
  - `DEGRADED`: descriptive analysis remains valid but some metrics are unavailable (e.g. `SHORT_HISTORY`, `ZERO_VOLUME`).
  - `INSUFFICIENT_DATA`: `STALE`, `INVALID_OHLC`, `SYMBOL_MISMATCH`, unresolved `GAP` in the indicator window, `< 150` candles, or `PROVIDER_ERROR`.
- Provider errors stay distinguishable from insufficient data in logs and in the snapshot.

### 5.8 Analysis snapshot

Every result passed to Gemini must include:

```text
snapshot_id
provider, venue, symbol, market type
timeframe and as_of
input range: first/last open_time, candle count, SHA-256 of canonical candle serialization
calc_version and indicator parameters
facts: [{fact_id, label, value, unit, display_format, source, as_of}]
regime, market structure, events (with confirmation_time)
factor list: [{rule_id, description, direction, fact_ids}], explicitly non-probabilistic
quality status and flags
limitations
```

`direction` is one of `supports_bullish`, `supports_bearish`, `neutral`.

### 5.9 LLM response

Gemini may return only a validated structure:

```json
{
  "summary": "Harga terakhir {F1} berada di atas EMA 200 ({F4}).",
  "supporting_factors": ["..."],
  "contradicting_factors": ["..."],
  "bullish_scenario": "...",
  "bearish_scenario": "...",
  "limitations": ["..."],
  "fact_ids": ["F1", "F4"]
}
```

Validation rules:

- Regime, structure, prices, and critical metrics are rendered by the backend. The schema has no `market_condition` field.
- Numbers appear only through `{F<n>}` placeholders; the backend substitutes formatted values (Indonesian number format).
- Digits outside placeholders are allowed only in allowlisted tokens: indicator labels whose parameters exist in the snapshot (e.g. `EMA 200`, `RSI 14`, `MACD 12/26/9`), supported timeframes (`1h`, `4h`, `1d`), and supported symbols/pairs. Any other digit sequence is rejected.
- Every placeholder must reference a fact in the current snapshot, and every referenced fact must appear in `fact_ids`.
- Scenarios may reference only levels that exist as facts (e.g. a confirmed swing high). No LLM-created targets.
- Overconfidence lexicon, case-insensitive with word boundaries (initial list; extend with tests): `pasti`, `dijamin`, `jaminan`, `garansi`, `tanpa risiko`, `bebas risiko`, `wajib beli`, `wajib jual`, `beli sekarang`, `jual sekarang`, `all in`, `all-in`, `cuan pasti`, `profit pasti`, `guaranteed`, `risk-free`, `sure win`, `to the moon`. Negated uncertainty forms such as `tidak pasti` and `ketidakpastian` are allowed.
- Any violation discards the narration; the user receives the deterministic template output, and the validation failure is logged.

## 6. Delivery phases

### Phase 0: Confirm product choices — Done

Decisions are recorded in `docs/decisions/0001-product-scope.md`. Open actions (Binance reachability from the VPS region, Gemini model pin and billing, beta allowlist) are tracked there and in the phases below.

### Phase 1: Bootstrap the repository

Implement:

- Git repository initialized before any `.env` file exists.
- Python project using `pyproject.toml` with `src/` layout and `requires-python = ">=3.12"`.
- Dependencies reconciled with the existing `.venv`:
  - runtime: `python-telegram-bot` 22.x, `google-genai`, `ccxt`, `pandas`, `numpy`, `pydantic`, `pydantic-settings`, `sqlalchemy` 2.x, `psycopg` 3, `tenacity`;
  - dev: `pytest`, `pytest-asyncio`, `ruff`, `mypy`, `pandas-stubs`.
- Remove `uvloop` and `aiosqlite` from the environment unless a concrete need is documented. Integration tests use PostgreSQL.
- Recreate `.venv` from `pyproject.toml` to prove reproducibility; record exact versions in `requirements.lock`.
- Configuration loaded from environment variables via `pydantic-settings`; secrets typed as `SecretStr`.
- `.env.example` with placeholders only.
- `.gitignore` protecting `.env`, local databases, logs, packets, and model artifacts.
- Ruff (lint + format), mypy, and pytest configuration; `live` pytest marker for opt-in network tests.
- Minimal health command or application startup check.
- Commands documented in `AGENTS.md`.

Do not add ML, OpenBB, `gs-quant`, Redis, or a queue yet.

Verify:

```text
configuration fails clearly when required secrets are absent
secrets are not shown in repr/str of the config object
lint and format checks pass
type checks pass
test suite passes
fresh venv installs from pyproject.toml and tests pass
git ls-files contains no .env or secret-looking values
```

### Phase 2: Define schemas and symbol normalization

Implement typed schemas for:

- candles (§5.2) with the time semantics of §5.1;
- provider metadata;
- quality flags and status (§5.7);
- facts and analysis snapshot (§5.8);
- LLM response (§5.9);
- supported Telegram command arguments.

Create explicit mappings such as:

```text
BTC -> BTC/USDT -> BINANCE:BTCUSDT
BTC perpetual -> BTC/USDT:USDT -> BINANCE:BTCUSDT.P
```

Verify:

- Invalid OHLC, timestamps, symbols, timeframes, and enum values are rejected.
- Naive (timezone-less) datetimes are rejected.
- Spot and perpetual symbols cannot be silently mixed.
- All timestamps normalize to UTC; `close_time = open_time + timeframe`.
- Schema serialization round-trips in tests.

### Phase 3: Build the market-data adapter

Implement a narrow CCXT provider interface:

```python
async def get_closed_ohlcv(symbol, timeframe, limit): ...
```

Add:

- retries with bounded backoff (`tenacity`) for transient errors only;
- provider timeout;
- closed-candle filtering (`close_time <= received_time`);
- duplicate and gap detection;
- stale-data detection (§5.3);
- `received_time` and `available_time` population (§5.1);
- unit metadata;
- fixture-based tests without live network access.

Verify:

- Open candle is excluded.
- Duplicate candles are handled deterministically.
- Missing intervals produce a `GAP` flag; nothing is interpolated.
- Provider timeout returns a controlled `PROVIDER_ERROR`.
- One optional `live` smoke test works without being required in CI.
- The live smoke test passes from the target VPS region (ADR 0001 open action). If blocked, evaluate `data-api.binance.vision` or another region before Phase 8.

### Phase 4: Implement deterministic technical analysis

Implement and test every metric in §5.6:

- returns;
- EMA 20/50/200;
- RSI 14;
- MACD 12/26/9;
- ATR 14 and ATR%;
- ADX and DMI 14;
- relative volume 20;
- ATR percentile;
- multi-timeframe RSI alignment (§5.5);
- trend/range/high-volatility regime;
- confirmed swing structure;
- failed breakout;
- previous-period levels.

Rules:

- Use closed candles only.
- Follow §5.6 exactly; record parameters and `CALC_VERSION` in metadata.
- Do not label rule outputs as probabilities.
- Record confirmation time for delayed pivots.

Verify:

- Unit tests cover warm-up periods, flat prices, zero volume, missing values, and short histories (§5.4 thresholds).
- Golden fixtures match independently calculated expected values.
- A small parity suite compares RSI, EMA, MACD, ATR, and ADX with TradingView values on a fixed 1000-candle fixture within documented tolerances.
- No feature reads a future row: truncating the input at any bar leaves all earlier outputs unchanged.
- Higher-timeframe values never use a candle with `close_time > as_of`.

### Phase 5: Build the analysis service and CLI (Milestone A)

Combine validated data and indicators into one `AnalysisSnapshot` (§5.8). Apply quality status rules (§5.7). Build the factor list from documented rules.

Render a deterministic section containing:

- symbol, venue, and timeframe;
- last price;
- `as_of` in UTC and WIB, and source;
- indicator values;
- regime, structure, events;
- quality status and flags.

Persist snapshots with input range and SHA-256 hash; candles are stored in `market_candles`.

CLI:

```text
python -m crypto_chatbot.cli analyze BTC 4h --fixture tests/fixtures/<file>
python -m crypto_chatbot.cli analyze BTC 4h            # live
```

Verify:

- Same input fixture always produces the same snapshot and hash.
- Recomputing a snapshot from stored candles yields identical facts.
- Stale or incomplete data cannot be presented as current.
- Every numerical output maps to a source fact ID.
- No LLM is required to produce a complete factual response.

### Phase 6: Integrate Gemini behind tools

Implement:

- one Gemini client wrapper using the pinned `GEMINI_MODEL` (paid tier; ADR 0001 open action);
- narrow tool definitions such as `analyze_coin` and `compare_coins` with validated arguments;
- structured output via a response schema matching §5.9;
- placeholder renderer and output guard (§5.9);
- prompt versioning;
- token, timeout, per-user daily, and global daily call limits;
- fallback to deterministic output when Gemini fails or validation rejects the output.

The model must not:

- fetch arbitrary URLs;
- execute code or SQL;
- write numbers outside `{F<n>}` placeholders and allowlisted tokens;
- create price targets or levels;
- expose secrets or internal prompts;
- turn factor counts or rule outputs into probabilities;
- issue guaranteed recommendations.

Verify:

- Mocked Gemini tests cover success, timeout, malformed JSON, unknown fact IDs, raw digits, and overconfidence lexicon hits (including allowed `tidak pasti`).
- A generated number absent from placeholders and allowlist is rejected.
- Gemini failure still returns a useful factual response.
- Prompt-injection text inside news, webhook data, or user messages is treated as quoted data.
- Daily call caps stop Gemini calls and fall back to deterministic output.

### Phase 7: Add Telegram commands and bounded memory

Transport: long polling (ADR 0001). Render with HTML parse mode, escape all dynamic text, and split messages below 4096 characters at section boundaries.

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

Access control:

- Only Telegram user IDs in `ALLOWED_TELEGRAM_USER_IDS` are served; others receive a short refusal and nothing is stored.
- Per-user rate limit (default 10 requests/minute) and daily Gemini cap.

Memory layers:

- recent messages: last 20 messages per user;
- explicit preferences: watchlist, language, default timeframe, display currency;
- no LLM summary in MVP.

Memory rules:

- Assistant turns are stored as placeholder narration plus `snapshot_id`, not rendered numbers.
- The fact set given to Gemini comes only from snapshots produced in the current turn.
- Do not store inferred financial circumstances as user facts.

Verify:

- Non-allowlisted users are refused and nothing is persisted for them.
- Users cannot read each other's conversations or preferences.
- `/forget` deletes the documented scope (§8).
- Context length remains bounded at 20 messages.
- Old numbers from memory cannot pass output validation.
- Invalid and unsupported commands return actionable errors.
- Rendered messages never exceed 4096 characters and survive HTML escaping of hostile input.
- Telegram and Gemini tokens never appear in logs.

### Phase 8: End-to-end MVP validation (Milestone B)

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
- Gemini unavailable or over daily cap;
- malformed symbol;
- request flood;
- non-allowlisted user;
- database unavailable.

Acceptance criteria:

- Every market number has a backend source.
- Responses contain source and UTC/WIB timestamps.
- No unsupported numerical claim reaches Telegram.
- The bot never describes rule outputs as a success probability.
- All automated checks pass.
- Binance is reachable from the production VPS.
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

- HTTPS webhook on port 443;
- authentication: TradingView cannot send custom headers, so a shared secret is placed in the JSON body and compared in constant time; additionally restrict source IPs to TradingView's published webhook addresses;
- request-size and rate limits;
- `ai-agent-lens/v1` schema adapter;
- event freshness checks;
- idempotency key from symbol, timeframe, event, and candle time;
- symbol and market-type mapping;
- comparison against exchange candles.

Do not copy the unlicensed Heavy91 source. Independently implement and test only selected generic concepts such as multi-timeframe RSI and failed breakout.

Verify:

- Requests with a missing or wrong body secret, or from a non-allowlisted IP, are rejected.
- Duplicate packets are ignored.
- Unknown schema versions are rejected.
- TradingView/exchange mismatches are flagged.
- Body and query secrets are redacted from logs.
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
- first-seen time (`available_time`);
- affected entities;
- sentiment label and confidence;
- model/version used.

Verify:

- Duplicate or syndicated articles are not counted repeatedly.
- Articles with `available_time` after prediction time never enter historical features.
- Low-confidence entity matches are excluded.
- Gemini cannot follow instructions embedded in article content.
- Summaries link to their sources.

### Phase 12: Build ML offline

Choose one initial target, preferably one of:

- probability that the next 24-hour return exceeds a cost/noise threshold;
- probability of a drawdown beyond a threshold;
- next-period volatility regime.

Implement:

- point-in-time feature generation using `available_time` (§5.1) and pivot `confirmation_time`;
- naive baseline;
- logistic-regression baseline;
- purged walk-forward validation with embargo;
- cost assumptions (fees, spread, slippage, funding);
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
- switch Telegram to webhook: HTTPS, `secret_token` set in `setWebhook`, and constant-time verification of the `X-Telegram-Bot-Api-Secret-Token` header on every update;
- replace the allowlist with public-access controls and abuse limits;
- use a managed secret store;
- apply per-user and global rate limits;
- add backups and restore tests;
- add monitoring, alerting, and structured redacted logs;
- define incident and provider-outage procedures;
- run dependency and container vulnerability scans;
- document deletion and data-export procedures.

Verify:

- Webhook updates without a valid secret-token header are rejected.
- Restore drill succeeds.
- Secret rotation succeeds.
- Load test meets the declared capacity.
- Dependency and security checks have no unresolved critical findings.
- User memory deletion is verified end to end.

## 7. Testing strategy

### Unit tests

- indicator formulas (§5.6);
- time semantics, freshness, and warm-up rules (§5.1–5.5);
- symbol mappings;
- quality status rules;
- output guard: placeholders, digit allowlist, overconfidence lexicon;
- position-independent descriptive scenarios;
- memory scoping;
- HTML escaping and message splitting.

### Integration tests

- CCXT adapter against recorded fixtures;
- PostgreSQL repositories;
- Gemini wrapper with mocked responses;
- Telegram update handling, including allowlist and rate limits;
- TradingView webhook adapter when added.

### Parity tests

Use fixed 1000-candle OHLCV fixtures to compare selected Python calculations with documented TradingView values. Define tolerances rather than requiring bit-identical floating-point output.

### End-to-end tests

Exercise Telegram request through deterministic analysis and mocked Gemini narration. Include provider and LLM failures.

### Live tests

Marked `live`, opt-in only, never required in CI.

### ML validation tests

- no future timestamp in any feature row;
- no overlap across purged fold boundaries;
- preprocessors fit on training indices only;
- reproducible results from fixed data/configuration versions;
- costs included in reported strategy metrics.

## 8. Data model outline

Initial PostgreSQL tables and retention:

| Table | Contents | Retention / deletion |
|---|---|---|
| `users` | Telegram numeric user ID, created time, allowlist status. No name, username, or phone. | Until account removal |
| `user_preferences` | Explicit preferences only. | Until `/forget` |
| `conversations`, `messages` | User messages; assistant narration with placeholders and `snapshot_id`. | 30 days; deleted by `/forget` |
| `market_candles` | Normalized public candles with provenance. | Retained (public data) |
| `analysis_snapshots` | Snapshot (§5.8) without user ID. | 90 days |
| `llm_interactions` | User ID, snapshot ID, model ID, prompt version, validation result, token counts, latency. No secrets. | 30 days; deleted by `/forget` |

`/forget` deletes the user's preferences, messages, conversations, and LLM interactions. Expired rows are deleted by `python -m crypto_chatbot.cli purge-expired`, run by host cron or a systemd timer.

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

- Use UTC everywhere internally; NTP-synchronized host clock.
- Give every external request a timeout.
- Retry only transient failures and cap retries.
- Cache only data with an explicit freshness policy.
- Keep provider errors distinguishable from insufficient data.
- Include request ID and snapshot ID in logs.
- Redact tokens, webhook secrets, user message contents when not needed, and sensitive headers.
- The MVP uses no exchange credentials; future exchange keys must be read-only.
- Enforce Gemini per-user and global daily caps; alert when the global cap is near.

## 10. Milestones

### Milestone A: Deterministic CLI prototype (end of Phase 5)

`python -m crypto_chatbot.cli analyze` analyzes one supported symbol from a fixture and then from live CCXT data. No Telegram or Gemini dependency is required to verify calculations.

### Milestone B: Telegram research MVP (end of Phase 8)

Telegram commands, PostgreSQL memory, deterministic analysis, Gemini narration, access control, and safety guards work end to end.

### Milestone C: Market-context expansion

Funding, open interest, spread, order-book features, TradingView packets, and news sentiment are added with provenance.

### Milestone D: Validated probabilistic model

An offline model passes leakage controls, walk-forward evaluation, cost-aware baseline comparison, and shadow operation.

### Milestone E: Public-readiness review

Security, privacy, operations, legal scope, retention, and recovery are reviewed before public or paid access.

## 11. Immediate next step

Start Phase 1. Then complete Phases 2–5 to reach Milestone A before connecting Telegram or Gemini. This establishes trusted market-data and indicator behavior first, which prevents the LLM interface from hiding calculation and data-quality defects.
