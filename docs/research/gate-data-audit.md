# Gate.io & Sentiment Data Audit

Verification log captured before implementation against live HTTP/WebSocket responses and downloaded samples on **2026-11-27** (UTC ~`10:20–10:24Z`). All checks used read-only public endpoints without credentials.

> [!NOTE]
> Live system clock in the verification container was `2026-11-27T10:20:00Z`–`10:24:00Z` (`epoch ~1795774800`), matching Gate server timestamps (`t: 1795774833`).

---

## 1. Gate.io Futures REST API (`https://api.gateio.ws/api/v4`)

### 1.1 Contract Specifications — `GET /futures/usdt/contracts` & `/futures/usdt/contracts/{contract}`

- **Status**: `200 OK`
- **Symbol format**: `BASE_USDT` (e.g., `BTC_USDT`, `ETH_USDT`, `SOL_USDT`, `LINK_USDT`).
- **Verified fields (`BTC_USDT`)**:
  - `name`: `"BTC_USDT"`
  - `type`: `"direct"` (USDT-margined linear perpetual)
  - `quanto_multiplier`: string decimal representing **base-asset units per 1 contract**:
    - `BTC_USDT`: `"0.0001"` (1 contract = `0.0001 BTC`)
    - `ETH_USDT`: `"0.01"` (1 contract = `0.01 ETH`)
    - `SOL_USDT`: `"1"` (1 contract = `1 SOL`)
    - `LINK_USDT`: `"1"` (1 contract = `1 LINK`)
  - `leverage_min`: `"1"`, `leverage_max`: `"200"` (`LINK_USDT`: `"75"`)
  - `maintenance_rate`: `"0.003"`
  - `risk_limit_base`: `"500000"`, `risk_limit_step`: `"1499500000"`, `risk_limit_max`: `"1500000000"`
  - `funding_interval`: integer seconds (`28800` = `8h`)
  - `funding_rate`: `"0.000028"`, `funding_rate_indicative`: `"0.000028"`
  - `mark_price`: `"61624.39"`, `index_price`: `"61652.74"`, `last_price`: `"61615.1"`
  - `position_size`: `318529596` (open contracts), `in_delisting`: `false`

---

### 1.2 Futures Candlesticks — `GET /futures/usdt/candlesticks`

- **Status**: `200 OK` for `interval` in `5m`, `15m`, `1h`, `4h`, `1d`.
- **Query params**: `contract`, `interval`, `limit` / `from` / `to` (Unix seconds, UTC).
- **Ordering**: ascending by `t`.
- **Open-candle warning**: **The latest element in the returned array can be the still-forming open bar**!
  - Verified at `2026-11-27T10:23:11Z`: `limit=3` for `5m` returned `[10:10:00Z, 10:15:00Z, 10:20:00Z]`; `10:20:00Z` closes at `10:25:00Z` and had changing live volume (`v: 605443`).
  - Therefore closed-candle filter **must** enforce `close_time = open_time + interval <= as_of`!
- **Field definitions**:
  - `t`: integer bar open time (Unix seconds, UTC)
  - `o`, `h`, `l`, `c`: string decimal prices (`"61623.2"`, etc.)
  - `v`: integer/numeric **contract count** (NOT base units!). Verified on `BTC_USDT` `1h`: `v = 47089285` contracts (`* 0.0001 = 4708.9285 BTC`). Verified on `LINK_USDT` `5m`: `v = 650` contracts (`* 1 = 650 LINK`).
  - `sum`: string decimal **quote volume in USDT** (`"291265470.34431"` on `BTC_USDT` 1h bar `v=47089285`, ~`4708.9285 BTC * ~$61,854 = $291.26M`).
- **History depth**: verified `from=1767225600` (Jan 1, 2026) returned `200 OK` directly from REST.

---

### 1.3 Contract Stats (Positioning / Liquidation / OI) — `GET /futures/usdt/contract_stats`

- **Status**: `200 OK` for `interval=5m` and `interval=1h`.
- **Query params**: `contract`, `interval` (`5m`, `1h`), `from` (Unix seconds), `limit` (max tested `limit=1000` returned `998` 5m rows).
- **Verified history depth**:
  - `from = now - 180 days` -> `200 OK` (180-day retention confirmed for both `5m` and `1h`!).
  - `from = now - 365 days` -> `400 Bad Request`: `{"label": "INVALID_PARAM_VALUE", "message": "from time exceeds 180-day limit"}`.
- **Verified JSON keys & units (`BTC_USDT` `5m` sample `time=1795774500`)**:
  - `time`: integer Unix seconds (UTC), bucket start time (`1795774500` = `2026-11-27T10:15:00Z`). Note: at `10:23:11Z`, `stats[-1]` had `time = 10:20:00Z` (latest in-progress/recent 5m point). Closed-bar policy for `contract_stats` intervals should respect `observed_time + interval <= as_of` (or mark open bucket carefully / poll after bar close).
  - `open_interest`: integer **contracts** (`637089156`).
  - `open_interest_usd`: float USD notional (`3926879946.326304`).
    - **Cross-check**: `637089156 * 0.0001 (quanto_multiplier) * 61637.84 (mark_price) = 3926879946.326304` — exact to float precision!
  - `mark_price`: float (`61637.84`).
  - `lsr_taker`: float (`1.5311786953396056`), equals `long_taker_size / short_taker_size` (`48987 / 31993 = 1.5311786953396056`).
  - `lsr_account`: float (`1.4225679685620338`), equals `long_users / short_users` (`17738 / 12469 = 1.4225679685620338`).
  - `long_users`, `short_users`: integer account counts (`17738`, `12469`).
  - `top_lsr_account`: float (`0.4302325581395349`), equals `top_long_account / top_short_account` (`37 / 86`).
  - `top_long_account`, `top_short_account`: integer counts (`37`, `86`).
  - `top_lsr_size`: float (`0.9664237335561044`), equals `top_long_size / top_short_size` (`34459111 / 35656317`).
  - `top_long_size`, `top_short_size`: integer contracts (`34459111`, `35656317`).
  - `long_taker_size`, `short_taker_size`: float/int taker volumes for the bucket.
    - *Caveat*: on `LINK_USDT` `10:15:00Z` bucket captured at `10:23:11Z`, `long_taker_size=377, short_taker_size=119` (sum `496` vs `candlesticks.v=650`), whereas `trades` in `[t0, t0+300)` summed to `650 == candlesticks.v`. So `long_taker_size`/`short_taker_size` in `contract_stats` is Gate's internal sampled/bucketed metric while `/futures/usdt/trades` is the exact execution ledger for taker flow (`D-05`).
  - `long_liq_size`, `short_liq_size`: liquidation size (contracts).
  - `long_liq_amount`, `short_liq_amount`: liquidation base/amount field (`0` in quiet sample).
  - `long_liq_usd`, `short_liq_usd`, `long_liq_usd_new`, `short_liq_usd_new`: liquidation USD values (`0` in quiet sample).
  - `last_funding_rate`: string decimal (`"0.0000234"`).

---

### 1.4 Funding Rate History — `GET /futures/usdt/funding_rate`

- **Status**: `200 OK`
- **Query params**: `contract`, `from`, `to` (Unix seconds), `limit`.
- **Response items**: list sorted newest-first (`t` descending): `{"r": "0.0000379", "t": 1795766400}` (`t` = settlement Unix seconds, 8h spacing `28800s`; `r` = string decimal funding rate).
- **Verified limits & pagination**:
  - `limit=100` / `limit=1000` both cap at `90` entries per page (`30 days * 3 settlements/day = 90`).
  - Paginates via `to = oldest_t - 1` cleanly (`oldest_t - 1` returned page 2 of 90 rows back to `2026-09-28T16:00:00Z`).
  - `from = -179 days` (`t=1782892801`) returned `200 OK`; deeper (`Jan 2024`) returned `400 INVALID_PARAM_VALUE: "from time exceeds 180-day limit"`.
  - Conclusion: REST retains ~180 days (paginated via `to` in chunks of 90); older history comes from `download.gatedata.org` `funding_applies`.

---

### 1.5 Risk Limit / Maintenance Margin Tiers — `GET /futures/usdt/risk_limit_tiers`

- **Status**: `200 OK` (`risk_limit_table` returned `400` requiring `table_id`, whereas `risk_limit_tiers?contract=BTC_USDT` works directly without `table_id`).
- **Response**: list of `19` tier dicts for `BTC_USDT`, ordered ascending by `tier`:
  - `tier`: integer (`1`, `2`, ...)
  - `risk_limit`: string USDT max position notional (`"500000"`, `"1000000"`, ...)
  - `maintenance_rate`: string decimal MMR (`"0.003"`, `"0.0035"`, ...)
  - `initial_rate`: string decimal IMR (`"0.005"`, `"0.006666"`, ...)
  - `leverage_max`: string max leverage (`"200"`, `"150.01"`, ...)
  - `deduction`: string USDT maintenance margin deduction (`"0"`, `"250"`, ...)

---

### 1.6 Futures Trades (Execution Ledger) — `GET /futures/usdt/trades`

- **Status**: `200 OK`
- **Sample item**:
  ```json
  {
    "id": 898355598,
    "contract": "BTC_USDT",
    "create_time": 1795774824.936,
    "create_time_ms": 1795774824.936,
    "size": -16,
    "price": "61607.5"
  }
  ```
- **Sign & unit convention (verified)**:
  - `size`: integer/numeric **signed contracts**:
    - **`size > 0`**: **taker buy** (buyer lifted ask).
    - **`size < 0`**: **taker sell** (seller hit bid; magnitude = `-size`).
  - **Cross-check on `LINK_USDT` `5m` bar `t=1795774500`**:
    - `sum(size for size > 0) = 539`, `sum(-size for size < 0) = 111`, total = `650` contracts (`42` trades).
    - Matching `GET /futures/usdt/candlesticks` 5m bar: `v = 650` contracts — **0.00% discrepancy (exact contract match)**.
  - *Note*: `create_time_ms` in the observed response is fractional seconds (`1795774824.936`), identical to `create_time`, not integer milliseconds — parse `create_time` as float seconds!

---

## 2. Bulk Historical Archive (`https://download.gatedata.org/futures_usdt/...`)

URL pattern: `https://download.gatedata.org/futures_usdt/{dataset}/{YYYYMM}/{CONTRACT}-{YYYYMM}.csv.gz` (no CSV header row).

- **Available datasets verified for `202610` (`BTC_USDT`)**:
  - `candlesticks_10s` (`200`), `candlesticks_1m` (`200`), `candlesticks_5m` (`200`), `candlesticks_1h` (`200`), `candlesticks_4h` (`200`), `candlesticks_1d` (`200`).
  - **Missing intervals (`404`)**: `candlesticks_15m`, `candlesticks_30m`, `candlesticks_8h` (confirming why `15m` / `30m` cannot come directly from `candlesticks_*` bulk download and must be resampled or fetched from REST).
  - `trades` (`200`), `mark_prices` (`200`), `funding_applies` (`200`).
  - **Missing datasets (`404`)**: `funding_rates`, `contract_stats` (historical `contract_stats` is NOT on `download.gatedata.org`, making `D-06` forward polling time-critical).
- **Depth verified**: `candlesticks_1h` returned `200` for `202001`, `202201`, `202401`, `202601`, `202609`, `202610`. Current partial month (`202611`) returns `404` (only completed months are published).

### CRITICAL CSV Column Order Verification

1. **`candlesticks_{tf}`**:
   - Sample row (`BTC_USDT-202610.csv.gz` at `1790812800`):
     `1790812800,13530341,83472,83618.3,83381.2,83578.7`
   - Cross-checked against live REST `GET /futures/usdt/candlesticks` for `t=1790812800`:
     `{"t": 1790812800, "v": 13530341, "c": "83472", "h": "83618.3", "l": "83381.2", "o": "83578.7"}`
   - **Exact column layout**: `[0]=timestamp_s, [1]=volume_contracts, [2]=close, [3]=high, [4]=low, [5]=open` (**NOT** OHLC order! `close` is column 2 and `open` is column 5!).
   - Does **not** contain `sum` (quote volume); must compute `base_volume = volume_contracts * quanto_multiplier` and estimate `quote_volume = base_volume * close` flagged `QUOTE_VOL_ESTIMATED`.
2. **`trades`**:
   - Sample rows (`BTC_USDT-202610`):
     `1790812800.013589,844545515,83578.7,1`
     `1790812800.577279,844545519,83578.6,-3`
   - **Column layout**: `[0]=create_time_s (float), [1]=trade_id (int), [2]=price (decimal), [3]=signed_size_contracts (pos=buy, neg=sell)`.
   - Notice row 0 price `83578.7` matches the 1h candle `open = 83578.7` at `1790812800`.
3. **`funding_applies`**:
   - Sample rows: `1790812800,0.000088`, `1790841600,0.000007` (8h cadence = `28800s`).
   - **Column layout**: `[0]=timestamp_s, [1]=funding_rate`.
4. **`mark_prices`**:
   - Sample rows: `1790812801.153758,83620.41,83582.77,83574.1` (1s cadence; `[0]=timestamp_s, [1]=mark_price, [2]=index_price, [3]=last_price` — unverified against historical tick archive; store verbatim if archiving raw CSV, do not derive trading features from cols 2–3 without confirmation).

---

## 3. Gate.io Futures WebSocket (`wss://fx-ws.gateio.ws/v4/ws/usdt`)

- **Status**: connected cleanly, subscribed to `futures.candlesticks` with `payload: ["1m", "BTC_USDT"]`.
- **Verified update payload**:
  ```json
  {"t":1795774800,"c":"61611.5","h":"61642.9","l":"61607.5","o":"61630.5","a":"456040.56613","n":"1m_BTC_USDT","w":false,"v":74005}
  ```
- **Verified field semantics**:
  - `t`: open timestamp (Unix seconds)
  - `o`, `h`, `l`, `c`: string OHLC prices
  - `v`: integer **contracts** (`74005`)
  - `a`: string **quote volume (USDT)** (`"456040.56613"`, matches `74005 * 0.0001 * ~6162`)
  - `n`: `"{interval}_{contract}"` (`"1m_BTC_USDT"`)
  - `w`: boolean closed/window-end indicator (`false` while forming; bar must only be marked closed when `w is true` **or** superseded by a strictly newer `t` / `close_time <= as_of`).

---

## 4. Alternative.me Fear & Greed (`https://api.alternative.me/fng/`)

- **Status**: `200 OK` (no API key required).
- **Tested query params**: `limit=5&format=json` and `limit=0&format=json`.
- **Verified full history depth**: `3218` daily records from `timestamp=1517443200` (`2018-02-01T00:00:00Z`) to `1795737600` (`2026-11-27T00:00:00Z`).
- **Response schema**:
  - Top-level: `name`, `data` (descending by `timestamp`), `metadata.error`.
  - Item fields: `value` (string int `0–100`), `value_classification` (`"Extreme Fear"`, `"Fear"`, etc.), `timestamp` (string Unix seconds UTC at daily 00:00Z), `time_until_update` (string seconds, present on newest item only).

---

## 5. CryptoPanic API (`https://cryptopanic.com/api/...`) — BLOCKED / UNVERIFIED LIVE

- **Result**: Every tested endpoint (`/api/v1/posts/`, `/api/developer/v2/posts/`, `/api/free/v1/posts/`, with and without `auth_token=test` and custom headers) returned **`HTTP 403 Forbidden`** with a Cloudflare HTML challenge page (`<!DOCTYPE html> <!--[if lt IE 7]>...`).
- **Implications for S-03 / S-05**:
  - Live unauthenticated requests from this environment are blocked by Cloudflare/API-key enforcement (`auth_token` required per CryptoPanic API v2 docs).
  - Exact live field quirks (e.g., whether v2 free/developer tier exposes `votes`, `instruments`, `source`, and domain limits) **could not be verified against a live 200 JSON payload**.
  - Implementation of `D-06`, `D-01`, `D-02`, `D-04`, `D-05`, `D-07`, `S-01`, `S-02`, `D-03`, `D-08`, `D-09` proceeds first using verified Gate.io and Alternative.me schemas. `S-03`/`S-05` will either require a valid `CRYPTOPANIC_API_KEY` from the operator (or fixture/fallback RSS/public feed approval) before making live calls.
