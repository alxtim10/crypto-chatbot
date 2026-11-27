"""CCXT Binance spot OHLCV adapter with bounded retries, timeout, and quality checks."""

from __future__ import annotations

import asyncio
import math
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any, Protocol

import ccxt
import ccxt.async_support as ccxt_async
from tenacity import (
    AsyncRetrying,
    retry_if_exception,
    stop_after_attempt,
    wait_exponential_jitter,
)

from crypto_chatbot.market_data.base import (
    DEFAULT_CANDLE_LIMIT,
    DEFAULT_FRESHNESS_GRACE,
    MAX_CANDLE_LIMIT,
    MAX_HISTORY_CANDLE_LIMIT,
    FuturesTickerBatch,
    FuturesTickerSnapshot,
    OhlcvBatch,
)
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import (
    SymbolMapping,
    resolve_default_perp_symbol,
    resolve_symbol,
)
from crypto_chatbot.market_data.venue_health import is_venue_block_error
from crypto_chatbot.schemas.market import (
    MarketType,
    ProviderMetadata,
    QualityFlag,
    Timeframe,
)

DEFAULT_TIMEOUT_SECONDS = 10.0
DEFAULT_MAX_ATTEMPTS = 3
DEFAULT_RETRY_INITIAL_SECONDS = 0.5
DEFAULT_RETRY_MAX_SECONDS = 4.0
DEFAULT_BINANCE_PUBLIC_API_URL = "https://data-api.binance.vision/api/v3"

_TRANSIENT_EXCEPTIONS = (ccxt.NetworkError, TimeoutError)
_BINANCE_1000_PERP_BASES: frozenset[str] = frozenset(
    {
        "BONK",
        "BTTC",
        "CAT",
        "CHEEMS",
        "FLOKI",
        "LUNC",
        "MOG",
        "PEPE",
        "RATS",
        "SATS",
        "SHIB",
        "WHY",
        "X",
        "XEC",
    }
)


class CcxtOhlcvClient(Protocol):
    """Minimal async CCXT client surface needed by `CcxtMarketDataProvider`."""

    id: str

    async def fetch_ohlcv(
        self,
        symbol: str,
        timeframe: str = "1m",
        since: int | None = None,
        limit: int | None = None,
    ) -> list[list[Any]]: ...

    async def close(self) -> None: ...


SUPPORTED_PERP_EXCHANGES: frozenset[str] = frozenset({"binance", "gate", "bingx"})


class CcxtMarketDataProvider:
    """Public perpetual futures (or spot) OHLCV provider backed by CCXT async_support."""

    def __init__(
        self,
        *,
        client: CcxtOhlcvClient | None = None,
        exchange_id: str = "binance",
        market_type: MarketType = MarketType.PERPETUAL,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_initial_seconds: float = DEFAULT_RETRY_INITIAL_SECONDS,
        retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
        freshness_grace: timedelta = DEFAULT_FRESHNESS_GRACE,
        public_api_base_url: str = DEFAULT_BINANCE_PUBLIC_API_URL,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if retry_initial_seconds < 0 or retry_max_seconds < 0:
            raise ValueError("retry backoff durations must be non-negative")
        if not public_api_base_url.startswith("https://") or any(
            ch.isspace() for ch in public_api_base_url
        ):
            raise ValueError("public_api_base_url must be a valid https:// URL without whitespace")
        normalized_exchange = exchange_id.strip().lower()
        if normalized_exchange not in SUPPORTED_PERP_EXCHANGES:
            raise ValueError(
                f"unsupported exchange_id '{exchange_id}'; "
                f"expected one of {sorted(SUPPORTED_PERP_EXCHANGES)}"
            )

        self._exchange_id = normalized_exchange
        self._market_type = market_type
        self._owns_client = client is None
        self._client: CcxtOhlcvClient = (
            client
            if client is not None
            else self._build_default_client(
                timeout_seconds, public_api_base_url, market_type, normalized_exchange
            )
        )
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._retry_initial_seconds = retry_initial_seconds
        self._retry_max_seconds = retry_max_seconds
        self._freshness_grace = freshness_grace
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))

    @staticmethod
    def _build_default_client(
        timeout_seconds: float,
        public_api_base_url: str,
        market_type: MarketType = MarketType.PERPETUAL,
        exchange_id: str = "binance",
    ) -> CcxtOhlcvClient:
        # Public market data only: never pass exchange credentials (ADR 0001 / ADR 0013).
        if market_type is MarketType.PERPETUAL:
            if exchange_id == "gate":
                exchange_gate: CcxtOhlcvClient = ccxt_async.gate(
                    {
                        "enableRateLimit": True,
                        "timeout": int(timeout_seconds * 1000),
                        "options": {
                            "defaultType": "swap",
                            "fetchMarkets": {"types": ["swap"]},
                        },
                    }
                )
                return exchange_gate
            if exchange_id == "bingx":
                exchange_bingx: CcxtOhlcvClient = ccxt_async.bingx(
                    {
                        "enableRateLimit": True,
                        "timeout": int(timeout_seconds * 1000),
                        "options": {"defaultType": "swap"},
                    }
                )
                return exchange_bingx
            exchange_perp: CcxtOhlcvClient = ccxt_async.binanceusdm(
                {
                    "enableRateLimit": True,
                    "timeout": int(timeout_seconds * 1000),
                    "options": {
                        "defaultType": "swap",
                        "fetchMarkets": {"types": ["linear"]},
                    },
                }
            )
            return exchange_perp

        exchange_spot: CcxtOhlcvClient = ccxt_async.binance(
            {
                "enableRateLimit": True,
                "timeout": int(timeout_seconds * 1000),
                "urls": {"api": {"public": public_api_base_url}},
                "options": {
                    "defaultType": "spot",
                    "fetchMarkets": {"types": ["spot"]},
                },
            }
        )
        return exchange_spot

    async def _fetch_once(
        self,
        ccxt_symbol: str,
        timeframe: Timeframe,
        fetch_limit: int,
        *,
        since: int | None = None,
    ) -> Sequence[Sequence[Any]]:
        kwargs: dict[str, Any] = {
            "timeframe": timeframe.value,
            "limit": fetch_limit,
        }
        if since is not None:
            kwargs["since"] = since
        return await asyncio.wait_for(
            self._client.fetch_ohlcv(ccxt_symbol, **kwargs),
            timeout=self._timeout_seconds,
        )

    async def _fetch_with_retries(
        self,
        ccxt_symbol: str,
        timeframe: Timeframe,
        fetch_limit: int,
        *,
        since: int | None = None,
    ) -> Sequence[Sequence[Any]]:
        retrying = AsyncRetrying(
            retry=retry_if_exception(
                lambda exc: isinstance(exc, _TRANSIENT_EXCEPTIONS) and not is_venue_block_error(exc)
            ),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(
                initial=self._retry_initial_seconds,
                max=self._retry_max_seconds,
                jitter=min(self._retry_initial_seconds, 0.25),
            ),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                return await self._fetch_once(
                    ccxt_symbol,
                    timeframe,
                    fetch_limit,
                    since=since,
                )
        raise RuntimeError("unreachable")  # pragma: no cover

    def _resolve_venue_mapping(
        self,
        symbol: str,
    ) -> tuple[SymbolMapping, list[QualityFlag]]:
        mapping = resolve_symbol(symbol, market_type=self._market_type)
        if self._market_type is MarketType.PERPETUAL and self._exchange_id != "binance":
            provider_sym = f"{self._exchange_id.upper()}:{mapping.base}{mapping.quote}.P"
            mapping = mapping.model_copy(
                update={
                    "venue": self._exchange_id,
                    "provider_symbol": provider_sym,
                }
            )
        extra_flags: list[QualityFlag] = []
        client_id = getattr(self._client, "id", "").lower()
        if client_id and not (
            client_id.startswith(mapping.venue) or mapping.venue.startswith(client_id)
        ):
            extra_flags.append(QualityFlag.SYMBOL_MISMATCH)
        return mapping, extra_flags

    def _resolve_1000_perp_fallback(self, base: str) -> SymbolMapping | None:
        if self._market_type is not MarketType.PERPETUAL or base not in _BINANCE_1000_PERP_BASES:
            return None
        fallback_mapping = resolve_symbol(
            f"1000{base}",
            market_type=MarketType.PERPETUAL,
        )
        if self._exchange_id != "binance":
            fallback_mapping = fallback_mapping.model_copy(
                update={
                    "venue": self._exchange_id,
                    "provider_symbol": (
                        f"{self._exchange_id.upper()}:"
                        f"{fallback_mapping.base}{fallback_mapping.quote}.P"
                    ),
                }
            )
        return fallback_mapping

    async def _fetch_chunk_with_fallback(
        self,
        *,
        mapping: SymbolMapping,
        timeframe: Timeframe,
        fetch_limit: int,
        since: int | None = None,
        allow_1000_fallback: bool = True,
    ) -> tuple[Sequence[Sequence[Any]] | None, SymbolMapping, OhlcvBatch | None]:
        try:
            rows = await self._fetch_with_retries(
                mapping.ccxt_symbol,
                timeframe,
                fetch_limit,
                since=since,
            )
            return rows, mapping, None
        except ccxt.BadSymbol as exc:
            fallback_mapping = (
                self._resolve_1000_perp_fallback(mapping.base) if allow_1000_fallback else None
            )
            if fallback_mapping is None:
                return (
                    None,
                    mapping,
                    self._provider_error_batch(
                        mapping=mapping,
                        timeframe=timeframe,
                        message=(
                            f"Provider error ({type(exc).__name__}) fetching "
                            f"{mapping.canonical_symbol} {timeframe.value}"
                        ),
                    ),
                )
            try:
                rows = await self._fetch_with_retries(
                    fallback_mapping.ccxt_symbol,
                    timeframe,
                    fetch_limit,
                    since=since,
                )
                return rows, fallback_mapping, None
            except TimeoutError:
                return (
                    None,
                    fallback_mapping,
                    self._provider_error_batch(
                        mapping=fallback_mapping,
                        timeframe=timeframe,
                        message=(
                            f"Provider timeout fetching {fallback_mapping.canonical_symbol} "
                            f"{timeframe.value} after {self._max_attempts} attempt(s)"
                        ),
                    ),
                )
            except ccxt.BaseError as fallback_exc:
                return (
                    None,
                    fallback_mapping,
                    self._provider_error_batch(
                        mapping=fallback_mapping,
                        timeframe=timeframe,
                        message=(
                            f"Provider error ({type(fallback_exc).__name__}) fetching "
                            f"{fallback_mapping.canonical_symbol} {timeframe.value}"
                        ),
                    ),
                )
        except TimeoutError:
            return (
                None,
                mapping,
                self._provider_error_batch(
                    mapping=mapping,
                    timeframe=timeframe,
                    message=(
                        f"Provider timeout fetching {mapping.canonical_symbol} {timeframe.value} "
                        f"after {self._max_attempts} attempt(s)"
                    ),
                ),
            )
        except ccxt.BaseError as exc:
            return (
                None,
                mapping,
                self._provider_error_batch(
                    mapping=mapping,
                    timeframe=timeframe,
                    message=(
                        f"Provider error ({type(exc).__name__}) fetching "
                        f"{mapping.canonical_symbol} {timeframe.value}"
                    ),
                ),
            )

    async def get_closed_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = DEFAULT_CANDLE_LIMIT,
    ) -> OhlcvBatch:
        """Fetch closed perpetual futures candles with retries, timeout, and quality checks."""
        if limit < 1 or limit > MAX_CANDLE_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_CANDLE_LIMIT}")

        mapping, extra_flags = self._resolve_venue_mapping(symbol)
        # Request one extra row when below venue cap so dropping the open candle preserves `limit`.
        fetch_limit = min(limit + 1, MAX_CANDLE_LIMIT)
        raw_rows, mapping, error_batch = await self._fetch_chunk_with_fallback(
            mapping=mapping,
            timeframe=timeframe,
            fetch_limit=fetch_limit,
        )
        if error_batch is not None or raw_rows is None:
            return error_batch or self._provider_error_batch(
                mapping=mapping,
                timeframe=timeframe,
                message=f"Provider error fetching {mapping.canonical_symbol} {timeframe.value}",
            )

        received_time = self._clock()
        return normalize_ohlcv_rows(
            raw_rows,
            symbol=mapping,
            timeframe=timeframe,
            received_time=received_time,
            provider="ccxt",
            limit=limit,
            now=received_time,
            freshness_grace=self._freshness_grace,
            extra_flags=extra_flags,
        )

    async def get_closed_ohlcv_history(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = DEFAULT_CANDLE_LIMIT,
    ) -> OhlcvBatch:
        """Fetch paginated closed OHLCV history up to `MAX_HISTORY_CANDLE_LIMIT` candles."""
        if limit < 1 or limit > MAX_HISTORY_CANDLE_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_HISTORY_CANDLE_LIMIT}")
        if limit <= MAX_CANDLE_LIMIT:
            return await self.get_closed_ohlcv(symbol, timeframe, limit=limit)

        mapping, extra_flags = self._resolve_venue_mapping(symbol)
        step_ms = int(timeframe.duration.total_seconds() * 1000)
        now = self._clock()
        now_ms = int(now.timestamp() * 1000)
        cursor_ms = now_ms - (limit + 2) * step_ms
        rows_by_ts: dict[int, Sequence[Any]] = {}

        max_pages = 60
        for page_idx in range(max_pages):
            remaining = (limit + 2) - len(rows_by_ts) + 1
            chunk_limit = max(2, min(MAX_CANDLE_LIMIT, remaining))
            chunk, mapping, error_batch = await self._fetch_chunk_with_fallback(
                mapping=mapping,
                timeframe=timeframe,
                fetch_limit=chunk_limit,
                since=cursor_ms,
                allow_1000_fallback=(page_idx == 0),
            )
            if error_batch is not None or chunk is None:
                return error_batch or self._provider_error_batch(
                    mapping=mapping,
                    timeframe=timeframe,
                    message=f"Provider error fetching {mapping.canonical_symbol} {timeframe.value}",
                )
            if not chunk:
                break

            page_max_ts: int | None = None
            for row in chunk:
                if not row:
                    continue
                try:
                    ts = int(row[0])
                except (TypeError, ValueError):
                    continue
                rows_by_ts[ts] = row
                if page_max_ts is None or ts > page_max_ts:
                    page_max_ts = ts

            if page_max_ts is None:
                break
            next_cursor = page_max_ts + step_ms
            if next_cursor <= cursor_ms or page_max_ts >= now_ms - step_ms:
                break
            if len(rows_by_ts) >= limit + 2:
                break
            cursor_ms = next_cursor
            if self._retry_initial_seconds > 0:
                await asyncio.sleep(0.1)

        sorted_rows = [rows_by_ts[ts] for ts in sorted(rows_by_ts)]
        received_time = self._clock()
        return normalize_ohlcv_rows(
            sorted_rows,
            symbol=mapping,
            timeframe=timeframe,
            received_time=received_time,
            provider="ccxt",
            limit=limit,
            now=received_time,
            freshness_grace=self._freshness_grace,
            extra_flags=extra_flags,
        )

    async def get_funding_history(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
    ) -> tuple[tuple[datetime, float], ...]:
        """Fetch point-in-time historical funding settlements `(timestamp_utc, rate)`."""
        fetch_fn = getattr(self._client, "fetch_funding_rate_history", None)
        if not callable(fetch_fn):
            return ()
        mapping, _ = self._resolve_venue_mapping(symbol)
        start_ms = int(start.astimezone(UTC).timestamp() * 1000)
        end_ms = int(end.astimezone(UTC).timestamp() * 1000)
        if end_ms <= start_ms:
            return ()

        by_ts: dict[int, float] = {}
        cursor_ms = start_ms
        for _ in range(30):
            try:
                rows = await asyncio.wait_for(
                    fetch_fn(mapping.ccxt_symbol, since=cursor_ms, limit=500),
                    timeout=self._timeout_seconds,
                )
            except (TimeoutError, ccxt.BaseError, TypeError, ValueError):
                break
            if not rows:
                break
            page_max_ts: int | None = None
            for item in rows:
                if not isinstance(item, Mapping):
                    continue
                ts_raw = item.get("timestamp")
                rate_raw = item.get("fundingRate")
                if not isinstance(ts_raw, (int, float)) or rate_raw is None:
                    continue
                ts_int = int(ts_raw)
                rate_f = self._to_finite_float(rate_raw)
                if rate_f is None:
                    continue
                if start_ms <= ts_int <= end_ms:
                    by_ts[ts_int] = rate_f
                if page_max_ts is None or ts_int > page_max_ts:
                    page_max_ts = ts_int
            if page_max_ts is None or page_max_ts >= end_ms or page_max_ts + 1 <= cursor_ms:
                break
            cursor_ms = page_max_ts + 1
            if self._retry_initial_seconds > 0:
                await asyncio.sleep(0.1)

        return tuple(
            (datetime.fromtimestamp(ts / 1000.0, tz=UTC), by_ts[ts]) for ts in sorted(by_ts)
        )

    def _provider_error_batch(
        self,
        *,
        mapping: SymbolMapping,
        timeframe: Timeframe,
        message: str,
    ) -> OhlcvBatch:
        received_time = self._clock()
        metadata = ProviderMetadata(
            provider="ccxt",
            venue=mapping.venue,
            market_type=mapping.market_type,
            base_unit=mapping.base,
            quote_currency=mapping.quote,
        )
        return OhlcvBatch(
            metadata=metadata,
            canonical_symbol=mapping.canonical_symbol,
            provider_symbol=mapping.provider_symbol,
            timeframe=timeframe,
            received_time=received_time,
            candles=(),
            quality_flags=(QualityFlag.PROVIDER_ERROR,),
            error_message=message,
        )

    @staticmethod
    def _to_finite_float(val: object) -> float | None:
        if val is None or isinstance(val, bool):
            return None
        try:
            num = float(val)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return None
        return num if math.isfinite(num) else None

    async def _fetch_tickers_once(self) -> dict[str, Any]:
        fetch_fn = getattr(self._client, "fetch_tickers", None)
        if fetch_fn is None:
            raise ccxt.NotSupported(f"{self._exchange_id} client does not implement fetch_tickers")
        raw = await asyncio.wait_for(fetch_fn(), timeout=self._timeout_seconds)
        return dict(raw) if isinstance(raw, Mapping) else {}

    async def _fetch_tickers_with_retries(self) -> dict[str, Any]:
        retrying = AsyncRetrying(
            retry=retry_if_exception(
                lambda exc: isinstance(exc, _TRANSIENT_EXCEPTIONS) and not is_venue_block_error(exc)
            ),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential_jitter(
                initial=self._retry_initial_seconds,
                max=self._retry_max_seconds,
                jitter=min(self._retry_initial_seconds, 0.25),
            ),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                return await self._fetch_tickers_once()
        raise RuntimeError("unreachable")  # pragma: no cover

    async def get_futures_tickers(self) -> FuturesTickerBatch:
        """Fetch 24h futures market tickers (price, % change, USD quote volume)."""
        received_time = self._clock()
        try:
            raw_tickers = await self._fetch_tickers_with_retries()
        except TimeoutError:
            return FuturesTickerBatch(
                venue=self._exchange_id,
                received_time=self._clock(),
                quality_flags=(QualityFlag.PROVIDER_ERROR,),
                error_message=f"Provider timeout fetching futures tickers on {self._exchange_id}",
            )
        except ccxt.BaseError as exc:
            return FuturesTickerBatch(
                venue=self._exchange_id,
                received_time=self._clock(),
                quality_flags=(QualityFlag.PROVIDER_ERROR,),
                error_message=(
                    f"Provider error ({type(exc).__name__}) fetching tickers on {self._exchange_id}"
                ),
            )

        observed_default = self._clock()
        snapshots: list[FuturesTickerSnapshot] = []
        seen_symbols: set[str] = set()

        for ccxt_sym, payload in raw_tickers.items():
            if not isinstance(ccxt_sym, str) or not ccxt_sym.endswith("/USDT:USDT"):
                continue
            if not isinstance(payload, Mapping):
                continue

            try:
                mapping = resolve_default_perp_symbol(ccxt_sym)
            except ValueError:
                continue
            if mapping.canonical_symbol in seen_symbols:
                continue

            last_price = self._to_finite_float(payload.get("last"))
            if last_price is None or last_price <= 0.0:
                last_price = self._to_finite_float(payload.get("close"))
            if last_price is None or last_price <= 0.0:
                continue

            change_pct = self._to_finite_float(payload.get("percentage"))
            if change_pct is None:
                open_price = self._to_finite_float(payload.get("open"))
                if open_price is not None and open_price > 0.0:
                    change_pct = ((last_price - open_price) / open_price) * 100.0
            if change_pct is None:
                continue

            quote_vol = self._to_finite_float(payload.get("quoteVolume"))
            if quote_vol is None or quote_vol <= 0.0:
                info = payload.get("info")
                if isinstance(info, Mapping):
                    for key in (
                        "volume_24h_settle",
                        "volume_24h_quote",
                        "quoteVolume",
                        "quote_volume",
                        "vol24h",
                    ):
                        cand = self._to_finite_float(info.get(key))
                        if cand is not None and cand > 0.0:
                            quote_vol = cand
                            break
            if quote_vol is None or quote_vol <= 0.0:
                base_vol = self._to_finite_float(payload.get("baseVolume"))
                if base_vol is not None and base_vol > 0.0:
                    quote_vol = base_vol * last_price
            if quote_vol is None or quote_vol < 0.0:
                quote_vol = 0.0

            ts_ms = self._to_finite_float(payload.get("timestamp"))
            if ts_ms is not None and ts_ms > 0.0:
                obs_time = datetime.fromtimestamp(ts_ms / 1000.0, tz=UTC)
            else:
                obs_time = observed_default

            provider_sym = f"{self._exchange_id.upper()}:{mapping.base}{mapping.quote}.P"
            seen_symbols.add(mapping.canonical_symbol)
            snapshots.append(
                FuturesTickerSnapshot(
                    venue=self._exchange_id,
                    canonical_symbol=mapping.canonical_symbol,
                    provider_symbol=provider_sym,
                    last_price=last_price,
                    change_24h_pct=change_pct,
                    quote_volume_24h_usd=quote_vol,
                    observed_time=obs_time,
                )
            )

        return FuturesTickerBatch(
            venue=self._exchange_id,
            received_time=received_time,
            tickers=tuple(snapshots),
        )

    async def close(self) -> None:
        if self._owns_client:
            await self._client.close()

    async def __aenter__(self) -> CcxtMarketDataProvider:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()
