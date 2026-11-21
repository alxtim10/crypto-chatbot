"""CCXT Binance derivatives & order-book liquidity adapter (funding, OI, depth, liquidations)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any, Protocol, TypeVar

import ccxt
import ccxt.async_support as ccxt_async
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from crypto_chatbot.market_data.ccxt_provider import (
    DEFAULT_BINANCE_PUBLIC_API_URL,
    DEFAULT_MAX_ATTEMPTS,
    DEFAULT_RETRY_INITIAL_SECONDS,
    DEFAULT_RETRY_MAX_SECONDS,
    DEFAULT_TIMEOUT_SECONDS,
)
from crypto_chatbot.market_data.derivatives_quality import (
    DEFAULT_FUNDING_MAX_AGE,
    DEFAULT_OI_UNIT_TOLERANCE,
    _parse_interval_hours,
    _to_finite_float,
    normalize_funding_rate_payload,
    normalize_liquidation_events,
    normalize_open_interest_payload,
)
from crypto_chatbot.market_data.orderbook import (
    DEFAULT_DEPTH_BANDS_BPS,
    OrderBookSynchronizer,
    extract_sequence_id,
    normalize_order_book_payload,
)
from crypto_chatbot.market_data.symbols import SymbolMapping, resolve_symbol
from crypto_chatbot.schemas.derivatives import (
    FundingRateBatch,
    LiquidationBatch,
    OpenInterestBatch,
    OrderBookBatch,
    OrderBookDiff,
)
from crypto_chatbot.schemas.market import MarketType, QualityFlag, Timeframe

BINANCE_DEFAULT_FUNDING_INTERVAL_HOURS = 8.0
_TRANSIENT_EXCEPTIONS = (ccxt.NetworkError, TimeoutError)
_T = TypeVar("_T")


class CcxtSpotOrderBookClient(Protocol):
    """Minimal async CCXT spot client surface for L2 order-book snapshots."""

    id: str

    async def fetch_order_book(
        self,
        symbol: str,
        limit: int | None = None,
    ) -> dict[str, Any]: ...

    async def close(self) -> None: ...


class CcxtFuturesDerivativesClient(Protocol):
    """Minimal async CCXT futures client surface for public funding, OI, and order books."""

    id: str

    async def fetch_funding_rate(self, symbol: str) -> dict[str, Any]: ...

    async def fetch_funding_intervals(
        self,
        symbols: Sequence[str] | None = None,
    ) -> dict[str, Any]: ...

    async def fetch_open_interest(self, symbol: str) -> dict[str, Any]: ...

    async def fetch_open_interest_history(
        self,
        symbol: str,
        timeframe: str = "4h",
        since: int | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]: ...

    async def fetch_order_book(
        self,
        symbol: str,
        limit: int | None = None,
    ) -> dict[str, Any]: ...

    async def close(self) -> None: ...


LiquidationFetcher = Callable[[SymbolMapping, int], Awaitable[Sequence[Mapping[str, Any]]]]


class CcxtDerivativesLiquidityProvider:
    """Public Binance derivatives (perpetual) and L2 order-book provider backed by CCXT."""

    def __init__(
        self,
        *,
        spot_client: CcxtSpotOrderBookClient | None = None,
        futures_client: CcxtFuturesDerivativesClient | None = None,
        liquidation_fetcher: LiquidationFetcher | None = None,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        retry_initial_seconds: float = DEFAULT_RETRY_INITIAL_SECONDS,
        retry_max_seconds: float = DEFAULT_RETRY_MAX_SECONDS,
        spot_public_api_url: str = DEFAULT_BINANCE_PUBLIC_API_URL,
        depth_bands_bps: Sequence[int] = DEFAULT_DEPTH_BANDS_BPS,
        oi_unit_tolerance: float = DEFAULT_OI_UNIT_TOLERANCE,
        funding_max_age: timedelta = DEFAULT_FUNDING_MAX_AGE,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if retry_initial_seconds < 0 or retry_max_seconds < 0:
            raise ValueError("retry backoff durations must be non-negative")
        if not spot_public_api_url.startswith("https://") or any(
            ch.isspace() for ch in spot_public_api_url
        ):
            raise ValueError("spot_public_api_url must be a valid https:// URL without whitespace")

        self._owns_spot_client = spot_client is None
        self._owns_futures_client = futures_client is None
        self._spot_client: CcxtSpotOrderBookClient = (
            spot_client
            if spot_client is not None
            else self._build_default_spot_client(timeout_seconds, spot_public_api_url)
        )
        self._futures_client: CcxtFuturesDerivativesClient = (
            futures_client
            if futures_client is not None
            else self._build_default_futures_client(timeout_seconds)
        )
        self._liquidation_fetcher = liquidation_fetcher
        self._timeout_seconds = timeout_seconds
        self._max_attempts = max_attempts
        self._retry_initial_seconds = retry_initial_seconds
        self._retry_max_seconds = retry_max_seconds
        self._depth_bands_bps = tuple(depth_bands_bps)
        self._oi_unit_tolerance = oi_unit_tolerance
        self._funding_max_age = funding_max_age
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._synchronizers: dict[tuple[MarketType, str], OrderBookSynchronizer] = {}

    @staticmethod
    def _build_default_spot_client(
        timeout_seconds: float,
        spot_public_api_url: str,
    ) -> CcxtSpotOrderBookClient:
        # Public market data only: never pass exchange credentials (ADR 0001 / ADR 0002).
        client: CcxtSpotOrderBookClient = ccxt_async.binance(
            {
                "enableRateLimit": True,
                "timeout": int(timeout_seconds * 1000),
                "urls": {"api": {"public": spot_public_api_url}},
                "options": {
                    "defaultType": "spot",
                    "fetchMarkets": {"types": ["spot"]},
                },
            }
        )
        return client

    @staticmethod
    def _build_default_futures_client(
        timeout_seconds: float,
    ) -> CcxtFuturesDerivativesClient:
        # Public USD-M futures data only: no credentials.
        client: CcxtFuturesDerivativesClient = ccxt_async.binanceusdm(
            {
                "enableRateLimit": True,
                "timeout": int(timeout_seconds * 1000),
                "options": {
                    "defaultType": "swap",
                    "fetchMarkets": {"types": ["swap"]},
                },
            }
        )
        return client

    async def _call_with_retries(self, op: Callable[[], Awaitable[_T]]) -> _T:
        retrying = AsyncRetrying(
            retry=retry_if_exception_type(_TRANSIENT_EXCEPTIONS),
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
                return await asyncio.wait_for(op(), timeout=self._timeout_seconds)
        raise RuntimeError("unreachable")  # pragma: no cover

    def _venue_mismatch_flags(
        self,
        client_id: str,
        mapping: SymbolMapping,
        *,
        market_type: MarketType,
    ) -> list[QualityFlag]:
        cleaned = client_id.strip().lower()
        if not cleaned:
            return []
        valid_ids = (
            {mapping.venue}
            if market_type is MarketType.SPOT
            else {mapping.venue, f"{mapping.venue}usdm"}
        )
        return [QualityFlag.SYMBOL_MISMATCH] if cleaned not in valid_ids else []

    async def get_funding_rate(self, symbol: str) -> FundingRateBatch:
        """Fetch latest perpetual funding rate and interval comparability metadata."""
        mapping = resolve_symbol(symbol, market_type=MarketType.PERPETUAL)
        extra_flags = self._venue_mismatch_flags(
            getattr(self._futures_client, "id", ""),
            mapping,
            market_type=MarketType.PERPETUAL,
        )

        try:
            raw_funding, raw_intervals = await asyncio.gather(
                self._call_with_retries(
                    lambda: self._futures_client.fetch_funding_rate(mapping.ccxt_symbol)
                ),
                self._call_with_retries(
                    lambda: self._futures_client.fetch_funding_intervals([mapping.ccxt_symbol])
                ),
            )
        except TimeoutError:
            return FundingRateBatch(
                canonical_symbol=mapping.canonical_symbol,
                provider_symbol=mapping.provider_symbol,
                venue=mapping.venue,
                received_time=self._clock(),
                observation=None,
                quality_flags=(QualityFlag.PROVIDER_ERROR, QualityFlag.DERIVATIVES_UNAVAILABLE),
                error_message=(
                    f"Provider timeout fetching funding rate for {mapping.canonical_symbol} "
                    f"after {self._max_attempts} attempt(s)"
                ),
            )
        except ccxt.BaseError as exc:
            return FundingRateBatch(
                canonical_symbol=mapping.canonical_symbol,
                provider_symbol=mapping.provider_symbol,
                venue=mapping.venue,
                received_time=self._clock(),
                observation=None,
                quality_flags=(QualityFlag.PROVIDER_ERROR, QualityFlag.DERIVATIVES_UNAVAILABLE),
                error_message=(
                    f"Provider error ({type(exc).__name__}) fetching funding rate for "
                    f"{mapping.canonical_symbol}"
                ),
            )

        interval_hours = _parse_interval_hours(raw_funding.get("interval"))
        if interval_hours is None and isinstance(raw_intervals, Mapping):
            entry = raw_intervals.get(mapping.ccxt_symbol)
            if isinstance(entry, Mapping):
                interval_hours = _parse_interval_hours(entry.get("interval"))
                if interval_hours is None and isinstance(entry.get("info"), Mapping):
                    interval_hours = _parse_interval_hours(
                        entry["info"].get("fundingIntervalHours")
                    )
        if interval_hours is None:
            interval_hours = BINANCE_DEFAULT_FUNDING_INTERVAL_HOURS

        received_time = self._clock()
        return normalize_funding_rate_payload(
            raw_funding,
            symbol=mapping,
            received_time=received_time,
            funding_interval_hours=interval_hours,
            provider="ccxt",
            now=received_time,
            freshness_max_age=self._funding_max_age,
            extra_flags=extra_flags,
        )

    def _futures_contract_specs(self, ccxt_symbol: str) -> tuple[float, bool]:
        markets = getattr(self._futures_client, "markets", None)
        if isinstance(markets, Mapping):
            market = markets.get(ccxt_symbol)
            if isinstance(market, Mapping):
                csize = _to_finite_float(market.get("contractSize"))
                is_inverse = bool(market.get("inverse", False))
                return (csize if csize is not None and csize > 0 else 1.0), is_inverse
        return 1.0, False

    async def get_open_interest(
        self,
        symbol: str,
        timeframe: Timeframe = Timeframe.H4,
        history_limit: int = 30,
        *,
        reference_price: float | None = None,
    ) -> OpenInterestBatch:
        """Fetch latest + historical perpetual open interest and normalize units."""
        if history_limit < 1 or history_limit > 500:
            raise ValueError("history_limit must be between 1 and 500")
        mapping = resolve_symbol(symbol, market_type=MarketType.PERPETUAL)
        extra_flags = self._venue_mismatch_flags(
            getattr(self._futures_client, "id", ""),
            mapping,
            market_type=MarketType.PERPETUAL,
        )

        try:
            raw_latest, raw_history = await asyncio.gather(
                self._call_with_retries(
                    lambda: self._futures_client.fetch_open_interest(mapping.ccxt_symbol)
                ),
                self._call_with_retries(
                    lambda: self._futures_client.fetch_open_interest_history(
                        mapping.ccxt_symbol,
                        timeframe=timeframe.value,
                        limit=history_limit,
                    )
                ),
            )
        except TimeoutError:
            return OpenInterestBatch(
                canonical_symbol=mapping.canonical_symbol,
                provider_symbol=mapping.provider_symbol,
                venue=mapping.venue,
                timeframe=timeframe,
                received_time=self._clock(),
                latest=None,
                history=(),
                quality_flags=(QualityFlag.PROVIDER_ERROR, QualityFlag.DERIVATIVES_UNAVAILABLE),
                error_message=(
                    f"Provider timeout fetching open interest for {mapping.canonical_symbol} "
                    f"after {self._max_attempts} attempt(s)"
                ),
            )
        except ccxt.BaseError as exc:
            return OpenInterestBatch(
                canonical_symbol=mapping.canonical_symbol,
                provider_symbol=mapping.provider_symbol,
                venue=mapping.venue,
                timeframe=timeframe,
                received_time=self._clock(),
                latest=None,
                history=(),
                quality_flags=(QualityFlag.PROVIDER_ERROR, QualityFlag.DERIVATIVES_UNAVAILABLE),
                error_message=(
                    f"Provider error ({type(exc).__name__}) fetching open interest for "
                    f"{mapping.canonical_symbol}"
                ),
            )

        contract_size, is_inverse = self._futures_contract_specs(mapping.ccxt_symbol)
        received_time = self._clock()
        return normalize_open_interest_payload(
            raw_latest,
            raw_history,
            symbol=mapping,
            timeframe=timeframe,
            received_time=received_time,
            contract_size=contract_size,
            is_inverse=is_inverse,
            reference_price=reference_price,
            unit_tolerance=self._oi_unit_tolerance,
            provider="ccxt",
            extra_flags=extra_flags,
        )

    async def get_order_book(
        self,
        symbol: str,
        *,
        market_type: MarketType = MarketType.SPOT,
        limit: int = 100,
        pending_diffs: Sequence[OrderBookDiff] = (),
    ) -> OrderBookBatch:
        """Fetch L2 order book and resynchronize cleanly when sequence gaps occur."""
        if limit < 5 or limit > 5000:
            raise ValueError("limit must be between 5 and 5000")

        mapping = resolve_symbol(symbol, market_type=market_type)
        client: CcxtSpotOrderBookClient | CcxtFuturesDerivativesClient = (
            self._spot_client if market_type is MarketType.SPOT else self._futures_client
        )
        extra_flags = self._venue_mismatch_flags(
            getattr(client, "id", ""),
            mapping,
            market_type=market_type,
        )
        sync_key = (market_type, mapping.canonical_symbol)
        sync = self._synchronizers.setdefault(sync_key, OrderBookSynchronizer())

        async def _fetch_raw_book() -> dict[str, Any]:
            return await self._call_with_retries(
                lambda: client.fetch_order_book(mapping.ccxt_symbol, limit=limit)
            )

        def _error_batch(flags: tuple[QualityFlag, ...], message: str) -> OrderBookBatch:
            return OrderBookBatch(
                canonical_symbol=mapping.canonical_symbol,
                provider_symbol=mapping.provider_symbol,
                venue=mapping.venue,
                market_type=market_type,
                received_time=self._clock(),
                snapshot=None,
                resync_count=sync.resync_count,
                quality_flags=tuple(dict.fromkeys([*extra_flags, *flags])),
                error_message=message,
            )

        if pending_diffs:
            if not sync.synchronized:
                try:
                    raw_init = await _fetch_raw_book()
                except TimeoutError:
                    return _error_batch(
                        (QualityFlag.PROVIDER_ERROR, QualityFlag.ORDERBOOK_UNAVAILABLE),
                        f"Provider timeout initializing order book for {mapping.canonical_symbol}",
                    )
                except ccxt.BaseError as exc:
                    return _error_batch(
                        (QualityFlag.PROVIDER_ERROR, QualityFlag.ORDERBOOK_UNAVAILABLE),
                        f"Provider error ({type(exc).__name__}) initializing order book",
                    )

                init_batch = normalize_order_book_payload(
                    raw_init,
                    symbol=mapping,
                    received_time=self._clock(),
                    depth_bands_bps=self._depth_bands_bps,
                    resync_count=sync.resync_count,
                    extra_flags=extra_flags,
                )
                if init_batch.snapshot is None or init_batch.snapshot.sequence_id is None:
                    return _error_batch(
                        (QualityFlag.ORDERBOOK_UNAVAILABLE,),
                        init_batch.error_message or "Initial order book missing sequence_id.",
                    )
                sync.load_snapshot(
                    sequence_id=init_batch.snapshot.sequence_id,
                    bids=init_batch.snapshot.bids,
                    asks=init_batch.snapshot.asks,
                    observed_time=init_batch.snapshot.observed_time,
                )

            last_batch: OrderBookBatch | None = None
            for diff in pending_diffs:
                last_batch = await sync.apply_diff_or_resync(
                    diff,
                    resync_fetch=_fetch_raw_book,
                    symbol=mapping,
                    received_time=self._clock(),
                    provider="ccxt",
                    depth_bands_bps=self._depth_bands_bps,
                )
                if last_batch.has_provider_error:
                    return last_batch
            if last_batch is not None:
                return last_batch

        try:
            raw_book = await _fetch_raw_book()
        except TimeoutError:
            return _error_batch(
                (QualityFlag.PROVIDER_ERROR, QualityFlag.ORDERBOOK_UNAVAILABLE),
                (
                    f"Provider timeout fetching order book for {mapping.canonical_symbol} "
                    f"after {self._max_attempts} attempt(s)"
                ),
            )
        except ccxt.BaseError as exc:
            return _error_batch(
                (QualityFlag.PROVIDER_ERROR, QualityFlag.ORDERBOOK_UNAVAILABLE),
                f"Provider error ({type(exc).__name__}) fetching order book for "
                f"{mapping.canonical_symbol}",
            )

        new_seq = extract_sequence_id(raw_book)
        batch_flags = list(extra_flags)
        if (
            sync.last_update_id is not None
            and new_seq is not None
            and new_seq < sync.last_update_id
        ):
            sync.had_sequence_gap = True
            sync.resync_count += 1
            batch_flags.append(QualityFlag.SEQUENCE_GAP)
            try:
                raw_book = await _fetch_raw_book()
            except (TimeoutError, ccxt.BaseError) as exc:
                return _error_batch(
                    (
                        QualityFlag.SEQUENCE_GAP,
                        QualityFlag.PROVIDER_ERROR,
                        QualityFlag.ORDERBOOK_UNAVAILABLE,
                    ),
                    f"Order-book sequence regression resync failed ({type(exc).__name__}).",
                )
            retry_seq = extract_sequence_id(raw_book)
            if retry_seq is not None and retry_seq < sync.last_update_id:
                return _error_batch(
                    (
                        QualityFlag.SEQUENCE_GAP,
                        QualityFlag.PROVIDER_ERROR,
                        QualityFlag.ORDERBOOK_UNAVAILABLE,
                    ),
                    "Order-book sequence ID continued regressing after resynchronization.",
                )

        batch = normalize_order_book_payload(
            raw_book,
            symbol=mapping,
            received_time=self._clock(),
            provider="ccxt",
            depth_bands_bps=self._depth_bands_bps,
            resync_count=sync.resync_count,
            extra_flags=batch_flags,
        )
        if batch.snapshot is not None and batch.snapshot.sequence_id is not None:
            sync.load_snapshot(
                sequence_id=batch.snapshot.sequence_id,
                bids=batch.snapshot.bids,
                asks=batch.snapshot.asks,
                observed_time=batch.snapshot.observed_time,
            )
        return batch

    async def get_liquidations(
        self,
        symbol: str,
        *,
        limit: int = 50,
    ) -> LiquidationBatch:
        """Fetch forced liquidations only when a reliable source is configured."""
        if limit < 1 or limit > 500:
            raise ValueError("limit must be between 1 and 500")
        mapping = resolve_symbol(symbol, market_type=MarketType.PERPETUAL)
        received_time = self._clock()
        if self._liquidation_fetcher is None:
            return LiquidationBatch(
                canonical_symbol=mapping.canonical_symbol,
                provider_symbol=mapping.provider_symbol,
                venue=mapping.venue,
                received_time=received_time,
                source_available=False,
                events=(),
                quality_flags=(),
                error_message=(
                    "Binance public REST API does not expose a reliable public liquidation "
                    "endpoint; liquidation metrics left unavailable."
                ),
            )

        fetcher = self._liquidation_fetcher
        try:
            raw_events = await self._call_with_retries(lambda: fetcher(mapping, limit))
        except TimeoutError:
            return LiquidationBatch(
                canonical_symbol=mapping.canonical_symbol,
                provider_symbol=mapping.provider_symbol,
                venue=mapping.venue,
                received_time=self._clock(),
                source_available=False,
                events=(),
                quality_flags=(QualityFlag.PROVIDER_ERROR,),
                error_message=(
                    f"Provider timeout fetching liquidations for {mapping.canonical_symbol}"
                ),
            )
        except Exception as exc:
            return LiquidationBatch(
                canonical_symbol=mapping.canonical_symbol,
                provider_symbol=mapping.provider_symbol,
                venue=mapping.venue,
                received_time=self._clock(),
                source_available=False,
                events=(),
                quality_flags=(QualityFlag.PROVIDER_ERROR,),
                error_message=f"Provider error ({type(exc).__name__}) fetching liquidations",
            )

        return normalize_liquidation_events(
            raw_events,
            symbol=mapping,
            received_time=self._clock(),
            provider="ccxt",
            source_available=True,
        )

    async def close(self) -> None:
        if self._owns_spot_client:
            await self._spot_client.close()
        if self._owns_futures_client:
            await self._futures_client.close()

    async def __aenter__(self) -> CcxtDerivativesLiquidityProvider:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()
