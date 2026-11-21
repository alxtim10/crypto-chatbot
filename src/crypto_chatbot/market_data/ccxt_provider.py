"""CCXT Binance spot OHLCV adapter with bounded retries, timeout, and quality checks."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta
from types import TracebackType
from typing import Any, Protocol

import ccxt
import ccxt.async_support as ccxt_async
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from crypto_chatbot.market_data.base import (
    DEFAULT_CANDLE_LIMIT,
    DEFAULT_FRESHNESS_GRACE,
    MAX_CANDLE_LIMIT,
    OhlcvBatch,
)
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import SymbolMapping, resolve_mvp_spot_symbol
from crypto_chatbot.schemas.market import (
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


class CcxtMarketDataProvider:
    """Public Binance spot OHLCV provider backed by CCXT async_support."""

    def __init__(
        self,
        *,
        client: CcxtOhlcvClient | None = None,
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

        self._owns_client = client is None
        self._client: CcxtOhlcvClient = (
            client
            if client is not None
            else self._build_default_client(timeout_seconds, public_api_base_url)
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
    ) -> CcxtOhlcvClient:
        # Public spot market data only: never pass exchange credentials in MVP (ADR 0001).
        exchange: CcxtOhlcvClient = ccxt_async.binance(
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
        return exchange

    async def _fetch_once(
        self,
        ccxt_symbol: str,
        timeframe: Timeframe,
        fetch_limit: int,
    ) -> Sequence[Sequence[Any]]:
        return await asyncio.wait_for(
            self._client.fetch_ohlcv(
                ccxt_symbol,
                timeframe=timeframe.value,
                limit=fetch_limit,
            ),
            timeout=self._timeout_seconds,
        )

    async def _fetch_with_retries(
        self,
        ccxt_symbol: str,
        timeframe: Timeframe,
        fetch_limit: int,
    ) -> Sequence[Sequence[Any]]:
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
                return await self._fetch_once(ccxt_symbol, timeframe, fetch_limit)
        raise RuntimeError("unreachable")  # pragma: no cover

    async def get_closed_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = DEFAULT_CANDLE_LIMIT,
    ) -> OhlcvBatch:
        """Fetch closed spot candles with retries, timeout, and deterministic quality checks."""
        if limit < 1 or limit > MAX_CANDLE_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_CANDLE_LIMIT}")

        mapping = resolve_mvp_spot_symbol(symbol)
        # Request one extra row when below venue cap so dropping the open candle preserves `limit`.
        fetch_limit = min(limit + 1, MAX_CANDLE_LIMIT)

        extra_flags: list[QualityFlag] = []
        client_id = getattr(self._client, "id", "").lower()
        if client_id and client_id != mapping.venue:
            extra_flags.append(QualityFlag.SYMBOL_MISMATCH)

        try:
            raw_rows = await self._fetch_with_retries(
                mapping.ccxt_symbol,
                timeframe,
                fetch_limit,
            )
        except TimeoutError:
            return self._provider_error_batch(
                mapping=mapping,
                timeframe=timeframe,
                message=(
                    f"Provider timeout fetching {mapping.canonical_symbol} {timeframe.value} "
                    f"after {self._max_attempts} attempt(s)"
                ),
            )
        except ccxt.BaseError as exc:
            return self._provider_error_batch(
                mapping=mapping,
                timeframe=timeframe,
                message=(
                    f"Provider error ({type(exc).__name__}) fetching "
                    f"{mapping.canonical_symbol} {timeframe.value}"
                ),
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
