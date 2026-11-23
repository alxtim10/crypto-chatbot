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
from crypto_chatbot.market_data.symbols import (
    SymbolMapping,
    resolve_symbol,
)
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
        """Fetch closed perpetual futures candles with retries, timeout, and quality checks."""
        if limit < 1 or limit > MAX_CANDLE_LIMIT:
            raise ValueError(f"limit must be between 1 and {MAX_CANDLE_LIMIT}")

        mapping = resolve_symbol(symbol, market_type=self._market_type)
        if self._market_type is MarketType.PERPETUAL and self._exchange_id != "binance":
            provider_sym = f"{self._exchange_id.upper()}:{mapping.base}{mapping.quote}.P"
            mapping = mapping.model_copy(
                update={
                    "venue": self._exchange_id,
                    "provider_symbol": provider_sym,
                }
            )
        # Request one extra row when below venue cap so dropping the open candle preserves `limit`.
        fetch_limit = min(limit + 1, MAX_CANDLE_LIMIT)

        extra_flags: list[QualityFlag] = []
        client_id = getattr(self._client, "id", "").lower()
        if client_id and not (
            client_id.startswith(mapping.venue) or mapping.venue.startswith(client_id)
        ):
            extra_flags.append(QualityFlag.SYMBOL_MISMATCH)

        try:
            raw_rows = await self._fetch_with_retries(
                mapping.ccxt_symbol,
                timeframe,
                fetch_limit,
            )
        except ccxt.BadSymbol as exc:
            if (
                self._market_type is MarketType.PERPETUAL
                and mapping.base in _BINANCE_1000_PERP_BASES
            ):
                fallback_mapping = resolve_symbol(
                    f"1000{mapping.base}",
                    market_type=MarketType.PERPETUAL,
                )
                try:
                    raw_rows = await self._fetch_with_retries(
                        fallback_mapping.ccxt_symbol,
                        timeframe,
                        fetch_limit,
                    )
                    mapping = fallback_mapping
                except TimeoutError:
                    return self._provider_error_batch(
                        mapping=fallback_mapping,
                        timeframe=timeframe,
                        message=(
                            f"Provider timeout fetching {fallback_mapping.canonical_symbol} "
                            f"{timeframe.value} after {self._max_attempts} attempt(s)"
                        ),
                    )
                except ccxt.BaseError as fallback_exc:
                    return self._provider_error_batch(
                        mapping=fallback_mapping,
                        timeframe=timeframe,
                        message=(
                            f"Provider error ({type(fallback_exc).__name__}) fetching "
                            f"{fallback_mapping.canonical_symbol} {timeframe.value}"
                        ),
                    )
            else:
                return self._provider_error_batch(
                    mapping=mapping,
                    timeframe=timeframe,
                    message=(
                        f"Provider error ({type(exc).__name__}) fetching "
                        f"{mapping.canonical_symbol} {timeframe.value}"
                    ),
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
