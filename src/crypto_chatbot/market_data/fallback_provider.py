"""Multi-exchange auto-fallback providers and factory helpers."""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime
from types import TracebackType
from typing import TypeVar

from crypto_chatbot.market_data.base import (
    DEFAULT_CANDLE_LIMIT,
    DerivativesLiquidityProvider,
    FuturesTickerBatch,
    FuturesTickerProvider,
    HistoryMarketDataProvider,
    MarketDataProvider,
    OhlcvBatch,
)
from crypto_chatbot.market_data.ccxt_provider import (
    SUPPORTED_PERP_EXCHANGES,
    CcxtMarketDataProvider,
)
from crypto_chatbot.market_data.derivatives_provider import CcxtDerivativesLiquidityProvider
from crypto_chatbot.schemas.derivatives import (
    FundingRateBatch,
    LiquidationBatch,
    OpenInterestBatch,
    OrderBookBatch,
)
from crypto_chatbot.schemas.market import MarketType, QualityFlag, Timeframe

logger = logging.getLogger(__name__)

_BatchT = TypeVar(
    "_BatchT",
    FundingRateBatch,
    OpenInterestBatch,
    OrderBookBatch,
    LiquidationBatch,
)


class FallbackMarketDataProvider:
    """Try multiple `MarketDataProvider` instances in priority order until one succeeds."""

    def __init__(self, providers: Sequence[MarketDataProvider]) -> None:
        if not providers:
            raise ValueError("FallbackMarketDataProvider requires at least one provider")
        self._providers: tuple[MarketDataProvider, ...] = tuple(providers)

    async def get_closed_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = DEFAULT_CANDLE_LIMIT,
    ) -> OhlcvBatch:
        return await self._get_ohlcv_chain(symbol, timeframe, limit=limit, use_history=False)

    async def get_closed_ohlcv_history(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = DEFAULT_CANDLE_LIMIT,
    ) -> OhlcvBatch:
        return await self._get_ohlcv_chain(symbol, timeframe, limit=limit, use_history=True)

    async def _get_ohlcv_chain(
        self,
        symbol: str,
        timeframe: Timeframe,
        *,
        limit: int,
        use_history: bool,
    ) -> OhlcvBatch:
        errors: list[str] = []
        last_batch: OhlcvBatch | None = None
        first_empty_valid: OhlcvBatch | None = None

        for provider in self._providers:
            if use_history and isinstance(provider, HistoryMarketDataProvider):
                batch = await provider.get_closed_ohlcv_history(symbol, timeframe, limit=limit)
            else:
                batch = await provider.get_closed_ohlcv(symbol, timeframe, limit=limit)
            last_batch = batch
            if not batch.has_provider_error:
                if batch.candles:
                    return batch
                if first_empty_valid is None:
                    first_empty_valid = batch
                errors.append(f"{batch.metadata.venue}: empty candles")
                logger.info(
                    "Fallback OHLCV skipping empty response from %s for %s %s",
                    batch.metadata.venue,
                    symbol,
                    timeframe.value,
                )
                continue

            err = batch.error_message or "provider error"
            errors.append(f"{batch.metadata.venue}: {err}")
            logger.warning(
                "Fallback OHLCV provider %s failed for %s %s: %s",
                batch.metadata.venue,
                symbol,
                timeframe.value,
                err,
            )

        if first_empty_valid is not None:
            return first_empty_valid

        if last_batch is None:
            raise RuntimeError("FallbackMarketDataProvider has no providers configured")
        summary = " | ".join(errors) if errors else "All OHLCV providers failed"
        return last_batch.model_copy(update={"error_message": summary})

    async def get_futures_tickers(self) -> FuturesTickerBatch:
        errors: list[str] = []
        last_batch: FuturesTickerBatch | None = None
        for provider in self._providers:
            if not isinstance(provider, FuturesTickerProvider):
                continue
            batch = await provider.get_futures_tickers()
            last_batch = batch
            if not batch.has_provider_error and batch.tickers:
                return batch
            err = batch.error_message or "empty tickers"
            errors.append(f"{batch.venue}: {err}")
            logger.warning("Fallback futures tickers provider %s failed: %s", batch.venue, err)

        if last_batch is not None:
            summary = " | ".join(errors) if errors else "All ticker providers failed"
            return last_batch.model_copy(update={"error_message": summary})
        return FuturesTickerBatch(
            venue="unknown",
            received_time=datetime.now(UTC),
            quality_flags=(QualityFlag.PROVIDER_ERROR,),
            error_message="No FuturesTickerProvider configured in fallback chain",
        )

    async def close(self) -> None:
        for provider in self._providers:
            await provider.close()

    async def __aenter__(self) -> FallbackMarketDataProvider:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()


class FallbackDerivativesLiquidityProvider:
    """Try multiple `DerivativesLiquidityProvider` instances in priority order."""

    def __init__(self, providers: Sequence[DerivativesLiquidityProvider]) -> None:
        if not providers:
            raise ValueError("FallbackDerivativesLiquidityProvider requires at least one provider")
        self._providers: tuple[DerivativesLiquidityProvider, ...] = tuple(providers)

    async def _try_chain(
        self,
        action: str,
        call: Callable[[DerivativesLiquidityProvider], Awaitable[_BatchT]],
    ) -> _BatchT:
        errors: list[str] = []
        last_batch: _BatchT | None = None
        for provider in self._providers:
            batch = await call(provider)
            last_batch = batch
            if QualityFlag.PROVIDER_ERROR not in batch.quality_flags:
                return batch
            err = batch.error_message or "provider error"
            errors.append(f"{batch.venue}: {err}")
            logger.warning(
                "Fallback derivatives %s provider %s failed: %s",
                action,
                batch.venue,
                err,
            )

        if last_batch is None:
            raise RuntimeError("FallbackDerivativesLiquidityProvider has no providers configured")
        summary = " | ".join(errors) if errors else f"All providers failed for {action}"
        return last_batch.model_copy(update={"error_message": summary})

    async def get_funding_rate(self, symbol: str) -> FundingRateBatch:
        return await self._try_chain(
            "funding_rate",
            lambda p: p.get_funding_rate(symbol),
        )

    async def get_open_interest(
        self,
        symbol: str,
        timeframe: Timeframe = Timeframe.H4,
        history_limit: int = 30,
        *,
        reference_price: float | None = None,
    ) -> OpenInterestBatch:
        return await self._try_chain(
            "open_interest",
            lambda p: p.get_open_interest(
                symbol,
                timeframe=timeframe,
                history_limit=history_limit,
                reference_price=reference_price,
            ),
        )

    async def get_order_book(
        self,
        symbol: str,
        *,
        market_type: MarketType = MarketType.SPOT,
        limit: int = 100,
    ) -> OrderBookBatch:
        return await self._try_chain(
            "order_book",
            lambda p: p.get_order_book(symbol, market_type=market_type, limit=limit),
        )

    async def get_liquidations(
        self,
        symbol: str,
        *,
        limit: int = 50,
    ) -> LiquidationBatch:
        return await self._try_chain(
            "liquidations",
            lambda p: p.get_liquidations(symbol, limit=limit),
        )

    async def close(self) -> None:
        for provider in self._providers:
            await provider.close()

    async def __aenter__(self) -> FallbackDerivativesLiquidityProvider:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.close()


def normalize_exchange_chain(exchange_chain: Sequence[str]) -> tuple[str, ...]:
    """Normalize and deduplicate an exchange priority chain, defaulting to `binance`."""
    cleaned: list[str] = []
    for raw in exchange_chain:
        ex = raw.strip().lower()
        if not ex:
            continue
        if ex not in SUPPORTED_PERP_EXCHANGES:
            raise ValueError(
                f"unsupported exchange '{raw}'; expected one of {sorted(SUPPORTED_PERP_EXCHANGES)}"
            )
        if ex not in cleaned:
            cleaned.append(ex)
    return tuple(cleaned) if cleaned else ("binance",)


def build_futures_providers(
    exchange_chain: Sequence[str],
) -> tuple[MarketDataProvider, DerivativesLiquidityProvider]:
    """Build single or fallback OHLCV + derivatives providers for an ordered exchange chain."""
    chain = normalize_exchange_chain(exchange_chain)
    if len(chain) == 1:
        ex = chain[0]
        return (
            CcxtMarketDataProvider(exchange_id=ex),
            CcxtDerivativesLiquidityProvider(exchange_id=ex),
        )

    ohlcv_providers = [CcxtMarketDataProvider(exchange_id=ex) for ex in chain]
    deriv_providers = [CcxtDerivativesLiquidityProvider(exchange_id=ex) for ex in chain]
    return (
        FallbackMarketDataProvider(ohlcv_providers),
        FallbackDerivativesLiquidityProvider(deriv_providers),
    )
