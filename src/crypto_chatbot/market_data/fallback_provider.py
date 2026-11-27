"""Ordered multi-exchange fallback wrappers for OHLCV & derivatives liquidity (D-01)."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol, TypeVar

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
from crypto_chatbot.market_data.symbols import resolve_default_perp_symbol
from crypto_chatbot.market_data.venue_health import (
    DEFAULT_VENUE_COOLDOWN_MINUTES,
    VenueHealthCache,
    is_venue_block_error,
    normalize_venue_id,
)
from crypto_chatbot.schemas.derivatives import (
    FundingRateBatch,
    LiquidationBatch,
    OpenInterestBatch,
    OrderBookBatch,
)
from crypto_chatbot.schemas.market import (
    MarketType,
    ProviderMetadata,
    QualityFlag,
    Timeframe,
)

_BatchT = TypeVar(
    "_BatchT",
    FundingRateBatch,
    OpenInterestBatch,
    OrderBookBatch,
    LiquidationBatch,
)


def _provider_venue(provider: object, fallback_index: int = 0) -> str:
    raw = getattr(provider, "venue", None) or getattr(provider, "_exchange_id", None)
    if isinstance(raw, str) and raw.strip():
        return normalize_venue_id(raw)
    client = getattr(provider, "_client", None)
    client_id = getattr(client, "id", None)
    if isinstance(client_id, str) and client_id.strip():
        return normalize_venue_id(client_id)
    batch = getattr(provider, "_batch", None)
    if isinstance(batch, OhlcvBatch):
        return normalize_venue_id(batch.metadata.venue)
    return f"provider-{fallback_index}"


def _error_batch_for_request(
    symbol: str,
    timeframe: Timeframe,
    *,
    venue: str,
    error_message: str,
) -> OhlcvBatch:
    mapping = resolve_default_perp_symbol(symbol)
    return OhlcvBatch(
        metadata=ProviderMetadata(
            provider="ccxt",
            venue=venue,
            market_type=mapping.market_type,
            base_unit=mapping.base,
            quote_currency=mapping.quote,
        ),
        canonical_symbol=mapping.canonical_symbol,
        provider_symbol=mapping.provider_symbol,
        timeframe=timeframe,
        received_time=datetime.now(UTC),
        candles=(),
        quality_flags=(QualityFlag.PROVIDER_ERROR,),
        error_message=error_message,
    )


@dataclass(frozen=True, slots=True)
class BundleFetchResult:
    """Result of an atomic single-venue multi-timeframe OHLCV bundle fetch."""

    venue: str
    provider: MarketDataProvider | None
    batches: dict[tuple[str, Timeframe], OhlcvBatch]
    succeeded: bool
    error_message: str | None = None


class BundleMarketDataProvider(Protocol):
    """Protocol implemented by providers capable of atomic single-venue TF bundle fetching."""

    async def get_closed_ohlcv_bundle(
        self,
        requests: Sequence[tuple[str, Timeframe, int]],
        *,
        require_non_empty: bool = True,
    ) -> BundleFetchResult: ...


class FallbackMarketDataProvider:
    """Try multiple `MarketDataProvider` instances in priority order with health cooldown."""

    def __init__(
        self,
        providers: Sequence[MarketDataProvider],
        *,
        health_cache: VenueHealthCache | None = None,
        cooldown_minutes: int = DEFAULT_VENUE_COOLDOWN_MINUTES,
    ) -> None:
        if not providers:
            raise ValueError("FallbackMarketDataProvider requires at least one provider")
        self._providers: tuple[MarketDataProvider, ...] = tuple(providers)
        self._health_cache = health_cache or VenueHealthCache(
            default_cooldown_minutes=cooldown_minutes
        )

    @property
    def health_cache(self) -> VenueHealthCache:
        return self._health_cache

    @property
    def providers(self) -> tuple[MarketDataProvider, ...]:
        return self._providers

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

    async def _fetch_single_from_provider(
        self,
        provider: MarketDataProvider,
        symbol: str,
        timeframe: Timeframe,
        limit: int,
        *,
        use_history: bool = False,
    ) -> OhlcvBatch:
        if use_history and isinstance(provider, HistoryMarketDataProvider):
            return await provider.get_closed_ohlcv_history(symbol, timeframe, limit=limit)
        return await provider.get_closed_ohlcv(symbol, timeframe, limit=limit)

    async def get_closed_ohlcv_bundle(
        self,
        requests: Sequence[tuple[str, Timeframe, int]],
        *,
        require_non_empty: bool = True,
    ) -> BundleFetchResult:
        """Fetch ALL requested `(symbol, timeframe, limit)` from ONE venue atomically."""
        if not requests:
            raise ValueError("requests must not be empty")

        errors: list[str] = []
        last_batches: dict[tuple[str, Timeframe], OhlcvBatch] | None = None
        last_venue = _provider_venue(self._providers[0], 0)

        for idx, provider in enumerate(self._providers):
            venue = _provider_venue(provider, idx)
            if not self._health_cache.is_available(venue):
                reason = self._health_cache.blocked_reason(venue) or "cooldown active"
                errors.append(f"{venue}: skipped ({reason})")
                continue

            results = await asyncio.gather(
                *(
                    self._fetch_single_from_provider(provider, sym, tf, limit)
                    for sym, tf, limit in requests
                ),
                return_exceptions=True,
            )

            candidate_batches: dict[tuple[str, Timeframe], OhlcvBatch] = {}
            failed_reason: str | None = None
            for (sym, tf, _), item in zip(requests, results, strict=True):
                if isinstance(item, BaseException):
                    msg = str(item) or type(item).__name__
                    failed_reason = f"{tf.value}: {msg}"
                    if isinstance(item, Exception) and is_venue_block_error(item):
                        self._health_cache.mark_unavailable(venue, reason=msg)
                    candidate_batches[(sym, tf)] = _error_batch_for_request(
                        sym,
                        tf,
                        venue=venue,
                        error_message=failed_reason,
                    )
                    continue

                candidate_batches[(sym, tf)] = item
                batch_venue = normalize_venue_id(item.metadata.venue)
                venue = batch_venue or venue
                if item.has_provider_error:
                    msg = item.error_message or "provider error"
                    if is_venue_block_error(msg):
                        self._health_cache.mark_unavailable(venue, reason=msg)
                    if failed_reason is None:
                        failed_reason = f"{tf.value}: {msg}"
                elif require_non_empty and not item.candles and failed_reason is None:
                    failed_reason = f"{tf.value}: empty closed candles"

            last_batches = candidate_batches
            last_venue = venue
            if failed_reason is None:
                return BundleFetchResult(
                    venue=venue,
                    provider=provider,
                    batches=candidate_batches,
                    succeeded=True,
                )
            errors.append(f"{venue}: {failed_reason}")

        summary = "; ".join(errors) if errors else "All venues unavailable"
        if last_batches is None:
            fallback_batches = {
                (sym, tf): _error_batch_for_request(
                    sym,
                    tf,
                    venue=last_venue,
                    error_message=summary,
                )
                for sym, tf, _ in requests
            }
            return BundleFetchResult(
                venue=last_venue,
                provider=None,
                batches=fallback_batches,
                succeeded=False,
                error_message=summary,
            )
        annotated = {
            key: (
                batch.model_copy(update={"error_message": summary})
                if batch.has_provider_error
                else batch
            )
            for key, batch in last_batches.items()
        }
        return BundleFetchResult(
            venue=last_venue,
            provider=None,
            batches=annotated,
            succeeded=False,
            error_message=summary,
        )

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

        for idx, provider in enumerate(self._providers):
            venue = _provider_venue(provider, idx)
            if not self._health_cache.is_available(venue):
                reason = self._health_cache.blocked_reason(venue) or "cooldown active"
                errors.append(f"{venue}: skipped ({reason})")
                continue

            try:
                batch = await self._fetch_single_from_provider(
                    provider,
                    symbol,
                    timeframe,
                    limit,
                    use_history=use_history,
                )
            except Exception as exc:
                msg = str(exc) or type(exc).__name__
                if is_venue_block_error(exc):
                    self._health_cache.mark_unavailable(venue, reason=msg)
                errors.append(f"{venue}: {msg}")
                continue

            last_batch = batch
            batch_venue = normalize_venue_id(batch.metadata.venue)
            if not batch.has_provider_error:
                if batch.candles:
                    return batch
                if first_empty_valid is None:
                    first_empty_valid = batch
                errors.append(f"{batch_venue}: empty candles")
                continue

            err = batch.error_message or "provider error"
            if is_venue_block_error(err):
                self._health_cache.mark_unavailable(batch_venue, reason=err)
            errors.append(f"{batch_venue}: {err}")

        if first_empty_valid is not None:
            return first_empty_valid
        summary = "; ".join(errors) if errors else "All providers unavailable"
        if last_batch is None:
            return _error_batch_for_request(
                symbol,
                timeframe,
                venue=_provider_venue(self._providers[0], 0),
                error_message=summary,
            )
        return last_batch.model_copy(update={"error_message": summary})

    async def get_futures_tickers(self) -> FuturesTickerBatch:
        errors: list[str] = []
        last_batch: FuturesTickerBatch | None = None
        for idx, provider in enumerate(self._providers):
            if not isinstance(provider, FuturesTickerProvider):
                continue
            venue = _provider_venue(provider, idx)
            if not self._health_cache.is_available(venue):
                reason = self._health_cache.blocked_reason(venue) or "cooldown active"
                errors.append(f"{venue}: skipped ({reason})")
                continue
            batch = await provider.get_futures_tickers()
            last_batch = batch
            if not batch.has_provider_error and batch.tickers:
                return batch
            err = batch.error_message or "no tickers returned"
            if is_venue_block_error(err):
                self._health_cache.mark_unavailable(normalize_venue_id(batch.venue), reason=err)
            errors.append(f"{batch.venue}: {err}")
        if last_batch is not None:
            return last_batch.model_copy(update={"error_message": "; ".join(errors)})
        return FuturesTickerBatch(
            venue=_provider_venue(self._providers[0], 0),
            received_time=datetime.now(UTC),
            quality_flags=(QualityFlag.PROVIDER_ERROR,),
            error_message="; ".join(errors) or "No FuturesTickerProvider configured",
        )

    async def close(self) -> None:
        for provider in self._providers:
            await provider.close()

    async def __aenter__(self) -> FallbackMarketDataProvider:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        del exc_type, exc, tb
        await self.close()


class FallbackDerivativesLiquidityProvider:
    """Try multiple `DerivativesLiquidityProvider` instances in priority order."""

    def __init__(
        self,
        providers: Sequence[DerivativesLiquidityProvider],
        *,
        health_cache: VenueHealthCache | None = None,
        cooldown_minutes: int = DEFAULT_VENUE_COOLDOWN_MINUTES,
    ) -> None:
        if not providers:
            raise ValueError("FallbackDerivativesLiquidityProvider requires at least one provider")
        self._providers: tuple[DerivativesLiquidityProvider, ...] = tuple(providers)
        self._health_cache = health_cache or VenueHealthCache(
            default_cooldown_minutes=cooldown_minutes
        )

    @property
    def health_cache(self) -> VenueHealthCache:
        return self._health_cache

    def provider_for_venue(self, venue: str) -> DerivativesLiquidityProvider | None:
        target = normalize_venue_id(venue)
        for idx, provider in enumerate(self._providers):
            if _provider_venue(provider, idx) == target:
                return provider
        return None

    async def _try_chain(
        self,
        action: str,
        call: Callable[[DerivativesLiquidityProvider], Awaitable[_BatchT]],
        *,
        on_all_cooldown: Callable[[str, str], _BatchT],
    ) -> _BatchT:
        errors: list[str] = []
        last_batch: _BatchT | None = None
        for idx, provider in enumerate(self._providers):
            venue = _provider_venue(provider, idx)
            if not self._health_cache.is_available(venue):
                reason = self._health_cache.blocked_reason(venue) or "cooldown active"
                errors.append(f"{venue}: skipped ({reason})")
                continue
            batch = await call(provider)
            last_batch = batch
            if QualityFlag.PROVIDER_ERROR not in batch.quality_flags:
                return batch
            err = batch.error_message or "provider error"
            if is_venue_block_error(err):
                self._health_cache.mark_unavailable(normalize_venue_id(batch.venue), reason=err)
            errors.append(f"{batch.venue}: {err}")
        summary = f"All {action} providers failed ({'; '.join(errors)})"
        if last_batch is None:
            return on_all_cooldown(_provider_venue(self._providers[0], 0), summary)
        return last_batch.model_copy(update={"error_message": summary})

    async def get_funding_rate(self, symbol: str) -> FundingRateBatch:
        return await self._try_chain(
            "funding_rate",
            lambda p: p.get_funding_rate(symbol),
            on_all_cooldown=lambda v, msg: FundingRateBatch(
                canonical_symbol=resolve_default_perp_symbol(symbol).canonical_symbol,
                provider_symbol=resolve_default_perp_symbol(symbol).provider_symbol,
                venue=v,
                received_time=datetime.now(UTC),
                quality_flags=(QualityFlag.PROVIDER_ERROR,),
                error_message=msg,
            ),
        )

    async def get_open_interest(
        self,
        symbol: str,
        timeframe: Timeframe = Timeframe.H4,
        history_limit: int = 24,
        *,
        reference_price: float | None = None,
    ) -> OpenInterestBatch:
        return await self._try_chain(
            "open_interest",
            lambda p: p.get_open_interest(
                symbol,
                timeframe,
                history_limit,
                reference_price=reference_price,
            ),
            on_all_cooldown=lambda v, msg: OpenInterestBatch(
                canonical_symbol=resolve_default_perp_symbol(symbol).canonical_symbol,
                provider_symbol=resolve_default_perp_symbol(symbol).provider_symbol,
                venue=v,
                timeframe=timeframe,
                received_time=datetime.now(UTC),
                quality_flags=(QualityFlag.PROVIDER_ERROR,),
                error_message=msg,
            ),
        )

    async def get_order_book(
        self,
        symbol: str,
        *,
        market_type: MarketType = MarketType.PERPETUAL,
        limit: int = 100,
    ) -> OrderBookBatch:
        return await self._try_chain(
            "order_book",
            lambda p: p.get_order_book(symbol, market_type=market_type, limit=limit),
            on_all_cooldown=lambda v, msg: OrderBookBatch(
                canonical_symbol=resolve_default_perp_symbol(symbol).canonical_symbol,
                provider_symbol=resolve_default_perp_symbol(symbol).provider_symbol,
                venue=v,
                market_type=market_type,
                received_time=datetime.now(UTC),
                quality_flags=(QualityFlag.PROVIDER_ERROR,),
                error_message=msg,
            ),
        )

    async def get_liquidations(
        self,
        symbol: str,
        *,
        limit: int = 20,
    ) -> LiquidationBatch:
        return await self._try_chain(
            "liquidations",
            lambda p: p.get_liquidations(symbol, limit=limit),
            on_all_cooldown=lambda v, msg: LiquidationBatch(
                canonical_symbol=resolve_default_perp_symbol(symbol).canonical_symbol,
                provider_symbol=resolve_default_perp_symbol(symbol).provider_symbol,
                venue=v,
                received_time=datetime.now(UTC),
                quality_flags=(QualityFlag.PROVIDER_ERROR,),
                error_message=msg,
            ),
        )

    async def close(self) -> None:
        for provider in self._providers:
            await provider.close()

    async def __aenter__(self) -> FallbackDerivativesLiquidityProvider:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        del exc_type, exc, tb
        await self.close()


def normalize_exchange_chain(
    exchange_chain: Sequence[str],
    *,
    default: tuple[str, ...] = ("binance",),
) -> tuple[str, ...]:
    """Normalize and deduplicate an exchange priority chain, mapping aliases like `gateio`."""
    cleaned: list[str] = []
    for raw in exchange_chain:
        ex = normalize_venue_id(raw)
        if not ex:
            continue
        if ex not in SUPPORTED_PERP_EXCHANGES:
            raise ValueError(
                f"unsupported exchange '{raw}'; expected one of {sorted(SUPPORTED_PERP_EXCHANGES)}"
            )
        if ex not in cleaned:
            cleaned.append(ex)
    return tuple(cleaned) if cleaned else default


def build_futures_providers(
    exchange_chain: Sequence[str],
    *,
    cooldown_minutes: int = DEFAULT_VENUE_COOLDOWN_MINUTES,
    health_cache: VenueHealthCache | None = None,
) -> tuple[MarketDataProvider, DerivativesLiquidityProvider]:
    """Build single or fallback OHLCV + derivatives providers sharing one health cache."""
    chain = normalize_exchange_chain(exchange_chain)
    shared_cache = health_cache or VenueHealthCache(default_cooldown_minutes=cooldown_minutes)
    if len(chain) == 1:
        ex = chain[0]
        return (
            CcxtMarketDataProvider(exchange_id=ex),
            CcxtDerivativesLiquidityProvider(exchange_id=ex),
        )

    ohlcv_providers = [CcxtMarketDataProvider(exchange_id=ex) for ex in chain]
    deriv_providers = [CcxtDerivativesLiquidityProvider(exchange_id=ex) for ex in chain]
    return (
        FallbackMarketDataProvider(
            ohlcv_providers,
            health_cache=shared_cache,
            cooldown_minutes=cooldown_minutes,
        ),
        FallbackDerivativesLiquidityProvider(
            deriv_providers,
            health_cache=shared_cache,
            cooldown_minutes=cooldown_minutes,
        ),
    )
