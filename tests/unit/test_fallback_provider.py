from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import ValidationError

from crypto_chatbot.config import Settings
from crypto_chatbot.market_data.base import OhlcvBatch
from crypto_chatbot.market_data.fallback_provider import (
    FallbackDerivativesLiquidityProvider,
    FallbackMarketDataProvider,
    build_futures_providers,
    normalize_exchange_chain,
)
from crypto_chatbot.schemas.derivatives import (
    FundingRateBatch,
    FundingRateObservation,
    LiquidationBatch,
    OpenInterestBatch,
    OrderBookBatch,
)
from crypto_chatbot.schemas.market import (
    MarketCandle,
    MarketType,
    ProviderMetadata,
    QualityFlag,
    Timeframe,
)

NOW = datetime(2026, 11, 23, 12, 0, tzinfo=UTC)


def _sample_ohlcv_batch(*, venue: str, has_error: bool = False) -> OhlcvBatch:
    meta = ProviderMetadata(
        provider="ccxt",
        venue=venue,
        market_type=MarketType.PERPETUAL,
        base_unit="BTC",
        quote_currency="USDT",
    )
    if has_error:
        return OhlcvBatch(
            metadata=meta,
            canonical_symbol="BTC/USDT:USDT",
            provider_symbol=f"{venue.upper()}:BTCUSDT.P",
            timeframe=Timeframe.H4,
            received_time=NOW,
            candles=(),
            quality_flags=(QualityFlag.PROVIDER_ERROR,),
            error_message=f"{venue} down",
        )
    candle = MarketCandle(
        provider="ccxt",
        venue=venue,
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol=f"{venue.upper()}:BTCUSDT.P",
        market_type=MarketType.PERPETUAL,
        timeframe=Timeframe.H4,
        open_time=NOW - timedelta(hours=8),
        close_time=NOW - timedelta(hours=4),
        received_time=NOW,
        available_time=NOW - timedelta(hours=4),
        open=90000.0,
        high=90500.0,
        low=89500.0,
        close=90200.0,
        base_volume=100.0,
        base_unit="BTC",
        quote_currency="USDT",
        is_closed=True,
    )
    return OhlcvBatch(
        metadata=meta,
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol=f"{venue.upper()}:BTCUSDT.P",
        timeframe=Timeframe.H4,
        received_time=NOW,
        candles=(candle,),
        quality_flags=(),
    )


class _StubMarketProvider:
    def __init__(self, batch: OhlcvBatch) -> None:
        self._batch = batch
        self.calls = 0
        self.closed = False

    async def get_closed_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = 1000,
    ) -> OhlcvBatch:
        del symbol, timeframe, limit
        self.calls += 1
        return self._batch

    async def close(self) -> None:
        self.closed = True


class _StubDerivProvider:
    def __init__(self, venue: str, *, fail: bool = False) -> None:
        self._venue = venue
        self._fail = fail
        self.calls = 0
        self.closed = False

    async def get_funding_rate(self, symbol: str) -> FundingRateBatch:
        del symbol
        self.calls += 1
        flags = (QualityFlag.PROVIDER_ERROR,) if self._fail else ()
        obs = (
            None
            if self._fail
            else FundingRateObservation(
                provider="ccxt",
                venue=self._venue,
                canonical_symbol="BTC/USDT:USDT",
                provider_symbol=f"{self._venue.upper()}:BTCUSDT.P",
                observed_time=NOW,
                received_time=NOW,
                available_time=NOW,
                funding_rate=0.0001,
                funding_interval_hours=8.0,
                rate_per_8h=0.0001,
                annualized_rate=0.1095,
            )
        )
        return FundingRateBatch(
            canonical_symbol="BTC/USDT:USDT",
            provider_symbol=f"{self._venue.upper()}:BTCUSDT.P",
            venue=self._venue,
            received_time=NOW,
            observation=obs,
            quality_flags=flags,
            error_message=f"{self._venue} funding err" if self._fail else None,
        )

    async def get_open_interest(
        self,
        symbol: str,
        timeframe: Timeframe = Timeframe.H4,
        history_limit: int = 30,
        *,
        reference_price: float | None = None,
    ) -> OpenInterestBatch:
        del symbol, history_limit, reference_price
        self.calls += 1
        return OpenInterestBatch(
            canonical_symbol="BTC/USDT:USDT",
            provider_symbol=f"{self._venue.upper()}:BTCUSDT.P",
            venue=self._venue,
            timeframe=timeframe,
            received_time=NOW,
            quality_flags=(QualityFlag.PROVIDER_ERROR,) if self._fail else (),
            error_message=f"{self._venue} oi err" if self._fail else None,
        )

    async def get_order_book(
        self,
        symbol: str,
        *,
        market_type: MarketType = MarketType.SPOT,
        limit: int = 100,
    ) -> OrderBookBatch:
        del symbol, market_type, limit
        self.calls += 1
        return OrderBookBatch(
            canonical_symbol="BTC/USDT:USDT",
            provider_symbol=f"{self._venue.upper()}:BTCUSDT.P",
            venue=self._venue,
            market_type=MarketType.PERPETUAL,
            received_time=NOW,
            quality_flags=(QualityFlag.PROVIDER_ERROR,) if self._fail else (),
            error_message=f"{self._venue} ob err" if self._fail else None,
        )

    async def get_liquidations(
        self,
        symbol: str,
        *,
        limit: int = 50,
    ) -> LiquidationBatch:
        del symbol, limit
        self.calls += 1
        return LiquidationBatch(
            canonical_symbol="BTC/USDT:USDT",
            provider_symbol=f"{self._venue.upper()}:BTCUSDT.P",
            venue=self._venue,
            received_time=NOW,
            source_available=not self._fail,
            quality_flags=(QualityFlag.PROVIDER_ERROR,) if self._fail else (),
            error_message=f"{self._venue} liq err" if self._fail else None,
        )

    async def close(self) -> None:
        self.closed = True


@pytest.mark.asyncio
async def test_fallback_market_provider_skips_secondary_when_primary_succeeds() -> None:
    p1 = _StubMarketProvider(_sample_ohlcv_batch(venue="gate", has_error=False))
    p2 = _StubMarketProvider(_sample_ohlcv_batch(venue="bingx", has_error=False))

    async with FallbackMarketDataProvider([p1, p2]) as fb:
        batch = await fb.get_closed_ohlcv("BTC/USDT:USDT", Timeframe.H4, limit=10)

    assert batch.metadata.venue == "gate"
    assert not batch.has_provider_error
    assert p1.calls == 1
    assert p2.calls == 0
    assert p1.closed
    assert p2.closed


@pytest.mark.asyncio
async def test_fallback_market_provider_falls_back_on_error_preserving_venue_metadata() -> None:
    p1 = _StubMarketProvider(_sample_ohlcv_batch(venue="gate", has_error=True))
    p2 = _StubMarketProvider(_sample_ohlcv_batch(venue="bingx", has_error=False))

    fb = FallbackMarketDataProvider([p1, p2])
    batch = await fb.get_closed_ohlcv("BTC/USDT:USDT", Timeframe.H4, limit=10)

    assert p1.calls == 1
    assert p2.calls == 1
    assert batch.metadata.venue == "bingx"
    assert batch.provider_symbol == "BINGX:BTCUSDT.P"
    assert not batch.has_provider_error


@pytest.mark.asyncio
async def test_fallback_market_provider_returns_error_summary_when_all_fail() -> None:
    p1 = _StubMarketProvider(_sample_ohlcv_batch(venue="gate", has_error=True))
    p2 = _StubMarketProvider(_sample_ohlcv_batch(venue="bingx", has_error=True))

    fb = FallbackMarketDataProvider([p1, p2])
    batch = await fb.get_closed_ohlcv("BTC/USDT:USDT", Timeframe.H4, limit=10)

    assert batch.has_provider_error
    assert batch.error_message is not None
    assert "gate: gate down" in batch.error_message
    assert "bingx: bingx down" in batch.error_message


@pytest.mark.asyncio
async def test_fallback_derivatives_provider_falls_back_to_healthy_exchange() -> None:
    d1 = _StubDerivProvider("gate", fail=True)
    d2 = _StubDerivProvider("binance", fail=False)

    async with FallbackDerivativesLiquidityProvider([d1, d2]) as fb:
        funding = await fb.get_funding_rate("BTC/USDT:USDT")
        oi = await fb.get_open_interest("BTC/USDT:USDT")
        ob = await fb.get_order_book("BTC/USDT:USDT")
        liq = await fb.get_liquidations("BTC/USDT:USDT")

    assert funding.venue == "binance"
    assert oi.venue == "binance"
    assert ob.venue == "binance"
    assert liq.venue == "binance"
    assert QualityFlag.PROVIDER_ERROR not in funding.quality_flags
    assert d1.calls == 4
    assert d2.calls == 4
    assert d1.closed
    assert d2.closed


def test_settings_exchange_chain_normalization_and_validation() -> None:
    base_kwargs: dict[str, Any] = {
        "_env_file": None,
        "telegram_bot_token": "123:abc",
        "gemini_api_key": "gem-key",
        "gemini_model": "gemini-2.5-flash",
        "database_url": "postgresql+psycopg://u:p@localhost:5432/db",
        "allowed_telegram_user_ids": "111",
        "futures_exchange": "gate",
        "futures_exchange_fallbacks": "bingx, gate, binance",
    }
    settings = Settings(**base_kwargs)
    assert settings.futures_exchange_chain == ("gate", "bingx", "binance")

    assert normalize_exchange_chain(["GATE", "bingx", "gate"]) == ("gate", "bingx")
    assert normalize_exchange_chain([]) == ("binance",)

    with pytest.raises(ValidationError):
        Settings(**{**base_kwargs, "futures_exchange_fallbacks": "kraken"})

    with pytest.raises(ValueError, match="unsupported exchange"):
        normalize_exchange_chain(["okx"])


@pytest.mark.asyncio
async def test_build_futures_providers_returns_single_or_fallback() -> None:
    p_single, d_single = build_futures_providers(["gate"])
    assert not isinstance(p_single, FallbackMarketDataProvider)
    assert not isinstance(d_single, FallbackDerivativesLiquidityProvider)
    await p_single.close()
    await d_single.close()

    p_fb, d_fb = build_futures_providers(["gate", "bingx"])
    assert isinstance(p_fb, FallbackMarketDataProvider)
    assert isinstance(d_fb, FallbackDerivativesLiquidityProvider)
    await p_fb.close()
    await d_fb.close()


@pytest.mark.asyncio
async def test_ccxt_1000_perp_fallback_preserves_non_binance_venue_metadata() -> None:
    import ccxt

    from crypto_chatbot.market_data.ccxt_provider import CcxtMarketDataProvider

    class _FakeGateClient:
        id = "gateio"

        def __init__(self) -> None:
            self.symbols_called: list[str] = []

        async def fetch_ohlcv(
            self,
            symbol: str,
            timeframe: str = "4h",
            since: int | None = None,
            limit: int | None = 100,
        ) -> list[list[Any]]:
            del timeframe, since, limit
            self.symbols_called.append(symbol)
            if symbol == "PEPE/USDT:USDT":
                raise ccxt.BadSymbol("PEPE/USDT:USDT not found")
            t_ms = int((NOW - timedelta(hours=8)).timestamp() * 1000)
            return [[t_ms, 0.01, 0.012, 0.009, 0.011, 5000.0]]

        async def close(self) -> None:
            return None

    client = _FakeGateClient()
    prov = CcxtMarketDataProvider(
        exchange_id="gate",
        market_type=MarketType.PERPETUAL,
        client=client,
        clock=lambda: NOW,
    )
    batch = await prov.get_closed_ohlcv("PEPE", Timeframe.H4, limit=5)
    assert client.symbols_called == ["PEPE/USDT:USDT", "1000PEPE/USDT:USDT"]
    assert batch.metadata.venue == "gate"
    assert batch.provider_symbol == "GATE:1000PEPEUSDT.P"
    assert len(batch.candles) == 1
    assert batch.candles[0].venue == "gate"
