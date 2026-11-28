from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from crypto_chatbot.analysis.service import AnalysisService
from crypto_chatbot.config import Settings
from crypto_chatbot.market_data.base import DEFAULT_CANDLE_LIMIT, OhlcvBatch
from crypto_chatbot.market_data.fallback_provider import (
    FallbackDerivativesLiquidityProvider,
    FallbackMarketDataProvider,
)
from crypto_chatbot.market_data.venue_health import VenueHealthCache, is_venue_block_error
from crypto_chatbot.schemas.analysis import AnalysisSnapshot
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

NOW = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)


class _InMemoryMarketRepo:
    def __init__(self) -> None:
        self.candles: list[MarketCandle] = []
        self.snapshots: list[AnalysisSnapshot] = []

    def save_candles(self, candles: Sequence[MarketCandle]) -> int:
        self.candles.extend(candles)
        return len(candles)

    def load_candles(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        market_type: MarketType,
        timeframe: Timeframe,
        start_open_time: datetime | None = None,
        end_open_time: datetime | None = None,
        end_close_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[MarketCandle, ...]:
        del venue, canonical_symbol, market_type, timeframe
        del start_open_time, end_open_time, end_close_time, limit
        return ()

    def save_snapshot(self, snapshot: AnalysisSnapshot) -> None:
        self.snapshots.append(snapshot)

    def get_snapshot(self, snapshot_id: str) -> AnalysisSnapshot | None:
        del snapshot_id
        return None


def _make_batch(
    *,
    venue: str,
    tf: Timeframe,
    count: int = 80,
    error_message: str | None = None,
) -> OhlcvBatch:
    meta = ProviderMetadata(
        provider="ccxt",
        venue=venue,
        market_type=MarketType.PERPETUAL,
        base_unit="BTC",
        quote_currency="USDT",
    )
    if error_message is not None:
        return OhlcvBatch(
            metadata=meta,
            canonical_symbol="BTC/USDT:USDT",
            provider_symbol=f"{venue.upper()}:BTCUSDT.P",
            timeframe=tf,
            received_time=NOW,
            candles=(),
            quality_flags=(QualityFlag.PROVIDER_ERROR,),
            error_message=error_message,
        )
    candles: list[MarketCandle] = []
    step = tf.duration
    for i in range(count):
        close_t = NOW - (count - 1 - i) * step
        open_t = close_t - step
        base = 60000.0 + i * 10.0
        candles.append(
            MarketCandle(
                provider="ccxt",
                venue=venue,
                canonical_symbol="BTC/USDT:USDT",
                provider_symbol=f"{venue.upper()}:BTCUSDT.P",
                market_type=MarketType.PERPETUAL,
                timeframe=tf,
                open_time=open_t,
                close_time=close_t,
                received_time=NOW,
                available_time=close_t,
                open=base,
                high=base + 25.0,
                low=base - 25.0,
                close=base + 12.0,
                base_volume=150.0,
                base_unit="BTC",
                quote_currency="USDT",
                is_closed=True,
            )
        )
    return OhlcvBatch(
        metadata=meta,
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol=f"{venue.upper()}:BTCUSDT.P",
        timeframe=tf,
        received_time=NOW,
        candles=tuple(candles),
        quality_flags=(),
    )


class _VenueOhlcvStub:
    def __init__(
        self,
        venue: str,
        *,
        failing_tfs: dict[Timeframe, str] | None = None,
    ) -> None:
        self.venue = venue
        self._failing_tfs = failing_tfs or {}
        self.calls: list[Timeframe] = []

    async def get_closed_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        limit: int = DEFAULT_CANDLE_LIMIT,
    ) -> OhlcvBatch:
        del symbol
        self.calls.append(timeframe)
        if timeframe in self._failing_tfs:
            return _make_batch(
                venue=self.venue,
                tf=timeframe,
                error_message=self._failing_tfs[timeframe],
            )
        return _make_batch(venue=self.venue, tf=timeframe, count=limit)

    async def close(self) -> None:
        return None


class _VenueDerivStub:
    def __init__(self, venue: str) -> None:
        self.venue = venue
        self.calls = 0

    async def get_funding_rate(self, symbol: str) -> FundingRateBatch:
        del symbol
        self.calls += 1
        return FundingRateBatch(
            canonical_symbol="BTC/USDT:USDT",
            provider_symbol=f"{self.venue.upper()}:BTCUSDT.P",
            venue=self.venue,
            received_time=NOW,
            observation=FundingRateObservation(
                provider="ccxt",
                venue=self.venue,
                canonical_symbol="BTC/USDT:USDT",
                provider_symbol=f"{self.venue.upper()}:BTCUSDT.P",
                market_type=MarketType.PERPETUAL,
                observed_time=NOW - timedelta(hours=1),
                received_time=NOW,
                available_time=NOW - timedelta(hours=1),
                funding_rate=0.0001,
                funding_interval_hours=8.0,
                rate_per_8h=0.0001,
                annualized_rate=0.1095,
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
        del symbol, history_limit, reference_price
        return OpenInterestBatch(
            canonical_symbol="BTC/USDT:USDT",
            provider_symbol=f"{self.venue.upper()}:BTCUSDT.P",
            venue=self.venue,
            timeframe=timeframe,
            received_time=NOW,
        )

    async def get_order_book(
        self,
        symbol: str,
        *,
        market_type: MarketType = MarketType.PERPETUAL,
        limit: int = 100,
    ) -> OrderBookBatch:
        del symbol, limit
        return OrderBookBatch(
            canonical_symbol="BTC/USDT:USDT",
            provider_symbol=f"{self.venue.upper()}:BTCUSDT.P",
            venue=self.venue,
            market_type=market_type,
            received_time=NOW,
        )

    async def get_liquidations(self, symbol: str, *, limit: int = 20) -> LiquidationBatch:
        del symbol, limit
        return LiquidationBatch(
            canonical_symbol="BTC/USDT:USDT",
            provider_symbol=f"{self.venue.upper()}:BTCUSDT.P",
            venue=self.venue,
            received_time=NOW,
        )

    async def close(self) -> None:
        return None


def _base_settings_kwargs() -> dict[str, Any]:
    return {
        "_env_file": None,
        "telegram_bot_token": "123:abc",
        "gemini_api_key": "gem-key",
        "gemini_model": "gemini-2.5-flash",
        "database_url": "postgresql+psycopg://u:p@localhost:5432/db",
        "allowed_telegram_user_ids": "111",
    }


def test_d01_config_defaults_gateio_primary_and_binance_disabled() -> None:
    settings = Settings(**_base_settings_kwargs())
    assert settings.primary_venue == "gateio"
    assert settings.binance_enabled is False
    assert settings.venue_cooldown_min == 360
    assert settings.futures_exchange_chain == ("gate",)

    with_fallback = Settings(
        **_base_settings_kwargs(),
        venue_fallbacks=("bingx", "binance"),
    )
    # binance filtered out while binance_enabled=False
    assert with_fallback.futures_exchange_chain == ("gate", "bingx")

    with_binance = Settings(
        **_base_settings_kwargs(),
        venue_fallbacks=("bingx", "binance"),
        binance_enabled=True,
    )
    assert with_binance.futures_exchange_chain == ("gate", "bingx", "binance")


@pytest.mark.asyncio
async def test_d01_single_venue_bundle_never_mixes_when_1h_succeeds_and_4h_fails() -> None:
    # Venue A (gate) succeeds on 1h and 1d, fails on 4h.
    venue_a = _VenueOhlcvStub(
        "gate",
        failing_tfs={Timeframe.H4: "gate 4h timeout"},
    )
    venue_b = _VenueOhlcvStub("bingx")
    deriv_a = _VenueDerivStub("gate")
    deriv_b = _VenueDerivStub("bingx")

    health = VenueHealthCache(default_cooldown_minutes=360)
    provider = FallbackMarketDataProvider([venue_a, venue_b], health_cache=health)
    deriv_provider = FallbackDerivativesLiquidityProvider([deriv_a, deriv_b], health_cache=health)
    repo = _InMemoryMarketRepo()

    service = AnalysisService(
        provider=provider,
        repository=repo,
        derivatives_provider=deriv_provider,
    )
    snapshot = await service.analyze("BTC", Timeframe.H4, include_mtf=True, persist=True)

    # Entire bundle & persisted rows must come exclusively from venue B (bingx).
    assert snapshot.venue == "bingx"
    assert snapshot.derivatives is not None
    assert snapshot.derivatives.funding_venue == "bingx"
    assert repo.candles
    assert {c.venue for c in repo.candles} == {"bingx"}
    assert {s.venue for s in repo.snapshots} == {"bingx"}
    assert deriv_a.calls == 0
    assert deriv_b.calls == 1

    # Multi-style path must also obey the exact same atomic single-venue guarantee.
    snaps = await service.analyze_multi_style("BTC", (Timeframe.H1, Timeframe.H4))
    assert snaps[Timeframe.H1].venue == "bingx"
    assert snaps[Timeframe.H4].venue == "bingx"


@pytest.mark.asyncio
async def test_d01_blocked_venue_enters_cooldown_and_is_skipped_without_calling() -> None:
    now_mono = [1000.0]
    health = VenueHealthCache(default_cooldown_minutes=360, clock=lambda: now_mono[0])

    venue_a = _VenueOhlcvStub(
        "binance",
        failing_tfs={
            Timeframe.H4: "HTTP 451 Service unavailable from a restricted location",
        },
    )
    venue_b = _VenueOhlcvStub("gate")
    provider = FallbackMarketDataProvider([venue_a, venue_b], health_cache=health)
    service = AnalysisService(provider=provider)

    snap1 = await service.analyze("BTC", Timeframe.H4, include_mtf=True, persist=False)
    assert snap1.venue == "gate"
    assert len(venue_a.calls) > 0
    assert not health.is_available("binance")
    assert health.cooldown_remaining_seconds("binance") == pytest.approx(360 * 60.0)

    # Next request: venue A must be skipped completely (0 calls, instant skip).
    venue_a.calls.clear()
    snap2 = await service.analyze("BTC", Timeframe.H4, include_mtf=True, persist=False)
    assert snap2.venue == "gate"
    assert venue_a.calls == []

    # After 360 minutes pass, cooldown clears and venue A is eligible again.
    now_mono[0] += 360 * 60.0 + 1.0
    assert health.is_available("binance")


@pytest.mark.parametrize(
    ("msg", "expected"),
    [
        ("HTTP 451 Unavailable For Legal Reasons", True),
        ("403 Forbidden cloudflare", True),
        ("Connection refused by peer", True),
        (ConnectionRefusedError("refused"), True),
        ("FOO/USDT:USDT not found", False),
    ],
)
def test_d01_is_venue_block_error_classifier(msg: Exception | str, expected: bool) -> None:
    assert is_venue_block_error(msg) is expected
