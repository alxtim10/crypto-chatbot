from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from crypto_chatbot.analysis.momentum_screener import (
    evaluate_15m_momentum_setup,
    render_momentum_screener_report,
    screen_momentum_movers,
)
from crypto_chatbot.bot.commands import ScreenIntent, resolve_followup_intent
from crypto_chatbot.market_data.base import (
    FuturesTickerBatch,
    FuturesTickerSnapshot,
    OhlcvBatch,
)
from crypto_chatbot.market_data.fallback_provider import FallbackMarketDataProvider
from crypto_chatbot.schemas.analysis import SetupStatus
from crypto_chatbot.schemas.market import (
    MarketCandle,
    MarketType,
    ProviderMetadata,
    QualityFlag,
    Timeframe,
)

NOW = datetime(2026, 11, 24, 9, 0, tzinfo=UTC)


def _candle_15m(
    idx: int,
    *,
    o: float,
    h: float,
    low_p: float,
    c: float,
    symbol: str = "SUI/USDT:USDT",
    venue: str = "gate",
    total: int = 32,
) -> MarketCandle:
    open_t = NOW - (total - idx) * timedelta(minutes=15)
    close_t = open_t + timedelta(minutes=15)
    base = symbol.split("/")[0]
    return MarketCandle(
        provider="ccxt",
        venue=venue,
        canonical_symbol=symbol,
        provider_symbol=f"{venue.upper()}:{base}USDT.P",
        market_type=MarketType.PERPETUAL,
        timeframe=Timeframe.M15,
        open_time=open_t,
        close_time=close_t,
        received_time=close_t,
        available_time=close_t,
        open=o,
        high=h,
        low=low_p,
        close=c,
        base_volume=10_000.0,
        base_unit=base,
        quote_currency="USDT",
        is_closed=True,
    )


def _uptrend_history(
    *,
    count: int = 160,
    symbol: str = "SUI/USDT:USDT",
    last_two: tuple[tuple[float, float, float, float], tuple[float, float, float, float]],
) -> list[MarketCandle]:
    total = count + 2
    candles: list[MarketCandle] = []
    step = 0.002
    price = last_two[0][0] - count * step
    for i in range(count):
        o = price
        c = price + step
        candles.append(
            _candle_15m(
                i,
                o=o,
                h=c + 0.002,
                low_p=o - 0.002,
                c=c,
                symbol=symbol,
                total=total,
            )
        )
        price = c
    p1 = last_two[0]
    p2 = last_two[1]
    candles.append(
        _candle_15m(count, o=p1[0], h=p1[1], low_p=p1[2], c=p1[3], symbol=symbol, total=total)
    )
    candles.append(
        _candle_15m(count + 1, o=p2[0], h=p2[1], low_p=p2[2], c=p2[3], symbol=symbol, total=total)
    )
    return candles


def _downtrend_history(
    *,
    count: int = 160,
    symbol: str = "RED/USDT:USDT",
    last_two: tuple[tuple[float, float, float, float], tuple[float, float, float, float]],
) -> list[MarketCandle]:
    total = count + 2
    candles: list[MarketCandle] = []
    step = 0.003
    price = last_two[0][0] + count * step
    for i in range(count):
        o = price
        c = price - step
        candles.append(
            _candle_15m(
                i,
                o=o,
                h=o + 0.003,
                low_p=c - 0.003,
                c=c,
                symbol=symbol,
                total=total,
            )
        )
        price = c
    p1 = last_two[0]
    p2 = last_two[1]
    candles.append(
        _candle_15m(count, o=p1[0], h=p1[1], low_p=p1[2], c=p1[3], symbol=symbol, total=total)
    )
    candles.append(
        _candle_15m(count + 1, o=p2[0], h=p2[1], low_p=p2[2], c=p2[3], symbol=symbol, total=total)
    )
    return candles


def test_evaluate_15m_momentum_bullish_and_bearish_patterns() -> None:
    from crypto_chatbot.analysis.momentum_screener import MomentumPattern

    # 1. Bullish Engulfing in uptrend -> READY LONG, SL below min(prev.low, curr.low), TP2 = 2R
    bull_eng_candles = _uptrend_history(
        last_two=(
            (1.245, 1.248, 1.230, 1.236),  # red pullback bar with lower wick 1.230
            (1.235, 1.262, 1.233, 1.258),  # strong green engulfing closing above 1.245
        )
    )
    bull_eng = evaluate_15m_momentum_setup(
        bull_eng_candles,
        side="long",
        change_24h_pct=18.5,
        quote_volume_24h_usd=25_000_000.0,
    )
    assert bull_eng.status is SetupStatus.READY
    assert bull_eng.pattern is MomentumPattern.BULLISH_ENGULFING
    assert bull_eng.trend_aligned is True
    assert bull_eng.entry_price is not None
    assert bull_eng.entry_price == pytest.approx(1.258)
    assert bull_eng.stop_loss is not None
    assert bull_eng.stop_loss < 1.230  # strictly below min(prev.low, curr.low)
    assert bull_eng.tp2_2r is not None
    risk_be = bull_eng.entry_price - bull_eng.stop_loss
    assert bull_eng.tp2_2r == pytest.approx(bull_eng.entry_price + 2.0 * risk_be)

    # 2. Bullish Marubozu in uptrend -> READY LONG, SL below curr.low
    bull_maru_candles = _uptrend_history(
        last_two=(
            (1.235, 1.242, 1.233, 1.240),
            (1.240, 1.272, 1.239, 1.270),  # tiny wicks, huge body
        )
    )
    bull_maru = evaluate_15m_momentum_setup(
        bull_maru_candles,
        side="long",
        change_24h_pct=14.0,
        quote_volume_24h_usd=12_000_000.0,
    )
    assert bull_maru.status is SetupStatus.READY
    assert bull_maru.pattern is MomentumPattern.BULLISH_MARUBOZU
    assert bull_maru.stop_loss is not None
    assert bull_maru.stop_loss < 1.239

    # 3. Bearish Engulfing in downtrend -> READY SHORT, SL above max(prev.high, curr.high)
    bear_eng_candles = _downtrend_history(
        last_two=(
            (1.545, 1.565, 1.540, 1.558),  # green bounce bar with upper wick 1.565
            (1.560, 1.562, 1.522, 1.525),  # strong red engulfing closing below 1.545
        )
    )
    bear_eng = evaluate_15m_momentum_setup(
        bear_eng_candles,
        side="short",
        change_24h_pct=-19.2,
        quote_volume_24h_usd=18_000_000.0,
    )
    assert bear_eng.status is SetupStatus.READY
    assert bear_eng.pattern is MomentumPattern.BEARISH_ENGULFING
    assert bear_eng.entry_price is not None
    assert bear_eng.stop_loss is not None
    assert bear_eng.stop_loss > 1.565
    assert bear_eng.tp2_2r is not None
    risk_se = bear_eng.stop_loss - bear_eng.entry_price
    assert bear_eng.tp2_2r == pytest.approx(bear_eng.entry_price - 2.0 * risk_se)

    # 4. Bearish Marubozu in downtrend -> READY SHORT, SL above curr.high
    bear_maru_candles = _downtrend_history(
        last_two=(
            (1.560, 1.563, 1.545, 1.548),
            (1.548, 1.550, 1.502, 1.505),  # strong bearish marubozu
        )
    )
    bear_maru = evaluate_15m_momentum_setup(
        bear_maru_candles,
        side="short",
        change_24h_pct=-11.5,
        quote_volume_24h_usd=8_000_000.0,
    )
    assert bear_maru.status is SetupStatus.READY
    assert bear_maru.pattern is MomentumPattern.BEARISH_MARUBOZU
    assert bear_maru.stop_loss is not None
    assert bear_maru.stop_loss > 1.550

    # 5. Counter-trend Bullish Engulfing (during downtrend) -> WAIT
    counter_candles = _downtrend_history(
        last_two=(
            (1.560, 1.562, 1.545, 1.548),
            (1.547, 1.572, 1.546, 1.569),
        )
    )
    counter = evaluate_15m_momentum_setup(
        counter_candles,
        side="long",
        change_24h_pct=8.0,
        quote_volume_24h_usd=9_000_000.0,
    )
    assert counter.status is SetupStatus.WAIT
    assert counter.pattern is MomentumPattern.BULLISH_ENGULFING
    assert counter.trend_aligned is False

    # 6. Weak/Doji candle -> WAIT
    doji_candles = _uptrend_history(
        last_two=(
            (1.240, 1.245, 1.238, 1.243),
            (1.243, 1.250, 1.236, 1.244),
        )
    )
    doji = evaluate_15m_momentum_setup(
        doji_candles,
        side="long",
        change_24h_pct=10.0,
        quote_volume_24h_usd=9_000_000.0,
    )
    assert doji.status is SetupStatus.WAIT
    assert doji.pattern is MomentumPattern.NONE


class _StubScreenerService:
    def __init__(self, ticker_batch: FuturesTickerBatch, ohlcv_map: dict[str, OhlcvBatch]) -> None:
        self._ticker_batch = ticker_batch
        self._ohlcv_map = ohlcv_map

    async def get_futures_tickers(self) -> FuturesTickerBatch:
        return self._ticker_batch

    async def get_closed_ohlcv(
        self,
        symbol: str,
        timeframe: Timeframe,
        *,
        limit: int = 100,
    ) -> OhlcvBatch:
        del timeframe, limit
        return self._ohlcv_map[symbol]


@pytest.mark.asyncio
async def test_screen_momentum_movers_filters_volume_and_surfaces_gainers_losers() -> None:
    tickers = (
        # Illiquid gainer (< $5M) must be excluded!
        FuturesTickerSnapshot(
            venue="gate",
            canonical_symbol="ILLIQ/USDT:USDT",
            provider_symbol="GATE:ILLIQUSDT.P",
            last_price=1.0,
            change_24h_pct=45.0,
            quote_volume_24h_usd=1_200_000.0,
            observed_time=NOW,
        ),
        FuturesTickerSnapshot(
            venue="gate",
            canonical_symbol="SUI/USDT:USDT",
            provider_symbol="GATE:SUIUSDT.P",
            last_price=1.258,
            change_24h_pct=18.5,
            quote_volume_24h_usd=42_000_000.0,
            observed_time=NOW,
        ),
        FuturesTickerSnapshot(
            venue="gate",
            canonical_symbol="RED/USDT:USDT",
            provider_symbol="GATE:REDUSDT.P",
            last_price=1.525,
            change_24h_pct=-19.2,
            quote_volume_24h_usd=15_500_000.0,
            observed_time=NOW,
        ),
    )
    ticker_batch = FuturesTickerBatch(venue="gate", received_time=NOW, tickers=tickers)

    sui_candles = _uptrend_history(
        symbol="SUI/USDT:USDT",
        last_two=(
            (1.245, 1.248, 1.230, 1.236),
            (1.235, 1.262, 1.233, 1.258),
        ),
    )
    red_candles = _downtrend_history(
        symbol="RED/USDT:USDT",
        last_two=(
            (1.545, 1.565, 1.540, 1.558),
            (1.560, 1.562, 1.522, 1.525),
        ),
    )

    ohlcv_map = {
        "SUI/USDT:USDT": OhlcvBatch(
            metadata=ProviderMetadata(
                provider="ccxt",
                venue="gate",
                market_type=MarketType.PERPETUAL,
                base_unit="SUI",
                quote_currency="USDT",
            ),
            canonical_symbol="SUI/USDT:USDT",
            provider_symbol="GATE:SUIUSDT.P",
            timeframe=Timeframe.M15,
            received_time=NOW,
            candles=tuple(sui_candles),
        ),
        "RED/USDT:USDT": OhlcvBatch(
            metadata=ProviderMetadata(
                provider="ccxt",
                venue="gate",
                market_type=MarketType.PERPETUAL,
                base_unit="RED",
                quote_currency="USDT",
            ),
            canonical_symbol="RED/USDT:USDT",
            provider_symbol="GATE:REDUSDT.P",
            timeframe=Timeframe.M15,
            received_time=NOW,
            candles=tuple(red_candles),
        ),
    }

    report = await screen_momentum_movers(_StubScreenerService(ticker_batch, ohlcv_map))  # type: ignore[arg-type]
    assert report.total_pairs_seen == 3
    assert report.liquid_pairs_count == 2
    assert [t.canonical_symbol for t in report.top_gainers] == ["SUI/USDT:USDT"]
    assert [t.canonical_symbol for t in report.top_losers] == ["RED/USDT:USDT"]
    assert len(report.long_signals) == 1
    assert report.long_signals[0].symbol == "SUI/USDT:USDT"
    assert len(report.short_signals) == 1
    assert report.short_signals[0].symbol == "RED/USDT:USDT"

    rendered = render_momentum_screener_report(report, html_mode=True)
    assert "Top Gainers" in rendered
    assert "Top Losers" in rendered
    assert "Bullish Engulfing (15m)" in rendered
    assert "Bearish Engulfing (15m)" in rendered
    assert "1% per trade" in rendered
    assert "20-30x trading" in rendered


def test_nlp_routes_screening_queries_to_screen_intent() -> None:
    assert resolve_followup_intent("screening top gainers dong", recent_messages=()) == (
        ScreenIntent(side="long")
    )
    assert resolve_followup_intent("cari top losers buat short", recent_messages=()) == (
        ScreenIntent(side="short")
    )
    assert resolve_followup_intent(
        "coba screening market yang momentum 15m engulfing", recent_messages=()
    ) == ScreenIntent(side="all")


class _FailingTickerStub:
    async def get_closed_ohlcv(
        self, symbol: str, timeframe: Timeframe, limit: int = 100
    ) -> OhlcvBatch:
        raise NotImplementedError

    async def get_futures_tickers(self) -> FuturesTickerBatch:
        return FuturesTickerBatch(
            venue="gate",
            received_time=NOW,
            quality_flags=(QualityFlag.PROVIDER_ERROR,),
            error_message="gate rate limited",
        )

    async def close(self) -> None:
        return None


class _HealthyTickerStub:
    async def get_closed_ohlcv(
        self, symbol: str, timeframe: Timeframe, limit: int = 100
    ) -> OhlcvBatch:
        raise NotImplementedError

    async def get_futures_tickers(self) -> FuturesTickerBatch:
        return FuturesTickerBatch(
            venue="bingx",
            received_time=NOW,
            tickers=(
                FuturesTickerSnapshot(
                    venue="bingx",
                    canonical_symbol="BTC/USDT:USDT",
                    provider_symbol="BINGX:BTCUSDT.P",
                    last_price=92_000.0,
                    change_24h_pct=4.5,
                    quote_volume_24h_usd=500_000_000.0,
                    observed_time=NOW,
                ),
            ),
        )

    async def close(self) -> None:
        return None


@pytest.mark.asyncio
async def test_fallback_provider_falls_back_for_futures_tickers() -> None:
    fb = FallbackMarketDataProvider([_FailingTickerStub(), _HealthyTickerStub()])
    batch = await fb.get_futures_tickers()
    assert not batch.has_provider_error
    assert batch.venue == "bingx"
    assert len(batch.tickers) == 1
