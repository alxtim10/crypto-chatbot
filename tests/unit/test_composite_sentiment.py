from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from crypto_chatbot.analysis.service import AnalysisService
from crypto_chatbot.bot.templates import render_compact_snapshot, render_deterministic_snapshot
from crypto_chatbot.market_data.base import DEFAULT_CANDLE_LIMIT, OhlcvBatch
from crypto_chatbot.market_data.symbols import resolve_default_perp_symbol
from crypto_chatbot.schemas.analysis import PositioningPanel
from crypto_chatbot.schemas.derivatives import FundingRateBatch, FundingRateObservation
from crypto_chatbot.schemas.market import MarketCandle, MarketType, ProviderMetadata, Timeframe
from crypto_chatbot.schemas.news import (
    FearGreedObservation,
    SentimentLabel,
    SentimentObservation,
)


def _sample_candles(count: int = 250) -> tuple[MarketCandle, ...]:
    sym = resolve_default_perp_symbol("BTC")
    t0 = datetime(2026, 11, 1, 0, 0, tzinfo=UTC)
    rx = t0 + timedelta(hours=count)
    return tuple(
        MarketCandle(
            provider="gate_rest",
            venue="gate",
            canonical_symbol=sym.canonical_symbol,
            provider_symbol="GATE:BTCUSDT.P",
            market_type=MarketType.PERPETUAL,
            timeframe=Timeframe.H1,
            open_time=t0 + timedelta(hours=i),
            close_time=t0 + timedelta(hours=i + 1),
            received_time=rx,
            available_time=t0 + timedelta(hours=i + 1),
            open=60000.0 + i * 5.0,
            high=60020.0 + i * 5.0,
            low=59990.0 + i * 5.0,
            close=60005.0 + i * 5.0,
            base_volume=10.0,
            quote_volume=600050.0,
            taker_buy_base_volume=5.5,
            base_unit="BTC",
            quote_currency="USDT",
            is_closed=True,
        )
        for i in range(count)
    )


def test_composite_sentiment_panel_renders_all_parity_tags_and_attributions() -> None:
    candles = _sample_candles()
    as_of = candles[-1].close_time
    batch = OhlcvBatch(
        metadata=ProviderMetadata(
            provider="gate_rest",
            venue="gate",
            market_type=MarketType.PERPETUAL,
            base_unit="BTC",
            quote_currency="USDT",
        ),
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol="GATE:BTCUSDT.P",
        timeframe=Timeframe.H1,
        received_time=candles[-1].received_time,
        candles=candles,
    )
    fg = FearGreedObservation(
        source="alternative.me",
        observed_time=as_of - timedelta(hours=6),
        received_time=as_of - timedelta(hours=5),
        available_time=as_of - timedelta(hours=5),
        value=24,
        classification="Extreme Fear",
        normalized_score=-0.52,
    )
    news_obs = (
        SentimentObservation(
            observation_id="obs-" + "a" * 20,
            article_id="news-" + "b" * 20,
            canonical_url="https://coindesk.com/btc-etf",
            publisher="CoinDesk",
            title="Bitcoin ETF inflows hit weekly high",
            canonical_symbol="BTC/USDT:USDT",
            base_asset="BTC",
            entity_confidence=0.95,
            sentiment_label=SentimentLabel.POSITIVE,
            sentiment_score=0.7,
            sentiment_confidence=0.9,
            published_time=as_of - timedelta(hours=2),
            available_time=as_of - timedelta(hours=1),
        ),
    )
    from crypto_chatbot.analysis.service import build_analysis_snapshot
    from crypto_chatbot.schemas.derivatives import DerivativesContext

    deriv_ctx = DerivativesContext(
        funding=FundingRateBatch(
            venue="gate",
            canonical_symbol="BTC/USDT:USDT",
            provider_symbol="GATE:BTCUSDT.P",
            received_time=as_of,
            observation=FundingRateObservation(
                provider="gate_rest",
                venue="gate",
                canonical_symbol="BTC/USDT:USDT",
                provider_symbol="GATE:BTCUSDT.P",
                observed_time=as_of,
                next_funding_time=as_of + timedelta(hours=8),
                received_time=as_of,
                available_time=as_of,
                funding_rate=0.0001,
                funding_interval_hours=8.0,
                rate_per_8h=0.0001,
                annualized_rate=0.0001 * (8760.0 / 8.0),
            ),
        )
    )
    pos = PositioningPanel(
        status="READY",
        parity_mode="FORWARD_ONLY",
        canonical_symbol="BTC/USDT:USDT",
        venue="gate",
        as_of=as_of,
        top_lsr_account=1.85,
        lsr_taker=1.25,
        oi_24h_delta_pct=3.4,
    )
    snap = build_analysis_snapshot(
        batch,
        derivatives_context=deriv_ctx,
        news_observations=news_obs,
        positioning=pos,
        fear_greed=fg,
    )

    for text in (
        render_deterministic_snapshot(snap, html_mode=False),
        render_compact_snapshot(snap, html_mode=False),
    ):
        assert "Panel Sentimen" in text
        assert "[BACKTESTED]" in text
        assert "[FORWARD_ONLY]" in text
        assert "[CONTEXT_ONLY]" in text
        assert "Alternative.me" in text
        assert "Extreme Fear" in text
        assert "CoinDesk" in text


@pytest.mark.asyncio
async def test_analysis_service_loads_causal_fear_greed_without_future_leak() -> None:
    candles = _sample_candles()
    as_of = candles[-1].close_time
    past_fg = FearGreedObservation(
        observed_time=as_of - timedelta(days=1),
        received_time=as_of - timedelta(days=1),
        available_time=as_of - timedelta(days=1),
        value=32,
        classification="Fear",
        normalized_score=-0.36,
    )
    future_fg = FearGreedObservation(
        observed_time=as_of + timedelta(hours=2),
        received_time=as_of + timedelta(hours=2),
        available_time=as_of + timedelta(hours=2),
        value=88,
        classification="Extreme Greed",
        normalized_score=0.76,
    )

    class _FgRepo:
        def save_observations(self, observations: object) -> int:
            del observations
            return 0

        def load_observations(
            self,
            *,
            source: str = "alternative.me",
            end_available_time: datetime | None = None,
            limit: int | None = None,
        ) -> tuple[FearGreedObservation, ...]:
            del source, limit
            rows = [past_fg, future_fg]
            if end_available_time is not None:
                rows = [r for r in rows if r.available_time <= end_available_time]
            return tuple(rows)

    class _Provider:
        async def get_closed_ohlcv(
            self, sym: str, tf: Timeframe, limit: int = DEFAULT_CANDLE_LIMIT
        ) -> OhlcvBatch:
            del sym, tf, limit
            return OhlcvBatch(
                metadata=ProviderMetadata(
                    provider="gate_rest",
                    venue="gate",
                    market_type=MarketType.PERPETUAL,
                    base_unit="BTC",
                    quote_currency="USDT",
                ),
                canonical_symbol="BTC/USDT:USDT",
                provider_symbol="GATE:BTCUSDT.P",
                timeframe=Timeframe.H1,
                received_time=candles[-1].received_time,
                candles=candles,
            )

        async def close(self) -> None:
            return None

    svc = AnalysisService(provider=_Provider(), fear_greed_repository=_FgRepo())
    snap = await svc.analyze(
        "BTC",
        Timeframe.H1,
        limit=250,
        include_mtf=False,
        include_derivatives=False,
        persist=False,
    )
    assert snap.fear_greed is not None
    assert snap.fear_greed.value == 32
    assert snap.fear_greed.classification == "Fear"
