"""Unit tests for composite sentiment panel wiring, symbol normalization, and reason codes."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from crypto_chatbot.analysis.service import AnalysisService
from crypto_chatbot.bot.templates import render_compact_snapshot, render_deterministic_snapshot
from crypto_chatbot.market_data.base import OhlcvBatch
from crypto_chatbot.schemas.derivatives import ContractStatsObservation
from crypto_chatbot.schemas.market import (
    MarketCandle,
    MarketType,
    ProviderMetadata,
    Timeframe,
)
from crypto_chatbot.schemas.news import (
    FearGreedObservation,
    SentimentLabel,
    SentimentObservation,
)

AS_OF = datetime(2026, 11, 28, 14, 30, tzinfo=UTC)


def _make_15m_batch(end_close: datetime = AS_OF, count: int = 150) -> OhlcvBatch:
    dur = Timeframe.M15.duration
    candles: list[MarketCandle] = []
    for i in range(count):
        close_t = end_close - dur * (count - 1 - i)
        open_t = close_t - dur
        p = 90_000.0 + i * 10.0
        candles.append(
            MarketCandle(
                provider="gate_direct",
                venue="gate",
                canonical_symbol="BTC/USDT:USDT",
                provider_symbol="BTC_USDT",
                market_type=MarketType.PERPETUAL,
                timeframe=Timeframe.M15,
                open_time=open_t,
                close_time=close_t,
                received_time=close_t,
                available_time=close_t,
                open=p,
                high=p + 50.0,
                low=p - 50.0,
                close=p + 5.0,
                base_volume=100.0,
                quote_volume=100.0 * p,
                base_unit="BTC",
                quote_currency="USDT",
                is_closed=True,
            )
        )
    return OhlcvBatch(
        metadata=ProviderMetadata(
            provider="gate_direct",
            venue="gate",
            market_type=MarketType.PERPETUAL,
            base_unit="BTC",
            quote_currency="USDT",
        ),
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol="BTC_USDT",
        timeframe=Timeframe.M15,
        received_time=end_close,
        candles=tuple(candles),
    )


class _StaticProvider:
    venue = "gate"

    def __init__(self, batch: OhlcvBatch) -> None:
        self._batch = batch

    async def get_closed_ohlcv(
        self, symbol: str, timeframe: Timeframe, limit: int = 1000
    ) -> OhlcvBatch:
        del symbol, timeframe, limit
        return self._batch


class _FakeContractStatsRepo:
    def __init__(self, rows: Sequence[ContractStatsObservation], *, fail: bool = False) -> None:
        self._rows = list(rows)
        self._fail = fail

    def save_contract_stats(self, observations: Sequence[ContractStatsObservation]) -> int:
        del observations
        return 0

    def load_contract_stats(
        self,
        *,
        venue: str,
        canonical_symbol: str,
        interval: str = "5m",
        start_time: datetime | None = None,
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[ContractStatsObservation, ...]:
        del interval
        if self._fail:
            raise RuntimeError("db down")
        out = [
            r
            for r in self._rows
            if r.venue == venue
            and r.canonical_symbol == canonical_symbol
            and (start_time is None or r.observed_time >= start_time)
            and (end_available_time is None or r.available_time <= end_available_time)
        ]
        if limit is not None:
            out = out[-limit:]
        return tuple(out)

    def list_contract_stats_times(self, **kwargs: Any) -> tuple[datetime, ...]:
        del kwargs
        return ()


class _FakeFearGreedRepo:
    def __init__(self, rows: Sequence[FearGreedObservation], *, fail: bool = False) -> None:
        self._rows = list(rows)
        self._fail = fail

    def save_observations(self, observations: Sequence[FearGreedObservation]) -> int:
        del observations
        return 0

    def load_observations(
        self,
        *,
        source: str = "alternative.me",
        end_available_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[FearGreedObservation, ...]:
        if self._fail:
            raise RuntimeError("fg db down")
        out = [
            r
            for r in self._rows
            if r.source == source
            and (end_available_time is None or r.available_time <= end_available_time)
        ]
        if limit is not None:
            out = out[-limit:]
        return tuple(out)


class _FakeNewsRepo:
    def __init__(self, rows: Sequence[SentimentObservation], *, fail: bool = False) -> None:
        self._rows = list(rows)
        self._fail = fail

    def save_articles(self, articles: Sequence[Any]) -> int:
        del articles
        return 0

    def save_observations(self, observations: Sequence[SentimentObservation]) -> int:
        del observations
        return 0

    def load_observations(
        self,
        *,
        canonical_symbol: str,
        start_available_time: datetime | None = None,
        end_available_time: datetime | None = None,
        min_entity_confidence: float = 0.7,
        limit: int | None = None,
    ) -> tuple[SentimentObservation, ...]:
        if self._fail:
            raise RuntimeError("news db down")
        out = [
            r
            for r in self._rows
            if r.canonical_symbol == canonical_symbol
            and r.entity_confidence >= min_entity_confidence
            and (start_available_time is None or r.available_time >= start_available_time)
            and (end_available_time is None or r.available_time <= end_available_time)
        ]
        if limit is not None:
            out = out[-limit:]
        return tuple(out)


def _make_contract_stat(ts: datetime) -> ContractStatsObservation:
    return ContractStatsObservation(
        venue="gateio",
        canonical_symbol="BTC/USDT:USDT",
        provider_symbol="BTC_USDT",
        interval="5m",
        observed_time=ts - timedelta(minutes=5),
        received_time=ts,
        available_time=ts,
        lsr_taker=1.25,
        lsr_account=1.18,
        top_lsr_account=1.42,
        top_lsr_size=1.10,
        open_interest=10_000.0,
        open_interest_usd=900_000_000.0,
        mark_price=90_000.0,
    )


def _make_fg(ts: datetime) -> FearGreedObservation:
    return FearGreedObservation(
        source="alternative.me",
        observed_time=ts,
        received_time=ts,
        available_time=ts,
        value=28,
        classification="Fear",
        normalized_score=-0.44,
        is_backfilled=True,
    )


def _make_news(ts: datetime) -> SentimentObservation:
    # Stored with spot symbol "BTC/USDT" (just like `sync-news`) while analyze queries BTC/USDT:USDT
    return SentimentObservation(
        observation_id=f"obs-test-{int(ts.timestamp())}",
        article_id=f"art-test-{int(ts.timestamp())}",
        canonical_url="https://example.com/btc",
        publisher="CoinDesk",
        title="Bitcoin rallies on ETF inflows",
        canonical_symbol="BTC/USDT",
        base_asset="BTC",
        entity_confidence=0.95,
        sentiment_label=SentimentLabel.POSITIVE,
        sentiment_score=0.65,
        sentiment_confidence=0.85,
        model_version="crypto-lexicon-v1.0.0",
        is_probabilistic=False,
        published_time=ts - timedelta(minutes=5),
        available_time=ts,
        scored_at=ts,
    )


@pytest.mark.asyncio
async def test_sentiment_panels_populate_when_available_before_as_of() -> None:
    """With rows <= as_of (and news stored as BTC/USDT), all 3 panels populate with real values."""
    # 11 bars of 5m contract_stats -> WARMING_UP with insufficient_history (11 bars) + live L/S
    stats_rows = [_make_contract_stat(AS_OF - timedelta(minutes=5 * i)) for i in range(11)]
    fg_rows = [_make_fg(datetime(2026, 11, 28, 0, 0, tzinfo=UTC))]
    news_rows = [_make_news(AS_OF - timedelta(minutes=14))]

    svc = AnalysisService(
        provider=_StaticProvider(_make_15m_batch()),  # type: ignore[arg-type]
        contract_stats_repository=_FakeContractStatsRepo(stats_rows),
        fear_greed_repository=_FakeFearGreedRepo(fg_rows),
        news_repository=_FakeNewsRepo(news_rows),
    )
    snap = await svc.analyze(
        "BTC", Timeframe.M15, include_mtf=False, include_derivatives=False, persist=False
    )

    assert snap.positioning is not None
    assert snap.positioning.status == "WARMING_UP"
    assert snap.positioning.reason_code == "insufficient_history"
    assert snap.positioning.observation_count == 11
    assert snap.positioning.top_lsr_account == pytest.approx(1.42)

    assert snap.fear_greed is not None
    assert snap.fear_greed.value == 28
    assert snap.fear_greed.classification == "Fear"

    assert snap.sentiment is not None
    assert snap.sentiment.article_count == 1
    assert snap.sentiment.dominant_label is SentimentLabel.POSITIVE

    assert snap.sentiment_diagnostics is not None
    assert snap.sentiment_diagnostics.positioning.reason == "insufficient_history"
    assert snap.sentiment_diagnostics.fear_greed.available is True
    assert snap.sentiment_diagnostics.news.available is True

    rendered = render_deterministic_snapshot(snap, html_mode=False)
    assert "insufficient_history (11 bars)" in rendered
    assert "Top L/S=1,42" in rendered
    assert "Fear & Greed [CONTEXT_ONLY]: 28/100 (Fear)" in rendered
    assert "Berita 24h [FORWARD_ONLY]: Positif" in rendered


@pytest.mark.asyncio
async def test_sentiment_panels_report_not_yet_available_when_rows_are_after_as_of() -> None:
    """When rows exist only AFTER as_of, every panel returns UNAVAILABLE (not_yet_available)."""
    future_t = AS_OF + timedelta(minutes=10)
    svc = AnalysisService(
        provider=_StaticProvider(_make_15m_batch()),  # type: ignore[arg-type]
        contract_stats_repository=_FakeContractStatsRepo([_make_contract_stat(future_t)]),
        fear_greed_repository=_FakeFearGreedRepo([_make_fg(future_t)]),
        news_repository=_FakeNewsRepo([_make_news(future_t)]),
    )
    snap = await svc.analyze(
        "BTC", Timeframe.M15, include_mtf=False, include_derivatives=False, persist=False
    )

    assert snap.fear_greed is None
    assert snap.sentiment is None
    assert snap.positioning is not None
    assert snap.positioning.status == "UNAVAILABLE"
    assert snap.positioning.reason_code == "not_yet_available"

    assert snap.sentiment_diagnostics is not None
    assert snap.sentiment_diagnostics.positioning.reason == "not_yet_available"
    assert snap.sentiment_diagnostics.fear_greed.reason == "not_yet_available"
    assert snap.sentiment_diagnostics.news.reason == "not_yet_available"

    rendered = render_compact_snapshot(snap, html_mode=False)
    assert (
        "Positioning Derivatif [FORWARD_ONLY]: UNAVAILABLE (reason=not_yet_available)" in rendered
    )
    assert "Fear & Greed [CONTEXT_ONLY]: UNAVAILABLE (reason=not_yet_available)" in rendered
    assert "Berita 24h [FORWARD_ONLY]: UNAVAILABLE (reason=not_yet_available)" in rendered


@pytest.mark.asyncio
async def test_sentiment_panels_report_no_data_not_wired_and_error_reasons() -> None:
    """Distinct reason codes: not_wired, no_data, and error are all surfaced cleanly."""
    # 1. not_wired
    svc_unwired = AnalysisService(provider=_StaticProvider(_make_15m_batch()))  # type: ignore[arg-type]
    snap_unwired = await svc_unwired.analyze(
        "BTC", Timeframe.M15, include_mtf=False, include_derivatives=False, persist=False
    )
    assert snap_unwired.sentiment_diagnostics is not None
    assert snap_unwired.sentiment_diagnostics.positioning.reason == "not_wired"
    assert snap_unwired.sentiment_diagnostics.fear_greed.reason == "not_wired"
    assert snap_unwired.sentiment_diagnostics.news.reason == "not_wired"

    # 2. no_data
    svc_empty = AnalysisService(
        provider=_StaticProvider(_make_15m_batch()),  # type: ignore[arg-type]
        contract_stats_repository=_FakeContractStatsRepo([]),
        fear_greed_repository=_FakeFearGreedRepo([]),
        news_repository=_FakeNewsRepo([]),
    )
    snap_empty = await svc_empty.analyze(
        "BTC", Timeframe.M15, include_mtf=False, include_derivatives=False, persist=False
    )
    assert snap_empty.sentiment_diagnostics is not None
    assert snap_empty.sentiment_diagnostics.positioning.reason == "no_data"
    assert snap_empty.sentiment_diagnostics.fear_greed.reason == "no_data"
    assert snap_empty.sentiment_diagnostics.news.reason == "no_data"

    # 3. error (graceful degradation, analyze still succeeds!)
    svc_err = AnalysisService(
        provider=_StaticProvider(_make_15m_batch()),  # type: ignore[arg-type]
        contract_stats_repository=_FakeContractStatsRepo([], fail=True),
        fear_greed_repository=_FakeFearGreedRepo([], fail=True),
        news_repository=_FakeNewsRepo([], fail=True),
    )
    snap_err = await svc_err.analyze(
        "BTC", Timeframe.M15, include_mtf=False, include_derivatives=False, persist=False
    )
    assert snap_err.sentiment_diagnostics is not None
    assert snap_err.sentiment_diagnostics.positioning.reason == "error"
    assert snap_err.sentiment_diagnostics.fear_greed.reason == "error"
    assert snap_err.sentiment_diagnostics.news.reason == "error"
