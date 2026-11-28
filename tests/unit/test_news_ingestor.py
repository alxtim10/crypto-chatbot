from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from crypto_chatbot.news.ingestor import (
    VERIFIED_PUBLIC_RSS_FEEDS,
    PublicRssNewsClient,
    build_article_vote_snapshots,
    build_news_attribution,
    sync_public_rss_news,
)
from crypto_chatbot.news.pipeline import deduplicate_articles
from crypto_chatbot.schemas.news import (
    MIN_ENTITY_MATCH_CONFIDENCE,
    NewsVoteSnapshot,
    NormalizedNewsArticle,
    SentimentObservation,
)


class _FakeNewsRepo:
    def __init__(self) -> None:
        self.articles: dict[str, NormalizedNewsArticle] = {}
        self.observations: list[SentimentObservation] = []
        self.snapshots: list[NewsVoteSnapshot] = []

    def save_articles(self, articles: Sequence[NormalizedNewsArticle]) -> int:
        cnt = 0
        for a in articles:
            if a.article_id not in self.articles:
                self.articles[a.article_id] = a
                cnt += 1
        return cnt

    def save_observations(self, observations: Sequence[SentimentObservation]) -> int:
        items = list(observations)
        self.observations.extend(items)
        return len(items)

    def load_observations(
        self,
        *,
        canonical_symbol: str,
        start_available_time: datetime | None = None,
        end_available_time: datetime | None = None,
        min_entity_confidence: float = MIN_ENTITY_MATCH_CONFIDENCE,
        limit: int | None = None,
    ) -> tuple[SentimentObservation, ...]:
        del canonical_symbol, start_available_time, end_available_time, min_entity_confidence, limit
        return ()

    def save_vote_snapshots(self, snapshots: Sequence[NewsVoteSnapshot]) -> int:
        # Strictly append-only: never overwrite existing snapshots!
        existing_ids = {s.snapshot_id for s in self.snapshots}
        added = 0
        for s in snapshots:
            if s.snapshot_id not in existing_ids:
                self.snapshots.append(s)
                existing_ids.add(s.snapshot_id)
                added += 1
        return added

    def load_vote_snapshots(
        self,
        *,
        article_id: str,
        end_snapshot_time: datetime | None = None,
        limit: int | None = None,
    ) -> tuple[NewsVoteSnapshot, ...]:
        del limit
        rows = [s for s in self.snapshots if s.article_id == article_id]
        if end_snapshot_time is not None:
            rows = [s for s in rows if s.snapshot_time <= end_snapshot_time]
        return tuple(sorted(rows, key=lambda s: s.snapshot_time))


_SAMPLE_RSS = """<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0">
  <channel>
    <title>CoinDesk</title>
    <item>
      <title>Bitcoin surges past resistance as ETF inflows accelerate</title>
      <link>https://www.coindesk.com/markets/btc-surges/?utm_source=x</link>
      <description>BTC rallies strongly.</description>
      <pubDate>Fri, 27 Nov 2026 12:00:00 +0000</pubDate>
    </item>
    <item>
      <title>Ethereum devs finalize upgrade milestone cleanly</title>
      <link>https://www.coindesk.com/tech/eth-upgrade/</link>
      <description>ETH staking rises.</description>
      <pubDate>Fri, 27 Nov 2026 14:00:00 +0000</pubDate>
    </item>
  </channel>
</rss>
"""


@pytest.mark.asyncio
async def test_public_rss_ingestor_sets_first_seen_available_and_append_only_snapshots() -> None:
    fetch_t1 = datetime(2026, 11, 27, 16, 0, tzinfo=UTC)
    fetch_t2 = fetch_t1 + timedelta(minutes=30)
    current_t = fetch_t1

    def _handler(request: httpx.Request) -> httpx.Response:
        if str(request.url).startswith("https://www.coindesk.com"):
            return httpx.Response(200, text=_SAMPLE_RSS)
        return httpx.Response(404, text="not found")

    client = PublicRssNewsClient(
        feeds=(("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),),
        min_request_interval_seconds=0.0,
        transport=httpx.MockTransport(_handler),
        clock=lambda: current_t,
    )
    repo = _FakeNewsRepo()
    try:
        summary1, arts1, _obs1 = await sync_public_rss_news(
            client=client,
            repository=repo,
            clock=lambda: current_t,
        )
        assert summary1.feeds_succeeded == 1
        assert summary1.dedup_articles == 2
        assert summary1.inserted_snapshots == 2
        # available_time must equal first_seen_time (fetch_t1), NOT older pubDate!
        assert all(a.first_seen_time == fetch_t1 for a in arts1)
        assert all(a.available_time == fetch_t1 for a in arts1)

        # Second poll 30 minutes later appends a NEW vote snapshot without overwriting!
        current_t = fetch_t2
        summary2, arts2, _ = await sync_public_rss_news(
            client=client,
            repository=repo,
            clock=lambda: current_t,
        )
        assert summary2.inserted_articles == 0
        assert summary2.inserted_snapshots == 2
        assert len(repo.snapshots) == 4

        art_id = arts2[0].article_id
        causal_at_t1 = repo.load_vote_snapshots(article_id=art_id, end_snapshot_time=fetch_t1)
        causal_at_t2 = repo.load_vote_snapshots(article_id=art_id, end_snapshot_time=fetch_t2)
        assert [s.snapshot_time for s in causal_at_t1] == [fetch_t1]
        assert [s.snapshot_time for s in causal_at_t2] == [fetch_t1, fetch_t2]
    finally:
        await client.close()


def test_attribution_and_verified_feed_list() -> None:
    assert len(VERIFIED_PUBLIC_RSS_FEEDS) == 14
    assert build_news_attribution("cryptopanic", publisher="CoinDesk") == (
        "Source: CryptoPanic (CoinDesk)"
    )
    assert build_news_attribution("rss", publisher="Decrypt") == "Source: Decrypt (rss)"

    # Verify build_article_vote_snapshots attribution text
    now = datetime(2026, 11, 27, 18, 0, tzinfo=UTC)
    from crypto_chatbot.news.pipeline import parse_rss_xml

    raw = parse_rss_xml(_SAMPLE_RSS, publisher="CoinDesk", first_seen_time=now)
    arts, _ = deduplicate_articles(raw)
    snaps = build_article_vote_snapshots(arts, snapshot_time=now)
    assert len(snaps) == 2
    assert snaps[0].attribution_text == "Source: CoinDesk (rss)"
