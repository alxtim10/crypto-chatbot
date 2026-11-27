"""Public RSS news feed ingestor & append-only vote snapshot helpers (S-03)."""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx
from tenacity import AsyncRetrying, retry_if_exception, stop_after_attempt, wait_exponential

from crypto_chatbot.news.pipeline import (
    build_sentiment_observations,
    deduplicate_articles,
    parse_rss_xml,
)
from crypto_chatbot.schemas.news import (
    NewsVoteSnapshot,
    NormalizedNewsArticle,
    RawNewsItem,
    SentimentObservation,
)
from crypto_chatbot.storage.repositories import NewsRepository, NewsVoteSnapshotRepository

logger = logging.getLogger(__name__)

# Live-verified HTTP 200 public crypto RSS 2.0 feeds (docs/research/gate-data-audit.md §6).
VERIFIED_PUBLIC_RSS_FEEDS: tuple[tuple[str, str], ...] = (
    ("CoinDesk", "https://www.coindesk.com/arc/outboundfeeds/rss/"),
    ("CoinTelegraph", "https://cointelegraph.com/rss"),
    ("Decrypt", "https://decrypt.co/feed"),
    ("The Block", "https://www.theblock.co/rss.xml"),
    ("U.Today", "https://u.today/rss"),
    ("BeInCrypto", "https://beincrypto.com/feed/"),
    ("NewsBTC", "https://www.newsbtc.com/feed/"),
    ("AMBCrypto", "https://ambcrypto.com/feed/"),
    ("CryptoPotato", "https://cryptopotato.com/feed/"),
    ("Bitcoinist", "https://bitcoinist.com/feed/"),
    ("The Daily Hodl", "https://dailyhodl.com/feed/"),
    ("ZyCrypto", "https://zycrypto.com/feed/"),
    ("CoinJournal", "https://coinjournal.net/news/feed/"),
    ("CryptoNews", "https://cryptonews.com/news/feed/"),
)


class _RetryableNewsHttpError(RuntimeError):
    """Transient HTTP 429/5xx error when polling news feeds."""


def build_news_attribution(source_provider: str, *, publisher: str) -> str:
    """Format mandatory provider attribution string (e.g. CryptoPanic or RSS publisher)."""
    provider_clean = source_provider.strip().lower()
    if provider_clean == "cryptopanic":
        return f"Source: CryptoPanic ({publisher.strip()})"
    return f"Source: {publisher.strip()} ({source_provider.strip()})"


@dataclass(frozen=True, slots=True)
class NewsSyncResult:
    """Summary returned after syncing public RSS feeds into PostgreSQL."""

    feeds_attempted: int
    feeds_succeeded: int
    raw_items: int
    dedup_articles: int
    inserted_articles: int
    inserted_observations: int
    inserted_snapshots: int


class PublicRssNewsClient:
    """Rate-limited async RSS fetcher over live-verified public crypto feeds."""

    def __init__(
        self,
        *,
        feeds: Sequence[tuple[str, str]] = VERIFIED_PUBLIC_RSS_FEEDS,
        timeout_seconds: float = 12.0,
        max_attempts: int = 2,
        min_request_interval_seconds: float = 0.25,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], datetime] | None = None,
        mono_clock: Callable[[], float] | None = None,
    ) -> None:
        self._feeds = tuple(feeds)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            follow_redirects=True,
            transport=transport,
            headers={"User-Agent": "crypto-chatbot/1.0 (+research-bot)"},
        )
        self._max_attempts = max(1, max_attempts)
        self._min_interval = max(0.0, float(min_request_interval_seconds))
        self._clock = clock or (lambda: datetime.now(UTC))
        self._mono_clock = mono_clock or time.monotonic
        self._last_request_at: float | None = None

    async def close(self) -> None:
        await self._client.aclose()

    async def _respect_rate_limit(self) -> None:
        if self._min_interval <= 0.0:
            self._last_request_at = self._mono_clock()
            return
        if self._last_request_at is not None:
            elapsed = self._mono_clock() - self._last_request_at
            if elapsed < self._min_interval:
                await asyncio.sleep(self._min_interval - elapsed)
        self._last_request_at = self._mono_clock()

    async def fetch_feed(self, publisher: str, url: str) -> tuple[RawNewsItem, ...]:
        """Fetch and parse one RSS feed, stamping `first_seen_time = fetch_time`."""
        retrying = AsyncRetrying(
            retry=retry_if_exception(
                lambda exc: isinstance(exc, (httpx.TransportError, _RetryableNewsHttpError))
            ),
            stop=stop_after_attempt(self._max_attempts),
            wait=wait_exponential(multiplier=0.3, min=0.2, max=1.5),
            reraise=True,
        )
        async for attempt in retrying:
            with attempt:
                await self._respect_rate_limit()
                response = await self._client.get(url)
                if response.status_code in {429, 500, 502, 503, 504}:
                    raise _RetryableNewsHttpError(
                        f"RSS HTTP {response.status_code} for {publisher}"
                    )
                response.raise_for_status()
                fetch_time = self._clock().astimezone(UTC)
                return parse_rss_xml(
                    response.text,
                    publisher=publisher,
                    source_provider="rss",
                    first_seen_time=fetch_time,
                )
        raise RuntimeError("unreachable")  # pragma: no cover

    async def fetch_all(self) -> tuple[list[RawNewsItem], int]:
        """Fetch all configured feeds sequentially (rate-limited) without crashing on one error."""
        collected: list[RawNewsItem] = []
        succeeded = 0
        for publisher, url in self._feeds:
            try:
                items = await self.fetch_feed(publisher, url)
                collected.extend(items)
                succeeded += 1
            except Exception as exc:
                logger.warning("RSS feed fetch skipped for %s (%s): %s", publisher, url, exc)
        return collected, succeeded


def build_article_vote_snapshots(
    articles: Sequence[NormalizedNewsArticle],
    *,
    snapshot_time: datetime,
) -> tuple[NewsVoteSnapshot, ...]:
    """Emit append-only vote/engagement snapshots for each fetched article."""
    snap_utc = (
        snapshot_time.astimezone(UTC)
        if snapshot_time.tzinfo is not None
        else snapshot_time.replace(tzinfo=UTC)
    )
    snapshots: list[NewsVoteSnapshot] = []
    for art in articles:
        sid_src = f"{art.article_id}|{art.source_provider}|{snap_utc.isoformat()}"
        sid = "nvs-" + hashlib.sha256(sid_src.encode("utf-8")).hexdigest()[:24]
        snapshots.append(
            NewsVoteSnapshot(
                snapshot_id=sid,
                article_id=art.article_id,
                source_provider=art.source_provider,
                snapshot_time=snap_utc,
                attribution_text=build_news_attribution(
                    art.source_provider,
                    publisher=art.publisher,
                ),
            )
        )
    return tuple(snapshots)


async def sync_public_rss_news(
    *,
    client: PublicRssNewsClient,
    repository: NewsRepository | None = None,
    vote_repository: NewsVoteSnapshotRepository | None = None,
    clock: Callable[[], datetime] | None = None,
) -> tuple[NewsSyncResult, tuple[NormalizedNewsArticle, ...], tuple[SentimentObservation, ...]]:
    """Fetch public RSS headlines, deduplicate with `first_seen_time`, score, and persist."""
    now_utc = (clock or (lambda: datetime.now(UTC)))().astimezone(UTC)
    raw_items, feeds_ok = await client.fetch_all()
    articles, _dropped = deduplicate_articles(raw_items)
    observations = build_sentiment_observations(articles, scored_at=now_utc)
    snapshots = build_article_vote_snapshots(articles, snapshot_time=now_utc)

    ins_art = repository.save_articles(articles) if repository is not None else 0
    ins_obs = repository.save_observations(observations) if repository is not None else 0
    snap_repo: NewsVoteSnapshotRepository | None = vote_repository or (
        repository if isinstance(repository, NewsVoteSnapshotRepository) else None
    )
    ins_snap = snap_repo.save_vote_snapshots(snapshots) if snap_repo is not None else 0

    summary = NewsSyncResult(
        feeds_attempted=len(client._feeds),
        feeds_succeeded=feeds_ok,
        raw_items=len(raw_items),
        dedup_articles=len(articles),
        inserted_articles=ins_art,
        inserted_observations=ins_obs,
        inserted_snapshots=ins_snap,
    )
    return summary, articles, observations
