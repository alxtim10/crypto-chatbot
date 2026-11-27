"""Traceable Phase 11 news ingestion and non-probabilistic sentiment analysis."""

from crypto_chatbot.news.fear_greed import (
    DEFAULT_FNG_API_URL,
    FearGreedClient,
    FearGreedSyncResult,
    normalize_fear_greed_payload,
    sync_fear_greed,
)
from crypto_chatbot.news.ingestor import (
    VERIFIED_PUBLIC_RSS_FEEDS,
    NewsSyncResult,
    PublicRssNewsClient,
    build_article_vote_snapshots,
    build_news_attribution,
    sync_public_rss_news,
)
from crypto_chatbot.news.pipeline import (
    aggregate_sentiment,
    build_sentiment_observations,
    canonicalize_url,
    classify_article_sentiment,
    compute_content_fingerprint,
    deduplicate_articles,
    match_entities,
    parse_rss_xml,
)

__all__ = [
    "DEFAULT_FNG_API_URL",
    "VERIFIED_PUBLIC_RSS_FEEDS",
    "FearGreedClient",
    "FearGreedSyncResult",
    "NewsSyncResult",
    "PublicRssNewsClient",
    "aggregate_sentiment",
    "build_article_vote_snapshots",
    "build_news_attribution",
    "build_sentiment_observations",
    "canonicalize_url",
    "classify_article_sentiment",
    "compute_content_fingerprint",
    "deduplicate_articles",
    "match_entities",
    "normalize_fear_greed_payload",
    "parse_rss_xml",
    "sync_fear_greed",
    "sync_public_rss_news",
]
