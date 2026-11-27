"""Traceable Phase 11 news ingestion and non-probabilistic sentiment analysis."""

from crypto_chatbot.news.fear_greed import (
    DEFAULT_FNG_API_URL,
    FearGreedClient,
    FearGreedSyncResult,
    normalize_fear_greed_payload,
    sync_fear_greed,
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
    "FearGreedClient",
    "FearGreedSyncResult",
    "aggregate_sentiment",
    "build_sentiment_observations",
    "canonicalize_url",
    "classify_article_sentiment",
    "compute_content_fingerprint",
    "deduplicate_articles",
    "match_entities",
    "normalize_fear_greed_payload",
    "parse_rss_xml",
    "sync_fear_greed",
]
