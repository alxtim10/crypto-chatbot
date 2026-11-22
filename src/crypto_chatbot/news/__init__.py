"""Traceable Phase 11 news ingestion and non-probabilistic sentiment analysis."""

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
    "aggregate_sentiment",
    "build_sentiment_observations",
    "canonicalize_url",
    "classify_article_sentiment",
    "compute_content_fingerprint",
    "deduplicate_articles",
    "match_entities",
    "parse_rss_xml",
]
