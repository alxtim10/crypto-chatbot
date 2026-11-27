"""Validated schemas for traceable Phase 11 news ingestion and sentiment aggregation."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from crypto_chatbot.schemas.market import FiniteFloat, UtcDatetime

SENTIMENT_MODEL_VERSION = "crypto-lexicon-v1.0.0"
SENTIMENT_SCORER_ID = "crypto-lexicon"
MIN_ENTITY_MATCH_CONFIDENCE = 0.65

ScorerClass = Literal["lexicon_deterministic", "llm"]


class SentimentLabel(StrEnum):
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


class RawNewsItem(BaseModel):
    """Untrusted raw news article payload before canonicalization and deduplication."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    publisher: str = Field(min_length=1)
    url: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str = ""
    published_time: UtcDatetime
    first_seen_time: UtcDatetime
    source_provider: str = Field(default="fixture", min_length=1)
    raw_payload: dict[str, Any] = Field(default_factory=dict)


class EntityMatch(BaseModel):
    """Asset/coin match extracted from an article with explicit confidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_symbol: str = Field(min_length=1)
    base_asset: str = Field(min_length=1)
    confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    match_reason: str = Field(min_length=1)


class NormalizedNewsArticle(BaseModel):
    """Deduplicated news article with canonical URL, UTC timestamps, and entity matches."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    article_id: str = Field(min_length=16, max_length=64)
    canonical_url: str = Field(min_length=1)
    content_fingerprint: str = Field(min_length=16, max_length=64)
    publisher: str = Field(min_length=1)
    title: str = Field(min_length=1)
    summary: str = ""
    published_time: UtcDatetime
    first_seen_time: UtcDatetime
    available_time: UtcDatetime
    source_provider: str = Field(min_length=1)
    affected_entities: tuple[EntityMatch, ...] = ()
    raw_payload: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_article(self) -> NormalizedNewsArticle:
        if not self.canonical_url.startswith(("http://", "https://")):
            raise ValueError("canonical_url must start with http:// or https://")
        if self.available_time < self.published_time:
            raise ValueError("available_time cannot be earlier than published_time")
        if self.available_time < self.first_seen_time:
            raise ValueError("available_time cannot be earlier than first_seen_time")
        return self


class SentimentObservation(BaseModel):
    """Per-article, per-coin non-probabilistic sentiment observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observation_id: str = Field(min_length=16, max_length=64)
    article_id: str = Field(min_length=16, max_length=64)
    canonical_url: str = Field(min_length=1)
    publisher: str = Field(min_length=1)
    title: str = Field(min_length=1)
    canonical_symbol: str = Field(min_length=1)
    base_asset: str = Field(min_length=1)
    entity_confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    sentiment_label: SentimentLabel
    sentiment_score: FiniteFloat = Field(ge=-1.0, le=1.0)
    sentiment_confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    model_version: str = Field(default=SENTIMENT_MODEL_VERSION, min_length=1)
    is_probabilistic: Literal[False] = False
    published_time: UtcDatetime
    available_time: UtcDatetime
    scored_at: UtcDatetime | None = None
    scorer_id: str = Field(default=SENTIMENT_SCORER_ID, min_length=1)
    scorer_version: str = Field(default=SENTIMENT_MODEL_VERSION, min_length=1)
    scorer_class: ScorerClass = "lexicon_deterministic"
    is_backfilled: bool = False

    @model_validator(mode="after")
    def _validate_observation(self) -> SentimentObservation:
        if not self.canonical_url.startswith(("http://", "https://")):
            raise ValueError("canonical_url must start with http:// or https://")
        if self.available_time < self.published_time:
            raise ValueError("available_time cannot be earlier than published_time")
        if self.is_backfilled and self.scorer_class == "llm":
            raise ValueError("LLM-based sentiment scores cannot be backfilled (PRD v1.7 §6.2)")
        return self


class SentimentBucket(BaseModel):
    """Chronological sentiment aggregation bucket inside the causal lookback window."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    bucket_start: UtcDatetime
    bucket_end: UtcDatetime
    article_count: int = Field(ge=0)
    net_score: FiniteFloat = Field(ge=-1.0, le=1.0)
    mean_confidence: FiniteFloat = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_bucket_bounds(self) -> SentimentBucket:
        if self.bucket_end <= self.bucket_start:
            raise ValueError("bucket_end must be greater than bucket_start")
        return self


class NewsSourceLink(BaseModel):
    """Verified article source attribution rendered deterministically alongside summaries."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    publisher: str = Field(min_length=1)
    title: str = Field(min_length=1)
    canonical_url: str = Field(min_length=1)
    published_time: UtcDatetime
    available_time: UtcDatetime
    sentiment_label: SentimentLabel
    sentiment_confidence: FiniteFloat = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_url(self) -> NewsSourceLink:
        if not self.canonical_url.startswith(("http://", "https://")):
            raise ValueError("canonical_url must start with http:// or https://")
        return self


class SentimentMetrics(BaseModel):
    """Causal time-bucketed news sentiment metrics attached to `AnalysisSnapshot`."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    canonical_symbol: str = Field(min_length=1)
    window_hours: int = Field(default=24, ge=1)
    bucket_hours: int = Field(default=4, ge=1)
    model_version: str = Field(default=SENTIMENT_MODEL_VERSION, min_length=1)
    is_probabilistic: Literal[False] = False
    article_count: int = Field(ge=0)
    positive_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    neutral_count: int = Field(ge=0)
    net_sentiment_score: FiniteFloat = Field(ge=-1.0, le=1.0)
    mean_confidence: FiniteFloat = Field(ge=0.0, le=1.0)
    dominant_label: SentimentLabel
    latest_available_time: UtcDatetime | None = None
    sentiment_backfilled: bool = False
    buckets: tuple[SentimentBucket, ...] = ()
    sources: tuple[NewsSourceLink, ...] = ()
