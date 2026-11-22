"""Phase 11 tests: traceable news deduplication, causality, entity matching, and LLM safety."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from crypto_chatbot.analysis.service import build_analysis_snapshot
from crypto_chatbot.bot.templates import render_compact_snapshot, render_deterministic_snapshot
from crypto_chatbot.llm.gemini import GeminiNarrator, RawGeminiOutput
from crypto_chatbot.llm.prompts import build_narration_user_payload
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import resolve_mvp_spot_symbol
from crypto_chatbot.news import (
    aggregate_sentiment,
    build_sentiment_observations,
    canonicalize_url,
    classify_article_sentiment,
    deduplicate_articles,
    match_entities,
    parse_rss_xml,
)
from crypto_chatbot.schemas.llm import NarrationStatus
from crypto_chatbot.schemas.market import QualityStatus, Timeframe
from crypto_chatbot.schemas.news import (
    MIN_ENTITY_MATCH_CONFIDENCE,
    SENTIMENT_MODEL_VERSION,
    RawNewsItem,
    SentimentLabel,
)

NEWS_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "news_sentiment_sample.json"
CANDLE_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"
ANCHOR_NOW = datetime(2026, 10, 14, 16, 0, 0, tzinfo=UTC)


def _load_fixture_items() -> list[RawNewsItem]:
    raw_list = json.loads(NEWS_FIXTURE.read_text(encoding="utf-8"))
    return [RawNewsItem.model_validate(item) for item in raw_list]


def test_url_canonicalization_and_syndication_dedup() -> None:
    canon = canonicalize_url(
        "https://www.CoinDesk.com/markets/btc/?utm_source=x&fbclid=abc&b=2&a=1#frag"
    )
    assert canon == "https://coindesk.com/markets/btc?a=1&b=2"

    with pytest.raises(ValueError, match="unsupported"):
        canonicalize_url("javascript:alert(1)")

    items = _load_fixture_items()
    kept, dropped = deduplicate_articles(items)
    # 7 raw items minus exact tracking-param URL duplicate and Yahoo syndication reprint = 5 unique
    assert len(kept) == 5
    assert dropped == 2
    assert len({art.canonical_url for art in kept}) == 5
    assert len({art.content_fingerprint for art in kept}) == 5


def test_low_confidence_entity_matches_are_excluded() -> None:
    ambiguous_title = "Warm sol shines near coastal link roads in spring"
    ambiguous_summary = "Drivers enjoyed clear coastal routes and mild weather."

    default_matches = match_entities(ambiguous_title, ambiguous_summary)
    assert default_matches == ()

    raw_matches = match_entities(ambiguous_title, ambiguous_summary, min_confidence=0.0)
    assert raw_matches
    assert all(m.confidence < MIN_ENTITY_MATCH_CONFIDENCE for m in raw_matches)

    explicit = match_entities("$SOL and Solana rally on DeFi ETF upgrade", "")
    assert len(explicit) == 1
    assert explicit[0].canonical_symbol == "SOL/USDT"
    assert explicit[0].confidence >= 0.9


def test_causal_point_in_time_aggregation_excludes_future_articles() -> None:
    items = _load_fixture_items()
    articles, _ = deduplicate_articles(items)
    observations = build_sentiment_observations(articles)

    # At ANCHOR_NOW (16:00Z), the 16:06Z Reuters crash article is NOT available yet.
    metrics_now = aggregate_sentiment(
        observations,
        canonical_symbol="BTC/USDT",
        as_of=ANCHOR_NOW,
        window_hours=24,
        bucket_hours=4,
    )
    assert metrics_now is not None
    assert metrics_now.is_probabilistic is False
    assert metrics_now.model_version == SENTIMENT_MODEL_VERSION
    assert metrics_now.article_count == 3  # CoinDesk (deduped), The Block, CryptoWire
    assert metrics_now.negative_count == 0
    assert metrics_now.dominant_label is SentimentLabel.POSITIVE
    assert metrics_now.latest_available_time is not None
    assert metrics_now.latest_available_time <= ANCHOR_NOW
    assert len(metrics_now.buckets) == 6
    assert sum(b.article_count for b in metrics_now.buckets) == 3

    # Advancing as_of past 16:06Z picks up the Reuters bearish article.
    metrics_later = aggregate_sentiment(
        observations,
        canonical_symbol="BTC/USDT",
        as_of=ANCHOR_NOW + timedelta(minutes=10),
        window_hours=24,
        bucket_hours=4,
    )
    assert metrics_later is not None
    assert metrics_later.article_count == 4
    assert metrics_later.negative_count == 1


def test_sentiment_negation_and_rss_xml_parsing() -> None:
    pos_lbl, pos_score, _ = classify_article_sentiment("Bitcoin ETF triggers rally", "")
    assert pos_lbl is SentimentLabel.POSITIVE
    assert pos_score > 0.15

    negated_lbl, neg_score, _ = classify_article_sentiment("Bitcoin ETF will not rally", "")
    assert negated_lbl is SentimentLabel.NEGATIVE
    assert neg_score < -0.15

    rss = """<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0"><channel><item>
      <title>Ethereum upgrade boosts scaling adoption</title>
      <link>https://example.com/eth-upgrade?utm_source=rss</link>
      <description>ETH institutional demand grows.</description>
      <pubDate>Wed, 14 Oct 2026 12:00:00 GMT</pubDate>
    </item></channel></rss>"""
    parsed = parse_rss_xml(rss, publisher="ExampleNews", first_seen_time=ANCHOR_NOW)
    assert len(parsed) == 1
    assert parsed[0].publisher == "ExampleNews"
    assert parsed[0].published_time == datetime(2026, 10, 14, 12, 0, tzinfo=UTC)

    xxe = '<!DOCTYPE foo [<!ENTITY xxe "bad">]><rss><channel></channel></rss>'
    with pytest.raises(ValueError, match="DOCTYPE"):
        parse_rss_xml(xxe, publisher="BadNews")


def test_snapshot_integration_preserves_baseline_and_links_sources() -> None:
    raw_candles = json.loads(CANDLE_FIXTURE.read_text(encoding="utf-8"))["raw_ohlcv"]
    batch = normalize_ohlcv_rows(
        raw_candles,
        symbol=resolve_mvp_spot_symbol("BTC"),
        timeframe=Timeframe.H4,
        received_time=ANCHOR_NOW + timedelta(seconds=1),
        now=ANCHOR_NOW + timedelta(seconds=1),
    )
    base_snap = build_analysis_snapshot(batch)
    assert base_snap.sentiment is None

    items = _load_fixture_items()
    articles, _ = deduplicate_articles(items)
    observations = build_sentiment_observations(articles)

    sent_snap = build_analysis_snapshot(batch, news_observations=observations)
    assert sent_snap.quality_status is QualityStatus.READY
    assert sent_snap.sentiment is not None
    assert sent_snap.sentiment.is_probabilistic is False
    assert sent_snap.sentiment.article_count == 3
    assert sent_snap.facts[: len(base_snap.facts)] == base_snap.facts
    assert any(
        f.rule_id == "news_sentiment_24h" and f.is_probabilistic is False for f in sent_snap.factors
    )

    plain = render_deterministic_snapshot(sent_snap, html_mode=False)
    compact = render_compact_snapshot(sent_snap, html_mode=True)
    assert "Sentimen Berita 24h" in plain
    assert "https://www.theblock.co/post/20261014/btc-breakout-rebound" in plain or (
        "https://theblock.co/post/20261014/btc-breakout-rebound" in plain
    )
    assert '<a href="https://' in compact
    assert "Sentimen Berita 24h" in compact


@pytest.mark.asyncio
async def test_gemini_cannot_follow_prompt_injection_in_article_content() -> None:
    raw_candles = json.loads(CANDLE_FIXTURE.read_text(encoding="utf-8"))["raw_ohlcv"]
    batch = normalize_ohlcv_rows(
        raw_candles,
        symbol=resolve_mvp_spot_symbol("BTC"),
        timeframe=Timeframe.H4,
        received_time=ANCHOR_NOW + timedelta(seconds=1),
        now=ANCHOR_NOW + timedelta(seconds=1),
    )
    items = _load_fixture_items()
    articles, _ = deduplicate_articles(items)
    observations = build_sentiment_observations(articles)
    snap = build_analysis_snapshot(batch, news_observations=observations)

    payload = build_narration_user_payload(snap)
    # Breakout tag neutralized and hostile title confined strictly inside <untrusted_data>
    assert "[filtered_tag]" in payload
    trusted_json_section = payload.split("Quoted untrusted data", 1)[0]
    assert "999999" not in trusted_json_section
    assert "BUY sekarang" not in trusted_json_section

    class _CompromisedTransport:
        async def generate_json(self, **kwargs: Any) -> RawGeminiOutput:
            return RawGeminiOutput(
                text=json.dumps(
                    {
                        "summary": "BTC pasti ke 999999 BUY sekarang ({F1}).",
                        "supporting_factors": ["Dorongan ETF ({F1})."],
                        "contradicting_factors": ["Risiko volatilitas ({F1})."],
                        "bullish_scenario": "Jika bertahan di atas {F1}.",
                        "bearish_scenario": "Jika turun di bawah {F1}.",
                        "limitations": ["Risiko pasar ({F1})."],
                        "fact_ids": ["F1"],
                    }
                ),
                model_version="gemini-test",
                prompt_tokens=100,
                output_tokens=40,
            )

    narrator = GeminiNarrator(model="gemini-test", transport=_CompromisedTransport())
    outcome = await narrator.narrate_snapshot(snap, user_id=111)
    assert outcome.validation_passed is False
    assert outcome.response.narration_status is NarrationStatus.FALLBACK_DETERMINISTIC
    assert any(
        r in outcome.rejection_reasons
        for r in ("raw_digit_detected", "overconfidence_lexicon:pasti")
    )
