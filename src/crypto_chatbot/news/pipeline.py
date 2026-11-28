"""Traceable Phase 11 news & sentiment pipeline (no external dependencies)."""

from __future__ import annotations

import hashlib
import re
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

from crypto_chatbot.schemas.news import (
    MIN_ENTITY_MATCH_CONFIDENCE,
    SENTIMENT_MODEL_VERSION,
    SENTIMENT_SCORER_ID,
    EntityMatch,
    NewsSourceLink,
    NormalizedNewsArticle,
    RawNewsItem,
    ScorerClass,
    SentimentBucket,
    SentimentLabel,
    SentimentMetrics,
    SentimentObservation,
)

_TRACKING_QUERY_KEYS: frozenset[str] = frozenset(
    {
        "campaign",
        "cmpid",
        "fbclid",
        "gclid",
        "guccounter",
        "mc_cid",
        "mc_eid",
        "ref",
        "si",
        "source",
    }
)
_WIRE_PREFIX_RE = re.compile(
    r"^(?:reuters|coindesk|cointelegraph|decrypt|the block|bloomberg|cnbc|"
    r"breaking|update|exclusive|opinion)\s*[-:|]\s*",
    re.IGNORECASE,
)
_TRAILING_WIRE_RE = re.compile(r"\s+(?:[-|—])\s+[A-Za-z0-9 .]{2,30}$")
_WORD_RE = re.compile(r"[a-z0-9]+")
_RAW_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")
_EXPLICIT_DYN_TICKER_RE = re.compile(r"(?:\$([A-Z]{2,10})\b|\b([A-Z0-9]{2,10})/USDT(?::USDT)?\b)")

_STOP_WORDS: frozenset[str] = frozenset(
    {
        "a",
        "after",
        "amid",
        "an",
        "and",
        "are",
        "as",
        "at",
        "by",
        "for",
        "from",
        "in",
        "is",
        "of",
        "on",
        "or",
        "over",
        "the",
        "to",
        "via",
        "with",
    }
)

_KNOWN_ASSETS: dict[str, tuple[str, ...]] = {
    "BTC": ("bitcoin", "xbt"),
    "ETH": ("ethereum",),
    "SOL": ("solana",),
    "BNB": ("binance coin", "bnb chain"),
    "XRP": ("ripple", "xrp ledger"),
    "DOGE": ("dogecoin",),
    "PEPE": ("pepecoin", "pepe coin"),
    "ADA": ("cardano",),
    "AVAX": ("avalanche",),
    "LINK": ("chainlink",),
    "SUI": ("sui network", "sui blockchain"),
    "ARB": ("arbitrum",),
    "OP": ("optimism",),
    "TON": ("toncoin",),
    "LTC": ("litecoin",),
    "NEAR": ("near protocol",),
    "DOT": ("polkadot",),
    "TRX": ("tron",),
    "SHIB": ("shiba inu",),
}

_AMBIGUOUS_TICKERS: frozenset[str] = frozenset(
    {
        "APE",
        "ARB",
        "GAS",
        "GMT",
        "KEY",
        "LINK",
        "MANA",
        "NEAR",
        "ONE",
        "OP",
        "SAND",
        "SOL",
        "SUI",
        "TON",
    }
)
_IGNORED_DYNAMIC_TOKENS: frozenset[str] = frozenset(
    {
        "AI",
        "CEO",
        "CPI",
        "ETF",
        "EUR",
        "FED",
        "FOMC",
        "GDP",
        "IDR",
        "IPO",
        "SEC",
        "USD",
        "USDC",
        "USDT",
    }
)
_CRYPTO_CONTEXT_RE = re.compile(
    r"\b(?:altcoin|binance|blockchain|coin|crypto|cryptocurrency|defi|etf|exchange|"
    r"futures|market|perp|perpetual|price|rally|sec|staking|token|trading|wallet)\b",
    re.IGNORECASE,
)

_NEGATION_WORDS: frozenset[str] = frozenset(
    {
        "avoids",
        "avoided",
        "batal",
        "belum",
        "bukan",
        "denied",
        "denies",
        "deny",
        "fail",
        "failed",
        "fails",
        "gagal",
        "neither",
        "never",
        "no",
        "nor",
        "not",
        "reject",
        "rejected",
        "rejects",
        "tak",
        "tidak",
        "without",
    }
)
_POSITIVE_WORDS: frozenset[str] = frozenset(
    {
        "accumulate",
        "accumulates",
        "accumulation",
        "adoption",
        "approval",
        "approved",
        "breakout",
        "bullish",
        "gain",
        "gains",
        "growth",
        "inflow",
        "inflows",
        "institutional",
        "kuat",
        "menguat",
        "naik",
        "outperform",
        "partnership",
        "rallied",
        "rallies",
        "rally",
        "rebound",
        "record",
        "rekor",
        "soar",
        "soars",
        "surge",
        "surged",
        "surges",
        "upgrade",
        "upgraded",
    }
)
_NEGATIVE_WORDS: frozenset[str] = frozenset(
    {
        "anjlok",
        "ban",
        "banned",
        "bankrupt",
        "bankruptcy",
        "bearish",
        "bocor",
        "collapse",
        "collapsed",
        "crash",
        "crashed",
        "crashes",
        "downgrade",
        "exploit",
        "exploited",
        "fear",
        "fraud",
        "gugatan",
        "hack",
        "hacked",
        "halt",
        "halted",
        "insolvent",
        "jatuh",
        "lawsuit",
        "liquidation",
        "liquidations",
        "outflow",
        "outflows",
        "plunge",
        "plunged",
        "plunges",
        "sanction",
        "sanctions",
        "selloff",
        "slump",
        "slumps",
        "sued",
        "turun",
    }
)


def canonicalize_url(url: str) -> str:
    """Normalize HTTP/HTTPS article URLs by stripping fragments, `www.`, and tracking params."""
    raw = url.strip()
    parsed = urlparse(raw)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"unsupported or invalid article URL: '{url}'")

    host, sep, port = parsed.netloc.lower().partition(":")
    if host.startswith("www.") and len(host) > 4:
        host = host[4:]
    netloc = f"{host}{sep}{port}" if sep else host

    path = parsed.path or "/"
    if len(path) > 1:
        path = path.rstrip("/") or "/"

    kept_params = sorted(
        (k, v)
        for k, v in parse_qsl(parsed.query, keep_blank_values=False)
        if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_QUERY_KEYS
    )
    query = urlencode(kept_params, doseq=True)
    return urlunparse((scheme, netloc, path, "", query, ""))


def _strip_wire_affixes(title: str) -> str:
    cleaned = title.strip()
    while True:
        updated = _WIRE_PREFIX_RE.sub("", cleaned, count=1)
        if updated == cleaned:
            break
        cleaned = updated.strip()
    return _TRAILING_WIRE_RE.sub("", cleaned).strip()


def compute_content_fingerprint(title: str) -> str:
    """Compute a normalized SHA-256 fingerprint over a headline with wire labels stripped."""
    stripped = _strip_wire_affixes(title).lower()
    tokens = _WORD_RE.findall(stripped)
    normalized = " ".join(tokens)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _title_token_set(title: str) -> frozenset[str]:
    stripped = _strip_wire_affixes(title).lower()
    return frozenset(tok for tok in _WORD_RE.findall(stripped) if tok not in _STOP_WORDS)


def _jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    if not a or not b:
        return 0.0
    union_len = len(a | b)
    return len(a & b) / union_len if union_len else 0.0


def match_entities(
    title: str,
    summary: str = "",
    *,
    min_confidence: float = MIN_ENTITY_MATCH_CONFIDENCE,
) -> tuple[EntityMatch, ...]:
    """Identify affected crypto assets deterministically with confidence filtering."""
    combined = f"{title}\n{summary}"
    has_crypto_context = bool(_CRYPTO_CONTEXT_RE.search(combined))
    best_by_base: dict[str, tuple[float, str]] = {}

    def record(base: str, confidence: float, reason: str) -> None:
        if base in _IGNORED_DYNAMIC_TOKENS:
            return
        prev = best_by_base.get(base)
        if prev is None or confidence > prev[0]:
            best_by_base[base] = (confidence, reason)

    for base, aliases in _KNOWN_ASSETS.items():
        explicit_re = re.compile(rf"(?:\${base}\b|\b{base}/USDT(?::USDT)?\b)", re.IGNORECASE)
        if explicit_re.search(title):
            record(base, 0.95, "title_explicit_ticker")
            continue
        if explicit_re.search(summary):
            record(base, 0.90, "summary_explicit_ticker")

        title_alias_hit = any(
            re.search(rf"\b{re.escape(alias)}\b", title, re.IGNORECASE) for alias in aliases
        )
        if title_alias_hit:
            record(base, 0.95, "title_full_name")
            continue

        summary_alias_hit = any(
            re.search(rf"\b{re.escape(alias)}\b", summary, re.IGNORECASE) for alias in aliases
        )
        if summary_alias_hit:
            record(base, 0.85, "summary_full_name")

        upper_title_hit = bool(re.search(rf"\b{re.escape(base)}\b", title))
        upper_summary_hit = bool(re.search(rf"\b{re.escape(base)}\b", summary))
        if upper_title_hit:
            if base in _AMBIGUOUS_TICKERS:
                conf = 0.75 if has_crypto_context else 0.40
                record(base, conf, "title_uppercase_ambiguous_ticker")
            else:
                record(base, 0.85, "title_uppercase_ticker")
        elif upper_summary_hit:
            if base in _AMBIGUOUS_TICKERS:
                conf = 0.68 if has_crypto_context else 0.35
                record(base, conf, "summary_uppercase_ambiguous_ticker")
            else:
                record(base, 0.70, "summary_uppercase_ticker")
        elif base in _AMBIGUOUS_TICKERS and re.search(
            rf"\b{re.escape(base)}\b", combined, re.IGNORECASE
        ):
            record(base, 0.40, "ambiguous_lowercase_word")

    for m in _EXPLICIT_DYN_TICKER_RE.finditer(combined):
        dyn = (m.group(1) or m.group(2) or "").upper()
        if dyn:
            record(dyn, 0.90, "dynamic_explicit_ticker")

    matches = [
        EntityMatch(
            canonical_symbol=f"{base}/USDT",
            base_asset=base,
            confidence=conf,
            match_reason=reason,
        )
        for base, (conf, reason) in sorted(best_by_base.items())
        if conf >= min_confidence
    ]
    return tuple(matches)


def _score_token_stream(tokens: Sequence[str], *, weight: float) -> tuple[float, float, int]:
    pos = 0.0
    neg = 0.0
    hits = 0
    for idx, raw in enumerate(tokens):
        tok = raw.lower()
        window_start = max(0, idx - 2)
        negated = any(tokens[j].lower() in _NEGATION_WORDS for j in range(window_start, idx))
        if tok in _POSITIVE_WORDS:
            hits += 1
            if negated:
                neg += weight
            else:
                pos += weight
        elif tok in _NEGATIVE_WORDS:
            hits += 1
            if negated:
                pos += weight
            else:
                neg += weight
    return pos, neg, hits


def classify_article_sentiment(
    title: str,
    summary: str = "",
) -> tuple[SentimentLabel, float, float]:
    """Deterministically score headline + summary with negation-aware crypto lexicon."""
    title_tokens = _RAW_TOKEN_RE.findall(title)
    summary_tokens = _RAW_TOKEN_RE.findall(summary)

    pos_t, neg_t, hits_t = _score_token_stream(title_tokens, weight=1.5)
    pos_s, neg_s, hits_s = _score_token_stream(summary_tokens, weight=1.0)
    pos = pos_t + pos_s
    neg = neg_t + neg_s
    total_hits = hits_t + hits_s

    raw_score = (pos - neg) / (pos + neg + 1.5) if total_hits > 0 else 0.0
    score = max(-1.0, min(1.0, raw_score))
    if score >= 0.15:
        label = SentimentLabel.POSITIVE
    elif score <= -0.15:
        label = SentimentLabel.NEGATIVE
    else:
        label = SentimentLabel.NEUTRAL

    confidence = min(0.95, 0.35 + 0.15 * total_hits + 0.30 * abs(score)) if total_hits > 0 else 0.25
    return label, score, confidence


def deduplicate_articles(
    items: Sequence[RawNewsItem],
    *,
    syndication_window: timedelta = timedelta(hours=24),
    jaccard_threshold: float = 0.80,
    min_entity_confidence: float = MIN_ENTITY_MATCH_CONFIDENCE,
) -> tuple[tuple[NormalizedNewsArticle, ...], int]:
    """Canonicalize URLs/timestamps and drop exact or syndicated near-duplicate items."""
    prepared: list[tuple[str, str, datetime, RawNewsItem]] = []
    for item in items:
        canon_url = canonicalize_url(item.url)
        fp = compute_content_fingerprint(item.title)
        pub_utc = item.published_time.astimezone(UTC)
        seen_utc = item.first_seen_time.astimezone(UTC)
        avail_utc = max(pub_utc, seen_utc)
        prepared.append((canon_url, fp, avail_utc, item))

    prepared.sort(key=lambda entry: (entry[2], entry[0]))

    seen_urls: set[str] = set()
    seen_fps: set[str] = set()
    kept_tokens: list[tuple[datetime, frozenset[str]]] = []
    kept: list[NormalizedNewsArticle] = []
    dropped = 0
    window_secs = syndication_window.total_seconds()

    for canon_url, fp, avail_utc, item in prepared:
        tokens = _title_token_set(item.title)
        if canon_url in seen_urls or fp in seen_fps:
            dropped += 1
            continue

        is_syndicated = False
        if len(tokens) >= 4:
            for prev_avail, prev_tokens in kept_tokens:
                if (
                    abs((avail_utc - prev_avail).total_seconds()) <= window_secs
                    and _jaccard_similarity(tokens, prev_tokens) >= jaccard_threshold
                ):
                    is_syndicated = True
                    break
        if is_syndicated:
            dropped += 1
            continue

        seen_urls.add(canon_url)
        seen_fps.add(fp)
        kept_tokens.append((avail_utc, tokens))

        article_id = "news-" + hashlib.sha256(canon_url.encode("utf-8")).hexdigest()[:24]
        entities = match_entities(
            item.title,
            item.summary,
            min_confidence=min_entity_confidence,
        )
        kept.append(
            NormalizedNewsArticle(
                article_id=article_id,
                canonical_url=canon_url,
                content_fingerprint=fp,
                publisher=item.publisher.strip(),
                title=item.title.strip(),
                summary=item.summary.strip(),
                published_time=item.published_time.astimezone(UTC),
                first_seen_time=item.first_seen_time.astimezone(UTC),
                available_time=avail_utc,
                source_provider=item.source_provider,
                affected_entities=entities,
                raw_payload=dict(item.raw_payload),
            )
        )

    return tuple(kept), dropped


def build_sentiment_observations(
    articles: Sequence[NormalizedNewsArticle],
    *,
    min_entity_confidence: float = MIN_ENTITY_MATCH_CONFIDENCE,
    scored_at: datetime | None = None,
    scorer_id: str = SENTIMENT_SCORER_ID,
    scorer_version: str = SENTIMENT_MODEL_VERSION,
    scorer_class: ScorerClass = "lexicon_deterministic",
    is_backfilled: bool = False,
) -> tuple[SentimentObservation, ...]:
    """Score normalized articles and emit causal observations per high-confidence entity.

    Raises `ValueError` if `is_backfilled=True` with `scorer_class='llm'` (PRD v1.7 §6.2).
    """
    if is_backfilled and scorer_class == "llm":
        raise ValueError("LLM-based sentiment scores must never be backfilled (PRD v1.7 §6.2)")
    observations: list[SentimentObservation] = []
    for article in articles:
        qualifying = [e for e in article.affected_entities if e.confidence >= min_entity_confidence]
        if not qualifying:
            continue
        label, score, confidence = classify_article_sentiment(article.title, article.summary)
        eff_scored_at = (
            scored_at.astimezone(UTC) if scored_at is not None else article.available_time
        )
        for entity in qualifying:
            raw_obs_key = f"{article.article_id}|{entity.canonical_symbol}|{scorer_version}"
            obs_id = "sent-" + hashlib.sha256(raw_obs_key.encode("utf-8")).hexdigest()[:24]
            observations.append(
                SentimentObservation(
                    observation_id=obs_id,
                    article_id=article.article_id,
                    canonical_url=article.canonical_url,
                    publisher=article.publisher,
                    title=article.title,
                    canonical_symbol=entity.canonical_symbol,
                    base_asset=entity.base_asset,
                    entity_confidence=entity.confidence,
                    sentiment_label=label,
                    sentiment_score=score,
                    sentiment_confidence=confidence,
                    model_version=scorer_version,
                    is_probabilistic=False,
                    published_time=article.published_time,
                    available_time=article.available_time,
                    scored_at=eff_scored_at,
                    scorer_id=scorer_id,
                    scorer_version=scorer_version,
                    scorer_class=scorer_class,
                    is_backfilled=is_backfilled,
                )
            )
    return tuple(observations)


def is_sentiment_observation_eligible(
    obs: SentimentObservation,
    as_of: datetime,
    *,
    allow_lexicon_backfill: bool = True,
) -> bool:
    """Return True iff `obs` satisfies PRD v1.7 §6.2 point-in-time eligibility at `as_of`."""
    as_of_utc = as_of.astimezone(UTC)
    avail = obs.available_time.astimezone(UTC)
    if avail > as_of_utc:
        return False
    eff_scored = (obs.scored_at or obs.available_time).astimezone(UTC)
    if eff_scored <= as_of_utc:
        return True
    return bool(
        allow_lexicon_backfill and obs.scorer_class == "lexicon_deterministic" and obs.is_backfilled
    )


def _symbol_match_base(symbol: str) -> str:
    raw = symbol.strip().upper()
    if "/" in raw:
        return raw.split("/", 1)[0]
    if "_" in raw:
        return raw.split("_", 1)[0]
    if ":" in raw:
        return raw.split(":", 1)[0]
    if raw.endswith("USDT") and len(raw) > 4:
        return raw[:-4]
    return raw


def aggregate_sentiment(
    observations: Sequence[SentimentObservation],
    *,
    canonical_symbol: str,
    as_of: datetime,
    window_hours: int = 24,
    bucket_hours: int = 4,
    min_entity_confidence: float = MIN_ENTITY_MATCH_CONFIDENCE,
    allow_lexicon_backfill: bool = True,
) -> SentimentMetrics | None:
    """Aggregate causal sentiment observations strictly inside `(as_of - window, as_of]`."""
    as_of_utc = as_of.astimezone(UTC)
    window_start = as_of_utc - timedelta(hours=window_hours)
    target_base = _symbol_match_base(canonical_symbol)

    by_article: dict[str, SentimentObservation] = {}
    for obs in observations:
        if (
            obs.canonical_symbol != canonical_symbol
            and _symbol_match_base(obs.canonical_symbol) != target_base
            and obs.base_asset.strip().upper() != target_base
        ):
            continue
        if obs.entity_confidence < min_entity_confidence:
            continue
        avail = obs.available_time.astimezone(UTC)
        if avail <= window_start or avail > as_of_utc:
            continue
        if not is_sentiment_observation_eligible(
            obs,
            as_of_utc,
            allow_lexicon_backfill=allow_lexicon_backfill,
        ):
            continue
        prev = by_article.get(obs.article_id)
        if prev is None or obs.sentiment_confidence > prev.sentiment_confidence:
            by_article[obs.article_id] = obs

    if not by_article:
        return None

    filtered = list(by_article.values())
    pos_count = sum(1 for o in filtered if o.sentiment_label is SentimentLabel.POSITIVE)
    neg_count = sum(1 for o in filtered if o.sentiment_label is SentimentLabel.NEGATIVE)
    neu_count = sum(1 for o in filtered if o.sentiment_label is SentimentLabel.NEUTRAL)

    weight_sum = sum(o.sentiment_confidence for o in filtered)
    if weight_sum > 0:
        net_score = sum(o.sentiment_score * o.sentiment_confidence for o in filtered) / weight_sum
    else:
        net_score = sum(o.sentiment_score for o in filtered) / len(filtered)
    net_score = max(-1.0, min(1.0, net_score))
    mean_conf = sum(o.sentiment_confidence for o in filtered) / len(filtered)

    if net_score >= 0.15:
        dominant = SentimentLabel.POSITIVE
    elif net_score <= -0.15:
        dominant = SentimentLabel.NEGATIVE
    else:
        dominant = SentimentLabel.NEUTRAL

    buckets: list[SentimentBucket] = []
    bucket_delta = timedelta(hours=bucket_hours)
    cursor = window_start
    while cursor < as_of_utc:
        b_end = min(as_of_utc, cursor + bucket_delta)
        in_bucket = [o for o in filtered if cursor < o.available_time.astimezone(UTC) <= b_end]
        if in_bucket:
            bw = sum(o.sentiment_confidence for o in in_bucket)
            b_score = (
                sum(o.sentiment_score * o.sentiment_confidence for o in in_bucket) / bw
                if bw > 0
                else sum(o.sentiment_score for o in in_bucket) / len(in_bucket)
            )
            b_score = max(-1.0, min(1.0, b_score))
            b_conf = sum(o.sentiment_confidence for o in in_bucket) / len(in_bucket)
        else:
            b_score = 0.0
            b_conf = 0.0
        buckets.append(
            SentimentBucket(
                bucket_start=cursor,
                bucket_end=b_end,
                article_count=len(in_bucket),
                net_score=b_score,
                mean_confidence=b_conf,
            )
        )
        cursor = b_end

    ranked_sources = sorted(
        filtered,
        key=lambda o: (o.available_time, o.sentiment_confidence, o.canonical_url),
        reverse=True,
    )[:3]
    sources = tuple(
        NewsSourceLink(
            publisher=o.publisher,
            title=o.title,
            canonical_url=o.canonical_url,
            published_time=o.published_time,
            available_time=o.available_time,
            sentiment_label=o.sentiment_label,
            sentiment_confidence=o.sentiment_confidence,
        )
        for o in ranked_sources
    )

    latest_avail = max(o.available_time.astimezone(UTC) for o in filtered)
    has_backfilled = any(o.is_backfilled for o in filtered)
    return SentimentMetrics(
        canonical_symbol=canonical_symbol,
        window_hours=window_hours,
        bucket_hours=bucket_hours,
        model_version=SENTIMENT_MODEL_VERSION,
        is_probabilistic=False,
        article_count=len(filtered),
        positive_count=pos_count,
        negative_count=neg_count,
        neutral_count=neu_count,
        net_sentiment_score=net_score,
        mean_confidence=mean_conf,
        dominant_label=dominant,
        latest_available_time=latest_avail,
        sentiment_backfilled=has_backfilled,
        buckets=tuple(buckets),
        sources=sources,
    )


def _parse_pub_time(raw_date: str | None, fallback: datetime) -> datetime:
    if not raw_date or not raw_date.strip():
        return fallback
    text = raw_date.strip()
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        return dt.astimezone(UTC) if dt.tzinfo else dt.replace(tzinfo=UTC)
    except ValueError:
        pass
    try:
        dt_rfc = parsedate_to_datetime(text)
        return dt_rfc.astimezone(UTC) if dt_rfc.tzinfo else dt_rfc.replace(tzinfo=UTC)
    except (TypeError, ValueError):
        return fallback


def parse_rss_xml(
    xml_text: str,
    *,
    publisher: str,
    source_provider: str = "rss",
    first_seen_time: datetime | None = None,
) -> tuple[RawNewsItem, ...]:
    """Parse static RSS 2.0 or Atom XML safely (disallowing DOCTYPE/ENTITY declarations)."""
    lowered = xml_text.lower()
    if "<!doctype" in lowered or "<!entity" in lowered:
        raise ValueError("RSS/Atom XML cannot contain DOCTYPE or ENTITY declarations")
    root = ET.fromstring(xml_text)  # noqa: S314
    seen_utc = (first_seen_time or datetime.now(UTC)).astimezone(UTC)

    items: list[RawNewsItem] = []
    # RSS 2.0 `<item>` elements
    for elem in root.findall(".//item"):
        title = (elem.findtext("title") or "").strip()
        link = (elem.findtext("link") or "").strip()
        desc = (elem.findtext("description") or "").strip()
        pub_raw = elem.findtext("pubDate")
        if not title or not link:
            continue
        pub_utc = _parse_pub_time(pub_raw, seen_utc)
        items.append(
            RawNewsItem(
                publisher=publisher,
                url=link,
                title=title,
                summary=desc,
                published_time=pub_utc,
                first_seen_time=seen_utc,
                source_provider=source_provider,
            )
        )

    # Atom `<entry>` elements
    atom_ns = "{http://www.w3.org/2005/Atom}"
    for entry in root.findall(f".//{atom_ns}entry"):
        title = (entry.findtext(f"{atom_ns}title") or "").strip()
        link_el = entry.find(f"{atom_ns}link")
        link = (link_el.attrib.get("href") or "").strip() if link_el is not None else ""
        desc = (entry.findtext(f"{atom_ns}summary") or "").strip()
        pub_raw = entry.findtext(f"{atom_ns}updated") or entry.findtext(f"{atom_ns}published")
        if not title or not link:
            continue
        pub_utc = _parse_pub_time(pub_raw, seen_utc)
        items.append(
            RawNewsItem(
                publisher=publisher,
                url=link,
                title=title,
                summary=desc,
                published_time=pub_utc,
                first_seen_time=seen_utc,
                source_provider=source_provider,
            )
        )

    return tuple(items)
