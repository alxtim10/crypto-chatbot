# ADR 0005: Traceable News Ingestion & Non-Probabilistic Sentiment Pipeline

- Status: Accepted
- Date: 2026-11-22
- Supersedes: none (extends ADR 0001, ADR 0002, ADR 0003, and ADR 0004)

## Context

Phase 11 adds a point-in-time, auditable news and sentiment pipeline (`PROJECT_PLAN.md` §6 Phase 11) so macro/coin headlines can serve as a non-probabilistic supporting variable for research context and setup risk notes before Phase 12 offline ML experiments.

## Decisions

| Area | Decision | Rationale |
|---|---|---|
| Pipeline architecture | Implement deterministic multi-stage pipeline in `crypto_chatbot.news.pipeline`: URL & UTC timestamp canonicalization -> URL, fingerprint, and syndication near-duplicate deduplication -> rule-based coin/entity matching (`MIN_ENTITY_MATCH_CONFIDENCE = 0.65`) -> crypto-lexicon sentiment classification with negation handling (`SENTIMENT_MODEL_VERSION = "crypto-lexicon-v1.0.0"`) -> causal time-bucket aggregation (`available_time <= as_of`). | Reproducible, testable offline without external APIs or new ML dependencies in `pyproject.toml`. |
| Causality & deduplication | Set `available_time = max(published_time, first_seen_time)`. Deduplicate by canonicalized URL (tracking parameters stripped), normalized headline SHA-256 fingerprint, and token-set Jaccard similarity (`>= 0.80`) within a 24h syndication window. Queries filter strictly on `available_time <= as_of`. | Prevents syndicated reprints from inflating counts and eliminates future news leakage. |
| Entity precision | Require explicit ticker (`$BTC`, `BTC/USDT`), unambiguous uppercase symbol, or full asset name above `0.65` confidence; reject ambiguous common words (e.g., lowercase `sol`, `link`, `near`, `op`). | Prevents unrelated general news from polluting coin sentiment features. |
| Separation from ML probability | Both `SentimentObservation` and `SentimentMetrics` carry `is_probabilistic: Literal[False] = False` and explicit `model_version`. Sentiment only enriches risk notes and supporting factors (`news_sentiment_24h`), never inventing probabilities or overriding deterministic Entry/SL/TP prices. | Keeps rule-based lexical scores strictly separate from Phase 12 calibrated probabilities. |
| Prompt-injection & links | Raw article titles/summaries never enter `AnalysisSnapshot` trusted JSON facts for Gemini; article headlines passed for context are fenced inside `<untrusted_data>...</untrusted_data>` with tag breakout filtering. Source URLs (`http://`/`https://` only) are rendered deterministically with HTML escaping in backend templates (`render_deterministic_snapshot`, `render_compact_snapshot`), while Gemini's output guard continues rejecting LLM-emitted URLs. | Prevents embedded instructions in untrusted news copy from hijacking Gemini while guaranteeing source attribution links. |

## Consequences

- Normalized news articles and per-entity sentiment observations persist in PostgreSQL tables `news_articles` and `sentiment_observations` with unique constraints preventing duplicate writes.
- Existing `AnalysisSnapshot` hashes and Fact IDs remain identical when no news observations are attached; when attached, sentiment facts are appended after TradingView facts.
