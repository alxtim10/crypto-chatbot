"""Unit tests for PRD v1.7 Wave 3 tasks (T-21, T-22, T-23, T-24)."""

from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from crypto_chatbot.analysis.backtest import BACKTEST_PROFILE, ParityTag
from crypto_chatbot.analysis.forward_proof import (
    ForwardLedgerEntry,
    evaluate_forward_proof,
    freeze_variant_params,
    load_frozen_params,
)
from crypto_chatbot.cli import main as cli_main
from crypto_chatbot.llm.gemini import GeminiNarrator, RawGeminiOutput
from crypto_chatbot.llm.guard import validate_narration
from crypto_chatbot.news.pipeline import (
    aggregate_sentiment,
    build_sentiment_observations,
    is_sentiment_observation_eligible,
)
from crypto_chatbot.schemas.analysis import (
    AnalysisSnapshot,
    Fact,
    FactStatus,
    InputRange,
    MarketStructure,
    Regime,
    TechnicalMetrics,
)
from crypto_chatbot.schemas.llm import LlmAnalysisNarration, NarrationStatus
from crypto_chatbot.schemas.market import MarketType, QualityStatus, Timeframe
from crypto_chatbot.schemas.news import (
    EntityMatch,
    NormalizedNewsArticle,
    SentimentLabel,
    SentimentObservation,
)


def _make_snapshot(
    *,
    symbol: str = "1000PEPE/USDT",
    facts: tuple[Fact, ...] | None = None,
) -> AnalysisSnapshot:
    now = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
    default_facts = (
        Fact(
            fact_id="F1",
            label="Last close",
            value=0.0123,
            unit="USDT",
            display_format=".4f",
            source="indicators",
            as_of=now,
            status=FactStatus.OK,
        ),
        Fact(
            fact_id="F2",
            label="Unavailable metric",
            value=0.0,
            unit="ratio",
            display_format=".2f",
            source="orderflow",
            as_of=now,
            status=FactStatus.NONE,
        ),
    )
    return AnalysisSnapshot(
        snapshot_id="snap-test-1",
        canonical_symbol=symbol,
        provider="bingx",
        provider_symbol=symbol,
        venue="bingx",
        market_type=MarketType.SPOT,
        timeframe=Timeframe.H4,
        as_of=now,
        calc_version="v1.7.0",
        quality_status=QualityStatus.READY,
        regime=Regime.TRENDING_UP,
        market_structure=MarketStructure.BULLISH,
        input_range=InputRange(
            first_open_time=now - timedelta(hours=400),
            last_open_time=now - timedelta(hours=4),
            candle_count=100,
            sha256="a" * 64,
        ),
        technical=TechnicalMetrics(last_close=0.0123),
        facts=facts if facts is not None else default_facts,
    )


def _make_narration(
    *,
    summary: str = "Harga 1000PEPE pada timeframe 4h berada di {F1}.",
    fact_ids: tuple[str, ...] = ("F1",),
) -> LlmAnalysisNarration:
    return LlmAnalysisNarration(
        summary=summary,
        supporting_factors=("Momentum mendukung di {F1}.",),
        contradicting_factors=("Volatilitas tetap perlu diperhatikan.",),
        bullish_scenario="Bertahan di atas level referensi {F1}.",
        bearish_scenario="Gagal bertahan di bawah {F1}.",
        limitations=("Analisis berbasis candle tutup.",),
        fact_ids=fact_ids,
    )


class _StubGeminiTransport:
    def __init__(self, payloads: list[str]) -> None:
        self._payloads = list(payloads)
        self.calls: list[str] = []

    async def generate_json(
        self,
        *,
        model: str,
        system_instruction: str,
        contents: str,
        response_json_schema: dict[str, Any],
        max_output_tokens: int,
        temperature: float,
    ) -> RawGeminiOutput:
        self.calls.append(contents)
        idx = min(len(self.calls) - 1, len(self._payloads) - 1)
        return RawGeminiOutput(text=self._payloads[idx], model_version=model)


# ---------------------------------------------------------------------------
# T-24: Narration Output Guard & Retry-then-Fallback Pipeline
# ---------------------------------------------------------------------------


def test_t24_guard_rejects_raw_digits_unknown_and_none_facts_and_allows_1000pepe_4h() -> None:
    snap = _make_snapshot(symbol="1000PEPE/USDT")

    # 1000PEPE and 4h pass the guard cleanly
    ok = validate_narration(_make_narration(), snap)
    assert ok.is_valid is True
    assert ok.reasons == ()

    # Raw digit rejected
    raw_num = validate_narration(
        _make_narration(summary="Harga 1000PEPE naik ke 99500 pada 4h."),
        snap,
    )
    assert raw_num.is_valid is False
    assert "raw_digit_detected" in raw_num.reasons

    # Unknown {F99} rejected
    unknown_f = validate_narration(
        _make_narration(summary="Level referensi {F99} terlihat kuat.", fact_ids=("F1", "F99")),
        snap,
    )
    assert unknown_f.is_valid is False
    assert "unknown_fact_id:F99" in unknown_f.reasons
    assert "malformed_or_unknown_placeholder" in unknown_f.reasons

    # Fact with status NONE rejected
    none_f = validate_narration(
        _make_narration(summary="Orderflow menunjukkan {F2}.", fact_ids=("F1", "F2")),
        snap,
    )
    assert none_f.is_valid is False
    assert "fact_status_none:F2" in none_f.reasons


def test_t24_narrator_retries_once_and_falls_back_after_two_violations() -> None:
    snap = _make_snapshot(symbol="1000PEPE/USDT")
    bad_json = _make_narration(summary="Harga naik ke 12345 sekarang.").model_dump_json()
    good_json = _make_narration().model_dump_json()

    # 1st attempt violates, 2nd attempt succeeds -> GENERATED
    t_recover = _StubGeminiTransport([bad_json, good_json])
    narrator_recover = GeminiNarrator(model="gemini-2.5-flash", transport=t_recover)
    outcome_ok = asyncio.run(narrator_recover.narrate_snapshot(snap))
    assert outcome_ok.validation_passed is True
    assert outcome_ok.response.narration_status is NarrationStatus.GENERATED
    assert len(t_recover.calls) == 2
    assert "[GUARD_RETRY_FEEDBACK]" in t_recover.calls[1]

    # Both attempts violate -> deterministic template fallback
    t_fail = _StubGeminiTransport([bad_json, bad_json])
    narrator_fail = GeminiNarrator(model="gemini-2.5-flash", transport=t_fail)
    outcome_fb = asyncio.run(narrator_fail.narrate_snapshot(snap))
    assert outcome_fb.validation_passed is False
    assert outcome_fb.response.narration_status is NarrationStatus.FALLBACK_DETERMINISTIC
    assert outcome_fb.response.narration is None
    assert len(t_fail.calls) == 2


# ---------------------------------------------------------------------------
# T-23: Sentiment Point-in-Time Eligibility & Block LLM Backfill
# ---------------------------------------------------------------------------


def test_t23_sentiment_scored_at_leakage_guard_and_llm_backfill_block() -> None:
    t0 = datetime(2026, 2, 1, 10, 0, tzinfo=UTC)
    pred_time = datetime(2026, 2, 1, 12, 0, tzinfo=UTC)
    future_scored = datetime(2026, 2, 1, 16, 0, tzinfo=UTC)

    # LLM observation scored AFTER prediction_time must NOT be visible at pred_time
    llm_future_obs = SentimentObservation(
        observation_id="obs-" + "1" * 20,
        article_id="art-" + "1" * 20,
        canonical_url="https://example.com/btc-1",
        publisher="Example",
        title="Bitcoin rally continues",
        canonical_symbol="BTC/USDT",
        base_asset="BTC",
        entity_confidence=0.95,
        sentiment_label=SentimentLabel.POSITIVE,
        sentiment_score=0.8,
        sentiment_confidence=0.9,
        published_time=t0,
        available_time=t0,
        scored_at=future_scored,
        scorer_id="gemini-sentiment",
        scorer_version="gemini-2.5-flash",
        scorer_class="llm",
        is_backfilled=False,
    )
    assert is_sentiment_observation_eligible(llm_future_obs, pred_time) is False
    agg = aggregate_sentiment([llm_future_obs], canonical_symbol="BTC/USDT", as_of=pred_time)
    assert agg is None

    # Lexicon deterministic backfill IS eligible and sets sentiment_backfilled=True
    lex_backfill = llm_future_obs.model_copy(
        update={
            "observation_id": "obs-" + "2" * 20,
            "scorer_id": "crypto-lexicon",
            "scorer_class": "lexicon_deterministic",
            "is_backfilled": True,
        }
    )
    assert is_sentiment_observation_eligible(lex_backfill, pred_time) is True
    agg_lex = aggregate_sentiment([lex_backfill], canonical_symbol="BTC/USDT", as_of=pred_time)
    assert agg_lex is not None
    assert agg_lex.sentiment_backfilled is True

    # Attempting to backfill with an LLM scorer must raise ValueError
    article = NormalizedNewsArticle(
        article_id="art-" + "3" * 20,
        canonical_url="https://example.com/btc-3",
        content_fingerprint="f" * 32,
        publisher="Example",
        title="Bitcoin surges on adoption",
        published_time=t0,
        first_seen_time=t0,
        available_time=t0,
        source_provider="rss",
        affected_entities=(
            EntityMatch(
                canonical_symbol="BTC/USDT",
                base_asset="BTC",
                confidence=0.95,
                match_reason="explicit_name",
            ),
        ),
    )
    with pytest.raises(ValueError, match="LLM-based sentiment"):
        build_sentiment_observations(
            [article],
            scored_at=future_scored,
            scorer_class="llm",
            is_backfilled=True,
        )


# ---------------------------------------------------------------------------
# T-22: Parameter Freeze & Forward Evaluation
# ---------------------------------------------------------------------------


def test_t22_freeze_params_cli_and_forward_evaluator(tmp_path: Path) -> None:
    out_file = tmp_path / "frozen_params.json"
    rc = cli_main(["freeze-params", "v3", "--output", str(out_file)])
    assert rc == 0
    assert out_file.exists()

    loaded = load_frozen_params(out_file)
    assert loaded.variant == "v3"
    assert len(loaded.params_sha256) == 64

    frozen_at = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    frozen = freeze_variant_params("v3", frozen_at=frozen_at)

    # Mix of mismatched hash, pre-freeze, and insufficient valid entries -> NOT_MET
    mixed_entries = [
        ForwardLedgerEntry(
            setup_id="bad-hash",
            timeframe=Timeframe.H4,
            emitted_at=frozen_at + timedelta(days=2),
            settled_at=frozen_at + timedelta(days=3),
            net_r_1x=1.0,
            exec_cost_r=0.05,
            params_sha256="0" * 64,
        ),
        ForwardLedgerEntry(
            setup_id="pre-freeze",
            timeframe=Timeframe.H4,
            emitted_at=frozen_at - timedelta(days=1),
            settled_at=frozen_at + timedelta(days=1),
            net_r_1x=1.0,
            exec_cost_r=0.05,
            params_sha256=frozen.params_sha256,
        ),
        ForwardLedgerEntry(
            setup_id="valid-1",
            timeframe=Timeframe.H4,
            emitted_at=frozen_at + timedelta(days=5),
            settled_at=frozen_at + timedelta(days=6),
            net_r_1x=0.8,
            exec_cost_r=0.05,
            params_sha256=frozen.params_sha256,
        ),
    ]
    eval_not_met = evaluate_forward_proof(
        mixed_entries,
        frozen=frozen,
        timeframe=Timeframe.H4,
        bootstrap_resamples=200,
    )
    assert eval_not_met.status == "NOT_MET"
    assert eval_not_met.excluded_hash_mismatch == 1
    assert eval_not_met.excluded_before_freeze == 1
    assert eval_not_met.eligible_settled_filled == 1
    assert len(eval_not_met.reasons) >= 1

    # 110 positive entries spanning 70 days -> MET
    passing_entries = [
        ForwardLedgerEntry(
            setup_id=f"ok-{i}",
            timeframe=Timeframe.H4,
            emitted_at=frozen_at + timedelta(hours=12 * i + 1),
            settled_at=frozen_at + timedelta(hours=12 * i + 16),
            net_r_1x=0.35 + (0.05 if i % 2 == 0 else -0.05),
            exec_cost_r=0.05,
            params_sha256=frozen.params_sha256,
            forward_only_guards_fired=("l2_orderbook_imbalance_guard",) if i % 5 == 0 else (),
        )
        for i in range(130)
    ]
    eval_met = evaluate_forward_proof(
        passing_entries,
        frozen=frozen,
        timeframe=Timeframe.H4,
        bootstrap_resamples=300,
    )
    assert eval_met.status == "MET"
    assert eval_met.reasons == ()
    assert eval_met.with_forward_only_expectancy_r is not None
    assert eval_met.without_forward_only_expectancy_r is not None


# ---------------------------------------------------------------------------
# T-21: Parity Profile, Benchmark Report & Banned Claims Grep
# ---------------------------------------------------------------------------


def test_t21_parity_profile_and_banned_words_discipline() -> None:
    assert BACKTEST_PROFILE["ema200_gate"] is ParityTag.BACKTESTED
    assert BACKTEST_PROFILE["l2_orderbook_imbalance_guard"] is ParityTag.FORWARD_ONLY
    assert BACKTEST_PROFILE["dxy_macro_context"] is ParityTag.CONTEXT_ONLY

    repo_root = Path(__file__).resolve().parents[1]
    benchmark_doc = repo_root / "docs" / "research" / "benchmark-v1.7.md"
    assert benchmark_doc.exists()
    bench_text = benchmark_doc.read_text(encoding="utf-8")
    assert "trial_id" in bench_text or "trial-" in bench_text

    # No banned words in updated quant PRD, benchmark report, or user templates
    banned_re = re.compile(r"\b(?:deployed\s+winner|verified|proof)\b", re.IGNORECASE)
    for rel_path in (
        "docs/PRD_QUANT_ML_BACKTEST.md",
        "docs/research/benchmark-v1.7.md",
        "src/crypto_chatbot/bot/templates.py",
    ):
        text = (repo_root / rel_path).read_text(encoding="utf-8")
        assert banned_re.search(text) is None, f"Banned claim word found in {rel_path}"
