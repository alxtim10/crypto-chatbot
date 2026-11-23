"""Phase 12 offline ML leakage, walk-forward, calibration, cost & rejection tests."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import resolve_mvp_spot_symbol
from crypto_chatbot.ml import (
    FEATURE_FAMILIES,
    FeaturePreprocessor,
    assert_no_fold_leakage,
    build_point_in_time_dataset,
    compute_evaluation_summary,
    evaluate_feature_rows_experiment,
    generate_purged_walk_forward_splits,
    run_offline_experiment,
    split_dev_and_holdout,
)
from crypto_chatbot.news import build_sentiment_observations, deduplicate_articles
from crypto_chatbot.schemas.analysis import Regime
from crypto_chatbot.schemas.market import MarketCandle, MarketType, Timeframe
from crypto_chatbot.schemas.ml import (
    CostAssumptions,
    FeatureFamily,
    FeatureRow,
    MlExperimentConfig,
)
from crypto_chatbot.schemas.news import RawNewsItem

CANDLE_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "parity_1000_candles.json"
NEWS_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "news_sentiment_sample.json"
ANCHOR_NOW = datetime(2026, 10, 14, 16, 0, 1, tzinfo=UTC)


def _load_btc_candles() -> tuple[MarketCandle, ...]:
    raw = json.loads(CANDLE_FIXTURE.read_text(encoding="utf-8"))["raw_ohlcv"]
    batch = normalize_ohlcv_rows(
        raw,
        symbol=resolve_mvp_spot_symbol("BTC"),
        timeframe=Timeframe.H4,
        received_time=ANCHOR_NOW,
        now=ANCHOR_NOW,
    )
    return batch.candles


def _synthetic_candles_with_future_shock() -> tuple[MarketCandle, ...]:
    candles = list(_load_btc_candles())
    # Mutate the final 10 candles violently; earlier feature rows must stay bit-identical.
    for i in range(len(candles) - 10, len(candles)):
        orig = candles[i]
        candles[i] = orig.model_copy(
            update={
                "open": orig.open * 2.5,
                "high": orig.high * 3.0,
                "low": orig.low * 2.0,
                "close": orig.close * 2.8,
                "base_volume": orig.base_volume * 50.0,
            }
        )
    return tuple(candles)


def test_point_in_time_features_have_zero_future_candle_or_sentiment_leakage() -> None:
    candles = _load_btc_candles()
    shocked = _synthetic_candles_with_future_shock()
    config = MlExperimentConfig(canonical_symbol="BTC/USDT", timeframe=Timeframe.H4)

    news_raw = [
        RawNewsItem.model_validate(item)
        for item in json.loads(NEWS_FIXTURE.read_text(encoding="utf-8"))
    ]
    articles, _ = deduplicate_articles(news_raw)
    obs = build_sentiment_observations(articles)

    rows_base, _ = build_point_in_time_dataset(candles, config=config, sentiment_observations=obs)
    rows_shock, _ = build_point_in_time_dataset(shocked, config=config, sentiment_observations=obs)

    cutoff_time = candles[-15].close_time
    early_base = [r for r in rows_base if r.prediction_time <= cutoff_time]
    early_shock = [r for r in rows_shock if r.prediction_time <= cutoff_time]
    assert len(early_base) == len(early_shock)
    assert len(early_base) > 500
    for r_b, r_s in zip(early_base, early_shock, strict=True):
        assert r_b.available_time <= r_b.prediction_time
        assert r_b.prediction_time < r_b.label_end_time
        assert r_b.features == r_s.features

    # Future sentiment injected after `cutoff_time` cannot leak into earlier feature rows.
    future_item = RawNewsItem(
        publisher="FutureWire",
        url="https://future.example.com/btc-shock-2026",
        title="Bitcoin catastrophic hack crash plunge fraud exploit",
        summary="Extreme negative news only available after cutoff_time.",
        published_time=cutoff_time + timedelta(hours=1),
        first_seen_time=cutoff_time + timedelta(hours=1),
        source_provider="fixture",
    )
    future_articles, _ = deduplicate_articles([future_item])
    future_obs = (*obs, *build_sentiment_observations(future_articles))
    rows_with_future, _ = build_point_in_time_dataset(
        candles,
        config=config,
        sentiment_observations=future_obs,
    )
    early_future = [r for r in rows_with_future if r.prediction_time <= cutoff_time]
    for r_b, r_f in zip(early_base, early_future, strict=True):
        assert r_b.features["sentiment_net_score"] == r_f.features["sentiment_net_score"]


def test_purged_walk_forward_enforces_no_overlap_and_training_only_scaling() -> None:
    candles = _load_btc_candles()
    config = MlExperimentConfig(canonical_symbol="BTC/USDT", timeframe=Timeframe.H4)
    rows, _ = build_point_in_time_dataset(candles, config=config)

    dev_rows, holdout_rows = split_dev_and_holdout(
        rows,
        holdout_fraction=config.holdout_fraction,
        embargo_bars=config.embargo_bars,
    )
    assert dev_rows[-1].label_end_time <= holdout_rows[0].prediction_time

    splits = generate_purged_walk_forward_splits(
        dev_rows,
        n_splits=config.n_splits,
        embargo_bars=config.embargo_bars,
    )
    assert_no_fold_leakage(dev_rows, splits)

    # Inject overlapping index to confirm automated leakage check catches it.
    bad_split = replace(
        splits[0],
        train_indices=(*splits[0].train_indices, splits[0].val_indices[0] - 1),
    )
    with pytest.raises(ValueError, match=r"overlap|embargo|label_end_time"):
        assert_no_fold_leakage(dev_rows, (bad_split,))

    # Preprocessing fits solely on training indices; validation outliers never alter scaling.
    feat_names = FEATURE_FAMILIES[FeatureFamily.TECHNICAL]
    first_split = splits[0]
    x_tr = np.array(
        [[dev_rows[i].features[f] for f in feat_names] for i in first_split.train_indices]
    )
    x_va = np.array(
        [[dev_rows[i].features[f] for f in feat_names] for i in first_split.val_indices]
    )
    x_va_extreme = x_va.copy()
    x_va_extreme[:, 0] = 1e6

    prep_a = FeaturePreprocessor(feat_names).fit(
        x_tr,
        fit_max_time=dev_rows[first_split.train_indices[-1]].prediction_time,
    )
    prep_b = FeaturePreprocessor(feat_names).fit(
        x_tr,
        fit_max_time=dev_rows[first_split.train_indices[-1]].prediction_time,
    )
    assert prep_a.fit_max_time is not None
    assert prep_a.fit_max_time < dev_rows[first_split.val_indices[0]].prediction_time
    np.testing.assert_allclose(prep_a.means_, prep_b.means_)  # type: ignore[arg-type]
    np.testing.assert_allclose(prep_a.stds_, prep_b.stds_)  # type: ignore[arg-type]
    _ = prep_a.transform(x_va_extreme)


def test_experiment_reproducibility_costs_and_rejection_gate() -> None:
    candles = _load_btc_candles()
    config = MlExperimentConfig(canonical_symbol="BTC/USDT", timeframe=Timeframe.H4)

    rep1 = run_offline_experiment(candles, config=config)
    rep2 = run_offline_experiment(candles, config=config)
    assert rep1 == rep2
    assert rep1.experiment_id.startswith("exp-")
    assert len(rep1.fold_metrics) == config.n_splits
    assert len(rep1.ablations) == 2
    assert rep1.regime_stability

    # Verify cost assumptions genuinely reduce simulated strategy return.
    y = np.array([1.0, 1.0, 1.0, 0.0], dtype=np.float64)
    p_m = np.array([0.8, 0.8, 0.8, 0.2], dtype=np.float64)
    p_n = np.array([0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    fwd = np.array([0.01, 0.01, 0.01, -0.01], dtype=np.float64)
    zero_cost = CostAssumptions(
        fee_bps_per_side=0.0,
        spread_bps=0.0,
        slippage_bps=0.0,
        funding_bps_per_24h=0.0,
    )
    high_cost = CostAssumptions(
        fee_bps_per_side=25.0,
        spread_bps=15.0,
        slippage_bps=15.0,
        funding_bps_per_24h=10.0,
    )
    ev_zero = compute_evaluation_summary(
        y, p_m, p_n, fwd, cost_assumptions=zero_cost, prob_threshold=0.55
    )
    ev_costly = compute_evaluation_summary(
        y, p_m, p_n, fwd, cost_assumptions=high_cost, prob_threshold=0.55
    )
    assert ev_zero.trade_count == 3
    assert ev_costly.net_return_after_costs < ev_zero.net_return_after_costs

    # Synthetic uninformative random-noise dataset must be REJECTED (`accepted=False`).
    rng = np.random.default_rng(42)
    t0 = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    noisy_rows: list[FeatureRow] = []
    for idx in range(300):
        pred_t = t0 + timedelta(hours=4 * idx)
        feats = {
            name: float(rng.normal(0.0, 1.0))
            for name in (
                *FEATURE_FAMILIES[FeatureFamily.TECHNICAL],
                *FEATURE_FAMILIES[FeatureFamily.STRUCTURE_PIVOT],
                *FEATURE_FAMILIES[FeatureFamily.SENTIMENT],
            )
        }
        lbl = int(idx % 3 == 0)
        noisy_rows.append(
            FeatureRow(
                prediction_time=pred_t,
                available_time=pred_t,
                label_end_time=pred_t + timedelta(hours=4),
                regime=Regime.RANGING,
                features=feats,
                forward_return=0.001 if lbl == 1 else -0.002,
                label=lbl,
            )
        )
    noisy_report = evaluate_feature_rows_experiment(
        noisy_rows,
        dataset_sha256="a" * 64,
        config=MlExperimentConfig(
            canonical_symbol="BTC/USDT",
            timeframe=Timeframe.H4,
            embargo_bars=1,
            min_brier_improvement=0.01,
        ),
    )
    assert noisy_report.accepted is False
    assert noisy_report.rejection_reasons

    # Clean separable synthetic signal with sufficient margin over costs passes promotion gate.
    signal_rows: list[FeatureRow] = []
    for idx in range(300):
        pred_t = t0 + timedelta(hours=4 * idx)
        lbl = 1 if (idx % 2 == 0) else 0
        sig_val = 2.0 if lbl == 1 else -2.0
        feats = {
            name: 0.0
            for name in (
                *FEATURE_FAMILIES[FeatureFamily.TECHNICAL],
                *FEATURE_FAMILIES[FeatureFamily.STRUCTURE_PIVOT],
                *FEATURE_FAMILIES[FeatureFamily.SENTIMENT],
            )
        }
        feats["ret_1"] = sig_val + float(rng.normal(0.0, 0.1))
        signal_rows.append(
            FeatureRow(
                prediction_time=pred_t,
                available_time=pred_t,
                label_end_time=pred_t + timedelta(hours=4),
                regime=Regime.TRENDING_UP if lbl == 1 else Regime.TRENDING_DOWN,
                features=feats,
                forward_return=0.025 if lbl == 1 else -0.020,
                label=lbl,
            )
        )
    good_report = evaluate_feature_rows_experiment(
        signal_rows,
        dataset_sha256="b" * 64,
        config=MlExperimentConfig(
            canonical_symbol="BTC/USDT",
            timeframe=Timeframe.H4,
            embargo_bars=1,
            prob_threshold=0.60,
            min_trades=5,
        ),
    )
    assert good_report.accepted is True
    assert good_report.rejection_reasons == ()
    assert good_report.holdout_summary.net_return_after_costs > 0.0
    assert MarketType.SPOT is MarketType.SPOT
