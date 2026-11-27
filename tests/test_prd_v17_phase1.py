"""Unit tests for PRD v1.7 Phase P0-C and P1 tasks (T-12 through T-20)."""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from crypto_chatbot.analysis.backtest import (
    BacktestTrade,
    _compute_cost_sensitivity,
    _compute_signed_funding_r,
    _simulate_setup_v17,
)
from crypto_chatbot.analysis.bootstrap import (
    compute_max_concurrent_positions,
    compute_portfolio_and_block_bootstrap,
    stationary_block_bootstrap_ci,
)
from crypto_chatbot.ml.evaluation import (
    PromotionV2Config,
    compute_ece,
    compute_evaluation_summary,
    decide_promotion_v2,
)
from crypto_chatbot.ml.features import (
    ALL_FEATURE_NAMES,
    FEATURE_FAMILIES,
    build_point_in_time_dataset,
    select_available_feature_families,
)
from crypto_chatbot.ml.models import L2LogisticRegression, select_l2_reg_grid
from crypto_chatbot.ml.shadow import _confidence_label, compute_shadow_monitoring_report
from crypto_chatbot.ml.validation import (
    PurgedSplit,
    assert_no_fold_leakage,
    compute_uniqueness_weights,
)
from crypto_chatbot.schemas.analysis import Regime, SetupBias, SetupStatus, TradeSetup
from crypto_chatbot.schemas.market import MarketCandle, MarketType, QualityStatus, Timeframe
from crypto_chatbot.schemas.ml import (
    FEATURE_VERSION,
    CostAssumptions,
    FeatureFamily,
    FeatureRow,
    MlExperimentConfig,
    ShadowPredictionRecord,
)


def _make_candle(
    idx: int,
    *,
    open_: float,
    high: float,
    low: float,
    close: float,
    volume: float = 100.0,
    taker_buy_base_volume: float | None = None,
    start: datetime | None = None,
    timeframe: Timeframe = Timeframe.H1,
) -> MarketCandle:
    base = start or datetime(2026, 1, 1, tzinfo=UTC)
    dur = timeframe.duration
    open_time = base + idx * dur
    close_time = open_time + dur
    return MarketCandle(
        canonical_symbol="BTC/USDT",
        provider="bingx",
        venue="bingx",
        provider_symbol="BTC/USDT:USDT",
        market_type=MarketType.SPOT,
        timeframe=timeframe,
        open_time=open_time,
        close_time=close_time,
        available_time=close_time,
        received_time=close_time,
        open=open_,
        high=high,
        low=low,
        close=close,
        base_volume=volume,
        base_unit="BTC",
        quote_currency="USDT",
        taker_buy_base_volume=taker_buy_base_volume,
        is_closed=True,
    )


def _make_setup(
    *,
    bias: SetupBias = SetupBias.LONG,
    entry_low: float = 99.0,
    entry_high: float = 101.0,
    entry_mid: float = 100.0,
    stop_loss: float = 95.0,
    tp1: float = 105.0,
    tp2: float = 110.0,
    tp3: float = 115.0,
) -> TradeSetup:
    risk = abs(entry_mid - stop_loss)
    return TradeSetup(
        status=SetupStatus.READY,
        bias=bias,
        entry_low=entry_low,
        entry_high=entry_high,
        entry_mid=entry_mid,
        stop_loss=stop_loss,
        tp1=tp1,
        tp2=tp2,
        tp3=tp3,
        rr_tp1=abs(tp1 - entry_mid) / risk,
        rr_tp2=abs(tp2 - entry_mid) / risk,
        rr_tp3=abs(tp3 - entry_mid) / risk,
    )


def _make_trade(
    day_offset: int,
    *,
    hold_hours: int = 4,
    gross_r: float = 1.0,
    exec_cost_r: float = 0.05,
    funding_r: float = 0.0,
) -> BacktestTrade:
    sig = datetime(2026, 1, 1, tzinfo=UTC) + timedelta(days=day_offset)
    ex = sig + timedelta(hours=hold_hours)
    return BacktestTrade(
        signal_time=sig,
        exit_time=ex,
        bias=SetupBias.LONG,
        regime=Regime.TRENDING_UP,
        entry_low=99.0,
        entry_high=101.0,
        entry_mid=100.0,
        stop_loss=95.0,
        tp1=105.0,
        tp2=110.0,
        tp3=115.0,
        outcome="hit_tp2",
        gross_r=gross_r,
        net_r=gross_r - exec_cost_r + funding_r,
        exec_cost_r=exec_cost_r,
        funding_r=funding_r,
        funding_assumed=False,
        be_exit=False,
        bars_held=hold_hours,
    )


# ---------------------------------------------------------------------------
# T-19: Stationary Block Bootstrap & Portfolio Summary
# ---------------------------------------------------------------------------


def test_t19_stationary_block_bootstrap_determinism_and_ar1_wider_ci() -> None:
    rng = np.random.default_rng(123)
    n = 250
    iid = rng.normal(0.1, 1.0, size=n)

    ci1 = stationary_block_bootstrap_ci(iid, resamples=1000, mean_block_length=5.0, seed=42)
    ci2 = stationary_block_bootstrap_ci(iid, resamples=1000, mean_block_length=5.0, seed=42)
    assert ci1 == ci2

    # On IID series, block bootstrap CI width should be close to IID bootstrap CI width
    iid_ci = stationary_block_bootstrap_ci(iid, resamples=1500, mean_block_length=1.0, seed=42)
    blk_ci = stationary_block_bootstrap_ci(iid, resamples=1500, mean_block_length=5.0, seed=42)
    iid_width = iid_ci[1] - iid_ci[0]
    blk_width = blk_ci[1] - blk_ci[0]
    assert abs(blk_width - iid_width) / iid_width < 0.15

    # On AR(1) with rho=0.6, block bootstrap CI must be noticeably wider than IID CI
    ar1 = np.zeros(n, dtype=np.float64)
    eps = rng.normal(0.0, 1.0, size=n)
    for t in range(1, n):
        ar1[t] = 0.6 * ar1[t - 1] + eps[t]
    ar_iid = stationary_block_bootstrap_ci(ar1, resamples=1500, mean_block_length=1.0, seed=42)
    ar_blk = stationary_block_bootstrap_ci(ar1, resamples=1500, mean_block_length=5.0, seed=42)
    assert (ar_blk[1] - ar_blk[0]) > (ar_iid[1] - ar_iid[0])


def test_t19_portfolio_max_concurrent_and_drawdown() -> None:
    t0 = _make_trade(0, hold_hours=10, gross_r=-1.0, exec_cost_r=0.0)
    t1 = _make_trade(0, hold_hours=6, gross_r=-0.5, exec_cost_r=0.0)
    t2 = _make_trade(1, hold_hours=4, gross_r=2.0, exec_cost_r=0.0)
    t3 = _make_trade(2, hold_hours=4, gross_r=-1.0, exec_cost_r=0.0)
    t4 = _make_trade(3, hold_hours=4, gross_r=0.5, exec_cost_r=0.0)
    t5 = _make_trade(4, hold_hours=4, gross_r=0.5, exec_cost_r=0.0)

    trades = [t0, t1, t2, t3, t4, t5]
    assert compute_max_concurrent_positions(trades) == 2

    summary = compute_portfolio_and_block_bootstrap(trades, resamples=300, seed=42)
    assert summary.n_days == 5
    assert summary.max_concurrent_positions == 2
    assert summary.portfolio_max_drawdown_r == pytest.approx(1.5)
    assert summary.block_bootstrap is not None


# ---------------------------------------------------------------------------
# T-18: Pending Setup Lifecycle & Intrabar Order
# ---------------------------------------------------------------------------


def test_t18_tp1_before_fill_cancels_setup() -> None:
    setup = _make_setup()
    future = [
        # Bar 1: touches TP1 (105) without touching entry_high (101)
        _make_candle(1, open_=102.5, high=106.0, low=102.0, close=105.5),
        _make_candle(2, open_=105.5, high=106.0, low=99.0, close=100.0),
    ]
    sim = _simulate_setup_v17(
        setup, future, start_bar_index=0, max_hold_bars=10, entry_expiry_bars=6
    )
    assert sim is not None
    assert sim.cancel_reason == "NO_FILL_TP1_FIRST"
    assert sim.fill_time is None


def test_t18_same_bar_fill_and_tp1_gives_no_tp1_credit() -> None:
    setup = _make_setup()
    future = [
        # Bar 1: touches entry zone [99, 101] AND touches TP1 (105) on same bar
        _make_candle(1, open_=102.0, high=106.0, low=99.5, close=104.0),
        # Bar 2: immediately hits original SL (95) -> proves TP1 was NOT credited & BE not active
        _make_candle(2, open_=104.0, high=104.5, low=94.0, close=94.5),
    ]
    sim = _simulate_setup_v17(
        setup, future, start_bar_index=0, max_hold_bars=10, entry_expiry_bars=6
    )
    assert sim is not None
    assert sim.cancel_reason is None
    assert sim.outcome == "hit_sl"
    assert sim.be_exit is False
    assert sim.gross_r == pytest.approx(-1.2)


def test_t18_tp1_on_bar_k_activates_breakeven_sl_from_bar_k_plus_1() -> None:
    setup = _make_setup()
    future = [
        # Bar 1: fills inside [99, 101] (open=101.5 > 101 => fills at entry_high=101.0)
        _make_candle(1, open_=101.5, high=102.0, low=99.5, close=101.0),
        # Bar 2: hits TP1 (105) cleanly
        _make_candle(2, open_=101.0, high=106.0, low=100.5, close=105.5),
        # Bar 3: drops to 99.5 (below BE fill_price 101.0) -> breakeven SL hit!
        _make_candle(3, open_=105.0, high=105.0, low=99.5, close=99.8),
    ]
    sim = _simulate_setup_v17(
        setup, future, start_bar_index=0, max_hold_bars=10, entry_expiry_bars=6
    )
    assert sim is not None
    assert sim.outcome == "hit_tp1"
    assert sim.be_exit is True
    # Fill at entry_high=101.0, TP1=105.0, risk=5.0 => 40% * (4/5)R = 0.32R; remaining 60% at BE
    assert sim.gross_r == pytest.approx(0.32)


def test_t18_entry_expiry_bars_cancels_as_expired() -> None:
    setup = _make_setup()
    future = [
        _make_candle(i + 1, open_=102.5, high=103.5, low=102.0, close=103.0) for i in range(8)
    ]
    sim = _simulate_setup_v17(
        setup, future, start_bar_index=0, max_hold_bars=10, entry_expiry_bars=3
    )
    assert sim is not None
    assert sim.cancel_reason == "NO_FILL_EXPIRED"


# ---------------------------------------------------------------------------
# T-20: Signed Historical Funding & Cost Sensitivity Sweep
# ---------------------------------------------------------------------------


def test_t20_signed_historical_funding_long_pays_short_receives() -> None:
    fill_t = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    settle_t = datetime(2026, 1, 1, 8, 0, tzinfo=UTC)
    exit_t = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    funding_hist = [(settle_t, 0.0001)]  # +0.01% positive funding
    rem_hist = [(fill_t, 1.0), (settle_t, 1.0), (exit_t, 1.0)]

    long_fund, long_assumed = _compute_signed_funding_r(
        is_long=True,
        fill_time=fill_t,
        exit_time=exit_t,
        fill_price=100.0,
        risk=5.0,
        remaining_history=rem_hist,
        funding_history=funding_hist,
    )
    short_fund, short_assumed = _compute_signed_funding_r(
        is_long=False,
        fill_time=fill_t,
        exit_time=exit_t,
        fill_price=100.0,
        risk=5.0,
        remaining_history=rem_hist,
        funding_history=funding_hist,
    )
    expected_mag = (0.0001 * 100.0) / 5.0
    assert long_assumed is False
    assert short_assumed is False
    assert long_fund == pytest.approx(-expected_mag)
    assert short_fund == pytest.approx(+expected_mag)

    # Missing funding history triggers assumed fallback
    _f, assumed = _compute_signed_funding_r(
        is_long=True,
        fill_time=fill_t,
        exit_time=exit_t,
        fill_price=100.0,
        risk=5.0,
        remaining_history=rem_hist,
        funding_history=None,
    )
    assert assumed is True


def test_t20_cost_sensitivity_and_breakeven_multiplier() -> None:
    trades = [
        _make_trade(0, gross_r=0.30, exec_cost_r=0.10, funding_r=0.0),
        _make_trade(1, gross_r=0.10, exec_cost_r=0.10, funding_r=0.0),
    ]
    # Mean gross_r = 0.20, mean exec_cost_r = 0.10 => break-even multiplier = 2.0
    points, be_mult = _compute_cost_sensitivity(trades)
    assert [p.multiplier for p in points] == [1.0, 1.25, 1.5, 2.0]
    assert be_mult == pytest.approx(2.0)
    assert points[0].net_expectancy_r == pytest.approx(0.10)
    assert points[-1].net_expectancy_r == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# T-13: Base-Rate-Centered Shadow Confidence & Monitoring Alerts
# ---------------------------------------------------------------------------


def test_t13_base_rate_centered_confidence_and_monitoring() -> None:
    # b=0.40, p=0.50 -> edge = +0.10 >= 0.08 -> moderate (NOT abstained)
    assert _confidence_label(0.50, abstained=False, base_rate=0.40) == "moderate"
    assert _confidence_label(0.56, abstained=False, base_rate=0.40) == "high"
    assert _confidence_label(0.45, abstained=False, base_rate=0.40) == "low"

    now = datetime(2026, 2, 1, tzinfo=UTC)
    records = [
        ShadowPredictionRecord(
            prediction_id=f"pred-{i}",
            snapshot_id=f"snap-{i}",
            experiment_id="exp-1",
            canonical_symbol="BTC/USDT",
            timeframe=Timeframe.H4,
            target_name="next_24h_net_positive",
            horizon_hours=24,
            prediction_time=now + timedelta(hours=i),
            model_version="l2-logreg-platt-v1.0.1",
            feature_version=FEATURE_VERSION,
            quality_status=QualityStatus.READY,
            regime="trending_up",
            calibrated_probability=0.90 if i % 2 == 0 else 0.10,
            confidence_label="high",
            abstained=False,
            latency_ms=1,
            realized_label=0 if i % 2 == 0 else 1,  # anti-correlated => BSS < 0
            settled_at=now + timedelta(hours=i + 24),
        )
        for i in range(35)
    ]
    report = compute_shadow_monitoring_report(records, reference_base_rate=0.5)
    assert report.settled_predictions == 35
    assert "bss_below_zero" in report.alerts
    assert "insufficient_settled_samples" not in report.alerts
    # ECE alert requires settled >= 50 by default
    assert "calibration_ece_above_threshold" not in report.alerts


# ---------------------------------------------------------------------------
# T-14 & T-15: Uniqueness Weights, Embargo Check, Mean-Loss Invariance, Grid
# ---------------------------------------------------------------------------


def test_t14_uniqueness_weights_and_embargo_enforcement() -> None:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)

    def _row(start_h: int, end_h: int) -> FeatureRow:
        return FeatureRow(
            prediction_time=t0 + timedelta(hours=start_h),
            available_time=t0 + timedelta(hours=start_h),
            label_end_time=t0 + timedelta(hours=end_h),
            features={"ret_1": 0.01},
            label=1,
            forward_return=0.01,
        )

    # First two rows completely overlap [0, 4]; third is isolated [10, 14]
    rows = [_row(0, 4), _row(0, 4), _row(10, 14)]
    raw_w = compute_uniqueness_weights(rows, normalize=False)
    assert raw_w[0] == pytest.approx(0.5)
    assert raw_w[1] == pytest.approx(0.5)
    assert raw_w[2] == pytest.approx(1.0)

    # Embargo check raises when gap < min_embargo_bars
    split = PurgedSplit(fold_index=0, train_indices=(0,), val_indices=(2,))
    # gap = 2 - 0 - 1 = 1 < 3
    with pytest.raises(ValueError, match="embargo"):
        assert_no_fold_leakage(rows, [split], min_embargo_bars=3)


def test_t15_weighted_mean_loss_duplication_invariance_and_grid_selection() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=(80, 4))
    logits = 0.8 * x[:, 0] - 0.5 * x[:, 1] + 0.2
    y = (rng.random(80) < 1.0 / (1.0 + np.exp(-logits))).astype(np.float64)

    m1 = L2LogisticRegression(l2_reg=0.1).fit(x, y)
    m2 = L2LogisticRegression(l2_reg=0.1).fit(
        np.vstack([x, x]),
        np.concatenate([y, y]),
    )
    assert abs(m1.bias_ - m2.bias_) < 1e-6
    assert np.allclose(m1.weights_, m2.weights_, atol=1e-6)

    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    pred_t = [t0 + timedelta(hours=i) for i in range(80)]
    end_t = [t0 + timedelta(hours=i + 2) for i in range(80)]
    best_lam, scores = select_l2_reg_grid(
        x,
        y,
        feature_names=("f0", "f1", "f2", "f3"),
        pred_times_train=pred_t,
        label_end_times_train=end_t,
    )
    assert best_lam in (0.001, 0.01, 0.1, 1.0)
    assert set(scores.keys()) == {0.001, 0.01, 0.1, 1.0}
    assert all(math.isfinite(v) for v in scores.values())


# ---------------------------------------------------------------------------
# T-16: Feature Set v1.2, Missing Flags, Family Auto-Exclusion
# ---------------------------------------------------------------------------


def test_t16_feature_v12_no_constant_fallbacks_and_family_auto_exclusion() -> None:
    assert FEATURE_VERSION == "ml-feat-v1.2.0"
    assert "cvd_ratio_20" in FEATURE_FAMILIES[FeatureFamily.ORDERFLOW]
    assert "cvd_z_100" in FEATURE_FAMILIES[FeatureFamily.ORDERFLOW]
    assert "cvd_zscore_20" not in FEATURE_FAMILIES[FeatureFamily.ORDERFLOW]
    assert "rvol_missing" in ALL_FEATURE_NAMES
    assert "sentiment_missing" in ALL_FEATURE_NAMES
    assert "orderflow_missing" in FEATURE_FAMILIES[FeatureFamily.ORDERFLOW]

    src_text = (
        Path(__file__).resolve().parents[1] / "src" / "crypto_chatbot" / "ml" / "features.py"
    ).read_text(encoding="utf-8")
    assert "cvd_zscore_20" not in src_text

    # Build candles without taker_buy_base_volume and without sentiment
    candles = [
        _make_candle(
            i,
            open_=100.0 + 0.1 * i,
            high=101.0 + 0.1 * i,
            low=99.0 + 0.1 * i,
            close=100.5 + 0.1 * i,
            volume=100.0 + (i % 7) * 5.0,
            taker_buy_base_volume=None,
            timeframe=Timeframe.H4,
        )
        for i in range(260)
    ]
    cfg = MlExperimentConfig(
        canonical_symbol="BTC/USDT",
        timeframe=Timeframe.H4,
        horizon_hours=24,
    )
    rows, _sha = build_point_in_time_dataset(candles, config=cfg)
    assert len(rows) > 0
    sample = rows[-1].features
    assert sample["orderflow_missing"] == 1.0
    assert sample["cvd_ratio_20"] is None
    assert sample["cvd_z_100"] is None
    assert sample["sentiment_missing"] == 1.0
    assert sample["sentiment_net_score"] is None

    selected, excluded = select_available_feature_families(
        rows,
        (
            FeatureFamily.TECHNICAL,
            FeatureFamily.STRUCTURE_PIVOT,
            FeatureFamily.ORDERFLOW,
            FeatureFamily.SENTIMENT,
        ),
    )
    assert FeatureFamily.ORDERFLOW in excluded
    assert FeatureFamily.SENTIMENT in excluded
    assert "cvd_ratio_20" not in selected
    assert "sentiment_net_score" not in selected


# ---------------------------------------------------------------------------
# T-12: Promotion Gates v2 (G1-G8) & Weighted Evaluation
# ---------------------------------------------------------------------------


def test_t12_weighted_ece_and_promotion_v2_gates() -> None:
    y = np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float64)
    p = np.array([0.8, 0.2, 0.8, 0.2], dtype=np.float64)
    w = np.array([2.0, 1.0, 2.0, 1.0], dtype=np.float64)
    ece = compute_ece(y, p, n_bins=5, sample_weight=w)
    assert 0.0 <= ece <= 1.0

    summary = compute_evaluation_summary(
        y,
        p,
        np.full(4, 0.5),
        np.array([0.02, -0.01, 0.03, -0.02]),
        cost_assumptions=CostAssumptions(),
        prob_threshold=0.55,
        sample_weight=w,
        base_rate=0.50,
        base_rate_delta=0.05,
    )
    assert summary.model_brier < summary.naive_brier

    rng = np.random.default_rng(42)
    n = 120
    oof_y = np.tile([0.0, 1.0], n // 2)
    oof_pn = np.full(n, 0.5)
    oof_pm = np.where(oof_y > 0.5, 0.72, 0.28)
    oof_sel_1x = [0.25 + float(rng.normal(0.0, 0.05)) for _ in range(60)]
    oof_sel_15x = [r - 0.05 for r in oof_sel_1x]
    folds = [(0.1, 0.2), (0.08, 0.15), (0.12, 0.18), (-0.01, -0.05)]
    cfg = PromotionV2Config(bootstrap_resamples=300)

    passed = decide_promotion_v2(
        oof_y_true=oof_y,
        oof_p_model=oof_pm,
        oof_p_naive=oof_pn,
        oof_selected_r_1x=oof_sel_1x,
        oof_selected_r_15x=oof_sel_15x,
        oof_all_setup_expectancy=0.05,
        fold_bss_and_expectancy=folds,
        holdout_selected_trades=35,
        holdout_bss=0.05,
        holdout_selected_expectancy=0.12,
        holdout_weighted_ece=0.06,
        dsr=0.97,
        cpcv_profitable_fraction=0.75,
        config=cfg,
    )
    assert passed.promoted is True
    assert len(passed.gates) == 8
    assert passed.rejection_reasons == ()

    # Failing any single gate must reject
    fail_g3 = decide_promotion_v2(
        oof_y_true=oof_y,
        oof_p_model=oof_pm,
        oof_p_naive=oof_pn,
        oof_selected_r_1x=oof_sel_1x[:10],
        oof_selected_r_15x=oof_sel_15x[:10],
        oof_all_setup_expectancy=0.05,
        fold_bss_and_expectancy=folds,
        holdout_selected_trades=35,
        holdout_bss=0.05,
        holdout_selected_expectancy=0.12,
        holdout_weighted_ece=0.06,
        dsr=0.97,
        cpcv_profitable_fraction=0.75,
        config=cfg,
    )
    assert fail_g3.promoted is False
    assert any("G3" in r for r in fail_g3.rejection_reasons)

    fail_g6 = decide_promotion_v2(
        oof_y_true=oof_y,
        oof_p_model=oof_pm,
        oof_p_naive=oof_pn,
        oof_selected_r_1x=oof_sel_1x,
        oof_selected_r_15x=oof_sel_15x,
        oof_all_setup_expectancy=0.05,
        fold_bss_and_expectancy=folds,
        holdout_selected_trades=35,
        holdout_bss=0.05,
        holdout_selected_expectancy=0.12,
        holdout_weighted_ece=0.06,
        dsr=0.80,
        cpcv_profitable_fraction=0.75,
        config=cfg,
    )
    assert fail_g6.promoted is False
    assert any("G6" in r for r in fail_g6.rejection_reasons)

    fail_g8 = decide_promotion_v2(
        oof_y_true=oof_y,
        oof_p_model=oof_pm,
        oof_p_naive=oof_pn,
        oof_selected_r_1x=oof_sel_1x,
        oof_selected_r_15x=oof_sel_15x,
        oof_all_setup_expectancy=0.05,
        fold_bss_and_expectancy=folds,
        holdout_selected_trades=35,
        holdout_bss=0.05,
        holdout_selected_expectancy=0.12,
        holdout_weighted_ece=0.15,
        dsr=0.97,
        cpcv_profitable_fraction=0.75,
        config=cfg,
    )
    assert fail_g8.promoted is False
    assert any("G8" in r for r in fail_g8.rejection_reasons)
