"""Unit tests for PRD v1.7 Wave 4 Task T-17 (Meta-Labeling & ADR 0010)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from crypto_chatbot.analysis.backtest import BacktestConfig
from crypto_chatbot.cli import main as cli_main
from crypto_chatbot.market_data.quality import normalize_ohlcv_rows
from crypto_chatbot.market_data.symbols import resolve_mvp_spot_symbol
from crypto_chatbot.ml.evaluation import PromotionV2Config
from crypto_chatbot.ml.meta_labels import (
    META_MODEL_VERSION,
    SETUP_META_FEATURES,
    MetaLabelDataset,
    MetaLabelEvent,
    align_features_for_direction,
    build_meta_label_dataset,
    run_meta_labeling_experiment,
)
from crypto_chatbot.ml.validation import (
    assert_no_fold_leakage,
    generate_purged_walk_forward_splits,
    split_dev_and_holdout,
)
from crypto_chatbot.schemas.analysis import (
    MarketStructure,
    Regime,
    SetupBias,
    SetupStatus,
    TradeSetup,
)
from crypto_chatbot.schemas.market import MarketCandle, Timeframe
from crypto_chatbot.schemas.ml import CostAssumptions, FeatureRow

FIXTURE_1000 = Path(__file__).resolve().parent / "fixtures" / "parity_1000_candles.json"


def _load_btc_4h_candles() -> tuple[MarketCandle, ...]:
    raw = json.loads(FIXTURE_1000.read_text(encoding="utf-8"))["raw_ohlcv"]
    last_open = datetime.fromtimestamp(float(raw[-1][0]) / 1000.0, tz=UTC)
    rec = last_open + Timeframe.H4.duration + timedelta(seconds=1)
    batch = normalize_ohlcv_rows(
        raw,
        symbol=resolve_mvp_spot_symbol("BTC"),
        timeframe=Timeframe.H4,
        received_time=rec,
        now=rec,
    )
    return batch.candles


def test_t17_direction_aligned_and_setup_features() -> None:
    base = {
        "ret_1": 0.02,
        "cvd_ratio_20": 0.40,
        "taker_buy_ratio": 0.65,
        "atr_pct": 0.015,
    }
    long_setup = TradeSetup(
        status=SetupStatus.READY,
        bias=SetupBias.LONG,
        entry_low=99.0,
        entry_high=101.0,
        entry_mid=100.0,
        entry_source="bullish_fvg",
        stop_loss=95.0,
        tp1=105.0,
        tp2=110.0,
        tp3=115.0,
        rr_tp1=1.0,
        rr_tp2=2.0,
        rr_tp3=3.0,
    )
    short_setup = TradeSetup(
        status=SetupStatus.READY,
        bias=SetupBias.SHORT,
        entry_low=99.0,
        entry_high=101.0,
        entry_mid=100.0,
        entry_source="bearish_ob",
        stop_loss=105.0,
        tp1=95.0,
        tp2=90.0,
        tp3=85.0,
        rr_tp1=1.0,
        rr_tp2=2.0,
        rr_tp3=3.0,
    )

    f_long = align_features_for_direction(
        base,
        bias=SetupBias.LONG,
        setup=long_setup,
        atr_14=2.5,
        regime=Regime.TRENDING_UP,
        market_structure=MarketStructure.BULLISH,
        cost_assumptions=CostAssumptions(),
        funding_8h=0.0002,
        raw_score=6.0,
    )
    f_short = align_features_for_direction(
        base,
        bias=SetupBias.SHORT,
        setup=short_setup,
        atr_14=2.5,
        regime=Regime.TRENDING_DOWN,
        market_structure=MarketStructure.BEARISH,
        cost_assumptions=CostAssumptions(),
        funding_8h=0.0002,
        raw_score=-6.0,
    )

    # Signed features flip sign for SHORT vs LONG
    assert f_long["ret_1"] == pytest.approx(+0.02)
    assert f_short["ret_1"] == pytest.approx(-0.02)
    assert f_long["cvd_ratio_20"] == pytest.approx(+0.40)
    assert f_short["cvd_ratio_20"] == pytest.approx(-0.40)
    assert f_long["taker_buy_ratio"] == pytest.approx(+0.15)
    assert f_short["taker_buy_ratio"] == pytest.approx(-0.15)
    assert f_long["funding_8h_aligned"] == pytest.approx(+0.0002)
    assert f_short["funding_8h_aligned"] == pytest.approx(-0.0002)
    # Unsigned feature stays positive
    assert f_long["atr_pct"] == pytest.approx(0.015)
    assert f_short["atr_pct"] == pytest.approx(0.015)

    for feat_name in SETUP_META_FEATURES:
        assert feat_name in f_long
        assert feat_name in f_short
    assert f_long["entry_src_fvg"] == 1.0
    assert f_short["entry_src_ob"] == 1.0
    assert f_long["structure_aligned"] == 1.0
    assert f_short["structure_aligned"] == 1.0


def test_t17_dataset_excludes_and_counts_no_fill_and_insufficient_data() -> None:
    candles = _load_btc_4h_candles()
    ds = build_meta_label_dataset(
        {"BTC/USDT": candles},
        timeframe=Timeframe.H4,
        config=BacktestConfig(warmup_bars=600, entry_expiry_bars=6, require_ema200=True),
    )

    assert ds.total_ready_setups >= ds.filled_events_count
    no_fill_total = ds.no_fill_tp1_first_count + ds.no_fill_expired_count + ds.no_fill_other_count
    assert ds.filled_events_count + no_fill_total == ds.total_ready_setups
    assert no_fill_total > 0  # parity fixture has NO_FILL_TP1_FIRST / EXPIRED events

    for ev in ds.events:
        assert ev.row.available_time == ev.row.prediction_time
        assert ev.row.label_end_time > ev.row.prediction_time

    # Fewer than 300 events -> INSUFFICIENT_DATA
    res = run_meta_labeling_experiment(ds, min_dev_events=300)
    assert res.status == "INSUFFICIENT_DATA"
    assert res.promoted is False
    assert res.model_version == META_MODEL_VERSION
    assert any("INSUFFICIENT_DATA" in r for r in res.rejection_reasons)


def test_t17_meta_experiment_leakage_control_and_adr_present() -> None:
    # ADR 0010 must exist
    adr_path = Path(__file__).resolve().parents[1] / "docs" / "decisions" / "0010-meta-labeling.md"
    assert adr_path.exists()

    # Build synthetic 460-event chronological dataset with strict non-overlapping exit times
    rng = np.random.default_rng(42)
    t0 = datetime(2025, 1, 1, tzinfo=UTC)
    events: list[MetaLabelEvent] = []
    for i in range(460):
        pred_t = t0 + timedelta(hours=12 * i)
        exit_t = pred_t + timedelta(hours=8)
        sig = float(rng.normal(0.0, 1.0))
        prob = 1.0 / (1.0 + np.exp(-(1.2 * sig)))
        y = 1 if rng.random() < prob else 0
        net_r = 0.6 if y == 1 else -0.5
        row = FeatureRow(
            prediction_time=pred_t,
            available_time=pred_t,
            label_end_time=exit_t,
            regime=Regime.TRENDING_UP,
            features={"ret_1": sig, "sl_pct": 0.02, "exec_cost_r": 0.05},
            forward_return=net_r,
            label=y,
        )
        events.append(
            MetaLabelEvent(
                symbol="BTC/USDT",
                timeframe=Timeframe.H4,
                bias=SetupBias.LONG,
                row=row,
                net_r_1x=net_r,
                net_r_15x=net_r - 0.025,
                exec_cost_r=0.05,
            )
        )

    ds = MetaLabelDataset(
        timeframe=Timeframe.H4,
        symbols=("BTC/USDT",),
        events=tuple(events),
        total_ready_setups=500,
        filled_events_count=460,
        no_fill_tp1_first_count=30,
        no_fill_expired_count=10,
        no_fill_other_count=0,
        dataset_sha256="b" * 64,
    )

    # Verify leakage control with embargo_bars = max(3, 6 + 24) = 30
    rows = tuple(e.row for e in ds.events)
    dev_rows, _hold = split_dev_and_holdout(rows, holdout_fraction=0.20, embargo_bars=30)
    splits = generate_purged_walk_forward_splits(dev_rows, n_splits=4, embargo_bars=30)
    assert_no_fold_leakage(dev_rows, splits, min_embargo_bars=30)

    res = run_meta_labeling_experiment(
        ds,
        min_dev_events=300,
        entry_expiry_bars=6,
        max_hold_bars=24,
        promotion_config=PromotionV2Config(bootstrap_resamples=200),
    )
    assert res.status == "READY"
    assert res.model_version == META_MODEL_VERSION
    assert res.dev_events_count >= 300
    assert res.selected_l2_reg in (0.001, 0.01, 0.1, 1.0)


def test_t17_cli_meta_target_flag() -> None:
    with patch("sys.stdout", new_callable=StringIO) as fake_out:
        rc = cli_main(
            [
                "ml-experiment",
                "BTC",
                "4h",
                "--target",
                "meta",
                "--fixture",
                str(FIXTURE_1000),
                "--json",
            ]
        )
    assert rc == 0
    payload = json.loads(fake_out.getvalue())
    assert payload["status"] == "INSUFFICIENT_DATA"
    assert payload["model_version"] == META_MODEL_VERSION
