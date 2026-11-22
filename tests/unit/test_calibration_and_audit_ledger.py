"""Unit tests for Track 2 quantitative calibration audit, maker/taker split, and ledger CLI."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from io import StringIO
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from crypto_chatbot.cli import main
from crypto_chatbot.ml.evaluation import (
    compute_brier_skill_score,
    compute_calibration_bins,
    compute_evaluation_summary,
)
from crypto_chatbot.schemas.market import QualityStatus, Timeframe
from crypto_chatbot.schemas.ml import (
    CostAssumptions,
    ShadowPredictionRecord,
)
from crypto_chatbot.storage.repositories import TradeSetupRecord


def test_brier_skill_score_math() -> None:
    # Typical skill > 0
    assert pytest.approx(compute_brier_skill_score(0.20, 0.25), abs=1e-5) == 0.20
    # No skill = 0
    assert pytest.approx(compute_brier_skill_score(0.25, 0.25), abs=1e-5) == 0.0
    # Negative skill < 0
    assert pytest.approx(compute_brier_skill_score(0.30, 0.25), abs=1e-5) == -0.20

    # Edge cases when climatology has zero variance (all positive or all negative)
    assert compute_brier_skill_score(0.0, 0.0) == 0.0
    assert compute_brier_skill_score(0.05, 0.0) == -1.0


def test_calibration_bins_partition_and_metrics() -> None:
    y_true = np.array([0, 0, 1, 1, 1, 0, 1, 1, 0, 1], dtype=np.float64)
    p_pred = np.array([0.1, 0.15, 0.35, 0.45, 0.65, 0.70, 0.85, 0.90, 0.92, 0.95], dtype=np.float64)
    net_r = np.array(
        [0.01, -0.02, 0.03, 0.04, 0.02, -0.01, 0.05, 0.06, -0.03, 0.07], dtype=np.float64
    )

    bins = compute_calibration_bins(y_true, p_pred, net_returns=net_r, n_bins=5)
    assert len(bins) == 5

    # Sum of counts across all bins equals sample count
    total_count = sum(b.sample_count for b in bins)
    assert total_count == len(y_true)

    # Bin ranges are contiguous [0.0, 0.2], [0.2, 0.4], ..., [0.8, 1.0]
    assert bins[0].bin_low == 0.0
    assert bins[-1].bin_high == 1.0
    for b in bins:
        if b.sample_count > 0:
            assert b.mean_predicted_prob is not None
            assert b.empirical_positive_rate is not None
            assert b.calibration_gap is not None
            assert pytest.approx(b.calibration_gap, abs=1e-5) == abs(
                b.empirical_positive_rate - b.mean_predicted_prob
            )


def test_maker_vs_taker_cost_monotonicity() -> None:
    taker_costs = CostAssumptions(
        fee_bps_per_side=10.0,
        spread_bps=5.0,
        slippage_bps=5.0,
        execution_style="taker",
    )
    maker_costs = CostAssumptions(
        fee_bps_per_side=10.0,
        maker_fee_bps_per_side=2.0,
        spread_bps=5.0,
        slippage_bps=5.0,
        execution_style="maker",
    )

    # Maker roundtrip friction must be strictly smaller than taker friction
    assert maker_costs.round_trip_cost_bps < taker_costs.round_trip_cost_bps

    # Simulate evaluation summary with trades
    y_true = np.array([1, 1, 1, 0, 1] * 10, dtype=np.float64)
    p_model = np.array([0.7, 0.8, 0.65, 0.55, 0.75] * 10, dtype=np.float64)
    p_naive = np.array([0.5] * len(y_true), dtype=np.float64)
    fwd_ret = np.array([0.02, 0.03, 0.01, -0.01, 0.02] * 10, dtype=np.float64)

    summary = compute_evaluation_summary(
        y_true,
        p_model,
        p_naive,
        fwd_ret,
        cost_assumptions=taker_costs,
        prob_threshold=0.60,
    )

    assert summary.maker_net_return_after_costs is not None
    assert summary.taker_net_return_after_costs is not None
    # Maker net return must be strictly greater than taker net return
    assert summary.maker_net_return_after_costs > summary.taker_net_return_after_costs


def test_cli_setup_stats_json_output() -> None:
    ref_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    mock_setups = (
        TradeSetupRecord(
            snapshot_id="snap-1",
            venue="binance",
            canonical_symbol="BTC/USDT",
            timeframe=Timeframe.H4,
            as_of=ref_time,
            status="active",
            bias="bullish",
            entry_source="h4_sweep",
            wait_reason=None,
            reference_price=50000.0,
            entry_low=49500.0,
            entry_high=50000.0,
            stop_loss=48900.0,
            tp1=51500.0,
            tp2=53000.0,
            tp3=55000.0,
            rr_tp1=1.5,
            rr_tp2=3.0,
            rr_tp3=5.0,
            outcome_status="tp1",
            settled_at=ref_time,
            realized_r=1.5,
        ),
        TradeSetupRecord(
            snapshot_id="snap-2",
            venue="binance",
            canonical_symbol="BTC/USDT",
            timeframe=Timeframe.H4,
            as_of=ref_time,
            status="active",
            bias="bearish",
            entry_source="h4_bos",
            wait_reason=None,
            reference_price=50000.0,
            entry_low=50000.0,
            entry_high=50500.0,
            stop_loss=51200.0,
            tp1=48500.0,
            tp2=47000.0,
            tp3=45000.0,
            rr_tp1=1.5,
            rr_tp2=3.0,
            rr_tp3=5.0,
            outcome_status="sl",
            settled_at=ref_time,
            realized_r=-1.0,
        ),
    )

    mock_repo = MagicMock()
    mock_repo.list_recent_setups.return_value = mock_setups

    with (
        patch("crypto_chatbot.cli._resolve_database_url", return_value="postgresql://dummy"),
        patch("crypto_chatbot.cli.create_db_engine"),
        patch("crypto_chatbot.cli.init_db"),
        patch("crypto_chatbot.cli.PostgresSetupLogRepository", return_value=mock_repo),
        patch("sys.stdout", new_callable=StringIO) as fake_out,
    ):
        code = main(["setup-stats", "--symbol", "BTC", "--timeframe", "4h", "--json"])
        assert code == 0
        parsed = json.loads(fake_out.getvalue())
        assert parsed["total_setups"] == 2
        assert parsed["status_counts"]["tp1"] == 1
        assert parsed["status_counts"]["sl"] == 1
        assert parsed["hit_rate_tp_vs_sl"] == 0.50
        assert parsed["mean_realized_r"] == 0.25


def test_cli_shadow_report_json_output() -> None:
    ref_time = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    mock_records = tuple(
        ShadowPredictionRecord(
            prediction_id=f"pred-{i}",
            snapshot_id=f"snap-{i}",
            experiment_id="exp-test",
            canonical_symbol="BTC/USDT",
            timeframe=Timeframe.H4,
            target_name="next_24h_return_exceeds_cost",
            horizon_hours=24,
            prediction_time=ref_time,
            model_version="l2-logreg-platt-v1.0.0",
            feature_version="ml-feat-v1.1.0",
            quality_status=QualityStatus.READY,
            calibrated_probability=0.70,
            confidence_label="HIGH",
            abstained=False,
            latency_ms=12,
            realized_label=1 if i < 7 else 0,
            settled_at=ref_time,
        )
        for i in range(10)
    )

    mock_repo = MagicMock()
    mock_repo.list_predictions.return_value = mock_records

    with (
        patch("crypto_chatbot.cli._resolve_database_url", return_value="postgresql://dummy"),
        patch("crypto_chatbot.cli.create_db_engine"),
        patch("crypto_chatbot.cli.init_db"),
        patch("crypto_chatbot.cli.PostgresModelPredictionRepository", return_value=mock_repo),
        patch("sys.stdout", new_callable=StringIO) as fake_out,
    ):
        code = main(["shadow-report", "--symbol", "BTC", "--json"])
        assert code == 0
        parsed = json.loads(fake_out.getvalue())
        assert parsed["total_predictions"] == 10
        assert parsed["settled_predictions"] == 10
        assert parsed["is_healthy"] is True
        assert len(parsed["calibration_bins"]) == 5
