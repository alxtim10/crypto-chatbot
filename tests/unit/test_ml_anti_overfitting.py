from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import pytest

from crypto_chatbot.ml.statistics import (
    adjust_sharpe_for_structure,
    deflated_sharpe_ratio,
    degrees_of_freedom_adjusted_sharpe,
    expected_max_sharpe_under_null,
    minimum_backtest_length,
    probabilistic_sharpe_ratio,
    profitable_path_fraction,
    sample_sharpe_ratio,
)
from crypto_chatbot.ml.validation import generate_cpcv_paths
from crypto_chatbot.schemas.ml import FeatureRow


def test_sample_sharpe_ratio_zero_when_flat() -> None:
    assert sample_sharpe_ratio([0.01, 0.01, 0.01]) == pytest.approx(0.0)


def test_probabilistic_sharpe_ratio_monotonic_in_sample_size() -> None:
    small = probabilistic_sharpe_ratio(0.1, 0.0, 50)
    large = probabilistic_sharpe_ratio(0.1, 0.0, 5000)
    assert 0.0 <= small <= 1.0
    assert large > small  # more data -> more confident the Sharpe beats zero


def test_expected_max_sharpe_increases_with_trials() -> None:
    few = expected_max_sharpe_under_null(5, 1.0)
    many = expected_max_sharpe_under_null(200, 1.0)
    assert many > few
    assert expected_max_sharpe_under_null(1, 1.0) == pytest.approx(0.0)


def test_deflated_sharpe_more_trials_lowers_significance() -> None:
    rng = np.random.default_rng(42)
    returns = rng.normal(0.0015, 0.01, 600)
    dsr_one = deflated_sharpe_ratio(returns, trials=1)
    dsr_many = deflated_sharpe_ratio(returns, trials=200)
    assert dsr_many.deflated_sharpe_ratio <= dsr_one.deflated_sharpe_ratio
    assert dsr_many.expected_max_sharpe_under_null > dsr_one.expected_max_sharpe_under_null


def test_minimum_backtest_length_decreases_with_sharpe() -> None:
    low_sr = minimum_backtest_length(0.5)
    high_sr = minimum_backtest_length(1.5)
    assert low_sr is not None
    assert high_sr is not None
    assert low_sr > high_sr
    assert minimum_backtest_length(0.0) is None


def test_degrees_of_freedom_penalty() -> None:
    assert degrees_of_freedom_adjusted_sharpe(2.0, num_parameters=0, num_observations=200) == (
        pytest.approx(2.0)
    )
    assert degrees_of_freedom_adjusted_sharpe(
        2.0, num_parameters=200, num_observations=200
    ) == pytest.approx(0.0)
    penalized = degrees_of_freedom_adjusted_sharpe(2.0, num_parameters=50, num_observations=200)
    assert 0.0 < penalized < 2.0


def test_adjust_sharpe_for_structure_reports_moments() -> None:
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0, 0.01, 400)
    adj = adjust_sharpe_for_structure(returns)
    assert np.isfinite(adj.autocorr_adjusted_sharpe)
    assert np.isfinite(adj.non_normal_adjusted_sharpe)
    assert -1.0 <= adj.lag1_autocorr <= 1.0


def test_profitable_path_fraction() -> None:
    assert profitable_path_fraction([0.1, -0.2, 0.3, 0.05]) == pytest.approx(0.75)
    assert profitable_path_fraction([]) == pytest.approx(0.0)


def _feature_rows(count: int) -> list[FeatureRow]:
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    rng = np.random.default_rng(1)
    rows: list[FeatureRow] = []
    for i in range(count):
        pt = t0 + timedelta(hours=4 * i)
        fwd = float(rng.normal(0.0, 0.01))
        rows.append(
            FeatureRow(
                prediction_time=pt,
                available_time=pt - timedelta(minutes=1),
                label_end_time=pt + timedelta(hours=4),
                features={"f1": float(rng.normal())},
                forward_return=fwd,
                label=1 if fwd > 0 else 0,
            )
        )
    return rows


def test_generate_cpcv_paths_count_and_purge() -> None:
    rows = _feature_rows(60)
    paths = generate_cpcv_paths(rows, n_groups=6, test_groups_per_path=2, embargo_bars=3)
    # C(6, 2) = 15 combinations should all succeed with bidirectional purging (including group 0).
    assert len(paths) == 15
    for path in paths:
        train_set = set(path.train_indices)
        test_set = set(path.test_indices)
        assert not (train_set & test_set)
        for tr in path.train_indices:
            for te in path.test_indices:
                overlaps = (
                    rows[tr].prediction_time < rows[te].label_end_time
                    and rows[tr].label_end_time > rows[te].prediction_time
                )
                assert not overlaps


def test_generate_cpcv_paths_rejects_small_data() -> None:
    rows = _feature_rows(10)
    with pytest.raises(ValueError, match="insufficient rows"):
        generate_cpcv_paths(rows, n_groups=6, test_groups_per_path=2)
