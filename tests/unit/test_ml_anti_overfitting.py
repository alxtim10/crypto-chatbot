from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from crypto_chatbot.ml.models import CalibratedLogisticPipeline
from crypto_chatbot.ml.statistics import (
    adjust_sharpe_for_structure,
    deflated_sharpe_ratio,
    degrees_of_freedom_adjusted_sharpe,
    expected_max_sharpe_under_null,
    minimum_backtest_length,
    minimum_track_record_length,
    probabilistic_sharpe_ratio,
    profitable_path_fraction,
    raw_kurtosis,
    sample_sharpe_ratio,
)
from crypto_chatbot.ml.trials import (
    JsonlTrialRegistry,
    TrialRecord,
    get_family_trial_sharpes,
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
    assert dsr_one.deflated_sharpe_ratio == pytest.approx(dsr_one.probabilistic_sharpe_ratio)
    dsr_many = deflated_sharpe_ratio(returns, trials=200, variance_of_trial_sharpes=0.05**2)
    assert dsr_many.deflated_sharpe_ratio <= dsr_one.deflated_sharpe_ratio
    assert dsr_many.expected_max_sharpe_under_null > dsr_one.expected_max_sharpe_under_null
    with pytest.raises(ValueError, match="required when trials > 1"):
        deflated_sharpe_ratio(returns, trials=10)
    with pytest.raises(ValueError, match="mixed observation frequencies"):
        deflated_sharpe_ratio(
            returns,
            trials=2,
            variance_of_trial_sharpes=0.01,
            obs_frequency="4h",
            trial_frequencies=("4h", "1d"),
        )


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
    assert adj.autocorr_adjusted_sharpe is not None
    assert np.isfinite(adj.autocorr_adjusted_sharpe)
    assert adj.ac_undefined is False
    assert adj.non_normal_adjusted_sharpe is None
    assert np.isfinite(adj.sr_std_error)
    assert adj.sr_std_error > 0.0
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


def test_prd_v17_golden_psr_kurtosis_se_bartlett_dsr_mintrl() -> None:
    # T-01: PSR normal golden value within 1e-9
    expected_psr = 0.5 * (
        1.0 + math.erf((0.1 * math.sqrt(999.0) / math.sqrt(1.005)) / math.sqrt(2))
    )
    actual_psr = probabilistic_sharpe_ratio(0.1, 0.0, 1000, skewness=0.0, raw_kurtosis=3.0)
    assert abs(actual_psr - expected_psr) < 1e-9

    # T-01: raw_kurtosis returns 3.0 +/- 0.05 on 1e5 seeded normal samples
    rng = np.random.default_rng(12345)
    normal_samples = rng.normal(0.0, 1.0, 100_000)
    assert abs(raw_kurtosis(normal_samples) - 3.0) < 0.05

    # T-02: sr_std_error matches sqrt((1 + SR^2/2)/(n-1)) on near-normal series
    adj = adjust_sharpe_for_structure(normal_samples[:1000])
    assert adj.non_normal_adjusted_sharpe is None
    approx_normal_se = math.sqrt((1.0 + 0.5 * (adj.raw_sharpe**2)) / 999.0)
    assert adj.sr_std_error == pytest.approx(approx_normal_se, rel=0.05)

    # T-03: Negative lag-1 autocorrelation (-0.6) never raises or yields NaN
    alt = np.array([((-1) ** i) * 0.01 + 0.001 for i in range(200)], dtype=np.float64)
    adj_alt = adjust_sharpe_for_structure(alt, bartlett_lags=1)
    assert adj_alt.autocorr_adjusted_sharpe is None or math.isfinite(
        adj_alt.autocorr_adjusted_sharpe
    )

    # T-05: Golden DSR with M=100, sigma_trials=0.05, SR=0.1, n=1000
    exp_max = expected_max_sharpe_under_null(100, 0.05**2)
    ref_dsr = probabilistic_sharpe_ratio(0.1, exp_max, 1000, skewness=0.0, raw_kurtosis=3.0)
    assert 0.0 <= ref_dsr <= 1.0
    assert ref_dsr < actual_psr

    # T-06: MinTRL exact formula and SR <= SR* -> None
    m_res = minimum_track_record_length(
        0.2,
        benchmark_sharpe=0.0,
        obs_per_year=365.0,
        confidence=0.95,
        skewness=0.0,
        raw_kurtosis=3.0,
    )
    assert m_res is not None
    z95 = 1.6448536269514722
    expected_n = 1.0 + (1.0 + 0.5 * (0.2**2)) * ((z95 / 0.2) ** 2)
    assert m_res.n_obs == pytest.approx(expected_n, rel=1e-4)
    assert m_res.years == pytest.approx(expected_n / 365.0, rel=1e-4)
    assert minimum_track_record_length(0.0, benchmark_sharpe=0.0) is None
    assert minimum_track_record_length(-0.1, benchmark_sharpe=0.0) is None


def test_prd_v17_t04_trial_registry_and_t07_no_refit_calibration(tmp_path: Path) -> None:
    # T-04: Trial Registry records distinct trial_id with identical params_sha256
    reg = JsonlTrialRegistry(tmp_path / "trials.jsonl")
    t1 = reg.record_trial(
        TrialRecord.create(
            family="setup_backtest",
            variant_params={"warmup": 600, "v": "v3"},
            data_window="2024-09-17..2026-11-25",
            symbols=("BTC/USDT:USDT",),
            timeframe="1d",
            obs_frequency="per_trade",
            sr_per_obs=0.121,
            n_obs=23,
        )
    )
    t2 = reg.record_trial(
        TrialRecord.create(
            family="setup_backtest",
            variant_params={"v": "v3", "warmup": 600},
            data_window="2024-09-17..2026-11-25",
            symbols=("BTC/USDT:USDT",),
            timeframe="1d",
            obs_frequency="per_trade",
            sr_per_obs=0.145,
            n_obs=23,
        )
    )
    assert t1.trial_id != t2.trial_id
    assert t1.params_sha256 == t2.params_sha256
    count, sharpes = get_family_trial_sharpes(
        reg, family="setup_backtest", obs_frequency="per_trade"
    )
    assert count == 2
    assert sharpes == (0.121, 0.145)

    # T-07: Platt calibrator inputs match final stored model logits identically (no refit)
    rng = np.random.default_rng(7)
    n = 80
    x = rng.normal(0.0, 1.0, size=(n, 3))
    y = (x[:, 0] + 0.5 * x[:, 1] + rng.normal(0.0, 0.5, size=n) > 0.0).astype(np.float64)
    t0 = datetime(2026, 1, 1, tzinfo=UTC)
    pred_times = [t0 + timedelta(hours=4 * i) for i in range(n)]
    end_times = [pt + timedelta(hours=4) for pt in pred_times]

    pipe = CalibratedLogisticPipeline(feature_names=("a", "b", "c"), l2_reg=1.0).fit(
        x, y, pred_times_train=pred_times, label_end_times_train=end_times
    )
    cal_count = max(8, n // 5)
    split_idx = n - cal_count
    x_cal_scaled = pipe.preprocessor.transform(x[split_idx:])
    stored_logits = pipe.model.predict_logits(x_cal_scaled)
    np.testing.assert_allclose(pipe.calibration_logits_, stored_logits, rtol=0.0, atol=1e-12)
    assert pipe.base_rate_ == pytest.approx(float(np.mean(y)))
