"""Anti-overfitting statistics for strategy evaluation.

Deterministic implementations of standard methods that adjust naive performance metrics for
selection bias and the statistical properties of returns:

- Sharpe-ratio adjustment for autocorrelation (Lo, 2002) and non-normality.
- Deflated / probabilistic Sharpe ratio for multiple testing (Bailey & Lopez de Prado).
- Minimum backtest length implied by an observed Sharpe ratio.
- Degrees-of-freedom penalty for parameter count relative to sample size.

These are statistical measurements used to gate promotion of an offline model or rule-based
strategy. They are never exposed to end users as success probabilities for a live trade.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from pydantic import BaseModel, ConfigDict, Field

_EPS = 1e-12
# Euler-Mascheroni constant, used in the expected-maximum-Sharpe approximation.
_EULER_GAMMA = 0.5772156649015329


class SharpeAdjustment(BaseModel):
    """Sharpe ratio adjusted for Bartlett autocorrelation and non-normal standard error."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    raw_sharpe: float
    autocorr_adjusted_sharpe: float | None = None
    ac_undefined: bool = False
    sr_std_error: float
    non_normal_adjusted_sharpe: float | None = None  # Deprecated in v1.7 (replaced by sr_std_error)
    skewness: float
    raw_kurtosis: float
    excess_kurtosis: float
    lag1_autocorr: float


class MinTrackRecordResult(BaseModel):
    """Minimum track record length in observations and equivalent years."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    n_obs: float
    years: float


class DeflatedSharpeResult(BaseModel):
    """Probabilistic and deflated Sharpe ratios accounting for multiple testing."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    observed_sharpe: float
    trials: int = Field(ge=1)
    sample_size: int = Field(ge=2)
    expected_max_sharpe_under_null: float
    probabilistic_sharpe_ratio: float = Field(ge=0.0, le=1.0)
    deflated_sharpe_ratio: float = Field(ge=0.0, le=1.0)
    is_significant: bool


def _as_1d_float64(values: ArrayLike, name: str) -> NDArray[np.float64]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    return arr


def _normal_cdf(x: float) -> float:
    """Standard normal CDF via the error function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _normal_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if not (0.0 < p < 1.0):
        raise ValueError("p must lie in (0, 1)")
    a = (
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    )
    b = (
        -5.447609879822406e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    )
    c = (
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    )
    d = (
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    )
    p_low = 0.02425
    p_high = 1.0 - p_low
    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
        )
    if p <= p_high:
        q = p - 0.5
        r = q * q
        return (
            (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
            * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
        )
    q = math.sqrt(-2.0 * math.log(1.0 - p))
    return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
        (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0
    )


def raw_kurtosis(returns: ArrayLike) -> float:
    """Raw fourth standardized moment (equals 3.0 under a normal distribution)."""
    arr = _as_1d_float64(returns, "returns")
    finite = arr[np.isfinite(arr)]
    if len(finite) < 2:
        raise ValueError("at least 2 finite returns are required")
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1))
    if std <= _EPS:
        return 3.0
    centered = finite - mean
    return float(np.mean(centered**4) / (std**4))


def sample_sharpe_ratio(returns: ArrayLike) -> float:
    """Per-period Sharpe ratio: mean / sample std (ddof=1). Zero when std is ~0."""
    arr = _as_1d_float64(returns, "returns")
    finite = arr[np.isfinite(arr)]
    if len(finite) < 2:
        raise ValueError("at least 2 finite returns are required")
    std = float(np.std(finite, ddof=1))
    if std <= _EPS:
        return 0.0
    return float(np.mean(finite)) / std


def adjust_sharpe_for_structure(
    returns: ArrayLike,
    *,
    bartlett_lags: int = 1,
) -> SharpeAdjustment:
    """Compute Bartlett autocorrelation-adjusted Sharpe and non-normal SR standard error."""
    if bartlett_lags < 1:
        raise ValueError("bartlett_lags must be >= 1")
    arr = _as_1d_float64(returns, "returns")
    finite = arr[np.isfinite(arr)]
    n = len(finite)
    if n < 3:
        raise ValueError("at least 3 finite returns are required")

    raw = sample_sharpe_ratio(finite)
    mean = float(np.mean(finite))
    std = float(np.std(finite, ddof=1))

    if std <= _EPS:
        return SharpeAdjustment(
            raw_sharpe=0.0,
            autocorr_adjusted_sharpe=0.0,
            ac_undefined=False,
            sr_std_error=1.0 / math.sqrt(n - 1),
            non_normal_adjusted_sharpe=None,
            skewness=0.0,
            raw_kurtosis=3.0,
            excess_kurtosis=0.0,
            lag1_autocorr=0.0,
        )

    centered = finite - mean
    denom = float(np.sum(centered**2))
    max_k = min(bartlett_lags, n - 1)
    rhos: list[float] = []
    for k in range(1, max_k + 1):
        rk = float(np.sum(centered[k:] * centered[:-k]) / denom) if denom > _EPS else 0.0
        rhos.append(min(0.999, max(-0.999, rk)))
    rho1 = rhos[0] if rhos else 0.0

    # Bartlett long-run variance factor: eta = 1 + 2 * sum_{k=1}^L (1 - k/(L+1)) * rho_k
    eta = 1.0 + 2.0 * sum(
        (1.0 - float(k) / float(bartlett_lags + 1)) * rhos[k - 1] for k in range(1, max_k + 1)
    )
    if eta <= 0.0 or not math.isfinite(eta):
        autocorr_adjusted: float | None = None
        ac_undefined = True
    else:
        autocorr_adjusted = raw / math.sqrt(eta)
        ac_undefined = False

    skew = float(np.mean(centered**3) / (std**3))
    kurt = float(np.mean(centered**4) / (std**4))
    excess_kurt = kurt - 3.0
    var_numer = 1.0 - skew * raw + ((kurt - 1.0) / 4.0) * (raw**2)
    sr_se = math.sqrt(max(_EPS, var_numer) / float(n - 1))

    return SharpeAdjustment(
        raw_sharpe=raw,
        autocorr_adjusted_sharpe=autocorr_adjusted,
        ac_undefined=ac_undefined,
        sr_std_error=sr_se,
        non_normal_adjusted_sharpe=None,
        skewness=skew,
        raw_kurtosis=kurt,
        excess_kurtosis=excess_kurt,
        lag1_autocorr=rho1,
    )


def expected_max_sharpe_under_null(trials: int, variance_of_trial_sharpes: float) -> float:
    """Expected maximum Sharpe across `trials` independent strategies with zero true Sharpe."""
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if trials == 1:
        return 0.0
    sigma = math.sqrt(max(0.0, variance_of_trial_sharpes))
    if sigma <= _EPS:
        return 0.0
    term1 = (1.0 - _EULER_GAMMA) * _normal_ppf(1.0 - 1.0 / trials)
    term2 = _EULER_GAMMA * _normal_ppf(1.0 - 1.0 / (trials * math.e))
    return sigma * (term1 + term2)


def probabilistic_sharpe_ratio(
    observed_sharpe: float,
    benchmark_sharpe: float,
    sample_size: int,
    *,
    skewness: float = 0.0,
    raw_kurtosis: float | None = None,
    excess_kurtosis: float | None = None,
) -> float:
    """Probability that the true Sharpe exceeds a benchmark (Bailey & Lopez de Prado).

    Denominator uses raw kurtosis `gamma_4`: `1 - gamma_3 * SR + ((gamma_4 - 1) / 4) * SR^2`.
    When neither `raw_kurtosis` nor `excess_kurtosis` is supplied, normal kurtosis `3.0` is used.
    """
    if sample_size < 2:
        raise ValueError("sample_size must be >= 2")
    if raw_kurtosis is not None and excess_kurtosis is not None:
        raise ValueError("specify at most one of raw_kurtosis or excess_kurtosis")
    if raw_kurtosis is not None:
        gamma4 = float(raw_kurtosis)
    elif excess_kurtosis is not None:
        gamma4 = float(excess_kurtosis) + 3.0
    else:
        gamma4 = 3.0

    denom = 1.0 - skewness * observed_sharpe + ((gamma4 - 1.0) / 4.0) * (observed_sharpe**2)
    if denom <= _EPS:
        return 1.0 if observed_sharpe > benchmark_sharpe else 0.0
    numer = (observed_sharpe - benchmark_sharpe) * math.sqrt(sample_size - 1)
    return _normal_cdf(numer / math.sqrt(denom))


def deflated_sharpe_ratio(
    returns: ArrayLike,
    *,
    trials: int,
    variance_of_trial_sharpes: float | None = None,
    trial_sharpes: Sequence[float] | None = None,
    obs_frequency: str | None = None,
    trial_frequencies: Sequence[str] | None = None,
    significance: float = 0.95,
) -> DeflatedSharpeResult:
    """Deflated Sharpe ratio: PSR against the expected maximum Sharpe under the null.

    `trials` is the number of independent strategy configurations evaluated at the same
    observation frequency. For `trials == 1`, `E[max SR_0] = 0` so `DSR == PSR(0)`.
    For `trials > 1`, either `variance_of_trial_sharpes` or `trial_sharpes` (>= 2 values)
    must be provided explicitly (no unit-mismatched default of 1.0).
    """
    if trials < 1:
        raise ValueError("trials must be >= 1")
    if trial_frequencies is not None:
        freq_set = {f.strip().lower() for f in trial_frequencies if f.strip()}
        if obs_frequency is not None:
            freq_set.add(obs_frequency.strip().lower())
        if len(freq_set) > 1:
            raise ValueError(
                f"mixed observation frequencies cannot be combined in DSR: {sorted(freq_set)}"
            )

    arr = _as_1d_float64(returns, "returns")
    finite = arr[np.isfinite(arr)]
    n = len(finite)
    if n < 2:
        raise ValueError("at least 2 finite returns are required")

    adj = adjust_sharpe_for_structure(finite) if n >= 3 else None
    observed = sample_sharpe_ratio(finite)
    skew = adj.skewness if adj is not None else 0.0
    kurt = adj.raw_kurtosis if adj is not None else 3.0

    if trials == 1:
        expected_max = 0.0
    else:
        if trial_sharpes is not None:
            if len(trial_sharpes) < 2:
                raise ValueError("trial_sharpes must contain at least 2 entries when trials > 1")
            var_trials = float(np.var(np.asarray(trial_sharpes, dtype=np.float64), ddof=1))
        elif variance_of_trial_sharpes is not None:
            if variance_of_trial_sharpes < 0.0:
                raise ValueError("variance_of_trial_sharpes must be non-negative")
            var_trials = float(variance_of_trial_sharpes)
        else:
            raise ValueError(
                "variance_of_trial_sharpes or trial_sharpes is required when trials > 1"
            )
        expected_max = expected_max_sharpe_under_null(trials, var_trials)

    psr_vs_zero = probabilistic_sharpe_ratio(observed, 0.0, n, skewness=skew, raw_kurtosis=kurt)
    dsr = probabilistic_sharpe_ratio(observed, expected_max, n, skewness=skew, raw_kurtosis=kurt)
    return DeflatedSharpeResult(
        observed_sharpe=observed,
        trials=trials,
        sample_size=n,
        expected_max_sharpe_under_null=expected_max,
        probabilistic_sharpe_ratio=min(1.0, max(0.0, psr_vs_zero)),
        deflated_sharpe_ratio=min(1.0, max(0.0, dsr)),
        is_significant=dsr >= significance,
    )


def minimum_track_record_length(
    observed_sharpe: float,
    *,
    benchmark_sharpe: float = 0.0,
    obs_per_year: float = 365.0,
    confidence: float = 0.95,
    skewness: float = 0.0,
    raw_kurtosis: float = 3.0,
) -> MinTrackRecordResult | None:
    """Minimum Track Record Length (Bailey & Lopez de Prado) in observations and years.

    `n* = 1 + [1 - gamma_3 * SR + ((gamma_4 - 1) / 4) * SR^2] * (z_{1-alpha} / (SR - SR*))^2`.
    Returns `None` when `observed_sharpe <= benchmark_sharpe`.
    """
    if obs_per_year <= 0.0:
        raise ValueError("obs_per_year must be > 0")
    if not (0.5 < confidence < 1.0):
        raise ValueError("confidence must lie in (0.5, 1.0)")
    diff = observed_sharpe - benchmark_sharpe
    if diff <= _EPS:
        return None
    z = _normal_ppf(confidence)
    moment_factor = (
        1.0 - skewness * observed_sharpe + ((raw_kurtosis - 1.0) / 4.0) * (observed_sharpe**2)
    )
    if moment_factor <= _EPS:
        return None
    n_obs = 1.0 + moment_factor * ((z / diff) ** 2)
    if not math.isfinite(n_obs) or n_obs <= 0.0:
        return None
    return MinTrackRecordResult(n_obs=n_obs, years=n_obs / obs_per_year)


def minimum_backtest_length(
    observed_sharpe: float,
    *,
    periods_per_year: float = 365.0,
    confidence: float = 0.95,
    benchmark_sharpe: float = 0.0,
    skewness: float = 0.0,
    raw_kurtosis: float = 3.0,
) -> float | None:
    """Backwards-compatible wrapper returning MinTRL in years (or `None` if SR <= SR*)."""
    res = minimum_track_record_length(
        observed_sharpe,
        benchmark_sharpe=benchmark_sharpe,
        obs_per_year=periods_per_year,
        confidence=confidence,
        skewness=skewness,
        raw_kurtosis=raw_kurtosis,
    )
    return res.years if res is not None else None


def degrees_of_freedom_adjusted_sharpe(
    raw_sharpe: float,
    *,
    num_parameters: int,
    num_observations: int,
) -> float:
    """Penalize Sharpe for free parameters: `SR * sqrt(1 - k/N)`; 0 when k >= N."""
    if num_parameters < 0:
        raise ValueError("num_parameters must be >= 0")
    if num_observations < 1:
        raise ValueError("num_observations must be >= 1")
    if num_parameters >= num_observations:
        return 0.0
    return raw_sharpe * math.sqrt(1.0 - num_parameters / num_observations)


def profitable_path_fraction(path_returns: Sequence[float]) -> float:
    """Fraction of CPCV paths with a strictly positive total return."""
    if not path_returns:
        return 0.0
    positive = sum(1 for r in path_returns if r > 0.0)
    return positive / len(path_returns)
