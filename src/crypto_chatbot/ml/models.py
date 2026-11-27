"""Pure-NumPy deterministic baselines, logistic regression, Platt calibration & scaling."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

import numpy as np
from numpy.typing import NDArray


def _sigmoid_array(z: NDArray[np.float64]) -> NDArray[np.float64]:
    clipped = np.clip(z, -30.0, 30.0)
    return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=np.float64)


class FeaturePreprocessor:
    """Training-fold-only median imputer and z-score scaler (prevents leakage)."""

    def __init__(self, feature_names: Sequence[str]) -> None:
        self.feature_names: tuple[str, ...] = tuple(feature_names)
        self.medians_: NDArray[np.float64] | None = None
        self.means_: NDArray[np.float64] | None = None
        self.stds_: NDArray[np.float64] | None = None
        self.fit_row_count: int = 0
        self.fit_max_time: datetime | None = None
        self.is_fitted: bool = False

    def fit(
        self,
        features_x: NDArray[np.float64],
        *,
        fit_max_time: datetime | None = None,
    ) -> FeaturePreprocessor:
        if features_x.ndim != 2 or features_x.shape[0] < 1:
            raise ValueError("features_x must be a non-empty 2D matrix")
        if features_x.shape[1] != len(self.feature_names):
            raise ValueError("feature column count mismatch")

        n_cols = features_x.shape[1]
        medians = np.zeros(n_cols, dtype=np.float64)
        imputed = features_x.astype(np.float64, copy=True)

        for j in range(n_cols):
            col = imputed[:, j]
            finite_mask = np.isfinite(col)
            med = float(np.median(col[finite_mask])) if np.any(finite_mask) else 0.0
            medians[j] = med
            if not np.all(finite_mask):
                col[~finite_mask] = med

        means = np.mean(imputed, axis=0, dtype=np.float64)
        stds = np.std(imputed, axis=0, dtype=np.float64)
        safe_stds = np.where(stds < 1e-8, 1.0, stds)

        self.medians_ = medians
        self.means_ = np.asarray(means, dtype=np.float64)
        self.stds_ = np.asarray(safe_stds, dtype=np.float64)
        self.fit_row_count = int(features_x.shape[0])
        self.fit_max_time = fit_max_time
        self.is_fitted = True
        return self

    def transform(self, features_x: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.is_fitted or self.medians_ is None or self.means_ is None or self.stds_ is None:
            raise RuntimeError("FeaturePreprocessor must be fit on training rows before transform")
        if features_x.ndim != 2 or features_x.shape[1] != len(self.feature_names):
            raise ValueError("feature matrix shape mismatch")

        imputed = features_x.astype(np.float64, copy=True)
        for j in range(len(self.feature_names)):
            col = imputed[:, j]
            bad = ~np.isfinite(col)
            if np.any(bad):
                col[bad] = self.medians_[j]

        scaled = (imputed - self.means_) / self.stds_
        return np.asarray(np.clip(scaled, -10.0, 10.0), dtype=np.float64)


class NaiveBaseRateModel:
    """Zero-parameter baseline predicting the training positive base rate."""

    def __init__(self) -> None:
        self.base_rate_: float = 0.5
        self.is_fitted: bool = False

    def fit(self, y: NDArray[np.float64]) -> NaiveBaseRateModel:
        if len(y) < 1:
            raise ValueError("y cannot be empty")
        self.base_rate_ = float(np.clip(np.mean(y), 1e-6, 1.0 - 1e-6))
        self.is_fitted = True
        return self

    def predict_proba(self, n_rows: int) -> NDArray[np.float64]:
        if not self.is_fitted:
            raise RuntimeError("NaiveBaseRateModel must be fit first")
        return np.full(n_rows, self.base_rate_, dtype=np.float64)


class L2LogisticRegression:
    """Deterministic L2-regularized binary logistic regression solved via damped IRLS."""

    def __init__(
        self,
        *,
        l2_reg: float = 1.0,
        max_iter: int = 50,
        tol: float = 1e-7,
    ) -> None:
        if l2_reg <= 0.0:
            raise ValueError("l2_reg must be positive")
        self.l2_reg = float(l2_reg)
        self.max_iter = max_iter
        self.tol = tol
        self.weights_: NDArray[np.float64] = np.zeros(0, dtype=np.float64)
        self.bias_: float = 0.0
        self.is_fitted: bool = False

    def fit(
        self,
        features_x: NDArray[np.float64],
        y: NDArray[np.float64],
        *,
        sample_weight: NDArray[np.float64] | None = None,
    ) -> L2LogisticRegression:
        if features_x.ndim != 2 or features_x.shape[0] != len(y) or features_x.shape[0] < 1:
            raise ValueError("invalid X/y dimensions for L2LogisticRegression")

        n_samples, n_features = features_x.shape
        if sample_weight is not None:
            if len(sample_weight) != n_samples:
                raise ValueError("sample_weight length must match y")
            sw = np.asarray(sample_weight, dtype=np.float64)
            sw_sum = float(np.sum(sw))
            if sw_sum <= 0.0 or np.any(sw < 0.0):
                raise ValueError("sample_weight must be non-negative with positive sum")
            norm_w = sw / sw_sum
        else:
            norm_w = np.full(n_samples, 1.0 / float(n_samples), dtype=np.float64)

        pos_rate = float(np.sum(norm_w * y))
        if pos_rate <= 0.0 or pos_rate >= 1.0:
            p_clamped = float(np.clip(pos_rate, 1e-4, 1.0 - 1e-4))
            self.bias_ = float(np.log(p_clamped / (1.0 - p_clamped)))
            self.weights_ = np.zeros(n_features, dtype=np.float64)
            self.is_fitted = True
            return self

        x_aug = np.hstack([np.ones((n_samples, 1), dtype=np.float64), features_x])
        theta = np.zeros(n_features + 1, dtype=np.float64)
        theta[0] = float(np.log(pos_rate / (1.0 - pos_rate)))

        reg_vec = np.zeros(n_features + 1, dtype=np.float64)
        reg_vec[1:] = self.l2_reg

        for _ in range(self.max_iter):
            logits = np.clip(x_aug @ theta, -30.0, 30.0)
            probs = _sigmoid_array(logits)
            w_diag = norm_w * np.clip(probs * (1.0 - probs), 1e-6, 0.25)

            grad = x_aug.T @ (norm_w * (probs - y)) + reg_vec * theta
            hess = (x_aug.T * w_diag) @ x_aug + np.diag(reg_vec + 1e-6)
            try:
                step = np.linalg.solve(hess, grad)
            except np.linalg.LinAlgError:
                step = 0.1 * grad

            theta = theta - step
            if float(np.max(np.abs(step))) < self.tol:
                break

        self.bias_ = float(theta[0])
        self.weights_ = np.asarray(theta[1:], dtype=np.float64)
        self.is_fitted = True
        return self

    def predict_logits(self, features_x: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.is_fitted:
            raise RuntimeError("L2LogisticRegression must be fit before prediction")
        logits = self.bias_ + features_x @ self.weights_
        return np.asarray(np.clip(logits, -30.0, 30.0), dtype=np.float64)

    def predict_proba(self, features_x: NDArray[np.float64]) -> NDArray[np.float64]:
        probs = _sigmoid_array(self.predict_logits(features_x))
        return np.asarray(np.clip(probs, 1e-6, 1.0 - 1e-6), dtype=np.float64)


class PlattCalibrator:
    """1D logistic probability calibrator fit exclusively on an inner training slice."""

    def __init__(self, *, a: float = 1.0, b: float = 0.0) -> None:
        self.a = float(a)
        self.b = float(b)

    def fit(self, logits: NDArray[np.float64], y: NDArray[np.float64]) -> PlattCalibrator:
        pos_rate = float(np.mean(y))
        if len(y) < 6 or pos_rate <= 0.0 or pos_rate >= 1.0:
            self.a = 1.0
            self.b = 0.0
            return self
        lr = L2LogisticRegression(l2_reg=0.1)
        lr.fit(logits.reshape(-1, 1), y)
        self.a = float(lr.weights_[0]) if abs(float(lr.weights_[0])) > 1e-6 else 1.0
        self.b = float(lr.bias_)
        return self

    def transform(self, logits: NDArray[np.float64]) -> NDArray[np.float64]:
        calibrated = _sigmoid_array(self.a * logits + self.b)
        return np.asarray(np.clip(calibrated, 1e-6, 1.0 - 1e-6), dtype=np.float64)


class CalibratedLogisticPipeline:
    """Training-only preprocessor + L2 logistic model + inner causal Platt calibrator."""

    def __init__(
        self,
        *,
        feature_names: Sequence[str],
        l2_reg: float = 1.0,
    ) -> None:
        self.preprocessor = FeaturePreprocessor(feature_names)
        self.model = L2LogisticRegression(l2_reg=l2_reg)
        self.calibrator = PlattCalibrator()
        self.l2_reg = l2_reg
        self.base_rate_: float = 0.5
        self.calibration_logits_: NDArray[np.float64] = np.zeros(0, dtype=np.float64)

    def fit(
        self,
        raw_x_train: NDArray[np.float64],
        y_train: NDArray[np.float64],
        *,
        pred_times_train: Sequence[datetime],
        label_end_times_train: Sequence[datetime],
        sample_weight: NDArray[np.float64] | None = None,
    ) -> CalibratedLogisticPipeline:
        max_time = max(pred_times_train)
        x_scaled = self.preprocessor.fit(raw_x_train, fit_max_time=max_time).transform(raw_x_train)
        if sample_weight is not None and float(np.sum(sample_weight)) > 0.0:
            self.base_rate_ = float(
                np.clip(np.average(y_train, weights=sample_weight), 1e-6, 1.0 - 1e-6)
            )
        else:
            self.base_rate_ = float(np.clip(np.mean(y_train), 1e-6, 1.0 - 1e-6))

        n = len(y_train)
        cal_count = max(8, n // 5)
        split_idx = n - cal_count
        calibrated_inner = False
        if split_idx >= 12:
            cal_start_time = pred_times_train[split_idx]
            inner_idx = [i for i in range(split_idx) if label_end_times_train[i] <= cal_start_time]
            if len(inner_idx) >= 10:
                y_inner = y_train[inner_idx]
                y_cal = y_train[split_idx:]
                w_inner = sample_weight[inner_idx] if sample_weight is not None else None
                inner_rate = float(np.mean(y_inner))
                cal_rate = float(np.mean(y_cal))
                if 0.0 < inner_rate < 1.0 and 0.0 < cal_rate < 1.0:
                    # Final predictor is the inner model composed with Platt calibrator (no refit).
                    self.model.fit(x_scaled[inner_idx], y_inner, sample_weight=w_inner)
                    cal_logits = self.model.predict_logits(x_scaled[split_idx:])
                    self.calibration_logits_ = cal_logits
                    self.calibrator.fit(cal_logits, y_cal)
                    calibrated_inner = True

        if not calibrated_inner:
            self.model.fit(x_scaled, y_train, sample_weight=sample_weight)
            self.calibrator = PlattCalibrator(a=1.0, b=0.0)
            self.calibration_logits_ = np.zeros(0, dtype=np.float64)
        return self

    def predict_proba(self, raw_x_eval: NDArray[np.float64]) -> NDArray[np.float64]:
        x_scaled = self.preprocessor.transform(raw_x_eval)
        logits = self.model.predict_logits(x_scaled)
        return self.calibrator.transform(logits)


L2_REG_GRID: tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)


def select_l2_reg_grid(
    raw_x_train: NDArray[np.float64],
    y_train: NDArray[np.float64],
    *,
    feature_names: Sequence[str],
    pred_times_train: Sequence[datetime],
    label_end_times_train: Sequence[datetime],
    sample_weight: NDArray[np.float64] | None = None,
    grid: Sequence[float] = L2_REG_GRID,
    n_splits: int = 3,
) -> tuple[float, dict[float, float]]:
    """Select L2 regularization lambda via purged inner walk-forward weighted Brier score."""
    if not grid:
        raise ValueError("grid must be non-empty")
    n = len(y_train)
    if n < 30 or n_splits < 1:
        default_lam = float(grid[-1])
        return default_lam, {float(lam): float("nan") for lam in grid}

    val_size = max(8, n // (n_splits + 2))
    split_scores: dict[float, list[float]] = {float(lam): [] for lam in grid}

    for k in range(n_splits):
        val_start = n - (n_splits - k) * val_size
        val_end = val_start + val_size
        if val_start < 12 or val_end > n:
            continue
        val_start_time = pred_times_train[val_start]
        train_idx = [i for i in range(val_start) if label_end_times_train[i] <= val_start_time]
        if len(train_idx) < 10:
            continue

        y_tr = y_train[train_idx]
        if float(np.mean(y_tr)) <= 0.0 or float(np.mean(y_tr)) >= 1.0:
            continue
        y_va = y_train[val_start:val_end]
        w_tr = sample_weight[train_idx] if sample_weight is not None else None
        w_va = sample_weight[val_start:val_end] if sample_weight is not None else None
        norm_w_va = (
            w_va / float(np.sum(w_va))
            if w_va is not None and float(np.sum(w_va)) > 0.0
            else np.full(len(y_va), 1.0 / float(len(y_va)), dtype=np.float64)
        )

        prep = FeaturePreprocessor(feature_names)
        x_tr = prep.fit(raw_x_train[train_idx]).transform(raw_x_train[train_idx])
        x_va = prep.transform(raw_x_train[val_start:val_end])

        for lam in grid:
            model = L2LogisticRegression(l2_reg=float(lam))
            model.fit(x_tr, y_tr, sample_weight=w_tr)
            p_va = model.predict_proba(x_va)
            brier = float(np.sum(norm_w_va * (p_va - y_va) ** 2))
            split_scores[float(lam)].append(brier)

    mean_scores: dict[float, float] = {}
    for lam, scores in split_scores.items():
        mean_scores[lam] = float(np.mean(scores)) if scores else float("inf")

    best_lam = min(mean_scores, key=lambda lam: (mean_scores[lam], lam))
    if not np.isfinite(mean_scores[best_lam]):
        best_lam = float(grid[-1])
    return best_lam, mean_scores
