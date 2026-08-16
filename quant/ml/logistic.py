"""Pure-numpy L2-regularized logistic regression.

The interpretable baseline every fancier model must beat (spec §17).
Deterministic full-batch gradient descent from zero-initialized weights;
feature standardization is computed on TRAINING data only and stored with
the model, so prediction-time inputs are transformed with frozen
statistics and can never leak information backward.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class LogisticConfig:
    l2: float = 1.0
    learning_rate: float = 0.1
    epochs: int = 500
    tolerance: float = 1e-8
    fit_intercept: bool = True

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not math.isfinite(self.l2) or self.l2 < 0:
            errors.append("L2 penalty must be finite and non-negative.")
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0:
            errors.append("Learning rate must be finite and positive.")
        if self.epochs < 1:
            errors.append("At least one epoch is required.")
        if not math.isfinite(self.tolerance) or self.tolerance <= 0:
            errors.append("Tolerance must be finite and positive.")
        return errors


class LogisticModel:
    """Deterministic binary classifier with stored standardization."""

    def __init__(self, config: Optional[LogisticConfig] = None):
        self.config = config or LogisticConfig()
        errors = self.config.validate()
        if errors:
            raise ValueError("Invalid logistic configuration: " + "; ".join(errors))
        self._weights: Optional[np.ndarray] = None
        self._intercept: float = 0.0
        self._mean: Optional[np.ndarray] = None
        self._std: Optional[np.ndarray] = None
        self._constant_columns: Optional[np.ndarray] = None

    @property
    def is_fitted(self) -> bool:
        return self._weights is not None

    @property
    def coefficients(self) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Model is not fitted.")
        return self._weights.copy()

    def _standardize(self, X: np.ndarray) -> np.ndarray:
        return (X - self._mean) / self._std

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be a 2-D array.")
        if len(X) != len(y):
            raise ValueError("X and y must have equal length.")
        if len(X) < 2:
            raise ValueError("At least two samples are required.")
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("Training data must be finite (no NaN/inf).")
        if not np.isin(y, (0.0, 1.0)).all():
            raise ValueError("Labels must be 0 or 1.")

        self._mean = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        # A constant column carries no information; standardizing it would
        # divide by zero, so its scale is pinned to 1 and its weight is
        # forced to stay 0 throughout training.
        self._constant_columns = std == 0
        std = np.where(self._constant_columns, 1.0, std)
        self._std = std

        Z = self._standardize(X)
        n, k = Z.shape
        weights = np.zeros(k)
        intercept = 0.0
        cfg = self.config
        previous_loss = math.inf
        for _ in range(cfg.epochs):
            logits = Z @ weights + (intercept if cfg.fit_intercept else 0.0)
            probs = 1.0 / (1.0 + np.exp(-logits))
            error = probs - y
            gradient = Z.T @ error / n + cfg.l2 * weights / n
            gradient[self._constant_columns] = 0.0
            weights -= cfg.learning_rate * gradient
            if cfg.fit_intercept:
                intercept -= cfg.learning_rate * float(error.mean())
            eps = 1e-12
            loss = float(
                -(y * np.log(probs + eps) + (1 - y) * np.log(1 - probs + eps)).mean()
                + cfg.l2 * float(weights @ weights) / (2 * n)
            )
            if abs(previous_loss - loss) < cfg.tolerance:
                break
            previous_loss = loss
        self._weights = weights
        self._intercept = intercept
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._weights is None:
            raise RuntimeError("Model is not fitted.")
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if not np.isfinite(X).all():
            raise ValueError("Prediction inputs must be finite.")
        Z = self._standardize(X)
        logits = Z @ self._weights + (
            self._intercept if self.config.fit_intercept else 0.0
        )
        return 1.0 / (1.0 + np.exp(-logits))

    def feature_zscores(self, x: np.ndarray) -> np.ndarray:
        """Standardized values of one input row under the FROZEN training
        statistics — the out-of-distribution signal consumers gate on."""
        if self._mean is None:
            raise RuntimeError("Model is not fitted.")
        x = np.asarray(x, dtype=float)
        return (x - self._mean) / self._std
