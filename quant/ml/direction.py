"""Specialist direction model: P(close rises over the next H bars).

One narrow task, one interpretable model, calibration measured — never a
single magical predictor (spec §17).  The causality rules are structural:

* the label at bar N is computed from close[N+H] vs close[N], so the last
  H rows carry NO label and are excluded from training entirely;
* the chronological split drops H additional rows between train and
  validation (embargo) so no training label's forward window overlaps a
  validation feature timestamp;
* prediction-time inputs are standardized with statistics frozen at
  training, and any feature outside the trained distribution flags the
  prediction OOD — consumers treat OOD as ABSTAIN, never as a number.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

from quant.ml.calibration import (
    PoolAdjacentViolators,
    brier_score,
    expected_calibration_error,
)
from quant.ml.logistic import LogisticConfig, LogisticModel


@dataclass(frozen=True)
class DirectionModelConfig:
    feature_columns: tuple[str, ...]
    horizon_bars: int = 5
    l2: float = 1.0
    epochs: int = 500
    learning_rate: float = 0.1
    train_fraction: float = 0.7
    calibration_bins: int = 10
    ood_z_limit: float = 4.0
    min_train_samples: int = 100

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.feature_columns:
            errors.append("At least one feature column is required.")
        if self.horizon_bars < 1:
            errors.append("Horizon must be at least one bar.")
        if not (0.5 <= self.train_fraction < 1.0):
            errors.append("Training fraction must be in [0.5, 1).")
        if self.calibration_bins < 2:
            errors.append("Calibration needs at least two bins.")
        if not math.isfinite(self.ood_z_limit) or self.ood_z_limit <= 0:
            errors.append("OOD z-limit must be finite and positive.")
        if self.min_train_samples < 20:
            errors.append("Minimum training sample must be at least 20.")
        return errors


@dataclass
class TrainReport:
    train_samples: int
    validation_samples: int
    excluded_unlabeled_rows: int
    embargo_rows: int
    validation_brier: float
    validation_ece: float
    validation_accuracy: float
    validation_base_rate: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class DirectionPrediction:
    prob_up: float
    calibrated_prob: float
    ood: bool
    ood_reasons: list[str] = field(default_factory=list)


class DirectionModel:
    """Direction classifier over a causal feature frame."""

    def __init__(self, config: DirectionModelConfig):
        errors = config.validate()
        if errors:
            raise ValueError("Invalid direction-model configuration: " + "; ".join(errors))
        self.config = config
        self._model: Optional[LogisticModel] = None
        self._calibrator: Optional[PoolAdjacentViolators] = None
        self.report: Optional[TrainReport] = None

    @property
    def is_trained(self) -> bool:
        return self._model is not None

    def train(self, features: pd.DataFrame) -> TrainReport:
        cfg = self.config
        missing = set(cfg.feature_columns).difference(features.columns)
        if missing:
            raise ValueError(f"Feature frame lacks columns: {sorted(missing)}")
        if "close" not in features.columns:
            raise ValueError("Feature frame must include a close column for labels.")

        close = features["close"].astype(float)
        # Label at N: does close rise over the next H bars?  The final H
        # rows have no future close and are excluded, not guessed.
        future = close.shift(-cfg.horizon_bars)
        labels = (future > close).astype(float)
        usable = features[list(cfg.feature_columns)].assign(_label=labels)
        labeled = usable.iloc[: len(usable) - cfg.horizon_bars].dropna()
        excluded = len(features) - len(labeled)
        if len(labeled) < cfg.min_train_samples:
            raise ValueError(
                f"Only {len(labeled)} labeled samples; "
                f"{cfg.min_train_samples} required."
            )

        split = int(len(labeled) * cfg.train_fraction)
        train_frame = labeled.iloc[:split]
        # Embargo: drop H rows after the split so no training label's
        # forward window overlaps a validation feature timestamp.
        validation_frame = labeled.iloc[split + cfg.horizon_bars :]
        if len(validation_frame) < cfg.calibration_bins * 2:
            raise ValueError(
                "Validation slice too small after the embargo; extend history."
            )

        X_train = train_frame[list(cfg.feature_columns)].to_numpy(dtype=float)
        y_train = train_frame["_label"].to_numpy(dtype=float)
        X_val = validation_frame[list(cfg.feature_columns)].to_numpy(dtype=float)
        y_val = validation_frame["_label"].to_numpy(dtype=float)

        model = LogisticModel(
            LogisticConfig(l2=cfg.l2, learning_rate=cfg.learning_rate, epochs=cfg.epochs)
        )
        model.fit(X_train, y_train)
        raw_val = model.predict_proba(X_val)
        calibrator = PoolAdjacentViolators().fit(raw_val, y_val)
        calibrated = calibrator.transform(raw_val)

        self._model = model
        self._calibrator = calibrator
        self.report = TrainReport(
            train_samples=len(train_frame),
            validation_samples=len(validation_frame),
            excluded_unlabeled_rows=excluded,
            embargo_rows=cfg.horizon_bars,
            validation_brier=brier_score(calibrated, y_val),
            validation_ece=expected_calibration_error(
                calibrated, y_val, cfg.calibration_bins
            ),
            validation_accuracy=float(((raw_val >= 0.5) == (y_val == 1.0)).mean()),
            validation_base_rate=float(y_val.mean()),
        )
        return self.report

    def predict(self, row: pd.Series) -> DirectionPrediction:
        if self._model is None or self._calibrator is None:
            raise RuntimeError("Model is not trained.")
        cfg = self.config
        values = []
        reasons: list[str] = []
        for column in cfg.feature_columns:
            raw = row.get(column)
            try:
                value = float(raw)
            except (TypeError, ValueError):
                value = math.nan
            if not math.isfinite(value):
                reasons.append(f"missing_feature:{column}")
                value = 0.0  # placeholder for the math; the OOD flag rules
            values.append(value)
        x = np.asarray(values, dtype=float)
        zscores = self._model.feature_zscores(x)
        for column, z in zip(cfg.feature_columns, zscores):
            if abs(float(z)) > cfg.ood_z_limit:
                reasons.append(f"feature_out_of_distribution:{column}")
        prob = float(self._model.predict_proba(x)[0])
        calibrated = float(self._calibrator.transform([prob])[0])
        return DirectionPrediction(
            prob_up=prob,
            calibrated_prob=calibrated,
            ood=bool(reasons),
            ood_reasons=sorted(set(reasons)),
        )
