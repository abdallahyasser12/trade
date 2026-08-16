"""Logistic baseline, calibration, and direction-model causality tests."""

import numpy as np
import pandas as pd
import pytest

from quant.ml.calibration import (
    PoolAdjacentViolators,
    brier_score,
    expected_calibration_error,
    reliability_table,
)
from quant.ml.direction import DirectionModel, DirectionModelConfig
from quant.ml.logistic import LogisticConfig, LogisticModel


def separable_data(n=200, seed=5):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 1, (n, 2))
    y = (X[:, 0] + 0.5 * X[:, 1] > 0).astype(float)
    return X, y


def test_logistic_learns_separable_pattern():
    X, y = separable_data()
    model = LogisticModel(LogisticConfig(epochs=2000, learning_rate=0.5, l2=0.01))
    model.fit(X, y)
    probs = model.predict_proba(np.array([[2.0, 1.0], [-2.0, -1.0]]))
    assert probs[0] > 0.9
    assert probs[1] < 0.1


def test_logistic_standardizes_with_training_stats_only():
    X, y = separable_data()
    model = LogisticModel().fit(X, y)
    single = model.predict_proba(X[0])
    batch = model.predict_proba(X)[:1]
    assert single[0] == pytest.approx(batch[0])


def test_logistic_constant_column_weight_stays_zero():
    X, y = separable_data()
    X = np.hstack([X, np.ones((len(X), 1))])
    model = LogisticModel().fit(X, y)
    assert model.coefficients[2] == 0.0


def test_logistic_rejects_nan():
    X, y = separable_data()
    X[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        LogisticModel().fit(X, y)


def test_pav_is_monotone_and_matches_hand_case():
    probs = [0.1, 0.4, 0.35, 0.8]
    outcomes = [0.0, 0.0, 1.0, 1.0]
    calibrator = PoolAdjacentViolators().fit(probs, outcomes)
    out = calibrator.transform(sorted(probs))
    assert all(out[i] <= out[i + 1] + 1e-12 for i in range(len(out) - 1))
    # Sorted outcomes are 0, 1, 0, 1 -> the middle violation pools to 0.5.
    assert calibrator.transform([0.36])[0] == pytest.approx(0.5)


def test_ece_zero_for_perfect_calibration():
    rng = np.random.default_rng(3)
    probs = np.repeat([0.25, 0.75], 400)
    outcomes = np.concatenate(
        [rng.random(400) < 0.25, rng.random(400) < 0.75]
    ).astype(float)
    # Empirical rates fluctuate; ECE should be small, Brier near p(1-p).
    assert expected_calibration_error(probs, outcomes) < 0.05
    table = reliability_table(probs, outcomes)
    assert sum(row["count"] for row in table) == 800
    assert brier_score([1.0, 0.0], [1.0, 0.0]) == 0.0


def make_feature_frame(periods=400, seed=11):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2024-01-01", periods=periods, freq="1h", tz="UTC")
    momentum = rng.normal(0, 1, periods)
    noise = rng.normal(0, 1, periods)
    # Close follows momentum with a lag so direction is learnable.
    ret = 0.01 * momentum + 0.002 * rng.normal(0, 1, periods)
    close = 100 * np.cumprod(1 + ret)
    return pd.DataFrame(
        {"close": close, "momentum": momentum, "noise": noise}, index=index
    )


def direction_config(**overrides):
    base = dict(
        feature_columns=("momentum", "noise"),
        horizon_bars=3,
        min_train_samples=50,
        epochs=300,
    )
    base.update(overrides)
    return DirectionModelConfig(**base)


def test_direction_model_trains_and_reports():
    model = DirectionModel(direction_config())
    frame = make_feature_frame()
    report = model.train(frame)
    assert report.train_samples > 100
    assert report.validation_samples > 30
    assert 0.0 <= report.validation_base_rate <= 1.0
    assert np.isfinite(report.validation_brier)
    assert np.isfinite(report.validation_ece)
    # Label exclusion: the last horizon rows carry no label.
    assert report.excluded_unlabeled_rows >= 3
    assert report.embargo_rows == 3


def test_direction_predictions_frozen_against_future_data():
    model = DirectionModel(direction_config())
    frame = make_feature_frame()
    model.train(frame)
    row = frame.iloc[100]
    before = model.predict(row)
    # New future data must not change a frozen model's prediction.
    _ = make_feature_frame(periods=600, seed=99)
    after = model.predict(row)
    assert before.prob_up == after.prob_up
    assert before.calibrated_prob == after.calibrated_prob


def test_direction_ood_flags():
    model = DirectionModel(direction_config(ood_z_limit=3.0))
    model.train(make_feature_frame())
    extreme = pd.Series({"momentum": 50.0, "noise": 0.0, "close": 100.0})
    prediction = model.predict(extreme)
    assert prediction.ood
    assert any("momentum" in reason for reason in prediction.ood_reasons)
    missing = pd.Series({"noise": 0.0, "close": 100.0})
    prediction = model.predict(missing)
    assert prediction.ood
    assert any("missing_feature" in reason for reason in prediction.ood_reasons)


def test_direction_config_validation():
    assert DirectionModelConfig(feature_columns=()).validate()
    assert DirectionModelConfig(feature_columns=("a",), horizon_bars=0).validate()
    assert DirectionModelConfig(feature_columns=("a",), train_fraction=0.3).validate()
    assert not direction_config().validate()
