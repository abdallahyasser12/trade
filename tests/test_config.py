"""Configuration safety, masking, finiteness, and Alpaca timeframe mapping."""

from datetime import date

import pytest

from config import (
    AlpacaCredentials,
    BacktestConfig,
    EngineConfig,
    RiskConfig,
    StrategyConfig,
    timeframe_to_alpaca,
)


def test_credential_repr_and_fingerprint_never_expose_secret():
    creds = AlpacaCredentials("PK123456", "super-secret-value")
    rendered = repr(creds)
    assert "super-secret-value" not in rendered
    assert "PK123456" not in rendered
    assert "super-secret-value" not in creds.fingerprint
    assert creds.fingerprint != AlpacaCredentials("PK123456", "other-secret").fingerprint


@pytest.mark.parametrize(
    "config",
    [
        StrategyConfig(atr_multiplier=float("nan")),
        RiskConfig(risk_per_trade_pct=float("nan")),
        EngineConfig(poll_interval_seconds=float("inf")),
        BacktestConfig(
            start_date=date(2024, 1, 1),
            end_date=date(2024, 2, 1),
            initial_capital=float("nan"),
        ),
    ],
)
def test_non_finite_configuration_is_rejected(config):
    assert config.validate()


@pytest.mark.parametrize(
    "name, amount, unit_value",
    [
        ("1Min", 1, "Min"),
        ("5Min", 5, "Min"),
        ("15Min", 15, "Min"),
        ("1Hour", 1, "Hour"),
        ("1Day", 1, "Day"),
    ],
)
def test_timeframe_mapping_uses_alpaca_py_objects(name, amount, unit_value):
    timeframe = timeframe_to_alpaca(name)
    assert timeframe.amount_value == amount
    assert timeframe.unit_value.value == unit_value
