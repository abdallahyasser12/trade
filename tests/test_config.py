"""Configuration safety, masking, finiteness, and Alpaca timeframe mapping."""

from datetime import date

import pytest

from config import (
    AlpacaCredentials,
    BacktestConfig,
    EngineConfig,
    RiskConfig,
    StrategyConfig,
    result_context_fingerprint,
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


def test_smart_strategy_warmup_covers_two_stage_and_market_dependencies():
    config = StrategyConfig(
        enable_adx_filter=True,
        adx_period=14,
        enable_market_regime_filter=True,
        regime_symbol="SPY",
        regime_timeframe="1Day",
        regime_ma_period=200,
        regime_slope_period=20,
    )
    # Same symbol/timeframe means the live symbol fetch itself must be long
    # enough for the 200-bar MA plus its 20-bar slope and prior transition.
    assert config.warmup_bars() >= 222


def test_relative_strength_rejects_mismatched_timeframes():
    config = StrategyConfig(
        timeframe="15Min",
        enable_relative_strength_filter=True,
        regime_timeframe="1Day",
    )
    assert any("timeframes to match" in error for error in config.validate())


def test_result_context_fingerprint_changes_with_every_result_dimension():
    strategy = StrategyConfig()
    risk = RiskConfig()
    backtest = BacktestConfig(
        start_date=date(2024, 1, 1), end_date=date(2024, 12, 31)
    )
    baseline = result_context_fingerprint(
        strategy, risk, backtest, data_feed="iex", experiment={"folds": 3}
    )
    assert baseline == result_context_fingerprint(
        strategy, risk, backtest, data_feed="iex", experiment={"folds": 3}
    )
    assert baseline != result_context_fingerprint(
        StrategyConfig(short_ma_period=19),
        risk,
        backtest,
        data_feed="iex",
        experiment={"folds": 3},
    )
    assert baseline != result_context_fingerprint(
        strategy, risk, backtest, data_feed="sip", experiment={"folds": 3}
    )
    assert baseline != result_context_fingerprint(
        strategy, risk, backtest, data_feed="iex", experiment={"folds": 4}
    )


def test_adaptive_risk_soft_limit_must_precede_hard_halt():
    config = RiskConfig(
        enable_adaptive_risk=True,
        drawdown_risk_reduction_start_pct=3.0,
        max_daily_loss_pct=3.0,
    )
    assert any("below the maximum session loss" in error for error in config.validate())


def test_four_hour_timeframe_is_wired_through_every_map():
    """4Hour must exist in every timeframe map or engines break at runtime."""

    from config import (
        INTRADAY_TIMEFRAMES,
        TIMEFRAMES,
        bars_per_year,
        default_poll_interval,
        timeframe_to_timedelta,
    )

    assert "4Hour" in TIMEFRAMES
    assert "4Hour" in INTRADAY_TIMEFRAMES
    assert timeframe_to_timedelta("4Hour").total_seconds() == 4 * 3600
    assert bars_per_year("4Hour") == 252 * 2.0
    assert default_poll_interval("4Hour") == 180.0
    alpaca = timeframe_to_alpaca("4Hour")
    assert alpaca.amount_value == 4
    assert alpaca.unit_value.value == "Hour"
