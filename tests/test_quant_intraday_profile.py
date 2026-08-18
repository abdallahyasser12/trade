from types import SimpleNamespace

import pandas as pd

from config import AccountMode
from quant.intraday import (
    HIGH_VELOCITY_CRYPTO_UNIVERSE,
    evaluate_intraday_result,
    make_high_velocity_engine_config,
    make_high_velocity_risk_config,
    make_high_velocity_strategy_config,
)


def test_high_velocity_profile_is_short_bar_and_broad_universe():
    cfg = make_high_velocity_strategy_config()
    assert cfg.timeframe == "5Min"
    assert len(cfg.symbols) == 12
    assert cfg.symbols == HIGH_VELOCITY_CRYPTO_UNIVERSE
    assert cfg.validate() == []


def test_trade_frequency_does_not_raise_risk_limits():
    risk = make_high_velocity_risk_config()
    assert risk.risk_per_trade_pct == 0.15
    assert risk.max_session_loss_pct == 2.0
    assert risk.max_portfolio_heat_pct == 0.90
    assert risk.max_live_total_exposure_pct == 10.0
    assert risk.max_cost_to_risk == 0.35
    assert risk.validate() == []


def test_realtime_profile_requires_fresh_quotes_and_fast_exposure_polling():
    engine = make_high_velocity_engine_config(account_mode=AccountMode.PAPER)
    assert engine.poll_interval_seconds == 5.0
    assert engine.position_open_poll_interval_seconds == 1.0
    assert engine.quote_max_age_seconds == 8.0
    assert engine.validate() == []


def test_two_percent_is_measurement_not_a_promotion_shortcut():
    index = pd.date_range("2026-08-01", periods=289, freq="5min", tz="UTC")
    curve = pd.Series([100_000.0] * 288 + [101_000.0], index=index)
    result = SimpleNamespace(equity_curve=curve, num_trades=12, total_return_pct=1.0)
    evaluation = evaluate_intraday_result(result)
    assert evaluation.activity_target_met
    assert not evaluation.aspirational_return_met
    assert 11.9 <= evaluation.trades_per_day <= 12.1
