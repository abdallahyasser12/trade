"""Deterministic tests for chronological research and promotion gates."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from config import BacktestConfig, RiskConfig, StrategyConfig
from research import (
    PromotionCriteria,
    WalkForwardConfig,
    build_candidate_configs,
    run_walk_forward,
)


def strategy_config(**overrides) -> StrategyConfig:
    params = dict(
        symbol="TEST",
        timeframe="1Day",
        short_ma_period=2,
        long_ma_period=4,
        rsi_period=2,
        rsi_overbought=100.0,
        atr_period=2,
        volume_period=2,
        min_volume_ratio=0.1,
        enable_volatility_filter=False,
    )
    params.update(overrides)
    return StrategyConfig(**params)


def cyclical_bars(n: int = 120) -> pd.DataFrame:
    x = np.arange(n, dtype=float)
    close = 100.0 + 0.05 * x + 5.0 * np.sin(2.0 * np.pi * x / 12.0)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + 1.0
    low = np.minimum(open_, close) - 1.0
    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": np.full(n, 1_000_000.0),
        },
        index=pd.date_range("2020-01-01", periods=n, freq="D", tz="UTC"),
    )


def backtest_config() -> BacktestConfig:
    return BacktestConfig(
        start_date=date(2020, 1, 1),
        end_date=date(2021, 1, 1),
        initial_capital=100_000.0,
        slippage_bps=1.0,
        commission_per_trade=0.10,
    )


def permissive_criteria() -> PromotionCriteria:
    return PromotionCriteria(
        min_folds=2,
        min_oos_trades=1,
        min_positive_fold_ratio=0.0,
        min_compounded_return_pct=-100.0,
        min_excess_return_pct=-100.0,
        min_aggregate_sharpe=-100.0,
        min_profit_factor=0.0001,
        max_fold_drawdown_pct=100.0,
        min_stressed_return_pct=-100.0,
    )


def test_candidate_grid_is_deterministic_bounded_and_skips_invalid_ma_pairs():
    base = strategy_config()
    configs = build_candidate_configs(
        base,
        {"short_ma_period": [2, 5], "long_ma_period": [4, 6]},
        max_candidates=4,
    )
    assert [(c.short_ma_period, c.long_ma_period) for c in configs] == [
        (2, 4),
        (2, 6),
        (5, 6),
    ]
    with pytest.raises(ValueError, match="Unsupported tunable"):
        build_candidate_configs(base, {"symbol": ["SPY"]})
    with pytest.raises(ValueError, match="maximum"):
        build_candidate_configs(
            base,
            {"short_ma_period": [2, 3, 4], "long_ma_period": [5, 6]},
            max_candidates=5,
        )


def test_walk_forward_folds_are_non_overlapping_embargoed_and_deterministic():
    bars = cyclical_bars()
    walk = WalkForwardConfig(
        train_bars=50,
        test_bars=20,
        step_bars=20,
        embargo_bars=1,
        min_train_trades=2,
        max_candidates=4,
        stress_slippage_bps=15.0,
        stress_commission_per_trade=1.0,
    )
    args = dict(
        price_data=bars,
        base_strategy_config=strategy_config(),
        risk_config=RiskConfig(max_position_pct=100.0),
        backtest_config=backtest_config(),
        parameter_grid={"short_ma_period": [2, 3], "long_ma_period": [4]},
        walk_config=walk,
        promotion_criteria=permissive_criteria(),
    )
    first = run_walk_forward(**args)
    second = run_walk_forward(**args)

    assert len(first.folds) == 3
    assert first.candidate_count == 2
    for prior, fold in zip(first.folds, first.folds[1:]):
        assert fold.test_start > prior.test_end
    for fold in first.folds:
        train_end_pos = bars.index.get_loc(fold.train_end)
        test_start_pos = bars.index.get_loc(fold.test_start)
        assert test_start_pos - train_end_pos - 1 == walk.embargo_bars
        assert fold.test_result.config_snapshot["active_start"] == fold.test_start.isoformat()
        assert fold.test_result.config_snapshot["liquidate_at_end"] is True
        assert fold.stressed_result.config_snapshot["backtest"]["slippage_bps"] == 15.0

    assert first.compounded_return_pct == pytest.approx(second.compounded_return_pct)
    assert [
        (f.selected_config.short_ma_period, f.selected_config.long_ma_period)
        for f in first.folds
    ] == [
        (f.selected_config.short_ma_period, f.selected_config.long_ma_period)
        for f in second.folds
    ]


def test_default_promotion_gate_refuses_insufficient_evidence():
    bars = cyclical_bars(75)
    result = run_walk_forward(
        bars,
        strategy_config(),
        RiskConfig(max_position_pct=100.0),
        backtest_config(),
        {"short_ma_period": [2], "long_ma_period": [4]},
        WalkForwardConfig(
            train_bars=50,
            test_bars=20,
            step_bars=20,
            embargo_bars=1,
            min_train_trades=2,
        ),
        PromotionCriteria(),
    )
    assert result.promotion.promoted is False
    assert any("folds" in failure for failure in result.promotion.failures)
