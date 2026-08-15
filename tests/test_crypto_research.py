"""Bounded chronological selection and promotion-gate integration tests."""

import numpy as np
import pandas as pd
import pytest

from crypto_config import CryptoBacktestConfig, CryptoRiskConfig
from crypto_research import (
    CryptoPromotionCriteria,
    CryptoWalkForwardConfig,
    build_crypto_candidates,
    run_crypto_walk_forward,
)
from tests.crypto_helpers import compact_crypto_config


def cyclical_bars(periods=300):
    pattern = [0.0, 0.2, -0.2, -0.1, 1.0]
    close = np.asarray(
        [100.0 + index * 0.1 + pattern[index % len(pattern)] for index in range(periods)]
    )
    index = pd.date_range("2025-01-01", periods=periods, freq="h", tz="UTC")
    open_price = np.r_[close[0], close[:-1]]
    return pd.DataFrame(
        {
            "open": open_price,
            "high": np.maximum(open_price, close) + 0.5,
            "low": np.minimum(open_price, close) - 0.5,
            "close": close,
            "volume": 1_000.0,
        },
        index=index,
    )


def test_crypto_candidate_grid_is_bounded_and_rejects_unknown_rules():
    base = compact_crypto_config()
    candidates = build_crypto_candidates(
        base,
        {"breakout_period": [3, 4], "min_adx": [0, 5]},
        max_candidates=4,
    )
    assert len(candidates) == 4
    with pytest.raises(ValueError, match="maximum"):
        build_crypto_candidates(
            base, {"breakout_period": range(10)}, max_candidates=4
        )
    with pytest.raises(ValueError, match="Unsupported"):
        build_crypto_candidates(base, {"future_return": [1]}, max_candidates=4)


def test_crypto_walk_forward_uses_nonoverlapping_tests_and_shared_backtest():
    bars = cyclical_bars()
    backtest = CryptoBacktestConfig(
        start_date=bars.index[0].date(),
        end_date=bars.index[-1].date() + pd.Timedelta(days=1),
        initial_capital=10_000,
        slippage_bps=1,
        taker_fee_bps=1,
        spread_bps=1,
    )
    report = run_crypto_walk_forward(
        {"BTC/USD": bars},
        compact_crypto_config(),
        CryptoRiskConfig(max_positions=1),
        backtest,
        {},
        CryptoWalkForwardConfig(
            train_bars=100,
            test_bars=40,
            step_bars=40,
            min_train_trades=1,
            max_candidates=2,
        ),
        CryptoPromotionCriteria(
            min_folds=2,
            min_oos_trades=1,
            min_positive_fold_ratio=0,
            min_compounded_return_pct=-99,
            min_excess_return_pct=-99,
            min_aggregate_sharpe=-999,
            min_profit_factor=0.01,
            max_fold_drawdown_pct=99,
            min_stressed_return_pct=-99,
        ),
    )
    assert len(report.folds) == 4
    assert report.total_oos_trades > 0
    assert all(
        left.test_end < right.test_start
        for left, right in zip(report.folds, report.folds[1:])
    )
    assert report.protocol["execution"].startswith("shared next-bar crypto backtest")
    assert report.stressed_compounded_return_pct <= report.compounded_return_pct + 1e-9
    assert report.promotion.promoted is False


def test_walk_forward_rejects_insufficient_aligned_history():
    bars = cyclical_bars(120)
    config = CryptoBacktestConfig(
        start_date=bars.index[0].date(),
        end_date=bars.index[-1].date() + pd.Timedelta(days=1),
    )
    with pytest.raises(ValueError, match="needs"):
        run_crypto_walk_forward(
            {"BTC/USD": bars},
            compact_crypto_config(),
            CryptoRiskConfig(max_positions=1),
            config,
            {},
            CryptoWalkForwardConfig(train_bars=100, test_bars=40, step_bars=40),
            CryptoPromotionCriteria(),
        )


def _fold(strategy_return_pct, benchmark_return_pct):
    """Minimal fold stub carrying only what the promotion gate reads."""

    from types import SimpleNamespace

    def result(total):
        return SimpleNamespace(
            total_return_pct=total,
            benchmark_return_pct=benchmark_return_pct,
            num_trades=20,
            max_drawdown_pct=5.0,
            sharpe_ratio=1.5,
            profit_factor=2.0,
            equity_curve=pd.Series(
                np.linspace(100.0, 100.0 * (1 + total / 100.0), 60),
                index=pd.date_range("2025-01-01", periods=60, freq="h", tz="UTC"),
            ),
            trades=[],
            config_snapshot={"backtest": {"risk_free_rate": 0.0}},
        )

    return SimpleNamespace(
        test_result=result(strategy_return_pct),
        baseline_result=result(0.0),
        stressed_result=result(strategy_return_pct),
        selected_config=compact_crypto_config(),
    )


def test_promotion_rejects_a_strategy_that_loses_to_buy_and_hold():
    from crypto_research import evaluate_crypto_promotion

    # Profitable in absolute terms, but the benchmark tripled it every fold.
    folds = [_fold(5.0, 30.0) for _ in range(3)]
    decision, metrics = evaluate_crypto_promotion(folds, CryptoPromotionCriteria())
    assert not decision.promoted
    assert metrics["benchmark_excess"] < 0
    assert any("buy-and-hold" in failure for failure in decision.failures)


def test_promotion_reports_benchmark_excess_when_the_strategy_wins():
    from crypto_research import evaluate_crypto_promotion

    folds = [_fold(30.0, 5.0) for _ in range(3)]
    decision, metrics = evaluate_crypto_promotion(folds, CryptoPromotionCriteria())
    assert metrics["benchmark_excess"] > 0
    assert metrics["benchmark_beating_ratio"] == 1.0
    assert not any("buy-and-hold" in failure for failure in decision.failures)
