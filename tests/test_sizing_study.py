"""A sizing sweep must measure deployment, not the caps that block it."""

import math

import numpy as np
import pandas as pd
import pytest

from crypto_config import CryptoBacktestConfig, CryptoRiskConfig, CryptoStrategyConfig
from cross_sectional_research import run_cross_sectional_backtest
from sizing_study import (
    STUDY_MAX_HEAT_PCT,
    STUDY_MAX_POSITION_PCT,
    SizingGrid,
    annualised_stats,
    binding_constraint,
    implied_position_pct,
    run_sizing_study,
    scaled_risk_config,
)

SYMBOLS = ("BTC/USD", "ETH/USD", "LTC/USD", "DOGE/USD")


def _bars(periods=900, seed=0, drift=0.0015, vol=0.02):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift, vol, periods)))
    open_price = np.r_[close[0], close[:-1]]
    index = pd.date_range("2023-01-01", periods=periods, freq="D", tz="UTC")
    return pd.DataFrame(
        {
            "open": open_price,
            "high": np.maximum(open_price, close) * 1.01,
            "low": np.minimum(open_price, close) * 0.99,
            "close": close,
            "volume": rng.uniform(900.0, 1_100.0, periods),
        },
        index=index,
    )


def _universe():
    return {s: _bars(seed=i) for i, s in enumerate(SYMBOLS)}


def test_raising_risk_alone_would_be_blocked_by_the_heat_cap():
    """The defect this guards: a sweep that measures its own ceiling.

    max_portfolio_heat_pct defaults to risk x positions, so raising risk
    without raising heat means the extra budget is never spent and every point
    returns the same book.
    """

    base = CryptoRiskConfig()
    assert base.max_portfolio_heat_pct == pytest.approx(
        base.risk_per_trade_pct * base.max_positions
    )
    naive = base.__class__(**{**base.to_dict(), "risk_per_trade_pct": 2.0})
    assert naive.max_portfolio_heat_pct < naive.risk_per_trade_pct * naive.max_positions

    scaled = scaled_risk_config(base, 2.0, 3)
    assert scaled.max_portfolio_heat_pct == pytest.approx(6.0)
    assert not scaled.validate()


def test_every_ceiling_scales_together_and_stays_valid():
    base = CryptoRiskConfig()
    for risk in (0.35, 0.7, 1.05, 1.4, 2.0):
        for positions in (1, 3, 8, 12):
            scaled = scaled_risk_config(base, risk, positions)
            assert not scaled.validate(), (risk, positions, scaled.validate())
            assert scaled.risk_per_trade_pct == risk
            assert scaled.max_positions == positions
            assert scaled.max_portfolio_heat_pct >= scaled.risk_per_trade_pct
            assert scaled.max_position_pct <= scaled.max_total_exposure_pct
            assert scaled.max_portfolio_heat_pct <= STUDY_MAX_HEAT_PCT
            assert scaled.max_position_pct <= STUDY_MAX_POSITION_PCT


def test_the_live_exposure_ceiling_is_never_raised():
    """A research sweep has no business touching the live safety cap."""

    base = CryptoRiskConfig()
    for risk in (0.35, 2.0):
        for positions in (3, 12):
            scaled = scaled_risk_config(base, risk, positions)
            assert scaled.max_live_total_exposure_pct <= base.max_live_total_exposure_pct
            assert scaled.max_live_order_notional == base.max_live_order_notional


def test_implied_position_size_falls_out_of_cost_to_risk():
    """This is what makes 'the book is 90% cash' measurable rather than asserted."""

    class Result:
        round_trip_cost_bps = 75.0
        median_cost_to_risk = 0.05  # => 1500bps stop distance => 15%

    assert implied_position_pct(Result(), 0.35) == pytest.approx(2.3333, rel=1e-3)
    assert implied_position_pct(Result(), 2.0) == pytest.approx(13.3333, rel=1e-3)


def test_degenerate_cost_to_risk_yields_no_implied_size():
    class Result:
        round_trip_cost_bps = 75.0
        median_cost_to_risk = None

    assert implied_position_pct(Result(), 1.0) is None

    class Zero:
        round_trip_cost_bps = 75.0
        median_cost_to_risk = 0.0

    assert implied_position_pct(Zero(), 1.0) is None

    class Infinite:
        round_trip_cost_bps = 75.0
        median_cost_to_risk = math.inf

    assert implied_position_pct(Infinite(), 1.0) is None


def test_the_binding_constraint_is_named():
    """A flat return is ambiguous unless you know which ceiling stopped it."""

    risk = scaled_risk_config(CryptoRiskConfig(), 0.35, 3)
    assert binding_constraint(risk, 2.3, 3).startswith("none")
    assert "per-position cap" in binding_constraint(risk, 99.0, 3)
    assert binding_constraint(risk, None, 3).startswith("unknown")


def test_sharpe_is_scale_invariant_which_is_why_the_study_reads_it():
    """Doubling every return leaves Sharpe alone; that is the whole diagnostic."""

    index = pd.date_range("2023-01-01", periods=400, freq="D", tz="UTC")
    rng = np.random.default_rng(7)
    returns = rng.normal(0.001, 0.02, len(index))
    single = pd.Series(100.0 * np.exp(np.cumsum(returns)), index=index)
    doubled = pd.Series(100.0 * np.exp(np.cumsum(returns * 2)), index=index)

    _, vol_a, sharpe_a = annualised_stats(single, "1Day")
    _, vol_b, sharpe_b = annualised_stats(doubled, "1Day")
    assert vol_b == pytest.approx(vol_a * 2, rel=0.02)
    assert sharpe_b == pytest.approx(sharpe_a, rel=0.10)


def test_annualisation_rejects_an_unknown_timeframe():
    index = pd.date_range("2023-01-01", periods=50, freq="D", tz="UTC")
    curve = pd.Series(np.linspace(100, 120, 50), index=index)
    with pytest.raises(ValueError, match="Unsupported timeframe"):
        annualised_stats(curve, "3Weeks")


def test_a_degenerate_curve_reports_no_sharpe_rather_than_a_fake_one():
    index = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
    flat = pd.Series(100.0, index=index)
    _, volatility, sharpe = annualised_stats(flat, "1Day")
    assert volatility == 0.0 and sharpe is None
    assert annualised_stats(pd.Series(dtype="float64"), "1Day") == (0.0, 0.0, None)


def test_an_invalid_grid_is_rejected_before_any_backtest_runs():
    assert any("(0, 2]" in e for e in SizingGrid(risk_per_trade_pcts=(5.0,)).validate())
    assert any("[1, 12]" in e for e in SizingGrid(position_counts=(99,)).validate())
    assert any("at least one" in e for e in SizingGrid(risk_per_trade_pcts=()).validate())
    with pytest.raises(ValueError, match="Invalid sizing grid"):
        run_sizing_study(
            run_cross_sectional_backtest,
            _universe(),
            CryptoStrategyConfig(symbols=SYMBOLS, timeframe="1Day"),
            CryptoRiskConfig(),
            CryptoBacktestConfig(),
            grid=SizingGrid(risk_per_trade_pcts=(99.0,)),
        )


def test_the_study_runs_end_to_end_and_deployment_actually_rises():
    universe = _universe()
    study = run_sizing_study(
        run_cross_sectional_backtest,
        universe,
        CryptoStrategyConfig(symbols=SYMBOLS, timeframe="1Day"),
        CryptoRiskConfig(),
        CryptoBacktestConfig(
            start_date=universe["BTC/USD"].index[0].date(),
            end_date=universe["BTC/USD"].index[-1].date(),
        ),
        grid=SizingGrid(risk_per_trade_pcts=(0.35, 1.40), position_counts=(3,)),
    )
    assert len(study.points) == 2
    low, high = study.points
    assert low.risk_per_trade_pct < high.risk_per_trade_pct
    # The point of the whole exercise: more risk must actually deploy more.
    assert high.implied_position_pct > low.implied_position_pct
    assert high.annualised_volatility_pct > low.annualised_volatility_pct
    assert study.benchmark_sharpe is not None
    assert "Sharpe" in study.verdict()


def test_the_verdict_distinguishes_holding_from_decaying():
    universe = _universe()
    study = run_sizing_study(
        run_cross_sectional_backtest,
        universe,
        CryptoStrategyConfig(symbols=SYMBOLS, timeframe="1Day"),
        CryptoRiskConfig(),
        CryptoBacktestConfig(
            start_date=universe["BTC/USD"].index[0].date(),
            end_date=universe["BTC/USD"].index[-1].date(),
        ),
        grid=SizingGrid(risk_per_trade_pcts=(0.35, 1.40), position_counts=(3,)),
    )
    verdict = study.verdict()
    assert ("held under deployment" in verdict) or ("decayed under deployment" in verdict)

    # And the judgement is driven by the numbers, not hardcoded either way.
    from sizing_study import SizingStudy

    def synthetic(low_sharpe, high_sharpe):
        points = [
            type(study.points[0])(
                risk_per_trade_pct=risk, max_positions=3, heat_pct=1.05, num_trades=10,
                total_return_pct=1.0, max_drawdown_pct=1.0, sharpe_ratio=sharpe,
                profit_factor=1.2, annualised_volatility_pct=5.0, exposure_time_pct=10.0,
                implied_position_pct=2.0, binding_constraint="none",
            )
            for risk, sharpe in ((0.35, low_sharpe), (1.40, high_sharpe))
        ]
        return SizingStudy(points, 100.0, 0.2, 60.0, SizingGrid())

    assert synthetic(1.0, 0.95).quality_holds() is True
    assert synthetic(1.0, 0.30).quality_holds() is False
    assert synthetic(1.0, None).quality_holds() is None


def test_the_sizing_script_does_not_reach_into_a_sibling_script():
    """The import that broke a CI run, guarded.

    _aligned used to do `from run_crypto_research import ...`, which only
    resolved when the scripts directory happened to be on sys.path. That is a
    dependency on process start-up state rather than on anything declared, and
    it is why both scripts now share universe_alignment.
    """

    from pathlib import Path

    source = (
        Path(__file__).resolve().parent.parent / "scripts" / "run_sizing_study.py"
    ).read_text()
    assert "from run_crypto_research import" not in source
    assert "from universe_alignment import" in source


def test_an_unexpected_crash_is_reported_rather_than_swallowed():
    """CI publishes stdout; a bare non-zero exit leaves nothing to diagnose."""

    import importlib.util
    import sys
    from pathlib import Path

    script = Path(__file__).resolve().parent.parent / "scripts" / "run_sizing_study.py"
    spec = importlib.util.spec_from_file_location("run_sizing_study_guard", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_sizing_study_guard"] = module
    spec.loader.exec_module(module)

    def boom(argv=None):
        raise RuntimeError("synthetic failure")

    original = module.main
    module.main = boom
    try:
        assert module._guarded_main([]) == 2
    finally:
        module.main = original
