"""Cross-sectional momentum: causality, comparability, and trade frequency."""

import numpy as np
import pandas as pd
import pytest

from cross_sectional import (
    CrossSectionalConfig,
    CrossSectionalMomentumStrategy,
    CrossSectionalStrategyConfig,
    with_cross_config,
)
from cross_sectional_research import (
    _candidate_builder,
    run_cross_sectional_backtest,
    run_cross_sectional_walk_forward,
)
from crypto_config import CryptoBacktestConfig, CryptoRiskConfig, CryptoStrategyConfig
from crypto_research import CryptoPromotionCriteria, CryptoWalkForwardConfig
from models import BotAction

SYMBOLS = ("BTC/USD", "ETH/USD", "LTC/USD", "DOGE/USD")


def _bars(periods=900, seed=0, drift=0.05, vol=0.02):
    rng = np.random.default_rng(seed)
    close = 100.0 * np.exp(np.cumsum(rng.normal(drift / 100.0, vol, periods)))
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


def _universe(periods=900):
    return {
        symbol: _bars(periods=periods, seed=i, drift=0.20 - 0.10 * i)
        for i, symbol in enumerate(SYMBOLS)
    }


def _strategy(**overrides):
    config = CryptoStrategyConfig(symbols=SYMBOLS, timeframe="1Day")
    return CrossSectionalMomentumStrategy(config, CrossSectionalConfig(**overrides))


def test_ranks_are_a_permutation_of_the_eligible_universe():
    """Every symbol is ranked against the same bar, never a neighbouring one."""

    strategy = _strategy()
    prepared = strategy.prepare_universe(_universe())
    index = prepared["BTC/USD"].index
    for timestamp in (index[-1], index[-50], index[-200]):
        ranks = sorted(
            prepared[symbol].loc[timestamp, "rank"] for symbol in SYMBOLS
        )
        assert ranks == [1.0, 2.0, 3.0, 4.0]


def test_score_is_causal_and_ignores_every_future_bar():
    """Truncating history must not change any score already computed."""

    strategy = _strategy()
    universe = _universe()
    full = strategy.prepare_universe(universe)
    cutoff = universe["BTC/USD"].index[-100]
    truncated = strategy.prepare_universe(
        {symbol: frame.loc[:cutoff] for symbol, frame in universe.items()}
    )
    for symbol in SYMBOLS:
        a = full[symbol].loc[:cutoff, "score"].dropna()
        b = truncated[symbol].loc[:cutoff, "score"].dropna()
        assert len(a) == len(b)
        pd.testing.assert_series_equal(a, b, check_names=False)


def test_volatility_scaling_makes_symbols_comparable():
    """A quiet and a wild symbol with the same risk-adjusted move rank equally.

    This is the property the old score lacked: it summed a raw fractional
    return with a normalised ADX, so a big move in a volatile name always
    outranked an equivalent risk-adjusted move in a calm one.
    """

    strategy = _strategy(lookback=20, vol_lookback=20)
    quiet = _bars(seed=11, drift=0.10, vol=0.01)
    wild = _bars(seed=12, drift=0.40, vol=0.04)
    prepared = strategy.prepare_universe(
        {
            "BTC/USD": quiet,
            "ETH/USD": wild,
            "LTC/USD": _bars(seed=13),
            "DOGE/USD": _bars(seed=14),
        }
    )
    # Wild has 4x the drift AND 4x the vol, so its risk-adjusted score should
    # be of the same order as quiet's, not 4x larger.
    quiet_score = prepared["BTC/USD"]["score"].dropna()
    wild_score = prepared["ETH/USD"]["score"].dropna()
    ratio = abs(wild_score.mean()) / max(abs(quiet_score.mean()), 1e-9)
    assert 0.2 < ratio < 5.0, f"vol scaling failed to equalise magnitudes: {ratio}"


def test_only_top_k_above_the_score_floor_can_buy():
    strategy = _strategy(top_k=1, min_score=0.0)
    prepared = strategy.prepare_universe(_universe())
    last = prepared["BTC/USD"].index[-1]
    decisions = strategy.evaluate_prepared(prepared, last)
    buys = [d for d in decisions if d.action is BotAction.BUY]
    assert len(buys) <= 1
    for buy in buys:
        assert prepared[buy.symbol].loc[last, "rank"] == 1.0


def test_a_uniformly_bad_universe_produces_no_entry():
    """The floor exists so the book can decline 'best of a bad lot'."""

    strategy = _strategy(min_score=99.0)
    prepared = strategy.prepare_universe(_universe())
    last = prepared["BTC/USD"].index[-1]
    decisions = strategy.evaluate_prepared(prepared, last)
    assert not [d for d in decisions if d.action is BotAction.BUY]


def test_exit_is_stickier_than_entry():
    """Turnover was the measured killer, so leaving needs a worse rank."""

    strategy = _strategy(top_k=1, exit_rank_buffer=2)
    prepared = strategy.prepare_universe(_universe())
    last = prepared["BTC/USD"].index[-1]
    ranked = sorted(SYMBOLS, key=lambda s: prepared[s].loc[last, "rank"])
    second = ranked[1]  # rank 2: not good enough to enter, good enough to keep
    entering = strategy.evaluate_prepared(prepared, last, held_symbols=frozenset())
    holding = strategy.evaluate_prepared(prepared, last, held_symbols={second})
    entry_action = next(d.action for d in entering if d.symbol == second)
    hold_action = next(d.action for d in holding if d.symbol == second)
    assert entry_action is BotAction.HOLD  # would not open it
    assert hold_action is not BotAction.EXIT  # but does not dump it either


def test_degenerate_inputs_fail_closed_rather_than_trading():
    """A flat series has zero volatility; that must never become infinite score."""

    strategy = _strategy()
    flat = _bars(seed=1)
    flat["close"] = 100.0
    flat["open"] = 100.0
    flat["high"] = 100.0
    flat["low"] = 100.0
    universe = _universe()
    universe["BTC/USD"] = flat
    prepared = strategy.prepare_universe(universe)
    scores = prepared["BTC/USD"]["score"]
    assert not np.isinf(scores.to_numpy(dtype=float)).any()
    last = prepared["BTC/USD"].index[-1]
    decisions = strategy.evaluate_prepared(prepared, last)
    flat_decision = next(d for d in decisions if d.symbol == "BTC/USD")
    assert flat_decision.action is not BotAction.BUY


def test_it_actually_trades_where_the_breakout_conjunction_did_not():
    """The diagnosed failure was 2 trades in five years of daily data."""

    universe = _universe()
    result = run_cross_sectional_backtest(
        universe,
        CryptoStrategyConfig(symbols=SYMBOLS, timeframe="1Day"),
        CryptoRiskConfig(),
        CryptoBacktestConfig(
            start_date=universe["BTC/USD"].index[0].date(),
            end_date=universe["BTC/USD"].index[-1].date(),
        ),
    )
    assert result.num_trades > 20, (
        f"only {result.num_trades} trades: frequency is still accidental"
    )
    # And the shared cost machinery still applies to it.
    assert result.round_trip_cost_bps == pytest.approx(75.0)
    assert result.median_cost_to_risk is not None


def test_candidates_that_differ_only_cross_sectionally_are_distinct():
    """The bug this guards: a search that runs N identical backtests.

    The audited builder de-duplicates by ``to_dict``. If the cross-sectional
    parameters did not reach that dict, every candidate would collapse to one
    and the reported candidate count would describe a search that never ran.
    """

    build = _candidate_builder(CrossSectionalConfig())
    candidates = build(
        CryptoStrategyConfig(symbols=SYMBOLS),
        {"cross": [CrossSectionalConfig(lookback=20), CrossSectionalConfig(lookback=45)]},
        max_candidates=32,
    )
    assert len(candidates) == 2
    assert {c.cross.lookback for c in candidates} == {20, 45}
    identities = {repr(sorted(c.to_dict().items())) for c in candidates}
    assert len(identities) == 2, "cross parameters never reached the dedup identity"


def test_the_cross_axis_multiplies_the_base_axis():
    build = _candidate_builder(CrossSectionalConfig())
    candidates = build(
        CryptoStrategyConfig(symbols=SYMBOLS),
        {
            "cross": [CrossSectionalConfig(top_k=1), CrossSectionalConfig(top_k=3)],
            "atr_stop_multiplier": [2.0, 3.0],
        },
        max_candidates=32,
    )
    assert len(candidates) == 4
    assert {(c.cross.top_k, c.atr_stop_multiplier) for c in candidates} == {
        (1, 2.0), (1, 3.0), (3, 2.0), (3, 3.0)
    }


def test_the_candidate_cap_still_binds_after_the_cross_axis_is_added():
    """Delegating to the audited builder must not smuggle candidates past it."""

    build = _candidate_builder(CrossSectionalConfig())
    grid = {
        "cross": [CrossSectionalConfig(lookback=lb) for lb in (10, 20, 30)],
        "atr_stop_multiplier": [2.0, 2.5, 3.0],
    }
    with pytest.raises(ValueError, match="maximum is"):
        build(CryptoStrategyConfig(symbols=SYMBOLS), grid, max_candidates=8)
    # And the base allowlist is still enforced, not bypassed.
    with pytest.raises(ValueError, match="Unsupported crypto tunables"):
        build(
            CryptoStrategyConfig(symbols=SYMBOLS),
            {"cross": [CrossSectionalConfig()], "not_a_real_parameter": [1]},
            max_candidates=8,
        )


def test_a_searched_candidate_is_evaluated_with_its_own_parameters():
    """Each candidate must be judged on the parameters it carries."""

    slow = with_cross_config(
        CryptoStrategyConfig(symbols=SYMBOLS, timeframe="1Day"),
        CrossSectionalConfig(lookback=45, vol_lookback=45),
    )
    fast = with_cross_config(slow, CrossSectionalConfig(lookback=10, vol_lookback=10))
    assert isinstance(fast, CrossSectionalStrategyConfig)
    assert CrossSectionalMomentumStrategy(slow).cross_config.lookback == 45
    assert CrossSectionalMomentumStrategy(fast).cross_config.lookback == 10

    universe = _universe()
    backtest = CryptoBacktestConfig(
        start_date=universe["BTC/USD"].index[0].date(),
        end_date=universe["BTC/USD"].index[-1].date(),
    )
    slow_result = run_cross_sectional_backtest(
        universe, slow, CryptoRiskConfig(), backtest
    )
    fast_result = run_cross_sectional_backtest(
        universe, fast, CryptoRiskConfig(), backtest
    )
    assert slow_result.num_trades != fast_result.num_trades, (
        "both candidates produced identical books: the parameters were ignored"
    )


def test_walk_forward_searches_the_cross_axis_and_reports_what_it_searched():
    universe = _universe(periods=1_400)
    grid = {"cross": [CrossSectionalConfig(lookback=lb) for lb in (15, 30)]}
    result = run_cross_sectional_walk_forward(
        universe,
        CryptoStrategyConfig(symbols=SYMBOLS, timeframe="1Day"),
        CryptoRiskConfig(),
        CryptoBacktestConfig(
            start_date=universe["BTC/USD"].index[0].date(),
            end_date=universe["BTC/USD"].index[-1].date(),
        ),
        grid,
        CryptoWalkForwardConfig(
            train_bars=400, test_bars=150, step_bars=150, min_train_trades=1
        ),
        CryptoPromotionCriteria(),
    )
    assert result.candidate_count == 2
    assert result.protocol["decision_engine"] == "cross_sectional_momentum"
    assert [entry["lookback"] for entry in result.protocol["cross_sectional_grid"]] == [15, 30]
    assert result.protocol["cross_sectional_selected"]["lookback"] in (15, 30)
    for fold in result.folds:
        assert fold.selected_config.cross.lookback in (15, 30)


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError, match="cross-sectional"):
        CrossSectionalMomentumStrategy(
            CryptoStrategyConfig(symbols=SYMBOLS), CrossSectionalConfig(top_k=0)
        )
    assert any(
        "falling assets" in error
        for error in CrossSectionalConfig(min_score=-1.0).validate()
    )
    # A bad cross-sectional candidate must fail the combined config too, so the
    # audited builder's validity filter can drop it.
    assert CrossSectionalStrategyConfig(
        symbols=SYMBOLS, cross=CrossSectionalConfig(top_k=0)
    ).validate()
    assert not CrossSectionalStrategyConfig(symbols=SYMBOLS).validate()
    with pytest.raises(ValueError, match="CrossSectionalConfig instances"):
        _candidate_builder(CrossSectionalConfig())(
            CryptoStrategyConfig(symbols=SYMBOLS),
            {"cross": [{"lookback": 20}]},
            max_candidates=8,
        )
