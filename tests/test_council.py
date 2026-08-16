"""The council must combine information, not amplify turnover.

Every test here encodes something that was measured rather than assumed. The
council was built, measured against its own members, found worse than all of
them, diagnosed, and fixed; these pin the diagnosis so it cannot regress.
"""

import numpy as np
import pandas as pd
import pytest

from council import (
    MEMBER_REGISTRY,
    CouncilConfig,
    CouncilStrategy,
    CouncilStrategyConfig,
    _rank_to_signed_unit,
)
from council_research import (
    _candidate_builder,
    run_council_backtest,
    with_council_config,
)
from crypto_config import CryptoBacktestConfig, CryptoRiskConfig, CryptoStrategyConfig
from models import BotAction

SYMBOLS = ("BTC/USD", "ETH/USD", "LTC/USD", "DOGE/USD")


def _bars(periods=900, seed=0, drift=0.001, vol=0.02):
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


def _universe(periods=900):
    return {s: _bars(periods=periods, seed=i) for i, s in enumerate(SYMBOLS)}


def _strategy(**overrides):
    return CouncilStrategy(
        CryptoStrategyConfig(symbols=SYMBOLS, timeframe="1Day"),
        CouncilConfig(**overrides),
    )


def test_member_scores_are_rank_normalised_not_summed_raw():
    """Members score on incompatible scales; summing them weights by units.

    cross-sectional spans about [-2, +4] and breadth about [-1.4, +6.2]. Raw
    summation would weight breadth roughly twice as heavily for no reason but
    its scale - the same dimensional-incoherence defect that sank the original
    strategy.
    """

    wide = pd.DataFrame(
        {"a": [1.0, 100.0], "b": [2.0, 200.0], "c": [3.0, 300.0], "d": [4.0, 400.0]}
    )
    scaled = _rank_to_signed_unit(wide)
    # Identical ordering, wildly different magnitudes, identical output.
    assert scaled.iloc[0].tolist() == scaled.iloc[1].tolist()
    assert scaled.iloc[0].min() == -1.0 and scaled.iloc[0].max() == 1.0


def test_an_abstaining_member_does_not_vote_zero():
    """NaN must mean 'no opinion', never 'neutral opinion'."""

    wide = pd.DataFrame({"a": [1.0], "b": [np.nan], "c": [3.0]})
    scaled = _rank_to_signed_unit(wide)
    assert pd.isna(scaled.loc[0, "b"])
    assert not pd.isna(scaled.loc[0, "a"])


def test_a_single_eligible_symbol_ranks_neutral_not_best():
    """With one scorable symbol the ranking carries no information."""

    wide = pd.DataFrame({"a": [1.0], "b": [np.nan], "c": [np.nan]})
    assert _rank_to_signed_unit(wide).loc[0, "a"] == 0.0


def test_below_quorum_the_council_has_no_opinion():
    """Fail closed: better no trade than a trade on one member's view."""

    strategy = _strategy(min_voters=4)
    prepared = strategy.prepare_universe(_universe())
    frame = prepared["BTC/USD"]
    early = frame.index[50]  # most members still warming up
    assert pd.isna(frame.loc[early, "council_score"])
    decision = next(
        d for d in strategy.evaluate_prepared(prepared, early) if d.symbol == "BTC/USD"
    )
    assert decision.action is BotAction.HOLD


def test_exit_threshold_must_sit_below_entry_threshold():
    """Without the gap the book churns across the boundary at 75bps a trip."""

    errors = CouncilConfig(entry_threshold=0.1, exit_threshold=0.1).validate()
    assert any("strictly below" in e for e in errors)
    assert not CouncilConfig(entry_threshold=0.3, exit_threshold=-0.1).validate()


def test_the_single_member_veto_is_not_the_default():
    """Measured: it turned the best configuration into the worst.

    With four decorrelated members P(at least one issues EXIT) is 94.4%, so a
    one-member veto makes holding nearly impossible. It drove 198 trades
    against the best member's 69 and collapsed profit factor 3.49 -> 0.51.
    """

    assert CouncilConfig().min_vetoers == 0


def test_a_veto_quorum_larger_than_the_council_is_rejected():
    """It could never fire, so it is a silently dead safety control."""

    errors = CouncilConfig(min_vetoers=99).validate()
    assert any("never fire" in e for e in errors)


def test_unanimity_is_reachable_as_a_setting_even_though_it_never_trades():
    """Kept selectable so the search can demonstrate the problem, not assume it."""

    strategy = _strategy(vote_mode="unanimous")
    assert strategy._required_proposers() == len(CouncilConfig().members)
    assert _strategy(vote_mode="majority")._required_proposers() == 3
    assert _strategy(min_proposers=1)._required_proposers() == 1


def test_scores_are_causal_under_truncation():
    """Truncating history must not change any already-computed council score."""

    strategy = _strategy()
    universe = _universe()
    full = strategy.prepare_universe(universe)
    cutoff = universe["BTC/USD"].index[-100]
    truncated = strategy.prepare_universe(
        {s: f.loc[:cutoff] for s, f in universe.items()}
    )
    for symbol in SYMBOLS:
        a = full[symbol].loc[:cutoff, "council_score"].dropna()
        b = truncated[symbol].loc[:cutoff, "council_score"].dropna()
        assert len(a) == len(b)
        assert np.allclose(a.to_numpy(float), b.to_numpy(float))


def test_ranks_are_a_permutation_of_the_scorable_universe():
    strategy = _strategy()
    prepared = strategy.prepare_universe(_universe())
    last = prepared["BTC/USD"].index[-1]
    ranks = sorted(prepared[s].loc[last, "council_rank"] for s in SYMBOLS)
    assert ranks == [1.0, 2.0, 3.0, 4.0]


def test_required_bars_is_gated_by_the_slowest_member():
    """Anything less starves a voter and silently shrinks the council."""

    strategy = _strategy()
    assert strategy.required_bars() == max(
        m.required_bars() for m in strategy.members.values()
    )


def test_one_decision_per_symbol_with_buys_ordered_first():
    strategy = _strategy(min_proposers=1)
    prepared = strategy.prepare_universe(_universe())
    last = prepared["BTC/USD"].index[-1]
    decisions = strategy.evaluate_prepared(prepared, last)
    assert [d.symbol for d in decisions].count("BTC/USD") == 1
    assert len(decisions) == len(SYMBOLS)
    actions = [d.action is BotAction.BUY for d in decisions]
    assert actions == sorted(actions, reverse=True), "BUYs must sort first"


def test_a_decision_records_who_proposed_and_who_vetoed():
    """The council must be auditable after the fact, not a black box."""

    strategy = _strategy(min_proposers=1)
    prepared = strategy.prepare_universe(_universe())
    last = prepared["BTC/USD"].index[-1]
    decision = next(
        d for d in strategy.evaluate_prepared(prepared, last) if d.symbol == "BTC/USD"
    )
    assert "proposers" in decision.details and "vetoes" in decision.details
    assert isinstance(decision.details["proposers"], list)


def test_invalid_configurations_are_rejected():
    assert any("distinct" in e for e in CouncilConfig(members=("cross", "cross")).validate())
    assert any("Unknown" in e for e in CouncilConfig(members=("nope",)).validate())
    assert any("min_proposers" in e for e in CouncilConfig(min_proposers=99).validate())
    assert any("Vote mode" in e for e in CouncilConfig(vote_mode="telepathy").validate())
    assert any("weights" in e for e in CouncilConfig(weights=(1.0,)).validate())
    with pytest.raises(ValueError, match="council"):
        CouncilStrategy(
            CryptoStrategyConfig(symbols=SYMBOLS), CouncilConfig(vote_mode="telepathy")
        )


def test_council_candidates_are_distinct_so_the_search_is_real():
    """The no-op-search defect, guarded for the council too."""

    build = _candidate_builder(CouncilConfig())
    candidates = build(
        CryptoStrategyConfig(symbols=SYMBOLS),
        {"council": [CouncilConfig(min_proposers=1), CouncilConfig(min_proposers=3)]},
        max_candidates=32,
    )
    assert len(candidates) == 2
    assert len({repr(sorted(c.to_dict().items())) for c in candidates}) == 2
    assert {c.council.min_proposers for c in candidates} == {1, 3}


def test_the_candidate_cap_still_binds_and_the_allowlist_is_enforced():
    build = _candidate_builder(CouncilConfig())
    with pytest.raises(ValueError, match="maximum is"):
        build(
            CryptoStrategyConfig(symbols=SYMBOLS),
            {
                "council": [CouncilConfig(top_k=k) for k in (1, 2, 3)],
                "atr_stop_multiplier": [2.0, 2.5, 3.0],
            },
            max_candidates=8,
        )
    with pytest.raises(ValueError, match="Unsupported crypto tunables"):
        build(
            CryptoStrategyConfig(symbols=SYMBOLS),
            {"council": [CouncilConfig()], "not_real": [1]},
            max_candidates=8,
        )


def test_a_searched_candidate_is_evaluated_with_its_own_council():
    loose = with_council_config(
        CryptoStrategyConfig(symbols=SYMBOLS, timeframe="1Day"),
        CouncilConfig(min_proposers=1),
    )
    strict = with_council_config(loose, CouncilConfig(min_proposers=4))
    assert isinstance(strict, CouncilStrategyConfig)
    assert CouncilStrategy(loose).council_config.min_proposers == 1
    assert CouncilStrategy(strict).council_config.min_proposers == 4

    universe = _universe()
    backtest = CryptoBacktestConfig(
        start_date=universe["BTC/USD"].index[0].date(),
        end_date=universe["BTC/USD"].index[-1].date(),
    )
    loose_result = run_council_backtest(universe, loose, CryptoRiskConfig(), backtest)
    strict_result = run_council_backtest(universe, strict, CryptoRiskConfig(), backtest)
    assert loose_result.num_trades > strict_result.num_trades, (
        "demanding more agreement must not trade more"
    )


def test_every_registered_member_can_actually_be_constructed():
    """A registry entry that cannot be built is a landmine for the search."""

    config = CryptoStrategyConfig(symbols=SYMBOLS, timeframe="1Day")
    for name in MEMBER_REGISTRY:
        strategy = CouncilStrategy(
            config,
            CouncilConfig(members=(name,), min_voters=1, min_proposers=1),
        )
        assert name in strategy.members
        assert strategy.required_bars() > 0
