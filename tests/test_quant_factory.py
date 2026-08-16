"""Walk-forward factory tests: fold mechanics, gates, and recording."""

import numpy as np
import pandas as pd
import pytest

from crypto_models import CryptoSignalDecision
from models import BotAction
from quant.research.factory import (
    QuantPromotionCriteria,
    QuantResearchReport,
    QuantWalkForwardConfig,
    record_research_report,
    run_quant_walk_forward,
)
from quant.research.simulator import SimulationConfig
from quant.store import QuantStore


class PeriodicStrategy:
    """Deterministic protocol strategy: enter every N bars, exit M later.

    The parameters are the 'candidate grid'; behavior is stateless per
    timestamp so walk-forward folds are exactly reproducible.
    """

    def __init__(self, entry_every=10, hold_bars=5, stop_atr=2.0):
        self.entry_every = entry_every
        self.hold_bars = hold_bars
        self.stop_atr = stop_atr
        self.config = type("Cfg", (), {"benchmark_symbol": "TEST", "symbols": ("TEST",)})()

    def required_bars(self) -> int:
        return 10

    def prepare_symbol(self, bars):
        out = bars.copy()
        out["atr"] = 1.0
        return out

    def prepare_universe(self, bars_by_symbol):
        return {s: self.prepare_symbol(f) for s, f in bars_by_symbol.items()}

    def evaluate(self, bars_by_symbol, held_symbols=frozenset()):
        prepared = self.prepare_universe(bars_by_symbol)
        common = None
        for frame in prepared.values():
            common = frame.index if common is None else common.intersection(frame.index)
        return self.evaluate_prepared(prepared, common[-1], held_symbols)

    def evaluate_prepared(self, prepared, timestamp, held_symbols=frozenset()):
        decisions = []
        for symbol, frame in prepared.items():
            position = int(frame.index.get_indexer([pd.Timestamp(timestamp)])[0])
            close = float(frame.iloc[position]["close"])
            if symbol in held_symbols:
                if position % self.entry_every == self.hold_bars:
                    action, reason = BotAction.EXIT, "periodic_exit"
                else:
                    action, reason = BotAction.HOLD, "holding"
                decisions.append(
                    CryptoSignalDecision(
                        symbol=symbol, action=action, timestamp=None, reason=reason,
                        close=close, atr=1.0,
                    )
                )
            elif position % self.entry_every == 0:
                decisions.append(
                    CryptoSignalDecision(
                        symbol=symbol, action=BotAction.BUY, timestamp=None,
                        reason="periodic_entry", close=close, atr=1.0,
                        stop_price=close - self.stop_atr,
                    )
                )
            else:
                decisions.append(
                    CryptoSignalDecision(
                        symbol=symbol, action=BotAction.HOLD, timestamp=None,
                        reason="waiting", close=close, atr=1.0,
                    )
                )
        return decisions

    def ratchet_stop(self, current_stop, highest, close, atr_value):
        return current_stop, max(highest, close)


def uptrend_bars(periods=520, seed=2):
    rng = np.random.default_rng(seed)
    index = pd.date_range("2023-01-01", periods=periods, freq="1h", tz="UTC")
    close = 100.0 * np.cumprod(1.0 + 0.001 + rng.normal(0, 0.001, periods))
    return pd.DataFrame(
        {
            "open": close,
            "high": close * 1.002,
            "low": close * 0.998,
            "close": close,
            "volume": np.full(periods, 1_000.0),
        },
        index=index,
    )


def factory_kwargs(**overrides):
    base = dict(
        strategy_id="periodic_test",
        strategy_factory=lambda params: PeriodicStrategy(**params),
        candidate_params=[{"entry_every": 10, "hold_bars": 5}, {"entry_every": 20, "hold_bars": 5}],
        bars_by_symbol={"TEST": uptrend_bars()},
        sim_config=SimulationConfig(
            slippage_bps=0.0, spread_bps=0.0, fee_bps=0.0, annualization_factor=8760.0
        ),
        walk_config=QuantWalkForwardConfig(
            train_bars=150, test_bars=80, step_bars=80, embargo_bars=2
        ),
        criteria=QuantPromotionCriteria(),
    )
    base.update(overrides)
    return base


def test_walk_forward_produces_folds_and_gates():
    report = run_quant_walk_forward(**factory_kwargs())
    assert isinstance(report, QuantResearchReport)
    assert len(report.folds) >= 3
    # Every gate individually visible with value/threshold/verdict.
    gate_names = {gate.name for gate in report.gates}
    assert {
        "min_folds",
        "min_oos_trades",
        "min_positive_fold_ratio",
        "min_compounded_return_pct",
        "min_benchmark_excess_return_pct",
        "min_aggregate_sharpe",
        "min_profit_factor",
        "max_fold_drawdown_pct",
        "min_stressed_return_pct",
        "min_deflated_sharpe_probability",
        "max_ruin_probability",
    } == gate_names
    assert report.passed == all(g.passed for g in report.gates)
    assert report.total_oos_trades > 0


def test_folds_are_chronological_with_embargo():
    report = run_quant_walk_forward(**factory_kwargs())
    for fold in report.folds:
        assert fold.train_end < fold.test_start
        # Embargo: at least embargo_bars strictly between train end and test start.
        assert fold.test_start > fold.train_end
    starts = [fold.test_start for fold in report.folds]
    assert starts == sorted(starts)


def test_oos_trades_confined_to_test_window():
    report = run_quant_walk_forward(**factory_kwargs())
    for fold in report.folds:
        for trade in fold.test_result.trades:
            entry = pd.Timestamp(trade.entry_time)
            assert entry >= fold.test_start
            assert entry <= fold.test_end


def test_stressed_costs_never_beat_base():
    kwargs = factory_kwargs()
    kwargs["walk_config"] = QuantWalkForwardConfig(
        train_bars=150, test_bars=80, step_bars=80, embargo_bars=2,
        stress_slippage_bps=50.0, stress_spread_bps=30.0,
    )
    report = run_quant_walk_forward(**kwargs)
    assert report.stressed_compounded_return_pct <= report.compounded_return_pct + 1e-9


def test_trial_count_includes_prior_trials():
    report = run_quant_walk_forward(**factory_kwargs(prior_trials=100))
    assert report.trial_count == 2 + 100


def test_candidate_cap_enforced():
    kwargs = factory_kwargs()
    kwargs["candidate_params"] = [{"entry_every": n} for n in range(10, 50)]
    kwargs["walk_config"] = QuantWalkForwardConfig(max_candidates=8)
    with pytest.raises(ValueError, match="exceed the configured cap"):
        run_quant_walk_forward(**kwargs)


def test_insufficient_history_raises():
    kwargs = factory_kwargs()
    kwargs["bars_by_symbol"] = {"TEST": uptrend_bars(periods=100)}
    with pytest.raises(ValueError, match="walk-forward fold"):
        run_quant_walk_forward(**kwargs)


def test_report_recording_accumulates_lineage():
    store = QuantStore(":memory:")
    report = run_quant_walk_forward(**factory_kwargs())
    run_id = record_research_report(store, report, git_commit="abc123", seed=7)
    assert run_id > 0
    runs = store.get_research_runs("periodic_test")
    assert len(runs) == 1
    assert runs[0]["git_commit"] == "abc123"
    assert store.total_trials("periodic_test") == 2
    # A second run keeps growing the recorded trial lineage.
    record_research_report(store, report)
    assert store.total_trials("periodic_test") == 4
    store.close()
