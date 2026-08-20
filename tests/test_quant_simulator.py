"""Execution-semantics tests for the quant research simulator.

A scripted stub strategy drives exact scenarios so every fill rule
(next-bar open execution, gap-through-stop, same-bar entry protection,
management-instruction timing, MFE/MAE) is checked against hand-computed
numbers.
"""

import math

import numpy as np
import pandas as pd
import pytest

from crypto_models import CryptoSignalDecision
from models import BotAction
from quant.exits import ExitPolicyConfig
from quant.research.simulator import SimulationConfig, run_simulation


def make_bars(rows, start="2024-01-01", freq="1h"):
    """rows: list of (open, high, low, close) tuples; volume constant."""
    index = pd.date_range(start=start, periods=len(rows), freq=freq, tz="UTC")
    frame = pd.DataFrame(rows, columns=["open", "high", "low", "close"], dtype=float)
    frame["volume"] = 1_000.0
    frame.index = index
    return frame


class ScriptedStrategy:
    """UniverseStrategy stub: emits scripted decisions at index positions."""

    def __init__(self, script, warmup=3, atr_value=2.0):
        # script: {(symbol, position): (action, close, stop_price, reason)}
        self.script = script
        self.warmup = warmup
        self.atr_value = atr_value
        self.symbols = sorted({symbol for symbol, _ in script}) or ["TEST"]

    def required_bars(self):
        return self.warmup

    def prepare_symbol(self, bars):
        out = bars.copy()
        out["atr"] = self.atr_value
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
            entry = self.script.get((symbol, position))
            if entry is None:
                decisions.append(
                    CryptoSignalDecision(
                        symbol=symbol, action=BotAction.HOLD, timestamp=None, reason="scripted_hold"
                    )
                )
                continue
            action, close, stop, reason = entry
            decisions.append(
                CryptoSignalDecision(
                    symbol=symbol,
                    action=action,
                    timestamp=None,
                    reason=reason,
                    close=close,
                    stop_price=stop,
                    atr=self.atr_value,
                )
            )
        return decisions

    def ratchet_stop(self, current_stop, highest_completed_close, completed_close, current_atr):
        return current_stop, max(highest_completed_close, completed_close)


def frictionless_config(**overrides):
    base = dict(
        initial_capital=10_000.0,
        risk_per_trade_pct=1.0,
        slippage_bps=0.0,
        spread_bps=0.0,
        fee_bps=0.0,
        max_positions=2,
        annualization_factor=8760.0,
    )
    base.update(overrides)
    return SimulationConfig(**base)


def no_trailing_policy():
    return ExitPolicyConfig(enable_trailing_stop=False)


def test_entry_fills_at_next_bar_open_with_costs():
    rows = [(100, 101, 99, 100)] * 4 + [
        (100, 101, 99, 100),   # bar 4: scripted BUY close=100 stop=95
        (102, 103, 101, 102),  # bar 5: entry fills at open 102
        (103, 104, 102, 103),
        (103, 104, 102, 103),
    ]
    bars = make_bars(rows)
    strategy = ScriptedStrategy({("TEST", 4): (BotAction.BUY, 100.0, 95.0, "scripted_buy")})
    config = frictionless_config(slippage_bps=100.0)  # 1% slippage for visibility
    result = run_simulation(
        strategy, {"TEST": bars}, config=config, exit_policy=no_trailing_policy()
    )
    assert result.entries_taken == 1
    trade = result.trades[0]
    assert trade.entry_price == pytest.approx(102.0 * 1.01)
    # Stop preserves the signal's distance (100 - 95 = 5) below the fill.
    assert trade.initial_stop_price == pytest.approx(trade.entry_price - 5.0)
    # Position sizing: 1% of 10k = 100 risk, distance 5 -> 20 units.
    assert trade.qty == pytest.approx(100.0 / 5.0)


def test_stop_hit_intrabar_fills_at_stop():
    rows = [(100, 101, 99, 100)] * 4 + [
        (100, 101, 99, 100),   # bar 4: BUY signal, stop distance 5
        (100, 101, 99, 100),   # bar 5: entry at open 100, stop 95
        (100, 100, 90, 96),    # bar 6: low 90 pierces stop 95 -> fill at 95
        (96, 97, 95, 96),
    ]
    bars = make_bars(rows)
    strategy = ScriptedStrategy({("TEST", 4): (BotAction.BUY, 100.0, 95.0, "buy")})
    result = run_simulation(
        strategy, {"TEST": bars}, config=frictionless_config(), exit_policy=no_trailing_policy()
    )
    trade = result.trades[0]
    assert trade.is_closed
    assert trade.exit_legs[0].reason == "stop_hit"
    assert trade.exit_legs[0].price == pytest.approx(95.0)
    assert trade.net_pnl == pytest.approx((95.0 - 100.0) * trade.qty)
    assert trade.r_multiple == pytest.approx(-1.0)


def test_gap_below_stop_fills_at_open_not_stop():
    rows = [(100, 101, 99, 100)] * 4 + [
        (100, 101, 99, 100),   # BUY signal
        (100, 101, 99, 100),   # entry at 100, stop 95
        (90, 91, 89, 90),      # gap open 90 < stop 95 -> fill at the open
        (90, 91, 89, 90),
    ]
    bars = make_bars(rows)
    strategy = ScriptedStrategy({("TEST", 4): (BotAction.BUY, 100.0, 95.0, "buy")})
    result = run_simulation(
        strategy, {"TEST": bars}, config=frictionless_config(), exit_policy=no_trailing_policy()
    )
    trade = result.trades[0]
    assert trade.exit_legs[0].reason == "stop_gap"
    assert trade.exit_legs[0].price == pytest.approx(90.0)
    assert trade.r_multiple == pytest.approx(-2.0)


def test_same_bar_entry_protection():
    rows = [(100, 101, 99, 100)] * 4 + [
        (100, 101, 99, 100),   # BUY signal, stop distance 5
        (100, 101, 94, 96),    # entry at 100, low 94 pierces stop 95 same bar
        (96, 97, 95, 96),
        (96, 97, 95, 96),
    ]
    bars = make_bars(rows)
    strategy = ScriptedStrategy({("TEST", 4): (BotAction.BUY, 100.0, 95.0, "buy")})
    result = run_simulation(
        strategy, {"TEST": bars}, config=frictionless_config(), exit_policy=no_trailing_policy()
    )
    trade = result.trades[0]
    assert trade.exit_legs[0].reason == "stop_hit_entry_bar"
    assert trade.exit_legs[0].price == pytest.approx(95.0)


def test_exit_signal_executes_next_open():
    rows = [(100, 101, 99, 100)] * 4 + [
        (100, 101, 99, 100),   # BUY signal
        (100, 101, 99, 100),   # entry at 100
        (101, 102, 100, 101),  # scripted EXIT on this completed bar
        (103, 104, 102, 103),  # exit fills at open 103
        (103, 104, 102, 103),
    ]
    bars = make_bars(rows)
    strategy = ScriptedStrategy(
        {
            ("TEST", 4): (BotAction.BUY, 100.0, 95.0, "buy"),
            ("TEST", 6): (BotAction.EXIT, 101.0, None, "scripted_exit"),
        }
    )
    result = run_simulation(
        strategy, {"TEST": bars}, config=frictionless_config(), exit_policy=no_trailing_policy()
    )
    trade = result.trades[0]
    assert trade.exit_legs[0].reason == "scripted_exit"
    assert trade.exit_legs[0].price == pytest.approx(103.0)
    assert trade.r_multiple == pytest.approx(3.0 / 5.0)


def test_mfe_mae_measured_on_bars_while_open():
    rows = [(100, 101, 99, 100)] * 4 + [
        (100, 101, 99, 100),    # BUY signal
        (100, 101, 98, 100),    # entry bar: high 101, low 98
        (100, 110, 97, 105),    # high 110, low 97
        (105, 106, 104, 105),   # scripted EXIT evaluated here
        (105, 106, 104, 105),   # exit at open
    ]
    bars = make_bars(rows)
    strategy = ScriptedStrategy(
        {
            ("TEST", 4): (BotAction.BUY, 100.0, 90.0, "buy"),
            ("TEST", 7): (BotAction.EXIT, 105.0, None, "exit"),
        }
    )
    result = run_simulation(
        strategy, {"TEST": bars}, config=frictionless_config(), exit_policy=no_trailing_policy()
    )
    trade = result.trades[0]
    assert trade.mfe_pct == pytest.approx((110.0 - 100.0) / 100.0 * 100.0)
    assert trade.mae_pct == pytest.approx((100.0 - 97.0) / 100.0 * 100.0)


def test_max_positions_rejection_counted():
    rows = [(100, 101, 99, 100)] * 10
    bars_a = make_bars(rows)
    strategy = ScriptedStrategy(
        {
            ("AAA", 4): (BotAction.BUY, 100.0, 95.0, "buy"),
            ("BBB", 4): (BotAction.BUY, 100.0, 95.0, "buy"),
        }
    )
    result = run_simulation(
        strategy,
        {"AAA": bars_a, "BBB": bars_a.copy()},
        config=frictionless_config(max_positions=1),
        exit_policy=no_trailing_policy(),
    )
    assert result.entries_taken == 1
    assert result.entries_rejected.get("max_positions") == 1


def test_signal_on_final_bar_never_executes():
    rows = [(100, 101, 99, 100)] * 6
    bars = make_bars(rows)
    strategy = ScriptedStrategy({("TEST", 5): (BotAction.BUY, 100.0, 95.0, "buy")})
    result = run_simulation(
        strategy, {"TEST": bars}, config=frictionless_config(), exit_policy=no_trailing_policy()
    )
    assert result.signals_seen == 1
    assert result.entries_taken == 0
    assert result.num_trades == 0


def test_equity_accounting_reconciles():
    rows = [(100, 101, 99, 100)] * 4 + [
        (100, 101, 99, 100),
        (100, 101, 99, 100),   # entry at 100
        (100, 100, 90, 96),    # stop out at 95
        (96, 97, 95, 96),
    ]
    bars = make_bars(rows)
    strategy = ScriptedStrategy({("TEST", 4): (BotAction.BUY, 100.0, 95.0, "buy")})
    config = frictionless_config(fee_bps=10.0)
    result = run_simulation(
        strategy, {"TEST": bars}, config=config, exit_policy=no_trailing_policy()
    )
    trade = result.trades[0]
    assert result.ending_equity == pytest.approx(
        config.initial_capital + trade.net_pnl, rel=1e-9
    )
    assert result.total_fees == pytest.approx(trade.fees_total)


def test_partial_take_produces_two_legs():
    # Entry then a +2R bar triggers the 50% partial (policy), which fills
    # on the following open; the rest exits via scripted EXIT.
    rows = [(100, 101, 99, 100)] * 4 + [
        (100, 101, 99, 100),    # BUY signal, stop distance 5
        (100, 101, 99, 100),    # entry at 100, stop 95
        (110, 111, 109, 110),   # completed close 110 = +2R -> partial queued
        (111, 112, 110, 111),   # partial fills at open 111
        (111, 112, 110, 111),   # scripted EXIT here
        (112, 113, 111, 112),   # remainder exits at open 112
        (112, 113, 111, 112),
    ]
    bars = make_bars(rows)
    strategy = ScriptedStrategy(
        {
            ("TEST", 4): (BotAction.BUY, 100.0, 95.0, "buy"),
            ("TEST", 8): (BotAction.EXIT, 111.0, None, "exit"),
        }
    )
    policy = ExitPolicyConfig(
        enable_trailing_stop=False,
        enable_partial_take=True,
        partial_take_r=2.0,
        partial_take_fraction=0.5,
    )
    result = run_simulation(
        strategy, {"TEST": bars}, config=frictionless_config(), exit_policy=policy
    )
    trade = result.trades[0]
    assert len(trade.exit_legs) == 2
    first, second = trade.exit_legs
    assert first.reason == "partial_profit_take"
    assert first.price == pytest.approx(111.0)
    assert first.qty == pytest.approx(trade.qty * 0.5)
    assert second.reason == "exit"
    assert second.price == pytest.approx(112.0)
    assert trade.is_closed


def test_benchmark_curve_is_buy_and_hold():
    rows = [(100, 101, 99, 100 + i) for i in range(10)]
    bars = make_bars(rows)
    strategy = ScriptedStrategy({}, warmup=3)
    strategy.symbols = ["TEST"]
    result = run_simulation(
        strategy, {"TEST": bars}, config=frictionless_config(), exit_policy=no_trailing_policy()
    )
    closes = bars["close"].iloc[2:]  # evaluated window starts at warmup-1
    expected = closes.iloc[-1] / closes.iloc[0] - 1.0
    assert result.benchmark_return_pct == pytest.approx(expected * 100.0)


def test_insufficient_history_raises():
    rows = [(100, 101, 99, 100)] * 3
    bars = make_bars(rows)
    strategy = ScriptedStrategy({}, warmup=10)
    with pytest.raises(ValueError, match="aligned history"):
        run_simulation(
            strategy, {"TEST": bars}, config=frictionless_config(), exit_policy=no_trailing_policy()
        )


def test_result_r_multiples_are_finite():
    rows = [(100, 101, 99, 100)] * 4 + [
        (100, 101, 99, 100),
        (100, 101, 99, 100),
        (100, 100, 90, 96),
        (96, 97, 95, 96),
    ]
    bars = make_bars(rows)
    strategy = ScriptedStrategy({("TEST", 4): (BotAction.BUY, 100.0, 95.0, "buy")})
    result = run_simulation(
        strategy, {"TEST": bars}, config=frictionless_config(), exit_policy=no_trailing_policy()
    )
    assert result.r_multiples
    assert all(math.isfinite(r) for r in result.r_multiples)
    assert isinstance(result.config_snapshot["exit_policy"], dict)


def test_profit_target_full_exit_fills_next_open():
    # Entry, then a +2R completed close queues the full profit-target exit,
    # which fills on the following open.
    rows = [(100, 101, 99, 100)] * 4 + [
        (100, 101, 99, 100),    # BUY signal, stop distance 5
        (100, 101, 99, 100),    # entry at 100, stop 95
        (110, 111, 109, 110),   # completed close 110 = +2R -> target queued
        (111, 112, 110, 111),   # exit fills at open 111
        (111, 112, 110, 111),
    ]
    bars = make_bars(rows)
    strategy = ScriptedStrategy({("TEST", 4): (BotAction.BUY, 100.0, 95.0, "buy")})
    policy = ExitPolicyConfig(
        enable_trailing_stop=False,
        enable_profit_target=True,
        profit_target_r=1.5,
        profit_target_fraction=1.0,
    )
    result = run_simulation(
        strategy, {"TEST": bars}, config=frictionless_config(), exit_policy=policy
    )
    trade = result.trades[0]
    assert len(trade.exit_legs) == 1
    assert trade.exit_legs[0].reason == "profit_target"
    assert trade.exit_legs[0].price == pytest.approx(111.0)
    assert trade.is_closed


def test_scalp_policy_runs_with_hold_wiring_and_journals_config():
    # The scalp preset enables hold-consuming behaviors; with tiny scripted
    # frames the hold assessment is neutral (unmeasurable), so the wiring must
    # be exercised without changing hold-independent behavior or crashing.
    from quant.exits import scalp_exit_policy

    rows = [(100, 101, 99, 100)] * 4 + [
        (100, 101, 99, 100),
        (100, 101, 99, 100),    # entry at 100, stop 95
        (107, 108, 106, 107),   # +1.4R: above 1.2R target -> partial queued
        (108, 109, 107, 108),   # partial fills at open 108
        (108, 109, 107, 108),
        (108, 109, 107, 108),
    ]
    bars = make_bars(rows)
    strategy = ScriptedStrategy({("TEST", 4): (BotAction.BUY, 100.0, 95.0, "buy")})
    result = run_simulation(
        strategy, {"TEST": bars}, config=frictionless_config(), exit_policy=scalp_exit_policy()
    )
    trade = result.trades[0]
    target_legs = [leg for leg in trade.exit_legs if leg.reason == "profit_target"]
    assert len(target_legs) == 1
    assert target_legs[0].qty == pytest.approx(trade.qty * 0.6)
    assert result.config_snapshot["hold_intelligence"] is not None
