from types import SimpleNamespace

from scripts.run_adaptive_intraday_research import _add_robustness_gates


def report():
    return SimpleNamespace(gates=[], protocol={}, passed=True)


def good_diagnostics():
    return {
        "trades_per_day": 8.0,
        "expectancy_r": 0.05,
        "active_symbols": 8,
        "min_active_symbols": 6,
        "symbol_profit_concentration": 0.30,
        "fold_profit_concentration": 0.25,
    }


def test_adaptive_extra_research_gates_pass_only_for_broad_positive_activity():
    r = report()
    _add_robustness_gates(r, good_diagnostics())
    assert r.passed is True
    assert all(gate.passed for gate in r.gates)
    assert {gate.name for gate in r.gates} >= {
        "min_trades_per_day",
        "max_trades_per_day",
        "min_oos_expectancy_r",
        "min_active_symbols",
        "max_symbol_profit_concentration",
        "max_fold_profit_concentration",
    }


def test_too_few_trades_fails_even_when_returns_otherwise_look_good():
    diagnostics = good_diagnostics()
    diagnostics["trades_per_day"] = 5.99
    r = report()
    _add_robustness_gates(r, diagnostics)
    assert r.passed is False
    gate = next(g for g in r.gates if g.name == "min_trades_per_day")
    assert gate.passed is False


def test_non_positive_expectancy_cannot_pass_on_trade_count_alone():
    diagnostics = good_diagnostics()
    diagnostics["expectancy_r"] = 0.0
    r = report()
    _add_robustness_gates(r, diagnostics)
    assert r.passed is False
    gate = next(g for g in r.gates if g.name == "min_oos_expectancy_r")
    assert gate.passed is False


def test_one_symbol_explaining_most_profit_fails_concentration_gate():
    diagnostics = good_diagnostics()
    diagnostics["symbol_profit_concentration"] = 0.80
    r = report()
    _add_robustness_gates(r, diagnostics)
    assert r.passed is False
