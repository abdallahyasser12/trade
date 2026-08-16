"""VWAP-reversion strategy tests: stretch entries only in chop."""

import pandas as pd
import pytest

from models import BotAction
from quant.contracts import Market
from quant.strategies.vwap_reversion import VwapReversionConfig, VwapReversionStrategy


def compact_config(**overrides):
    base = dict(
        symbols=("BTC/USD",),
        market=Market.CRYPTO,
        timeframe="5Min",
        atr_period=5,
        atr_percentile_lookback=20,
        volume_period=5,
        max_atr_percentile=100.0,
        entry_z=-2.0,
        exit_z=0.0,
        max_efficiency_ratio=0.35,
        min_rsi=0.0,
        max_rsi=70.0,
        rsi_period=14,
        efficiency_period=20,
        min_session_bar=2,
    )
    base.update(overrides)
    return VwapReversionConfig(**base)


def make_bars(closes, start="2024-01-02 00:00", spread=0.4):
    index = pd.date_range(start=start, periods=len(closes), freq="5min", tz="UTC")
    frame = pd.DataFrame(
        {
            "open": closes,
            "high": [c + spread for c in closes],
            "low": [c - spread for c in closes],
            "close": closes,
            "volume": [1_000.0] * len(closes),
        },
        index=index,
        dtype=float,
    )
    return frame


def choppy_drift_down(flat=32, drift=28):
    """Sideways chop, then a slow zigzag drift below VWAP: net move small
    per bar, path long — exactly the low-efficiency stretch the strategy
    is hypothesized to buy."""
    closes = [100.0 + (0.2 if i % 2 else -0.2) for i in range(flat)]
    level = 100.0
    for i in range(drift):
        level -= 0.18
        closes.append(level + (0.35 if i % 2 else -0.35))
    return closes


def straight_line_down(flat=32, drop=12):
    """A deep stretch delivered as one efficient trend leg — the chop-only
    filter must refuse it (buying a clean downtrend is not a dislocation
    trade).  The leg is steep so the efficiency window exceeds the chop
    ceiling before the z-stretch first triggers."""
    closes = [100.0 + (0.2 if i % 2 else -0.2) for i in range(flat)]
    level = 100.0
    for _ in range(drop):
        level -= 0.9
        closes.append(level)
    return closes


def scan_for_buy(strategy, bars):
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decisions = []
    for position in range(strategy.required_bars() - 1, len(bars)):
        decision = strategy.evaluate_prepared(prepared, bars.index[position], set())[0]
        decisions.append(decision)
    return prepared, decisions


def test_choppy_stretch_generates_buy():
    strategy = VwapReversionStrategy(compact_config())
    bars = make_bars(choppy_drift_down())
    _, decisions = scan_for_buy(strategy, bars)
    buys = [d for d in decisions if d.action is BotAction.BUY]
    assert buys, "expected at least one stretch entry in the choppy drift"
    decision = buys[0]
    assert decision.reason == "vwap_overextension"
    assert decision.details["vwap_z"] <= -2.0
    assert 0 < decision.stop_price < decision.close


def test_established_downtrend_is_refused_by_chop_filter():
    # A causal efficiency window cannot identify a trend on its very first
    # bars (the window is still chop-dominated), so the guarantee under
    # test is: once the leg is established — efficiency above the ceiling —
    # every stretched bar is refused, and the refusals dominate.
    strategy = VwapReversionStrategy(compact_config())
    bars = make_bars(straight_line_down())
    _, decisions = scan_for_buy(strategy, bars)
    stretched = [d for d in decisions if d.filters and d.filters.get("stretch")]
    assert stretched
    blocked = [d for d in stretched if not d.filters.get("chop_only")]
    assert len(blocked) >= len(stretched) - 1, "chop filter must refuse the leg"
    for decision in blocked:
        assert decision.action is BotAction.HOLD
        assert "chop_only" in decision.details.get("failed_filters", [])


def test_held_position_exits_on_reversion():
    strategy = VwapReversionStrategy(compact_config())
    closes = choppy_drift_down()
    # Snap back above the running VWAP: reversion complete.
    closes.extend([101.0, 102.0])
    bars = make_bars(closes)
    prepared = strategy.prepare_universe({"BTC/USD": bars})
    decision = strategy.evaluate_prepared(prepared, bars.index[-1], {"BTC/USD"})[0]
    assert decision.action is BotAction.EXIT
    assert decision.reason == "reversion_complete"


def test_causality_under_future_appends():
    strategy = VwapReversionStrategy(compact_config())
    bars = make_bars(choppy_drift_down())
    prepared_full = strategy.prepare_universe({"BTC/USD": bars})
    cut = len(bars) - 6
    prepared_short = strategy.prepare_universe({"BTC/USD": bars.iloc[:cut]})
    ts = bars.index[cut - 1]
    full = strategy.evaluate_prepared(prepared_full, ts, set())[0]
    short = strategy.evaluate_prepared(prepared_short, ts, set())[0]
    assert full.action == short.action
    assert full.reason == short.reason


def test_immature_session_vwap_blocked():
    strategy = VwapReversionStrategy(compact_config(min_session_bar=10_000))
    bars = make_bars(choppy_drift_down())
    _, decisions = scan_for_buy(strategy, bars)
    assert not [d for d in decisions if d.action is BotAction.BUY]


def test_config_validation():
    assert compact_config(entry_z=1.0).validate()
    assert compact_config(exit_z=-5.0).validate()
    assert compact_config(max_efficiency_ratio=0.0).validate()
    assert not compact_config().validate()
