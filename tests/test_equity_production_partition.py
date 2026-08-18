from types import SimpleNamespace

import pytest

from equity_marketwide_production_engine import (
    EntryGatedShadowExecutor,
    partition_locked_universe,
)
from models import BotAction


def test_shadow_held_symbol_remains_managed_but_not_entry_eligible():
    locked = ("SPY", "AAPL", "MSFT", "NVDA")
    entry, management = partition_locked_universe(
        locked,
        ("SPY", "AAPL", "MSFT"),
        {"NVDA"},
    )
    assert entry == frozenset({"SPY", "AAPL", "MSFT"})
    assert management == locked


def test_collapsed_entry_breadth_still_manages_shadow_position():
    locked = ("SPY", "AAPL", "MSFT", "NVDA")
    entry, management = partition_locked_universe(
        locked,
        ("SPY", "AAPL"),
        {"NVDA", "UNRESEARCHED"},
    )
    assert entry == frozenset()
    assert management == ("SPY", "AAPL", "NVDA")
    assert "UNRESEARCHED" not in management


def test_management_partition_preserves_researched_order_and_validates_breadth():
    entry, management = partition_locked_universe(
        ("SPY", "MSFT", "AAPL"),
        ("AAPL", "SPY", "MSFT"),
        set(),
    )
    assert entry == frozenset({"SPY", "MSFT", "AAPL"})
    assert management == ("SPY", "MSFT", "AAPL")
    with pytest.raises(ValueError, match="positive"):
        partition_locked_universe(("SPY",), ("SPY",), set(), min_entry_breadth=0)


def test_shadow_executor_blocks_fresh_buy_outside_entry_set_but_preserves_exit():
    class Delegate:
        def held_symbols(self, strategy_id):
            return frozenset({"NVDA"}) if strategy_id == "strategy" else frozenset()

        def process_cycle(self, *, decisions, **kwargs):
            self.decisions = list(decisions)
            return {"opened": 0, "closed": 1}

    delegate = Delegate()
    wrapper = EntryGatedShadowExecutor(delegate, lambda: frozenset({"AAPL"}))
    buy_allowed = SimpleNamespace(
        symbol="AAPL",
        action=BotAction.BUY,
        reason="signal",
        filters={},
        details={},
    )
    buy_blocked = SimpleNamespace(
        symbol="NVDA",
        action=BotAction.BUY,
        reason="signal",
        filters={},
        details={},
    )
    exit_held = SimpleNamespace(
        symbol="NVDA",
        action=BotAction.EXIT,
        reason="exit",
        filters={},
        details={},
    )

    result = wrapper.process_cycle(
        decisions=[buy_allowed, buy_blocked, exit_held],
        strategy_id="strategy",
    )

    assert result == {"opened": 0, "closed": 1}
    assert buy_allowed.action is BotAction.BUY
    assert buy_blocked.action is BotAction.HOLD
    assert buy_blocked.reason.startswith("current_universe_entry_block:")
    assert buy_blocked.filters["current_entry_eligible"] is False
    assert exit_held.action is BotAction.EXIT
    assert wrapper.held_symbols("strategy") == frozenset({"NVDA"})
