import pytest

from equity_marketwide_production_engine import partition_locked_universe


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
