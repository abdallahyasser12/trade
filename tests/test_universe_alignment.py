"""One late listing must not silently discard everyone else's history."""

import numpy as np
import pandas as pd
import pytest

from universe_alignment import (
    align_to_common_index,
    aligned_length,
    drop_limiting_symbols,
)


def _frame(index):
    return pd.DataFrame({"close": np.linspace(100.0, 120.0, len(index))}, index=index)


def _index(periods=2_000, start="2021-01-01"):
    return pd.date_range(start, periods=periods, freq="h", tz="UTC")


def test_aligned_length_is_the_intersection():
    full = _index()
    bars = {"BTC/USD": _frame(full), "ETH/USD": _frame(full[-500:])}
    assert aligned_length(bars) == 500
    assert aligned_length({}) == 0


def test_a_late_listing_is_dropped_and_the_window_recovered():
    full = _index()
    bars = {
        "BTC/USD": _frame(full),
        "ETH/USD": _frame(full),
        "SOL/USD": _frame(full),
        "LATE/USD": _frame(full[-500:]),
    }
    kept, dropped = drop_limiting_symbols(bars, "BTC/USD")
    assert [symbol for symbol, _ in dropped] == ["LATE/USD"]
    assert aligned_length(kept) == 2_000
    assert "recovers 1500 aligned bars" in dropped[0][1]


def test_selection_is_by_impact_not_by_start_date():
    """A symbol missing scattered days truncates just as much as a late one."""

    full = _index()
    bars = {
        "BTC/USD": _frame(full),
        "ETH/USD": _frame(full),
        "SOL/USD": _frame(full),
        "GAPPY/USD": _frame(full.delete(slice(100, 900))),
    }
    kept, dropped = drop_limiting_symbols(bars, "BTC/USD")
    assert [symbol for symbol, _ in dropped] == ["GAPPY/USD"]
    assert aligned_length(kept) == 2_000


def test_the_benchmark_survives_even_when_it_is_the_offender():
    full = _index()
    bars = {
        "BTC/USD": _frame(full[-500:]),
        "ETH/USD": _frame(full),
        "SOL/USD": _frame(full),
        "LTC/USD": _frame(full),
    }
    kept, _ = drop_limiting_symbols(bars, "BTC/USD")
    assert "BTC/USD" in kept


def test_the_universe_never_falls_below_the_floor():
    """A cross-sectional ranking over two names is not a ranking."""

    full = _index()
    bars = {
        "BTC/USD": _frame(full),
        "ETH/USD": _frame(full),
        "LATE/USD": _frame(full[-500:]),
    }
    kept, dropped = drop_limiting_symbols(bars, "BTC/USD")
    assert dropped == []
    assert set(kept) == set(bars)


def test_alignment_refuses_a_disjoint_universe():
    a = _index(periods=100, start="2021-01-01")
    b = _index(periods=100, start="2024-01-01")
    with pytest.raises(ValueError, match="no common timestamps"):
        align_to_common_index({"BTC/USD": _frame(a), "ETH/USD": _frame(b)})


def test_alignment_returns_identical_indices():
    full = _index()
    bars = {"BTC/USD": _frame(full), "ETH/USD": _frame(full[-500:])}
    aligned, common = align_to_common_index(bars)
    assert len(common) == 500
    for frame in aligned.values():
        assert frame.index.equals(common)
