import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from quant.liquidity_universe import (
    ADAPTIVE_LOCKED_UNIVERSE_KEY,
    ADAPTIVE_USD_CANDIDATE_POOL,
    LiquiditySelectionConfig,
    lock_deployment_universe,
    locked_deployment_universe,
    require_locked_universe,
    score_quote,
    select_liquid_universe,
)

NOW = datetime(2026, 8, 18, 2, 0, tzinfo=timezone.utc)


def quote(*, bid=100.0, ask=100.1, bid_size=100.0, ask_size=100.0, age=1.0):
    return SimpleNamespace(
        timestamp=NOW - timedelta(seconds=age),
        bid_price=bid,
        ask_price=ask,
        bid_size=bid_size,
        ask_size=ask_size,
    )


class FakeStore:
    def __init__(self):
        self.values = {}

    def set_kv(self, key, value):
        self.values[key] = value

    def get_kv(self, key):
        return self.values.get(key)


def test_quote_score_rejects_stale_wide_and_thin_markets():
    config = LiquiditySelectionConfig(
        max_quote_age_seconds=10.0,
        max_spread_bps=30.0,
        min_side_depth_usd=500.0,
    )
    assert not score_quote("BTC/USD", quote(age=20), config, NOW).eligible
    assert not score_quote("BTC/USD", quote(bid=100, ask=101), config, NOW).eligible
    assert not score_quote(
        "BTC/USD",
        quote(bid_size=1.0, ask_size=1.0),
        config,
        NOW,
    ).eligible


def test_selector_keeps_benchmark_and_ranks_only_eligible_quotes():
    config = LiquiditySelectionConfig(
        max_symbols=6,
        min_symbols=6,
        max_quote_age_seconds=30.0,
        max_spread_bps=40.0,
        min_side_depth_usd=100.0,
    )
    quotes = {}
    good = (
        "BTC/USD",
        "ETH/USD",
        "SOL/USD",
        "XRP/USD",
        "LINK/USD",
        "UNI/USD",
        "AAVE/USD",
    )
    for index, symbol in enumerate(good):
        quotes[symbol] = quote(ask=100.05 + index * 0.005, age=index + 1)
    for symbol in ADAPTIVE_USD_CANDIDATE_POOL:
        quotes.setdefault(symbol, quote(age=120.0))

    selected, metrics = select_liquid_universe(quotes, config=config, now=NOW)
    assert len(selected) == 6
    assert selected[0] == "BTC/USD"
    assert set(selected).issubset(set(good))
    assert all(metric.symbol in ADAPTIVE_USD_CANDIDATE_POOL for metric in metrics)


def test_selector_fails_closed_when_btc_is_not_executable():
    config = LiquiditySelectionConfig(min_symbols=1, max_symbols=6)
    quotes = {symbol: quote() for symbol in ADAPTIVE_USD_CANDIDATE_POOL}
    quotes["BTC/USD"] = quote(age=999.0)
    with pytest.raises(ValueError, match="BTC/USD"):
        select_liquid_universe(quotes, config=config, now=NOW)


def test_locked_universe_round_trips_and_runtime_mismatch_is_rejected():
    store = FakeStore()
    universe = (
        "BTC/USD",
        "ETH/USD",
        "SOL/USD",
        "XRP/USD",
        "LINK/USD",
        "UNI/USD",
    )
    lock_deployment_universe(store, universe)
    assert ADAPTIVE_LOCKED_UNIVERSE_KEY in store.values
    assert json.loads(store.values[ADAPTIVE_LOCKED_UNIVERSE_KEY]) == list(universe)
    assert locked_deployment_universe(store) == universe
    assert require_locked_universe(store) == universe
    with pytest.raises(ValueError, match="does not exactly match"):
        require_locked_universe(store, tuple(reversed(universe)))


def test_runtime_requires_a_promoted_universe_lock():
    with pytest.raises(ValueError, match="not locked"):
        require_locked_universe(FakeStore())
