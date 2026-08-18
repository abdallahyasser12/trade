import pytest

from quant.equity_manifest import (
    EQUITY_STRATEGY_VERSION,
    EquityDeploymentManifest,
    lock_deployment_manifest,
    require_common_locked_universe,
)
from quant.equity_marketwide import MarketWideEquityConfig

SHA = "a" * 40


class MemoryStore:
    def __init__(self):
        self.values = {}

    def set_kv(self, key, value):
        self.values[key] = value

    def get_kv(self, key):
        return self.values.get(key)


def manifest(strategy_id, universe=("AAPL", "MSFT", "SPY"), config=None):
    cfg = config or MarketWideEquityConfig(
        timeframe="15Min",
        max_scan_symbols=48,
        shard_size=48,
    )
    return EquityDeploymentManifest(
        research_run_id=7,
        git_commit=SHA,
        strategy_id=strategy_id,
        strategy_version=EQUITY_STRATEGY_VERSION,
        timeframe=cfg.timeframe,
        universe=tuple(universe),
        runtime_config=cfg.to_dict(),
    )


def test_manifest_round_trip_and_current_selection_is_restrictive_only():
    original = manifest("opening_range_breakout")
    restored = EquityDeploymentManifest.from_json(original.to_json())
    assert restored == original
    assert restored.restrict_current_selection(("NVDA", "SPY", "AAPL")) == ("AAPL", "SPY")


def test_common_locked_universe_requires_exact_cross_strategy_identity():
    store = MemoryStore()
    cfg = MarketWideEquityConfig(timeframe="15Min", max_scan_symbols=48, shard_size=48)
    for strategy_id in ("opening_range_breakout", "vwap_reversion"):
        lock_deployment_manifest(store, manifest(strategy_id, config=cfg))
    assert require_common_locked_universe(
        store,
        {"opening_range_breakout", "vwap_reversion"},
        runtime_config=cfg,
        git_commit=SHA,
    ) == ("AAPL", "MSFT", "SPY")


def test_common_locked_universe_rejects_commit_config_and_universe_drift():
    store = MemoryStore()
    cfg = MarketWideEquityConfig(timeframe="15Min", max_scan_symbols=48, shard_size=48)
    lock_deployment_manifest(store, manifest("opening_range_breakout", config=cfg))
    lock_deployment_manifest(
        store,
        manifest("vwap_reversion", universe=("AAPL", "NVDA", "SPY"), config=cfg),
    )
    with pytest.raises(ValueError, match="different locked universes"):
        require_common_locked_universe(
            store,
            {"opening_range_breakout", "vwap_reversion"},
            runtime_config=cfg,
            git_commit=SHA,
        )

    with pytest.raises(ValueError, match="Runtime git commit"):
        require_common_locked_universe(
            store,
            {"opening_range_breakout"},
            runtime_config=cfg,
            git_commit="b" * 40,
        )

    changed = MarketWideEquityConfig(
        timeframe="15Min",
        max_scan_symbols=48,
        shard_size=48,
        risk_per_trade_pct=0.10,
    )
    with pytest.raises(ValueError, match="configuration differs"):
        require_common_locked_universe(
            store,
            {"opening_range_breakout"},
            runtime_config=changed,
            git_commit=SHA,
        )
