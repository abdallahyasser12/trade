import pytest

from quant.adaptive_manifest import (
    AdaptiveManifest,
    load_research_manifest,
    lock_deployment_manifest,
    locked_deployment_manifest,
    record_research_manifest,
)

COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
UNIVERSE = (
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
    "XRP/USD",
    "LINK/USD",
    "UNI/USD",
)
PARAMS = {
    "fast_lookback": 3,
    "medium_lookback": 12,
    "slow_lookback": 36,
    "vol_lookback": 48,
    "volume_lookback": 24,
    "breakout_lookback": 20,
    "reversion_lookback": 24,
    "trend_fast_span": 18,
    "trend_slow_span": 54,
    "atr_baseline_lookback": 72,
    "top_k": 4,
    "exit_rank_buffer": 2,
    "min_entry_score": 0.24,
    "exit_score": 0.08,
    "min_volume_ratio": 0.65,
    "max_atr_ratio": 2.75,
    "stop_atr_multiplier": 1.65,
    "trailing_atr_multiplier": 2.05,
    "stress_block_probability": 0.70,
    "min_trend_probability": 0.42,
}


class Store:
    def __init__(self):
        self.values = {}

    def set_kv(self, key, value):
        self.values[key] = value

    def get_kv(self, key):
        return self.values.get(key)


def manifest(run_id=7, **overrides):
    values = {
        "research_run_id": run_id,
        "git_commit": COMMIT,
        "timeframe": "5Min",
        "universe": UNIVERSE,
        "params": dict(PARAMS),
    }
    values.update(overrides)
    return AdaptiveManifest(**values).require_valid()


def test_manifest_round_trips_through_research_and_deployment_storage():
    store = Store()
    source = manifest()
    record_research_manifest(store, source)
    assert load_research_manifest(store, source.research_run_id) == source
    lock_deployment_manifest(store, source)
    assert locked_deployment_manifest(store) == source


def test_measured_cost_rerun_may_have_new_run_id_but_not_new_strategy():
    original = manifest(run_id=7)
    rerun = manifest(run_id=8)
    assert original.same_strategy(rerun)


def test_code_timeframe_universe_or_params_change_is_not_same_strategy():
    original = manifest()
    assert not original.same_strategy(manifest(run_id=8, git_commit=OTHER_COMMIT))
    assert not original.same_strategy(manifest(run_id=8, timeframe="15Min"))
    assert not original.same_strategy(
        manifest(run_id=8, universe=tuple(reversed(UNIVERSE)))
    )
    changed = dict(PARAMS)
    changed["min_entry_score"] = 0.32
    assert not original.same_strategy(manifest(run_id=8, params=changed))


def test_runtime_identity_fails_closed_on_git_or_cli_assertion_mismatch():
    source = manifest()
    source.assert_runtime_identity(
        git_commit=COMMIT,
        timeframe="5Min",
        universe=UNIVERSE,
    )
    with pytest.raises(ValueError, match="git commit"):
        source.assert_runtime_identity(git_commit=OTHER_COMMIT)
    with pytest.raises(ValueError, match="timeframe"):
        source.assert_runtime_identity(git_commit=COMMIT, timeframe="15Min")
    with pytest.raises(ValueError, match="universe"):
        source.assert_runtime_identity(
            git_commit=COMMIT,
            universe=tuple(reversed(UNIVERSE)),
        )


def test_manifest_rejects_partial_git_sha_or_invalid_adaptive_params():
    with pytest.raises(ValueError, match="git_commit"):
        manifest(git_commit="abc")
    bad = dict(PARAMS)
    bad["fast_lookback"] = 100
    with pytest.raises(ValueError, match="adaptive params"):
        manifest(params=bad)
