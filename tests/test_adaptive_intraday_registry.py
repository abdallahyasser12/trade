from quant.intraday import make_high_velocity_strategy_config
from quant.strategies.intraday_adaptive import AdaptiveIntradayStrategy
from quant.strategies.registry import build_default_registry


def test_adaptive_v2_is_registered_with_isolated_version_identity():
    registry = build_default_registry()
    adaptive = registry.get("crypto_intraday_adaptive").metadata
    legacy = registry.get("crypto_intraday_momentum").metadata
    assert adaptive.version == "2.0.0"
    assert legacy.version == "1.0.0"
    assert adaptive.strategy_id != legacy.strategy_id
    assert "adaptive" in adaptive.tags
    assert "legacy" in legacy.tags


def test_registry_constructs_adaptive_strategy_on_high_velocity_config():
    registry = build_default_registry()
    config = make_high_velocity_strategy_config()
    strategy = registry.create("crypto_intraday_adaptive", config)
    assert isinstance(strategy, AdaptiveIntradayStrategy)
