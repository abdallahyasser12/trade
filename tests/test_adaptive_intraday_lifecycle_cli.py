from scripts.manage_strategy_lifecycle import (
    ADAPTIVE_STATE_DB,
    _default_state_db,
    _requirements,
)


def test_adaptive_strategy_uses_high_velocity_promotion_requirements():
    req = _requirements("crypto_intraday_adaptive")
    assert req.min_research_trades >= 100
    assert req.min_shadow_trades >= 100
    assert req.min_paper_trades >= 200
    assert req.min_canary_trades >= 100
    assert req.min_measured_execution_samples >= 50


def test_adaptive_strategy_defaults_to_separate_execution_state_database():
    assert _default_state_db("crypto_intraday_adaptive") == ADAPTIVE_STATE_DB
    assert "adaptive" in ADAPTIVE_STATE_DB
