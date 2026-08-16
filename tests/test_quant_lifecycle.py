"""Lifecycle ladder, operator-ack enforcement, and degradation tests."""

import pytest

from quant.contracts import StrategyHealth, StrategyState
from quant.lifecycle import (
    DegradationMonitor,
    DegradationThresholds,
    LifecycleManager,
    PromotionEvidence,
    PromotionRequirements,
)
from quant.store import QuantStore

SID = "donchian_trend"


@pytest.fixture()
def store():
    s = QuantStore(":memory:")
    yield s
    s.close()


@pytest.fixture()
def manager(store):
    return LifecycleManager(
        store,
        PromotionRequirements(
            min_research_trades=10,
            min_shadow_trades=5,
            min_paper_trades=5,
            min_measured_execution_samples=5,
            min_canary_trades=5,
        ),
    )


def health(trades=20, drawdown=2.0, state=StrategyState.SHADOW):
    return StrategyHealth(
        strategy_id=SID,
        state=state,
        trades=trades,
        wins=12,
        losses=trades - 12,
        hit_rate=0.6,
        expectancy_r=0.3,
        avg_r=0.3,
        profit_factor=1.5,
        sharpe=0.4,
        sortino=0.6,
        max_drawdown_pct=drawdown,
        tail_loss_r=-1.2,
        net_pnl=1_000.0,
    )


def passing_research_run(trades=30):
    return {
        "id": 1,
        "passed": 1,
        "trial_count": 16,
        "result_json": f'{{"total_oos_trades": {trades}}}',
    }


def test_default_state_is_research(manager):
    assert manager.state("unknown_strategy") is StrategyState.RESEARCH
    assert manager.risk_multiplier("unknown_strategy") == 0.0


def test_full_ladder_walk(manager, store):
    # RESEARCH -> SHADOW
    decision = manager.request_promotion(
        SID, PromotionEvidence(research_run=passing_research_run())
    )
    assert decision.approved, decision.failures
    assert manager.state(SID) is StrategyState.SHADOW

    # SHADOW -> PAPER
    decision = manager.request_promotion(
        SID, PromotionEvidence(shadow_health=health())
    )
    assert decision.approved, decision.failures
    assert manager.state(SID) is StrategyState.PAPER

    # PAPER -> LIVE_CANARY (needs ack + measured costs + rerun)
    evidence = PromotionEvidence(
        paper_health=health(state=StrategyState.PAPER),
        measured_execution_samples=10,
        research_rerun_with_measured_costs={"passed": True},
    )
    decision = manager.request_promotion(SID, evidence, operator_ack=True)
    assert decision.approved, decision.failures
    assert manager.state(SID) is StrategyState.LIVE_CANARY
    assert manager.risk_multiplier(SID) == pytest.approx(0.25)

    # LIVE_CANARY -> LIVE
    decision = manager.request_promotion(
        SID,
        PromotionEvidence(canary_health=health(state=StrategyState.LIVE_CANARY)),
        operator_ack=True,
    )
    assert decision.approved, decision.failures
    assert manager.state(SID) is StrategyState.LIVE
    history = store.get_promotion_history(SID)
    assert len(history) == 4


def test_failed_research_blocks_shadow(manager):
    decision = manager.request_promotion(
        SID, PromotionEvidence(research_run={"passed": 0, "trial_count": 4})
    )
    assert not decision.approved
    assert "research_run_missing_or_failed" in decision.failures
    assert manager.state(SID) is StrategyState.RESEARCH


def test_insufficient_research_trades_blocks_shadow(manager):
    decision = manager.request_promotion(
        SID, PromotionEvidence(research_run=passing_research_run(trades=3))
    )
    assert not decision.approved
    assert any("research_trades_3_below" in f for f in decision.failures)


def test_live_entry_without_ack_refused(manager):
    manager.request_promotion(SID, PromotionEvidence(research_run=passing_research_run()))
    manager.request_promotion(SID, PromotionEvidence(shadow_health=health()))
    evidence = PromotionEvidence(
        paper_health=health(),
        measured_execution_samples=10,
        research_rerun_with_measured_costs={"passed": True},
    )
    decision = manager.request_promotion(SID, evidence, operator_ack=False)
    assert not decision.approved
    assert "operator_ack_required_for_live_states" in decision.failures
    assert manager.state(SID) is StrategyState.PAPER


def test_shadow_drawdown_gate(manager):
    manager.request_promotion(SID, PromotionEvidence(research_run=passing_research_run()))
    decision = manager.request_promotion(
        SID, PromotionEvidence(shadow_health=health(drawdown=50.0))
    )
    assert not decision.approved
    assert any("shadow_drawdown" in f for f in decision.failures)


def test_demotion_never_needs_ack_and_is_recorded(manager, store):
    manager.request_promotion(SID, PromotionEvidence(research_run=passing_research_run()))
    manager.demote(SID, StrategyState.QUARANTINE, reason="drift breach")
    assert manager.state(SID) is StrategyState.QUARANTINE
    assert manager.risk_multiplier(SID) == 0.0
    history = store.get_promotion_history(SID)
    assert history[0]["to_state"] == "QUARANTINE"


def test_demote_refuses_privilege_increase(manager):
    with pytest.raises(ValueError, match="cannot raise privilege"):
        manager.demote(SID, StrategyState.LIVE, reason="nice try")


def test_quarantined_strategy_cannot_promote(manager):
    manager.request_promotion(SID, PromotionEvidence(research_run=passing_research_run()))
    manager.demote(SID, StrategyState.QUARANTINE, reason="degraded")
    decision = manager.request_promotion(
        SID, PromotionEvidence(research_run=passing_research_run())
    )
    assert not decision.approved
    assert any("explicit_demote" in f for f in decision.failures)


def test_degradation_monitor_actions():
    monitor = DegradationMonitor(DegradationThresholds())
    ok = health(trades=30)
    assert monitor.evaluate(ok).action == "none"

    soft = health(trades=30)
    soft.profit_factor = 0.5
    verdict = monitor.evaluate(soft)
    assert verdict.action == "reduce_risk"
    assert "profit_factor_collapse" in verdict.reasons

    two_soft = health(trades=30)
    two_soft.profit_factor = 0.5
    two_soft.expectancy_r = -0.5
    assert monitor.evaluate(two_soft).action == "quarantine"

    drawdown = health(trades=30, drawdown=20.0)
    verdict = monitor.evaluate(drawdown)
    assert verdict.action == "quarantine"
    assert "unexpected_drawdown" in verdict.reasons

    small = health(trades=5)
    small.profit_factor = 0.1
    verdict = monitor.evaluate(small)
    assert verdict.action == "none"
    assert "insufficient_sample" in verdict.reasons


def test_cost_inflation_quarantines():
    monitor = DegradationMonitor()
    verdict = monitor.evaluate(health(trades=30), cost_inflation_ratio=3.0)
    assert verdict.action == "quarantine"
    assert "execution_cost_inflation" in verdict.reasons
