from quant.contracts import StrategyHealth, StrategyState
from quant.lifecycle import LifecycleManager, PromotionEvidence, PromotionRequirements
from quant.store import QuantStore

SID = "forward_health"


def health(state, *, trades=100, expectancy=0.2, pf=1.3, drawdown=2.0):
    return StrategyHealth(
        strategy_id=SID,
        state=state,
        trades=trades,
        wins=55,
        losses=max(0, trades - 55),
        hit_rate=0.55,
        expectancy_r=expectancy,
        avg_r=expectancy,
        profit_factor=pf,
        sharpe=0.2,
        sortino=0.3,
        max_drawdown_pct=drawdown,
        tail_loss_r=-1.0,
        net_pnl=100.0,
    )


def research():
    return {
        "id": 1,
        "passed": 1,
        "trial_count": 10,
        "result_json": '{"total_oos_trades": 100}',
    }


def manager(store):
    return LifecycleManager(
        store,
        PromotionRequirements(
            min_research_trades=10,
            min_shadow_trades=10,
            min_paper_trades=10,
            min_measured_execution_samples=10,
            min_canary_trades=10,
        ),
    )


def test_negative_shadow_expectancy_blocks_paper():
    store = QuantStore(":memory:")
    try:
        lifecycle = manager(store)
        assert lifecycle.request_promotion(SID, PromotionEvidence(research_run=research())).approved
        decision = lifecycle.request_promotion(
            SID,
            PromotionEvidence(
                shadow_health=health(StrategyState.SHADOW, expectancy=-0.01, pf=0.98)
            ),
        )
        assert not decision.approved
        assert any("shadow_expectancy" in failure for failure in decision.failures)
        assert any("shadow_profit_factor" in failure for failure in decision.failures)
        assert lifecycle.state(SID) is StrategyState.SHADOW
    finally:
        store.close()


def test_losing_paper_sample_cannot_promote_live_canary_even_with_ack_and_costs():
    store = QuantStore(":memory:")
    try:
        lifecycle = manager(store)
        assert lifecycle.request_promotion(SID, PromotionEvidence(research_run=research())).approved
        assert lifecycle.request_promotion(
            SID,
            PromotionEvidence(shadow_health=health(StrategyState.SHADOW)),
        ).approved
        decision = lifecycle.request_promotion(
            SID,
            PromotionEvidence(
                paper_health=health(StrategyState.PAPER, expectancy=-0.05, pf=0.7),
                measured_execution_samples=100,
                research_rerun_with_measured_costs={"passed": True},
            ),
            operator_ack=True,
        )
        assert not decision.approved
        assert any("paper_expectancy" in failure for failure in decision.failures)
        assert any("paper_profit_factor" in failure for failure in decision.failures)
        assert lifecycle.state(SID) is StrategyState.PAPER
    finally:
        store.close()


def test_losing_canary_sample_cannot_promote_full_live():
    store = QuantStore(":memory:")
    try:
        lifecycle = manager(store)
        assert lifecycle.request_promotion(SID, PromotionEvidence(research_run=research())).approved
        assert lifecycle.request_promotion(
            SID, PromotionEvidence(shadow_health=health(StrategyState.SHADOW))
        ).approved
        assert lifecycle.request_promotion(
            SID,
            PromotionEvidence(
                paper_health=health(StrategyState.PAPER),
                measured_execution_samples=100,
                research_rerun_with_measured_costs={"passed": True},
            ),
            operator_ack=True,
        ).approved
        decision = lifecycle.request_promotion(
            SID,
            PromotionEvidence(
                canary_health=health(StrategyState.LIVE_CANARY, expectancy=-0.02, pf=0.8)
            ),
            operator_ack=True,
        )
        assert not decision.approved
        assert any("canary_expectancy" in failure for failure in decision.failures)
        assert lifecycle.state(SID) is StrategyState.LIVE_CANARY
    finally:
        store.close()
