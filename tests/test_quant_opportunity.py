"""Abstention layer and global-ranker behavior tests."""

from datetime import datetime, timezone

import pytest

from quant.abstention import AbstentionConfig, AbstentionPolicy
from quant.contracts import (
    DecisionOutcome,
    ExecutionEstimate,
    Market,
    OpportunityCandidate,
    QuantAction,
)
from quant.execution import ExecutionQualityConfig, ExecutionQualityEngine
from quant.opportunity import (
    GlobalOpportunityRanker,
    OpportunityScoringConfig,
    RankerInput,
    score_components,
)

NOW = datetime(2026, 2, 1, tzinfo=timezone.utc)


def make_candidate(symbol="BTC/USD", alpha=40.0, confidence=0.7, action=QuantAction.BUY):
    return OpportunityCandidate(
        symbol=symbol,
        market=Market.CRYPTO,
        timeframe="1Hour",
        strategy_id="donchian_trend",
        strategy_version="1.0.0",
        action=action,
        timestamp=NOW,
        close=100.0,
        stop_price=95.0,
        atr=2.0,
        expected_gross_alpha_bps=alpha,
        confidence=confidence,
        signal_reason="breakout",
    )


def make_estimate(symbol="BTC/USD", viable=True, round_trip=10.0, reason=None):
    return ExecutionEstimate(
        symbol=symbol,
        spread_bps=4.0,
        slippage_bps=3.0,
        fee_bps=0.0,
        round_trip_bps=round_trip,
        participation_pct=0.1,
        cost_source="assumed",
        sample_count=0,
        viable=viable,
        reason=reason,
    )


def make_ranker(min_confidence=0.5, min_net_alpha=1.0):
    return GlobalOpportunityRanker(
        OpportunityScoringConfig(min_net_alpha_bps=min_net_alpha),
        AbstentionPolicy(AbstentionConfig(min_confidence=min_confidence)),
        ExecutionQualityEngine(ExecutionQualityConfig(uncertainty_margin_bps=5.0)),
    )


# ---------------------------------------------------------------------------
# Abstention
# ---------------------------------------------------------------------------


def test_abstention_each_reason():
    policy = AbstentionPolicy(
        AbstentionConfig(min_confidence=0.6, max_feature_z=3.0, max_data_age_bars=1,
                         min_model_agreement=0.7)
    )
    low = policy.evaluate(make_candidate(confidence=0.3))
    assert low.abstain and "low_confidence" in low.reasons
    ood = policy.evaluate(make_candidate(), model_ood=True)
    assert "out_of_distribution" in ood.reasons
    extreme = policy.evaluate(make_candidate(), feature_zscores={"mom": 9.0})
    assert "out_of_distribution" in extreme.reasons
    stale = policy.evaluate(make_candidate(), data_age_bars=5)
    assert "stale_data" in stale.reasons
    disagree = policy.evaluate(make_candidate(), model_disagreement=0.5)
    assert "model_disagreement" in disagree.reasons
    contradicts = policy.evaluate(make_candidate(), model_prob_up=0.3)
    assert "model_contradicts" in contradicts.reasons


def test_abstention_nan_fails_closed():
    policy = AbstentionPolicy()
    verdict = policy.evaluate(make_candidate(), model_prob_up=float("nan"))
    assert verdict.abstain
    assert "non_finite_input" in verdict.reasons


def test_abstention_clean_pass():
    policy = AbstentionPolicy(AbstentionConfig(min_confidence=0.5))
    verdict = policy.evaluate(make_candidate(confidence=0.8), model_prob_up=0.7)
    assert not verdict.abstain
    assert verdict.reasons == []


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


def test_score_components_neutral_defaults():
    config = OpportunityScoringConfig()
    components = score_components(
        config, make_candidate(confidence=1.0), net_alpha_bps=10.0
    )
    assert components["regime_multiplier"] == 1.0
    assert components["reliability_multiplier"] == pytest.approx(0.5)
    assert components["score"] == pytest.approx(10.0 * 0.5)


def test_score_penalties_shrink_but_never_flip_sign():
    config = OpportunityScoringConfig()
    components = score_components(
        config,
        make_candidate(confidence=1.0),
        net_alpha_bps=10.0,
        drawdown_pct=100.0,
        atr_percentile=99.0,
    )
    assert 0.0 < components["score"] < 10.0


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------


def test_ranker_orders_by_score_and_assigns_ranks():
    ranker = make_ranker()
    inputs = [
        RankerInput(candidate=make_candidate("AAA/USD", alpha=20.0), execution_estimate=make_estimate("AAA/USD")),
        RankerInput(candidate=make_candidate("BBB/USD", alpha=80.0), execution_estimate=make_estimate("BBB/USD")),
    ]
    decisions = ranker.rank(inputs)
    executes = [d for d in decisions if d.outcome is DecisionOutcome.EXECUTE]
    assert [d.candidate.symbol for d in executes] == ["BBB/USD", "AAA/USD"]
    assert [d.rank for d in executes] == [1, 2]


def test_ranker_net_alpha_gate():
    ranker = make_ranker()
    # Gross 12 - round trip 10 - margin 5 = -3 -> rejected.
    inputs = [
        RankerInput(candidate=make_candidate(alpha=12.0), execution_estimate=make_estimate())
    ]
    decisions = ranker.rank(inputs)
    assert decisions[0].outcome is DecisionOutcome.REJECT
    assert "net_alpha_below_minimum" in decisions[0].rejection_reasons
    assert decisions[0].expected_net_alpha_bps == pytest.approx(-3.0)


def test_ranker_execution_not_viable():
    ranker = make_ranker()
    inputs = [
        RankerInput(
            candidate=make_candidate(alpha=100.0),
            execution_estimate=make_estimate(viable=False, reason="spread_above_maximum"),
        )
    ]
    decisions = ranker.rank(inputs)
    assert decisions[0].outcome is DecisionOutcome.REJECT
    assert "execution_not_viable" in decisions[0].rejection_reasons
    assert "spread_above_maximum" in decisions[0].rejection_reasons


def test_ranker_abstains_low_confidence():
    ranker = make_ranker(min_confidence=0.9)
    inputs = [
        RankerInput(candidate=make_candidate(confidence=0.6), execution_estimate=make_estimate())
    ]
    decisions = ranker.rank(inputs)
    assert decisions[0].outcome is DecisionOutcome.ABSTAIN
    assert "low_confidence" in decisions[0].rejection_reasons


def test_ranker_every_input_yields_exactly_one_decision():
    ranker = make_ranker()
    inputs = [
        RankerInput(candidate=make_candidate("AAA/USD", alpha=50.0), execution_estimate=make_estimate("AAA/USD")),
        RankerInput(candidate=make_candidate("BBB/USD", alpha=2.0), execution_estimate=make_estimate("BBB/USD")),
        RankerInput(
            candidate=make_candidate("CCC/USD", action=QuantAction.HOLD),
            execution_estimate=make_estimate("CCC/USD"),
        ),
        RankerInput(candidate=make_candidate("DDD/USD", confidence=0.1), execution_estimate=make_estimate("DDD/USD")),
    ]
    decisions = ranker.rank(inputs)
    assert len(decisions) == 4
    outcomes = {d.candidate.symbol: d.outcome for d in decisions}
    assert outcomes["AAA/USD"] is DecisionOutcome.EXECUTE
    assert outcomes["BBB/USD"] is DecisionOutcome.REJECT
    assert outcomes["CCC/USD"] is DecisionOutcome.REJECT
    assert outcomes["DDD/USD"] is DecisionOutcome.ABSTAIN


def test_ranker_invalid_candidate_rejected_with_validation_errors():
    ranker = make_ranker()
    bad = make_candidate()
    bad.stop_price = 200.0  # stop above close
    inputs = [RankerInput(candidate=bad, execution_estimate=make_estimate())]
    decisions = ranker.rank(inputs)
    assert decisions[0].outcome is DecisionOutcome.REJECT
    assert any("stop" in reason for reason in decisions[0].rejection_reasons)


def test_ranker_deterministic_tie_break():
    ranker = make_ranker()
    inputs = [
        RankerInput(candidate=make_candidate("ZZZ/USD", alpha=50.0), execution_estimate=make_estimate("ZZZ/USD")),
        RankerInput(candidate=make_candidate("AAA/USD", alpha=50.0), execution_estimate=make_estimate("AAA/USD")),
    ]
    decisions = ranker.rank(inputs)
    executes = [d for d in decisions if d.outcome is DecisionOutcome.EXECUTE]
    assert [d.candidate.symbol for d in executes] == ["AAA/USD", "ZZZ/USD"]
