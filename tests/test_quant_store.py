"""Persistence contract tests for the quant desk store."""

from datetime import datetime, timedelta, timezone

import pytest

from quant.contracts import (
    AttributionRecord,
    DecisionOutcome,
    Market,
    OpportunityCandidate,
    OpportunityDecision,
    QuantAction,
    StrategyState,
)
from quant.store import QuantStore


NOW = datetime(2026, 1, 5, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def store():
    s = QuantStore(":memory:")
    yield s
    s.close()


def make_candidate(symbol="BTC/USD", action=QuantAction.BUY):
    return OpportunityCandidate(
        symbol=symbol,
        market=Market.CRYPTO,
        timeframe="1Hour",
        strategy_id="donchian_trend",
        strategy_version="1.0.0",
        action=action,
        timestamp=NOW,
        close=50_000.0,
        stop_price=48_000.0,
        atr=800.0,
        expected_gross_alpha_bps=40.0,
        confidence=0.7,
        signal_reason="confirmed_breakout",
    )


def make_decision(outcome=DecisionOutcome.EXECUTE, rank=1):
    return OpportunityDecision(
        candidate=make_candidate(),
        outcome=outcome,
        rank=rank,
        expected_cost_bps=12.0,
        expected_net_alpha_bps=23.0,
        opportunity_score=16.1,
        risk_allocation_pct=0.35,
        rejection_reasons=[] if outcome is DecisionOutcome.EXECUTE else ["net_alpha_below_minimum"],
        decided_at=NOW,
    )


def make_attribution(strategy_id="donchian_trend", r_multiple=1.5, shadow=False, regime="STRONG_BULL"):
    return AttributionRecord(
        strategy_id=strategy_id,
        strategy_version="1.0.0",
        symbol="BTC/USD",
        market=Market.CRYPTO,
        timeframe="1Hour",
        session_id="s1",
        signal_timestamp=NOW,
        entry_timestamp=NOW,
        exit_timestamp=NOW + timedelta(hours=8),
        qty=0.5,
        entry_price=50_000.0,
        exit_price=53_000.0,
        initial_stop_price=48_000.0,
        fees=25.0,
        expected_cost_bps=12.0,
        realized_slippage_bps=8.0,
        gross_pnl=1_500.0,
        net_pnl=1_475.0,
        r_multiple=r_multiple,
        max_favorable_excursion_pct=7.0,
        max_adverse_excursion_pct=1.2,
        entry_reason="confirmed_breakout",
        exit_reason="trailing_stop",
        regime_at_entry=regime,
        opportunity_score=16.1,
        shadow=shadow,
    )


def test_decision_round_trip_includes_rejections(store):
    store.record_decision(make_decision())
    store.record_decision(make_decision(outcome=DecisionOutcome.REJECT, rank=None))
    all_rows = store.get_decisions()
    assert len(all_rows) == 2
    rejects = store.get_decisions(outcome="REJECT")
    assert len(rejects) == 1
    assert "net_alpha_below_minimum" in rejects[0]["rejection_reasons"]
    counts = store.decision_counts_since(NOW - timedelta(days=1))
    assert counts == {"EXECUTE": 1, "REJECT": 1}


def test_attribution_round_trip_and_shadow_filter(store):
    store.record_attribution(make_attribution())
    store.record_attribution(make_attribution(shadow=True))
    assert len(store.get_attributions()) == 2
    real_only = store.get_attributions(include_shadow=False)
    assert len(real_only) == 1
    assert real_only[0]["shadow"] == 0
    assert real_only[0]["mfe_pct"] == pytest.approx(7.0)


def test_lifecycle_state_and_history_stay_consistent(store):
    assert store.get_strategy_state("donchian_trend") is None
    store.set_strategy_state(
        "donchian_trend",
        StrategyState.RESEARCH,
        StrategyState.SHADOW,
        reason="research PASS",
        evidence={"run_id": 1},
    )
    state = store.get_strategy_state("donchian_trend")
    assert state["state"] == "SHADOW"
    history = store.get_promotion_history("donchian_trend")
    assert len(history) == 1
    assert history[0]["from_state"] == "RESEARCH"
    assert history[0]["to_state"] == "SHADOW"


def test_research_runs_accumulate_trial_counts(store):
    for trials in (16, 32):
        store.record_research_run(
            strategy_id="donchian_trend",
            params={"breakout_period": 20},
            universe=["BTC/USD", "ETH/USD"],
            data_window="2023-01-01..2025-06-30",
            candidate_count=trials,
            trial_count=trials,
            passed=False,
            gates={"min_folds": {"value": 2, "threshold": 3, "passed": False}},
        )
    assert store.total_trials("donchian_trend") == 48
    assert store.total_trials() == 48
    runs = store.get_research_runs("donchian_trend")
    assert len(runs) == 2
    assert runs[0]["passed"] == 0


def test_features_asof_respects_availability(store):
    observed = NOW
    published_late = NOW + timedelta(hours=6)
    store.record_features(
        timestamp=observed,
        symbol="AAPL",
        market=Market.US_EQUITY,
        timeframe="1Day",
        source="fundamentals",
        feature_version="1.0.0",
        payload={"eps_surprise": 0.12},
        available_at=published_late,
    )
    # Before publication: invisible even though the observation timestamp passed.
    before = store.get_features_asof(
        symbol="AAPL",
        market=Market.US_EQUITY,
        timeframe="1Day",
        source="fundamentals",
        feature_version="1.0.0",
        asof=NOW + timedelta(hours=1),
    )
    assert before == []
    after = store.get_features_asof(
        symbol="AAPL",
        market=Market.US_EQUITY,
        timeframe="1Day",
        source="fundamentals",
        feature_version="1.0.0",
        asof=published_late,
    )
    assert len(after) == 1
    assert after[0]["payload"]["eps_surprise"] == pytest.approx(0.12)


def test_features_reject_availability_before_observation(store):
    with pytest.raises(ValueError, match="available_at"):
        store.record_features(
            timestamp=NOW,
            symbol="AAPL",
            market=Market.US_EQUITY,
            timeframe="1Day",
            source="news",
            feature_version="1.0.0",
            payload={},
            available_at=NOW - timedelta(hours=1),
        )


def test_shadow_position_crud(store):
    data = dict(
        strategy_id="vwap_reversion",
        symbol="ETH/USD",
        qty=2.0,
        entry_price=3_000.0,
        entry_time=NOW.isoformat(),
        stop_price=2_900.0,
        initial_stop_price=2_900.0,
        highest_close=3_010.0,
        lowest_close=2_995.0,
    )
    store.upsert_shadow_position(data)
    rows = store.get_shadow_positions("vwap_reversion")
    assert len(rows) == 1
    data["stop_price"] = 2_950.0
    store.upsert_shadow_position(data)
    rows = store.get_shadow_positions("vwap_reversion")
    assert len(rows) == 1
    assert rows[0]["stop_price"] == pytest.approx(2_950.0)
    store.delete_shadow_position("vwap_reversion", "ETH/USD")
    assert store.get_shadow_positions() == []


def test_shadow_position_requires_complete_fields(store):
    with pytest.raises(ValueError, match="missing fields"):
        store.upsert_shadow_position({"strategy_id": "x", "symbol": "ETH/USD"})


def test_newer_schema_refused(tmp_path):
    path = str(tmp_path / "quant.db")
    store = QuantStore(path)
    store._execute("PRAGMA user_version = 99")
    store.close()
    with pytest.raises(RuntimeError, match="newer build"):
        QuantStore(path)


def test_model_registry_upsert(store):
    store.register_model(
        model_id="direction_5bar",
        version="1.0.0",
        params={"l2": 1.0},
        metrics={"brier": 0.24},
        calibration={"ece": 0.03},
    )
    store.register_model(
        model_id="direction_5bar",
        version="1.0.0",
        params={"l2": 2.0},
        metrics={"brier": 0.22},
    )
    models = store.get_models("direction_5bar")
    assert len(models) == 1
    assert '"l2": 2.0' in models[0]["params_json"]


def test_drift_and_risk_events(store):
    store.record_drift(
        subject="strategy:donchian_trend",
        metric="expectancy_z",
        value=-2.4,
        threshold=-2.0,
        breached=True,
    )
    drift = store.get_drift("strategy:donchian_trend")
    assert len(drift) == 1
    assert drift[0]["breached"] == 1
    store.record_risk_event(
        source="scanner", kind="stale_data", message="quote older than limit"
    )
    events = store.get_risk_events()
    assert len(events) == 1
    assert events[0]["kind"] == "stale_data"
