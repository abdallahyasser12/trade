"""End-to-end wiring proof: edge intelligence inside a REAL scan cycle.

The unit tests around ``assess_edge`` cannot catch a silently-neutral edge
layer (wrong timestamps, starved frames, dropped ranker input all degrade to
"measurable=False, multiplier 1.0" by design).  This file runs the actual
``EvidenceQuantScanner.scan`` path against an in-memory store and asserts the
assessment was MEASURED and its multiplier actually reached the ranked score.
"""

from datetime import datetime, timedelta, timezone

from crypto_models import CryptoQuote
from quant.contracts import DecisionOutcome, Market, StrategyState
from quant.evidence_scanner import EvidenceQuantScanner
from quant.execution import ExecutionQualityConfig, ExecutionQualityEngine
from quant.lifecycle import LifecycleManager
from quant.portfolio import PortfolioBook
from quant.scanner import ScanStrategy
from quant.store import QuantStore

from tests.test_quant_scanner_production import (
    NOW,
    SID,
    ScriptedStrategy,
    bars,
    seed_edge,
)


def _scanner(store):
    return EvidenceQuantScanner(
        store=store,
        lifecycle=LifecycleManager(store),
        execution_engine=ExecutionQualityEngine(
            ExecutionQualityConfig(
                uncertainty_margin_bps=0.0,
                assumed_slippage_bps=1.0,
                crypto_assumed_spread_bps=1.0,
                crypto_taker_fee_bps=0.0,
            )
        ),
    )


def _scan(store, frame):
    scan = _scanner(store)
    store.set_strategy_state(SID, StrategyState.RESEARCH, StrategyState.PAPER, reason="test")
    seed_edge(store)
    quote = CryptoQuote(
        symbol="ETH/USD",
        timestamp=NOW - timedelta(seconds=5),
        bid_price=float(frame["close"].iloc[-1]) * 0.9999,
        ask_price=float(frame["close"].iloc[-1]) * 1.0001,
        bid_size=500.0,
        ask_size=500.0,
    )
    return scan.scan(
        strategies=[ScanStrategy(SID, ScriptedStrategy(), Market.CRYPTO, "5Min")],
        bars_by_symbol={"ETH/USD": frame},
        equity=100_000.0,
        book=PortfolioBook(equity=100_000.0),
        quotes={"ETH/USD": quote},
        now=NOW,
    )


def test_edge_assessment_is_measured_and_reaches_the_ranked_score():
    store = QuantStore(":memory:")
    try:
        report = _scan(store, bars(periods=120))
        executed = [
            decision
            for decision in report.decisions
            if decision.outcome is DecisionOutcome.EXECUTE
            and decision.candidate.action.value == "BUY"
        ]
        assert executed, [d.rejection_reasons for d in report.decisions]
        details = executed[0].candidate.details

        payload = details["edge_intelligence"]
        assert payload["measurable"] is True, payload["reasons"]
        assert isinstance(details["edge_score"], float)
        assert 0.0 <= details["edge_score"] <= 100.0

        components = details["score_components"]
        assert components["setup_quality"] == payload["quality_multiplier"]
        assert components["setup_quality"] <= 1.0
    finally:
        store.close()


def test_starved_history_journals_unmeasured_and_stays_neutral():
    store = QuantStore(":memory:")
    try:
        report = _scan(store, bars(periods=50))  # >= strategy warmup, < edge min_bars
        executed = [
            decision
            for decision in report.decisions
            if decision.outcome is DecisionOutcome.EXECUTE
            and decision.candidate.action.value == "BUY"
        ]
        assert executed
        details = executed[0].candidate.details
        payload = details["edge_intelligence"]
        assert payload["measurable"] is False
        assert "insufficient_edge_history" in payload["reasons"]
        # Unmeasurable journals None, never a fake mid score, and stays
        # perfectly neutral in the ranked score.
        assert details["edge_score"] is None
        assert details["score_components"]["setup_quality"] == 1.0
    finally:
        store.close()
