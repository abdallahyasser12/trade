"""Quant Desk dashboard tab: read-only views over the desk's own store.

Follows the dashboard's KV-panel contract: local SQLite reads only, no
broker or market-data calls, cheap enough for Streamlit's rerun cadence,
and defensive against missing/legacy data (an empty desk database renders
guidance, not errors).

Everything shown here is measurement, not promise: lifecycle states,
journaled decisions with their reasons, marketplace statistics with their
sample sizes, per-gate research verdicts, drift findings, and execution
quality.  Where a number is unmeasured it is rendered as unmeasured —
never as zero.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import pandas as pd
import streamlit as st

from quant.contracts import StrategyState
from quant.lifecycle import LifecycleManager
from quant.marketplace import compute_strategy_health
from quant.store import DEFAULT_QUANT_DB_PATH, QuantStore
from quant.strategies.registry import build_default_registry
from state_store import StateStore


@st.cache_resource
def get_quant_store() -> QuantStore:
    return QuantStore(DEFAULT_QUANT_DB_PATH)


def _fmt(value: Optional[float], pattern: str = "{:.2f}", none_text: str = "unmeasured") -> str:
    if value is None:
        return none_text
    try:
        return pattern.format(float(value))
    except (TypeError, ValueError):
        return none_text


def _render_market_state(qstore: QuantStore) -> None:
    st.subheader("Market state")
    raw = qstore.get_kv("last_regime")
    if not raw:
        st.info(
            "No regime assessment recorded yet. The scanner writes one per "
            "cycle; run a scan or a shadow session to populate this panel."
        )
        return
    try:
        regime = json.loads(raw)
    except ValueError:
        st.warning("Stored regime snapshot is unreadable.")
        return
    probabilities: dict[str, Any] = regime.get("probabilities", {})
    columns = st.columns(4)
    trend_labels = ["STRONG_BULL", "WEAK_BULL", "RANGE", "WEAK_BEAR", "STRONG_BEAR"]
    dominant = max(trend_labels, key=lambda label: probabilities.get(label, 0.0))
    columns[0].metric("Dominant trend", dominant, f"P={probabilities.get(dominant, 0):.2f}")
    columns[1].metric("Vol expansion", f"{probabilities.get('VOL_EXPANSION', 0):.2f}")
    columns[2].metric("Liquidity stress", f"{probabilities.get('LIQUIDITY_STRESS', 0):.2f}")
    columns[3].metric("Confidence", f"{regime.get('confidence', 0):.2f}")
    with st.expander("All regime probabilities and inputs"):
        st.json(regime)


def _render_opportunity_board(qstore: QuantStore) -> None:
    st.subheader("Opportunity board (latest decisions)")
    rows = qstore.get_decisions(limit=100)
    if not rows:
        st.info("No journaled decisions yet.")
        return
    frame = pd.DataFrame(rows)
    display = frame[
        [
            "decided_at",
            "symbol",
            "market",
            "strategy_id",
            "timeframe",
            "action",
            "outcome",
            "rank",
            "expected_net_alpha_bps",
            "opportunity_score",
            "risk_allocation_pct",
            "reason",
            "rejection_reasons",
        ]
    ].rename(columns={"expected_net_alpha_bps": "net_alpha_bps"})
    st.dataframe(display, use_container_width=True, height=320)
    counts = frame["outcome"].value_counts().to_dict()
    st.caption(
        "Outcomes in view: "
        + ", ".join(f"{name}: {count}" for name, count in sorted(counts.items()))
        + ". Rejections and abstentions are retained deliberately — the "
        "reasons column is the desk's explainability record."
    )


def _render_strategy_health(qstore: QuantStore) -> None:
    st.subheader("Strategy marketplace")
    registry = build_default_registry()
    lifecycle = LifecycleManager(qstore)
    records = []
    for metadata in registry.list():
        state = lifecycle.state(metadata.strategy_id)
        health = compute_strategy_health(
            qstore, metadata.strategy_id, state, include_shadow=True
        )
        records.append(
            {
                "strategy": metadata.strategy_id,
                "state": state.value,
                "markets": ", ".join(market.value for market in metadata.markets),
                "horizon": ", ".join(metadata.natural_timeframes),
                "trades": health.trades,
                "hit_rate": _fmt(health.hit_rate, "{:.0%}"),
                "expectancy_R": _fmt(health.expectancy_r),
                "profit_factor": _fmt(health.profit_factor),
                "max_dd_R": _fmt(health.max_drawdown_pct),
                "risk_multiplier": lifecycle.risk_multiplier(metadata.strategy_id),
            }
        )
    st.dataframe(pd.DataFrame(records), use_container_width=True)
    st.caption(
        "States: RESEARCH/SHADOW deploy no capital; PAPER trades paper only; "
        "any live state requires explicit operator acknowledgement through "
        "the lifecycle manager — nothing on this page can change a state."
    )
    history = qstore.get_promotion_history(limit=20)
    if history:
        with st.expander("Recent lifecycle transitions"):
            st.dataframe(pd.DataFrame(history), use_container_width=True)


def _render_research(qstore: QuantStore) -> None:
    st.subheader("Research runs and promotion gates")
    runs = qstore.get_research_runs(limit=20)
    if not runs:
        st.info(
            "No recorded research runs. Run the research factory "
            "(quant.research.factory.run_quant_walk_forward) to populate "
            "this panel; every run records its full gate detail and its "
            "candidate count for multiple-testing accounting."
        )
        return
    for run in runs:
        verdict = "PASS" if run["passed"] else "FAIL"
        title = (
            f"{run['run_at'][:19]} — {run['strategy_id']} — {verdict} — "
            f"{run['candidate_count']} candidates"
        )
        with st.expander(title):
            try:
                gates = json.loads(run["gates_json"])
                st.dataframe(
                    pd.DataFrame(list(gates.values())), use_container_width=True
                )
            except (ValueError, TypeError):
                st.write(run["gates_json"])
            st.caption(
                f"Cumulative recorded trials for this strategy: "
                f"{qstore.total_trials(run['strategy_id'])} (feeds the "
                "deflated-Sharpe correction on future runs)."
            )


def _render_execution_quality(state_store: StateStore) -> None:
    st.subheader("Execution quality (measured)")
    quality = state_store.execution_quality()
    columns = st.columns(4)
    columns[0].metric("Fills measured", quality["samples"])
    if quality["samples"] > 0:
        columns[1].metric(
            "Avg slippage (bps)", _fmt(quality["avg_slippage_bps"])
        )
        columns[2].metric(
            "Worst slippage (bps)", _fmt(quality["worst_slippage_bps"])
        )
        columns[3].metric("Fees paid", _fmt(quality["total_fees"], "${:,.2f}"))
        st.caption(
            "The execution gate uses max(measured, assumed) slippage: a "
            "worse measurement tightens gating immediately; a better one "
            "only takes effect after research is re-run with it."
        )
    else:
        st.caption("No fills measured yet — estimates fall back to assumptions.")


def _render_drift_and_risk(qstore: QuantStore) -> None:
    st.subheader("Drift and desk risk events")
    drift = qstore.get_drift(limit=50)
    if drift:
        frame = pd.DataFrame(drift)
        breached = frame[frame["breached"] == 1]
        if not breached.empty:
            st.error(f"{len(breached)} drift breach(es) recorded — review before scaling.")
        st.dataframe(frame, use_container_width=True, height=220)
    else:
        st.caption("No drift measurements recorded.")
    events = qstore.get_risk_events(limit=50)
    if events:
        st.dataframe(pd.DataFrame(events), use_container_width=True, height=180)


def render_quant_desk(state_store: StateStore) -> None:
    """The Quant Desk tab: measurement views over the desk store."""
    st.caption(
        "The quant desk layer discovers, validates, ranks, and retires "
        "strategies. It never places orders directly and never promotes a "
        "strategy into a live state without recorded evidence plus an "
        "explicit operator acknowledgement."
    )
    try:
        qstore = get_quant_store()
    except Exception as error:
        st.error(f"Quant store unavailable: {error}")
        return
    _render_market_state(qstore)
    st.divider()
    _render_opportunity_board(qstore)
    st.divider()
    _render_strategy_health(qstore)
    st.divider()
    _render_research(qstore)
    st.divider()
    _render_execution_quality(state_store)
    st.divider()
    _render_drift_and_risk(qstore)
