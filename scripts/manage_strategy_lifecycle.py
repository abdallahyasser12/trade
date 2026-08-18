#!/usr/bin/env python3
"""Inspect or promote a Quant Desk strategy without editing SQLite by hand.

Each forward rung is measured only from attributions recorded since that rung
began. High-velocity intraday thresholds come from one centralized policy module
shared with its runtime.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from quant.contracts import StrategyState
from quant.intraday_lifecycle import intraday_promotion_requirements
from quant.lifecycle import LifecycleManager, PromotionEvidence, PromotionRequirements
from quant.marketplace import compute_strategy_health
from quant.store import DEFAULT_QUANT_DB_PATH, QuantStore
from state_store import StateStore

DEFAULT_STATE_DB = str(Path("data") / "intraday_trading.db")
LIVE_ACK = "I_ACCEPT_LIVE_RISK"
INTRADAY_STRATEGY_ID = "crypto_intraday_momentum"


def _requirements(strategy_id: str) -> PromotionRequirements:
    return (
        intraday_promotion_requirements()
        if strategy_id == INTRADAY_STRATEGY_ID
        else PromotionRequirements()
    )


def _manager(store: QuantStore, strategy_id: str) -> LifecycleManager:
    return LifecycleManager(store, requirements=_requirements(strategy_id))


def _parse_time(value) -> datetime | None:
    if not value:
        return None
    timestamp = datetime.fromisoformat(str(value))
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


def _latest_passing_research(store: QuantStore, strategy_id: str):
    for run in store.get_research_runs(strategy_id, limit=500):
        if bool(run.get("passed")):
            return run
    return None


def _research_run_by_id(store: QuantStore, strategy_id: str, run_id: int):
    for run in store.get_research_runs(strategy_id, limit=1000):
        if int(run.get("id") or 0) == run_id:
            return run
    return None


def _modeled_slippage(run) -> float | None:
    raw = run.get("cost_assumptions_json") if isinstance(run, dict) else None
    if not raw:
        return None
    try:
        costs = json.loads(raw) if isinstance(raw, str) else raw
        sim = (costs or {}).get("sim") or {}
        value = sim.get("slippage_bps")
        return float(value) if value is not None else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _state_health(store, strategy_id, state, state_row, include_shadow):
    return compute_strategy_health(
        store,
        strategy_id,
        state,
        include_shadow=include_shadow,
        since=_parse_time((state_row or {}).get("since")),
    )


def show_status(quant_store, state_store, strategy_id) -> int:
    lifecycle = _manager(quant_store, strategy_id)
    state = lifecycle.state(strategy_id)
    row = quant_store.get_strategy_state(strategy_id)
    include_shadow = state is StrategyState.SHADOW
    health = _state_health(
        quant_store,
        strategy_id,
        state,
        row,
        include_shadow,
    )
    latest = _latest_passing_research(quant_store, strategy_id)
    quality = state_store.execution_quality()
    req = lifecycle.requirements
    print(f"strategy={strategy_id}")
    print(f"state={state.value} since={(row or {}).get('since') or 'not-recorded'}")
    print(
        f"current-state trades={health.trades} expectancy_R={health.expectancy_r} "
        f"PF={health.profit_factor} drawdown_R={health.max_drawdown_pct}"
    )
    print(
        "requirements: "
        f"research_trades>={req.min_research_trades}; "
        f"shadow trades>={req.min_shadow_trades}, E[R]>={req.min_shadow_expectancy_r:.3f}, "
        f"PF>={req.min_shadow_profit_factor:.2f}, DD<={req.max_shadow_drawdown_r:.1f}R; "
        f"paper trades>={req.min_paper_trades}, E[R]>={req.min_paper_expectancy_r:.3f}, "
        f"PF>={req.min_paper_profit_factor:.2f}, DD<={req.max_paper_drawdown_r:.1f}R; "
        f"executions>={req.min_measured_execution_samples}; "
        f"canary trades>={req.min_canary_trades}, E[R]>={req.min_canary_expectancy_r:.3f}, "
        f"PF>={req.min_canary_profit_factor:.2f}, DD<={req.max_canary_drawdown_r:.1f}R"
    )
    print(
        f"execution samples={quality['samples']} avg_slippage_bps="
        f"{quality['avg_slippage_bps']} worst={quality['worst_slippage_bps']}"
    )
    if latest:
        print(
            f"latest passing research id={latest['id']} run_at={latest['run_at']} "
            f"candidate_trials={latest['trial_count']}"
        )
    else:
        print("latest passing research: none")
    return 0


def promote(args, quant_store, state_store) -> int:
    strategy_id = args.strategy
    lifecycle = _manager(quant_store, strategy_id)
    state = lifecycle.state(strategy_id)
    state_row = quant_store.get_strategy_state(strategy_id)
    evidence = PromotionEvidence()
    operator_ack = False

    if state is StrategyState.RESEARCH:
        run = _latest_passing_research(quant_store, strategy_id)
        if run is None:
            print("Promotion refused: no passing real research run.", file=sys.stderr)
            return 1
        evidence.research_run = run

    elif state is StrategyState.SHADOW:
        evidence.shadow_health = _state_health(
            quant_store,
            strategy_id,
            state,
            state_row,
            include_shadow=True,
        )

    elif state is StrategyState.PAPER:
        if args.live_ack != LIVE_ACK:
            print(
                f"Promotion refused: PAPER->LIVE_CANARY requires --live-ack {LIVE_ACK}",
                file=sys.stderr,
            )
            return 1
        if args.measured_research_run_id is None:
            print(
                "Promotion refused: --measured-research-run-id is required for "
                "PAPER->LIVE_CANARY.",
                file=sys.stderr,
            )
            return 1
        run = _research_run_by_id(
            quant_store,
            strategy_id,
            args.measured_research_run_id,
        )
        if run is None or not bool(run.get("passed")):
            print(
                "Promotion refused: selected measured-cost research run is missing/failed.",
                file=sys.stderr,
            )
            return 1
        quality = state_store.execution_quality()
        measured_samples = int(quality.get("samples") or 0)
        measured_slippage = quality.get("avg_slippage_bps")
        modeled_slippage = _modeled_slippage(run)
        if measured_samples < lifecycle.requirements.min_measured_execution_samples:
            print(
                f"Promotion refused: only {measured_samples} measured execution samples; "
                f"{lifecycle.requirements.min_measured_execution_samples} required.",
                file=sys.stderr,
            )
            return 1
        if measured_slippage is None or modeled_slippage is None:
            print(
                "Promotion refused: measured/modelled slippage is unavailable.",
                file=sys.stderr,
            )
            return 1
        if modeled_slippage + 1e-12 < max(0.0, float(measured_slippage)):
            print(
                f"Promotion refused: research modeled {modeled_slippage:.2f} bps slippage "
                f"but measured execution is {float(measured_slippage):.2f} bps. "
                "Rerun research with at least the measured cost.",
                file=sys.stderr,
            )
            return 1
        evidence.paper_health = _state_health(
            quant_store,
            strategy_id,
            state,
            state_row,
            include_shadow=False,
        )
        evidence.measured_execution_samples = measured_samples
        evidence.research_rerun_with_measured_costs = run
        evidence.operational_defects = args.operational_defects
        operator_ack = True

    elif state is StrategyState.LIVE_CANARY:
        if args.live_ack != LIVE_ACK:
            print(
                f"Promotion refused: LIVE_CANARY->LIVE requires --live-ack {LIVE_ACK}",
                file=sys.stderr,
            )
            return 1
        evidence.canary_health = _state_health(
            quant_store,
            strategy_id,
            state,
            state_row,
            include_shadow=False,
        )
        operator_ack = True

    else:
        print(
            f"Promotion refused from {state.value}; "
            "demotion/recovery must be handled explicitly.",
            file=sys.stderr,
        )
        return 1

    decision = lifecycle.request_promotion(
        strategy_id,
        evidence,
        operator_ack=operator_ack,
        reason=args.reason or "operator_evidence_promotion",
    )
    if not decision.approved:
        print(
            f"Promotion {decision.from_state.value}->{decision.to_state.value} REFUSED: "
            + ", ".join(decision.failures),
            file=sys.stderr,
        )
        return 1
    print(
        f"Promotion APPROVED: {decision.from_state.value}->{decision.to_state.value}."
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("status", "promote"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--strategy", required=True)
        cmd.add_argument("--quant-db", default=DEFAULT_QUANT_DB_PATH)
        cmd.add_argument("--state-db", default=DEFAULT_STATE_DB)
        if name == "promote":
            cmd.add_argument("--measured-research-run-id", type=int, default=None)
            cmd.add_argument("--operational-defects", type=int, default=0)
            cmd.add_argument("--live-ack", default="")
            cmd.add_argument("--reason", default="")
    args = parser.parse_args()
    if getattr(args, "operational_defects", 0) < 0:
        print("operational-defects cannot be negative.", file=sys.stderr)
        return 2

    quant_store = QuantStore(args.quant_db)
    state_store = StateStore(args.state_db)
    try:
        if args.command == "status":
            return show_status(quant_store, state_store, args.strategy)
        return promote(args, quant_store, state_store)
    finally:
        quant_store.close()


if __name__ == "__main__":
    raise SystemExit(main())
