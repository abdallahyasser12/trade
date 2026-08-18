#!/usr/bin/env python3
"""Inspect/promote market-wide US-equity strategies through PAPER.

RESEARCH -> SHADOW requires a passing real-data research row plus its immutable
manifest.  The promoted manifest must match the checked-out commit and any other
already-forward market-wide strategy.  SHADOW -> PAPER requires measured forward
strategy health.  Promotion beyond PAPER is deliberately refused because the stock
broker boundary in this repository does not permit live US-equity orders.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant.adaptive_manifest import current_git_commit
from quant.contracts import StrategyState
from quant.equity_manifest import (
    EQUITY_STRATEGY_IDS,
    load_research_manifest,
    lock_deployment_manifest,
    locked_deployment_manifest,
)
from quant.lifecycle import LifecycleManager, PromotionEvidence, PromotionRequirements
from quant.marketplace import compute_strategy_health
from quant.store import DEFAULT_QUANT_DB_PATH, QuantStore

STRATEGIES = (
    "opening_range_breakout",
    "vwap_reversion",
    "volatility_squeeze",
)


def _parse_time(value):
    if not value:
        return None
    stamp = datetime.fromisoformat(str(value))
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=timezone.utc)
    return stamp.astimezone(timezone.utc)


def _run_by_id(store: QuantStore, strategy_id: str, run_id: int):
    for run in store.get_research_runs(strategy_id, limit=1000):
        if int(run.get("id") or 0) == int(run_id):
            return run
    return None


def _latest_passing(store: QuantStore, strategy_id: str):
    for run in store.get_research_runs(strategy_id, limit=500):
        if bool(run.get("passed")):
            return run
    return None


def _health(store: QuantStore, strategy_id: str, state: StrategyState):
    row = store.get_strategy_state(strategy_id)
    return compute_strategy_health(
        store,
        strategy_id,
        state,
        include_shadow=state is StrategyState.SHADOW,
        since=_parse_time((row or {}).get("since")),
    )


def _assert_manifest_compatible_with_desk(store: QuantStore, strategy_id: str, manifest) -> None:
    for other_id in STRATEGIES:
        if other_id == strategy_id:
            continue
        other_state = LifecycleManager(store).state(other_id)
        if other_state is StrategyState.RESEARCH:
            continue
        other = locked_deployment_manifest(store, other_id)
        if other is None:
            raise ValueError(
                f"{other_id} is {other_state.value} but has no locked market-wide equity manifest."
            )
        if (
            other.git_commit != manifest.git_commit
            or other.timeframe != manifest.timeframe
            or other.universe != manifest.universe
            or other.runtime_config != manifest.runtime_config
        ):
            raise ValueError(
                f"{strategy_id} research identity differs from already-forward {other_id}; "
                "all market-wide equity strategies must share one commit/config/universe."
            )


def show_status(store: QuantStore, strategy_id: str) -> int:
    lifecycle = LifecycleManager(store, requirements=PromotionRequirements())
    state = lifecycle.state(strategy_id)
    health = _health(store, strategy_id, state)
    manifest = locked_deployment_manifest(store, strategy_id)
    latest = _latest_passing(store, strategy_id)
    print(f"strategy={strategy_id}")
    print(f"state={state.value}")
    print(
        f"state_health trades={health.trades} expectancy_R={health.expectancy_r} "
        f"PF={health.profit_factor} drawdown_R={health.max_drawdown_pct}"
    )
    if manifest:
        print(
            f"locked_manifest=run#{manifest.research_run_id} commit={manifest.git_commit} "
            f"timeframe={manifest.timeframe} universe={len(manifest.universe)} symbols"
        )
        print("locked_symbols=" + ",".join(manifest.universe))
    else:
        print("locked_manifest=none")
    if latest:
        print(
            f"latest_passing_research=run#{latest['id']} trials={latest.get('trial_count')} "
            f"run_at={latest.get('run_at')}"
        )
    else:
        print("latest_passing_research=none")
    if state is StrategyState.PAPER:
        print("next_state=unsupported (live US-equity execution is intentionally blocked)")
    return 0


def promote(store: QuantStore, strategy_id: str, run_id: int | None) -> int:
    lifecycle = LifecycleManager(store, requirements=PromotionRequirements())
    state = lifecycle.state(strategy_id)
    if state is StrategyState.PAPER:
        print(
            "Promotion refused: market-wide US-equity execution is intentionally capped at PAPER.",
            file=sys.stderr,
        )
        return 1
    if state not in {StrategyState.RESEARCH, StrategyState.SHADOW}:
        print(f"Promotion refused from {state.value}.", file=sys.stderr)
        return 1

    evidence = PromotionEvidence()
    manifest_to_lock = None
    if state is StrategyState.RESEARCH:
        run = _run_by_id(store, strategy_id, run_id) if run_id else _latest_passing(store, strategy_id)
        if run is None or not bool(run.get("passed")):
            print("Promotion refused: selected real-data research run is missing or failed.", file=sys.stderr)
            return 1
        selected_run_id = int(run.get("id") or 0)
        manifest = load_research_manifest(store, strategy_id, selected_run_id)
        if manifest is None:
            print("Promotion refused: passing research row has no immutable equity manifest.", file=sys.stderr)
            return 1
        commit = current_git_commit(ROOT)
        if commit is None or commit.lower() != manifest.git_commit.lower():
            print(
                "Promotion refused: checked-out git commit differs from the passing research manifest.",
                file=sys.stderr,
            )
            return 1
        if str(run.get("git_commit") or "").lower() != manifest.git_commit.lower():
            print("Promotion refused: research row and manifest git commits differ.", file=sys.stderr)
            return 1
        try:
            _assert_manifest_compatible_with_desk(store, strategy_id, manifest)
        except ValueError as error:
            print(f"Promotion refused: {error}", file=sys.stderr)
            return 1
        evidence.research_run = run
        manifest_to_lock = manifest
    else:
        evidence.shadow_health = _health(store, strategy_id, StrategyState.SHADOW)

    decision = lifecycle.request_promotion(
        strategy_id,
        evidence,
        reason="marketwide_equity_manifest_bound_promotion",
    )
    if not decision.approved:
        print("Promotion refused: " + ", ".join(decision.failures), file=sys.stderr)
        return 1
    if manifest_to_lock is not None:
        lock_deployment_manifest(store, manifest_to_lock)
    print(f"promoted {strategy_id}: {decision.from_state.value} -> {decision.to_state.value}")
    return 0


def demote(store: QuantStore, strategy_id: str, target: str, reason: str) -> int:
    to_state = StrategyState(target)
    if to_state not in {StrategyState.RESEARCH, StrategyState.SHADOW}:
        print("Demotion target must be RESEARCH or SHADOW for this paper-only desk.", file=sys.stderr)
        return 1
    lifecycle = LifecycleManager(store, requirements=PromotionRequirements())
    lifecycle.demote(strategy_id, to_state, reason=reason)
    print(f"demoted {strategy_id} -> {to_state.value}: {reason}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("status", "promote", "demote"))
    parser.add_argument("--strategy", choices=STRATEGIES, required=True)
    parser.add_argument("--research-run-id", type=int)
    parser.add_argument("--to", choices=("RESEARCH", "SHADOW"), default="SHADOW")
    parser.add_argument("--reason", default="operator_demote")
    parser.add_argument("--db", default=DEFAULT_QUANT_DB_PATH)
    args = parser.parse_args()

    if args.strategy not in EQUITY_STRATEGY_IDS:
        print("Unsupported market-wide equity strategy.", file=sys.stderr)
        return 2
    store = QuantStore(args.db)
    try:
        if args.command == "status":
            return show_status(store, args.strategy)
        if args.command == "promote":
            return promote(store, args.strategy, args.research_run_id)
        return demote(store, args.strategy, args.to, args.reason)
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
