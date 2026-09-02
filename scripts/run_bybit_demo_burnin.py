#!/usr/bin/env python3
"""Run the frozen Bybit Spot Demo operational burn-in.

Examples:
    python scripts/run_bybit_demo_burnin.py --hours 72
    python scripts/run_bybit_demo_burnin.py --hours 120

The runner never constructs BYBIT_LIVE and cannot accept live credentials.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broker import BrokerError
from bybit_burnin import (
    BybitBurnInConfig,
    BybitDemoBurnInService,
    credentials_from_environment,
)
from bybit_report import BybitBurnInReport
from quant.exits import ExitPolicyConfig


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, choices=(72.0, 120.0), default=120.0)
    parser.add_argument("--capital-usdt", type=float, default=1_000.0)
    parser.add_argument("--state-db", default="data/bybit_demo_trading.db")
    parser.add_argument("--venue-db", default="data/bybit_demo_venue.db")
    parser.add_argument("--quant-db", default="data/quant_desk.db")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--enable-researched-smart-exits",
        action="store_true",
        help=(
            "Exercise the existing smart-exit consumers during Demo. This enables "
            "policy gates but does not tune their thresholds and is not a profitability claim."
        ),
    )
    args = parser.parse_args()

    policy = ExitPolicyConfig()
    if args.enable_researched_smart_exits:
        # Existing, already-researchable policy thresholds only. The burn-in runner
        # never searches or changes them to make a desired return appear.
        policy = replace(
            policy,
            enable_partial_take=True,
            enable_break_even=True,
            enable_time_stop=True,
            enable_smart_exit=True,
            enable_adaptive_trail=True,
        )

    config = BybitBurnInConfig(
        hours=args.hours,
        strategy_capital_budget_usdt=args.capital_usdt,
        exit_policy=policy,
    )
    try:
        key, secret = credentials_from_environment()
        service = BybitDemoBurnInService(
            key,
            secret,
            config,
            engine_db=args.state_db,
            venue_db=args.venue_db,
            quant_db=args.quant_db,
        )
        try:
            run_id = service.prepare()
            print(
                f"BYBIT DEMO burn-in {run_id}: {args.hours:.0f}h, "
                f"capital={args.capital_usdt:.2f} USDT, universe={config.universe}, "
                "Spot only, isLeverage=0, live disabled."
            )
            code = service.run()
        finally:
            service.close()
        report = BybitBurnInReport(args.venue_db, args.state_db)
        json_path, md_path, result = report.write(run_id, args.reports_dir)
        print(f"JSON report: {json_path}")
        print(f"Markdown report: {md_path}")
        print(f"Operational verdict: {result['verdict']['operational']}")
        print(f"Edge verdict: {result['verdict']['edge']}")
        print("Strategy proven: NO")
        return code
    except (BrokerError, RuntimeError, ValueError) as exc:
        print(f"Bybit Demo burn-in refused/failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
