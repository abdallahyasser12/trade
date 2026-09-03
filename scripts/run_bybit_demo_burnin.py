#!/usr/bin/env python3
"""Run the frozen Bybit Spot Demo operational burn-in.

Normal operators start this service from the dashboard. These commands remain the
exact service equivalents used for acceptance/recovery:

    python scripts/run_bybit_demo_burnin.py --hours 72
    python scripts/run_bybit_demo_burnin.py --hours 120

The runner never constructs BYBIT_LIVE and cannot accept live credentials. SIGTERM is
a graceful operator stop. SIGUSR1 (where available) arms the emergency kill switch.
"""
from __future__ import annotations

import argparse
import os
import signal
import sys
import threading
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broker import BrokerError
from bybit_active_report import ActiveSafeBybitBurnInReport
from bybit_burnin import (
    BybitBurnInConfig,
    BybitDemoBurnInService,
    credentials_from_environment,
)
from bybit_entry_intelligence import wire_bybit_quality_layers
from bybit_operator_control import OperatorControl
from bybit_report import BybitBurnInReport
from quant.exits import ExitPolicyConfig

DAILY_REPORT_SECONDS = 24 * 60 * 60


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hours", type=float, choices=(72.0, 120.0), default=120.0)
    parser.add_argument("--capital-usdt", type=float, default=1_000.0)
    parser.add_argument(
        "--symbols",
        nargs="+",
        choices=("BTC/USD", "ETH/USD", "SOL/USD"),
        default=["BTC/USD", "ETH/USD", "SOL/USD"],
        help="Explicit permitted Spot universe. No coins are added automatically.",
    )
    parser.add_argument("--state-db", default="data/bybit_demo_trading.db")
    parser.add_argument("--venue-db", default="data/bybit_demo_venue.db")
    parser.add_argument("--quant-db", default="data/quant_desk.db")
    parser.add_argument("--control-db", default="data/bybit_operator.db")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument(
        "--enable-researched-smart-exits",
        action="store_true",
        help=(
            "Exercise the existing smart-exit consumers during Demo. This enables "
            "policy gates but does not tune thresholds and is not a profitability claim."
        ),
    )
    args = parser.parse_args()

    policy = ExitPolicyConfig()
    if args.enable_researched_smart_exits:
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
        universe=tuple(args.symbols),
        strategy_capital_budget_usdt=args.capital_usdt,
        exit_policy=policy,
    )
    control = OperatorControl(
        args.control_db,
        engine_db=args.state_db,
        venue_db=args.venue_db,
    )
    service = None
    daily_stop = threading.Event()
    daily_thread = None
    run_id = None
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
        # Bybit-specific quality is negative-only and sits on top of the existing
        # EvidenceQuantScanner. It adds live depth/impact/target-room rejection and
        # feeds the accepted target plan into the existing TradeManagementPolicy.
        # It does not create a second strategy or a second exit system.
        wire_bybit_quality_layers(service)

        def graceful_stop(_signum, _frame):
            if service is not None:
                service.request_stop()

        def emergency_stop(_signum, _frame):
            if service is not None:
                service.emergency_stop()

        signal.signal(signal.SIGTERM, graceful_stop)
        if hasattr(signal, "SIGUSR1"):
            signal.signal(signal.SIGUSR1, emergency_stop)

        run_id = service.prepare()
        control.mark_running(os.getpid(), run_id)

        def daily_reports() -> None:
            reporter = ActiveSafeBybitBurnInReport(args.venue_db, args.state_db)
            day = 1
            while not daily_stop.wait(DAILY_REPORT_SECONDS):
                try:
                    label = f"day_{day}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
                    json_path, md_path, _ = reporter.write_snapshot(
                        run_id,
                        args.reports_dir,
                        label=label,
                    )
                    print(f"Daily JSON report: {json_path}", flush=True)
                    print(f"Daily Markdown report: {md_path}", flush=True)
                    day += 1
                except Exception as exc:
                    print(f"Daily report generation failed: {exc}", file=sys.stderr, flush=True)

        daily_thread = threading.Thread(
            target=daily_reports,
            name=f"bybit-daily-report-{run_id[-8:]}",
            daemon=True,
        )
        daily_thread.start()

        print(
            f"BYBIT DEMO burn-in {run_id}: {args.hours:.0f}h, "
            f"capital={args.capital_usdt:.2f} USDT, universe={config.universe}, "
            "Spot only, isLeverage=0, live disabled. Dashboard/backend process ownership active.",
            flush=True,
        )
        code = service.run()
        service.close()
        service = None

        daily_stop.set()
        if daily_thread is not None:
            daily_thread.join(2.0)

        report = BybitBurnInReport(args.venue_db, args.state_db)
        json_path, md_path, result = report.write(run_id, args.reports_dir)
        print(f"JSON report: {json_path}")
        print(f"Markdown report: {md_path}")
        print(f"Operational verdict: {result['verdict']['operational']}")
        print(f"Edge verdict: {result['verdict']['edge']}")
        print("Strategy proven: NO")
        control.mark_stopped(f"run {run_id} exited with code {code}")
        return code
    except (BrokerError, RuntimeError, ValueError) as exc:
        control.mark_stopped(str(exc))
        print(f"Bybit Demo burn-in refused/failed: {exc}", file=sys.stderr)
        return 2
    finally:
        daily_stop.set()
        if daily_thread is not None and daily_thread.is_alive():
            daily_thread.join(2.0)
        if service is not None:
            service.close()
        control.close()


if __name__ == "__main__":
    raise SystemExit(main())
