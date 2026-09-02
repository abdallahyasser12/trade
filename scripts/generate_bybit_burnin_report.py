#!/usr/bin/env python3
"""Generate JSON/Markdown evidence for an existing Bybit Demo burn-in run."""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from bybit_report import BybitBurnInReport


def _latest_run(venue_db: str) -> str:
    con = sqlite3.connect(venue_db)
    try:
        row = con.execute(
            "SELECT run_id FROM burnin_runs ORDER BY start_time DESC LIMIT 1"
        ).fetchone()
    finally:
        con.close()
    if not row:
        raise ValueError("No Bybit burn-in runs found")
    return str(row[0])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--venue-db", default="data/bybit_demo_venue.db")
    parser.add_argument("--state-db", default="data/bybit_demo_trading.db")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()
    try:
        run_id = args.run_id or _latest_run(args.venue_db)
        json_path, md_path, report = BybitBurnInReport(
            args.venue_db, args.state_db
        ).write(run_id, args.reports_dir)
        print(json_path)
        print(md_path)
        print(report["verdict"]["operational"])
        print(report["verdict"]["edge"])
        print("strategy_proven=false")
        return 0
    except Exception as exc:
        print(f"Report generation failed: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
