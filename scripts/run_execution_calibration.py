#!/usr/bin/env python3
"""Report what execution actually cost, against what the simulator assumed.

Every backtest in this repository charges 10 bps of slippage per side, and that
number was chosen rather than measured. This reads the engine's execution
ledger and says whether the assumption holds.

Read the sample count first. Below thirty fills there is no recommendation at
all, because an average of four fills is an anecdote wearing a number's
clothing.

Reads only. It never places an order, never modifies the ledger, and never
changes a configuration file.

Exit codes:
  0  the assumption holds, or there is not yet enough evidence to say
  1  the assumption is optimistic: measured execution is worse than assumed
  2  the report could not be produced (missing database, bad arguments)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from execution_calibration import (
    MIN_SAMPLES_FOR_RECOMMENDATION,
    calibrate_from_store,
)
from state_store import StateStore

RULE = "=" * 78


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure realised execution cost against the assumed value.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", default="trading_bot.db", help="Path to the state store.")
    parser.add_argument(
        "--assumed-slippage-bps",
        type=float,
        default=10.0,
        help="The per-side slippage the simulator charges, to compare against.",
    )
    parser.add_argument(
        "--session-id",
        default="",
        help="Restrict to one engine session. Blank uses every recorded fill.",
    )
    parser.add_argument("--json-out", default="")
    return parser.parse_args(argv)


def _fail(message: str) -> int:
    print(f"\nERROR: {message}", file=sys.stderr)
    return 2


def _row(label: str, stats) -> str:
    def fmt(value, places=1):
        return "n/a" if value is None else f"{value:.{places}f}"

    return (
        f"  {label:<14}{stats.samples:>8}{fmt(stats.mean_bps):>10}"
        f"{fmt(stats.median_bps):>10}{fmt(stats.p90_bps):>10}"
        f"{fmt(stats.worst_bps):>10}{fmt(stats.skewness, 2):>8}"
        # Withheld per population, not just overall: an exit split built from
        # two fills must not publish a planning number just because the
        # overall ledger is large enough.
        f"{(fmt(stats.planning_bps) if stats.sufficient else 'thin'):>11}"
    )


def main(argv=None) -> int:
    args = _parse_args(argv)

    if not args.assumed_slippage_bps >= 0:
        return _fail("--assumed-slippage-bps must be non-negative.")
    database = Path(args.db)
    if not database.exists():
        return _fail(
            f"No state store at {database}. The ledger is written by the live or "
            "paper engine, so there is nothing to calibrate until one has run."
        )

    try:
        store = StateStore(str(database))
        calibration = calibrate_from_store(
            store,
            assumed_slippage_bps=args.assumed_slippage_bps,
            session_id=args.session_id or None,
        )
    except Exception as exc:  # noqa: BLE001 - surface any store failure plainly
        return _fail(f"Could not read the execution ledger: {exc}")

    print(RULE)
    print("EXECUTION CALIBRATION — measured cost versus assumed cost")
    print(RULE)
    print(f"  Ledger        {database}")
    print(f"  Session       {args.session_id or 'all recorded sessions'}")
    print(f"  Assumed       {args.assumed_slippage_bps:g} bps per side")
    print("\n  Positive slippage always means the fill was worse than the decision")
    print("  price, for a buy and a sell alike.")

    print(f"\n{RULE}\nSLIPPAGE DISTRIBUTION (bps)\n{RULE}")
    print(f"  {'population':<14}{'fills':>8}{'mean':>10}{'median':>10}"
          f"{'p90':>10}{'worst':>10}{'skew':>8}{'planning':>11}")
    print(_row("overall", calibration.overall))
    print(_row("entries", calibration.entries))
    print(_row("exits", calibration.exits))
    if calibration.unclassified.samples:
        print(_row("unclassified", calibration.unclassified))

    print("\n  'planning' is max(Student-t upper bound on the mean, p90), withheld")
    print(f"  below {MIN_SAMPLES_FOR_RECOMMENDATION} fills. It errs high on purpose:")
    print("  the plain confidence bound covers only ~74% on heavily skewed data.")

    for warning in (
        calibration.measurability_warning(), calibration.coverage_warning()
    ):
        if warning:
            print(f"\n  NOTE: {warning}")

    fee_bps = calibration.realised_fee_bps()
    if fee_bps is not None:
        print(f"\n  Realised fees  {fee_bps:.1f} bps of traded notional"
              f"   (${calibration.total_fees:,.2f} on ${calibration.total_notional:,.0f})")

    print(f"\n{RULE}\nVERDICT\n{RULE}")
    print(f"  {calibration.verdict()}")
    if calibration.sufficient and calibration.understated:
        print("\n  Re-run research with the measured figure before trusting any")
        print("  cost-to-risk number:")
        print(f"    --slippage-bps {calibration.recommended_slippage_bps:.1f}")
    elif not calibration.sufficient:
        needed = MIN_SAMPLES_FOR_RECOMMENDATION - calibration.overall.samples
        print(f"\n  {needed} more recorded fills are needed before this can answer.")
    print(RULE)

    if args.json_out:
        Path(args.json_out).write_text(
            json.dumps(
                {
                    "database": str(database),
                    "session_id": args.session_id or None,
                    "calibration": calibration.to_dict(),
                },
                indent=2,
                default=str,
            )
        )
        print(f"\nWrote {args.json_out}")

    # Exit 1 only when the assumption is measurably optimistic. Insufficient
    # evidence is not a failure - it is the normal state before the engine has
    # traded enough - and must not be conflated with one.
    return 1 if calibration.understated else 0


def _guarded_main(argv=None) -> int:
    try:
        return main(argv)
    except Exception:  # noqa: BLE001 - last resort, re-raised as a report
        import traceback

        print("\nUNEXPECTED FAILURE — full traceback follows:", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(_guarded_main())
