#!/usr/bin/env python3
"""Measure whether a signal's quality survives being given more capital.

Ten walk-forward experiments produced exactly one configuration with positive
out-of-sample expectancy, and it failed promotion by losing to buy-and-hold
while holding roughly 82-95% cash. This script answers the follow-up: if the
same signal is deployed harder, does its risk-adjusted quality hold?

Read the Sharpe column, not the return column. Sizing scales a signal, it does
not improve one, so the highest risk level will show the highest return
whenever expectancy is positive at all. What carries information is whether
Sharpe and profit factor stay flat as risk rises (an edge that was merely
under-deployed) or decay (something non-linear that more capital makes worse).

Market data only — this script never constructs a broker and can never place
an order. The live exposure caps are never modified by anything it does.

Exit codes:
  0  the study completed
  2  the study could not be completed (credentials, data, or configuration)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from config import load_credentials, timeframe_to_timedelta
from council import CouncilConfig
from council_research import run_council_backtest
from cross_sectional import CrossSectionalConfig
from cross_sectional_research import run_cross_sectional_backtest
from crypto_backtest import run_crypto_backtest
from crypto_config import (
    DEFAULT_CRYPTO_UNIVERSE,
    CryptoBacktestConfig,
    CryptoRiskConfig,
    CryptoStrategyConfig,
)
from crypto_data_feed import AlpacaCryptoDataFeed
from engine_research import SOLO_ENGINES, run_injected_backtest
from sizing_study import SizingGrid, run_sizing_study
from universe_alignment import align_to_common_index, drop_limiting_symbols

RULE = "=" * 78


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure how a signal's quality scales with deployment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--symbols", default=",".join(DEFAULT_CRYPTO_UNIVERSE))
    parser.add_argument("--benchmark", default="BTC/USD")
    parser.add_argument(
        "--timeframe", default="1Day", choices=["15Min", "1Hour", "4Hour", "1Day"]
    )
    parser.add_argument("--start", default="2021-01-01")
    parser.add_argument("--end", default="")
    parser.add_argument(
        "--engine",
        default="cross",
        choices=["base", "cross", "council", "meanrev", "volregime", "breadth"],
        help="Which decision layer to study. Defaults to the only engine with "
             "measured positive out-of-sample expectancy.",
    )
    parser.add_argument(
        "--risk-levels",
        default="0.35,0.70,1.05,1.40,2.00",
        help="Comma-separated risk_per_trade_pct values. 0.35 is the current "
             "default and 2.0 is the config's hard ceiling.",
    )
    parser.add_argument(
        "--position-counts",
        default="3,5,8",
        help="Comma-separated max_positions values.",
    )
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--taker-fee-bps", type=float, default=25.0)
    parser.add_argument("--spread-bps", type=float, default=5.0)
    parser.add_argument(
        "--drop-late-listings",
        action="store_true",
        help="Drop the symbols that most truncate the aligned window.",
    )
    parser.add_argument("--json-out", default="")
    return parser.parse_args(argv)


def _fail(message: str) -> int:
    print(f"\nERROR: {message}", file=sys.stderr)
    return 2


def _floats(raw: str, label: str) -> tuple[float, ...]:
    try:
        return tuple(float(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated numbers: {exc}") from exc


def _ints(raw: str, label: str) -> tuple[int, ...]:
    try:
        return tuple(int(part) for part in raw.split(",") if part.strip())
    except ValueError as exc:
        raise ValueError(f"{label} must be comma-separated integers: {exc}") from exc


def _backtest_fn(engine: str):
    """One callable with the audited signature, whichever engine was asked for."""

    if engine == "cross":
        def run(*args, **kwargs):
            return run_cross_sectional_backtest(
                *args, cross_config=CrossSectionalConfig(), **kwargs
            )
        return run
    if engine == "council":
        def run(*args, **kwargs):
            return run_council_backtest(*args, council_config=CouncilConfig(), **kwargs)
        return run
    if engine in SOLO_ENGINES and engine != "base":
        factory = SOLO_ENGINES[engine]

        def run(*args, **kwargs):
            return run_injected_backtest(factory, *args, **kwargs)
        return run
    return run_crypto_backtest


def _aligned(bars: dict, benchmark: str, drop: bool):
    """Intersect to common timestamps, optionally dropping the worst offender."""

    dropped: list[tuple[str, str]] = []
    if drop:
        bars, dropped = drop_limiting_symbols(bars, benchmark)
    aligned, common = align_to_common_index(bars)
    return aligned, dropped, len(common)


def main(argv=None) -> int:
    args = _parse_args(argv)

    try:
        risk_levels = _floats(args.risk_levels, "--risk-levels")
        position_counts = _ints(args.position_counts, "--position-counts")
    except ValueError as exc:
        return _fail(str(exc))

    grid = SizingGrid(
        risk_per_trade_pcts=risk_levels, position_counts=position_counts
    )
    grid_errors = grid.validate()
    if grid_errors:
        return _fail("Invalid sizing grid:\n  - " + "\n  - ".join(grid_errors))

    credentials = load_credentials()
    if not credentials.api_key or not credentials.secret_key:
        return _fail(
            "Alpaca credentials are not configured. Set ALPACA_PAPER_API_KEY and "
            "ALPACA_PAPER_SECRET_KEY."
        )

    symbols = tuple(s.strip() for s in args.symbols.split(",") if s.strip())
    if args.benchmark not in symbols:
        return _fail(f"Benchmark {args.benchmark} must be part of the universe.")

    try:
        start_date = date.fromisoformat(args.start)
        end_date = (
            date.fromisoformat(args.end)
            if args.end
            else datetime.now(timezone.utc).date() - timedelta(days=1)
        )
    except ValueError as exc:
        return _fail(f"Dates must be YYYY-MM-DD: {exc}")
    if start_date >= end_date:
        return _fail("The start date must precede the end date.")

    strategy = CryptoStrategyConfig(
        symbols=symbols, timeframe=args.timeframe, benchmark_symbol=args.benchmark
    )
    base_risk = CryptoRiskConfig()
    backtest = CryptoBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        slippage_bps=args.slippage_bps,
        taker_fee_bps=args.taker_fee_bps,
        spread_bps=args.spread_bps,
    )
    for label, config in (
        ("strategy", strategy), ("risk", base_risk), ("backtest", backtest)
    ):
        errors = config.validate()
        if errors:
            return _fail(f"Invalid {label} configuration:\n  - " + "\n  - ".join(errors))

    print(RULE)
    print("SIZING STUDY — does quality survive deployment?")
    print(RULE)
    print(f"  Engine      {args.engine}")
    print(f"  Universe    {', '.join(symbols)}   benchmark {args.benchmark}")
    print(f"  Timeframe   {args.timeframe}   window {start_date} .. {end_date}")
    print(f"  Grid        risk {list(risk_levels)}  x  positions {list(position_counts)}")
    print("\n  Market data only. This script never constructs a broker, and never")
    print("  modifies the live exposure caps.")

    duration = timeframe_to_timedelta(args.timeframe)
    activation = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    fetch_start = activation - duration * (strategy.warmup_bars() + 5)
    fetch_end = datetime.combine(
        end_date + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc
    )

    feed = AlpacaCryptoDataFeed(credentials.api_key, credentials.secret_key)
    print(f"\n  Downloading bars from {fetch_start:%Y-%m-%d}…")
    try:
        bars = {
            symbol: feed.get_bars(symbol, args.timeframe, start=fetch_start, end=fetch_end)
            for symbol in symbols
        }
    except Exception as exc:  # noqa: BLE001 - surface any vendor failure plainly
        return _fail(f"Could not download bars: {exc}")
    for symbol, frame in bars.items():
        if frame.empty:
            return _fail(f"Alpaca returned no bars for {symbol}.")

    try:
        bars, dropped, aligned = _aligned(bars, args.benchmark, args.drop_late_listings)
    except ValueError as exc:
        return _fail(str(exc))
    for symbol, reason in dropped:
        print(f"    dropped {symbol:<10} {reason}")
    if dropped:
        # The universe the strategy is told about must match the bars it is
        # handed. Dropping a symbol from one and not the other is how this
        # failed in CI: the strategy still demanded nine symbols while eight
        # remained, and prepare_universe raised from deep inside the study.
        strategy = replace(strategy, symbols=tuple(bars))
    print(f"    aligned    {aligned:>7} bars across {len(bars)} symbols")
    if aligned < strategy.warmup_bars() + 30:
        return _fail(
            f"{aligned} aligned bars is too few to trade after a "
            f"{strategy.warmup_bars()}-bar warmup."
        )

    print("\n  Running one backtest per deployment level…")
    try:
        study = run_sizing_study(
            _backtest_fn(args.engine),
            bars,
            strategy,
            base_risk,
            backtest,
            grid=grid,
            active_start=pd.Timestamp(activation),
        )
    except Exception as exc:  # noqa: BLE001 - a study failure is not a crash
        return _fail(f"The sizing study could not complete: {exc}")

    print(f"\n{RULE}\nDEPLOYMENT LADDER\n{RULE}")
    print(f"  {'risk%':>6}{'pos':>5}{'heat%':>7}{'trades':>8}{'return%':>9}"
          f"{'maxDD%':>8}{'sharpe':>8}{'PF':>7}{'vol%':>7}{'pos%':>7}  binding")
    for point in study.points:
        fmt = lambda v, p=2: "n/a" if v is None else f"{v:.{p}f}"
        print(
            f"  {point.risk_per_trade_pct:>6.2f}{point.max_positions:>5}"
            f"{point.heat_pct:>7.2f}{point.num_trades:>8}"
            f"{point.total_return_pct:>9.2f}{point.max_drawdown_pct:>8.2f}"
            f"{fmt(point.sharpe_ratio):>8}{fmt(point.profit_factor):>7}"
            f"{point.annualised_volatility_pct:>7.1f}"
            f"{fmt(point.implied_position_pct, 1):>7}  {point.binding_constraint}"
        )

    print(f"\n{RULE}\nSCALE-FREE COMPARISON\n{RULE}")
    bench_sharpe = study.benchmark_sharpe
    print(f"  Benchmark {args.benchmark:<10} return {study.benchmark_return_pct:+.2f}%"
          f"   volatility {study.benchmark_volatility_pct:.1f}%"
          f"   Sharpe {'n/a' if bench_sharpe is None else f'{bench_sharpe:.2f}'}")
    best = max(
        (p for p in study.points if p.sharpe_ratio is not None),
        key=lambda p: p.sharpe_ratio,
        default=None,
    )
    if best is not None and bench_sharpe is not None:
        print(f"  Best strategy point         Sharpe {best.sharpe_ratio:.2f}"
              f"   at risk {best.risk_per_trade_pct:g}% / {best.max_positions} positions")
        if best.sharpe_ratio > bench_sharpe:
            print("\n  >> Higher Sharpe than the benchmark: at MATCHED volatility this")
            print("     signal would out-earn holding it. That match is not available")
            print("     here — it needs leverage the risk limits forbid and this")
            print("     repository has never tested — so this is a diagnostic about")
            print("     the signal, not a promotion argument.")
        else:
            print("\n  >> The benchmark wins on risk-adjusted terms too, not merely on")
            print("     raw return. Deployment is not the binding problem.")

    print(f"\n{RULE}\nVERDICT\n{RULE}")
    print(f"  {study.verdict()}")
    print(RULE)

    if args.json_out:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "engine": args.engine,
                "symbols": list(bars),
                "benchmark": args.benchmark,
                "timeframe": args.timeframe,
                "start": str(start_date),
                "end": str(end_date),
                "dropped": [symbol for symbol, _ in dropped],
                "aligned_bars": aligned,
            },
            "study": study.to_dict(),
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nWrote {args.json_out}")
    return 0


def _guarded_main(argv=None) -> int:
    """Turn an unexpected crash into a readable report rather than a bare exit.

    CI captures stdout and stderr into the published report, so an uncaught
    exception here is the difference between a diagnosable failure and 600
    bytes of banner followed by nothing.
    """

    try:
        return main(argv)
    except Exception:  # noqa: BLE001 - last resort, re-raised as a report
        import traceback

        print("\nUNEXPECTED FAILURE — full traceback follows:", file=sys.stderr)
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(_guarded_main())
