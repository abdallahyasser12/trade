#!/usr/bin/env python3
"""Run the quant-desk walk-forward factory against US equities.

``scripts/run_quant_research.py`` is crypto-only by construction: all
three of its strategy factories hardcode ``Market.CRYPTO``, its bars come
from ``AlpacaCryptoDataFeed``, and it annualizes with
``crypto_bars_per_year``.  There was no equity equivalent, so the equity
side of the desk — including ``opening_range_breakout``, the one
equities-only strategy in the registry — had no research entry point at
all.  This is that entry point.

Three things it does differently from the crypto CLI, each because
equities are not crypto:

**Annualization uses the equity calendar.**  ``config.bars_per_year``
gives 252*7 = 1764 hourly bars a year; ``crypto_bars_per_year`` gives
8766.  Annualizing hourly equity returns with the crypto factor
overstates Sharpe by sqrt(8766/1764) ~ 2.2x, which is precisely the kind
of error that manufactures a passing gate.

**Bars come through the research fetcher, not the live feed.**
``quant.data.EquityHistoryFetcher`` chunks on calendar boundaries,
caches, filters to regular hours, and reports completeness against the
real NYSE session calendar.  A symbol whose history is incomplete is
reported, not silently researched as though it simply had no signals.

**Data quality gates the run.**  A universe where symbols are MISSING or
materially INCOMPLETE fails closed by default (``--allow-incomplete`` to
override, which is recorded in the run's provenance).

Exit codes follow the research-CLI convention: 0 = all gates PASS
(eligible for SHADOW evaluation, nothing more), 1 = gates FAIL, 2 = could
not run.

``--synthetic`` runs the identical pipeline on deterministic
session-shaped data.  That validates the machinery — folds, gates,
stress, deflated Sharpe, recording — and is NEVER evidence of edge; those
runs are recorded under a strategy id suffixed ``_synthetic`` so they can
never satisfy a real promotion request.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import bars_per_year, load_credentials, validate_symbol
from quant.contracts import Market
from quant.data.cache import DEFAULT_CACHE_DIR, BarCache
from quant.data.calendar import regular_session_bounds, trading_days
from quant.data.equity_history import (
    SUPPORTED_TIMEFRAMES,
    EquityHistoryFetcher,
)
from quant.data.provenance import DataQuality
from quant.exits import ExitPolicyConfig
from quant.research.factory import (
    QuantPromotionCriteria,
    QuantWalkForwardConfig,
    record_research_report,
    run_quant_walk_forward,
)
from quant.research.simulator import SimulationConfig
from quant.store import DEFAULT_QUANT_DB_PATH, QuantStore

ENGINES = ("orb", "vwap", "donchian", "squeeze")

# Liquid, long-history, and deliberately small. A default universe is a
# convenience, not a recommendation: see scripts/discover_stock_universe
# discussion in the report for why point-in-time membership matters.
DEFAULT_UNIVERSE = ("SPY", "QQQ", "IWM")

_STRATEGY_IDS = {
    "orb": "opening_range_breakout",
    "vwap": "vwap_reversion",
    "donchian": "donchian_trend",
    "squeeze": "volatility_squeeze",
}


def _fail(message: str) -> int:
    print(f"\nCOULD NOT RUN: {message}", file=sys.stderr)
    return 2


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent.parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


# ---------------------------------------------------------------------------
# Strategy grids
# ---------------------------------------------------------------------------


def _strategy_factory(engine: str, symbols: tuple[str, ...], timeframe: str, benchmark: str):
    """Return (build, grid) for one engine, all bound to US_EQUITY.

    Grids are deliberately small and centered on documented defaults. A
    wide grid is not free: every candidate multiplies the run's
    multiple-testing burden (candidates x folds), which the deflated
    Sharpe correctly penalizes.
    """
    common = dict(
        symbols=symbols, market=Market.US_EQUITY, timeframe=timeframe,
        benchmark_symbol=benchmark,
    )

    if engine == "orb":
        from quant.strategies.opening_range import (
            OpeningRangeBreakoutStrategy,
            OpeningRangeConfig,
        )

        def build(params: dict):
            return OpeningRangeBreakoutStrategy(OpeningRangeConfig(**common, **params))

        grid = [
            {"opening_range_bars": bars, "atr_stop_multiplier": stop}
            for bars in (2, 4, 6)
            for stop in (1.5, 2.0)
        ]
        return build, grid

    if engine == "vwap":
        from quant.strategies.vwap_reversion import (
            VwapReversionConfig,
            VwapReversionStrategy,
        )

        def build(params: dict):
            return VwapReversionStrategy(VwapReversionConfig(**common, **params))

        grid = [
            {"entry_z": entry, "atr_stop_multiplier": stop}
            for entry in (-1.5, -2.0, -2.5)
            for stop in (1.5, 2.0)
        ]
        return build, grid

    if engine == "donchian":
        from quant.strategies.donchian_trend import (
            DonchianTrendConfig,
            DonchianTrendStrategy,
        )

        def build(params: dict):
            return DonchianTrendStrategy(DonchianTrendConfig(**common, **params))

        grid = [
            {"breakout_period": breakout, "atr_stop_multiplier": stop}
            for breakout in (15, 20, 30)
            for stop in (2.0, 3.0)
        ]
        return build, grid

    from quant.strategies.volatility_squeeze import (
        VolatilitySqueezeConfig,
        VolatilitySqueezeStrategy,
    )

    def build(params: dict):
        return VolatilitySqueezeStrategy(VolatilitySqueezeConfig(**common, **params))

    grid = [
        {"squeeze_percentile_max": squeeze, "atr_stop_multiplier": stop}
        for squeeze in (20.0, 30.0)
        for stop in (2.0, 3.0)
    ]
    return build, grid


# ---------------------------------------------------------------------------
# Bars
# ---------------------------------------------------------------------------


def _synthetic_equity_bars(
    symbols: tuple[str, ...], timeframe: str, sessions: int
) -> tuple[dict, dict]:
    """Deterministic bars laid out on REAL trading sessions.

    Session-shaped rather than a flat date_range: an opening-range
    strategy is meaningless on a series with no session boundaries, and a
    flat range would silently include weekends and holidays.
    """
    from config import timeframe_to_timedelta

    end = date(2024, 12, 31)
    start = end - timedelta(days=int(sessions * 1.6) + 20)
    days = trading_days(start, end)[-sessions:]
    if not days:
        raise ValueError("synthetic run produced no trading sessions")

    stamps: list[datetime] = []
    for day in days:
        if timeframe == "1Day":
            stamps.append(datetime.combine(day, time(4, 0), tzinfo=timezone.utc))
            continue
        open_utc, close_utc = regular_session_bounds(day)
        duration = timeframe_to_timedelta(timeframe)
        cursor = open_utc
        while cursor < close_utc:
            stamps.append(cursor)
            cursor += duration

    index = pd.DatetimeIndex(stamps, name="timestamp")
    out = {}
    for offset, symbol in enumerate(symbols):
        rng = np.random.default_rng(17 + offset)
        n = len(index)
        t = np.arange(n)
        base = 100.0 * np.exp(0.05 * t / max(1, n))
        cycle = 1.0 + 0.04 * np.sin(t / (29.0 + 7 * offset))
        noise = np.exp(np.cumsum(rng.normal(0, 0.004, n)))
        close = base * cycle * noise
        frame = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.0008, n)),
                "high": close * (1 + np.abs(rng.normal(0, 0.003, n))),
                "low": close * (1 - np.abs(rng.normal(0, 0.003, n))),
                "close": close,
                "volume": rng.uniform(1_000, 20_000, n),
            },
            index=index,
        )
        frame["high"] = frame[["open", "high", "close"]].max(axis=1)
        frame["low"] = frame[["open", "low", "close"]].min(axis=1)
        out[symbol] = frame
    return out, {}


def _real_equity_bars(
    symbols: tuple[str, ...],
    timeframe: str,
    start: date,
    end: date,
    *,
    feed: str,
    cache_dir: str | None,
    refresh: bool,
) -> tuple[dict, dict]:
    """Fetch real bars through the research fetcher, with provenance."""
    from data_feed import AlpacaDataFeed

    credentials = load_credentials()
    if not credentials.is_configured:
        raise RuntimeError(
            "No Alpaca credentials configured. Set ALPACA_PAPER_API_KEY and "
            "ALPACA_PAPER_SECRET_KEY, or use --synthetic for a "
            "machinery-validation run (which is never evidence of edge)."
        )
    source = AlpacaDataFeed(credentials.api_key, credentials.secret_key, feed=feed)
    fetcher = EquityHistoryFetcher(
        source,
        cache=BarCache(cache_dir) if cache_dir else None,
        feed_name=feed,
    )
    histories = fetcher.fetch_many(
        symbols, timeframe, start, end, skip_unusable=False
    )
    bars = {s: h.bars for s, h in histories.items() if not h.bars.empty}
    provenance = {s: h.provenance.to_dict() for s, h in histories.items()}
    print(
        f"Fetched {len(bars)}/{len(symbols)} symbols "
        f"({fetcher.api_calls} API calls, {fetcher.cache_hits} cache hits)"
    )
    return bars, provenance


def _quality_problems(provenance: dict) -> list[str]:
    problems = []
    for symbol, record in sorted(provenance.items()):
        quality = record.get("quality")
        if quality in (DataQuality.OK.value, DataQuality.STALE.value):
            continue
        coverage = record.get("completeness", {}).get("coverage", 0.0)
        problems.append(f"{symbol}: {quality} (coverage {coverage:.1%})")
    return problems


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--engine", choices=ENGINES, default="orb")
    parser.add_argument(
        "--timeframe", default="15Min", choices=list(SUPPORTED_TIMEFRAMES),
        help="4Hour is unsupported for equities: a 6.5h session does not "
             "divide into 4-hour buckets.",
    )
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_UNIVERSE))
    parser.add_argument(
        "--benchmark", default="",
        help="Must be one of --symbols; defaults to the first.",
    )
    parser.add_argument("--start", help="YYYY-MM-DD (market-local).")
    parser.add_argument("--end", help="YYYY-MM-DD (market-local).")
    parser.add_argument(
        "--days", type=int, default=365,
        help="Calendar days back from today when --start is omitted.",
    )
    parser.add_argument("--feed", default="iex", choices=("iex", "sip"))
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE_DIR)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--refresh", action="store_true", help="Bypass cached bars.")
    parser.add_argument(
        "--allow-incomplete", action="store_true",
        help="Proceed despite MISSING/INCOMPLETE symbols. Recorded in provenance.",
    )
    parser.add_argument("--synthetic", action="store_true",
                        help="Deterministic session-shaped data; validates the pipeline only.")
    parser.add_argument("--synthetic-sessions", type=int, default=400)
    # Walk-forward window. Exposed because the right window depends on the
    # timeframe: 500 training bars is two years of daily bars but under
    # three weeks of 15-minute bars, and the fold count that falls out of
    # it drives both runtime and the multiple-testing burden.
    parser.add_argument("--train-bars", type=int, default=500)
    parser.add_argument("--test-bars", type=int, default=150)
    parser.add_argument("--step-bars", type=int, default=150)
    parser.add_argument("--embargo-bars", type=int, default=1)
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.35)
    parser.add_argument("--slippage-bps", type=float, default=3.0)
    parser.add_argument("--spread-bps", type=float, default=2.0)
    parser.add_argument(
        "--fee-bps", type=float, default=0.0,
        help="US equities are commission-free on Alpaca; the cost is spread "
             "and slippage, set above.",
    )
    parser.add_argument("--seed", type=int, default=None,
                        help="Recorded in the run's lineage for reproducibility.")
    parser.add_argument("--db", default=DEFAULT_QUANT_DB_PATH)
    parser.add_argument("--json", help="Write the full report to this path.")
    parser.add_argument("--no-record", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    symbols = tuple(dict.fromkeys(args.symbols))
    if not symbols:
        return _fail("at least one symbol is required")
    for symbol in symbols:
        error = validate_symbol(symbol)
        if error:
            return _fail(f"{symbol}: {error}")

    benchmark = args.benchmark or symbols[0]
    if benchmark not in symbols:
        return _fail(f"benchmark {benchmark!r} must be one of --symbols")

    try:
        end = date.fromisoformat(args.end) if args.end else date.today()
        start = (
            date.fromisoformat(args.start)
            if args.start
            else end - timedelta(days=args.days)
        )
    except ValueError as error:
        return _fail(f"bad date: {error}")
    if end < start:
        return _fail(f"--end {end} precedes --start {start}")

    provenance: dict = {}
    try:
        if args.synthetic:
            bars, provenance = _synthetic_equity_bars(
                symbols, args.timeframe, args.synthetic_sessions
            )
        else:
            bars, provenance = _real_equity_bars(
                symbols, args.timeframe, start, end,
                feed=args.feed,
                cache_dir=None if args.no_cache else args.cache_dir,
                refresh=args.refresh,
            )
    except Exception as error:
        return _fail(str(error))

    if not bars:
        return _fail("no usable bars for any requested symbol")

    problems = _quality_problems(provenance)
    if problems:
        print("\nData-quality problems:")
        for problem in problems:
            print(f"  - {problem}")
        if not args.allow_incomplete:
            return _fail(
                "refusing to research an incomplete universe. Researching "
                "around missing history makes a strategy look like it had no "
                "signals when it simply had no data. Re-run with "
                "--allow-incomplete to proceed anyway (it is recorded)."
            )
        print("  proceeding anyway (--allow-incomplete)")

    usable = tuple(s for s in symbols if s in bars)
    if benchmark not in usable:
        return _fail(f"benchmark {benchmark} has no usable bars")

    build, grid = _strategy_factory(args.engine, usable, args.timeframe, benchmark)
    strategy_id = _STRATEGY_IDS[args.engine]
    if args.synthetic:
        strategy_id += "_synthetic"

    walk_config = QuantWalkForwardConfig(
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=args.step_bars,
        embargo_bars=args.embargo_bars,
    )
    walk_errors = walk_config.validate()
    if walk_errors:
        return _fail("; ".join(walk_errors))

    store = QuantStore(args.db)
    try:
        prior = store.total_trials(strategy_id)
        report = run_quant_walk_forward(
            strategy_id=strategy_id,
            strategy_factory=build,
            candidate_params=grid,
            bars_by_symbol=bars,
            sim_config=SimulationConfig(
                risk_per_trade_pct=args.risk_per_trade_pct,
                slippage_bps=args.slippage_bps,
                spread_bps=args.spread_bps,
                fee_bps=args.fee_bps,
                # The equity calendar, NOT crypto's 24/7 factor. Using
                # crypto's would overstate hourly Sharpe by ~2.2x.
                annualization_factor=bars_per_year(args.timeframe),
            ),
            exit_policy=ExitPolicyConfig(
                enable_trailing_stop=True, enable_runner=True
            ),
            walk_config=walk_config,
            criteria=QuantPromotionCriteria(),
            prior_trials=prior,
        )
    except ValueError as error:
        store.close()
        return _fail(str(error))

    label = "SYNTHETIC (machinery validation only)" if args.synthetic else "REAL DATA"
    print(f"\n=== Equity walk-forward: {strategy_id} [{label}] ===")
    print(f"universe={list(usable)}  timeframe={args.timeframe}  "
          f"window={start.isoformat()}..{end.isoformat()}")
    print(f"folds={len(report.folds)}  candidates={report.candidate_count}"
          f"  trials_this_run={report.trials_this_run}"
          f"  cumulative_trials={report.trial_count}"
          f"  oos_trades={report.total_oos_trades}")
    print(f"compounded={report.compounded_return_pct:+.2f}%"
          f"  benchmark={report.benchmark_compounded_return_pct:+.2f}%"
          f"  stressed={report.stressed_compounded_return_pct:+.2f}%")
    print("  (benchmark is computed WITHOUT transaction costs; the "
          "comparison is not like-for-like)")

    print("\nGates:")
    for gate in report.gates:
        verdict = "PASS" if gate.passed else "FAIL"
        value = "unmeasured" if gate.value is None else f"{gate.value:.4f}"
        print(f"  [{verdict}] {gate.name:<38} value={value:<12} "
              f"threshold={gate.threshold:.4f}")

    if args.json:
        payload = {
            "strategy_id": strategy_id,
            "market": Market.US_EQUITY.value,
            "timeframe": args.timeframe,
            "universe": list(usable),
            "window": {"start": start.isoformat(), "end": end.isoformat()},
            "synthetic": bool(args.synthetic),
            "folds": len(report.folds),
            "oos_trades": report.total_oos_trades,
            "compounded_return_pct": report.compounded_return_pct,
            "benchmark_return_pct": report.benchmark_compounded_return_pct,
            "stressed_return_pct": report.stressed_compounded_return_pct,
            "trials_this_run": report.trials_this_run,
            "cumulative_trials": report.trial_count,
            "passed": report.passed,
            "gates": report.gates_dict(),
            "protocol": report.protocol,
            "data_provenance": provenance,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nWrote {args.json}")

    if not args.no_record:
        run_id = record_research_report(
            store, report, git_commit=_git_commit(), seed=args.seed
        )
        print(f"\nRecorded as research run #{run_id} in {args.db} "
              f"(cumulative trials now {store.total_trials(strategy_id)}).")
    store.close()

    if report.passed:
        print("\nVERDICT: PASS — eligible for SHADOW evaluation only. "
              "This is not a live unlock and not a profitability claim."
              + (" Synthetic data: NOT evidence of edge." if args.synthetic else ""))
        return 0
    print("\nVERDICT: FAIL — gates above show exactly why.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
