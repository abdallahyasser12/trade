#!/usr/bin/env python3
"""Run the crypto walk-forward protocol and print the honest verdict.

Answers the two questions the dashboard could not previously ask:

  1. What does a round trip cost, measured against the risk each trade takes?
  2. Did the strategy actually beat simply holding the benchmark?

Reads paper credentials from the environment (``ALPACA_PAPER_API_KEY`` /
``ALPACA_PAPER_SECRET_KEY``, or the legacy ``ALPACA_API_KEY`` /
``ALPACA_SECRET_KEY``), so it works unchanged from a shell with a ``.env``
or from CI with repository secrets. Market data only — this script never
constructs a broker and can never place an order.

Exit codes:
  0  promotion gates passed
  1  promotion gates failed (a normal, informative outcome)
  2  the run could not be completed (credentials, data, or configuration)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from advanced_crypto_research import run_regime_ensemble_walk_forward
from council import CouncilConfig
from council_research import run_council_walk_forward
from cross_sectional import CrossSectionalConfig
from cross_sectional_research import run_cross_sectional_walk_forward
from engine_research import SOLO_ENGINES, run_injected_walk_forward
from config import load_credentials, timeframe_to_timedelta
from cost_model import break_even_win_rate, cost_to_risk, round_trip_cost_bps
from crypto_config import (
    DEFAULT_CRYPTO_UNIVERSE,
    CryptoBacktestConfig,
    CryptoRiskConfig,
    CryptoStrategyConfig,
)
from crypto_data_feed import AlpacaCryptoDataFeed
from crypto_research import (
    CryptoPromotionCriteria,
    CryptoWalkForwardConfig,
    run_crypto_walk_forward,
)
from regime_ensemble import EnsembleConfig
from universe_alignment import aligned_length, drop_limiting_symbols

RULE = "=" * 78


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run crypto walk-forward research and print the verdict.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--symbols",
        default=",".join(DEFAULT_CRYPTO_UNIVERSE),
        help="Comma-separated Alpaca USD crypto pairs.",
    )
    parser.add_argument("--benchmark", default="BTC/USD", help="Buy-and-hold comparator.")
    parser.add_argument(
        "--timeframe",
        default="1Hour",
        choices=["15Min", "1Hour", "4Hour", "1Day"],
        help="Bar size. Widening this is the standard response to high cost-per-R.",
    )
    parser.add_argument("--start", default="2023-01-01", help="First research date (UTC).")
    parser.add_argument("--end", default="", help="Last research date (UTC). Blank = yesterday.")
    parser.add_argument(
        "--engine",
        default="base",
        choices=["base", "ensemble", "cross", "council", "meanrev", "volregime", "breadth"],
        help="Decision engine to walk forward. 'base' is the audited breakout/trend "
             "strategy. 'ensemble' is the regime-aware multi-sleeve blend. 'cross' is "
             "cross-sectional risk-adjusted momentum — a different hypothesis (relative "
             "ranking rather than an absolute filter conjunction), with six parameters "
             "instead of fifteen.",
    )
    parser.add_argument(
        "--max-cost-to-risk",
        type=float,
        default=None,
        help="Admission gate in units of R. Omit to use the config default (1.0). "
             "Tighten toward 0.25 to see the honest picture.",
    )
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--taker-fee-bps", type=float, default=25.0)
    parser.add_argument("--spread-bps", type=float, default=5.0)
    parser.add_argument("--train-bars", type=int, default=1_500)
    parser.add_argument("--test-bars", type=int, default=500)
    parser.add_argument("--min-train-trades", type=int, default=5)
    parser.add_argument(
        "--drop-late-listings",
        action="store_true",
        help="Drop symbols whose history starts late enough to truncate the "
             "aligned window for everything else. The universe is intersected "
             "to common timestamps, so one 2023 listing in a 2021 universe "
             "costs every symbol those years. Trades breadth for window length.",
    )
    parser.add_argument(
        "--max-positions",
        type=int,
        default=None,
        help="Concurrent position slots. Omit to use the config default (3). "
             "With a wide universe the portfolio heat cap, not this count, is "
             "what bounds total open risk.",
    )
    parser.add_argument(
        "--embargo-bars",
        type=int,
        default=None,
        help="Gap between each training window and its test window. Omit to "
             "derive it from the strategy's indicator warmup (currently 226 "
             "bars), so no test bar's features overlap training data. The "
             "library default of 1 bar leaks feature windows across the "
             "boundary and understates out-of-sample honesty.",
    )
    parser.add_argument(
        "--json-out", default="", help="Optional path to write the full verdict as JSON."
    )
    return parser.parse_args(argv)


def _parameter_grid(strategy: CryptoStrategyConfig, timeframe: str) -> dict:
    """Bounded search space, scaled to the bar size.

    The hourly-tuned entry defaults almost never fire on slow bars — measured
    directly: two out-of-sample trades across five years of daily data, at
    two symbols and again at eight. Slow timeframes therefore search shorter
    breakout windows and softer trend filters. Every key stays inside the
    audited TUNABLE_CRYPTO_PARAMETERS allowlist and the candidate count stays
    well under the protocol's 32-candidate cap; selection remains strictly
    in-sample per fold, judged out of sample.
    """

    if timeframe in ("4Hour", "1Day"):
        return {
            "breakout_period": sorted({10, 15, strategy.breakout_period}),
            "min_adx": sorted({15.0, strategy.min_adx}),
            "min_efficiency_ratio": sorted({0.15, strategy.min_efficiency_ratio}),
        }
    return {
        "breakout_period": [strategy.breakout_period, strategy.breakout_period + 10],
        "min_adx": [strategy.min_adx, strategy.min_adx + 5.0],
    }


def _cross_parameter_grid(strategy: CryptoStrategyConfig) -> dict:
    """Search what the cross-sectional engine actually reads.

    It never looks at breakout_period, min_adx or min_efficiency_ratio, so the
    grid above would have produced twelve behaviourally identical candidates
    per fold: twelve times the compute, an argmax decided by tie-break order,
    and a reported candidate count implying a search that never happened.

    Real candidates instead. The momentum horizon and the volatility horizon
    are tied together deliberately — measure the move over the same window you
    measure its noise — which halves the surface for free.

    The stop multiplier is the second axis because it sets the denominator of
    cost-to-risk: a wider stop means a smaller position for the same risk
    budget and therefore less friction per unit of R, and friction per unit of
    R is the constraint that has decided every experiment so far. The wide
    value has to be genuinely wider than the default or the axis collapses.

    The benchmark-regime gate is the third axis, and it earned its place by
    killing a run. With the gate always on, a long-only book is flat whenever
    BTC trades below its 200-day average — so a training window sitting inside
    the 2022 bear market produces zero trades for every candidate, no
    candidate can meet the in-sample evidence bar, and the entire walk-forward
    aborts on its first fold. Searching the gate means a bear fold still has
    something to select, and it turns "is the regime filter earning its keep?"
    from an assumption into a question each fold answers with its own data.
    """

    return {
        "cross": [
            CrossSectionalConfig(
                lookback=window,
                vol_lookback=window,
                top_k=top_k,
                require_benchmark_bull=gated,
            )
            for window in (20, 30, 45)
            for top_k in (2, 3)
            for gated in (True, False)
        ],
        "atr_stop_multiplier": sorted({strategy.atr_stop_multiplier, 4.0}),
    }


def _drop_late_listings(bars: dict, benchmark: str, *, min_symbols: int = 3):
    """Thin alias; the implementation is shared with the sizing study."""

    return drop_limiting_symbols(bars, benchmark, min_symbols=min_symbols)


def _aligned_length(bars: dict) -> int:
    return aligned_length(bars)


def _json_grid(grid) -> dict:
    """Grid values are plain numbers except the cross-sectional axis."""

    return {
        key: [
            value.to_dict() if hasattr(value, "to_dict") else value for value in values
        ]
        for key, values in grid.items()
    }


def _fail(message: str) -> int:
    print(f"\nERROR: {message}", file=sys.stderr)
    return 2


def _fmt(value, suffix="", places=2):
    if value is None:
        return "n/a"
    if isinstance(value, float) and not math.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{places}f}{suffix}"


def _cost_section(strategy: CryptoStrategyConfig, backtest: CryptoBacktestConfig, result) -> dict:
    """What a round trip costs, in bps and in units of the risk taken."""

    round_trip = round_trip_cost_bps(
        slippage_bps=backtest.slippage_bps,
        spread_bps=backtest.spread_bps,
        taker_fee_bps=backtest.taker_fee_bps,
    )
    print(f"\n{RULE}\nCOST\n{RULE}")
    print(f"  Round-trip friction        {round_trip:.1f} bps"
          f"  (2 x [{backtest.slippage_bps:g} slip + {backtest.spread_bps / 2:g} half-spread"
          f" + {backtest.taker_fee_bps:g} fee])")

    samples = [
        fold.test_result.median_cost_to_risk
        for fold in result.folds
        if fold.test_result.median_cost_to_risk is not None
    ]
    median_r = float(pd.Series(samples).median()) if samples else None
    if median_r is None:
        print("  Cost per trade             n/a (no trades were taken)")
        print("  Break-even win rate        n/a")
        verdict = "no trades"
    else:
        win_rate = break_even_win_rate(median_r, reward_to_risk=2.0)
        print(f"  Cost per trade             {median_r:.2f} R"
              f"   (friction as a share of entry-minus-stop)")
        print("  Break-even win rate        "
              + ("unreachable" if math.isnan(win_rate) else f"{win_rate:.1%}")
              + "   at 2R reward:risk")
        print("  Trend systems typically win 35-45% of trades.")
        if math.isnan(win_rate) or win_rate > 0.50:
            verdict = "PROHIBITIVE"
            print(f"\n  >> Cost is {median_r:.2f}R per trade. The required win rate is above"
                  f"\n     what trend-following realistically achieves. Widen --timeframe"
                  f"\n     (4Hour/1Day) or use maker entries. Do not loosen the gate.")
        elif median_r > 0.25:
            verdict = "HIGH"
            print(f"\n  >> Cost is {median_r:.2f}R per trade — a large share of what each"
                  f"\n     trade risks. Treat any edge shown below as fragile.")
        else:
            verdict = "ACCEPTABLE"
            print(f"\n  >> Cost is {median_r:.2f}R per trade, within a workable range.")
    return {"round_trip_bps": round_trip, "median_cost_to_risk": median_r, "verdict": verdict}


def _benchmark_section(result, benchmark_symbol: str) -> dict:
    """Did the strategy beat simply holding the asset?"""

    print(f"\n{RULE}\nVERSUS BUY-AND-HOLD {benchmark_symbol}\n{RULE}")
    print(f"  {'Fold':<6}{'Window':<26}{'Strategy':>12}{'Buy & hold':>13}{'Excess':>11}")
    for number, fold in enumerate(result.folds, start=1):
        strat = fold.test_result.total_return_pct
        bench = fold.test_result.benchmark_return_pct
        window = (f"{fold.test_start:%Y-%m-%d}..{fold.test_end:%Y-%m-%d}")
        flag = "" if strat > bench else "  <-- lost to holding"
        print(f"  {number:<6}{window:<26}{strat:>11.2f}%{bench:>12.2f}%"
              f"{strat - bench:>+10.2f}%{flag}")
    print(f"  {'-' * 68}")
    print(f"  {'ALL':<6}{'compounded':<26}"
          f"{result.compounded_return_pct:>11.2f}%"
          f"{result.benchmark_compounded_return_pct:>12.2f}%"
          f"{result.benchmark_excess_return_pct:>+10.2f}%")
    print(f"\n  Folds beating buy-and-hold  {result.benchmark_beating_fold_ratio:.0%}")

    if result.benchmark_excess_return_pct > 0:
        print("\n  >> The strategy outperformed holding the benchmark out of sample.")
    else:
        print("\n  >> Holding the benchmark outperformed the strategy out of sample."
              "\n     All the complexity is currently costing you money versus buying"
              "\n     and doing nothing.")
    return {
        "strategy_pct": result.compounded_return_pct,
        "benchmark_pct": result.benchmark_compounded_return_pct,
        "excess_pct": result.benchmark_excess_return_pct,
        "beating_fold_ratio": result.benchmark_beating_fold_ratio,
    }


def main(argv=None) -> int:
    args = _parse_args(argv)

    credentials = load_credentials()
    if not credentials.api_key or not credentials.secret_key:
        return _fail(
            "No Alpaca paper credentials found. Set ALPACA_PAPER_API_KEY and\n"
            "ALPACA_PAPER_SECRET_KEY (or the legacy ALPACA_API_KEY /\n"
            "ALPACA_SECRET_KEY) in the environment, a .env file, or CI secrets."
        )

    symbols = tuple(s.strip().upper() for s in args.symbols.split(",") if s.strip())
    try:
        start_date = date.fromisoformat(args.start)
        end_date = (
            date.fromisoformat(args.end)
            if args.end
            else (datetime.now(timezone.utc) - timedelta(days=1)).date()
        )
    except ValueError as exc:
        return _fail(f"Bad date: {exc}")

    strategy = CryptoStrategyConfig(
        symbols=symbols, benchmark_symbol=args.benchmark, timeframe=args.timeframe
    )
    risk_kwargs = {}
    if args.max_cost_to_risk is not None:
        risk_kwargs["max_cost_to_risk"] = args.max_cost_to_risk
    if args.max_positions is not None:
        risk_kwargs["max_positions"] = args.max_positions
    risk = CryptoRiskConfig(**risk_kwargs)
    backtest = CryptoBacktestConfig(
        start_date=start_date,
        end_date=end_date,
        slippage_bps=args.slippage_bps,
        taker_fee_bps=args.taker_fee_bps,
        spread_bps=args.spread_bps,
    )
    embargo = (
        args.embargo_bars if args.embargo_bars is not None else strategy.warmup_bars()
    )
    walk = CryptoWalkForwardConfig(
        train_bars=args.train_bars,
        test_bars=args.test_bars,
        step_bars=args.test_bars,
        embargo_bars=embargo,
        min_train_trades=args.min_train_trades,
    )
    criteria = CryptoPromotionCriteria()

    errors = strategy.validate() + risk.validate() + backtest.validate() + walk.validate()
    if errors:
        return _fail("Invalid configuration:\n  - " + "\n  - ".join(errors))

    print(RULE)
    print("CRYPTO WALK-FORWARD RESEARCH")
    print(RULE)
    print(f"  Universe    {', '.join(symbols)}   benchmark {args.benchmark}")
    print(f"  Engine      {args.engine}")
    print(f"  Timeframe   {args.timeframe}")
    print(f"  Window      {start_date} .. {end_date}")
    print(f"  Folds       {args.train_bars} train / {args.test_bars} test bars, "
          f"embargo {embargo} (warmup-derived)")
    print(f"  Cost gate   max_cost_to_risk = {risk.max_cost_to_risk:.2f} R")
    print("\n  Market data only. This script never constructs a broker.")

    # Fetch enough history before the activation date to warm up indicators.
    duration = timeframe_to_timedelta(args.timeframe)
    activation = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    fetch_start = activation - duration * (strategy.warmup_bars() + walk.train_bars + 5)
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
        print(f"    {symbol:<10} {len(frame):>7} bars  "
              f"{frame.index[0]:%Y-%m-%d} .. {frame.index[-1]:%Y-%m-%d}")

    # The simulator requires exactly aligned indices across the universe, so a
    # single late listing truncates every other symbol down to its own start.
    # That is easy to do by accident and expensive: one 2023 listing in an
    # otherwise 2021 universe costs two and a half years of history for all of
    # them. Name the symbol responsible rather than letting the fold arithmetic
    # fail later with something vague.
    if args.drop_late_listings:
        bars, dropped = _drop_late_listings(bars, args.benchmark)
        for symbol, reason in dropped:
            print(f"    dropped {symbol:<10} {reason}")
        if dropped:
            symbols = tuple(bars)
            strategy = replace(strategy, symbols=symbols)

    common = None
    for frame in bars.values():
        common = frame.index if common is None else common.intersection(frame.index)
    if common is None or len(common) == 0:
        return _fail("The symbols share no common timestamps.")
    # Measured before the reassignment below, or every frame is already the
    # aligned length and the comparison can never fire.
    longest = max(len(frame) for frame in bars.values())
    binding, first = max(
        ((symbol, frame.index[0]) for symbol, frame in bars.items()),
        key=lambda item: item[1],
    )
    bars = {symbol: frame.loc[common] for symbol, frame in bars.items()}
    if len(common) < longest:
        lost = longest - len(common)
        print(
            f"    NOTE: alignment cost {lost} bars ({lost / longest:.0%} of the "
            f"longest series); {binding} starts {first:%Y-%m-%d}."
        )
        if not args.drop_late_listings:
            print("          Pass --drop-late-listings to trade breadth for window length.")

    needed = walk.train_bars + walk.embargo_bars + walk.test_bars
    if len(common) < needed:
        return _fail(
            f"{len(common)} aligned bars is too few for one fold: "
            f"{walk.train_bars} train + {walk.embargo_bars} embargo + "
            f"{walk.test_bars} test = {needed} required.\n"
            "Widen --start, shorten the folds, or drop the late-listing symbol "
            "named above (--drop-late-listings)."
        )
    print(f"    aligned    {len(common):>7} bars across {len(bars)} symbols")

    if args.engine == "cross":
        grid = _cross_parameter_grid(strategy)
    elif args.engine == "council":
        grid = _council_parameter_grid()
    elif args.engine in SOLO_ENGINES and args.engine != "base":
        # One candidate, no in-sample selection, and therefore no
        # multiple-testing burden to discount. These runs exist to be the
        # comparison baseline the council must beat, so they are deliberately
        # given the cleanest possible protocol rather than the best chance of
        # looking good.
        grid = {}
    else:
        grid = _parameter_grid(strategy, args.timeframe)

    print("\n  Running bounded chronological walk-forward…")
    try:
        if args.engine == "cross":
            result = run_cross_sectional_walk_forward(
                bars, strategy, risk, backtest, grid, walk, criteria,
            )
        elif args.engine == "council":
            result = run_council_walk_forward(
                bars, strategy, risk, backtest, grid, walk, criteria,
            )
        elif args.engine in SOLO_ENGINES and args.engine != "base":
            result = run_injected_walk_forward(
                SOLO_ENGINES[args.engine],
                bars, strategy, risk, backtest, grid, walk, criteria,
                engine_name=args.engine,
            )
        elif args.engine == "ensemble":
            result = run_regime_ensemble_walk_forward(
                bars, strategy, risk, backtest, grid, walk, criteria,
                ensemble_config=EnsembleConfig(),
            )
        else:
            result = run_crypto_walk_forward(
                bars, strategy, risk, backtest, grid, walk, criteria
            )
    except ValueError as exc:
        return _fail(
            f"Walk-forward could not complete: {exc}\n"
            "Three causes account for almost every occurrence:\n"
            "  - too little aligned history for the requested fold sizes (the note\n"
            "    above says which symbol cost the most bars; --drop-late-listings\n"
            "    removes it),\n"
            "  - a cost gate strict enough that no candidate met --min-train-trades,\n"
            "  - a training window sitting entirely inside a bear market, where a\n"
            "    regime-gated long-only book never trades and so cannot be selected\n"
            "    on. Lengthen --train-bars so each window spans both regimes."
        )

    cost = _cost_section(strategy, backtest, result)
    benchmark = _benchmark_section(result, args.benchmark)

    print(f"\n{RULE}\nOUT-OF-SAMPLE EVIDENCE\n{RULE}")
    print(f"  Folds                      {len(result.folds)}")
    print(f"  Candidates per fold        {result.candidate_count}")
    print(f"  OOS trades                 {result.total_oos_trades}")
    print(f"  Aggregate Sharpe           {_fmt(result.aggregate_sharpe)}")
    print(f"  Profit factor              {_fmt(result.profit_factor)}")
    print(f"  Worst fold drawdown        -{result.worst_fold_drawdown_pct:.2f}%")
    print(f"  Positive folds             {result.positive_fold_ratio:.0%}")
    print(f"  Stressed OOS return        {result.stressed_compounded_return_pct:+.2f}%")

    if args.engine == "cross":
        # Which parameters each fold selected in-sample. Agreement across folds
        # is weak evidence of a stable optimum; disagreement means the search
        # is fitting fold-specific noise and the result should be read as such.
        print("\n  Selected per fold (in-sample):")
        for index, fold in enumerate(result.folds, start=1):
            chosen = getattr(fold.selected_config, "cross", None)
            if chosen is None:
                continue
            print(
                f"    fold {index:<2} lookback {chosen.lookback:>3}"
                f"  top_k {chosen.top_k}"
                f"  stop {fold.selected_config.atr_stop_multiplier:g}x ATR"
                f"  regime gate {'on' if chosen.require_benchmark_bull else 'off'}"
            )

    print(f"\n{RULE}\nPROMOTION VERDICT\n{RULE}")
    if result.promotion.promoted:
        print("  PASS — eligible for controlled PAPER testing only.")
        print("\n  This does not unlock live trading and does not copy anything into a")
        print("  running engine. Supervise it through several market regimes first.")
    else:
        print("  FAIL — do not promote this configuration.")
        for failure in result.promotion.failures:
            print(f"    - {failure}")
        print("\n  A failure here is information, not a defect. The correct response is a")
        print("  different timeframe, universe, or cost structure — never a looser gate.")
    print(RULE)

    if args.json_out:
        payload = {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "config": {
                "symbols": list(symbols),
                "benchmark": args.benchmark,
                "engine": args.engine,
                "timeframe": args.timeframe,
                "start": str(start_date),
                "end": str(end_date),
                "max_cost_to_risk": risk.max_cost_to_risk,
                "slippage_bps": args.slippage_bps,
                "taker_fee_bps": args.taker_fee_bps,
                "spread_bps": args.spread_bps,
                "embargo_bars": embargo,
                "grid": _json_grid(grid),
            },
            "cost": cost,
            "benchmark": benchmark,
            "evidence": {
                "folds": len(result.folds),
                "candidates": result.candidate_count,
                "oos_trades": result.total_oos_trades,
                "aggregate_sharpe": result.aggregate_sharpe,
                "profit_factor": (
                    result.profit_factor
                    if result.profit_factor is None or math.isfinite(result.profit_factor)
                    else None
                ),
                "worst_fold_drawdown_pct": result.worst_fold_drawdown_pct,
                "positive_fold_ratio": result.positive_fold_ratio,
                "stressed_return_pct": result.stressed_compounded_return_pct,
            },
            "promotion": {
                "promoted": result.promotion.promoted,
                "failures": list(result.promotion.failures),
            },
        }
        Path(args.json_out).write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nWrote {args.json_out}")

    return 0 if result.promotion.promoted else 1


if __name__ == "__main__":
    raise SystemExit(main())
