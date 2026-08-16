#!/usr/bin/env python3
"""Run the quant-desk walk-forward factory for one strategy family.

Exit codes follow the research-CLI convention: 0 = all gates PASS
(eligible for SHADOW, nothing more), 1 = gates FAIL, 2 = could not run.

With Alpaca credentials configured, bars come from the real crypto feed.
``--synthetic`` runs the identical pipeline on deterministic synthetic
data instead — that validates the machinery (folds, gates, stress,
deflated Sharpe, recording), and is clearly labeled as machinery
validation: synthetic results are NEVER evidence of edge and are recorded
under a strategy id suffixed ``_synthetic`` so they can never satisfy a
real promotion request.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import load_credentials
from crypto_config import crypto_bars_per_year
from quant.contracts import Market
from quant.exits import ExitPolicyConfig
from quant.research.factory import (
    QuantPromotionCriteria,
    QuantWalkForwardConfig,
    record_research_report,
    run_quant_walk_forward,
)
from quant.research.simulator import SimulationConfig
from quant.store import DEFAULT_QUANT_DB_PATH, QuantStore

ENGINES = ("donchian", "squeeze", "vwap")

DEFAULT_UNIVERSE = ("BTC/USD", "ETH/USD", "SOL/USD")


def _fail(message: str) -> int:
    print(f"\nCOULD NOT RUN: {message}", file=sys.stderr)
    return 2


def _git_commit() -> str | None:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                capture_output=True, text=True, check=True,
                cwd=Path(__file__).resolve().parent.parent,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def _strategy_factory(engine: str, symbols: tuple[str, ...], timeframe: str):
    if engine == "donchian":
        from quant.strategies.donchian_trend import (
            DonchianTrendConfig,
            DonchianTrendStrategy,
        )

        def build(params: dict):
            return DonchianTrendStrategy(
                DonchianTrendConfig(
                    symbols=symbols, market=Market.CRYPTO, timeframe=timeframe,
                    **params,
                )
            )

        grid = [
            {"breakout_period": breakout, "atr_stop_multiplier": stop}
            for breakout in (15, 20, 30)
            for stop in (2.0, 3.0)
        ]
        return build, grid
    if engine == "squeeze":
        from quant.strategies.volatility_squeeze import (
            VolatilitySqueezeConfig,
            VolatilitySqueezeStrategy,
        )

        def build(params: dict):
            return VolatilitySqueezeStrategy(
                VolatilitySqueezeConfig(
                    symbols=symbols, market=Market.CRYPTO, timeframe=timeframe,
                    **params,
                )
            )

        grid = [
            {"squeeze_percentile_max": squeeze, "atr_stop_multiplier": stop}
            for squeeze in (20.0, 30.0)
            for stop in (2.0, 3.0)
        ]
        return build, grid
    from quant.strategies.vwap_reversion import (
        VwapReversionConfig,
        VwapReversionStrategy,
    )

    def build(params: dict):
        return VwapReversionStrategy(
            VwapReversionConfig(
                symbols=symbols, market=Market.CRYPTO, timeframe=timeframe,
                **params,
            )
        )

    grid = [
        {"entry_z": entry, "atr_stop_multiplier": stop}
        for entry in (-1.5, -2.0, -2.5)
        for stop in (1.5, 2.0)
    ]
    return build, grid


def _synthetic_bars(symbols: tuple[str, ...], periods: int, timeframe: str) -> dict:
    """Deterministic trending-with-pullbacks series per symbol."""
    freq = {"1Hour": "1h", "4Hour": "4h", "1Day": "1D", "15Min": "15min"}[timeframe]
    index = pd.date_range("2023-01-01", periods=periods, freq=freq, tz="UTC")
    out = {}
    for offset, symbol in enumerate(symbols):
        rng = np.random.default_rng(11 + offset)
        t = np.arange(periods)
        base = 100.0 * np.exp(0.04 * t / 100.0)
        cycle = 1.0 + 0.06 * np.sin(t / (37.0 + 5 * offset))
        noise = np.exp(np.cumsum(rng.normal(0, 0.008, periods)))
        close = base * cycle * noise
        out[symbol] = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.001, periods)),
                "high": close * (1 + np.abs(rng.normal(0, 0.004, periods))),
                "low": close * (1 - np.abs(rng.normal(0, 0.004, periods))),
                "close": close,
                "volume": rng.uniform(500, 5_000, periods),
            },
            index=index,
        )
        frame = out[symbol]
        frame["high"] = frame[["open", "high", "close"]].max(axis=1)
        frame["low"] = frame[["open", "low", "close"]].min(axis=1)
    return out


def _real_bars(symbols: tuple[str, ...], timeframe: str, days: int) -> dict:
    from crypto_data_feed import AlpacaCryptoDataFeed

    credentials = load_credentials()
    if not credentials.is_configured:
        raise RuntimeError(
            "No Alpaca credentials configured; use --synthetic for a "
            "machinery-validation run."
        )
    feed = AlpacaCryptoDataFeed(credentials.api_key, credentials.secret_key)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    return {
        symbol: feed.get_bars(symbol, timeframe, start, end) for symbol in symbols
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=ENGINES, default="donchian")
    parser.add_argument("--timeframe", default="1Hour")
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_UNIVERSE))
    parser.add_argument("--days", type=int, default=730)
    parser.add_argument("--synthetic", action="store_true",
                        help="deterministic synthetic data; validates the pipeline only")
    parser.add_argument("--synthetic-bars", type=int, default=2_500)
    parser.add_argument("--db", default=DEFAULT_QUANT_DB_PATH)
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()

    symbols = tuple(args.symbols)
    try:
        if args.synthetic:
            bars = _synthetic_bars(symbols, args.synthetic_bars, args.timeframe)
        else:
            bars = _real_bars(symbols, args.timeframe, args.days)
    except Exception as error:
        return _fail(str(error))

    build, grid = _strategy_factory(args.engine, symbols, args.timeframe)
    strategy_id = {
        "donchian": "donchian_trend",
        "squeeze": "volatility_squeeze",
        "vwap": "vwap_reversion",
    }[args.engine]
    if args.synthetic:
        strategy_id += "_synthetic"

    store = QuantStore(args.db)
    try:
        prior = store.total_trials(strategy_id)
        report = run_quant_walk_forward(
            strategy_id=strategy_id,
            strategy_factory=build,
            candidate_params=grid,
            bars_by_symbol=bars,
            sim_config=SimulationConfig(
                risk_per_trade_pct=0.35,
                slippage_bps=10.0,
                spread_bps=5.0,
                fee_bps=25.0,
                annualization_factor=crypto_bars_per_year(args.timeframe),
            ),
            exit_policy=ExitPolicyConfig(
                enable_trailing_stop=True,
                enable_runner=True,
            ),
            walk_config=QuantWalkForwardConfig(),
            criteria=QuantPromotionCriteria(),
            prior_trials=prior,
        )
    except ValueError as error:
        store.close()
        return _fail(str(error))

    label = "SYNTHETIC (machinery validation only)" if args.synthetic else "REAL DATA"
    print(f"\n=== Quant walk-forward: {strategy_id} [{label}] ===")
    print(f"folds={len(report.folds)}  candidates={report.candidate_count}"
          f"  cumulative_trials={report.trial_count}  oos_trades={report.total_oos_trades}")
    print(f"compounded={report.compounded_return_pct:+.2f}%"
          f"  benchmark={report.benchmark_compounded_return_pct:+.2f}%"
          f"  stressed={report.stressed_compounded_return_pct:+.2f}%")
    print("\nGates:")
    for gate in report.gates:
        verdict = "PASS" if gate.passed else "FAIL"
        value = "unmeasured" if gate.value is None else f"{gate.value:.4f}"
        print(f"  [{verdict}] {gate.name:<38} value={value:<12} threshold={gate.threshold:.4f}")

    if not args.no_record:
        run_id = record_research_report(
            store, report, git_commit=_git_commit(), seed=None
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
