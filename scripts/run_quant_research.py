#!/usr/bin/env python3
"""Run leakage-controlled Quant Desk walk-forward research.

Exit codes: 0 = all promotion gates PASS (eligible for SHADOW only),
1 = research completed but gates FAIL, 2 = could not run.

``--engine intraday`` is the high-velocity 5-minute hypothesis. Its ~2%/day
objective is reported only after OOS evaluation; it is never used for candidate
selection or promotion, preventing an optimize-until-pretty overfit loop.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

from config import load_credentials
from crypto_config import CryptoStrategyConfig, crypto_bars_per_year
from quant.contracts import Market
from quant.exits import ExitPolicyConfig
from quant.intraday import DEFAULT_INTRADAY_OBJECTIVE, make_high_velocity_strategy_config
from quant.research.factory import (
    QuantPromotionCriteria,
    QuantWalkForwardConfig,
    record_research_report,
    run_quant_walk_forward,
)
from quant.research.simulator import SimulationConfig
from quant.store import DEFAULT_QUANT_DB_PATH, QuantStore

ENGINES = ("donchian", "squeeze", "vwap", "intraday")
DEFAULT_UNIVERSE = ("BTC/USD", "ETH/USD", "SOL/USD")


def _fail(message: str) -> int:
    print(f"\nCOULD NOT RUN: {message}", file=sys.stderr)
    return 2


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parent.parent,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _strategy_factory(engine: str, symbols: tuple[str, ...], timeframe: str):
    if engine == "donchian":
        from quant.strategies.donchian_trend import DonchianTrendConfig, DonchianTrendStrategy

        def build(params):
            return DonchianTrendStrategy(
                DonchianTrendConfig(symbols=symbols, market=Market.CRYPTO, timeframe=timeframe, **params)
            )

        grid = [
            {"breakout_period": breakout, "atr_stop_multiplier": stop}
            for breakout in (15, 20, 30)
            for stop in (2.0, 3.0)
        ]
        return build, grid

    if engine == "squeeze":
        from quant.strategies.volatility_squeeze import VolatilitySqueezeConfig, VolatilitySqueezeStrategy

        def build(params):
            return VolatilitySqueezeStrategy(
                VolatilitySqueezeConfig(symbols=symbols, market=Market.CRYPTO, timeframe=timeframe, **params)
            )

        grid = [
            {"squeeze_percentile_max": squeeze, "atr_stop_multiplier": stop}
            for squeeze in (20.0, 30.0)
            for stop in (2.0, 3.0)
        ]
        return build, grid

    if engine == "vwap":
        from quant.strategies.vwap_reversion import VwapReversionConfig, VwapReversionStrategy

        def build(params):
            return VwapReversionStrategy(
                VwapReversionConfig(symbols=symbols, market=Market.CRYPTO, timeframe=timeframe, **params)
            )

        grid = [
            {"entry_z": entry, "atr_stop_multiplier": stop}
            for entry in (-1.5, -2.0, -2.5)
            for stop in (1.5, 2.0)
        ]
        return build, grid

    from quant.strategies.intraday_momentum import IntradayMomentumConfig, IntradayMomentumStrategy

    base = make_high_velocity_strategy_config(symbols=symbols, timeframe=timeframe)

    def build(params):
        return IntradayMomentumStrategy(base, IntradayMomentumConfig(**params))

    # Deliberately bounded grid: 18 distinct hypotheses, not hundreds of tiny
    # parameter nudges. Stop/trailing values are fixed so selection focuses on
    # signal horizon/quality instead of fitting exits to one market epoch.
    top_k = min(3, len(symbols))
    grid = [
        {
            "fast_lookback": fast,
            "slow_lookback": slow,
            "vol_lookback": 36,
            "volume_lookback": 24,
            "top_k": top_k,
            "exit_rank_buffer": 1,
            "min_score": score,
            "min_volume_ratio": 0.8,
            "stop_atr_multiplier": 1.8,
            "trailing_atr_multiplier": 2.2,
            "benchmark_fast_ma": 24,
            "benchmark_slow_ma": 72,
            "require_benchmark_bull": regime,
        }
        for fast, slow in ((2, 9), (3, 12), (4, 18))
        for score in (0.20, 0.35, 0.50)
        for regime in (False, True)
    ]
    return build, grid


def _synthetic_bars(symbols: tuple[str, ...], periods: int, timeframe: str) -> dict:
    freq = {
        "5Min": "5min",
        "15Min": "15min",
        "1Hour": "1h",
        "4Hour": "4h",
        "1Day": "1D",
    }[timeframe]
    index = pd.date_range("2023-01-01", periods=periods, freq=freq, tz="UTC")
    out = {}
    for offset, symbol in enumerate(symbols):
        rng = np.random.default_rng(11 + offset)
        t = np.arange(periods)
        base = 100.0 * np.exp(0.04 * t / 100.0)
        cycle = 1.0 + 0.06 * np.sin(t / (37.0 + 5 * offset))
        noise = np.exp(np.cumsum(rng.normal(0, 0.008, periods)))
        close = base * cycle * noise
        frame = pd.DataFrame(
            {
                "open": close * (1 + rng.normal(0, 0.001, periods)),
                "high": close * (1 + np.abs(rng.normal(0, 0.004, periods))),
                "low": close * (1 - np.abs(rng.normal(0, 0.004, periods))),
                "close": close,
                "volume": rng.uniform(500, 5_000, periods),
            },
            index=index,
        )
        frame["high"] = frame[["open", "high", "close"]].max(axis=1)
        frame["low"] = frame[["open", "low", "close"]].min(axis=1)
        out[symbol] = frame
    return out


def _real_bars(symbols: tuple[str, ...], timeframe: str, days: int) -> dict:
    from crypto_data_feed import AlpacaCryptoDataFeed

    credentials = load_credentials()
    if not credentials.is_configured:
        raise RuntimeError("No Alpaca credentials configured; use --synthetic only for machinery validation.")
    feed = AlpacaCryptoDataFeed(credentials.api_key, credentials.secret_key)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    return {symbol: feed.get_bars(symbol, timeframe, start, end) for symbol in symbols}


def _research_policy(engine: str):
    if engine != "intraday":
        return (
            0.35,
            QuantWalkForwardConfig(),
            QuantPromotionCriteria(),
            10.0,
            5.0,
            25.0,
        )
    return (
        0.15,
        QuantWalkForwardConfig(
            train_bars=2_016,      # 7 days of 5-minute bars
            test_bars=576,         # 2 OOS days
            step_bars=576,
            embargo_bars=12,       # one hour separation
            min_train_trades=20,
            max_candidates=32,
            stress_slippage_bps=20.0,
            stress_spread_bps=20.0,
            stress_fee_bps=25.0,
        ),
        QuantPromotionCriteria(
            min_folds=5,
            min_oos_trades=100,
            min_positive_fold_ratio=0.60,
            min_compounded_return_pct=0.0,
            min_benchmark_excess_return_pct=0.0,
            min_aggregate_sharpe=0.75,
            min_profit_factor=1.15,
            max_fold_drawdown_pct=10.0,
            min_stressed_return_pct=0.0,
            min_deflated_sharpe_probability=0.75,
            max_ruin_probability=0.02,
        ),
        8.0,
        10.0,
        25.0,
    )


def _intraday_oos_diagnostics(report):
    test_days = 0.0
    for fold in report.folds:
        test_days += max(
            (fold.test_end - fold.test_start).total_seconds() / 86_400.0,
            1.0 / 288.0,
        )
    trades_per_day = report.total_oos_trades / test_days if test_days > 0 else 0.0
    growth = max(0.0, 1.0 + report.compounded_return_pct / 100.0)
    daily = (growth ** (1.0 / test_days) - 1.0) * 100.0 if growth > 0 and test_days > 0 else -100.0
    return test_days, trades_per_day, daily


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=ENGINES, default="donchian")
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--symbols", nargs="+", default=list(DEFAULT_UNIVERSE))
    parser.add_argument("--days", type=int, default=None)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--synthetic-bars", type=int, default=None)
    parser.add_argument("--db", default=DEFAULT_QUANT_DB_PATH)
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()

    timeframe = args.timeframe or ("5Min" if args.engine == "intraday" else "1Hour")
    days = args.days or (90 if args.engine == "intraday" else 730)
    synthetic_bars = args.synthetic_bars or (10_000 if args.engine == "intraday" else 2_500)
    symbols = tuple(args.symbols)
    if args.engine == "intraday" and len(symbols) < 3:
        return _fail("Intraday cross-sectional research requires at least three symbols.")

    try:
        bars = (
            _synthetic_bars(symbols, synthetic_bars, timeframe)
            if args.synthetic
            else _real_bars(symbols, timeframe, days)
        )
        build, grid = _strategy_factory(args.engine, symbols, timeframe)
    except Exception as error:
        return _fail(str(error))

    strategy_id = {
        "donchian": "donchian_trend",
        "squeeze": "volatility_squeeze",
        "vwap": "vwap_reversion",
        "intraday": "crypto_intraday_momentum",
    }[args.engine]
    if args.synthetic:
        strategy_id += "_synthetic"

    risk_pct, walk, criteria, slippage, spread, fee = _research_policy(args.engine)
    store = QuantStore(args.db)
    try:
        report = run_quant_walk_forward(
            strategy_id=strategy_id,
            strategy_factory=build,
            candidate_params=grid,
            bars_by_symbol=bars,
            sim_config=SimulationConfig(
                risk_per_trade_pct=risk_pct,
                slippage_bps=slippage,
                spread_bps=spread,
                fee_bps=fee,
                annualization_factor=crypto_bars_per_year(timeframe),
            ),
            exit_policy=ExitPolicyConfig(enable_trailing_stop=True, enable_runner=True),
            walk_config=walk,
            criteria=criteria,
            prior_trials=store.total_trials(strategy_id),
        )
    except ValueError as error:
        store.close()
        return _fail(str(error))

    label = "SYNTHETIC (machinery only)" if args.synthetic else "REAL DATA"
    print(f"\n=== Quant walk-forward: {strategy_id} [{label}] ===")
    print(
        f"folds={len(report.folds)} candidates={report.candidate_count} "
        f"cumulative_trials={report.trial_count} oos_trades={report.total_oos_trades}"
    )
    print(
        f"compounded={report.compounded_return_pct:+.2f}% "
        f"benchmark={report.benchmark_compounded_return_pct:+.2f}% "
        f"stressed={report.stressed_compounded_return_pct:+.2f}%"
    )
    if args.engine == "intraday":
        test_days, trades_per_day, daily = _intraday_oos_diagnostics(report)
        objective = DEFAULT_INTRADAY_OBJECTIVE
        print(
            f"intraday OOS: {test_days:.1f} test-days, {trades_per_day:.2f} trades/day, "
            f"geometric {daily:+.3f}%/day"
        )
        print(
            f"activity target {objective.target_trades_per_day_min:.0f}-"
            f"{objective.target_trades_per_day_max:.0f}/day: "
            f"{'MET' if objective.target_trades_per_day_min <= trades_per_day <= objective.target_trades_per_day_max else 'NOT MET'}"
        )
        print(
            f"aspirational {objective.aspirational_daily_return_pct:.1f}%/day: "
            f"{'OBSERVED OOS' if daily >= objective.aspirational_daily_return_pct else 'NOT OBSERVED'} "
            "(diagnostic only; never a selection gate)"
        )

    print("\nGates:")
    for gate in report.gates:
        verdict = "PASS" if gate.passed else "FAIL"
        value = "unmeasured" if gate.value is None else f"{gate.value:.4f}"
        print(f"  [{verdict}] {gate.name:<38} value={value:<12} threshold={gate.threshold:.4f}")

    if not args.no_record:
        run_id = record_research_report(store, report, git_commit=_git_commit(), seed=None)
        print(f"\nRecorded research run #{run_id}; cumulative trials={store.total_trials(strategy_id)}")
    store.close()

    if report.passed:
        print("\nVERDICT: PASS — eligible for SHADOW only; not a live unlock or profit claim.")
        return 0
    print("\nVERDICT: FAIL — do not promote. Gates above are authoritative.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
