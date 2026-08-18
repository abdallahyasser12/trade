#!/usr/bin/env python3
"""Research the adaptive v2 intraday strategy on real Alpaca data.

The runner is intentionally harder to pass than the generic research command.  It
adds breadth/concentration/activity gates on top of chronological walk-forward,
stressed execution costs, DSR multiple-testing correction and Monte Carlo ruin.

The 2%/day objective is diagnostic only.  A run may pass because it demonstrates
robust positive expectancy even when 2%/day is not observed; conversely, a run that
prints 2%/day but fails robustness cannot promote.
"""

from __future__ import annotations

import argparse
import math
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_credentials
from crypto_config import crypto_bars_per_year
from crypto_data_feed import AlpacaCryptoDataFeed
from intraday_crypto_engine import ADAPTIVE_STRATEGY_ID
from quant.exits import ExitPolicyConfig
from quant.intraday import DEFAULT_INTRADAY_OBJECTIVE, HIGH_VELOCITY_CRYPTO_UNIVERSE, make_high_velocity_strategy_config
from quant.research.factory import (
    GateResult,
    QuantPromotionCriteria,
    QuantWalkForwardConfig,
    record_research_report,
    run_quant_walk_forward,
)
from quant.research.simulator import SimulationConfig
from quant.store import DEFAULT_QUANT_DB_PATH, QuantStore
from quant.strategies.intraday_adaptive import AdaptiveIntradayConfig, AdaptiveIntradayStrategy


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


def _real_bars(symbols: tuple[str, ...], timeframe: str, days: int):
    credentials = load_credentials()
    if not credentials.is_configured:
        raise RuntimeError("Alpaca credentials are required for real adaptive research.")
    feed = AlpacaCryptoDataFeed(credentials.api_key, credentials.secret_key)
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    bars = {
        symbol: feed.get_bars(symbol, timeframe, start, end)
        for symbol in symbols
    }
    empty = [symbol for symbol, frame in bars.items() if frame is None or frame.empty]
    if empty:
        raise RuntimeError(f"No historical bars returned for: {empty}")
    return bars


def _factory(symbols: tuple[str, ...], timeframe: str):
    base = make_high_velocity_strategy_config(symbols=symbols, timeframe=timeframe)

    def build(params):
        return AdaptiveIntradayStrategy(base, AdaptiveIntradayConfig(**params))

    # Structural grid only.  Eighteen hypotheses are enough to test robustness
    # without turning the optimizer into a parameter-mining engine.
    structures = (
        (2, 9, 30, 16, 48),
        (3, 12, 36, 18, 54),
        (4, 15, 45, 24, 72),
    )
    grid = [
        {
            "fast_lookback": fast,
            "medium_lookback": medium,
            "slow_lookback": slow,
            "vol_lookback": max(48, slow),
            "volume_lookback": 24,
            "breakout_lookback": 20,
            "reversion_lookback": 24,
            "trend_fast_span": trend_fast,
            "trend_slow_span": trend_slow,
            "atr_baseline_lookback": 72,
            "top_k": min(4, len(symbols)),
            "exit_rank_buffer": 2,
            "min_entry_score": score,
            "exit_score": 0.08,
            "min_volume_ratio": 0.65,
            "max_atr_ratio": 2.75,
            "stop_atr_multiplier": stop,
            "trailing_atr_multiplier": 2.05,
            "stress_block_probability": 0.70,
            "min_trend_probability": 0.42,
        }
        for fast, medium, slow, trend_fast, trend_slow in structures
        for score in (0.20, 0.26, 0.32)
        for stop in (1.50, 1.80)
    ]
    return build, grid


def _diagnostics(report, universe_size: int):
    test_days = sum(
        max((fold.test_end - fold.test_start).total_seconds() / 86_400.0, 1.0 / 288.0)
        for fold in report.folds
    )
    trades_per_day = report.total_oos_trades / test_days if test_days > 0 else 0.0
    growth = max(0.0, 1.0 + report.compounded_return_pct / 100.0)
    geometric_daily = (
        (growth ** (1.0 / test_days) - 1.0) * 100.0
        if growth > 0 and test_days > 0
        else -100.0
    )

    symbol_pnl = defaultdict(float)
    traded_symbols = set()
    positive_fold_profit = []
    for fold in report.folds:
        positive_fold_profit.append(max(0.0, fold.test_result.total_return_pct))
        for trade in fold.test_result.trades:
            if trade.net_pnl is None:
                continue
            symbol_pnl[trade.symbol] += float(trade.net_pnl)
            traded_symbols.add(trade.symbol)

    positive_symbol_pnl = {symbol: max(0.0, pnl) for symbol, pnl in symbol_pnl.items()}
    positive_total = sum(positive_symbol_pnl.values())
    symbol_concentration = (
        max(positive_symbol_pnl.values()) / positive_total
        if positive_total > 0 and positive_symbol_pnl
        else 1.0
    )
    fold_profit_total = sum(positive_fold_profit)
    fold_concentration = (
        max(positive_fold_profit) / fold_profit_total
        if fold_profit_total > 0 and positive_fold_profit
        else 1.0
    )
    min_active_symbols = min(6, max(3, math.ceil(universe_size / 2)))
    return {
        "test_days": test_days,
        "trades_per_day": trades_per_day,
        "geometric_daily_return_pct": geometric_daily,
        "active_symbols": len(traded_symbols),
        "min_active_symbols": min_active_symbols,
        "symbol_profit_concentration": symbol_concentration,
        "fold_profit_concentration": fold_concentration,
    }


def _add_robustness_gates(report, diagnostics):
    extras = [
        GateResult(
            "min_trades_per_day",
            diagnostics["trades_per_day"],
            4.0,
            diagnostics["trades_per_day"] >= 4.0,
        ),
        GateResult(
            "min_active_symbols",
            float(diagnostics["active_symbols"]),
            float(diagnostics["min_active_symbols"]),
            diagnostics["active_symbols"] >= diagnostics["min_active_symbols"],
        ),
        GateResult(
            "max_symbol_profit_concentration",
            diagnostics["symbol_profit_concentration"],
            0.55,
            diagnostics["symbol_profit_concentration"] <= 0.55,
        ),
        GateResult(
            "max_fold_profit_concentration",
            diagnostics["fold_profit_concentration"],
            0.45,
            diagnostics["fold_profit_concentration"] <= 0.45,
        ),
    ]
    report.gates.extend(extras)
    report.protocol["adaptive_diagnostics"] = dict(diagnostics)
    report.passed = report.passed and all(gate.passed for gate in extras)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--timeframe", choices=("5Min", "15Min"), default="5Min")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--db", default=DEFAULT_QUANT_DB_PATH)
    parser.add_argument("--no-record", action="store_true")
    args = parser.parse_args()

    if args.days < 45:
        print("Adaptive research requires at least 45 calendar days.", file=sys.stderr)
        return 2
    symbols = tuple(args.symbols or HIGH_VELOCITY_CRYPTO_UNIVERSE)
    if len(symbols) < 6:
        print("Adaptive research requires at least six symbols for breadth evidence.", file=sys.stderr)
        return 2

    try:
        bars = _real_bars(symbols, args.timeframe, args.days)
        build, grid = _factory(symbols, args.timeframe)
    except Exception as error:
        print(f"COULD NOT RUN: {error}", file=sys.stderr)
        return 2

    store = QuantStore(args.db)
    try:
        report = run_quant_walk_forward(
            strategy_id=ADAPTIVE_STRATEGY_ID,
            strategy_factory=build,
            candidate_params=grid,
            bars_by_symbol=bars,
            sim_config=SimulationConfig(
                risk_per_trade_pct=0.15,
                max_positions=6,
                max_total_exposure_pct=30.0,
                slippage_bps=10.0,
                spread_bps=8.0,
                fee_bps=25.0,
                annualization_factor=crypto_bars_per_year(args.timeframe),
            ),
            exit_policy=ExitPolicyConfig(enable_trailing_stop=True, enable_runner=True),
            walk_config=QuantWalkForwardConfig(
                train_bars=4_032,
                test_bars=1_152,
                step_bars=1_152,
                embargo_bars=36,
                min_train_trades=40,
                max_candidates=32,
                stress_slippage_bps=25.0,
                stress_spread_bps=25.0,
                stress_fee_bps=50.0,
            ),
            criteria=QuantPromotionCriteria(
                min_folds=8,
                min_oos_trades=300,
                min_positive_fold_ratio=0.65,
                min_compounded_return_pct=0.0,
                min_benchmark_excess_return_pct=0.0,
                min_aggregate_sharpe=0.85,
                min_profit_factor=1.18,
                max_fold_drawdown_pct=8.0,
                min_stressed_return_pct=0.0,
                min_deflated_sharpe_probability=0.80,
                max_ruin_probability=0.01,
            ),
            prior_trials=store.total_trials(ADAPTIVE_STRATEGY_ID),
        )
        diagnostics = _diagnostics(report, len(symbols))
        _add_robustness_gates(report, diagnostics)

        print(f"\n=== Adaptive intraday v2 — {ADAPTIVE_STRATEGY_ID} ===")
        print(
            f"folds={len(report.folds)} candidates={report.candidate_count} "
            f"trials={report.trial_count} OOS_trades={report.total_oos_trades}"
        )
        print(
            f"OOS={report.compounded_return_pct:+.2f}% benchmark="
            f"{report.benchmark_compounded_return_pct:+.2f}% stressed="
            f"{report.stressed_compounded_return_pct:+.2f}% PF={report.profit_factor}"
        )
        print(
            f"activity={diagnostics['trades_per_day']:.2f}/day across "
            f"{diagnostics['active_symbols']} symbols; symbol concentration="
            f"{diagnostics['symbol_profit_concentration']:.3f}; fold concentration="
            f"{diagnostics['fold_profit_concentration']:.3f}"
        )
        print(
            f"geometric OOS={diagnostics['geometric_daily_return_pct']:+.3f}%/day; "
            f"2% aspiration="
            f"{'OBSERVED OOS' if diagnostics['geometric_daily_return_pct'] >= DEFAULT_INTRADAY_OBJECTIVE.aspirational_daily_return_pct else 'NOT OBSERVED'}"
        )
        print("\nGates:")
        for gate in report.gates:
            value = "unmeasured" if gate.value is None else f"{gate.value:.4f}"
            print(
                f"  [{'PASS' if gate.passed else 'FAIL'}] {gate.name:<40} "
                f"value={value:<12} threshold={gate.threshold:.4f}"
            )
        print(f"\nselected latest params={report.latest_selected_params}")

        if not args.no_record:
            run_id = record_research_report(
                store,
                report,
                git_commit=_git_commit(),
                hypothesis_id="adaptive_intraday_v2",
            )
            print(f"recorded research run #{run_id}")

        if report.passed:
            print("VERDICT: PASS — SHADOW eligibility only; no live-profit claim.")
            return 0
        print("VERDICT: FAIL — remain RESEARCH.")
        return 1
    except ValueError as error:
        print(f"COULD NOT RUN: {error}", file=sys.stderr)
        return 2
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
