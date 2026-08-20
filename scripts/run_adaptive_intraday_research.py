#!/usr/bin/env python3
"""Research adaptive intraday v2 on real Alpaca data.

Normal mode selects a currently executable universe (or an explicit 6-12 symbol
universe), evaluates a bounded 18-hypothesis grid and records an immutable research
manifest containing the exact git commit, timeframe, ordered universe and selected
parameters.

``--locked-config`` is the pre-CANARY verification mode.  It does not optimize: it
re-evaluates the exact promoted deployment manifest under current/measured costs.
The requested 2%/day number remains an untouched-OOS diagnostic only.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_credentials
from crypto_config import crypto_bars_per_year
from crypto_data_feed import AlpacaCryptoDataFeed
from intraday_crypto_engine import ADAPTIVE_STRATEGY_ID, ADAPTIVE_STRATEGY_VERSION
from quant.adaptive_manifest import (
    AdaptiveManifest,
    current_git_commit,
    locked_deployment_manifest,
    record_research_manifest,
)
from quant.exits import ExitPolicyConfig, scalp_exit_policy
from quant.intraday import DEFAULT_INTRADAY_OBJECTIVE, make_high_velocity_strategy_config
from quant.liquidity_universe import (
    ADAPTIVE_USD_CANDIDATE_POOL,
    LiquiditySelectionConfig,
    select_liquid_universe,
)
from quant.research.factory import (
    GateResult,
    QuantPromotionCriteria,
    QuantWalkForwardConfig,
    record_research_report,
    run_quant_walk_forward,
)
from quant.research.simulator import SimulationConfig
from quant.store import DEFAULT_QUANT_DB_PATH, QuantStore
from quant.strategies.intraday_adaptive import (
    AdaptiveIntradayConfig,
    AdaptiveIntradayStrategy,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _feed() -> AlpacaCryptoDataFeed:
    credentials = load_credentials()
    if not credentials.is_configured:
        raise RuntimeError("Alpaca credentials are required for real adaptive research.")
    return AlpacaCryptoDataFeed(credentials.api_key, credentials.secret_key)


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _p75(values: list[float]) -> float:
    values = sorted(value for value in values if math.isfinite(value) and value >= 0)
    if not values:
        return 0.0
    index = min(len(values) - 1, max(0, math.ceil(0.75 * len(values)) - 1))
    return values[index]


def _current_spread_p75(feed: AlpacaCryptoDataFeed, symbols: tuple[str, ...]) -> float:
    quotes = feed.get_latest_quotes(list(symbols))
    spreads: list[float] = []
    for symbol in symbols:
        quote = quotes.get(symbol)
        if quote is None:
            continue
        spread = _finite(getattr(quote, "spread_bps", None))
        if spread is None:
            bid = _finite(getattr(quote, "bid_price", None))
            ask = _finite(getattr(quote, "ask_price", None))
            if bid is not None and ask is not None and bid > 0 and ask > bid:
                spread = (ask - bid) / ((ask + bid) / 2.0) * 10_000.0
        if spread is not None:
            spreads.append(spread)
    return _p75(spreads)


def _liquidity_selected_symbols(feed: AlpacaCryptoDataFeed):
    quotes = feed.get_latest_quotes(list(ADAPTIVE_USD_CANDIDATE_POOL))
    symbols, metrics = select_liquid_universe(
        quotes,
        config=LiquiditySelectionConfig(
            max_symbols=12,
            min_symbols=6,
            max_quote_age_seconds=120.0,
            max_spread_bps=40.0,
            min_side_depth_usd=250.0,
        ),
        now=datetime.now(timezone.utc),
    )
    by_symbol = {metric.symbol: metric for metric in metrics}
    spread_floor = _p75(
        [
            float(by_symbol[symbol].spread_bps)
            for symbol in symbols
            if by_symbol[symbol].spread_bps is not None
        ]
    )
    return symbols, metrics, spread_floor


def _real_bars(feed, symbols: tuple[str, ...], timeframe: str, days: int):
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    bars = {symbol: feed.get_bars(symbol, timeframe, start, end) for symbol in symbols}
    empty = [symbol for symbol, frame in bars.items() if frame is None or frame.empty]
    if empty:
        raise RuntimeError(f"No historical bars returned for: {empty}")
    return bars


def _factory(symbols: tuple[str, ...], timeframe: str):
    base = make_high_velocity_strategy_config(symbols=symbols, timeframe=timeframe)

    def build(params):
        return AdaptiveIntradayStrategy(base, AdaptiveIntradayConfig(**params))

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
    r_values = []
    for fold in report.folds:
        positive_fold_profit.append(max(0.0, fold.test_result.total_return_pct))
        for trade in fold.test_result.trades:
            if trade.net_pnl is not None:
                symbol_pnl[trade.symbol] += float(trade.net_pnl)
                traded_symbols.add(trade.symbol)
            r_value = trade.r_multiple
            if r_value is not None and math.isfinite(r_value):
                r_values.append(float(r_value))
    positive_symbol_pnl = {symbol: max(0.0, pnl) for symbol, pnl in symbol_pnl.items()}
    positive_total = sum(positive_symbol_pnl.values())
    fold_profit_total = sum(positive_fold_profit)
    return {
        "test_days": test_days,
        "trades_per_day": trades_per_day,
        "geometric_daily_return_pct": geometric_daily,
        "active_symbols": len(traded_symbols),
        "min_active_symbols": min(6, max(3, math.ceil(universe_size / 2))),
        "symbol_profit_concentration": (
            max(positive_symbol_pnl.values()) / positive_total
            if positive_total > 0 and positive_symbol_pnl
            else 1.0
        ),
        "fold_profit_concentration": (
            max(positive_fold_profit) / fold_profit_total
            if fold_profit_total > 0 and positive_fold_profit
            else 1.0
        ),
        "expectancy_r": sum(r_values) / len(r_values) if r_values else None,
    }


def _add_robustness_gates(report, diagnostics):
    objective = DEFAULT_INTRADAY_OBJECTIVE
    expectancy_r = diagnostics["expectancy_r"]
    extras = [
        GateResult(
            "min_trades_per_day",
            diagnostics["trades_per_day"],
            objective.target_trades_per_day_min,
            diagnostics["trades_per_day"] >= objective.target_trades_per_day_min,
        ),
        GateResult(
            "max_trades_per_day",
            diagnostics["trades_per_day"],
            50.0,
            diagnostics["trades_per_day"] <= 50.0,
        ),
        GateResult(
            "min_oos_expectancy_r",
            expectancy_r,
            0.03,
            expectancy_r is not None and expectancy_r >= 0.03,
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


def _finite_nonnegative(label: str, value: float) -> str | None:
    if not math.isfinite(value) or value < 0:
        return f"{label} must be finite and non-negative."
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--timeframe", choices=("5Min", "15Min"), default=None)
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--locked-config", action="store_true")
    parser.add_argument("--db", default=DEFAULT_QUANT_DB_PATH)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument("--spread-bps", type=float, default=8.0)
    parser.add_argument("--fee-bps", type=float, default=25.0)
    parser.add_argument("--no-record", action="store_true")
    parser.add_argument(
        "--exit-profile",
        choices=("standard", "scalp"),
        default="standard",
        help=(
            "Exit-management hypothesis to measure: 'standard' keeps the "
            "trailing+runner policy every promotion to date was measured "
            "under; 'scalp' measures quick profit capture with "
            "hold-intelligence exits (quant/exits.scalp_exit_policy)."
        ),
    )
    args = parser.parse_args()

    if args.days < 45:
        print("Adaptive research requires at least 45 calendar days.", file=sys.stderr)
        return 2
    if args.exit_profile != "standard" and args.locked_config:
        print(
            "Locked-config verification must reproduce the promoted configuration "
            "exactly; it always uses the standard exit policy. Drop --exit-profile.",
            file=sys.stderr,
        )
        return 2
    for label, value in (
        ("slippage-bps", args.slippage_bps),
        ("spread-bps", args.spread_bps),
        ("fee-bps", args.fee_bps),
    ):
        error = _finite_nonnegative(label, value)
        if error:
            print(error, file=sys.stderr)
            return 2

    git_commit = current_git_commit(REPO_ROOT)
    if git_commit is None:
        print("COULD NOT RUN: exact git commit is required for reproducible research.", file=sys.stderr)
        return 2

    store = QuantStore(args.db)
    try:
        locked = locked_deployment_manifest(store) if args.locked_config else None
        if args.locked_config and locked is None:
            print("COULD NOT RUN: --locked-config requires a promoted deployment manifest.", file=sys.stderr)
            return 2
        if locked is not None:
            try:
                locked.assert_runtime_identity(git_commit=git_commit)
            except ValueError as error:
                print(f"COULD NOT RUN: {error}", file=sys.stderr)
                return 2
            if args.timeframe is not None and args.timeframe != locked.timeframe:
                print("COULD NOT RUN: --timeframe differs from locked manifest.", file=sys.stderr)
                return 2
            if args.symbols is not None and tuple(args.symbols) != locked.universe:
                print("COULD NOT RUN: --symbols differs from locked manifest.", file=sys.stderr)
                return 2
            timeframe = locked.timeframe
            symbols = locked.universe
        else:
            timeframe = args.timeframe or "5Min"
            symbols = tuple(args.symbols) if args.symbols else None

        feed = _feed()
        if locked is None and symbols is None:
            symbols, metrics, observed_spread_p75 = _liquidity_selected_symbols(feed)
            by_symbol = {metric.symbol: metric for metric in metrics}
            print("Universe source: current executable-liquidity selection.")
            for symbol in symbols:
                metric = by_symbol[symbol]
                print(
                    f"  {symbol:<12} spread={metric.spread_bps:.2f}bps "
                    f"age={metric.quote_age_seconds:.1f}s "
                    f"side_depth=${metric.side_depth_usd:.0f} score={metric.score:.3f}"
                )
        else:
            source = "locked deployment manifest" if locked is not None else "explicit --symbols"
            print(f"Universe source: {source}.")
            observed_spread_p75 = _current_spread_p75(feed, symbols)

        if not 6 <= len(symbols) <= 12 or "BTC/USD" not in symbols:
            raise ValueError("Adaptive research universe must contain BTC/USD and 6-12 symbols.")
        bars = _real_bars(feed, symbols, timeframe, args.days)
        build, grid = _factory(symbols, timeframe)
        if locked is not None:
            grid = [dict(locked.params)]
            print("Locked-config verification: optimization disabled; one exact parameter set.")

        effective_spread_bps = max(args.spread_bps, observed_spread_p75)
        report = run_quant_walk_forward(
            strategy_id=ADAPTIVE_STRATEGY_ID,
            strategy_factory=build,
            candidate_params=grid,
            bars_by_symbol=bars,
            sim_config=SimulationConfig(
                risk_per_trade_pct=0.15,
                max_positions=6,
                max_total_exposure_pct=30.0,
                slippage_bps=args.slippage_bps,
                spread_bps=effective_spread_bps,
                fee_bps=args.fee_bps,
                annualization_factor=crypto_bars_per_year(timeframe),
            ),
            exit_policy=(
                scalp_exit_policy()
                if args.exit_profile == "scalp"
                else ExitPolicyConfig(enable_trailing_stop=True, enable_runner=True)
            ),
            walk_config=QuantWalkForwardConfig(
                train_bars=4_032,
                test_bars=1_152,
                step_bars=1_152,
                embargo_bars=36,
                min_train_trades=40,
                max_candidates=32,
                stress_slippage_bps=max(25.0, args.slippage_bps * 1.5),
                stress_spread_bps=max(30.0, effective_spread_bps * 1.5),
                stress_fee_bps=max(50.0, args.fee_bps * 1.5),
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
        report.protocol["universe"] = list(symbols)
        report.protocol["timeframe"] = timeframe
        report.protocol["strategy_version"] = ADAPTIVE_STRATEGY_VERSION
        report.protocol["git_commit"] = git_commit
        report.protocol["locked_config_verification"] = bool(locked)

        diagnostics = _diagnostics(report, len(symbols))
        _add_robustness_gates(report, diagnostics)

        print(f"\n=== Adaptive intraday v2 — {ADAPTIVE_STRATEGY_ID} ===")
        print(f"git_commit={git_commit}")
        print(f"timeframe={timeframe} universe={', '.join(symbols)}")
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
            f"{diagnostics['active_symbols']} symbols; E[R]={diagnostics['expectancy_r']}R; "
            f"symbol concentration={diagnostics['symbol_profit_concentration']:.3f}; "
            f"fold concentration={diagnostics['fold_profit_concentration']:.3f}"
        )
        print(
            f"geometric OOS={diagnostics['geometric_daily_return_pct']:+.3f}%/day; "
            f"2% aspiration="
            f"{'OBSERVED OOS' if diagnostics['geometric_daily_return_pct'] >= DEFAULT_INTRADAY_OBJECTIVE.aspirational_daily_return_pct else 'NOT OBSERVED'}"
        )
        print(
            f"cost model: slippage={args.slippage_bps:.2f}bps, "
            f"spread={effective_spread_bps:.2f}bps, fee={args.fee_bps:.2f}bps"
        )
        print("\nGates:")
        for gate in report.gates:
            value = "unmeasured" if gate.value is None else f"{gate.value:.4f}"
            print(
                f"  [{'PASS' if gate.passed else 'FAIL'}] {gate.name:<40} "
                f"value={value:<12} threshold={gate.threshold:.4f}"
            )
        print(f"\nselected params={report.latest_selected_params}")

        if not args.no_record:
            run_id = record_research_report(
                store,
                report,
                git_commit=git_commit,
                hypothesis_id="adaptive_intraday_v2_locked" if locked else "adaptive_intraday_v2",
            )
            manifest = AdaptiveManifest(
                research_run_id=run_id,
                git_commit=git_commit,
                timeframe=timeframe,
                universe=tuple(symbols),
                params=dict(report.latest_selected_params),
            ).require_valid()
            record_research_manifest(store, manifest)
            print(f"recorded research run #{run_id} + immutable manifest")

        if report.passed:
            print("VERDICT: PASS — lifecycle eligibility only; no live-profit claim.")
            return 0
        print("VERDICT: FAIL — remain at current lifecycle state.")
        return 1
    except ValueError as error:
        print(f"COULD NOT RUN: {error}", file=sys.stderr)
        return 2
    finally:
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
