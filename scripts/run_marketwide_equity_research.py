#!/usr/bin/env python3
"""Run real-data walk-forward research for the market-wide US-equity desk.

The discovery stage considers every active/tradable Alpaca US equity and uses only
current executability/liquidity fields to choose the research universe. Historical
return never chooses symbols. One common ordered universe is then frozen for all three
strategy families and recorded in immutable manifests with the exact git commit and
runtime configuration.

Exit codes: 0 = every requested strategy passed, 1 = research ran but at least one
strategy failed promotion gates, 2 = the research protocol could not be completed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import AccountMode, bars_per_year, load_credentials
from equity_market_data import AlpacaEquityMarketData, EquityMarketDataConfig
from quant.adaptive_manifest import current_git_commit
from quant.equity_manifest import (
    EQUITY_STRATEGY_IDS,
    EQUITY_STRATEGY_VERSION,
    EquityDeploymentManifest,
    record_research_manifest,
)
from quant.equity_marketwide import (
    MarketWideEquityConfig,
    MarketWideEquitySelector,
    build_equity_scan_strategies,
)
from quant.exits import ExitPolicyConfig
from quant.research.factory import (
    QuantPromotionCriteria,
    QuantWalkForwardConfig,
    record_research_report,
    run_quant_walk_forward,
)
from quant.research.simulator import SimulationConfig
from quant.store import DEFAULT_QUANT_DB_PATH, QuantStore

STRATEGY_ORDER = (
    "opening_range_breakout",
    "vwap_reversion",
    "volatility_squeeze",
)
MAX_DEPLOYABLE_UNIVERSE = 48


def _fail(message: str) -> int:
    print(f"COULD NOT RUN: {message}", file=sys.stderr)
    return 2


def _latest_quote_asof(quotes: dict[str, object]) -> datetime:
    stamps: list[pd.Timestamp] = []
    for quote in quotes.values():
        raw = getattr(quote, "timestamp", None)
        if raw is None:
            continue
        try:
            stamp = pd.Timestamp(raw)
        except (TypeError, ValueError):
            continue
        if stamp.tzinfo is not None:
            stamps.append(stamp.tz_convert("UTC"))
    if not stamps:
        raise ValueError("No timezone-aware equity quote timestamps were returned.")
    return max(stamps).to_pydatetime()


def _aligned_research_universe(
    ordered_symbols: tuple[str, ...],
    bars: dict[str, pd.DataFrame],
    *,
    minimum_common_bars: int,
    minimum_symbols: int,
) -> tuple[tuple[str, ...], dict[str, pd.DataFrame]]:
    """Greedily retain liquid names that preserve the fixed research horizon.

    The input order is current-liquidity order. A symbol may be dropped only for
    insufficient/alignment data; no return, Sharpe, hit rate or strategy output enters
    this decision.
    """
    kept: list[str] = []
    common: pd.DatetimeIndex | None = None
    for symbol in ordered_symbols:
        frame = bars.get(symbol)
        if frame is None or len(frame) < minimum_common_bars:
            continue
        prospective = frame.index if common is None else common.intersection(frame.index)
        prospective = prospective.sort_values()
        if len(prospective) < minimum_common_bars:
            continue
        kept.append(symbol)
        common = prospective
    if common is None or len(kept) < minimum_symbols:
        raise ValueError(
            f"Only {len(kept)} symbols have {minimum_common_bars}+ exactly aligned "
            f"completed bars; at least {minimum_symbols} are required."
        )
    trimmed = {
        symbol: bars[symbol].loc[bars[symbol].index.isin(common)].copy()
        for symbol in kept
    }
    return tuple(kept), trimmed


def _strategy_factory(
    strategy_id: str,
    universe: tuple[str, ...],
    config: MarketWideEquityConfig,
):
    def build(_params):
        entries = build_equity_scan_strategies(universe, config=config)
        matching = [entry for entry in entries if entry.strategy_id == strategy_id]
        if len(matching) != 1:
            raise ValueError(
                f"Research requires one unsharded {strategy_id} instance; got {len(matching)}."
            )
        # OwnershipAwareStrategy is a production routing guard. Research needs the
        # underlying hypothesis so its own held-symbol exit logic is exercised.
        return matching[0].strategy.delegate

    return build


def _research_one(
    store: QuantStore,
    strategy_id: str,
    universe: tuple[str, ...],
    bars: dict[str, pd.DataFrame],
    runtime_config: MarketWideEquityConfig,
    git_commit: str,
):
    walk = QuantWalkForwardConfig(
        train_bars=3_000,
        test_bars=500,
        step_bars=500,
        embargo_bars=2,
        min_train_trades=12,
        max_candidates=4,
        stress_slippage_bps=20.0,
        stress_spread_bps=15.0,
        stress_fee_bps=0.0,
    )
    criteria = QuantPromotionCriteria(
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
    )
    report = run_quant_walk_forward(
        strategy_id=strategy_id,
        strategy_factory=_strategy_factory(strategy_id, universe, runtime_config),
        # This production path validates the versioned structural hypothesis exactly;
        # it does not optimize a grid and then silently deploy different parameters.
        candidate_params=[{}],
        bars_by_symbol=bars,
        sim_config=SimulationConfig(
            risk_per_trade_pct=runtime_config.risk_per_trade_pct,
            max_positions=runtime_config.max_positions,
            max_total_exposure_pct=runtime_config.max_total_exposure_pct,
            slippage_bps=10.0,
            spread_bps=5.0,
            fee_bps=0.0,
            annualization_factor=bars_per_year(runtime_config.timeframe),
        ),
        exit_policy=ExitPolicyConfig(enable_trailing_stop=True, enable_runner=True),
        walk_config=walk,
        criteria=criteria,
        prior_trials=store.total_trials(strategy_id),
    )
    run_id = record_research_report(
        store,
        report,
        git_commit=git_commit,
        seed=None,
    )
    manifest = EquityDeploymentManifest(
        research_run_id=run_id,
        git_commit=git_commit,
        strategy_id=strategy_id,
        strategy_version=EQUITY_STRATEGY_VERSION,
        timeframe=runtime_config.timeframe,
        universe=universe,
        runtime_config=runtime_config.to_dict(),
    )
    record_research_manifest(store, manifest)
    return report, run_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--strategy",
        choices=("all", *STRATEGY_ORDER),
        default="all",
    )
    parser.add_argument("--timeframe", choices=("5Min", "15Min"), default="15Min")
    parser.add_argument("--feed", choices=("iex", "sip"), default="iex")
    parser.add_argument("--universe-size", type=int, default=MAX_DEPLOYABLE_UNIVERSE)
    parser.add_argument("--history-bars", type=int, default=6_500)
    parser.add_argument("--db", default=DEFAULT_QUANT_DB_PATH)
    args = parser.parse_args()

    if not 12 <= args.universe_size <= MAX_DEPLOYABLE_UNIVERSE:
        return _fail(
            f"--universe-size must be between 12 and {MAX_DEPLOYABLE_UNIVERSE}; "
            "the deployable desk is intentionally one strategy shard."
        )
    if args.history_bars < 5_750:
        return _fail("--history-bars must be at least 5750 for five untouched OOS folds.")

    git_commit = current_git_commit(ROOT)
    if git_commit is None:
        return _fail("A full git commit SHA is required before deployable research can be recorded.")
    credentials = load_credentials(account_mode=AccountMode.PAPER)
    if not credentials.is_configured:
        return _fail("Alpaca paper credentials are required for real US-equity research.")

    runtime_config = MarketWideEquityConfig(
        timeframe=args.timeframe,
        max_scan_symbols=args.universe_size,
        shard_size=args.universe_size,
    )
    errors = runtime_config.validate()
    if errors:
        return _fail("; ".join(errors))

    try:
        market_data = AlpacaEquityMarketData(
            credentials.api_key,
            credentials.secret_key,
            EquityMarketDataConfig(feed=args.feed),
        )
        assets = market_data.list_active_us_equities()
        tradable = tuple(asset.symbol for asset in assets if asset.active and asset.tradable)
        quotes = market_data.get_latest_quotes(tradable)
        quote_asof = _latest_quote_asof(quotes)
        daily = market_data.get_recent_bars_many(
            tradable,
            "1Day",
            runtime_config.liquidity_lookback_days,
            now=datetime.now(timezone.utc),
        )
        universe_report = MarketWideEquitySelector(runtime_config).select(
            assets,
            quotes,
            daily,
            now=quote_asof,
        )
        if len(universe_report.selected) < 12:
            return _fail(
                f"Only {len(universe_report.selected)} symbols passed current "
                "executability/liquidity filters."
            )
        raw_history = market_data.get_recent_bars_many(
            universe_report.selected,
            runtime_config.timeframe,
            args.history_bars,
            now=datetime.now(timezone.utc),
            tolerance_seconds=2.0,
        )
        universe, bars = _aligned_research_universe(
            universe_report.selected,
            raw_history,
            minimum_common_bars=5_750,
            minimum_symbols=12,
        )
    except Exception as error:
        return _fail(str(error))

    print(
        f"Research universe: {len(universe)} symbols from {universe_report.discovered} active assets; "
        f"timeframe={runtime_config.timeframe}; quote_asof={quote_asof.isoformat()}"
    )
    print("Locked ordered universe:", ", ".join(universe))

    requested = STRATEGY_ORDER if args.strategy == "all" else (args.strategy,)
    if any(strategy_id not in EQUITY_STRATEGY_IDS for strategy_id in requested):
        return _fail("Unsupported strategy selection.")

    store = QuantStore(args.db)
    all_passed = True
    try:
        for strategy_id in requested:
            try:
                report, run_id = _research_one(
                    store,
                    strategy_id,
                    universe,
                    bars,
                    runtime_config,
                    git_commit,
                )
            except ValueError as error:
                return _fail(f"{strategy_id}: {error}")
            verdict = "PASS" if report.passed else "FAIL"
            print(
                f"\n[{verdict}] {strategy_id} run#{run_id}: folds={len(report.folds)} "
                f"OOS trades={report.total_oos_trades} return={report.compounded_return_pct:+.2f}% "
                f"benchmark={report.benchmark_compounded_return_pct:+.2f}% "
                f"stressed={report.stressed_compounded_return_pct:+.2f}%"
            )
            for gate in report.gates:
                value = "unmeasured" if gate.value is None else f"{gate.value:.4f}"
                print(f"  {'PASS' if gate.passed else 'FAIL'} {gate.name}: {value}")
            all_passed = all_passed and report.passed
    finally:
        store.close()
    return 0 if all_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
