#!/usr/bin/env python3
"""Run the high-velocity strategy in real-time SHADOW or broker PAPER mode.

The runtime is deliberately promotion-aware:

* SHADOW requires the strategy lifecycle to already be SHADOW (earned by a
  passing real-data research run + explicit promotion).
* PAPER requires lifecycle PAPER and uses the production broker engine.
* LIVE is intentionally not exposed by this CLI. Real-money startup remains a
  separate, operator-acknowledged LIVE_CANARY/LIVE action.

Quotes refresh every five seconds in SHADOW and every five seconds while flat /
one second while exposed in PAPER. Completed OHLCV history is cached until a new
5-minute bar is safely complete, avoiding API pressure.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from broker import AlpacaPaperBroker, BrokerError
from config import AccountMode, load_credentials
from crypto_data_feed import AlpacaCryptoDataFeed
from intraday_crypto_engine import STRATEGY_ID, STRATEGY_VERSION, make_intraday_momentum_crypto_engine
from quant.contracts import Market, StrategyState
from quant.execution import ExecutionQualityEngine
from quant.intraday import (
    HIGH_VELOCITY_CRYPTO_UNIVERSE,
    make_high_velocity_engine_config,
    make_high_velocity_risk_config,
    make_high_velocity_strategy_config,
)
from quant.lifecycle import LifecycleManager
from quant.portfolio import PortfolioBook
from quant.realtime_feed import CompletedBarCachingCryptoFeed
from quant.scanner import QuantScanner, ScanStrategy
from quant.store import DEFAULT_QUANT_DB_PATH, QuantStore
from quant.strategies.intraday_momentum import IntradayMomentumStrategy
from state_store import StateStore

DEFAULT_STATE_DB = str(Path("data") / "intraday_trading.db")
REALTIME_QUOTES_KEY = "intraday_realtime_quotes"


def _quote_snapshot(quotes) -> str:
    payload = {}
    for symbol, quote in quotes.items():
        timestamp = getattr(quote, "timestamp", None)
        payload[symbol] = {
            "timestamp": timestamp.isoformat() if timestamp is not None else None,
            "bid": getattr(quote, "bid_price", None),
            "ask": getattr(quote, "ask_price", None),
            "spread_bps": getattr(quote, "spread_bps", None),
        }
    return json.dumps(payload, sort_keys=True)


def _make_market_stack(symbols):
    credentials = load_credentials(account_mode=AccountMode.PAPER)
    if not credentials.is_configured:
        raise RuntimeError(
            "Alpaca paper credentials are required for real-time market data."
        )
    broker = AlpacaPaperBroker(credentials.api_key, credentials.secret_key)
    backend = AlpacaCryptoDataFeed(credentials.api_key, credentials.secret_key)
    return broker, CompletedBarCachingCryptoFeed(backend)


def _shadow_loop(symbols, quant_store: QuantStore, poll_seconds: float) -> int:
    lifecycle = LifecycleManager(quant_store)
    state = lifecycle.state(STRATEGY_ID)
    if state is not StrategyState.SHADOW:
        print(
            f"Refusing SHADOW runtime: {STRATEGY_ID} is {state.value}; "
            "promote a passing real-data research run to SHADOW first.",
            file=sys.stderr,
        )
        return 2

    broker, feed = _make_market_stack(symbols)
    account = broker.verify_account()
    strategy_config = make_high_velocity_strategy_config(symbols=symbols)
    strategy = IntradayMomentumStrategy(strategy_config)
    scanner = QuantScanner(
        store=quant_store,
        lifecycle=lifecycle,
        execution_engine=ExecutionQualityEngine(),
    )
    entry = ScanStrategy(
        strategy_id=STRATEGY_ID,
        strategy=strategy,
        market=Market.CRYPTO,
        timeframe=strategy_config.timeframe,
        version=STRATEGY_VERSION,
        base_risk_pct=make_high_velocity_risk_config().risk_per_trade_pct,
    )
    last_bar = None
    print(
        f"SHADOW {STRATEGY_ID}: {len(symbols)} symbols, 5Min decisions, "
        f"quotes every {poll_seconds:g}s. Ctrl+C stops."
    )
    try:
        while True:
            now = datetime.now(timezone.utc)
            quotes = feed.get_latest_quotes(list(symbols))
            quant_store.set_kv(REALTIME_QUOTES_KEY, _quote_snapshot(quotes))
            bars = {
                symbol: feed.get_recent_bars(
                    symbol,
                    strategy_config.timeframe,
                    strategy.required_bars() + 5,
                    now=now,
                    tolerance_seconds=2.0,
                )
                for symbol in symbols
            }
            if all(len(frame) >= strategy.required_bars() for frame in bars.values()):
                common = None
                for frame in bars.values():
                    common = frame.index if common is None else common.intersection(frame.index)
                if common is not None and len(common):
                    bar = common[-1]
                    if last_bar is None or bar > last_bar:
                        report = scanner.scan(
                            strategies=[entry],
                            bars_by_symbol=bars,
                            equity=account.equity,
                            book=PortfolioBook(equity=account.equity),
                            quotes=quotes,
                            benchmark_symbol="BTC/USD",
                            session_id="intraday_shadow",
                            now=now,
                        )
                        counts = report.shadow_counts.get(STRATEGY_ID, {})
                        print(
                            f"{bar.isoformat()} shadow cycle: "
                            + ", ".join(f"{key}={value}" for key, value in counts.items())
                        )
                        last_bar = bar
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\nSHADOW stopped by operator.")
        return 0


def _paper_loop(
    symbols,
    quant_store: QuantStore,
    state_db: str,
    max_runtime_hours: float | None,
) -> int:
    lifecycle = LifecycleManager(quant_store)
    state = lifecycle.state(STRATEGY_ID)
    if state not in {
        StrategyState.PAPER,
        StrategyState.LIVE_CANARY,
        StrategyState.LIVE,
        StrategyState.REDUCED_RISK,
    }:
        print(
            f"Refusing PAPER runtime: {STRATEGY_ID} is {state.value}; "
            "complete SHADOW evidence and promote to PAPER first.",
            file=sys.stderr,
        )
        return 2

    broker, feed = _make_market_stack(symbols)
    strategy_config = make_high_velocity_strategy_config(symbols=symbols)
    risk_config = make_high_velocity_risk_config()
    engine_config = make_high_velocity_engine_config(account_mode=AccountMode.PAPER)
    if max_runtime_hours is not None:
        engine_config = replace(
            engine_config,
            max_runtime_seconds=max_runtime_hours * 3600.0,
        )
    state_store = StateStore(state_db)
    engine = make_intraday_momentum_crypto_engine(
        broker,
        feed,
        state_store,
        strategy_config,
        risk_config,
        engine_config,
        quant_store=quant_store,
        lifecycle=lifecycle,
    )
    print(
        f"PAPER {STRATEGY_ID}: {len(symbols)} symbols, risk/trade "
        f"{risk_config.risk_per_trade_pct:.2f}%, quote loop 5s flat / 1s exposed."
    )
    engine.start()
    try:
        while engine.is_running:
            status = engine.status
            if status.error:
                print(f"Engine error: {status.error}", file=sys.stderr)
            engine.join(1.0)
    except KeyboardInterrupt:
        print("\nGraceful PAPER stop requested...")
        engine.stop()
        engine.join(10.0)
    status = engine.status
    print(f"PAPER stopped: state={status.state.value}, reason={status.stop_reason}")
    return 1 if status.error else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shadow", "paper"), default="shadow")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--quant-db", default=DEFAULT_QUANT_DB_PATH)
    parser.add_argument("--state-db", default=DEFAULT_STATE_DB)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-runtime-hours", type=float, default=None)
    args = parser.parse_args()

    if args.poll_seconds < 1.0:
        print("poll-seconds must be at least 1 second.", file=sys.stderr)
        return 2
    if args.max_runtime_hours is not None and args.max_runtime_hours <= 0:
        print("max-runtime-hours must be positive.", file=sys.stderr)
        return 2
    symbols = tuple(args.symbols or HIGH_VELOCITY_CRYPTO_UNIVERSE)
    if len(symbols) < 3 or len(symbols) > 12:
        print("Intraday runtime universe must contain 3-12 symbols.", file=sys.stderr)
        return 2

    quant_store = QuantStore(args.quant_db)
    try:
        if args.mode == "shadow":
            return _shadow_loop(symbols, quant_store, args.poll_seconds)
        return _paper_loop(
            symbols,
            quant_store,
            args.state_db,
            args.max_runtime_hours,
        )
    except (BrokerError, RuntimeError, ValueError) as error:
        print(f"Runtime refused/failed: {error}", file=sys.stderr)
        return 2
    finally:
        quant_store.close()


if __name__ == "__main__":
    raise SystemExit(main())
