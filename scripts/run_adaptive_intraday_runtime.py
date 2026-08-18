#!/usr/bin/env python3
"""Run adaptive intraday v2 in SHADOW, PAPER, or guarded LIVE mode.

Quotes are fetched on every loop and persisted as non-blocking telemetry.  Completed
5-minute history is cached until the next safely completed bar.  LIVE remains behind
all existing broker, lifecycle, operator-ack and canary controls.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broker import AlpacaLiveBroker, AlpacaPaperBroker, BrokerError
from config import AccountMode, load_credentials
from crypto_data_feed import AlpacaCryptoDataFeed
from intraday_crypto_engine import (
    ADAPTIVE_STRATEGY_ID,
    ADAPTIVE_STRATEGY_VERSION,
    make_intraday_adaptive_crypto_engine,
)
from quant.contracts import Market, StrategyState
from quant.evidence_scanner import EvidenceQuantScanner
from quant.execution import ExecutionQualityEngine
from quant.intraday import (
    HIGH_VELOCITY_CRYPTO_UNIVERSE,
    make_high_velocity_engine_config,
    make_high_velocity_risk_config,
    make_high_velocity_strategy_config,
)
from quant.intraday_lifecycle import intraday_promotion_requirements
from quant.lifecycle import LifecycleManager
from quant.portfolio import PortfolioBook
from quant.realtime_feed import CompletedBarCachingCryptoFeed
from quant.scanner import ScanStrategy
from quant.store import DEFAULT_QUANT_DB_PATH, QuantStore
from quant.strategies.intraday_adaptive import AdaptiveIntradayStrategy
from state_store import StateStore

DEFAULT_STATE_DB = str(Path("data") / "adaptive_intraday_trading.db")
DEFAULT_LIVE_STATE_DB = str(Path("data") / "adaptive_intraday_live_trading.db")
REALTIME_QUOTES_KEY = "adaptive_intraday_realtime_quotes"
LIVE_ACK = "I_ACCEPT_LIVE_RISK"


def _lifecycle(store: QuantStore) -> LifecycleManager:
    return LifecycleManager(store, requirements=intraday_promotion_requirements())


def _quote_snapshot(quotes) -> str:
    payload = {}
    now = datetime.now(timezone.utc)
    for symbol, quote in quotes.items():
        timestamp = getattr(quote, "timestamp", None)
        bid_size = getattr(quote, "bid_size", None)
        ask_size = getattr(quote, "ask_size", None)
        imbalance = None
        try:
            bid = float(bid_size) if bid_size is not None else math.nan
            ask = float(ask_size) if ask_size is not None else math.nan
            total = bid + ask
            if math.isfinite(total) and total > 0:
                imbalance = (bid - ask) / total
        except (TypeError, ValueError):
            pass
        age = None
        if timestamp is not None:
            try:
                ts = timestamp if timestamp.tzinfo is not None else timestamp.replace(tzinfo=timezone.utc)
                age = max(0.0, (now - ts.astimezone(timezone.utc)).total_seconds())
            except (AttributeError, TypeError, ValueError):
                pass
        payload[symbol] = {
            "timestamp": timestamp.isoformat() if timestamp is not None else None,
            "quote_age_seconds": age,
            "bid": getattr(quote, "bid_price", None),
            "ask": getattr(quote, "ask_price", None),
            "bid_size": bid_size,
            "ask_size": ask_size,
            "depth_imbalance": imbalance,
            "spread_bps": getattr(quote, "spread_bps", None),
        }
    return json.dumps(payload, sort_keys=True)


def _make_market_stack(mode: AccountMode, risk_config, quant_store: QuantStore):
    credentials = load_credentials(account_mode=mode)
    if not credentials.is_configured:
        raise RuntimeError(f"Separate Alpaca {mode.value} credentials are required.")
    if mode is AccountMode.LIVE:
        broker = AlpacaLiveBroker(
            credentials.api_key,
            credentials.secret_key,
            max_live_order_notional=risk_config.max_live_order_notional,
        )
    else:
        broker = AlpacaPaperBroker(credentials.api_key, credentials.secret_key)
    backend = AlpacaCryptoDataFeed(credentials.api_key, credentials.secret_key)

    def observe(quotes):
        quant_store.set_kv(REALTIME_QUOTES_KEY, _quote_snapshot(quotes))

    return broker, CompletedBarCachingCryptoFeed(backend, quote_observer=observe)


def _shadow_loop(symbols, timeframe, quant_store: QuantStore, poll_seconds: float) -> int:
    lifecycle = _lifecycle(quant_store)
    state = lifecycle.state(ADAPTIVE_STRATEGY_ID)
    if state is not StrategyState.SHADOW:
        print(
            f"Refusing SHADOW runtime: {ADAPTIVE_STRATEGY_ID} is {state.value}; "
            "a passing real-data research run must promote it first.",
            file=sys.stderr,
        )
        return 2

    risk_config = make_high_velocity_risk_config()
    broker, feed = _make_market_stack(AccountMode.PAPER, risk_config, quant_store)
    account = broker.verify_account()
    strategy_config = make_high_velocity_strategy_config(symbols=symbols, timeframe=timeframe)
    strategy = AdaptiveIntradayStrategy(strategy_config)
    scanner = EvidenceQuantScanner(
        store=quant_store,
        lifecycle=lifecycle,
        execution_engine=ExecutionQualityEngine(),
    )
    entry = ScanStrategy(
        strategy_id=ADAPTIVE_STRATEGY_ID,
        strategy=strategy,
        market=Market.CRYPTO,
        timeframe=timeframe,
        version=ADAPTIVE_STRATEGY_VERSION,
        base_risk_pct=risk_config.risk_per_trade_pct,
    )
    last_bar = None
    print(
        f"SHADOW {ADAPTIVE_STRATEGY_ID}: {len(symbols)} symbols, {timeframe} decisions, "
        f"fresh quotes every {poll_seconds:g}s. Ctrl+C stops."
    )
    try:
        while True:
            now = datetime.now(timezone.utc)
            quotes = feed.get_latest_quotes(list(symbols))
            bars = {
                symbol: feed.get_recent_bars(
                    symbol,
                    timeframe,
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
                            session_id="adaptive_intraday_shadow",
                            now=now,
                        )
                        counts = report.shadow_counts.get(ADAPTIVE_STRATEGY_ID, {})
                        print(
                            f"{bar.isoformat()} shadow: "
                            + ", ".join(f"{key}={value}" for key, value in counts.items())
                        )
                        last_bar = bar
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\nAdaptive SHADOW stopped by operator.")
        return 0


def _broker_loop(
    *,
    symbols,
    timeframe,
    quant_store: QuantStore,
    state_db: str,
    account_mode: AccountMode,
    max_runtime_hours: float | None,
) -> int:
    lifecycle = _lifecycle(quant_store)
    state = lifecycle.state(ADAPTIVE_STRATEGY_ID)
    if account_mode is AccountMode.PAPER:
        allowed = {
            StrategyState.PAPER,
            StrategyState.LIVE_CANARY,
            StrategyState.LIVE,
            StrategyState.REDUCED_RISK,
        }
        required = "PAPER"
    else:
        allowed = {
            StrategyState.LIVE_CANARY,
            StrategyState.LIVE,
            StrategyState.REDUCED_RISK,
        }
        required = "LIVE_CANARY/LIVE"
    if state not in allowed:
        print(
            f"Refusing {account_mode.value.upper()}: {ADAPTIVE_STRATEGY_ID} is "
            f"{state.value}; {required} is required.",
            file=sys.stderr,
        )
        return 2

    risk_config = make_high_velocity_risk_config()
    broker, feed = _make_market_stack(account_mode, risk_config, quant_store)
    strategy_config = make_high_velocity_strategy_config(symbols=symbols, timeframe=timeframe)
    engine_config = make_high_velocity_engine_config(account_mode=account_mode)
    if max_runtime_hours is not None:
        engine_config = replace(engine_config, max_runtime_seconds=max_runtime_hours * 3600.0)
    state_store = StateStore(state_db)
    engine = make_intraday_adaptive_crypto_engine(
        broker,
        feed,
        state_store,
        strategy_config,
        risk_config,
        engine_config,
        quant_store=quant_store,
        lifecycle=lifecycle,
    )
    label = account_mode.value.upper()
    effective_risk = risk_config.risk_per_trade_pct * lifecycle.risk_multiplier(ADAPTIVE_STRATEGY_ID)
    print(
        f"{label} {ADAPTIVE_STRATEGY_ID}: {len(symbols)} symbols, {timeframe}, "
        f"base risk={risk_config.risk_per_trade_pct:.3f}%/trade, lifecycle risk<="
        f"{effective_risk:.3f}%/trade, quote/protection loop 5s flat / 1s exposed."
    )
    engine.start()
    try:
        while engine.is_running:
            status = engine.status
            if status.error:
                print(f"Engine error: {status.error}", file=sys.stderr)
            engine.join(1.0)
    except KeyboardInterrupt:
        print(f"\nGraceful {label} stop requested...")
        engine.stop()
        engine.join(10.0)
    status = engine.status
    print(f"{label} stopped: state={status.state.value}, reason={status.stop_reason}")
    return 1 if status.error else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("shadow", "paper", "live"), default="shadow")
    parser.add_argument("--timeframe", choices=("5Min", "15Min"), default="5Min")
    parser.add_argument("--symbols", nargs="+", default=None)
    parser.add_argument("--quant-db", default=DEFAULT_QUANT_DB_PATH)
    parser.add_argument("--state-db", default=DEFAULT_STATE_DB)
    parser.add_argument("--live-state-db", default=DEFAULT_LIVE_STATE_DB)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--max-runtime-hours", type=float, default=None)
    parser.add_argument("--live-ack", default="")
    args = parser.parse_args()

    if args.poll_seconds < 1.0:
        print("poll-seconds must be at least 1 second.", file=sys.stderr)
        return 2
    if args.max_runtime_hours is not None and args.max_runtime_hours <= 0:
        print("max-runtime-hours must be positive.", file=sys.stderr)
        return 2
    if args.mode == "live" and args.live_ack != LIVE_ACK:
        print(f"LIVE requires --live-ack {LIVE_ACK}", file=sys.stderr)
        return 2
    symbols = tuple(args.symbols or HIGH_VELOCITY_CRYPTO_UNIVERSE)
    if len(symbols) < 6 or len(symbols) > 12:
        print("Adaptive runtime universe must contain 6-12 symbols.", file=sys.stderr)
        return 2

    quant_store = QuantStore(args.quant_db)
    try:
        if args.mode == "shadow":
            return _shadow_loop(symbols, args.timeframe, quant_store, args.poll_seconds)
        return _broker_loop(
            symbols=symbols,
            timeframe=args.timeframe,
            quant_store=quant_store,
            state_db=args.state_db if args.mode == "paper" else args.live_state_db,
            account_mode=AccountMode.PAPER if args.mode == "paper" else AccountMode.LIVE,
            max_runtime_hours=args.max_runtime_hours,
        )
    except (BrokerError, RuntimeError, ValueError) as error:
        print(f"Runtime refused/failed: {error}", file=sys.stderr)
        return 2
    finally:
        quant_store.close()


if __name__ == "__main__":
    raise SystemExit(main())
