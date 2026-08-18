#!/usr/bin/env python3
"""Run high-velocity intraday strategy in SHADOW, PAPER, or guarded LIVE mode.

LIVE is available only through this Quant-controlled runner. The legacy Streamlit
crypto LIVE config is rejected before construction. LIVE additionally requires:
separate live credentials, ENABLE_LIVE_TRADING=true, lifecycle state
LIVE_CANARY/LIVE/REDUCED_RISK, historical operator-acknowledged promotion, and the
explicit CLI phrase ``--live-ack I_ACCEPT_LIVE_RISK``.

Quotes refresh every five seconds while flat and every one second while broker
capital is exposed. Completed OHLCV history is cached until a new 5-minute bar is
safely complete, so responsiveness does not multiply historical-data requests.
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

from broker import AlpacaLiveBroker, AlpacaPaperBroker, BrokerError
from config import AccountMode, load_credentials
from crypto_data_feed import AlpacaCryptoDataFeed
from intraday_crypto_engine import (
    STRATEGY_ID,
    STRATEGY_VERSION,
    make_intraday_momentum_crypto_engine,
)
from quant.contracts import Market, StrategyState
from quant.execution import ExecutionQualityEngine
from quant.intraday import (
    HIGH_VELOCITY_CRYPTO_UNIVERSE,
    make_high_velocity_engine_config,
    make_high_velocity_risk_config,
    make_high_velocity_strategy_config,
)
from quant.lifecycle import LifecycleManager, PromotionRequirements
from quant.portfolio import PortfolioBook
from quant.realtime_feed import CompletedBarCachingCryptoFeed
from quant.scanner import QuantScanner, ScanStrategy
from quant.store import DEFAULT_QUANT_DB_PATH, QuantStore
from quant.strategies.intraday_momentum import IntradayMomentumStrategy
from state_store import StateStore

DEFAULT_STATE_DB = str(Path("data") / "intraday_trading.db")
DEFAULT_LIVE_STATE_DB = str(Path("data") / "intraday_live_trading.db")
REALTIME_QUOTES_KEY = "intraday_realtime_quotes"
LIVE_ACK = "I_ACCEPT_LIVE_RISK"


def _intraday_lifecycle(store: QuantStore) -> LifecycleManager:
    return LifecycleManager(
        store,
        requirements=PromotionRequirements(
            min_research_trades=100,
            min_shadow_trades=100,
            max_shadow_drawdown_r=6.0,
            min_paper_trades=200,
            min_measured_execution_samples=50,
            min_canary_trades=100,
            max_open_operational_defects=0,
        ),
    )


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


def _make_market_stack(mode: AccountMode, risk_config):
    credentials = load_credentials(account_mode=mode)
    if not credentials.is_configured:
        raise RuntimeError(
            f"Separate Alpaca {mode.value} credentials are required for this runtime."
        )
    if mode is AccountMode.LIVE:
        broker = AlpacaLiveBroker(
            credentials.api_key,
            credentials.secret_key,
            max_live_order_notional=risk_config.max_live_order_notional,
        )
    else:
        broker = AlpacaPaperBroker(credentials.api_key, credentials.secret_key)
    backend = AlpacaCryptoDataFeed(credentials.api_key, credentials.secret_key)
    return broker, CompletedBarCachingCryptoFeed(backend)


def _shadow_loop(symbols, quant_store: QuantStore, poll_seconds: float) -> int:
    lifecycle = _intraday_lifecycle(quant_store)
    state = lifecycle.state(STRATEGY_ID)
    if state is not StrategyState.SHADOW:
        print(
            f"Refusing SHADOW runtime: {STRATEGY_ID} is {state.value}; "
            "promote a passing real-data research run to SHADOW first.",
            file=sys.stderr,
        )
        return 2

    risk_config = make_high_velocity_risk_config()
    broker, feed = _make_market_stack(AccountMode.PAPER, risk_config)
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
        base_risk_pct=risk_config.risk_per_trade_pct,
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
                            + ", ".join(
                                f"{key}={value}" for key, value in counts.items()
                            )
                        )
                        last_bar = bar
            time.sleep(poll_seconds)
    except KeyboardInterrupt:
        print("\nSHADOW stopped by operator.")
        return 0


def _broker_loop(
    *,
    symbols,
    quant_store: QuantStore,
    state_db: str,
    account_mode: AccountMode,
    max_runtime_hours: float | None,
) -> int:
    lifecycle = _intraday_lifecycle(quant_store)
    state = lifecycle.state(STRATEGY_ID)
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
            f"Refusing {account_mode.value.upper()} runtime: {STRATEGY_ID} is "
            f"{state.value}; lifecycle state {required} is required.",
            file=sys.stderr,
        )
        return 2

    risk_config = make_high_velocity_risk_config()
    broker, feed = _make_market_stack(account_mode, risk_config)
    strategy_config = make_high_velocity_strategy_config(symbols=symbols)
    engine_config = make_high_velocity_engine_config(account_mode=account_mode)
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
    label = account_mode.value.upper()
    multiplier = lifecycle.risk_multiplier(STRATEGY_ID)
    effective_risk = risk_config.risk_per_trade_pct * multiplier
    print(
        f"{label} {STRATEGY_ID}: {len(symbols)} symbols, configured risk/trade "
        f"{risk_config.risk_per_trade_pct:.3f}%, lifecycle effective risk <= "
        f"{effective_risk:.3f}%, quote loop 5s flat / 1s exposed."
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
    parser.add_argument(
        "--mode", choices=("shadow", "paper", "live"), default="shadow"
    )
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
        print(
            f"LIVE requires the explicit phrase: --live-ack {LIVE_ACK}",
            file=sys.stderr,
        )
        return 2
    symbols = tuple(args.symbols or HIGH_VELOCITY_CRYPTO_UNIVERSE)
    if len(symbols) < 3 or len(symbols) > 12:
        print("Intraday runtime universe must contain 3-12 symbols.", file=sys.stderr)
        return 2

    quant_store = QuantStore(args.quant_db)
    try:
        if args.mode == "shadow":
            return _shadow_loop(symbols, quant_store, args.poll_seconds)
        if args.mode == "paper":
            return _broker_loop(
                symbols=symbols,
                quant_store=quant_store,
                state_db=args.state_db,
                account_mode=AccountMode.PAPER,
                max_runtime_hours=args.max_runtime_hours,
            )
        return _broker_loop(
            symbols=symbols,
            quant_store=quant_store,
            state_db=args.live_state_db,
            account_mode=AccountMode.LIVE,
            max_runtime_hours=args.max_runtime_hours,
        )
    except (BrokerError, RuntimeError, ValueError) as error:
        print(f"Runtime refused/failed: {error}", file=sys.stderr)
        return 2
    finally:
        quant_store.close()


if __name__ == "__main__":
    raise SystemExit(main())
