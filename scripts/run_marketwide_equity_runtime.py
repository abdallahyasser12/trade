#!/usr/bin/env python3
"""Run the manifest-locked market-wide US-equity Quant Desk runtime.

Broker-backed equity execution remains paper-only. Strategy lifecycle state controls
whether each hypothesis is skipped (RESEARCH), virtual-only (SHADOW), or allowed to
reach the paper broker (PAPER). Any forward-state strategy must match its promoted
research commit/config/universe exactly. Ctrl-C requests a graceful stop and leaves
protective GTC stops in place for still-open bot-owned positions.
"""

from __future__ import annotations

import argparse
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from broker import AlpacaPaperBroker
from config import AccountMode, load_credentials
from equity_market_data import AlpacaEquityMarketData, EquityMarketDataConfig
from equity_marketwide_deployment import ManifestLockedMarketWideEquityTradeEngine
from equity_marketwide_engine import clear_marketwide_equity_kill_switch
from quant.equity_lifecycle import marketwide_equity_promotion_requirements
from quant.equity_marketwide import MarketWideEquityConfig
from quant.lifecycle import LifecycleManager
from quant.store import DEFAULT_QUANT_DB_PATH, QuantStore
from state_store import StateStore


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manifest-locked US-equity intraday scanner + guarded paper execution."
    )
    parser.add_argument("--timeframe", choices=("5Min", "15Min"), default="15Min")
    parser.add_argument("--feed", choices=("iex", "sip"), default="iex")
    parser.add_argument(
        "--max-scan-symbols",
        type=int,
        default=48,
        help="Must exactly match the promoted research configuration in SHADOW/PAPER.",
    )
    parser.add_argument("--max-positions", type=int, default=6)
    parser.add_argument("--risk-per-trade-pct", type=float, default=0.15)
    parser.add_argument("--max-session-loss-pct", type=float, default=2.0)
    parser.add_argument(
        "--state-db",
        default=str(Path("data") / "equity_marketwide_trading.db"),
        help="Dedicated crash-safe execution database.",
    )
    parser.add_argument(
        "--quant-db",
        default=DEFAULT_QUANT_DB_PATH,
        help="Quant Desk evidence/lifecycle database. Use the same DB as research/promotions.",
    )
    parser.add_argument(
        "--acknowledge-kill-switch",
        action="store_true",
        help="Explicitly clear a previously armed market-wide equity kill switch before startup.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    credentials = load_credentials(account_mode=AccountMode.PAPER)
    if not credentials.is_configured:
        print(
            "Paper Alpaca credentials are missing. Set ALPACA_PAPER_API_KEY and "
            "ALPACA_PAPER_SECRET_KEY (or the legacy paper variables).",
            file=sys.stderr,
        )
        return 2

    config = MarketWideEquityConfig(
        timeframe=args.timeframe,
        max_scan_symbols=args.max_scan_symbols,
        shard_size=min(48, args.max_scan_symbols),
        max_positions=args.max_positions,
        risk_per_trade_pct=args.risk_per_trade_pct,
        max_session_loss_pct=args.max_session_loss_pct,
    )
    errors = config.validate()
    if errors:
        print("Invalid runtime configuration: " + "; ".join(errors), file=sys.stderr)
        return 2

    store = StateStore(args.state_db)
    quant_store = QuantStore(args.quant_db)
    engine = None
    try:
        if args.acknowledge_kill_switch:
            clear_marketwide_equity_kill_switch(store)
        broker = AlpacaPaperBroker(credentials.api_key, credentials.secret_key)
        market_data = AlpacaEquityMarketData(
            credentials.api_key,
            credentials.secret_key,
            EquityMarketDataConfig(feed=args.feed),
        )
        lifecycle = LifecycleManager(
            quant_store,
            requirements=marketwide_equity_promotion_requirements(),
        )
        engine = ManifestLockedMarketWideEquityTradeEngine(
            broker,
            market_data,
            store,
            config=config,
            quant_store=quant_store,
            lifecycle=lifecycle,
        )

        stopping = False

        def request_stop(_signum, _frame):
            nonlocal stopping
            if stopping:
                return
            stopping = True
            print("Graceful stop requested; protective bot stops are left in place.")
            engine.stop()

        signal.signal(signal.SIGINT, request_stop)
        signal.signal(signal.SIGTERM, request_stop)

        engine.start()
        while engine.is_running:
            status = engine.status
            if status.error:
                print(f"Engine error: {status.error}", file=sys.stderr)
            time.sleep(1.0)
        engine.join()
        final = engine.status
        print(
            f"state={final.state.value} selected={final.selected_symbols} "
            f"open_bot_positions={final.open_bot_positions} "
            f"last_candidates={final.last_executable_candidates}"
        )
        return 1 if final.error else 0
    finally:
        if engine is not None and engine.is_running:
            engine.stop()
            engine.join(5.0)
        quant_store.close()
        store.close()


if __name__ == "__main__":
    raise SystemExit(main())
