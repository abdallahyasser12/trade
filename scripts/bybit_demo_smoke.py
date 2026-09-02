#!/usr/bin/env python3
"""Opt-in economic smoke against Bybit DEMO Spot only.

No automated CI invokes this script.  It requires an explicit confirmation string and
BYBIT_DEMO credentials.  It buys the smallest safely representable BTCUSDT quantity
within the configured notional cap, reconciles the fill, then sells only proven
bot-owned inventory and verifies that no bot-owned residual remains.
"""
from __future__ import annotations

import argparse
import math
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from broker import BrokerError, make_client_order_id
from bybit_burnin import credentials_from_environment
from bybit_market_data import BybitPrivateWebSocketManager, BybitSpotDataFeed
from bybit_spot import BybitSpotBroker, decimal_ceil
from bybit_state import BybitStateStore
from crypto_venue import CryptoExecutionEnvironment

ACK = "I_ACCEPT_BYBIT_DEMO_ORDER"


def _wait_order(broker, client_id: str, timeout_seconds: float = 30.0):
    deadline = time.monotonic() + timeout_seconds
    last = None
    while time.monotonic() < deadline:
        last = broker.get_order_by_client_id(client_id)
        if last is not None and last.is_terminal:
            return last
        time.sleep(0.5)
    return last


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--symbol", choices=("BTC/USD", "ETH/USD", "SOL/USD"), default="BTC/USD")
    parser.add_argument("--max-notional-usdt", type=float, default=25.0)
    parser.add_argument("--state-db", default="data/bybit_demo_smoke.db")
    args = parser.parse_args()
    if args.confirm != ACK:
        print(f"Economic Demo smoke requires --confirm {ACK}", file=sys.stderr)
        return 2
    if not math.isfinite(args.max_notional_usdt) or args.max_notional_usdt <= 0:
        print("max-notional-usdt must be positive", file=sys.stderr)
        return 2

    key, secret = credentials_from_environment()
    state = BybitStateStore(args.state_db)
    broker = BybitSpotBroker(
        key,
        secret,
        environment=CryptoExecutionEnvironment.BYBIT_DEMO,
        state=state,
        strategy_capital_budget_usdt=args.max_notional_usdt,
        permitted_symbols=(args.symbol,),
        allow_live=False,
    )
    feed = BybitSpotDataFeed(broker.http, symbols=(args.symbol,), timeframe="5Min")
    private = BybitPrivateWebSocketManager(
        key,
        secret,
        environment=CryptoExecutionEnvironment.BYBIT_DEMO,
        on_execution=lambda _message: broker.sync_bot_executions(args.symbol),
    )
    session = "smoke" + uuid.uuid4().hex[:8]
    entry_id = make_client_order_id(session, "entry", datetime.now(timezone.utc), args.symbol)
    exit_id = None
    try:
        env = broker.verify_environment()
        metadata = broker.instrument_provider.get_instrument_metadata(args.symbol, force_refresh=True)
        feed.start()
        private.start()
        quote = feed.get_latest_quotes([args.symbol])[args.symbol]
        price = quote.ask_price
        qty = decimal_ceil(
            max(metadata.min_order_qty, metadata.min_order_amount / price),
            metadata.quantity_step,
        )
        notional = qty * price
        if notional > args.max_notional_usdt + 1e-9:
            raise BrokerError(
                f"Dynamic Bybit minimum requires {notional:.4f} USDT, above smoke cap "
                f"{args.max_notional_usdt:.4f}. Raise the explicit Demo cap if desired."
            )
        print(
            f"DEMO verified: host={env['rest_host']}, Spot only, isLeverage=0; "
            f"submitting {qty:.12g} {args.symbol} (~{notional:.4f} USDT)."
        )
        entry = broker.submit_crypto_market_order(
            args.symbol,
            qty,
            "buy",
            entry_id,
            trade_increment=metadata.quantity_step,
            min_order_size=metadata.min_order_qty,
            reference_price=price,
        )
        entry = _wait_order(broker, entry_id) or entry
        broker.sync_bot_executions(args.symbol)
        position = broker.get_position(args.symbol)
        if position is None or position.qty <= 0:
            if entry.is_open and entry.broker_order_id:
                broker.cancel_order(entry.broker_order_id)
            raise BrokerError(f"Demo entry did not produce reconciled bot-owned inventory ({entry.status})")

        exit_id = make_client_order_id(
            session,
            "exit",
            datetime.now(timezone.utc),
            args.symbol,
        )
        quote = feed.get_latest_quotes([args.symbol])[args.symbol]
        exit_order = broker.submit_crypto_market_order(
            args.symbol,
            position.qty,
            "sell",
            exit_id,
            trade_increment=metadata.quantity_step,
            min_order_size=metadata.min_order_qty,
            reference_price=quote.bid_price,
        )
        exit_order = _wait_order(broker, exit_id) or exit_order
        broker.sync_bot_executions(args.symbol)
        residual = broker.get_position(args.symbol)
        if residual is not None and residual.qty > metadata.quantity_step / 10.0:
            raise BrokerError(
                f"Demo smoke left {residual.qty:.12g} proven bot-owned residual; manual review required"
            )
        print(
            "DEMO_SMOKE_PASS: account/metadata/public data/private stream/order entry/"
            "execution reconciliation/close verified; no bot-owned residual."
        )
        return 0
    except Exception as exc:
        print(f"DEMO_SMOKE_FAIL: {exc}", file=sys.stderr)
        return 2
    finally:
        private.stop()
        feed.stop()
        broker.close()
        state.close()


if __name__ == "__main__":
    raise SystemExit(main())
