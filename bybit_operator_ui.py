"""Simple Streamlit operator UI for the Bybit Spot Demo burn-in.

Run with:
    streamlit run bybit_operator_ui.py

The UI is deliberately a control/observability shell.  The detached backend process
owns trading, reconciliation and protection; closing the browser does not stop it.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st

from bybit_active_report import ActiveSafeBybitBurnInReport
from bybit_operator_control import OperatorControl

VENUE_DB = "data/bybit_demo_venue.db"
ENGINE_DB = "data/bybit_demo_trading.db"
CONTROL_DB = "data/bybit_operator.db"
SUPPORTED = ("BTC/USD", "ETH/USD", "SOL/USD")

st.set_page_config(page_title="Trade Bot · Bybit Demo", page_icon="📈", layout="wide")


def _query(path: str, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    con = sqlite3.connect(path, timeout=2.0)
    con.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in con.execute(sql, params).fetchall()]
    except sqlite3.Error:
        return []
    finally:
        con.close()


def _latest_run() -> Optional[dict[str, Any]]:
    rows = _query(
        VENUE_DB,
        "SELECT * FROM burnin_runs ORDER BY CASE WHEN status='running' THEN 0 ELSE 1 END,start_time DESC LIMIT 1",
    )
    return rows[0] if rows else None


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    try:
        return json.loads(row.get("payload_json") or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}


def _run_config(run_id: str) -> dict[str, Any]:
    rows = _query(
        VENUE_DB,
        """SELECT payload_json FROM burnin_events
           WHERE run_id=? AND event_type IN ('run_start','run_resume')
           ORDER BY timestamp LIMIT 1""",
        (run_id,),
    )
    return (_payload(rows[0]).get("config") or {}) if rows else {}


def _latest_decisions(run_id: str) -> dict[str, dict[str, Any]]:
    rows = _query(
        VENUE_DB,
        """SELECT * FROM burnin_events WHERE run_id=? AND event_type='decision'
           ORDER BY timestamp DESC,id DESC LIMIT 200""",
        (run_id,),
    )
    result = {}
    for row in rows:
        symbol = row.get("symbol")
        if symbol and symbol not in result:
            result[symbol] = _payload(row)
    return result


def _latest_health(run_id: str) -> dict[str, Any]:
    rows = _query(
        VENUE_DB,
        "SELECT * FROM health_samples WHERE run_id=? ORDER BY timestamp DESC LIMIT 1",
        (run_id,),
    )
    return rows[0] if rows else {}


def _latest_prices(decisions: dict[str, dict[str, Any]]) -> dict[str, float]:
    result = {}
    for symbol, decision in decisions.items():
        micro = decision.get("microstructure") or {}
        bid, ask = micro.get("bid"), micro.get("ask")
        try:
            if bid and ask:
                result[symbol] = (float(bid) + float(ask)) / 2.0
            elif decision.get("close"):
                result[symbol] = float(decision["close"])
        except (TypeError, ValueError):
            pass
    return result


def _inventory() -> list[dict[str, Any]]:
    rows = _query(
        VENUE_DB,
        """SELECT i.*,m.original_qty,m.phase,m.highest_price,m.lowest_price,
                  m.initial_stop_price
           FROM owned_inventory i LEFT JOIN management_state m
             ON m.strategy_symbol=i.strategy_symbol
           WHERE i.qty>0 ORDER BY i.strategy_symbol""",
    )
    return rows


def _today_realized() -> float:
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    rows = _query(
        ENGINE_DB,
        "SELECT realized_pnl FROM trades WHERE timestamp>=? AND realized_pnl IS NOT NULL",
        (start,),
    )
    return sum(float(row["realized_pnl"] or 0.0) for row in rows)


def _fmt_money(value: Optional[float]) -> str:
    if value is None or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):,.2f} USDT"


def _fmt_pct(value: Optional[float], fraction: bool = False) -> str:
    if value is None:
        return "—"
    number = float(value) * (100.0 if fraction else 1.0)
    return f"{number:.2f}%"


def _public_24h(symbols: tuple[str, ...]) -> dict[str, dict[str, float]]:
    try:
        from pybit.unified_trading import HTTP
        from crypto_venue import bybit_instrument

        http = HTTP(testnet=False)
        result = {}
        for symbol in symbols:
            response = http.get_tickers(category="spot", symbol=bybit_instrument(symbol).venue_symbol)
            rows = ((response.get("result") or {}).get("list") or [])
            if not rows:
                continue
            row = rows[0]
            result[symbol] = {
                "price": float(row.get("lastPrice") or 0.0),
                "change_24h": float(row.get("price24hPcnt") or 0.0) * 100.0,
            }
        return result
    except Exception:
        return {}


def _header(run: Optional[dict[str, Any]], control_status: dict[str, Any]) -> None:
    st.title("BYBIT DEMO")
    st.caption("Spot only · no leverage · no margin · no derivatives · live trading disabled")
    health = _latest_health(run["run_id"]) if run else {}
    connected = bool(health) and bool(health.get("public_ws_up")) and bool(health.get("private_ws_up"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Connection", "Connected" if connected else "Disconnected")
    c2.metric("Bot", "Running" if control_status.get("process_alive") else "Stopped")
    if run:
        started = datetime.fromisoformat(run["start_time"])
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        uptime = max(0.0, (datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds() / 3600.0)
        c3.metric("Uptime", f"{uptime:.1f} h")
        c4.metric("Health", health.get("health") or "UNKNOWN")
    else:
        c3.metric("Uptime", "—")
        c4.metric("Health", "STOPPED")


def render_home(run: Optional[dict[str, Any]], control: OperatorControl, control_status: dict[str, Any]) -> None:
    _header(run, control_status)
    config = _run_config(run["run_id"]) if run else {}
    decisions = _latest_decisions(run["run_id"]) if run else {}
    prices = _latest_prices(decisions)
    inventory = _inventory()
    exposure = sum(float(row["qty"]) * prices.get(row["strategy_symbol"], 0.0) for row in inventory)

    a, b, c, d = st.columns(4)
    a.metric("Capital allocated", _fmt_money(config.get("strategy_capital_budget_usdt")))
    b.metric("Today net P&L", _fmt_money(_today_realized()))
    c.metric("Open exposure", _fmt_money(exposure))
    with d:
        st.write("Kill Switch")
        if st.button("EMERGENCY STOP", type="primary", use_container_width=True):
            try:
                control.emergency_stop()
                st.error("Emergency kill switch requested.")
            except Exception as exc:
                st.error(str(exc))

    st.subheader("Top opportunities")
    ordered = sorted(
        decisions.items(),
        key=lambda item: float(item[1].get("score") or -1e9),
        reverse=True,
    )
    columns = st.columns(max(1, min(3, len(ordered) or 3)))
    if not ordered:
        st.info("No completed-bar decisions recorded yet.")
    for index, (symbol, decision) in enumerate(ordered[:3]):
        action = decision.get("operator_decision") or ("BUY SETUP" if decision.get("action") == "BUY" else "NO TRADE")
        entry = decision.get("close")
        stop = decision.get("stop_price")
        target = (decision.get("details") or {}).get("target_price")
        rr = None
        try:
            if target and stop and entry and float(entry) > float(stop):
                rr = (float(target) - float(entry)) / (float(entry) - float(stop))
        except (TypeError, ValueError, ZeroDivisionError):
            rr = None
        with columns[index % len(columns)]:
            st.markdown(f"### {symbol.split('/')[0]}")
            st.metric("Decision", str(action).replace("_", " "))
            st.write(f"Score: **{float(decision.get('score') or 0.0):.1f}**")
            st.write(f"Price: **{float(entry):,.4f}**" if entry else "Price: —")
            st.write("Entry: current setup price after all gates" if action == "BUY_SETUP" else "Entry: —")
            st.write(f"Target: **{float(target):,.4f}**" if target else "Target: dynamic / policy-based")
            st.write(f"Stop: **{float(stop):,.4f}**" if stop else "Stop: —")
            st.write(f"Risk/reward: **{rr:.2f}R**" if rr is not None else "Risk/reward: dynamic")
            st.caption(str(decision.get("reason") or "No setup reason recorded"))

    st.subheader("Open positions")
    if not inventory:
        st.caption("No bot-owned Spot inventory is open.")
    else:
        table = []
        for row in inventory:
            symbol = row["strategy_symbol"]
            current = prices.get(symbol)
            entry = float(row["cost_basis_quote"]) / float(row["qty"]) if float(row["qty"]) > 0 else None
            pnl = (current - entry) * float(row["qty"]) if current and entry else None
            table.append(
                {
                    "symbol": symbol,
                    "entry": entry,
                    "current": current,
                    "net P&L (approx)": pnl,
                    "stop": row.get("initial_stop_price"),
                    "target": "dynamic",
                    "phase": row.get("phase") or "NORMAL_POSITION",
                }
            )
        st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)

    st.subheader("Recent activity")
    if run:
        rows = _query(
            VENUE_DB,
            """SELECT timestamp,event_type,symbol,payload_json FROM burnin_events
               WHERE run_id=? ORDER BY timestamp DESC,id DESC LIMIT 15""",
            (run["run_id"],),
        )
        activity = []
        for row in rows:
            payload = _payload(row)
            activity.append(
                {
                    "time": row["timestamp"],
                    "symbol": row.get("symbol") or "—",
                    "event": row["event_type"],
                    "detail": payload.get("reason") or payload.get("action") or payload.get("kind") or "",
                }
            )
        st.dataframe(pd.DataFrame(activity), use_container_width=True, hide_index=True)


def render_markets(run: Optional[dict[str, Any]]) -> None:
    st.title("Markets")
    symbols = tuple(json.loads(run["universe_json"])) if run else SUPPORTED
    decisions = _latest_decisions(run["run_id"]) if run else {}
    public = _public_24h(symbols)
    cols = st.columns(len(symbols))
    for idx, symbol in enumerate(symbols):
        decision = decisions.get(symbol, {})
        details = decision.get("details") or {}
        micro = decision.get("microstructure") or {}
        ticker = public.get(symbol, {})
        price = ticker.get("price") or decision.get("close")
        bull = details.get("bull_probability")
        momentum = details.get("momentum_sleeve")
        atr_ratio = details.get("atr_ratio")
        with cols[idx]:
            st.subheader(symbol.split("/")[0])
            st.metric("Price", f"{float(price):,.4f}" if price else "—", f"{ticker.get('change_24h', 0):+.2f}%" if ticker else None)
            st.write(f"Regime: **{decision.get('regime') or 'UNKNOWN'}**")
            st.write(f"Spread: **{float(micro.get('spread_bps')):.2f} bps**" if micro.get("spread_bps") is not None else "Spread: —")
            st.write(f"Trend: **{'Bullish' if bull is not None and float(bull) >= 0.55 else 'Mixed'}**")
            st.write(f"Momentum: **{float(momentum):.2f}**" if momentum is not None else "Momentum: —")
            st.write(f"Volatility ratio: **{float(atr_ratio):.2f}**" if atr_ratio is not None else "Volatility: —")
            st.write(f"Setup score: **{float(decision.get('score') or 0.0):.1f}**")
            with st.expander("Why?"):
                st.write(f"Trend: {'Bullish' if bull is not None and float(bull) >= 0.55 else 'Mixed'}")
                st.write(f"Momentum: {'Healthy' if momentum is not None and float(momentum) >= 0.4 else 'Limited'}")
                st.write(f"Liquidity: {'Excellent' if micro.get('spread_bps') is not None and float(micro['spread_bps']) <= 10 else 'Check spread/depth'}")
                st.write(f"Market regime: {decision.get('regime') or 'Unknown'}")
                st.write(f"Expected costs: {'Acceptable' if micro.get('spread_bps') is not None and float(micro['spread_bps']) <= 25 else 'Needs validation'}")
                st.write(f"Decision: {decision.get('operator_decision') or decision.get('action') or 'NO TRADE'}")
                with st.expander("Advanced indicators"):
                    st.json({"details": details, "microstructure": micro})


def render_performance(run: Optional[dict[str, Any]]) -> None:
    st.title("Performance")
    if not run:
        st.info("No Bybit burn-in exists yet.")
        return
    try:
        report = ActiveSafeBybitBurnInReport(VENUE_DB, ENGINE_DB).build(run["run_id"])
        tr = report["trading"]
        sm = report["small_move_hypothesis"]
        top = st.columns(4)
        top[0].metric("Burn-in P&L", _fmt_money(tr["net_pnl_usdt"]))
        top[1].metric("Trades", tr["total_trades"])
        top[2].metric("Win rate", _fmt_pct(tr["win_rate"], fraction=True))
        pf = tr["profit_factor"]
        top[3].metric("Profit factor", "—" if pf is None else ("∞" if math.isinf(pf) else f"{pf:.2f}"))
        more = st.columns(4)
        more[0].metric("Expectancy", _fmt_money(tr["expectancy_usdt"]))
        more[1].metric("Drawdown", _fmt_pct(tr["maximum_drawdown_pct"]))
        more[2].metric("Fees", _fmt_money(tr["fees_quote_estimate"]))
        avg_w = tr["average_winner_usdt"]
        avg_l = tr["average_loser_usdt"]
        more[3].metric("Avg win / loss", f"{avg_w:.2f} / {avg_l:.2f}" if avg_w is not None and avg_l is not None else "—")

        curve = _query(
            ENGINE_DB,
            "SELECT timestamp,equity FROM equity_snapshots ORDER BY timestamp",
        )
        if curve:
            frame = pd.DataFrame(curve)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            st.line_chart(frame.set_index("timestamp")["equity"])

        st.subheader("Small-move reach rate")
        c1, c2, c3 = st.columns(3)
        c1.metric("+0.5%", _fmt_pct(sm["reach_0_50_pct"], fraction=True))
        c2.metric("+0.75%", _fmt_pct(sm["reach_0_75_pct"], fraction=True))
        c3.metric("+1.0%", _fmt_pct(sm["reach_1_00_pct"], fraction=True))
        st.caption("Losing trades are included. A 3–5 day Demo run does not prove a durable edge.")
    except Exception as exc:
        st.warning(f"Performance evidence is not available yet: {exc}")


def render_settings(run: Optional[dict[str, Any]], control: OperatorControl, control_status: dict[str, Any]) -> None:
    st.title("Settings")
    st.write("Venue: **Bybit**")
    st.write("Environment: **DEMO** — live/real-money construction is disabled by default")
    configured = bool(os.getenv("BYBIT_DEMO_API_KEY")) and bool(os.getenv("BYBIT_DEMO_API_SECRET"))
    st.write(f"Demo credentials: **{'configured on server' if configured else 'not configured'}**")
    capital = st.number_input("Capital allocation (USDT)", min_value=20.0, value=1_000.0, step=100.0)
    risk = st.selectbox("Risk profile", ["Conservative burn-in"], disabled=True)
    allowed_coins = st.multiselect("Allowed coins", ["BTC", "ETH", "SOL"], default=["BTC", "ETH", "SOL"])
    hours = st.radio("Burn-in duration", [72, 120], horizontal=True, format_func=lambda x: f"{x // 24} days")
    egp_rate = st.number_input("Optional EGP reference rate (display only)", min_value=0.0, value=0.0, step=0.1)
    if egp_rate > 0:
        st.caption(f"Display reference only: allocated capital ≈ EGP {capital * egp_rate:,.0f}. Trading math remains USDT-based.")
    st.write("Notifications: use the existing mobile notification integration from Advanced / Research if configured.")

    start, stop = st.columns(2)
    with start:
        if st.button("Start Bybit Demo", type="primary", use_container_width=True, disabled=control_status.get("process_alive", False)):
            if not configured:
                st.error("Configure BYBIT_DEMO_API_KEY and BYBIT_DEMO_API_SECRET on the server first.")
            elif not allowed_coins:
                st.error("Select at least one permitted Spot coin.")
            else:
                symbols = tuple(f"{coin}/USD" for coin in allowed_coins)
                try:
                    pid = control.start_detached(
                        hours=int(hours),
                        capital_usdt=float(capital),
                        symbols=symbols,
                    )
                    st.success(f"Detached Demo backend started (PID {pid}).")
                except Exception as exc:
                    st.error(str(exc))
    with stop:
        if st.button("Stop gracefully", use_container_width=True, disabled=not control_status.get("process_alive", False)):
            try:
                control.request_stop()
                st.info("Graceful stop requested; protection/reconciliation remain backend-owned during shutdown.")
            except Exception as exc:
                st.error(str(exc))

    with st.expander("Advanced / Research"):
        st.warning("These controls are intentionally outside normal operation.")
        st.code("streamlit run app.py", language="bash")
        st.write("Walk-forward grids, Quant Desk internals, lifecycle administration, raw diagnostics and exit-policy experiments remain in the existing research application.")
        st.write("Smart-exit behaviors are wired in production but are feature-gated until chronological research justifies enabling them for a burn-in.")


control = OperatorControl(CONTROL_DB)
try:
    run = _latest_run()
    control_status = control.status()
    page = st.sidebar.radio("Navigation", ["HOME", "MARKETS", "PERFORMANCE", "SETTINGS"])
    st.sidebar.caption("BYBIT DEMO · Spot only")
    if page == "HOME":
        render_home(run, control, control_status)
    elif page == "MARKETS":
        render_markets(run)
    elif page == "PERFORMANCE":
        render_performance(run)
    else:
        render_settings(run, control, control_status)
finally:
    control.close()
