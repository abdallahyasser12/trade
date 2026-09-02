"""Primary Streamlit operator dashboard for the Bybit Spot Demo burn-in.

Trading never runs in Streamlit. Start launches a detached server-side burn-in service;
all protection, reconciliation and WebSocket handling remain backend-owned if the browser
closes or reruns. Secrets remain process environment values and are never copied into
browser/session state.
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
from intraday_crypto_engine import ADAPTIVE_STRATEGY_ID
from quant.lifecycle import LifecycleManager
from quant.store import QuantStore

VENUE_DB = "data/bybit_demo_venue.db"
ENGINE_DB = "data/bybit_demo_trading.db"
CONTROL_DB = "data/bybit_operator.db"
QUANT_DB = "data/quant_desk.db"
SUPPORTED = ("BTC/USD", "ETH/USD", "SOL/USD")
DEMO_EXECUTION_STATES = {"PAPER", "LIVE_CANARY", "LIVE", "REDUCED_RISK"}

st.set_page_config(
    page_title="Trade Bot · Bybit Demo",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _query(path: str, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    if not Path(path).exists():
        return []
    connection = sqlite3.connect(path, timeout=2.0)
    connection.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in connection.execute(sql, params).fetchall()]
    except sqlite3.Error:
        return []
    finally:
        connection.close()


def _latest_run() -> Optional[dict[str, Any]]:
    rows = _query(
        VENUE_DB,
        """SELECT * FROM burnin_runs
           ORDER BY CASE WHEN status='running' THEN 0 ELSE 1 END,start_time DESC LIMIT 1""",
    )
    return rows[0] if rows else None


def _payload(row: dict[str, Any]) -> dict[str, Any]:
    try:
        value = json.loads(row.get("payload_json") or "{}")
    except (TypeError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


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
           ORDER BY timestamp DESC,id DESC LIMIT 300""",
        (run_id,),
    )
    result: dict[str, dict[str, Any]] = {}
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


def _inventory() -> list[dict[str, Any]]:
    return _query(
        VENUE_DB,
        """SELECT i.*,m.original_qty,m.phase,m.highest_price,m.lowest_price,
                  m.initial_stop_price,m.current_stop_price
           FROM owned_inventory i LEFT JOIN management_state m
             ON m.strategy_symbol=i.strategy_symbol
           WHERE i.qty>0 ORDER BY i.strategy_symbol""",
    )


def _latest_prices(decisions: dict[str, dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for symbol, decision in decisions.items():
        micro = decision.get("microstructure") or {}
        try:
            bid = float(micro.get("bid"))
            ask = float(micro.get("ask"))
            if bid > 0 and ask >= bid:
                result[symbol] = (bid + ask) / 2.0
                continue
        except (TypeError, ValueError):
            pass
        try:
            close = float(decision.get("close"))
            if close > 0:
                result[symbol] = close
        except (TypeError, ValueError):
            pass
    return result


def _today_realized() -> float:
    now = datetime.now(timezone.utc)
    start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
    rows = _query(
        ENGINE_DB,
        "SELECT realized_pnl FROM trades WHERE timestamp>=? AND realized_pnl IS NOT NULL",
        (start,),
    )
    return sum(float(row["realized_pnl"] or 0.0) for row in rows)


def _lifecycle_state() -> str:
    store = QuantStore(QUANT_DB)
    try:
        return LifecycleManager(store).state(ADAPTIVE_STRATEGY_ID).value
    except Exception:
        return "UNKNOWN"
    finally:
        store.close()


def _fmt_money(value: Optional[float]) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    return f"{number:,.2f} USDT" if math.isfinite(number) else "—"


def _fmt_pct(value: Optional[float], *, fraction: bool = False) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "—"
    if not math.isfinite(number):
        return "—"
    if fraction:
        number *= 100.0
    return f"{number:.2f}%"


@st.cache_data(ttl=5, show_spinner=False)
def _public_24h(symbols: tuple[str, ...]) -> dict[str, dict[str, float]]:
    """Public-only market snapshot. No API credentials are involved."""
    try:
        from pybit.unified_trading import HTTP
        from crypto_venue import bybit_instrument

        http = HTTP(testnet=False)
        result = {}
        for symbol in symbols:
            response = http.get_tickers(
                category="spot", symbol=bybit_instrument(symbol).venue_symbol
            )
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
    st.caption(
        "Spot only · isLeverage=0 · no margin · no derivatives · long-only · live trading disabled"
    )
    health = _latest_health(run["run_id"]) if run else {}
    public_up = bool(health.get("public_ws_up"))
    private_up = bool(health.get("private_ws_up"))
    connected = public_up and private_up
    cols = st.columns(5)
    cols[0].metric("Venue", "BYBIT DEMO")
    cols[1].metric("Connection", "Connected" if connected else "Disconnected")
    cols[2].metric("Bot", "Running" if control_status.get("process_alive") else "Stopped")
    if run:
        started = datetime.fromisoformat(run["start_time"])
        if started.tzinfo is None:
            started = started.replace(tzinfo=timezone.utc)
        uptime = max(
            0.0,
            (datetime.now(timezone.utc) - started.astimezone(timezone.utc)).total_seconds()
            / 3600.0,
        )
        cols[3].metric("Uptime", f"{uptime:.1f} h")
        cols[4].metric("Health", health.get("health") or "UNKNOWN")
    else:
        cols[3].metric("Uptime", "—")
        cols[4].metric("Health", "STOPPED")

    if health:
        details = st.columns(4)
        details[0].caption(f"Public WS: {'UP' if public_up else 'DOWN'}")
        details[1].caption(f"Private WS: {'UP' if private_up else 'DOWN'}")
        details[2].caption(
            f"Clock skew: {float(health.get('clock_skew_ms') or 0.0):.0f} ms"
        )
        details[3].caption(
            f"Unprotected qty: {float(health.get('unprotected_qty') or 0.0):.8f}"
        )


def _decision_label(decision: dict[str, Any]) -> str:
    label = str(decision.get("operator_decision") or "NO_TRADE").replace("_", " ")
    if label not in {"BUY SETUP", "WATCH", "NO TRADE", "AVOID"}:
        return "NO TRADE"
    return label


def render_home(
    run: Optional[dict[str, Any]],
    control: OperatorControl,
    control_status: dict[str, Any],
) -> None:
    _header(run, control_status)
    config = _run_config(run["run_id"]) if run else {}
    decisions = _latest_decisions(run["run_id"]) if run else {}
    prices = _latest_prices(decisions)
    inventory = _inventory()
    exposure = sum(
        float(row["qty"] or 0.0) * prices.get(row["strategy_symbol"], 0.0)
        for row in inventory
    )

    summary = st.columns(4)
    summary[0].metric(
        "Capital allocated", _fmt_money(config.get("strategy_capital_budget_usdt"))
    )
    summary[1].metric("Today net P&L", _fmt_money(_today_realized()))
    summary[2].metric("Open exposure", _fmt_money(exposure))
    kill = control_status.get("kill_switch") or {}
    summary[3].metric("Kill switch", "ARMED" if kill.get("armed") else "Ready")

    if kill.get("armed"):
        st.error(
            f"Emergency kill switch is armed: {kill.get('reason') or 'UNKNOWN'}. "
            "The backend cannot start until this is explicitly acknowledged in Settings."
        )

    st.subheader("Top opportunities")
    ordered = sorted(
        decisions.items(),
        key=lambda item: float(item[1].get("score") or -1e9),
        reverse=True,
    )
    if not ordered:
        st.info("No completed-bar decision has been recorded yet. Zero trades is valid.")
    else:
        columns = st.columns(min(3, len(ordered)))
        for index, (symbol, decision) in enumerate(ordered[:3]):
            details = decision.get("details") or {}
            entry = decision.get("close")
            stop = decision.get("stop_price")
            target = details.get("target_price")
            rr = None
            try:
                if target and stop and entry and float(entry) > float(stop):
                    rr = (float(target) - float(entry)) / (float(entry) - float(stop))
            except (TypeError, ValueError, ZeroDivisionError):
                rr = None
            with columns[index]:
                st.markdown(f"### {symbol.split('/')[0]}")
                st.metric("Decision", _decision_label(decision))
                st.write(f"Setup score: **{float(decision.get('score') or 0.0):.1f}**")
                st.write(f"Price: **{float(entry):,.4f}**" if entry else "Price: —")
                st.write(
                    f"Target: **{float(target):,.4f}**"
                    if target
                    else "Target: structure / ATR / R based"
                )
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
            qty = float(row["qty"] or 0.0)
            current = prices.get(symbol)
            entry = float(row["cost_basis_quote"] or 0.0) / qty if qty > 0 else None
            pnl = (current - entry) * qty if current and entry else None
            table.append(
                {
                    "symbol": symbol,
                    "qty": qty,
                    "entry": entry,
                    "current": current,
                    "net P&L approx": pnl,
                    "stop": row.get("current_stop_price") or row.get("initial_stop_price"),
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
               WHERE run_id=? ORDER BY timestamp DESC,id DESC LIMIT 20""",
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
                    "detail": payload.get("reason")
                    or payload.get("action")
                    or payload.get("kind")
                    or "",
                }
            )
        st.dataframe(pd.DataFrame(activity), use_container_width=True, hide_index=True)


def render_markets(run: Optional[dict[str, Any]]) -> None:
    st.title("Markets")
    symbols = tuple(json.loads(run["universe_json"])) if run else SUPPORTED
    decisions = _latest_decisions(run["run_id"]) if run else {}
    public = _public_24h(symbols)
    columns = st.columns(len(symbols))
    for index, symbol in enumerate(symbols):
        decision = decisions.get(symbol, {})
        details = decision.get("details") or {}
        micro = decision.get("microstructure") or {}
        ticker = public.get(symbol, {})
        price = ticker.get("price") or decision.get("close")
        bull = details.get("bull_probability")
        momentum = details.get("momentum_sleeve")
        atr_ratio = details.get("atr_ratio")
        spread = micro.get("spread_bps")
        with columns[index]:
            st.subheader(symbol.split("/")[0])
            st.metric(
                "Price",
                f"{float(price):,.4f}" if price else "—",
                f"{ticker.get('change_24h', 0):+.2f}%" if ticker else None,
            )
            st.write(f"Regime: **{decision.get('regime') or 'UNKNOWN'}**")
            st.write(f"Spread: **{float(spread):.2f} bps**" if spread is not None else "Spread: —")
            st.write(
                f"Trend: **{'Bullish' if bull is not None and float(bull) >= 0.55 else 'Mixed'}**"
            )
            st.write(
                f"Momentum: **{float(momentum):.2f}**" if momentum is not None else "Momentum: —"
            )
            st.write(
                f"Volatility: **{float(atr_ratio):.2f}×**" if atr_ratio is not None else "Volatility: —"
            )
            st.write(f"Setup score: **{float(decision.get('score') or 0.0):.1f}**")
            with st.expander("Why?"):
                st.write(
                    f"Trend: {'Bullish' if bull is not None and float(bull) >= 0.55 else 'Mixed'}"
                )
                st.write(
                    f"Momentum: {'Healthy' if momentum is not None and float(momentum) >= 0.4 else 'Limited'}"
                )
                st.write(
                    f"Liquidity: {'Excellent' if spread is not None and float(spread) <= 10 else 'Check spread/depth'}"
                )
                st.write(f"Market regime: {decision.get('regime') or 'Unknown'}")
                st.write(
                    f"Expected costs: {'Acceptable' if spread is not None and float(spread) <= 25 else 'Needs validation'}"
                )
                st.write(f"Decision: {_decision_label(decision)}")
                with st.expander("Advanced indicators"):
                    st.json({"details": details, "microstructure": micro})


def render_performance(run: Optional[dict[str, Any]]) -> None:
    st.title("Performance")
    if not run:
        st.info("No Bybit burn-in exists yet.")
        return
    try:
        report = ActiveSafeBybitBurnInReport(VENUE_DB, ENGINE_DB).build(run["run_id"])
        trading = report["trading"]
        small = report["small_move_hypothesis"]
        top = st.columns(4)
        top[0].metric("Burn-in P&L", _fmt_money(trading["net_pnl_usdt"]))
        top[1].metric("Trades", trading["total_trades"])
        top[2].metric("Win rate", _fmt_pct(trading["win_rate"], fraction=True))
        pf = trading["profit_factor"]
        top[3].metric(
            "Profit factor",
            "—" if pf is None else ("∞" if math.isinf(float(pf)) else f"{float(pf):.2f}"),
        )
        more = st.columns(4)
        more[0].metric("Expectancy", _fmt_money(trading["expectancy_usdt"]))
        more[1].metric("Drawdown", _fmt_pct(trading["maximum_drawdown_pct"]))
        more[2].metric("Fees", _fmt_money(trading["fees_quote_estimate"]))
        average_win = trading["average_winner_usdt"]
        average_loss = trading["average_loser_usdt"]
        more[3].metric(
            "Avg win / loss",
            f"{float(average_win):.2f} / {float(average_loss):.2f}"
            if average_win is not None and average_loss is not None
            else "—",
        )

        curve = _query(
            ENGINE_DB,
            "SELECT timestamp,equity FROM equity_snapshots ORDER BY timestamp",
        )
        if curve:
            frame = pd.DataFrame(curve)
            frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
            st.line_chart(frame.set_index("timestamp")["equity"])

        st.subheader("Small-move reach rate")
        reach = st.columns(3)
        reach[0].metric("+0.5%", _fmt_pct(small["reach_0_50_pct"], fraction=True))
        reach[1].metric("+0.75%", _fmt_pct(small["reach_0_75_pct"], fraction=True))
        reach[2].metric("+1.0%", _fmt_pct(small["reach_1_00_pct"], fraction=True))

        verdict = report.get("verdict") or {}
        st.info(
            f"Operational: {verdict.get('operational', 'IN_PROGRESS')} · "
            f"Edge: {verdict.get('edge', 'EDGE_INCONCLUSIVE')} · Strategy proven: NO"
        )
        st.caption(
            "Losing trades are never hidden. A 72–120 hour Demo run is operational/preliminary evidence, not proof of durable profitability."
        )
    except Exception as exc:
        st.warning(f"Performance evidence is not available yet: {exc}")


def render_settings(
    run: Optional[dict[str, Any]],
    control: OperatorControl,
    control_status: dict[str, Any],
) -> None:
    st.title("Settings")
    st.write("Venue: **Bybit**")
    st.write("Environment: **DEMO**")
    st.caption("LIVE / real-money construction remains disabled and is not available here.")

    configured = bool(os.getenv("BYBIT_DEMO_API_KEY")) and bool(
        os.getenv("BYBIT_DEMO_API_SECRET")
    )
    lifecycle = _lifecycle_state()
    eligible = lifecycle in DEMO_EXECUTION_STATES
    status = st.columns(3)
    status[0].metric("Credentials", "Configured" if configured else "Missing")
    status[1].metric("Lifecycle", lifecycle)
    status[2].metric(
        "Backend", "Running" if control_status.get("process_alive") else "Stopped"
    )
    if not eligible:
        st.warning(
            "The adaptive strategy is not yet lifecycle-eligible for broker-backed Demo execution. "
            "Research/shadow promotion must complete first; the dashboard will not auto-promote it."
        )

    kill = control_status.get("kill_switch") or {}
    if kill.get("armed"):
        st.error(
            f"Kill switch ARMED · reason={kill.get('reason') or 'UNKNOWN'} · "
            f"at={kill.get('at') or 'unknown'}"
        )
        acknowledge = st.checkbox(
            "I reviewed the emergency reason and want to clear the restart latch",
            value=False,
        )
        if st.button(
            "Acknowledge kill switch",
            disabled=not acknowledge or control_status.get("process_alive", False),
        ):
            try:
                control.acknowledge_kill_switch()
                st.success("Kill switch acknowledged. Refresh before starting a new run.")
            except Exception as exc:
                st.error(str(exc))

    capital = st.number_input(
        "Capital allocation (USDT)", min_value=20.0, max_value=1_000_000.0, value=1_000.0, step=100.0
    )
    st.selectbox("Risk profile", ["Conservative burn-in"], disabled=True)
    allowed_coins = st.multiselect(
        "Allowed coins", ["BTC", "ETH", "SOL"], default=["BTC", "ETH", "SOL"]
    )
    hours = st.radio(
        "Burn-in duration",
        [72, 120],
        horizontal=True,
        format_func=lambda value: f"{value // 24} days",
    )
    egp_rate = st.number_input(
        "Optional EGP reference rate (display only)", min_value=0.0, value=0.0, step=0.1
    )
    if egp_rate > 0:
        st.caption(
            f"Reference only: allocated capital ≈ EGP {capital * egp_rate:,.0f}. Trading mathematics remains USDT-based."
        )

    can_start = (
        configured
        and eligible
        and bool(allowed_coins)
        and not control_status.get("process_alive", False)
        and not kill.get("armed")
    )
    buttons = st.columns(3)
    with buttons[0]:
        if st.button("Start Bybit Demo", type="primary", use_container_width=True, disabled=not can_start):
            symbols = tuple(f"{coin}/USD" for coin in allowed_coins)
            try:
                pid = control.start_detached(
                    hours=int(hours),
                    capital_usdt=float(capital),
                    symbols=symbols,
                )
                st.success(f"Bybit Demo backend started as server process {pid}.")
            except Exception as exc:
                st.error(str(exc))
    with buttons[1]:
        if st.button(
            "Stop gracefully",
            use_container_width=True,
            disabled=not control_status.get("process_alive", False),
        ):
            try:
                control.request_stop()
                st.info("Graceful stop requested. Backend reconciliation/protection owns shutdown.")
            except Exception as exc:
                st.error(str(exc))
    with buttons[2]:
        if st.button("EMERGENCY STOP", use_container_width=True, type="primary"):
            try:
                control.emergency_stop()
                st.error("Emergency kill switch armed and backend halt requested.")
            except Exception as exc:
                st.error(str(exc))

    if control_status.get("note"):
        st.caption(f"Operator state: {control_status['note']}")

    with st.expander("Advanced / Research"):
        st.warning("Research controls are intentionally separated from normal operation.")
        st.markdown("[Open Advanced / Research dashboard](/research)")
        st.write(
            "Walk-forward grids, raw Quant Desk internals, lifecycle administration, diagnostics and strategy experiments remain available there."
        )
        st.write(
            "The operator dashboard never weakens research/promotion gates to increase trade count."
        )


def main() -> None:
    control = OperatorControl(CONTROL_DB, engine_db=ENGINE_DB)
    try:
        run = _latest_run()
        control_status = control.status()
        st.sidebar.markdown("### Trade Bot")
        st.sidebar.caption("BYBIT DEMO · Spot only · live disabled")
        page = st.sidebar.radio(
            "Navigation", ["HOME", "MARKETS", "PERFORMANCE", "SETTINGS"], label_visibility="collapsed"
        )
        if st.sidebar.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
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


if __name__ == "__main__":
    main()
