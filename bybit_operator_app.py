"""Production entrypoint for the simple Bybit Demo operator dashboard.

This composes the existing read-only HOME/MARKETS/PERFORMANCE surfaces with the
acceptance-aware SETTINGS flow. The browser never owns the trading loop; economic
smoke and burn-in processes remain server-side and receive Demo credentials only via
their process environment.
"""
from __future__ import annotations

import bybit_operator_ui as base
import streamlit as st

from bybit_operator_control import OperatorControl


def _smoke_badge(status: dict) -> str:
    value = str(status.get("status") or "NOT_RUN").upper()
    if value == "PASS":
        return "✅ PASS"
    if value == "FAIL":
        return "❌ FAIL"
    if value == "STALE":
        return "⚠️ STALE"
    return "⏳ NOT RUN"


def _render_demo_smoke(
    control: OperatorControl,
    *,
    credential_ready: bool,
    backend_running: bool,
    kill_armed: bool,
) -> dict:
    st.subheader("2 · Verify tiny Demo order path")
    smoke = control.demo_smoke_status()
    cols = st.columns(3)
    cols[0].metric("Demo smoke", _smoke_badge(smoke))
    cols[1].metric("Economic venue", "BYBIT DEMO")
    cols[2].metric("Order mode", "Spot · no leverage")
    st.caption(
        "This is the acceptance smoke required before a 3–5 day run. It buys the "
        "smallest dynamically valid Demo Spot quantity, reconciles it, closes only "
        "proven bot-owned inventory, and must finish with no residual. It never uses Live."
    )
    if smoke.get("detail") and smoke.get("status") in {"FAIL", "STALE"}:
        with st.expander("Last smoke detail"):
            st.code(str(smoke.get("detail"))[-3000:])

    smoke_left, smoke_right = st.columns(2)
    with smoke_left:
        smoke_symbol = st.selectbox(
            "Smoke coin",
            ["BTC", "ETH", "SOL"],
            index=0,
            disabled=backend_running,
            help="Demo only. The smoke opens and closes a tiny Spot position.",
        )
    with smoke_right:
        smoke_cap = st.number_input(
            "Maximum Demo smoke notional (USDT)",
            min_value=5.0,
            max_value=100.0,
            value=25.0,
            step=5.0,
            disabled=backend_running,
        )
    acknowledge = st.checkbox(
        "I understand this will place and close one tiny Bybit Demo Spot order",
        value=False,
        disabled=backend_running,
    )
    can_smoke = (
        credential_ready
        and not backend_running
        and not kill_armed
        and acknowledge
    )
    if st.button(
        "Run Demo economic smoke",
        type="primary",
        use_container_width=True,
        disabled=not can_smoke,
    ):
        try:
            with st.spinner("Running Demo Spot smoke and reconciling residual inventory…"):
                result = control.run_demo_smoke(
                    symbol=f"{smoke_symbol}/USD",
                    max_notional_usdt=float(smoke_cap),
                )
            if result.get("ok"):
                st.success("Demo smoke PASSED. Account, market data, order, execution, close and residual checks succeeded.")
            else:
                st.error("Demo smoke FAILED. The burn-in remains blocked until the smoke passes.")
            with st.expander("Smoke result"):
                st.code(str(result.get("detail") or "No detail"))
        except Exception as exc:
            st.error(f"Demo smoke refused/failed: {exc}")
    return control.demo_smoke_status()


def render_settings(
    run,
    control: OperatorControl,
    control_status: dict,
) -> None:
    st.title("Settings")
    st.caption(
        "Four simple steps: connect Demo API → pass the tiny Demo smoke → choose run settings → start. "
        "LIVE trading is not available here."
    )
    backend_running = bool(control_status.get("process_alive"))
    kill = control_status.get("kill_switch") or {}

    credential_status = base._render_credentials(control, backend_running)

    st.divider()
    smoke = _render_demo_smoke(
        control,
        credential_ready=bool(credential_status.get("configured")),
        backend_running=backend_running,
        kill_armed=bool(kill.get("armed")),
    )

    st.divider()
    st.subheader("3 · Choose run settings")
    lifecycle = base._lifecycle_state()
    eligible = lifecycle in base.DEMO_EXECUTION_STATES
    status = st.columns(4)
    status[0].metric("Credentials", "Ready" if credential_status.get("configured") else "Missing")
    status[1].metric("Demo smoke", _smoke_badge(smoke))
    status[2].metric("Strategy", lifecycle)
    status[3].metric("Backend", "Running" if backend_running else "Stopped")
    if not eligible:
        st.warning(
            "Strategy lifecycle is not yet eligible for broker-backed Demo execution. "
            "Use Advanced / Research to complete the existing evidence gate; this dashboard will not bypass it."
        )

    if kill.get("armed"):
        st.error(
            f"Kill switch ARMED · {kill.get('reason') or 'UNKNOWN'} · "
            f"{kill.get('at') or 'unknown time'}"
        )
        acknowledge_kill = st.checkbox(
            "I reviewed the emergency reason and want to clear the restart latch",
            value=False,
        )
        if st.button(
            "Acknowledge kill switch",
            disabled=not acknowledge_kill or backend_running,
        ):
            try:
                control.acknowledge_kill_switch()
                st.success("Kill switch acknowledged. Re-run the readiness checks before starting.")
            except Exception as exc:
                st.error(str(exc))

    settings_left, settings_right = st.columns(2)
    with settings_left:
        capital = st.number_input(
            "Capital allocation (USDT)",
            min_value=20.0,
            max_value=1_000_000.0,
            value=1_000.0,
            step=100.0,
            disabled=backend_running,
            help="Strategy budget cap only; never sizes automatically from the full Demo wallet.",
        )
        allowed_coins = st.multiselect(
            "Allowed coins",
            ["BTC", "ETH", "SOL"],
            default=["BTC", "ETH", "SOL"],
            disabled=backend_running,
        )
    with settings_right:
        hours = st.radio(
            "Burn-in duration",
            [72, 120],
            horizontal=True,
            format_func=lambda value: f"{value // 24} days",
            disabled=backend_running,
        )
        smart_exits = st.toggle(
            "Enable researched smart-exit consumers",
            value=True,
            disabled=backend_running,
            help="Uses the existing TradeManagementPolicy; it does not tune for a return quota.",
        )
        notifications = st.toggle(
            "Phone push notifications",
            value=control.notifications_enabled(),
            help="Fill/safety notifications only. Trading never depends on push delivery.",
        )
        if notifications != control.notifications_enabled():
            control.set_notifications_enabled(notifications)

    with st.expander("Optional display preferences"):
        egp_rate = st.number_input(
            "EGP reference rate (display only)", min_value=0.0, value=0.0, step=0.1
        )
        if egp_rate > 0:
            st.caption(
                f"Reference only: allocated capital ≈ EGP {capital * egp_rate:,.0f}. "
                "Trading mathematics remains USDT-based."
            )

    st.divider()
    st.subheader("4 · Start / stop")
    checks = {
        "Demo API": bool(credential_status.get("configured")),
        "Demo smoke": bool(smoke.get("ok")),
        "Strategy lifecycle": eligible,
        "Coin universe": bool(allowed_coins),
        "Safety latch clear": not bool(kill.get("armed")),
    }
    st.caption(" · ".join(f"{'✅' if ok else '❌'} {name}" for name, ok in checks.items()))
    can_start = all(checks.values()) and not backend_running
    buttons = st.columns(3)
    with buttons[0]:
        if st.button(
            "Start Bybit Demo",
            type="primary",
            use_container_width=True,
            disabled=not can_start,
        ):
            symbols = tuple(f"{coin}/USD" for coin in allowed_coins)
            try:
                with st.spinner("Starting the detached Demo backend…"):
                    pid = control.start_detached(
                        hours=int(hours),
                        capital_usdt=float(capital),
                        symbols=symbols,
                        smart_exits=bool(smart_exits),
                    )
                st.success(
                    f"Bybit Demo backend started as server process {pid}. "
                    "You can close the browser; the backend keeps running."
                )
            except Exception as exc:
                st.error(f"Start refused: {exc}")
    with buttons[1]:
        if st.button(
            "Stop gracefully",
            use_container_width=True,
            disabled=not backend_running,
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
            "Walk-forward grids, lifecycle administration, raw Quant Desk diagnostics and strategy experiments remain there."
        )
        st.write(
            "The simple dashboard never weakens research/promotion gates to increase trade count."
        )


base.render_settings = render_settings


if __name__ == "__main__":
    base.main()
