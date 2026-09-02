from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_demo_runner_has_no_live_selector_or_live_credential_reader():
    runner = _read("scripts/run_bybit_demo_burnin.py")
    assert "BYBIT_LIVE_API_KEY" not in runner
    assert "BYBIT_LIVE_API_SECRET" not in runner
    assert '"--live"' not in runner
    assert "BYBIT_LIVE" not in runner


def test_demo_smoke_has_no_live_selector():
    smoke = _read("scripts/bybit_demo_smoke.py")
    assert "BYBIT_LIVE_API_KEY" not in smoke
    assert "BYBIT_LIVE_API_SECRET" not in smoke
    assert '"--live"' not in smoke
    assert "BYBIT_LIVE" not in smoke


def test_operator_ui_never_reads_or_displays_secret_value():
    ui = _read("bybit_operator_ui.py")
    # Presence checks may use the environment variable names; the value must never
    # be copied into Streamlit state or rendered.
    assert "BYBIT_DEMO_API_SECRET" in ui
    assert "os.getenv(\"BYBIT_DEMO_API_SECRET\")" in ui
    assert "st.session_state" not in ui
    assert "api_secret =" not in ui
    assert "api_key =" not in ui


def test_operator_child_process_never_accepts_secret_arguments():
    control = _read("bybit_operator_control.py")
    assert "BYBIT_DEMO_API_KEY" not in control
    assert "BYBIT_DEMO_API_SECRET" not in control
    assert '"--api-key"' not in control
    assert '"--api-secret"' not in control


def test_bybit_runtime_contains_no_margin_or_leverage_toggle_endpoint_calls():
    runtime = "\n".join(
        _read(path)
        for path in (
            "bybit_spot.py",
            "bybit_burnin.py",
            "bybit_management.py",
            "bybit_market_data.py",
        )
    )
    banned_calls = (
        ".set_leverage(",
        ".set_spot_margin_mode(",
        ".toggle_margin_trade(",
        ".borrow(",
        ".repay(",
    )
    for call in banned_calls:
        assert call not in runtime


def test_production_root_is_simple_bybit_dashboard_and_research_is_separate():
    compose = _read("docker-compose.production.yml")
    caddy = _read("Caddyfile")
    assert "bybit_operator_ui.py" in compose
    assert "--server.baseUrlPath=research" in compose
    assert "research-dashboard:8502" in caddy
    assert "reverse_proxy dashboard:8501" in caddy


def test_environment_template_warns_against_withdrawal_permission():
    env = _read(".env.example")
    assert "DO NOT grant withdrawal permission" in env
    assert "BYBIT_DEMO_API_KEY=" in env
    assert "BYBIT_DEMO_API_SECRET=" in env
