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


def test_dashboard_started_runner_wires_bybit_quality_layer_before_prepare():
    runner = _read("scripts/run_bybit_demo_burnin.py")
    assert "from bybit_entry_intelligence import wire_bybit_quality_layers" in runner
    wire = runner.index("wire_bybit_quality_layers(service)")
    prepare = runner.index("run_id = service.prepare()")
    assert wire < prepare


def test_demo_smoke_has_no_live_selector():
    smoke = _read("scripts/bybit_demo_smoke.py")
    assert "BYBIT_LIVE_API_KEY" not in smoke
    assert "BYBIT_LIVE_API_SECRET" not in smoke
    assert '"--live"' not in smoke
    assert "BYBIT_LIVE" not in smoke


def test_operator_ui_uses_password_fields_and_clears_credential_form():
    ui = _read("bybit_operator_ui.py")
    app = _read("bybit_operator_app.py")
    assert 'type="password"' in ui
    assert 'clear_on_submit=True' in ui
    assert "Test & Save" in ui
    assert "Test saved connection" in ui
    assert "st.session_state" not in ui
    assert "BYBIT_LIVE_API_KEY" not in ui + app
    assert "BYBIT_LIVE_API_SECRET" not in ui + app
    assert "Run Demo economic smoke" in app
    assert "control.demo_smoke_status()" in app
    assert '"Demo smoke": bool(smoke.get("ok"))' in app


def test_operator_child_process_never_accepts_secret_arguments():
    control = _read("bybit_operator_control.py")
    assert '"--api-key"' not in control
    assert '"--api-secret"' not in control
    assert "child_env[\"BYBIT_DEMO_API_KEY\"]" in control
    assert "child_env[\"BYBIT_DEMO_API_SECRET\"]" in control
    assert "json.dump(payload" in control
    assert "0o600" in control
    assert "require_demo_smoke: bool = True" in control
    assert "Bybit Demo economic smoke has not passed" in control
    assert "BYBIT_LIVE_API_KEY" not in control
    assert "BYBIT_LIVE_API_SECRET" not in control


def test_dashboard_saved_secret_volume_is_isolated_from_other_services():
    compose = _read("docker-compose.production.yml")
    dashboard = compose.split("  research-dashboard:", 1)[0]
    remainder = compose.split("  research-dashboard:", 1)[1]
    assert "trade_secrets:/app/secrets" in dashboard
    assert "trade_secrets:/app/secrets" not in remainder
    assert "trade_secrets:" in compose
    ignore = _read(".gitignore")
    assert "secrets/" in ignore


def test_production_dashboards_require_caddy_basic_auth_but_mobile_api_does_not():
    compose = _read("docker-compose.production.yml")
    caddy = _read("Caddyfile")
    env = _read(".env.example")
    assert "TRADE_DASHBOARD_PASSWORD: ${TRADE_DASHBOARD_PASSWORD}" in compose
    assert "caddy hash-password --plaintext" in compose
    assert "TRADE_DASHBOARD_PASSWORD is required" in compose
    assert caddy.count("basic_auth") == 2
    research = caddy.split("handle /research*", 1)[1].split("handle {", 1)[0]
    root = caddy.rsplit("handle {", 1)[1]
    api = caddy.split("handle_path /api/*", 1)[1].split("handle /research*", 1)[0]
    assert "basic_auth" in research
    assert "basic_auth" in root
    assert "basic_auth" not in api
    assert "operator {$TRADE_DASHBOARD_PASSWORD_HASH}" in caddy
    assert "TRADE_DASHBOARD_PASSWORD=" in env


def test_bybit_runtime_contains_no_margin_or_leverage_toggle_endpoint_calls():
    runtime = "\n".join(
        _read(path)
        for path in (
            "bybit_spot.py",
            "bybit_burnin.py",
            "bybit_management.py",
            "bybit_market_data.py",
            "bybit_entry_intelligence.py",
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


def test_production_root_is_acceptance_aware_bybit_dashboard_and_research_is_separate():
    compose = _read("docker-compose.production.yml")
    caddy = _read("Caddyfile")
    assert "bybit_operator_app.py" in compose
    assert "--server.baseUrlPath=research" in compose
    assert "research-dashboard:8502" in caddy
    assert "reverse_proxy dashboard:8501" in caddy


def test_mobile_push_observes_bybit_db_but_never_mounts_dashboard_secret_vault():
    compose = _read("docker-compose.production.yml")
    assert "MOBILE_PUSH_SOURCE_DB_PATH: /app/data/bybit_demo_trading.db" in compose
    assert "MOBILE_PUSH_PREFERENCE_DB_PATH: /app/data/bybit_operator.db" in compose
    mobile = compose.split("  mobile-api:", 1)[1].split("  caddy:", 1)[0]
    assert "/app/secrets" not in mobile


def test_environment_template_warns_against_withdrawal_permission():
    env = _read(".env.example")
    assert "DO NOT grant withdrawal permission" in env
    assert "BYBIT_DEMO_API_KEY=" in env
    assert "BYBIT_DEMO_API_SECRET=" in env
