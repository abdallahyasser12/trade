from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

import bybit_operator_control as module
from bybit_operator_control import OperatorControl


class _Process:
    pid = 424242


def _runner_root(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    scripts = root / "scripts"
    scripts.mkdir(parents=True)
    (scripts / "run_bybit_demo_burnin.py").write_text("print('stub')\n", encoding="utf-8")
    return root


def test_dashboard_start_freezes_selected_universe_without_secrets(tmp_path, monkeypatch):
    root = _runner_root(tmp_path)
    captured = {}

    def fake_popen(command, **kwargs):
        captured["command"] = list(command)
        captured["kwargs"] = kwargs
        return _Process()

    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    control = OperatorControl(
        str(tmp_path / "operator.db"),
        engine_db=str(tmp_path / "engine.db"),
        venue_db=str(tmp_path / "venue.db"),
    )
    try:
        pid = control.start_detached(
            hours=72,
            capital_usdt=750,
            symbols=("BTC/USD", "SOL/USD"),
            repo_root=str(root),
            log_path=str(tmp_path / "bot.log"),
        )
        assert pid == _Process.pid
        command = captured["command"]
        assert "--hours" in command and "72" in command
        assert "--capital-usdt" in command and "750.0" in command
        assert "BTC/USD" in command and "SOL/USD" in command
        assert "ETH/USD" not in command
        joined = " ".join(command)
        assert "BYBIT_DEMO_API_KEY" not in joined
        assert "BYBIT_DEMO_API_SECRET" not in joined
        assert "live" not in joined.lower()
        assert captured["kwargs"]["start_new_session"] is True
    finally:
        control.close()


def test_emergency_latch_blocks_restart_until_explicit_ack(tmp_path, monkeypatch):
    root = _runner_root(tmp_path)
    monkeypatch.setattr(OperatorControl, "_owned_backend_alive", classmethod(lambda cls, pid: False))
    control = OperatorControl(
        str(tmp_path / "operator.db"),
        engine_db=str(tmp_path / "engine.db"),
        venue_db=str(tmp_path / "venue.db"),
    )
    try:
        control.emergency_stop()
        kill = control.read_kill_switch()
        assert kill and kill["armed"] is True
        assert kill["reason"] == "EMERGENCY_STOP"
        with pytest.raises(RuntimeError, match="kill switch is armed"):
            control.start_detached(
                hours=72,
                capital_usdt=100,
                symbols=("BTC/USD",),
                repo_root=str(root),
            )
        control.acknowledge_kill_switch()
        assert control.read_kill_switch() is None
    finally:
        control.close()


def test_corrupt_kill_switch_fails_closed(tmp_path):
    engine = tmp_path / "engine.db"
    connection = sqlite3.connect(engine)
    connection.execute("CREATE TABLE kv(key TEXT PRIMARY KEY,value TEXT)")
    connection.execute("INSERT INTO kv VALUES('crypto_kill_switch','not-json')")
    connection.commit()
    connection.close()
    control = OperatorControl(
        str(tmp_path / "operator.db"),
        engine_db=str(engine),
        venue_db=str(tmp_path / "venue.db"),
    )
    try:
        assert control.read_kill_switch()["armed"] is True
    finally:
        control.close()


def test_dashboard_projection_uses_engine_authoritative_current_stop(tmp_path):
    engine = tmp_path / "engine.db"
    venue = tmp_path / "venue.db"
    e = sqlite3.connect(engine)
    e.execute("CREATE TABLE kv(key TEXT PRIMARY KEY,value TEXT)")
    e.execute(
        "INSERT INTO kv VALUES(?,?)",
        ("bot_position:BTC/USD", json.dumps({"stop_price": 101.25})),
    )
    e.commit()
    e.close()
    v = sqlite3.connect(venue)
    v.execute(
        """CREATE TABLE management_state(
        strategy_symbol TEXT PRIMARY KEY, initial_stop_price REAL, updated_at TEXT
        )"""
    )
    v.execute("INSERT INTO management_state VALUES('BTC/USD',95.0,'old')")
    v.commit()
    v.close()

    control = OperatorControl(
        str(tmp_path / "operator.db"), engine_db=str(engine), venue_db=str(venue)
    )
    try:
        control.status()
    finally:
        control.close()
    v = sqlite3.connect(venue)
    try:
        value = v.execute(
            "SELECT current_stop_price FROM management_state WHERE strategy_symbol='BTC/USD'"
        ).fetchone()[0]
        assert value == pytest.approx(101.25)
    finally:
        v.close()
