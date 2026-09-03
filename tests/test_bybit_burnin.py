from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from bybit_active_report import ActiveSafeBybitBurnInReport
from bybit_burnin import BurnInRiskOverlay, BybitBurnInConfig
from state_store import StateStore


def test_burnin_config_defaults_to_zero_quota_five_entry_ceiling_and_three_symbols():
    config = BybitBurnInConfig()
    assert config.max_entries_rolling_24h == 5
    assert config.universe == ("BTC/USD", "ETH/USD", "SOL/USD")
    assert config.strategy_capital_budget_usdt == pytest.approx(1000.0)
    assert config.validate() == []


def test_config_hash_changes_when_strategy_parameter_changes():
    first = BybitBurnInConfig(hours=72, strategy_capital_budget_usdt=1000)
    second = BybitBurnInConfig(hours=72, strategy_capital_budget_usdt=900)
    assert first.config_hash != second.config_hash


def test_risk_overlay_blocks_rolling_loss_limit(tmp_path):
    store = StateStore(str(tmp_path / "engine.db"))
    now = datetime.now(timezone.utc)
    try:
        store.record_trade(
            "session",
            "BTC/USD",
            "FILL_SELL",
            0.1,
            100.0,
            -21.0,
            "exit",
            None,
            event_key="loss-1",
        )
        overlay = BurnInRiskOverlay(
            store,
            BybitBurnInConfig(
                strategy_capital_budget_usdt=1000,
                rolling_24h_loss_limit_pct=2.0,
            ),
            now_fn=lambda: now,
        )
        allowed, reason = overlay.check_entry(1000)
        assert allowed is False
        assert "rolling_24h_loss_limit" in reason
    finally:
        store.close()


def test_consecutive_loss_cooldown_does_not_create_minimum_trade_quota(tmp_path):
    store = StateStore(str(tmp_path / "engine.db"))
    now = datetime.now(timezone.utc)
    try:
        overlay = BurnInRiskOverlay(
            store,
            BybitBurnInConfig(strategy_capital_budget_usdt=1000),
            now_fn=lambda: now,
        )
        allowed, reason = overlay.check_entry(1000)
        assert allowed is True
        assert reason == "risk_overlay_passed"
    finally:
        store.close()


def _minimal_report_databases(tmp_path):
    venue = tmp_path / "venue.db"
    engine = tmp_path / "engine.db"
    start = datetime.now(timezone.utc) - timedelta(hours=1)
    planned = start + timedelta(hours=120)
    v = sqlite3.connect(venue)
    v.executescript(
        """
        CREATE TABLE burnin_runs(
          run_id TEXT PRIMARY KEY,git_sha TEXT,strategy_version TEXT,config_hash TEXT,
          universe_json TEXT,start_time TEXT,planned_end_time TEXT,environment TEXT,
          status TEXT,restart_count INTEGER,last_heartbeat TEXT,stopped_at TEXT,stop_reason TEXT
        );
        CREATE TABLE burnin_events(
          id INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT,event_key TEXT,event_type TEXT,
          timestamp TEXT,symbol TEXT,strategy TEXT,decision_id TEXT,intent_id TEXT,
          order_link_id TEXT,order_id TEXT,execution_id TEXT,payload_json TEXT
        );
        CREATE TABLE health_samples(
          id INTEGER PRIMARY KEY AUTOINCREMENT,run_id TEXT,timestamp TEXT,provider TEXT,
          health TEXT,public_ws_up INTEGER,private_ws_up INTEGER,public_ws_reconnects INTEGER,
          private_ws_reconnects INTEGER,rest_failures INTEGER,rate_limit_events INTEGER,
          clock_skew_ms REAL,quote_age_ms REAL,spread_bps REAL,pending_intents INTEGER,
          unprotected_qty REAL,payload_json TEXT
        );
        CREATE TABLE venue_orders(
          internal_intent_id TEXT,venue TEXT,order_link_id TEXT,order_id TEXT,
          strategy_symbol TEXT,venue_symbol TEXT,purpose TEXT,attempt INTEGER,
          requested_qty REAL,filled_qty REAL,status TEXT,created_at TEXT,updated_at TEXT
        );
        CREATE TABLE venue_executions(
          execution_id TEXT,internal_intent_id TEXT,order_link_id TEXT,order_id TEXT,
          strategy_symbol TEXT,venue_symbol TEXT,side TEXT,qty REAL,price REAL,fee REAL,
          fee_asset TEXT,execution_at TEXT,raw_hash TEXT
        );
        CREATE TABLE management_state(
          strategy_symbol TEXT PRIMARY KEY,original_qty REAL,remaining_fraction REAL,
          bars_held INTEGER,phase TEXT,partial_taken INTEGER,target_taken INTEGER,
          signal_reason TEXT,entry_price REAL,initial_stop_price REAL,highest_price REAL,
          lowest_price REAL,threshold_025_at TEXT,threshold_050_at TEXT,
          threshold_075_at TEXT,threshold_100_at TEXT,updated_at TEXT
        );
        """
    )
    v.execute(
        "INSERT INTO burnin_runs VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (
            "run-1",
            "abc",
            "2.0.0",
            "cfg",
            json.dumps(["BTC/USD"]),
            start.isoformat(),
            planned.isoformat(),
            "BYBIT_DEMO",
            "running",
            0,
            datetime.now(timezone.utc).isoformat(),
            None,
            None,
        ),
    )
    v.commit()
    v.close()
    StateStore(str(engine)).close()
    return venue, engine, start


def test_active_report_never_counts_future_planned_hours(tmp_path):
    venue, engine, start = _minimal_report_databases(tmp_path)
    report = ActiveSafeBybitBurnInReport(str(venue), str(engine)).build("run-1")
    elapsed = report["run_integrity"]["hours_elapsed"]
    assert elapsed >= 0.9
    assert elapsed < 2.0
    assert report["verdict"]["strategy_proven"] is False
