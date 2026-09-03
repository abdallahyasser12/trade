from __future__ import annotations

import json
from types import SimpleNamespace

import pandas as pd
import pytest

from bybit_entry_intelligence import (
    BybitEntryQualityConfig,
    BybitEvidenceQuantScanner,
    TargetAwareBybitTradeManagementExecutor,
    _target_plan,
)
from bybit_management import BybitTradeManagementExecutor
from quant.evidence_scanner import EvidenceQuantScanner
from quant.exits import ExitPolicyConfig, TradeManagementPolicy


def _bars(prior_high: float = 104.0) -> pd.DataFrame:
    index = pd.date_range("2026-01-01", periods=7, freq="5min", tz="UTC")
    close = [96.0, 97.0, 98.0, 99.0, 99.2, 99.5, 100.0]
    high = [max(value + 0.4, prior_high if i == 4 else 0.0) for i, value in enumerate(close)]
    low = [value - 0.5 for value in close]
    return pd.DataFrame({"high": high, "low": low, "close": close}, index=index)


def test_target_plan_uses_structure_atr_r_and_cost_room_not_fixed_percent():
    frame = _bars(104.0)
    plan = _target_plan(
        frame,
        frame.index[-1],
        entry_price=100.0,
        stop_price=98.0,
        atr=2.0,
        signal_reason="adaptive_pullback_entry",
        round_trip_cost_bps=10.0,
        config=BybitEntryQualityConfig(resistance_lookback=5),
    )
    assert plan["tradeable"] is True
    assert plan["selected_family"] in {"atr", "r", "structure"}
    assert plan["target_price"] == pytest.approx(102.5)
    assert plan["target_r"] == pytest.approx(1.25)
    assert plan["net_target_r_after_costs"] < plan["target_r"]
    assert plan["research_family_prices"]["0.50pct"] == pytest.approx(100.5)
    assert plan["target_price"] != plan["research_family_prices"]["0.50pct"]


def test_target_plan_rejects_when_nearby_resistance_leaves_insufficient_room():
    frame = _bars(100.4)
    plan = _target_plan(
        frame,
        frame.index[-1],
        entry_price=100.0,
        stop_price=98.0,
        atr=2.0,
        signal_reason="adaptive_pullback_entry",
        round_trip_cost_bps=10.0,
        config=BybitEntryQualityConfig(resistance_lookback=5),
    )
    assert plan["tradeable"] is False
    assert plan["selected_family"] == "structure"
    assert plan["reason"] == "insufficient_target_room_after_costs"
    assert plan["target_r"] < 1.0


def test_scanner_freezes_only_executable_target_and_clears_rejected_stale_plan(monkeypatch):
    good_plan = {
        "tradeable": True,
        "target_r": 1.25,
        "target_fraction": 0.5,
        "target_price": 102.5,
    }
    rejected_plan = {
        "tradeable": True,
        "target_r": 1.1,
        "target_fraction": 0.5,
        "target_price": 51.0,
    }
    btc_candidate = SimpleNamespace(
        symbol="BTC/USD",
        timestamp=pd.Timestamp("2026-01-01T00:00:00Z"),
        strategy_id="crypto_intraday_adaptive",
        signal_reason="adaptive_pullback_entry",
        details={"bybit_target_plan": good_plan},
    )
    eth_candidate = SimpleNamespace(
        symbol="ETH/USD",
        timestamp=pd.Timestamp("2026-01-01T00:00:00Z"),
        strategy_id="crypto_intraday_adaptive",
        signal_reason="adaptive_momentum_entry",
        details={"bybit_target_plan": rejected_plan},
    )
    executable = SimpleNamespace(candidate=btc_candidate, risk_allocation_pct=0.15)
    rejected = SimpleNamespace(candidate=eth_candidate, risk_allocation_pct=0.0)
    fake_report = SimpleNamespace(
        executable=[executable],
        decisions=[executable, rejected],
    )

    class Store:
        def __init__(self):
            self.values = {"bybit_target_plan:ETH/USD": "stale"}

        def set_kv(self, key, value):
            self.values[key] = value

        def delete_kv(self, key):
            self.values.pop(key, None)

    store = Store()
    monkeypatch.setattr(EvidenceQuantScanner, "scan", lambda self, *args, **kwargs: fake_report)
    scanner = BybitEvidenceQuantScanner.__new__(BybitEvidenceQuantScanner)
    scanner.target_plan_store = store

    returned = scanner.scan()
    assert returned is fake_report
    frozen = json.loads(store.values["bybit_target_plan:BTC/USD"])
    assert frozen["frozen_for_execution"] is True
    assert frozen["target_price"] == pytest.approx(102.5)
    assert frozen["risk_allocation_pct"] == pytest.approx(0.15)
    assert "bybit_target_plan:ETH/USD" not in store.values


def test_target_aware_executor_feeds_frozen_plan_into_existing_policy_only(monkeypatch):
    plan = {
        "tradeable": True,
        "frozen_for_execution": True,
        "target_r": 1.4,
        "target_fraction": 0.5,
    }

    class Store:
        def get_kv(self, key):
            assert key == "bybit_target_plan:BTC/USD"
            return json.dumps(plan)

    observed = {}

    def fake_parent_manage(self, symbol, frame, timestamp):
        observed["symbol"] = symbol
        observed["enabled"] = self.policy.config.enable_profit_target
        observed["target_r"] = self.policy.config.profit_target_r
        observed["fraction"] = self.policy.config.profit_target_fraction
        return []

    monkeypatch.setattr(BybitTradeManagementExecutor, "manage_symbol", fake_parent_manage)
    original = TradeManagementPolicy(ExitPolicyConfig())
    executor = TargetAwareBybitTradeManagementExecutor(
        SimpleNamespace(store=Store()),
        None,
        None,
        original,
        strategy_id="crypto_intraday_adaptive",
    )
    result = executor.manage_symbol(
        "BTC/USD", pd.DataFrame(), pd.Timestamp("2026-01-01", tz="UTC")
    )
    assert result == []
    assert observed == {
        "symbol": "BTC/USD",
        "enabled": True,
        "target_r": pytest.approx(1.4),
        "fraction": pytest.approx(0.5),
    }
    assert executor.policy is original
