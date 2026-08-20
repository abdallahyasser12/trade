import pytest

from datetime import datetime, timedelta, timezone

from quant.contracts import StrategyState
from quant.evidence_scanner import (
    CURRENT_RUNG_AUTHORITY_TRADES,
    adaptive_deployment_health,
)


class Lifecycle:
    def __init__(self, state):
        self._state = state

    def state(self, _strategy_id):
        return self._state


class Store:
    def __init__(self, state_since, history, rows):
        self.state_since = state_since
        self.history = history
        self.rows = rows

    def get_strategy_state(self, _strategy_id):
        return {
            "state": StrategyState.PAPER.value,
            "since": self.state_since.isoformat(),
        }

    def get_promotion_history(self, _strategy_id, limit=500):
        return self.history[:limit]

    def get_attributions(self, strategy_id=None, include_shadow=True, limit=1000):
        assert strategy_id == "crypto_intraday_adaptive"
        rows = self.rows
        if not include_shadow:
            rows = [row for row in rows if not row.get("shadow", 0)]
        return rows[:limit]


def row(timestamp, r_value, *, shadow):
    return {
        "recorded_at": timestamp.isoformat(),
        "r_multiple": r_value,
        "net_pnl": r_value * 10.0,
        "shadow": 1 if shadow else 0,
        "regime_at_entry": "RANGE",
    }


def setup_rows(current_real_count=0):
    t0 = datetime(2026, 8, 17, 0, 0, tzinfo=timezone.utc)
    shadow_start = t0 + timedelta(hours=1)
    paper_start = t0 + timedelta(hours=12)
    history = [
        {
            "from_state": "SHADOW",
            "to_state": "PAPER",
            "timestamp": paper_start.isoformat(),
        },
        {
            "from_state": "RESEARCH",
            "to_state": "SHADOW",
            "timestamp": shadow_start.isoformat(),
        },
    ]
    rows = [
        row(shadow_start - timedelta(minutes=1), 50.0, shadow=True),
    ]
    rows.extend(
        row(shadow_start + timedelta(minutes=index + 1), 0.10, shadow=True)
        for index in range(100)
    )
    rows.extend(
        row(paper_start + timedelta(minutes=index + 1), 0.08, shadow=False)
        for index in range(current_real_count)
    )
    # QuantStore returns newest first.
    rows.sort(key=lambda item: item["recorded_at"], reverse=True)
    return paper_start, history, rows


def test_new_paper_rung_bridges_only_immediately_prior_shadow_evidence():
    paper_start, history, rows = setup_rows(current_real_count=0)
    health, source = adaptive_deployment_health(
        Store(paper_start, history, rows),
        Lifecycle(StrategyState.PAPER),
        "crypto_intraday_adaptive",
    )
    assert source == "bridge_shadow_to_paper"
    assert health.trades == 100
    # The deliberately huge attribution before SHADOW began must not leak in.
    assert health.expectancy_r == pytest.approx(0.10)


def test_current_rung_becomes_authoritative_after_40_real_trades():
    paper_start, history, rows = setup_rows(
        current_real_count=CURRENT_RUNG_AUTHORITY_TRADES
    )
    health, source = adaptive_deployment_health(
        Store(paper_start, history, rows),
        Lifecycle(StrategyState.PAPER),
        "crypto_intraday_adaptive",
    )
    assert source == "current_rung"
    assert health.trades == CURRENT_RUNG_AUTHORITY_TRADES
    assert health.expectancy_r == pytest.approx(0.08)
