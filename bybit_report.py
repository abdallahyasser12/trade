"""Evidence-backed Bybit Demo burn-in reporting.

The report intentionally separates operational acceptance from preliminary edge
observations.  A 3–5 day run can never produce a "strategy proven" verdict here.
"""
from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any, Optional


def _dt(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    try:
        result = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    if result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    return result.astimezone(timezone.utc)


def _f(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _pct(values: list[float], p: float) -> Optional[float]:
    clean = sorted(v for v in values if math.isfinite(v))
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    rank = (len(clean) - 1) * p
    low = int(math.floor(rank))
    high = int(math.ceil(rank))
    if low == high:
        return clean[low]
    weight = rank - low
    return clean[low] * (1 - weight) + clean[high] * weight


def _stats(values: list[float]) -> dict[str, Optional[float]]:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    if not clean:
        return {"count": 0, "min": None, "p50": None, "p95": None, "max": None, "mean": None}
    return {
        "count": len(clean),
        "min": min(clean),
        "p50": _pct(clean, 0.50),
        "p95": _pct(clean, 0.95),
        "max": max(clean),
        "mean": mean(clean),
    }


def _rows(path: str, sql: str, params: tuple = ()) -> list[dict[str, Any]]:
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    try:
        return [dict(row) for row in con.execute(sql, params).fetchall()]
    finally:
        con.close()


def _one(path: str, sql: str, params: tuple = ()) -> Optional[dict[str, Any]]:
    rows = _rows(path, sql, params)
    return rows[0] if rows else None


@dataclass
class Episode:
    symbol: str
    entry_time: datetime
    exit_time: Optional[datetime]
    entry_price: float
    exit_price: Optional[float]
    original_qty: float
    gross_pnl: float
    fees_quote_estimate: float
    net_pnl: float
    holding_seconds: Optional[float]
    mfe_pct: Optional[float] = None
    mae_pct: Optional[float] = None
    reached_025: bool = False
    reached_050: bool = False
    reached_075: bool = False
    reached_100: bool = False
    giveback_pct_points: Optional[float] = None
    exit_reason: Optional[str] = None
    setup_score: Optional[float] = None
    regime: Optional[str] = None
    strategy_family: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "entry_time": self.entry_time.isoformat(),
            "exit_time": self.exit_time.isoformat() if self.exit_time else None,
            "entry_price": self.entry_price,
            "exit_price": self.exit_price,
            "original_qty": self.original_qty,
            "gross_pnl": self.gross_pnl,
            "fees_quote_estimate": self.fees_quote_estimate,
            "net_pnl": self.net_pnl,
            "holding_seconds": self.holding_seconds,
            "mfe_pct": self.mfe_pct,
            "mae_pct": self.mae_pct,
            "reached_025": self.reached_025,
            "reached_050": self.reached_050,
            "reached_075": self.reached_075,
            "reached_100": self.reached_100,
            "giveback_pct_points": self.giveback_pct_points,
            "exit_reason": self.exit_reason,
            "setup_score": self.setup_score,
            "regime": self.regime,
            "strategy_family": self.strategy_family,
        }


class BybitBurnInReport:
    def __init__(self, venue_db: str, engine_db: str) -> None:
        self.venue_db = venue_db
        self.engine_db = engine_db

    def _run(self, run_id: str) -> dict[str, Any]:
        row = _one(self.venue_db, "SELECT * FROM burnin_runs WHERE run_id=?", (run_id,))
        if row is None:
            raise ValueError(f"Unknown Bybit burn-in run {run_id!r}")
        return row

    def _events(self, run_id: str) -> list[dict[str, Any]]:
        rows = _rows(
            self.venue_db,
            "SELECT * FROM burnin_events WHERE run_id=? ORDER BY timestamp,id",
            (run_id,),
        )
        for row in rows:
            try:
                row["payload"] = json.loads(row.pop("payload_json"))
            except Exception:
                row["payload"] = {}
        return rows

    def _window(self, run: dict[str, Any]) -> tuple[datetime, datetime]:
        start = _dt(run["start_time"])
        if start is None:
            raise ValueError("Burn-in start time is invalid")
        end = _dt(run.get("stopped_at")) or _dt(run.get("planned_end_time")) or datetime.now(timezone.utc)
        return start, end

    def _executions(self, start: datetime, end: datetime) -> list[dict[str, Any]]:
        return _rows(
            self.venue_db,
            """SELECT e.*,o.purpose,o.created_at AS order_created_at
               FROM venue_executions e
               JOIN venue_orders o ON o.internal_intent_id=e.internal_intent_id
               WHERE e.execution_at>=? AND e.execution_at<=?
               ORDER BY e.execution_at,e.execution_id""",
            (start.isoformat(), end.isoformat()),
        )

    def _decisions(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [e for e in events if e["event_type"] == "decision"]

    def _management(self, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [e for e in events if e["event_type"] == "position_management"]

    def _episodes(
        self,
        executions: list[dict[str, Any]],
        decisions: list[dict[str, Any]],
        management: list[dict[str, Any]],
    ) -> list[Episode]:
        by_symbol: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in executions:
            by_symbol[row["strategy_symbol"]].append(row)
        decision_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in decisions:
            decision_rows[event.get("symbol")].append(event)
        management_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for event in management:
            management_rows[event.get("symbol")].append(event)

        episodes: list[Episode] = []
        for symbol, rows in by_symbol.items():
            qty = 0.0
            cost = 0.0
            entry_time: Optional[datetime] = None
            entry_qty = 0.0
            total_gross = 0.0
            total_fees = 0.0
            last_exit_price: Optional[float] = None
            last_reason: Optional[str] = None
            for row in rows:
                side = str(row["side"]).lower()
                row_qty = _f(row["qty"])
                price = _f(row["price"])
                fee = max(0.0, _f(row.get("fee")))
                fee_asset = str(row.get("fee_asset") or "").upper()
                base = symbol.split("/")[0].upper()
                fee_quote = fee * price if fee_asset == base else fee
                when = _dt(row["execution_at"])
                if when is None or row_qty <= 0 or price <= 0:
                    continue
                if side == "buy":
                    if qty <= 1e-12:
                        entry_time = when
                        entry_qty = 0.0
                        total_gross = 0.0
                        total_fees = 0.0
                        last_exit_price = None
                        last_reason = None
                    acquired = max(0.0, row_qty - (fee if fee_asset == base else 0.0))
                    qty += acquired
                    cost += row_qty * price + (fee if fee_asset and fee_asset != base else 0.0)
                    entry_qty += acquired
                    total_fees += fee_quote
                    continue
                if side != "sell" or qty <= 1e-12:
                    continue
                sold = min(qty, row_qty + (fee if fee_asset == base else 0.0))
                avg_cost = cost / qty if qty > 0 else 0.0
                removed_cost = avg_cost * sold
                gross = row_qty * price - removed_cost
                total_gross += gross
                total_fees += fee_quote
                qty = max(0.0, qty - sold)
                cost = max(0.0, cost - removed_cost)
                last_exit_price = price
                last_reason = str(row.get("purpose") or "exit")
                if qty <= 1e-10 and entry_time is not None:
                    avg_entry = (removed_cost / sold) if sold > 0 else 0.0
                    # Weighted episode entry is all buy cost divided by original qty;
                    # reconstruct from realized gross/proceeds when possible.
                    proceeds = total_gross + (avg_cost * sold)
                    _ = proceeds  # documents intent; episode entry below uses decision/order data fallback.
                    episode = Episode(
                        symbol=symbol,
                        entry_time=entry_time,
                        exit_time=when,
                        entry_price=avg_cost if avg_cost > 0 else avg_entry,
                        exit_price=last_exit_price,
                        original_qty=entry_qty,
                        gross_pnl=total_gross,
                        fees_quote_estimate=total_fees,
                        net_pnl=total_gross - total_fees,
                        holding_seconds=(when - entry_time).total_seconds(),
                        exit_reason=last_reason,
                    )
                    episodes.append(episode)
                    qty = 0.0
                    cost = 0.0
                    entry_time = None
                    entry_qty = 0.0

        # Attach setup context and excursion evidence without peeking past each exit.
        for episode in episodes:
            eligible = [
                event for event in decision_rows.get(episode.symbol, [])
                if (_dt(event["timestamp"]) or episode.entry_time) <= episode.entry_time
                and str((event.get("payload") or {}).get("action")) == "BUY"
            ]
            if eligible:
                payload = eligible[-1].get("payload") or {}
                episode.setup_score = _f(payload.get("score"), math.nan)
                if not math.isfinite(episode.setup_score or math.nan):
                    episode.setup_score = None
                episode.regime = payload.get("regime")
                details = payload.get("details") or {}
                sleeve_scores = {
                    "momentum": _f(details.get("momentum_sleeve"), -math.inf),
                    "breakout": _f(details.get("breakout_sleeve"), -math.inf),
                    "pullback": _f(details.get("pullback_sleeve"), -math.inf),
                    "mean_reversion": _f(details.get("reversion_sleeve"), -math.inf),
                }
                episode.strategy_family = max(sleeve_scores, key=sleeve_scores.get)
            highest = episode.entry_price
            lowest = episode.entry_price
            thresholds = {"threshold_025_at": False, "threshold_050_at": False, "threshold_075_at": False, "threshold_100_at": False}
            for event in management_rows.get(episode.symbol, []):
                ts = _dt(event["timestamp"])
                if ts is None or ts < episode.entry_time or (episode.exit_time and ts > episode.exit_time):
                    continue
                payload = event.get("payload") or {}
                for state in (payload.get("before"), payload.get("after")):
                    if not isinstance(state, dict):
                        continue
                    highest = max(highest, _f(state.get("highest_price"), highest))
                    low_candidate = _f(state.get("lowest_price"), lowest)
                    if low_candidate > 0:
                        lowest = min(lowest, low_candidate)
                    for key in thresholds:
                        thresholds[key] = thresholds[key] or bool(state.get(key))
            if episode.entry_price > 0:
                episode.mfe_pct = (highest / episode.entry_price - 1.0) * 100.0
                episode.mae_pct = (lowest / episode.entry_price - 1.0) * 100.0
                if episode.exit_price is not None:
                    exit_return = (episode.exit_price / episode.entry_price - 1.0) * 100.0
                    episode.giveback_pct_points = max(0.0, (episode.mfe_pct or 0.0) - exit_return)
            episode.reached_025 = thresholds["threshold_025_at"] or (episode.mfe_pct or 0.0) >= 0.25
            episode.reached_050 = thresholds["threshold_050_at"] or (episode.mfe_pct or 0.0) >= 0.50
            episode.reached_075 = thresholds["threshold_075_at"] or (episode.mfe_pct or 0.0) >= 0.75
            episode.reached_100 = thresholds["threshold_100_at"] or (episode.mfe_pct or 0.0) >= 1.00
        return episodes

    @staticmethod
    def _performance(episodes: list[Episode], capital: float) -> dict[str, Any]:
        closed = [e for e in episodes if e.exit_time is not None]
        wins = [e for e in closed if e.net_pnl > 1e-9]
        losses = [e for e in closed if e.net_pnl < -1e-9]
        breakeven = len(closed) - len(wins) - len(losses)
        gross_profit = sum(e.net_pnl for e in wins)
        gross_loss = abs(sum(e.net_pnl for e in losses))
        net = sum(e.net_pnl for e in closed)
        gross = sum(e.gross_pnl for e in closed)
        fees = sum(e.fees_quote_estimate for e in closed)
        avg_win = mean([e.net_pnl for e in wins]) if wins else None
        avg_loss = mean([e.net_pnl for e in losses]) if losses else None
        expectancy = mean([e.net_pnl for e in closed]) if closed else None
        return {
            "total_trades": len(closed),
            "wins": len(wins),
            "losses": len(losses),
            "break_even": breakeven,
            "win_rate": len(wins) / len(closed) if closed else None,
            "average_winner_usdt": avg_win,
            "average_loser_usdt": avg_loss,
            "expectancy_usdt": expectancy,
            "profit_factor": (gross_profit / gross_loss if gross_loss > 0 else (math.inf if gross_profit > 0 else None)),
            "net_pnl_usdt": net,
            "gross_pnl_usdt": gross,
            "net_return_pct": (net / capital * 100.0 if capital > 0 else None),
            "gross_return_pct": (gross / capital * 100.0 if capital > 0 else None),
            "fees_quote_estimate": fees,
            "average_holding_seconds": mean([e.holding_seconds for e in closed if e.holding_seconds is not None]) if closed else None,
        }

    @staticmethod
    def _equity_drawdown(episodes: list[Episode], capital: float) -> float:
        equity = capital
        peak = capital
        max_dd = 0.0
        for episode in sorted((e for e in episodes if e.exit_time), key=lambda e: e.exit_time):
            equity += episode.net_pnl
            peak = max(peak, equity)
            if peak > 0:
                max_dd = max(max_dd, (peak - equity) / peak * 100.0)
        return max_dd

    @staticmethod
    def _group_results(episodes: list[Episode], field: str) -> dict[str, dict[str, Any]]:
        groups: dict[str, list[Episode]] = defaultdict(list)
        for episode in episodes:
            key = getattr(episode, field) or "UNKNOWN"
            groups[str(key)].append(episode)
        result = {}
        for key, rows in sorted(groups.items()):
            pnl = [row.net_pnl for row in rows]
            result[key] = {
                "trades": len(rows),
                "net_pnl_usdt": sum(pnl),
                "win_rate": sum(1 for value in pnl if value > 0) / len(pnl) if pnl else None,
                "avg_net_pnl_usdt": mean(pnl) if pnl else None,
            }
        return result

    @staticmethod
    def _score_buckets(episodes: list[Episode]) -> dict[str, dict[str, Any]]:
        buckets = {"<40": [], "40-59": [], "60-79": [], "80+": [], "UNKNOWN": []}
        for episode in episodes:
            score = episode.setup_score
            if score is None:
                buckets["UNKNOWN"].append(episode)
            elif score < 40:
                buckets["<40"].append(episode)
            elif score < 60:
                buckets["40-59"].append(episode)
            elif score < 80:
                buckets["60-79"].append(episode)
            else:
                buckets["80+"].append(episode)
        return {
            key: {
                "trades": len(rows),
                "net_pnl_usdt": sum(e.net_pnl for e in rows),
                "win_rate": sum(1 for e in rows if e.net_pnl > 0) / len(rows) if rows else None,
            }
            for key, rows in buckets.items()
        }

    def build(self, run_id: str) -> dict[str, Any]:
        run = self._run(run_id)
        start, end = self._window(run)
        events = self._events(run_id)
        executions = self._executions(start, end)
        decisions = self._decisions(events)
        management = self._management(events)
        episodes = self._episodes(executions, decisions, management)
        start_event = next((e for e in events if e["event_type"] in {"run_start", "run_resume"}), None)
        config = ((start_event or {}).get("payload") or {}).get("config") or {}
        capital = _f(config.get("strategy_capital_budget_usdt"), 1_000.0)
        perf = self._performance(episodes, capital)
        perf["maximum_drawdown_pct"] = self._equity_drawdown(episodes, capital)
        elapsed_hours = max(0.0, (end - start).total_seconds() / 3600.0)
        expected_hours = max(0.0, (_dt(run["planned_end_time"]) - start).total_seconds() / 3600.0) if _dt(run["planned_end_time"]) else elapsed_hours
        perf["trades_per_day"] = perf["total_trades"] / max(elapsed_hours / 24.0, 1 / 24.0)

        health = _rows(
            self.venue_db,
            "SELECT * FROM health_samples WHERE run_id=? ORDER BY timestamp",
            (run_id,),
        )
        health_states = [str(row["health"]) for row in health]
        quote_ages = [_f(row.get("quote_age_ms"), math.nan) for row in health]
        spreads = [_f(row.get("spread_bps"), math.nan) for row in health]
        clock_skews = [abs(_f(row.get("clock_skew_ms"), math.nan)) for row in health]
        public_reconnects = max([int(row.get("public_ws_reconnects") or 0) for row in health] or [0])
        private_reconnects = max([int(row.get("private_ws_reconnects") or 0) for row in health] or [0])
        rest_failures = max([int(row.get("rest_failures") or 0) for row in health] or [0])
        rate_limits = max([int(row.get("rate_limit_events") or 0) for row in health] or [0])
        unprotected = sum(1 for row in health if _f(row.get("unprotected_qty")) > 1e-12)

        order_rows = _rows(
            self.venue_db,
            "SELECT * FROM venue_orders WHERE created_at>=? AND created_at<=? ORDER BY created_at",
            (start.isoformat(), end.isoformat()),
        )
        order_latencies = []
        for row in order_rows:
            created, updated = _dt(row.get("created_at")), _dt(row.get("updated_at"))
            if created and updated:
                order_latencies.append(max(0.0, (updated - created).total_seconds() * 1000.0))
        execution_latencies = []
        order_created = {row["internal_intent_id"]: _dt(row.get("created_at")) for row in order_rows}
        for execution in executions:
            created = order_created.get(execution["internal_intent_id"])
            executed = _dt(execution["execution_at"])
            if created and executed:
                execution_latencies.append(max(0.0, (executed - created).total_seconds() * 1000.0))

        engine_executions = _rows(
            self.engine_db,
            "SELECT * FROM executions WHERE timestamp>=? AND timestamp<=? ORDER BY timestamp",
            (start.isoformat(), end.isoformat()),
        )
        slippage = [_f(row.get("slippage_bps"), math.nan) for row in engine_executions]

        closed = [episode for episode in episodes if episode.exit_time]
        reach = lambda attr: (sum(1 for e in closed if getattr(e, attr)) / len(closed) if closed else None)
        givebacks = [e.giveback_pct_points for e in closed if e.giveback_pct_points is not None]
        mfe_values = [e.mfe_pct for e in closed if e.mfe_pct is not None]
        mae_values = [e.mae_pct for e in closed if e.mae_pct is not None]

        partial_events = 0
        for event in management:
            for instruction in (event.get("payload") or {}).get("instructions") or []:
                if instruction.get("kind") == "PARTIAL_EXIT":
                    partial_events += 1

        crashes = sum(1 for e in events if e["event_type"] in {"service_exception", "engine_error"})
        duplicate_order_incidents = sum(1 for e in events if e["event_type"] == "duplicate_order_incident")
        reconciliation_incidents = sum(1 for e in events if "reconciliation" in e["event_type"])
        operational_accept = (
            run.get("status") not in {"failed"}
            and elapsed_hours >= expected_hours * 0.95
            and crashes == 0
            and unprotected == 0
            and duplicate_order_incidents == 0
            and not any(state == "HALTED" for state in health_states)
        )
        operational_verdict = "OPERATIONAL_ACCEPT" if operational_accept else "OPERATIONAL_REJECT"

        if len(closed) < 10:
            edge_verdict = "EDGE_INCONCLUSIVE"
        elif (perf["expectancy_usdt"] or 0.0) > 0 and (perf["net_pnl_usdt"] or 0.0) > 0 and (perf["profit_factor"] or 0.0) > 1.0:
            edge_verdict = "EDGE_PRELIMINARY_POSITIVE"
        else:
            edge_verdict = "EDGE_PRELIMINARY_NEGATIVE"

        report = {
            "run_id": run_id,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "disclaimer": "3–5 day Demo evidence is operational/preliminary and does not prove profitability or a durable edge.",
            "run_integrity": {
                "hours_expected": expected_hours,
                "hours_elapsed": elapsed_hours,
                "code_sha": run["git_sha"],
                "strategy_version": run["strategy_version"],
                "config_hash": run["config_hash"],
                "universe": json.loads(run["universe_json"]),
                "environment": run["environment"],
                "restarts": int(run.get("restart_count") or 0),
                "run_status": run.get("status"),
                "stop_reason": run.get("stop_reason"),
                "crashes": crashes,
                "health_samples": len(health),
            },
            "scanning": {
                "decisions_recorded": len(decisions),
                "buy_setups": sum(1 for e in decisions if str((e.get("payload") or {}).get("action")) == "BUY"),
                "no_trade_or_watch": sum(1 for e in decisions if str((e.get("payload") or {}).get("action")) != "BUY"),
                "minimum_required_trades": 0,
                "maximum_entries_rolling_24h": config.get("max_entries_rolling_24h", 5),
            },
            "orders": {
                "intents": len(order_rows),
                "entries": sum(1 for row in order_rows if row["purpose"] == "entry"),
                "protective": sum(1 for row in order_rows if row["purpose"] == "stop"),
                "partial_exit_intents": sum(1 for row in order_rows if row["purpose"] == "partial"),
                "rejected": sum(1 for row in order_rows if row["status"] == "rejected"),
                "partial_fills": sum(1 for row in order_rows if row["status"] == "partially_filled"),
                "duplicate_intent_incidents": duplicate_order_incidents,
            },
            "trading": {**perf, "episodes": [episode.to_dict() for episode in episodes]},
            "small_move_hypothesis": {
                "reach_0_25_pct": reach("reached_025"),
                "reach_0_50_pct": reach("reached_050"),
                "reach_0_75_pct": reach("reached_075"),
                "reach_1_00_pct": reach("reached_100"),
                "mfe_pct_distribution": _stats(mfe_values),
                "mae_pct_distribution": _stats(mae_values),
                "giveback_pct_points_distribution": _stats(givebacks),
                "partial_profit_events": partial_events,
                "by_symbol": self._group_results(closed, "symbol"),
                "by_strategy_family": self._group_results(closed, "strategy_family"),
                "by_regime": self._group_results(closed, "regime"),
                "by_setup_score_bucket": self._score_buckets(closed),
            },
            "execution_quality": {
                "quote_age_ms": _stats(quote_ages),
                "spread_bps": _stats(spreads),
                "clock_skew_abs_ms": _stats(clock_skews),
                "order_ack_latency_ms": _stats(order_latencies),
                "execution_latency_ms": _stats(execution_latencies),
                "slippage_bps": _stats(slippage),
                "public_ws_reconnects": public_reconnects,
                "private_ws_reconnects": private_reconnects,
                "rest_failures": rest_failures,
                "rate_limit_events": rate_limits,
                "reconciliation_incidents": reconciliation_incidents,
                "duplicate_order_incidents": duplicate_order_incidents,
                "uncovered_position_samples": unprotected,
            },
            "verdict": {
                "operational": operational_verdict,
                "edge": edge_verdict,
                "strategy_proven": False,
            },
        }
        return report

    @staticmethod
    def markdown(report: dict[str, Any]) -> str:
        ri = report["run_integrity"]
        tr = report["trading"]
        sm = report["small_move_hypothesis"]
        ex = report["execution_quality"]
        verdict = report["verdict"]

        def percent(value):
            return "n/a" if value is None else f"{value * 100:.1f}%"

        lines = [
            f"# Bybit Demo Burn-In — {report['run_id']}",
            "",
            f"> {report['disclaimer']}",
            "",
            "## Run integrity",
            "",
            f"- Expected hours: **{ri['hours_expected']:.2f}**",
            f"- Elapsed hours: **{ri['hours_elapsed']:.2f}**",
            f"- Git SHA: `{ri['code_sha']}`",
            f"- Config hash: `{ri['config_hash']}`",
            f"- Restarts: **{ri['restarts']}**",
            f"- Crashes: **{ri['crashes']}**",
            "",
            "## Trading",
            "",
            f"- Trades: **{tr['total_trades']}** (wins {tr['wins']}, losses {tr['losses']}, break-even {tr['break_even']})",
            f"- Win rate: **{percent(tr['win_rate'])}**",
            f"- Net P&L: **{tr['net_pnl_usdt']:.4f} USDT**",
            f"- Net return: **{tr['net_return_pct']:.4f}%**" if tr['net_return_pct'] is not None else "- Net return: **n/a**",
            f"- Expectancy: **{tr['expectancy_usdt']:.4f} USDT/trade**" if tr['expectancy_usdt'] is not None else "- Expectancy: **n/a**",
            f"- Profit factor: **{tr['profit_factor']:.3f}**" if tr['profit_factor'] is not None and math.isfinite(tr['profit_factor']) else f"- Profit factor: **{tr['profit_factor']}**",
            f"- Maximum drawdown: **{tr['maximum_drawdown_pct']:.3f}%**",
            f"- Fees estimate: **{tr['fees_quote_estimate']:.4f} USDT**",
            f"- Trades/day: **{tr['trades_per_day']:.3f}** (minimum required = 0)",
            "",
            "## Small-move hypothesis",
            "",
            f"- Reached +0.25% MFE: **{percent(sm['reach_0_25_pct'])}**",
            f"- Reached +0.50% MFE: **{percent(sm['reach_0_50_pct'])}**",
            f"- Reached +0.75% MFE: **{percent(sm['reach_0_75_pct'])}**",
            f"- Reached +1.00% MFE: **{percent(sm['reach_1_00_pct'])}**",
            f"- Partial-profit instructions executed: **{sm['partial_profit_events']}**",
            "",
            "## Execution quality",
            "",
            f"- Public WS reconnects: **{ex['public_ws_reconnects']}**",
            f"- Private WS reconnects: **{ex['private_ws_reconnects']}**",
            f"- REST failures: **{ex['rest_failures']}**",
            f"- Rate-limit events: **{ex['rate_limit_events']}**",
            f"- Duplicate-order incidents: **{ex['duplicate_order_incidents']}**",
            f"- Uncovered-position health samples: **{ex['uncovered_position_samples']}**",
            "",
            "## Verdict",
            "",
            f"**{verdict['operational']}**",
            "",
            f"**{verdict['edge']}**",
            "",
            "**Strategy proven: NO.** A 3–5 day Demo run is not sufficient evidence of a durable edge.",
            "",
        ]
        return "\n".join(lines)

    def write(self, run_id: str, reports_dir: str = "reports") -> tuple[Path, Path, dict[str, Any]]:
        report = self.build(run_id)
        out = Path(reports_dir)
        out.mkdir(parents=True, exist_ok=True)
        json_path = out / f"bybit_burnin_{run_id}.json"
        md_path = out / f"bybit_burnin_{run_id}.md"
        json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
        md_path.write_text(self.markdown(report), encoding="utf-8")
        return json_path, md_path, report
