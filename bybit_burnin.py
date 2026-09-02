"""Bybit Demo operational burn-in orchestration.

A burn-in is an operational acceptance experiment, not a profitability claim.  This
service preserves the existing adaptive intraday strategy, Quant Scanner, lifecycle,
portfolio governor, execution engine, protection/reconciliation state, and exit
policy; it only supplies Bybit Spot adapters plus mission-specific conservative gates
and evidence capture.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from broker import BrokerError
from bybit_management import BybitTradeManagementExecutor, PolicyManagedStrategy
from bybit_market_data import BybitPrivateWebSocketManager, BybitSpotDataFeed
from bybit_spot import BybitSpotBroker
from bybit_state import BybitStateStore
from config import AccountMode
from crypto_config import CryptoEngineConfig, CryptoRiskConfig, CryptoStrategyConfig
from crypto_venue import CryptoExecutionEnvironment, DEFAULT_BYBIT_SPOT_UNIVERSE
from intraday_crypto_engine import (
    ADAPTIVE_STRATEGY_ID,
    ADAPTIVE_STRATEGY_VERSION,
    IntradayAdaptiveCryptoTradeEngine,
)
from models import EngineState, StopReason
from quant.adaptive_manifest import current_git_commit
from quant.contracts import StrategyState
from quant.exits import ExitPolicyConfig, TradeManagementPolicy
from quant.intraday import (
    make_high_velocity_engine_config,
    make_high_velocity_risk_config,
    make_high_velocity_strategy_config,
)
from quant.intraday_lifecycle import intraday_promotion_requirements
from quant.lifecycle import LifecycleManager
from quant.store import DEFAULT_QUANT_DB_PATH, QuantStore
from quant.strategies.intraday_adaptive import AdaptiveIntradayConfig
from state_store import StateStore

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_BYBIT_ENGINE_DB = "data/bybit_demo_trading.db"
DEFAULT_BYBIT_VENUE_DB = "data/bybit_demo_venue.db"
DEFAULT_BYBIT_QUANT_DB = DEFAULT_QUANT_DB_PATH

_ALLOWED_DEMO_LIFECYCLE = frozenset(
    {
        StrategyState.PAPER,
        StrategyState.LIVE_CANARY,
        StrategyState.LIVE,
        StrategyState.REDUCED_RISK,
    }
)


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


@dataclass(frozen=True)
class BybitBurnInConfig:
    hours: float = 120.0
    universe: tuple[str, ...] = DEFAULT_BYBIT_SPOT_UNIVERSE
    timeframe: str = "5Min"
    strategy_capital_budget_usdt: float = 1_000.0
    max_entries_rolling_24h: int = 5
    risk_per_trade_pct: float = 0.15
    max_position_pct: float = 7.5
    max_total_exposure_pct: float = 20.0
    max_concurrent_positions: int = 2
    rolling_24h_loss_limit_pct: float = 2.0
    max_consecutive_losses: int = 3
    cooldown_minutes_after_losses: int = 60
    max_pairwise_correlation: float = 0.80
    max_portfolio_heat_pct: float = 0.60
    max_spread_bps: float = 25.0
    quote_max_age_seconds: float = 5.0
    max_clock_skew_ms: float = 1_500.0
    health_sample_seconds: float = 5.0
    # Existing policy gates.  Plain trailing is the production-compatible default;
    # all smarter behaviors are supported but remain explicit experiment choices.
    exit_policy: ExitPolicyConfig = field(default_factory=ExitPolicyConfig)

    def validate(self) -> list[str]:
        errors: list[str] = []
        if self.hours < 1 or self.hours > 240:
            errors.append("Burn-in hours must be within [1, 240].")
        if not self.universe:
            errors.append("Burn-in universe cannot be empty.")
        if tuple(self.universe) != tuple(dict.fromkeys(self.universe)):
            errors.append("Burn-in universe cannot contain duplicates.")
        if not 1 <= self.max_entries_rolling_24h <= 50:
            errors.append("Rolling entry cap must be in [1, 50].")
        if self.max_concurrent_positions < 1:
            errors.append("At least one concurrent position must be permitted.")
        if self.max_concurrent_positions > len(self.universe):
            errors.append("Concurrent positions cannot exceed universe size.")
        if not (0 < self.risk_per_trade_pct <= 1.0):
            errors.append("Burn-in risk per trade must be in (0, 1]%.")
        if not (0 < self.max_position_pct <= self.max_total_exposure_pct <= 100):
            errors.append("Position/total exposure limits are invalid.")
        if not (0 < self.rolling_24h_loss_limit_pct <= 5):
            errors.append("Rolling 24h loss limit must be in (0, 5]%.")
        if self.max_consecutive_losses < 1:
            errors.append("Maximum consecutive losses must be positive.")
        if self.cooldown_minutes_after_losses < 1:
            errors.append("Loss cooldown must be positive.")
        if self.strategy_capital_budget_usdt <= 0:
            errors.append("Strategy capital budget must be positive.")
        if self.max_portfolio_heat_pct < self.risk_per_trade_pct:
            errors.append("Portfolio heat must not be below per-trade risk.")
        errors.extend(self.exit_policy.validate())
        return errors

    def frozen_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["universe"] = list(self.universe)
        return value

    @property
    def config_hash(self) -> str:
        raw = json.dumps(self.frozen_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class BurnInRiskOverlay:
    """Additional entry-only risk gates; exits are never blocked by this overlay."""
    def __init__(
        self,
        engine_store: StateStore,
        config: BybitBurnInConfig,
        *,
        now_fn=utc_now,
    ) -> None:
        self.store = engine_store
        self.config = config
        self.now_fn = now_fn

    def _recent_realized(self) -> list[dict[str, Any]]:
        cutoff = self.now_fn() - timedelta(hours=24)
        rows = []
        for row in self.store.get_trades(limit=5_000):
            if row.get("realized_pnl") is None or row.get("action") != "FILL_SELL":
                continue
            try:
                ts = datetime.fromisoformat(str(row.get("timestamp")))
            except (TypeError, ValueError):
                continue
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            if ts.astimezone(timezone.utc) >= cutoff:
                rows.append(row)
        rows.sort(key=lambda row: str(row.get("timestamp") or ""))
        return rows

    def check_entry(self, strategy_capital_budget_usdt: float) -> tuple[bool, str]:
        rows = self._recent_realized()
        realized = sum(_finite(row.get("realized_pnl")) for row in rows)
        floor = -strategy_capital_budget_usdt * self.config.rolling_24h_loss_limit_pct / 100.0
        if realized <= floor:
            return False, (
                f"rolling_24h_loss_limit: realized {realized:.2f} <= {floor:.2f} USDT"
            )
        consecutive = 0
        last_loss_at: Optional[datetime] = None
        for row in reversed(rows):
            pnl = _finite(row.get("realized_pnl"))
            if pnl < 0:
                consecutive += 1
                if last_loss_at is None:
                    try:
                        last_loss_at = datetime.fromisoformat(str(row.get("timestamp")))
                    except (TypeError, ValueError):
                        last_loss_at = None
            else:
                break
        if consecutive >= self.config.max_consecutive_losses:
            if last_loss_at is None:
                return False, "consecutive_loss_cooldown: loss timestamp unavailable"
            if last_loss_at.tzinfo is None:
                last_loss_at = last_loss_at.replace(tzinfo=timezone.utc)
            until = last_loss_at.astimezone(timezone.utc) + timedelta(
                minutes=self.config.cooldown_minutes_after_losses
            )
            if self.now_fn() < until:
                return False, f"consecutive_loss_cooldown until {until.isoformat()}"
        return True, "risk_overlay_passed"


class BurnInBybitSpotBroker(BybitSpotBroker):
    def __init__(self, *args, entry_overlay: Optional[BurnInRiskOverlay] = None, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.entry_overlay = entry_overlay

    def submit_crypto_market_order(self, symbol, qty, side, client_order_id, **kwargs):
        if str(side).lower() == "buy" and self.entry_overlay is not None:
            allowed, reason = self.entry_overlay.check_entry(self.strategy_capital_budget_usdt)
            if not allowed:
                raise BrokerError(f"Bybit Demo entry blocked by burn-in risk overlay: {reason}")
        return super().submit_crypto_market_order(symbol, qty, side, client_order_id, **kwargs)


class BybitDemoIntradayEngine(IntradayAdaptiveCryptoTradeEngine):
    """Existing Quant-controlled engine plus the existing post-entry policy executor."""
    def __init__(
        self,
        *args,
        venue_state: BybitStateStore,
        exit_policy: TradeManagementPolicy,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Prevent the legacy strategy-level ATR ratchet from becoming a second exit
        # system.  Entry evaluation and all Quant Scanner behavior still delegate.
        self.strategy = PolicyManagedStrategy(self.strategy)
        self.trade_manager = BybitTradeManagementExecutor(
            self,
            self.broker,
            venue_state,
            exit_policy,
            strategy_id=ADAPTIVE_STRATEGY_ID,
        )
        self._last_managed_bar: Optional[pd.Timestamp] = None
        self._management_observer = None

    def set_management_observer(self, observer) -> None:
        self._management_observer = observer

    def run_iteration(self) -> None:
        super().run_iteration()
        status = self.status
        if status.last_bar_time is None:
            return
        bar = pd.Timestamp(status.last_bar_time)
        if self._last_managed_bar is not None and bar <= self._last_managed_bar:
            return
        # Manage only held symbols and only from completed bars.  If management
        # cannot be evaluated, existing protective stops remain authoritative.
        positions = {p.symbol: p for p in self.broker.list_positions() if p.qty > 0}
        for symbol in sorted(positions):
            frame = self.data_feed.get_recent_bars(
                symbol,
                self.strategy_config.timeframe,
                max(80, int(self.strategy_config.atr_period) + 10),
                now=self._now_fn(),
                tolerance_seconds=self.engine_config.bar_completion_tolerance_seconds,
            )
            if frame.empty or bar not in frame.index:
                continue
            before = self.trade_manager.venue_state.get_management_state(symbol)
            instructions = self.trade_manager.manage_symbol(symbol, frame, bar)
            after = self.trade_manager.venue_state.get_management_state(symbol)
            if self._management_observer is not None:
                self._management_observer(
                    symbol=symbol,
                    bar=bar,
                    before=before,
                    after=after,
                    instructions=instructions,
                )
        self._last_managed_bar = bar


class BybitDemoBurnInService:
    def __init__(
        self,
        api_key: str,
        api_secret: str,
        config: BybitBurnInConfig,
        *,
        engine_db: str = DEFAULT_BYBIT_ENGINE_DB,
        venue_db: str = DEFAULT_BYBIT_VENUE_DB,
        quant_db: str = DEFAULT_BYBIT_QUANT_DB,
        git_sha: Optional[str] = None,
        adaptive_config: Optional[AdaptiveIntradayConfig] = None,
        http_client=None,
        public_ws_manager=None,
        private_ws_factory=None,
    ) -> None:
        errors = config.validate()
        if errors:
            raise ValueError("Invalid Bybit burn-in configuration: " + "; ".join(errors))
        self.config = config
        self.engine_store = StateStore(engine_db)
        self.venue_state = BybitStateStore(venue_db)
        self.quant_store = QuantStore(quant_db)
        self.lifecycle = LifecycleManager(
            self.quant_store, requirements=intraday_promotion_requirements()
        )
        state = self.lifecycle.state(ADAPTIVE_STRATEGY_ID)
        if state not in _ALLOWED_DEMO_LIFECYCLE:
            raise BrokerError(
                f"Bybit Demo burn-in requires lifecycle state PAPER or a later non-research "
                f"state; {ADAPTIVE_STRATEGY_ID} is {state.value}. No auto-promotion is allowed."
            )
        self.git_sha = git_sha or current_git_commit(REPO_ROOT)
        if not self.git_sha:
            raise BrokerError("Exact git SHA is required for a frozen burn-in identity")
        self.adaptive_config = adaptive_config or AdaptiveIntradayConfig()
        adaptive_errors = self.adaptive_config.validate()
        if adaptive_errors:
            raise ValueError("Invalid adaptive configuration: " + "; ".join(adaptive_errors))

        risk_config = self._risk_config()
        engine_config = self._engine_config()
        strategy_config = self._strategy_config()
        self.entry_overlay = BurnInRiskOverlay(self.engine_store, config)
        self.broker = BurnInBybitSpotBroker(
            api_key,
            api_secret,
            environment=CryptoExecutionEnvironment.BYBIT_DEMO,
            state=self.venue_state,
            strategy_capital_budget_usdt=config.strategy_capital_budget_usdt,
            permitted_symbols=config.universe,
            max_entries_rolling_24h=config.max_entries_rolling_24h,
            max_clock_skew_ms=config.max_clock_skew_ms,
            allow_live=False,
            http_client=http_client,
            entry_overlay=self.entry_overlay,
        )
        self.feed = BybitSpotDataFeed(
            self.broker.http,
            symbols=config.universe,
            timeframe=config.timeframe,
            websocket_manager=public_ws_manager,
        )
        self.engine = BybitDemoIntradayEngine(
            self.broker,
            self.feed,
            self.engine_store,
            strategy_config,
            risk_config,
            engine_config,
            adaptive_config=self.adaptive_config,
            quant_store=self.quant_store,
            lifecycle=self.lifecycle,
            venue_state=self.venue_state,
            exit_policy=TradeManagementPolicy(config.exit_policy),
        )
        self.engine.set_management_observer(self._record_management)
        self.private_ws = BybitPrivateWebSocketManager(
            api_key,
            api_secret,
            environment=CryptoExecutionEnvironment.BYBIT_DEMO,
            on_order=self._on_private_order,
            on_execution=self._on_private_execution,
            on_wallet=self._on_private_wallet,
            websocket_factory=private_ws_factory,
        )
        self.run_id: Optional[str] = None
        self._last_recorded_bar: Optional[str] = None
        self._stop = threading.Event()

    def _strategy_config(self) -> CryptoStrategyConfig:
        return make_high_velocity_strategy_config(
            symbols=self.config.universe,
            timeframe=self.config.timeframe,
        )

    def _risk_config(self) -> CryptoRiskConfig:
        base = make_high_velocity_risk_config()
        return replace(
            base,
            risk_per_trade_pct=self.config.risk_per_trade_pct,
            max_position_pct=self.config.max_position_pct,
            max_total_exposure_pct=self.config.max_total_exposure_pct,
            max_session_loss_pct=min(base.max_session_loss_pct, self.config.rolling_24h_loss_limit_pct),
            max_positions=self.config.max_concurrent_positions,
            max_pairwise_correlation=self.config.max_pairwise_correlation,
            max_portfolio_heat_pct=self.config.max_portfolio_heat_pct,
        )

    def _engine_config(self) -> CryptoEngineConfig:
        base = make_high_velocity_engine_config(account_mode=AccountMode.PAPER)
        return replace(
            base,
            max_runtime_seconds=self.config.hours * 3600.0,
            quote_max_age_seconds=self.config.quote_max_age_seconds,
            max_spread_bps=self.config.max_spread_bps,
        )

    def _run_identity(self) -> tuple[str, datetime, datetime, bool]:
        now = utc_now()
        active = self.venue_state.active_run()
        if active is not None:
            expected_universe = json.dumps(list(self.config.universe), sort_keys=True)
            if (
                active["git_sha"] != self.git_sha
                or active["strategy_version"] != ADAPTIVE_STRATEGY_VERSION
                or active["config_hash"] != self.config.config_hash
                or active["universe_json"] != expected_universe
                or active["environment"] != CryptoExecutionEnvironment.BYBIT_DEMO.value
            ):
                raise BrokerError(
                    "A different frozen Bybit burn-in is already marked running; "
                    "stop/classify it before starting changed code or parameters."
                )
            start = datetime.fromisoformat(active["start_time"])
            planned = datetime.fromisoformat(active["planned_end_time"])
            return active["run_id"], start, planned, True
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        run_id = f"bybit-demo-{stamp}-{self.config.config_hash[:10]}"
        return run_id, now, now + timedelta(hours=self.config.hours), False

    def prepare(self) -> str:
        if self.run_id:
            return self.run_id
        run_id, start, planned, expected_resume = self._run_identity()
        resumed = self.venue_state.create_or_resume_run(
            run_id=run_id,
            git_sha=self.git_sha,
            strategy_version=ADAPTIVE_STRATEGY_VERSION,
            config_hash=self.config.config_hash,
            universe=self.config.universe,
            start_time=start,
            planned_end_time=planned,
            environment=CryptoExecutionEnvironment.BYBIT_DEMO.value,
        )
        if resumed != expected_resume:
            raise RuntimeError("Burn-in resume state changed unexpectedly")
        self.run_id = run_id
        self.venue_state.record_event(
            run_id,
            "run_resume" if resumed else "run_start",
            {
                "git_sha": self.git_sha,
                "strategy_version": ADAPTIVE_STRATEGY_VERSION,
                "config_hash": self.config.config_hash,
                "config": self.config.frozen_dict(),
                "minimum_required_trades": 0,
                "maximum_entries_rolling_24h": self.config.max_entries_rolling_24h,
                "profitability_claim": False,
                "live_enabled": False,
            },
            event_key=f"{run_id}:{'resume' if resumed else 'start'}:{int(time.time())}",
            strategy=ADAPTIVE_STRATEGY_ID,
        )
        return run_id

    def _event(self, event_type: str, payload: dict[str, Any], **kwargs) -> None:
        if self.run_id:
            self.venue_state.record_event(self.run_id, event_type, payload, **kwargs)

    def _on_private_order(self, message: dict[str, Any]) -> None:
        for row in message.get("data") or []:
            self._event(
                "private_order",
                {
                    "symbol": row.get("symbol"),
                    "side": row.get("side"),
                    "status": row.get("orderStatus"),
                    "qty": row.get("qty"),
                    "cumExecQty": row.get("cumExecQty"),
                    "updatedTime": row.get("updatedTime"),
                },
                event_key=(
                    f"ws-order:{row.get('orderId')}:{row.get('orderStatus')}:"
                    f"{row.get('cumExecQty')}:{row.get('updatedTime')}"
                ),
                order_link_id=str(row.get("orderLinkId") or "") or None,
                order_id=str(row.get("orderId") or "") or None,
            )

    def _on_private_execution(self, message: dict[str, Any]) -> None:
        self.broker.sync_bot_executions()
        for row in message.get("data") or []:
            link = str(row.get("orderLinkId") or "")
            mapping = self.venue_state.order_by_link(link) if link else None
            if mapping is None:
                continue
            self._event(
                "execution",
                {
                    "side": row.get("side"),
                    "qty": row.get("execQty"),
                    "price": row.get("execPrice"),
                    "fee": row.get("execFee"),
                    "fee_currency": row.get("feeCurrency"),
                    "is_maker": row.get("isMaker"),
                    "exec_time": row.get("execTime"),
                },
                event_key=f"execution:{row.get('execId')}",
                symbol=mapping["strategy_symbol"],
                strategy=ADAPTIVE_STRATEGY_ID,
                intent_id=mapping["internal_intent_id"],
                order_link_id=link,
                order_id=str(row.get("orderId") or "") or None,
                execution_id=str(row.get("execId") or "") or None,
            )

    def _on_private_wallet(self, message: dict[str, Any]) -> None:
        # Deliberately store no complete wallet payload; only the fact/time of a
        # wallet event. Manual balances are not adopted by the trading engine.
        self._event(
            "private_wallet_event",
            {"topic": message.get("topic"), "creationTime": message.get("creationTime")},
        )

    def _record_management(self, *, symbol, bar, before, after, instructions) -> None:
        payload = {
            "bar": pd.Timestamp(bar).isoformat(),
            "before": before,
            "after": after,
            "instructions": [
                {
                    "kind": item.kind.value,
                    "reason": item.reason,
                    "new_stop": item.new_stop,
                    "exit_fraction": item.exit_fraction,
                }
                for item in instructions
            ],
        }
        self._event(
            "position_management",
            payload,
            event_key=f"management:{symbol}:{pd.Timestamp(bar).isoformat()}",
            symbol=symbol,
            strategy=ADAPTIVE_STRATEGY_ID,
        )

    def _record_decisions(self) -> None:
        status = self.engine.status
        if status.last_bar_time is None:
            return
        bar = status.last_bar_time.isoformat()
        if bar == self._last_recorded_bar:
            return
        self._last_recorded_bar = bar
        signals = status.last_signals or {}
        for symbol in self.config.universe:
            decision = signals.get(symbol)
            if decision is None:
                self._event(
                    "decision",
                    {
                        "action": "NO_TRADE",
                        "reason": "no_decision_available",
                        "bar": bar,
                    },
                    event_key=f"decision:{bar}:{symbol}",
                    symbol=symbol,
                    strategy=ADAPTIVE_STRATEGY_ID,
                    decision_id=f"{bar}:{symbol}",
                )
                continue
            action = decision.action.value
            operator_decision = "BUY_SETUP" if action == "BUY" else "NO_TRADE"
            if action == "HOLD" and decision.score is not None and decision.score > 0:
                operator_decision = "WATCH"
            payload = decision.to_dict()
            payload["operator_decision"] = operator_decision
            payload["bar"] = bar
            try:
                context = self.feed.market_context(symbol)
                payload["microstructure"] = context.to_dict()
            except Exception as exc:
                payload["microstructure_error"] = str(exc)[:180]
            self._event(
                "decision",
                payload,
                event_key=f"decision:{bar}:{symbol}",
                symbol=symbol,
                strategy=ADAPTIVE_STRATEGY_ID,
                decision_id=f"{bar}:{symbol}",
            )

    def _health(self) -> dict[str, Any]:
        broker_health = self.broker.broker_health()
        market_health = self.feed.market_health()
        private_health = self.private_ws.health()
        unprotected = 0.0
        try:
            for position in self.broker.list_positions():
                local = self.engine._load_position(position.symbol)
                if local is None or (
                    not local.stop_client_order_id and not local.pending_exit_client_order_id
                ):
                    unprotected += position.qty
        except Exception:
            unprotected = math.inf
        health = "HEALTHY"
        if broker_health.get("health") == "HALTED" or not math.isfinite(unprotected) or unprotected > 1e-12:
            health = "HALTED"
        elif (
            broker_health.get("health") != "HEALTHY"
            or not market_health.get("public_ws_up")
            or not private_health.get("private_ws_up")
        ):
            health = "DEGRADED"
        sample = {
            **broker_health,
            **market_health,
            **private_health,
            "provider": "bybit",
            "environment": "BYBIT_DEMO",
            "health": health,
            "unprotected_qty": unprotected,
            "engine_state": self.engine.status.state.value,
        }
        return sample

    def request_stop(self, reason: StopReason = StopReason.USER_STOP) -> None:
        self._stop.set()
        self.engine.stop(reason)

    def emergency_stop(self) -> None:
        self._stop.set()
        self.engine.emergency_stop()

    def run(self) -> int:
        run_id = self.prepare()
        self.broker.verify_environment()
        self.feed.start()
        self.private_ws.start()
        self.engine.start()
        last_health = 0.0
        exit_code = 0
        try:
            while not self._stop.is_set():
                status = self.engine.status
                self._record_decisions()
                now_mono = time.monotonic()
                if now_mono - last_health >= self.config.health_sample_seconds:
                    sample = self._health()
                    self.venue_state.record_health(run_id, sample)
                    self.venue_state.heartbeat_run(run_id)
                    last_health = now_mono
                    if sample["health"] == "HALTED":
                        self.engine.emergency_stop()
                        self._event("operational_halt", sample)
                        exit_code = 2
                        break
                if status.state in {EngineState.STOPPED, EngineState.ERROR, EngineState.HALTED}:
                    if status.error:
                        self._event("engine_error", {"error": status.error})
                        exit_code = 2
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            self.request_stop()
        except Exception as exc:
            self.engine.emergency_stop()
            self._event("service_exception", {"error": str(exc)[:300]})
            exit_code = 2
        finally:
            self.engine.stop()
            self.engine.join(15.0)
            self.private_ws.stop()
            self.feed.stop()
            status = self.engine.status
            reason = (
                status.stop_reason.value if status.stop_reason is not None else "SERVICE_STOP"
            )
            final_state = "failed" if exit_code else "stopped"
            self.venue_state.stop_run(run_id, reason, status=final_state)
        return exit_code

    def close(self) -> None:
        try:
            self.private_ws.stop()
        finally:
            try:
                self.feed.stop()
            finally:
                self.quant_store.close()
                self.engine_store.close()
                self.venue_state.close()


def credentials_from_environment() -> tuple[str, str]:
    key = os.getenv("BYBIT_DEMO_API_KEY", "").strip()
    secret = os.getenv("BYBIT_DEMO_API_SECRET", "").strip()
    if not key or not secret:
        raise BrokerError(
            "BYBIT_DEMO_API_KEY and BYBIT_DEMO_API_SECRET are required for Demo burn-in"
        )
    return key, secret
