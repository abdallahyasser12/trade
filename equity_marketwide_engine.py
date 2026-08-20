"""Production paper runtime for market-wide US-equity opportunity rotation.

The engine continuously discovers the liquid executable US-equity universe, evaluates
multiple independently versioned intraday hypotheses, lets Quant Desk rank every BUY
by measured post-cost expectancy, exits risk first, and then recycles available risk
into the best remaining candidates.

Safety boundaries are intentional:
* equity execution is PAPER-only; ``broker.submit_market_order`` already hard-blocks
  live equities and this engine independently verifies paper mode at startup;
* strategies remain governed by Quant lifecycle state (RESEARCH/SHADOW/PAPER/...);
* every order has a durable local intent before broker submission and a deterministic
  client id for restart idempotency;
* only positions carrying this engine's explicit ownership record are exited or have
  stops managed; unrelated account positions remain untouched but count as exposure;
* exits cancel and confirm the bot stop before submitting a market sell;
* entries stop on session-loss, PDT, near-close, stale data, quote, portfolio heat,
  cost/expectancy or lifecycle failures.
"""

from __future__ import annotations

import json
import math
import threading
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Callable, Mapping, Optional

import pandas as pd

from broker import AlpacaBroker, BrokerError, make_client_order_id
from config import AccountMode
from data_feed import DataFeedError, is_data_stale
from equity_market_data import AlpacaEquityMarketData
from models import EngineState, OrderRecord, StopReason
from quant.contracts import DecisionOutcome, Market, QuantAction
from quant.equity_marketwide import (
    EquityUniverseReport,
    MarketWideEquityConfig,
    MarketWideEquitySelector,
    build_equity_scan_strategies,
    required_intraday_bars,
)
from quant.evidence_scanner import EvidenceQuantScanner
from quant.execution import ExecutionQualityEngine
from quant.lifecycle import LifecycleManager
from quant.portfolio import (
    BookPosition,
    PortfolioBook,
    PortfolioLimitsConfig,
    sector_for,
)
from quant.store import QuantStore
from state_store import StateStore


EQUITY_MARKETWIDE_SESSION_SYMBOL = "__US_EQUITY_MARKETWIDE__"
EQUITY_MARKETWIDE_KILL_SWITCH_KEY = "equity_marketwide_kill_switch"
EQUITY_MARKETWIDE_STATUS_KEY = "equity_marketwide_engine_status"
EQUITY_MARKETWIDE_UNIVERSE_KEY = "equity_marketwide_universe"
EQUITY_MARKETWIDE_LAST_DECISIONS_KEY = "equity_marketwide_last_decisions"
_PENDING_ENTRY_PREFIX = "equity_marketwide_pending_entry:"
_OWNERSHIP_MARKET = Market.US_EQUITY.value
_MANAGED_STRATEGIES = frozenset(
    {"opening_range_breakout", "vwap_reversion", "volatility_squeeze"}
)
_PDT_EQUITY_THRESHOLD = 25_000.0
_PDT_MAX_DAY_TRADES = 3
_QTY_EPS = 1e-9


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _is_finite_positive(value) -> bool:
    try:
        return math.isfinite(float(value)) and float(value) > 0
    except (TypeError, ValueError):
        return False


@dataclass
class MarketWideEquityEngineStatus:
    state: EngineState = EngineState.STOPPED
    session_id: Optional[str] = None
    started_at: Optional[datetime] = None
    last_heartbeat: Optional[datetime] = None
    last_bar_time: Optional[datetime] = None
    selected_symbols: int = 0
    discovered_symbols: int = 0
    open_bot_positions: int = 0
    last_executable_candidates: int = 0
    market_open: Optional[bool] = None
    halt_reason: Optional[str] = None
    stop_reason: Optional[StopReason] = None
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "state": self.state.value,
            "session_id": self.session_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "last_heartbeat": (
                self.last_heartbeat.isoformat() if self.last_heartbeat else None
            ),
            "last_bar_time": self.last_bar_time.isoformat() if self.last_bar_time else None,
            "selected_symbols": self.selected_symbols,
            "discovered_symbols": self.discovered_symbols,
            "open_bot_positions": self.open_bot_positions,
            "last_executable_candidates": self.last_executable_candidates,
            "market_open": self.market_open,
            "halt_reason": self.halt_reason,
            "stop_reason": self.stop_reason.value if self.stop_reason else None,
            "error": self.error,
            "warnings": list(self.warnings),
        }


class MarketWideEquityTradeEngine:
    """Multi-position, multi-strategy US-equity paper engine."""

    def __init__(
        self,
        broker: AlpacaBroker,
        market_data: AlpacaEquityMarketData,
        store: StateStore,
        *,
        config: MarketWideEquityConfig = MarketWideEquityConfig(),
        quant_store: Optional[QuantStore] = None,
        lifecycle: Optional[LifecycleManager] = None,
        scanner: Optional[EvidenceQuantScanner] = None,
        now_fn: Callable[[], datetime] = _utc_now,
    ) -> None:
        errors = config.validate()
        if errors:
            raise ValueError("Invalid market-wide equity configuration: " + "; ".join(errors))
        if scanner is not None and lifecycle is not None and scanner.lifecycle is not lifecycle:
            raise ValueError("Supplied scanner and lifecycle must share one lifecycle instance.")
        if scanner is not None and quant_store is not None and scanner.store is not quant_store:
            raise ValueError("Supplied scanner and quant_store must share one store instance.")

        self.broker = broker
        self.market_data = market_data
        self.store = store
        self.config = config
        self.now_fn = now_fn
        self.session_id = uuid.uuid4().hex
        self.quant_store = quant_store or (scanner.store if scanner is not None else QuantStore())
        self._owns_quant_store = quant_store is None and scanner is None
        self.lifecycle = lifecycle or (scanner.lifecycle if scanner is not None else LifecycleManager(self.quant_store))
        if self.lifecycle.store is not self.quant_store:
            raise ValueError("LifecycleManager and QuantStore must reference the same store.")

        limits = PortfolioLimitsConfig(
            max_portfolio_heat_pct=max(1.0, config.risk_per_trade_pct * config.max_positions),
            max_gross_exposure_pct=config.max_total_exposure_pct,
            max_market_exposure_pct=config.max_total_exposure_pct,
            max_sector_exposure_pct=min(25.0, config.max_total_exposure_pct),
            max_strategy_heat_pct=max(0.5, config.risk_per_trade_pct * 3.0),
            max_new_risk_per_cycle_pct=max(
                config.risk_per_trade_pct,
                config.risk_per_trade_pct * config.max_positions,
            ),
            max_pairwise_correlation=0.80,
            correlation_lookback=60,
            min_correlation_observations=30,
            reject_unknown_correlation=False,
        )
        self.execution_quality = ExecutionQualityEngine(state_store=store)
        self.scanner = scanner or EvidenceQuantScanner(
            store=self.quant_store,
            lifecycle=self.lifecycle,
            execution_engine=self.execution_quality,
            portfolio_config=limits,
        )
        self.selector = MarketWideEquitySelector(config)

        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.RLock()
        self._started_at: Optional[datetime] = None
        self._starting_equity: Optional[float] = None
        self._peak_equity: Optional[float] = None
        self._session_halted = False
        self._universe: Optional[EquityUniverseReport] = None
        self._last_universe_refresh: Optional[datetime] = None
        self._status = MarketWideEquityEngineStatus(
            state=EngineState.STOPPED,
            session_id=self.session_id,
        )

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    @property
    def status(self) -> MarketWideEquityEngineStatus:
        with self._lock:
            return MarketWideEquityEngineStatus(**{**self._status.__dict__, "warnings": list(self._status.warnings)})

    def start(self) -> None:
        with self._lock:
            if self.is_running:
                return
            self._stop_event.clear()
            self._status.state = EngineState.STARTING
            self._persist_status()
            self._thread = threading.Thread(
                target=self._run,
                name=f"equity-marketwide-{self.session_id[:8]}",
                daemon=True,
            )
            self._thread.start()

    def stop(self, reason: StopReason = StopReason.USER_STOP) -> None:
        with self._lock:
            self._status.stop_reason = reason
            if self._status.state in {EngineState.STARTING, EngineState.RUNNING}:
                self._status.state = EngineState.STOPPING
            self._persist_status()
        self._stop_event.set()

    def emergency_stop(self) -> None:
        now = self.now_fn()
        self.store.set_kv(
            EQUITY_MARKETWIDE_KILL_SWITCH_KEY,
            json.dumps({"armed": True, "reason": "EMERGENCY_STOP", "at": now.isoformat()}),
        )
        self.stop(StopReason.EMERGENCY_STOP)

    def join(self, timeout: Optional[float] = None) -> None:
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def close(self) -> None:
        if self.is_running:
            self.stop()
            self.join(5.0)
        if self._owns_quant_store:
            self.quant_store.close()

    def _run(self) -> None:
        reason = StopReason.USER_STOP
        try:
            self._startup()
            with self._lock:
                self._status.state = EngineState.RUNNING
                self._persist_status()
            while not self._stop_event.is_set():
                try:
                    self.run_iteration()
                except (BrokerError, DataFeedError) as exc:
                    self._warn(str(exc))
                except Exception as exc:  # noqa: BLE001 - engine process boundary
                    self.store.log(
                        "CRITICAL",
                        f"Market-wide equity engine fatal error: {exc}",
                        context={"traceback": traceback.format_exc(limit=20)},
                        session_id=self.session_id,
                    )
                    with self._lock:
                        self._status.state = EngineState.ERROR
                        self._status.error = str(exc)
                        self._status.stop_reason = StopReason.FATAL_ERROR
                        self._persist_status()
                    self.store.set_kv(
                        EQUITY_MARKETWIDE_KILL_SWITCH_KEY,
                        json.dumps(
                            {"armed": True, "reason": "FATAL_ERROR", "at": self.now_fn().isoformat()}
                        ),
                    )
                    reason = StopReason.FATAL_ERROR
                    break
                # Decisions are bar-driven; order/stop reconciliation is not.  Five
                # seconds gives pending market orders and canceled stops prompt care.
                if self._stop_event.wait(5.0):
                    break
            with self._lock:
                reason = self._status.stop_reason or reason
        except Exception as exc:  # noqa: BLE001
            reason = StopReason.FATAL_ERROR
            with self._lock:
                self._status.state = EngineState.ERROR
                self._status.error = str(exc)
                self._status.stop_reason = reason
                self._persist_status()
            self.store.set_kv(
                EQUITY_MARKETWIDE_KILL_SWITCH_KEY,
                json.dumps({"armed": True, "reason": "STARTUP_ERROR", "at": self.now_fn().isoformat()}),
            )
            self.store.log("CRITICAL", f"Market-wide equity startup failed: {exc}", session_id=self.session_id)
        finally:
            self._finalize(reason)

    def _startup(self) -> None:
        if getattr(self.broker, "account_mode", None) is not AccountMode.PAPER:
            raise BrokerError("Market-wide equity execution is paper-only; live equity startup is refused.")
        raw_kill = self.store.get_kv(EQUITY_MARKETWIDE_KILL_SWITCH_KEY)
        if raw_kill:
            try:
                if json.loads(raw_kill).get("armed"):
                    raise BrokerError("Market-wide equity kill switch is armed; acknowledge it before restart.")
            except json.JSONDecodeError as exc:
                raise BrokerError("Market-wide equity kill-switch state is corrupt.") from exc
        if not self.store.acquire_engine_lock():
            raise BrokerError("Another trading engine already owns this state database.")
        try:
            account = self.broker.verify_paper_account()
            if not account.can_trade:
                raise BrokerError("Alpaca paper account is not active and tradable.")
            now = self.now_fn()
            if now.tzinfo is None:
                raise ValueError("Engine clock must return timezone-aware datetimes.")
            active = self.store.get_active_session()
            if active and active["session_id"] != self.session_id:
                self.store.end_session(active["session_id"], now, "RECOVERED_AFTER_PROCESS_RESTART")
            self._started_at = now
            self._starting_equity = account.equity
            self._peak_equity = account.equity
            self.store.create_session(
                self.session_id,
                EQUITY_MARKETWIDE_SESSION_SYMBOL,
                self.config.timeframe,
                now,
                None,
                account.equity,
                json.dumps({"marketwide_equity": self.config.to_dict()}, sort_keys=True),
            )
            self.store.record_equity(self.session_id, now, account.equity, account.buying_power)
            self._reconcile_orders(now)
            self._reconcile_positions(now)
            self._ensure_all_stops(now)
            with self._lock:
                self._status.started_at = now
                self._status.state = EngineState.RUNNING
                self._persist_status()
            self.store.log(
                "INFO",
                "Market-wide US-equity paper session started.",
                context={"timeframe": self.config.timeframe},
                session_id=self.session_id,
            )
        except BaseException:
            self.store.release_engine_lock()
            raise

    def _finalize(self, reason: StopReason) -> None:
        now = self.now_fn()
        try:
            session = self.store.get_session(self.session_id)
            if session and session.get("status") == "running":
                self.store.end_session(self.session_id, now, reason.value)
        except Exception:
            pass
        finally:
            self.store.release_engine_lock()
        with self._lock:
            if self._status.state is not EngineState.ERROR:
                self._status.state = EngineState.HALTED if self._session_halted else EngineState.STOPPED
            self._status.stop_reason = reason
            self._status.last_heartbeat = now
            self._persist_status()

    def run_iteration(self) -> None:
        now = self.now_fn()
        if now.tzinfo is None:
            raise ValueError("Engine clock must return timezone-aware datetimes.")
        account = self.broker.get_account()
        if not account.can_trade:
            raise BrokerError("Alpaca account became blocked or inactive.")
        self._peak_equity = max(self._peak_equity or account.equity, account.equity)
        self._apply_session_loss_gate(account.equity, now)

        self._reconcile_orders(now)
        broker_positions = self._reconcile_positions(now)
        self._ensure_all_stops(now, broker_positions)

        clock = self.broker.get_clock()
        with self._lock:
            self._status.market_open = clock.is_open
        if not clock.is_open:
            self._heartbeat(now, None, broker_positions)
            return

        self._refresh_universe_if_due(now)
        selected = tuple(self._universe.selected if self._universe else ())
        owned = self._owned_positions(broker_positions)
        scan_symbols = tuple(dict.fromkeys((*selected, *owned.keys())))
        if not scan_symbols:
            self._warn("No liquid market-wide equity symbols are currently eligible.")
            self._heartbeat(now, None, broker_positions)
            return

        quotes = self.market_data.get_latest_quotes(scan_symbols)
        owned_by_strategy: dict[str, set[str]] = {}
        for symbol, record in owned.items():
            strategy_id = str(record.get("strategy_id") or "")
            if strategy_id in _MANAGED_STRATEGIES:
                owned_by_strategy.setdefault(strategy_id, set()).add(symbol)
        provisional = build_equity_scan_strategies(
            scan_symbols,
            config=self.config,
            owned_by_strategy=owned_by_strategy,
        )
        lookback = required_intraday_bars(provisional, self.config.intraday_lookback_bars)
        bars = self.market_data.get_recent_bars_many(
            scan_symbols,
            self.config.timeframe,
            lookback,
            now=now,
            tolerance_seconds=2.0,
        )
        bars, aligned_timestamp = self._aligned_fresh_bars(bars, provisional, now)
        if aligned_timestamp is None:
            self._warn("No sufficiently broad exactly-aligned completed equity bar; entries blocked.")
            self._heartbeat(now, None, broker_positions)
            return

        good_symbols = tuple(symbol for symbol in scan_symbols if symbol in bars)
        good_owned_by_strategy = {
            strategy_id: {symbol for symbol in symbols if symbol in bars}
            for strategy_id, symbols in owned_by_strategy.items()
        }
        strategies = build_equity_scan_strategies(
            good_symbols,
            config=self.config,
            owned_by_strategy=good_owned_by_strategy,
        )
        if not strategies:
            self._heartbeat(now, aligned_timestamp.to_pydatetime(), broker_positions)
            return

        # A completed market-wide bar is economic work only once. Deterministic order
        # ids also protect the crash window before these markers are written.
        if all(
            self.store.is_bar_processed(symbol, self.config.timeframe, aligned_timestamp)
            for symbol in good_symbols
        ):
            self._heartbeat(now, aligned_timestamp.to_pydatetime(), broker_positions)
            return

        book = self._portfolio_book(broker_positions, bars, account.equity)
        drawdown_pct = (
            max(0.0, ((self._peak_equity or account.equity) - account.equity) / (self._peak_equity or account.equity) * 100.0)
            if (self._peak_equity or account.equity) > 0
            else 0.0
        )
        benchmark = "SPY" if "SPY" in bars else good_symbols[0]
        report = self.scanner.scan(
            strategies=strategies,
            bars_by_symbol=bars,
            equity=account.equity,
            book=book,
            quotes=quotes,
            benchmark_symbol=benchmark,
            drawdown_pct=drawdown_pct,
            session_id=self.session_id,
            now=now,
        )
        self._persist_decisions(report)

        # Risk-reducing exits always precede fresh entries.
        exit_rows = [
            decision
            for decision in report.decisions
            if decision.outcome is DecisionOutcome.EXECUTE
            and decision.candidate.action is QuantAction.EXIT
        ]
        for decision in exit_rows:
            symbol = decision.candidate.symbol
            local = self.store.get_bot_position(symbol)
            if not self._is_ours(local):
                continue
            if local.get("strategy_id") != decision.candidate.strategy_id:
                continue
            position = self._position_map().get(symbol)
            if position is None or position.qty <= 0:
                continue
            self._initiate_exit(
                symbol,
                position,
                decision.candidate.strategy_id,
                decision.candidate.signal_reason,
                aligned_timestamp,
                now,
            )

        self._reconcile_orders(now)
        broker_positions = self._reconcile_positions(now)
        self._ratchet_stops(strategies, bars, aligned_timestamp, broker_positions, now)
        self._ensure_all_stops(now, broker_positions)

        account = self.broker.get_account()
        if not self._session_halted and self._entries_allowed_by_clock(account, clock, now):
            self._execute_ranked_entries(
                report.executable,
                quotes,
                aligned_timestamp,
                account,
                broker_positions,
                now,
            )

        for symbol in good_symbols:
            self.store.mark_bar_processed(
                symbol,
                self.config.timeframe,
                aligned_timestamp,
                self.session_id,
            )
        self._heartbeat(now, aligned_timestamp.to_pydatetime(), self._position_map())

    def _refresh_universe_if_due(self, now: datetime) -> None:
        if (
            self._universe is not None
            and self._last_universe_refresh is not None
            and (now - self._last_universe_refresh).total_seconds() < self.config.universe_refresh_seconds
        ):
            return
        assets = self.market_data.list_active_us_equities()
        candidate_symbols = tuple(asset.symbol for asset in assets if asset.active and asset.tradable)
        quotes = self.market_data.get_latest_quotes(candidate_symbols)
        daily = self.market_data.get_recent_bars_many(
            candidate_symbols,
            "1Day",
            self.config.liquidity_lookback_days,
            now=now,
            tolerance_seconds=2.0,
        )
        report = self.selector.select(assets, quotes, daily, now=now)
        self._universe = report
        self._last_universe_refresh = now
        self.store.set_kv(EQUITY_MARKETWIDE_UNIVERSE_KEY, json.dumps(report.to_dict(), sort_keys=True))
        with self._lock:
            self._status.selected_symbols = len(report.selected)
            self._status.discovered_symbols = report.discovered
            self._persist_status()
        self.store.log(
            "INFO",
            f"Market-wide universe refreshed: {len(report.selected)} selected from {report.discovered} active assets.",
            context={"rejected": len(report.rejected)},
            session_id=self.session_id,
        )

    def _aligned_fresh_bars(self, bars, strategies, now):
        required_by_symbol: dict[str, int] = {}
        for entry in strategies:
            required = int(entry.strategy.required_bars())
            for symbol in getattr(entry.strategy.config, "symbols", ()):  # structural strategy contract
                required_by_symbol[symbol] = max(required_by_symbol.get(symbol, 0), required)

        usable: dict[str, pd.DataFrame] = {}
        latest_counts: dict[pd.Timestamp, int] = {}
        for symbol, frame in bars.items():
            required = required_by_symbol.get(symbol, self.config.intraday_lookback_bars)
            if frame is None or len(frame) < required:
                continue
            latest = frame.index[-1]
            if is_data_stale(
                latest.to_pydatetime(),
                self.config.timeframe,
                now,
                staleness_bars=2,
            ):
                continue
            usable[symbol] = frame
            latest_counts[latest] = latest_counts.get(latest, 0) + 1
        if not latest_counts:
            return {}, None
        target = sorted(latest_counts, key=lambda ts: (-latest_counts[ts], -ts.value))[0]
        aligned = {symbol: frame for symbol, frame in usable.items() if target in frame.index and frame.index[-1] == target}
        # Avoid treating a tiny surviving fragment as "market-wide" after a feed outage.
        minimum = min(10, max(3, len(self._universe.selected) // 10 if self._universe else 3))
        if len(aligned) < minimum:
            return {}, None
        return aligned, target

    def _portfolio_book(self, broker_positions, bars, equity: float) -> PortfolioBook:
        positions: list[BookPosition] = []
        for symbol, position in broker_positions.items():
            if position.qty <= 0 or position.asset_class not in (None, "us_equity"):
                continue
            mark = position.current_price
            frame = bars.get(symbol)
            if not _is_finite_positive(mark) and frame is not None and not frame.empty:
                mark = float(frame["close"].iloc[-1])
            if not _is_finite_positive(mark):
                mark = position.avg_entry_price
            local = self.store.get_bot_position(symbol)
            if self._is_ours(local) and _is_finite_positive(local.get("stop_price")):
                positions.append(
                    BookPosition.from_stop(
                        symbol,
                        Market.US_EQUITY,
                        str(local.get("strategy_id")),
                        qty=float(position.qty),
                        price=float(mark),
                        stop_price=float(local["stop_price"]),
                    )
                )
            else:
                # Unowned account positions are never touched but must consume gross
                # exposure. Risk-at-stop is unknown, so do not fabricate a stop.
                positions.append(
                    BookPosition(
                        symbol=symbol,
                        market=Market.US_EQUITY,
                        sector=sector_for(symbol, Market.US_EQUITY),
                        strategy_id="external",
                        notional=float(position.qty) * float(mark),
                        risk_at_stop=0.0,
                    )
                )
        return PortfolioBook(equity=float(equity), positions=positions)

    def _execute_ranked_entries(self, decisions, quotes, bar_timestamp, account, broker_positions, now):
        positions = dict(broker_positions)
        open_orders = self.broker.list_open_orders()
        pending_buys = {order.symbol for order in open_orders if order.side == "buy" and order.is_open}
        bot_owned_count = len(self._owned_positions(positions))
        bot_notional = sum(
            abs(float(position.market_value or (position.qty * position.avg_entry_price)))
            for symbol, position in positions.items()
            if self._is_ours(self.store.get_bot_position(symbol))
        )
        cash = float(account.cash)
        buying_power = float(account.buying_power)

        for ranked in decisions:
            if ranked.candidate.market is not Market.US_EQUITY:
                continue
            symbol = ranked.candidate.symbol
            if symbol in positions or symbol in pending_buys:
                continue
            if bot_owned_count >= self.config.max_positions:
                break
            quote = quotes.get(symbol)
            if quote is None:
                continue
            try:
                entry_price = float(quote.ask_price)
            except (AttributeError, TypeError, ValueError):
                continue
            stop_price = ranked.candidate.stop_price
            if not _is_finite_positive(entry_price) or not _is_finite_positive(stop_price):
                continue
            stop_price = float(stop_price)
            if not 0 < stop_price < entry_price:
                continue
            risk_pct = min(float(ranked.risk_allocation_pct or 0.0), self.config.risk_per_trade_pct)
            if risk_pct <= 0:
                continue
            risk_amount = float(account.equity) * risk_pct / 100.0
            qty_risk = math.floor(risk_amount / (entry_price - stop_price))
            qty_position = math.floor(
                float(account.equity) * self.config.max_position_pct / 100.0 / entry_price
            )
            remaining_exposure = max(
                0.0,
                float(account.equity) * self.config.max_total_exposure_pct / 100.0 - bot_notional,
            )
            qty_exposure = math.floor(remaining_exposure / entry_price)
            qty_cash = math.floor(max(0.0, min(cash, buying_power)) / entry_price)
            qty = int(min(qty_risk, qty_position, qty_exposure, qty_cash))
            if qty <= 0:
                continue

            client_id = make_client_order_id(
                self.session_id,
                "entry",
                bar_timestamp.to_pydatetime(),
                symbol,
            )
            intent = OrderRecord(
                broker_order_id="",
                client_order_id=client_id,
                symbol=symbol,
                purpose="entry",
                side="buy",
                requested_qty=qty,
                filled_qty=0.0,
                status="intent",
                submitted_at=now,
                updated_at=now,
                reference_price=entry_price,
            )
            self.store.record_order(intent, self.session_id)
            self.store.set_kv(
                _PENDING_ENTRY_PREFIX + symbol,
                json.dumps(
                    {
                        "client_order_id": client_id,
                        "strategy_id": ranked.candidate.strategy_id,
                        "strategy_version": ranked.candidate.strategy_version,
                        "stop_price": stop_price,
                        "entry_reason": ranked.candidate.signal_reason,
                        "signal_timestamp": (
                            ranked.candidate.timestamp.isoformat() if ranked.candidate.timestamp else None
                        ),
                    },
                    sort_keys=True,
                ),
            )
            order = self._submit_idempotent(
                client_id,
                lambda: self.broker.submit_market_order(symbol, qty, "buy", client_id),
            )
            self._sync_order(order, intent, now)
            pending_buys.add(symbol)
            notional = qty * entry_price
            cash = max(0.0, cash - notional)
            buying_power = max(0.0, buying_power - notional)
            bot_notional += notional
            bot_owned_count += 1

    def _initiate_exit(self, symbol, position, strategy_id, reason, bar_timestamp, now):
        if not self._cancel_stop_for_exit(symbol):
            self.store.log(
                "WARNING",
                f"{symbol} exit deferred until protective stop cancellation is confirmed.",
                session_id=self.session_id,
            )
            return
        qty = int(math.floor(float(position.qty) + _QTY_EPS))
        if qty <= 0:
            return
        reference = position.current_price or position.avg_entry_price
        client_id = make_client_order_id(
            self.session_id,
            "exit",
            bar_timestamp.to_pydatetime(),
            symbol,
        )
        intent = OrderRecord(
            broker_order_id="",
            client_order_id=client_id,
            symbol=symbol,
            purpose="exit",
            side="sell",
            requested_qty=qty,
            filled_qty=0.0,
            status="intent",
            submitted_at=now,
            updated_at=now,
            reference_price=float(reference) if _is_finite_positive(reference) else None,
        )
        self.store.record_order(intent, self.session_id)
        local = self.store.get_bot_position(symbol) or {}
        local["pending_exit_client_order_id"] = client_id
        local["pending_exit_reason"] = reason
        local["strategy_id"] = strategy_id
        self.store.set_bot_position(symbol, local)
        order = self._submit_idempotent(
            client_id,
            lambda: self.broker.submit_market_order(symbol, qty, "sell", client_id),
        )
        self._sync_order(order, intent, now)

    def _submit_idempotent(self, client_id: str, submit_fn):
        existing = self.broker.get_order_by_client_id(client_id)
        if existing is not None:
            return existing
        try:
            return submit_fn()
        except BrokerError:
            recovered = self.broker.get_order_by_client_id(client_id)
            if recovered is not None:
                return recovered
            raise

    def _reconcile_orders(self, now: datetime) -> None:
        rows = self.store.get_orders(session_id=self.session_id, limit=2000)
        for row in rows:
            local = self.store.get_order_by_client_id(str(row["client_order_id"]))
            if local is None or local.is_terminal:
                continue
            broker_order = self.broker.get_order_by_client_id(local.client_order_id)
            if broker_order is None:
                # A pre-submit intent can legitimately have no broker row after a
                # crash. Keep it; the deterministic next decision can resubmit it.
                continue
            self._sync_order(broker_order, local, now)

    def _sync_order(self, order: OrderRecord, previous: Optional[OrderRecord], now: datetime) -> None:
        previous_filled = float(previous.filled_qty if previous else 0.0)
        reference = previous.reference_price if previous else None
        if reference is not None and order.reference_price is None:
            order.reference_price = reference
        delta = max(0.0, float(order.filled_qty) - previous_filled)
        self.store.record_order(order, self.session_id)
        if delta <= _QTY_EPS or order.fill_price is None:
            return

        fill_price = float(order.fill_price)
        event_key = f"{order.client_order_id}:{float(order.filled_qty):.10g}"
        self.store.record_execution(
            self.session_id,
            order.client_order_id,
            order.symbol,
            order.purpose,
            order.side,
            reference,
            fill_price,
            delta,
            timestamp=order.updated_at or now,
            event_key=event_key,
        )
        if order.side == "buy":
            self.store.record_trade(
                self.session_id,
                order.symbol,
                "BUY",
                delta,
                fill_price,
                None,
                "marketwide_entry_fill",
                None,
                order.updated_at or now,
                event_key=event_key,
            )
            self._adopt_entry_fill(order.symbol, order, now)
        elif order.side == "sell":
            local = self.store.get_bot_position(order.symbol)
            entry_price = float(local.get("entry_price", 0.0)) if self._is_ours(local) else 0.0
            realized = (fill_price - entry_price) * delta if entry_price > 0 else None
            reason = (
                str(local.get("pending_exit_reason") or "strategy_exit")
                if order.purpose == "exit" and local
                else "protective_stop_fill"
            )
            self.store.record_trade(
                self.session_id,
                order.symbol,
                "SELL",
                delta,
                fill_price,
                realized,
                reason,
                None,
                order.updated_at or now,
                event_key=event_key,
            )

    def _adopt_entry_fill(self, symbol: str, order: OrderRecord, now: datetime) -> None:
        raw = self.store.get_kv(_PENDING_ENTRY_PREFIX + symbol)
        if not raw:
            return
        try:
            context = json.loads(raw)
        except json.JSONDecodeError:
            return
        position = self.broker.get_position(symbol)
        if position is None or position.qty <= 0:
            return
        stop = context.get("stop_price")
        if not _is_finite_positive(stop) or float(stop) >= position.avg_entry_price:
            # Keep the pending context and refuse to invent downside protection.
            raise BrokerError(f"Cannot adopt {symbol} entry: persisted stop is invalid for the fill price.")
        existing = self.store.get_bot_position(symbol)
        record = existing if self._is_ours(existing) else {}
        record.update(
            {
                "market": _OWNERSHIP_MARKET,
                "strategy_id": str(context.get("strategy_id")),
                "strategy_version": str(context.get("strategy_version") or "1.0.0"),
                "qty": float(position.qty),
                "entry_price": float(position.avg_entry_price),
                "entry_time": (order.updated_at or now).isoformat(),
                "entry_client_order_id": order.client_order_id,
                "entry_reason": context.get("entry_reason"),
                "stop_price": float(stop),
                "initial_stop_price": float(stop),
                "highest_close": float(position.avg_entry_price),
                "stop_client_order_id": record.get("stop_client_order_id"),
                "stop_attempt": int(record.get("stop_attempt") or 0),
                "pending_exit_client_order_id": None,
                "pending_exit_reason": None,
            }
        )
        self.store.set_bot_position(symbol, record)
        if order.is_terminal:
            self.store.delete_kv(_PENDING_ENTRY_PREFIX + symbol)
        self._ensure_stop(symbol, position, now)

    def _reconcile_positions(self, now: datetime):
        positions = self._position_map()
        known = self._known_owned_symbols()
        for symbol in known:
            local = self.store.get_bot_position(symbol)
            if not self._is_ours(local):
                continue
            broker_position = positions.get(symbol)
            if broker_position is not None and broker_position.qty > 0:
                local["qty"] = float(broker_position.qty)
                self.store.set_bot_position(symbol, local)
                continue
            # Broker is flat. First reconcile the bot sell that may have closed it.
            for key in ("pending_exit_client_order_id", "stop_client_order_id"):
                client_id = local.get(key)
                if not client_id:
                    continue
                order = self.broker.get_order_by_client_id(str(client_id))
                previous = self.store.get_order_by_client_id(str(client_id))
                if order is not None:
                    self._sync_order(order, previous, now)
            # Never leave an orphaned sell stop after an external/manual close.
            stop_id = local.get("stop_client_order_id")
            if stop_id:
                stop_order = self.broker.get_order_by_client_id(str(stop_id))
                if stop_order is not None and stop_order.is_open:
                    self.broker.cancel_order(stop_order.broker_order_id)
                    refreshed = self.broker.get_order_by_client_id(str(stop_id))
                    if refreshed is not None and refreshed.is_open:
                        raise BrokerError(f"{symbol} is flat but its bot stop is still open; cleanup deferred.")
            self.store.clear_bot_position(symbol)
            self.store.delete_kv(_PENDING_ENTRY_PREFIX + symbol)
        with self._lock:
            self._status.open_bot_positions = len(self._owned_positions(positions))
        return positions

    def _known_owned_symbols(self) -> set[str]:
        symbols: set[str] = set()
        for row in self.store.get_orders(limit=5000):
            symbol = str(row.get("symbol") or "")
            if symbol:
                local = self.store.get_bot_position(symbol)
                if self._is_ours(local):
                    symbols.add(symbol)
        return symbols

    def _owned_positions(self, positions: Mapping[str, object]) -> dict[str, dict]:
        out: dict[str, dict] = {}
        for symbol, position in positions.items():
            if position.qty <= 0:
                continue
            local = self.store.get_bot_position(symbol)
            if self._is_ours(local):
                out[symbol] = local
        return out

    @staticmethod
    def _is_ours(local: Optional[dict]) -> bool:
        return bool(
            isinstance(local, dict)
            and local.get("market") == _OWNERSHIP_MARKET
            and local.get("strategy_id") in _MANAGED_STRATEGIES
        )

    def _position_map(self):
        return {
            position.symbol: position
            for position in self.broker.list_positions()
            if position.asset_class in (None, "us_equity")
        }

    def _ensure_all_stops(self, now: datetime, positions=None) -> None:
        positions = positions or self._position_map()
        for symbol, local in self._owned_positions(positions).items():
            self._ensure_stop(symbol, positions[symbol], now)

    def _ensure_stop(self, symbol: str, position, now: datetime) -> None:
        local = self.store.get_bot_position(symbol)
        if not self._is_ours(local):
            return
        stop_price = local.get("stop_price")
        if not _is_finite_positive(stop_price):
            raise BrokerError(f"Bot-owned {symbol} position has no valid protective stop price.")
        qty = int(math.floor(float(position.qty) + _QTY_EPS))
        if qty <= 0:
            return
        existing_id = local.get("stop_client_order_id")
        if existing_id:
            existing = self.broker.get_order_by_client_id(str(existing_id))
            if existing is not None and existing.is_open:
                same_qty = abs(float(existing.requested_qty) - qty) <= _QTY_EPS
                same_stop = existing.stop_price is not None and abs(float(existing.stop_price) - float(stop_price)) <= 0.011
                if same_qty and same_stop:
                    return
                self.broker.cancel_order(existing.broker_order_id)
                refreshed = self.broker.get_order_by_client_id(str(existing_id))
                if refreshed is not None:
                    self.store.record_order(refreshed, self.session_id)
                    if refreshed.is_open:
                        return

        attempt = int(local.get("stop_attempt") or 0) + 1
        client_id = make_client_order_id(self.session_id, "stop", now, symbol, attempt=attempt)
        intent = OrderRecord(
            broker_order_id="",
            client_order_id=client_id,
            symbol=symbol,
            purpose="stop",
            side="sell",
            requested_qty=qty,
            filled_qty=0.0,
            status="intent",
            stop_price=float(stop_price),
            submitted_at=now,
            updated_at=now,
            reference_price=float(position.current_price or position.avg_entry_price),
        )
        self.store.record_order(intent, self.session_id)
        order = self._submit_idempotent(
            client_id,
            lambda: self.broker.submit_stop_order(symbol, qty, float(stop_price), client_id),
        )
        self.store.record_order(order, self.session_id)
        local["stop_client_order_id"] = client_id
        local["stop_attempt"] = attempt
        self.store.set_bot_position(symbol, local)

    def _cancel_stop_for_exit(self, symbol: str) -> bool:
        local = self.store.get_bot_position(symbol)
        if not self._is_ours(local):
            return False
        client_id = local.get("stop_client_order_id")
        if not client_id:
            return True
        order = self.broker.get_order_by_client_id(str(client_id))
        if order is None or order.is_terminal:
            local["stop_client_order_id"] = None
            self.store.set_bot_position(symbol, local)
            return True
        self.broker.cancel_order(order.broker_order_id)
        refreshed = self.broker.get_order_by_client_id(str(client_id))
        if refreshed is not None:
            self.store.record_order(refreshed, self.session_id)
            if refreshed.is_open:
                return False
        local["stop_client_order_id"] = None
        self.store.set_bot_position(symbol, local)
        return True

    def _ratchet_stops(self, strategies, bars, timestamp, positions, now):
        for symbol, local in self._owned_positions(positions).items():
            frame = bars.get(symbol)
            if frame is None or timestamp not in frame.index:
                continue
            owner = next(
                (
                    entry
                    for entry in strategies
                    if entry.strategy_id == local.get("strategy_id")
                    and symbol in getattr(entry.strategy.config, "symbols", ())
                ),
                None,
            )
            if owner is None:
                continue
            prepared = owner.strategy.prepare_symbol(frame)
            row = prepared.loc[timestamp]
            close = float(row.get("close"))
            atr = float(row.get("atr"))
            if not all(math.isfinite(value) and value > 0 for value in (close, atr)):
                continue
            current_stop = float(local["stop_price"])
            highest = float(local.get("highest_close") or local.get("entry_price") or close)
            next_stop, next_highest = owner.strategy.ratchet_stop(current_stop, highest, close, atr)
            next_stop = min(float(next_stop), close * (1.0 - 1e-9))
            local["highest_close"] = max(highest, float(next_highest))
            if next_stop > current_stop + 0.009:
                local["stop_price"] = next_stop
                self.store.set_bot_position(symbol, local)
                self._ensure_stop(symbol, positions[symbol], now)
            else:
                self.store.set_bot_position(symbol, local)

    def _apply_session_loss_gate(self, equity: float, now: datetime) -> None:
        if self._session_halted or not self._starting_equity or self._starting_equity <= 0:
            return
        drawdown = (self._starting_equity - equity) / self._starting_equity * 100.0
        if drawdown < self.config.max_session_loss_pct:
            return
        self._session_halted = True
        reason = (
            f"Session equity drawdown {drawdown:.2f}% reached the "
            f"{self.config.max_session_loss_pct:.2f}% limit."
        )
        self.store.set_session_halted(self.session_id, reason)
        self.store.set_kv(
            EQUITY_MARKETWIDE_KILL_SWITCH_KEY,
            json.dumps({"armed": True, "reason": "DAILY_LOSS_HALT", "at": now.isoformat(), "message": reason}),
        )
        with self._lock:
            self._status.halt_reason = reason
            self._status.state = EngineState.HALTED
            self._persist_status()
        self.store.log("CRITICAL", reason, session_id=self.session_id)

    def _entries_allowed_by_clock(self, account, clock, now: datetime) -> bool:
        if not clock.is_open:
            return False
        seconds_to_close = (clock.next_close - now).total_seconds()
        if seconds_to_close <= self.config.near_close_entry_block_minutes * 60:
            self._warn("Fresh equity entries are blocked near the market close.")
            return False
        if account.equity < _PDT_EQUITY_THRESHOLD and account.daytrade_count >= _PDT_MAX_DAY_TRADES:
            self._warn("Fresh intraday entries are blocked by the conservative PDT gate.")
            return False
        return True

    def _persist_decisions(self, report) -> None:
        payload = []
        for decision in report.decisions[:250]:
            payload.append(
                {
                    "symbol": decision.candidate.symbol,
                    "strategy_id": decision.candidate.strategy_id,
                    "action": decision.candidate.action.value,
                    "outcome": decision.outcome.value,
                    "rank": decision.rank,
                    "net_alpha_bps": decision.expected_net_alpha_bps,
                    "score": decision.opportunity_score,
                    "risk_allocation_pct": decision.risk_allocation_pct,
                    "rejection_reasons": list(decision.rejection_reasons),
                    "signal_reason": decision.candidate.signal_reason,
                }
            )
        self.store.set_kv(EQUITY_MARKETWIDE_LAST_DECISIONS_KEY, json.dumps(payload, sort_keys=True))
        with self._lock:
            self._status.last_executable_candidates = len(report.executable)
            self._persist_status()

    def _heartbeat(self, now: datetime, bar_time: Optional[datetime], positions) -> None:
        account = self.broker.get_account()
        self.store.record_equity(self.session_id, now, account.equity, account.buying_power)
        with self._lock:
            self._status.last_heartbeat = now
            self._status.last_bar_time = bar_time
            self._status.open_bot_positions = len(self._owned_positions(positions))
            self._persist_status()

    def _warn(self, message: str) -> None:
        with self._lock:
            warnings = [message] + [value for value in self._status.warnings if value != message]
            self._status.warnings = warnings[:10]
            self._persist_status()
        self.store.log("WARNING", message, session_id=self.session_id)

    def _persist_status(self) -> None:
        self.store.set_kv(EQUITY_MARKETWIDE_STATUS_KEY, json.dumps(self._status.to_dict(), sort_keys=True))


def clear_marketwide_equity_kill_switch(store: StateStore) -> None:
    """Explicit operator acknowledgement hook used by CLI/UI control surfaces."""
    store.delete_kv(EQUITY_MARKETWIDE_KILL_SWITCH_KEY)
