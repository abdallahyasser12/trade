"""Shadow executor: virtual decisions and fills for non-deployed strategies.

A strategy in SHADOW state trades nothing, but its decisions still become
virtual positions with virtual fills, stops, and attributions (spec §25).
That record is the evidence the lifecycle ladder demands before PAPER —
and it is how a demoted strategy proves its edge returned.

Honesty rules:

* virtual fills pay the same modeled costs the backtester charges
  (slippage + half spread as price adjustment, fee on notional);
* virtual equity is a FIXED notional, deliberately non-compounding —
  shadow measures decision quality per trade, not wealth paths;
* excursions are close-based (the shadow book sees completed closes);
* every attribution row is marked ``shadow=True`` so no marketplace
  statistic can ever mix virtual and real fills by accident.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Optional

import pandas as pd

from crypto_models import CryptoSignalDecision
from models import BotAction
from quant.contracts import AttributionRecord, Market
from quant.store import QuantStore
from quant.strategies.base import UniverseStrategy

BPS = 10_000.0


@dataclass(frozen=True)
class ShadowConfig:
    virtual_equity: float = 100_000.0
    risk_per_trade_pct: float = 0.5
    slippage_bps: float = 10.0
    spread_bps: float = 10.0
    fee_bps: float = 25.0

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not math.isfinite(self.virtual_equity) or self.virtual_equity <= 0:
            errors.append("Virtual equity must be finite and positive.")
        if not (0 < self.risk_per_trade_pct <= 2):
            errors.append("Shadow risk per trade must be in (0, 2]%.")
        for label, value in {
            "slippage": self.slippage_bps,
            "spread": self.spread_bps,
            "fee": self.fee_bps,
        }.items():
            if not math.isfinite(value) or value < 0 or value > 2_000:
                errors.append(f"Shadow {label} must be within [0, 2000] bps.")
        return errors


class ShadowExecutor:
    """Maintains virtual positions per (strategy, symbol) in the quant store."""

    def __init__(self, store: QuantStore, config: Optional[ShadowConfig] = None):
        self.store = store
        self.config = config or ShadowConfig()
        errors = self.config.validate()
        if errors:
            raise ValueError("Invalid shadow configuration: " + "; ".join(errors))

    def held_symbols(self, strategy_id: str) -> frozenset[str]:
        return frozenset(
            row["symbol"] for row in self.store.get_shadow_positions(strategy_id)
        )

    def process_cycle(
        self,
        *,
        strategy_id: str,
        strategy_version: str,
        strategy: UniverseStrategy,
        market: Market,
        timeframe: str,
        decisions: list[CryptoSignalDecision],
        prepared: Mapping[str, pd.DataFrame],
        timestamp: pd.Timestamp,
    ) -> dict[str, int]:
        """Apply one completed bar to the strategy's virtual book.

        Order mirrors the real engines: stop checks first (against this
        bar's completed close — the shadow book has no intrabar data),
        then strategy exits, then stop ratchets, then entries.
        """
        cfg = self.config
        buy_cost = 1.0 + (cfg.slippage_bps + cfg.spread_bps / 2.0) / BPS
        sell_cost = 1.0 - (cfg.slippage_bps + cfg.spread_bps / 2.0) / BPS
        fee_rate = cfg.fee_bps / BPS
        counts = {"entries": 0, "exits": 0, "stop_exits": 0, "updates": 0}

        decisions_by_symbol = {decision.symbol: decision for decision in decisions}
        positions = {
            row["symbol"]: row for row in self.store.get_shadow_positions(strategy_id)
        }

        for symbol, row in positions.items():
            frame = prepared.get(symbol)
            if frame is None or timestamp not in frame.index:
                continue
            bar = frame.loc[timestamp]
            close = float(bar["close"])
            atr_value = float(bar["atr"]) if "atr" in frame.columns else math.nan
            if not math.isfinite(close) or close <= 0:
                continue
            highest = max(float(row["highest_close"]), close)
            lowest = min(float(row["lowest_close"]), close)

            decision = decisions_by_symbol.get(symbol)
            stop_breached = close <= float(row["stop_price"])
            strategy_exit = decision is not None and decision.action is BotAction.EXIT
            if stop_breached or strategy_exit:
                reason = "shadow_stop_hit" if stop_breached else decision.reason
                exit_price = (
                    float(row["stop_price"]) if stop_breached else close
                ) * sell_cost
                self._close(
                    strategy_id,
                    strategy_version,
                    market,
                    timeframe,
                    row,
                    qty=float(row["qty"]),
                    exit_price=exit_price,
                    fee_rate=fee_rate,
                    reason=reason,
                    highest=highest,
                    lowest=lowest,
                    timestamp=timestamp,
                )
                counts["stop_exits" if stop_breached else "exits"] += 1
                continue

            new_stop = float(row["stop_price"])
            if math.isfinite(atr_value) and atr_value > 0:
                new_stop, highest = strategy.ratchet_stop(
                    new_stop, highest, close, atr_value
                )
            updated = dict(row)
            updated.update(
                stop_price=new_stop, highest_close=highest, lowest_close=lowest
            )
            self.store.upsert_shadow_position(updated)
            counts["updates"] += 1

        held = self.held_symbols(strategy_id)
        for decision in decisions:
            if decision.action is not BotAction.BUY or decision.symbol in held:
                continue
            close = decision.close
            stop = decision.stop_price
            if (
                close is None
                or stop is None
                or not math.isfinite(close)
                or not math.isfinite(stop)
                or stop <= 0
                or stop >= close
            ):
                continue
            fill = close * buy_cost
            distance = close - stop
            qty = (cfg.virtual_equity * cfg.risk_per_trade_pct / 100.0) / distance
            entry_fee = qty * fill * fee_rate
            self.store.upsert_shadow_position(
                {
                    "strategy_id": strategy_id,
                    "symbol": decision.symbol,
                    "qty": qty,
                    "entry_price": fill,
                    "entry_time": (
                        decision.timestamp or timestamp.to_pydatetime()
                    ).isoformat(),
                    "stop_price": fill - distance,
                    "initial_stop_price": fill - distance,
                    "highest_close": close,
                    "lowest_close": close,
                    "entry_fees": entry_fee,
                    "signal": decision.to_dict(),
                }
            )
            counts["entries"] += 1
        return counts

    def _close(
        self,
        strategy_id: str,
        strategy_version: str,
        market: Market,
        timeframe: str,
        row: dict,
        *,
        qty: float,
        exit_price: float,
        fee_rate: float,
        reason: str,
        highest: float,
        lowest: float,
        timestamp: pd.Timestamp,
    ) -> None:
        entry_price = float(row["entry_price"])
        exit_fee = qty * exit_price * fee_rate
        total_fees = float(row["entry_fees"] or 0.0) + exit_fee
        gross = (exit_price - entry_price) * qty
        net = gross - total_fees
        initial_stop = float(row["initial_stop_price"])
        risk = (entry_price - initial_stop) * qty
        entry_regime = None
        signal_json = row.get("signal_json")
        if signal_json:
            try:
                import json

                entry_regime = json.loads(signal_json).get("regime")
            except (ValueError, AttributeError):
                entry_regime = None
        self.store.record_attribution(
            AttributionRecord(
                strategy_id=strategy_id,
                strategy_version=strategy_version,
                symbol=row["symbol"],
                market=market,
                timeframe=timeframe,
                session_id=None,
                signal_timestamp=datetime.fromisoformat(row["entry_time"]),
                entry_timestamp=datetime.fromisoformat(row["entry_time"]),
                exit_timestamp=timestamp.to_pydatetime()
                if hasattr(timestamp, "to_pydatetime")
                else datetime.now(timezone.utc),
                qty=qty,
                entry_price=entry_price,
                exit_price=exit_price,
                initial_stop_price=initial_stop,
                fees=total_fees,
                expected_cost_bps=None,
                realized_slippage_bps=None,
                gross_pnl=gross,
                net_pnl=net,
                r_multiple=net / risk if risk > 0 else None,
                max_favorable_excursion_pct=(
                    max(0.0, (highest - entry_price) / entry_price * 100.0)
                    if entry_price > 0
                    else None
                ),
                max_adverse_excursion_pct=(
                    max(0.0, (entry_price - lowest) / entry_price * 100.0)
                    if entry_price > 0
                    else None
                ),
                entry_reason="shadow_entry",
                exit_reason=reason,
                regime_at_entry=entry_regime,
                shadow=True,
            )
        )
        self.store.delete_shadow_position(strategy_id, row["symbol"])
