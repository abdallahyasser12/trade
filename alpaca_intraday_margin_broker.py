"""Current Alpaca Trading API account adapter for the market-wide equity desk.

Alpaca removed the legacy Trading API ``pattern_day_trader`` and ``daytrade_count``
account fields on 2026-07-06 after moving to FINRA's intraday-margin framework. The
repository's older normalized ``AccountInfo`` shape is retained for compatibility, but
this adapter never requires those removed wire fields: it populates the deprecated
compatibility slots with safe inert values and treats current ``buying_power`` plus
broker pre-trade checks as the execution authority.
"""

from __future__ import annotations

from broker import AlpacaPaperBroker, BrokerError, _sanitized_broker_message
from models import AccountInfo


def _numeric(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class AlpacaIntradayMarginPaperBroker(AlpacaPaperBroker):
    """Paper broker compatible with Alpaca's post-PDT Trading API schema."""

    def get_account(self) -> AccountInfo:
        try:
            acct = self._client.get_account()
        except Exception as exc:
            raise BrokerError(_sanitized_broker_message(exc)) from exc
        raw_status = getattr(acct, "status", "")
        status = raw_status.value if hasattr(raw_status, "value") else str(raw_status)
        return AccountInfo(
            equity=_numeric(getattr(acct, "equity", 0.0)),
            cash=_numeric(getattr(acct, "cash", 0.0)),
            buying_power=_numeric(getattr(acct, "buying_power", 0.0)),
            status=status,
            currency=str(getattr(acct, "currency", None) or "USD"),
            # Compatibility-only fields. They were removed from Alpaca's Trading API
            # in July 2026 and must not drive market-wide execution decisions.
            pattern_day_trader=False,
            daytrade_count=0,
            trading_blocked=bool(getattr(acct, "trading_blocked", False)),
        )
