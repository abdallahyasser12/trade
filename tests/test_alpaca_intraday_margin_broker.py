from types import SimpleNamespace

from alpaca_intraday_margin_broker import AlpacaIntradayMarginPaperBroker


class FakeClient:
    def get_account(self):
        # Intentionally omit deprecated PDT/day-trade fields: Alpaca removed these
        # from the Trading API account schema in July 2026.
        return SimpleNamespace(
            equity="12345.67",
            cash="4567.89",
            buying_power="9135.78",
            status=SimpleNamespace(value="ACTIVE"),
            currency="USD",
            trading_blocked=False,
        )


def test_current_alpaca_account_schema_does_not_require_removed_pdt_fields():
    broker = object.__new__(AlpacaIntradayMarginPaperBroker)
    broker._client = FakeClient()

    account = broker.get_account()

    assert account.equity == 12345.67
    assert account.cash == 4567.89
    assert account.buying_power == 9135.78
    assert account.can_trade is True
    assert account.pattern_day_trader is False
    assert account.daytrade_count == 0
