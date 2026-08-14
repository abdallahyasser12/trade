# Production Crypto Extension Plan

## Objective and evidence standard

This extension adds a deterministic, long-only Alpaca spot-crypto system without
weakening the audited stock paper bot. It is designed for controlled paper testing
first. Live-money execution is an explicitly unlocked deployment mode, not a claim
of profitability or a substitute for independent operational review.

A configuration is eligible for paper testing only after chronological,
next-bar-execution backtests and cost-stressed walk-forward gates pass. A good
historical result never unlocks live trading automatically.

## Runtime architecture

```text
Streamlit
  |-- Stock workspace (existing audited paper-only engine)
  `-- Crypto workspace
        |-- immutable AccountMode + Crypto*Config snapshot
        |-- CryptoRegimeStrategy (shared by live and backtest)
        |-- CryptoRiskManager (shared by live and backtest)
        |-- CryptoBacktester / CryptoTradeEngine
        |-- AlpacaCryptoDataFeed
        |-- AlpacaBroker (exact paper/live endpoint verification)
        `-- StateStore (orders, fills, processed bars, positions, latches)
```

The strategy never imports Alpaca. The backtester never imports the broker. Live
execution and simulation differ only in execution mechanics; signal preparation,
regime classification, ranking, initial-stop arithmetic, and sizing are shared.

## Decision model

- Universe: a validated, USD-quoted allowlist; defaults to BTC/USD, ETH/USD, and
  SOL/USD. Every symbol must be active, tradable, fractionable crypto at Alpaca.
- Data: completed bars only. A bar stamped `T` is usable only after
  `T + timeframe + publication tolerance`. Missing or stale data blocks entries.
- Regime: long exposure is allowed only when the BTC benchmark is above its slow
  average and that average is rising. High ATR percentile is a stress regime and
  blocks new exposure.
- Entry: a new Donchian breakout transition must be confirmed by the fast/slow
  trend, RSI band, ADX, Kaufman efficiency, positive momentum, non-zero relative
  volume, benchmark regime, and relative strength for non-BTC assets.
- Ranking: simultaneous candidates are ordered by a deterministic composite of
  relative momentum, ADX, path efficiency, volume confirmation, and inverse
  volatility. No LLM, news text, or mutable online learning can place an order.
- Exit: trend failure or regime stress schedules a next-bar exit. ATR initial and
  completed-bar trailing stops can only tighten.

## Portfolio and execution controls

- Long-only spot; no margin, leverage, borrowing, shorting, martingale, grid
  averaging, or averaging down.
- Fractional quantity is floored to Alpaca's asset-specific trade increment and
  must exceed both the dynamic minimum size and configured minimum notional.
- Quantity is the minimum allowed by risk-at-stop, per-position exposure, remaining
  portfolio exposure, cash/buying power, drawdown scaling, volatility scaling, and
  the live per-order notional cap.
- Correlation and maximum-position gates prevent a nominally diversified portfolio
  from becoming one concentrated crypto-beta trade.
- The session-loss latch is irreversible for that engine session; exits and
  protection continue while entries remain disabled.
- Market orders use GTC, as required for Alpaca crypto. Protective orders use a GTC
  sell stop-limit with the limit below the stop. Because a stop-limit can remain
  unfilled through a gap, the live watchdog reconciles broker truth, cancels only
  the bot-owned stale protective order, confirms cancellation/terminal state, and
  submits one deterministic-id market exit. It never sizes a sell above the
  broker-reported long quantity.
- Submission timeouts are reconciled by deterministic client order ID before any
  retry. Partial fills immediately become authoritative and protection is resized
  to the broker position; replacement always cancels and confirms the old order
  first.
- A delayed GTC entry is canceled and confirmed after a bounded timeout so an old
  signal cannot create exposure much later; a raced fill is reconciled before the
  next decision.

## Live-account safety boundary

Paper is the default. Live mode requires all of the following:

1. `ENABLE_LIVE_TRADING=true` in the process environment.
2. Separate `ALPACA_LIVE_API_KEY` and `ALPACA_LIVE_SECRET_KEY`; paper variables are
   never reused or silently promoted.
3. No active engine while switching account mode.
4. A checked risk acknowledgement and exact typed phrase `LIVE` in the current
   Streamlit process. The unlock is cleared on mode change and process restart.
5. Exact SDK base URL equality with `https://api.alpaca.markets`, an ACTIVE,
   unblocked account, and successful crypto asset verification before Start.
6. Hard live order-notional and total-exposure caps, with absolute code ceilings of
   $10,000 per buy and 20% total exposure. These are enforced in the risk manager
   and the per-order cap is checked again at the broker boundary.

There is no URL override and no environment variable that can redirect an Alpaca
client to a third-party or unintended endpoint. The UI always renders a red live
money banner when live mode is selected.

## Simulation and promotion acceptance gates

- Signals use data through bar N; fills occur no earlier than bar N+1 open.
- Normal stop: stop price less sell slippage. Gap below stop: gap open less sell
  slippage; the simulator never grants an impossible stop price.
- Cash is debited by entry cost plus percentage fee and credited by exit proceeds
  minus fee. Equity is cash plus current open-position market value exactly once.
- Walk-forward selection uses training data only, an embargo, non-overlapping test
  folds, a bounded grid, a fixed baseline, and stressed fees/slippage/spread.
- Promotion requires minimum folds/trades, positive-fold ratio, return above the
  fixed baseline, Sharpe, profit factor, drawdown, and stressed-return gates.
- The dashboard displays the gate result but never applies a selected candidate to
  a running engine.

## Verification matrix

Automated tests cover validation and credential isolation; exact endpoint checks;
fractional flooring; zero/negative values; completed/stale bars; causal breakout and
benchmark alignment; repeated breakout suppression; portfolio exposure and
correlation; next-bar fills; gap stops; accounting; duplicate bars; timeout-after-
acceptance; partial fills; stop replacement/cancel races; restart reconciliation;
session-loss latching; controller single-engine behavior; and disabled-by-default
live UI rendering.

No test or smoke check in this repository places a live order. Read-only account and
market-data checks are permitted; any economic order requires an explicit user start
after all runtime gates pass.
