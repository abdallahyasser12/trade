# Production Plan — Regime-Aware Trend Strategy

Status: implemented and covered by automated regression tests.

Scope: the US-equity workspace, which remains Alpaca paper trading only. The
separate crypto extension is specified in `CRYPTO_TRADING_PLAN.md`. The objective
of this stock plan is to improve selectivity,
execution integrity, and risk-adjusted robustness. It is not to maximize an
in-sample equity curve, predict prices, or claim guaranteed profit.

## 1. Safety and decision invariants

These constraints are part of the design, not optional tuning choices:

- The broker remains hard-wired to Alpaca paper trading. Research never submits
  an order.
- A session trades one user-selected symbol, long-only, with whole shares and no
  leverage introduced by the strategy.
- A BUY still requires one fresh bullish crossover:
  `previous_short <= previous_long AND current_short > current_long`.
- A bearish crossover still exits. A new bullish-to-bearish benchmark regime
  transition may also exit when enabled.
- Live, backtest, and research call the same `TrendStrategy` and `RiskManager`.
  `research.py` contains no alternate signal or sizing implementation.
- Every decision uses completed bars only. A signal on bar N can fill no earlier
  than bar N+1 in the simulator.
- Separate benchmark bars become usable only after their bar duration has
  elapsed. Same-timeframe symbol and benchmark returns must share exact
  timestamps.
- Missing, stale, unsorted, duplicated, misaligned, or insufficient benchmark
  data fails closed: no new entry.
- Runtime configuration is frozen at engine start. Walk-forward output is never
  auto-applied to a running or future paper session.
- Risk adaptation may reduce the configured risk budget but can never increase
  it above 1.0 times the base budget.
- A protective stop may tighten but never loosen. The broker stop is replaced
  only after cancellation is confirmed and always matches broker-truth position
  quantity.

## 2. Signal architecture

The fresh moving-average crossover is the trigger. Confirmation filters answer
whether the new trend is strong and orderly enough to enter:

1. **RSI band:** `entry_floor <= RSI < overbought`. This rejects both weak
   momentum and an already stretched move.
2. **Volume:** `current_volume / rolling_average_volume >= minimum_ratio`.
3. **Volatility:** the causal rolling percentile of ATR must not exceed its
   configured ceiling.
4. **ADX:** Wilder-smoothed directional movement must produce
   `ADX >= minimum_ADX`. ADX measures trend strength, not direction.
5. **Kaufman efficiency ratio:**
   `abs(close_N - close_(N-p)) / sum(abs(one_bar_changes))` must meet its floor.
   It approaches 1 for a clean directional path and 0 for noise.
6. **Broad-market regime:** the completed benchmark close must be above its SMA,
   and `SMA_N - SMA_(N-slope_period) > 0`.
7. **Optional relative strength:** over identical timestamps,
   `symbol_period_return - benchmark_period_return >= minimum`.

Every enabled entry filter must pass. This deliberately favors interpretable,
orthogonal evidence over a large feature set or opaque model. Momentum can suffer
sharp regime-dependent crashes, so the market-regime and volatility controls are
risk defenses rather than claims of higher returns. See the NBER paper
[Momentum Crashes](https://www.nber.org/papers/w20439).

## 3. Risk, sizing, and exits

The initial protective stop is based on the actual entry fill:

`initial_stop = actual_fill - ATR_at_entry * initial_ATR_multiplier`

Base sizing remains the minimum of risk, exposure, and buying-power caps:

```text
risk_budget       = equity * risk_per_trade_pct * risk_scale
risk_per_share    = entry - stop
shares_by_risk    = floor(risk_budget / risk_per_share)
shares_by_exposure= floor(equity * max_position_pct / entry)
shares_by_cash    = floor(buying_power / entry)
quantity          = min(all three caps)
```

Adaptive risk uses the harsher of two linear reductions:

- peak-equity drawdown, beginning at a soft threshold and reaching the minimum
  scale at the configured session-loss threshold;
- ATR percentile above the configured volatility target, reaching the minimum
  scale at the 100th percentile.

The scale is clamped to `[minimum_risk_scale, 1]`. Invalid account state fails to
the minimum scale. The existing hard session-loss circuit breaker remains
authoritative.

After each completed bar, the optional ratchet candidate is:

`highest_completed_close - current_ATR * trailing_ATR_multiplier`

The next stop is the maximum of the initial stop, current stop, trailing
candidate, and—after the configured R multiple is reached—the entry price. A
level calculated from bar N is eligible only from bar N+1. If the completed
close is already at or below the candidate, the engine follows its cancel-stop,
confirm, and strategy-exit lifecycle instead of submitting an invalid stop above
the market.

## 4. Chronological research protocol

The Walk-Forward Research tab is a promotion gate, not an optimizer that chooses
what to trade automatically.

For every fold:

1. Use a fixed-length historical training window.
2. Evaluate a deterministic, allow-listed grid capped at 64 candidates in the
   UI and 256 in the library.
3. Reject invalid candidates and candidates with too few training trades.
4. Select one candidate using training data only. The score combines Sharpe,
   return-to-drawdown, capped profit factor, a drawdown penalty, and an explicit
   multiple-testing penalty.
5. Skip at least one full embargo bar.
6. Evaluate the selected candidate once on the next untouched test window.
7. Evaluate the fixed current configuration on the same window as a baseline.
8. Re-run the selected candidate with worse slippage and commissions.
9. Force fold-end liquidation so separate folds do not hide open risk or share
   state.
10. Advance by at least the test length, so out-of-sample windows never overlap.

Warmup rows may precede a fold but cannot trade or contribute to fold equity.
Indicators, signals, sizing, next-bar fills, gap stops, trailing stops, costs,
and P&L all flow through the production backtester.

This constrained design addresses the selection-bias problem described in
[The Probability of Backtest Overfitting](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253)
and the dangers of repeated trial-and-error described in
[Pseudo-Mathematics and Financial Charlatanism](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2308659).

## 5. Promotion gates

A result is marked PASS only when every configured gate passes:

- minimum number of non-overlapping out-of-sample folds;
- minimum number of closed out-of-sample trades;
- minimum fraction of profitable folds;
- positive compounded out-of-sample return;
- minimum excess return versus the fixed baseline;
- minimum aggregate out-of-sample Sharpe ratio using the configured risk-free
  rate;
- minimum aggregate profit factor;
- maximum permitted drawdown in any fold; and
- positive compounded return under stressed transaction costs.

A weighted score cannot compensate for a failed gate. The UI lists every
failure. Changing any strategy, risk, cost, date, feed, grid, or protocol input
hides the prior result until the experiment is run again.

PASS means only “eligible for controlled forward paper testing.” It does not
mean profitable, statistically proven, or suitable for live money.

## 6. Operational rollout

The accepted operating sequence is:

1. Run a normal causal backtest with realistic nonzero costs.
2. Run chronological walk-forward validation over enough history for at least
   three complete out-of-sample folds.
3. Reject the candidate if any promotion gate fails.
4. If all gates pass, freeze that configuration and run supervised Alpaca paper
   trading through multiple market regimes for at least 6–12 weeks.
5. Compare paper fills, slippage, trade frequency, drawdown, stop behavior, and
   filter reasons against the backtest assumptions.
6. Stop or revert on reconciliation errors, unexpected exposure, gate-breaking
   drawdown, or persistent paper/backtest divergence.

This stock subsystem intentionally has no live-money deployment stage.
Multi-symbol stock execution, online learning, news models, and automatic
parameter promotion remain deliberate safety boundaries, not unfinished stock
work. The repository's crypto workspace is a separate implementation with its
own portfolio exposure, correlation, fractional order, recovery, and gated-live
design; it does not redirect or relax this paper-only stock engine.

## 7. Acceptance evidence

The implementation is accepted only when all of the following are green:

- compile every Python module;
- run the full automated suite;
- deterministic tests for filter pass/fail, fresh-cross behavior, benchmark
  causality/alignment, regime exits, ADX/efficiency warmup, adaptive sizing,
  trailing-stop timing, stop replacement, lost-response idempotency, partial
  fills, and chronological folds;
- Streamlit empty-state smoke test with no credentials;
- no paper order required for validation; and
- no secret, database, virtual environment, or generated cache tracked.

Historical profitability and forward paper profitability are separate empirical
questions. They must be reported as not established until real data and elapsed
paper-trading evidence satisfy the gates above.
