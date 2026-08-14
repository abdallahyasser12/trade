# Regime-Aware Multi-Strategy Crypto Engine v2

## Purpose

This branch adds an opt-in second-generation decision layer for the existing
Alpaca crypto system. It does **not** replace broker reconciliation, order
idempotency, portfolio risk limits, live-mode gating, stop protection, or the
audited causal backtester.

The goal is to improve adaptability across market conditions while keeping the
decision process deterministic, inspectable, and resistant to overfitting. It
does not guarantee profit, win rate, Sharpe ratio, or future performance.

## Architecture

```text
completed aligned bars
        |
CryptoRegimeStrategy (existing audited breakout model)
        |
RegimeAwareEnsembleStrategy
        |-- cross-sectional regime classifier
        |-- breakout sleeve
        |-- trend-pullback sleeve
        |-- guarded mean-reversion sleeve
        `-- deterministic regime weights + confidence gate
        |
        +--> RegimeAwareCryptoTradeEngine
        |       `-- existing CryptoTradeEngine execution / risk / recovery
        |
        +--> run_regime_ensemble_backtest
        |       `-- exact existing crypto backtest code object
        |
        `--> run_regime_ensemble_walk_forward
                `-- exact existing chronological research protocol
```

## Regime classifier

The classifier uses only the latest completed, exactly aligned bars. It combines:

- BTC benchmark structure versus fast, slow, and regime moving averages;
- benchmark regime-MA slope;
- benchmark momentum;
- ADX trend strength;
- Kaufman efficiency;
- ATR percentile;
- cross-sectional breadth: fraction of the configured universe with
  `close > slow_ma` and `fast_ma > slow_ma`;
- cross-sectional momentum dispersion.

It emits one of:

- `BULL_EXPANSION`
- `BULL_NORMAL`
- `RANGE`
- `RISK_OFF`
- `STRESS`
- `UNKNOWN`

`RISK_OFF`, `STRESS`, and `UNKNOWN` cannot open new positions. Existing exits are
never suppressed.

## Strategy sleeves

### Breakout

The existing causal Donchian-breakout strategy remains the primary trend sleeve.
A fresh breakout is still required by that sleeve and its existing filters remain
part of the score.

### Pullback continuation

The pullback sleeve looks for an orderly retracement toward the fast average while
the asset remains above the slow average, the benchmark regime remains bullish,
momentum has not materially failed, and volatility-adjusted distance, RSI, ADX,
efficiency, and relative volume remain acceptable.

It does not average down an existing position. It can only create a new long
position when no position is already held.

### Guarded mean reversion

Mean reversion receives material weight only in `RANGE`, and a small weight in
`BULL_NORMAL`. It requires:

- benchmark regime still bullish;
- asset close not materially below the slow average;
- close below the fast average;
- RSI below the configured ceiling;
- ATR percentile below the stress threshold;
- bounded ATR-normalized deviation;
- optional one-bar rebound evidence.

It is disabled in `BULL_EXPANSION`, `RISK_OFF`, `STRESS`, and `UNKNOWN`.

## Regime routing

Default sleeve weights are:

| Regime | Breakout | Pullback | Mean reversion |
|---|---:|---:|---:|
| Bull expansion | 70% | 30% | 0% |
| Bull normal | 45% | 45% | 10% |
| Range | 10% | 25% | 65% |
| Risk-off / stress / unknown | 0% | 0% | 0% |

A BUY requires both the weighted ensemble score and regime-confidence threshold.

Every emitted decision records the full regime snapshot and sleeve scores in
`CryptoSignalDecision.details["ensemble"]` so paper/live behavior is auditable.

## Execution safety

`RegimeAwareCryptoTradeEngine` subclasses the production
`CryptoTradeEngine` and replaces only its strategy object. The parent engine
continues to own:

- paper/live account-mode verification;
- separate live credentials and live unlock requirements;
- account/asset validation;
- completed/stale-bar enforcement;
- quote age and spread gates;
- session-loss latch;
- ATR/drawdown-aware sizing;
- per-position and aggregate exposure caps;
- maximum position count;
- correlation gate;
- deterministic client order IDs;
- timeout-after-acceptance reconciliation;
- delayed-entry cancellation;
- partial-fill reconciliation;
- protective stop-limit replacement;
- stop watchdog market exits;
- restart recovery and SQLite persistence.

The ensemble cannot bypass those controls.

## Backtest parity

`run_regime_ensemble_backtest` executes the existing
`crypto_backtest.run_crypto_backtest` **code object** with a private copy of its
globals where only the strategy constructor is replaced. It does not monkeypatch
the process-wide module.

Therefore the ensemble uses the same:

- causal completed-bar preparation;
- next-bar fills;
- adverse gap-stop behavior;
- slippage/spread/fee accounting;
- fractional quantity flooring;
- portfolio-risk manager;
- correlation and exposure gates;
- forced end liquidation;
- equity and P&L accounting.

Classic and ensemble backtests can run in parallel without sharing a mutable
strategy binding.

## Walk-forward parity

`run_regime_ensemble_walk_forward` applies the same isolated binding technique to
the existing research function. Candidate construction, train-only selection,
embargo, non-overlapping out-of-sample folds, multiple-testing penalty, stressed
costs, fixed baseline, and hard promotion gates remain unchanged.

A PASS still means only “eligible for controlled forward paper testing.”

## Activation

### Backtest

```python
from advanced_crypto_backtest import run_regime_ensemble_backtest
from regime_ensemble import EnsembleConfig

result = run_regime_ensemble_backtest(
    bars_by_symbol,
    strategy_config,
    risk_config,
    backtest_config,
    ensemble_config=EnsembleConfig(),
)
```

### Walk-forward

```python
from advanced_crypto_research import run_regime_ensemble_walk_forward

result = run_regime_ensemble_walk_forward(
    bars_by_symbol,
    strategy_config,
    risk_config,
    backtest_config,
    parameter_grid,
    walk_config,
    criteria,
)
```

### Paper/live engine

```python
from advanced_crypto_engine import RegimeAwareCryptoTradeEngine

engine = RegimeAwareCryptoTradeEngine(
    broker,
    data_feed,
    store,
    strategy_config,
    risk_config,
    engine_config,
)
```

Paper remains the recommended first deployment. Live execution keeps every
existing live-mode gate and should not be enabled merely because an ensemble
backtest or walk-forward run passes.

## Promotion sequence

1. Compare classic and ensemble results on identical data/cost assumptions.
2. Run the existing chronological walk-forward gates through the ensemble adapter.
3. Reject the ensemble if it only improves training performance.
4. Stress costs and confirm drawdown and trade-frequency stability.
5. Forward paper trade through materially different volatility/trend regimes.
6. Compare expected versus realized slippage, fill behavior, regime labels,
   sleeve selection, drawdown, and stop behavior.
7. Promote to live only under the repository's existing explicit live gates and
   conservative caps.

## Deliberate boundaries

This implementation does not use:

- LLM-generated orders;
- mutable online learning;
- martingale;
- grid averaging;
- leverage;
- shorting;
- automatic parameter promotion;
- future-bar data;
- news sentiment as an execution trigger.

Those exclusions are intentional safety and overfitting controls.
