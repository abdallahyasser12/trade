# Trend & Crypto Bot — Alpaca

A deterministic, long-only trading research and execution application with a
Streamlit dashboard, Alpaca `alpaca-py`, chronological backtests, walk-forward
promotion gates, SQLite recovery state, and background engines.

The repository contains two deliberately separate workspaces plus a
cross-market research layer:

- **US stocks:** the independently audited moving-average trend bot remains
  **Alpaca paper-only**.
- **Spot crypto:** a 24/7 multi-asset regime/breakout system defaults to paper and
  exposes an explicitly gated live-account mode. Live mode uses real money and is
  disabled in a normal installation.
- **Quant desk (`quant/`):** a multi-market strategy registry, probabilistic
  regime engine, opportunity ranker, portfolio-heat governor, execution-cost
  gate, shadow executor, drift monitors, and an evidence-based strategy
  lifecycle (RESEARCH → SHADOW → PAPER → LIVE_CANARY → LIVE) with its own
  walk-forward research factory (Monte Carlo stress + deflated-Sharpe
  multiple-testing control). It journals every decision — including
  rejections and abstentions — and never promotes a strategy into a live
  state without recorded evidence plus an explicit operator acknowledgement.
  Architecture, methodology, and the honest validation status are in
  [`QUANT_DESK.md`](QUANT_DESK.md). All new strategies ship in RESEARCH
  state and deploy no capital.

No strategy is guaranteed to make money. Historical or paper performance is not a
forecast. Crypto can move sharply, liquidity can disappear, and Alpaca crypto
stop-limit orders can remain unfilled through a gap. This software is not investment
advice and is not described as live-money production-ready merely because live API
access exists.

An audit of every subsystem against professional systematic-desk practice, with
a ranked and sequenced improvement plan, is in
[`ELITE_TRADER_ROADMAP.md`](ELITE_TRADER_ROADMAP.md).

The implemented stock design is documented in
[`SMART_STRATEGY_PLAN.md`](SMART_STRATEGY_PLAN.md). The crypto architecture, safety
boundary, execution model, and acceptance gates are documented in
[`CRYPTO_TRADING_PLAN.md`](CRYPTO_TRADING_PLAN.md).

To run this on an iPhone with an always-on backend, follow
[`GET_THE_APP.md`](GET_THE_APP.md): `scripts/deploy_backend.sh` starts and verifies
the server side, then either the **Build unsigned iOS IPA** workflow (free Apple
ID sideloading, no paid membership, no push notifications, 7-day re-signing) or
the **Build production iOS app** workflow (paid Apple Developer account, signed
install link or TestFlight) produces the app itself.

## Installation

Requirements:

- Python 3.11 or newer
- Alpaca paper credentials for paper order execution
- Separate Alpaca live credentials only if you deliberately enable crypto live mode

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
streamlit run app.py
```

Run verification with:

```bash
python -m compileall .
python -m pytest -q
```

Do not commit `.env`, SQLite databases, API keys, virtual environments, or generated
caches.

## Credentials and account modes

Paper keys use the namespaced variables below. The legacy paper names remain supported
so existing installations continue to work:

```dotenv
ALPACA_PAPER_API_KEY=
ALPACA_PAPER_SECRET_KEY=

# Legacy paper aliases:
ALPACA_API_KEY=
ALPACA_SECRET_KEY=
```

Crypto live mode requires all three separate settings:

```dotenv
ENABLE_LIVE_TRADING=true
ALPACA_LIVE_API_KEY=
ALPACA_LIVE_SECRET_KEY=
```

Paper credentials are never used as live credentials. The broker accepts no URL
override. Before Start, it compares the SDK's base URL for exact equality with either
`https://paper-api.alpaca.markets` or `https://api.alpaca.markets`, then requires an
ACTIVE, unblocked account.

The Streamlit live switch is intentionally more than one click. It requires the
deployment flag, separate keys, an inactive engine, a checked risk acknowledgement,
the exact typed phrase `LIVE`, account verification, crypto asset verification, and
hard live notional/exposure caps. The unlock exists only in the current Streamlit
process and is cleared on mode change or restart. A red **LIVE ACCOUNT — REAL MONEY**
banner is always rendered in that mode.

## Recommended operating sequence

1. Start in Paper mode.
2. Configure a small USD crypto universe and realistic nonzero fees, spread, and
   slippage.
3. Run the causal portfolio backtest over enough history for indicator warmup.
4. Inspect individual trades, drawdown, exposure time, fee burden, and the BTC
   benchmark—not only total return.
5. Run bounded chronological walk-forward research with at least three non-overlapping
   test folds.
6. Reject the configuration if any promotion gate fails.
7. If every gate passes, freeze it and run supervised paper trading through different
   market regimes. Compare observed fills and reconciliation behavior with simulation.
8. Keep live mode disabled until independent review, operational monitoring, and a
   personally acceptable loss budget are in place.

Walk-forward PASS means only “eligible for controlled paper testing.” The application
never copies a selected candidate into a running engine or unlocks live trading.

## Runtime architecture

```text
Streamlit
  |-- Stock configuration -> TrendStrategy + RiskManager
  |                         -> Backtester OR paper TradeEngine
  |                         -> AlpacaPaperBroker / StockDataFeed
  |
  `-- Crypto configuration + frozen AccountMode
                            -> CryptoRegimeStrategy + CryptoRiskManager
                            -> CryptoBacktester / WalkForward / CryptoTradeEngine
                            -> AlpacaBroker / CryptoDataFeed

All engines -> StateStore -> SQLite -> dashboard state/logs
```

The process-level `EngineController` permits at most one active background engine, so
Streamlit reruns cannot create a second stock or crypto loop. Start is idempotent and
polling/backoff waits are interruptible.

Exclusion also holds **across processes**: stock-engine startup takes an OS-level lock
on a sibling of the SQLite file and holds it until shutdown, so a second app instance —
or an engine orphaned by a source reload — refuses to trade against the same state
database. The lock dies with its process, so a crash can never leave it stuck.

After an emergency stop, daily-loss halt, or fatal error the kill switch is enforced by
the stock engine itself, not only by the dashboard's Start button: startup refuses until
the unclean stop is acknowledged.

## Stock strategy and execution

The stock workspace remains the audited deterministic system:

- A BUY needs a **fresh** crossover: previous short MA <= previous long MA and current
  short MA > current long MA.
- EXIT is the exact mirror: previous short MA >= previous long MA and current short MA
  < current long MA.
- RSI, volume, ATR percentile, ADX, Kaufman efficiency, broad-market regime, and
  relative strength provide causal confirmation.
- Risk sizing is the minimum of risk-at-stop, exposure, and buying-power shares.
- Fills are reconciled before local position state becomes authoritative; partial fills
  are protected at broker-reported quantity.
- Stock decisions use completed regular-session bars and Alpaca's market clock. New
  entries fail closed on market-status, stale-data, benchmark, or clock-skew errors.

The stock trading client is always constructed with `paper=True`; the crypto account
switch cannot redirect the stock engine to live money.

## Crypto analyst model

The crypto system is deterministic and long-only spot. It does not use leverage,
shorting, martingale, grid averaging, averaging down, mutable online learning, an LLM,
or news sentiment to place orders.

For each configured USD pair it calculates, from completed bars only:

- fast, slow, and BTC benchmark regime moving averages;
- RSI with Wilder smoothing;
- ATR with Wilder smoothing and a causal ATR percentile;
- ADX with Wilder smoothing;
- Kaufman path-efficiency ratio;
- relative volume, momentum, and BTC-relative strength;
- a prior-window Donchian high computed with `close.shift(1)`.

A long entry requires a **new** breakout transition plus fast/slow trend, rising BTC
regime, RSI band, ADX, efficiency, nonzero-volume confirmation, positive momentum,
volatility, and relative-strength filters. Simultaneous candidates receive a stable
cross-sectional score and symbol tie-break. Remaining in an uptrend alone does not emit
BUY repeatedly.

Held positions exit on fast-trend failure, benchmark-regime failure, or volatility
stress. Initial stop arithmetic is based on the actual fill and signal-bar ATR. A
completed-bar trailing stop can only tighten and first applies to a future bar.

## Crypto portfolio risk

Fractional sizing is floored to Alpaca's asset-specific `min_trade_increment` and must
also meet the dynamic `min_order_size` and configured minimum notional. Quantity is the
minimum allowed by:

```text
risk budget / (entry - stop)
per-position exposure / entry
remaining total portfolio exposure / entry
cash and buying power / entry
live order-notional cap / entry          # live buys only
```

Risk is scaled down as peak-equity drawdown or ATR percentile rises; it never scales
above the configured base. Maximum concurrent positions and a trailing-return
correlation gate reduce concentration. Missing correlation evidence fails closed.

The session-loss latch triggers at the exact configured threshold and cannot silently
reactivate if equity later rebounds. It blocks new entries while reconciliation,
protective stops, and exits continue.

## Crypto order lifecycle and recovery

Alpaca supports crypto market, limit, and stop-limit orders with GTC/IOC time in force.
This implementation uses fractional GTC market orders and GTC sell stop-limit protection.

- Submitting an order is never treated as a fill.
- Every order carries a deterministic, <=48-character `trendbot-` client ID containing
  a sanitized symbol, purpose, session namespace, and bar/attempt identity.
- If submission times out after broker acceptance, the engine queries that exact client
  ID before doing anything else. Reprocessing the same bar reuses the same economic ID.
- A GTC entry that remains incomplete beyond the configured timeout is canceled and
  confirmed so it cannot fill long after the signal; any raced partial fill is then
  reconciled and protected.
- Broker-reported fill quantity and average price are authoritative. Incremental fill
  events are persisted idempotently.
- A partially filled BUY is protected at the filled broker quantity. If more shares
  fill, the old bot-owned stop is canceled and confirmed before a correctly sized stop
  replaces it.
- Before a strategy market EXIT, the engine cancels and confirms its protective stop,
  then re-reads position quantity. A stop fill racing with cancellation therefore
  reduces or eliminates the exit instead of creating an oversized sell or short.
- The bot lists/cancels only orders in its own `trendbot-` namespace and cancels them
  individually. It never performs account-wide cancellation.

A stop-limit can trigger but fail to execute if price moves below its limit. The 24/7
watchdog records the first breach, waits the configured grace period, refreshes the
order, cancels and confirms only that bot-owned stop, re-reads broker quantity, and
submits one deterministic market exit for the remaining long quantity. That reduces
risk but cannot guarantee a price or protect against an Alpaca/API/network outage.

On restart, broker truth wins:

- persisted position + matching broker position/stop is reconciled and retained;
- a broker-flat symbol clears local position state and cancels an orphan bot stop;
- a broker position without local state is adopted only when refreshed bot-owned fills
  prove that exact net quantity;
- an unexplained/manual position causes startup to fail rather than being traded;
- a pending order is refreshed by its deterministic client ID before another economic
  intent is considered.

## Historical simulation

Both stock and crypto backtests enforce the same causality rule:

```text
bar N closes
-> indicators and signal use data <= N
-> execution occurs no earlier than bar N+1 open
```

For a long stop `S`, if the next bar opens at or below `S`, the simulated fill is the
adverse open after sell costs—not `S`. Otherwise an intrabar low touching `S` fills at
`S` after sell costs. Trailing levels calculated after bar N cannot trigger against bar
N's low.

The crypto simulator models cash, fractional quantity, directional slippage, half-spread
on each side, percentage taker fees, multiple open positions, and forced final
liquidation. Equity is `cash + market value(open positions)` exactly once. Realized P&L
subtracts entry and exit fees exactly once.

The walk-forward protocol selects a bounded candidate grid using training data only,
skips an embargo bar, evaluates non-overlapping test windows, compares the fixed
baseline, and reruns with stressed costs. Promotion has independent gates for fold/trade
count, positive-fold ratio, compounded and excess return, Sharpe, profit factor,
drawdown, and stressed return. One score cannot hide a failed gate.

## Persistence and security

`StateStore` uses SQLite WAL mode, a shared connection guarded by an `RLock`, short
transactions with rollback, timezone-aware ISO timestamps, and uniqueness constraints
for processed bars, client order IDs, and fill event keys. Streamlit session state is
never authoritative for positions or orders; the live unlock is intentionally the only
ephemeral safety acknowledgement.

API secrets are loaded only from environment variables. They are never rendered,
logged, written to SQLite, included in configuration snapshots, or used in result
fingerprints. Broker error messages are sanitized. The application uses no `eval`,
`exec`, shell command, or symbol-derived filesystem operation.

## Important limitations

- Profit is not promised. A strategy can pass every historical gate and still lose.
- The live engine uses completed-bar REST polling and latest quote snapshots, not a
  co-located low-latency execution stack. It is unsuitable for latency-sensitive
  scalping.
- Bar-volume zero can represent a midpoint-derived Alpaca crypto bar. The price bar is
  retained for continuity but fails the entry liquidity filter.
- The backtest cannot reproduce queue position, order-book impact, partial fills,
  exchange outages, fee-tier changes, or stop-limit nonexecution exactly.
- Crypto is 24/7, but Alpaca maintenance or API/data outages can still prevent an exit.
  Broker-side protection and the watchdog reduce risk; they do not eliminate it.
- Manually altering bot positions or orders while an engine runs can trigger a fail-safe
  halt. Stop the engine before manual account intervention whenever possible.
- Live configuration is a capability for explicitly authorized controlled use, not a
  certification that the repository is suitable for unattended live-money operation.

## Files

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit stock/crypto research, verification, and engine controls. |
| `config.py` | Stock config, account modes, isolated credential loading. |
| `crypto_config.py` | Frozen crypto strategy, portfolio risk, engine, and simulation config. |
| `indicators.py` | Causal SMA, Wilder RSI/ATR/ADX, efficiency, volume, ATR percentile. |
| `strategy.py` / `risk_manager.py` | Shared stock decision and risk implementation. |
| `crypto_strategy.py` / `crypto_risk.py` | Shared crypto decision, ranking, sizing, correlation, and loss latch. |
| `backtest.py` / `research.py` | Stock backtest and walk-forward promotion. |
| `crypto_backtest.py` / `crypto_research.py` | Multi-asset crypto simulation and promotion. |
| `trade_engine.py` | Audited paper stock order lifecycle. |
| `crypto_engine.py` | 24/7 crypto reconciliation, partial fills, stops, watchdog, restart. |
| `broker.py` | Exact environment verification and stock/crypto Alpaca requests. |
| `data_feed.py` / `crypto_data_feed.py` | Completed bars, staleness, retries, crypto quotes. |
| `state_store.py` | SQLite sessions, bars, orders, fills, equity, logs, positions. |
| `models.py` / `crypto_models.py` | Typed boundary structures. |
| `mobile_api.py` / `mobile_push.py` | Read-only mobile API and server-side push observer. |
| `scripts/deploy_backend.sh` | One-command production stack bring-up and app connection details. |
| `scripts/verify_backend.sh` | External reachability and authentication check for the mobile API. |

## Troubleshooting

- **Paper account will not verify:** confirm paper keys and exact names in `.env`. A 401
  means Alpaca rejected the credentials.
- **Live remains disabled:** set `ENABLE_LIVE_TRADING=true`, configure separate live
  keys, restart Streamlit, select Live, acknowledge risk, type `LIVE`, and verify.
- **Live selection resets:** an active engine freezes mode; stop it first. A process
  restart intentionally clears the typed live unlock.
- **Crypto asset rejected:** use an active Alpaca USD pair in `BASE/USD` format. The app
  also requires tradable, fractionable, non-marginable, non-shortable metadata and
  dynamic increments.
- **No crypto entry:** inspect every filter and quote guard. Zero volume, wide/stale
  quote, stale/misaligned bars, regime stress, correlation, exposure, or loss latch all
  fail closed.
- **Not enough history:** widen the period or use a slower timeframe. The regime and ATR
  percentile require a substantial causal warmup.
- **Walk-forward cannot form folds:** the requested train + embargo + test windows exceed
  aligned history, or no candidate produced the minimum training trade count.
- **Kill switch blocks Start:** inspect logs and Alpaca account/order state, then use the
  explicit dashboard acknowledgement and re-verify the account.
- **Manual position blocks startup:** the bot cannot prove ownership. Resolve it manually
  in Alpaca; the application will not adopt or liquidate an unexplained position.

Use Paper mode unless you independently understand and accept every live-money failure
mode above.
