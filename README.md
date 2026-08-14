# Trend Bot — Alpaca Paper Trading (Educational)

A deterministic, long-only trading bot for **Alpaca PAPER trading**, with a Streamlit
dashboard. It exists so you can experiment with a fixed, rule-based strategy — backtest it
on historical data, then watch the exact same rules run against Alpaca's simulated paper
market — without any real money involved.

**Paper-only by design.** The broker client is created with `paper=True` and no URL
override, and at startup it additionally verifies that it is pointed at
`paper-api.alpaca.markets` before doing anything. There is no live-trading switch anywhere
in the code. This is an educational tool for learning how a mechanical strategy behaves;
it is not a product for trading real money.

---

## Important Warning

- This project is **educational** and works **only** with Alpaca **paper** (simulated)
  trading. No real money is ever at risk while using it as shipped.
- **No strategy guarantees profit.** This one applies a handful of fixed mathematical
  rules; it can and will lose (simulated) money in many market conditions.
- **Historical performance does not predict future performance.** A backtest that looks
  good over one period says nothing about the next period.
- **Backtests differ materially from live trading.** Real markets involve slippage,
  partial fills, liquidity gaps, data outages, and fees that a simulation only
  approximates (see "Why paper and backtest results differ from real trading" below).
- Do **not** adapt this code for live money unless you independently understand every
  rule, every assumption, and every failure mode, and accept all of the risks yourself.

## What this bot does NOT do

- It does **not** predict the market. It reacts to past prices with fixed arithmetic.
- It does **not** learn or adapt. The rules never change while it runs.
- It knows **nothing** about news, earnings, fundamentals, or anything outside the
  price/volume bars it downloads.
- It applies exactly **four fixed mathematical rules** — a moving-average crossover, an
  RSI entry filter, a volume confirmation, and an ATR-based volatility filter/stop — and
  nothing else.
- There is **no AI or machine learning anywhere** in this project.

---

## How the strategy works (in plain language)

All decisions are made on **completed** price bars (never a bar that is still forming),
for **one symbol at a time**, and the bot only ever goes **long** (buys shares hoping to
sell them later; it never bets on prices falling).

### 1. Moving averages detect a change in trend

A moving average smooths price by averaging the last N closing prices (defaults: 20 bars
for the "short" average, 50 for the "long" one). The entry trigger is a **new crossover**,
not merely "short average is above long average":

- **BUY signal:** on the previous bar `short MA <= long MA`, and on the current bar
  `short MA > long MA`. The short average has just crossed above the long one.
- **EXIT signal:** the mirror image — previous bar `short MA >= long MA`, current bar
  `short MA < long MA`.

Requiring a *new* cross means the bot buys at most once per trend change instead of
buying again on every bar of an existing uptrend.

### 2. RSI avoids chasing an overextended move (entry filter only)

RSI (Relative Strength Index) scores recent momentum from 0 to 100. A crossover that
happens after price has already surged tends to arrive "stretched", so the bot only takes
a BUY signal when **RSI is below the overbought threshold (default 70)**. RSI is **only an
entry filter — there is no RSI-based exit rule**. Positions are exited by the bearish
crossover or the protective stop, never by RSI.

### 3. Volume confirmation

A crossover on unusually thin trading is treated as weaker evidence. An entry additionally
requires the current bar's **volume >= its rolling average volume × a multiplier**
(defaults: 20-bar average, multiplier 1.0). Note the free-tier data caveat below — IEX
volume is only a fraction of total market volume.

### 4. ATR measures volatility and sets the protective stop

ATR (Average True Range) is the typical size of a bar's price movement — a volatility
yardstick. It is used twice:

- **Volatility filter (optional, on by default):** entries are skipped when the current
  ATR sits above the 90th percentile of its own recent history — i.e. when the market is
  unusually turbulent for this symbol.
- **Protective stop:** immediately after an entry fills, the bot places a broker-side
  stop-loss order at `stop = fill price − ATR × multiplier` (default multiplier 2.0).
  The stop is **fixed from entry — it never trails** upward as price moves.

### 5. Position sizing works backward from the stop distance

The bot computes three independent share caps and takes the smallest:

1. **Risk-budget shares:** `floor(equity × risk% ÷ (entry − stop))` — if the stop fills,
   the loss is at most the configured percentage of equity (default 1%).
2. **Exposure-cap shares:** `floor(equity × max position% ÷ entry price)` (default 20%).
3. **Buying-power shares:** `floor(buying power ÷ entry price)`.

Quantities are **whole shares only, always rounded down (floored), never up** — rounding
up would silently increase risk just to force a trade. Fractional shares are deliberately
not used, even for symbols that support them: whole-share floor arithmetic keeps sizing
simple, auditable, and conservative. Entries that come out at 0 shares, below the minimum
share count, or below the minimum position value are rejected rather than resized.

---

## Requirements & setup

- **Python 3.11 or newer**
- A free Alpaca account with **paper trading** API keys

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Configure credentials
cp .env.example .env
# Edit .env and set ALPACA_API_KEY and ALPACA_SECRET_KEY
# (use PAPER keys, from the Alpaca dashboard -> Paper Trading -> API Keys)

# 4. Launch the dashboard
streamlit run app.py
```

Your browser opens the local dashboard automatically (typically at
`http://localhost:8501`). Credentials stay in the local `.env` file; the app never
displays or logs the secret key.

Run the test suite with:

```bash
python -m pytest -q
```

## Recommended workflow

1. **Configure** — on the *Configure & Backtest* tab, pick a symbol, timeframe, indicator
   periods, and risk limits.
2. **Backtest** — run a historical simulation over your chosen date range.
3. **Inspect the drawdown and the trade list** — look at the worst peak-to-trough equity
   drop and read through individual trades, not just the summary numbers.
4. **Understand the assumptions** — expand the "Simulation assumptions" panel and read the
   execution/backtest assumptions below until nothing surprises you.
5. **Switch to the *Run Live (Paper)* tab.**
6. **Confirm the PAPER TRADING banner** is visible — it is shown in the sidebar and on the
   live tab.
7. **Choose a short runtime first** (e.g. a fixed duration of one hour) rather than
   "until I stop it".
8. **Start the bot.**
9. **Observe** — watch the latest signal panel (each filter shows PASS/FAIL), orders,
   fills, and the session equity curve.
10. **Stop safely** with the *Stop Bot* button; use *EMERGENCY STOP* only when you want
    the engine terminated immediately.

Note: a session's configuration is **frozen at start** — to change parameters, stop the
bot and start a new session.

---

## Architecture

```
Streamlit UI -> Configuration -> TrendStrategy + RiskManager -> Backtest or TradeEngine ->
AlpacaPaperBroker / AlpacaDataFeed -> SQLite StateStore -> Dashboard
```

`TrendStrategy` and `RiskManager` are the **same classes** in both the backtester and the
live engine — a single implementation of the rules, with tests asserting decision parity.
There is no separate "live version" of the strategy that could drift from what you
backtested.

| File | Responsibility |
| --- | --- |
| `app.py` | Streamlit dashboard: configuration forms, backtest results, live controls, logs. Presentation only — no trading logic. |
| `config.py` | Typed configuration objects, validation, timeframe mapping, `.env` credential loading. |
| `indicators.py` | Deterministic, causal indicator math (SMA, Wilder RSI, ATR, volume ratio, ATR percentile). |
| `strategy.py` | `TrendStrategy` — the only implementation of the trading rules. |
| `risk_manager.py` | `RiskManager` — stop price, position sizing, session loss limit. |
| `backtest.py` | Historical simulation and statistics (drawdown, Sharpe, win rate). |
| `trade_engine.py` | Live paper-trading loop: polling, order lifecycle, reconciliation, safety guards. |
| `broker.py` | All Alpaca trading-API access; hardcodes the paper endpoint; only touches orders in its own `trendbot-` namespace. |
| `data_feed.py` | Historical/recent bars, completed-bar detection, staleness rules, retry/backoff. |
| `state_store.py` | SQLite persistence (sessions, orders, trades, equity, logs) shared by engine and UI. |
| `models.py` | Shared dataclasses and enums. |

## Execution and backtest assumptions

The backtest is honest about what it can and cannot know:

- **Next-bar execution:** a signal generated on completed bar N fills at bar N+1's
  **open** — never at bar N's close, which the strategy could not have traded at.
- **Conservative stop fills:** if a bar *opens* at or below the stop (a gap), the fill is
  the open price — you do **not** get the stop price; otherwise, if the bar's low touches
  the stop, the fill is the stop price capped at the bar's open (a stop exit is never
  booked at a better price than the bar opened at, which matters on the entry bar itself
  when slippage is configured).
- **Directional slippage:** buys fill slightly higher, sells slightly lower — slippage
  always works against you.
- **Slippage and commission are configurable and default to 0**, and that assumption is
  stated in the results panel. Zero-cost fills are optimistic; try nonzero values.
- **Split-adjusted bars:** without adjustment, a stock split (say 4-for-1) makes the price
  chart drop 75% overnight and creates fake crossovers; split adjustment removes these
  artifacts.
- **Completed (closed) bars only**, in both backtest and live — the bot never acts on a
  bar that is still forming.
- **Regular market hours only:** the live engine trades only while the market is open; no
  extended-hours trading.
- **One symbol per session**, and the configuration is frozen once a session starts.
- The backtest models a **cash account**: buying power equals cash, whole shares only.

## Why paper and backtest results differ from real trading

- **Fills and slippage:** a simulation fills you at a clean reference price; real markets
  fill you at whatever the order book offers at that instant.
- **Liquidity and partial fills:** large orders (or thin symbols) may fill in pieces at
  worsening prices, or not at all. The engine handles partial fills, but a backtest
  assumes complete fills.
- **Outages and delays:** data feeds stall, brokers rate-limit, clocks drift. The engine
  guards against these (see below) by *not trading*, which a backtest never has to do.
- **IEX vs SIP data:** Alpaca's **free tier serves IEX data only** — one exchange, a
  fraction of consolidated (SIP) market volume. This matters here because the **volume
  confirmation filter** compares bar volume to its average: on IEX both numbers are small
  and their ratio behaves differently than volume figures you might see on other
  platforms or charting tools. The feed is configurable (`iex`/`sip`; SIP requires a paid
  data subscription) and the active feed is shown in the sidebar.

## Pattern Day Trading (PDT), for beginners

FINRA (the US brokerage regulator) flags an account as a **pattern day trader** when it
makes **4 or more day trades within 5 business days** (a day trade = buying and selling
the same symbol in the same day) while holding **under $25,000** in equity. Flagged
accounts face trading restrictions. Alpaca's paper environment **simulates this rule**, so
you can learn about it safely.

The bot's handling: on intraday timeframes, when account equity is below $25,000 and the
account is flagged or already has 3 day trades in the rolling window, new entries are
**blocked with reason `pdt_restriction`** rather than risking a fourth. The dashboard's
live tab shows the account's PDT flag and current day-trade count.

## Safety behaviors

- **Session loss halt:** when equity falls a configured percentage (default 3%) below the
  session's starting equity, the bot **halts new entries** and stops the session. It does
  **not** auto-liquidate — an open position and its broker-side protective stop are left
  in place.
- **Emergency Stop leaves the protective stop in place.** This is deliberate: the stop
  order lives at the broker (GTC), so an open position keeps its downside protection even
  after the engine process is gone.
- **Unclean-stop acknowledgement:** after an emergency stop, a session loss halt, or a
  fatal error, a persisted "kill switch" prevents restarting until you review the state
  and click the acknowledgement button in the UI.
- **Position ownership boundary:** the bot refuses to start if a position already exists
  in the configured symbol that it did not open. It only manages positions and orders it
  created (all its orders carry a `trendbot-` client-order-id prefix).
- **Partial-fill containment:** after an entry partially fills, the filled quantity is
  persisted and protected, the unfilled remainder is canceled, and any fill that races
  with cancellation causes the stop to be replaced at the exact final quantity. Partial
  exits and stop fills use cumulative-fill accounting so the bot never sells more than
  broker-reported position truth.
- **Exact stop ownership and quantity:** restart reconciliation adopts only a bot-owned
  sell stop whose remaining quantity exactly matches the bot position. Duplicate,
  wrong-side, or mismatched bot stops are canceled individually; unrelated orders are
  never canceled.
- **Entries blocked near the close:** by default, no new entries in the final 30 minutes
  of the regular session (configurable).
- **Optional flat-before-close (off by default):** when off, positions are held overnight
  and exposed to **opening-gap risk** — the next day's open can be far from the stop
  price, and a gap through the stop fills at the worse open price. When on, the bot sells
  its position shortly before the close.
- **Clock-skew guard:** if the local clock disagrees with Alpaca's by more than a
  threshold (default 30 s), trading pauses — bar-completion logic depends on local time.
- **Stale-data guard:** if the newest completed bar is too old (default: 3 bar-lengths
  intraday), the bot skips the trading decision instead of acting on outdated prices.
- **Single engine per state database:** engine startup takes an OS-level lock on a
  sibling of the SQLite file and holds it until shutdown. A second app instance (or an
  engine orphaned by a source reload) refuses to trade against the same database. The
  lock dies with its process, so a crash can never leave it stuck.
- **Kill switch enforced by the engine:** after an emergency stop, daily-loss halt, or
  fatal error, the engine itself refuses to start until the unclean stop is
  acknowledged in the dashboard — not just the Start button.
- Other engine invariants: each bar is processed at most once (persisted across
  restarts), the same submission attempt has a deterministic order ID while a confirmed
  terminal retry gets a distinct bounded ID, and fills are always confirmed from broker
  state before any dependent action.

## Glossary

- **Moving average (MA):** the average of the last N closing prices; smooths out noise to
  show the general direction.
- **Crossover:** the moment one moving average moves from one side of another to the
  other side — used here as a trend-change trigger.
- **RSI (Relative Strength Index):** a 0–100 momentum score; high values mean price has
  recently risen a lot relative to its declines.
- **ATR (Average True Range):** the typical size of a bar's price movement; a measure of
  volatility.
- **Stop loss:** a resting order to sell if price falls to a set level, limiting further
  loss on a position.
- **Drawdown:** how far equity has fallen from its previous peak, in percent.
- **Sharpe ratio:** average return divided by the variability of returns — a
  volatility-adjusted summary statistic, meaningful only with many observations.
- **Slippage:** the difference between the price you expected and the price your order
  actually filled at.
- **PDT (pattern day trading):** the FINRA rule described above, limiting frequent
  intraday round-trips in accounts under $25,000.
- **Paper trading:** simulated trading with fake money against real market data — how
  this entire project operates.

## Troubleshooting

- **Missing or invalid credentials:** the dashboard shows a setup box; copy
  `.env.example` to `.env`, add your paper keys, and restart Streamlit. A 401 error means
  Alpaca rejected the keys — re-copy them from the dashboard's Paper Trading section.
- **Data subscription errors:** requesting SIP data (or very recent minutes) on the free
  tier is rejected by Alpaca. Switch the data feed to `iex`.
- **PDT rejections (`pdt_restriction`):** you are on an intraday timeframe with equity
  under $25,000 and the day-trade limit is in reach. Use the `1Day` timeframe, or wait
  for the 5-business-day window to roll off. Paper equity is simulated, so you can also
  reset the paper account from Alpaca's dashboard.
- **Market closed:** entries are blocked outside regular US market hours
  (reason `market_closed`); the bot idles until the next open.
- **Symbol not tradable:** the symbol must be an active, tradable US equity on Alpaca;
  check spelling (e.g. `BRK.B`) or pick another symbol.
- **Insufficient buying power / position too small:** the sizing caps produced 0 shares
  or a position under the configured minimums. Lower the entry price target (cheaper
  symbol), or review risk-per-trade and minimum-position settings.
- **"Not enough historical data" when backtesting:** the configured indicator periods
  need more warmup bars than the date range provides — widen the date range or shorten
  the indicator periods (the error message states the exact bar count needed).
- **Stale "unclean stop" message blocks starting:** the previous session ended with an
  emergency stop, loss halt, or error. Review the Logs/State tab, then click
  *"I understand — clear and allow restart"* on the live tab.
- **Resetting bot state:** all persistent state lives in a SQLite file at
  `data/trading_bot.db`. Deleting it (while the app is stopped) resets the bot's local
  state — sessions, orders, trades, logs, and position bookkeeping. It does not change
  anything at Alpaca; check your paper account for leftover positions or orders.

## A note on statistics and parameter choice

With few trades, summary statistics are noise: the dashboard explicitly flags win rate
and Sharpe ratio as **statistically meaningless below roughly 30 completed trades**, and
withholds the Sharpe ratio entirely below 20 return observations. Parameters are chosen
**manually, by design** — there is no auto-optimization, because searching many parameter
combinations until a backtest looks good mostly finds settings that fit past noise
(overfitting), not settings that mean anything about the future. Results are specific to
the exact parameters and date range you chose; small changes can produce large result
changes.

---

*Educational software for Alpaca PAPER trading. Nothing in this repository is investment
advice, and nothing about it implies any expected outcome.*
