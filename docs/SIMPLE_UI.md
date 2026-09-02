# Simple operator UI

The primary production dashboard is `bybit_operator_ui.py`. It intentionally exposes the minimum information an operator needs to supervise a Bybit Spot Demo burn-in.

The legacy research-heavy application remains available at `/research`; none of its underlying capabilities are deleted.

## Normal navigation

```text
HOME
MARKETS
PERFORMANCE
SETTINGS
```

The normal operator should not need to understand walk-forward grids, Quant Desk database internals, lifecycle administration or raw strategy research knobs.

## HOME

The top section makes environment and safety state obvious:

- BYBIT DEMO
- Spot only / no leverage / no derivatives / live disabled
- public/private connection state
- backend running/stopped
- uptime
- HEALTHY / DEGRADED / HALTED
- clock skew
- unprotected quantity

Operational summary:

- strategy capital allocation
- today net P&L
- open bot-owned exposure
- persistent emergency kill-switch state

Top opportunities show at most the strongest configured symbols with an operator-friendly decision:

- BUY SETUP
- WATCH
- NO TRADE
- AVOID

Cards show score, price, dynamic target description, stop, risk/reward when measurable, and a one-line reason.

Open positions show only proven bot-owned Spot inventory, not unexplained wallet holdings. The current protective stop is projected from engine-authoritative persisted position state.

Recent activity includes decisions, no-trade decisions, order/execution events, position management and operational events.

## MARKETS

The configured BTC / ETH / SOL cards show:

- price
- 24-hour move
- regime
- spread
- trend
- momentum
- volatility
- setup score

The **Why?** expander converts the model state into plain-language trend, momentum, liquidity, regime, cost and decision explanations. Raw details/microstructure remain available only in the nested Advanced expander.

## PERFORMANCE

The dashboard never hides losing trades. It displays the current time-causal burn-in evidence:

- net P&L
- trade count
- win rate
- profit factor
- expectancy
- maximum drawdown
- fees
- average winner/loss
- equity curve
- +0.5%, +0.75% and +1.0% MFE reach rates

Operational and preliminary-edge verdicts are shown separately. The screen always states that a 3–5 day run does not prove a durable strategy.

## SETTINGS

Normal settings are limited to:

- Venue: Bybit
- Environment: Demo
- server credential status (configured/missing only; values never displayed)
- strategy lifecycle eligibility
- capital allocation
- conservative risk profile
- allowed BTC / ETH / SOL universe
- 72h / 120h duration
- optional display-only EGP reference
- Start Bybit Demo
- Stop gracefully
- EMERGENCY STOP

The Start button is disabled unless credentials exist, the strategy is lifecycle-eligible, at least one permitted symbol is selected, no backend is already running, and the kill switch is clear.

## Emergency latch

Emergency Stop persists the engine kill latch before signalling the backend. Starting remains blocked even if the process disappears.

Clearing the latch requires an explicit acknowledgement in Settings and is refused while the backend is still running.

## Backend ownership

Streamlit is presentation/control only. Start creates a detached server-side process. No API secret is passed on the command line or written into Streamlit session state.

Closing the browser does not stop orders, protection, reconciliation or WebSocket handling.

## Advanced / Research

The Settings expander links to `/research`, which hosts the existing `app.py` UI in production Docker. It retains research, backtests, walk-forward work, Quant Desk views, lifecycle administration, Alpaca workflows and diagnostics without cluttering ordinary Bybit operation.

## Production routing

`docker-compose.production.yml` runs:

- `bybit_operator_ui.py` on port 8501 as the root dashboard;
- `app.py` on port 8502 with Streamlit base path `/research`;
- the existing mobile API on port 8000.

Caddy routes `/api/*` to the API, `/research*` to the research dashboard, and everything else to the simple Bybit operator dashboard.
