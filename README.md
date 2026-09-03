# Trade Bot

A research-first trading platform with a production-oriented **Bybit Spot Demo** operator path and preserved Alpaca/Quant Desk research workflows.

The current Bybit mission is deliberately constrained:

- Spot only
- `category=spot`
- `isLeverage=0`
- long-only ownership-aware execution
- BTC / ETH / SOL permitted universe by default
- no margin borrowing
- no leverage
- no perpetuals/futures/inverse/options
- no shorts
- no lending/borrowing execution
- no martingale, loss doubling, averaging down or grid-loss recovery
- Bybit Live disabled

A 3–5 day Demo burn-in is an operational acceptance experiment plus preliminary forward evidence. It does **not** prove profitability or a durable edge.

## Primary operator dashboard

Normal operation is dashboard-first:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt -r requirements-mobile-api.txt
streamlit run bybit_operator_app.py
```

Then open **SETTINGS** and follow four steps:

1. Paste your **Bybit Demo API key + secret** and click **Test & Save**.
2. Explicitly acknowledge and run the tiny **Demo economic smoke**. The 3–5 day burn-in stays blocked until this passes for the currently configured credentials.
3. Choose the capital allocation, BTC/ETH/SOL subset and 3-day or 5-day burn-in.
4. Click **Start Bybit Demo**.

Create the key under **Bybit Demo Trading**, not Testnet or Live. Grant only the read/trade permissions needed for Spot Demo. **Do not grant withdrawal permission.**

Dashboard-saved credentials are written to a server-only credential vault with restrictive permissions. They are not stored in SQLite, reports, logs, URLs or subprocess arguments, and are never rendered back to the browser. Production Docker mounts the vault only into the operator dashboard container. Existing deployments may still use `BYBIT_DEMO_API_KEY` / `BYBIT_DEMO_API_SECRET` in a private `.env` as an optional fallback.

The dashboard navigation is intentionally small:

```text
HOME
MARKETS
PERFORMANCE
SETTINGS
```

The browser is not the trading process: Start launches a detached server-side backend, so closing the browser does not terminate protection, reconciliation or WebSocket supervision.

The existing comprehensive research application remains available separately as Advanced / Research.

## Production Docker

Set `TRADE_DOMAIN`, a strong `TRADE_DASHBOARD_PASSWORD`, and the existing mobile token/settings in your private `.env`, then run:

```bash
docker compose -f docker-compose.production.yml up -d --build
```

Production routing:

- `/` → authenticated simple Bybit Demo operator dashboard
- `/research` → authenticated existing full research/Alpaca/Quant Desk Streamlit application
- `/api/*` → existing mobile API, which keeps its own bearer-token authentication

Caddy requires HTTP Basic Auth for both browser dashboards. Username is `operator`; the password is `TRADE_DASHBOARD_PASSWORD`. Caddy derives a one-way hash at container startup and fails closed if the password is missing. Do not reuse the mobile bearer token or Bybit API secret as the dashboard password.

Persistent volumes are separated for runtime state, reports, logs and the dashboard-only secret vault.

## Architecture

The venue adapter is below the strategy/Quant Desk boundary:

```text
Operator dashboard
        |
Detached Bybit Demo service
        |
Quant-controlled adaptive crypto engine
        |
+-- existing regime/adaptive strategy
+-- existing EvidenceQuantScanner / Edge Intelligence
+-- Bybit-only depth / impact / target-room rejection layer
+-- existing portfolio + risk governor
+-- existing lifecycle/degradation supervision
+-- existing execution-quality journal
+-- existing TradeManagementPolicy / exit intelligence
        |
Venue-neutral crypto contracts
        |
+-- BybitSpotBroker / instrument metadata
+-- public Spot WebSocket + REST recovery
+-- private Demo WebSocket
+-- deterministic orderLinkId reconciliation
+-- bot-owned Spot inventory ledger
```

The existing Alpaca broker/data-feed path remains intact for regression, paper workflows and research.

See [`docs/TRADING_ENGINE_ARCHITECTURE.md`](docs/TRADING_ENGINE_ARCHITECTURE.md).

## Bybit Demo topology

The implementation uses current V5 semantics:

- public Spot WebSocket for order book/trades/klines;
- Demo REST for account/instrument/order submit-amend-cancel-query and reconciliation;
- private Demo WebSocket for order/execution/wallet events;
- REST recovery/backfill after uncertain WebSocket state.

Demo WebSocket order entry is not fabricated.

`pybit` is pinned to the exact tested version in `requirements.txt`.

## Instrument and order safety

The canonical instrument layer maps the existing strategy identity to the venue explicitly, for example:

```text
strategy_symbol  BTC/USD
canonical_pair   BTC/USDT
venue_symbol     BTCUSDT
venue            bybit
```

Tick size, quantity step and venue minimums are read dynamically from Bybit instrument metadata. They are not hardcoded into strategy code.

Every economic intent has a deterministic Bybit `orderLinkId` within the venue length contract. A request timeout is reconciled by that same ID; an HTTP response is never treated as a fill.

## Bot-owned Spot quantity

A wallet balance is not treated as a derivative-style position.

The bot may sell only:

```text
min(proven bot-owned execution quantity, currently available wallet quantity)
```

Manual/external holdings are not adopted merely because they are in the wallet. An unexplained wallet deficit fails closed.

## Quant decision path

The production entry path remains evidence/risk driven:

```text
completed causal data
  -> adaptive setup/regime
  -> EvidenceQuantScanner + Edge Intelligence
  -> spread/slippage/fee/participation cost gate
  -> Bybit multi-level depth + impact proxy + structural target-room gate
  -> portfolio/risk governor
  -> burn-in rolling-loss/cooldown/entry ceiling
  -> Bybit Spot Demo order intent
```

The Bybit-specific layer is negative-only: favorable depth/order flow cannot manufacture alpha; adverse liquidity, stale/gapped books, excessive impact or insufficient target room can only demote/reject a candidate.

Target selection is not a fixed `+0.5%` order. An accepted target plan combines initial R, ATR, recent realized movement, completed-bar resistance and expected round-trip costs. The +0.25/+0.50/+0.75/+1.00 families remain research measurements in the burn-in report.

BTC, ETH and SOL compete for capital. The maximum allowed high-quality entries per rolling 24h is five by default. The minimum required trades is zero.

The software does not target a daily trade quota or guaranteed daily return.

## Smart exits

The Bybit production path executes the existing `quant.exits.TradeManagementPolicy`; it does not introduce another exit system.

Supported policy mechanics include:

- initial protective stop
- partial profit taking
- break-even transition
- stagnation/time exit
- smart full exit
- adaptive trailing stop
- runner management
- emergency market exit
- accepted structure/ATR/R target plan fed into the same policy

The legacy strategy ATR ratchet is made inert on this path so the existing policy is the single source of post-entry decisions. Broker protection/reconciliation remains owned by the existing engine.

## Notifications

If the existing mobile/Expo push service is configured, SETTINGS exposes a **Phone push notifications** switch. Device tokens stay in the existing mobile API database while the push observer watches the dedicated Bybit engine database for new fills, safety locks and engine-state changes. Notification delivery is observational only and can never block or modify trading/protection.

## Burn-in evidence

A run freezes:

- run ID
- Git SHA
- strategy version
- configuration hash
- universe
- start/planned end
- environment

Restart recovery can resume only an exact matching identity.

Every completed decision is journaled, including **NO TRADE**. Operational samples include WebSocket health/reconnects, stale/gap evidence, spread, clock skew, rate limits, REST failures, pending intents and unprotected quantity.

Trade evidence includes execution IDs, fees/slippage, P&L, R where measurable, MFE/MAE, holding time, regime/setup context, and +0.25/+0.50/+0.75/+1.00 reach timing.

Daily time-causal JSON/Markdown snapshots are emitted while the run is active. The final report is written on shutdown.

Final reports separate:

- `OPERATIONAL_ACCEPT` / `OPERATIONAL_REJECT`
- `EDGE_PRELIMINARY_POSITIVE` / `EDGE_INCONCLUSIVE` / `EDGE_PRELIMINARY_NEGATIVE`

and always state `strategy_proven: false` for this burn-in horizon.

## Exact 3-day and 5-day service commands

The dashboard launches equivalent backend semantics, but the exact service commands remain:

```bash
python scripts/run_bybit_demo_burnin.py --hours 72
python scripts/run_bybit_demo_burnin.py --hours 120
```

Normal dashboard operation additionally requires a credential-bound Demo-smoke PASS before it will start either run.

## Demo economic smoke

Normal operators run this from **SETTINGS → Verify tiny Demo order path**. The equivalent service command is:

```bash
python scripts/bybit_demo_smoke.py --confirm I_ACCEPT_BYBIT_DEMO_ORDER
```

It validates account/metadata/public/private connectivity, submits only a dynamically-valid tiny Demo Spot order inside the smoke cap, reconciles it, closes only proven bot-owned quantity and refuses to pass with residual bot-owned exposure.

Changing dashboard-saved credentials invalidates the stored smoke PASS. The next burn-in cannot start until the smoke passes again for the new credential pair.

No automated test places a Live order.

## One-command acceptance evidence

When GitHub-hosted Actions is unavailable, run the same repository acceptance surface locally:

```bash
python scripts/run_bybit_acceptance.py
```

It byte-compiles the repository, runs the dedicated Bybit/dashboard safety suite, runs the entire Alpaca/Quant/mobile/Bybit regression suite, checks tracked-secret hygiene, validates Docker Compose when Docker is installed, and writes:

```text
reports/bybit_acceptance_<SHA>.json
reports/bybit_acceptance_<SHA>.md
```

To append the explicit real Demo smoke after the hermetic suite passes:

```bash
python scripts/run_bybit_acceptance.py \
  --demo-smoke \
  --confirm-demo-order I_ACCEPT_BYBIT_DEMO_ORDER
```

This still requires real Bybit Demo credentials in the server environment. It never selects Bybit Live.

## Emergency stop

The dashboard **EMERGENCY STOP** writes a persistent kill latch before signalling the backend. Restart remains blocked until the operator reviews the reason and explicitly acknowledges the latch while the backend is stopped.

Use **Stop gracefully** for ordinary shutdown.

## Documentation

- [`docs/BYBIT_PRODUCTION_AUDIT.md`](docs/BYBIT_PRODUCTION_AUDIT.md) — source-of-truth architecture audit
- [`docs/BYBIT_SETUP.md`](docs/BYBIT_SETUP.md) — dashboard credential and Demo setup
- [`docs/BYBIT_DEMO_BURNIN.md`](docs/BYBIT_DEMO_BURNIN.md) — burn-in procedure and evidence
- [`docs/TRADING_ENGINE_ARCHITECTURE.md`](docs/TRADING_ENGINE_ARCHITECTURE.md) — execution/data/ownership boundaries
- [`docs/SIMPLE_UI.md`](docs/SIMPLE_UI.md) — normal operator UX
- [`QUANT_DESK.md`](QUANT_DESK.md) — existing Quant Desk
- [`ADAPTIVE_INTRADAY_V2.md`](ADAPTIVE_INTRADAY_V2.md) — existing adaptive intraday research
- [`INTRADAY_PRODUCTION.md`](INTRADAY_PRODUCTION.md) — existing intraday production/lifecycle background

## Testing

Hermetic unit/integration tests cover the legacy Alpaca workflows and the Bybit venue path, including Spot/leverage invariants, symbol mapping, dynamic metadata changes, deterministic IDs, timeout reconciliation, wallet ownership, execution idempotency, order-book gaps/staleness, WebSocket duplicate tolerance, target-room intelligence, smart-exit single-policy wiring, secure dashboard credential storage, credential-bound Demo-smoke gating, production dashboard authentication, notification separation, active-report causality, dashboard process ownership and emergency-latch behavior.

Run locally:

```bash
python scripts/run_bybit_acceptance.py
```

CI uses no trading credentials and cannot place economic orders.

## Security

Never commit `.env`, `secrets/` or API keys. The application does not require withdrawal permission. Bybit API secrets must not appear in SQLite, reports, logs, subprocess arguments, frontend status, or API responses.

Production browser dashboards are protected by Caddy Basic Auth over HTTPS and fail closed when `TRADE_DASHBOARD_PASSWORD` is missing. The mobile API remains independently bearer-token authenticated.

Live trading remains outside the current Bybit acceptance boundary and requires a separate future review.
