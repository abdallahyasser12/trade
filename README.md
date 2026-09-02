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
cp .env.example .env
streamlit run bybit_operator_ui.py
```

Set only the Bybit Demo credentials in `.env`:

```text
BYBIT_DEMO_API_KEY=...
BYBIT_DEMO_API_SECRET=...
```

Create the key from **Bybit Demo Trading** with only the read/trade permissions needed for Spot Demo. **Do not grant withdrawal permission.**

The dashboard navigation is intentionally small:

```text
HOME
MARKETS
PERFORMANCE
SETTINGS
```

From SETTINGS you choose the strategy capital allocation, 72h/120h burn-in duration and permitted BTC/ETH/SOL subset, then click **Start Bybit Demo**. The browser is not the trading process: the UI launches a detached server-side backend, so closing the browser does not terminate protection or reconciliation.

The existing comprehensive research application remains available separately as Advanced / Research.

## Production Docker

```bash
docker compose -f docker-compose.production.yml up -d --build
```

Production routing:

- `/` → simple Bybit Demo operator dashboard
- `/research` → existing full research/Alpaca/Quant Desk Streamlit application
- `/api/*` → existing mobile API

Runtime data, burn-in reports and logs use persistent Docker volumes.

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
  -> execution-quality/cost gate
  -> portfolio/risk governor
  -> burn-in rolling-loss/cooldown/entry ceiling
  -> Bybit Spot Demo order intent
```

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

The legacy strategy ATR ratchet is made inert on this path so the existing policy is the single source of post-entry decisions. Broker protection/reconciliation remains owned by the existing engine.

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

The dashboard launches equivalent backend semantics, but the exact acceptance commands remain:

```bash
python scripts/run_bybit_demo_burnin.py --hours 72
python scripts/run_bybit_demo_burnin.py --hours 120
```

## Demo economic smoke

The smoke is explicit, tiny, Spot-only and Demo-only:

```bash
python scripts/bybit_demo_smoke.py --acknowledge-demo-order I_ACCEPT_BYBIT_DEMO_ORDER
```

It validates account/metadata/public/private connectivity, submits only a dynamically-valid tiny Demo Spot order inside the smoke cap, reconciles it, closes only proven bot-owned quantity and refuses to pass with residual bot-owned exposure.

No automated test places a Live order.

## Emergency stop

The dashboard **EMERGENCY STOP** writes a persistent kill latch before signalling the backend. Restart remains blocked until the operator reviews the reason and explicitly acknowledges the latch while the backend is stopped.

Use **Stop gracefully** for ordinary shutdown.

## Documentation

- [`docs/BYBIT_PRODUCTION_AUDIT.md`](docs/BYBIT_PRODUCTION_AUDIT.md) — source-of-truth architecture audit
- [`docs/BYBIT_SETUP.md`](docs/BYBIT_SETUP.md) — Demo credentials and dashboard setup
- [`docs/BYBIT_DEMO_BURNIN.md`](docs/BYBIT_DEMO_BURNIN.md) — burn-in procedure and evidence
- [`docs/TRADING_ENGINE_ARCHITECTURE.md`](docs/TRADING_ENGINE_ARCHITECTURE.md) — execution/data/ownership boundaries
- [`docs/SIMPLE_UI.md`](docs/SIMPLE_UI.md) — normal operator UX
- [`QUANT_DESK.md`](QUANT_DESK.md) — existing Quant Desk
- [`ADAPTIVE_INTRADAY_V2.md`](ADAPTIVE_INTRADAY_V2.md) — existing adaptive intraday research
- [`INTRADAY_PRODUCTION.md`](INTRADAY_PRODUCTION.md) — existing intraday production/lifecycle background

## Testing

Hermetic unit/integration tests cover the legacy Alpaca workflows and the Bybit venue path, including Spot/leverage invariants, symbol mapping, dynamic metadata changes, deterministic IDs, timeout reconciliation, wallet ownership, execution idempotency, order-book gaps/staleness, WebSocket duplicate tolerance, smart-exit policy wiring, active-report causality, dashboard process ownership and emergency-latch behavior.

Run locally:

```bash
python -m compileall -q .
python -m pytest -q
```

CI uses no trading credentials and cannot place economic orders.

## Security

Never commit `.env` or API keys. The application does not require withdrawal permission. Bybit API secrets must not appear in SQLite, reports, logs, frontend state or API responses.

Live trading remains outside the current Bybit acceptance boundary and requires a separate future review.
