# Native Mobile Companion

## Goal

The mobile implementation gives the existing Trade repository a simple native
iOS/Android companion without creating a second trading engine or putting Alpaca
credentials on a phone.

The architecture deliberately separates **monitoring** from **money-moving
controls**:

```text
Alpaca
  ^
  | existing broker/data modules
  |
TradeEngine / CryptoTradeEngine
  |
  +--> SQLite StateStore <---- mobile_api.py <---- Expo native app
  |
  `--> existing Streamlit dashboard <------------ "Open full dashboard"
```

The native app reads the same durable state that already survives Streamlit
reruns and engine restarts. The existing Streamlit application remains the one
place that owns Start/Stop workflows, kill-switch acknowledgement, strategy
configuration, backtests, and the guarded crypto live-account unlock.

This is intentional: the mobile layer cannot bypass the repository's existing
live-trading safety gates.

## What the mobile app shows

The first native screen is intentionally small:

- latest persisted equity and change over the returned window;
- stock engine state, symbol, timeframe, last signal, errors, and safety-lock
  state;
- crypto engine state, account mode, universe, timeframe, most actionable recent
  signal, errors, and safety-lock state;
- recent persisted trades;
- recent engine logs;
- pull-to-refresh plus a 30-second foreground refresh;
- a button that opens the configured existing Streamlit dashboard for full
  controls.

The companion does **not** submit orders, start an engine, clear a kill switch,
or unlock a live account.

## Security model

### Broker secrets stay server-side

Do not place any of these in the Expo app, `app.json`, an `EXPO_PUBLIC_*`
variable, TestFlight metadata, or a mobile build:

- `ALPACA_PAPER_API_KEY`
- `ALPACA_PAPER_SECRET_KEY`
- `ALPACA_LIVE_API_KEY`
- `ALPACA_LIVE_SECRET_KEY`

The phone only needs:

1. the URL of `mobile_api.py`; and
2. a separate `MOBILE_API_TOKEN`.

The app stores those connection values with `expo-secure-store`.

### API authentication

Every data endpoint requires:

```http
Authorization: Bearer <MOBILE_API_TOKEN>
```

The server:

- refuses configured tokens shorter than 32 characters;
- uses constant-time token comparison;
- returns `503` from protected endpoints when authentication is not configured;
- exposes only `/health` without authentication;
- disables interactive OpenAPI/Redoc endpoints by default;
- exposes no mutation/trading endpoint.

Generate a token locally, for example:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Store the result only in the server's secret environment.

### HTTPS

For any non-local deployment, terminate TLS in front of the FastAPI service and
use an `https://` mobile API URL. The app rejects plain HTTP outside localhost or
RFC1918 private-network development addresses.

Do not expose a raw development Uvicorn port directly to the public Internet.

### CORS

A native app does not require browser CORS. `MOBILE_ALLOWED_ORIGINS` therefore
defaults to empty. If the Expo web build is intentionally used, set an explicit
comma-separated origin allowlist. Wildcard credentialed origins are not used.

## Backend setup

Install the existing bot dependencies plus the isolated mobile API dependencies:

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-mobile-api.txt
```

Configure the service environment:

```bash
export MOBILE_API_TOKEN='<generated token>'
export MOBILE_API_DB_PATH='data/trading_bot.db'
export MOBILE_DASHBOARD_URL='https://your-existing-streamlit-host.example'
```

`MOBILE_API_DB_PATH` must resolve to the **same SQLite database** used by the
trading dashboard/engine. The default already matches `config.DEFAULT_DB_PATH`.

For local LAN development only:

```bash
uvicorn mobile_api:app --host 0.0.0.0 --port 8000
```

For an Internet-facing deployment, run the ASGI service behind your normal HTTPS
reverse proxy / ingress and restrict network access as appropriate.

Health check:

```bash
curl http://127.0.0.1:8000/health
```

Authenticated overview:

```bash
curl \
  -H "Authorization: Bearer $MOBILE_API_TOKEN" \
  http://127.0.0.1:8000/v1/overview
```

## API surface

### `GET /health`

Public liveness only. It contains no account, engine, signal, trade, or log data.

### `GET /v1/overview`

Authenticated. Returns:

- persisted stock engine status;
- persisted stock last signal;
- stock safety-lock state;
- persisted crypto engine status;
- persisted crypto last-signal map;
- crypto safety-lock state;
- latest persisted session metadata;
- configured full-dashboard URL;
- explicit capability flags showing that mobile trading commands are disabled.

No Alpaca network request is made for this endpoint.

### `GET /v1/trades?limit=25`

Authenticated. Returns the newest persisted fill events. `limit` is bounded to
1–100.

### `GET /v1/logs?limit=40`

Authenticated. Returns newest persisted engine logs. `limit` is bounded to
1–100.

### `GET /v1/equity?limit=120`

Authenticated. Returns the newest points from the latest session's persisted
equity curve. `limit` is bounded to 2–500.

## Native app setup

The mobile project lives under `mobile/` and is intentionally independent from
the Python environment.

```bash
cd mobile
npm install
npm run typecheck
npm run ios
```

On first launch, enter:

- **Mobile API URL** — the HTTPS URL serving `mobile_api.py`;
- **Access token** — the exact `MOBILE_API_TOKEN` configured on the server.

The settings can later be edited or removed from the app. Removing them deletes
the values from Expo SecureStore.

## Current mobile dependency baseline

The project is pinned to an Expo SDK 57-compatible baseline:

- Expo `~57.0.8`
- React `19.2.3`
- React Native `0.86.2`
- Expo SecureStore `~57.0.0`
- TypeScript `~6.0.3`

Before shipping a new binary, run:

```bash
cd mobile
npx expo install --check
npm run typecheck
```

and resolve any SDK compatibility warnings before building.

## Backend dependency baseline

The isolated API requirements are pinned in `requirements-mobile-api.txt`:

- FastAPI `0.139.2`
- Uvicorn `0.51.0`

They are separate from the core bot requirements so existing Streamlit-only
installations are not forced to install an HTTP API server.

## Test coverage

`tests/test_mobile_api.py` covers:

- public health response contains no trading state;
- missing/incorrect bearer tokens are rejected;
- short configured tokens are rejected;
- stock/crypto status and signal snapshots come from the durable store;
- kill-switch state is visible to the companion;
- the API advertises `trading_commands=false`;
- activity limits are bounded;
- equity history returns only the requested newest window.

The test module uses `pytest.importorskip` for FastAPI/httpx so the historical core
suite can still run in installations that intentionally omit the optional mobile
API dependencies.

## Why trading commands are not duplicated in the native API

Remote order/start/live-unlock endpoints would create a second control path that
must perfectly reproduce all of the existing Streamlit confirmation, credential,
account-mode, and process-controller semantics. That would increase the chance of
a safety bypass.

Instead, the native companion has one explicit bridge back to the existing full
dashboard. This keeps the simple phone experience while preserving a single
authoritative path for money-moving actions.

A future native-control phase should only be considered after the engine itself
has a process-independent authenticated command bus with durable command IDs,
replay protection, explicit authorization scopes, audit records, and the same
live-unlock policy as the web dashboard.

## Release checklist

Before distributing a TestFlight/internal build:

1. Run the complete Python test suite with `requirements-mobile-api.txt`
   installed.
2. Run `npm install`, `npm run typecheck`, and Expo dependency checks in
   `mobile/`.
3. Run the app against a paper-only deployment first.
4. Confirm no Alpaca credential exists in the JS bundle, app config, build logs,
   or mobile source.
5. Confirm the production mobile API URL is HTTPS.
6. Rotate `MOBILE_API_TOKEN` if it was ever pasted into an issue, chat, build log,
   screenshot, or source file.
7. Confirm `MOBILE_DASHBOARD_URL` points to the intended protected dashboard.
8. Verify the phone shows the same persisted engine status, signals, trades, and
   safety locks as the Streamlit dashboard.
9. Keep crypto live mode disabled until the existing repository's separate live
   qualification gates are satisfied.

The mobile companion changes presentation and read access. It does not change the
strategy's expected return and makes no profitability guarantee.
