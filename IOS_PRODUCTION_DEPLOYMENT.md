# Standalone iOS Production Deployment

This repository can run as a normal signed iOS app without Expo Go. The trading
runtime remains on an always-on backend, so closing, suspending, or locking the
iPhone does not stop the bot. The phone is a secure native client of that backend.

## Why the trading engine is server-side

iOS does not allow a general JavaScript app to execute continuously in the
background. `expo-background-task` uses Apple's scheduled background task system;
wake-ups are opportunistic and are not a continuous-process guarantee. The mobile
app therefore never owns the trading loop. The Docker/Streamlit backend owns the
engine, Alpaca credentials, reconciliation, protective stops, and persisted state.

The native app uses background tasks only for lightweight connectivity refresh and
uses Expo push notifications for engine-state changes, safety locks, and new fills.
Push delivery and iOS background scheduling are conveniences; neither is part of
the order-safety path.

## 1. Backend host

Use a machine that stays online while the bot is expected to run. A small VPS,
always-on desktop/server, or another persistent Docker host is suitable. If that
machine is powered off, the trading process is no longer running; keeping the
phone open cannot replace the backend.

Install Docker Engine and Docker Compose. Point a DNS A/AAAA record for a domain
you control at the host.

Copy the environment template and fill only local secrets:

```bash
cp .env.example .env
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Set in `.env`:

```text
TRADE_DOMAIN=your-domain.example
MOBILE_API_TOKEN=<the generated value>
ALPACA_PAPER_API_KEY=<paper key>
ALPACA_PAPER_SECRET_KEY=<paper secret>
ENABLE_LIVE_TRADING=false
```

Start the production stack:

```bash
docker compose -f docker-compose.production.yml up -d --build
```

Caddy terminates HTTPS automatically and exposes:

- `https://<TRADE_DOMAIN>/` — the existing Streamlit control dashboard.
- `https://<TRADE_DOMAIN>/api/health` — public API liveness only.
- `https://<TRADE_DOMAIN>/api/v1/...` — bearer-authenticated mobile API.

The Streamlit dashboard and mobile API share the same durable `trade_data` Docker
volume. Both services restart automatically with `restart: unless-stopped`. The
mobile API is one worker so its push observer is singular. The trading engine's
cross-process lock remains authoritative and prevents two engine processes from
trading against the same database.

Verify the stack:

```bash
docker compose -f docker-compose.production.yml ps
curl -fsS "https://${TRADE_DOMAIN}/api/health"
```

Start with Alpaca paper mode and verify the dashboard/engine lifecycle before any
live-money enablement.

## 2. Native app connection

The app never stores Alpaca credentials. On first launch enter:

```text
Mobile API URL: https://<TRADE_DOMAIN>/api
Access token:   <MOBILE_API_TOKEN>
```

The URL and token are stored with Expo SecureStore. The app polls while open,
registers an iOS Expo push token when notification permission is granted, and
registers a lightweight scheduled background task.

The backend sends notifications for:

- stock/crypto engine state transitions;
- newly armed safety locks;
- newly persisted trade fills.

Notification failure cannot submit, cancel, resize, or otherwise alter an order.

## 3. Create the EAS project once

An Expo account and Apple Developer account are required to produce/sign an App
Store/TestFlight iOS build. From the mobile directory:

```bash
cd mobile
npm install
npx eas-cli@latest login
npx eas-cli@latest init
```

`eas init` links this repository to the Expo project and writes the EAS project ID
used by `getExpoPushTokenAsync`. No project ID or Apple credential is hard-coded in
this repository.

Validate dependencies and TypeScript:

```bash
npm run typecheck
npm run doctor
```

## 4. Build a real iPhone app

Preview/internal signed build:

```bash
npm run build:ios:preview
```

Production App Store/TestFlight build:

```bash
npm run build:ios
```

The `production` EAS profile auto-increments the remote iOS build number. EAS can
manage the signing certificate/provisioning profile after the Apple Developer
account is linked.

Submit the latest production build to App Store Connect/TestFlight:

```bash
npm run submit:ios
```

After Apple finishes processing the build, install it through TestFlight like a
normal iOS app. Expo Go is not required.

## 5. GitHub build workflow

For manual cloud builds from GitHub, add an `EXPO_TOKEN` repository secret and run
**Build production iOS app** from GitHub Actions. The workflow validates the EAS
project and queues a non-interactive signed production build.

Apple/EAS credentials are external account credentials and are intentionally not
committed. The build cannot be signed until those accounts are linked once.

## 6. Runtime behavior when the phone is closed

Once the backend engine is running:

1. The trading loop, state database, broker reconciliation, protective stops, and
   session controls execute on the backend host.
2. The iPhone may be locked, backgrounded, or the Trade app may be closed.
3. Expo/Apple push infrastructure can still deliver server-originated alerts.
4. iOS may occasionally wake the app's background health task, but no trading
   decision depends on that wake-up.
5. Opening Trade later reads the same durable server state and catches up on
   trades, logs, equity, signals, and engine state.

If the user force-quits the app, iOS background JavaScript should not be expected
to keep executing. This is safe because the app is not the engine.

## 7. Live-money boundary

This production packaging does not weaken the existing live safeguards. Live crypto
still requires the server feature flag, separate live Alpaca credentials, the
existing dashboard acknowledgement/typed unlock, account/endpoint verification,
and live exposure/notional ceilings. The native API has no order-entry endpoint
and cannot silently unlock live mode.

This separation is deliberate: the app is a real always-available client, while
the server is the only authority allowed to operate the trading engine.
