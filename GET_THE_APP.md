# Getting the iOS app onto your iPhone

This is the shortest path from this repository to a running app plus a running
backend. [IOS_PRODUCTION_DEPLOYMENT.md](IOS_PRODUCTION_DEPLOYMENT.md) explains
the architecture and the safety boundaries; this file is the checklist.

## What can and cannot be automated

The build, the install link, and the whole backend are automated here. Two
things cannot be produced by this repository, by CI, or by any script, because
Apple issues them to a named human:

1. **An Apple Developer Program membership** (99 USD/year). Without it no `.ipa`
   can be signed, and an unsigned iOS app cannot be installed on any iPhone.
2. **An Expo account access token**, which is what lets the build run in the
   cloud instead of on a Mac you own.

There is also no way around Apple's distribution rules: iOS refuses to install
an app from an arbitrary download URL. A working install link is always either
an **internal-distribution link** (installs only on iPhones whose UDID is
registered with your Apple Developer account) or **TestFlight**. Anything that
claims otherwise is describing a jailbreak or a revoked enterprise certificate.

Once those two accounts exist, every build publishes an install link on its own.

## Step 1 — Backend (do this first)

The trading engine lives on the server, not the phone, so the app is useless
until the backend answers. On an always-on host with Docker installed and a
domain already pointed at it:

```bash
git clone <this repo> && cd trade
scripts/deploy_backend.sh your-domain.example
```

The script creates `.env`, generates a `MOBILE_API_TOKEN`, builds and starts the
stack, waits for both containers to report healthy, and prints the two values
the phone asks for. Fill in your Alpaca paper keys in `.env` and re-run it if it
warned that they were empty.

Confirm the backend from the outside before touching the phone:

```bash
scripts/verify_backend.sh
```

That checks the public health endpoint, checks an authenticated read, and checks
that an unauthenticated caller is rejected.

## Step 2 — Link the Expo project once

```bash
cd mobile
npm install
npx eas-cli@latest login
npx eas-cli@latest init
```

`eas init` writes the EAS project ID used for push tokens. Commit the resulting
change to `mobile/app.json`.

Then create an access token at <https://expo.dev/settings/access-tokens> and add
it to this repository as the **`EXPO_TOKEN`** secret
(Settings → Secrets and variables → Actions).

The first iOS build asks EAS to create signing credentials; log in with your
Apple Developer account when prompted and EAS manages the certificate and
provisioning profile from then on. Register your iPhone's UDID with that
account so internal builds can install on it.

## Step 3 — Get the link

Builds run in GitHub Actions under **Build production iOS app**:

| Trigger | Profile | Result |
| --- | --- | --- |
| Push to `main` touching `mobile/**` | `preview` | Install link in the run summary |
| Run the workflow manually | your choice | Install link in the run summary |
| Push a tag like `ios-v1.1.0` | `production` | Link also attached to the GitHub Release |
| Manual run with *submit* checked | `production` | Uploaded to TestFlight |

Open the run's summary page and use the **install page** link on the iPhone
itself. Safari hands the profile to iOS and the app installs like any other app.

Until `EXPO_TOKEN` exists, the workflow does not fail noisily — it writes a
summary explaining what is missing and stops.

To build from your own machine instead:

```bash
cd mobile
npm run build:ios:preview   # internal install link
npm run build:ios           # App Store / TestFlight build
npm run submit:ios          # send the latest production build to TestFlight
```

## Step 4 — First launch

The app asks for:

```text
Mobile API URL: https://your-domain.example/api
Access token:   <the token printed by deploy_backend.sh>
```

Both are stored in Expo SecureStore. Grant notification permission if you want
push alerts for engine state changes, safety locks, and fills.

After that the phone can be locked, backgrounded, or the app force-quit. The
engine keeps running on the backend, because the backend is the engine. If the
backend host is powered off, the bot is not running, and no phone setting
changes that.
