# Getting the iOS app onto your iPhone

This is the shortest path from this repository to a running app plus a running
backend. [IOS_PRODUCTION_DEPLOYMENT.md](IOS_PRODUCTION_DEPLOYMENT.md) explains
the architecture and the safety boundaries; this file is the checklist.

## Two routes

| | Free Apple ID (sideload) | Paid Apple Developer |
| --- | --- | --- |
| Cost | Free | 99 USD/year |
| Workflow | **Build unsigned iOS IPA** | **Build production iOS app** |
| You get | An `.ipa` you sign yourself | An install link / TestFlight |
| App expires | **Every 7 days** | No |
| Push notifications | **No** | Yes |
| Needs a computer to install | Yes | No |

Both routes need the backend from step 1. Pick your route in step 2.

Apple's distribution rules apply either way: iOS never installs an app straight
from an arbitrary download URL. Anything claiming otherwise is describing a
jailbreak or a revoked enterprise certificate.

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

## Step 2, free route — Sideload with a free Apple ID

Any Apple ID can sign apps for **your own devices**, with no paid membership.
This is what AltStore, SideStore, and Sideloadly automate. It is legitimate and
uses Apple's own tooling, but Apple caps it deliberately:

- **The app stops opening after 7 days.** You reconnect it to re-sign. AltStore
  and SideStore can automate this refresh; Sideloadly is manual.
- **Push notifications do not work.** The `aps-environment` entitlement requires
  a paid membership. Your backend still detects engine state changes, safety
  locks, and fills, but the phone will not be alerted. You see everything when
  you open the app. For a trading tool this is the limitation that matters —
  decide whether unattended alerts are something you need.
- **Three sideloaded apps** per Apple ID at a time, and **ten new App IDs per
  week**, so avoid repeated reinstalls in one sitting.
- You need a computer for the initial install (and for refreshes, unless you use
  SideStore's on-device refresh).

Background refresh still works: it uses ordinary background modes, not a paid
entitlement. And none of this touches trading safety, because the engine runs on
the backend, not the phone.

Run the **Build unsigned iOS IPA** workflow in GitHub Actions (Actions tab →
select it → Run workflow), or push a tag like `sideload-v1`. It compiles the app
on a macOS runner with signing disabled and uploads the `.ipa` as a run
artifact. Nothing is sent to Apple and no account is needed for this step.

> This repository is private, so macOS runner minutes bill at GitHub's 10x rate.
> A build takes roughly 15–25 minutes of macOS time. Only rebuild when the app
> code changes — re-signing an existing `.ipa` weekly costs nothing.

Then, on your computer:

1. Download the artifact and unzip it to get the `.ipa`.
2. Install [Sideloadly](https://sideloadly.io) (Windows/macOS) or
   [AltStore/SideStore](https://altstore.io).
3. Plug in the iPhone, drag in the `.ipa`, sign in with your Apple ID.
4. On the iPhone: Settings → General → VPN & Device Management → trust your
   developer certificate.

Use a throwaway Apple ID if you would rather not put your main account into a
third-party sideloading tool. Skip to step 4.

To build it yourself on a Mac instead, no CI needed:

```bash
cd mobile
npm install
npx expo prebuild --platform ios --clean
open ios/*.xcworkspace   # select your personal team in Signing, then Run
```

## Step 2, paid route — Link the Expo project once

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

## Step 3, paid route — Get the link

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
push alerts for engine state changes, safety locks, and fills. On a
free-Apple-ID sideload that registration fails silently by design — the app
catches it and keeps working, it just never receives push.

After that the phone can be locked, backgrounded, or the app force-quit. The
engine keeps running on the backend, because the backend is the engine. If the
backend host is powered off, the bot is not running, and no phone setting
changes that.
