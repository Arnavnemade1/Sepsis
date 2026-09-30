# AegisOS clinical app (`frontend/`)

The product's face: five role workspaces over one running AegisOS backend. Next.js 16
(App Router), React 19, Tailwind v4 and three.js r160, built as a **static export** and
deployed to Vercel.

```bash
cp .env.example .env.local   # NEXT_PUBLIC_AEGIS_BACKEND=http://127.0.0.1:8100 (make serve)
npm install
npm run dev                  # http://localhost:3000
npm run typecheck
npm run build                # → out/
```

## Workspaces

| Route | File | Reads |
|:--|:--|:--|
| `/` | `app/page.tsx` | `/aegis/sys/accounts`, `POST /aegis/sys/login` |
| `/command/` | `app/command/page.tsx` | `WS /aegis/cluster/stream` |
| `/bed/?id=` | `app/bed/page.tsx` | `WS /aegis/stream` (live patient) or `/aegis/patient/snapshot?bed=` (cohort bed), `/aegis/cohort/{bed}/chart`, `/aegis/timeline`, `/aegis/decision`, `/aegis/doctor/*` |
| `/pharmacy/?id=` | `app/pharmacy/page.tsx` | `/aegis/pk/options?bed=`, `POST /aegis/pharmacopoeia/screen` |
| `/lab/` | `app/lab/page.tsx` | `/aegis/lab/evidence`, `/aegis/lab` |
| `/os/` | `app/os/page.tsx` | `/aegis/sys/*`, `/aegis/fs/*`, `/aegis/kernel/devices`, `/aegis/services` |

Which role lands where, and what the nav offers, is one table in `lib/ecosystem.ts`. The
backend gate stays the authority on what a role may *do*.

## Layout

```text
app/          one directory per workspace; globals.css holds the design tokens
components/   shell.tsx (top bar, nav, backend status, user) · ui.tsx (plates, stats, meters)
              feed.tsx · twin.tsx · terminal.tsx · bed/ (bedside panels, chart, selector)
engine/       twin3d.js — the anatomical twin, ported from webui/twin3d.js
lib/          api.ts (backend + gate token) · hooks.ts (useFeed, usePoll, useSession)
              types.ts (backend contracts) · clinical.ts (tone + formatting) · ecosystem.ts
public/       vendor/models (organ meshes, CC BY 4.0) · _headers (Cloudflare fallback)
vercel.json   headers, ignored-build step (Vercel)
```

## Rules this code keeps

- **Colour is signal.** The stock Tailwind palette is deleted in `globals.css`
  (`--color-*: initial`), so `bg-blue-500` does not compile. Green, amber, orange and red
  mean clinical state; the brand gradient is for identity moments only; a healthy reading is
  neutral, not green. `lib/clinical.ts` is the only place a state hue is chosen.
- **A bar's fill is its value.** `Meter` and every chart draw the number, never a multiple of
  it. Risk charts keep a fixed 0–1 axis.
- **Say what a number is.** A cohort bed's score is the fused, personalized cohort risk, not
  the calibrated detector, so it shows "no conformal claim". The 8 h projection continues the
  *population* line because that is what the forecaster scores. Backend honesty statements are
  rendered verbatim.
- **The gate token travels in headers.** `api()` sends `Authorization: Bearer`; sockets offer
  the `aegis.session.<token>` subprotocol. Never a query string.
- **No `?id=` means a sensible default, not an error.** Bedside falls back to the live
  session; Pharmacy to the first cohort bed.

## Traps found building it

- **The backend signs everyone out on restart** unless `AEGIS_SESSION_SECRET` is set (the
  secret is random per boot). `UserChip` asks `/aegis/sys/whoami` and drops a token the kernel
  no longer knows, instead of showing a name the gate treats as a guest.
- **Hidden tabs must go quiet.** `useFeed` closes its socket and `usePoll` stops while
  `document.hidden`; the backend is on Render's free tier and a backgrounded dashboard used to
  drain it for hours.
- **The twin is a singleton.** `engine/twin3d.js` owns one WebGL renderer and ~18 MB of meshes.
  `start()` on an already-started engine re-parents the canvas (`attach`), and unmount calls
  `detach`, which parks the render loop. Do not create a second renderer per mount.
- **Two "15% budget" numbers exist** in the detector manifest: `detect_at_far15` and
  `alarm_budget.det15`. "Caught within a 15% alarm budget" is the second, and it is what the
  top-level README publishes.
- **Artifacts carry their own detector stamp.** The Lab shows `matches_detector` on every
  block; an artifact scored on an earlier fit is displayed, never passed off as current.
- **A static export cannot pre-render unknown ids**, which is why patients are query strings,
  and why pages using `useSearchParams` are wrapped in `<Suspense>`.

## Deploying

Live at **https://aegis-clinical-beryl.vercel.app**: the Vercel project `aegis-clinical`,
root directory `frontend`, git-linked to `main`. The backend URL comes from `.env.production`
and the headers from `vercel.json`, so a build needs no dashboard configuration. A push that
does not touch `frontend/` skips the build. Cloudflare Pages is the documented fallback
(`public/_headers` mirrors `vercel.json`). See [`../deploy/README.md`](../deploy/README.md).
