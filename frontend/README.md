# AegisOS app (`frontend/`)

The product's face: a realtime hospital paging OS. It has an operator console, a pager for
every clinician, and the patient, pharmacy, research and system workspaces as drill-downs,
all over one running AegisOS backend. Next.js 16 (App Router), React 19, Tailwind v4,
`motion` and three.js r160, built as a **static export** and deployed to Vercel.

```bash
cp .env.example .env.local   # NEXT_PUBLIC_AEGIS_BACKEND=http://127.0.0.1:8100 (make serve)
npm install
npm run dev                  # http://localhost:3000
npm run typecheck
npm run build                # → out/
```

## Routes

| Route | File | Reads |
|:--|:--|:--|
| `/` | `app/page.tsx` | `WS /aegis/paging/stream` (live numbers), `POST /aegis/sys/login` |
| `/operator/` | `app/operator/page.tsx` | `WS /aegis/paging/stream`, `/aegis/paging/hospital`; writes `POST /aegis/paging/alerts`, `…/pages/{id}/approve · escalate · cancel · resolve`, `…/clinicians/{id}/status`, `…/recs/{id}/ack`, `…/settings` |
| `/clinician/?id=` | `app/clinician/page.tsx` | `WS /aegis/paging/stream?clinician=<id>` (the pager is open), its own device session; writes `…/pages/{id}/respond`, `…/resolve`, `…/clinicians/{id}/status` |
| `/patient/?id=` | `app/patient/page.tsx` | `WS /aegis/stream` (live patient) or `/aegis/patient/snapshot?bed=` (cohort bed), `/aegis/cohort/{bed}/chart`, `/aegis/timeline`, `/aegis/decision`, `/aegis/doctor/*` |
| `/pharmacy/?id=` | `app/pharmacy/page.tsx` | `/aegis/pk/options?bed=`, `POST /aegis/pharmacopoeia/screen` |
| `/lab/` | `app/lab/page.tsx` | `/aegis/lab/evidence`, `/aegis/lab` |
| `/os/` | `app/os/page.tsx` | `/aegis/sys/*`, `/aegis/fs/*`, `/aegis/kernel/devices`, `/aegis/services` |

`/command/` and `/bed/` redirect to `/operator/` and `/patient/`. Which role lands where is
one table in `lib/ecosystem.ts`. The backend gate stays the authority on what a role may *do*.

## Layout

```text
app/                one directory per route; globals.css holds the design tokens
components/paging/  floor-view.tsx (the isometric building) · queue.tsx (queue, feed, actions)
                    page-drawer.tsx (dispatch, triage, routing, certificate, timeline, SBAR)
                    composer.tsx (⌘K, text or voice) · panels.tsx (KPIs, cases, staff,
                    insights, policy, room drawer) · bits.tsx (priority chips, countdowns, avatars)
components/         shell.tsx (app bar, user menu) · ui.tsx (plates, buttons, drawer, modal)
                    feed.tsx · twin.tsx · terminal.tsx · bed/ (patient panels, chart, orders)
engine/             twin3d.js, the anatomical twin, ported from webui/twin3d.js
lib/                paging.ts (types, the event reducer, usePaging, labels, clock) ·
                    api.ts (backend, gate token, session renewal) · hooks.ts · clinical.ts ·
                    ecosystem.ts
public/             vendor/models (organ meshes, CC BY 4.0) · _headers (Cloudflare fallback)
vercel.json         headers, ignored-build step (Vercel)
```

## How the live state works

`usePaging()` opens `WS /aegis/paging/stream`. It receives one snapshot and then one batch
of events a second (`page`, `alert`, `clinician`, `rec`, `monitor`, `stats`, `notify`, `settings`), and folds
them into a reducer. Every screen renders from that one store, so the floor, the queue, the
drawer and the pager never disagree. Time comes from the server (`useServerNow` corrects for
clock offset), so countdowns are the backend's deadlines, not the browser's guess.

## Rules this code keeps

- **Colour is signal.** The stock Tailwind palette is deleted in `globals.css`
  (`--color-*: initial`), so `bg-blue-500` does not compile. Red, orange, amber and grey are
  page priority (P1–P4). Green, blue, violet, amber and grey are staff status. Indigo is the
  one accent, for selection, focus and the primary action. A healthy reading is neutral, not
  green.
- **Paper, not glass.** White plates on a warm canvas, one hairline border, one soft shadow.
  Depth belongs to the building view, and the terminal and the anatomical twin keep dark
  stages of their own.
- **Numbers are tabular** (`.num`), so timers and counts never jitter. Wall-clock time is
  24-hour (`clock()` in `lib/paging.ts`).
- **Say what a number is.** A cohort bed's score is the fused, personalized cohort risk, not
  the calibrated detector, so it shows "no conformal claim". The 8 h projection continues the
  *population* line, because that is what the forecaster scores. Synthetic pages, patients
  and responders are labelled as such everywhere they appear.
- **The gate token travels in headers.** `api()` sends `Authorization: Bearer`, and sockets
  offer the `aegis.session.<token>` subprotocol. It never goes in a query string. A pager
  carries its own clinician's token, apart from the operator's, so the two cannot collide in
  one browser.
- **Small screens get a different layout, not a squeezed one.** Below `lg` the queue becomes
  a tab (the one a phone opens on). On a phone the pager page *is* the pager: full screen,
  with no device mockup.

## Traps found building it

- **A backend restart forgets every session** unless `AEGIS_SESSION_SECRET` is set (the
  secret is random per boot). A forgotten token is served as *guest* in audit mode, so a
  signed-in nurse's alert used to be recorded as guest's. The fix has two parts. The paging
  routes refuse a rejected session with `EAUTH`. The client renews the session for the same
  person and retries once (`revalidate()` in `lib/api.ts`), and it re-checks on every
  WebSocket reconnect, because a reconnect is what a restart looks like from here. An *ended*
  session (signed out, revoked) is not renewed.
- **An absolutely positioned child of a `<button>` starts at the button's centred static
  position.** The switch thumb needs `left-0`, or `translate-x` pushes it out of the track.
- **A flex child that scrolls on one axis has no min-height**, so a column squashes it to
  nothing. The small-screen tab row and the cases table carry `shrink-0`.
- **`AnimatePresence mode="wait"` crossfades need no absolute stacking.** Below `lg` the floor
  view sits in normal flow, so the drawing's own aspect ratio sets the height.
- **Reading `localStorage` during render breaks hydration.** The static HTML says "guest", so
  the first client render must too. Use `useSession()` (set in an effect), never
  `readSession()` inline in JSX.
- **Hidden tabs must go quiet.** `useFeed`, `usePoll` and the paging socket close while
  `document.hidden`. The backend is on Render's free tier, and a backgrounded dashboard used
  to drain it for hours. A pager keeps its socket, because it is a pager.
- **The twin is a singleton.** `engine/twin3d.js` owns one WebGL renderer and ~18 MB of
  meshes. `start()` on an already-started engine re-parents the canvas (`attach`), and
  unmounting calls `detach`, which parks the render loop. Do not create a second renderer per
  mount.
- **A static export cannot pre-render unknown ids**, so patients and pagers take the id as a
  query string, and pages that use `useSearchParams` are wrapped in `<Suspense>`.

## Deploying

Live at **https://aegis-clinical-beryl.vercel.app**: the Vercel project `aegis-clinical`,
root directory `frontend`, git-linked to `main`. The backend URL comes from `.env.production`
and the headers from `vercel.json`, so a build needs no dashboard configuration. A push that
does not touch `frontend/` skips the build. Cloudflare Pages is the documented fallback
(`public/_headers` mirrors `vercel.json`). See [`../deploy/README.md`](../deploy/README.md).
