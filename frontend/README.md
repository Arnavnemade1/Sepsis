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
| `/` | `app/page.tsx` | nothing: it redirects to `/operator/` (at the edge on Vercel via `vercel.json`, client-side elsewhere) |
| `/operator/?view=overview·pages·staff·sentinel·policy` | `app/operator/page.tsx` | `WS /aegis/paging/stream`, `/aegis/paging/hospital`; writes `POST /aegis/paging/alerts`, `…/pages/{id}/approve · escalate · cancel · resolve`, `…/clinicians/{id}/status`, `…/recs/{id}/ack`, `…/settings` |
| `/clinician/?id=` | `app/clinician/page.tsx` | `WS /aegis/paging/stream?clinician=<id>` (the pager is open), its own device session; writes `…/pages/{id}/respond`, `…/resolve`, `…/clinicians/{id}/status` |
| `/room/` · `/room/?id=` | `app/room/page.tsx`, `components/rooms/*` | the board reads `/aegis/rooms` every 5 s; a room reads `/aegis/rooms/{room}` every 4 s (record, vitals series, labs, findings, pages, care team, devices, timeline) |
| `/patient/` · `/patient/?id=` | `app/patient/page.tsx`, `components/bed/*` | the census reads `/aegis/cluster/snapshot` and the paging stream; the record reads `WS /aegis/stream` (live patient) or `/aegis/patient/snapshot?bed=` (cohort bed), `/aegis/cohort/{bed}/chart`, `/aegis/timeline`, `/aegis/decision`, `/aegis/doctor/*` |
| `/pharmacy/?id=` | `app/pharmacy/page.tsx` | `/aegis/pk/options?bed=`, `POST /aegis/pharmacopoeia/screen` |
| `/lab/` | `app/lab/page.tsx` | `/aegis/lab/evidence`, `/aegis/lab` |
| `/integrations/?view=overview·mappings·endpoints` | `app/integrations/page.tsx`, `components/integrations/*` | `/aegis/interop` every 2.5 s, `/aegis/interop/{messages,mappings}/{id}`, `/aegis/interop/sim`, `/aegis/interop/audit`, `/aegis/interop/bench`, `/aegis/fhir/R4/metadata`, `/aegis/cds-services`, `/aegis/interop/webhooks`; writes `…/mappings/{id}/adopt · reject · restore · reevaluate`, `POST /aegis/interop/hl7`, `…/sim/drill`, `POST /aegis/cds-services/{id}`, `…/webhooks` |
| `/os/?view=architecture·kernel·journal·gate·shell` | `app/os/page.tsx`, `components/os/*` | `/aegis/proc` (the kernel tree in one call), `/aegis/sys/{gate,abi,who,journal,ps,boot}`, `/aegis/kernel/devices`, `/aegis/services`, `/aegis/devices`, `/aegis/consensus`, `/aegis/orchestrator`, `/aegis/health/prover`, `/aegis/lab/evidence`, plus the paging stream |

`/command/` and `/bed/` redirect to `/operator/` and `/patient/`. Which role lands where is
one table in `lib/ecosystem.ts`. The backend gate stays the authority on what a role may *do*.

## Layout

```text
app/                one directory per route; globals.css holds the design tokens
components/paging/  floor-view.tsx (the isometric building) · queue.tsx (queue, feed, actions)
                    page-drawer.tsx (dispatch, triage, routing, certificate, timeline, SBAR)
                    composer.tsx (N, text or voice) · panels.tsx (KPIs with hour sparklines, pages, staff,
                    insights, policy) · bits.tsx (priority chips, countdowns, avatars)
components/integrations/  overview.tsx (interfaces, message stream, quarantines, review, drills, validation) ·
                    mappings.tsx (every local code and its status) · drawers.tsx (one HL7 message as received; one
                    mapping with its evidence and the decision) · evidence-chart.tsx (values against the reference) ·
                    endpoints.tsx (HL7 console, FHIR, CDS Hooks try-out, webhooks)
components/rooms/   unit.tsx (one room as a unit: header, scores, detections, vitals, labs, record,
                    linked objects, timeline) · body.tsx (the anatomical twin lit by the room, pins,
                    systems) · board.tsx (every room as a tile) · drawer.tsx (the overview's watchlist and
                    room drawer) · bits.tsx (severity, NEWS2, allergy and code chips)
components/         shell.tsx (sidebar, top bar, status, user) · nav.ts (the section model) · command.tsx (⌘K palette)
                    os/ (architecture, kernel, journal, gate, shell views) · bed/census.tsx · ui.tsx (panels, tags,
                    buttons, inspector sheet, modal, mini charts, Restricted)
                    feed.tsx · twin.tsx · terminal.tsx · bed/ (patient panels, chart, orders)
engine/             twin3d.js, the anatomical twin, ported from webui/twin3d.js (plus pinned system
                    labels, hover/highlight, camera focus, an idle sway and MSAA through the composer)
lib/                paging.ts (types, the event reducer, usePaging, labels, clock) · rooms.ts (room-unit
                    contracts, severity hues, useBoard/useUnit) ·
                    api.ts (backend, gate token, session renewal) · hooks.ts · clinical.ts ·
                    ecosystem.ts
public/             vendor/models (organ meshes, CC BY 4.0) · _headers (Cloudflare fallback)
vercel.json         headers, ignored-build step (Vercel)
```

## How the live state works

`usePaging()` opens `WS /aegis/paging/stream`. It receives one snapshot and then one batch
of events a second (`page`, `alert`, `clinician`, `rec`, `monitor`, `rooms`, `stats`, `notify`, `settings`), and folds
them into a reducer. Every screen renders from that one store, so the floor, the queue, the
drawer and the pager never disagree. Time comes from the server (`useServerNow` corrects for
clock offset), so countdowns are the backend's deadlines, not the browser's guess.

## Rules this code keeps

- **An operations console, not a website.** There is no landing page, no hero, no serif display
  type, no gradients or glass, and no decorative icons. One system bar (connection latency,
  prover health, clock), a grouped sidebar and a ⌘K command palette frame every screen. The palette is stealth
  grey (`#212429` canvas, `#282c32` panels): dark enough for a wall display, not black.
- **Colour is signal.** The stock Tailwind palette is deleted in `globals.css`
  (`--color-*: initial`), so `bg-blue-500` does not compile. Red, orange, amber and grey are
  page priority (P1–P4), and the same four hues are a room monitor finding's severity
  (critical, urgent, watch, information), so a room and its page always speak one colour. Green, blue, violet, amber and grey are staff status. Indigo is the
  one accent, for selection, focus and the primary action. A healthy reading is neutral, not
  green.
- **Flat and exact.** Panels are a hairline border on a slightly lighter grey with 6–8 px radii.
  Depth is kept for overlays (menus, the inspector, the composer). The terminal and the
  anatomical twin have darker stages of their own.
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

- **The demo opens signed in, and sign-out has to stick.** `useEnsureSession()` signs in as the
  charge nurse when there is no session, unless the person signed out (the `aegis_signed_out`
  flag). Without that flag, signing out would undo itself on the next render.

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
  mount. A page that changes the engine's shared state (pins, the HUD, the camera, its
  listeners) puts it back on unmount: the room view does, so the patient record's twin never
  inherits a room's pins or a zoomed camera.
- **A static export cannot pre-render unknown ids**, so patients and pagers take the id as a
  query string, and pages that use `useSearchParams` are wrapped in `<Suspense>`.

## Deploying

Live at **https://aegis-clinical-beryl.vercel.app**: the Vercel project `aegis-clinical`,
root directory `frontend`, git-linked to `main`. The backend URL comes from `.env.production`
and the headers from `vercel.json`, so a build needs no dashboard configuration. A push that
does not touch `frontend/` skips the build. Cloudflare Pages is the documented fallback
(`public/_headers` mirrors `vercel.json`). See [`../deploy/README.md`](../deploy/README.md).
