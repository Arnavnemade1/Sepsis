# Deploying AegisOS

AegisOS is **four deployables**. What runs where today:

| Piece | What it is | Where it runs |
|-------|------------|---------------|
| **Clinical app** (`frontend/`) | The role-routed Next.js app: Command, Bedside, Pharmacy, Lab, System. A static export that talks to the backend from the browser | **Vercel**, project `aegis-clinical` → https://aegis-clinical-beryl.vercel.app |
| **Backend** (`backend/`) | FastAPI: the REST surface (`backend/api/routes/`), the `/aegis/stream` and `/aegis/cluster/stream` **WebSockets**, the kernel, the ICU wing, real-data ingest, Supabase persistence | **Render**, from [`render.yaml`](../render.yaml) → https://sepsis-backend-omd1.onrender.com |
| **Legacy shell** (`webui/`) | The previous single-page OS shell, kept deployed until every feature has a home in `frontend/` ([`PARITY.md`](../frontend/PARITY.md)) | **Vercel**, project `sepsis` → https://sepsis-three.vercel.app (being retired) |
| **Public site** (`site/`) | The marketing site ([`site/README.md`](../site/README.md)) | **Vercel**, project `sepsis-jwf1` → https://sepsis-jwf1.vercel.app |

All three Vercel projects are git-linked to `main`: a push redeploys them (the clinical app only when `frontend/` changed since its last successful deploy, compared against `VERCEL_GIT_PREVIOUS_SHA`; see `ignoreCommand` in [`frontend/vercel.json`](../frontend/vercel.json). Comparing only `HEAD^` skipped a push whose last commit happened not to touch `frontend/`). Render redeploys only for backend changes (`buildFilter` in `render.yaml`).

> **Why split this way?** The backend must run somewhere persistent: a serverless host cannot hold a WebSocket open or run the background drive loop. The frontends are static; the anatomical twin alone is ~18 MB of organ meshes, and serving that from the backend would burn its bandwidth quota, so they live on a static host. Vercel's free tier allows 100 GB a month; the meshes are cached `immutable`, so a returning visitor downloads them once. If bandwidth ever becomes the constraint, **Cloudflare Pages** (uncapped) is the documented fallback below, and `frontend/public/_headers` is kept for it. See **Bandwidth budget**.

---

## 1 · Deploy the backend to Render

1. Go to **render.com → New → Blueprint** and connect this GitHub repo. Render reads [`render.yaml`](../render.yaml) and creates the `sepsis-backend` web service.
2. In the service's **Environment**, set the secrets (these are NOT in the repo):
   - `SUPABASE_URL` — `https://oyukwcuwiyiuslmmbvou.supabase.co`
   - `SUPABASE_KEY` — your Supabase **service-role** key (from `backend/.env`)
   - `GEMINI_API_KEY` — primary LLM provider, and the only one that carries the
     AI-doctor **voice consult** (`/aegis/doctor/live`). Without it the consult
     panel still works, but typed-only: the call button disables itself and says
     so. **This is the one variable the voice call cannot start without** —
     voice relays through the backend precisely so the key never ships to the
     browser, which means it has to exist *here*, not in the frontend build.
   - `OPENROUTER_API_KEY` — text-only backup provider, so a Gemini quota wall
     degrades the narrative layer instead of removing it.
   - `OPENROUTER_MODEL` — set to `nvidia/nemotron-3-ultra-550b-a55b:free`. If an
     older value is still set it silently pins the backup to that model; check
     `primary_model` under `openrouter` in `/aegis/doctor/status`.

   Confirm all three landed:

   ```bash
   curl -s https://<your-service>.onrender.com/aegis/doctor/status
   ```

   `voice` must be `true`. If it is `false`, `voice_reason` says why.
3. Deploy. You get a URL like **`https://sepsis-backend.onrender.com`**.
4. Verify: open `https://sepsis-backend.onrender.com/aegis/health` → `{"status":"ok", ...}`.

> Render's **free** tier supports WebSockets but **sleeps after ~15 min idle**; the first request after idle cold-starts in ~30–60 s. Upgrade to a paid instance to keep it always-on for a live demo/judging.

> The backend also serves the legacy shell (`webui/`) at its root, so the Render URL is a working app on its own. The clinical app (`frontend/`) is deployed separately, below.

## 2 · Deploy the clinical app (`frontend/`)

### On Vercel (how it runs today)

The project `aegis-clinical` builds `frontend/` with the Next.js preset (Node 24), git-linked to `main`. Everything a build needs is in the repo:

- [`frontend/.env.production`](../frontend/.env.production) sets `NEXT_PUBLIC_AEGIS_BACKEND` to the Render URL. It is public (the browser calls it directly) and baked in at build time; a local `.env.local` still wins for development.
- [`frontend/vercel.json`](../frontend/vercel.json) caches the organ meshes for a year, sets the security headers, allows camera and microphone on the app's own origin (the optical biosensor and the voice consult), and skips the build when a push does not touch `frontend/`.

To recreate it: **vercel.com → Add New → Project → import `Arnavnemade1/Sepsis` → Root Directory `frontend`** (the preset is detected), no environment variables needed. From a terminal the same thing is `POST /v11/projects` with `{"name", "framework": "nextjs", "rootDirectory": "frontend", "gitRepository": {"type": "github", "repo": "Arnavnemade1/Sepsis"}}` via `vercel api`.

Deployment Protection stays on **Standard**: the production domain is public, per-commit preview URLs require a Vercel login.

### On Cloudflare Pages (the uncapped-bandwidth fallback)

1. **dash.cloudflare.com → Workers & Pages → Create → Pages → Connect to Git**, select this repo.
2. Build settings:
   - **Framework preset:** `Next.js (Static HTML Export)`
   - **Root directory:** `frontend`
   - **Build command:** `npm run build`
   - **Build output directory:** `out`
   - **Environment variable:** `NEXT_PUBLIC_AEGIS_BACKEND` = your Render URL (e.g. `https://sepsis-backend-omd1.onrender.com`). It is baked in at build time.
   - **Node:** 20.9 or newer (`NODE_VERSION=20` if the image defaults lower).
3. Deploy. Every route is pre-rendered (`/`, `/command/`, `/bed/`, `/pharmacy/`, `/lab/`, `/os/`); per-patient pages take the patient as a query string (`/bed/?id=bed_6`), which is why no server is needed.

Cloudflare reads [`frontend/public/_headers`](../frontend/public/_headers), which mirrors the headers in `vercel.json`. CORS on the backend is already `*`.

> **Point a deployed app at a different backend without rebuilding:** in the browser console,
> `localStorage.setItem('aegis_backend','https://<service>.onrender.com'); location.reload()`

### The legacy shell (`webui/`), while it is being retired

It runs as the Vercel project `sepsis` (root `webui`, no build step). It reads its backend from [`webui/config.js`](../webui/config.js). On Cloudflare Pages it would be **Framework preset** `None`, **Build command** empty, **Build output directory** `webui`, with its cache policy in [`webui/_headers`](../webui/_headers).

## 3 · Verify the live site

Open https://aegis-clinical-beryl.vercel.app and confirm:
- The launcher lists the demo staff (it reads `/aegis/sys/accounts`). If it says the backend is waking, a free-tier Render service is cold-starting (~30–60 s); the page retries on its own.
- Sign in as **Priya Patel, RN**: **Command** shows twelve beds and the ward-feed pill reads **live**.
- Open a bed: the anatomical twin loads and the risk, vitals and decision populate.
- **System → Shell**: `uname -a` answers with the running kernel's boot id.

---

## Local development

```bash
make setup && make serve                        # backend on http://127.0.0.1:8100
cd frontend && cp .env.example .env.local        # points the app at :8100
npm install && npm run dev                        # the app on http://localhost:3000
```

---

## Bandwidth budget (keeping Render free-tier alive)

Render's bandwidth is the scarce resource. Four levers keep egress low:

1. **Static assets off Render.** Both frontends are served by Cloudflare Pages (unlimited), not Render. Render serves only the JSON API and the WebSockets.
2. **MediaPipe models from Google's CDN.** The ~16.6 MB of `.task` models load from `storage.googleapis.com`, not our hosts ([`webui/camera.js`](../webui/camera.js)). The `vendor/models/*.task` copies remain only as an offline fallback.
3. **Immutable caching.** Both Cloudflare ([`webui/_headers`](../webui/_headers)) and the backend's fallback static mount (`_SmartStatic` in [`backend/aegis/server.py`](../backend/aegis/server.py)) cache `vendor/` for a year, so repeat visits cost ~0 bytes. (This replaced an old `no-store`-everything policy that re-served 35 MB per reload.)
4. **WebSocket throttle + tab-aware pause.** Stream ticks are `TICK_SECONDS=3.0` / `CLUSTER_TICK_SECONDS=2.0`; clients close the socket and stop polling on hidden tabs (`visibilitychange` in [`webui/os.js`](../webui/os.js) / [`webui/tactical.js`](../webui/tactical.js)), so a backgrounded dashboard no longer drains bandwidth for hours. The clinical app does the same in `useFeed`/`usePoll` ([`frontend/lib/hooks.ts`](../frontend/lib/hooks.ts)).

## Notes
- **Secrets** live only in Render's dashboard / local `backend/.env` (gitignored) — never committed.
- **Set `AEGIS_SESSION_SECRET`** on Render. Without it the session-signing secret is random per boot, so every restart (and every free-tier cold start) signs every user out. The app notices and returns them to sign-in, but a fixed secret avoids it.
- **Database stays under free-tier limits**: only real (camera-tagged) data is persisted; an hourly `pg_cron` job (`aegis_retain`) rolls raw telemetry >7 days into 1-minute summaries and prunes the rest. See [`supabase/migrations/`](../supabase/migrations/).
