# Deploying AegisOS

AegisOS is **three deployables** with different hosting needs:

| Piece | What it is | Where it runs |
|-------|------------|---------------|
| **Clinical app** (`frontend/`) | The role-routed Next.js app: Command, Bedside, Pharmacy, Lab, System. A static export that talks to the backend from the browser | **Cloudflare Pages** (static, **unlimited bandwidth**) |
| **Backend** (`backend/`) | FastAPI: the REST surface (`backend/api/routes/`), the `/aegis/stream` and `/aegis/cluster/stream` **WebSockets**, the kernel, the ICU wing, real-data ingest, Supabase persistence | **Render** (persistent web service, from [`render.yaml`](../render.yaml)) |
| **Legacy shell** (`webui/`) | The previous single-page OS shell, kept deployed until every feature has a home in `frontend/` | Cloudflare Pages (being retired) |

> **Why split this way?** The backend must run somewhere persistent: a serverless host cannot hold a WebSocket open or run the background drive loop. The frontends are static; the anatomical twin alone is ~18 MB of organ meshes, and serving that from the backend burns its bandwidth quota, so both live on **Cloudflare Pages**, the only major free host with *uncapped* bandwidth. See **Bandwidth budget** below.

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

## 2 · Deploy the clinical app (`frontend/`) to Cloudflare Pages

1. **dash.cloudflare.com → Workers & Pages → Create → Pages → Connect to Git**, select this repo.
2. Build settings:
   - **Framework preset:** `Next.js (Static HTML Export)`
   - **Root directory:** `frontend`
   - **Build command:** `npm run build`
   - **Build output directory:** `out`
   - **Environment variable:** `NEXT_PUBLIC_AEGIS_BACKEND` = your Render URL (e.g. `https://sepsis-backend-omd1.onrender.com`). It is baked in at build time.
   - **Node:** 20.9 or newer (`NODE_VERSION=20` if the image defaults lower).
3. Deploy. Every route is pre-rendered (`/`, `/command/`, `/bed/`, `/pharmacy/`, `/lab/`, `/os/`); per-patient pages take the patient as a query string (`/bed/?id=bed_6`), which is why no server is needed.

Cloudflare reads [`frontend/public/_headers`](../frontend/public/_headers): the organ meshes and hashed build output are cached `immutable` for a year, pages revalidate so a deploy lands immediately, and camera/microphone are allowed on the app's own origin (the optical biosensor and the voice consult). CORS on the backend is already `*`.

> **Point a deployed app at a different backend without rebuilding:** in the browser console,
> `localStorage.setItem('aegis_backend','https://<service>.onrender.com'); location.reload()`

### The legacy shell (`webui/`), while it is being retired

Same Pages flow with **Framework preset** `None`, **Build command** empty, **Build output directory** `webui`. It reads its backend from [`webui/config.js`](../webui/config.js) and its cache policy from [`webui/_headers`](../webui/_headers).

## 3 · Verify the live site

Open the Pages URL and confirm:
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
