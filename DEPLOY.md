# Deploying AegisOS / HyperSepsisTwin-Ω

The app is **two pieces** with different hosting needs:

| Piece | What it is | Where it runs |
|-------|------------|---------------|
| **Frontend** (`webui/`) | Static OS shell + the fully client-side optical biosensor (camera rPPG, detection, 3D twin) | **Cloudflare Pages** (static, **unlimited bandwidth**) |
| **Backend** (`backend/`) | FastAPI server: `/aegis/stream` **WebSocket**, simulator/risk brain, real-data ingest, Supabase persistence | **Render** (persistent web service) |

> **Why split this way?** The backend must run somewhere persistent — a serverless host can't hold a WebSocket open or run the background drive loop. The frontend, by contrast, is ~35 MB of static assets (3D organ GLBs, three.js); serving those from the backend burns its bandwidth quota fast, so the frontend lives on **Cloudflare Pages** (the only major free host with *uncapped* bandwidth) and the backend keeps only the API + WebSocket. See **Bandwidth budget** below.

---

## 1 · Deploy the backend to Render

1. Go to **render.com → New → Blueprint** and connect this GitHub repo. Render reads [`render.yaml`](render.yaml) and creates the `sepsis-backend` web service.
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

> The backend also serves the frontend itself, so `https://sepsis-backend.onrender.com` is a complete working app on its own — Vercel is optional.

## 2 · Deploy the frontend to Cloudflare Pages

First point the frontend at the backend — edit one line in [`webui/config.js`](webui/config.js):

```js
window.AEGIS_BACKEND = "https://sepsis-backend.onrender.com";
```

Then create the Pages project:

1. **dash.cloudflare.com → Workers & Pages → Create → Pages → Connect to Git**, select this repo.
2. Build settings:
   - **Framework preset:** `None`
   - **Build command:** *(leave empty — it's pre-built static)*
   - **Build output directory:** `webui`
3. Deploy. You get a URL like **`https://aegis-os.pages.dev`**.

Cloudflare reads [`webui/_headers`](webui/_headers) automatically: the ~35 MB of `vendor/` assets are cached `immutable` for a year (repeat visitors download them once), while JS/CSS/HTML revalidate so deploys take effect immediately. CORS on the backend is already `*`, so the new origin works with no backend change.

> **Quick test without deploying:** on any served copy of the frontend, run in the console
> `localStorage.setItem('aegis_backend','https://sepsis-backend.onrender.com'); location.reload()`

## 3 · Verify the live site

Open the Cloudflare Pages URL and confirm:
- The connection pill reads **`linked`** (not `offline`).
- The **Sepsis Risk Core**, **Live Telemetry**, **Multi-Agent**, etc. populate and update.
- Enabling the **Optical Biosensor** streams real readings → the pill flips to **`LIVE · CAMERA`** and rows land in Supabase (`telemetry_logs`, tagged `source='camera'`).

---

## Local development

```bash
pip install -r requirements.txt
python -m backend.aegis.server      # serves API + WebSocket + frontend at http://127.0.0.1:8100
```

`webui/config.js` left empty ⇒ the frontend uses the same origin, so local just works.

## Bandwidth budget (keeping Render free-tier alive)

Render's bandwidth is the scarce resource. Four levers keep egress low:

1. **Static assets off Render.** The 35 MB frontend is served by Cloudflare Pages (unlimited), not Render. Render serves only JSON API + the WebSocket.
2. **MediaPipe models from Google's CDN.** The ~16.6 MB of `.task` models load from `storage.googleapis.com`, not our hosts ([`webui/camera.js`](webui/camera.js)). The `vendor/models/*.task` copies remain only as an offline fallback.
3. **Immutable caching.** Both Cloudflare ([`webui/_headers`](webui/_headers)) and the backend's fallback static mount (`_SmartStatic` in [`backend/aegis/server.py`](backend/aegis/server.py)) cache `vendor/` for a year, so repeat visits cost ~0 bytes. (This replaced an old `no-store`-everything policy that re-served 35 MB per reload.)
4. **WebSocket throttle + tab-aware pause.** Stream ticks are `TICK_SECONDS=3.0` / `CLUSTER_TICK_SECONDS=2.0`; clients close the socket and stop polling on hidden tabs (`visibilitychange` in [`webui/os.js`](webui/os.js) / [`webui/tactical.js`](webui/tactical.js)), so a backgrounded dashboard no longer drains bandwidth for hours.

## Notes
- **Secrets** live only in Render's dashboard / local `backend/.env` (gitignored) — never committed.
- **Database stays under free-tier limits**: only real (camera-tagged) data is persisted; an hourly `pg_cron` job (`aegis_retain`) rolls raw telemetry >7 days into 1-minute summaries and prunes the rest. See [`supabase/migrations/`](supabase/migrations/).
