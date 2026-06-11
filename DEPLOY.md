# Deploying AegisOS / HyperSepsisTwin-Ω

The app is **two pieces** with different hosting needs:

| Piece | What it is | Where it runs |
|-------|------------|---------------|
| **Frontend** (`webui/`) | Static OS shell + the fully client-side optical biosensor (camera rPPG, detection, 3D twin) | **Vercel** (static) |
| **Backend** (`backend/`) | FastAPI server: `/aegis/stream` **WebSocket**, simulator/risk brain, real-data ingest, Supabase persistence | **Render** (persistent web service) |

> **Why not all on Vercel?** Vercel is serverless — it cannot hold a WebSocket open or run the background drive loop. The live `*.vercel.app` site is the frontend only; without a backend the desktop panels stay empty and the console shows `wss://…/aegis/stream 404`. The backend must run somewhere persistent.

---

## 1 · Deploy the backend to Render

1. Go to **render.com → New → Blueprint** and connect this GitHub repo. Render reads [`render.yaml`](render.yaml) and creates the `sepsis-backend` web service.
2. In the service's **Environment**, set the secrets (these are NOT in the repo):
   - `SUPABASE_URL` — `https://oyukwcuwiyiuslmmbvou.supabase.co`
   - `SUPABASE_KEY` — your Supabase **service-role** key (from `backend/.env`)
   - `OPENROUTER_API_KEY` — optional (LLM narrative)
3. Deploy. You get a URL like **`https://sepsis-backend.onrender.com`**.
4. Verify: open `https://sepsis-backend.onrender.com/aegis/health` → `{"status":"ok", ...}`.

> Render's **free** tier supports WebSockets but **sleeps after ~15 min idle**; the first request after idle cold-starts in ~30–60 s. Upgrade to a paid instance to keep it always-on for a live demo/judging.

> The backend also serves the frontend itself, so `https://sepsis-backend.onrender.com` is a complete working app on its own — Vercel is optional.

## 2 · Point the Vercel frontend at the backend

Edit one line in [`webui/config.js`](webui/config.js):

```js
window.AEGIS_BACKEND = "https://sepsis-backend.onrender.com";
```

Commit + push → Vercel redeploys. The frontend now opens its WebSocket and REST calls against Render.

> **Quick test without redeploying:** in the browser console on the Vercel site run
> `localStorage.setItem('aegis_backend','https://sepsis-backend.onrender.com'); location.reload()`

## 3 · Verify the live site

Open the Vercel URL and confirm:
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

## Notes
- **Secrets** live only in Render's dashboard / local `backend/.env` (gitignored) — never committed.
- **Database stays under free-tier limits**: only real (camera-tagged) data is persisted; an hourly `pg_cron` job (`aegis_retain`) rolls raw telemetry >7 days into 1-minute summaries and prunes the rest. See [`supabase/migrations/`](supabase/migrations/).
