# `site/` — the public website

The marketing/research face of AegisOS. Next.js 16 (App Router) + Tailwind v4 +
Motion + Remotion. Entirely additive: it does not touch `webui/` (the live OS
console) or `backend/`.

```bash
npm install
npm run dev          # http://localhost:3000
npm run build        # all four routes prerender static
npm run typecheck
```

## The four things that are not obvious

### 1. `sync:results` is NOT a `prebuild` hook — on purpose

`data/results.json` is **committed**, and it is the build's only data input.

Vercel's Root Directory sandbox cannot read files outside its root and rejects
`..`, so `../artifacts/*.json` does not exist at build time. Wiring the sync
script to `prebuild` would therefore break the deploy. Instead:

```bash
npm run sync:results   # local only; re-reads ../artifacts and rewrites data/results.json
```

It exits 0 with a warning when `../artifacts` is unreachable, so it is harmless
if it ever does run in CI. Re-run it after any evaluation changes, and commit the
result — otherwise the site quietly reports stale numbers.

### 2. Design tokens live in two places, and a script keeps them honest

`app/globals.css` (`@theme`) feeds the website. `lib/tokens.ts` feeds the Remotion
composition, because Remotion's bundler cannot see `globals.css` and a
`var(--color-*)` inside a composition resolves against a `:root` that does not
exist during rendering.

```bash
npm run check:tokens   # fails if the two disagree
```

The stock Tailwind palette is deleted (`--color-*: initial`), so `bg-blue-500`
does not compile. That is deliberate — it is the build-level enforcement of the
"colour is signal, never decoration" rule from `webui/os.css:1-23`. Read that
header before changing a colour.

### 3. Remotion is split across two entry points

| Path | Used by | Notes |
|---|---|---|
| `remotion/Explainer.tsx` | both | plain React; **inline styles only**, no Tailwind |
| `remotion/index.ts` | Remotion CLI | `registerRoot` — the Next app never imports it |
| `components/explainer/` | the website | `<Player>` behind `"use client"` + `dynamic(ssr:false)` |

`@remotion/bundler` does not work inside Next.js, so MP4 rendering can never be
part of `next build`:

```bash
npm run video:studio   # scrub the composition; the fastest way to check the loop seam
npm run video:render   # → public/media/explainer.mp4 (downloads headless Chrome on first run)
```

Remotion is **not** MIT. Its free licence covers individuals, non-profits, and
for-profits with up to 3 employees.

### 4. Deployment

Vercel, **Root Directory = `site`**, everything else default. Enable
"skip unaffected projects" or every backend commit rebuilds this.

- **Do not add a `package.json` at the repo root.** Cloudflare Pages builds
  `webui/` with an empty build command; a root manifest would make its build
  system start guessing at a framework.
- **Do not proxy the console through `rewrites()`.** It would push 38 MB of
  three.js and organ GLBs through this site's bandwidth, bypass
  `webui/_headers`' immutable caching, and break `webui/config.js`'s same-origin
  backend resolution. `NEXT_PUBLIC_CONSOLE_URL` is a plain outbound link.
- The code is kept static-export-compatible (no Server Actions, no Route
  Handlers, no `cookies()`/`headers()`), so `output: "export"` remains a
  one-line escape hatch to Cloudflare Pages.

`render.yaml` carries a `buildFilter` so `site/**` commits no longer redeploy the
Python backend.

## Environment

| Variable | Default | Purpose |
|---|---|---|
| `NEXT_PUBLIC_CONSOLE_URL` | the Render service | where "Launch console" points |
| `NEXT_PUBLIC_BACKEND_URL` | the Render service | reserved for the live-health probe |

## Hero video

`public/media/hero.mp4` — Pexels, free for commercial use, no attribution
required ([source](https://www.pexels.com/video/sea-of-clouds-on-mountain-peaks-5721853/)).
1280×720, 22 s, 2.5 MB.

It is graded in CSS to ~32% saturation. That is not a style choice: `--low` is a
saturated green, and ungraded nature footage would put a clinical state colour
behind the whole page. It is also **conditionally mounted** — on a phone, on
Save-Data, or on a 2G link the browser downloads zero video bytes.
