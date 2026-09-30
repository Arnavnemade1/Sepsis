import path from "node:path";

import type { NextConfig } from "next";

/**
 * The AegisOS clinical frontend is a STATIC EXPORT. It talks to the FastAPI
 * backend (Render) over REST + WebSocket from the browser, so it needs no server
 * of its own and deploys to Cloudflare Pages, whose bandwidth is uncapped — which
 * matters, because the anatomical twin ships ~35 MB of organ meshes.
 *
 * That is why every route that needs an id takes it as a query string
 * (`/bed?id=bed_3`) rather than a dynamic segment: a static export cannot
 * pre-render an unknown set of beds.
 *
 * Response headers (caching, security) live in public/_headers, which Cloudflare
 * reads; `headers()` here would be silently ignored by `output: "export"`.
 */
const config: NextConfig = {
  output: "export",
  reactStrictMode: true,
  trailingSlash: true,
  images: { unoptimized: true },
  // Pin the build root: Next otherwise walks up, finds site/'s lockfile, and warns.
  turbopack: { root: path.resolve(__dirname) },
};

export default config;
