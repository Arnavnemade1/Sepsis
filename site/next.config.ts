import path from "node:path";

import type { NextConfig } from "next";

/**
 * Deliberately minimal.
 *
 * NOT needed — and each omission is load-bearing, so don't "helpfully" add them:
 *   • transpilePackages   — @remotion/player ships proper dual ESM/CJS exports.
 *   • serverExternalPackages — @remotion/renderer is a devDependency and never
 *     runs on the host; it exists only for the local `video:render` script.
 *   • a vercel.json        — headers live here instead, so they also apply under
 *     `next start` and survive a move off Vercel.
 *
 * The whole app is kept static-export-compatible (no Server Actions, no Route
 * Handlers, no cookies()/headers(), all data imported at build time), so
 * `output: "export"` remains a one-line escape hatch to Cloudflare Pages.
 */
const config: NextConfig = {
  reactStrictMode: true,

  // Pin the build root to site/. Without this Next walks up looking for a
  // workspace root, finds an unrelated package-lock.json in the user's home
  // directory, and warns. Pinning also keeps file tracing correct when Vercel
  // builds with Root Directory = site.
  turbopack: { root: path.resolve(__dirname) },

  async headers() {
    return [
      {
        // Hero video, poster and the rendered explainer are content-addressed by
        // deploy — they never change under a fixed URL, so cache them hard. This
        // mirrors webui/_headers' treatment of vendor/.
        source: "/media/:path*",
        headers: [
          { key: "Cache-Control", value: "public, max-age=31536000, immutable" },
        ],
      },
      {
        source: "/:path*",
        headers: [
          { key: "X-Content-Type-Options", value: "nosniff" },
          { key: "Referrer-Policy", value: "strict-origin-when-cross-origin" },
          { key: "Permissions-Policy", value: "camera=(), microphone=(), geolocation=()" },
        ],
      },
    ];
  },
};

export default config;
