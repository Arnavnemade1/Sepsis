import { IBM_Plex_Sans, IBM_Plex_Mono } from "next/font/google";

/**
 * Self-hosted at build time by next/font/google — the WOFF2 is downloaded during
 * `next build` and served from /_next/static/media with an immutable cache
 * header. No runtime request to fonts.googleapis.com, and Next emits a
 * size-adjusted fallback face so CLS stays ~0. This is a strict upgrade over the
 * console's current <link> to the Google Fonts CDN.
 *
 * Caveat worth knowing: the font fetch happens at BUILD time, so a fully offline
 * build fails. Fine on Vercel; relevant if this ever moves to an air-gapped CI.
 */

/** Variable font (axes: wght, wdth). Omitting `weight` gets the real variable
 *  file — one request, the full 100-700 range. */
export const plexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  display: "swap",
  variable: "--font-plex-sans",
});

/** NOT a variable font on Google Fonts — it publishes discrete weights only.
 *  Omitting `weight` here is a hard BUILD FAILURE, not a silent fallback.
 *  Ship only the three weights the telemetry type actually uses. */
export const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  display: "swap",
  variable: "--font-plex-mono",
});

/**
 * THE HEADLINE FACE — the single swap point for the whole site.
 *
 * globals.css declares `--font-display: var(--font-display-face, var(--font-plex-sans))`,
 * so `font-display` resolves to whatever sets --font-display-face, and falls back
 * to Plex Sans when nothing does.
 *
 * To change every headline on the site: import a different face from
 * next/font/google below, give it `variable: "--font-display-face"`, and add it
 * to fontVars. To revert: delete it. No component changes either way.
 *
 * Currently deliberately Plex Sans — the identity says the site should look like
 * a sibling of the instrument, and Plex Sans at 600 with -0.03em tracking holds
 * up at display sizes.
 */
export const displayFace = plexSans;

/** Applied to <html>. */
export const fontVars = [plexSans.variable, plexMono.variable].join(" ");
