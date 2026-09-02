/**
 * Every outward-facing URL and constant in one place.
 */

export const site = {
  name: "AegisOS",
  /** Used in <title> templates and the OG card. */
  tagline: "Continuous sepsis detection you can verify.",
  description:
    "AegisOS turns ordinary sensors into a real-time digital twin that detects sepsis, explains its reasoning, and synthesizes formally-verified treatment protocols. Validated across hospitals on PhysioNet CinC-2019, with the claims we could not defend retracted in public.",
  /** Research codename, used once on /about. */
  codename: "HyperSepsisTwin-Ω",
  repo: "https://github.com/arnavnemade/Sepsis",
} as const;

/**
 * The live OS console (webui/). A plain outbound link — never a Next rewrite:
 * proxying it would push 44 MB of three.js and organ GLBs through this site's
 * bandwidth, bypass webui/_headers' one-year immutable caching so repeat
 * visitors re-download everything, and break webui/config.js's same-origin
 * backend resolution.
 *
 * Defaults to the Render service, which serves the full shell itself. Point
 * NEXT_PUBLIC_CONSOLE_URL at a Cloudflare Pages domain to move it with no code
 * change.
 */
export const consoleUrl =
  process.env.NEXT_PUBLIC_CONSOLE_URL ?? "https://sepsis-backend-omd1.onrender.com";

/**
 * The FastAPI backend. Only used for the optional, non-blocking live health
 * probe — every number this site renders comes from build-time data.
 *
 * Free tier: sleeps after ~15 min idle, cold-starts in 30-60 s. Anything that
 * touches this must tolerate that without blocking paint.
 */
export const backendUrl =
  process.env.NEXT_PUBLIC_BACKEND_URL ?? "https://sepsis-backend-omd1.onrender.com";

export const nav = [
  { href: "/technology", label: "Technology" },
  { href: "/research", label: "Research" },
  { href: "/about", label: "About" },
] as const;

/** Shown wherever a claim could be mistaken for a clinical one. */
export const disclaimer =
  "Research and educational use only. Not a medical device; not for clinical use.";
