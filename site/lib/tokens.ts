/**
 * ============================================================================
 * AEGIS — CLINICAL INSTRUMENT · design tokens as TypeScript
 * ----------------------------------------------------------------------------
 * Ported verbatim from webui/os.css:25-91. Read webui/os.css:1-23 before
 * changing a colour — the three rules that identity enforces apply here too:
 *
 *   1. COLOUR IS SIGNAL, NEVER DECORATION.
 *   2. STRUCTURE IS HAIRLINES, NOT GLOW.
 *   3. NUMBERS ARE THE INTERFACE.
 *
 * WHY THIS FILE EXISTS AT ALL (rather than everything reading CSS variables):
 * the Remotion composition is bundled by Remotion's own bundler, which has no
 * access to app/globals.css. A `var(--color-high)` inside a composition would
 * resolve against a :root that does not exist in the render environment, and
 * silently paint nothing. So the composition uses inline styles fed from here.
 *
 * app/globals.css and this file must therefore agree. `npm run check:tokens`
 * asserts that they do, so the video can never drift from the website.
 * ============================================================================
 */

/** Achromatic graphite. Deliberately NO blue cast, so clinical hues sit on a
 *  neutral ground and read at full strength. */
export const surface = {
  bg: "#0a0b0c",
  bg2: "#0e0f11",
  s1: "#111214", // panel
  s2: "#16181b", // raised / inset
  s3: "#1d1f23", // hover / control
} as const;

/** Structure: hairlines do the work shadows used to. */
export const line = {
  line: "rgba(255,255,255,0.09)",
  lineSoft: "rgba(255,255,255,0.05)",
  lineBright: "rgba(255,255,255,0.18)",
} as const;

export const text = {
  ink: "#f2f3f5",
  muted: "#a0a4ac",
  dim: "#6d7178",
  faint: "#4b4f56",
} as const;

/** CLINICAL STATE — the only saturated colour in the product.
 *  If a pixel is amber, a patient is amber. Never use these decoratively. */
export const state = {
  low: "#40b880",
  med: "#e2a33c",
  dys: "#de833a",
  high: "#e5544c",
  nodata: "#4b4f56",
} as const;

/** The same signal lifted to a text-legible tint. A state's fill and its label
 *  are the same hue by construction, so nothing drifts out of sync. */
export const stateInk = {
  low: "#7fdcae",
  med: "#f0c46a",
  dys: "#f0a76a",
  high: "#ff9d97",
} as const;

/** Affordance accent: focus, selection, active nav. NOT a brand wash, and never
 *  means "urgent" — that is state.high's job. */
export const accent = {
  accent: "#7aa2f7",
  accentInk: "#a8c4ff",
} as const;

/** Instrument-tight, not pill-soft. */
export const radius = {
  sm: 4,
  md: 6,
  lg: 8,
  xl: 10,
} as const;

/** Strict 4px rhythm. */
export const space = [0, 4, 8, 12, 16, 24, 32] as const;

/** Motion: quick and mechanical; nothing bounces. */
export const motion = {
  ease: [0.2, 0.7, 0.2, 1] as const,
  easeCss: "cubic-bezier(.2,.7,.2,1)",
  fast: 110,
  base: 160,
  slow: 240,
} as const;

export const font = {
  sans: '"IBM Plex Sans", system-ui, -apple-system, "Segoe UI", Roboto, sans-serif',
  mono: '"IBM Plex Mono", ui-monospace, "SF Mono", Menlo, monospace',
} as const;

/** Flat map consumed by scripts/check-tokens.mjs. Every entry here must appear
 *  verbatim in app/globals.css's @theme block. */
export const TOKEN_HEXES: Record<string, string> = {
  bg: surface.bg,
  bg2: surface.bg2,
  s1: surface.s1,
  s2: surface.s2,
  s3: surface.s3,
  ink: text.ink,
  muted: text.muted,
  dim: text.dim,
  faint: text.faint,
  low: state.low,
  med: state.med,
  dys: state.dys,
  high: state.high,
  nodata: state.nodata,
  "low-ink": stateInk.low,
  "med-ink": stateInk.med,
  "dys-ink": stateInk.dys,
  "high-ink": stateInk.high,
  accent: accent.accent,
  "accent-ink": accent.accentInk,
};
