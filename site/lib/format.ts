/**
 * Number formatters, matching backend/ml/scripts/build_report.py:138-142 so the
 * site and the generated report render the same value the same way.
 *
 * Rule 3 — numbers are the interface. An em dash for a missing value, never
 * "N/A", never 0, and never a silently rounded-away difference.
 */

const bad = (v: number | null | undefined): v is null | undefined =>
  v === null || v === undefined || Number.isNaN(v);

/** Axis tick label — build_report.py:138. */
export function fmtTick(v: number): string {
  if (Math.abs(v) >= 1000) return `${(v / 1000).toFixed(0)}k`;
  if (v === 0) return "0";
  if (Math.abs(v) < 1) return String(+v.toFixed(2));
  return String(+v.toFixed(v % 1 ? 1 : 0));
}

export const f2 = (v?: number | null): string => (bad(v) ? "—" : v.toFixed(2));
export const f3 = (v?: number | null): string => (bad(v) ? "—" : v.toFixed(3));
export const f4 = (v?: number | null): string => (bad(v) ? "—" : v.toFixed(4));

/** Percentage with no decimals — for rates a clinician reads at a glance. */
export const pct = (v?: number | null, dp = 0): string =>
  bad(v) ? "—" : `${(v * 100).toFixed(dp)}%`;

/** A signed delta. The sign is the point, so it is always shown. */
export const signed = (v?: number | null, dp = 3): string =>
  bad(v) ? "—" : `${v >= 0 ? "+" : "−"}${Math.abs(v).toFixed(dp)}`;

/** "0.686–0.739" */
export const ci = (pair?: readonly number[] | null, dp = 3): string =>
  !pair || pair.length < 2 ? "—" : `${pair[0].toFixed(dp)}–${pair[1].toFixed(dp)}`;

/** A CONFIDENCE INTERVAL ON A DELTA, e.g. "[−0.033, −0.009]".
 *
 *  `ci` renders "0.686–0.739", which is right for a metric that cannot go
 *  negative and wrong for a difference that can: an en-dash range whose bounds
 *  carry their own minus signs reads as "-0.0333–-0.0092". Brackets and a comma
 *  remove the ambiguity, and the true minus (U+2212) matches `signed` so the two
 *  never appear side by side in different glyphs. */
export const ciSigned = (pair?: readonly number[] | null, dp = 4): string =>
  !pair || pair.length < 2 ? "—" : `[${signed(pair[0], dp)}, ${signed(pair[1], dp)}]`;

export const hours = (v?: number | null): string => (bad(v) ? "—" : `${v.toFixed(0)} h`);

/** 232797 → "232,797" */
export const count = (v?: number | null): string =>
  bad(v) ? "—" : v.toLocaleString("en-US");

/** p-values: below the resolution of the run, say so rather than print 0.000. */
export const pval = (v?: number | null): string =>
  bad(v) ? "—" : v < 0.001 ? "<0.001" : v.toFixed(3);
