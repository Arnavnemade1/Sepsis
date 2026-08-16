/**
 * Chart scales and tick math.
 *
 * Ported from backend/ml/scripts/build_report.py:105-161 (the `sx`/`sy`/`tk`/
 * `fmt` helpers and the forest-plot geometry) so the two renderings of the same
 * evaluation — the generated report and this site — stay visually consistent.
 *
 * No d3, no chart library. Every axis on this site is a probability or a rate in
 * [0,1]; a linear scale is three lines, and hand-rolled SVG is what lets Motion
 * animate a curve's pathLength so it draws itself like a plotter.
 */

export type Pt = readonly [number, number];
export type Range = readonly [number, number];

export interface Margins {
  l: number;
  r: number;
  t: number;
  b: number;
}

/** build_report.py's default plot margins. */
export const M: Margins = { l: 44, r: 14, t: 10, b: 34 };

const clamp = (v: number, [lo, hi]: Range) => Math.max(lo, Math.min(hi, v));

/** Linear scale with domain clamping, matching build_report.py:108-110. */
export function scaleX(domain: Range, width: number, m: Margins = M) {
  return (x: number) =>
    m.l + ((clamp(x, domain) - domain[0]) / (domain[1] - domain[0])) * (width - m.l - m.r);
}

export function scaleY(domain: Range, height: number, m: Margins = M) {
  return (y: number) =>
    height -
    m.b -
    ((clamp(y, domain) - domain[0]) / (domain[1] - domain[0])) * (height - m.t - m.b);
}

/** Evenly spaced ticks across a range — build_report.py:137. */
export function ticks(range: Range, n = 5): number[] {
  const step = (range[1] - range[0]) / n;
  return Array.from({ length: n + 1 }, (_, i) => range[0] + step * i);
}

/** Build an SVG path `d` from points already in screen space. */
export function path(pts: readonly Pt[]): string {
  return pts.map(([x, y], i) => `${i ? "L" : "M"}${x.toFixed(2)} ${y.toFixed(2)}`).join(" ");
}

/** Zip two parallel arrays (the shape every curve arrives in) into points. */
export function zip(xs: readonly number[], ys: readonly number[]): Pt[] {
  const n = Math.min(xs.length, ys.length);
  return Array.from({ length: n }, (_, i) => [xs[i], ys[i]] as const);
}

/** Map data points through scales into screen space, then to a path. */
export function toPath(
  pts: readonly Pt[],
  sx: (x: number) => number,
  sy: (y: number) => number,
): string {
  return path(pts.map(([x, y]) => [sx(x), sy(y)] as const));
}
