"use client";

import { useMemo, useState } from "react";

import { TONE_RGB } from "@/lib/clinical";

export type RiskSeries = {
  /** x positions (ICU hour, or seconds for the live session) */
  x: number[];
  risk: (number | null)[];
  population?: (number | null)[];
  /** Projection after the last x. It continues the POPULATION line: the
   *  forecaster scores projected vitals on the population fusion, so drawing it
   *  off the effective-risk line would invent a jump at "now". */
  forecast?: { x: number; mean: number; lo: number; hi: number }[];
  alarm: number;
  medium?: number;
  xLabel: string;
};

/**
 * Risk over time against the line that actually alarms. The y-axis is always
 * 0–1: rescaling to the data would make a 3% wobble look like a crisis.
 */
export function RiskChart({ s, height = 220 }: { s: RiskSeries; height?: number }) {
  const [hover, setHover] = useState<number | null>(null);
  const W = 720;
  const H = height;
  const pad = { l: 34, r: 12, t: 12, b: 24 };
  const xs = [...s.x, ...(s.forecast ?? []).map((f) => f.x)];
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs, x0 + 1);
  const X = (v: number) => pad.l + ((v - x0) / (x1 - x0)) * (W - pad.l - pad.r);
  const Y = (p: number) => pad.t + (1 - p) * (H - pad.t - pad.b);

  const line = (vals: (number | null)[]) => {
    let d = "";
    let pen = false;
    vals.forEach((v, i) => {
      if (v == null) {
        pen = false;
        return;
      }
      d += `${pen ? "L" : "M"}${X(s.x[i]).toFixed(1)},${Y(v).toFixed(1)}`;
      pen = true;
    });
    return d;
  };

  const band = useMemo(() => {
    const f = s.forecast ?? [];
    if (f.length < 2) return "";
    const top = f.map((p) => `${X(p.x).toFixed(1)},${Y(p.hi).toFixed(1)}`).join(" L");
    const bot = [...f].reverse().map((p) => `${X(p.x).toFixed(1)},${Y(p.lo).toFixed(1)}`).join(" L");
    return `M${top} L${bot} Z`;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [s]);

  const last = s.risk.length - 1;
  const hi = hover ?? last;
  const lvl = (p: number | null | undefined) =>
    p == null ? "nodata" : p >= s.alarm ? "high" : s.medium != null && p >= s.medium ? "med" : "neutral";

  return (
    <div className="relative">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        className="h-auto w-full select-none"
        onMouseLeave={() => setHover(null)}
        onMouseMove={(e) => {
          const r = (e.currentTarget as SVGSVGElement).getBoundingClientRect();
          const vx = ((e.clientX - r.left) / r.width) * W;
          let best = 0;
          s.x.forEach((xv, i) => {
            if (Math.abs(X(xv) - vx) < Math.abs(X(s.x[best]) - vx)) best = i;
          });
          setHover(best);
        }}
      >
        {[0, 0.25, 0.5, 0.75, 1].map((g) => (
          <g key={g}>
            <line x1={pad.l} x2={W - pad.r} y1={Y(g)} y2={Y(g)} stroke="rgb(255 255 255 / .05)" />
            <text x={pad.l - 6} y={Y(g) + 3.5} textAnchor="end" fontSize="10" fill="#6e7482">
              {Math.round(g * 100)}
            </text>
          </g>
        ))}
        <line x1={pad.l} x2={W - pad.r} y1={Y(s.alarm)} y2={Y(s.alarm)} stroke={`rgb(${TONE_RGB.high} / .55)`} strokeDasharray="4 4" />
        <text x={W - pad.r} y={Y(s.alarm) - 5} textAnchor="end" fontSize="10" fill="#ff9d97">
          alarm {Math.round(s.alarm * 100)}%
        </text>
        {band && <path d={band} fill="rgb(164 169 182 / .08)" />}
        {s.forecast && s.forecast.length > 1 && (
          <path
            d={`M${s.forecast.map((f) => `${X(f.x).toFixed(1)},${Y(f.mean).toFixed(1)}`).join(" L")}`}
            fill="none"
            stroke="rgb(164 169 182 / .6)"
            strokeDasharray="3 3"
            strokeWidth={1.4}
          />
        )}
        {s.population && <path d={line(s.population)} fill="none" stroke="rgb(164 169 182 / .35)" strokeWidth={1.2} />}
        <path d={line(s.risk)} fill="none" stroke="#f4f5f8" strokeWidth={1.8} strokeLinejoin="round" />
        {s.risk[hi] != null && (
          <g>
            <line x1={X(s.x[hi])} x2={X(s.x[hi])} y1={pad.t} y2={H - pad.b} stroke="rgb(255 255 255 / .12)" />
            <circle cx={X(s.x[hi])} cy={Y(s.risk[hi]!)} r={3.5} fill={`rgb(${TONE_RGB[lvl(s.risk[hi])]})`} />
          </g>
        )}
        <text x={pad.l} y={H - 6} fontSize="10" fill="#6e7482">
          {s.xLabel} {Math.round(x0)}
        </text>
        <text x={W - pad.r} y={H - 6} textAnchor="end" fontSize="10" fill="#6e7482">
          {Math.round(x1)}
        </text>
      </svg>
      <div className="num pointer-events-none absolute left-10 top-1 text-[11.5px] text-muted">
        {s.xLabel} {Math.round(s.x[hi] ?? 0)} ·{" "}
        <span className="text-ink">{s.risk[hi] != null ? `${(s.risk[hi]! * 100).toFixed(1)}%` : "—"}</span>
        {s.population?.[hi] != null && <span className="text-dim"> · population {(s.population[hi]! * 100).toFixed(1)}%</span>}
      </div>
    </div>
  );
}
