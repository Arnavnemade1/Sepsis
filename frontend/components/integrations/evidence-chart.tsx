"use client";

import { useMemo } from "react";

import type { MappingDetail } from "@/lib/interop";

/**
 * The values a feed sent, read through a conversion, against the concept's reference
 * distribution: the band is where the reference's middle 95% and middle 50% sit, the
 * tick is its median, each dot one value. A unit error is a cloud of dots far from the
 * band; a mixture is a second cloud. Log axis for skewed analytes.
 */
export function EvidenceChart({ m, factor, offset, height = 92 }: { m: MappingDetail; factor: number; offset: number; height?: number }) {
  const ref = m.reference;
  const info = m.concept_info;
  const W = 560;
  const H = height;
  const pad = { l: 8, r: 8, t: 14, b: 22 };
  const data = useMemo(() => {
    if (!ref || !info) return null;
    const q = (p: string) => ref.quantiles[p] ?? ref.quantiles[String(Number(p))];
    const vals = m.samples.map((s) => ({ v: s.raw * factor + offset, delivered: s.delivered }));
    const log = info.scale === "log";
    const pos = (x: number) => (log ? Math.log(Math.max(x, 1e-9)) : x);
    const refLo = q("0.001") ?? ref.tail[0];
    const refHi = q("0.999") ?? ref.tail[1];
    const finite = vals.map((d) => d.v).filter((v) => Number.isFinite(v) && (!log || v > 0));
    const lo = Math.min(pos(refLo), ...finite.map(pos));
    const hi = Math.max(pos(refHi), ...finite.map(pos));
    const span = hi - lo || 1;
    const x = (v: number) => pad.l + ((pos(v) - lo) / span) * (W - pad.l - pad.r);
    return { q, vals, log, x, lo, hi, refLo, refHi, win: info.window };
  }, [m.samples, ref, info, factor, offset, W, pad.l, pad.r]);
  if (!data || !ref || !info) return <div className="text-xs text-dim">No reference distribution for this concept.</div>;
  const { q, vals, x } = data;
  const y0 = pad.t;
  const y1 = H - pad.b;
  const mid = (y0 + y1) / 2;
  const ticks = [q("0.025"), q("0.5"), q("0.975")].filter((v): v is number => v != null);
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="h-auto w-full" role="img" aria-label={`Values against the ${info.label} reference`}>
      {/* the physiological window */}
      <rect x={x(Math.max(data.win[0], data.log ? 1e-9 : data.win[0]))} y={y0 - 6} width={Math.max(1, x(data.win[1]) - x(Math.max(data.win[0], data.log ? 1e-9 : data.win[0])))} height={y1 - y0 + 12} fill="#2f343b" opacity={0.35} />
      {q("0.025") != null && q("0.975") != null && (
        <rect x={x(q("0.025")!)} y={y0} width={Math.max(1, x(q("0.975")!) - x(q("0.025")!))} height={y1 - y0} fill="#4a5059" opacity={0.55} rx={2} />
      )}
      {q("0.25") != null && q("0.75") != null && (
        <rect x={x(q("0.25")!)} y={y0 + 6} width={Math.max(1, x(q("0.75")!) - x(q("0.25")!))} height={y1 - y0 - 12} fill="#6b727c" opacity={0.7} rx={2} />
      )}
      {q("0.5") != null && <line x1={x(q("0.5")!)} x2={x(q("0.5")!)} y1={y0 - 3} y2={y1 + 3} stroke="#e8eaed" strokeWidth={1.4} />}
      {vals.map((d, i) => {
        if (!Number.isFinite(d.v) || (data.log && d.v <= 0)) return null;
        const jitter = ((i * 7919) % 13) - 6;
        return <circle key={i} cx={x(d.v)} cy={mid + jitter * 1.6} r={2.1} fill={d.delivered ? "#6eaaff" : "#e5484d"} opacity={0.75} />;
      })}
      {ticks.map((t, i) => (
        <text key={i} x={x(t)} y={H - 6} textAnchor="middle" className="fill-[#868d97] text-2xs" style={{ fontFamily: "var(--font-mono, monospace)" }}>
          {t >= 100 ? t.toFixed(0) : t >= 1 ? t.toFixed(1) : t.toPrecision(2)}
        </text>
      ))}
      <text x={pad.l} y={10} className="fill-[#868d97] text-2xs">
        reference 95% / 50% band · median · window shaded{data.log ? " · log axis" : ""}
      </text>
    </svg>
  );
}
