"use client";

import { useEffect, useRef, useState } from "react";

import { cx } from "@/components/ui";
import { BAND_LABEL, BAND_SEV, KIND_LABEL, SEV_INK, SEV_LABEL, SEV_RGB, type RoomKind, type Severity } from "@/lib/rooms";

/** The width of an element, for SVG drawn at true pixel size (no stretched strokes). */
export function useWidth<T extends HTMLElement>(fallback = 320) {
  const ref = useRef<T>(null);
  const [w, setW] = useState(fallback);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver(([e]) => {
      const next = Math.round(e.contentRect.width);
      if (next > 0) setW(next);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

export function SevDot({ sev, size = 7, className }: { sev: Severity | null | undefined; size?: number; className?: string }) {
  if (!sev) return <span className={cx("inline-block shrink-0 rounded-full border border-line-bright", className)} style={{ width: size, height: size }} />;
  return <span className={cx("inline-block shrink-0 rounded-full", className)} style={{ width: size, height: size, background: `rgb(${SEV_RGB[sev]})` }} />;
}

export function SevTag({ sev, children }: { sev: Severity; children?: React.ReactNode }) {
  if (sev === "info")
    return <span className="inline-flex h-5 items-center rounded-[4px] border border-line-bright bg-s2 px-1.5 text-[11px] font-medium text-muted">{children ?? SEV_LABEL[sev]}</span>;
  return (
    <span
      className="inline-flex h-5 items-center rounded-[4px] border px-1.5 text-[11px] font-medium"
      style={{ color: SEV_INK[sev], borderColor: `rgb(${SEV_RGB[sev]} / .4)`, background: `rgb(${SEV_RGB[sev]} / .12)` }}
    >
      {children ?? SEV_LABEL[sev]}
    </span>
  );
}

/** A NEWS2 aggregate, coloured by its clinical-response band. */
export function News2Badge({ score, band, size = "md", title }: { score: number | null | undefined; band: string | null | undefined; size?: "sm" | "md" | "lg"; title?: string }) {
  const dims = size === "lg" ? "h-9 min-w-9 text-[20px] px-2" : size === "sm" ? "h-5 min-w-5 text-[11px] px-1" : "h-6 min-w-6 text-[12.5px] px-1.5";
  if (score == null) return <span className={cx("grid place-items-center rounded-[5px] border border-line font-mono text-faint", dims)}>—</span>;
  const sev = BAND_SEV[band ?? "low"] ?? "info";
  const neutral = sev === "info";
  return (
    <span
      title={title ?? `NEWS2 ${score}: ${BAND_LABEL[band ?? "low"] ?? band} band`}
      className={cx("num grid place-items-center rounded-[5px] border font-mono font-semibold", dims, neutral && "border-line-bright bg-s2 text-ink")}
      style={neutral ? undefined : { color: SEV_INK[sev], borderColor: `rgb(${SEV_RGB[sev]} / .5)`, background: `rgb(${SEV_RGB[sev]} / .14)` }}
    >
      {score}
    </span>
  );
}

export function KindTag({ kind }: { kind: RoomKind | null | undefined }) {
  if (!kind) return null;
  return (
    <span
      className={cx(
        "inline-flex h-5 items-center gap-1.5 whitespace-nowrap rounded-[4px] border px-1.5 text-[11px] font-medium",
        kind === "synthetic" ? "border-line-bright bg-s2 text-muted" : "border-accent/40 bg-accent-soft text-accent-ink",
      )}
      title={kind === "synthetic" ? "Generated for the demo hospital: not a real person" : kind === "replay" ? "PhysioNet/CinC 2019, de-identified" : "Camera, wearable or simulator"}
    >
      <span className={cx("h-1.5 w-1.5 rounded-full", kind === "synthetic" ? "bg-dim" : "bg-accent-ink")} />
      {KIND_LABEL[kind]}
    </span>
  );
}

/** Allergies as the record states them. Shown wherever an order could be written. */
export function AllergyChips({ allergies, compact = false }: { allergies: { substance: string; reaction?: string; severity?: string; kind?: string }[] | string[] | undefined; compact?: boolean }) {
  const list = (allergies ?? []).map((a) => (typeof a === "string" ? { substance: a } : a)) as { substance: string; reaction?: string; severity?: string; kind?: string }[];
  if (!list.length) return <span className="inline-flex h-5 items-center rounded-[4px] border border-line-bright bg-s2 px-1.5 text-[11px] text-muted">No known allergies</span>;
  return (
    <>
      {list.map((a) => {
        const severe = a.severity === "severe";
        const intolerance = a.kind === "intolerance";
        return (
          <span
            key={a.substance}
            className={cx("inline-flex h-5 items-center gap-1 whitespace-nowrap rounded-[4px] border px-1.5 text-[11px] font-medium", intolerance && "border-line-bright bg-s2 text-muted")}
            style={intolerance ? undefined : { color: SEV_INK[severe ? "critical" : "urgent"], borderColor: `rgb(${SEV_RGB[severe ? "critical" : "urgent"]} / .45)`, background: `rgb(${SEV_RGB[severe ? "critical" : "urgent"]} / .1)` }}
            title={intolerance ? "Intolerance, not an allergy" : "Allergy on the record"}
          >
            {intolerance ? "Intolerance" : "Allergy"}: {a.substance}
            {!compact && a.reaction && a.reaction !== "unknown" && <span className="font-normal opacity-80">· {a.reaction}</span>}
          </span>
        );
      })}
    </>
  );
}

export function CodeChip({ code }: { code: string | null | undefined }) {
  if (!code) return <span className="inline-flex h-5 items-center rounded-[4px] border border-dashed border-line-bright px-1.5 text-[11px] text-dim">Code status not recorded</span>;
  const limited = code !== "Full code";
  return (
    <span
      className={cx("inline-flex h-5 items-center whitespace-nowrap rounded-[4px] border px-1.5 text-[11px] font-semibold", !limited && "border-line-bright bg-s2 font-medium text-muted")}
      style={limited ? { color: SEV_INK.watch, borderColor: `rgb(${SEV_RGB.watch} / .5)`, background: `rgb(${SEV_RGB.watch} / .12)` } : undefined}
    >
      {code}
    </span>
  );
}

/** A small line chart at true pixel width with an optional shaded band (a normal range). */
export function Spark({ data, w, h, color = "#a8afb8", band, domain, dot }: {
  data: (number | null | undefined)[];
  w: number;
  h: number;
  color?: string;
  band?: [number, number] | null;
  domain?: [number, number];
  dot?: string;
}) {
  const vals = data.filter((v): v is number => v != null && !Number.isNaN(v));
  if (vals.length < 2) return <svg width={w} height={h} aria-hidden />;
  let lo = domain?.[0] ?? Math.min(...vals);
  let hi = domain?.[1] ?? Math.max(...vals);
  if (band && !domain) {
    lo = Math.min(lo, band[0]);
    hi = Math.max(hi, band[1]);
  }
  const pad = (hi - lo || 1) * 0.12;
  lo -= pad;
  hi += pad;
  const y = (v: number) => h - 1 - ((v - lo) / (hi - lo)) * (h - 2);
  const step = w / Math.max(1, data.length - 1);
  let d = "";
  let pen = false;
  let last: [number, number] | null = null;
  data.forEach((v, i) => {
    if (v == null || Number.isNaN(v)) {
      pen = false;
      return;
    }
    const p: [number, number] = [i * step, y(v)];
    d += `${pen ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`;
    pen = true;
    last = p;
  });
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="block overflow-visible" aria-hidden>
      {band && <rect x={0} y={y(band[1])} width={w} height={Math.max(1, y(band[0]) - y(band[1]))} fill="rgb(255 255 255 / .045)" />}
      <path d={d} fill="none" stroke={color} strokeWidth={1.4} strokeLinejoin="round" strokeLinecap="round" />
      {last && dot && <circle cx={(last as [number, number])[0]} cy={(last as [number, number])[1]} r={2.6} fill={dot} stroke="var(--color-s1)" strokeWidth={1.2} />}
    </svg>
  );
}
