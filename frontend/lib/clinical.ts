import type { Tier, WardTier } from "./types";

/**
 * Clinical state → colour. These are the ONLY places a state hue is chosen; they
 * mirror --color-low/med/dys/high in globals.css and the hand-duplicated literals
 * in engine/twin3d.js. A healthy reading is neutral, not green: colour on this
 * product means something needs attention.
 */
export type Tone = "low" | "med" | "dys" | "high" | "nodata" | "neutral";

export const TONE_TEXT: Record<Tone, string> = {
  low: "text-low-ink",
  med: "text-med-ink",
  dys: "text-dys-ink",
  high: "text-high-ink",
  nodata: "text-dim",
  neutral: "text-ink",
};
export const TONE_BG: Record<Tone, string> = {
  low: "bg-low",
  med: "bg-med",
  dys: "bg-dys",
  high: "bg-high",
  nodata: "bg-nodata",
  neutral: "bg-dim",
};
export const TONE_RGB: Record<Tone, string> = {
  low: "47 182 124",
  med: "238 177 42",
  dys: "245 118 46",
  high: "229 72 77",
  nodata: "86 93 104",
  neutral: "162 169 179",
};

export function tierTone(t: Tier | WardTier | string | null | undefined): Tone {
  switch (t) {
    case "HIGH":
    case "CRITICAL":
      return "high";
    case "MEDIUM":
    case "WATCH":
      return "med";
    case "LOW":
    case "STABLE":
      return "low";
    default:
      return "nodata";
  }
}

export function severityTone(s: number | null | undefined, monitored = true): Tone {
  if (!monitored || s == null) return "nodata";
  if (s >= 0.66) return "high";
  if (s >= 0.4) return "dys";
  if (s >= 0.15) return "med";
  return "neutral";
}

export function pct(p: number | null | undefined, digits = 1): string {
  if (p == null || Number.isNaN(p)) return "—";
  return `${(p * 100).toFixed(digits)}%`;
}

export function num(v: number | null | undefined, digits = 0): string {
  if (v == null || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function ago(tsSeconds: number | null | undefined): string {
  if (!tsSeconds) return "—";
  const s = Math.max(0, Date.now() / 1000 - tsSeconds);
  if (s < 5) return "just now";
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  return `${Math.round(s / 3600)}h ago`;
}

export function title(s: string): string {
  const t = s.replace(/[_-]+/g, " ").trim();
  return t.charAt(0).toUpperCase() + t.slice(1).toLowerCase();
}

/** Vital-sign display metadata: label, unit, precision, and the normal band. */
export const VITALS: Record<string, { label: string; unit: string; d: number; lo?: number; hi?: number }> = {
  heart_rate: { label: "Heart rate", unit: "bpm", d: 0, lo: 50, hi: 100 },
  respiratory_rate: { label: "Resp. rate", unit: "/min", d: 0, lo: 10, hi: 22 },
  spo2: { label: "SpO₂", unit: "%", d: 0, lo: 94 },
  map: { label: "MAP", unit: "mmHg", d: 0, lo: 65 },
  systolic_bp: { label: "Systolic", unit: "mmHg", d: 0, lo: 100 },
  skin_temp: { label: "Temp", unit: "°C", d: 1, lo: 36, hi: 38 },
  hrv: { label: "HRV", unit: "ms", d: 0, lo: 20 },
  sweat_lactate: { label: "Lactate", unit: "mmol/L", d: 1, hi: 2 },
};

export function vitalTone(key: string, v: number | null | undefined): Tone {
  const m = VITALS[key];
  if (!m || v == null) return "nodata";
  if ((m.lo != null && v < m.lo) || (m.hi != null && v > m.hi)) return "med";
  return "neutral";
}
