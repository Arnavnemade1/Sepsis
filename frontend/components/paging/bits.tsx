"use client";

import { cx } from "@/components/ui";
import { initials, PRIORITY_LABEL, SOURCE_LABEL, STATUS_LABEL, type Source, type StaffStatus } from "@/lib/paging";

export const P_RGB: Record<number, string> = { 1: "229 72 77", 2: "240 106 29", 3: "226 163 54", 4: "139 144 154" };
export const P_INK: Record<number, string> = { 1: "#c32a2f", 2: "#b8500c", 3: "#926000", 4: "#5b616b" };
export const ST_RGB: Record<StaffStatus, string> = {
  available: "43 154 102",
  paging: "28 126 240",
  in_procedure: "139 92 246",
  on_case: "217 146 26",
  off_shift: "163 167 174",
};

export function PriorityChip({ p, size = "md", label = false }: { p: number; size?: "sm" | "md" | "lg"; label?: boolean }) {
  const sz = size === "lg" ? "h-7 px-2.5 text-[13px]" : size === "sm" ? "h-[18px] px-1.5 text-[10.5px]" : "h-[22px] px-2 text-[11.5px]";
  return (
    <span
      className={cx("inline-flex shrink-0 items-center gap-1 rounded-md font-mono font-semibold tracking-tight", sz)}
      style={{ background: `rgb(${P_RGB[p]} / ${p === 4 ? 0.12 : 0.13})`, color: P_INK[p], boxShadow: `inset 0 0 0 1px rgb(${P_RGB[p]} / 0.22)` }}
      title={`P${p} · ${PRIORITY_LABEL[p]}`}
    >
      P{p}
      {label && <span className="font-sans font-medium tracking-normal">{PRIORITY_LABEL[p]}</span>}
    </span>
  );
}

export function StatusDot({ status, size = 8, pulse = false }: { status: StaffStatus; size?: number; pulse?: boolean }) {
  const c = `rgb(${ST_RGB[status]})`;
  return (
    <span className="relative inline-flex shrink-0" style={{ width: size, height: size }} title={STATUS_LABEL[status]}>
      {(pulse || status === "paging") && (
        <span className="absolute inset-0 rounded-full" style={{ background: c, animation: "ring 1.6s cubic-bezier(.16,1,.3,1) infinite" }} />
      )}
      <span className="relative rounded-full" style={{ width: size, height: size, background: c, boxShadow: "0 0 0 1.5px #fff" }} />
    </span>
  );
}

export function Avatar({ name, status, size = 28, present }: { name: string; status?: StaffStatus; size?: number; present?: boolean }) {
  // a stable, quiet tint per person (never a signal colour)
  let h = 0;
  for (const ch of name) h = (h * 31 + ch.charCodeAt(0)) % 360;
  const hue = [210, 225, 245, 265, 190, 175][h % 6];
  return (
    <span className="relative inline-flex shrink-0">
      <span
        className="grid place-items-center rounded-full font-medium text-ink"
        style={{ width: size, height: size, fontSize: size * 0.36, background: `hsl(${hue} 30% 92%)`, boxShadow: "inset 0 0 0 1px rgb(17 20 24 / 0.06)" }}
      >
        {initials(name)}
      </span>
      {status && (
        <span className="absolute -bottom-0.5 -right-0.5">
          <StatusDot status={status} size={Math.max(7, size * 0.3)} />
        </span>
      )}
      {present && (
        <span className="absolute -left-0.5 -top-0.5 h-2.5 w-2.5 rounded-full border-2 border-white bg-accent" title="Device online" />
      )}
    </span>
  );
}

export function SourceTag({ source, synthetic }: { source: Source; synthetic?: boolean }) {
  const icon =
    source === "detector" ? (
      <svg width="11" height="11" viewBox="0 0 16 16" fill="none" aria-hidden>
        <path d="M1 8h3l2-5 3 10 2-5h4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ) : source === "voice" ? (
      <svg width="11" height="11" viewBox="0 0 16 16" fill="none" aria-hidden>
        <rect x="5.5" y="1.5" width="5" height="8.5" rx="2.5" stroke="currentColor" strokeWidth="1.5" />
        <path d="M3 8a5 5 0 0 0 10 0M8 13v2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ) : null;
  return (
    <span className={cx("inline-flex items-center gap-1 text-[11px]", source === "detector" ? "text-accent-ink" : "text-dim")}>
      {icon}
      {SOURCE_LABEL[source]}
      {synthetic && source !== "demo" && <span className="text-faint">· synthetic</span>}
    </span>
  );
}

/** Time left before a page escalates, as a ring that empties. */
export function Countdown({ remaining, total, size = 34, priority = 2 }: { remaining: number; total: number; size?: number; priority?: number }) {
  const r = (size - 4) / 2;
  const c = 2 * Math.PI * r;
  const frac = total > 0 ? Math.max(0, Math.min(1, remaining / total)) : 0;
  const urgent = remaining <= 10;
  const s = Math.max(0, Math.round(remaining));
  return (
    <span className="relative inline-grid shrink-0 place-items-center" style={{ width: size, height: size }} title={`${s}s before it escalates`}>
      <svg width={size} height={size} className="-rotate-90" aria-hidden>
        <circle cx={size / 2} cy={size / 2} r={r} fill="none" stroke="rgb(17 20 24 / 0.08)" strokeWidth="2.5" />
        <circle
          cx={size / 2}
          cy={size / 2}
          r={r}
          fill="none"
          stroke={urgent ? "rgb(229 72 77)" : `rgb(${P_RGB[priority]})`}
          strokeWidth="2.5"
          strokeLinecap="round"
          strokeDasharray={c}
          strokeDashoffset={c * (1 - frac)}
          style={{ transition: "stroke-dashoffset 1s linear" }}
        />
      </svg>
      <span className={cx("num absolute font-mono text-[10px] font-medium", urgent ? "text-high-ink" : "text-muted")}>
        {s >= 60 ? `${Math.floor(s / 60)}m` : `${s}s`}
      </span>
    </span>
  );
}

export function StatusText({ status }: { status: StaffStatus }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-[12px] text-muted">
      <StatusDot status={status} size={7} />
      {STATUS_LABEL[status]}
    </span>
  );
}

export const LEGEND: { status: StaffStatus; label: string }[] = [
  { status: "available", label: "Available" },
  { status: "paging", label: "Being paged" },
  { status: "in_procedure", label: "In procedure" },
  { status: "on_case", label: "On a case" },
  { status: "off_shift", label: "Off shift" },
];
