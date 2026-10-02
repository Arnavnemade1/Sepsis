"use client";

import { cx } from "@/components/ui";
import { initials, PRIORITY_LABEL, SOURCE_LABEL, STATUS_LABEL, type Source, type StaffStatus } from "@/lib/paging";

/** Priority colour as an "r g b" triple (for alpha) and as text on the console's grey. */
export const P_RGB: Record<number, string> = { 1: "229 72 77", 2: "245 118 46", 3: "238 177 42", 4: "120 129 141" };
export const P_INK: Record<number, string> = { 1: "#ff6b70", 2: "#ff9357", 3: "#f4c55a", 4: "#a8afb8" };
const P_SOLID: Record<number, { bg: string; fg: string }> = {
  1: { bg: "#e5484d", fg: "#ffffff" },
  2: { bg: "#f5762e", fg: "#1f1206" },
  3: { bg: "#eeb12a", fg: "#1f1704" },
  4: { bg: "#4b525b", fg: "#dfe3e8" },
};
export const ST_RGB: Record<StaffStatus, string> = {
  available: "47 182 124",
  paging: "61 142 245",
  in_procedure: "156 109 240",
  on_case: "31 179 166",
  off_shift: "95 102 111",
};

/** A solid priority badge: the one place a priority's colour is spent at full strength. */
export function PriorityChip({ p, size = "md", label = false }: { p: number; size?: "sm" | "md" | "lg"; label?: boolean }) {
  const c = P_SOLID[p] ?? P_SOLID[4];
  const sz = size === "lg" ? "h-6 px-2 text-[12px]" : size === "sm" ? "h-[18px] px-1 text-[10.5px]" : "h-5 px-1.5 text-[11px]";
  return (
    <span
      className={cx("inline-flex shrink-0 items-center gap-1.5 rounded-[4px] font-mono font-semibold leading-none", sz)}
      style={{ background: c.bg, color: c.fg }}
      title={`P${p} · ${PRIORITY_LABEL[p]}`}
    >
      P{p}
      {label && <span className="font-sans font-medium">{PRIORITY_LABEL[p]}</span>}
    </span>
  );
}

export function StatusDot({ status, size = 8, ring, still = false }: { status: StaffStatus; size?: number; ring?: string; still?: boolean }) {
  return (
    <span
      className={cx("inline-block shrink-0 rounded-full", status === "paging" && !still && "blink")}
      style={{ width: size, height: size, background: `rgb(${ST_RGB[status]})`, boxShadow: ring ? `0 0 0 2px ${ring}` : undefined }}
      title={STATUS_LABEL[status]}
    />
  );
}

export function Avatar({ name, status, size = 24, present, ring = "var(--color-s1)" }: { name: string; status?: StaffStatus; size?: number; present?: boolean; ring?: string }) {
  return (
    <span className="relative inline-flex shrink-0">
      <span className="grid place-items-center rounded-full bg-s4 font-semibold text-muted" style={{ width: size, height: size, fontSize: Math.max(9, size * 0.36) }}>
        {initials(name)}
      </span>
      {status && (
        <span className="absolute -bottom-px -right-px">
          <StatusDot status={status} size={Math.max(6, Math.round(size * 0.3))} ring={ring} />
        </span>
      )}
      {present && <span className="absolute -left-px -top-px h-2 w-2 rounded-full bg-accent" style={{ boxShadow: `0 0 0 2px ${ring}` }} title="Pager open on a device" />}
    </span>
  );
}

export function SourceTag({ source, synthetic }: { source: Source; synthetic?: boolean }) {
  const icon =
    source === "detector" ? (
      <svg width="11" height="11" viewBox="0 0 16 16" fill="none" aria-hidden>
        <path d="M1 8h3l2-5 3 10 2-5h4" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ) : source === "monitor" ? (
      <svg width="11" height="11" viewBox="0 0 16 16" fill="none" aria-hidden>
        <rect x="1.5" y="2.5" width="13" height="9" rx="1.5" stroke="currentColor" strokeWidth="1.5" />
        <path d="M3.5 8h2l1.2-2.5L8.5 10l1.2-2h2.8M6 14h4" stroke="currentColor" strokeWidth="1.4" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    ) : source === "voice" ? (
      <svg width="11" height="11" viewBox="0 0 16 16" fill="none" aria-hidden>
        <rect x="5.5" y="1.5" width="5" height="8.5" rx="2.5" stroke="currentColor" strokeWidth="1.5" />
        <path d="M3 8a5 5 0 0 0 10 0M8 13v2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
      </svg>
    ) : null;
  return (
    <span className={cx("inline-flex shrink-0 items-center gap-1 text-[11px]", source === "detector" || source === "monitor" ? "text-accent-ink" : "text-dim")}>
      {icon}
      {SOURCE_LABEL[source]}
      {synthetic && source !== "demo" && <span className="text-faint">· synthetic</span>}
    </span>
  );
}

export function fmtClock(sec: number): string {
  const s = Math.max(0, Math.round(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

/** Time left before a page escalates: m:ss, red in the last ten seconds. */
export function Countdown({ remaining, className }: { remaining: number; total?: number; size?: number; priority?: number; className?: string }) {
  const urgent = remaining <= 10;
  return (
    <span className={cx("num font-mono font-medium", urgent ? "text-high-ink" : "text-ink", className)} title={`${Math.max(0, Math.round(remaining))} s before it escalates`}>
      {fmtClock(remaining)}
    </span>
  );
}

/** The same deadline as a draining line, under a row or a pager. */
export function TimerBar({ remaining, total, priority, className }: { remaining: number; total: number; priority: number; className?: string }) {
  const frac = total > 0 ? Math.max(0, Math.min(1, remaining / total)) : 0;
  return (
    <span className={cx("block h-[2px] overflow-hidden bg-s3", className)} aria-hidden>
      <span
        className="block h-full"
        style={{ width: `${frac * 100}%`, background: remaining <= 10 ? "#e5484d" : `rgb(${P_RGB[priority]})`, transition: "width 1s linear" }}
      />
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
