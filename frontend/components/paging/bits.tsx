"use client";

import { Microphone, Monitor, Pulse } from "@/components/icons";
import { cx } from "@/components/ui";
import { initials, PRIORITY_LABEL, SOURCE_LABEL, STATUS_LABEL, type Source, type StaffStatus } from "@/lib/paging";

/** Priority colour as an "r g b" triple (for alpha) and as text on the console's grey. */
export const P_RGB: Record<number, string> = { 1: "229 72 77", 2: "245 118 46", 3: "238 177 42", 4: "120 129 141" };
export const P_INK: Record<number, string> = { 1: "#ff6b70", 2: "#ff9357", 3: "#f4c55a", 4: "#a8afb8" };
/** P1 is the only solid chip on the console; P2–P4 are tints, so the eye lands on P1 first. */
const P_CHIP: Record<number, { bg: string; fg: string }> = {
  1: { bg: "#e5484d", fg: "#ffffff" },
  2: { bg: "rgb(245 118 46 / 0.16)", fg: "#ff9a62" },
  3: { bg: "rgb(238 177 42 / 0.15)", fg: "#f4c55a" },
  4: { bg: "rgb(255 255 255 / 0.07)", fg: "#b0b5bc" },
};
export const ST_RGB: Record<StaffStatus, string> = {
  available: "47 182 124",
  paging: "61 142 245",
  in_procedure: "156 109 240",
  on_case: "31 179 166",
  off_shift: "95 102 111",
};

/** The priority badge: the one place a priority's colour is spent. */
export function PriorityChip({ p, size = "md", label = false }: { p: number; size?: "sm" | "md" | "lg"; label?: boolean }) {
  const c = P_CHIP[p] ?? P_CHIP[4];
  const sz = size === "lg" ? "h-6 px-2 text-xs" : size === "sm" ? "h-[18px] px-[5px] text-2xs" : "h-5 px-1.5 text-2xs";
  return (
    <span
      className={cx("num inline-flex shrink-0 items-center gap-1.5 rounded-[5px] font-semibold leading-none", sz)}
      style={{ background: c.bg, color: c.fg }}
      title={`P${p} · ${PRIORITY_LABEL[p]}`}
    >
      P{p}
      {label && <span className="font-medium">{PRIORITY_LABEL[p]}</span>}
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
      <span className="grid place-items-center rounded-full bg-s4 font-semibold text-muted shadow-[inset_0_0_0_1px_rgb(255_255_255/.06)]" style={{ width: size, height: size, fontSize: Math.max(9, size * 0.38) }}>
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
  const I = source === "detector" ? Pulse : source === "monitor" ? Monitor : source === "voice" ? Microphone : null;
  return (
    <span className={cx("inline-flex shrink-0 items-center gap-1 text-xs", source === "detector" || source === "monitor" ? "text-muted" : "text-dim")}>
      {I && <I size={13} className="text-dim" aria-hidden />}
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
    <span className={cx("num font-medium", urgent ? "text-high-ink" : "text-ink", className)} title={`${Math.max(0, Math.round(remaining))} s before it escalates`}>
      {fmtClock(remaining)}
    </span>
  );
}

/** The same deadline as a draining line, under a row or a pager. */
export function TimerBar({ remaining, total, priority, className }: { remaining: number; total: number; priority: number; className?: string }) {
  const frac = total > 0 ? Math.max(0, Math.min(1, remaining / total)) : 0;
  return (
    <span className={cx("block h-[2px] overflow-hidden bg-white/[0.06]", className)} aria-hidden>
      <span
        className="block h-full"
        style={{ width: `${frac * 100}%`, background: remaining <= 10 ? "#e5484d" : `rgb(${P_RGB[priority]})`, transition: "width 1s linear" }}
      />
    </span>
  );
}

export function StatusText({ status }: { status: StaffStatus }) {
  return (
    <span className="inline-flex items-center gap-1.5 text-xs text-muted">
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
