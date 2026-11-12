"use client";

import { AnimatePresence, motion } from "motion/react";
import { useEffect, type ReactNode } from "react";

import { TONE_BG, TONE_RGB, TONE_TEXT, type Tone } from "@/lib/clinical";

import { Close, Lock } from "./icons";

export function cx(...c: (string | false | null | undefined)[]): string {
  return c.filter(Boolean).join(" ");
}

/* ------------------------------------------------------------------ */
/* Surfaces                                                            */
/* ------------------------------------------------------------------ */
/** The one surface: a flat panel with an optional header row. */
export function Panel({ title, meta, right, children, className, bodyClass, flush = false }: {
  title?: ReactNode;
  meta?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClass?: string;
  flush?: boolean;
}) {
  return (
    <section className={cx("panel flex min-w-0 flex-col", className)}>
      {(title || right) && (
        <header className="flex min-h-11 shrink-0 flex-wrap items-center gap-x-3 gap-y-2 border-b border-line px-4 py-2.5">
          {/* the title keeps at least 11rem; below that the controls wrap under it instead of
              squeezing it to one word per line, and scroll sideways if still too wide */}
          <div className="min-w-0 flex-1 basis-44">
            {title && <h2 className="truncate text-sm font-semibold tracking-tight text-ink">{title}</h2>}
            {meta && <div className="mt-0.5 text-xs leading-snug text-dim">{meta}</div>}
          </div>
          {right && <div className="flex max-w-full shrink-0 items-center gap-2 overflow-x-auto [scrollbar-width:none]">{right}</div>}
        </header>
      )}
      <div className={cx("min-w-0 flex-1", !flush && "p-4", bodyClass)}>{children}</div>
    </section>
  );
}

/** The drill-down pages' name for a panel; `eyebrow` is its description line. */
export function Plate({ title, eyebrow, right, children, className, bodyClass }: {
  title?: ReactNode;
  eyebrow?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClass?: string;
}) {
  return (
    <Panel title={title} meta={eyebrow} right={right} className={className} bodyClass={bodyClass}>
      {children}
    </Panel>
  );
}

export function SectionLabel({ children, right, className }: { children: ReactNode; right?: ReactNode; className?: string }) {
  return (
    <div className={cx("flex items-center gap-2 pb-2", className)}>
      <span className="text-xs font-medium text-muted">{children}</span>
      {right && <span className="ml-auto text-xs text-dim">{right}</span>}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Readings                                                            */
/* ------------------------------------------------------------------ */
export function Stat({ label, value, unit, tone = "neutral", sub, size = "md" }: {
  label: ReactNode;
  value: ReactNode;
  unit?: ReactNode;
  tone?: Tone;
  sub?: ReactNode;
  size?: "sm" | "md" | "lg" | "xl";
}) {
  const sz = { sm: "text-md", md: "text-xl", lg: "text-2xl", xl: "text-3xl" }[size];
  return (
    <div className="min-w-0">
      <div className="truncate text-xs text-dim">{label}</div>
      <div className={cx("num mt-1 font-medium leading-none tracking-tighter", sz, TONE_TEXT[tone])}>
        {value}
        {unit && <span className="ml-1 text-xs font-normal tracking-normal text-dim">{unit}</span>}
      </div>
      {sub && <div className="mt-1.5 truncate text-xs text-dim">{sub}</div>}
    </div>
  );
}

/** A row of headline readings sharing hairline dividers: the top of most screens.
 *  One shape everywhere, so a number reads the same on every screen. */
export function StatStrip({ items, className, cols = 6 }: {
  items: { k: ReactNode; v: ReactNode; sub?: ReactNode; tone?: string; mark?: ReactNode; title?: string }[];
  className?: string;
  cols?: 4 | 5 | 6;
}) {
  const grid = { 4: "sm:grid-cols-4", 5: "sm:grid-cols-3 xl:grid-cols-5", 6: "sm:grid-cols-3 xl:grid-cols-6" }[cols];
  return (
    // dividers are drawn by each tile (right + bottom) and clipped at the frame, so a short last
    // row ends in panel, not in a grey block
    <div className={cx("grid shrink-0 grid-cols-2 overflow-hidden rounded-lg border border-line bg-s1", grid, className)}>
      {items.map((it, i) => (
        <div key={i} className="min-w-0 px-4 py-3.5 shadow-[1px_0_0_0_var(--color-line),0_1px_0_0_var(--color-line)]" title={it.title}>
          <div className="truncate text-xs text-dim">{it.k}</div>
          <div className={cx("num mt-1.5 flex items-center gap-2 text-xl font-medium leading-none tracking-tighter", it.tone ?? "text-ink")}>
            {it.mark}
            <span className="min-w-0 truncate">{it.v}</span>
          </div>
          {it.sub != null && <div className="mt-2 truncate text-xs text-dim">{it.sub}</div>}
        </div>
      ))}
    </div>
  );
}

/** A small labelled tag. Neutral tags are grey; toned tags carry clinical state only. */
export function Tag({ tone = "neutral", children, dot = false, className, title }: {
  tone?: Tone;
  children: ReactNode;
  dot?: boolean;
  live?: boolean;
  className?: string;
  title?: string;
}) {
  const neutral = tone === "neutral" || tone === "nodata";
  return (
    <span
      title={title}
      className={cx(
        "inline-flex h-5 items-center gap-1.5 whitespace-nowrap rounded-sm px-1.5 text-2xs font-medium leading-none",
        neutral ? "bg-white/[0.06] text-muted" : TONE_TEXT[tone],
        className,
      )}
      style={neutral ? undefined : { background: `rgb(${TONE_RGB[tone]} / 0.13)` }}
    >
      {dot && <span className={cx("h-1.5 w-1.5 rounded-full", TONE_BG[tone])} />}
      {children}
    </span>
  );
}
export const Pill = Tag;

export function Meter({ value, tone = "neutral", marks = [], className }: {
  value: number | null | undefined;
  tone?: Tone;
  marks?: number[];
  className?: string;
}) {
  const v = value == null ? 0 : Math.max(0, Math.min(1, value));
  return (
    <div className={cx("relative h-1 overflow-hidden rounded-full bg-white/[0.07]", className)}>
      <div className={cx("h-full rounded-full transition-[width] duration-500", TONE_BG[tone])} style={{ width: `${v * 100}%` }} />
      {marks.map((m) => (
        <span key={m} className="absolute top-0 h-full w-px bg-ink/50" style={{ left: `${m * 100}%` }} />
      ))}
    </div>
  );
}

export function Sparkline({ data, tone = "neutral", w = 120, h = 28, domain }: {
  data: (number | null)[];
  tone?: Tone;
  w?: number;
  h?: number;
  domain?: [number, number];
}) {
  const vals = data.filter((v): v is number => v != null && !Number.isNaN(v));
  if (vals.length < 2) return <svg width={w} height={h} aria-hidden />;
  const [lo, hi] = domain ?? [Math.min(...vals), Math.max(...vals)];
  const span = hi - lo || 1;
  const step = w / Math.max(1, data.length - 1);
  let d = "";
  let pen = false;
  data.forEach((v, i) => {
    if (v == null || Number.isNaN(v)) {
      pen = false;
      return;
    }
    d += `${pen ? "L" : "M"}${(i * step).toFixed(1)},${(h - 2 - ((v - lo) / span) * (h - 4)).toFixed(1)}`;
    pen = true;
  });
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="overflow-visible" aria-hidden>
      <path d={d} fill="none" stroke={`rgb(${TONE_RGB[tone]})`} strokeWidth={1.3} strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

/** A row of tiny bars: counts per time bucket, oldest first. */
export function MiniBars({ data, color = "#868d97", w = 76, h = 22, title }: { data: number[]; color?: string; w?: number; h?: number; title?: string }) {
  const max = Math.max(1, ...data);
  const bw = w / Math.max(1, data.length);
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} aria-hidden={!title} role={title ? "img" : undefined} aria-label={title}>
      {data.map((v, i) => {
        const bh = v > 0 ? Math.max(2, (v / max) * (h - 1)) : 1;
        return <rect key={i} x={i * bw + 0.75} y={h - bh} width={Math.max(1, bw - 1.5)} height={bh} rx={0.75} fill={v > 0 ? color : "#3e444c"} />;
      })}
    </svg>
  );
}

/** A tiny line over time buckets; gaps (null) break the line. */
export function MiniLine({ data, color = "#6eaaff", w = 76, h = 22 }: { data: (number | null)[]; color?: string; w?: number; h?: number }) {
  const vals = data.filter((v): v is number => v != null);
  if (vals.length < 2) return <svg width={w} height={h} aria-hidden />;
  const lo = Math.min(...vals), hi = Math.max(...vals);
  const span = hi - lo || 1;
  const step = w / Math.max(1, data.length - 1);
  let d = "";
  let pen = false;
  data.forEach((v, i) => {
    if (v == null) {
      pen = false;
      return;
    }
    d += `${pen ? "L" : "M"}${(i * step).toFixed(1)},${(h - 2 - ((v - lo) / span) * (h - 4)).toFixed(1)}`;
    pen = true;
  });
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="overflow-visible" aria-hidden>
      <path d={d} fill="none" stroke={color} strokeWidth={1.4} strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

export function Empty({ children, icon }: { children: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex min-h-24 flex-col items-center justify-center gap-2.5 px-4 py-8 text-center text-sm text-dim">
      {icon && <span className="text-faint">{icon}</span>}
      <span className="max-w-xs">{children}</span>
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cx("animate-pulse rounded-md bg-white/[0.04]", className)} />;
}

export function Kv({ k, v, mono = false }: { k: ReactNode; v: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1.5 text-sm">
      <span className="shrink-0 text-dim">{k}</span>
      <span className={cx("num min-w-0 truncate text-right text-ink", mono && "font-mono text-xs")}>{v}</span>
    </div>
  );
}

/** A read this role is not entitled to. In audit mode the gate serves it and records a
 *  would-deny; in enforce mode it refuses it. Either way the interface says so. */
export function Restricted({ cap, role, children }: { cap: string; role: string | null; children?: ReactNode }) {
  return (
    <div className="flex items-start gap-2.5 rounded-md bg-med/[0.08] px-3 py-2.5 text-xs leading-relaxed text-muted">
      <Lock size={14} className="mt-[2px] shrink-0 text-med-ink" aria-hidden />
      <span>
        Requires <span className="font-mono text-xs text-med-ink">{cap}</span>
        {role ? <>, which the {role} role does not hold</> : null}. {children ?? "In audit mode the gate serves this read and records it as a would-deny; in enforce mode it is refused."}
      </span>
    </div>
  );
}

export function Kbd({ children }: { children: ReactNode }) {
  return <span className="kbd">{children}</span>;
}

/* ------------------------------------------------------------------ */
/* Controls                                                            */
/* ------------------------------------------------------------------ */
export function Button({ children, onClick, variant = "ghost", size = "md", disabled, type = "button", className, title }: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "ghost" | "primary" | "accent" | "quiet" | "danger" | "success";
  size?: "sm" | "md" | "lg";
  disabled?: boolean;
  type?: "button" | "submit";
  className?: string;
  title?: string;
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      title={title}
      className={cx(
        "inline-flex select-none items-center justify-center gap-1.5 whitespace-nowrap rounded-md font-medium transition-[background-color,color,box-shadow] duration-100 active:translate-y-px disabled:pointer-events-none disabled:opacity-40",
        size === "sm" && "h-7 px-2.5 text-xs",
        size === "md" && "h-8 px-3 text-sm",
        size === "lg" && "h-10 px-4 text-base",
        variant === "primary" && "bg-ink text-[#17191c] shadow-[0_1px_2px_rgb(0_0_0/.3)] hover:bg-white",
        variant === "accent" && "bg-accent text-white shadow-[0_1px_2px_rgb(0_0_0/.3)] hover:bg-[#5598f6]",
        variant === "success" && "bg-low text-[#0b1d14] hover:bg-[#3cc98b]",
        variant === "ghost" && "bg-white/[0.06] text-ink shadow-[inset_0_0_0_1px_rgb(255_255_255/.07)] hover:bg-white/[0.1]",
        variant === "quiet" && "text-muted hover:bg-white/[0.06] hover:text-ink",
        variant === "danger" && "bg-high/[0.12] text-high-ink hover:bg-high/[0.18]",
        className,
      )}
    >
      {children}
    </button>
  );
}

export function Switch({ checked, onChange, disabled, label }: {
  checked: boolean;
  onChange: (v: boolean) => void;
  disabled?: boolean;
  label?: string;
}) {
  return (
    <button
      role="switch"
      aria-checked={checked}
      aria-label={label}
      disabled={disabled}
      onClick={() => onChange(!checked)}
      className={cx("relative h-[18px] w-8 shrink-0 rounded-full transition-colors duration-150 disabled:opacity-40", checked ? "bg-accent" : "bg-white/[0.12]")}
    >
      {/* anchored left: an absolute child of a <button> otherwise starts at its centred static position */}
      <span className={cx("absolute left-0 top-[2px] h-[14px] w-[14px] rounded-full bg-white shadow-[0_1px_2px_rgb(0_0_0/.35)] transition-transform duration-200 ease-out", checked ? "translate-x-[16px]" : "translate-x-[2px]")} />
    </button>
  );
}

export function Segmented<T extends string>({ value, options, onChange, size = "md" }: {
  value: T;
  options: { value: T; label: ReactNode }[];
  onChange: (v: T) => void;
  size?: "sm" | "md";
}) {
  return (
    <div role="group" className="inline-flex shrink-0 gap-px rounded-md bg-black/[0.18] p-[3px] shadow-[inset_0_0_0_1px_rgb(255_255_255/.05)]">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          aria-pressed={value === o.value}
          className={cx(
            "whitespace-nowrap rounded-[5px] font-medium transition-colors duration-100",
            size === "sm" ? "h-6 px-2 text-xs" : "h-[26px] px-2.5 text-xs",
            value === o.value ? "bg-s4 text-ink shadow-[0_1px_2px_rgb(0_0_0/.3)]" : "text-dim hover:text-ink",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function IconButton({ children, onClick, label, className }: { children: ReactNode; onClick?: () => void; label: string; className?: string }) {
  return (
    <button onClick={onClick} aria-label={label} title={label} className={cx("grid h-7 w-7 place-items-center rounded-md text-dim transition-colors hover:bg-white/[0.07] hover:text-ink", className)}>
      {children}
    </button>
  );
}

export const CloseIcon = () => <Close size={15} aria-hidden />;

/* ------------------------------------------------------------------ */
/* Overlays                                                            */
/* ------------------------------------------------------------------ */
/** A sheet docked to the right edge: the inspector. */
export function Drawer({ open, onClose, children, width = 520 }: {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  width?: number;
}) {
  useEffect(() => {
    if (!open) return;
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [open, onClose]);
  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div
            className="fixed inset-0 z-50 bg-black/40"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.15 }}
            onClick={onClose}
          />
          <motion.aside
            className="fixed inset-y-0 right-0 z-50 flex max-w-full flex-col border-l border-line bg-s1 shadow-[-32px_0_64px_-24px_rgb(0_0_0/.6)] lg:inset-y-2 lg:right-2 lg:rounded-xl lg:border"
            style={{ width }}
            initial={{ x: 32, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 32, opacity: 0 }}
            transition={{ duration: 0.24, ease: [0.16, 1, 0.3, 1] }}
          >
            {children}
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}

export function Modal({ open, onClose, children, width = 640 }: {
  open: boolean;
  onClose: () => void;
  children: ReactNode;
  width?: number;
}) {
  useEffect(() => {
    if (!open) return;
    const esc = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", esc);
    return () => window.removeEventListener("keydown", esc);
  }, [open, onClose]);
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[60] flex items-start justify-center bg-black/50 px-3 pt-[12vh]"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.12 }}
          onMouseDown={(e) => e.target === e.currentTarget && onClose()}
        >
          <motion.div
            className="pop w-full overflow-hidden"
            style={{ maxWidth: width }}
            initial={{ y: 8, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            exit={{ y: 4, opacity: 0 }}
            transition={{ duration: 0.2, ease: [0.16, 1, 0.3, 1] }}
          >
            {children}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
