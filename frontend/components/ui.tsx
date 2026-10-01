"use client";

import { AnimatePresence, motion } from "motion/react";
import { useEffect, type ReactNode } from "react";

import { TONE_BG, TONE_RGB, TONE_TEXT, type Tone } from "@/lib/clinical";

export function cx(...c: (string | false | null | undefined)[]): string {
  return c.filter(Boolean).join(" ");
}

/* ------------------------------------------------------------------ */
/* Surfaces                                                            */
/* ------------------------------------------------------------------ */
export function Plate({ title, eyebrow, right, children, className, bodyClass }: {
  title?: ReactNode;
  eyebrow?: ReactNode;
  right?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClass?: string;
}) {
  return (
    <section className={cx("plate rise flex min-w-0 flex-col", className)}>
      {(title || right) && (
        <header className="flex min-h-12 items-center gap-3 border-b border-line-soft px-4 py-2.5">
          <div className="min-w-0 flex-1">
            {eyebrow && <div className="text-[11.5px] text-dim">{eyebrow}</div>}
            {title && <h2 className="truncate text-[14px] font-semibold tracking-tight text-ink">{title}</h2>}
          </div>
          {right && <div className="flex shrink-0 items-center gap-2">{right}</div>}
        </header>
      )}
      <div className={cx("min-w-0 flex-1 p-4", bodyClass)}>{children}</div>
    </section>
  );
}

export function SectionLabel({ children, right }: { children: ReactNode; right?: ReactNode }) {
  return (
    <div className="flex items-center gap-2 px-1 pb-2">
      <span className="font-mono text-[10.5px] font-medium uppercase tracking-[0.12em] text-dim">{children}</span>
      {right && <span className="ml-auto text-[11px] text-dim">{right}</span>}
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
  const sz = { sm: "text-[18px]", md: "text-[24px]", lg: "text-[32px]", xl: "text-[52px]" }[size];
  return (
    <div className="min-w-0">
      <div className="truncate text-[11.5px] text-dim">{label}</div>
      <div className={cx("num font-medium leading-tight tracking-tight", sz, TONE_TEXT[tone])}>
        {value}
        {unit && <span className="ml-1 text-[12px] font-normal text-dim">{unit}</span>}
      </div>
      {sub && <div className="truncate text-[11.5px] text-muted">{sub}</div>}
    </div>
  );
}

export function Pill({ tone = "neutral", children, dot = false, live = false, className }: {
  tone?: Tone;
  children: ReactNode;
  dot?: boolean;
  live?: boolean;
  className?: string;
}) {
  return (
    <span
      className={cx("inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2 py-0.5 text-[11.5px] font-medium", TONE_TEXT[tone], className)}
      style={{ borderColor: `rgb(${TONE_RGB[tone]} / 0.22)`, background: `rgb(${TONE_RGB[tone]} / 0.08)` }}
    >
      {dot && (
        <span className={cx("h-1.5 w-1.5 rounded-full", TONE_BG[tone], live && "live-dot")} style={{ color: `rgb(${TONE_RGB[tone]} / 0.5)` }} />
      )}
      {children}
    </span>
  );
}

export function Meter({ value, tone = "neutral", marks = [], className }: {
  value: number | null | undefined;
  tone?: Tone;
  marks?: number[];
  className?: string;
}) {
  const v = value == null ? 0 : Math.max(0, Math.min(1, value));
  return (
    <div className={cx("relative h-1.5 overflow-hidden rounded-full bg-s3", className)}>
      <div className={cx("h-full rounded-full transition-[width] duration-500", TONE_BG[tone])} style={{ width: `${v * 100}%` }} />
      {marks.map((m) => (
        <span key={m} className="absolute top-0 h-full w-px bg-ink/40" style={{ left: `${m * 100}%` }} />
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
      <path d={d} fill="none" stroke={`rgb(${TONE_RGB[tone]})`} strokeWidth={1.4} strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

export function Empty({ children, icon }: { children: ReactNode; icon?: ReactNode }) {
  return (
    <div className="flex min-h-24 flex-col items-center justify-center gap-2 px-4 py-6 text-center text-[13px] text-dim">
      {icon}
      {children}
    </div>
  );
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cx("animate-pulse rounded-md bg-s3/70", className)} />;
}

export function Kv({ k, v, mono = false }: { k: ReactNode; v: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1 text-[12.5px]">
      <span className="shrink-0 text-dim">{k}</span>
      <span className={cx("num min-w-0 truncate text-right text-ink", mono && "font-mono text-[11.5px]")}>{v}</span>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Controls                                                            */
/* ------------------------------------------------------------------ */
export function Button({ children, onClick, variant = "ghost", size = "md", disabled, type = "button", className, title }: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "ghost" | "primary" | "accent" | "quiet" | "danger";
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
        "inline-flex items-center justify-center gap-1.5 whitespace-nowrap rounded-lg font-medium transition duration-150 active:translate-y-px disabled:cursor-not-allowed disabled:opacity-40",
        size === "sm" && "h-7 px-2.5 text-[12px]",
        size === "md" && "h-8 px-3 text-[12.5px]",
        size === "lg" && "h-11 px-5 text-[14px]",
        variant === "primary" && "bg-ink text-white shadow-[0_1px_0_rgb(255_255_255/.12)_inset,0_1px_2px_rgb(0_0_0/.2)] hover:bg-ink/88",
        variant === "accent" && "bg-accent text-white shadow-[0_1px_2px_rgb(68_83_214/.35)] hover:bg-accent-ink",
        variant === "ghost" && "border border-line bg-s1 text-ink shadow-[0_1px_1px_rgb(16_18_24/.04)] hover:border-line-bright hover:bg-s2",
        variant === "quiet" && "text-muted hover:bg-s2 hover:text-ink",
        variant === "danger" && "border border-line bg-s1 text-high-ink hover:border-high/40 hover:bg-high/5",
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
      className={cx("relative h-5 w-9 shrink-0 rounded-full transition-colors disabled:opacity-40", checked ? "bg-accent" : "bg-s3")}
    >
      {/* anchored left: an absolute child of a <button> otherwise starts at its centred static position */}
      <span className={cx("absolute left-0 top-0.5 h-4 w-4 rounded-full bg-white shadow transition-transform", checked ? "translate-x-[18px]" : "translate-x-0.5")} />
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
    <div className="inline-flex rounded-lg border border-line bg-s2 p-0.5">
      {options.map((o) => (
        <button
          key={o.value}
          onClick={() => onChange(o.value)}
          className={cx(
            "relative rounded-md font-medium transition",
            size === "sm" ? "h-6 px-2 text-[11.5px]" : "h-7 px-3 text-[12.5px]",
            value === o.value ? "text-ink" : "text-dim hover:text-ink",
          )}
        >
          {value === o.value && (
            <motion.span layoutId={`seg-${options.map((x) => x.value).join("-")}`} className="absolute inset-0 rounded-md bg-s1 shadow-[0_1px_2px_rgb(16_18_24/.1)]" transition={{ type: "spring", bounce: 0, duration: 0.35 }} />
          )}
          <span className="relative">{o.label}</span>
        </button>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Overlays                                                            */
/* ------------------------------------------------------------------ */
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
            className="fixed inset-0 z-50 bg-ink/10"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
          />
          <motion.aside
            className="fixed inset-y-2 right-2 z-50 flex max-w-[calc(100vw-16px)] flex-col overflow-hidden rounded-2xl border border-line bg-s1 shadow-[0_30px_80px_-20px_rgb(16_18_24/.35)]"
            style={{ width }}
            initial={{ x: 40, opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={{ x: 40, opacity: 0 }}
            transition={{ type: "spring", bounce: 0, duration: 0.4 }}
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
          className="fixed inset-0 z-[60] flex items-start justify-center bg-ink/20 px-4 pt-[12vh] backdrop-blur-[2px]"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onMouseDown={(e) => e.target === e.currentTarget && onClose()}
        >
          <motion.div
            className="w-full overflow-hidden rounded-2xl border border-line bg-s1 shadow-[0_40px_120px_-30px_rgb(16_18_24/.45)]"
            style={{ maxWidth: width }}
            initial={{ y: 12, scale: 0.98, opacity: 0 }}
            animate={{ y: 0, scale: 1, opacity: 1 }}
            exit={{ y: 8, scale: 0.98, opacity: 0 }}
            transition={{ type: "spring", bounce: 0, duration: 0.35 }}
          >
            {children}
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
