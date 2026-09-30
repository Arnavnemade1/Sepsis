import type { ReactNode } from "react";

import { TONE_BG, TONE_RGB, TONE_TEXT, type Tone } from "@/lib/clinical";

export function cx(...c: (string | false | null | undefined)[]): string {
  return c.filter(Boolean).join(" ");
}

/** A glass plate with an optional header row. The unit every workspace is built from. */
export function Plate({
  title,
  eyebrow,
  right,
  children,
  className,
  bodyClass,
}: {
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
        <header className="flex min-h-11 items-center gap-3 border-b border-line-soft px-4 py-2.5">
          <div className="min-w-0 flex-1">
            {eyebrow && <div className="text-[11px] tracking-wide text-dim">{eyebrow}</div>}
            {title && <h2 className="truncate text-[14px] font-medium text-ink">{title}</h2>}
          </div>
          {right && <div className="flex shrink-0 items-center gap-2">{right}</div>}
        </header>
      )}
      <div className={cx("min-w-0 flex-1 p-4", bodyClass)}>{children}</div>
    </section>
  );
}

/** A reading: small label, big light tabular number, dim unit. */
export function Stat({
  label,
  value,
  unit,
  tone = "neutral",
  sub,
  size = "md",
}: {
  label: ReactNode;
  value: ReactNode;
  unit?: ReactNode;
  tone?: Tone;
  sub?: ReactNode;
  size?: "sm" | "md" | "lg" | "xl";
}) {
  const sz = { sm: "text-[18px]", md: "text-[24px]", lg: "text-[34px]", xl: "text-[52px]" }[size];
  return (
    <div className="min-w-0">
      <div className="truncate text-[11.5px] text-dim">{label}</div>
      <div className={cx("num font-light leading-tight tracking-tight", sz, TONE_TEXT[tone])}>
        {value}
        {unit && <span className="ml-1 text-[12px] font-normal text-dim">{unit}</span>}
      </div>
      {sub && <div className="truncate text-[11.5px] text-muted">{sub}</div>}
    </div>
  );
}

export function Pill({ tone = "neutral", children, dot = false, live = false }: {
  tone?: Tone;
  children: ReactNode;
  dot?: boolean;
  live?: boolean;
}) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1.5 whitespace-nowrap rounded-full border px-2 py-0.5 text-[11.5px]",
        TONE_TEXT[tone],
      )}
      style={{
        borderColor: `rgb(${TONE_RGB[tone]} / 0.28)`,
        background: `rgb(${TONE_RGB[tone]} / 0.08)`,
      }}
    >
      {dot && (
        <span
          className={cx("h-1.5 w-1.5 rounded-full", TONE_BG[tone], live && "live-dot")}
          style={{ color: `rgb(${TONE_RGB[tone]} / 0.6)` }}
        />
      )}
      {children}
    </span>
  );
}

/** A bar whose fill IS the value (0–1). Never a cosmetic multiplier. */
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
        <span key={m} className="absolute top-0 h-full w-px bg-ink/50" style={{ left: `${m * 100}%` }} />
      ))}
    </div>
  );
}

/** A tiny line of the last N readings. Gaps (null) break the line, honestly. */
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
    const x = i * step;
    const y = h - 2 - ((v - lo) / span) * (h - 4);
    d += `${pen ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)}`;
    pen = true;
  });
  return (
    <svg width={w} height={h} viewBox={`0 0 ${w} ${h}`} className="overflow-visible" aria-hidden>
      <path d={d} fill="none" stroke={`rgb(${TONE_RGB[tone]})`} strokeWidth={1.4} strokeLinejoin="round" strokeLinecap="round" />
    </svg>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="flex min-h-24 items-center justify-center px-4 text-center text-[13px] text-dim">{children}</div>;
}

export function Skeleton({ className }: { className?: string }) {
  return <div className={cx("animate-pulse rounded-md bg-s2", className)} />;
}

export function Kv({ k, v, mono = false }: { k: ReactNode; v: ReactNode; mono?: boolean }) {
  return (
    <div className="flex items-baseline justify-between gap-3 py-1 text-[12.5px]">
      <span className="shrink-0 text-dim">{k}</span>
      <span className={cx("num min-w-0 truncate text-right text-ink", mono && "font-mono text-[11.5px]")}>{v}</span>
    </div>
  );
}

export function Button({
  children,
  onClick,
  variant = "ghost",
  disabled,
  type = "button",
  className,
}: {
  children: ReactNode;
  onClick?: () => void;
  variant?: "ghost" | "primary" | "quiet";
  disabled?: boolean;
  type?: "button" | "submit";
  className?: string;
}) {
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled}
      className={cx(
        "inline-flex items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-[12.5px] font-medium transition duration-150 disabled:cursor-not-allowed disabled:opacity-40",
        variant === "primary" && "brand-fill text-bg hover:brightness-110",
        variant === "ghost" && "border border-line bg-s2 text-ink hover:border-line-bright hover:bg-s3",
        variant === "quiet" && "text-muted hover:bg-s2 hover:text-ink",
        className,
      )}
    >
      {children}
    </button>
  );
}
