import type { ReactNode } from "react";

/* ── Section scaffolding ─────────────────────────────────────────────────── */

export function Section({
  id,
  eyebrow,
  title,
  lede,
  children,
  className = "",
}: {
  id?: string;
  eyebrow?: string;
  title?: ReactNode;
  lede?: ReactNode;
  children?: ReactNode;
  className?: string;
}) {
  return (
    <section id={id} className={`scroll-mt-24 ${className}`}>
      {eyebrow && <p className="eyebrow mb-3">{eyebrow}</p>}
      {title && (
        <h2 className="font-display text-[clamp(1.5rem,3vw,2rem)] font-semibold leading-[1.15] tracking-[-0.02em] text-ink text-balance">
          {title}
        </h2>
      )}
      {lede && (
        <p className="mt-3 max-w-2xl text-[15px] leading-relaxed text-muted text-pretty">{lede}</p>
      )}
      {children}
    </section>
  );
}

/**
 * A hairline panel.
 *
 * Translucent rather than solid, because the site's footage runs behind every
 * page: an opaque panel would punch a rectangular hole in it. Frosted glass over
 * moving texture keeps the surface calm enough to read dense numbers on while
 * letting the background stay continuous. Still flat — no glow, no gradient.
 */
export function Panel({
  children,
  className = "",
}: {
  children: ReactNode;
  className?: string;
}) {
  return (
    <div
      className={`rounded-lg border border-line bg-s1/62 backdrop-blur-xl ${className}`}
    >
      {children}
    </div>
  );
}

/* ── Metric display ──────────────────────────────────────────────────────── */

export function MetricTile({
  value,
  label,
  detail,
  tone = "ink",
  size = "lg",
}: {
  value: ReactNode;
  label: string;
  detail?: string;
  tone?: "ink" | "accent" | "low" | "med" | "high";
  /** `sm` for readings that carry two figures (e.g. "2.21% / 1.41%") — at the
   *  large size they wrap and knock the tile out of the row's rhythm. */
  size?: "lg" | "sm";
}) {
  const toneClass = {
    ink: "text-ink",
    accent: "text-accent-ink",
    low: "text-low-ink",
    med: "text-med-ink",
    high: "text-high-ink",
  }[tone];

  const sizeClass = size === "lg" ? "text-[28px]" : "text-[20px]";

  return (
    <div className="flex h-full flex-col rounded-lg border border-line bg-s1/62 p-5 backdrop-blur-xl">
      <div
        className={`num font-mono ${sizeClass} leading-none font-medium ${toneClass}`}
      >
        {value}
      </div>
      <div className="mt-2.5 text-[13px] font-medium text-ink">{label}</div>
      {detail && <div className="mt-1 text-[12px] leading-snug text-dim">{detail}</div>}
    </div>
  );
}

/** Provenance line. Every number on this site can name the file it came from. */
export function SourceNote({ children }: { children: ReactNode }) {
  return (
    <p className="mt-3 font-mono text-[11px] leading-relaxed text-faint">{children}</p>
  );
}

/** A verdict chip. Clinical hues are correct here — these encode a real state
 *  (a result promoted or refused, a subsystem validated or not). */
export function StatusChip({
  tone,
  children,
}: {
  tone: "low" | "med" | "high" | "nodata" | "accent";
  children: ReactNode;
}) {
  const map = {
    low: "border-low/35 bg-low/10 text-low-ink",
    med: "border-med/35 bg-med/10 text-med-ink",
    high: "border-high/35 bg-high/10 text-high-ink",
    nodata: "border-line bg-s2 text-dim",
    accent: "border-accent/35 bg-accent/10 text-accent-ink",
  }[tone];

  return (
    <span
      className={`inline-flex shrink-0 items-center gap-1.5 rounded-sm border px-2 py-[3px] font-mono text-[10.5px] uppercase tracking-[0.08em] ${map}`}
    >
      {children}
    </span>
  );
}

/* ── Tables ──────────────────────────────────────────────────────────────── */

export function DataTable({
  head,
  children,
  caption,
}: {
  head: ReactNode[];
  children: ReactNode;
  caption?: string;
}) {
  return (
    // The page body must never scroll sideways — wide tables scroll inside here.
    <div className="scroll-x rounded-lg border border-line bg-s1/62 backdrop-blur-xl">
      <table className="w-full min-w-[560px] border-collapse text-[13px]">
        {caption && <caption className="sr-only">{caption}</caption>}
        <thead>
          <tr className="border-b border-line">
            {head.map((h, i) => (
              <th
                key={i}
                scope="col"
                className={`px-4 py-3 font-mono text-[10.5px] font-medium uppercase tracking-[0.08em] text-dim ${
                  i === 0 ? "text-left" : "text-right"
                }`}
              >
                {h}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>{children}</tbody>
      </table>
    </div>
  );
}

export function Row({
  cells,
  emphasis = false,
}: {
  cells: ReactNode[];
  emphasis?: boolean;
}) {
  return (
    <tr
      className={`border-b border-line-soft last:border-0 ${
        emphasis ? "bg-s2/60" : ""
      }`}
    >
      {cells.map((c, i) => (
        <td
          key={i}
          className={`px-4 py-3 ${i === 0 ? "text-left" : "num text-right font-mono"} ${
            emphasis && i === 0 ? "font-medium text-ink" : i === 0 ? "text-muted" : "text-ink"
          }`}
        >
          {c}
        </td>
      ))}
    </tr>
  );
}
