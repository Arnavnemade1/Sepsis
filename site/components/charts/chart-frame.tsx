import type { ReactNode } from "react";

import { ChartReveal } from "@/components/motion/chart-reveal";

import { M, scaleX, scaleY, ticks, type Range, type Margins } from "@/lib/scales";
import { fmtTick } from "@/lib/format";

export interface ChartFrameProps {
  /** Accessible headline — what the chart shows AND its key number. */
  label: string;
  xDomain: Range;
  yDomain: Range;
  xLabel?: string;
  yLabel?: string;
  width?: number;
  height?: number;
  margins?: Margins;
  xTicks?: number[];
  yTicks?: number[];
  /** Suppress the horizontal gridlines where a chart carries its own reference
   *  geometry and the rules would just add noise. */
  grid?: boolean;
  children: (s: {
    sx: (x: number) => number;
    sy: (y: number) => number;
    w: number;
    h: number;
    m: Margins;
  }) => ReactNode;
  defs?: ReactNode;
  /** Screen-reader table of the underlying values. A ROC curve read aloud is
   *  meaningless; "AUROC 0.712, 95% CI 0.686–0.739" is not. */
  table?: ReactNode;
}

/**
 * The shared plot skeleton: viewBox, margins, gridlines, ticks, axis titles.
 * Geometry follows backend/ml/scripts/build_report.py:105-136.
 *
 * Server component — the scale math and path strings are computed at build time
 * and only the finished `d` strings reach the client.
 */
export function ChartFrame({
  label,
  xDomain,
  yDomain,
  xLabel,
  yLabel,
  width = 460,
  height = 300,
  margins = M,
  xTicks,
  yTicks,
  grid = true,
  children,
  defs,
  table,
}: ChartFrameProps) {
  const m = margins;
  const sx = scaleX(xDomain, width, m);
  const sy = scaleY(yDomain, height, m);
  const xt = xTicks ?? ticks(xDomain);
  const yt = yTicks ?? ticks(yDomain);
  const midY = (m.t + height - m.b) / 2;

  return (
    <figure className="m-0">
      {/* The IntersectionObserver has to live on this div. Watching the SVG
          children directly (whileInView) never fires — see chart-reveal.tsx. */}
      <ChartReveal>
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={label}
        className="block w-full"
        style={{ overflow: "visible" }}
      >
        {defs && <defs>{defs}</defs>}

        {grid &&
          yt.map((v) => (
            <line
              key={`g${v}`}
              x1={m.l}
              y1={sy(v)}
              x2={width - m.r}
              y2={sy(v)}
              stroke="var(--color-line-soft)"
              strokeWidth={1}
            />
          ))}

        {yt.map((v) => (
          <text
            key={`y${v}`}
            x={m.l - 8}
            y={sy(v) + 3}
            textAnchor="end"
            className="fill-faint font-mono"
            fontSize={9.5}
          >
            {fmtTick(v)}
          </text>
        ))}

        {xt.map((v) => (
          <text
            key={`x${v}`}
            x={sx(v)}
            y={height - m.b + 16}
            textAnchor="middle"
            className="fill-faint font-mono"
            fontSize={9.5}
          >
            {fmtTick(v)}
          </text>
        ))}

        <line
          x1={m.l}
          y1={height - m.b}
          x2={width - m.r}
          y2={height - m.b}
          stroke="var(--color-line)"
          strokeWidth={1}
        />

        {xLabel && (
          <text
            x={(m.l + width - m.r) / 2}
            y={height - 2}
            textAnchor="middle"
            className="fill-dim font-mono"
            fontSize={9.5}
          >
            {xLabel}
          </text>
        )}
        {yLabel && (
          <text
            x={11}
            y={midY}
            textAnchor="middle"
            transform={`rotate(-90 11 ${midY})`}
            className="fill-dim font-mono"
            fontSize={9.5}
          >
            {yLabel}
          </text>
        )}

        {children({ sx, sy, w: width, h: height, m })}
      </svg>
      </ChartReveal>
      {table && <div className="sr-only">{table}</div>}
    </figure>
  );
}

/** A dashed reference line (chance diagonal, perfect calibration, treat-all). */
export function RefLine({ d }: { d: string }) {
  return (
    <path
      d={d}
      fill="none"
      stroke="var(--color-faint)"
      strokeWidth={1.2}
      strokeDasharray="4 4"
    />
  );
}

/**
 * A vertical gradient that fades a series colour into nothing — the fill under
 * a curve. Rule 2 forbids decorative gradients on panels; this one is not
 * decoration, it is how a curve communicates area (AUROC literally *is* the area
 * under the curve, so shading it is the honest reading of the chart).
 */
export function AreaGradient({ id, color }: { id: string; color: string }) {
  return (
    <linearGradient id={id} x1="0" y1="0" x2="0" y2="1">
      <stop offset="0%" stopColor={color} stopOpacity={0.28} />
      <stop offset="100%" stopColor={color} stopOpacity={0} />
    </linearGradient>
  );
}

/** A small callout that names a point on the plot, with a leader line. */
export function Annotation({
  x,
  y,
  dx = 10,
  dy = -14,
  title,
  detail,
  anchor = "start",
  color = "var(--color-ink)",
}: {
  x: number;
  y: number;
  dx?: number;
  dy?: number;
  title: string;
  detail?: string;
  anchor?: "start" | "end" | "middle";
  color?: string;
}) {
  return (
    <g>
      <line
        x1={x}
        y1={y}
        x2={x + dx}
        y2={y + dy}
        stroke="var(--color-line-bright)"
        strokeWidth={1}
      />
      <circle cx={x} cy={y} r={3} fill={color} stroke="var(--color-bg)" strokeWidth={1.5} />
      <text
        x={x + dx + (anchor === "end" ? -4 : 4)}
        y={y + dy}
        textAnchor={anchor}
        className="fill-ink font-mono"
        fontSize={10}
        fontWeight={500}
      >
        {title}
      </text>
      {detail && (
        <text
          x={x + dx + (anchor === "end" ? -4 : 4)}
          y={y + dy + 12}
          textAnchor={anchor}
          className="fill-dim font-mono"
          fontSize={9}
        >
          {detail}
        </text>
      )}
    </g>
  );
}

/** Direct series label, placed on the curve itself instead of in a legend. */
export function SeriesLabel({
  x,
  y,
  text: t,
  color,
  anchor = "start",
}: {
  x: number;
  y: number;
  text: string;
  color: string;
  anchor?: "start" | "end" | "middle";
}) {
  return (
    <text
      x={x}
      y={y}
      textAnchor={anchor}
      fill={color}
      className="font-mono"
      fontSize={10}
      fontWeight={600}
    >
      {t}
    </text>
  );
}
