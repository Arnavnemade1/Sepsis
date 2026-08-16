import type { ReactNode } from "react";

import { M, scaleX, scaleY, ticks, type Range } from "@/lib/scales";
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
  xTicks?: number[];
  yTicks?: number[];
  /** Rendered inside the plot area, already in screen space. */
  children: (s: {
    sx: (x: number) => number;
    sy: (y: number) => number;
    w: number;
    h: number;
  }) => ReactNode;
  /** Screen-reader table of the underlying values. A ROC curve read aloud is
   *  meaningless; "AUROC 0.712, 95% CI 0.686–0.739" is not. */
  table?: ReactNode;
}

/**
 * The shared plot skeleton: viewBox, margins, gridlines, ticks, axis titles.
 * Geometry matches backend/ml/scripts/build_report.py:105-136.
 *
 * Server component — the scale math and path strings are computed at build time
 * and only the finished `d` strings reach the client (see DrawPath).
 */
export function ChartFrame({
  label,
  xDomain,
  yDomain,
  xLabel,
  yLabel,
  width = 440,
  height = 300,
  xTicks,
  yTicks,
  children,
  table,
}: ChartFrameProps) {
  const sx = scaleX(xDomain, width);
  const sy = scaleY(yDomain, height);
  const xt = xTicks ?? ticks(xDomain);
  const yt = yTicks ?? ticks(yDomain);
  const midY = (M.t + height - M.b) / 2;

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        role="img"
        aria-label={label}
        className="block w-full"
        style={{ overflow: "visible" }}
      >
        {/* gridlines — hairlines, Rule 2 */}
        {yt.map((v) => (
          <g key={`y${v}`}>
            <line
              x1={M.l}
              y1={sy(v)}
              x2={width - M.r}
              y2={sy(v)}
              stroke="var(--color-line-soft)"
              strokeWidth={1}
            />
            <text
              x={M.l - 7}
              y={sy(v) + 3}
              textAnchor="end"
              className="fill-faint font-mono"
              fontSize={9.5}
            >
              {fmtTick(v)}
            </text>
          </g>
        ))}

        {xt.map((v) => (
          <text
            key={`x${v}`}
            x={sx(v)}
            y={height - M.b + 15}
            textAnchor="middle"
            className="fill-faint font-mono"
            fontSize={9.5}
          >
            {fmtTick(v)}
          </text>
        ))}

        {/* axis rules */}
        <line
          x1={M.l}
          y1={height - M.b}
          x2={width - M.r}
          y2={height - M.b}
          stroke="var(--color-line)"
          strokeWidth={1}
        />

        {xLabel && (
          <text
            x={(M.l + width - M.r) / 2}
            y={height - 3}
            textAnchor="middle"
            className="fill-dim font-mono"
            fontSize={10}
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
            fontSize={10}
          >
            {yLabel}
          </text>
        )}

        {children({ sx, sy, w: width, h: height })}
      </svg>
      {table && <div className="sr-only">{table}</div>}
    </figure>
  );
}

/** A dashed reference line (chance diagonal, perfect calibration, treat-all). */
export function RefLine({
  d,
  label,
  x,
  y,
}: {
  d: string;
  label?: string;
  x?: number;
  y?: number;
}) {
  return (
    <>
      <path
        d={d}
        fill="none"
        stroke="var(--color-faint)"
        strokeWidth={1.2}
        strokeDasharray="4 4"
      />
      {label && x !== undefined && y !== undefined && (
        <text
          x={x}
          y={y}
          textAnchor="end"
          className="fill-faint font-mono"
          fontSize={9.5}
        >
          {label}
        </text>
      )}
    </>
  );
}
