"use client";

import { motion } from "motion/react";

import { f3 } from "@/lib/format";

export interface ForestRow {
  label: string;
  sub?: string;
  value: number;
  ci: number[];
  /** The production model gets the accent; everything else stays neutral. */
  emphasis?: boolean;
}

/**
 * AUROC with 95% bootstrap confidence intervals, one row per (direction ×
 * model). Geometry ported from backend/ml/scripts/build_report.py:145-161.
 *
 * The whiskers are the entire point of leading with this chart: four point
 * estimates in a table look like four facts, whereas four intervals show how
 * much of the difference between them is real.
 */
export function ForestPlot({
  rows,
  domain = [0.5, 0.8],
}: {
  rows: ForestRow[];
  domain?: [number, number];
}) {
  const W = 620;
  const rh = 40;
  const H = rows.length * rh + 34;
  const m = { l: 168, r: 176 };
  const sx = (x: number) =>
    m.l + ((x - domain[0]) / (domain[1] - domain[0])) * (W - m.l - m.r);

  const gridVals = [0.5, 0.6, 0.7, 0.8].filter((v) => v >= domain[0] && v <= domain[1]);

  return (
    <figure className="m-0">
      <svg
        viewBox={`0 0 ${W} ${H}`}
        role="img"
        aria-label={`Cross-hospital AUROC with 95% confidence intervals: ${rows
          .map((r) => `${r.label} ${f3(r.value)} (${f3(r.ci[0])} to ${f3(r.ci[1])})`)
          .join("; ")}`}
        className="block w-full"
      >
        {gridVals.map((v) => (
          <g key={v}>
            <line
              x1={sx(v)}
              y1={22}
              x2={sx(v)}
              y2={H - 8}
              stroke="var(--color-line-soft)"
              strokeWidth={1}
            />
            <text
              x={sx(v)}
              y={14}
              textAnchor="middle"
              className="fill-faint font-mono"
              fontSize={9.5}
            >
              {v.toFixed(1)}
            </text>
          </g>
        ))}

        {/* chance */}
        <line
          x1={sx(0.5)}
          y1={22}
          x2={sx(0.5)}
          y2={H - 8}
          stroke="var(--color-faint)"
          strokeWidth={1.2}
          strokeDasharray="3 3"
        />

        {rows.map((r, i) => {
          const y = 34 + i * rh;
          const colour = r.emphasis ? "var(--color-accent)" : "var(--color-dim)";
          return (
            <motion.g
              key={r.label + i}
              initial={{ opacity: 0, x: -6 }}
              whileInView={{ opacity: 1, x: 0 }}
              viewport={{ once: true, amount: 0.4 }}
              transition={{ duration: 0.24, ease: [0.2, 0.7, 0.2, 1], delay: i * 0.06 }}
            >
              <text
                x={0}
                y={y + 1}
                className={r.emphasis ? "fill-ink" : "fill-muted"}
                fontSize={12}
                fontWeight={r.emphasis ? 600 : 400}
              >
                {r.label}
              </text>
              {r.sub && (
                <text x={0} y={y + 14} className="fill-faint font-mono" fontSize={9.5}>
                  {r.sub}
                </text>
              )}

              {/* interval */}
              <line x1={sx(r.ci[0])} y1={y} x2={sx(r.ci[1])} y2={y} stroke={colour} strokeWidth={2} />
              <line x1={sx(r.ci[0])} y1={y - 4} x2={sx(r.ci[0])} y2={y + 4} stroke={colour} strokeWidth={2} />
              <line x1={sx(r.ci[1])} y1={y - 4} x2={sx(r.ci[1])} y2={y + 4} stroke={colour} strokeWidth={2} />
              <circle
                cx={sx(r.value)}
                cy={y}
                r={r.emphasis ? 5 : 4}
                fill={colour}
                stroke="var(--color-s1)"
                strokeWidth={1.5}
              />

              <text
                x={W}
                y={y + 4}
                textAnchor="end"
                className="fill-muted font-mono"
                fontSize={11}
                style={{ fontVariantNumeric: "tabular-nums" }}
              >
                {f3(r.value)} [{f3(r.ci[0])}–{f3(r.ci[1])}]
              </text>
            </motion.g>
          );
        })}
      </svg>
    </figure>
  );
}
