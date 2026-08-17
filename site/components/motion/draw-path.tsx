"use client";

import { motion } from "motion/react";

import { useChartInView } from "./chart-reveal";

/**
 * A curve that draws itself like a plotter laying down ink. This is the concrete
 * reason the charts are hand-rolled SVG rather than a chart library — animating
 * `pathLength` means reaching past a library's API into its internal path refs,
 * whereas here it is one prop.
 *
 * These animate off `animate={...}` driven by a boolean from ChartReveal, NOT
 * `whileInView`. Motion implements `whileInView` with an IntersectionObserver on
 * the element itself, and IntersectionObserver does not reliably report SVG
 * child elements — so `whileInView` here left every curve permanently invisible.
 * See components/motion/chart-reveal.tsx.
 *
 * Only the finished `d` string crosses the server/client boundary; the scale
 * math and the underlying arrays stay on the server.
 */
export function DrawPath({
  d,
  stroke = "var(--color-accent)",
  width = 2,
  dash,
  delay = 0,
  duration = 0.9,
}: {
  d: string;
  stroke?: string;
  width?: number;
  dash?: string;
  delay?: number;
  duration?: number;
}) {
  const on = useChartInView();
  return (
    <motion.path
      d={d}
      fill="none"
      stroke={stroke}
      strokeWidth={width}
      strokeLinejoin="round"
      strokeLinecap="round"
      strokeDasharray={dash}
      initial={{ pathLength: 0, opacity: 0 }}
      animate={on ? { pathLength: 1, opacity: 1 } : { pathLength: 0, opacity: 0 }}
      transition={{
        pathLength: { duration, ease: [0.2, 0.7, 0.2, 1], delay },
        opacity: { duration: 0.12, delay },
      }}
    />
  );
}

/**
 * The shaded region under a curve. Fades and rises rather than "drawing",
 * because an area is a quantity, not a stroke — AUROC *is* this area, so it
 * should arrive as a mass.
 */
export function DrawArea({
  d,
  fill,
  delay = 0,
}: {
  d: string;
  fill: string;
  delay?: number;
}) {
  const on = useChartInView();
  return (
    <motion.path
      d={d}
      fill={fill}
      stroke="none"
      initial={{ opacity: 0 }}
      animate={on ? { opacity: 1 } : { opacity: 0 }}
      transition={{ duration: 0.7, ease: [0.2, 0.7, 0.2, 1], delay: delay + 0.25 }}
    />
  );
}

/** Scatter points that fade in after their curve has drawn. */
export function DrawDots({
  pts,
  fill = "var(--color-accent)",
  r = 3.2,
  delay = 0.35,
}: {
  pts: readonly (readonly [number, number])[];
  fill?: string;
  r?: number;
  delay?: number;
}) {
  const on = useChartInView();
  return (
    <>
      {pts.map(([cx, cy], i) => (
        <motion.circle
          key={i}
          cx={cx}
          cy={cy}
          r={r}
          fill={fill}
          stroke="var(--color-s1)"
          strokeWidth={1.5}
          initial={{ opacity: 0, scale: 0.6 }}
          animate={on ? { opacity: 1, scale: 1 } : { opacity: 0, scale: 0.6 }}
          transition={{ duration: 0.2, ease: [0.2, 0.7, 0.2, 1], delay: delay + i * 0.03 }}
        />
      ))}
    </>
  );
}
