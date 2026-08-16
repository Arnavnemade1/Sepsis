"use client";

import { motion } from "motion/react";

/**
 * A curve that draws itself when it scrolls into view, like a plotter laying
 * down ink. This is the concrete reason the charts are hand-rolled SVG rather
 * than a chart library — animating `pathLength` means reaching past a library's
 * API into its internal path refs, whereas here it is one prop.
 *
 * Only the finished `d` string crosses the server/client boundary. The scale
 * math and the underlying arrays stay on the server.
 *
 * Under prefers-reduced-motion, MotionConfig reducedMotion="user" (set in
 * MotionProvider) drops the transform animation and the path simply appears.
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
      whileInView={{ pathLength: 1, opacity: 1 }}
      viewport={{ once: true, amount: 0.3 }}
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
  return (
    <motion.path
      d={d}
      fill={fill}
      stroke="none"
      initial={{ opacity: 0 }}
      whileInView={{ opacity: 1 }}
      viewport={{ once: true, amount: 0.3 }}
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
          whileInView={{ opacity: 1, scale: 1 }}
          viewport={{ once: true, amount: 0.3 }}
          transition={{ duration: 0.2, ease: [0.2, 0.7, 0.2, 1], delay: delay + i * 0.03 }}
        />
      ))}
    </>
  );
}
