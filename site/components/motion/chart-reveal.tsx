"use client";

import { createContext, useContext, useRef, type ReactNode } from "react";

import { useSafeInView } from "./use-safe-in-view";

/**
 * Drives the draw-on animations inside a chart.
 *
 * WHY THIS EXISTS: the chart internals used `whileInView` directly on
 * `motion.path` / `motion.g` / `motion.circle`, and it never fired. Motion
 * implements `whileInView` with an IntersectionObserver on the element itself,
 * and IntersectionObserver does not reliably report SVG child elements — they
 * have a bounding box but not a CSS layout box. The result was silent and
 * total: every curve, area and forest row sat at `opacity: 0` forever, so the
 * charts rendered as empty axes.
 *
 * The page's other reveals were fine throughout, which is the tell — those wrap
 * `motion.div`, where the observer works normally.
 *
 * So the observer goes on a plain wrapping <div> and the result is handed down
 * by context. SVG children then animate off a boolean instead of observing
 * themselves.
 *
 * The default is `true`, deliberately. If a chart is ever rendered outside a
 * provider, it must appear immediately — an un-animated chart is a small loss,
 * an invisible one is a broken page.
 */
const InViewContext = createContext(true);

export function useChartInView() {
  return useContext(InViewContext);
}

export function ChartReveal({
  children,
  className,
}: {
  children: ReactNode;
  className?: string;
}) {
  const ref = useRef<HTMLDivElement>(null);
  // amount 0.15 rather than 0.3: these figures are tall, and requiring a third
  // of one on screen delayed the draw until the reader had scrolled past its top.
  const inView = useSafeInView(ref, { amount: 0.15 });

  return (
    <div ref={ref} className={className}>
      <InViewContext.Provider value={inView}>{children}</InViewContext.Provider>
    </div>
  );
}
