"use client";

import { useInView } from "motion/react";
import { createContext, useContext, useEffect, useRef, useState, type ReactNode } from "react";

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
  const inView = useInView(ref, { once: true, amount: 0.15, margin: "0px 0px -8% 0px" });
  const rescued = useFallbackReveal(ref, inView);

  return (
    <div ref={ref} className={className}>
      <InViewContext.Provider value={inView || rescued}>{children}</InViewContext.Provider>
    </div>
  );
}

/**
 * Safety net for a chart that is plainly on screen but whose observer never
 * reported it.
 *
 * An invisible chart is a broken page, and this exact failure has now happened
 * once for real (SVG children are not reliably observed). So rather than trust
 * the observer alone, this measures the element directly a short while after
 * mount: if it overlaps the viewport and `inView` is still false, reveal it.
 *
 * It deliberately does NOT reveal off-screen charts, so the draw-on effect
 * survives for every reader whose observer works normally.
 */
function useFallbackReveal(ref: React.RefObject<HTMLElement | null>, inView: boolean) {
  const [rescued, setRescued] = useState(false);

  useEffect(() => {
    if (inView || rescued) return;
    const check = () => {
      const el = ref.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      const onScreen = r.top < window.innerHeight && r.bottom > 0 && r.height > 0;
      if (onScreen) setRescued(true);
    };
    const t = setTimeout(check, 900);
    window.addEventListener("scroll", check, { passive: true });
    return () => {
      clearTimeout(t);
      window.removeEventListener("scroll", check);
    };
  }, [ref, inView, rescued]);

  return rescued;
}
