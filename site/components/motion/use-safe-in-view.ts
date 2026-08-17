"use client";

import { useInView } from "motion/react";
import { useEffect, useState, type RefObject } from "react";

/**
 * `useInView`, but content can never end up permanently invisible.
 *
 * Every reveal on this site animates from `opacity: 0`, which means an
 * IntersectionObserver that does not fire is not a missing animation — it is a
 * blank page. That has already happened once for real: the charts used
 * `whileInView` on SVG children, which IntersectionObserver does not reliably
 * report, and every curve stayed invisible forever.
 *
 * So the observer is treated as an optimisation, not a source of truth. If an
 * element's box overlaps the viewport but the observer still has not reported
 * it, this reveals it anyway — measured with getBoundingClientRect, which does
 * not depend on the observer working at all.
 *
 * Off-screen elements are deliberately NOT rescued, so the scroll-triggered
 * effect survives intact for every reader whose observer behaves normally.
 */
export function useSafeInView(
  ref: RefObject<HTMLElement | null>,
  { amount = 0.2, margin = "0px 0px -8% 0px" }: { amount?: number; margin?: string } = {},
) {
  const observed = useInView(ref, {
    once: true,
    amount,
    margin: margin as `${number}px ${number}px ${number}% ${number}px`,
  });
  const [rescued, setRescued] = useState(false);

  useEffect(() => {
    if (observed || rescued) return;

    const check = () => {
      const el = ref.current;
      if (!el) return;
      const r = el.getBoundingClientRect();
      if (r.height > 0 && r.top < window.innerHeight && r.bottom > 0) setRescued(true);
    };

    // One delayed check catches "already on screen at load"; the scroll
    // listener catches everything after that.
    const t = setTimeout(check, 900);
    window.addEventListener("scroll", check, { passive: true });
    window.addEventListener("resize", check, { passive: true });
    return () => {
      clearTimeout(t);
      window.removeEventListener("scroll", check);
      window.removeEventListener("resize", check);
    };
  }, [ref, observed, rescued]);

  return observed || rescued;
}
