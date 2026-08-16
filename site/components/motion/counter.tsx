"use client";

import {
  animate,
  useInView,
  useMotionValue,
  useMotionValueEvent,
  useReducedMotion,
} from "motion/react";
import { useEffect, useRef, useState } from "react";

/**
 * A number that counts up when it scrolls into view.
 *
 * Two details that are not cosmetic:
 *
 * 1. `minWidth` in ch + tabular figures. Without them, 0.712 counting up from
 *    0.000 changes width on nearly every frame and jitters the whole tile.
 *    That is precisely the failure webui/os.css Rule 3 exists to prevent.
 *
 * 2. useReducedMotion() is checked explicitly. MotionConfig's reducedMotion
 *    only suppresses transform and layout animations — a counting number is
 *    neither, so it would keep animating. Here it snaps straight to the value.
 */
export function Counter({
  to,
  decimals = 3,
  prefix = "",
  suffix = "",
  duration = 0.9,
  className = "",
}: {
  to: number;
  decimals?: number;
  prefix?: string;
  suffix?: string;
  duration?: number;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.6 });
  const reduced = useReducedMotion();
  const mv = useMotionValue(0);
  const final = to.toFixed(decimals);
  const [shown, setShown] = useState(reduced ? final : (0).toFixed(decimals));

  useMotionValueEvent(mv, "change", (v) => setShown(v.toFixed(decimals)));

  useEffect(() => {
    if (!inView) return;
    if (reduced) {
      setShown(final);
      return;
    }
    const controls = animate(mv, to, { duration, ease: [0.2, 0.7, 0.2, 1] });
    return () => controls.stop();
  }, [inView, reduced, to, mv, duration, final]);

  return (
    <span
      ref={ref}
      className={`num ${className}`}
      // reserve the final width up front so nothing reflows while counting
      style={{
        minWidth: `${(prefix + final + suffix).length}ch`,
        display: "inline-block",
      }}
    >
      {prefix}
      {shown}
      {suffix}
    </span>
  );
}
