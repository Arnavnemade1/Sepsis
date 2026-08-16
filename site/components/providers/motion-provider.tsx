"use client";

import { MotionConfig } from "motion/react";
import type { ReactNode } from "react";

/**
 * Global motion contract, set once.
 *
 * The identity says motion is "quick and mechanical; nothing bounces"
 * (webui/os.css:79), so the default transition is the instrument easing curve
 * --ease at --t-slow. NO SPRINGS anywhere in this codebase: a spring overshoots,
 * and an overshooting instrument reads as a toy.
 *
 * reducedMotion="user" makes Motion automatically skip transform and layout
 * animations while keeping opacity fades, for anyone who has asked their OS to
 * reduce motion. That covers almost everything; the three places it can't reach
 * (mounting the hero video, mounting the Remotion Player, running number
 * counters) call useReducedMotion() explicitly.
 */
export function MotionProvider({ children }: { children: ReactNode }) {
  return (
    <MotionConfig
      reducedMotion="user"
      transition={{ duration: 0.24, ease: [0.2, 0.7, 0.2, 1] }}
    >
      {children}
    </MotionConfig>
  );
}
