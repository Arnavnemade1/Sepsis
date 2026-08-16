"use client";

import { motion } from "motion/react";
import type { ReactNode } from "react";

/**
 * Per-navigation enter transition.
 *
 * template.tsx (unlike layout.tsx) creates a NEW instance on every navigation,
 * so `initial` fires each time — exactly the mount hook a page transition needs.
 *
 * Two things worth knowing:
 *
 * 1. "use client" here does NOT make the pages client-rendered. `children`
 *    arrives as an already-rendered RSC slot, so app/research/page.tsx stays a
 *    Server Component and keeps its static rendering.
 *
 * 2. There is deliberately NO exit animation. App Router unmounts the outgoing
 *    segment before the incoming one commits, so AnimatePresence has no window
 *    in which to run one — this is structural, not a tuning problem. The
 *    circulating FrozenRouter workaround reaches into Next internals, breaks
 *    scroll restoration and prefetching, and is unpredictable under React 19's
 *    concurrent renderer. Enter-only is also the better feel: an exit animation
 *    would put a mandatory dead delay between every click and any response.
 */
export default function Template({ children }: { children: ReactNode }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 4 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.18, ease: [0.2, 0.7, 0.2, 1] }}
    >
      {children}
    </motion.div>
  );
}
