"use client";

import dynamic from "next/dynamic";
import { useReducedMotion } from "motion/react";

import { W, H } from "@/remotion/constants";

/**
 * Client boundary for the Remotion Player.
 *
 * Two gates, both load-bearing:
 *
 * 1. `dynamic(..., { ssr: false })` — @remotion/player is client-only (its
 *    entrypoint throws a deliberate error if imported from a Server Component),
 *    and together with `remotion` core it is ~119 KB gzipped. Lazy-loading keeps
 *    that entirely off the critical path. Note this file must carry "use client"
 *    itself: App Router rejects `ssr: false` inside a Server Component.
 *
 * 2. `useReducedMotion()` — someone who has asked their OS to reduce motion
 *    should not be served a looping animation, and should not pay 119 KB to be
 *    shown a still. So the Player is never even imported for them.
 */
const ExplainerPlayer = dynamic(() => import("./explainer-player"), {
  ssr: false,
  loading: () => <Placeholder label="Loading…" />,
});

export function ExplainerFrame() {
  const reduced = useReducedMotion();
  if (reduced) return <Placeholder label="Animation disabled — reduced motion is on" />;
  return <ExplainerPlayer />;
}

/** Identical box either way, so nothing shifts when the Player mounts. */
function Placeholder({ label }: { label: string }) {
  return (
    <div
      className="flex items-center justify-center rounded-lg border border-line bg-s1/62 backdrop-blur-xl"
      style={{ aspectRatio: `${W} / ${H}` }}
    >
      <span className="font-mono text-[11px] tracking-[0.08em] text-faint uppercase">
        {label}
      </span>
    </div>
  );
}
