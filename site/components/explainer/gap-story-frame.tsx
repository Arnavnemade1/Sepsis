"use client";

import dynamic from "next/dynamic";
import { useReducedMotion } from "motion/react";

import { GAP_W, GAP_H } from "@/remotion/GapStory";

/**
 * Client boundary for the generalization-gap composition. Same two gates as the
 * main explainer: lazy so ~119 KB of Remotion never touches first paint, and
 * never mounted at all under reduced motion.
 */
const GapStoryPlayer = dynamic(() => import("./gap-story-player"), {
  ssr: false,
  loading: () => <Placeholder label="Loading…" />,
});

export function GapStoryFrame() {
  const reduced = useReducedMotion();
  if (reduced) return <Placeholder label="Animation disabled — reduced motion is on" />;
  return <GapStoryPlayer />;
}

function Placeholder({ label }: { label: string }) {
  return (
    <div
      className="flex items-center justify-center rounded-lg border border-line bg-s1/62 backdrop-blur-xl"
      style={{ aspectRatio: `${GAP_W} / ${GAP_H}` }}
    >
      <span className="font-mono text-[11px] tracking-[0.08em] text-faint uppercase">
        {label}
      </span>
    </div>
  );
}
