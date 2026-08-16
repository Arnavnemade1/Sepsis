"use client";

import { Player, type PlayerRef } from "@remotion/player";
import { useEffect, useRef } from "react";

import { GapStory, GAP_FPS, GAP_W, GAP_H, GAP_DURATION } from "@/remotion/GapStory";
import { generalization } from "@/lib/results";

export default function GapStoryPlayer() {
  const player = useRef<PlayerRef>(null);
  const box = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const p = player.current;
    const el = box.current;
    if (!p || !el) return;
    const io = new IntersectionObserver(
      ([e]) => (e.isIntersecting ? p.play() : p.pause()),
      { threshold: 0.25 },
    );
    io.observe(el);
    const onVis = () => document.hidden && p.pause();
    document.addEventListener("visibilitychange", onVis);
    return () => {
      io.disconnect();
      document.removeEventListener("visibilitychange", onVis);
    };
  }, []);

  return (
    <div
      ref={box}
      className="overflow-hidden rounded-lg border border-line bg-s1/62 backdrop-blur-xl"
      style={{ aspectRatio: `${GAP_W} / ${GAP_H}` }}
    >
      <Player
        ref={player}
        component={GapStory}
        inputProps={{
          internal: generalization.internalAuroc,
          external: generalization.externalAuroc,
          gap: generalization.gap,
        }}
        durationInFrames={GAP_DURATION}
        fps={GAP_FPS}
        compositionWidth={GAP_W}
        compositionHeight={GAP_H}
        style={{ width: "100%", height: "100%" }}
        loop
        autoPlay
        initiallyMuted
        controls={false}
        clickToPlay={false}
        spaceKeyToPlayOrPause={false}
        allowFullscreen={false}
        doubleClickToFullscreen={false}
        acknowledgeRemotionLicense
      />
    </div>
  );
}
