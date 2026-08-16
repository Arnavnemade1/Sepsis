"use client";

import { Player, type PlayerRef } from "@remotion/player";
import { useEffect, useRef } from "react";

import { Explainer } from "@/remotion/Explainer";
import { FPS, W, H, DURATION } from "@/remotion/constants";
import { headline, operatingPoint } from "@/lib/results";

/**
 * The embedded explainer.
 *
 * Why the Player rather than a rendered MP4: this design is built from 1px
 * hairlines and 11px tabular mono figures, and any sane video bitrate turns
 * those into mush and rings the digits. The Player renders real DOM and SVG at
 * device pixel ratio, so hairlines stay hairlines on a 2× display. It also reads
 * its numbers from the same data/results.json as the research page, so the video
 * cannot drift out of sync with the evaluation.
 */
export default function ExplainerPlayer() {
  const player = useRef<PlayerRef>(null);
  const box = useRef<HTMLDivElement>(null);

  // The Player re-renders React every frame. Leaving it running behind the fold
  // or in a hidden tab is a battery tax for nothing.
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
      className="overflow-hidden rounded-lg border border-line bg-s1"
      style={{ aspectRatio: `${W} / ${H}` }}
    >
      <Player
        ref={player}
        component={Explainer}
        inputProps={{
          auroc: headline.production.auroc,
          ece: headline.production.ece,
          leadHours: headline.production.medianLeadHours,
          threshold: operatingPoint.high,
        }}
        durationInFrames={DURATION}
        fps={FPS}
        compositionWidth={W}
        compositionHeight={H}
        // inline: the Player's own styles take precedence over Tailwind classes
        style={{ width: "100%", height: "100%" }}
        loop
        autoPlay
        initiallyMuted
        controls={false}
        clickToPlay={false}
        // Defaults to TRUE. Left on, pressing space anywhere on the page would
        // toggle this background animation instead of scrolling.
        spaceKeyToPlayOrPause={false}
        allowFullscreen={false}
        doubleClickToFullscreen={false}
        acknowledgeRemotionLicense
      />
    </div>
  );
}
