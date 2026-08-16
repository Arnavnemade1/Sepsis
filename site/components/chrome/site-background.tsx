"use client";

import { usePathname } from "next/navigation";
import { useEffect, useRef, useState } from "react";
import { motion, useReducedMotion, useScroll, useTransform } from "motion/react";

/**
 * The site-wide background: ONE fixed video behind every page.
 *
 * `position: fixed` means it never scrolls, which gives parallax for free and —
 * more importantly — means there is exactly one <video> element for the whole
 * site rather than one per route. Route changes do not restart it.
 *
 * Legibility is handled by a two-part scrim:
 *
 *   · a base ramp that is always present, dark at the very top (behind the nav)
 *     and at the very bottom;
 *   · a scroll-linked veil that thickens once you leave the hero, so the footage
 *     is a clear presence on the landing screen and a texture underneath dense
 *     content rather than a competitor to it.
 *
 * On routes other than "/" there is no hero, so the veil starts already closed.
 *
 * The same three gates as before still apply: reduced motion, narrow viewports
 * and Save-Data never mount the video at all and get the graphite ground only.
 */
export function SiteBackground() {
  const pathname = usePathname();
  /** Only "/" has a hero to reveal; elsewhere the veil starts closed. */
  const dimmed = pathname !== "/";
  const reduced = useReducedMotion();
  const [mounted, setMounted] = useState(false);
  const [playing, setPlaying] = useState(false);
  const vid = useRef<HTMLVideoElement>(null);

  const { scrollY } = useScroll();
  // Thicken the veil over the first screenful. On a sub-page it is already at
  // full strength, because there is no hero to reveal.
  const veil = useTransform(scrollY, [0, 620], dimmed ? [0.82, 0.82] : [0.12, 0.82], {
    clamp: true,
  });
  // A touch of scale so the footage breathes as you scroll rather than sitting
  // dead still behind moving content.
  const scale = useTransform(scrollY, [0, 1400], [1, 1.08], { clamp: true });

  useEffect(() => {
    if (reduced) return;
    const conn = (
      navigator as Navigator & {
        connection?: { saveData?: boolean; effectiveType?: string };
      }
    ).connection;
    const tooSlow =
      conn?.saveData === true || /^(slow-2g|2g)$/.test(conn?.effectiveType ?? "");
    if (tooSlow) return;

    const mq = window.matchMedia("(min-width: 768px)");
    const sync = () => mq.matches && setMounted(true);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, [reduced]);

  // Pause on a hidden tab. There is no IntersectionObserver here — a fixed
  // background is always "in view" for as long as the page is open.
  useEffect(() => {
    const v = vid.current;
    if (!v) return;
    const onVis = () => {
      if (document.hidden) v.pause();
      else void v.play().catch(() => {});
    };
    document.addEventListener("visibilitychange", onVis);
    return () => document.removeEventListener("visibilitychange", onVis);
  }, [mounted]);

  return (
    <div
      aria-hidden="true"
      className="pointer-events-none fixed inset-0 -z-10 overflow-hidden bg-bg"
    >
      <GeneratedGround />

      {mounted && (
        <motion.video
          ref={vid}
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          disablePictureInPicture
          tabIndex={-1}
          onPlaying={() => setPlaying(true)}
          style={{
            scale,
            // Graded, not decorated: --low is a saturated green, so ungraded
            // footage would put a clinical state colour behind every page.
            filter: "saturate(.28) contrast(1.1) brightness(.66)",
            opacity: playing ? 1 : 0,
          }}
          className="absolute inset-0 h-full w-full object-cover transition-opacity duration-[900ms] ease-instrument"
        >
          <source src="/media/hero.mp4" type="video/mp4" />
        </motion.video>
      )}

      {/* base ramp — dark behind the nav and at the foot of the screen */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(180deg, rgba(10,11,12,.86) 0%, rgba(10,11,12,.30) 18%, rgba(10,11,12,.30) 62%, rgba(10,11,12,.78) 100%)",
        }}
      />
      {/* left weighting, so copy has contrast without flattening the frame */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "linear-gradient(90deg, rgba(10,11,12,.58) 0%, rgba(10,11,12,.22) 48%, transparent 78%)",
        }}
      />
      {/* scroll-linked veil */}
      <motion.div className="absolute inset-0 bg-bg" style={{ opacity: veil }} />
    </div>
  );
}

/**
 * The always-present ground beneath the video: instant, zero bytes, and the
 * complete background for anyone who never gets the footage.
 */
function GeneratedGround() {
  return (
    <>
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(130% 85% at 50% 0%, #191c20 0%, #101215 42%, #0a0b0c 100%)",
        }}
      />
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(55% 40% at 20% 12%, rgba(122,162,247,.09) 0%, transparent 72%)",
        }}
      />
    </>
  );
}
