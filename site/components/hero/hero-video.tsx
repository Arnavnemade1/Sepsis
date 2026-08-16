"use client";

import { useEffect, useRef, useState } from "react";
import { useReducedMotion } from "motion/react";

/**
 * The looping hero background.
 *
 * Four rules this encodes:
 *
 * 1. THE GROUND IS ALWAYS PAINTED, and it is the LCP element. `GeneratedGround`
 *    below is a pure-CSS graphite ramp that renders instantly at zero bytes; the
 *    video sits on top and fades in only once it is actually playing. That makes
 *    a separate poster JPEG unnecessary — the thing a poster exists to prevent
 *    (a blank frame while video loads) cannot happen.
 *
 * 2. THE VIDEO IS CONDITIONALLY MOUNTED, not merely hidden. On a phone, on
 *    Save-Data, or on a slow connection the browser downloads ZERO video bytes.
 *    (`<source media>` is spec'd for this but has been unreliable in Safari; a
 *    mount gate is deterministic.) Bandwidth discipline is an explicit
 *    documented value in this project's DEPLOY.md.
 *
 * 3. THE FOOTAGE IS GRADED TOWARD MONOCHROME. `--low` is a saturated green, so
 *    ungraded nature footage would put that exact hue behind the whole page —
 *    precisely what "colour is signal, never decoration" forbids. Grading to
 *    ~35% saturation keeps it readable as landscape while making it impossible
 *    to mistake for clinical state.
 *
 * 4. IT PAUSES WHEN NOBODY IS WATCHING — off-screen or hidden tab — mirroring
 *    the visibilitychange throttling already in webui/os.js.
 */
export function HeroVideo({
  mp4 = "/media/hero.mp4",
  /** Optional. H.264 already covers every browser, so a WebM is a bytes
   *  optimisation rather than a fallback, and is omitted until encoded. */
  webm,
}: {
  mp4?: string;
  webm?: string;
}) {
  const reduced = useReducedMotion();
  const [mounted, setMounted] = useState(false);
  const [playing, setPlaying] = useState(false);
  const box = useRef<HTMLDivElement>(null);
  const vid = useRef<HTMLVideoElement>(null);

  useEffect(() => {
    if (reduced) return;

    const conn = (
      navigator as Navigator & {
        connection?: { saveData?: boolean; effectiveType?: string };
      }
    ).connection;
    // Block only on Save-Data and genuinely slow links. effectiveType is a
    // coarse RTT/throughput estimate, and plenty of perfectly capable desktop
    // connections report "3g" — treating that as too slow meant most visitors
    // silently never got the video at all.
    const tooSlow =
      conn?.saveData === true || /^(slow-2g|2g)$/.test(conn?.effectiveType ?? "");
    if (tooSlow) return;

    // Listen rather than sample once: a window dragged from phone width to
    // desktop width should pick the video up without a reload. Note this only
    // ever turns the video ON — having paid for the bytes, tearing it down on a
    // narrow resize would waste them.
    const mq = window.matchMedia("(min-width: 768px)");
    const sync = () => mq.matches && setMounted(true);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, [reduced]);

  useEffect(() => {
    const v = vid.current;
    const el = box.current;
    if (!v || !el) return;

    const io = new IntersectionObserver(
      ([e]) => {
        if (e.isIntersecting) void v.play().catch(() => {});
        else v.pause();
      },
      { threshold: 0.05 },
    );
    io.observe(el);

    const onVis = () => {
      if (document.hidden) v.pause();
      else void v.play().catch(() => {});
    };
    document.addEventListener("visibilitychange", onVis);

    return () => {
      io.disconnect();
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [mounted]);

  return (
    <div ref={box} className="absolute inset-0 bg-bg">
      <GeneratedGround />

      {mounted && (
        <video
          ref={vid}
          autoPlay
          muted
          loop
          playsInline
          preload="auto"
          disablePictureInPicture
          aria-hidden="true"
          tabIndex={-1}
          onPlaying={() => setPlaying(true)}
          className="absolute inset-0 h-full w-full object-cover transition-opacity duration-[600ms] ease-instrument"
          style={{
            filter: "saturate(.32) contrast(1.08) brightness(.72)",
            opacity: playing ? 1 : 0,
          }}
        >
          {webm && <source src={webm} type="video/webm" />}
          <source src={mp4} type="video/mp4" />
        </video>
      )}
    </div>
  );
}

/**
 * The always-present ground beneath the video.
 *
 * This is the ONE place in the codebase that paints a gradient. The hero needs a
 * luminance ramp for text contrast, and Rule 2's "no gradient fills" is about
 * decorative depth on panels, not about making type legible over footage. It is
 * achromatic apart from a whisper of the affordance accent, so it never competes
 * with a clinical hue.
 */
function GeneratedGround() {
  return (
    <div className="absolute inset-0 overflow-hidden" aria-hidden="true">
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
    </div>
  );
}
