"use client";

import { useEffect, useRef, useState } from "react";
import { motion, useReducedMotion, useScroll, useTransform, useMotionValueEvent } from "motion/react";

export interface Layer {
  n: string;
  name: string;
  body: string;
  status: "validated" | "verified" | "machinery" | "gated";
}

const CHIP: Record<Layer["status"], { cls: string; text: string }> = {
  validated: { cls: "border-low/35 bg-low/10 text-low-ink", text: "Validated on real data" },
  verified: { cls: "border-low/35 bg-low/10 text-low-ink", text: "Formally verified" },
  machinery: { cls: "border-med/35 bg-med/10 text-med-ink", text: "Machinery only" },
  gated: { cls: "border-line bg-s2 text-dim", text: "Gated on credentialed data" },
};

/**
 * Scroll-scrubbed product display.
 *
 * The clip is an exploded stack of layers with signal rising through it and
 * finally erupting — so scroll position drives `currentTime` directly, and the
 * eruption lands on the last layer, "certified egress". The picture and the
 * argument advance together instead of the video looping beside the text.
 *
 * Two details make this feel smooth rather than janky:
 *
 * 1. `currentTime` is NEVER written from the scroll handler. Scroll fires far
 *    more often than a video can seek, and seeking mid-seek is what produces
 *    the stutter this pattern is notorious for. Instead scroll updates a target,
 *    and a single rAF loop eases the real playhead toward it.
 * 2. Seeks below ~1/50 s are skipped entirely — below the frame interval they
 *    cost a decode and change nothing on screen.
 */
export function LayerScroller({ layers }: { layers: Layer[] }) {
  const outer = useRef<HTMLDivElement>(null);
  const video = useRef<HTMLVideoElement>(null);
  const reduced = useReducedMotion();

  const [active, setActive] = useState(0);
  const [ready, setReady] = useState(false);
  const target = useRef(0);

  const { scrollYProgress } = useScroll({
    target: outer,
    offset: ["start start", "end end"],
  });

  // which layer is being read right now
  const idx = useTransform(scrollYProgress, (p) =>
    Math.min(layers.length - 1, Math.floor(p * layers.length + 0.0001)),
  );
  useMotionValueEvent(idx, "change", (v) => setActive(v));
  useMotionValueEvent(scrollYProgress, "change", (p) => {
    target.current = p;
  });

  // ease the playhead toward the scroll target, once per frame
  useEffect(() => {
    if (reduced) return;
    let raf = 0;
    let current = 0;
    let last = performance.now();

    /** Seconds for the playhead to close ~63% of the gap to the scroll target. */
    const TAU = 0.09;

    const tick = (now: number) => {
      const dt = Math.min(0.25, (now - last) / 1000);
      last = now;

      const v = video.current;
      if (v && v.readyState >= 2 && v.duration) {
        // Time-based smoothing, NOT `gap * k` per frame. A per-frame constant
        // makes convergence depend on frame rate, so the picture trails badly
        // on a slow device or a throttled tab and snaps instantly on a 120Hz
        // one. Easing by elapsed time takes the same wall-clock duration
        // everywhere.
        const gap = target.current - current;
        current += gap * (1 - Math.exp(-dt / TAU));
        if (Math.abs(target.current - current) < 0.0005) current = target.current;

        const t = current * v.duration;
        if (Math.abs(v.currentTime - t) > 0.02) {
          // fastSeek drops the exact-frame guarantee we do not need here
          if (typeof v.fastSeek === "function") v.fastSeek(t);
          else v.currentTime = t;
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

    // A hidden tab throttles rAF to a crawl; resync on return rather than
    // easing through hundreds of stale frames.
    const onVis = () => {
      if (!document.hidden) {
        current = target.current;
        last = performance.now();
      }
    };
    document.addEventListener("visibilitychange", onVis);

    return () => {
      cancelAnimationFrame(raf);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, [reduced]);

  const glow = useTransform(scrollYProgress, [0, 0.55, 1], [0.25, 0.5, 0.85]);

  return (
    <div
      ref={outer}
      // one viewport per layer, plus one to read the last one in
      style={{ height: `${(layers.length + 1) * 62}vh` }}
      className="relative"
    >
      <div className="sticky top-0 flex h-[100svh] items-center">
        <div className="mx-auto grid w-full max-w-6xl items-center gap-10 px-5 sm:px-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,420px)] lg:gap-16">
          {/* ── the argument ─────────────────────────────────────────── */}
          <div className="order-2 lg:order-1">
            <p className="eyebrow">The stack</p>

            <div className="relative mt-5 min-h-[210px]">
              {layers.map((l, i) => (
                <motion.div
                  key={l.n}
                  aria-hidden={i !== active}
                  initial={false}
                  animate={{
                    opacity: i === active ? 1 : 0,
                    y: i === active ? 0 : 10,
                  }}
                  transition={{ duration: 0.28, ease: [0.2, 0.7, 0.2, 1] }}
                  className={`${i === active ? "" : "pointer-events-none"} absolute inset-0`}
                >
                  <div className="flex items-center gap-3">
                    <span className="num font-mono text-[12px] text-faint">{l.n}</span>
                    <span
                      className={`inline-flex items-center rounded-sm border px-2 py-[3px] font-mono text-[10.5px] uppercase tracking-[0.08em] ${CHIP[l.status].cls}`}
                    >
                      {CHIP[l.status].text}
                    </span>
                  </div>
                  <h3 className="mt-4 font-display text-[clamp(1.75rem,3.4vw,2.6rem)] font-semibold leading-[1.06] tracking-[-0.03em] text-ink text-balance">
                    {l.name}
                  </h3>
                  <p className="mt-4 max-w-lg text-[15px] leading-relaxed text-muted text-pretty">
                    {l.body}
                  </p>
                </motion.div>
              ))}
            </div>

            {/* progress rail — eight ticks, the current one lit */}
            <div className="mt-8 flex items-center gap-1.5">
              {layers.map((l, i) => (
                <div key={l.n} className="flex-1">
                  <div
                    className={`h-[2px] rounded-full transition-colors duration-240 ease-instrument ${
                      i === active
                        ? "bg-accent"
                        : i < active
                          ? "bg-line-bright"
                          : "bg-line"
                    }`}
                  />
                </div>
              ))}
            </div>
            <p className="num mt-3 font-mono text-[11px] text-faint">
              {String(active + 1).padStart(2, "0")} / {String(layers.length).padStart(2, "0")}
              <span className="ml-3 hidden sm:inline">scroll to advance</span>
            </p>
          </div>

          {/* ── the object ───────────────────────────────────────────── */}
          <div className="relative order-1 lg:order-2">
            <motion.div
              aria-hidden="true"
              className="pointer-events-none absolute inset-0 -z-10"
              style={{
                opacity: glow,
                background:
                  "radial-gradient(58% 45% at 50% 52%, rgba(122,162,247,.20) 0%, transparent 72%)",
              }}
            />
            <div className="relative mx-auto w-full max-w-[380px] overflow-hidden rounded-xl border border-line bg-s1/40 backdrop-blur-sm">
              <video
                ref={video}
                src="/media/technology.mp4"
                muted
                playsInline
                preload="auto"
                disablePictureInPicture
                tabIndex={-1}
                aria-hidden="true"
                onLoadedData={() => setReady(true)}
                className="block h-auto w-full"
                style={{ opacity: ready ? 1 : 0, transition: "opacity .5s var(--ease-instrument)" }}
              />
              {/* hairline inner edge, so the panel reads as an object not a hole */}
              <div className="pointer-events-none absolute inset-0 rounded-xl ring-1 ring-inset ring-white/5" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Reduced-motion / no-scrub fallback: the same eight layers as a plain list,
 * with a single still frame. Nobody who has asked for less motion should be
 * given a video whose playhead is wired to their scroll wheel.
 */
export function LayerList({ layers }: { layers: Layer[] }) {
  return (
    <div className="overflow-hidden rounded-lg border border-line bg-s1/62 backdrop-blur-xl">
      {layers.map((l, i) => (
        <div
          key={l.n}
          className={`flex flex-col gap-3 p-5 sm:flex-row sm:items-start sm:gap-6 sm:p-6 ${
            i > 0 ? "border-t border-line-soft" : ""
          }`}
        >
          <span className="num shrink-0 font-mono text-[12px] text-faint sm:pt-1">{l.n}</span>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-center gap-3">
              <h3 className="text-[15px] font-medium text-ink">{l.name}</h3>
              <span
                className={`inline-flex items-center rounded-sm border px-2 py-[3px] font-mono text-[10.5px] uppercase tracking-[0.08em] ${CHIP[l.status].cls}`}
              >
                {CHIP[l.status].text}
              </span>
            </div>
            <p className="mt-2 max-w-2xl text-[14px] leading-relaxed text-muted text-pretty">
              {l.body}
            </p>
          </div>
        </div>
      ))}
    </div>
  );
}
