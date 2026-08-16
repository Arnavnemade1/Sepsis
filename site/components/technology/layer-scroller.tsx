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
 * A leader line that stretches across whatever gutter is left and ends in a
 * head at the inner edge, pointing at the stack. Built from a flexing rule plus
 * a fixed triangle rather than a scaled SVG, so the head never distorts as the
 * gutter changes width.
 */
function Leader({ dir }: { dir: "left" | "right" }) {
  const head = (
    <svg width="7" height="8" viewBox="0 0 7 8" className="shrink-0" aria-hidden="true">
      <polygon points={dir === "left" ? "7,4 0,0 0,8" : "0,4 7,0 7,8"} fill="currentColor" />
    </svg>
  );
  return (
    <span className="flex min-w-[28px] flex-1 items-center" aria-hidden="true">
      {dir === "right" && head}
      <span className="h-px flex-1 bg-current" />
      {dir === "left" && head}
    </span>
  );
}

/**
 * Full-bleed scroll-scrubbed layer display.
 *
 * The clip is an exploded stack of layers with signal rising through it and
 * finally erupting, so scroll drives the playhead directly and the eruption
 * lands on the last layer, "certified egress".
 *
 * The video is `object-contain`, not `cover`: it is portrait, and covering a
 * landscape viewport would crop away the top and bottom of the very stack the
 * section is about. Fitting it to the height leaves a gutter either side, which
 * is exactly where the annotations belong — so the constraint and the layout
 * want the same thing.
 *
 * All eight callouts stay on screen, alternating sides, each with an arrow into
 * the stack. Scrolling lights the current one and dims the rest, so the section
 * reads as an annotated diagram even in a still.
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

  const idx = useTransform(scrollYProgress, (p) =>
    Math.min(layers.length - 1, Math.floor(p * layers.length + 0.0001)),
  );
  useMotionValueEvent(idx, "change", (v) => setActive(v));
  useMotionValueEvent(scrollYProgress, "change", (p) => {
    target.current = p;
  });

  useEffect(() => {
    if (reduced) return;
    let raf = 0;
    let current = 0;
    let last = performance.now();

    /** Seconds to close ~63% of the gap to the scroll target. */
    const TAU = 0.09;

    const tick = (now: number) => {
      const dt = Math.min(0.25, (now - last) / 1000);
      last = now;

      const v = video.current;
      if (v && v.readyState >= 2 && v.duration) {
        // Time-based smoothing, NOT `gap * k` per frame. A per-frame constant
        // makes convergence depend on frame rate, so the picture trails badly
        // on a slow device or a throttled tab and snaps on a 120Hz one.
        const gap = target.current - current;
        current += gap * (1 - Math.exp(-dt / TAU));
        if (Math.abs(target.current - current) < 0.0005) current = target.current;

        const t = current * v.duration;
        // currentTime is never written from the scroll handler itself: scroll
        // fires far more often than a video can seek, and seeking mid-seek is
        // what makes this pattern stutter.
        if (Math.abs(v.currentTime - t) > 0.02) {
          if (typeof v.fastSeek === "function") v.fastSeek(t);
          else v.currentTime = t;
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);

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

  const glow = useTransform(scrollYProgress, [0, 0.55, 1], [0.2, 0.45, 0.85]);

  return (
    <div
      ref={outer}
      style={{ height: `${(layers.length + 1) * 62}vh` }}
      className="relative"
    >
      <div
        className="sticky top-0 h-[100svh] overflow-hidden"
        style={
          {
            // The video is object-contain, so its rendered width is driven by
            // the stage HEIGHT and its 464:688 aspect. Deriving the gutter from
            // that keeps the callouts beside the stack instead of on top of it,
            // at any viewport shape.
            "--vid-w": "min(100vw, calc(100svh * 0.6744))",
            "--gutter": "calc((100vw - var(--vid-w)) / 2)",
          } as React.CSSProperties
        }
      >
        {/* ── the object, filling the stage ─────────────────────────── */}
        <motion.div
          aria-hidden="true"
          className="pointer-events-none absolute inset-0"
          style={{
            opacity: glow,
            background:
              "radial-gradient(46% 52% at 50% 50%, rgba(122,162,247,.20) 0%, transparent 70%)",
          }}
        />
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
          className="absolute inset-0 h-full w-full object-contain"
          style={{
            opacity: ready ? 1 : 0,
            transition: "opacity .6s var(--ease-instrument)",
          }}
        />
        {/* keeps type legible over the brighter frames near the eruption */}
        <div
          aria-hidden="true"
          className="absolute inset-0"
          style={{
            background:
              "linear-gradient(90deg, rgba(10,11,12,.86) 0%, rgba(10,11,12,.35) 26%, transparent 44%, transparent 56%, rgba(10,11,12,.35) 74%, rgba(10,11,12,.86) 100%)",
          }}
        />

        {/* ── heading ───────────────────────────────────────────────── */}
        <div className="pointer-events-none absolute inset-x-0 top-0 z-10 mx-auto max-w-6xl px-5 pt-24 sm:px-8">
          <p className="eyebrow !text-ink/70">The stack</p>
          <h2 className="mt-2 font-display text-[clamp(1.4rem,2.6vw,2rem)] font-semibold tracking-[-0.03em] text-ink">
            Eight layers, signal to certificate
          </h2>
        </div>

        {/* ── side callouts (desktop) ───────────────────────────────── */}
        <div className="absolute inset-0 z-10 hidden lg:block">
          {layers.map((l, i) => {
            const side: "left" | "right" = i % 2 === 0 ? "left" : "right";
            const row = Math.floor(i / 2);
            const top = 26 + row * 17; // % down the stage
            const on = i === active;
            return (
              <motion.div
                key={l.n}
                initial={false}
                animate={{ opacity: on ? 1 : 0.42, x: on ? 0 : side === "left" ? -6 : 6 }}
                transition={{ duration: 0.32, ease: [0.2, 0.7, 0.2, 1] }}
                className={`absolute ${side === "left" ? "left-6" : "right-6"}`}
                style={{
                  top: `${top}%`,
                  width: "clamp(230px, calc(var(--gutter) - 2.5rem), 360px)",
                }}
              >
                <div
                  className={`flex items-center gap-2.5 ${
                    side === "left" ? "" : "flex-row-reverse"
                  }`}
                >
                  <span
                    className={`num shrink-0 font-mono text-[11px] ${on ? "text-accent-ink" : "text-faint"}`}
                  >
                    {l.n}
                  </span>
                  <span
                    className={`inline-flex shrink-0 items-center rounded-sm border px-2 py-[2px] font-mono text-[10px] uppercase tracking-[0.08em] ${
                      on ? CHIP[l.status].cls : "border-line text-faint"
                    }`}
                  >
                    {CHIP[l.status].text}
                  </span>
                  <span className={on ? "text-accent" : "text-faint"}>
                    <Leader dir={side} />
                  </span>
                </div>

                <h3
                  className={`mt-2.5 font-display text-[22px] font-semibold leading-tight tracking-[-0.02em] ${
                    side === "left" ? "text-left" : "text-right"
                  } ${on ? "text-ink" : "text-dim"}`}
                >
                  {l.name}
                </h3>
                {/* Only the active layer carries prose. Eight paragraphs
                    competing at low opacity read as noise; eight labels with
                    one of them expanded reads as a diagram. */}
                <motion.p
                  initial={false}
                  animate={{ opacity: on ? 1 : 0 }}
                  transition={{ duration: 0.24, ease: [0.2, 0.7, 0.2, 1] }}
                  className={`mt-2 text-[13.5px] leading-relaxed text-muted text-pretty ${
                    side === "left" ? "text-left" : "text-right"
                  }`}
                >
                  {l.body}
                </motion.p>
              </motion.div>
            );
          })}
        </div>

        {/* ── mobile: one callout, pinned low over the video ────────── */}
        <div className="absolute inset-x-0 bottom-0 z-10 lg:hidden">
          <div className="border-t border-line bg-bg/85 px-5 pb-8 pt-5 backdrop-blur-xl">
            {layers.map((l, i) => (
              <motion.div
                key={l.n}
                initial={false}
                animate={{ opacity: i === active ? 1 : 0, y: i === active ? 0 : 8 }}
                transition={{ duration: 0.28, ease: [0.2, 0.7, 0.2, 1] }}
                className={i === active ? "" : "pointer-events-none absolute inset-x-5 top-5"}
              >
                <div className="flex items-center gap-2.5">
                  <span className="num font-mono text-[11px] text-accent-ink">{l.n}</span>
                  <span
                    className={`inline-flex items-center rounded-sm border px-2 py-[2px] font-mono text-[10px] uppercase tracking-[0.08em] ${CHIP[l.status].cls}`}
                  >
                    {CHIP[l.status].text}
                  </span>
                </div>
                <h3 className="mt-2 font-display text-[20px] font-semibold tracking-[-0.02em] text-ink">
                  {l.name}
                </h3>
                <p className="mt-1.5 text-[13.5px] leading-relaxed text-muted text-pretty">
                  {l.body}
                </p>
              </motion.div>
            ))}
          </div>
        </div>

        {/* ── progress rail ─────────────────────────────────────────── */}
        <div className="absolute inset-x-0 bottom-0 z-20 mx-auto hidden max-w-6xl px-5 pb-7 sm:px-8 lg:block">
          <div className="flex items-center gap-1.5">
            {layers.map((l, i) => (
              <div
                key={l.n}
                className={`h-[2px] flex-1 rounded-full transition-colors duration-240 ease-instrument ${
                  i === active ? "bg-accent" : i < active ? "bg-line-bright" : "bg-line"
                }`}
              />
            ))}
          </div>
          <p className="num mt-3 font-mono text-[11px] text-faint">
            {String(active + 1).padStart(2, "0")} / {String(layers.length).padStart(2, "0")}
            <span className="ml-3">scroll to advance</span>
          </p>
        </div>
      </div>
    </div>
  );
}

/**
 * Reduced-motion / no-scrub fallback: the same eight layers as a plain list.
 * Nobody who has asked for less motion should be given a video whose playhead
 * is wired to their scroll wheel.
 */
export function LayerList({ layers }: { layers: Layer[] }) {
  return (
    <div className="mx-auto max-w-6xl px-5 py-16 sm:px-8">
      <p className="eyebrow">The stack</p>
      <h2 className="mt-2 font-display text-[clamp(1.5rem,3vw,2rem)] font-semibold tracking-[-0.03em] text-ink">
        Eight layers, signal to certificate
      </h2>
      <div className="mt-8 overflow-hidden rounded-lg border border-line bg-s1/62 backdrop-blur-xl">
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
    </div>
  );
}
