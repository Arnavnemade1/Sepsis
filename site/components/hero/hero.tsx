"use client";

import Link from "next/link";
import { useRef } from "react";
import { motion, useScroll, useTransform } from "motion/react";

import { HeroVideo } from "./hero-video";
import { consoleUrl } from "@/lib/site-config";

/**
 * Scroll-linked hero.
 *
 * The scroll target must be `position: relative` or scrollYProgress stays
 * pinned at 0. Only y / opacity / scale are animated — those ride the
 * compositor, whereas animating height or background-position would drop frames
 * against a playing video.
 *
 * Deliberately NO useSpring smoothing on the scroll values: the identity says
 * motion is "quick and mechanical; nothing bounces". Raw scroll linkage reads
 * like an instrument; a lagging spring reads like a portfolio site.
 */
export function Hero() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end start"],
  });

  const videoY = useTransform(scrollYProgress, [0, 1], ["0%", "14%"]);
  const scrim = useTransform(scrollYProgress, [0, 0.7], [0.55, 1]);
  const copyY = useTransform(scrollYProgress, [0, 1], [0, -56]);
  const copyFade = useTransform(scrollYProgress, [0, 0.55], [1, 0]);

  return (
    <div ref={ref} className="relative h-[100svh] min-h-[620px] overflow-hidden">
      {/* overscan so the parallax never exposes an edge at the bottom of travel */}
      <motion.div style={{ y: videoY }} className="absolute inset-x-0 -inset-y-[8%]">
        <HeroVideo />
      </motion.div>

      {/* Scrim. Lands exactly on --bg so the next section joins seamlessly.
          Three stops, not two: this footage is BRIGHTEST at the top (sky), which
          is also where the nav sits, so the ramp has to be dark at the very top,
          open up through the middle where the peaks are, then close again. A
          simple light-to-dark ramp left the nav links illegible. */}
      <motion.div style={{ opacity: scrim }} className="absolute inset-0" aria-hidden="true">
        <div
          className="absolute inset-0"
          style={{
            background:
              "linear-gradient(180deg, rgba(10,11,12,.82) 0%, rgba(10,11,12,.46) 16%, rgba(10,11,12,.40) 34%, rgba(10,11,12,.80) 74%, #0a0b0c 100%)",
          }}
        />
        {/* left-weighted, so the headline gets contrast without flattening the
            whole frame */}
        <div
          className="absolute inset-0"
          style={{
            background:
              "linear-gradient(90deg, rgba(10,11,12,.62) 0%, rgba(10,11,12,.28) 45%, transparent 72%)",
          }}
        />
      </motion.div>

      <motion.div
        style={{ y: copyY, opacity: copyFade }}
        className="relative mx-auto flex h-full max-w-6xl flex-col justify-center px-5 sm:px-8"
      >
        <motion.p
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4, ease: [0.2, 0.7, 0.2, 1] }}
          // The shared .eyebrow colour (--dim) is tuned for flat graphite panels
          // and disappears against footage. Over video it needs to be lifted.
          className="eyebrow flex items-center gap-2 !text-ink/80"
        >
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-low" />
          Open research · CinC-2019 validated · Not a medical device
        </motion.p>

        <motion.h1
          initial={{ opacity: 0, y: 12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.06, ease: [0.2, 0.7, 0.2, 1] }}
          className="mt-6 max-w-[17ch] font-display text-[clamp(2.75rem,7.5vw,5.25rem)] font-semibold leading-[0.97] tracking-[-0.04em] text-ink text-balance"
        >
          Sepsis shows itself hours before anyone is watching.
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.12, ease: [0.2, 0.7, 0.2, 1] }}
          className="mt-6 max-w-xl text-[17px] leading-relaxed text-muted text-pretty sm:text-[18px]"
        >
          AegisOS listens continuously, gives a probability you can actually trust, and
          proves the treatment it proposes is safe before it ever reaches a clinician.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.18, ease: [0.2, 0.7, 0.2, 1] }}
          className="mt-9 flex flex-wrap items-center gap-3"
        >
          <Link
            href="/research"
            className="rounded-md bg-ink px-5 py-2.5 text-[14px] font-medium text-bg transition-colors duration-160 ease-instrument hover:bg-white"
          >
            Read the research
          </Link>
          <a
            href={consoleUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="rounded-md border border-line-bright px-5 py-2.5 text-[14px] font-medium text-ink transition-colors duration-160 ease-instrument hover:bg-s2"
          >
            Launch the console ↗
          </a>
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.28 }}
          className="num mt-10 font-mono text-[11.5px] tracking-[0.04em] text-dim"
        >
          PhysioNet/CinC 2019 · 40,335 patients · Z3-verified protocols
        </motion.p>
      </motion.div>
    </div>
  );
}
