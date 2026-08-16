"use client";

import Link from "next/link";
import { useRef } from "react";
import { motion, useScroll, useTransform } from "motion/react";

import { consoleUrl } from "@/lib/site-config";

/**
 * The hero.
 *
 * It no longer owns a video — the footage is a single fixed layer behind the
 * whole site (see components/chrome/site-background.tsx), which is what makes it
 * continuous across routes and gives parallax for free. All this does now is
 * hold the copy and fade it out as you scroll past.
 */
export function Hero() {
  const ref = useRef<HTMLDivElement>(null);
  const { scrollYProgress } = useScroll({
    target: ref,
    offset: ["start start", "end start"],
  });

  const copyY = useTransform(scrollYProgress, [0, 1], [0, -64]);
  const copyFade = useTransform(scrollYProgress, [0, 0.6], [1, 0]);

  return (
    <div ref={ref} className="relative h-[100svh] min-h-[620px]">
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
            className="rounded-md border border-line-bright bg-bg/30 px-5 py-2.5 text-[14px] font-medium text-ink backdrop-blur-sm transition-colors duration-160 ease-instrument hover:bg-s2/70"
          >
            Launch the console ↗
          </a>
        </motion.div>

        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.28 }}
          className="num mt-10 font-mono text-[11.5px] tracking-[0.04em] text-ink/55"
        >
          PhysioNet/CinC 2019 · 40,335 patients · Z3-verified protocols
        </motion.p>
      </motion.div>
    </div>
  );
}
