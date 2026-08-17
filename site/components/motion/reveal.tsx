"use client";

import { motion, type Variants } from "motion/react";
import { useRef, type ReactNode } from "react";

import { useSafeInView } from "./use-safe-in-view";

/**
 * Scroll reveals.
 *
 * y: 8, not 40 — eight pixels is two units of the 4px rhythm. A big slide-up on
 * a clinical instrument reads as a landing-page template.
 *
 * These are driven by `useSafeInView` rather than `whileInView`. The behaviour
 * is the same when the observer works; the difference is that content can never
 * be left permanently at `opacity: 0` if it does not. Every element here starts
 * invisible, so a silent observer failure is a blank page, not a missing
 * flourish — and that has already happened once on this site.
 *
 * Under prefers-reduced-motion these collapse to a plain opacity fade
 * automatically, via MotionConfig reducedMotion="user" in MotionProvider.
 */
const group: Variants = {
  hidden: {},
  show: { transition: { staggerChildren: 0.06, delayChildren: 0.04 } },
};

const item: Variants = {
  hidden: { opacity: 0, y: 8 },
  show: { opacity: 1, y: 0, transition: { duration: 0.24, ease: [0.2, 0.7, 0.2, 1] } },
};

export function Stagger({
  children,
  className,
  as = "div",
}: {
  children: ReactNode;
  className?: string;
  as?: "div" | "ul" | "section";
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useSafeInView(ref);
  const C = motion[as];

  return (
    <C
      ref={ref as never}
      className={className}
      variants={group}
      initial="hidden"
      animate={inView ? "show" : "hidden"}
    >
      {children}
    </C>
  );
}

export function Item({
  children,
  className,
  as = "div",
}: {
  children: ReactNode;
  className?: string;
  as?: "div" | "li";
}) {
  const C = motion[as];
  return (
    <C variants={item} className={className}>
      {children}
    </C>
  );
}

/** A single element that reveals on its own, without a parent Stagger. */
export function Reveal({
  children,
  className,
  delay = 0,
}: {
  children: ReactNode;
  className?: string;
  delay?: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useSafeInView(ref);

  return (
    <motion.div
      ref={ref}
      className={className}
      initial={{ opacity: 0, y: 8 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 8 }}
      transition={{ duration: 0.24, ease: [0.2, 0.7, 0.2, 1], delay }}
    >
      {children}
    </motion.div>
  );
}
