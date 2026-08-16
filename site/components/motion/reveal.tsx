"use client";

import { motion, type Variants } from "motion/react";
import type { ReactNode } from "react";

/**
 * Scroll reveals.
 *
 * y: 8, not 40 — eight pixels is two units of the 4px rhythm. A big slide-up on
 * a clinical instrument reads as a landing-page template.
 *
 * once: true — re-triggering on every scroll-past is the most common taste
 * failure in this pattern.
 *
 * Under prefers-reduced-motion these collapse to a plain opacity fade
 * automatically, via MotionConfig reducedMotion="user".
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
  const C = motion[as];
  return (
    <C
      className={className}
      variants={group}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, amount: 0.2, margin: "0px 0px -10% 0px" }}
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
  return (
    <motion.div
      className={className}
      initial={{ opacity: 0, y: 8 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, amount: 0.2, margin: "0px 0px -10% 0px" }}
      transition={{ duration: 0.24, ease: [0.2, 0.7, 0.2, 1], delay }}
    >
      {children}
    </motion.div>
  );
}
