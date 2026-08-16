"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { AnimatePresence, motion, useMotionValueEvent, useScroll } from "motion/react";
import { useState } from "react";

import { nav, consoleUrl, site } from "@/lib/site-config";
import { Wordmark } from "./wordmark";

/**
 * The top bar.
 *
 * Previously this was a bare fixed strip with no backdrop, so page content
 * scrolled *through* the nav links and both became unreadable. It now condenses
 * on scroll: at the top of the hero it is pure type over footage, and past the
 * first 24px it earns a blurred, translucent plate and a hairline.
 *
 * The active-route indicator is a sliding pill rather than an underline —
 * `layoutId` animates it between items, so the bar reads as one instrument
 * whose state moves, not four independent links.
 */
export function SiteHeader() {
  const pathname = usePathname();
  const { scrollY } = useScroll();
  const [condensed, setCondensed] = useState(false);
  const [menuOpen, setMenuOpen] = useState(false);

  useMotionValueEvent(scrollY, "change", (y) => setCondensed(y > 24));

  return (
    <header className="fixed inset-x-0 top-0 z-50">
      {/* The plate. Separated from the content so blurring it cannot blur the
          type sitting on top, and so it can fade independently. */}
      <motion.div
        aria-hidden="true"
        className="absolute inset-0 border-b backdrop-blur-xl"
        initial={false}
        animate={{
          opacity: condensed ? 1 : 0,
          backgroundColor: "rgba(10,11,12,0.72)",
          borderBottomColor: "rgba(255,255,255,0.09)",
        }}
        transition={{ duration: 0.24, ease: [0.2, 0.7, 0.2, 1] }}
        style={{ borderBottomColor: "transparent" }}
      />

      <motion.div
        className="relative mx-auto flex max-w-6xl items-center gap-4 px-5 sm:px-8"
        initial={false}
        animate={{ height: condensed ? 58 : 72 }}
        transition={{ duration: 0.24, ease: [0.2, 0.7, 0.2, 1] }}
      >
        <Link
          href="/"
          className="flex shrink-0 items-center gap-2.5 text-ink"
          aria-label={`${site.name} home`}
        >
          <Wordmark className="h-[18px] w-[18px] text-accent" />
          <span className="text-[15px] font-semibold tracking-[-0.015em]">{site.name}</span>
        </Link>

        {/* desktop nav */}
        <nav className="ml-auto hidden items-center gap-0.5 sm:flex" aria-label="Main">
          {nav.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={`relative rounded-md px-3 py-1.5 text-[13.5px] transition-colors duration-160 ease-instrument ${
                  active ? "text-ink" : "text-muted hover:text-ink"
                }`}
              >
                {active && (
                  <motion.span
                    layoutId="nav-pill"
                    className="absolute inset-0 rounded-md border border-line bg-s2/80"
                    transition={{ duration: 0.28, ease: [0.2, 0.7, 0.2, 1] }}
                  />
                )}
                <span className="relative">{item.label}</span>
              </Link>
            );
          })}
        </nav>

        <a
          href={consoleUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="ml-auto hidden rounded-md bg-ink px-3.5 py-2 text-[13px] font-medium text-bg transition-colors duration-160 ease-instrument hover:bg-white sm:ml-3 sm:block"
        >
          Launch console
        </a>

        {/* mobile trigger — the nav links had no small-screen affordance at all */}
        <button
          type="button"
          onClick={() => setMenuOpen((v) => !v)}
          aria-expanded={menuOpen}
          aria-controls="mobile-nav"
          aria-label={menuOpen ? "Close menu" : "Open menu"}
          className="ml-auto flex h-9 w-9 items-center justify-center rounded-md border border-line text-ink sm:hidden"
        >
          {/* Two lines that rotate into a cross.
              NOT animated via the `d` attribute — Motion cannot interpolate SVG
              path data, and asking it to emits `<path d="undefined">` on every
              render. Rotating and translating the lines is both valid and
              smoother. */}
          <svg viewBox="0 0 16 16" className="h-4 w-4" fill="none" aria-hidden="true">
            <motion.line
              x1="2"
              y1="8"
              x2="14"
              y2="8"
              stroke="currentColor"
              strokeWidth="1.4"
              strokeLinecap="round"
              style={{ originX: "8px", originY: "8px" }}
              animate={menuOpen ? { rotate: 45, y: 0 } : { rotate: 0, y: -3.5 }}
              transition={{ duration: 0.2, ease: [0.2, 0.7, 0.2, 1] }}
            />
            <motion.line
              x1="2"
              y1="8"
              x2="14"
              y2="8"
              stroke="currentColor"
              strokeWidth="1.4"
              strokeLinecap="round"
              style={{ originX: "8px", originY: "8px" }}
              animate={menuOpen ? { rotate: -45, y: 0 } : { rotate: 0, y: 3.5 }}
              transition={{ duration: 0.2, ease: [0.2, 0.7, 0.2, 1] }}
            />
          </svg>
        </button>
      </motion.div>

      <AnimatePresence>
        {menuOpen && (
          <motion.nav
            id="mobile-nav"
            aria-label="Main"
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            transition={{ duration: 0.2, ease: [0.2, 0.7, 0.2, 1] }}
            className="relative border-b border-line bg-bg/92 px-5 pb-5 backdrop-blur-xl sm:hidden"
          >
            <ul className="flex flex-col">
              {nav.map((item) => (
                <li key={item.href}>
                  <Link
                    href={item.href}
                    onClick={() => setMenuOpen(false)}
                    className={`block border-b border-line-soft py-3 text-[15px] ${
                      pathname === item.href ? "text-ink" : "text-muted"
                    }`}
                  >
                    {item.label}
                  </Link>
                </li>
              ))}
            </ul>
            <a
              href={consoleUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="mt-4 block rounded-md bg-ink px-4 py-2.5 text-center text-[14px] font-medium text-bg"
            >
              Launch console ↗
            </a>
          </motion.nav>
        )}
      </AnimatePresence>
    </header>
  );
}
