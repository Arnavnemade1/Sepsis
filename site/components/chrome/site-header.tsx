"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { motion } from "motion/react";

import { nav, consoleUrl, site } from "@/lib/site-config";
import { Wordmark } from "./wordmark";

/**
 * Hairline nav. No background blur wash until the page has scrolled — Rule 2
 * says structure is hairlines, so at the top of the hero the header is pure
 * type over video, and only the bottom hairline appears once content is behind
 * it.
 */
export function SiteHeader() {
  const pathname = usePathname();

  return (
    <header className="fixed inset-x-0 top-0 z-40">
      <div className="mx-auto flex h-16 max-w-6xl items-center gap-6 px-5 sm:px-8">
        <Link
          href="/"
          className="flex shrink-0 items-center gap-2.5 text-ink"
          aria-label={`${site.name} home`}
        >
          <Wordmark className="h-[18px] w-[18px] text-accent" />
          <span className="text-[15px] font-semibold tracking-[-0.01em]">{site.name}</span>
        </Link>

        <nav className="ml-auto hidden items-center gap-1 sm:flex" aria-label="Main">
          {nav.map((item) => {
            const active = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                aria-current={active ? "page" : undefined}
                className={`relative rounded-md px-3 py-2 text-[13.5px] transition-colors duration-160 ease-instrument ${
                  active ? "text-ink" : "text-muted hover:text-ink"
                }`}
              >
                {item.label}
                {active && (
                  <motion.span
                    layoutId="nav-active"
                    className="absolute inset-x-3 -bottom-px h-px bg-accent"
                    transition={{ duration: 0.24, ease: [0.2, 0.7, 0.2, 1] }}
                  />
                )}
              </Link>
            );
          })}
        </nav>

        <a
          href={consoleUrl}
          target="_blank"
          rel="noopener noreferrer"
          className="ml-auto rounded-md border border-line-bright bg-ink px-3.5 py-2 text-[13px] font-medium text-bg transition-colors duration-160 ease-instrument hover:bg-white sm:ml-0"
        >
          Launch console
        </a>
      </div>
    </header>
  );
}
