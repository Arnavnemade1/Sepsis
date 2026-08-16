"use client";

import { useEffect, useState } from "react";

export interface NavSection {
  id: string;
  label: string;
}

/**
 * Sticky in-page nav. Highlights the section currently in the upper part of the
 * viewport — rootMargin's negative bottom means a section only becomes "current"
 * once it has actually reached reading position, rather than the moment its
 * first pixel appears.
 */
export function SectionNav({ sections }: { sections: NavSection[] }) {
  const [active, setActive] = useState(sections[0]?.id);

  useEffect(() => {
    const els = sections
      .map((s) => document.getElementById(s.id))
      .filter((el): el is HTMLElement => Boolean(el));

    const io = new IntersectionObserver(
      (entries) => {
        const visible = entries
          .filter((e) => e.isIntersecting)
          .sort((a, b) => a.boundingClientRect.top - b.boundingClientRect.top);
        if (visible[0]) setActive(visible[0].target.id);
      },
      { rootMargin: "-88px 0px -62% 0px", threshold: 0 },
    );

    els.forEach((el) => io.observe(el));
    return () => io.disconnect();
  }, [sections]);

  return (
    <nav aria-label="Sections" className="sticky top-24">
      <p className="eyebrow mb-3">Contents</p>
      <ol className="space-y-0.5 border-l border-line">
        {sections.map((s, i) => {
          const on = active === s.id;
          return (
            <li key={s.id}>
              <a
                href={`#${s.id}`}
                aria-current={on ? "true" : undefined}
                className={`-ml-px flex gap-2.5 border-l py-1.5 pl-3 text-[12.5px] leading-snug transition-colors duration-160 ease-instrument ${
                  on
                    ? "border-accent text-ink"
                    : "border-transparent text-dim hover:border-line-bright hover:text-muted"
                }`}
              >
                <span className="num font-mono text-[10.5px] text-faint">
                  {String(i + 1).padStart(2, "0")}
                </span>
                {s.label}
              </a>
            </li>
          );
        })}
      </ol>
    </nav>
  );
}
