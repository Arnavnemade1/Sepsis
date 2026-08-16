import Link from "next/link";

import { nav, consoleUrl, site, disclaimer } from "@/lib/site-config";
import { Wordmark } from "./wordmark";

export function SiteFooter() {
  return (
    <footer className="border-t border-line">
      <div className="mx-auto max-w-6xl px-5 py-12 sm:px-8">
        <div className="flex flex-col gap-8 sm:flex-row sm:items-start sm:justify-between">
          <div className="max-w-sm">
            <div className="flex items-center gap-2.5 text-ink">
              <Wordmark className="h-[18px] w-[18px] text-accent" />
              <span className="text-[15px] font-semibold tracking-[-0.01em]">{site.name}</span>
            </div>
            <p className="mt-3 text-[13.5px] leading-relaxed text-dim">
              {site.description}
            </p>
          </div>

          <nav className="flex flex-col gap-2.5 text-[13.5px]" aria-label="Footer">
            {nav.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="text-muted transition-colors duration-160 ease-instrument hover:text-ink"
              >
                {item.label}
              </Link>
            ))}
            <a
              href={consoleUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted transition-colors duration-160 ease-instrument hover:text-ink"
            >
              Console ↗
            </a>
            <a
              href={site.repo}
              target="_blank"
              rel="noopener noreferrer"
              className="text-muted transition-colors duration-160 ease-instrument hover:text-ink"
            >
              GitHub ↗
            </a>
          </nav>
        </div>

        <div className="mt-10 flex flex-col gap-3 border-t border-line-soft pt-6 text-[12px] text-faint sm:flex-row sm:items-center sm:justify-between">
          <p>{disclaimer}</p>
          <p className="num">
            Data: PhysioNet/CinC 2019 (CC BY 4.0) · BIDMC PPG+ECG
          </p>
        </div>
      </div>
    </footer>
  );
}
