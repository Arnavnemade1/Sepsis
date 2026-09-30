"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { cx } from "@/components/ui";
import type { RosterPatient } from "@/lib/types";

/** The patient switcher that sits in the top bar of any per-patient workspace. */
export function BedSelector({
  roster,
  current,
  hrefFor = (id) => `/bed/?id=${encodeURIComponent(id)}`,
}: {
  roster: RosterPatient[];
  current: RosterPatient | null;
  hrefFor?: (id: string) => string;
}) {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const close = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", esc);
    };
  }, []);

  const dot = (p: RosterPatient) => (p.kind === "live" ? "bg-accent" : "bg-faint");
  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2.5 rounded-full border border-line bg-s2 py-1.5 pl-3 pr-2.5 text-[13px] hover:border-line-bright"
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        {current ? (
          <>
            <span className={cx("h-1.5 w-1.5 rounded-full", dot(current))} />
            <span className="max-w-[220px] truncate text-ink">{current.name}</span>
            <span className="text-[11.5px] text-dim">{current.kind === "live" ? "live session" : current.id.replace("bed_", "bed ")}</span>
          </>
        ) : (
          <span className="text-muted">Choose a patient</span>
        )}
        <svg width="10" height="10" viewBox="0 0 10 10" className="text-dim" aria-hidden>
          <path d="M2 3.5 5 6.5 8 3.5" stroke="currentColor" fill="none" strokeWidth="1.4" strokeLinecap="round" />
        </svg>
      </button>
      {open && (
        <div role="listbox" className="plate-float absolute left-1/2 top-11 z-50 max-h-[70vh] w-80 -translate-x-1/2 overflow-y-auto rounded-lg p-1.5 shadow-2xl">
          {roster.map((p) => (
            <button
              key={p.id}
              role="option"
              aria-selected={p.id === current?.id}
              onClick={() => {
                setOpen(false);
                router.push(hrefFor(p.id));
              }}
              className={cx(
                "flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left",
                p.id === current?.id ? "bg-s3" : "hover:bg-s2",
              )}
            >
              <span className={cx("h-1.5 w-1.5 shrink-0 rounded-full", dot(p))} />
              <span className="min-w-0 flex-1 truncate text-[13px] text-ink">{p.name}</span>
              <span className="text-[11px] text-dim">{p.kind === "live" ? "live session" : p.id.replace("bed_", "bed ")}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
