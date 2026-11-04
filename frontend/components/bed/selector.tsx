"use client";

import { useRouter } from "next/navigation";
import { useEffect, useRef, useState } from "react";

import { CaretDown } from "@/components/icons";
import { cx } from "@/components/ui";
import type { RosterPatient } from "@/lib/types";

/** The patient switcher that sits in the top bar of any per-patient workspace. */
export function BedSelector({
  roster,
  current,
  hrefFor = (id) => `/patient/?id=${encodeURIComponent(id)}`,
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
        className="field flex h-8 min-w-0 max-w-full items-center gap-2.5 pl-2.5 pr-2 text-sm"
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        {current ? (
          <>
            <span className={cx("h-1.5 w-1.5 shrink-0 rounded-full", dot(current))} />
            <span className="min-w-0 max-w-[220px] truncate text-ink">{current.name}</span>
            <span className="hidden shrink-0 whitespace-nowrap text-xs text-dim sm:inline">
              {current.kind === "live" ? "live session" : current.id.replace("bed_", "bed ")}
            </span>
          </>
        ) : (
          <span className="text-muted">Choose a patient</span>
        )}
        <CaretDown size={13} className="shrink-0 text-dim" aria-hidden />
      </button>
      {open && (
        <div role="listbox" className="pop absolute left-0 top-10 z-50 max-h-[70vh] w-80 max-w-[calc(100vw-24px)] overflow-y-auto p-1">
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
                "flex w-full items-center gap-2.5 rounded-md px-2.5 py-2 text-left transition-colors",
                p.id === current?.id ? "bg-white/[0.08]" : "hover:bg-white/[0.05]",
              )}
            >
              <span className={cx("h-1.5 w-1.5 shrink-0 rounded-full", dot(p))} />
              <span className="min-w-0 flex-1 truncate text-sm text-ink">{p.name}</span>
              <span className="shrink-0 whitespace-nowrap text-xs text-dim">{p.kind === "live" ? "live session" : p.id.replace("bed_", "bed ")}</span>
            </button>
          ))}
        </div>
      )}
    </div>
  );
}
