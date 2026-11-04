"use client";

import Link from "next/link";
import { useState } from "react";

import { Close, OpenLink } from "@/components/icons";
import { cx, Drawer, Panel, Skeleton } from "@/components/ui";
import { usePoll } from "@/lib/hooks";

/** One entry of /aegis/models (backend/aegis/models_registry.py). */
export type Model = {
  code: string;
  name: string;
  kind: "detector" | "support" | "language";
  decides: boolean;
  status: "serving" | "fallback" | "explains" | "retired" | "unavailable";
  serves: string;
  reads: string;
  method: string;
  id?: string | null;
  trained_on?: number | null;
  artifact?: string;
  measured?: {
    auroc?: number | null;
    hour_matched?: number | null;
    ece?: number | null;
    patients?: number | null;
    vs?: string;
    note?: string;
    source?: string;
  } | null;
  alarm?: { high: number; medium: number } | null;
  conformal?: { alpha: number } | null;
};
type Registry = { models: Model[]; deciding: string[]; note: string };

const STATUS: Record<Model["status"], { label: string; color: string }> = {
  serving: { label: "Serving", color: "#2fb67c" },
  fallback: { label: "Fallback", color: "#eeb12a" },
  explains: { label: "Explains", color: "#868d97" },
  retired: { label: "Retired", color: "#5f666f" },
  unavailable: { label: "Unavailable", color: "#e5484d" },
};

export function ModelCode({ code, strong = false }: { code: string; strong?: boolean }) {
  return (
    <span
      className={cx(
        "inline-grid h-5 min-w-9 place-items-center rounded-[5px] px-1.5 font-mono text-2xs font-semibold",
        strong ? "bg-ink/[0.14] text-ink" : "bg-white/[0.06] text-muted",
      )}
    >
      {code}
    </span>
  );
}

function headline(m: Model): string {
  const r = m.measured;
  if (!r) return "—";
  if (r.auroc != null) return `AUROC ${r.auroc.toFixed(3)}${r.hour_matched != null ? ` · ${r.hour_matched.toFixed(3)} hour-matched` : ""}`;
  return r.note ?? "—";
}

/** Every model the system runs, by name. Only the two detectors can page anyone. */
export function ModelsPanel() {
  const reg = usePoll<Registry>("/aegis/models", 30000);
  const [sel, setSel] = useState<string | null>(null);
  const models = reg.data?.models ?? [];
  const groups = [
    { label: "Can page", about: "each at its own validated alarm point", items: models.filter((m) => m.decides) },
    { label: "Explain, route or assist", about: "none of these can raise or lower a page on its own", items: models.filter((m) => !m.decides) },
  ];
  const selected = models.find((m) => m.code === sel) ?? null;
  return (
    <>
      <Panel title="Models" meta={`${models.length || "—"} models, named. ${reg.data?.note ?? ""}`} flush className="mb-3">
        {!reg.data ? (
          <div className="p-3.5">
            <Skeleton className="h-48" />
          </div>
        ) : (
          groups.map((g) => (
            <div key={g.label}>
              <div className="flex items-baseline gap-2 border-b border-line bg-white/[0.025] px-4 py-2 text-xs">
                <span className="font-medium text-muted">{g.label}</span>
                <span className="text-dim">{g.about}</span>
              </div>
              <ul className="divide-y divide-line-soft">
                {g.items.map((m) => (
                  <li key={m.code}>
                    <button
                      onClick={() => setSel(m.code)}
                      className="grid w-full grid-cols-[auto_minmax(0,1fr)] items-start gap-x-3 gap-y-1 px-4 py-2.5 text-left transition-colors hover:bg-s2 md:grid-cols-[auto_minmax(0,1.1fr)_minmax(0,1.3fr)_minmax(0,1fr)_auto] md:items-center"
                    >
                      <ModelCode code={m.code} strong={m.decides} />
                      <div className="min-w-0">
                        <div className="truncate text-sm font-medium text-ink">{m.name}</div>
                        <div className="truncate text-xs text-dim md:hidden">{m.serves}</div>
                      </div>
                      <div className="hidden min-w-0 truncate text-xs text-muted md:block" title={m.serves}>
                        {m.serves}
                      </div>
                      <div className="num col-span-2 min-w-0 truncate text-xs text-muted md:col-span-1" title={headline(m)}>
                        {headline(m)}
                      </div>
                      <span className="hidden items-center gap-1.5 justify-self-end text-xs text-dim md:inline-flex">
                        <span className="h-1.5 w-1.5 rounded-full" style={{ background: STATUS[m.status].color }} />
                        {STATUS[m.status].label}
                      </span>
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          ))
        )}
      </Panel>
      <Drawer open={!!selected} onClose={() => setSel(null)} width={460}>
        {selected && <ModelInspector m={selected} onClose={() => setSel(null)} />}
      </Drawer>
    </>
  );
}

function ModelInspector({ m, onClose }: { m: Model; onClose: () => void }) {
  const r = m.measured;
  const facts: [string, string][] = [
    ["Can page", m.decides ? "yes" : "no"],
    ["Status", STATUS[m.status].label],
    ["Serves", m.serves],
    ["Reads", m.reads],
    ["Method", m.method],
    ...(m.trained_on ? ([["Trained on", `${m.trained_on.toLocaleString()} patients`]] as [string, string][]) : []),
    ...(r?.auroc != null ? ([["AUROC", r.auroc.toFixed(3)]] as [string, string][]) : []),
    ...(r?.hour_matched != null ? ([["Hour-matched AUROC", r.hour_matched.toFixed(3)]] as [string, string][]) : []),
    ...(r?.ece != null ? ([["Calibration error", r.ece.toFixed(4)]] as [string, string][]) : []),
    ...(r?.patients ? ([["Judged on", `${r.patients.toLocaleString()} unseen patients`]] as [string, string][]) : []),
    ...(m.alarm ? ([["Pages at", `P ≥ ${(m.alarm.high * 100).toFixed(2)}% (index 0.70)`], ["Medium from", `P ≥ ${(m.alarm.medium * 100).toFixed(2)}%`]] as [string, string][]) : []),
    ...(m.conformal ? ([["Conformal sets", `α = ${m.conformal.alpha}`]] as [string, string][]) : []),
    ...(m.artifact ? ([["Artifact", m.artifact]] as [string, string][]) : []),
    ...(m.id ? ([["Fit", m.id]] as [string, string][]) : []),
  ];
  return (
    <>
      <header className="flex shrink-0 items-start gap-3 border-b border-line px-5 py-4">
        <div className="min-w-0 flex-1">
          <div className="text-xs text-dim">{m.kind === "detector" ? "Detector" : m.kind === "language" ? "Language model" : "Supporting model"}</div>
          <div className="mt-1 flex items-center gap-2">
            <ModelCode code={m.code} strong={m.decides} />
            <h2 className="min-w-0 truncate text-md font-semibold text-ink">{m.name}</h2>
          </div>
        </div>
        <button onClick={onClose} className="grid h-7 w-7 place-items-center rounded-md text-dim transition-colors hover:bg-white/[0.07] hover:text-ink" aria-label="Close">
          <Close size={14} />
        </button>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <dl className="well divide-y divide-white/[0.05] px-3.5">
          {facts.map(([k, v]) => (
            <div key={k} className="grid grid-cols-[112px_minmax(0,1fr)] gap-3 py-1.5 text-sm">
              <dt className="text-dim">{k}</dt>
              <dd className={cx("min-w-0 break-words text-ink", (k === "Fit" || k === "Artifact") && "font-mono text-xs")}>{v}</dd>
            </div>
          ))}
        </dl>
        {(r?.vs || r?.note) && (
          <p className="mt-3 text-sm leading-relaxed text-muted">
            {r?.vs && <>Compared with {r.vs}. </>}
            {r?.note && <>{r.note.charAt(0).toUpperCase() + r.note.slice(1)}.</>}
          </p>
        )}
        {r?.source && <div className="mt-3 text-xs text-dim">Source <span className="font-mono">{r.source}</span></div>}
      </div>
      {m.decides && (
        <footer className="shrink-0 border-t border-line px-5 py-3.5">
          <Link href="/lab/" className="inline-flex h-8 items-center gap-1.5 rounded-md bg-ink px-3 text-sm font-medium text-[#17191c] transition-colors hover:bg-white">
            Open evidence
            <OpenLink size={13} />
          </Link>
        </footer>
      )}
    </>
  );
}
