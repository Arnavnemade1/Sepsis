"use client";

import { useState } from "react";

import { Check, Flask, Plugs, Reset } from "@/components/icons";
import { Button, cx, Empty, MiniBars, Panel, Restricted, Skeleton, StatStrip, Tag } from "@/components/ui";
import { api } from "@/lib/api";
import { useCaps, usePoll } from "@/lib/hooks";
import { ACK_TONE, ago, STATUS_LABEL, STATUS_TONE, type MappingRow, type Overview, type SimStatus } from "@/lib/interop";

type Bench = {
  available: boolean;
  posthoc?: {
    note: string;
    h4_per_100k: Record<string, number>;
    h5: { correct_auto_adopted: number; unsafe_wrong_unit: { rate: number }; unsafe_wrong_concept: { rate: number } };
    delivered_faulty: Record<string, number>;
    fix_dauroc: Record<string, number | null>;
    detected: string[];
  } | null;
  clean?: { auroc: number; far: number; detection: number };
  verdicts?: Record<string, { pass: boolean } & Record<string, unknown>>;
  faults?: { id: string; what: string; silent_dauroc: number; silent_dfar: number; detected: boolean; faulty_to_quarantine: number | null; delivered_faulty: number; fix_dauroc: number | null; quarantine_dauroc: number }[];
};

type Audit = { findings?: { kind: string; concept?: string; invariant?: string; site: string; outside_window?: number; rate?: number; holdout?: { outside_window?: number; rate?: number } | null }[] };

export function IntegrationsOverview({ onMessage, onMapping }: { onMessage: (id: string) => void; onMapping: (id: string) => void }) {
  const { data: ov } = usePoll<Overview>("/aegis/interop", 2500);
  const { data: sim, reload: reloadSim } = usePoll<SimStatus>("/aegis/interop/sim", 5000);
  const { data: audit } = usePoll<Audit>("/aegis/interop/audit");
  const { data: bench } = usePoll<Bench>("/aegis/interop/bench");
  const { caps, role } = useCaps();
  const now = ov?.now ?? Date.now() / 1000;
  const c = ov?.counters ?? {};
  const live = ov?.interfaces.filter((i) => i.status === "live").length ?? 0;
  const perMin = ov?.interfaces.reduce((a, i) => a + i.per_min, 0) ?? 0;
  const errs5 = ov?.interfaces.reduce((a, i) => a + i.series.slice(-5).reduce((x, y) => x + y, 0), 0) ?? 0;

  return (
    <div>
      <StatStrip
        className="mb-5"
        items={[
          { k: "Interfaces", v: ov ? `${live} of ${ov.interfaces.length} live` : "—", sub: "inbound HL7 feeds" },
          { k: "Messages", v: ov ? `${perMin.toFixed(1)}/min` : "—", sub: ov ? `${(c.messages ?? 0).toLocaleString()} since boot` : "" },
          { k: "Mappings adopted", v: ov ? `${ov.mappings.by_status.adopted ?? 0} of ${ov.mappings.total}` : "—", sub: "each earned on evidence" },
          { k: "Waiting for review", v: ov ? String(ov.mappings.by_status.review ?? 0) : "—", sub: "inferred unit, ambiguous or refused" },
          { k: "Quarantined", v: ov ? String(ov.mappings.by_status.quarantined ?? 0) : "—", tone: ov?.mappings.by_status.quarantined ? "text-high-ink" : undefined, sub: ov ? `${c.retracted ?? 0} values withdrawn` : "" },
          { k: "Delivered / withheld", v: ov ? `${((c.delivered ?? 0) + (c.backfilled ?? 0)).toLocaleString()} / ${(c.withheld ?? 0).toLocaleString()}` : "—", sub: `${errs5} messages in the last 5 min` },
        ]}
      />

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.55fr)_minmax(0,1fr)]">
        <div className="min-w-0 space-y-4">
          <Panel title="Interfaces" meta="Every system that sends AegisOS data, and how its feed is doing" flush>
            {!ov ? (
              <div className="p-3.5">
                <Skeleton className="h-40" />
              </div>
            ) : (
              <>
                <div className="hidden overflow-x-auto md:block">
                  <table className="w-full min-w-[720px] text-sm">
                    <thead className="text-left text-xs text-dim">
                      <tr className="border-b border-line">
                        <th className="px-4 py-2.5 font-medium">Interface</th>
                        <th className="px-2 py-2.5 font-medium">Standard</th>
                        <th className="px-2 py-2.5 font-medium">Status</th>
                        <th className="px-2 py-2.5 font-medium">Traffic (30 min)</th>
                        <th className="px-2 py-2.5 text-right font-medium">Codes adopted</th>
                        <th className="whitespace-nowrap px-2 py-2.5 text-right font-medium">ACK AA / AE / AR</th>
                        <th className="px-4 py-2.5 text-right font-medium">p50</th>
                      </tr>
                    </thead>
                    <tbody>
                      {ov.interfaces.map((i) => {
                        const bi = ov.mappings.by_interface?.[i.id] ?? { total: 0 };
                        const attention = (bi.review ?? 0) + (bi.quarantined ?? 0);
                        return (
                          <tr key={i.id} className="border-b border-line-soft last:border-0">
                            <td className="px-4 py-3">
                              <div className="font-medium text-ink">{i.name}</div>
                              <div className="mt-0.5 font-mono text-2xs text-dim">
                                {i.sending_app}/{i.facility} · {i.kind}
                                {i.simulated ? " · simulated" : ""}
                              </div>
                            </td>
                            <td className="px-2 py-3 text-muted">
                              {i.standard}
                              <div className="text-xs text-dim">{i.transport}</div>
                            </td>
                            <td className="px-2 py-3">
                              <Tag tone={STATUS_TONE[i.status]} dot>
                                {STATUS_LABEL[i.status]}
                              </Tag>
                              <div className="mt-0.5 text-xs text-dim">{ago(i.last_message_at, now)}</div>
                            </td>
                            <td className="px-2 py-3">
                              <MiniBars data={i.series} w={96} h={20} title={`${i.per_min.toFixed(1)} messages per minute`} />
                              <div className="text-xs text-dim">{i.per_min.toFixed(1)}/min</div>
                            </td>
                            <td className="num whitespace-nowrap px-2 py-3 text-right text-ink">
                              {bi.total ? `${bi.adopted ?? 0} / ${bi.total}` : <span className="text-dim">—</span>}
                              {!bi.total && <div className="font-sans text-xs text-dim">no coded results</div>}
                              {attention > 0 && <div className="text-xs text-med-ink">{attention} need a person</div>}
                            </td>
                            <td className="num whitespace-nowrap px-2 py-3 text-right text-muted">
                              {i.stats.AA ?? 0} / <span className={i.stats.AE ? "text-med-ink" : ""}>{i.stats.AE ?? 0}</span> / <span className={i.stats.AR ? "text-high-ink" : ""}>{i.stats.AR ?? 0}</span>
                            </td>
                            <td className="num whitespace-nowrap px-4 py-3 text-right text-muted">{i.latency_us.p50 != null ? `${(i.latency_us.p50 / 1000).toFixed(2)} ms` : "—"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
                <div className="divide-y divide-line-soft md:hidden">
                  {ov.interfaces.map((i) => (
                    <div key={i.id} className="px-4 py-2.5">
                      <div className="flex items-center gap-2">
                        <span className="min-w-0 flex-1 truncate text-sm font-medium text-ink">{i.name}</span>
                        <Tag tone={STATUS_TONE[i.status]} dot>
                          {STATUS_LABEL[i.status]}
                        </Tag>
                      </div>
                      <div className="mt-1 flex items-center gap-3 text-xs text-dim">
                        <span>{i.standard}</span>
                        <span>{i.per_min.toFixed(1)}/min</span>
                        <span className="num ml-auto">
                          {i.stats.AA ?? 0}/{i.stats.AE ?? 0}/{i.stats.AR ?? 0}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </Panel>

          <Panel title="Message stream" meta="The latest messages and the acknowledgement each sender got back" flush>
            {!ov ? (
              <div className="p-3.5">
                <Skeleton className="h-56" />
              </div>
            ) : !ov.recent.length ? (
              <Empty icon={<Plugs size={16} />}>No messages yet.</Empty>
            ) : (
              <ul className="divide-y divide-line">
                {[...ov.recent].reverse().map((r) => (
                  <li key={r.id}>
                    <button onClick={() => onMessage(r.id)} className="flex w-full items-start gap-2.5 px-4 py-2.5 text-left transition-colors hover:bg-white/[0.025]">
                      <Tag tone={ACK_TONE[r.ack]} className="mt-px shrink-0 font-mono">
                        {r.ack}
                      </Tag>
                      <div className="min-w-0 flex-1">
                        <div className="flex items-baseline gap-2">
                          <span className="font-mono text-xs text-ink">{r.type ?? "?"}</span>
                          <span className="truncate text-xs text-dim">{r.interface}</span>
                          <span className="ml-auto shrink-0 text-xs text-dim">{ago(r.at, now)}</span>
                        </div>
                        <div className="truncate text-xs text-muted">{r.summary}</div>
                      </div>
                    </button>
                  </li>
                ))}
              </ul>
            )}
          </Panel>
        </div>

        <div className="min-w-0 space-y-4">
          <Queue title="Quarantined feeds" tone="high" rows={ov?.quarantined ?? null} empty="Every adopted feed passes its live evidence." onMapping={onMapping} />
          <Queue title="Waiting for review" tone="med" rows={ov?.review ?? null} empty="Nothing waits for a person." onMapping={onMapping} />
          <Drills sim={sim} may={caps.has("interop:configure") || caps.has("drill:device")} role={role} onDone={reloadSim} />
          <Validation bench={bench} audit={audit} />
        </div>
      </div>
    </div>
  );
}

function Queue({ title, tone, rows, empty, onMapping }: { title: string; tone: "high" | "med"; rows: MappingRow[] | null; empty: string; onMapping: (id: string) => void }) {
  return (
    <Panel title={title} meta={rows ? `${rows.length}` : undefined} flush>
      {!rows ? (
        <div className="p-3.5">
          <Skeleton className="h-16" />
        </div>
      ) : !rows.length ? (
        <Empty>{empty}</Empty>
      ) : (
        <ul className="divide-y divide-line">
          {rows.map((m) => (
            <li key={m.id}>
              <button onClick={() => onMapping(m.id)} className="w-full px-4 py-3 text-left transition-colors hover:bg-white/[0.025]">
                <div className="flex items-center gap-2">
                  <span className={cx("h-1.5 w-1.5 shrink-0 rounded-full", tone === "high" ? "bg-high" : "bg-med")} />
                  <span className="font-mono text-xs text-ink">{m.code}</span>
                  <span className="min-w-0 truncate text-xs text-muted">{m.text}</span>
                  <span className="ml-auto shrink-0 text-xs text-dim">{m.interface}</span>
                </div>
                <div className="mt-1 line-clamp-2 text-xs leading-snug text-dim first-letter:uppercase">{m.fault?.cause ?? m.reason}</div>
                {m.fault?.suggested && <div className="mt-1 text-xs text-low-ink">Fix found: read as {m.fault.suggested.label}</div>}
                {m.fault?.swap && <div className="mt-1 text-xs text-low-ink">Fix found: {m.fault.swap.label}</div>}
              </button>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}

function Drills({ sim, may, role, onDone }: { sim: SimStatus | null; may: boolean; role: string | null; onDone: () => void }) {
  const [busy, setBusy] = useState<string | null>(null);
  const run = async (kind: string, on: boolean) => {
    setBusy(kind);
    try {
      await api("/aegis/interop/sim/drill", { body: { kind, on } });
      onDone();
    } finally {
      setBusy(null);
    }
  };
  return (
    <Panel title="Fault drills" meta="Break the simulated hospital's interfaces the way real ones break, and watch the evidence catch it" right={sim && Object.keys(sim.faults).length ? <Button size="sm" variant="quiet" onClick={() => run("clear", true)}>Clear all</Button> : undefined}>
      {!sim ? (
        <Skeleton className="h-24" />
      ) : (
        <>
          {!may && (
            <div className="mb-2.5">
              <Restricted cap="interop:configure" role={role}>
                Drills change what the simulated hospital sends. Switch to the chief (or an attending) to run one.
              </Restricted>
            </div>
          )}
          <ul className="space-y-1.5">
            {sim.drills.map((d) => {
              const on = !!sim.faults[d.kind];
              return (
                <li key={d.kind} className="flex items-center gap-2">
                  <Flask size={13} className={on ? "text-med-ink" : "text-dim"} />
                  <span className={cx("min-w-0 flex-1 text-xs leading-snug", on ? "text-ink" : "text-muted")}>{d.label}</span>
                  <Button size="sm" variant={on ? "danger" : "ghost"} disabled={!may || busy === d.kind} onClick={() => run(d.kind, !on)}>
                    {on ? (
                      <>
                        <Reset size={12} /> Stop
                      </>
                    ) : (
                      "Run"
                    )}
                  </Button>
                </li>
              );
            })}
          </ul>
          <div className="mt-2.5 text-xs text-dim">
            {sim.patients.length} patients fed over HL7 · {sim.sent.toLocaleString()} messages sent · {sim.refused} refused
          </div>
        </>
      )}
    </Panel>
  );
}

function Validation({ bench, audit }: { bench: Bench | null; audit: Audit | null }) {
  const f = audit?.findings ?? [];
  return (
    <Panel title="Validation" meta="Measured on PhysioNet/CinC 2019; protocol registered before the run (docs/INTEROP.md)">
      <div className="space-y-3">
        <div>
          <div className="mb-1 text-xs font-medium text-dim">What the checks found in the benchmark data itself</div>
          {!f.length ? (
            <div className="text-xs text-dim">Audit not loaded.</div>
          ) : (
            <ul className="space-y-1.5 text-xs leading-snug text-muted">
              {f.map((x, i) => (
                <li key={i} className="flex gap-2">
                  <Check size={13} className="mt-0.5 shrink-0 text-low-ink" />
                  <span>
                    {x.kind === "window" ? (
                      <>
                        Hospital {x.site} <span className="font-mono">{x.concept}</span>: {(100 * (x.outside_window ?? 0)).toFixed(1)}% of values impossible as total calcium (ionized calcium in mmol/L recorded under it); holdout {(100 * (x.holdout?.outside_window ?? 0)).toFixed(1)}%
                      </>
                    ) : (
                      <>
                        Hospital {x.site} <span className="font-mono">{x.invariant}</span>: violated in {(100 * (x.rate ?? 0)).toFixed(1)}% of pairs (venous saturations recorded as arterial); holdout {(100 * (x.holdout?.rate ?? 0)).toFixed(1)}%
                      </>
                    )}
                  </span>
                </li>
              ))}
            </ul>
          )}
        </div>
        <div>
          <div className="mb-1 text-xs font-medium text-dim">Silent faults vs the gate (8,062 held-out patients)</div>
          {!bench?.available ? (
            <div className="text-xs text-dim">The registered benchmark has not been published yet.</div>
          ) : (
            <div className="space-y-1.5">
              <div className="flex flex-wrap gap-1.5">
                {Object.entries(bench.verdicts ?? {}).map(([k, v]) => {
                  const pass = typeof v.pass === "boolean" ? v.pass : Object.values(v as Record<string, { pass?: boolean }>).every((x) => x?.pass !== false);
                  return (
                    <Tag key={k} tone={pass ? "low" : "high"}>
                      {k.split("_")[0]} {pass ? "met" : "missed"}
                    </Tag>
                  );
                })}
              </div>
              {bench.posthoc && (
                <div className="well px-3 py-2.5 text-xs leading-snug text-muted">
                  <span className="font-medium text-ink">After the fixes the misses motivated</span> (exploratory, same patients): false quarantines per 100k{" "}
                  {Object.entries(bench.posthoc.h4_per_100k)
                    .map(([k, v]) => `${k.replace("_", " ")} ${v.toFixed(2)}`)
                    .join(" · ")}
                  ; wrong-concept adoptions {(100 * (bench.posthoc.h5.unsafe_wrong_concept?.rate ?? 0)).toFixed(1)}%, wrong-unit {(100 * (bench.posthoc.h5.unsafe_wrong_unit?.rate ?? 0)).toFixed(1)}%.
                </div>
              )}
              <div className="well max-h-56 overflow-auto">
                <table className="w-full min-w-[420px] text-xs">
                  <thead className="sticky top-0 bg-s1 text-left text-2xs text-dim">
                    <tr>
                      <th className="px-2 py-1 font-medium">Fault</th>
                      <th className="px-2 py-1 text-right font-medium">Silent ΔFAR</th>
                      <th className="px-2 py-1 text-right font-medium">Caught after</th>
                      <th className="px-2 py-1 text-right font-medium">Fix ΔAUROC</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(bench.faults ?? []).map((r) => (
                      <tr key={r.id} className="border-t border-line">
                        <td className="px-2 py-1 text-muted" title={r.what}>
                          {r.id}
                        </td>
                        <td className={cx("num px-2 py-1 text-right", Math.abs(r.silent_dfar) > 0.05 ? "text-high-ink" : "text-muted")}>{(100 * r.silent_dfar).toFixed(1)} pt</td>
                        <td className="num px-2 py-1 text-right text-muted">{r.detected ? `${r.faulty_to_quarantine} values` : "missed"}</td>
                        <td className="num px-2 py-1 text-right text-muted">{r.fix_dauroc != null ? r.fix_dauroc.toFixed(4) : "—"}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>
          )}
        </div>
      </div>
    </Panel>
  );
}
