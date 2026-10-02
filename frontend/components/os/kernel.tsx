"use client";

import { useMemo, useState } from "react";

import { cx, Empty, Panel, Segmented, Skeleton, Tag } from "@/components/ui";
import { usePoll } from "@/lib/hooks";

type Proc = { pid: number; ppid: number; name: string; kind: string; uid: string; state: string; cpu_ms: number; slices: number; last_ms: number; max_ms: number; duty: number; uptime_s: number; signals: string[] };
type Ps = { processes: Proc[]; status: { processes: number; uptime_s: number; by_kind: Record<string, number>; cpu_ms_total: number } };
type ProcTree = {
  tree: {
    "/proc/sched"?: { tick: number; mode: string; owner: string; procs: number; tiers: Record<string, number>; lane_budget: number; lanes_used: number; interrupts: number; anticipating: number; global_cost: number };
    "/proc/mem"?: { capacity_cells: number; used_cells: number; utilization: number; pointer_stable: boolean; bytes: number; base_pointer: string };
    "/proc/executor"?: { calls: number; busy: boolean; max_wait_ms: number; max_run_ms: number; violations: number; skipped_ticks: number };
    "/proc/ksched"?: { uptime_s: number; processes: number; by_state: Record<string, number>; cpu_ms_total: number };
  };
};
type Devices = {
  devices: { key: string; label: string; path: string; state: string; active: string; faults: number; failovers: number; restores: number; blocks: number; verified: boolean; max_steps: number; last_steps: number }[];
  supervisor: { healthy: number; degraded: number; failovers: number };
  programs: { name: string; description: string }[];
};
type Services = {
  services: { name: string; version: string; state: string; faults: number; restarts: number; ticks: number; healthy: boolean; last_ms: number; caps: { reads: number; emits: string[]; lanes: number } }[];
  events: { tick: number; kind: string; svc: string; detail: string; ts: number }[];
};
type Boot = { boot_seconds: number; phases: number; failed: unknown[]; python: string; platform: string; log: { phase: string; summary: string; ms: number; ok: boolean }[] };

const STATE_TONE: Record<string, string> = {
  RUNNING: "text-low-ink",
  READY: "text-ink",
  SLEEPING: "text-dim",
  ZOMBIE: "text-faint",
  FAULTED: "text-high-ink",
  STOPPED: "text-med-ink",
  DONE: "text-dim",
};

function fmtUptime(s: number): string {
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

export function Kernel() {
  const ps = usePoll<Ps>("/aegis/sys/ps", 3000);
  const proc = usePoll<ProcTree>("/aegis/proc", 4000);
  const devices = usePoll<Devices>("/aegis/kernel/devices", 6000);
  const services = usePoll<Services>("/aegis/services", 6000);
  const boot = usePoll<Boot>("/aegis/sys/boot");
  const [kind, setKind] = useState("all");

  const t = proc.data?.tree ?? {};
  const sched = t["/proc/sched"];
  const mem = t["/proc/mem"];
  const ex = t["/proc/executor"];
  const ks = t["/proc/ksched"];
  const total = ps.data?.status.cpu_ms_total ?? 0;
  const procs = useMemo(
    () => [...(ps.data?.processes ?? [])].filter((p) => kind === "all" || p.kind === kind).sort((a, b) => b.cpu_ms - a.cpu_ms),
    [ps.data, kind],
  );
  const kinds = Object.entries(ps.data?.status.by_kind ?? {}).filter(([, v]) => v > 0);

  const kpis: { k: string; v: string; sub: string }[] = [
    { k: "Uptime", v: ks ? fmtUptime(ks.uptime_s) : "—", sub: boot.data ? `booted in ${boot.data.boot_seconds.toFixed(1)} s` : "" },
    { k: "Processes", v: ks ? String(ks.processes) : "—", sub: ks ? `${ks.by_state.RUNNING ?? 0} running · ${ks.by_state.READY ?? 0} ready` : "" },
    { k: "CPU time", v: ks ? `${(ks.cpu_ms_total / 1000).toFixed(1)} s` : "—", sub: "accounted since boot" },
    { k: "Scheduler", v: sched ? sched.mode : "—", sub: sched ? `owner ${sched.owner} · tick ${sched.tick}` : "" },
    { k: "Memory fabric", v: mem ? `${(mem.utilization * 100).toFixed(1)}%` : "—", sub: mem ? `${mem.used_cells.toLocaleString()} of ${mem.capacity_cells.toLocaleString()} cells` : "" },
    { k: "Executor", v: ex ? String(ex.calls) : "—", sub: ex ? `max wait ${ex.max_wait_ms.toFixed(1)} ms · ${ex.violations} violations` : "" },
  ];

  const bootLog = boot.data?.log ?? [];
  const bootTotal = bootLog.reduce((a, p) => a + p.ms, 0) || 1;
  let acc = 0;

  return (
    <div>
      <div className="mb-5">
        <h1 className="text-[20px] font-semibold tracking-[-0.015em] text-ink">Kernel</h1>
        <p className="mt-1.5 max-w-3xl text-[13px] leading-relaxed text-muted">
          The process table with real CPU accounting, the cognitive scheduler, the memory fabric, verified device drivers and supervised services.
        </p>
      </div>

      <div className="mb-3 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-3 xl:grid-cols-6">
        {kpis.map((it) => (
          <div key={it.k} className="min-w-0 bg-s1 px-3.5 py-2.5">
            <div className="truncate text-[11px] text-dim">{it.k}</div>
            <div className="num mt-0.5 truncate font-mono text-[18px] font-medium text-ink">{it.v}</div>
            <div className="truncate text-[11px] text-dim">{it.sub}</div>
          </div>
        ))}
      </div>

      <div className="grid gap-3 xl:grid-cols-[minmax(0,1.45fr)_minmax(0,1fr)]">
        <Panel
          title="Processes"
          meta={ps.data ? `${procs.length} shown · sorted by CPU time` : undefined}
          right={
            <Segmented
              size="sm"
              value={kind}
              onChange={setKind}
              options={[{ value: "all", label: "All" }, ...kinds.map(([k, v]) => ({ value: k, label: `${k} ${v}` }))]}
            />
          }
          flush
        >
          {!ps.data ? (
            <div className="p-3.5">
              <Skeleton className="h-80" />
            </div>
          ) : (
            <div className="max-h-[560px] overflow-auto">
              <table className="w-full min-w-[720px] text-[12px]">
                <thead className="sticky top-0 z-[1] bg-s1">
                  <tr className="border-b border-line text-left text-[11px] text-dim">
                    <th className="px-3.5 py-2 font-medium">PID</th>
                    <th className="px-2 py-2 font-medium">Name</th>
                    <th className="px-2 py-2 font-medium">Kind</th>
                    <th className="px-2 py-2 font-medium">Owner</th>
                    <th className="px-2 py-2 font-medium">State</th>
                    <th className="px-2 py-2 text-right font-medium">CPU</th>
                    <th className="w-[140px] px-3.5 py-2 font-medium">Share</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-[11.5px]">
                  {procs.map((p) => {
                    const share = total > 0 ? p.cpu_ms / total : 0;
                    return (
                      <tr key={p.pid} className="border-b border-line-soft transition-colors last:border-b-0 hover:bg-s2">
                        <td className="num px-3.5 py-1.5 text-dim">{p.pid}</td>
                        <td className="max-w-[220px] truncate px-2 py-1.5 text-ink">{p.name}</td>
                        <td className="px-2 py-1.5 text-muted">{p.kind}</td>
                        <td className="px-2 py-1.5 text-muted">{p.uid}</td>
                        <td className={cx("px-2 py-1.5", STATE_TONE[p.state] ?? "text-muted")}>{p.state.toLowerCase()}</td>
                        <td className="num px-2 py-1.5 text-right text-muted">{p.cpu_ms >= 1000 ? `${(p.cpu_ms / 1000).toFixed(2)} s` : `${p.cpu_ms.toFixed(1)} ms`}</td>
                        <td className="px-3.5 py-1.5">
                          <div className="flex items-center gap-2">
                            <div className="h-1 flex-1 overflow-hidden rounded-[2px] bg-s3">
                              <div className="h-full bg-accent/80" style={{ width: `${Math.max(share > 0 ? 1.5 : 0, share * 100)}%` }} />
                            </div>
                            <span className="num w-10 text-right text-[11px] text-dim">{(share * 100).toFixed(1)}%</span>
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          )}
        </Panel>

        <div className="flex min-w-0 flex-col gap-3">
          <Panel title="Scheduler" meta="Beds compete for compute lanes; invariants proven in Z3" right={sched && <Tag>{sched.mode}</Tag>}>
            {!sched ? (
              <Skeleton className="h-28" />
            ) : (
              <div className="space-y-3">
                <div>
                  <div className="mb-1.5 flex items-center justify-between text-[11.5px]">
                    <span className="text-dim">Beds by tier</span>
                    <span className="num font-mono text-dim">{sched.procs} beds</span>
                  </div>
                  <div className="flex h-2 overflow-hidden rounded-[2px] bg-s3">
                    {[
                      ["CRITICAL", "#e5484d"],
                      ["WATCH", "#eeb12a"],
                      ["STABLE", "#4b525b"],
                    ].map(([k, c]) => (
                      <div key={k} style={{ width: `${((sched.tiers[k] ?? 0) / Math.max(1, sched.procs)) * 100}%`, background: c }} title={`${k.toLowerCase()} ${sched.tiers[k] ?? 0}`} />
                    ))}
                  </div>
                  <div className="mt-1.5 flex gap-4 text-[11.5px]">
                    <span className="text-high-ink">{sched.tiers.CRITICAL ?? 0} critical</span>
                    <span className="text-med-ink">{sched.tiers.WATCH ?? 0} watch</span>
                    <span className="text-dim">{sched.tiers.STABLE ?? 0} stable</span>
                  </div>
                </div>
                <div>
                  <div className="mb-1.5 flex items-center justify-between text-[11.5px]">
                    <span className="text-dim">Compute lanes</span>
                    <span className="num font-mono text-dim">
                      {sched.lanes_used}/{sched.lane_budget}
                    </span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded-[2px] bg-s3">
                    <div className="h-full bg-accent/80" style={{ width: `${(sched.lanes_used / Math.max(1, sched.lane_budget)) * 100}%` }} />
                  </div>
                </div>
                <div className="grid grid-cols-3 gap-px overflow-hidden rounded-md border border-line bg-line text-[11.5px]">
                  {[
                    ["Owner", sched.owner],
                    ["Interrupts", sched.interrupts],
                    ["Anticipating", sched.anticipating],
                  ].map(([k, v]) => (
                    <div key={String(k)} className="bg-s1 px-2.5 py-1.5">
                      <div className="text-dim">{k}</div>
                      <div className="num font-mono text-ink">{v}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </Panel>

          <Panel
            title="Device drivers"
            meta="/dev/bio · statically verified before they run"
            right={devices.data && <span className="num font-mono text-[11.5px] text-dim">{devices.data.devices.filter((d) => d.verified).length}/{devices.data.devices.length} verified</span>}
            flush
          >
            {!devices.data ? (
              <div className="p-3.5">
                <Skeleton className="h-20" />
              </div>
            ) : devices.data.devices.length === 0 ? (
              <Empty>No drivers bound.</Empty>
            ) : (
              <ul>
                {devices.data.devices.map((d) => (
                  <li key={d.path} className="border-b border-line-soft px-3.5 py-2 last:border-b-0">
                    <div className="flex items-center gap-2">
                      <span className={cx("h-1.5 w-1.5 rounded-full", d.state === "RUNNING" ? "bg-low" : "bg-dys")} />
                      <span className="min-w-0 truncate font-mono text-[11.5px] text-ink">{d.path}</span>
                      <span className="ml-auto shrink-0 text-[11px] text-dim">{d.label}</span>
                    </div>
                    <div className="mt-0.5 pl-3.5 font-mono text-[11px] text-dim">
                      {d.verified ? `verified · ≤${d.max_steps} steps (last ${d.last_steps})` : "UNVERIFIED"} · {d.blocks} blocks · {d.faults} faults · {d.failovers} failovers
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </Panel>

          <Panel title="Services" meta="aegisd: capability-confined, supervised" flush>
            {!services.data ? (
              <div className="p-3.5">
                <Skeleton className="h-20" />
              </div>
            ) : (
              <ul>
                {services.data.services.map((s) => (
                  <li key={s.name} className="flex items-center gap-2.5 border-b border-line-soft px-3.5 py-2 last:border-b-0">
                    <span className={cx("h-1.5 w-1.5 rounded-full", s.healthy && s.state === "RUNNING" ? "bg-low" : "bg-high")} />
                    <span className="font-mono text-[12px] text-ink">{s.name}</span>
                    <span className="font-mono text-[11px] text-dim">v{s.version}</span>
                    <span className="ml-auto font-mono text-[11px] text-dim">
                      {s.caps.reads} reads · {s.caps.lanes} lanes · {s.restarts} restarts · {s.last_ms.toFixed(1)} ms
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </Panel>
        </div>
      </div>

      <Panel
        title="Boot record"
        meta={boot.data ? `${boot.data.phases} phases in ${boot.data.boot_seconds.toFixed(2)} s · ${boot.data.platform} · Python ${boot.data.python}` : "Measured, not scripted"}
        className="mt-3"
      >
        {!boot.data ? (
          <Skeleton className="h-40" />
        ) : (
          <div className="space-y-1">
            {bootLog.map((p) => {
              const left = (acc / bootTotal) * 100;
              acc += p.ms;
              return (
                <div key={p.phase} className="grid grid-cols-[110px_minmax(0,1fr)_72px] items-center gap-3 text-[11.5px]" title={p.summary}>
                  <span className={cx("truncate font-mono", p.ok ? "text-muted" : "text-high-ink")}>{p.phase}</span>
                  <span className="relative h-3">
                    <span className="absolute inset-y-[5px] left-0 right-0 rounded-full bg-s2" />
                    <span
                      className={cx("absolute inset-y-0.5 rounded-[2px]", p.ok ? "bg-accent/75" : "bg-high")}
                      style={{ left: `${left}%`, width: `${Math.max(0.4, (p.ms / bootTotal) * 100)}%` }}
                    />
                  </span>
                  <span className="num text-right font-mono text-dim">{p.ms >= 100 ? `${p.ms.toFixed(0)} ms` : `${p.ms.toFixed(1)} ms`}</span>
                </div>
              );
            })}
          </div>
        )}
      </Panel>
    </div>
  );
}
