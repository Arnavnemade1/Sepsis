"use client";

import { useMemo, useState } from "react";

import { Page, PageHead } from "@/components/shell";
import { Terminal } from "@/components/terminal";
import { cx, Empty, Pill, Plate, Skeleton, Stat } from "@/components/ui";
import { title } from "@/lib/clinical";
import { usePoll } from "@/lib/hooks";
import { clock } from "@/lib/paging";

type Proc = { pid: number; ppid: number; name: string; kind: string; uid: string; state: string; cpu_ms: number; duty: number; uptime_s: number };
type Ps = { processes: Proc[]; status: { processes: number; uptime_s: number; by_kind: Record<string, number>; cpu_ms_total: number } };
type JournalEntry = { seq: number; ts: number; facility: string; severity: number; severity_name: string; event: string; uid: string; detail: Record<string, unknown>; hash: string };
type Journal = {
  status: { boot_id: string; seq: number; chain_ok: boolean; verified_window: number; by_facility: Record<string, number>; persisted: { sink: string; written: number; dropped: number } };
  entries: JournalEntry[];
};
type Boot = { booted: boolean; boot_seconds: number; phases: number; failed: unknown[]; slowest: { phase: string; ms: number }; python: string; platform: string; log: { phase: string; summary: string; ms: number; ok: boolean }[] };
type Gate = { mode: string; calls: number; denied: number; would_deny: number; overhead_us: { p50: number; p99: number }; abi: { syscalls: number; by_class: Record<string, number> } };
type Devices = { devices: { key: string; label: string; path: string; state: string; active: string; faults: number; failovers: number; verified: boolean; max_steps: number; last_steps: number }[] };
type Services = { services: { name: string; version: string; state: string; faults: number; restarts: number; healthy: boolean; last_ms: number; caps: { reads: number; emits: string[]; lanes: number } }[] };

export default function OsPage() {
  const ps = usePoll<Ps>("/aegis/sys/ps", 3000);
  const [fac, setFac] = useState("");
  const journal = usePoll<Journal>(`/aegis/sys/journal?limit=60${fac ? `&facility=${fac}` : ""}`, 3000);
  const boot = usePoll<Boot>("/aegis/sys/boot");
  const gate = usePoll<Gate>("/aegis/sys/gate", 5000);
  const devices = usePoll<Devices>("/aegis/kernel/devices", 5000);
  const services = usePoll<Services>("/aegis/services", 5000);

  const procs = useMemo(() => [...(ps.data?.processes ?? [])].sort((a, b) => b.cpu_ms - a.cpu_ms), [ps.data]);
  const facilities = Object.keys(journal.data?.status.by_facility ?? {});

  return (
    <Page>
      <PageHead title="System">
        The kernel behind the console: process table, hash-chained journal, syscall gate, boot record and verified device drivers. Every request
        from this application passes through the same gate.
      </PageHead>

      <div className="mb-3 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-5">
        <Tile loading={!boot.data}>
          {boot.data && (
            <Stat label="Boot" value={`${boot.data.boot_seconds.toFixed(1)}`} unit="s" sub={`${boot.data.phases} phases · ${boot.data.failed.length} failed`} tone={boot.data.failed.length ? "high" : "neutral"} />
          )}
        </Tile>
        <Tile loading={!ps.data}>
          {ps.data && (
            <Stat
              label="Processes"
              value={ps.data.status.processes}
              sub={Object.entries(ps.data.status.by_kind).filter(([, n]) => n).map(([k, n]) => `${n} ${k}`).join(" · ")}
            />
          )}
        </Tile>
        <Tile loading={!gate.data}>
          {gate.data && (
            <Stat
              label="Syscall gate"
              value={title(gate.data.mode)}
              sub={`${gate.data.abi.syscalls} syscalls · p99 ${gate.data.overhead_us.p99.toFixed(0)} µs`}
              tone={gate.data.mode === "enforce" ? "neutral" : "med"}
            />
          )}
        </Tile>
        <Tile loading={!journal.data}>
          {journal.data && (
            <Stat
              label="Journal"
              value={`#${journal.data.status.seq}`}
              sub={`${journal.data.status.chain_ok ? "chain verified" : "CHAIN BROKEN"} · ${journal.data.status.persisted.sink}`}
              tone={journal.data.status.chain_ok ? "neutral" : "high"}
            />
          )}
        </Tile>
        <Tile loading={!devices.data}>
          {devices.data && (
            <Stat
              label="Verified drivers"
              value={`${devices.data.devices.filter((d) => d.verified).length}/${devices.data.devices.length}`}
              sub={`${devices.data.devices.reduce((n, d) => n + d.faults, 0)} faults · ${devices.data.devices.reduce((n, d) => n + d.failovers, 0)} failovers`}
            />
          )}
        </Tile>
      </div>

      <div className="grid gap-3 xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <Plate title="Shell" eyebrow="Every command is a syscall; the gate decides what you may read">
          <Terminal />
        </Plate>
        <Plate
          title="Journal"
          eyebrow="Hash-chained kernel log"
          right={
            <select
              value={fac}
              onChange={(e) => setFac(e.target.value)}
              className="rounded-md border border-line bg-s2 px-2 py-1 text-[12px] text-ink"
              aria-label="Filter by facility"
            >
              <option value="">all facilities</option>
              {facilities.map((f) => (
                <option key={f} value={f}>
                  {f}
                </option>
              ))}
            </select>
          }
          bodyClass="p-0"
        >
          {!journal.data ? (
            <div className="p-4">
              <Skeleton className="h-96" />
            </div>
          ) : (
            <ul className="h-[472px] divide-y divide-line-soft overflow-y-auto font-mono text-[11.5px]">
              {[...journal.data.entries].reverse().map((e) => (
                <li key={e.seq} className="flex gap-3 px-4 py-1.5">
                  <span className="num shrink-0 text-faint">{clock(e.ts)}</span>
                  <span className={cx("w-16 shrink-0", e.severity <= 3 ? "text-high-ink" : e.severity <= 4 ? "text-med-ink" : "text-dim")}>{e.facility}</span>
                  <span className="shrink-0 text-ink">{e.event}</span>
                  <span className="min-w-0 truncate text-muted">{summarize(e.detail)}</span>
                </li>
              ))}
            </ul>
          )}
        </Plate>
      </div>

      <div className="mt-3 grid gap-3 xl:grid-cols-[minmax(0,1.25fr)_minmax(0,1fr)]">
        <Plate title="Process table" eyebrow="Sorted by CPU time" bodyClass="p-0">
          {!ps.data ? (
            <div className="p-4">
              <Skeleton className="h-64" />
            </div>
          ) : (
            <div className="max-h-[420px] overflow-auto">
              <table className="w-full min-w-[620px] text-[12px]">
                <thead className="sticky top-0 bg-s1 text-left text-[11px] text-dim">
                  <tr>
                    {["PID", "Name", "Kind", "Owner", "State", "CPU ms", "Duty"].map((h) => (
                      <th key={h} className="px-4 py-2 font-normal">
                        {h}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="font-mono text-[11.5px]">
                  {procs.map((p) => (
                    <tr key={p.pid} className="border-t border-line-soft">
                      <td className="num px-4 py-1.5 text-dim">{p.pid}</td>
                      <td className="px-4 py-1.5 text-ink">{p.name}</td>
                      <td className="px-4 py-1.5 text-muted">{p.kind}</td>
                      <td className="px-4 py-1.5 text-muted">{p.uid}</td>
                      <td className={cx("px-4 py-1.5", p.state === "RUNNING" ? "text-ink" : p.state === "FAULTED" ? "text-high-ink" : "text-dim")}>{p.state}</td>
                      <td className="num px-4 py-1.5 text-muted">{p.cpu_ms.toFixed(1)}</td>
                      <td className="num px-4 py-1.5 text-muted">{(p.duty * 100).toFixed(1)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Plate>

        <div className="flex min-w-0 flex-col gap-3">
          <Plate title="Boot record" eyebrow={boot.data ? `${boot.data.platform} · Python ${boot.data.python}` : "Measured, not scripted"}>
            {!boot.data ? (
              <Skeleton className="h-40" />
            ) : (
              <ul className="space-y-1.5">
                {boot.data.log.map((p) => {
                  const max = Math.max(...boot.data!.log.map((x) => x.ms), 1);
                  return (
                    <li key={p.phase} className="grid grid-cols-[92px_1fr_64px] items-center gap-3 text-[12px]">
                      <span className={cx("truncate font-mono text-[11.5px]", p.ok ? "text-ink" : "text-high-ink")}>{p.phase}</span>
                      <span className="relative h-1.5 overflow-hidden rounded-full bg-s3" title={p.summary}>
                        <span className={cx("absolute inset-y-0 left-0 rounded-full", p.ok ? "bg-muted" : "bg-high")} style={{ width: `${Math.max(1.5, (p.ms / max) * 100)}%` }} />
                      </span>
                      <span className="num text-right text-muted">{p.ms.toFixed(1)} ms</span>
                    </li>
                  );
                })}
              </ul>
            )}
          </Plate>

          <Plate title="Services and drivers" eyebrow="aegisd services · eBPF-verified device drivers">
            {!services.data || !devices.data ? (
              <Skeleton className="h-32" />
            ) : (
              <div className="space-y-4">
                <ul className="space-y-1.5">
                  {services.data.services.map((s) => (
                    <li key={s.name} className="flex items-center gap-3 text-[12.5px]">
                      <Pill tone={s.healthy ? "neutral" : "high"} dot live={s.state === "RUNNING"}>
                        {s.name}
                      </Pill>
                      <span className="text-dim">v{s.version}</span>
                      <span className="ml-auto text-[11.5px] text-muted">
                        {s.caps.reads} reads · {s.caps.lanes} lanes · {s.restarts} restarts
                      </span>
                    </li>
                  ))}
                </ul>
                {devices.data.devices.length === 0 ? (
                  <Empty>No drivers bound.</Empty>
                ) : (
                  <ul className="space-y-1.5 border-t border-line-soft pt-3">
                    {devices.data.devices.map((d) => (
                      <li key={d.path} className="flex items-center gap-3 text-[12px]">
                        <span className="min-w-0 truncate font-mono text-[11.5px] text-ink">{d.path}</span>
                        <span className="text-dim">{d.label}</span>
                        <span className={cx("ml-auto shrink-0", d.state === "RUNNING" ? "text-muted" : "text-dys-ink")}>
                          {d.verified ? `verified · ≤${d.max_steps} steps` : "UNVERIFIED"} · {d.state.toLowerCase()}
                        </span>
                      </li>
                    ))}
                  </ul>
                )}
              </div>
            )}
          </Plate>
        </div>
      </div>
    </Page>
  );
}

function Tile({ loading, children }: { loading: boolean; children: React.ReactNode }) {
  return <div className="panel px-3.5 py-3">{loading ? <Skeleton className="h-[58px]" /> : children}</div>;
}

function summarize(d: Record<string, unknown>): string {
  return Object.entries(d ?? {})
    .slice(0, 4)
    .map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : String(v)}`)
    .join(" ");
}
