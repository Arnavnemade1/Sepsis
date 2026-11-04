"use client";

import { useMemo, useState } from "react";

import { Check, Close, Search } from "@/components/icons";
import { PageHead } from "@/components/shell";
import { cx, Empty, Panel, Restricted, Segmented, Skeleton, StatStrip, Tag } from "@/components/ui";
import { useCaps, usePoll } from "@/lib/hooks";

type SyscallStat = { nr: number; syscall: string; class: string; cap: string; calls: number; denied: number; would_deny: number; p50_us: number };
type Gate = {
  mode: string;
  calls: number;
  denied: number;
  would_deny: number;
  overhead_us: { p50: number; p99: number; samples: number; measures: string };
  abi: { syscalls: number; by_class: Record<string, number> };
  busiest: SyscallStat[];
  refusals: SyscallStat[];
  policy: { sound: boolean; properties: { property: string; proven: boolean }[]; ms: number };
  coverage: { live: number; entries: number };
};
type Abi = { syscalls: { nr: number; name: string; method: string; path: string; cap: string; class: string }[]; coverage: { complete: boolean; missing: string[]; stale: string[] } };
type Who = {
  sessions: { sid: string; username: string; role: string; created: number; idle_s: number; expires_in_s: number; pid: number; locked: boolean; client: string; breakglass: unknown }[];
  status: { accounts: number; sessions: number; demo_login: boolean; breakglass_active: number; breakglass_unreviewed: number; secret: string };
};

const CLASS_TONE: Record<string, string> = { read: "text-muted", write: "text-dys-ink", egress: "text-high-ink", compute: "text-accent-ink", auth: "text-med-ink", stream: "text-low-ink" };

function dur(s: number): string {
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.round(s / 60)}m`;
  return `${(s / 3600).toFixed(1)}h`;
}

export function GateView() {
  const gate = usePoll<Gate>("/aegis/sys/gate", 4000);
  const abi = usePoll<Abi>("/aegis/sys/abi");
  const { caps, role, ready } = useCaps();
  const mayWho = caps.has("view:sessions");
  const who = usePoll<Who>(ready && mayWho ? "/aegis/sys/who" : null, 8000);
  const [q, setQ] = useState("");
  const [cls, setCls] = useState("all");
  const g = gate.data;
  const stats = useMemo(() => {
    const m: Record<number, SyscallStat> = {};
    for (const s of [...(g?.busiest ?? []), ...(g?.refusals ?? [])]) m[s.nr] = s;
    return m;
  }, [g]);
  const rows = useMemo(() => {
    const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    return (abi.data?.syscalls ?? []).filter(
      (s) => (cls === "all" || s.class === cls) && (!terms.length || terms.every((t) => `${s.nr} ${s.name} ${s.path} ${s.cap} ${s.method}`.toLowerCase().includes(t))),
    );
  }, [abi.data, q, cls]);

  return (
    <div>
      <PageHead
        title="Syscall gate"
        right={
          g && (
            <Tag tone={g.mode === "enforce" ? "low" : "med"} dot>
              {g.mode === "enforce" ? "Enforcing" : "Audit mode: refusals recorded, not applied"}
            </Tag>
          )
        }
      >
        One reference monitor every request crosses. Each route is a numbered syscall bound to a capability; the gate resolves it, authenticates the
        session and decides, in microseconds, before the handler runs.
      </PageHead>

      <StatStrip
        className="mb-5"
        items={[
          { k: "Mode", v: g ? g.mode.charAt(0).toUpperCase() + g.mode.slice(1) : "—", sub: "audit, enforce or off" },
          { k: "Calls", v: g ? g.calls.toLocaleString() : "—", sub: "since boot" },
          { k: "Would deny", v: g ? String(g.would_deny) : "—", sub: g ? `${g.denied} actually denied` : "" },
          { k: "Overhead p50", v: g ? `${g.overhead_us.p50.toFixed(1)} µs` : "—", sub: "resolve, authenticate, decide" },
          { k: "Overhead p99", v: g ? `${g.overhead_us.p99.toFixed(1)} µs` : "—", sub: g ? `${g.overhead_us.samples.toLocaleString()} samples` : "" },
          { k: "Coverage", v: g ? `${g.coverage.live}/${g.coverage.entries}` : "—", sub: "live routes with an entry" },
        ]}
      />

      <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <Panel
          title="Syscall table"
          meta={abi.data ? `${rows.length} of ${abi.data.syscalls.length} · ${abi.data.coverage.complete ? "coverage complete" : "coverage incomplete"}` : undefined}
          right={
            <label className="field flex h-7 items-center gap-1.5 px-2 text-dim">
              <Search size={14} />
              <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Search syscalls" className="w-28 bg-transparent text-xs text-ink outline-none placeholder:text-faint sm:w-40" aria-label="Search syscalls" />
            </label>
          }
          flush
        >
          <div className="border-b border-line px-4 py-2">
            <Segmented
              size="sm"
              value={cls}
              onChange={setCls}
              options={[{ value: "all", label: "All" }, ...Object.entries(g?.abi.by_class ?? {}).map(([k, v]) => ({ value: k, label: `${k} ${v}` }))]}
            />
          </div>
          {!abi.data ? (
            <div className="p-4">
              <Skeleton className="h-96" />
            </div>
          ) : rows.length === 0 ? (
            <Empty>No syscalls match.</Empty>
          ) : (
            <div className="max-h-[620px] overflow-auto">
              <table className="w-full min-w-[700px] text-xs">
                <thead className="sticky top-0 z-[1] bg-s1">
                  <tr className="border-b border-line text-left text-xs text-dim">
                    <th className="px-4 py-2 font-medium">Nr</th>
                    <th className="px-2 py-2 font-medium">Syscall</th>
                    <th className="px-2 py-2 font-medium">Route</th>
                    <th className="px-2 py-2 font-medium">Class</th>
                    <th className="px-2 py-2 font-medium">Capability</th>
                    <th className="px-4 py-2 text-right font-medium">Calls</th>
                  </tr>
                </thead>
                <tbody className="font-mono text-xs">
                  {rows.map((s) => (
                    <tr key={s.nr} className="border-b border-line-soft transition-colors last:border-b-0 hover:bg-s2">
                      <td className="num px-4 py-1.5 text-faint">{s.nr}</td>
                      <td className="px-2 py-1.5 text-ink">{s.name}</td>
                      <td className="max-w-[230px] truncate px-2 py-1.5 text-muted" title={`${s.method} ${s.path}`}>
                        <span className="text-dim">{s.method}</span> {s.path}
                      </td>
                      <td className={cx("px-2 py-1.5", CLASS_TONE[s.class] ?? "text-muted")}>{s.class}</td>
                      <td className="px-2 py-1.5 text-muted">{s.cap}</td>
                      <td className="num px-4 py-1.5 text-right text-dim">{stats[s.nr]?.calls ?? ""}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Panel>

        <div className="flex min-w-0 flex-col gap-3">
          <Panel
            title="Access policy"
            meta={g ? `Proved by Z3 against the whole table in ${g.policy.ms.toFixed(0)} ms` : "Proved by Z3 against the whole table"}
            right={g && <Tag tone={g.policy.sound ? "low" : "high"}>{g.policy.sound ? "sound" : "unsound"}</Tag>}
          >
            {!g ? (
              <Skeleton className="h-28" />
            ) : (
              <ul className="space-y-1.5">
                {g.policy.properties.map((p) => (
                  <li key={p.property} className="flex items-center gap-2 text-sm">
                    {p.proven ? <Check size={14} className="shrink-0 text-low" /> : <Close size={14} className="shrink-0 text-high" />}
                    <span className={p.proven ? "text-ink/90" : "text-high-ink"}>{p.property.replace(/_/g, " ")}</span>
                  </li>
                ))}
              </ul>
            )}
          </Panel>

          <Panel title="Sessions" meta={who.data ? `${who.data.status.sessions} active · ${who.data.status.accounts} accounts · secret ${who.data.status.secret}` : undefined} flush>
            {ready && !mayWho ? (
              <div className="p-4">
                <Restricted cap="view:sessions" role={role}>
                  The session list is not requested for this role.
                </Restricted>
              </div>
            ) : !who.data ? (
              <div className="p-4">
                <Skeleton className="h-20" />
              </div>
            ) : who.data.sessions.length === 0 ? (
              <Empty>No sessions.</Empty>
            ) : (
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-line text-left text-xs text-dim">
                    <th className="px-4 py-2 font-medium">User</th>
                    <th className="px-2 py-2 font-medium">Role</th>
                    <th className="px-2 py-2 font-medium">Idle</th>
                    <th className="px-4 py-2 text-right font-medium">Expires</th>
                  </tr>
                </thead>
                <tbody>
                  {who.data.sessions.map((s) => (
                    <tr key={s.sid} className="border-b border-line-soft last:border-b-0">
                      <td className="px-4 py-1.5">
                        <div className="font-mono text-xs text-ink">{s.username}</div>
                        <div className="font-mono text-2xs text-faint">{s.sid} · {s.client}</div>
                      </td>
                      <td className="px-2 py-1.5 text-muted">{s.role}</td>
                      <td className="num px-2 py-1.5 text-xs text-dim">{dur(s.idle_s)}</td>
                      <td className="num px-4 py-1.5 text-right text-xs text-dim">{dur(s.expires_in_s)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Panel>

          <Panel title="Would-deny record" meta="Calls the gate would refuse in enforce mode" flush>
            {!g ? (
              <div className="p-4">
                <Skeleton className="h-20" />
              </div>
            ) : g.refusals.length === 0 ? (
              <Empty>No refusals recorded.</Empty>
            ) : (
              <ul>
                {g.refusals.map((s) => (
                  <li key={s.nr} className="flex items-center gap-2.5 border-b border-line-soft px-4 py-2 last:border-b-0">
                    <span className="num text-xs text-faint">{s.nr}</span>
                    <span className="font-mono text-xs text-ink">{s.syscall}</span>
                    <span className="font-mono text-xs text-dim">{s.cap}</span>
                    <span className="num ml-auto text-xs text-med-ink">{s.would_deny}</span>
                  </li>
                ))}
              </ul>
            )}
          </Panel>
        </div>
      </div>
    </div>
  );
}
