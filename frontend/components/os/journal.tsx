"use client";

import { useMemo, useState } from "react";

import { Search } from "@/components/icons";
import { PageHead } from "@/components/shell";
import { cx, Empty, Panel, Restricted, Skeleton, StatStrip, Tag } from "@/components/ui";
import { useCaps, usePoll } from "@/lib/hooks";
import { clock } from "@/lib/paging";

type Entry = { seq: number; ts: number; facility: string; severity: number; severity_name: string; event: string; uid: string; pid: number; detail: Record<string, unknown>; hash: string };
type Journal = {
  status: {
    boot_id: string;
    seq: number;
    head: string;
    retained: number;
    ring: number;
    chain_ok: boolean;
    verified_window: number;
    verified_age_s: number;
    by_facility: Record<string, number>;
    persisted: { configured: string; sink: string; written: number; dropped: number; supabase_errors: number };
  };
  entries: Entry[];
};

const SEV: Record<string, string> = {
  emerg: "text-high-ink",
  alert: "text-high-ink",
  crit: "text-high-ink",
  err: "text-high-ink",
  warn: "text-med-ink",
  notice: "text-ink",
  info: "text-muted",
  debug: "text-dim",
};

function detailText(d: Record<string, unknown>): string {
  return Object.entries(d ?? {})
    .map(([k, v]) => `${k}=${typeof v === "object" ? JSON.stringify(v) : String(v)}`)
    .join("  ");
}

export function JournalView() {
  const [fac, setFac] = useState("");
  const [q, setQ] = useState("");
  const j = usePoll<Journal>(`/aegis/sys/journal?limit=200${fac ? `&facility=${fac}` : ""}`, 2500);
  const { caps, role, ready } = useCaps();
  const st = j.data?.status;
  const rows = useMemo(() => {
    const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    return [...(j.data?.entries ?? [])]
      .reverse()
      .filter((e) => !terms.length || terms.every((t) => `${e.facility} ${e.event} ${e.uid} ${detailText(e.detail)}`.toLowerCase().includes(t)));
  }, [j.data, q]);
  const facilities = Object.entries(st?.by_facility ?? {}).sort((a, b) => b[1] - a[1]);

  return (
    <div>
      <PageHead
        title="Journal"
        right={
          st && (
            <Tag tone={st.chain_ok ? "low" : "high"} dot>
              {st.chain_ok ? `Chain verified · ${st.verified_window} entries` : "Chain broken"}
            </Tag>
          )
        }
      >
        The kernel&apos;s hash-chained log. Each entry commits to the one before it, so an edited or deleted entry breaks the chain, and the chain is
        re-verified continuously.
      </PageHead>

      {ready && !caps.has("view:audit") && (
        <div className="mb-3">
          <Restricted cap="view:audit" role={role} />
        </div>
      )}
      <StatStrip
        className="mb-5"
        items={[
          { k: "Sequence", v: st ? `#${st.seq.toLocaleString()}` : "—", sub: "entries since boot" },
          { k: "Head", v: st ? <span className="font-mono text-base tracking-normal">{st.head}</span> : "—", sub: "hash of the newest entry" },
          { k: "Verified", v: st ? `${st.verified_window}` : "—", sub: st ? `${st.verified_age_s.toFixed(1)} s ago` : "" },
          { k: "Retained", v: st ? `${st.retained}/${st.ring}` : "—", sub: "ring in memory" },
          { k: "Sink", v: st ? st.persisted.sink : "—", sub: st ? `${st.persisted.written.toLocaleString()} written, ${st.persisted.dropped} dropped` : "" },
          { k: "Boot", v: st ? <span className="font-mono text-base tracking-normal">{st.boot_id}</span> : "—", sub: "boot identifier" },
        ]}
      />

      <Panel
        title="Entries"
        meta={j.data ? `${rows.length} shown${fac ? ` · facility ${fac}` : ""}` : undefined}
        right={
          <label className="field flex h-7 items-center gap-1.5 px-2 text-dim">
            <Search size={14} />
            <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Filter" className="w-28 bg-transparent text-xs text-ink outline-none placeholder:text-faint sm:w-44" aria-label="Filter entries" />
          </label>
        }
        flush
      >
        <div className="flex flex-wrap gap-1 border-b border-line px-4 py-2">
          <button onClick={() => setFac("")} className={cx("h-6 rounded-[5px] px-2 text-xs transition-colors", fac === "" ? "bg-white/[0.08] text-ink" : "text-dim hover:text-ink")}>
            All
          </button>
          {facilities.map(([f, c]) => (
            <button key={f} onClick={() => setFac(f === fac ? "" : f)} className={cx("h-6 rounded-[5px] px-2 font-mono text-xs transition-colors", fac === f ? "bg-white/[0.08] text-ink" : "text-dim hover:text-ink")}>
              {f} <span className="text-faint">{c}</span>
            </button>
          ))}
        </div>
        {!j.data ? (
          <div className="p-4">
            <Skeleton className="h-96" />
          </div>
        ) : rows.length === 0 ? (
          <Empty>No entries match.</Empty>
        ) : (
          <div className="max-h-[calc(100dvh-380px)] min-h-[360px] overflow-auto">
            <table className="w-full min-w-[860px] font-mono text-xs">
              <thead className="sticky top-0 z-[1] bg-s1">
                <tr className="border-b border-line text-left font-sans text-xs text-dim">
                  <th className="px-4 py-2 font-medium">Seq</th>
                  <th className="px-2 py-2 font-medium">Time</th>
                  <th className="px-2 py-2 font-medium">Facility</th>
                  <th className="px-2 py-2 font-medium">Level</th>
                  <th className="px-2 py-2 font-medium">Event</th>
                  <th className="px-2 py-2 font-medium">Detail</th>
                  <th className="px-4 py-2 text-right font-medium">Hash</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((e) => (
                  <tr key={e.seq} className="border-b border-line-soft align-top transition-colors last:border-b-0 hover:bg-s2">
                    <td className="num px-4 py-1.5 text-faint">{e.seq}</td>
                    <td className="num whitespace-nowrap px-2 py-1.5 text-dim">{clock(e.ts)}</td>
                    <td className="px-2 py-1.5 text-muted">{e.facility}</td>
                    <td className={cx("px-2 py-1.5", SEV[e.severity_name] ?? "text-muted")}>{e.severity_name}</td>
                    <td className="whitespace-nowrap px-2 py-1.5 text-ink">{e.event}</td>
                    <td className="max-w-[520px] truncate px-2 py-1.5 text-muted" title={detailText(e.detail)}>
                      {detailText(e.detail)}
                    </td>
                    <td className="px-4 py-1.5 text-right text-faint">{e.hash}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}
