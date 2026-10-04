"use client";

import { Search } from "lucide-react";
import { useMemo, useState } from "react";

import { cx, Empty, Panel, Segmented, Skeleton, Tag } from "@/components/ui";
import { usePoll } from "@/lib/hooks";
import { ago, proposerLabel, STATUS_LABEL, STATUS_TONE, type MappingRow } from "@/lib/interop";

const FILTERS = ["all", "adopted", "review", "quarantined", "learning", "unmapped"] as const;
type Filter = (typeof FILTERS)[number];

export function MappingsView({ onMapping }: { onMapping: (id: string) => void }) {
  const { data } = usePoll<{ mappings: MappingRow[] }>("/aegis/interop/mappings", 3000);
  const [f, setF] = useState<Filter>("all");
  const [q, setQ] = useState("");
  const [iface, setIface] = useState("all");
  const all = useMemo(() => data?.mappings ?? [], [data]);
  const ifaces = useMemo(() => Array.from(new Set(all.map((m) => m.interface))).sort(), [all]);
  const counts = useMemo(() => {
    const c: Record<string, number> = { all: all.length };
    for (const m of all) c[m.status === "proposing" ? "learning" : m.status === "rejected" ? "unmapped" : m.status] = (c[m.status === "proposing" ? "learning" : m.status === "rejected" ? "unmapped" : m.status] ?? 0) + 1;
    return c;
  }, [all]);
  const rows = useMemo(() => {
    const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    const order: Record<string, number> = { quarantined: 0, review: 1, proposing: 2, learning: 3, adopted: 4, unmapped: 5, rejected: 6 };
    return all
      .filter((m) => f === "all" || m.status === f || (f === "learning" && m.status === "proposing") || (f === "unmapped" && m.status === "rejected"))
      .filter((m) => iface === "all" || m.interface === iface)
      .filter((m) => !terms.length || terms.every((t) => `${m.code} ${m.text} ${m.concept ?? ""} ${m.interface} ${m.unit_written}`.toLowerCase().includes(t)))
      .sort((a, b) => order[a.status] - order[b.status] || a.interface.localeCompare(b.interface) || a.code.localeCompare(b.code));
  }, [all, f, q, iface]);

  return (
    <Panel
      title="Mappings"
      meta={data ? `${rows.length} of ${all.length} local codes · each adopted only on evidence from its own data` : undefined}
      right={
        <label className="flex h-7 items-center gap-1.5 rounded-md border border-line bg-bg px-2 text-dim focus-within:border-line-bright">
          <Search size={13} />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Code, analyte, unit" className="w-28 bg-transparent text-[12px] text-ink outline-none placeholder:text-faint sm:w-44" aria-label="Search mappings" />
        </label>
      }
      flush
    >
      <div className="flex flex-wrap items-center gap-2 border-b border-line px-3.5 py-2">
        <div className="max-w-full overflow-x-auto [scrollbar-width:none]">
          <Segmented size="sm" value={f} onChange={setF} options={FILTERS.map((k) => ({ value: k, label: `${k === "all" ? "All" : STATUS_LABEL[k]} ${counts[k] ?? 0}` }))} />
        </div>
        <select value={iface} onChange={(e) => setIface(e.target.value)} className="h-7 rounded-md border border-line bg-bg px-2 text-[12px] text-ink" aria-label="Interface">
          <option value="all">Every interface</option>
          {ifaces.map((i) => (
            <option key={i} value={i}>
              {i}
            </option>
          ))}
        </select>
      </div>
      {!data ? (
        <div className="p-3.5">
          <Skeleton className="h-96" />
        </div>
      ) : !rows.length ? (
        <Empty>No mappings match.</Empty>
      ) : (
        <>
          <div className="hidden max-h-[680px] overflow-auto md:block">
            <table className="w-full min-w-[740px] text-[12px]">
              <thead className="sticky top-0 z-[1] bg-s1 text-left text-[11px] text-dim">
                <tr className="border-b border-line">
                  <th className="px-3.5 py-2 font-medium">Local code</th>
                  <th className="px-2 py-2 font-medium">Status</th>
                  <th className="px-2 py-2 font-medium">AegisOS concept</th>
                  <th className="px-2 py-2 font-medium">Written → read as</th>
                  <th className="px-2 py-2 font-medium">Proposed by</th>
                  <th className="whitespace-nowrap px-2 py-2 text-right font-medium">Delivered / withheld</th>
                  <th className="hidden px-3.5 py-2 text-right font-medium xl:table-cell">Updated</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((m) => (
                  <tr key={m.id} onClick={() => onMapping(m.id)} className="cursor-pointer border-b border-line last:border-0 hover:bg-s2">
                    <td className="px-3.5 py-2">
                      <div className="font-mono text-[11.5px] text-ink">{m.code}</div>
                      <div className="max-w-[200px] truncate text-[11px] text-dim">
                        {m.text} · {m.interface}
                      </div>
                    </td>
                    <td className="px-2 py-2">
                      <Tag tone={STATUS_TONE[m.status]} dot>
                        {STATUS_LABEL[m.status]}
                      </Tag>
                    </td>
                    <td className="px-2 py-2 text-ink">
                      {m.concept ??
                        (m.lead ? (
                          <span className="text-dim" title="Proposed; the evidence has not confirmed it">
                            {m.lead.concept}?
                          </span>
                        ) : (
                          <span className="text-dim">—</span>
                        ))}
                    </td>
                    <td className="px-2 py-2 font-mono text-[11.5px] text-muted">{readAs(m)}</td>
                    <td className="px-2 py-2 text-muted">{proposerLabel(m.proposer ?? m.lead?.proposer)}</td>
                    <td className="num whitespace-nowrap px-2 py-2 text-right font-mono text-muted">
                      {(m.counts.delivered + m.counts.backfilled).toLocaleString()} / <span className={cx(m.counts.withheld > 0 && "text-high-ink")}>{m.counts.withheld.toLocaleString()}</span>
                    </td>
                    <td className="hidden px-3.5 py-2 text-right text-[11px] text-dim xl:table-cell">{ago(m.updated_at)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="divide-y divide-line md:hidden">
            {rows.map((m) => (
              <button key={m.id} onClick={() => onMapping(m.id)} className="block w-full px-3.5 py-2.5 text-left">
                <div className="flex items-center gap-2">
                  <span className="font-mono text-[12px] text-ink">{m.code}</span>
                  <span className="min-w-0 flex-1 truncate text-[12px] text-muted">{m.text}</span>
                  <Tag tone={STATUS_TONE[m.status]} dot>
                    {STATUS_LABEL[m.status]}
                  </Tag>
                </div>
                <div className="mt-1 text-[11.5px] text-dim">
                  {m.concept ? `${m.concept} · ` : m.lead ? `${m.lead.concept}? · ` : ""}
                  {readAs(m)} · {m.interface}
                </div>
              </button>
            ))}
          </div>
        </>
      )}
    </Panel>
  );
}

// the unit as written, and what it is read as when that is not the same thing
function readAs(m: MappingRow): string {
  const w = m.unit_written || "no unit";
  const c = m.conversion;
  if (!c || (c.factor === 1 && c.offset === 0)) return w;
  return c.label.includes("→") ? `${w} → ${c.label.split("→").pop()!.trim()}` : `${w} · ${c.label}`;
}
