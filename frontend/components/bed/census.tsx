"use client";

import { ArrowUpRight, TrendingUp } from "lucide-react";
import { useRouter } from "next/navigation";
import { useMemo } from "react";

import { PageHead } from "@/components/shell";
import { cx, MiniLine, Panel, Skeleton, Tag } from "@/components/ui";
import { title } from "@/lib/clinical";
import { usePoll } from "@/lib/hooks";
import { usePaging, useRoomPages } from "@/lib/paging";

type Bed = {
  bed_id: string;
  name: string;
  age: number | null;
  gender: string | null;
  mrn: string;
  icu_hour: number;
  los_total: number;
  fused_risk: number;
  population_risk: number;
  primary_system: string | null;
  drivers: string[];
  ews?: { ews: number; trace: number[]; rising: boolean; warming: boolean };
};
type Cluster = { beds: Bed[] };
type Roster = { patients: { id: string; name: string; kind: string }[] };

const ALARM = 0.7;
const WATCH = 0.6;

function riskTone(r: number) {
  return r >= ALARM ? "text-high-ink" : r >= WATCH ? "text-med-ink" : "text-ink";
}

export function Census() {
  const router = useRouter();
  const cluster = usePoll<Cluster>("/aegis/cluster/snapshot", 5000);
  const roster = usePoll<Roster>("/aegis/patients");
  const { state } = usePaging();
  const roomPages = useRoomPages(state.pages);

  const roomOf = useMemo(() => {
    const m: Record<string, string> = {};
    for (const [room, v] of Object.entries(state.monitor)) if (v.bed) m[v.bed] = room;
    return m;
  }, [state.monitor]);
  const liveRoom = Object.entries(state.monitor).find(([, v]) => v.live)?.[0] ?? null;
  const live = roster.data?.patients.find((p) => p.kind === "live") ?? null;
  const liveRisk = liveRoom ? state.monitor[liveRoom]?.risk ?? null : null;

  const beds = useMemo(() => [...(cluster.data?.beds ?? [])].sort((a, b) => b.fused_risk - a.fused_risk), [cluster.data]);
  const above = beds.filter((b) => b.fused_risk >= ALARM).length + (liveRisk != null && liveRisk >= ALARM ? 1 : 0);
  const watch = beds.filter((b) => b.fused_risk >= WATCH && b.fused_risk < ALARM).length;
  const rising = beds.filter((b) => b.ews?.rising).length;
  const withPage = beds.filter((b) => roomOf[b.bed_id] && roomPages[roomOf[b.bed_id]]).length;

  return (
    <div>
      <PageHead
        title="Patients"
        right={
          <div className="flex flex-wrap gap-2">
            <Tag tone={above ? "high" : "neutral"} dot={!!above}>
              {above} above alarm
            </Tag>
            <Tag tone={watch ? "med" : "neutral"} dot={!!watch}>
              {watch} on watch
            </Tag>
          </div>
        }
      >
        Every monitored bed, ranked by the risk the unit acts on: the fused, personalized score with the population score beside it. ICU rooms 301–312
        replay de-identified PhysioNet/CinC 2019 stays; room 402 is the live session.
      </PageHead>

      <div className="mb-3 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-4">
        {[
          ["Monitored", beds.length + (live ? 1 : 0), "12 replayed · 1 live"],
          ["Above the alarm line", above, `effective risk ≥ ${ALARM.toFixed(2)}`],
          ["Rising early warning", rising, "critical slowing down"],
          ["With an open page", withPage, "paged by nurse or detector"],
        ].map(([k, v, sub]) => (
          <div key={String(k)} className="min-w-0 bg-s1 px-3.5 py-2.5">
            <div className="truncate text-[11px] text-dim">{k}</div>
            <div className="num mt-0.5 text-[20px] font-medium text-ink">{v}</div>
            <div className="truncate text-[11px] text-dim">{sub}</div>
          </div>
        ))}
      </div>

      <Panel title="Census" meta="Sorted by effective risk · select a patient to open the record" flush>
        {!cluster.data ? (
          <div className="p-3.5">
            <Skeleton className="h-96" />
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full min-w-[1000px] text-[12.5px]">
              <thead>
                <tr className="border-b border-line text-left text-[11px] text-dim">
                  <th className="px-3.5 py-2 font-medium">Room</th>
                  <th className="px-2 py-2 font-medium">Patient</th>
                  <th className="px-2 py-2 font-medium">ICU hour</th>
                  <th className="w-[220px] px-2 py-2 font-medium">Effective risk</th>
                  <th className="px-2 py-2 font-medium">Early warning</th>
                  <th className="px-2 py-2 font-medium">Leading system</th>
                  <th className="px-2 py-2 font-medium">Drivers</th>
                  <th className="px-3.5 py-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {live && (
                  <tr onClick={() => router.push(`/patient/?id=${encodeURIComponent(live.id)}`)} className="group cursor-pointer border-b border-line transition-colors hover:bg-s2">
                    <td className="px-3.5 py-2.5 font-mono font-semibold text-ink">{liveRoom ?? "402"}</td>
                    <td className="px-2 py-2.5">
                      <div className="text-ink">{live.name}</div>
                      <div className="text-[11px] text-dim">Live monitored session</div>
                    </td>
                    <td className="px-2 py-2.5 font-mono text-[11.5px] text-dim">live</td>
                    <td className="px-2 py-2.5">
                      <RiskCell risk={liveRisk} />
                    </td>
                    <td className="px-2 py-2.5 text-[11.5px] text-dim">streaming</td>
                    <td className="px-2 py-2.5 text-muted">—</td>
                    <td className="px-2 py-2.5 text-[12px] text-dim">camera, wearable or simulator</td>
                    <td className="px-3.5 py-2.5">
                      <StatusCell risk={liveRisk} page={!!(liveRoom && roomPages[liveRoom])} />
                    </td>
                  </tr>
                )}
                {beds.map((b) => {
                  const room = roomOf[b.bed_id];
                  return (
                    <tr key={b.bed_id} onClick={() => router.push(`/patient/?id=${encodeURIComponent(b.bed_id)}`)} className="group cursor-pointer border-b border-line-soft transition-colors last:border-b-0 hover:bg-s2">
                      <td className="px-3.5 py-2.5">
                        <div className="font-mono font-semibold text-ink">{room ?? "—"}</div>
                        <div className="font-mono text-[10.5px] text-faint">{b.bed_id}</div>
                      </td>
                      <td className="px-2 py-2.5">
                        <div className="flex items-center gap-1.5 text-ink">
                          {b.name}
                          <ArrowUpRight size={12} className="text-faint opacity-0 transition-opacity group-hover:opacity-100" />
                        </div>
                        <div className="text-[11px] text-dim">
                          {b.age != null ? `${Math.round(b.age)} ${b.gender ?? ""}` : ""} · <span className="font-mono">{b.mrn}</span>
                        </div>
                      </td>
                      <td className="num px-2 py-2.5 font-mono text-[11.5px] text-muted">
                        {b.icu_hour}
                        <span className="text-faint"> / {b.los_total}</span>
                      </td>
                      <td className="px-2 py-2.5">
                        <RiskCell risk={b.fused_risk} population={b.population_risk} />
                      </td>
                      <td className="px-2 py-2.5">
                        <div className="flex items-center gap-2">
                          {b.ews ? <MiniLine data={b.ews.trace.slice(-16)} color={b.ews.rising ? "#eeb12a" : "#6c737e"} w={64} h={18} /> : null}
                          {b.ews?.rising && <TrendingUp size={13} className="text-med-ink" />}
                        </div>
                      </td>
                      <td className="px-2 py-2.5 text-muted">{b.primary_system ? title(b.primary_system) : "—"}</td>
                      <td className="max-w-[260px] truncate px-2 py-2.5 text-[12px] text-dim" title={b.drivers.join(" · ")}>
                        {b.drivers[0] ?? "—"}
                      </td>
                      <td className="px-3.5 py-2.5">
                        <StatusCell risk={b.fused_risk} page={!!(room && roomPages[room])} />
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </Panel>
    </div>
  );
}

function RiskCell({ risk, population }: { risk: number | null; population?: number }) {
  if (risk == null) return <span className="text-dim">—</span>;
  return (
    <div>
      <div className="flex items-baseline gap-2">
        <span className={cx("num font-mono text-[13px] font-semibold", riskTone(risk))}>{risk.toFixed(2)}</span>
        {population != null && <span className="num font-mono text-[11px] text-faint">pop {population.toFixed(2)}</span>}
      </div>
      <div className="relative mt-1 h-1 w-[150px] overflow-hidden rounded-[2px] bg-s3">
        <div className="h-full" style={{ width: `${Math.min(100, risk * 100)}%`, background: risk >= ALARM ? "#e5484d" : risk >= WATCH ? "#eeb12a" : "#868d97" }} />
        <span className="absolute inset-y-0 w-px bg-ink/60" style={{ left: `${ALARM * 100}%` }} />
      </div>
    </div>
  );
}

function StatusCell({ risk, page }: { risk: number | null; page: boolean }) {
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {risk == null ? (
        <Tag>no data</Tag>
      ) : risk >= ALARM ? (
        <Tag tone="high">Above alarm</Tag>
      ) : risk >= WATCH ? (
        <Tag tone="med">Watch</Tag>
      ) : (
        <Tag>Stable</Tag>
      )}
      {page && <Tag tone="dys">Page open</Tag>}
    </div>
  );
}
