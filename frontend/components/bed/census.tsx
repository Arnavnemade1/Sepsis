"use client";

import { ArrowUpRight, TrendingUp } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo } from "react";

import { PageHead } from "@/components/shell";
import { cx, MiniLine, Panel, Skeleton, Tag } from "@/components/ui";
import { INDEX, indexTier, pct, title, type IndexTier } from "@/lib/clinical";
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
  /** the detector's calibrated P(sepsis) behind `fused_risk` (absent without a detector) */
  probability?: number;
  primary_system: string | null;
  drivers: string[];
  ews?: { ews: number; trace: number[]; rising: boolean; warming: boolean };
};
type Cluster = { beds: Bed[] };
type Roster = { patients: { id: string; name: string; kind: string }[] };

// Every bed, the live room included, is read on the ward risk index: one scale, one
// rule (lib/clinical.ts). The paging line is the detector's validated HIGH point and is
// drawn as the tick on every bar.
const PAGING_LINE = INDEX.page;
const TIER_TEXT: Record<IndexTier, string> = { HIGH: "text-high-ink", MEDIUM: "text-med-ink", LOW: "text-ink" };
const TIER_BAR: Record<IndexTier, string> = { HIGH: "#e5484d", MEDIUM: "#eeb12a", LOW: "#868d97" };

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
  const liveP = liveRoom ? state.monitor[liveRoom]?.probability ?? null : null;

  const beds = useMemo(() => [...(cluster.data?.beds ?? [])].sort((a, b) => b.fused_risk - a.fused_risk), [cluster.data]);
  const risks = [...beds.map((b) => b.fused_risk), ...(liveRisk != null ? [liveRisk] : [])];
  const above = risks.filter((r) => r >= PAGING_LINE).length;
  const watch = risks.filter((r) => r >= INDEX.watch && r < PAGING_LINE).length;
  const rising = beds.filter((b) => b.ews?.rising).length;
  const withPage = beds.filter((b) => roomOf[b.bed_id] && roomPages[roomOf[b.bed_id]]).length;

  return (
    <div>
      <PageHead
        title="Patients"
        right={
          <div className="flex flex-wrap gap-2">
            <Tag tone={above ? "high" : "neutral"} dot={!!above}>
              {above} at the paging line
            </Tag>
            <Tag tone={watch ? "med" : "neutral"} dot={!!watch}>
              {watch} on watch
            </Tag>
          </div>
        }
      >
        Every monitored bed, ranked by the sepsis detector&rsquo;s score of the current hour. The score is a 0&ndash;1 index whose 0.70 line is the
        detector&rsquo;s validated alarm point; its calibrated probability is shown beneath. ICU rooms 301&ndash;312 replay de-identified PhysioNet/CinC
        2019 stays the detector never trained on; room 402 is the live session.
      </PageHead>

      <div className="mb-3 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-3 xl:grid-cols-5">
        {[
          ["Monitored", beds.length + (live ? 1 : 0), "12 replayed · 1 live"],
          ["At the paging line", above, `index ≥ ${PAGING_LINE.toFixed(2)}: the detector's alarm`],
          ["On watch", watch, `${INDEX.watch.toFixed(2)}–${PAGING_LINE.toFixed(2)}, approaching the line`],
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

      <Panel title="Census" meta="Sorted by risk · select a patient to open the record" flush>
        {!cluster.data ? (
          <div className="p-3.5">
            <Skeleton className="h-96" />
          </div>
        ) : (
          <>
          {/* a phone gets one row per bed: room, patient, risk, status */}
          <ul className="divide-y divide-line-soft md:hidden">
            {live && (
              <li>
                <Link href={`/patient/?id=${encodeURIComponent(live.id)}`} className="flex items-center gap-3 px-3.5 py-2.5 active:bg-s2">
                  <span className="w-9 shrink-0 font-mono font-semibold text-ink">{liveRoom ?? "402"}</span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-ink">{live.name}</span>
                    <span className="block text-[11px] text-dim">Live session</span>
                  </span>
                  <span className="shrink-0">
                    <RiskCell risk={liveRisk} probability={liveP} compact />
                  </span>
                </Link>
              </li>
            )}
            {beds.map((b) => {
              const room = roomOf[b.bed_id];
              return (
                <li key={b.bed_id}>
                  <Link href={`/patient/?id=${encodeURIComponent(b.bed_id)}`} className="flex items-center gap-3 px-3.5 py-2.5 active:bg-s2">
                    <span className="w-9 shrink-0 font-mono font-semibold text-ink">{room ?? "—"}</span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-ink">{b.name}</span>
                      <span className="block truncate text-[11px] text-dim">
                        ICU hour {b.icu_hour} · {b.drivers[0] ?? "no derangement named"}
                      </span>
                    </span>
                    <span className="shrink-0">
                      <RiskCell risk={b.fused_risk} probability={b.probability} compact />
                    </span>
                  </Link>
                </li>
              );
            })}
          </ul>
          <div className="hidden overflow-x-auto md:block">
            <table className="w-full min-w-[1000px] text-[12.5px]">
              <thead>
                <tr className="border-b border-line text-left text-[11px] text-dim">
                  <th className="px-3.5 py-2 font-medium">Room</th>
                  <th className="px-2 py-2 font-medium">Patient</th>
                  <th className="px-2 py-2 font-medium">ICU hour</th>
                  <th className="w-[220px] px-2 py-2 font-medium">Sepsis risk · paging line {PAGING_LINE.toFixed(2)}</th>
                  <th className="px-2 py-2 font-medium">Early warning</th>
                  <th className="px-2 py-2 font-medium">Leading system</th>
                  <th className="px-2 py-2 font-medium">Drivers</th>
                  <th className="px-3.5 py-2 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {live && (
                  <tr onClick={() => router.push(`/patient/?id=${encodeURIComponent(live.id)}`)} className="group cursor-pointer border-b border-line transition-colors hover:bg-s2">
                    <td className="px-3.5 py-2.5">
                      <Link
                        href={`/room/?id=${encodeURIComponent(liveRoom ?? "402")}`}
                        onClick={(e) => e.stopPropagation()}
                        className="font-mono font-semibold text-ink underline decoration-line-bright underline-offset-2 hover:decoration-ink"
                      >
                        {liveRoom ?? "402"}
                      </Link>
                    </td>
                    <td className="px-2 py-2.5">
                      <div className="text-ink">{live.name}</div>
                      <div className="text-[11px] text-dim">Live monitored session</div>
                    </td>
                    <td className="px-2 py-2.5 font-mono text-[11.5px] text-dim">live</td>
                    <td className="px-2 py-2.5">
                      <RiskCell risk={liveRisk} probability={liveP} />
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
                        {room ? (
                          <Link
                            href={`/room/?id=${encodeURIComponent(room)}`}
                            onClick={(e) => e.stopPropagation()}
                            className="font-mono font-semibold text-ink underline decoration-line-bright underline-offset-2 hover:decoration-ink"
                            title="Open the room: record, vitals, detections, timeline"
                          >
                            {room}
                          </Link>
                        ) : (
                          <div className="font-mono font-semibold text-ink">—</div>
                        )}
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
                        <RiskCell risk={b.fused_risk} probability={b.probability} />
                      </td>
                      <td className="px-2 py-2.5">
                        <div className="flex items-center gap-2">
                          {b.ews ? <MiniLine data={b.ews.trace.slice(-16)} color={b.ews.rising ? "#eeb12a" : "#6c737e"} w={64} h={18} /> : null}
                          {b.ews?.rising && <TrendingUp size={13} className="text-med-ink" />}
                        </div>
                      </td>
                      <td className="px-2 py-2.5 text-muted">{b.primary_system && b.primary_system !== "none" ? title(b.primary_system) : "—"}</td>
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
          </>
        )}
      </Panel>
    </div>
  );
}

function RiskCell({ risk, probability, compact = false }: { risk: number | null; probability?: number | null; compact?: boolean }) {
  if (risk == null) return <span className="text-dim">—</span>;
  const tier = indexTier(risk);
  if (compact)
    return (
      <div className="text-right">
        <div className={cx("num font-mono text-[13px] font-semibold", TIER_TEXT[tier])}>{risk.toFixed(2)}</div>
        {probability != null && <div className="num font-mono text-[10.5px] text-faint">P {pct(probability, 1)}</div>}
      </div>
    );
  return (
    <div>
      <div className="flex items-baseline gap-2">
        <span className={cx("num font-mono text-[13px] font-semibold", TIER_TEXT[tier])}>{risk.toFixed(2)}</span>
        {probability != null && (
          <span className="num font-mono text-[11px] text-faint" title="the detector's calibrated probability of the sepsis label">
            P {pct(probability, 1)}
          </span>
        )}
      </div>
      <div className="relative mt-1 h-1 w-[150px] overflow-hidden rounded-[2px] bg-s3">
        <div className="h-full" style={{ width: `${Math.min(100, risk * 100)}%`, background: TIER_BAR[tier] }} />
        <span className="absolute inset-y-0 w-px bg-ink/60" style={{ left: `${PAGING_LINE * 100}%` }} title="the detector pages at this line" />
      </div>
    </div>
  );
}

function StatusCell({ risk, page }: { risk: number | null; page: boolean }) {
  const tier = risk == null ? null : indexTier(risk);
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {tier == null ? <Tag>no data</Tag> : tier === "HIGH" ? <Tag tone="high">High risk</Tag> : tier === "MEDIUM" ? <Tag tone="med">Medium</Tag> : <Tag>Low</Tag>}
      {page && <Tag tone="dys">Page open</Tag>}
    </div>
  );
}
