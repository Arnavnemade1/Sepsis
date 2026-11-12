"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo } from "react";

import { OpenLink, TrendUp } from "@/components/icons";
import { PageHead } from "@/components/shell";
import { cx, MiniLine, Panel, Skeleton, StatStrip, Tag } from "@/components/ui";
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
  /** the named model that scored it (A40 / V7) */
  model?: string;
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
      <PageHead title="Patients">
        Monitored patients ranked by sepsis risk. The validated paging threshold is 0.70; calibrated probabilities appear below each score.
        ICU rooms 301&ndash;312 replay de-identified, held-out PhysioNet/CinC 2019 records; room 402 is live.
      </PageHead>

      <StatStrip
        cols={5}
        className="mb-5"
        items={[
          { k: "Monitored", v: beds.length + (live ? 1 : 0), sub: "12 replayed, 1 live" },
          { k: "At the paging line", v: above, tone: above ? "text-high-ink" : undefined, sub: `index ≥ ${PAGING_LINE.toFixed(2)}, the detector's alarm` },
          { k: "On watch", v: watch, tone: watch ? "text-med-ink" : undefined, sub: `${INDEX.watch.toFixed(2)}–${PAGING_LINE.toFixed(2)}, approaching the line` },
          { k: "Rising early warning", v: rising, sub: "critical slowing down" },
          { k: "With an open page", v: withPage, sub: "paged by nurse or detector" },
        ]}
      />

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
                <Link href={`/patient/?id=${encodeURIComponent(live.id)}`} className="flex items-center gap-3 px-4 py-2.5 active:bg-s2">
                  <span className="num w-9 shrink-0 font-semibold text-ink">{liveRoom ?? "402"}</span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-ink">{live.name}</span>
                    <span className="block text-xs text-dim">Live session</span>
                  </span>
                  <span className="shrink-0">
                    <RiskCell risk={liveRisk} probability={liveP} model="V7" compact />
                  </span>
                </Link>
              </li>
            )}
            {beds.map((b) => {
              const room = roomOf[b.bed_id];
              return (
                <li key={b.bed_id}>
                  <Link href={`/patient/?id=${encodeURIComponent(b.bed_id)}`} className="flex items-center gap-3 px-4 py-2.5 active:bg-s2">
                    <span className="num w-9 shrink-0 font-semibold text-ink">{room ?? "—"}</span>
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-ink">{b.name}</span>
                      <span className="block truncate text-xs text-dim">
                        ICU hour {b.icu_hour} · {b.drivers[0] ?? "no derangement named"}
                      </span>
                    </span>
                    <span className="shrink-0">
                      <RiskCell risk={b.fused_risk} probability={b.probability} model={b.model} compact />
                    </span>
                  </Link>
                </li>
              );
            })}
          </ul>
          <div className="hidden overflow-x-auto md:block">
            <table className="w-full min-w-[1000px] text-sm">
              <thead>
                <tr className="border-b border-line text-left text-xs text-dim">
                  <th className="px-4 py-2.5 font-medium">Room</th>
                  <th className="px-2 py-2.5 font-medium">Patient</th>
                  <th className="px-2 py-2.5 font-medium">ICU hour</th>
                  <th className="w-[220px] px-2 py-2.5 font-medium">Sepsis risk · paging line {PAGING_LINE.toFixed(2)}</th>
                  <th className="px-2 py-2.5 font-medium">Early warning</th>
                  <th className="px-2 py-2.5 font-medium">Leading system</th>
                  <th className="px-2 py-2.5 font-medium">Drivers</th>
                  <th className="px-4 py-2.5 font-medium">Status</th>
                </tr>
              </thead>
              <tbody>
                {live && (
                  <tr onClick={() => router.push(`/patient/?id=${encodeURIComponent(live.id)}`)} className="group cursor-pointer border-b border-line transition-colors hover:bg-s2">
                    <td className="px-4 py-3">
                      <Link
                        href={`/room/?id=${encodeURIComponent(liveRoom ?? "402")}`}
                        onClick={(e) => e.stopPropagation()}
                        className="num font-semibold text-ink underline decoration-line-bright underline-offset-[3px] hover:decoration-ink"
                      >
                        {liveRoom ?? "402"}
                      </Link>
                    </td>
                    <td className="px-2 py-3">
                      <div className="text-ink">{live.name}</div>
                      <div className="text-xs text-dim">Live monitored session</div>
                    </td>
                    <td className="px-2 py-3 text-xs text-dim">Live</td>
                    <td className="px-2 py-3">
                      <RiskCell risk={liveRisk} probability={liveP} model="V7" />
                    </td>
                    <td className="px-2 py-3 text-xs text-dim">Streaming</td>
                    <td className="px-2 py-2.5 text-muted">—</td>
                    <td className="px-2 py-2.5 text-xs text-dim">camera, wearable or simulator</td>
                    <td className="px-4 py-3">
                      <StatusCell risk={liveRisk} page={!!(liveRoom && roomPages[liveRoom])} />
                    </td>
                  </tr>
                )}
                {beds.map((b) => {
                  const room = roomOf[b.bed_id];
                  return (
                    <tr key={b.bed_id} onClick={() => router.push(`/patient/?id=${encodeURIComponent(b.bed_id)}`)} className="group cursor-pointer border-b border-line-soft transition-colors last:border-b-0 hover:bg-s2">
                      <td className="px-4 py-3">
                        {room ? (
                          <Link
                            href={`/room/?id=${encodeURIComponent(room)}`}
                            onClick={(e) => e.stopPropagation()}
                            className="num font-semibold text-ink underline decoration-line-bright underline-offset-[3px] hover:decoration-ink"
                            title="Open the room: record, vitals, detections, timeline"
                          >
                            {room}
                          </Link>
                        ) : (
                          <div className="num font-semibold text-ink">—</div>
                        )}
                        <div className="font-mono text-2xs text-faint">{b.bed_id}</div>
                      </td>
                      <td className="px-2 py-3">
                        <div className="flex items-center gap-1.5 text-ink">
                          {b.name}
                          <OpenLink size={12} className="text-faint opacity-0 transition-opacity group-hover:opacity-100" />
                        </div>
                        <div className="text-xs text-dim">
                          {b.age != null ? `${Math.round(b.age)} ${b.gender ?? ""}` : ""} · <span className="font-mono">{b.mrn}</span>
                        </div>
                      </td>
                      <td className="num px-2 py-2.5 text-xs text-muted">
                        {b.icu_hour}
                        <span className="text-faint"> / {b.los_total}</span>
                      </td>
                      <td className="px-2 py-3">
                        <RiskCell risk={b.fused_risk} probability={b.probability} model={b.model} />
                      </td>
                      <td className="px-2 py-3">
                        <div className="flex items-center gap-2">
                          {b.ews ? <MiniLine data={b.ews.trace.slice(-16)} color={b.ews.rising ? "#eeb12a" : "#6c737e"} w={64} h={18} /> : null}
                          {b.ews?.rising && <TrendUp size={13} className="text-med-ink" />}
                        </div>
                      </td>
                      <td className="px-2 py-2.5 text-muted">{b.primary_system && b.primary_system !== "none" ? title(b.primary_system) : "—"}</td>
                      <td className="max-w-[260px] truncate px-2 py-2.5 text-xs text-dim" title={b.drivers.join(" · ")}>
                        {b.drivers[0] ?? "—"}
                      </td>
                      <td className="px-4 py-3">
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

function RiskCell({ risk, probability, model, compact = false }: { risk: number | null; probability?: number | null; model?: string; compact?: boolean }) {
  if (risk == null) return <span className="text-dim">—</span>;
  const tier = indexTier(risk);
  if (compact)
    return (
      <div className="text-right">
        <div className={cx("num text-sm font-semibold", TIER_TEXT[tier])}>{risk.toFixed(2)}</div>
        {probability != null && (
          <div className="num text-2xs text-faint">
            P {pct(probability, 1)}
            {model ? ` · ${model}` : ""}
          </div>
        )}
      </div>
    );
  return (
    <div>
      <div className="flex items-baseline gap-2">
        <span className={cx("num text-sm font-semibold", TIER_TEXT[tier])}>{risk.toFixed(2)}</span>
        {probability != null && (
          <span className="num text-xs text-faint" title="the scoring model's calibrated probability of the sepsis label">
            P {pct(probability, 1)}
            {model ? ` · ${model}` : ""}
          </span>
        )}
      </div>
      <div className="relative mt-1.5 h-1 w-[150px] overflow-hidden rounded-full bg-white/[0.07]">
        <div className="h-full rounded-full" style={{ width: `${Math.min(100, risk * 100)}%`, background: TIER_BAR[tier] }} />
        <span className="absolute inset-y-0 w-px bg-ink/60" style={{ left: `${PAGING_LINE * 100}%` }} title="the detector pages at this line" />
      </div>
    </div>
  );
}

function StatusCell({ risk, page }: { risk: number | null; page: boolean }) {
  const tier = risk == null ? null : indexTier(risk);
  return (
    <div className="flex flex-wrap items-center gap-1.5">
      {tier == null ? <Tag>No data</Tag> : tier === "HIGH" ? <Tag tone="high">High risk</Tag> : tier === "MEDIUM" ? <Tag tone="med">Medium</Tag> : <Tag>Low</Tag>}
      {page && <Tag tone="dys">Page open</Tag>}
    </div>
  );
}
