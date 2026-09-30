"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import { FeedPill } from "@/components/feed";
import { Page, PageHead } from "@/components/shell";
import { Button, cx, Empty, Kv, Meter, Pill, Plate, Skeleton, Stat } from "@/components/ui";
import { ago, pct, tierTone, title, TONE_RGB } from "@/lib/clinical";
import { useFeed } from "@/lib/hooks";
import type { Cluster, TopologyNode, WardBed } from "@/lib/types";

export default function CommandPage() {
  const { data, status, lastAt } = useFeed<Cluster>("/aegis/cluster/stream");
  const [truth, setTruth] = useState(false);

  const rows = useMemo(() => {
    if (!data) return [];
    const topo = new Map(data.topology.map((t) => [t.node_id, t]));
    return data.beds
      .map((b) => ({ bed: b, node: topo.get(b.bed_id) }))
      .sort((a, b) => b.bed.fused_risk - a.bed.fused_risk);
  }, [data]);

  const k = data?.kstat;
  return (
    <Page context={<FeedPill status={status} lastAt={lastAt} label="Ward feed" />}>
      <PageHead
        eyebrow={data ? `ICU wing · ${data.beds.length} beds · ${data.data_mode === "real-cinc2019" ? "replayed PhysioNet/CinC 2019 records" : data.data_mode}` : "ICU wing"}
        title="Unit command"
        right={
          <Button variant={truth ? "ghost" : "quiet"} onClick={() => setTruth((t) => !t)}>
            {truth ? "Hide replay truth" : "Show replay truth"}
          </Button>
        }
      >
        Every bed, ranked by fused risk and re-ranked every tick. The scheduler gives the sickest and the fastest-worsening
        beds more attention, and every alert that leaves the kernel is formally verified first.
      </PageHead>

      <div className="mb-5 grid grid-cols-2 gap-3 md:grid-cols-3 xl:grid-cols-6">
        <KpiTile label="Critical" value={k?.tiers.CRITICAL} tone="high" sub="beds at the top tier" />
        <KpiTile label="Watch" value={k?.tiers.WATCH} tone="med" sub={k ? `${k.anticipating} anticipated early` : undefined} />
        <KpiTile label="Stable" value={k?.tiers.STABLE} tone="neutral" sub="no action needed" />
        <KpiTile
          label="Alerts certified"
          value={data?.safety.certified}
          sub={data ? `${pct(data.safety.pass_rate, 0)} pass · ${data.safety.blocked} blocked` : undefined}
        />
        <KpiTile
          label="Scheduler"
          value={k ? pct(k.utilization, 0) : undefined}
          sub={k ? `${title(k.mode)} · ${k.interrupts} preemptions` : undefined}
        />
        <KpiTile
          label="Sensor read p95"
          value={data ? `${data.devbio.latency.p95_us.toFixed(1)}` : undefined}
          unit="µs"
          sub={data ? `${data.devbio.mounted} verified device nodes` : undefined}
        />
      </div>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1fr)_380px]">
        <Plate
          title="Bed board"
          eyebrow="Sorted by fused risk"
          right={<span className="text-[11.5px] text-dim">click a bed to open it at the bedside</span>}
          bodyClass="p-3"
        >
          {!data ? (
            <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
              {Array.from({ length: 6 }).map((_, i) => (
                <Skeleton key={i} className="h-[150px] rounded-lg" />
              ))}
            </div>
          ) : (
            <div className="grid gap-3 sm:grid-cols-2 2xl:grid-cols-3">
              {rows.map(({ bed, node }) => (
                <BedCard key={bed.bed_id} bed={bed} node={node} truth={truth} />
              ))}
            </div>
          )}
        </Plate>

        <div className="flex min-w-0 flex-col gap-5">
          <FocusBundle data={data} />
          <EventLog data={data} />
          <Services data={data} />
        </div>
      </div>
    </Page>
  );
}

function KpiTile({ label, value, sub, tone = "neutral", unit }: {
  label: string;
  value: number | string | undefined;
  sub?: string;
  tone?: "high" | "med" | "neutral";
  unit?: string;
}) {
  return (
    <div className="plate rise px-4 py-3">
      {value === undefined ? (
        <Skeleton className="h-[58px]" />
      ) : (
        <Stat label={label} value={value} unit={unit} tone={value === 0 ? "neutral" : tone} sub={sub} />
      )}
    </div>
  );
}

function BedCard({ bed, node, truth }: { bed: WardBed; node?: TopologyNode; truth: boolean }) {
  const tone = tierTone(node?.tier);
  const slope = node?.slope ?? 0;
  const arrow = slope > 0.004 ? "↑" : slope < -0.004 ? "↓" : "→";
  return (
    <Link
      href={`/bed/?id=${bed.bed_id}`}
      className="group relative flex flex-col overflow-hidden rounded-lg border border-line bg-s2/60 p-3.5 transition duration-200 hover:-translate-y-px hover:border-line-bright hover:bg-s2"
    >
      <span
        className="absolute inset-y-0 left-0 w-[3px]"
        style={{ background: tone === "low" ? "transparent" : `rgb(${TONE_RGB[tone]})` }}
      />
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <div className="truncate text-[14px] font-medium text-ink">{bed.name}</div>
          <div className="num truncate text-[11.5px] text-dim">
            {bed.bed_id.replace("bed_", "Bed ")} · {bed.mrn} · {Math.round(bed.age)}
            {bed.gender} · ICU h{bed.icu_hour}
          </div>
        </div>
        <Pill tone={tone === "low" ? "neutral" : tone} dot>
          {title(node?.tier ?? "—")}
        </Pill>
      </div>

      <div className="mt-3 flex items-end gap-3">
        <div className="num text-[30px] font-light leading-none tracking-tight">
          {(bed.fused_risk * 100).toFixed(0)}
          <span className="text-[14px] text-dim">%</span>
        </div>
        <div className="num pb-0.5 text-[12px] text-muted" title="Change in fused risk per scheduler tick">
          {arrow} {(slope * 100).toFixed(1)} pt/tick
        </div>
        {node && node.foresight > 0.05 && (
          <span
            className="ml-auto pb-0.5 text-[11px] text-med-ink"
            title="Critical-slowing-down foresight: the scheduler is raising this bed's priority before its risk crosses"
          >
            early warning
          </span>
        )}
      </div>
      <Meter value={bed.fused_risk} tone={tone === "low" ? "neutral" : tone} className="mt-2" />

      <ul className="mt-3 space-y-0.5 text-[12px] text-muted">
        {(bed.drivers.length ? bed.drivers : ["No abnormal drivers"]).slice(0, 2).map((d) => (
          <li key={d} className="truncate">
            {d}
          </li>
        ))}
      </ul>

      <div className="mt-auto flex items-center gap-3 pt-3 text-[11px] text-dim">
        <span>{title(bed.primary_system)} lead</span>
        {node && <span className="num">{node.lanes} lanes</span>}
        {truth && (
          <span className={cx("ml-auto num", bed.will_be_septic ? "text-dys-ink" : "text-dim")}>
            {bed.will_be_septic
              ? bed.hours_to_onset != null && bed.hours_to_onset > 0
                ? `onset in ${bed.hours_to_onset}h`
                : "septic"
              : "never septic"}
          </span>
        )}
      </div>
    </Link>
  );
}

function FocusBundle({ data }: { data: Cluster | null }) {
  const b = data?.bundle;
  return (
    <Plate title="Certified focus" eyebrow="The bed the orchestrator is acting on">
      {!data ? (
        <Skeleton className="h-40" />
      ) : !b ? (
        <Empty>No bed needs a bundle right now.</Empty>
      ) : (
        <div>
          <div className="flex items-start gap-3">
            <div className="min-w-0 flex-1">
              <Link href={`/bed/?id=${b.bed_id}`} className="text-[15px] font-medium text-ink hover:underline">
                {b.name}
              </Link>
              <div className="text-[12px] text-dim">
                {title(b.severity_name)} · {title(b.primary_system)} lead
              </div>
            </div>
            <Stat label="Fused risk" value={pct(b.fused_risk, 0)} tone={tierTone(b.risk_level)} size="sm" />
          </div>
          <div className="mt-3 rounded-md border border-line-soft bg-s2/60 p-3">
            <Kv k="Antibiotic" v={b.protocol.antibiotic} />
            <Kv k="Fluids" v={`${b.protocol.fluid_rate} mL/kg/h`} />
            <Kv k="Vasopressor" v={`${b.protocol.vasopressor_dose} µg/kg/min`} />
          </div>
          <div className="mt-3 flex items-center gap-2">
            <Pill tone={b.protocol.certified ? "neutral" : "high"} dot>
              {b.protocol.certified ? "Protocol proven safe" : "Protocol refused"}
            </Pill>
            <span className="font-mono text-[11px] text-dim">z3 {b.protocol.z3}</span>
          </div>
          <details className="mt-2 text-[12px] text-muted">
            <summary className="cursor-pointer text-dim hover:text-ink">Proof obligations</summary>
            <p className="mt-2 break-words font-mono text-[11px] leading-relaxed text-muted">{b.protocol.proof}</p>
          </details>
        </div>
      )}
    </Plate>
  );
}

type LogRow = { ts: number; kind: string; text: string; tone: "high" | "med" | "neutral" | "nodata" };

function EventLog({ data }: { data: Cluster | null }) {
  const names = useMemo(() => new Map(data?.beds.map((b) => [b.bed_id, b.name]) ?? []), [data]);
  const rows: LogRow[] = useMemo(() => {
    if (!data) return [];
    const out: LogRow[] = [];
    for (const e of data.events ?? []) {
      out.push({
        ts: e.ts,
        kind: title(e.type),
        text: `${names.get(e.owner ?? "") ?? e.owner ?? "unit"} took priority · ${e.reason} · p ${pct(e.p, 0)}`,
        tone: e.type.includes("PREEMPT") ? "med" : "neutral",
      });
    }
    for (const s of data.safety.recent ?? []) {
      out.push({ ts: s.ts, kind: s.ok ? "Certified" : "Blocked", text: s.message, tone: s.ok ? "neutral" : "high" });
    }
    return out.sort((a, b) => b.ts - a.ts).slice(0, 14);
  }, [data, names]);
  return (
    <Plate title="Unit log" eyebrow="Scheduler preemptions and verified egress" bodyClass="p-0">
      {!data ? (
        <div className="p-4">
          <Skeleton className="h-40" />
        </div>
      ) : rows.length === 0 ? (
        <Empty>Quiet. Nothing has preempted or left the kernel yet.</Empty>
      ) : (
        <ul className="max-h-[360px] divide-y divide-line-soft overflow-y-auto">
          {rows.map((r, i) => (
            <li key={i} className="flex gap-3 px-4 py-2.5">
              <span className={cx("mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full", r.tone === "high" ? "bg-high" : r.tone === "med" ? "bg-med" : "bg-faint")} />
              <div className="min-w-0 flex-1">
                <div className="flex items-baseline gap-2">
                  <span className="text-[12px] font-medium text-ink">{r.kind}</span>
                  <span className="ml-auto shrink-0 text-[11px] text-dim">{ago(r.ts)}</span>
                </div>
                <div className="text-[12px] leading-snug text-muted">{r.text}</div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </Plate>
  );
}

function Services({ data }: { data: Cluster | null }) {
  return (
    <Plate title="Clinical services" eyebrow="aegisd — capability-confined apps on the ward" right={data && <span className="num text-[12px] text-muted">{data.services.running}/{data.services.n} running</span>}>
      {!data ? (
        <Skeleton className="h-16" />
      ) : (
        <div className="flex flex-wrap gap-2">
          {data.services.services.map((s) => (
            <Pill key={s.name} tone={s.healthy ? "neutral" : "high"} dot live={s.state === "RUNNING"}>
              {s.name}
            </Pill>
          ))}
        </div>
      )}
    </Plate>
  );
}
