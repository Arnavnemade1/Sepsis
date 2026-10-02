"use client";

import { ArrowLeft, ArrowUpRight, BellPlus, BookOpen, ExternalLink } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";

import { PriorityChip } from "@/components/paging/bits";
import { Button, cx, Empty, Panel, Segmented, Skeleton } from "@/components/ui";
import { clock, PAGE_STATUS_LABEL, type PageStatus } from "@/lib/paging";
import {
  BAND_LABEL,
  fmtVal,
  LAB_LABEL,
  labFlag,
  POINT_SEV,
  SEV_INK,
  SEV_LABEL,
  SEV_RANK,
  SEV_RGB,
  since,
  useUnit,
  type Finding,
  type RoomRecord,
  type Severity,
  type TimelineEvent,
  type Unit,
  type UnitPage,
  type Vitals,
} from "@/lib/rooms";

import { AllergyChips, CodeChip, KindTag, News2Badge, SevDot, SevTag, Spark, useWidth } from "./bits";
import { RoomBody } from "./body";

/* ------------------------------------------------------------------ */
/* One room, as one unit                                                */
/* ------------------------------------------------------------------ */
export function RoomUnit({ room }: { room: string }) {
  const { data: u, error } = useUnit(room, 4000);
  if (error && !u) return <Empty>Could not load room {room}: {error}</Empty>;
  if (!u) return <UnitSkeleton room={room} />;
  if ((u as unknown as { status?: string }).status === "error") return <Empty>There is no room {room} in this hospital.</Empty>;
  if (!u.occupied || !u.record) return <Unoccupied u={u} />;
  return <Occupied u={u as Required<Unit>} />;
}

function UnitSkeleton({ room }: { room: string }) {
  return (
    <div>
      <BackLink />
      <div className="mb-4 mt-2 flex items-end gap-3">
        <h1 className="font-mono text-[26px] font-semibold tracking-[-0.02em] text-ink">Room {room}</h1>
      </div>
      <Skeleton className="mb-3 h-16" />
      <div className="grid gap-3 wide:grid-cols-3 lg:grid-cols-2">
        <Skeleton className="h-[420px]" />
        <Skeleton className="h-[420px]" />
        <Skeleton className="h-[420px]" />
      </div>
    </div>
  );
}

function BackLink() {
  return (
    <Link href="/room/" className="inline-flex items-center gap-1.5 text-[12px] text-dim transition-colors hover:text-ink">
      <ArrowLeft size={13} /> All rooms
    </Link>
  );
}

function Unoccupied({ u }: { u: Unit }) {
  return (
    <div>
      <BackLink />
      <div className="mb-4 mt-2">
        <div className="text-[12px] text-dim">
          Floor {u.floor} · {u.zone_name}
        </div>
        <h1 className="font-mono text-[26px] font-semibold tracking-[-0.02em] text-ink">Room {u.room}</h1>
      </div>
      <Panel>
        <Empty>Unoccupied. The room is not being monitored.</Empty>
      </Panel>
    </div>
  );
}

function Occupied({ u }: { u: Required<Unit> }) {
  const rec = u.record;
  const replay = u.kind === "replay";
  // a system picked on the body narrows the detections to the findings that implicate it
  const [system, setSystem] = useState<string | null>(null);
  const pick = (r: string | null) => {
    setSystem(r);
    if (r && typeof window !== "undefined" && window.innerWidth < 1024) document.getElementById("detections")?.scrollIntoView({ behavior: "smooth", block: "start" });
  };
  const sysKeys = system ? new Set((u.body?.regions[system]?.findings ?? []).map((f) => f.key)) : null;
  const active = u.findings.filter((f) => f.severity !== "info" && (!sysKeys || sysKeys.has(f.key)));
  const tracked = u.findings.filter((f) => f.severity === "info" && (!sysKeys || sysKeys.has(f.key)));
  const openPage = u.pages.find((p) => p.active) ?? null;
  return (
    <div>
      <Header u={u} openPage={openPage} />
      <ScoreStrip u={u} active={active} openPage={openPage} />

      <div className="grid items-start gap-3 lg:grid-cols-2 wide:grid-cols-[minmax(0,1.05fr)_minmax(0,1.2fr)_minmax(0,0.95fr)]">
        {/* below lg the middle column dissolves into the grid so the body can lead the page,
            ahead of the detections; from lg up it is one column again */}
        <div className="order-2 flex min-w-0 flex-col gap-3 lg:order-none">
          <Detections u={u} active={active} tracked={tracked} system={system ? u.body?.regions[system]?.label ?? system : null} onClear={() => setSystem(null)} />
        </div>
        <div className="contents lg:flex lg:min-w-0 lg:flex-col lg:gap-3">
          <div className="order-1 min-w-0 lg:order-none">
            <RoomBody body={u.body} kind={u.kind} system={system} onSystem={pick} />
          </div>
          <div className="order-3 min-w-0 lg:order-none">
            <VitalsPanel u={u} />
          </div>
          <div className="order-4 min-w-0 lg:order-none">
            <LabsPanel u={u} />
          </div>
        </div>
        <div className="order-5 flex min-w-0 flex-col gap-3 lg:order-none lg:col-span-2 wide:col-span-1">
          <div className="grid gap-3 lg:grid-cols-2 wide:grid-cols-1">
            <RecordPanel rec={rec} now={u.now} replay={replay} />
            <LinkedPanel u={u} />
          </div>
        </div>
      </div>

      <div className="mt-3">
        <TimelinePanel u={u} />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Header: who is in the room, and what everyone must know first        */
/* ------------------------------------------------------------------ */
function Header({ u, openPage }: { u: Required<Unit>; openPage: UnitPage | null }) {
  const rec = u.record;
  const pt = rec.patient;
  const adm = rec.admission;
  const falls = u.findings.find((f) => f.key === "falls");
  const o2 = u.vitals?.o2_lpm;
  return (
    <div className="mb-4">
      <div className="flex flex-wrap items-center gap-x-3 gap-y-1">
        <BackLink />
        <span className="text-[12px] text-faint">/</span>
        <span className="text-[12px] text-dim">
          Floor {u.floor} · {u.zone_name}
        </span>
      </div>
      <div className="mt-2 flex flex-wrap items-start gap-x-6 gap-y-3">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-3 gap-y-1.5">
            <h1 className="whitespace-nowrap font-mono text-[26px] font-semibold leading-none tracking-[-0.02em] text-ink">Room {u.room}</h1>
            <span className="text-[17px] font-semibold tracking-[-0.01em] text-ink">{pt.name ?? "Patient"}</span>
            <span className="text-[13px] text-muted">
              {pt.age != null ? `${Math.round(pt.age)}` : ""}
              {pt.sex ?? ""}
              {pt.mrn && <span className="ml-2 font-mono text-[12px] text-dim">{pt.mrn}</span>}
            </span>
            <KindTag kind={u.kind} />
          </div>
          <div className="mt-1.5 text-[13px] text-muted">
            {adm.reason}
            {adm.day != null && !adm.icu_hour && <span className="text-dim"> · hospital day {adm.day}</span>}
            {adm.icu_hour != null && <span className="text-dim"> · ICU hour {adm.icu_hour}</span>}
            {pt.weight_kg != null && <span className="text-dim"> · {pt.weight_kg} kg</span>}
          </div>
          <div className="mt-2.5 flex flex-wrap items-center gap-1.5">
            <AllergyChips allergies={rec.allergies} />
            <CodeChip code={rec.code_status} />
            {rec.spo2_target && (
              <span className="inline-flex h-5 items-center rounded-[4px] border border-line-bright bg-s2 px-1.5 text-[11px] font-medium text-muted">
                SpO₂ target {rec.spo2_target.lo}–{rec.spo2_target.hi}%
              </span>
            )}
            {falls && (
              <span className="inline-flex h-5 items-center rounded-[4px] border border-line-bright bg-s2 px-1.5 text-[11px] font-medium text-muted">
                Falls risk · Morse {falls.score}
              </span>
            )}
            {!!o2 && (
              <span className="inline-flex h-5 items-center rounded-[4px] border border-line-bright bg-s2 px-1.5 text-[11px] font-medium text-muted">O₂ {o2} L/min</span>
            )}
            {rec.mobility && rec.mobility !== "independent" && (
              <span className="inline-flex h-5 items-center rounded-[4px] border border-line-bright bg-s2 px-1.5 text-[11px] font-medium text-muted">Mobility: {rec.mobility}</span>
            )}
          </div>
        </div>
        <div className="flex w-full shrink-0 flex-wrap items-center gap-2 sm:w-auto">
          {openPage && (
            <Link
              href={`/operator/?page=${encodeURIComponent(openPage.id)}`}
              className="inline-flex h-8 items-center gap-2 rounded-md border border-line-bright bg-s2 px-2.5 text-[12.5px] text-ink transition-colors hover:bg-s3"
            >
              <PriorityChip p={openPage.priority} size="sm" />
              {openPage.id}
              <span className="text-dim">{PAGE_STATUS_LABEL[openPage.status as PageStatus] ?? openPage.status}</span>
            </Link>
          )}
          {pt.bed && (
            <Link
              href={`/patient/?id=${encodeURIComponent(pt.bed)}`}
              className="inline-flex h-8 items-center gap-1.5 rounded-md border border-line-bright px-2.5 text-[12.5px] text-ink transition-colors hover:bg-s2"
            >
              ICU record <ArrowUpRight size={13} className="text-dim" />
            </Link>
          )}
          <Link href={`/operator/?compose=1&room=${encodeURIComponent(u.room)}`}>
            <Button variant="primary" size="sm">
              <BellPlus size={13} strokeWidth={2.2} /> Raise an alert
            </Button>
          </Link>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The scores the room is acting on                                     */
/* ------------------------------------------------------------------ */
function ScoreStrip({ u, active, openPage }: { u: Required<Unit>; active: Finding[]; openPage: UnitPage | null }) {
  const n = u.scores.news2;
  const replay = u.kind === "replay";
  const counts = { critical: 0, urgent: 0, watch: 0 } as Record<"critical" | "urgent" | "watch", number>;
  for (const f of active) if (f.severity !== "info") counts[f.severity as "critical" | "urgent" | "watch"] += 1;
  const last = u.vitals?.t;
  const cells: { k: string; v: React.ReactNode; sub: React.ReactNode }[] = [
    {
      k: "NEWS2",
      v: replay ? <span className="text-[15px] text-dim">Not used in ICU</span> : <span className="flex items-center gap-2"><News2Badge score={n?.score} band={n?.band} /><span className="text-[13px] text-muted">{n ? BAND_LABEL[n.band] : ""}</span></span>,
      sub: replay ? "SOFA and the sepsis detector instead" : n ? (n.complete ? `scale ${n.scale}${n.scale === 2 ? ", from the record" : ""}` : `incomplete: no ${n.missing.join(", ")}`) : "",
    },
    { k: "qSOFA", v: u.scores.qsofa != null ? `${u.scores.qsofa} / 3` : "—", sub: "RR ≥ 22 · SBP ≤ 100 · mentation" },
    { k: "Shock index", v: u.scores.shock_index != null ? u.scores.shock_index.toFixed(2) : "—", sub: "pulse ÷ systolic; ≥ 1.0 abnormal" },
    {
      k: "Sepsis risk",
      v: u.risk != null ? u.risk.toFixed(2) : "—",
      sub: u.risk != null ? "the detector pages at 0.70" : "a ward room: scored by the monitor",
    },
    {
      k: "Active findings",
      v: (
        <span className="flex items-center gap-2.5">
          {(["critical", "urgent", "watch"] as const).map((s) => (
            <span key={s} className="inline-flex items-center gap-1" title={SEV_LABEL[s]}>
              <SevDot sev={s} />
              <span className={cx("num font-mono text-[15px]", counts[s] ? "text-ink" : "text-faint")}>{counts[s]}</span>
            </span>
          ))}
        </span>
      ),
      sub: "critical · urgent · watch",
    },
    {
      k: "Page",
      v: openPage ? (
        <span className="flex items-center gap-2">
          <PriorityChip p={openPage.priority} size="sm" />
          <span className="font-mono text-[13.5px]">{openPage.id}</span>
        </span>
      ) : (
        "None open"
      ),
      sub: openPage ? PAGE_STATUS_LABEL[openPage.status as PageStatus] ?? openPage.status : u.monitoring.pages_from === "room monitor" ? "the room monitor pages here" : "paged by the sepsis detector",
    },
    {
      k: replay ? "Replay clock" : "Last sample",
      v: replay ? `ICU hour ${u.record.admission.icu_hour ?? "—"}` : last ? `${since(last, u.now)} ago` : "—",
      sub: replay ? "real stay, read up to this hour" : u.monitoring.sample_s ? `a set every ${u.monitoring.sample_s} s` : "live stream",
    },
  ];
  return (
    <div className="mb-3 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-4 xl:grid-cols-7">
      {cells.map((c) => (
        <div key={c.k} className="min-w-0 bg-s1 px-3.5 py-2.5">
          <div className="truncate text-[11px] text-dim">{c.k}</div>
          <div className="num mt-1 flex min-h-6 items-center truncate font-mono text-[17px] font-medium text-ink">{c.v}</div>
          <div className="mt-0.5 truncate text-[11px] text-dim">{c.sub}</div>
        </div>
      ))}
      <div className="bg-s1 xl:hidden" aria-hidden />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Detections                                                           */
/* ------------------------------------------------------------------ */
function Detections({ u, active, tracked, system, onClear }: { u: Required<Unit>; active: Finding[]; tracked: Finding[]; system: string | null; onClear: () => void }) {
  const [tab, setTab] = useState<"active" | "resolved">("active");
  // a paging finding belongs to its track's open episode: medication findings to pharmacy's, the rest to the room's
  const pageOf = useMemo(() => {
    const m: Record<string, string> = {};
    const phys = u.episode?.open ? u.episode.page : undefined;
    const med = u.med_episode?.open ? u.med_episode.page : undefined;
    for (const f of u.findings) {
      if (f.page == null) continue;
      const p = f.page_category === "medication" ? med : phys;
      if (p) m[f.key] = p;
    }
    return m;
  }, [u.episode, u.med_episode, u.findings]);
  return (
    <div id="detections" className="scroll-mt-3">
    <Panel
      title="Detections"
      meta={`${u.monitoring.detectors.length} detectors over this room's record and vitals · re-run every ${u.monitoring.eval_period_s} s`}
      right={
        <Segmented
          size="sm"
          value={tab}
          onChange={setTab}
          options={[
            { value: "active", label: `Active ${active.length}` },
            { value: "resolved", label: `Resolved ${u.resolved.length}` },
          ]}
        />
      }
      flush
    >
      {system && (
        <div className="flex items-center gap-2 border-b border-line bg-s2/50 px-3.5 py-1.5 text-[12px]">
          <span className="text-dim">Showing</span>
          <span className="font-medium text-ink">{system}</span>
          <span className="text-dim">findings only</span>
          <button onClick={onClear} className="ml-auto rounded px-1.5 text-[11.5px] text-accent-ink hover:underline">
            Show all
          </button>
        </div>
      )}
      {tab === "active" ? (
        <>
          {active.length === 0 ? (
            <div className="px-3.5 py-6 text-center text-[12.5px] text-dim">{system ? `No active finding implicates ${system.toLowerCase()} right now.` : "Nothing above information level. Every detector is running."}</div>
          ) : (
            <ul className="divide-y divide-line">
              {active.map((f) => (
                <FindingRow key={f.key} f={f} now={u.now} page={pageOf[f.key]} clockMode={u.clock} />
              ))}
            </ul>
          )}
          {tracked.length > 0 && (
            <div className="border-t border-line px-3.5 py-2.5">
              <div className="mb-1.5 text-[11px] font-medium text-dim">Also tracked</div>
              <ul className="space-y-1">
                {tracked.map((f) => (
                  <li key={f.key} className="grid grid-cols-[112px_minmax(0,1fr)] gap-2 text-[12px]">
                    <span className="truncate text-dim">{f.detector}</span>
                    <span className="truncate text-muted" title={`${f.response} (${f.source})`}>
                      {f.title.startsWith(f.detector) ? f.title.slice(f.detector.length).trim() || f.title : f.title}
                    </span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {u.faults.length > 0 && (
            <div className="border-t border-line px-3.5 py-2 text-[11.5px] text-high-ink">Detector faults: {u.faults.join("; ")}</div>
          )}
        </>
      ) : u.resolved.length === 0 ? (
        <div className="px-3.5 py-6 text-center text-[12.5px] text-dim">Nothing has resolved in this room yet.</div>
      ) : (
        <ul className="divide-y divide-line-soft">
          {u.resolved.map((f, i) => (
            <li key={`${f.key}-${i}`} className="flex items-start gap-2.5 px-3.5 py-2">
              <SevDot sev={f.peak ?? f.severity} className="mt-[5px]" />
              <div className="min-w-0 flex-1">
                <div className="text-[12.5px] text-ink">{f.title}</div>
                <div className="text-[11px] text-dim">
                  {f.detector} · peaked {SEV_LABEL[f.peak ?? f.severity].toLowerCase()} · {f.first_seen ? clock(f.first_seen, false) : ""}–{f.resolved_at ? clock(f.resolved_at, false) : ""}
                </div>
              </div>
            </li>
          ))}
        </ul>
      )}
    </Panel>
    </div>
  );
}

function FindingRow({ f, now, page, clockMode }: { f: Finding; now: number; page?: string; clockMode: Unit["clock"] }) {
  const [open, setOpen] = useState(SEV_RANK[f.severity] >= 2);
  return (
    <li className="relative">
      <span className="absolute inset-y-0 left-0 w-[3px]" style={{ background: `rgb(${SEV_RGB[f.severity]})` }} />
      <button onClick={() => setOpen((o) => !o)} className="flex w-full items-start gap-2.5 py-2.5 pl-4 pr-3.5 text-left transition-colors hover:bg-s2/60" aria-expanded={open}>
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-x-2 gap-y-1">
            <SevTag sev={f.severity} />
            <span className="text-[13px] font-medium leading-snug text-ink">{f.title}</span>
          </div>
          <div className="mt-1 flex flex-wrap items-center gap-x-2 text-[11px] text-dim">
            {!f.source.startsWith(f.detector) && (
              <>
                <span>{f.detector}</span>
                <span className="text-faint">·</span>
              </>
            )}
            <span className="truncate">{f.source}</span>
          </div>
        </div>
        <div className="shrink-0 text-right">
          <div className="num font-mono text-[11px] text-dim">{f.first_seen != null && clockMode === "wall" ? since(f.first_seen, now) : ""}</div>
          {f.page != null && (
            <div className="mt-0.5 text-[10.5px]" style={{ color: SEV_INK[f.severity] }}>
              {page ? `on ${page}` : `pages at P${f.page}`}
            </div>
          )}
        </div>
      </button>
      {open && (
        <div className="space-y-2 pb-3 pl-4 pr-3.5">
          {f.evidence.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {f.evidence.map((e, i) => (
                <span key={i} className="inline-flex h-6 items-center gap-1.5 rounded-[4px] border border-line bg-bg px-1.5 text-[11.5px]">
                  <span className="text-dim">{e.label}</span>
                  {e.value !== "" && e.value != null && (
                    <span className="num font-mono text-ink">
                      {fmtVal(e.value)}
                      {e.unit && <span className="ml-0.5 text-dim">{e.unit}</span>}
                    </span>
                  )}
                  {e.points != null && (
                    <span
                      className="num grid h-4 min-w-4 place-items-center rounded-[3px] px-0.5 font-mono text-[10px] font-semibold"
                      style={e.points > 0 ? { color: SEV_INK[POINT_SEV[Math.min(3, e.points)]], background: `rgb(${SEV_RGB[POINT_SEV[Math.min(3, e.points)]]} / .16)` } : { color: "var(--color-faint)", background: "var(--color-s2)" }}
                    >
                      {e.points}
                    </span>
                  )}
                </span>
              ))}
            </div>
          )}
          {f.context.length > 0 && (
            <ul className="space-y-1">
              {f.context.map((c) => (
                <li key={c} className="flex items-start gap-2 text-[12px] leading-snug text-muted">
                  <BookOpen size={12} className="mt-[2px] shrink-0 text-accent-ink" aria-label="From the record" />
                  <span>{c}</span>
                </li>
              ))}
            </ul>
          )}
          {f.response && (
            <div className="rounded-md border border-line bg-s2/60 px-2.5 py-1.5 text-[12px] leading-snug text-ink/90">
              <span className="text-dim">Response · </span>
              {f.response}
            </div>
          )}
        </div>
      )}
    </li>
  );
}

/* ------------------------------------------------------------------ */
/* Vitals                                                               */
/* ------------------------------------------------------------------ */
type Channel = { key: keyof Vitals; label: string; unit: string; digits?: number; normal: (scale2: boolean) => [number, number]; points: (v: number, scale2: boolean, o2: boolean) => number };
const CHANNELS: Channel[] = [
  { key: "hr", label: "Pulse", unit: "/min", normal: () => [51, 90], points: (v) => (v <= 40 ? 3 : v <= 50 ? 1 : v <= 90 ? 0 : v <= 110 ? 1 : v <= 130 ? 2 : 3) },
  { key: "sbp", label: "Systolic", unit: "mmHg", normal: () => [111, 219], points: (v) => (v <= 90 ? 3 : v <= 100 ? 2 : v <= 110 ? 1 : v <= 219 ? 0 : 3) },
  { key: "rr", label: "Resp. rate", unit: "/min", normal: () => [12, 20], points: (v) => (v <= 8 ? 3 : v <= 11 ? 1 : v <= 20 ? 0 : v <= 24 ? 2 : 3) },
  {
    key: "spo2",
    label: "SpO₂",
    unit: "%",
    normal: (s2) => (s2 ? [88, 92] : [96, 100]),
    points: (v, s2, o2) => (s2 ? (v <= 83 ? 3 : v <= 85 ? 2 : v <= 87 ? 1 : v <= 92 || !o2 ? 0 : v <= 94 ? 1 : v <= 96 ? 2 : 3) : v <= 91 ? 3 : v <= 93 ? 2 : v <= 95 ? 1 : 0),
  },
  { key: "temp_c", label: "Temperature", unit: "°C", digits: 1, normal: () => [36.1, 38.0], points: (v) => (v <= 35 ? 3 : v <= 36 ? 1 : v <= 38 ? 0 : v <= 39 ? 1 : 2) },
  { key: "map", label: "Mean arterial", unit: "mmHg", normal: () => [65, 110], points: (v) => (v < 65 ? 3 : 0) },
];

function VitalsPanel({ u }: { u: Required<Unit> }) {
  const [ref, w] = useWidth<HTMLDivElement>(420);
  const replay = u.kind === "replay";
  const series = u.series;
  const scale2 = u.record.spo2_target?.scale === 2;
  const base = u.record.baseline;
  const latest = u.vitals ?? {};
  const o2 = (latest.o2_lpm ?? 0) > 0;
  const chartW = Math.max(120, w - 96 - 84 - 24 - 28 - 6);
  const span = series.length ? (series.at(-1)!.t ?? 0) - (series[0].t ?? 0) : 0;
  const meta = replay
    ? `PhysioNet/CinC 2019, hourly, ICU hours ${series[0]?.t ?? 0}–${series.at(-1)?.t ?? 0}`
    : u.kind === "live"
      ? "Live session"
      : `Simulated monitor · a set every ${u.monitoring.sample_s ?? 10} s · last ${Math.max(1, Math.round(span / 60))} min`;
  const shown = CHANNELS.filter((c) => series.some((s) => s[c.key] != null));
  return (
    <Panel title="Vitals" meta={meta} flush>
      <div ref={ref}>
        {shown.length === 0 ? (
          <div className="px-3.5 py-6 text-center text-[12.5px] text-dim">No vitals yet.</div>
        ) : (
          <div className="divide-y divide-line-soft">
            {shown.map((c) => {
              const data = series.map((s) => (s[c.key] as number | undefined) ?? null);
              let cur: number | null = null;
              for (let i = data.length - 1; i >= 0; i--) {
                if (data[i] != null) {
                  cur = data[i];
                  break;
                }
              }
              const pts = cur == null ? 0 : c.points(cur, scale2, o2);
              const sev = POINT_SEV[Math.min(3, pts)];
              const b = base ? (base as unknown as Record<string, number>)[c.key as string] : undefined;
              return (
                <div key={c.key as string} className="grid grid-cols-[96px_minmax(0,1fr)_84px] items-center gap-3 px-3.5 py-2">
                  <div className="min-w-0">
                    <div className="truncate text-[11px] text-dim">{c.label}</div>
                    <div className="flex items-baseline gap-1">
                      <span className="num font-mono text-[18px] font-semibold leading-tight" style={{ color: pts > 0 ? SEV_INK[sev] : "var(--color-ink)" }}>
                        {cur == null ? "—" : c.digits ? cur.toFixed(c.digits) : Math.round(cur)}
                      </span>
                      <span className="text-[10.5px] text-dim">{c.unit}</span>
                    </div>
                  </div>
                  <Spark data={data} w={chartW} h={34} color="#a8afb8" band={c.normal(scale2)} dot={pts > 0 ? SEV_INK[sev] : "#e8eaed"} />
                  <div className="text-right text-[10.5px] leading-tight text-dim">
                    {!replay && c.key !== "map" && (
                      <div>
                        NEWS2 <span className="num font-mono" style={{ color: pts > 0 ? SEV_INK[sev] : undefined }}>{pts}</span>
                      </div>
                    )}
                    {b != null && (
                      <div title={base?.source}>
                        baseline <span className="num font-mono text-muted">{c.digits ? b.toFixed(1) : Math.round(b)}</span>
                      </div>
                    )}
                    {c.key === "spo2" && scale2 && <div>target 88–92</div>}
                  </div>
                </div>
              );
            })}
            {!replay && (latest.avpu != null || latest.o2_lpm != null) && (
              <div className="grid grid-cols-[96px_minmax(0,1fr)_84px] items-center gap-3 px-3.5 py-2">
                <div className="text-[11px] text-dim">Consciousness · O₂</div>
                <div className="text-[12.5px] text-ink">
                  {{ A: "Alert", C: latest.chronic_confusion ? "Confused (baseline)" : "New confusion", V: "Responds to voice", P: "Responds to pain", U: "Unresponsive" }[latest.avpu ?? "A"]}
                  <span className="text-dim"> · {o2 ? `oxygen ${latest.o2_lpm} L/min` : "room air"}</span>
                </div>
                <div />
              </div>
            )}
          </div>
        )}
        {!replay && u.news2_history.length > 1 && <News2History hist={u.news2_history} w={w} />}
      </div>
    </Panel>
  );
}

function News2History({ hist, w }: { hist: { t: number; score: number }[]; w: number }) {
  const h = 72;
  const pad = 14;
  const iw = Math.max(80, w - 28 - pad);
  const max = Math.max(9, ...hist.map((x) => x.score));
  const x = (i: number) => pad + (i / Math.max(1, hist.length - 1)) * iw;
  const y = (v: number) => h - 6 - (v / max) * (h - 12);
  const d = hist.map((p, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(p.score).toFixed(1)}`).join("");
  return (
    <div className="border-t border-line px-3.5 pb-2.5 pt-2">
      <div className="mb-1 flex items-center justify-between text-[11px] text-dim">
        <span>NEWS2 over the last {Math.max(1, Math.round((hist.at(-1)!.t - hist[0].t) / 60))} min</span>
        <span>
          <span className="text-dys-ink">— 5</span> urgent · <span className="text-high-ink">— 7</span> emergency
        </span>
      </div>
      <svg width={w - 28} height={h} className="block overflow-visible" aria-hidden>
        {[5, 7].map((t) => (
          <g key={t}>
            <line x1={pad} x2={pad + iw} y1={y(t)} y2={y(t)} stroke={t === 7 ? "rgb(229 72 77 / .55)" : "rgb(245 118 46 / .5)"} strokeDasharray="3 3" strokeWidth={1} />
            <text x={0} y={y(t) + 3} fontSize={9} fill="var(--color-dim)" fontFamily="var(--font-geist-mono)">
              {t}
            </text>
          </g>
        ))}
        <path d={d} fill="none" stroke="#e8eaed" strokeWidth={1.4} strokeLinejoin="round" />
        {hist.map((p, i) =>
          p.score >= 5 ? <circle key={i} cx={x(i)} cy={y(p.score)} r={1.8} fill={p.score >= 7 ? "#ff6b70" : "#ff9357"} /> : null,
        )}
      </svg>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Labs                                                                 */
/* ------------------------------------------------------------------ */
function LabsPanel({ u }: { u: Required<Unit> }) {
  const replay = u.kind === "replay";
  const rows = Object.entries(u.labs)
    .filter(([, xs]) => xs.length > 0)
    .map(([k, xs]) => {
      const cur = xs[xs.length - 1];
      const prev = xs.length > 1 ? xs[xs.length - 2] : null;
      const base = xs.find((x) => x.baseline);
      return { k, cur, prev, base, n: xs.length, flag: labFlag(k, cur.v) };
    })
    .sort((a, b) => Number(!!b.flag) - Number(!!a.flag) || a.k.localeCompare(b.k));
  return (
    <Panel title="Labs" meta={replay ? "Every result the stay has recorded so far" : "This admission's results, and the record's baseline"} flush>
      {rows.length === 0 ? (
        <div className="px-3.5 py-6 text-center text-[12.5px] text-dim">No results on the record.</div>
      ) : (
        <table className="w-full text-[12px]">
          <thead>
            <tr className="border-b border-line text-left text-[11px] text-dim">
              <th className="px-3.5 py-1.5 font-medium">Test</th>
              <th className="px-2 py-1.5 text-right font-medium">Latest</th>
              <th className="px-2 py-1.5 text-right font-medium">Before</th>
              <th className="hidden px-2 py-1.5 text-right font-medium sm:table-cell">Baseline</th>
              <th className="px-3.5 py-1.5 text-right font-medium">When</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const [label, unit] = LAB_LABEL[r.k] ?? [r.k, ""];
              const dir = r.prev ? (r.cur.v > r.prev.v ? "↑" : r.cur.v < r.prev.v ? "↓" : "→") : "";
              return (
                <tr key={r.k} className="border-b border-line-soft last:border-b-0">
                  <td className="px-3.5 py-1.5">
                    <span className="text-ink">{label}</span>
                    {unit && <span className="ml-1.5 text-[10.5px] text-faint">{unit}</span>}
                  </td>
                  <td className={cx("num px-2 py-1.5 text-right font-mono font-medium", r.flag ? "text-dys-ink" : "text-ink")}>
                    {fmtVal(r.cur.v, 1)}
                    {r.flag && <span className="ml-1 text-[10px]">{r.flag === "high" ? "H" : "L"}</span>}
                  </td>
                  <td className="num px-2 py-1.5 text-right font-mono text-dim">
                    {r.prev ? fmtVal(r.prev.v, 1) : "—"} <span className="text-faint">{dir}</span>
                  </td>
                  <td className="num hidden px-2 py-1.5 text-right font-mono text-dim sm:table-cell">{r.base ? fmtVal(r.base.v, 2) : "—"}</td>
                  <td className="num px-3.5 py-1.5 text-right font-mono text-[11px] text-dim">{replay ? `h ${r.cur.t}` : `${since(r.cur.t, u.now)} ago`}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
      )}
    </Panel>
  );
}

/* ------------------------------------------------------------------ */
/* The record                                                           */
/* ------------------------------------------------------------------ */
function RecordPanel({ rec, now, replay }: { rec: RoomRecord; now: number; replay: boolean }) {
  const [tab, setTab] = useState<"summary" | "orders" | "history">("summary");
  const active = rec.orders.filter((o) => o.status === "active");
  const held = rec.orders.filter((o) => o.status !== "active");
  const onOrders = new Set(rec.orders.filter((o) => o.status === "active").flatMap((o) => [o.name.toLowerCase(), (o.generic ?? "").toLowerCase()]));
  return (
    <Panel
      title="Record"
      meta={rec.provenance.label}
      right={
        <Segmented
          size="sm"
          value={tab}
          onChange={setTab}
          options={[
            { value: "summary", label: "Summary" },
            { value: "orders", label: `Orders ${active.length}` },
            { value: "history", label: `History ${rec.encounters.length}` },
          ]}
        />
      }
      flush
    >
      {tab === "summary" && (
        <div className="divide-y divide-line-soft">
          <Block title="Source">
            <p className="text-[12px] leading-snug text-muted">{rec.provenance.detail}</p>
          </Block>
          <Block title="Problems">
            <ul className="space-y-1">
              {rec.problems.map((p) => (
                <li key={p.name} className="flex items-baseline gap-2 text-[12.5px]">
                  <span className={cx("min-w-0 flex-1", p.status === "active" ? "font-medium text-ink" : "text-ink/90")}>{p.name}</span>
                  <span className="shrink-0 text-[11px] text-dim">{p.note ?? (p.since === "this admission" || p.since === "this stay" ? p.since : `since ${p.since}`)}</span>
                </li>
              ))}
            </ul>
          </Block>
          {rec.baseline ? (
            <Block title="Personal baseline" right={<span className="text-[11px] text-dim">{rec.baseline.source}</span>}>
              <div className="grid grid-cols-3 gap-px overflow-hidden rounded-md border border-line bg-line text-center">
                {[
                  ["Pulse", Math.round(rec.baseline.hr)],
                  ["BP", `${Math.round(rec.baseline.sbp)}/${Math.round(rec.baseline.dbp)}`],
                  ["RR", Math.round(rec.baseline.rr)],
                  ["SpO₂", Math.round(rec.baseline.spo2)],
                  ["Temp", rec.baseline.temp_c.toFixed(1)],
                  ["Mental", rec.baseline.mental === "alert" ? "Alert" : "Confused"],
                ].map(([k, v]) => (
                  <div key={String(k)} className="bg-s1 px-1.5 py-1.5">
                    <div className="text-[10px] text-dim">{k}</div>
                    <div className="num truncate font-mono text-[12.5px] text-ink">{v}</div>
                  </div>
                ))}
              </div>
              {rec.spo2_target && <p className="mt-2 text-[12px] leading-snug text-muted">SpO₂ target {rec.spo2_target.lo}–{rec.spo2_target.hi}%: {rec.spo2_target.why}.</p>}
            </Block>
          ) : (
            <Block title="Personal baseline">
              <p className="text-[12px] leading-snug text-dim">{replay ? "No earlier encounters in the dataset. Baselines come from this stay's first hours." : "None on the record."}</p>
            </Block>
          )}
          <Block title="Allergies">
            {rec.allergies.length === 0 ? (
              <p className="text-[12.5px] text-muted">No known allergies{replay ? " (not recorded in the dataset)" : ""}.</p>
            ) : (
              <ul className="space-y-1">
                {rec.allergies.map((a) => (
                  <li key={a.substance} className="flex items-baseline gap-2 text-[12.5px]">
                    <span className="min-w-0 flex-1 text-ink">{a.substance}</span>
                    <span className="text-[11.5px] text-muted">{a.reaction}</span>
                    <span className={cx("text-[11px]", a.severity === "severe" ? "text-high-ink" : "text-dim")}>{a.severity}</span>
                  </li>
                ))}
              </ul>
            )}
          </Block>
          {(rec.planned.length > 0 || rec.missing?.length || rec.linked.length > 0) && (
            <Block title={rec.planned.length ? "Planned" : "Sources"}>
              {rec.planned.map((p) => (
                <div key={p} className="text-[12.5px] text-ink">
                  {p}
                </div>
              ))}
              {rec.linked.map((l, i) => (
                <div key={i} className="text-[12px] text-muted">
                  Linked: {l.label ?? l.source}
                  {l.summary && <span className="text-dim"> · {l.summary}</span>}
                </div>
              ))}
              {rec.missing && rec.missing.length > 0 && (
                <p className="text-[12px] leading-snug text-dim">Not in this dataset: {rec.missing.join(", ")}. Link an external record from the ICU record to fill them.</p>
              )}
            </Block>
          )}
        </div>
      )}
      {tab === "orders" && (
        <div className="divide-y divide-line-soft">
          <Block title="Active orders">
            {active.length === 0 ? (
              <p className="text-[12.5px] text-dim">{replay ? "The dataset carries no medication orders." : "None."}</p>
            ) : (
              <ul className="space-y-1.5">
                {active.map((o) => (
                  <li key={o.name} className="text-[12.5px]">
                    <div className="flex items-baseline gap-2">
                      <span className="min-w-0 flex-1 font-medium text-ink">{o.name}</span>
                      {o.prn && <span className="text-[10.5px] text-dim">PRN</span>}
                      {o.infusion && <span className="text-[10.5px] text-dim">pump</span>}
                      {o.simulated && <span className="text-[10.5px] text-dim">simulated</span>}
                    </div>
                    <div className="text-[11.5px] text-muted">
                      {o.kind === "drug" ? `${o.dose ?? ""} ${o.route ?? ""} ${o.freq ?? ""}`.trim() : o.detail}
                      {o.last_given_at && <span className="text-dim"> · given {since(o.last_given_at, now)} ago</span>}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </Block>
          {held.length > 0 && (
            <Block title="Held">
              <ul className="space-y-1">
                {held.map((o) => (
                  <li key={o.name} className="text-[12.5px]">
                    <span className="text-muted line-through decoration-faint">{o.name}</span>
                    {o.note && <span className="text-[11.5px] text-dim"> · {o.note}</span>}
                  </li>
                ))}
              </ul>
            </Block>
          )}
          <Block title="Home medications">
            {rec.home_meds.length === 0 ? (
              <p className="text-[12.5px] text-dim">None recorded.</p>
            ) : (
              <ul className="space-y-1">
                {rec.home_meds.map((h) => {
                  const cont = onOrders.has(h.name.toLowerCase());
                  return (
                    <li key={h.name} className="flex items-baseline gap-2 text-[12.5px]">
                      <span className="min-w-0 flex-1 text-ink/90">
                        {h.name} <span className="text-[11.5px] text-dim">{`${h.dose} ${h.freq}`.trim()}</span>
                      </span>
                      <span className={cx("text-[11px]", cont ? "text-muted" : "text-dim")}>{cont ? "continued" : "not on orders"}</span>
                    </li>
                  );
                })}
              </ul>
            )}
          </Block>
        </div>
      )}
      {tab === "history" && (
        <div className="px-3.5 py-3">
          {rec.encounters.length === 0 ? (
            <p className="text-[12.5px] text-dim">{replay ? "PhysioNet/CinC 2019 has no prior encounters." : "No earlier encounters."}</p>
          ) : (
            <ol className="relative space-y-3 border-l border-line pl-4">
              {rec.encounters.map((e) => (
                <li key={e.at} className="relative">
                  <span className="absolute -left-[21px] top-1 h-2.5 w-2.5 rounded-full border-2 border-s1 bg-line-bright" />
                  <div className="flex flex-wrap items-baseline gap-x-2">
                    <span className="text-[12.5px] font-medium text-ink">{e.reason}</span>
                    <span className="text-[11px] text-dim">
                      {e.setting}
                      {e.los_days ? ` · ${e.los_days} d` : ""} · {new Date(e.at * 1000).toLocaleDateString("en-GB", { month: "short", year: "numeric" })}
                    </span>
                  </div>
                  <div className="num mt-0.5 font-mono text-[11px] text-muted">
                    HR {e.vitals.hr} · BP {e.vitals.sbp}/{e.vitals.dbp} · RR {e.vitals.rr} · SpO₂ {e.vitals.spo2} · {e.vitals.temp_c} °C
                  </div>
                  {e.note && <div className="mt-0.5 text-[11.5px] leading-snug text-dim">{e.note}</div>}
                </li>
              ))}
            </ol>
          )}
        </div>
      )}
    </Panel>
  );
}

function Block({ title, right, children }: { title: string; right?: React.ReactNode; children: React.ReactNode }) {
  return (
    <div className="px-3.5 py-2.5">
      <div className="mb-1.5 flex items-center gap-2">
        <span className="text-[11px] font-medium text-dim">{title}</span>
        {right && <span className="ml-auto">{right}</span>}
      </div>
      {children}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Linked objects: people, devices, pages                               */
/* ------------------------------------------------------------------ */
const STATUS_DOT: Record<string, string> = { available: "bg-st-available", paging: "bg-st-paging", in_procedure: "bg-st-procedure", on_case: "bg-st-case", off_shift: "bg-st-off" };

function LinkedPanel({ u }: { u: Required<Unit> }) {
  return (
    <Panel title="Linked" meta="The people, devices and pages attached to this room" flush>
      <div className="divide-y divide-line-soft">
        <Block title="Care team">
          <ul className="space-y-1.5">
            {u.care_team.map((c) => (
              <li key={`${c.id}-${c.role}`} className="flex items-center gap-2.5">
                <span className={cx("h-2 w-2 shrink-0 rounded-full", STATUS_DOT[c.status] ?? "bg-dim")} title={c.status.replace("_", " ")} />
                <Link href={`/clinician/?id=${encodeURIComponent(c.id)}`} className="min-w-0 flex-1 truncate text-[12.5px] text-ink hover:underline">
                  {c.name}
                </Link>
                <span className="shrink-0 text-[11px] text-dim">{c.role}</span>
              </li>
            ))}
          </ul>
        </Block>
        <Block title="Devices">
          <ul className="space-y-1.5">
            {u.devices.map((d) => (
              <li key={d.name + d.detail} className="flex items-start gap-2.5">
                <span className={cx("mt-[5px] h-1.5 w-1.5 shrink-0 rounded-full", d.status === "no signal" ? "bg-high" : "bg-low")} />
                <div className="min-w-0 flex-1">
                  <div className="text-[12.5px] text-ink">{d.name}</div>
                  <div className="truncate text-[11px] text-dim">{d.detail}</div>
                </div>
                <span className={cx("shrink-0 text-[11px]", d.status === "no signal" ? "text-high-ink" : "text-dim")}>{d.status}</span>
              </li>
            ))}
          </ul>
        </Block>
        <Block title="Pages">
          {u.pages.length === 0 ? (
            <p className="text-[12.5px] text-dim">No pages for this room.</p>
          ) : (
            <ul className="space-y-1">
              {u.pages.slice(0, 6).map((p) => (
                <li key={p.id}>
                  <Link href={`/operator/?page=${encodeURIComponent(p.id)}`} className="group flex items-center gap-2 rounded-[4px] py-0.5 transition-colors">
                    <PriorityChip p={p.priority} size="sm" />
                    <span className="min-w-0 flex-1 truncate text-[12px] text-ink/90 group-hover:text-ink">{p.text}</span>
                    <span className={cx("shrink-0 text-[11px]", p.active ? "text-ink" : "text-dim")}>{PAGE_STATUS_LABEL[p.status as PageStatus] ?? p.status}</span>
                  </Link>
                </li>
              ))}
            </ul>
          )}
        </Block>
      </div>
    </Panel>
  );
}

/** A time of day today; a date with it for anything older. */
function stamp(t: number, now: number): string {
  if (now - t < 20 * 3600) return clock(t);
  return new Date(t * 1000).toLocaleString("en-GB", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit", hour12: false });
}

/* ------------------------------------------------------------------ */
/* Timeline: everything that happened in this room, on one axis         */
/* ------------------------------------------------------------------ */
type Lane = "detections" | "pages" | "care" | "course";
const LANES: { key: Lane; label: string }[] = [
  { key: "detections", label: "Detections" },
  { key: "pages", label: "Pages & alerts" },
  { key: "care", label: "Orders & labs" },
  { key: "course", label: "Course & reports" },
];
const laneOf = (e: TimelineEvent): Lane =>
  e.kind === "finding" || e.kind === "escalated" || e.kind === "resolved"
    ? "detections"
    : e.kind === "page" || e.kind === "alert" || e.kind === "episode"
      ? "pages"
      : e.kind === "lab" || e.kind === "order" || e.kind === "record"
        ? "care"
        : "course";

function TimelinePanel({ u }: { u: Required<Unit> }) {
  const [filter, setFilter] = useState<"all" | Lane>("all");
  const [ref, w] = useWidth<HTMLDivElement>(900);
  const windowS = 60 * 60;
  const t1 = u.now;
  const t0 = t1 - windowS;
  const recent = u.timeline.filter((e) => e.t >= t0);
  const list = u.timeline.filter((e) => filter === "all" || laneOf(e) === filter);
  const labelW = 118;
  const iw = Math.max(100, w - labelW - 28);
  const x = (t: number) => labelW + ((t - t0) / windowS) * iw;
  const laneH = 26;
  const ticks = [0, 15, 30, 45, 60];
  return (
    <Panel
      title="Timeline"
      meta="Findings, pages, orders, labs and the patient's course, merged on one clock"
      right={
        <Segmented
          size="sm"
          value={filter}
          onChange={setFilter}
          options={[{ value: "all", label: "All" }, ...LANES.map((l) => ({ value: l.key, label: l.label.split(" ")[0] }))]}
        />
      }
      flush
    >
      <div ref={ref} className="hidden border-b border-line px-3.5 pb-2 pt-3 md:block">
        <svg width={w - 28} height={LANES.length * laneH + 18} className="block" aria-label="Last hour in this room">
          {LANES.map((l, i) => (
            <g key={l.key}>
              <text x={0} y={i * laneH + 15} fontSize={11} fill="var(--color-dim)">
                {l.label}
              </text>
              <line x1={labelW} x2={labelW + iw} y1={i * laneH + 11} y2={i * laneH + 11} stroke="var(--color-line)" strokeWidth={1} />
            </g>
          ))}
          {ticks.map((m) => (
            <g key={m}>
              <line x1={x(t0 + m * 60)} x2={x(t0 + m * 60)} y1={0} y2={LANES.length * laneH} stroke="var(--color-line-soft)" strokeWidth={1} />
              <text x={x(t0 + m * 60)} y={LANES.length * laneH + 13} fontSize={9.5} textAnchor={m === 60 ? "end" : m === 0 ? "start" : "middle"} fill="var(--color-faint)" fontFamily="var(--font-geist-mono)">
                {m === 60 ? "now" : `−${60 - m}m`}
              </text>
            </g>
          ))}
          {recent.map((e, i) => {
            const li = LANES.findIndex((l) => l.key === laneOf(e));
            const cy = li * laneH + 11;
            const cx0 = x(e.t);
            const col = e.kind === "resolved" ? "#6c737e" : e.kind === "page" || e.kind === "alert" ? `rgb(${SEV_RGB[e.severity === "info" && e.priority ? (["critical", "urgent", "watch", "info"] as Severity[])[e.priority - 1] : e.severity]})` : e.severity === "info" ? "#868d97" : `rgb(${SEV_RGB[e.severity]})`;
            return (
              <g key={i}>
                <title>{`${clock(e.t)} · ${e.text}`}</title>
                {e.kind === "page" || e.kind === "alert" ? (
                  <rect x={cx0 - 3.5} y={cy - 3.5} width={7} height={7} rx={1.5} fill={col} />
                ) : e.kind === "order" ? (
                  <path d={`M${cx0},${cy - 4.5}L${cx0 + 4.5},${cy}L${cx0},${cy + 4.5}L${cx0 - 4.5},${cy}Z`} fill={col} />
                ) : e.kind === "resolved" ? (
                  <circle cx={cx0} cy={cy} r={3.2} fill="var(--color-s1)" stroke={col} strokeWidth={1.4} />
                ) : (
                  <circle cx={cx0} cy={cy} r={e.kind === "lab" ? 2.6 : 3.6} fill={col} opacity={e.simulated ? 0.75 : 1} />
                )}
              </g>
            );
          })}
        </svg>
      </div>
      {list.length === 0 ? (
        <div className="px-3.5 py-6 text-center text-[12.5px] text-dim">Nothing yet.</div>
      ) : (
        <ul className="max-h-[360px] divide-y divide-line-soft overflow-y-auto">
          {list.slice(0, 80).map((e, i) => (
            <li key={i} className="grid grid-cols-[84px_120px_minmax(0,1fr)] items-start gap-2 px-3.5 py-1.5 max-sm:grid-cols-[72px_minmax(0,1fr)]">
              <span className="num pt-px font-mono text-[11px] text-dim" title={new Date(e.t * 1000).toLocaleString("en-GB")}>{stamp(e.t, u.now)}</span>
              <span className="truncate pt-px text-[11px] text-dim max-sm:hidden">
                <span
                  className="mr-1.5 inline-block h-1.5 w-1.5 rounded-full align-middle"
                  style={{ background: e.severity === "info" ? "#6c737e" : `rgb(${SEV_RGB[e.severity]})` }}
                />
                {LANES.find((l) => l.key === laneOf(e))?.label}
              </span>
              <span className={cx("text-[12.5px] leading-snug", e.kind === "resolved" ? "text-dim" : "text-ink/90")}>
                {e.text}
                {e.simulated && <span className="ml-1.5 text-[10.5px] text-faint">simulated</span>}
                {e.page && (
                  <Link href={`/operator/?page=${encodeURIComponent(e.page)}`} className="ml-1.5 inline-flex items-center gap-0.5 font-mono text-[11px] text-accent-ink hover:underline">
                    {e.page} <ExternalLink size={10} />
                  </Link>
                )}
              </span>
            </li>
          ))}
        </ul>
      )}
    </Panel>
  );
}
