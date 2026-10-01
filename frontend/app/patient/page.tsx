"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useMemo } from "react";

import { RiskChart, type RiskSeries } from "@/components/bed/chart";
import { Consult, DecisionCard, Organs, Reasoning, RiskHero, Vitals } from "@/components/bed/panels";
import { Orders } from "@/components/bed/orders";
import { BedSelector } from "@/components/bed/selector";
import { FeedPill } from "@/components/feed";
import { Page } from "@/components/shell";
import { BioTwin } from "@/components/twin";
import { Empty, Pill, Plate, Skeleton } from "@/components/ui";
import { title } from "@/lib/clinical";
import { useFeed, usePoll } from "@/lib/hooks";
import type { Decision, RosterPatient, Snapshot, Timeline } from "@/lib/types";

type Roster = { patients: RosterPatient[]; default: string };
type CohortChart = {
  hours: number[];
  series: Record<string, number[]>;
  risk: number[];
  population_risk: number[];
  alarm: number;
  icu_hour: number;
  forecast: { risk_mean: number[]; risk_lo: number[]; risk_hi: number[]; trajectory: string; earliest_breach_in: number | null };
};

// cohort chart channel → snapshot channel key
const CHART_KEY: Record<string, string> = {
  hr: "heart_rate",
  resp_rate: "respiratory_rate",
  spo2: "spo2",
  map: "map",
  temp_c: "skin_temp",
  lactate: "sweat_lactate",
};

export default function BedPage() {
  return (
    <Suspense>
      <Bedside />
    </Suspense>
  );
}

function Bedside() {
  const params = useSearchParams();
  const roster = usePoll<Roster>("/aegis/patients");
  const id = params.get("id") || roster.data?.default || null;
  const patient = roster.data?.patients.find((p) => p.id === id) ?? null;
  const live = patient?.kind === "live";
  const bed = live ? null : id;

  // The live session streams; a cohort bed is a replayed record, polled.
  const feed = useFeed<Snapshot>(roster.data && live ? "/aegis/stream" : null);
  const polled = usePoll<Snapshot>(roster.data && bed ? `/aegis/patient/snapshot?bed=${encodeURIComponent(bed)}` : null, 4000);
  const snap = live ? feed.data : polled.data;

  const decision = usePoll<Decision>(roster.data ? `/aegis/decision${bed ? `?bed=${encodeURIComponent(bed)}` : ""}` : null, 8000);
  const chart = usePoll<CohortChart>(bed ? `/aegis/cohort/${encodeURIComponent(bed)}/chart` : null, 15000);
  const timeline = usePoll<Timeline>(live ? "/aegis/timeline" : null, 10000);

  const series: RiskSeries | null = useMemo(() => {
    if (bed && chart.data) {
      const c = chart.data;
      const last = c.hours[c.hours.length - 1];
      return {
        x: c.hours,
        risk: c.risk,
        population: c.population_risk,
        forecast: c.forecast?.risk_mean?.map((m, i) => ({
          x: last + i + 1,
          mean: m,
          lo: c.forecast.risk_lo[i],
          hi: c.forecast.risk_hi[i],
        })),
        alarm: c.alarm,
        xLabel: "ICU hour",
      };
    }
    if (live && timeline.data && timeline.data.history.length > 1) {
      const t = timeline.data;
      return {
        x: t.history.map((h) => h.hour),
        risk: t.history.map((h) => h.probability),
        forecast: t.forecast.map((f) => ({ x: f.hour, mean: f.probability, lo: f.lo ?? f.probability, hi: f.hi ?? f.probability })),
        alarm: t.alarm_threshold,
        medium: t.medium_threshold,
        xLabel: "hour",
      };
    }
    return null;
  }, [bed, live, chart.data, timeline.data]);

  const trends = useMemo(() => {
    const out: Record<string, (number | null)[]> = {};
    if (bed && chart.data) {
      for (const [k, v] of Object.entries(chart.data.series)) if (CHART_KEY[k]) out[CHART_KEY[k]] = v.slice(-24);
    } else if (snap?.history?.channels) {
      for (const [k, v] of Object.entries(snap.history.channels)) out[k] = v.slice(-60);
    }
    return out;
  }, [bed, chart.data, snap]);

  const recorded = !live;
  const history = (snap?.patient.clinical_history ?? {}) as Record<string, unknown>;
  const flags = Object.entries(history)
    .filter(([, v]) => v === true)
    .map(([k]) => title(k));

  return (
    <Page
      context={
        <div className="flex items-center gap-3">
          {roster.data && <BedSelector roster={roster.data.patients} current={patient} />}
          {live ? (
            <FeedPill status={feed.status} lastAt={feed.lastAt} label={snap?.source === "simulator" ? "Simulator" : "Live"} />
          ) : (
            bed && <Pill tone="nodata">Replayed record</Pill>
          )}
        </div>
      }
    >
      {!snap ? (
        <div className="grid gap-3">
          <Skeleton className="h-36 rounded-lg" />
          <div className="grid gap-3 xl:grid-cols-3">
            <Skeleton className="h-[420px] rounded-lg" />
            <Skeleton className="h-[420px] rounded-lg" />
            <Skeleton className="h-[420px] rounded-lg" />
          </div>
        </div>
      ) : (
        <div className="grid gap-3">
          {/* identity + the one number */}
          <section className="panel flex flex-wrap items-end gap-x-10 gap-y-5 p-4">
            <div className="min-w-[240px]">
              <div className="text-[12px] text-dim">
                {live ? "Live monitored patient" : `${title(bed ?? "")} · ICU wing`}
                {snap.source && <span className="ml-2 text-faint">source · {snap.source}</span>}
              </div>
              <h1 className="mt-1 text-[24px] font-semibold tracking-[-0.01em]">{snap.patient.name}</h1>
              <div className="num mt-1 text-[13px] text-muted">
                {snap.patient.age != null && `${Math.round(snap.patient.age)} y`}
                {snap.patient.gender && ` · ${snap.patient.gender}`}
                {chart.data && ` · ICU hour ${chart.data.icu_hour}`}
              </div>
              {flags.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-1.5">
                  {flags.map((f) => (
                    <Pill key={f} tone="neutral">
                      {f}
                    </Pill>
                  ))}
                </div>
              )}
            </div>
            <div className="min-w-0 flex-1">
              <RiskHero snap={snap} recorded={recorded} />
            </div>
          </section>

          <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1.15fr)_minmax(0,1fr)]">
            <Plate title="Vitals" eyebrow={recorded ? "Hourly record, last 24 h" : "Streaming"}>
              <Vitals channels={(snap.body_state.channels ?? []) as never} trends={trends} />
            </Plate>
            <Plate
              title="Anatomical twin"
              eyebrow="Organs glow by their system's severity · drag to rotate"
              bodyClass="p-0"
              className="min-h-[460px]"
            >
              <div className="twin-viewport m-2 h-[444px]">
                <BioTwin snapshot={snap} className="h-full" />
              </div>
            </Plate>
            <Plate title="Organ systems" eyebrow="Unmeasured systems are shown as unknown, never as healthy">
              <Organs snap={snap} />
            </Plate>
          </div>

          <div className="grid gap-3 xl:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
            <Plate
              title="Risk over time"
              eyebrow={recorded ? "White: the risk the unit acts on · grey: population fusion · blue: its 8 h projection" : "Calibrated detector probability"}
              right={
                recorded && chart.data?.forecast ? (
                  <span className="text-[11.5px] text-dim">
                    population projection · {chart.data.forecast.trajectory}
                  </span>
                ) : timeline.data ? (
                  <span className="text-[11.5px] text-dim">{timeline.data.trajectory}</span>
                ) : null
              }
            >
              {series ? (
                <RiskChart s={series} />
              ) : (
                <Empty>{timeline.data?.refused ?? "Collecting history. A trend needs more than two readings."}</Empty>
              )}
            </Plate>
            <div className="flex min-w-0 flex-col gap-3">
              <Plate title="Decision" eyebrow="What the system recommends, and why">
                <DecisionCard d={decision.data} />
              </Plate>
              {bed && (
                <Plate title="Orders" eyebrow="Each write is applied only if the Z3 cage certifies it">
                  <Orders bed={bed} />
                </Plate>
              )}
            </div>
          </div>

          <div className="grid gap-3 xl:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
            <Plate
              title="Specialist panel"
              eyebrow={`${snap.reasoning?.opinions?.length ?? 0} reasoning agents deliberate on every tick`}
            >
              <Reasoning snap={snap} />
            </Plate>
            <Plate title="Dr. Aegis" eyebrow="Grounded consult on this patient" className="min-h-[420px]">
              <Consult bed={bed} name={snap.patient.name} />
            </Plate>
          </div>
        </div>
      )}
    </Page>
  );
}
