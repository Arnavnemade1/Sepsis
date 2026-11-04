"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useState } from "react";

import { BedSelector } from "@/components/bed/selector";
import { Close } from "@/components/icons";
import { Page, PageHead } from "@/components/shell";
import { Button, cx, Empty, Kv, Meter, Pill, Plate, Skeleton } from "@/components/ui";
import { api } from "@/lib/api";
import { pct, title, TONE_RGB } from "@/lib/clinical";
import { usePoll } from "@/lib/hooks";
import type { RosterPatient } from "@/lib/types";

type Roster = { patients: RosterPatient[]; default: string };

type Arm = {
  drug: string;
  label: string;
  drug_class: string;
  kind: string;
  unit: string;
  times_h: number[];
  plasma: number[];
  regimen: string;
  // concentration arms carry the PK metrics; the fluid arm (kind "volume") carries volumes
  metrics: {
    cmax?: number;
    tmax_h?: number;
    continuous?: boolean;
    clearance_l_h?: number;
    mic_mg_l?: number;
    pct_ft_above_mic?: number;
    infused_ml?: number;
    peak_plasma_expansion_ml?: number;
    plasma_retention_1h_post?: number;
    interstitial_ml_end?: number;
  };
  provenance: Record<string, string>;
  honesty: string;
};
type AdverseEvent = {
  key: string;
  label: string;
  drug: string;
  organ: string;
  severity: string;
  probability: number;
  ci_low: number;
  ci_high: number;
  baseline: number;
  onset_p50_h: number | null;
  monitoring: string;
  citation: string;
  evidence: string;
};
type Point = { text: string; kind: string; evidence: string };
type Option = {
  key: string;
  label: string;
  note: string;
  antibiotic: string | null;
  safe: boolean;
  verification: { z3_status: string; message: string; violations: string[] };
  arms: Arm[];
  adverse: { events: AdverseEvent[]; summary: { p_any_serious: number; independence_assumption?: string }; honesty: string };
  upsides: Point[];
  downsides: Point[];
  p_any_serious: number;
  n_serious: number;
};
type Cell = { text: string; v: number; bad: boolean };
type Options = {
  status: string;
  detail?: string;
  options: Option[];
  recommended: string;
  comparison: {
    headline: string;
    columns: { key: string; label: string; safe: boolean }[];
    rows: { key: string; label: string; note: string; cells: Record<string, Cell> }[];
    honesty: string;
  };
  patient: { weight_kg: number; egfr: number; hepatic_factor: number; map_mmhg: number };
  honesty: string;
  bed_name?: string;
  administered?: boolean;
};

export default function PharmacyPage() {
  return (
    <Suspense>
      <Pharmacy />
    </Suspense>
  );
}

function Pharmacy() {
  const params = useSearchParams();
  const roster = usePoll<Roster>("/aegis/patients");
  const id = params.get("id") || roster.data?.patients.find((p) => p.kind === "cohort")?.id || null;
  const patient = roster.data?.patients.find((p) => p.id === id) ?? null;
  const bed = patient?.kind === "live" ? null : id;
  const opts = usePoll<Options>(roster.data && id ? `/aegis/pk/options${bed ? `?bed=${encodeURIComponent(bed)}` : ""}` : null);
  const [pick, setPick] = useState<string | null>(null);
  useEffect(() => setPick(null), [id]);
  const data = opts.data?.status === "ok" ? opts.data : null;
  const chosen = data?.options.find((o) => o.key === (pick ?? data.recommended)) ?? data?.options[0];

  return (
    <Page
      context={
        roster.data && (
          <BedSelector roster={roster.data.patients} current={patient} hrefFor={(i) => `/pharmacy/?id=${encodeURIComponent(i)}`} />
        )
      }
    >
      <PageHead
        title={patient ? `Treatment options · ${patient.name}` : "Treatment options"}
        right={
          data && (
            <div className="flex flex-wrap gap-2">
              <Pill tone="neutral">eGFR {data.patient.egfr.toFixed(0)}</Pill>
              <Pill tone="neutral">{data.patient.weight_kg.toFixed(0)} kg</Pill>
              <Pill tone="neutral">MAP {data.patient.map_mmhg.toFixed(0)}</Pill>
            </div>
          )
        }
      >
        Each candidate order is projected through a pharmacokinetic model of this patient, scored against cited adverse-event rates, and
        checked against the safety invariants by Z3 before it can be given.
      </PageHead>

      {opts.loading && !data ? (
        <div className="grid grid-cols-1 gap-4">
          <Skeleton className="h-64 rounded-lg" />
          <Skeleton className="h-96 rounded-lg" />
        </div>
      ) : !data ? (
        <Plate>
          <Empty>{opts.data?.detail ?? opts.error ?? "Choose a patient."}</Empty>
        </Plate>
      ) : (
        <div className="grid grid-cols-1 gap-4">
          <Plate title="Compare the paths" eyebrow={data.comparison.headline} bodyClass="p-0">
            <div className="overflow-x-auto">
              <table className="w-full min-w-[860px] border-collapse text-sm">
                <thead>
                  <tr>
                    <th className="w-[180px] px-4 py-3 text-left font-normal text-dim" />
                    {data.comparison.columns.map((c) => {
                      const active = chosen?.key === c.key;
                      return (
                        <th key={c.key} className="px-2 py-2 text-left font-normal">
                          <button
                            onClick={() => setPick(c.key)}
                            className={cx(
                              "w-full rounded-md border px-2.5 py-2 text-left transition",
                              active ? "border-accent/50 bg-s3" : "border-transparent hover:bg-s2",
                            )}
                          >
                            <div className="text-sm text-ink">
                              {c.label}
                              {c.key === data.recommended && <span className="ml-1.5 text-2xs text-accent-ink">certified</span>}
                            </div>
                            <div className={cx("mt-0.5 text-xs", c.safe ? "text-dim" : "text-high-ink")}>
                              {c.safe ? "Z3: satisfies invariants" : "Z3: violates an invariant"}
                            </div>
                          </button>
                        </th>
                      );
                    })}
                  </tr>
                </thead>
                <tbody>
                  {data.comparison.rows.map((r) => (
                    <tr key={r.key} className="border-t border-line-soft">
                      <td className="px-4 py-2.5 align-top">
                        <div className="text-ink">{r.label}</div>
                        <div className="text-xs text-dim">{r.note}</div>
                      </td>
                      {data.comparison.columns.map((c) => {
                        const cell = r.cells[c.key];
                        return (
                          <td key={c.key} className={cx("px-4 py-2.5 align-top", chosen?.key === c.key && "bg-s2/40")}>
                            <span className={cx(cell?.bad ? "text-dys-ink" : "text-muted")}>{cell?.text ?? "—"}</span>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <p className="border-t border-line-soft px-4 py-3 text-xs leading-relaxed text-dim">{data.comparison.honesty}</p>
          </Plate>

          {chosen && <OptionDetail o={chosen} />}

          <InteractionScreen
            key={id ?? "none"}
            seed={chosen?.arms.map((a) => a.drug.split("-")[0]) ?? []}
            renal={data.patient.egfr < 45}
          />
          <p className="text-xs leading-relaxed text-dim">{data.honesty}</p>
        </div>
      )}
    </Page>
  );
}

function OptionDetail({ o }: { o: Option }) {
  const events = [...o.adverse.events].sort((a, b) => b.probability - a.probability).slice(0, 6);
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1.4fr)_minmax(0,1fr)]">
      <Plate
        title={o.label}
        eyebrow={o.note}
        right={
          <Pill tone={o.safe ? "neutral" : "high"} dot>
            {o.safe ? `Proven safe · z3 ${o.verification.z3_status}` : "Refused by the prover"}
          </Pill>
        }
      >
        {!o.safe && o.verification.violations.length > 0 && (
          <div className="mb-4 rounded-md bg-high/[0.08] p-3 text-sm text-high-ink">
            {o.verification.violations.join(" · ")}
          </div>
        )}
        {o.arms.length === 0 ? (
          <Empty>No drug is given on this path.</Empty>
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {o.arms.map((a) => (
              <div key={a.drug} className="min-w-0">
                <div className="flex items-baseline gap-2">
                  <span className="text-base text-ink">{a.label}</span>
                  <span className="text-xs text-dim">{title(a.drug_class)}</span>
                </div>
                <div className="text-xs text-muted">{a.regimen}</div>
                <ConcChart arm={a} />
                <div className="mt-1 grid grid-cols-2 gap-x-4">
                  {a.kind === "volume" ? (
                    <>
                      <Kv k="Infused" v={`${fmt(a.metrics.infused_ml, 0)} mL`} />
                      <Kv k="Peak plasma expansion" v={`${fmt(a.metrics.peak_plasma_expansion_ml, 0)} mL`} />
                      <Kv k="Still intravascular at 1 h" v={pct(a.metrics.plasma_retention_1h_post, 0)} />
                      <Kv k="Moved to tissue" v={`${fmt(a.metrics.interstitial_ml_end, 0)} mL`} />
                    </>
                  ) : (
                    <>
                      <Kv k={a.metrics.continuous ? "Steady state" : "Cmax"} v={`${fmt(a.metrics.cmax, 2)} ${a.unit}`} />
                      <Kv k="Tmax" v={`${fmt(a.metrics.tmax_h, 1)} h`} />
                      {a.metrics.pct_ft_above_mic != null && <Kv k="fT > MIC" v={`${fmt(a.metrics.pct_ft_above_mic, 0)}%`} />}
                      {a.metrics.clearance_l_h != null && <Kv k="Clearance" v={`${fmt(a.metrics.clearance_l_h, 1)} L/h`} />}
                    </>
                  )}
                </div>
                <div className="mt-1 text-2xs leading-snug text-faint">{Object.values(a.provenance).join(" · ")}</div>
              </div>
            ))}
          </div>
        )}
        {(o.upsides.length > 0 || o.downsides.length > 0) && (
          <div className="mt-3 grid grid-cols-1 gap-4 border-t border-line-soft pt-4 md:grid-cols-2">
            <PointList title="For" points={o.upsides} />
            <PointList title="Against" points={o.downsides} warn />
          </div>
        )}
      </Plate>

      <Plate
        title="Adverse events"
        eyebrow={`P(any serious) ${pct(o.p_any_serious, 0)} · ${o.n_serious} serious modelled`}
      >
        {events.length === 0 ? (
          <Empty>No modelled adverse events on this path.</Empty>
        ) : (
          <ul className="space-y-3">
            {events.map((e) => {
              const tone = e.probability >= 0.3 ? "high" : e.probability >= 0.1 ? "med" : "neutral";
              return (
                <li key={e.key}>
                  <div className="flex items-baseline gap-2">
                    <span className="text-sm text-ink">{e.label}</span>
                    <span className="text-xs text-dim">{e.drug}</span>
                    <span className="num ml-auto text-xs text-muted">
                      {pct(e.probability, 0)}{" "}
                      <span className="text-dim">
                        ({pct(e.ci_low, 0)}–{pct(e.ci_high, 0)})
                      </span>
                    </span>
                  </div>
                  <Meter value={e.probability} tone={tone} marks={[e.baseline]} className="mt-1" />
                  <div className="mt-1 text-xs leading-snug text-dim">
                    baseline {pct(e.baseline, 0)}
                    {e.onset_p50_h != null && ` · median onset ${Math.round(e.onset_p50_h)} h`} · {e.monitoring}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
        <p className="mt-4 text-xs leading-relaxed text-dim">{o.adverse.honesty}</p>
      </Plate>
    </div>
  );
}

function PointList({ title: t, points, warn }: { title: string; points: Point[]; warn?: boolean }) {
  return (
    <div>
      <div className="mb-1.5 text-xs text-dim">{t}</div>
      {points.length === 0 ? (
        <div className="text-xs text-faint">Nothing crossed a threshold (which is not the same as safe).</div>
      ) : (
        <ul className="space-y-1.5">
          {points.map((p) => (
            <li key={p.text} className="flex gap-2 text-sm leading-snug">
              <span className={cx("mt-[7px] h-1 w-1 shrink-0 rounded-full", warn ? "bg-dys" : "bg-faint")} />
              <span className="text-muted" title={p.evidence}>
                {p.text}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

function fmt(v: number | undefined, d: number): string {
  return v == null ? "—" : v.toFixed(d);
}

/** Plasma concentration over the projection horizon, with the MIC target when there is one. */
function ConcChart({ arm }: { arm: Arm }) {
  const W = 320;
  const H = 96;
  const t = arm.times_h;
  const y = arm.plasma;
  const mic = arm.metrics.mic_mg_l;
  const top = Math.max(...y, mic ?? 0) * 1.08 || 1;
  const X = (v: number) => (v / (t[t.length - 1] || 1)) * W;
  const Y = (v: number) => H - (v / top) * (H - 6);
  const d = y.map((v, i) => `${i ? "L" : "M"}${X(t[i]).toFixed(1)},${Y(v).toFixed(1)}`).join("");
  return (
    <svg viewBox={`0 0 ${W} ${H}`} className="mt-2 h-auto w-full" aria-label={`${arm.label} plasma concentration`}>
      <path d={`${d}L${W},${H}L0,${H}Z`} fill="rgb(61 142 245 / .12)" />
      <path d={d} fill="none" stroke="#6eaaff" strokeWidth={1.5} />
      {mic != null && (
        <>
          <line x1={0} x2={W} y1={Y(mic)} y2={Y(mic)} stroke={`rgb(${TONE_RGB.med} / .7)`} strokeDasharray="4 3" />
          {/* above the line, but never down on the time axis labels */}
          <text x={W - 2} y={Math.min(Y(mic) - 4, H - 15)} textAnchor="end" fontSize="9.5" fill="#f4c55a">
            MIC {mic} {arm.unit}
          </text>
        </>
      )}
      <text x={2} y={H - 3} fontSize="9.5" fill="#868d97">
        0 h
      </text>
      <text x={W - 2} y={H - 3} textAnchor="end" fontSize="9.5" fill="#868d97">
        {t[t.length - 1]} h
      </text>
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* Interaction screen: the drug list compiled to Z3 over the KB         */
/* ------------------------------------------------------------------ */
type Screen = {
  safe: boolean;
  drugs: string[];
  n_interactions: number;
  severity_counts: Record<string, number>;
  violations: string[];
  proof_tree: { interaction: string; severity: string; mechanism: string; source: string; constraint: string; z3_status: string; satisfied: boolean }[];
  warnings: { a: string; b: string; severity: string; mechanism: string; source: string }[];
  kb: { ddinter: { drugs: number; edges: number } };
};

function InteractionScreen({ seed, renal }: { seed: string[]; renal: boolean }) {
  const [drugs, setDrugs] = useState<string[]>(() => Array.from(new Set(seed.filter(Boolean))));
  const [add, setAdd] = useState("");
  const [res, setRes] = useState<Screen | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const run = useMemo(
    () => async (list: string[]) => {
      if (!list.length) {
        setRes(null);
        return;
      }
      setBusy(true);
      setErr(null);
      try {
        setRes(await api<Screen>("/aegis/pharmacopoeia/screen", { body: { drugs: list, context: { renal_failure: renal } } }));
      } catch (e) {
        setErr(e instanceof Error ? e.message : String(e));
      } finally {
        setBusy(false);
      }
    },
    [renal],
  );
  useEffect(() => {
    run(drugs);
  }, [drugs, run]);

  return (
    <Plate
      title="Interaction screen"
      eyebrow={
        res
          ? `${res.kb.ddinter.drugs.toLocaleString()} drugs · ${res.kb.ddinter.edges.toLocaleString()} documented interactions, each a Z3 constraint`
          : "Every hazard in the knowledge base becomes a Z3 constraint over the order"
      }
      right={
        res && (
          <Pill tone={res.safe ? "neutral" : "high"} dot>
            {res.safe ? "No hard constraint violated" : `${res.violations.length} hard constraint${res.violations.length > 1 ? "s" : ""} violated`}
          </Pill>
        )
      }
    >
      <div className="flex flex-wrap items-center gap-2">
        {drugs.map((d) => (
          <span key={d} className="inline-flex items-center gap-1.5 rounded-full bg-white/[0.06] py-1 pl-3 pr-1.5 text-sm shadow-[inset_0_0_0_1px_rgb(255_255_255/.07)]">
            {d}
            <button onClick={() => setDrugs(drugs.filter((x) => x !== d))} className="grid h-5 w-5 place-items-center rounded-full text-dim transition-colors hover:bg-white/[0.1] hover:text-ink" aria-label={`Remove ${d}`}>
              <Close size={12} />
            </button>
          </span>
        ))}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            const v = add.trim().toLowerCase();
            if (v && !drugs.includes(v)) setDrugs([...drugs, v]);
            setAdd("");
          }}
          className="flex gap-2"
        >
          <input
            value={add}
            onChange={(e) => setAdd(e.target.value)}
            placeholder="Add a drug (e.g. vancomycin)"
            className="w-56 field px-3 py-1.5 text-sm text-ink placeholder:text-faint"
          />
          <Button type="submit" disabled={!add.trim()}>
            Add
          </Button>
        </form>
        {renal && <Pill tone="med">Screening with renal impairment</Pill>}
        {busy && <span className="text-xs text-dim">proving…</span>}
      </div>
      {err && <div className="mt-3 text-sm text-high-ink">{err}</div>}
      {res && (
        <div className="mt-4 grid grid-cols-1 gap-4 lg:grid-cols-2">
          <div>
            <div className="mb-2 text-xs text-dim">Proof tree · hard constraints</div>
            {res.proof_tree.length === 0 ? (
              <div className="text-sm text-muted">No hard interaction constraints apply to this list.</div>
            ) : (
              <ul className="space-y-2.5">
                {res.proof_tree.map((p) => (
                  <li key={p.interaction} className="well p-3.5">
                    <div className="flex items-baseline gap-2">
                      <span className="text-sm text-ink">{p.interaction}</span>
                      <span className={cx("text-xs", p.satisfied ? "text-dim" : "text-high-ink")}>{p.satisfied ? "satisfied" : "violated"}</span>
                      <span className="ml-auto text-xs text-dim">{p.severity}</span>
                    </div>
                    <div className="text-xs text-muted">{p.mechanism}</div>
                    <div className="mt-1 font-mono text-2xs text-faint">
                      {p.constraint} · {p.source}
                    </div>
                  </li>
                ))}
              </ul>
            )}
          </div>
          <div>
            <div className="mb-2 text-xs text-dim">Documented interactions · warnings</div>
            {res.warnings.length === 0 ? (
              <div className="text-sm text-muted">None documented between these drugs.</div>
            ) : (
              <ul className="divide-y divide-line-soft">
                {res.warnings.map((w) => (
                  <li key={`${w.a}-${w.b}`} className="flex items-baseline gap-2 py-2 text-sm">
                    <span className="text-ink">{w.b ? `${w.a} + ${w.b}` : w.a}</span>
                    <span className="truncate text-dim">{w.mechanism}</span>
                    <span className={cx("ml-auto shrink-0 text-xs", w.severity === "Major" ? "text-dys-ink" : "text-muted")}>{w.severity}</span>
                  </li>
                ))}
              </ul>
            )}
          </div>
        </div>
      )}
    </Plate>
  );
}
