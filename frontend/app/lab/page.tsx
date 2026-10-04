"use client";

import { Page, PageHead } from "@/components/shell";
import { cx, Empty, Kv, Meter, Pill, Plate, Skeleton, Stat } from "@/components/ui";
import { pct } from "@/lib/clinical";
import { usePoll } from "@/lib/hooks";

type Metrics = {
  auroc: number;
  auprc: number;
  ece: number;
  utility: number;
  detect_at_far15: number;
  lead_at_far15: number;
  detection_at_spec90: number;
  false_alarm_at_spec90: number;
};
type Evidence = {
  detector: null | {
    id: string;
    promoted: boolean;
    created: string;
    feature_set: string;
    roster: string[] | null;
    calibration: string | null;
    train_patients: number;
    judged_patients: number;
    shipped: Metrics;
    predecessor: Metrics;
    gates: Record<string, boolean>;
    alarm_budget?: { candidate?: Record<string, number>; incumbent?: Record<string, number> };
  };
  operating_point: null | {
    high: number;
    medium: number;
    max_false_alarm_rate: number;
    detection_rate: number;
    false_alarm_rate: number;
    false_alarm_upper: number;
    median_lead_hours: number;
    n_patients: number;
    detector: string;
    matches_detector: boolean;
  };
  conformal: null | {
    alpha: number;
    nominal: number;
    coverage_septic: number;
    coverage_non_septic: number;
    abstain_rate: number;
    n_patients: number;
    detector: string;
    matches_detector: boolean;
  };
  ward: null | {
    detector: string;
    matches_detector: boolean;
    patients: number;
    rows: number;
    before: { auroc: number; hour_matched: number };
    after: { auroc: number; hour_matched: number };
    personalised: { auroc: number; hour_matched: number };
    delta: { delta: number; lo: number; hi: number; draws: number };
    index_map: { p_medium: number; p_high: number; index_medium: number; index_high: number };
    icu?: {
      detector: string;
      shipped: boolean;
      matches_operating_point: boolean;
      score: { auroc: number; hour_matched: number; ece: number };
      delta: { delta: number; lo: number; hi: number; draws: number };
      alarms: Record<"candidate" | "incumbent", { threshold: number; detection: number; false_alarm: number }>;
      budget15: { delta: number; lo: number; hi: number };
      high: number;
      medium: number;
      trees: number;
      train_patients: number;
    };
  };
  falsification: null | {
    generated: string;
    mutations: number;
    killed: number;
    skipped: number;
    coverage: number;
    survivors: { name: string; suite: string }[];
  };
};
type LabRecord = {
  campaign: string;
  agent: string;
  kind: string;
  title: string;
  status: string;
  accepted: boolean;
  observed_delta: number | null;
  mde?: number | null; // absent on records that never reached the gate
  alpha_spent: number;
  protocol: string;
};
type Lab = {
  proposers: { name: string; prior: string }[];
  roles: Record<string, string>;
  programme: { protocol: string; n_retracted_entries: number; retraction_note: string };
  record: LabRecord[];
  chain: { ok: boolean; n_entries: number; note: string };
  min_effect: number;
  ledger: { budget: { total: number; spent: number; n_tests: number; planned_tests: number } } | null;
  note: string;
};

export default function LabPage() {
  const ev = usePoll<Evidence>("/aegis/lab/evidence");
  const lab = usePoll<Lab>("/aegis/lab");
  const d = ev.data?.detector;

  return (
    <Page>
      <PageHead title="Evidence">
        The sepsis detector that raises pages: what is deployed, how it was validated on patients it never saw, where its alarm is set, and every
        proposed change the promotion gate refused.
      </PageHead>

      <div className="grid grid-cols-1 gap-3">
        <Plate
          title="V7 · vital-sign detector"
          eyebrow={d ? `${d.promoted ? "Promoted" : "Candidate"} ${d.created?.slice(0, 10)} · ${d.feature_set} features` : "From the promotion gate's manifest"}
          right={d && <span className="max-w-[460px] truncate font-mono text-[11px] text-dim">{d.id}</span>}
        >
          {!ev.data ? (
            <Skeleton className="h-36" />
          ) : !d ? (
            <Empty>No detector manifest on this backend.</Empty>
          ) : (
            <>
              <div className="grid grid-cols-2 gap-6 md:grid-cols-3 xl:grid-cols-6">
                <Stat label="AUROC" value={d.shipped.auroc.toFixed(3)} size="lg" sub={`was ${d.predecessor.auroc.toFixed(3)}`} />
                <Stat
                  label="Caught within a 15% alarm budget"
                  value={pct(d.alarm_budget?.candidate?.det15, 0)}
                  size="lg"
                  sub={`was ${pct(d.alarm_budget?.incumbent?.det15, 0)}`}
                />
                <Stat label="Calibration error (ECE)" value={d.shipped.ece.toFixed(3)} size="lg" sub={`was ${d.predecessor.ece.toFixed(3)}`} />
                <Stat label="Clinical utility" value={d.shipped.utility.toFixed(3)} size="lg" sub={`was ${d.predecessor.utility.toFixed(3)}`} />
                <Stat label="Trained on" value={d.train_patients.toLocaleString()} unit="patients" size="lg" />
                <Stat label="Judged on" value={d.judged_patients.toLocaleString()} unit="unseen" size="lg" />
              </div>
              <div className="mt-5 flex flex-wrap items-center gap-2 border-t border-line-soft pt-4 text-[12px] text-muted">
                <span className="text-dim">Members</span>
                {(d.roster ?? []).map((r) => (
                  <Pill key={r}>{r}</Pill>
                ))}
                {d.calibration && <Pill>{d.calibration} calibration</Pill>}
                <span className="ml-auto text-dim">
                  Gates passed:{" "}
                  {Object.entries(d.gates ?? {})
                    .filter(([, v]) => v)
                    .map(([k]) => k.replace(/_/g, " "))
                    .join(" · ")}
                </span>
              </div>
              <p className="mt-3 text-[12px] leading-relaxed text-dim">
                Both models were scored on the same {d.judged_patients.toLocaleString()} held-out patients, which neither was
                fitted on. They come from the same two hospitals as the training set; a model trained at one hospital and
                scored at the other does worse, and that cross-hospital number is the honest deployment figure.
              </p>
            </>
          )}
        </Plate>

        {ev.data?.ward && <WardPlate w={ev.data.ward} />}

        <div className="grid grid-cols-1 gap-3 xl:grid-cols-3">
          <Plate title="Operating point" eyebrow="Where the alarm is set, derived on out-of-fold patients" right={ev.data?.operating_point && <Stamp ok={ev.data.operating_point.matches_detector} />}>
            {!ev.data ? (
              <Skeleton className="h-32" />
            ) : !ev.data.operating_point ? (
              <Empty>No operating point artifact.</Empty>
            ) : (
              (() => {
                const o = ev.data!.operating_point!;
                return (
                  <>
                    <div className="grid grid-cols-2 gap-4">
                      <Stat label="Alarm threshold" value={pct(o.high, 2)} sub={`medium at ${pct(o.medium, 2)}`} />
                      <Stat label="Median lead time" value={o.median_lead_hours.toFixed(0)} unit="h" />
                    </div>
                    <div className="mt-4 space-y-3">
                      <Bar label="Septic patients flagged" v={o.detection_rate} />
                      <Bar label="Non-septic patients flagged" v={o.false_alarm_rate} cap={o.max_false_alarm_rate} note={`upper bound ${pct(o.false_alarm_upper, 1)} · budget ${pct(o.max_false_alarm_rate, 0)}`} />
                    </div>
                    <div className="mt-3 text-[11.5px] text-dim">{o.n_patients.toLocaleString()} out-of-fold patients</div>
                  </>
                );
              })()
            )}
          </Plate>

          <Plate title="Abstention" eyebrow="Conformal sets: when the model should say it doesn't know" right={ev.data?.conformal && <Stamp ok={ev.data.conformal.matches_detector} />}>
            {!ev.data?.conformal ? (
              <Skeleton className="h-32" />
            ) : (
              (() => {
                const c = ev.data!.conformal!;
                return (
                  <>
                    <div className="grid grid-cols-2 gap-4">
                      <Stat label="Requested coverage" value={pct(c.nominal, 0)} sub={`α = ${c.alpha}`} />
                      <Stat label="Hours it abstains" value={pct(c.abstain_rate, 0)} tone={c.abstain_rate > 0.5 ? "med" : "neutral"} />
                    </div>
                    <div className="mt-4 space-y-3">
                      <Bar label="Coverage, septic hours" v={c.coverage_septic} cap={c.nominal} />
                      <Bar label="Coverage, non-septic hours" v={c.coverage_non_septic} cap={c.nominal} />
                    </div>
                    <div className="mt-3 text-[11.5px] leading-relaxed text-dim">
                      Measured on {c.n_patients.toLocaleString()} locked patients. α is a request, not a result, and the
                      guarantee does not transfer across hospitals.
                    </div>
                  </>
                );
              })()
            )}
          </Plate>

          <Plate title="Falsification" eyebrow="Plant a known bug; the verification suite must catch it">
            {!ev.data?.falsification ? (
              <Skeleton className="h-32" />
            ) : (
              (() => {
                const f = ev.data!.falsification!;
                return (
                  <>
                    <div className="grid grid-cols-2 gap-4">
                      <Stat label="Planted defects caught" value={`${f.killed}/${f.mutations}`} size="lg" />
                      <Stat label="Mutation coverage" value={pct(f.coverage, 0)} size="lg" />
                    </div>
                    <p className="mt-4 text-[12px] leading-relaxed text-muted">
                      A check that has never been seen to fail is not evidence. Every suite behind a published claim is
                      mutated, and each surviving mutation is published as a gap.
                    </p>
                    {f.survivors.length > 0 && (
                      <ul className="mt-3 space-y-1 text-[11.5px] text-muted">
                        {f.survivors.map((sv) => (
                          <li key={sv.name}>
                            <span className="text-dys-ink">survived</span> <span className="font-mono">{sv.name}</span> · {sv.suite.split(".").pop()}
                          </li>
                        ))}
                      </ul>
                    )}
                    <div className="mt-2 text-[11.5px] text-dim">run {f.generated}</div>
                  </>
                );
              })()
            )}
          </Plate>
        </div>

        <Plate
          title="The refusal record"
          eyebrow="Agents propose changes; a locked gate on held-out patients is the only thing that can accept one"
          right={
            lab.data && (
              <div className="flex items-center gap-2">
                <Pill tone={lab.data.chain.ok ? "neutral" : "high"} dot>
                  {lab.data.chain.ok ? `${lab.data.chain.n_entries} verdicts hash-verified` : "Ledger chain broken"}
                </Pill>
                {lab.data.ledger && (
                  <Pill>
                    α spent {lab.data.ledger.budget.spent.toFixed(4)} / {lab.data.ledger.budget.total}
                  </Pill>
                )}
              </div>
            )
          }
          bodyClass="p-0"
        >
          {!lab.data ? (
            <div className="p-4">
              <Skeleton className="h-48" />
            </div>
          ) : (
            <>
              <div className="grid grid-cols-1 gap-4 border-b border-line-soft p-4 md:grid-cols-4">
                {lab.data.proposers.map((p) => (
                  <div key={p.name}>
                    <div className="text-[13px] text-ink">{p.name}</div>
                    <div className="text-[12px] text-muted">{p.prior}</div>
                  </div>
                ))}
              </div>
              <div className="overflow-x-auto">
                <table className="w-full min-w-[720px] text-[12.5px]">
                  <thead className="text-left text-[11.5px] text-dim">
                    <tr>
                      <th className="px-4 py-2 font-normal">Proposal</th>
                      <th className="px-4 py-2 font-normal">From</th>
                      <th className="px-4 py-2 font-normal">Observed Δ</th>
                      <th className="px-4 py-2 font-normal">Needed</th>
                      <th className="px-4 py-2 font-normal">Verdict</th>
                    </tr>
                  </thead>
                  <tbody>
                    {[...lab.data.record].reverse().slice(0, 16).map((r, i) => (
                      <tr key={i} className="border-t border-line-soft">
                        <td className="px-4 py-2 font-mono text-[11.5px] text-ink">{r.title}</td>
                        <td className="px-4 py-2 text-muted">{r.agent}</td>
                        <td className={cx("num px-4 py-2", (r.observed_delta ?? 0) > 0 ? "text-ink" : "text-muted")}>
                          {r.observed_delta == null ? "—" : `${r.observed_delta >= 0 ? "+" : ""}${r.observed_delta.toFixed(4)}`}
                        </td>
                        <td className="num px-4 py-2 text-dim">{typeof r.mde === "number" && Number.isFinite(r.mde) ? `+${r.mde.toFixed(4)}` : "—"}</td>
                        <td className="px-4 py-2">
                          <Pill tone={r.accepted ? "low" : "neutral"}>{r.status}</Pill>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="space-y-2 border-t border-line-soft p-4 text-[12px] leading-relaxed text-dim">
                <p>{lab.data.note}</p>
                {lab.data.programme.n_retracted_entries > 0 && <p>{lab.data.programme.retraction_note}</p>}
              </div>
            </>
          )}
        </Plate>

        <Plate title="How a claim gets in" eyebrow="The roles in the gate">
          {!lab.data ? (
            <Skeleton className="h-24" />
          ) : (
            <div className="grid grid-cols-1 gap-x-8 md:grid-cols-2">
              {Object.entries(lab.data.roles).map(([k, v]) => (
                <Kv key={k} k={k} v={<span className="whitespace-normal text-left text-muted">{v}</span>} />
              ))}
            </div>
          )}
        </Plate>
      </div>
    </Page>
  );
}

function Stamp({ ok }: { ok: boolean }) {
  return (
    <Pill tone={ok ? "neutral" : "med"} dot>
      {ok ? "Measured on the deployed detector" : "Measured on an earlier detector"}
    </Pill>
  );
}

function Bar({ label, v, cap, note }: { label: string; v: number; cap?: number; note?: string }) {
  return (
    <div>
      <div className="flex items-baseline justify-between text-[12px]">
        <span className="text-muted">{label}</span>
        <span className="num text-ink">{pct(v, 1)}</span>
      </div>
      <Meter value={v} marks={cap != null ? [cap] : []} className="mt-1" />
      {note && <div className="mt-1 text-[11px] text-dim">{note}</div>}
    </div>
  );
}

/** Which beds the detector scores, and what the ICU rooms were scored by before it did. */
function WardPlate({ w }: { w: NonNullable<Evidence["ward"]> }) {
  // AUROC on its meaningful range: 0.5 is chance
  const pos = (a: number) => `${Math.max(0, Math.min(1, (a - 0.5) / 0.5)) * 100}%`;
  const icu = w.icu?.shipped ? w.icu : null;
  const rows = [
    { k: "QR · bedside rules, the ICU rooms' risk until 2026-10-03", a: w.before, tone: "bg-faint" },
    { k: icu ? "V7 · vital-sign detector, what the live room reads" : "V7 · vital-sign detector, what every bed reads", a: w.after, tone: icu ? "bg-dim" : "bg-ink" },
    ...(icu ? [{ k: "A40 · all-variable detector, what the ICU rooms read", a: icu.score, tone: "bg-ink" }] : []),
  ];
  const high = icu ? icu.high : w.index_map.p_high;
  const medium = icu ? icu.medium : w.index_map.p_medium;
  return (
    <Plate title="What it scores" eyebrow="Every monitored bed: the twelve ICU rooms and the live room" right={<Stamp ok={w.matches_detector} />}>
      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <div>
          <div className="mb-3 flex flex-wrap items-baseline justify-between gap-x-3 text-[11.5px] text-dim">
            <span>
              AUROC on {w.patients.toLocaleString()} patients the detector never trained on, every hour
            </span>
            <span className="num shrink-0 font-mono">bars from 0.5 (chance) to 1.0</span>
          </div>
          <div className="space-y-3.5">
            {rows.map((r) => (
              <div key={r.k}>
                <div className="flex items-baseline justify-between gap-3 text-[12.5px]">
                  <span className="min-w-0 text-muted">{r.k}</span>
                  <span className="num shrink-0 font-mono text-ink">
                    {r.a.auroc.toFixed(3)} <span className="text-dim">· {r.a.hour_matched.toFixed(3)} hour-matched</span>
                  </span>
                </div>
                <div className="relative mt-1.5 h-1.5 overflow-hidden rounded-[2px] bg-s3">
                  <div className={cx("h-full", r.tone)} style={{ width: pos(r.a.auroc) }} />
                </div>
              </div>
            ))}
          </div>
          <p className="mt-4 text-[12px] leading-relaxed text-dim">
            The ICU rooms raise most detector pages, and until 2026-10-03 a hand-weighted qSOFA/SIRS score decided them while the validated
            detector served only the live room (+{w.delta.delta.toFixed(3)} [+{w.delta.lo.toFixed(3)}, +{w.delta.hi.toFixed(3)}] once it did).
            {icu && (
              <>
                {" "}They replay full ICU records, so they are now read by a LightGBM over all 40 variables, trained on the same{" "}
                {icu.train_patients.toLocaleString()} patients and judged on the same ones: +{icu.delta.delta.toFixed(3)} [+{icu.delta.lo.toFixed(3)}, +
                {icu.delta.hi.toFixed(3)}] over the stack, registered before it ran. The gain is in ranking: inside a fixed 15% alarm budget its
                detection is a tie ({icu.budget15.delta >= 0 ? "+" : ""}
                {icu.budget15.delta.toFixed(3)} [{icu.budget15.lo.toFixed(3)}, +{icu.budget15.hi.toFixed(3)}]).
              </>
            )}{" "}
            Own-baseline learning on top of the detector scored {w.personalised.auroc.toFixed(3)}, so it explains a record and no longer moves the alarm.
          </p>
        </div>
        <div className="border-t border-line-soft pt-4 lg:border-l lg:border-t-0 lg:pl-6 lg:pt-0">
          <div className="text-[11.5px] text-dim">{icu ? "The scale the ICU rooms are read on" : "The scale every room is read on"}</div>
          <div className="mt-3 space-y-2.5 text-[12.5px]">
            <div className="flex items-baseline justify-between gap-3">
              <span className="text-muted">Pages at index</span>
              <span className="num font-mono text-ink">
                {w.index_map.index_high.toFixed(2)} <span className="text-dim">= P {pct(high, 2)}</span>
              </span>
            </div>
            <div className="flex items-baseline justify-between gap-3">
              <span className="text-muted">Medium from</span>
              <span className="num font-mono text-ink">
                {w.index_map.index_medium.toFixed(2)} <span className="text-dim">= P {pct(medium, 2)}</span>
              </span>
            </div>
          </div>
          <p className="mt-3 text-[12px] leading-relaxed text-dim">
            The index is the calibrated probability mapped by a strictly monotone curve pinned at the scoring detector&rsquo;s own operating point,
            so the paging line is that detector&rsquo;s alarm and the ranking, and AUROC, are unchanged. Each detector has its own point; the live
            room&rsquo;s is the operating point below. Every room shows the probability beside its index.
          </p>
        </div>
      </div>
    </Plate>
  );
}
