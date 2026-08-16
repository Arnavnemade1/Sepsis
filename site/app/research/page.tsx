import type { Metadata } from "next";

import {
  meta,
  headline,
  generalization,
  calibration,
  operatingPoint,
  alarmPolicy,
  selfImprove,
  qhdc,
  phase7,
  directions,
  dirLabel,
} from "@/lib/results";
import { f2, f3, f4, ci, pct, count, signed, pval } from "@/lib/format";
import {
  Section,
  Panel,
  MetricTile,
  SourceNote,
  StatusChip,
  DataTable,
  Row,
} from "@/components/data/primitives";
import { Reveal, Stagger, Item } from "@/components/motion/reveal";
import { Counter } from "@/components/motion/counter";
import { SectionNav, type NavSection } from "@/components/research/section-nav";
import { ForestPlot } from "@/components/charts/forest-plot";
import { RocChart } from "@/components/charts/roc-chart";
import { PrChart } from "@/components/charts/pr-chart";
import { ReliabilityChart } from "@/components/charts/reliability-chart";
import { DecisionCurve } from "@/components/charts/decision-curve";
import { AlarmBurden } from "@/components/charts/alarm-burden";

export const metadata: Metadata = {
  title: "Research",
  description:
    "Cross-hospital external validation of AegisOS sepsis detection on PhysioNet/CinC 2019 — with the ceilings, the refused promotions, and the negative results included.",
};

const sections: NavSection[] = [
  { id: "protocol", label: "Protocol & data" },
  { id: "headline", label: "Headline results" },
  { id: "discrimination", label: "Discrimination" },
  { id: "calibration", label: "Calibration" },
  { id: "benefit", label: "Clinical benefit" },
  { id: "alarms", label: "Alarm burden" },
  { id: "gap", label: "The generalization gap" },
  { id: "operating-point", label: "The operating point" },
  { id: "negative", label: "What did not work" },
  { id: "integrity", label: "Integrity audit" },
  { id: "limits", label: "What this does not show" },
  { id: "reproduce", label: "Reproduce it" },
];

const AtoB = directions.AtoB;
const BtoA = directions.BtoA;

export default function ResearchPage() {
  return (
    <div className="mx-auto max-w-6xl px-5 pt-28 pb-24 sm:px-8">
      {/* ── Masthead ────────────────────────────────────────────────────── */}
      <header className="max-w-3xl">
        <p className="eyebrow flex items-center gap-2">
          <span className="inline-block h-1.5 w-1.5 rounded-full bg-low" />
          Real-data validation · reproducible
        </p>
        <h1 className="mt-5 font-display text-[clamp(2.25rem,5.5vw,3.5rem)] font-semibold leading-[1.02] tracking-[-0.035em] text-ink text-balance">
          Does it actually work?
        </h1>
        <p className="mt-5 text-[17px] leading-relaxed text-muted text-pretty">
          Every number below was earned on real, labelled ICU data with leakage-free
          validation, and each one is reported with the dataset and protocol that produced
          it. The ceilings, the refused promotions and the outright failures are here too —
          they are the most informative part of the record.
        </p>
        <p className="mt-4 text-[15px] leading-relaxed text-dim text-pretty">
          The honest deployment number is the{" "}
          <strong className="font-medium text-ink">cross-hospital</strong> figure, not the
          flattering within-hospital one. That is the number this page leads with.
        </p>
      </header>

      <div className="mt-16 grid gap-12 lg:grid-cols-[190px_minmax(0,1fr)] lg:gap-14">
        <aside className="hidden lg:block">
          <SectionNav sections={sections} />
        </aside>

        <div className="min-w-0 space-y-20">
          {/* ── 1 · Protocol ──────────────────────────────────────────── */}
          <Section
            id="protocol"
            eyebrow="01 · Protocol & data"
            title="Train on one hospital. Test on the other."
            lede={
              <>
                The easy way to report a medical model is a random split of one hospital&apos;s
                patients — which quietly lets the model learn that hospital&apos;s charting
                habits. This evaluation refuses that: it trains on one site and tests on the
                other, in both directions, so the reported number is what a new hospital
                would actually see on day one.
              </>
            }
          >
            <Stagger className="mt-7 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Item>
                <MetricTile
                  value={count(meta.setA.patients + meta.setB.patients)}
                  label="Patients"
                  detail={`${count(meta.setA.patients)} BIDMC + ${count(meta.setB.patients)} Emory`}
                />
              </Item>
              <Item>
                <MetricTile
                  value={count(meta.setA.rows + meta.setB.rows)}
                  label="Patient-hours"
                  detail="one row per hour of stay"
                />
              </Item>
              <Item>
                <MetricTile
                  size="sm"
                  value={`${pct(meta.setA.prevalence, 2)} / ${pct(meta.setB.prevalence, 2)}`}
                  label="Hourly prevalence"
                  detail="set A / set B — sepsis is rare, which is why AUPRC matters"
                />
              </Item>
              <Item>
                <MetricTile
                  value={`${meta.bootstrap}×`}
                  label="Bootstrap resamples"
                  detail="patient-level, for every confidence interval"
                />
              </Item>
            </Stagger>

            <Reveal className="mt-4">
              <Panel className="p-5">
                <dl className="grid gap-x-8 gap-y-3 text-[13px] sm:grid-cols-2">
                  <div>
                    <dt className="font-mono text-[10.5px] uppercase tracking-[0.08em] text-dim">
                      Dataset
                    </dt>
                    <dd className="mt-1 text-muted">
                      PhysioNet/CinC 2019. Set A = Beth Israel Deaconess (BIDMC), set B =
                      Emory. Labels start six hours before Sepsis-3 onset.
                    </dd>
                  </div>
                  <div>
                    <dt className="font-mono text-[10.5px] uppercase tracking-[0.08em] text-dim">
                      Feature set
                    </dt>
                    <dd className="mt-1 text-muted">
                      <code className="text-ink">{meta.featureSet}</code> — vitals a wearable
                      could plausibly capture. No labs, no chart-derived shortcuts.
                    </dd>
                  </div>
                  <div>
                    <dt className="font-mono text-[10.5px] uppercase tracking-[0.08em] text-dim">
                      Protocol
                    </dt>
                    <dd className="mt-1 text-muted">{meta.protocol}</dd>
                  </div>
                  <div>
                    <dt className="font-mono text-[10.5px] uppercase tracking-[0.08em] text-dim">
                      Generated
                    </dt>
                    <dd className="num mt-1 text-muted">{meta.generated}</dd>
                  </div>
                </dl>
              </Panel>
            </Reveal>
            <SourceNote>
              artifacts/rigorous_eval.json · reproduce with{" "}
              <code>python -m backend.ml.scripts.rigorous_eval</code>
            </SourceNote>
          </Section>

          {/* ── 2 · Headline ──────────────────────────────────────────── */}
          <Section
            id="headline"
            eyebrow="02 · Headline results"
            title="Cross-hospital discrimination"
            lede="Four point estimates in a table look like four facts. Four confidence intervals show how much of the difference between them is real — so this leads with the intervals."
          >
            <Reveal className="mt-8 rounded-lg border border-line bg-s1 p-5 sm:p-6">
              <ForestPlot
                rows={[
                  {
                    label: "Production (stacked)",
                    sub: dirLabel(AtoB),
                    value: AtoB.models.stacked.auroc,
                    ci: AtoB.models.stacked.aurocCi,
                    emphasis: true,
                  },
                  {
                    label: "Baseline (GBM)",
                    sub: dirLabel(AtoB),
                    value: AtoB.models.gbm.auroc,
                    ci: AtoB.models.gbm.aurocCi,
                  },
                  {
                    label: "Production (stacked)",
                    sub: dirLabel(BtoA),
                    value: BtoA.models.stacked.auroc,
                    ci: BtoA.models.stacked.aurocCi,
                    emphasis: true,
                  },
                  {
                    label: "Baseline (GBM)",
                    sub: dirLabel(BtoA),
                    value: BtoA.models.gbm.auroc,
                    ci: BtoA.models.gbm.aurocCi,
                  },
                ]}
              />
              <p className="mt-2 font-mono text-[11px] text-faint">
                AUROC with 95% bootstrap CI · dashed line is chance
              </p>
            </Reveal>

            <Stagger className="mt-4 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
              <Item>
                <MetricTile
                  value={<Counter to={headline.production.auroc} decimals={3} />}
                  label="Mean cross-hospital AUROC"
                  detail={`production model · baseline ${f3(headline.baseline.auroc)} (${signed(headline.delta)})`}
                  tone="accent"
                />
              </Item>
              <Item>
                <MetricTile
                  value={<Counter to={headline.production.ece} decimals={4} />}
                  label="Calibration error"
                  detail="the probability means what it says, off-site"
                  tone="accent"
                />
              </Item>
              <Item>
                <MetricTile
                  value={<Counter to={headline.production.medianLeadHours} decimals={1} suffix=" h" />}
                  label="Median lead time"
                  detail="before Sepsis-3 onset, among detected patients"
                />
              </Item>
              <Item>
                <MetricTile
                  value={<Counter to={headline.production.spread} decimals={4} />}
                  label="Direction spread"
                  detail="difference between the two directions — it transfers symmetrically"
                />
              </Item>
            </Stagger>

            <Reveal className="mt-4">
              <DataTable
                caption="Full cross-hospital results by direction and model"
                head={["Train → test", "Model", "AUROC", "95% CI", "AUPRC", "Brier", "ECE", "Utility"]}
              >
                {[
                  { d: AtoB, m: AtoB.models.stacked, n: "stacked", e: true },
                  { d: AtoB, m: AtoB.models.gbm, n: "gbm", e: false },
                  { d: BtoA, m: BtoA.models.stacked, n: "stacked", e: true },
                  { d: BtoA, m: BtoA.models.gbm, n: "gbm", e: false },
                ].map(({ d, m, n, e }, i) => (
                  <Row
                    key={i}
                    emphasis={e}
                    cells={[
                      dirLabel(d),
                      <span key="m" className="font-mono text-[12px] text-muted">
                        {n}
                        {e && <span className="ml-1.5 text-accent-ink">●</span>}
                      </span>,
                      f3(m.auroc),
                      <span key="ci" className="text-dim">
                        {ci(m.aurocCi)}
                      </span>,
                      f3(m.auprc),
                      f3(m.brier),
                      f4(m.ece),
                      f3(m.utility),
                    ]}
                  />
                ))}
              </DataTable>
              <p className="mt-2.5 flex items-center gap-2 font-mono text-[11px] text-faint">
                <span className="text-accent-ink">●</span> production model
              </p>
            </Reveal>
          </Section>

          {/* ── 3 · Discrimination ────────────────────────────────────── */}
          <Section
            id="discrimination"
            eyebrow="03 · Discrimination"
            title="ROC and precision-recall"
            lede="ROC is the number everyone quotes. Precision-recall is the number that survives contact with a 1.4% prevalence problem — so both are here, side by side, for both directions."
          >
            <div className="mt-8 grid gap-4 lg:grid-cols-2">
              {[
                { d: AtoB, key: "AtoB" },
                { d: BtoA, key: "BtoA" },
              ].map(({ d, key }) => (
                <Reveal key={key}>
                  <Panel className="p-5">
                    <h3 className="font-mono text-[11px] uppercase tracking-[0.08em] text-dim">
                      {dirLabel(d)}
                    </h3>
                    <div className="mt-4 space-y-5">
                      <RocChart
                        stacked={d.models.stacked}
                        gbm={d.models.gbm}
                        label={dirLabel(d)}
                      />
                      <PrChart
                        stacked={d.models.stacked}
                        gbm={d.models.gbm}
                        label={dirLabel(d)}
                      />
                    </div>
                  </Panel>
                </Reveal>
              ))}
            </div>
            <SourceNote>
              Accent curve is the production stacked ensemble; grey is the single-GBM
              baseline. Both drawn from the same held-out hospital.
            </SourceNote>

            <Reveal className="mt-6">
              <DataTable
                caption="Sensitivity at fixed specificity"
                head={["Train → test", "sens @ spec ≥ 0.85", "@ ≥ 0.90", "@ ≥ 0.95"]}
              >
                {[AtoB, BtoA].map((d, i) => (
                  <Row
                    key={i}
                    cells={[
                      dirLabel(d),
                      pct(d.models.stacked.sensAtSpec["0.85"], 1),
                      pct(d.models.stacked.sensAtSpec["0.90"], 1),
                      pct(d.models.stacked.sensAtSpec["0.95"], 1),
                    ]}
                  />
                ))}
              </DataTable>
              <SourceNote>
                Read this as the honest ceiling: hold false alarms down hard enough to be
                tolerable and roughly a quarter to a half of septic patients are still
                missed.
              </SourceNote>
            </Reveal>
          </Section>

          {/* ── 4 · Calibration ───────────────────────────────────────── */}
          <Section
            id="calibration"
            eyebrow="04 · Calibration"
            title="A probability that means what it says"
            lede="Discrimination only ranks patients. Calibration is what lets a clinician read 4% as four percent — and it is the property most likely to break when a model changes hospital."
          >
            <div className="mt-8 grid gap-4 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
              <Reveal>
                <Panel className="p-5">
                  <h3 className="font-mono text-[11px] uppercase tracking-[0.08em] text-dim">
                    {dirLabel(AtoB)}
                  </h3>
                  <div className="mt-4">
                    <ReliabilityChart model={AtoB.models.stacked} label={dirLabel(AtoB)} />
                  </div>
                </Panel>
              </Reveal>
              <Reveal delay={0.06}>
                <Panel className="flex h-full flex-col justify-center p-6">
                  <div className="flex items-baseline gap-3">
                    <span className="num font-mono text-[32px] leading-none text-dim line-through decoration-line-bright">
                      {f3(calibration.uncalibratedEce)}
                    </span>
                    <span className="text-dim">→</span>
                    <span className="num font-mono text-[32px] leading-none text-accent-ink">
                      <Counter to={calibration.calibratedEce} decimals={4} />
                    </span>
                  </div>
                  <p className="mt-4 text-[14px] leading-relaxed text-muted text-pretty">
                    Post-hoc calibration cut expected calibration error by roughly{" "}
                    <strong className="font-medium text-ink">
                      {calibration.foldReduction}×
                    </strong>{" "}
                    — and it holds across hospitals.
                  </p>
                  <SourceNote>
                    Left: {calibration.uncalibratedNote}. Right: {calibration.calibratedNote}.
                    Two different protocols, kept apart on purpose — a single
                    &ldquo;before/after&rdquo; number that mixed them would be flattering and
                    wrong.
                  </SourceNote>
                </Panel>
              </Reveal>
            </div>
            <Reveal className="mt-4">
              <Panel className="p-5">
                <p className="text-[13.5px] leading-relaxed text-dim text-pretty">
                  <strong className="font-medium text-muted">A limitation of this plot:</strong>{" "}
                  the cross-hospital run resolves only a couple of reliability bins, because
                  predicted probabilities on a ~1.4% prevalence problem cluster hard at the
                  low end. Few bins is a genuine weakness of the diagram — not evidence of a
                  suspiciously perfect fit.
                </p>
              </Panel>
            </Reveal>
          </Section>

          {/* ── 5 · Decision curve ────────────────────────────────────── */}
          <Section
            id="benefit"
            eyebrow="05 · Clinical benefit"
            title="Is it better than treating everyone?"
            lede="Decision-curve analysis compares the model against the two strategies that need no model at all. It only earns its place in the band of thresholds where it sits above both."
          >
            <div className="mt-8 grid gap-4 lg:grid-cols-2">
              {[AtoB, BtoA].map((d, i) => (
                <Reveal key={i} delay={i * 0.06}>
                  <Panel className="p-5">
                    <h3 className="font-mono text-[11px] uppercase tracking-[0.08em] text-dim">
                      {dirLabel(d)}
                    </h3>
                    <div className="mt-4">
                      <DecisionCurve model={d.models.stacked} label={dirLabel(d)} />
                    </div>
                    <p className="mt-2 font-mono text-[10.5px] text-faint">
                      accent = model · dashed = treat all / treat none
                    </p>
                  </Panel>
                </Reveal>
              ))}
            </div>
          </Section>

          {/* ── 6 · Alarm burden ──────────────────────────────────────── */}
          <Section
            id="alarms"
            eyebrow="06 · Alarm burden"
            title="The number that decides whether this is a product"
            lede="A detector that pages a nurse for a third of healthy patients is not deployable however good its AUROC is. This is the trade, plotted honestly."
          >
            <Stagger className="mt-8 grid gap-3 sm:grid-cols-3">
              <Item>
                <MetricTile
                  value={pct(headline.production.detectionRate, 1)}
                  label="Detection rate"
                  detail="septic patients alarmed before onset"
                  tone="low"
                />
              </Item>
              <Item>
                <MetricTile
                  value={pct(headline.production.falseAlarmRate, 1)}
                  label="False-alarm rate"
                  detail="non-septic patients who trigger at least one alarm"
                  tone="high"
                />
              </Item>
              <Item>
                <MetricTile
                  value={f2(headline.production.utility)}
                  label="Normalized utility"
                  detail="the CinC-2019 challenge metric, which penalises both misses and false alarms"
                />
              </Item>
            </Stagger>

            <div className="mt-4 grid gap-4 lg:grid-cols-2">
              {[AtoB, BtoA].map((d, i) => (
                <Reveal key={i} delay={i * 0.06}>
                  <Panel className="p-5">
                    <h3 className="font-mono text-[11px] uppercase tracking-[0.08em] text-dim">
                      {dirLabel(d)}
                    </h3>
                    <div className="mt-4">
                      <AlarmBurden
                        model={d.models.stacked}
                        label={dirLabel(d)}
                        shippedThreshold={operatingPoint.high}
                      />
                    </div>
                    <p className="mt-2 font-mono text-[10.5px] text-faint">
                      <span className="text-low-ink">●</span> detection ·{" "}
                      <span className="text-high-ink">●</span> false alarms
                    </p>
                  </Panel>
                </Reveal>
              ))}
            </div>

            <Reveal className="mt-4">
              <Panel className="border-high/25 p-5">
                <div className="flex flex-wrap items-center gap-3">
                  <StatusChip tone="high">Unsolved</StatusChip>
                  <h3 className="text-[14px] font-medium text-ink">
                    A {pct(headline.production.falseAlarmRate)} false-alarm rate is the real
                    bottleneck — and it is not policy-fixable.
                  </h3>
                </div>
                <p className="mt-3 text-[13.5px] leading-relaxed text-muted text-pretty">
                  Four families of alarm policy were searched against it — a plain threshold,
                  K consecutive hours, a leaky exceedance integrator, and a per-patient
                  z-score. None beat a plain threshold. The search refused to promote its own
                  challenger:
                </p>
                <p className="mt-3 border-l-2 border-line-bright pl-4 font-mono text-[12.5px] leading-relaxed text-dim">
                  {alarmPolicy.reason}
                </p>
                <SourceNote>
                  artifacts/alarm_policy.json · {count(alarmPolicy.nBoot)} bootstrap
                  resamples · shipped <code>{alarmPolicy.shipped}</code> vs challenger{" "}
                  <code>{alarmPolicy.challenger}</code>
                </SourceNote>
              </Panel>
            </Reveal>
          </Section>

          {/* ── 7 · Generalization gap ────────────────────────────────── */}
          <Section
            id="gap"
            eyebrow="07 · The generalization gap"
            title="The number most papers do not print"
            lede="Same model, same features, same code. The only thing that changes is which hospital the test patients came from."
          >
            <Reveal className="mt-8">
              <Panel className="p-6 sm:p-8">
                <div className="flex flex-wrap items-end gap-x-10 gap-y-6">
                  <div>
                    <div className="num font-mono text-[38px] leading-none text-dim">
                      {f3(generalization.internalAuroc)}
                    </div>
                    <div className="mt-2 text-[13px] font-medium text-muted">
                      Within hospital
                    </div>
                    <div className="mt-0.5 text-[12px] text-faint">
                      {generalization.internalNote}
                    </div>
                  </div>

                  <div className="pb-6 text-dim">→</div>

                  <div>
                    <div className="num font-mono text-[38px] leading-none text-ink">
                      <Counter to={generalization.externalAuroc} decimals={3} />
                    </div>
                    <div className="mt-2 text-[13px] font-medium text-ink">
                      Across hospitals
                    </div>
                    <div className="mt-0.5 text-[12px] text-faint">
                      {generalization.externalNote}
                    </div>
                  </div>

                  <div className="ml-auto">
                    <div className="num font-mono text-[38px] leading-none text-high-ink">
                      −{f3(generalization.gap)}
                    </div>
                    <div className="mt-2 text-[13px] font-medium text-high-ink">
                      The gap
                    </div>
                    <div className="mt-0.5 text-[12px] text-faint">
                      what a new hospital costs you
                    </div>
                  </div>
                </div>

                <p className="mt-7 max-w-2xl text-[14px] leading-relaxed text-muted text-pretty">
                  A model that reports only the left-hand number is not lying, exactly — but
                  it is describing a deployment that does not exist. Every headline on this
                  site uses the right-hand one.
                </p>
              </Panel>
            </Reveal>

            <Reveal className="mt-4">
              <Panel className="p-5">
                <p className="text-[13.5px] leading-relaxed text-dim text-pretty">
                  <strong className="font-medium text-muted">Two protocols, labelled.</strong>{" "}
                  An earlier evaluation of the single GBM on the full per-hospital data
                  reported a mean cross-hospital AUROC of{" "}
                  <span className="num text-muted">{f3(phase7.meanAuroc)}</span>. The newer
                  bootstrapped protocol above reports{" "}
                  <span className="num text-muted">{f3(headline.baseline.auroc)}</span> for the
                  same model on 6,000 + 6,000 patients. Both are real; they measure slightly
                  different things. They are never averaged together.
                </p>
                <SourceNote>
                  artifacts/external_validation_wearable.json (Phase 7) ·
                  artifacts/rigorous_eval.json (current)
                </SourceNote>
              </Panel>
            </Reveal>
          </Section>

          {/* ── 8 · Operating point ───────────────────────────────────── */}
          <Section
            id="operating-point"
            eyebrow="08 · The operating point"
            title="The threshold was a guess, and the guess was expensive"
            lede="When AUROC is capped by the data, the remaining wins are not in the score — they are in the decision. This is the largest single improvement in the project, and it changed no model at all."
          >
            <Reveal className="mt-8">
              <Panel className="p-6">
                <p className="text-[15px] leading-relaxed text-muted text-pretty">
                  The shipped alarm threshold of{" "}
                  <span className="num font-mono text-ink">0.12</span> was hand-set. Measured
                  against the detector&apos;s own output distribution, it sat at the{" "}
                  <strong className="font-medium text-ink">
                    {operatingPoint.heldOut.shippedPercentile}th percentile
                  </strong>{" "}
                  — only {pct(operatingPoint.heldOut.fracAboveShipped, 2)} of patient-hours
                  ever reached it. The detector was working; almost nothing was being allowed
                  through.
                </p>
              </Panel>
            </Reveal>

            <Reveal className="mt-4">
              <DataTable
                caption="Held-out comparison of operating points"
                head={["Operating point", "Threshold", "Utility", "Detection", "False alarms", "Lead"]}
              >
                {operatingPoint.heldOut.rows.map((r, i) => (
                  <Row
                    key={i}
                    emphasis={r.threshold === operatingPoint.high}
                    cells={[
                      <span key="l">
                        {r.label}
                        {r.threshold === operatingPoint.high && (
                          <span className="ml-2 text-accent-ink">● shipped</span>
                        )}
                      </span>,
                      r.threshold,
                      signed(r.utility, 4),
                      pct(r.detection, 1),
                      pct(r.far, 1),
                      `${r.lead} h`,
                    ]}
                  />
                ))}
              </DataTable>
            </Reveal>

            <Stagger className="mt-4 grid gap-3 sm:grid-cols-3">
              <Item>
                <MetricTile
                  value={f4(operatingPoint.heldOut.auroc)}
                  label="AUROC, all three rows"
                  detail="identical — the model never changed"
                />
              </Item>
              <Item>
                <MetricTile
                  value={signed(
                    operatingPoint.heldOut.rows[1].utility - operatingPoint.heldOut.rows[0].utility,
                    4,
                  )}
                  label="Held-out utility gained"
                  detail="≈ +14% over the hand-set threshold"
                  tone="low"
                />
              </Item>
              <Item>
                <MetricTile
                  value={`+${((operatingPoint.heldOut.rows[1].detection - operatingPoint.heldOut.rows[0].detection) * 100).toFixed(1)} pts`}
                  label="Detection gained"
                  detail="at an unchanged AUROC"
                  tone="low"
                />
              </Item>
            </Stagger>

            <SourceNote>
              artifacts/operating_point.json · derived out-of-fold on{" "}
              {count(operatingPoint.nPatients)} patients, then measured on a disjoint
              held-out set. Honest caveat: the {pct(operatingPoint.maxFalseAlarmRate)}{" "}
              false-alarm cap constrains the <em>derivation</em> (
              {pct(operatingPoint.oof.falseAlarmRate, 1)} out-of-fold), not the deployed rate
              — held-out came in at {pct(operatingPoint.heldOut.rows[1].far, 1)}.
            </SourceNote>
          </Section>

          {/* ── 9 · Negative results ──────────────────────────────────── */}
          <Section
            id="negative"
            eyebrow="09 · What did not work"
            title="Six things that failed, and stayed failed"
            lede="A research record with no negative results is not a record of research. These are kept because each one closed off a direction that looked promising — and because the machinery that refused them is the actual contribution."
          >
            {/* Self-improvement */}
            <Reveal className="mt-8">
              <Panel className="p-6">
                <div className="flex flex-wrap items-center gap-3">
                  <StatusChip tone="high">0 of {selfImprove.nTrials} accepted</StatusChip>
                  <h3 className="text-[15px] font-medium text-ink">
                    A recursive self-improvement search that improved nothing
                  </h3>
                </div>
                <p className="mt-3 text-[14px] leading-relaxed text-muted text-pretty">
                  Sixteen mutated model configurations were searched across four rounds against
                  a locked gate. Not one was promoted. The acceptance rule required both a real
                  effect (Δ &gt; 0.005) <em>and</em> statistical significance under a Bonferroni
                  correction (p &lt; 0.0125).
                </p>

                <div className="mt-5">
                  <DataTable
                    caption="Self-improvement trials, largest apparent gain first"
                    head={["Trial", "AUROC", "Δ vs champion", "p", "Verdict"]}
                  >
                    {selfImprove.trials.slice(0, 4).map((t) => (
                      <Row
                        key={t.id}
                        cells={[
                          <span key="t" className="font-mono text-[12px]">
                            #{t.id}
                          </span>,
                          f4(t.auroc),
                          signed(t.delta, 4),
                          pval(t.p),
                          <StatusChip key="v" tone={t.accepted ? "low" : "nodata"}>
                            {t.accepted ? "accepted" : "rejected"}
                          </StatusChip>,
                        ]}
                      />
                    ))}
                  </DataTable>
                </div>

                <p className="mt-4 text-[14px] leading-relaxed text-muted text-pretty">
                  Trial #{selfImprove.trials[0].id} is the instructive one. It scored{" "}
                  <span className="num text-ink">{signed(selfImprove.trials[0].delta, 4)}</span>{" "}
                  AUROC — a {(selfImprove.trials[0].delta * 100).toFixed(1)}-point gain that a
                  naive loop would have banked and shipped. At p ={" "}
                  <span className="num text-ink">{f3(selfImprove.trials[0].p)}</span> it is not
                  distinguishable from fold noise, so it was thrown away.
                </p>
                <p className="mt-3 border-l-2 border-line-bright pl-4 font-mono text-[12.5px] leading-relaxed text-dim">
                  {selfImprove.verdict}
                </p>
                <SourceNote>artifacts/self_improve_ledger.json — all 16 trials, with per-fold scores</SourceNote>
              </Panel>
            </Reveal>

            {/* Quantum */}
            <Reveal className="mt-4">
              <Panel className="p-6">
                <div className="flex flex-wrap items-center gap-3">
                  <StatusChip tone="high">No advantage</StatusChip>
                  <h3 className="text-[15px] font-medium text-ink">
                    Quantum hyperdimensional computing does not beat trees on tabular data
                  </h3>
                </div>
                <p className="mt-3 text-[14px] leading-relaxed text-muted text-pretty">
                  A quantum-inspired hyperdimensional encoder was implemented and ablated
                  against classical baselines. It lost. The framework is verified and the
                  ablation is published here because a negative result on a fashionable method
                  is worth more than a quiet omission.
                </p>
                <div className="mt-5">
                  <DataTable caption="QHDC ablation" head={["Model", "AUROC", "AUPRC", "Utility"]}>
                    {Object.entries(qhdc.ablation).map(([name, m]) => (
                      <Row
                        key={name}
                        emphasis={name.startsWith("GBM")}
                        cells={[
                          name,
                          <span key="a">
                            {f3(m.auroc.mean)}{" "}
                            <span className="text-faint">± {f3(m.auroc.sd)}</span>
                          </span>,
                          <span key="p">
                            {f3(m.auprc.mean)}{" "}
                            <span className="text-faint">± {f3(m.auprc.sd)}</span>
                          </span>,
                          <span key="u">
                            {f3(m.utility.mean)}{" "}
                            <span className="text-faint">± {f3(m.utility.sd)}</span>
                          </span>,
                        ]}
                      />
                    ))}
                  </DataTable>
                </div>
                <p className="mt-4 text-[13.5px] leading-relaxed text-dim text-pretty">
                  The dimensional sweep is the other half of the story: cost grows as 2
                  <sup>N</sup> while AUROC crawls —{" "}
                  {qhdc.sweep
                    .map((s) => `${s.qubits} qubits ${f3(s.auroc.mean)} (${s.seconds}s)`)
                    .join(" · ")}
                  .
                </p>
                <SourceNote>artifacts/qhdc_ablation.json</SourceNote>
              </Panel>
            </Reveal>

            {/* The rest */}
            <Stagger className="mt-4 grid gap-3 lg:grid-cols-2">
              {[
                {
                  title: "Multi-scale temporal features",
                  chip: "No gain",
                  body: "Expanding from 94 to 185 features moved AUROC by +0.006 — inside the noise — while AUPRC and utility went flat-to-worse at roughly 2× the compute. Tested, measured, not adopted.",
                },
                {
                  title: "AdaptHD-style archetype retraining",
                  chip: "Actively harmful",
                  body: "A published adaptive-retraining scheme measurably hurt this problem: AUROC fell from 0.634 to 0.586. It ships disabled by default.",
                },
                {
                  title: "HDC as a feature extractor",
                  chip: "Did not replicate",
                  body: "Looked like a +79% AUPRC improvement on one 800-patient split. Under proper cross-validation the effect vanished: 0.7235 ± 0.004 against 0.7263 ± 0.011.",
                },
                {
                  title: "Alarm-persistence debounce",
                  chip: "Bad trade",
                  body: "Requiring an alarm to persist trades away detection faster than it saves false alarms, because a six-hour pre-onset label window leaves very little room to wait.",
                },
              ].map((n) => (
                <Item key={n.title}>
                  <Panel className="h-full p-5">
                    <div className="flex flex-wrap items-center gap-2.5">
                      <StatusChip tone="nodata">{n.chip}</StatusChip>
                      <h4 className="text-[14px] font-medium text-ink">{n.title}</h4>
                    </div>
                    <p className="mt-2.5 text-[13.5px] leading-relaxed text-muted text-pretty">
                      {n.body}
                    </p>
                  </Panel>
                </Item>
              ))}
            </Stagger>
            <SourceNote>backend/ml/README.md · docs/PHASE7_REPORT.md</SourceNote>
          </Section>

          {/* ── 10 · Integrity ────────────────────────────────────────── */}
          <Section
            id="integrity"
            eyebrow="10 · Integrity audit"
            title="How do you know any of this is true?"
            lede="An adversarial suite plants a known lie in the evaluation and asserts that the guard catches it. It verifies the claims, not the code — and it once caught a flaw in itself."
          >
            <Stagger className="mt-8 grid gap-3 sm:grid-cols-2">
              {[
                {
                  n: "Leakage tripwire",
                  r: "Labels shuffled → CV AUROC 0.497",
                  d: "If the pipeline leaked, a model trained on noise would still score above chance. It scores exactly at chance.",
                },
                {
                  n: "Out-of-fold necessity",
                  r: "In-sample 0.999 vs OOF 0.560",
                  d: "Over-sharp by 0.440 AUROC. This is why every threshold on this site is derived out-of-fold.",
                },
                {
                  n: "Statistical gate",
                  r: "Planted noise p = 0.500, rejected",
                  d: "A planted +0.02 real gain came back p = 0.0002 and was accepted. The gate discriminates.",
                },
                {
                  n: "Locked-gate disjointness",
                  r: "481 search / 119 gate / 0 shared",
                  d: "The patients used to search for an improvement never appear in the set used to judge it.",
                },
              ].map((g) => (
                <Item key={g.n}>
                  <Panel className="h-full p-5">
                    <div className="flex items-start gap-3">
                      <StatusChip tone="low">Passed</StatusChip>
                      <div>
                        <h4 className="text-[14px] font-medium text-ink">{g.n}</h4>
                        <p className="num mt-1 font-mono text-[12px] text-accent-ink">{g.r}</p>
                        <p className="mt-2 text-[13px] leading-relaxed text-muted text-pretty">
                          {g.d}
                        </p>
                      </div>
                    </div>
                  </Panel>
                </Item>
              ))}
            </Stagger>
            <SourceNote>
              verify_integrity.py — 8 of 8 guards caught their planted pathology · run it with{" "}
              <code>make verify-integrity</code>
            </SourceNote>
          </Section>

          {/* ── 11 · Limits ───────────────────────────────────────────── */}
          <Section
            id="limits"
            eyebrow="11 · What this does not show"
            title="Read this before quoting anything above"
          >
            <Stagger className="mt-7 space-y-3">
              {[
                {
                  t: "The camera path is not validated end-to-end.",
                  b: "Optical pulse extraction is validated against synthetic ground truth and against real PPG-vs-ECG recordings, but camera → sepsis has never been measured against sepsis labels. Closing that needs a real rPPG video dataset.",
                },
                {
                  t: "Heart-rate-variability → sepsis is machinery, not a result.",
                  b: "The pipeline reports AUROC ≈ 0.9 on a physiology-grounded synthetic cohort. That is a statement about the code path, not about patients. The real test needs credentialed MIMIC-IV Waveform access; the adapter is built and waiting.",
                },
                {
                  t: "Detection is data-limited, not model-limited.",
                  b: "Adding laboratory values buys about +0.02 AUROC. Richer temporal features buy nothing. Wearable-only sepsis detection sits near a ceiling around AUROC 0.74–0.80, and no amount of model work has moved it.",
                },
                {
                  t: "Optical measurement is materially harder on darker skin.",
                  b: "Error rises from under 1 bpm on clean signal to roughly 20 bpm in the hardest cases. The system flags this through its quality gate rather than reporting a confident wrong number — but the disparity is real and unsolved.",
                },
                {
                  t: "The live demo cohort cannot support a performance claim.",
                  b: "The bundled demo runs on a handful of real de-identified stays. Twelve stays cannot support an AUROC, and the system refuses to compute one from them.",
                },
              ].map((l) => (
                <Item key={l.t}>
                  <Panel className="p-5">
                    <h4 className="text-[14px] font-medium text-ink">{l.t}</h4>
                    <p className="mt-2 text-[13.5px] leading-relaxed text-muted text-pretty">
                      {l.b}
                    </p>
                  </Panel>
                </Item>
              ))}
            </Stagger>
          </Section>

          {/* ── 12 · Reproduce ────────────────────────────────────────── */}
          <Section
            id="reproduce"
            eyebrow="12 · Reproduce it"
            title="Run the evaluation yourself"
            lede="Every figure on this page is regenerated by these commands. The data is public; the code is in the repository."
          >
            <Reveal className="mt-7">
              <Panel className="scroll-x p-5">
                <pre className="font-mono text-[12.5px] leading-relaxed text-muted">
                  <code>{`# fetch the public dataset (PhysioNet/CinC 2019, CC BY 4.0)
python -m backend.ml.data.cinc2019 --download

# train the production stacked ensemble
python -m backend.ml.scripts.train_stacked --feature-set wearable

# the cross-hospital evaluation behind every number above
python -m backend.ml.scripts.rigorous_eval --feature-set wearable --boot 200

# derive the operating point out-of-fold
make operating-point

# adversarial integrity audit — plants known lies, asserts the guards fire
make verify-integrity`}</code>
                </pre>
              </Panel>
            </Reveal>
            <SourceNote>
              Numbers on this page are synced from artifacts/ by{" "}
              <code>npm run sync:results</code>; a CI check fails if they drift from the
              committed evaluation.
            </SourceNote>
          </Section>
        </div>
      </div>
    </div>
  );
}
