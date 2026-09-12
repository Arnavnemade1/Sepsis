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
  frontier,
  directions,
  dirLabel,
  subgroup,
  scale,
  scaleCeiling,
  conformal,
  transfer,
  negatives,
  falsification,
} from "@/lib/results";
import { f2, f3, f4, ci, ciSigned, pct, count, signed } from "@/lib/format";
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
import { TrialScatter } from "@/components/charts/trial-scatter";
import { ThresholdSweep } from "@/components/charts/threshold-sweep";
import { GapSlope } from "@/components/charts/gap-slope";
import { AblationBars } from "@/components/charts/ablation-bars";
import { GapStoryFrame } from "@/components/explainer/gap-story-frame";
import { OutcomeWaffle } from "@/components/charts/outcome-waffle";
import { IntegrityBars } from "@/components/charts/integrity-bars";

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
  { id: "objective", label: "The wrong objective" },
  { id: "fairness", label: "Who it serves worse" },
  { id: "scale", label: "Does it scale" },
  { id: "invention", label: "What we invented" },
  { id: "abstention", label: "When it declines to answer" },
  { id: "corrections", label: "Three findings that were bugs" },
  { id: "negative", label: "What did not work" },
  { id: "integrity", label: "Integrity audit" },
  { id: "limits", label: "What this does not show" },
  { id: "reproduce", label: "Reproduce it" },
];

const AtoB = directions.AtoB;
const BtoA = directions.BtoA;

/* Frontier lookups hoisted: narrowing a `Record` through optional chaining does not
   carry into the JSX body, so each value is resolved once here and guarded once. */
const fr = frontier;
const frProd = fr ? fr.comparisons[fr.production] : undefined;
const frRedundant = fr ? fr.comparisons["stack-utility"] : undefined;
const frDiverse = fr ? fr.comparisons["xgb-utility"] : undefined;
const frBudgets = frRedundant?.detectionAtBudget;
const frBaseCand = fr ? fr.candidates[fr.baseline] : undefined;
const frProdCand = fr ? fr.candidates[fr.production] : undefined;

const ceiling = scaleCeiling();

/** Editorial copy for each published survivor, keyed by mutation name.
 *  Keyed rather than shared: two survivors are not the same finding, and a shared
 *  paragraph silently attributes one suite's defect to another. Anything without an
 *  entry falls back to the machine-generated description, which is dull but true. */
const SURVIVOR_COPY: Record<string, React.ReactNode> = {
  operating_point_no_margin: (
    <>
      does not defend its own bootstrap margin. Disabling it ships a false-alarm cap
      that holds only on the patients used to choose it — and every check still passes,
      because the check that looks like it defends this calls the margin function
      directly rather than through the code path that uses it.
    </>
  ),
  selfimprove_shipped_rule_drifts: (
    <>
      does not test the code it claims to. The suite defines its own copy of the
      acceptance rule — docstring: <em>&ldquo;the shipped rule&rdquo;</em> — and tests
      that, never once referencing the real improver. So the shipped rule&rsquo;s
      effect-size floor and significance bar can be set to accept anything and all 14
      checks still pass. That rule&rsquo;s verdict, <strong>0 of 16 candidates
      admitted</strong>, is the headline evidence for the refusal architecture.
    </>
  ),
};

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
            <Reveal className="mt-8 rounded-lg border border-line bg-s1/62 p-5 backdrop-blur-xl sm:p-7">
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
                {/* Retracted 2026-08-30: median lead time does not beat a random-score
                    null at matched detection rate (docs/LEAD_TIME_NULL.md). Replaced with
                    calibration error, which is a genuine property of the model. */}
                <MetricTile
                  value={<Counter to={headline.production.ece} decimals={4} />}
                  label="Calibration error"
                  detail="ECE across hospitals \u00b7 lower is better"
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
            <Reveal className="mt-8">
              <OutcomeWaffle />
            </Reveal>
            <SourceNote>
              Both panels are the production model at its shipped operating point, on the
              hospital it was not trained on. They are drawn at the same scale on purpose.
            </SourceNote>

            <div className="mt-10 space-y-4">
              {[
                { d: AtoB, key: "AtoB" },
                { d: BtoA, key: "BtoA" },
              ].map(({ d, key }) => (
                <Reveal key={key}>
                  <Panel className="p-5 sm:p-7">
                    <div className="flex flex-wrap items-baseline justify-between gap-3">
                      <h3 className="font-mono text-[11px] uppercase tracking-[0.1em] text-dim">
                        {dirLabel(d)}
                      </h3>
                      <p className="num font-mono text-[11px] text-faint">
                        AUROC {f3(d.models.stacked.auroc)} · AUPRC {f3(d.models.stacked.auprc)}
                      </p>
                    </div>
                    <div className="mt-6 space-y-10">
                      <RocChart
                        id={key}
                        stacked={d.models.stacked}
                        gbm={d.models.gbm}
                        label={dirLabel(d)}
                      />
                      <PrChart
                        id={key}
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
                      pct(d.models.stacked.sensAtSpec["0.85"].sensitivity, 1),
                      pct(d.models.stacked.sensAtSpec["0.90"].sensitivity, 1),
                      pct(d.models.stacked.sensAtSpec["0.95"].sensitivity, 1),
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
            <div className="mt-8 space-y-4">
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
            <Reveal className="mt-8">
              <Panel className="p-5 sm:p-7">
                <div className="space-y-10">
                  {[AtoB, BtoA].map((d, i) => (
                    <div key={i}>
                      <h3 className="mb-5 font-mono text-[11px] uppercase tracking-[0.1em] text-dim">
                        {dirLabel(d)}
                      </h3>
                      <DecisionCurve model={d.models.stacked} label={dirLabel(d)} />
                    </div>
                  ))}
                </div>
                <p className="mt-4 font-mono text-[10.5px] text-faint">
                  accent = model · dashed = treat all / treat none
                </p>
              </Panel>
            </Reveal>
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

            <Reveal className="mt-4">
              <Panel className="p-5 sm:p-7">
                <div className="space-y-10">
                  {[AtoB, BtoA].map((d, i) => (
                    <div key={i}>
                      <h3 className="mb-5 font-mono text-[11px] uppercase tracking-[0.1em] text-dim">
                        {dirLabel(d)}
                      </h3>
                      <AlarmBurden
                        id={`ab-${i}`}
                        model={d.models.stacked}
                        label={dirLabel(d)}
                        shippedThreshold={operatingPoint.high}
                      />
                    </div>
                  ))}
                </div>
                <p className="mt-4 font-mono text-[10.5px] text-faint">
                  <span className="text-low-ink">●</span> caught ·{" "}
                  <span className="text-high-ink">●</span> false alarms
                </p>
              </Panel>
            </Reveal>

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
                  {alarmPolicy.familiesSearched.length} families of alarm policy have been
                  searched against it — {alarmPolicy.familiesSearched.join(", ")}. No
                  first-alarm family beat a plain threshold. The search refused to promote
                  its own challenger:
                </p>
                <p className="mt-3 border-l-2 border-line-bright pl-4 font-mono text-[12.5px] leading-relaxed text-dim">
                  {alarmPolicy.reason}
                </p>
                {alarmPolicy.burden ? (
                  <p className="mt-3 text-[13.5px] leading-relaxed text-muted text-pretty">
                    The episode layer was refused for a sharper reason. Re-notifying less
                    genuinely does cut interruptions — {alarmPolicy.burden.championBurden} →{" "}
                    {alarmPolicy.burden.challengerBurden} per patient-day, on an interval
                    excluding zero — but it buys them by leaving an unresolved HIGH
                    unannounced for longer: {alarmPolicy.burden.championSilentHigh} →{" "}
                    {alarmPolicy.burden.challengerSilentHigh} hours per episode. The saving
                    was the silence, so the gate refused it.
                  </p>
                ) : null}
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
              <GapStoryFrame />
            </Reveal>

            <Reveal className="mt-4">
              <Panel className="p-6 sm:p-8">
                <GapSlope />
                <p className="mt-6 max-w-2xl text-[14px] leading-relaxed text-muted text-pretty">
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
              <Panel className="p-5 sm:p-6">
                <ThresholdSweep />
              </Panel>
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


          {/* ── 9 · The objective ─────────────────────────────────────── */}
          <Section
            id="objective"
            eyebrow="09 · The wrong objective"
            title="We were training on the wrong target."
            lede={
              <>
                The challenge label flips to 1 six hours before onset. The scoring function
                does not agree with it: alarming between twelve and six hours out is worth up
                to a full point, and the label calls those hours negative. On this cohort that
                is <strong className="font-medium text-ink">3,419 patient-hours holding
                roughly a fifth of all the credit available</strong> — that every model here
                had been trained to call a false positive.
              </>
            }
          >
            <Reveal className="mt-7">
              <Panel className="p-6">
                <p className="text-[14px] leading-relaxed text-muted text-pretty">
                  Re-deriving the target from the scoring function itself — alarm where it
                  pays, weighted by how much is at stake — lifts every individual booster we
                  tested. Then it did nothing at all to the ensemble, which turned out to be
                  the more useful result.
                </p>
              </Panel>
            </Reveal>

            {fr && (
              <>
                <Reveal className="mt-4">
                  <Panel className="p-6">
                    <div className="flex flex-wrap items-center gap-3">
                      <StatusChip tone="nodata">Within-site CV — not cross-hospital</StatusChip>
                    </div>
                    <p className="mt-3 text-[14px] leading-relaxed text-muted text-pretty">
                      Everything in this section is {fr.folds}-fold patient-grouped
                      cross-validation over {count(fr.patients)} patients
                      ({count(fr.rows)} hours), identical folds for every candidate. It is{" "}
                      <strong className="font-medium text-ink">not</strong> the cross-hospital
                      protocol the headline uses, so these numbers are not comparable with the
                      ones above and are systematically more flattering. The cross-hospital
                      confirmation of this change{" "}
                      <strong className="font-medium text-ink">has not been run yet</strong>.
                    </p>
                  </Panel>
                </Reveal>

                <Reveal className="mt-6">
                  <DataTable
                    caption="Ensemble candidates under identical patient-grouped folds"
                    head={["Candidate", "AUROC", "Rank corr. with incumbent", "False alarms", "ΔAUROC vs incumbent"]}
                  >
                    {Object.entries(fr.candidates).map(([name, c]) => {
                      const cmp = fr.comparisons[name];
                      const isProd = name === fr.production;
                      return (
                        <Row
                          key={name}
                          emphasis={isProd}
                          cells={[
                            <>
                              {name}
                              {isProd && (
                                <span className="ml-2 font-mono text-[10.5px] uppercase tracking-[0.08em] text-accent-ink">
                                  shipped
                                </span>
                              )}
                              {name === fr.baseline && (
                                <span className="ml-2 font-mono text-[10.5px] uppercase tracking-[0.08em] text-dim">
                                  previous
                                </span>
                              )}
                            </>,
                            `${f4(c.auroc)} ± ${f4(c.aurocSd)}`,
                            cmp ? f3(cmp.rankCorr) : "—",
                            f3(c.falseAlarm),
                            cmp ? (
                              <>
                                {signed(cmp.deltaAuroc, 4)}{" "}
                                <span className="text-dim">{ciSigned(cmp.ci)}</span>{" "}
                                {/* Three states, not two: an interval that excludes zero
                                    downward means significantly WORSE, and calling that
                                    "n.s." would hide the most interesting row here. */}
                                {cmp.ci[0] > 0 ? (
                                  <span className="text-accent-ink">better</span>
                                ) : cmp.ci[1] < 0 ? (
                                  <span className="text-high-ink">worse</span>
                                ) : (
                                  <span className="text-dim">n.s.</span>
                                )}
                              </>
                            ) : (
                              "—"
                            ),
                          ]}
                        />
                      );
                    })}
                  </DataTable>
                  <SourceNote>
                    artifacts/frontier_summary.json — {fr.bootstrap} paired
                    patient-level bootstrap draws per comparison
                  </SourceNote>
                </Reveal>

                <Reveal className="mt-6">
                  <Panel className="p-6">
                    <h3 className="text-[15px] font-medium text-ink">
                      Why re-aiming the whole ensemble changed nothing
                    </h3>
                    <p className="mt-3 text-[14px] leading-relaxed text-muted text-pretty">
                      Fitting every member to the new target produced a detector that ranks
                      patients almost identically to the old one — the two agree at a rank
                      correlation of{" "}
                      <span className="num text-ink">
                        {f3(frRedundant?.rankCorr)}
                      </span>
                      . The objective was redundant with what stacking already does. A
                      different <em>library</em> was not: the XGBoost member agrees only{" "}
                      <span className="num text-ink">
                        {f3(frDiverse?.rankCorr)}
                      </span>
                      , and on its own it is{" "}
                      <strong className="font-medium text-ink">significantly worse</strong>{" "}
                      than the ensemble it was added to ({signed(frDiverse?.deltaAuroc, 4)}{" "}
                      AUROC, <span className="text-dim">{ciSigned(frDiverse?.ci)}</span>). That is
                      the whole point: stacking is paid for disagreement, not for strength, so
                      the new objective earned its place on the one member whose mistakes the
                      ensemble did not already own — not on the members that were already
                      good.
                    </p>
                    <p className="mt-3 text-[14px] leading-relaxed text-muted text-pretty">
                      The shipped detector is the four previous members left exactly as they
                      were, plus that one. It gains{" "}
                      <span className="num text-ink">
                        {signed(frProd?.deltaAuroc, 4)}
                      </span>{" "}
                      AUROC{" "}
                      <span className="text-dim">{ciSigned(frProd?.ci)}</span>
                      , wins every fold individually, and cuts the false-alarm rate at matched
                      specificity from{" "}
                      <span className="num text-ink">
                        {f3(frBaseCand?.falseAlarm)}
                      </span>{" "}
                      to{" "}
                      <span className="num text-ink">
                        {f3(frProdCand?.falseAlarm)}
                      </span>
                      .
                    </p>
                    <SourceNote>
                      artifacts/detector_manifest.json — both promotion gates, on held-out
                      patients
                    </SourceNote>
                  </Panel>
                </Reveal>

                <Reveal className="mt-4">
                  <Panel className="p-6">
                    <div className="flex flex-wrap items-center gap-3">
                      <StatusChip tone="high">Nearly shipped a mirage</StatusChip>
                      <h3 className="text-[15px] font-medium text-ink">
                        The metric that agreed with us at exactly one threshold
                      </h3>
                    </div>
                    <p className="mt-3 text-[14px] leading-relaxed text-muted text-pretty">
                      The re-aimed ensemble looked like a large win on detection at a fixed
                      alarm budget. Swept across neighbouring budgets, it was significant at
                      one and absent at both others. A real gain does not appear at a 15%
                      alarm budget and vanish at 10% and 20% — so this was recorded as noise
                      and the change was refused on its own merits.
                    </p>
                    {frBudgets && (
                      <div className="mt-5">
                        <DataTable
                          caption="Detection gain at neighbouring alarm budgets"
                          head={["Alarm budget", "Δ detection", "95% CI", "Verdict"]}
                        >
                          {Object.entries(frBudgets).map(([budget, b]) => (
                            <Row
                              key={budget}
                              cells={[
                                `${budget}% of non-septic patients`,
                                signed(b.delta, 4),
                                ciSigned(b.ci),
                                b.significant ? (
                                  <span className="text-accent-ink">significant</span>
                                ) : (
                                  <span className="text-dim">n.s.</span>
                                ),
                              ]}
                            />
                          ))}
                        </DataTable>
                      </div>
                    )}
                    <SourceNote>
                      artifacts/frontier_summary.json — the same comparison at three budgets,
                      published together so the shape is visible
                    </SourceNote>
                  </Panel>
                </Reveal>
              </>
            )}
          </Section>

          {/* ── Fairness ─────────────────────────────────────────────── */}
          <Section
            id="fairness"
            eyebrow="10 · Who it serves worse"
            title="Every number above is a cohort average"
            lede="A cohort average is the shape of number that hides a disparity. Age and sex have been in the feature matrix since the first fit; this is the first pass that asks what the shipped detector does to different people at the one threshold it actually alarms on."
          >
            <Stagger className="mt-8 grid gap-3 sm:grid-cols-3">
              <Item>
                <MetricTile
                  value={pct(subgroup.cells.find((c) => c.name === "cohort")!.falseAlarm)}
                  label="Cohort false-alarm rate"
                  detail={`share of never-septic patients alarmed at least once, at threshold ${f3(subgroup.threshold)}`}
                />
              </Item>
              <Item>
                <MetricTile
                  value={pct(subgroup.cells.find((c) => c.name === subgroup.worstCell)!.falseAlarm)}
                  label={`Worst cell — ${subgroup.worstCell}`}
                  detail="the same threshold, a different group of people"
                  tone="high"
                />
              </Item>
              <Item>
                <MetricTile
                  value={subgroup.familySize}
                  label="Pre-specified tests"
                  detail={`Bonferroni-corrected at α = ${subgroup.alphaAdjusted.toFixed(4)}; only corrected findings are claimed`}
                />
              </Item>
            </Stagger>

            <Panel className="mt-8 overflow-x-auto p-0">
              <table className="w-full min-w-[46rem] border-collapse text-[13px]">
                <thead>
                  <tr className="border-b border-line text-left text-[11px] uppercase tracking-[0.08em] text-faint">
                    <th className="px-4 py-3 font-medium">Subgroup</th>
                    <th className="px-4 py-3 text-right font-medium">Patients</th>
                    <th className="px-4 py-3 text-right font-medium">AUROC</th>
                    <th className="px-4 py-3 text-right font-medium">Calibration</th>
                    <th className="px-4 py-3 text-right font-medium">Detection</th>
                    <th className="px-4 py-3 text-right font-medium">False alarms</th>
                    <th className="px-4 py-3 text-right font-medium" title="Uncorrected. Median lead time does not beat a random-score null at matched detection rate — see docs/LEAD_TIME_NULL.md. Shown only because the point of this table is that subgroup GAPS are flat.">Lead*</th>
                  </tr>
                </thead>
                <tbody className="tabular-nums">
                  {subgroup.cells.map((c) => {
                    const worst = c.name === subgroup.worstCell;
                    return (
                      <tr
                        key={c.name}
                        className={`border-b border-line/60 last:border-0 ${
                          c.kind === "all" ? "bg-s2/40 font-medium" : ""
                        }`}
                      >
                        <td className={`px-4 py-3 ${worst ? "text-high-ink" : "text-ink"}`}>
                          {c.name}
                        </td>
                        <td className="px-4 py-3 text-right text-muted">{count(c.patients)}</td>
                        <td className="px-4 py-3 text-right text-muted">{f3(c.auroc)}</td>
                        <td className="px-4 py-3 text-right text-muted">{c.calibration.toFixed(2)}</td>
                        <td className="px-4 py-3 text-right text-muted">{pct(c.detection)}</td>
                        <td className={`px-4 py-3 text-right ${worst ? "text-high-ink" : "text-muted"}`}>
                          {pct(c.falseAlarm)}
                        </td>
                        <td className="px-4 py-3 text-right text-muted">{c.leadHours} h</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </Panel>
            <SourceNote>
              Calibration is mean predicted risk &divide; observed rate: above 1.00 the model
              over-states risk for that group, which at a shared threshold becomes more alarms.
              Intersectional cells are omitted except the worst one; the full table is in
              <code className="mx-1">artifacts/subgroup_audit.json</code>.
            </SourceNote>

            <div className="mt-10 grid gap-3 lg:grid-cols-2">
              <Panel className="p-5">
                <StatusChip tone="high">Survives correction</StatusChip>
                <h4 className="mt-3 text-[14px] font-medium text-ink">
                  The alarm burden is not evenly shared
                </h4>
                <ul className="mt-3 space-y-2.5">
                  {subgroup.significant.map((g) => (
                    <li key={`${g.metric}-${g.a}`} className="text-[13.5px] leading-relaxed text-muted">
                      <span className="text-ink">
                        {g.a} &minus; {g.b}
                      </span>{" "}
                      <span className="tabular-nums">
                        {g.delta > 0 ? "+" : "−"}
                        {Math.abs(g.delta * 100).toFixed(1)} pts
                      </span>{" "}
                      <span className="text-faint tabular-nums">
                        [{(g.ci[0] * 100).toFixed(1)}, {(g.ci[1] * 100).toFixed(1)}]
                      </span>
                    </li>
                  ))}
                </ul>
                <p className="mt-4 text-[13px] leading-relaxed text-muted text-pretty">
                  Discrimination is statistically flat: every AUROC, detection and lead-time gap
                  has an interval including zero after correction. What is not flat is who absorbs
                  the false alarms.
                </p>
              </Panel>

              <Panel className="p-5">
                <StatusChip tone="med">Exploratory</StatusChip>
                <h4 className="mt-3 text-[14px] font-medium text-ink">
                  {subgroup.worstCell} — a hypothesis, not a finding
                </h4>
                <ul className="mt-3 space-y-2.5">
                  {subgroup.exploratory.map((g) => (
                    <li key={g.metric} className="text-[13.5px] leading-relaxed text-muted">
                      <span className="text-ink">{g.metric.replace(/_/g, " ")}</span>{" "}
                      <span className="tabular-nums">
                        {g.delta > 0 ? "+" : "−"}
                        {Math.abs(g.delta).toFixed(3)}
                      </span>{" "}
                      <span className="text-faint tabular-nums">
                        [{g.ci[0].toFixed(3)}, {g.ci[1].toFixed(3)}]
                      </span>
                    </li>
                  ))}
                </ul>
                <p className="mt-4 text-[13px] leading-relaxed text-muted text-pretty">
                  This cell was chosen by looking at the results, and nothing here corrects for
                  that choice. It is compared against its complement rather than the cohort that
                  contains it, held out of the correction family, and reported as a place to look
                  next — not as an established disparity.
                </p>
              </Panel>
            </div>

            <SourceNote>
              The threshold is global and never re-tuned per subgroup: re-tuning inside each group
              measures a system that does not exist and launders the disparity into the threshold.
              CinC 2019 carries age and sex and nothing else, so race, ethnicity, insurance and
              language &mdash; the axes a policy reviewer asks about first &mdash; are absent. This
              is a floor on what should be reported, not a complete fairness assessment.
            </SourceNote>
          </Section>

          {/* ── Scale ────────────────────────────────────────────────── */}
          <Section
            id="scale"
            eyebrow="11 · Does it scale"
            title="An operating system is a capacity claim"
            lede="The console had always been demonstrated at twelve beds, because the cohort builder defaults to twelve and nothing had asked for more. These numbers come from the real tick loop over real patients &mdash; the same code the live server runs, with no mock anywhere in the path."
          >
            <Stagger className="mt-8 grid gap-3 sm:grid-cols-3">
              <Item>
                <MetricTile
                  value={ceiling ? count(ceiling.beds) : "—"}
                  label="Beds within the real-time budget"
                  detail={`largest MEASURED ward whose p99 tick stays under ${scale.tickBudgetMs} ms — the crossing itself lies between this level and the next`}
                  tone="accent"
                />
              </Item>
              <Item>
                <MetricTile
                  value={`${scale.levels[scale.levels.length - 1].msPerBed.toFixed(2)} ms`}
                  label="Cost per bed per tick"
                  detail={`flat across a ${Math.round(scale.levels[scale.levels.length - 1].beds / scale.levels[0].beds)}× range in ward size, after the quadratic was removed`}
                />
              </Item>
              <Item>
                <MetricTile
                  value={`${scale.levels[scale.levels.length - 1].peakRssMb.toFixed(0)} MB`}
                  label="Peak resident memory"
                  detail="process high-water mark, not per-level cost — the honest ceiling on a small host"
                />
              </Item>
            </Stagger>

            <Panel className="mt-8 overflow-x-auto p-0">
              <table className="w-full min-w-[40rem] border-collapse text-[13px]">
                <thead>
                  <tr className="border-b border-line text-left text-[11px] uppercase tracking-[0.08em] text-faint">
                    <th className="px-4 py-3 font-medium">Beds</th>
                    <th className="px-4 py-3 text-right font-medium">Tick p50</th>
                    <th className="px-4 py-3 text-right font-medium">Run spread</th>
                    <th className="px-4 py-3 text-right font-medium">Tick p99</th>
                    <th className="px-4 py-3 text-right font-medium">ms / bed</th>
                    <th className="px-4 py-3 font-medium">Dominant phase</th>
                  </tr>
                </thead>
                <tbody className="tabular-nums">
                  {scale.levels.map((l) => {
                    const over = l.p99 > scale.tickBudgetMs;
                    return (
                      <tr key={l.beds} className="border-b border-line/60 last:border-0">
                        <td className="px-4 py-3 text-ink">{count(l.beds)}</td>
                        <td className="px-4 py-3 text-right text-muted">{l.p50.toFixed(1)} ms</td>
                        <td className="px-4 py-3 text-right text-faint">
                          {l.spreadPct === null ? "—" : `±${l.spreadPct}%`}
                        </td>
                        <td className={`px-4 py-3 text-right ${over ? "text-high-ink" : "text-muted"}`}>
                          {l.p99.toFixed(1)} ms{over ? " ✗" : ""}
                        </td>
                        <td className="px-4 py-3 text-right text-muted">{l.msPerBed.toFixed(2)}</td>
                        <td className="px-4 py-3 text-muted">
                          {l.topPhase.replace(/_/g, " ")}{" "}
                          <span className="text-faint">{pct(l.topPhaseShare)}</span>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </Panel>
            <SourceNote>
              Median of {scale.levels[0].repeats} rebuilt runs per level, with the run-to-run
              spread beside it &mdash; a single run of this harness moves by tens of percent, which
              is larger than several of the differences between rows. Measured on one core of a{" "}
              {scale.host.machine} host; these are numbers for this machine and this tick rate, not
              a deployment guarantee.
            </SourceNote>

            <div className="mt-10 grid gap-3 lg:grid-cols-2">
              <Panel className="p-5">
                <StatusChip tone="low">Removed</StatusChip>
                <h4 className="mt-3 text-[14px] font-medium text-ink">
                  The bottleneck was not the prover
                </h4>
                <p className="mt-3 text-[13.5px] leading-relaxed text-muted text-pretty">
                  The obvious suspect was the process-wide lock every SMT call serialises through.
                  Profiling said otherwise: verification runs once per tick, so its share{" "}
                  <em>falls</em> as the ward grows &mdash; 53% of the tick at one bed, 20% at five
                  hundred. The real cost was associative memory retrieval, which every bed ran
                  separately against a case store that itself grows with the ward. That is
                  quadratic in ward size, and it was 83% of the tick at a hundred beds.
                </p>
                <p className="mt-3 text-[13.5px] leading-relaxed text-muted text-pretty">
                  One matrix multiply for the whole ward replaced one per bed. Per-bed cost went
                  from climbing &mdash; 1.40 to 4.59 ms as the ward grew &mdash; to flat.
                </p>
              </Panel>

              <Panel className="p-5">
                <StatusChip tone="med">Found while measuring</StatusChip>
                <h4 className="mt-3 text-[14px] font-medium text-ink">
                  A correctness bug the profiler surfaced
                </h4>
                <p className="mt-3 text-[13.5px] leading-relaxed text-muted text-pretty">
                  The batched retrieval disagreed with the per-bed one. The cause was not the
                  batching: hypervector bundling resolved ties &mdash; dimensions summing to exactly
                  zero &mdash; with fresh randomness on every call. Ties only occur when an{" "}
                  <em>even</em> number of features contribute, which is to say on beds with missing
                  data. On a thirty-bed ward that was sixteen beds and roughly 12% of each of their
                  vectors.
                </p>
                <p className="mt-3 text-[13.5px] leading-relaxed text-muted text-pretty">
                  The associative memory was returning different precedents for an unchanged
                  patient, concentrated on the least-instrumented ones. Tie-breaking is now a fixed
                  pattern drawn once. The memory&rsquo;s previously published retrieval AUROC was
                  measured through that noise and is being re-measured.
                </p>
              </Panel>
            </div>
          </Section>

          {/* ── 9 · Negative results ──────────────────────────────────── */}
          <Section
            id="invention"
            eyebrow="12 · What we invented"
            title="Most of this is assembled. Some of it is not."
            lede="Saying which is which is what makes the rest credible. Stacked ensembles, conformal prediction, MiniRocket, hyperdimensional computing, label-shift estimation, Z3, federated averaging, transfer entropy, rPPG — all standard, none ours, none claimed. Four things are ours."
          >
            <Reveal className="mt-8">
              <div className="grid gap-4 md:grid-cols-2">
                {[
                  ["The refusal architecture",
                   "Verification that tests claims, not code. The integrity audit plants known lies and asserts the guards fire. The self-improvement search admitted 0 of 16 candidates. The alarm-policy artifact records a refusal as its result. Three derive scripts refuse to save an artifact that loses."],
                  ["Cross-modal HDC resonance gate",
                   "Hyperdimensional geometry used to arbitrate agreement between physically independent acquisition pathways rather than to classify. Real deterioration moves every pathway; a sensor glitch moves one. 91% artifact false-alarm removal, leave-one-patient-out."],
                  ["Portable proof certificates",
                   "Ed25519-signed SMT-LIB2 a third party re-checks without trusting this system — a signed lie is caught by the re-check, not by taking our word for the signature."],
                  ["Physiologic-liveness consensus",
                   "Byzantine fault-tolerant consensus whose liveness is established from physiology rather than from a timeout."],
                ].map(([t, d]) => (
                  <Panel key={t} className="p-5">
                    <h3 className="text-[14px] font-medium text-ink">{t}</h3>
                    <p className="mt-2 text-[13.5px] leading-relaxed text-muted text-pretty">{d}</p>
                  </Panel>
                ))}
              </div>
            </Reveal>

            <Reveal className="mt-10">
              <Panel className="p-6">
                <div className="flex flex-wrap items-center gap-3">
                  <StatusChip tone="low">
                    {pct(falsification.coverage)} falsification coverage
                  </StatusChip>
                  <h3 className="text-[15px] font-medium text-ink">
                    Proving our own checks can fail
                  </h3>
                </div>
                <p className="mt-3 text-[14px] leading-relaxed text-muted text-pretty">
                  There are 452 named checks in this repository and, until this,
                  nothing proved a single one was <em>capable</em> of failing. A check
                  that cannot fail is not a weak check — it is a lie with a green tick,
                  and it is worse than no check because it is counted as evidence. Four
                  such checks were found by hand in one session.
                </p>
                <p className="mt-3 text-[14px] leading-relaxed text-muted text-pretty">
                  This is <strong>not</strong> code-coverage mutation testing. That
                  perturbs syntax and asks whether any test fails. This perturbs the{" "}
                  <em>scientific invariant</em> — causality, the exchangeability unit,
                  class conditioning, fit identity — and asks whether a{" "}
                  <em>specific named check</em> fails. A surviving syntactic mutant
                  means weak coverage; a surviving semantic mutant means a published
                  claim nobody is defending. So the deliverable is the survivor list,
                  not the percentage.
                </p>

                <div className="scroll-x mt-6">
                  <DataTable
                    caption="Falsification coverage by suite"
                    head={["Suite", "Checks", "Mutations", "Killed", "Coverage"]}
                  >
                    {falsification.suites.map((s) => (
                      <Row
                        key={s.suite}
                        cells={[
                          s.suite, count(s.checks), count(s.mutations), count(s.killed),
                          s.mutations ? pct(s.killed / s.mutations) : "—",
                        ]}
                      />
                    ))}
                  </DataTable>
                </div>
                <p className="mt-3 text-[12px] leading-relaxed text-dim">
                  *Lead time is <b>uncorrected</b>. Measured against a random-score null at
                  matched detection rate it shows no excess, so it reflects how long
                  patients lie in the ICU rather than how early the model warns. It is kept
                  in these tables only because the point being made is that the subgroup
                  <i> gaps</i> are flat, which does not depend on the level.
                </p>

                <div className="mt-6 rounded-md border border-med/30 bg-med/5 p-4">
                  <p className="font-mono text-[10.5px] uppercase tracking-[0.14em] text-med-ink">
                    {falsification.survivors.length} survivor
                    {falsification.survivors.length === 1 ? "" : "s"} · published, not quietly fixed
                  </p>
                  {/* Copy is keyed to the mutation, never shared across survivors. The
                      first version of this hardcoded one survivor's story and rendered
                      it for every row — which, the moment a second survivor appeared,
                      published a false statement about a suite it had never described. */}
                  {falsification.survivors.map((s) => (
                    <p key={s.name} className="mt-2 text-[13.5px] leading-relaxed text-muted text-pretty">
                      <code className="font-mono text-[12.5px] text-ink">{s.suite}</code>{" "}
                      {SURVIVOR_COPY[s.name] ?? (
                        <>
                          has an undefended claim: no named check fails when{" "}
                          <code className="font-mono text-[12.5px] text-ink">{s.name}</code>{" "}
                          is applied. Expected to break{" "}
                          {s.missed.map((m) => `"${m}"`).join(", ")}.
                        </>
                      )}
                    </p>
                  ))}
                </div>

                <p className="mt-4 text-[14px] leading-relaxed text-dim text-pretty">
                  Coverage is over {count(falsification.mutations)} mutations across{" "}
                  {count(falsification.suites.length)} suites — not over all 452 checks.
                  The rest are unaudited. {pct(falsification.coverage)} is{" "}
                  {pct(falsification.coverage)} <em>of what was attempted</em>, and
                  saying otherwise would be the same species of lie the apparatus exists
                  to catch.
                </p>
              </Panel>
            </Reveal>
          </Section>

          <Section
            id="abstention"
            eyebrow="12 · When it declines to answer"
            title="The alarm can say it does not know"
            lede="A detector that only emits a number has no way to say the evidence is ambiguous, so an unresolvable hour gets rounded down to LOW and disappears. This layer gives it a third state, with a coverage claim that is measured rather than asserted."
          >
            <Reveal className="mt-8">
              <Panel className="p-6">
                <div className="flex flex-wrap items-center gap-3">
                  <StatusChip tone="low">
                    {pct(conformal.lockedCovPos)} coverage on locked patients
                  </StatusChip>
                  <h3 className="text-[15px] font-medium text-ink">
                    Conformal prediction sets at &alpha; = {conformal.alpha}
                  </h3>
                </div>
                <p className="mt-3 text-[14px] leading-relaxed text-muted text-pretty">
                  Each hour gets a set over{" "}
                  <code className="font-mono text-[13px] text-ink">&#123;0,1&#125;</code>, where{" "}
                  <code className="font-mono text-[13px] text-ink">&#123;0,1&#125;</code> is an
                  explicit refusal to decide. &alpha; is not a setting that can simply be
                  declared: on correlated ICU hours the delivered coverage does not track the
                  requested level, so the level is chosen on one split and confirmed on a locked
                  one. It misses {f4(1 - conformal.lockedCovPos)} against a requested{" "}
                  {conformal.alpha} — a{" "}
                  <span className="num text-ink">{conformal.miscoverageRatio}&times;</span>{" "}
                  overrun, inside the {conformal.overrunLimit}&times; gate.
                </p>

                <p className="mt-4 text-[14px] leading-relaxed text-muted text-pretty">
                  The finding is the abstention rate:{" "}
                  <span className="num text-ink">{pct(conformal.abstainRate)}</span> of hours are
                  ones the detector cannot separate at this confidence. That is what AUROC{" "}
                  {f3(headline.production.auroc)} at {pct(meta.setA.prevalence)} hourly prevalence
                  means stated per hour instead of per cohort.
                </p>

                <div className="scroll-x mt-6">
                  <DataTable
                    caption="What abstention does to the shipped alarm, bracketed both ways. *Lead time is uncorrected \u2014 it does not beat a random-score null at matched detection rate, so it is shown for completeness, not as a claim."
                    head={["Policy", "Utility", "Detection", "False alarm", "Lead (h)*"]}
                  >
                    {conformal.conversion.map((c) => (
                      <Row
                        key={c.policy}
                        cells={[
                          c.policy,
                          signed(c.utility, 4),
                          f3(c.detection),
                          f3(c.falseAlarm),
                          c.leadHours.toFixed(1),
                        ]}
                      />
                    ))}
                  </DataTable>
                </div>
                <p className="mt-4 text-[14px] leading-relaxed text-muted text-pretty">
                  Neither reading beats the shipped policy, so this is not a better firing rule
                  and is not claimed as one. What a ward does with a refusal is a clinical values
                  judgement, so both bounds are published and neither is picked here.
                </p>
              </Panel>
            </Reveal>
          </Section>

          <Section
            id="corrections"
            eyebrow="13 · Three findings that were bugs"
            title="The same arithmetic error, three times, wearing a lab coat"
            lede="Three separate results on this page used to be more dramatic than they are. Each read as a discovery about deployment. Each was the same mistake: a quantity read off one model's score distribution and applied to another's."
          >
            <Reveal className="mt-8">
              <Panel className="p-6">
                <div className="flex flex-wrap items-center gap-3">
                  <StatusChip tone="med">corrected</StatusChip>
                  <h3 className="text-[15px] font-medium text-ink">
                    The cross-hospital &ldquo;coverage collapse&rdquo; was a two-fit bug
                  </h3>
                </div>
                <p className="mt-3 text-[14px] leading-relaxed text-muted text-pretty">
                  This page previously reported that conformal coverage collapsed from 0.929 to
                  0.739 across hospitals, and concluded the guarantee did not travel. The external
                  harness fitted the detector <em>twice</em> &mdash; once to score the calibration
                  rows, again to score the target &mdash; so the bands were calibrated on one
                  model&rsquo;s scores and applied to a different model&rsquo;s. With a single
                  consistent fit, septic coverage goes{" "}
                  <span className="num text-ink">{f4(conformal.external.withinCovPos)}</span>{" "}
                  &rarr;{" "}
                  <span className="num text-ink">{f4(conformal.external.acrossCovPos)}</span>: a{" "}
                  {Math.abs(conformal.external.delta * 100).toFixed(1)}-point drop, not nineteen.
                </p>

                <div className="scroll-x mt-6">
                  <DataTable
                    caption="Transferred conformal bands versus bands calibrated on the target's own labels"
                    head={["Detector", "Direction", "Transferred", "Weighted", "Oracle (labels)"]}
                  >
                    {transfer.coverage.map((c) => (
                      <Row
                        key={`${c.detector}-${c.direction}`}
                        cells={[c.detector, c.direction, f4(c.plain), f4(c.weighted), f4(c.oracle)]}
                      />
                    ))}
                  </DataTable>
                </div>
                <p className="mt-4 text-[14px] leading-relaxed text-muted text-pretty">
                  Transferred bands land within 0.002 of oracle bands. In the last row the oracle
                  itself only reaches {f4(transfer.coverage[3]?.oracle ?? 0.8695)} &mdash; that
                  shortfall is the within-site finite-sample limit, not a transfer failure. There
                  was nothing there for adaptation to recover.
                </p>
              </Panel>
            </Reveal>

            <Reveal className="mt-6">
              <Panel className="p-6">
                <div className="flex flex-wrap items-center gap-3">
                  <StatusChip tone="high">significantly harmful</StatusChip>
                  <h3 className="text-[15px] font-medium text-ink">
                    Label-free adaptation, tested and refused
                  </h3>
                </div>
                <p className="mt-3 text-[14px] leading-relaxed text-muted text-pretty">
                  Hospital prevalence differs by 1.54&times;, which looked like label shift a new
                  site could correct without labels. A domain classifier separates the two
                  hospitals&rsquo; covariates at AUC{" "}
                  <span className="num text-ink">
                    {transfer.directions.map((d) => f3(d.domainAuc)).join(" / ")}
                  </span>{" "}
                  &mdash; that is covariate shift, which violates the assumption the correction
                  needs. Applying it anyway is significantly harmful in both directions.
                </p>
                <div className="scroll-x mt-6">
                  <DataTable
                    caption="Locked-half utility at a new hospital, adapted versus unadapted"
                    head={["Direction", "Unadapted", "Prior-corrected", "Oracle", "Reachable gap", "Δ vs unadapted"]}
                  >
                    {transfer.directions.map((d) => (
                      <Row
                        key={d.direction}
                        cells={[
                          d.direction,
                          signed(d.unadapted, 4),
                          signed(d.priorCorrected, 4),
                          signed(d.oracle, 4),
                          signed(d.reachableGap, 4),
                          `${signed(d.priorDelta, 4)} ${ciSigned(d.priorCi, 4)}`,
                        ]}
                      />
                    ))}
                  </DataTable>
                </div>
                <p className="mt-4 text-[14px] leading-relaxed text-muted text-pretty">
                  The reachable gap is the load-bearing column: even <em>with</em> target labels,
                  re-deriving the threshold buys{" "}
                  {transfer.directions.map((d) => signed(d.reachableGap, 4)).join(" and ")}. The
                  source threshold already transfers near-optimally, so the dramatic
                  cross-hospital utility collapse reported elsewhere was largely an artifact of
                  scoring a target hospital with a constant derived against a different fit.
                </p>
              </Panel>
            </Reveal>
          </Section>

          <Section
            id="negative"
            eyebrow="12 · What did not work"
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

                <div className="mt-6">
                  <TrialScatter />
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
                <div className="mt-6">
                  <AblationBars />
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
            eyebrow="13 · Integrity audit"
            title="How do you know any of this is true?"
            lede="An adversarial suite plants a known lie in the evaluation and asserts that the guard catches it. It verifies the claims, not the code — and it once caught a flaw in itself."
          >
            <Reveal className="mt-8">
              <Panel className="p-5 sm:p-7">
                <IntegrityBars />
              </Panel>
            </Reveal>

            <Stagger className="mt-4 grid gap-3 sm:grid-cols-2">
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
            eyebrow="14 · What this does not show"
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
            eyebrow="15 · Reproduce it"
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
