import Link from "next/link";

import { Hero } from "@/components/hero/hero";
import { ExplainerFrame } from "@/components/explainer/explainer-frame";
import { Section, Panel, MetricTile, SourceNote, StatusChip } from "@/components/data/primitives";
import { Reveal, Stagger, Item } from "@/components/motion/reveal";
import { Counter } from "@/components/motion/counter";
import { headline, generalization, meta, scale, scaleCeiling, subgroup } from "@/lib/results";
import { f3, pct, count } from "@/lib/format";
import { consoleUrl, disclaimer } from "@/lib/site-config";

const capabilities = [
  {
    title: "It watches continuously",
    body: "Sepsis is a trajectory, not a moment. AegisOS reads vitals every hour of a stay and reasons about the shape of the change, not a single threshold crossing.",
    chip: "Validated",
    tone: "low" as const,
  },
  {
    title: "The probability means what it says",
    body: "A calibrated 4% is four percent. Post-hoc calibration cut expected calibration error by ~58×, and it survives the move to a different hospital.",
    chip: "Validated",
    tone: "low" as const,
  },
  {
    title: "The treatment carries a proof",
    body: "Every proposed protocol is checked against formal safety invariants — fluid caps, vasopressor limits, allergy constraints — by an SMT solver before it is shown to anyone.",
    chip: "Verified",
    tone: "low" as const,
  },
  {
    title: "It says why, and where",
    body: "Per-organ severity, the drivers behind the score, and a plain-language narrative — so a clinician can disagree with it for a specific reason.",
    chip: "Validated",
    tone: "low" as const,
  },
];

const homeCeiling = scaleCeiling();

export default function Home() {
  return (
    <>
      <Hero />

      <div className="mx-auto max-w-6xl px-5 sm:px-8">
        {/* ── Stakes ──────────────────────────────────────────────────── */}
        <Section className="py-20 sm:py-28" eyebrow="The problem">
          <div className="grid gap-10 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)] lg:gap-16">
            <h2 className="font-display text-[clamp(1.75rem,4vw,2.75rem)] font-semibold leading-[1.08] tracking-[-0.03em] text-ink text-balance">
              Every hour of delay costs survival.
            </h2>
            <div className="space-y-5">
              <p className="text-[16px] leading-relaxed text-muted text-pretty">
                Sepsis kills roughly eleven million people a year, and the odds fall with
                every hour treatment is late. The earliest signs are physiological —
                autonomic, perfusion, inflammatory — and they appear in signals a wearable
                or a camera can capture continuously, long before anyone is at the bedside.
              </p>
              <p className="text-[16px] leading-relaxed text-muted text-pretty">
                The hard part was never noticing that. It is building something a clinician
                has reason to believe.
              </p>
            </div>
          </div>

          <Stagger className="mt-14 grid gap-3 sm:grid-cols-4">
            <Item>
              {/* Lead time used to headline here. It was audited on 2026-08-30 against a
                  random-score null at matched detection rate and did not survive — see
                  docs/LEAD_TIME_NULL.md. Detection rate is the honest headline: it is a
                  patient-level count, not a function of stay geometry. */}
              <MetricTile
                value={<Counter to={headline.production.detectionRate * 100} decimals={0} suffix="%" />}
                label="Septic patients caught"
                detail="alarmed before Sepsis-3 onset, across hospitals"
                tone="accent"
              />
            </Item>
            <Item>
              <MetricTile
                value={count(meta.setA.patients + meta.setB.patients)}
                label="Patients in the validation"
                detail="two hospitals, trained on one and tested on the other"
              />
            </Item>
            <Item>
              <MetricTile
                value={`${meta.bootstrap}×`}
                label="Bootstrap resamples"
                detail="patient-level, behind every confidence interval"
              />
            </Item>
            <Item>
              <MetricTile
                value={count(homeCeiling ? homeCeiling.beds : 0)}
                label="Beds it runs in real time"
                detail="measured on the real tick loop, not projected from one bed"
              />
            </Item>
          </Stagger>
        </Section>

        {/* ── What it is ──────────────────────────────────────────────── */}
        <Section
          className="border-t border-line py-20 sm:py-28"
          eyebrow="What it is"
          title="An operating system for one human body."
          lede="Not a dashboard on top of a model. A runtime: it ingests signal, maintains a live digital twin of the patient, decides, and refuses to emit anything it cannot certify."
        >
          <Stagger className="mt-12 grid gap-3 sm:grid-cols-2">
            {capabilities.map((c) => (
              <Item key={c.title}>
                <Panel className="h-full p-6">
                  <div className="flex items-start justify-between gap-4">
                    <h3 className="text-[15px] font-medium text-ink">{c.title}</h3>
                    <StatusChip tone={c.tone}>{c.chip}</StatusChip>
                  </div>
                  <p className="mt-3 text-[14px] leading-relaxed text-muted text-pretty">
                    {c.body}
                  </p>
                </Panel>
              </Item>
            ))}
          </Stagger>

          <Reveal className="mt-8">
            <Link
              href="/technology"
              className="inline-flex items-center gap-1.5 text-[14px] text-accent-ink transition-colors duration-160 ease-instrument hover:text-ink"
            >
              How the system works →
            </Link>
          </Reveal>
        </Section>

        {/* ── The loop, end to end ────────────────────────────────────── */}
        <Section
          className="border-t border-line py-20 sm:py-28"
          eyebrow="End to end"
          title="Signal in, certified protocol out."
          lede="Twenty seconds, no audio, real numbers. Every figure in it is read from the same evaluation data as the research page — the animation cannot claim something the measurements do not."
        >
          <Reveal className="mt-12">
            <ExplainerFrame />
          </Reveal>
          <SourceNote>
            Rendered live rather than played as video, so the hairlines and the
            tabular figures stay sharp at any pixel density.
          </SourceNote>
        </Section>

        {/* ── Honest numbers ──────────────────────────────────────────── */}
        <Section
          className="border-t border-line py-20 sm:py-28"
          eyebrow="The numbers"
          title="We lead with the number that survives a change of hospital."
          lede="It is easy to report a medical model on a random split of one hospital's patients, and it flatters every result. This one is trained on one site and tested on the other."
        >
          <Stagger className="mt-12 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
            <Item>
              <MetricTile
                value={<Counter to={headline.production.auroc} decimals={3} />}
                label="Cross-hospital AUROC"
                detail={`production model · single-GBM baseline ${f3(headline.baseline.auroc)}`}
                tone="accent"
              />
            </Item>
            <Item>
              <MetricTile
                value={<Counter to={headline.production.ece} decimals={4} />}
                label="Calibration error"
                detail="off-site — the probability still means what it says"
                tone="accent"
              />
            </Item>
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
                detail="the honest bottleneck — and not policy-fixable"
                tone="high"
              />
            </Item>
          </Stagger>

          <Reveal className="mt-4">
            <Panel className="p-6 sm:p-8">
              <div className="flex flex-wrap items-baseline gap-x-6 gap-y-3">
                <span className="num font-mono text-[30px] leading-none text-dim">
                  {f3(generalization.internalAuroc)}
                </span>
                <span className="text-dim">within hospital →</span>
                <span className="num font-mono text-[30px] leading-none text-ink">
                  {f3(generalization.externalAuroc)}
                </span>
                <span className="text-dim">across hospitals</span>
                <span className="num ml-auto font-mono text-[30px] leading-none text-high-ink">
                  −{f3(generalization.gap)}
                </span>
              </div>
              <p className="mt-5 max-w-2xl text-[14.5px] leading-relaxed text-muted text-pretty">
                Same model, same features, same code — only the hospital changes. Most
                write-ups print the left-hand number and stop. The gap is the part that
                tells you what deployment would actually feel like, so it is on the front
                page.
              </p>
              <SourceNote>
                artifacts/rigorous_eval.json · artifacts/compare_wearable.json
              </SourceNote>
            </Panel>
          </Reveal>

          <Reveal className="mt-8">
            <Link
              href="/research"
              className="inline-flex items-center gap-1.5 text-[14px] text-accent-ink transition-colors duration-160 ease-instrument hover:text-ink"
            >
              Read the full evaluation, including what failed →
            </Link>
          </Reveal>
        </Section>

        {/* ── Honesty pledge ──────────────────────────────────────────── */}
        <Section className="border-t border-line py-20 sm:py-28" eyebrow="How we report">
          <Reveal>
            <blockquote className="max-w-3xl">
              <p className="font-display text-[clamp(1.35rem,2.8vw,1.9rem)] font-medium leading-[1.3] tracking-[-0.02em] text-ink text-balance">
                Every detection number is reported with the real dataset and metric that
                earned it, and every capability the data cannot yet justify is labelled as
                such — never faked.
              </p>
            </blockquote>
          </Reveal>

          <Stagger className="mt-12 grid gap-3 sm:grid-cols-3">
            {[
              {
                t: "Negative results are published",
                b: "A self-improvement search ran 16 trials and accepted none of them. An alarm-policy search refused to promote its own challenger. Both are on the research page.",
              },
              {
                t: "The guards are audited adversarially",
                b: "A suite plants a known lie in the evaluation — shuffled labels, in-sample predictions — and asserts the guard catches it. Eight of eight fire.",
              },
              {
                t: "Numbers cannot drift from the code",
                b: "Every figure on this site is synced from the repository's own evaluation artifacts, and a CI check fails if the site and the evaluation disagree.",
              },
            ].map((x) => (
              <Item key={x.t}>
                <Panel className="h-full p-5">
                  <h3 className="text-[14px] font-medium text-ink">{x.t}</h3>
                  <p className="mt-2 text-[13.5px] leading-relaxed text-muted text-pretty">
                    {x.b}
                  </p>
                </Panel>
              </Item>
            ))}
          </Stagger>
        </Section>

        {/* ── CTA ─────────────────────────────────────────────────────── */}
        <Section className="border-t border-line py-20 sm:py-28">
          <Reveal>
            <div className="flex flex-col items-start gap-7">
              <h2 className="max-w-[20ch] font-display text-[clamp(1.75rem,4vw,2.75rem)] font-semibold leading-[1.08] tracking-[-0.03em] text-ink text-balance">
                See it running on real patient data.
              </h2>
              <div className="flex flex-wrap items-center gap-3">
                <a
                  href={consoleUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="rounded-md bg-ink px-5 py-2.5 text-[14px] font-medium text-bg transition-colors duration-160 ease-instrument hover:bg-white"
                >
                  Launch the console ↗
                </a>
                <Link
                  href="/research"
                  className="rounded-md border border-line-bright px-5 py-2.5 text-[14px] font-medium text-ink transition-colors duration-160 ease-instrument hover:bg-s2"
                >
                  Read the research
                </Link>
              </div>
              <p className="max-w-xl text-[13px] leading-relaxed text-dim">
                The console runs on a free hosting tier and sleeps when idle — the first
                load after a quiet period can take 30–60 seconds to wake. {disclaimer}
              </p>
            </div>
          </Reveal>
        </Section>
      </div>
    </>
  );
}
