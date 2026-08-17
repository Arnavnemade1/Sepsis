import type { Metadata } from "next";
import Link from "next/link";

import { Section, Panel, SourceNote, StatusChip } from "@/components/data/primitives";
import { Reveal, Stagger, Item } from "@/components/motion/reveal";
import { site, disclaimer } from "@/lib/site-config";
import { headline } from "@/lib/results";
import { f3 } from "@/lib/format";

export const metadata: Metadata = {
  title: "About",
  description:
    "Why AegisOS exists, the principle it is built on, and what it is not. A year-long research build validated on public ICU data.",
};

const phases = [
  {
    n: "01",
    t: "Real, measurable detection",
    b: "Data pipeline, leakage-free features, and an evaluation harness. The project's original hand-tuned detector was measured on real data for the first time and scored barely above chance.",
    done: true,
  },
  {
    n: "02",
    t: "Contactless optical biosensor",
    b: "Pulse, perfusion and respiratory signal recovered from ordinary camera video, with face-mesh regions of interest and a signal-quality gate.",
    done: true,
  },
  {
    n: "03",
    t: "Trustworthy probabilities",
    b: "Post-hoc calibration, a unified runtime, and an end-to-end verification suite. This is where the output stopped being a score and became a probability.",
    done: true,
  },
  {
    n: "04",
    t: "Camera-path validation",
    b: "Heart-rate-variability features validated against real simultaneous PPG and ECG recordings, plus the organ-system body-state model.",
    done: true,
  },
  {
    n: "05",
    t: "Accuracy & generalization",
    b: "Cross-hospital external validation, and the finding that detection here is data-limited rather than model-limited. Several promising ideas were tested and rejected.",
    done: true,
  },
  {
    n: "06",
    t: "Real HRV → sepsis, multi-site",
    b: "Training the HRV model on credentialed waveform data, multi-site training to shrink the cross-hospital gap, and real-camera optics validation.",
    done: false,
  },
];

export default function AboutPage() {
  return (
    <div className="mx-auto max-w-6xl px-5 pt-28 pb-24 sm:px-8">
      <header className="max-w-3xl">
        <p className="eyebrow">About</p>
        <h1 className="mt-5 font-display text-[clamp(2.25rem,5.5vw,3.5rem)] font-semibold leading-[1.02] tracking-[-0.035em] text-ink text-balance">
          Honesty over demo-shine.
        </h1>
        <p className="mt-5 text-[17px] leading-relaxed text-muted text-pretty">
          {site.name} is a year-long research build. It began with a detector that looked
          perfect in a demo and, when finally measured on real ICU data, scored barely
          above chance. Everything since has been rebuilt on real labelled data with
          leakage-free validation — and the failures have been kept in the record.
        </p>
      </header>

      {/* ── Origin ────────────────────────────────────────────────────── */}
      <Section className="mt-20" eyebrow="Why" title="The demo was an illusion">
        <div className="mt-8 grid gap-10 lg:grid-cols-[minmax(0,1.15fr)_minmax(0,1fr)] lg:gap-16">
          <div className="space-y-5">
            <p className="text-[16px] leading-relaxed text-muted text-pretty">
              The first version of this system detected sepsis flawlessly. It also happened
              that the simulator generating the patients and the detector reading them had
              been tuned against each other. Pointed at PhysioNet&apos;s real ICU data, the
              same detector scored an AUROC of about 0.64 and was essentially uncalibrated.
            </p>
            <p className="text-[16px] leading-relaxed text-muted text-pretty">
              That result set the direction for everything after it. If a number cannot
              name the real dataset and the real metric that produced it, it does not go on
              the page. If a capability cannot be justified by the data, it gets a label
              saying so rather than a quiet omission.
            </p>
            <p className="text-[16px] leading-relaxed text-muted text-pretty">
              The current honest deployment figure is a cross-hospital AUROC of{" "}
              <span className="num text-ink">{f3(headline.production.auroc)}</span> — not
              the flattering within-hospital number, and not a figure from a synthetic
              cohort.
            </p>
          </div>

          <Reveal>
            <Panel className="p-6">
              <p className="eyebrow mb-4">The rule</p>
              <ul className="space-y-4 text-[14px] leading-relaxed text-muted">
                {[
                  "Every number is reported with the dataset and protocol that earned it.",
                  "Negative results are published, not buried.",
                  "Anything the data cannot justify is labelled, not implied.",
                  "The guards that enforce this are themselves tested adversarially.",
                ].map((r) => (
                  <li key={r} className="flex gap-3">
                    <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-low" />
                    <span className="text-pretty">{r}</span>
                  </li>
                ))}
              </ul>
            </Panel>
          </Reveal>
        </div>
      </Section>

      {/* ── Timeline ──────────────────────────────────────────────────── */}
      <Section
        className="border-t border-line py-20 sm:py-28"
        eyebrow="Build record"
        title="Where it has got to"
      >
        <Stagger className="mt-10 overflow-hidden rounded-lg border border-line bg-s1/62 backdrop-blur-xl">
          {phases.map((p, i) => (
            <Item key={p.n}>
              <div
                className={`flex flex-col gap-3 p-5 sm:flex-row sm:items-start sm:gap-6 sm:p-6 ${
                  i > 0 ? "border-t border-line-soft" : ""
                }`}
              >
                <span className="num shrink-0 font-mono text-[12px] text-faint sm:pt-1">
                  {p.n}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-3">
                    <h3 className="text-[15px] font-medium text-ink">{p.t}</h3>
                    <StatusChip tone={p.done ? "low" : "nodata"}>
                      {p.done ? "Complete" : "Not started"}
                    </StatusChip>
                  </div>
                  <p className="mt-2 max-w-2xl text-[14px] leading-relaxed text-muted text-pretty">
                    {p.b}
                  </p>
                </div>
              </div>
            </Item>
          ))}
        </Stagger>
      </Section>

      {/* ── What it is not ────────────────────────────────────────────── */}
      <Section
        className="border-t border-line py-20 sm:py-28"
        eyebrow="Scope"
        title="What this is not"
      >
        <Stagger className="mt-10 grid gap-3 sm:grid-cols-2">
          {[
            {
              t: "Not a medical device",
              b: "It has no regulatory clearance of any kind and is not for clinical use. Nothing here should inform the care of a real patient.",
            },
            {
              t: "Not a finished product",
              b: "Several subsystems are working machinery whose clinical claim is not yet earned. Those are labelled throughout rather than presented as results.",
            },
            {
              t: "Not trained on private data",
              b: "All validation runs on public, de-identified datasets — PhysioNet/CinC 2019 and BIDMC — under their own licences.",
            },
            {
              t: "Not a benchmark record",
              b: "Wearable-only sepsis detection sits near a data ceiling. The contribution here is the rigour and the refusal machinery, not a leaderboard score.",
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

      {/* ── Colophon ──────────────────────────────────────────────────── */}
      <Section className="border-t border-line py-20 sm:py-28" eyebrow="Colophon">
        <Reveal>
          <Panel className="p-6 sm:p-8">
            <dl className="grid gap-x-10 gap-y-5 text-[14px] sm:grid-cols-2">
              <div>
                <dt className="font-mono text-[10.5px] uppercase tracking-[0.08em] text-dim">
                  Research codename
                </dt>
                <dd className="mt-1.5 text-muted">{site.codename}</dd>
              </div>
              <div>
                <dt className="font-mono text-[10.5px] uppercase tracking-[0.08em] text-dim">
                  Data
                </dt>
                <dd className="mt-1.5 text-muted">
                  PhysioNet/CinC 2019 (CC BY 4.0) · BIDMC PPG &amp; ECG · DDInter
                </dd>
              </div>
              <div>
                <dt className="font-mono text-[10.5px] uppercase tracking-[0.08em] text-dim">
                  Source
                </dt>
                <dd className="mt-1.5">
                  <a
                    href={site.repo}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-accent-ink transition-colors duration-160 ease-instrument hover:text-ink"
                  >
                    github.com/arnavnemade/Sepsis ↗
                  </a>
                </dd>
              </div>
              <div>
                <dt className="font-mono text-[10.5px] uppercase tracking-[0.08em] text-dim">
                  Licence
                </dt>
                <dd className="mt-1.5 text-muted">{disclaimer}</dd>
              </div>
            </dl>
            <SourceNote>
              Citation — Reyna MA, Josef C, Jeter R, Shashikumar SP, Moody B, Westover MB,
              Sharma A, Nemati S, Clifford GD. Early Prediction of Sepsis from Clinical Data.
              Computing in Cardiology, 2019.
            </SourceNote>
          </Panel>
        </Reveal>

        <Reveal className="mt-8">
          <Link
            href="/research"
            className="inline-flex items-center gap-1.5 text-[14px] text-accent-ink transition-colors duration-160 ease-instrument hover:text-ink"
          >
            See the evaluation →
          </Link>
        </Reveal>
      </Section>
    </div>
  );
}
