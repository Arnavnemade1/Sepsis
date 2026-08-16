import type { Metadata } from "next";
import Link from "next/link";

import {
  Section,
  Panel,
  SourceNote,
  StatusChip,
  DataTable,
  Row,
} from "@/components/data/primitives";
import { Reveal, Stagger, Item } from "@/components/motion/reveal";
import { consoleUrl } from "@/lib/site-config";
import { TechStack } from "@/components/technology/tech-stack";

export const metadata: Metadata = {
  title: "Technology",
  description:
    "How AegisOS works: a layered runtime that turns wearable and camera signal into a calibrated risk, a live digital twin, and a formally-verified treatment protocol.",
};

/** Status vocabulary. `validated` means measured against real labelled data;
 *  `machinery` means the code path works but the claim is not yet earned;
 *  `gated` means it waits on credentialed data. Nothing is left unlabelled. */
type Status = "validated" | "verified" | "machinery" | "gated";

const chip: Record<Status, { tone: "low" | "med" | "nodata" | "accent"; text: string }> = {
  validated: { tone: "low", text: "Validated on real data" },
  verified: { tone: "low", text: "Formally verified" },
  machinery: { tone: "med", text: "Machinery only" },
  gated: { tone: "nodata", text: "Gated on credentialed data" },
};

const layers = [
  {
    n: "01",
    name: "Signal abstraction",
    body: "Ring buffers and Kalman filtering over raw sensor input, at microsecond latency. Halves heart-rate jitter before anything downstream sees a number.",
    status: "validated" as Status,
  },
  {
    n: "02",
    name: "Feature construction",
    body: "A streaming feature builder that produces exactly the same features at serving time as at training time — the single most common source of silent medical-model failure.",
    status: "validated" as Status,
  },
  {
    n: "03",
    name: "Calibrated detection",
    body: "A stacked ensemble trained on PhysioNet/CinC 2019, with post-hoc calibration so the output is a probability rather than a ranking.",
    status: "validated" as Status,
  },
  {
    n: "04",
    name: "Neurosymbolic overrides",
    body: "SIRS and qSOFA criteria run alongside the learned model. Where a hard clinical rule fires, it is not silently outvoted by a gradient.",
    status: "validated" as Status,
  },
  {
    n: "05",
    name: "Body state",
    body: "Risk is fused into a per-organ severity model — brain, lungs, heart, liver, kidneys, vasculature, skin — which is what the digital twin renders.",
    status: "machinery" as Status,
  },
  {
    n: "06",
    name: "Protocol synthesis",
    body: "A genetic search proposes treatment protocols; an SMT solver proves each one against the safety invariants before it can leave this layer.",
    status: "verified" as Status,
  },
  {
    n: "07",
    name: "Consensus & escalation",
    body: "Multiple specialist agents debate the case and must reach Byzantine-fault-tolerant agreement, with a safety oracle able to escalate over them.",
    status: "machinery" as Status,
  },
  {
    n: "08",
    name: "Certified egress",
    body: "Nothing leaves the runtime without an attestation entry. A decision that cannot be certified is not emitted at all — the system fails closed.",
    status: "verified" as Status,
  },
];

const invariants = [
  ["Congestive heart failure fluid cap", "CHF ⇒ fluid_rate ≤ 5.0 mL/kg/hr"],
  ["Renal fluid cap", "Renal_Failure ⇒ fluid_rate ≤ 4.0 mL/kg/hr"],
  ["Vasopressor ceiling", "vasopressor_dose ≤ 0.4 mcg/kg/min"],
  ["Vasopressor justification", "vaso > 0.1 ⇒ MAP < 65 mmHg"],
  ["Drug allergy", "Penicillin_Allergy ⇒ antibiotic ≠ Pip/Tazo"],
  ["Sepsis intervention", "Risk_HIGH ⇒ antibiotic ≠ none"],
];

const subsystems = [
  {
    name: "Optical biosensor",
    status: "machinery" as Status,
    body: "A plain camera recovers pulse, perfusion and respiratory signal from facial video. Pulse extraction is validated against real simultaneous PPG and ECG recordings — heart-rate error around 1.2 bpm, SDNN correlation 0.88 after the robustness work.",
    caveat:
      "Camera → sepsis end-to-end has never been measured against sepsis labels, and optical measurement is materially harder on darker skin. The quality gate flags low-confidence readings rather than reporting a confident wrong number.",
  },
  {
    name: "Heart-rate-variability pipeline",
    status: "gated" as Status,
    body: "An 18-feature HRV suite including HeRO-style descriptors, feeding a sepsis model. On a physiology-grounded synthetic cohort it reports AUROC around 0.9.",
    caveat:
      "That number describes the code path, not patients. The real test needs credentialed MIMIC-IV Waveform access; the adapter is built and unit-tested, waiting on credentials.",
  },
  {
    name: "Federated learning",
    status: "validated" as Status,
    body: "Two hospitals train locally and share only signed weight updates. Federation recovers cross-site performance that neither local model reaches alone, and robust aggregation defeats a deliberately poisoned participant.",
    caveat:
      "This is data minimisation, not differential privacy — and the federated model is logistic regression, because FedAvg needs averageable weights and trees have none.",
  },
  {
    name: "Physiologic identity guard",
    status: "validated" as Status,
    body: "A Mahalanobis fingerprint over physiological dynamics detects when a monitoring stream has been swapped to a different patient, and locks dispensing when it fires.",
    caveat:
      "This is a stream-swap detector, not biometrics. It is not identification, and it is not claimed to be.",
  },
  {
    name: "Temporal causal graph",
    status: "machinery" as Status,
    body: "A tamper-evident clinical DAG records what happened and why, supports counterfactual replay, and keeps observed evidence firewalled from inferred evidence.",
    caveat:
      "The structure is engineered and verified; causal claims recovered from it are hypotheses for review, not established causation.",
  },
  {
    name: "Living pharmacopoeia",
    status: "machinery" as Status,
    body: "A fused interaction graph — 1,887 drugs and 143,000 edges from DDInter, plus a curated CYP450, QT and renal layer — compiles every hazard into a solver constraint with a proof tree.",
    caveat:
      "Exhaustive over the loaded knowledge base, which is not the same as exhaustive over pharmacology. It cannot warn about an interaction nobody has published.",
  },
];

export default function TechnologyPage() {
  return (
    <div className="mx-auto max-w-6xl px-5 pt-28 pb-24 sm:px-8">
      <header className="max-w-3xl">
        <p className="eyebrow">Technology</p>
        <h1 className="mt-5 font-display text-[clamp(2.25rem,5.5vw,3.5rem)] font-semibold leading-[1.02] tracking-[-0.035em] text-ink text-balance">
          A runtime, not a dashboard.
        </h1>
        <p className="mt-5 text-[17px] leading-relaxed text-muted text-pretty">
          AegisOS is structured as an operating system for a single patient: layers with
          clear contracts, a kernel that schedules and arbitrates, and a hard rule that
          nothing reaches a clinician without a proof attached. Every capability below
          carries a status label, including the ones that are not finished.
        </p>
      </header>

      {/* ── The stack ─────────────────────────────────────────────────── */}
      {/* Full-bleed: the scrubbed display needs the whole viewport to pin
          against, so it breaks out of the page's max-width container. */}
      <div className="relative left-1/2 mt-20 w-screen -translate-x-1/2 border-y border-line">
        <TechStack layers={layers} />
      </div>

      {/* ── Formal verification ───────────────────────────────────────── */}
      <Section
        className="mt-20 border-t border-line pt-20"
        eyebrow="Formal verification"
        title="The part that is actually proven"
        lede="Machine learning cannot be proven correct. Constraint satisfaction can. So the learned model is confined to producing a probability, and everything downstream of it — every dose, every drug choice — passes through a solver that either certifies it or refuses it."
      >
        <Reveal className="mt-10">
          <DataTable caption="Safety invariants proven by the SMT solver" head={["Invariant", "Constraint"]}>
            {invariants.map(([name, rule]) => (
              <Row
                key={name}
                cells={[
                  name,
                  <code key="r" className="font-mono text-[12px] text-accent-ink">
                    {rule}
                  </code>,
                ]}
              />
            ))}
          </DataTable>
        </Reveal>

        <Stagger className="mt-4 grid gap-3 sm:grid-cols-3">
          {[
            {
              t: "It fails closed",
              b: "If the prover is unavailable or a constraint cannot be discharged, the protocol is withheld. There is no path where an uncertified order is emitted because verification was slow.",
            },
            {
              t: "The proof is portable",
              b: "Each decision exports a signed SMT-LIB certificate that a third party can re-check with a stock solver, without trusting AegisOS. A tampered certificate fails the re-check.",
            },
            {
              t: "Hospitals can add rules",
              b: "A small safety DSL lets a hospital author its own constraints, which are compiled into the same solver and checked for consistency against the existing policy.",
            },
          ].map((x) => (
            <Item key={x.t}>
              <Panel className="h-full p-5">
                <h3 className="text-[14px] font-medium text-ink">{x.t}</h3>
                <p className="mt-2 text-[13.5px] leading-relaxed text-muted text-pretty">{x.b}</p>
              </Panel>
            </Item>
          ))}
        </Stagger>
        <SourceNote>
          Invariants as implemented in the runtime&apos;s prover cage · re-checkable with a
          stock SMT solver
        </SourceNote>
      </Section>

      {/* ── Subsystems ────────────────────────────────────────────────── */}
      <Section
        className="mt-20 border-t border-line pt-20"
        eyebrow="Subsystems"
        title="What each part can and cannot claim"
        lede="The temptation in a project this size is to let a working code path imply a validated result. Each subsystem below states its status first and its caveat last."
      >
        <Stagger className="mt-10 grid gap-3 lg:grid-cols-2">
          {subsystems.map((s) => (
            <Item key={s.name}>
              <Panel className="flex h-full flex-col p-6">
                <div className="flex flex-wrap items-center gap-3">
                  <h3 className="text-[15px] font-medium text-ink">{s.name}</h3>
                  <StatusChip tone={chip[s.status].tone}>{chip[s.status].text}</StatusChip>
                </div>
                <p className="mt-3 text-[14px] leading-relaxed text-muted text-pretty">
                  {s.body}
                </p>
                <p className="mt-4 border-l-2 border-line-bright pl-4 text-[13px] leading-relaxed text-dim text-pretty">
                  {s.caveat}
                </p>
              </Panel>
            </Item>
          ))}
        </Stagger>
      </Section>

      {/* ── CTA ───────────────────────────────────────────────────────── */}
      <Section className="mt-20 border-t border-line pt-20">
        <Reveal>
          <div className="flex flex-col items-start gap-6">
            <h2 className="max-w-[22ch] font-display text-[clamp(1.5rem,3.5vw,2.25rem)] font-semibold leading-[1.1] tracking-[-0.03em] text-ink text-balance">
              The evidence for all of this is on one page.
            </h2>
            <div className="flex flex-wrap items-center gap-3">
              <Link
                href="/research"
                className="rounded-md bg-ink px-5 py-2.5 text-[14px] font-medium text-bg transition-colors duration-160 ease-instrument hover:bg-white"
              >
                Read the research
              </Link>
              <a
                href={consoleUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="rounded-md border border-line-bright px-5 py-2.5 text-[14px] font-medium text-ink transition-colors duration-160 ease-instrument hover:bg-s2"
              >
                Launch the console ↗
              </a>
            </div>
          </div>
        </Reveal>
      </Section>
    </div>
  );
}
