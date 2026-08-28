import { memo } from "react";

/**
 * The subsystem inventory.
 *
 * WHY IT CARRIES A STATUS COLUMN
 * ------------------------------
 * A list of forty module names is a boast, not information — every project can
 * produce one. What makes this inventory worth reading is that each entry says what
 * kind of thing it is: running in the system, backed by a measurement, built and then
 * REFUTED by its own benchmark, or complete but blocked on data nobody here can
 * legally obtain. The refuted ones are listed on purpose. A research record that shows
 * only what worked is the one you cannot trust.
 *
 * Every entry names a real module in the repository. Nothing is aspirational.
 */

type Status = "live" | "measured" | "refuted" | "blocked";

interface Sub {
  name: string;
  path: string;
  status: Status;
  note: string;
}

interface Plane {
  id: string;
  title: string;
  blurb: string;
  subs: Sub[];
}

const STATUS: Record<Status, { label: string; cls: string; title: string }> = {
  live: {
    label: "live",
    cls: "border-low/40 text-low-ink bg-low/10",
    title: "Runs in the system on every tick",
  },
  measured: {
    label: "measured",
    cls: "border-accent/40 text-accent-ink bg-accent/10",
    title: "Has a number in artifacts/ produced by a verification suite",
  },
  refuted: {
    label: "refuted",
    cls: "border-med/40 text-med-ink bg-med/10",
    title: "Built, benchmarked, and NOT adopted — the negative is the result",
  },
  blocked: {
    label: "blocked",
    cls: "border-line-bright text-dim bg-s2",
    title: "Complete, but validation needs credentialed data (a user action)",
  },
};

const PLANES: Plane[] = [
  {
    id: "detection",
    title: "Detection",
    blurb:
      "The part that produces a number. Four separate attempts to raise it were run and none survived its own benchmark, which is why the ceiling is reported as data-limited rather than asserted to be.",
    subs: [
      { name: "Stacked ensemble", path: "ml/models/stacked.py", status: "live", note: "hgb+hgb2+et+lr+xgb@utility — the shipped detector" },
      { name: "Utility-aligned objective", path: "ml/eval/utility.py", status: "measured", note: "the scoring function's own decision rule, worth more than the library choice" },
      { name: "Derived operating point", path: "ml/operating_point.py", status: "live", note: "thresholds from out-of-fold data, not constants" },
      { name: "Conformal abstention", path: "ml/conformal.py", status: "live", note: "the alarm's third state, with a measured coverage claim" },
      { name: "Shift diagnosis", path: "ml/shift.py", status: "refuted", note: "label-free adaptation is significantly harmful here" },
      { name: "Causal MiniRocket", path: "ml/models/minirocket.py", status: "refuted", note: "+0.0012 AUROC, n.s. — a random convolutional basis buys nothing" },
      { name: "Forest-VSA", path: "ml/models/forest_vsa.py", status: "refuted", note: "leaf topology loses to the booster that produced it" },
      { name: "Causal sequence models", path: "ml/models/sequence.py", status: "refuted", note: "TCN / Mamba-2 SSM / transformer all lose to the boosters" },
      { name: "Self-improvement search", path: "ml/selfimprove.py", status: "refuted", note: "0 of 16 champion mutations admitted" },
    ],
  },
  {
    id: "kernel",
    title: "Kernel & safety",
    blurb:
      "Everything between a prediction and an action. No write reaches a patient model without passing a solver, an audit chain and a capability check.",
    subs: [
      { name: "HDKernel", path: "aegis/kernel/hdkernel.py", status: "live", note: "cooperative kernel — syscalls, IPC, process table" },
      { name: "HDSched", path: "aegis/kernel/hdsched.py", status: "measured", note: "benchmarkable dispatcher; ties best-tuned EDF, overhead disclosed" },
      { name: "Z3 trap gate", path: "aegis/syscalls.py", status: "live", note: "audited, solver-gated write path" },
      { name: "Clinical safety DSL", path: "aegis/clinical_dsl.py", status: "measured", note: "hospital-authored rules compiled to Z3, with counterexamples" },
      { name: "Proof export", path: "aegis/proofexport.py", status: "measured", note: "Ed25519-signed SMT-LIB2 certificates a third party re-checks" },
      { name: "RBAC rings", path: "aegis/rbac.py", status: "measured", note: "5 roles / 4 rings, soundness proven, hash-chained audit" },
      { name: "eBPF driver verifier", path: "aegis/kernel/ebpf.py", status: "measured", note: "static termination / memory / capability checks" },
      { name: "Temporal causal graph", path: "aegis/kernel/causal_graph.py", status: "live", note: "tamper-evident DAG with an evidence-type firewall" },
      { name: "Physiologic identity guard", path: "aegis/identity.py", status: "measured", note: "stream-swap detector, EER ~0.18 — explicitly not biometric" },
    ],
  },
  {
    id: "sensing",
    title: "Sensing",
    blurb:
      "Where new signal would have to come from, since more model on the same features has now failed four times.",
    subs: [
      { name: "Optical biosensor", path: "vision/pipeline.py", status: "live", note: "contactless rPPG, perfusion, distress signs" },
      { name: "HRV engine", path: "physio/hrv.py", status: "measured", note: "validated PPG-vs-ECG on real BIDMC" },
      { name: "PPG morphology", path: "physio/ppg_morphology.py", status: "measured", note: "dicrotic-notch vascular tone; monotone erosion response" },
      { name: "Cardio-respiratory coupling", path: "physio/coupling.py", status: "blocked", note: "transfer entropy + phase sync; needs MIMIC-IV Waveform for a detection claim" },
      { name: "Cross-modal resonance", path: "aegis/resonance.py", status: "measured", note: "91% artifact false-alarm removal, LOPO-validated" },
      { name: "MIMIC waveform adapter", path: "physio/data/mimic_waveform.py", status: "blocked", note: "written and ready; needs CITI training + a data-use agreement" },
    ],
  },
  {
    id: "reasoning",
    title: "Clinical reasoning",
    blurb:
      "The layers that turn a score into something a clinician can argue with — each one able to refuse.",
    subs: [
      { name: "Multi-agent plane", path: "aegis/agents.py", status: "live", note: "five named specialists debate to consensus" },
      { name: "BFT consensus brain", path: "aegis/consensus.py", status: "measured", note: "N=3f+1 quorum certificate + physiologic-liveness fallback" },
      { name: "Autonomous orchestrator", path: "aegis/orchestrator.py", status: "measured", note: "one Non-Maleficence invariant; no uncertified egress" },
      { name: "Living pharmacopoeia", path: "aegis/pharmacopoeia.py", status: "measured", note: "1,887 drugs / 143k interaction edges as Z3 constraints" },
      { name: "Treatment twin (PBPK)", path: "aegis/pharmacokinetics.py", status: "measured", note: "drug distribution + adverse-event projection" },
      { name: "Federated intelligence", path: "aegis/fedlearn.py", status: "measured", note: "falsifiable FedAvg: recovers cross-site AUROC without sharing records" },
      { name: "Scenario Lab", path: "aegis/scenarios.py", status: "measured", note: "synthetic cases, behaviour-only grading, data firewall" },
    ],
  },
];

const Badge = memo(function Badge({ status }: { status: Status }) {
  const s = STATUS[status];
  return (
    <span
      title={s.title}
      className={`shrink-0 rounded-sm border px-1.5 py-0.5 font-mono text-[10px] leading-none tracking-[0.08em] uppercase ${s.cls}`}
    >
      {s.label}
    </span>
  );
});

export function SubsystemMap() {
  const total = PLANES.reduce((n, p) => n + p.subs.length, 0);
  const counts = (["live", "measured", "refuted", "blocked"] as Status[]).map((k) => ({
    k,
    n: PLANES.reduce((n, p) => n + p.subs.filter((s) => s.status === k).length, 0),
  }));

  return (
    <div className="mt-8">
      <div className="flex flex-wrap items-center gap-x-5 gap-y-2 border-y border-line py-3">
        <span className="font-mono text-[11px] uppercase tracking-[0.14em] text-faint">
          {total} subsystems
        </span>
        {counts.map(({ k, n }) => (
          <span key={k} className="flex items-center gap-1.5">
            <Badge status={k} />
            <span className="num font-mono text-[11px] text-dim">{n}</span>
          </span>
        ))}
      </div>

      <div className="mt-8 grid gap-8 md:grid-cols-2">
        {PLANES.map((plane) => (
          <section key={plane.id} className="min-w-0">
            <h3 className="font-display text-[15px] font-semibold tracking-[-0.01em] text-ink">
              {plane.title}
            </h3>
            <p className="mt-1.5 text-[13px] leading-[1.55] text-dim">{plane.blurb}</p>
            <ul className="mt-4 space-y-0">
              {plane.subs.map((s) => (
                <li
                  key={s.path}
                  className="grid grid-cols-[1fr_auto] items-baseline gap-x-3 border-b border-line-soft py-2.5"
                >
                  <div className="min-w-0">
                    <span className="text-[13.5px] font-medium text-ink">{s.name}</span>
                    <span className="ml-2 font-mono text-[11px] text-faint">{s.path}</span>
                    <p className="mt-0.5 text-[12.5px] leading-[1.5] text-dim">{s.note}</p>
                  </div>
                  <Badge status={s.status} />
                </li>
              ))}
            </ul>
          </section>
        ))}
      </div>
    </div>
  );
}
