"use client";

import { ArrowUpRight, X } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";

import { useHealth } from "@/components/shell";
import { cx, Drawer, Kv, Tag } from "@/components/ui";
import { useCaps, usePoll } from "@/lib/hooks";
import { activePages, usePaging } from "@/lib/paging";

/* ------------------------------------------------------------------ */
/* Contracts (only the fields this view reads)                         */
/* ------------------------------------------------------------------ */
type ProcTree = {
  tree: {
    "/proc/cage"?: { certified: number; blocked: number; quarantined: number; pass_rate: number };
    "/proc/executor"?: { started: boolean; busy: boolean; calls: number; max_wait_ms: number; max_run_ms: number; violations: number };
    "/proc/journal"?: { seq: number; head: string; chain_ok: boolean; verified_window: number; retained: number; persisted: { configured: string; sink: string; written: number; dropped: number; supabase_errors: number } };
    "/proc/ksched"?: { uptime_s: number; processes: number; by_state: Record<string, number>; cpu_ms_total: number };
    "/proc/mem"?: { capacity_cells: number; used_cells: number; utilization: number; pointer_stable: boolean; bytes: number };
    "/proc/sched"?: { tick: number; mode: string; owner: string; procs: number; tiers: Record<string, number>; lane_budget: number; lanes_used: number; interrupts: number; anticipating: number };
    "/proc/llm"?: { available: boolean; prefer: string; primary_model: string; budget_remaining: number; calls: number };
    "/proc/radar"?: { ews: number; rising: boolean; warming: boolean; n: number };
    "/proc/resonance"?: { resonance: number; gated_risk: number; naive_risk: number; explain: string; alert_certified: boolean };
    "/proc/triage"?: { level: string; fused_risk: number; note_certified: boolean };
    "/proc/services"?: { n: number; running: number; quarantined: number };
    "/proc/boot"?: { booted: boolean; boot_seconds: number; phases: number; failed: unknown[]; slowest: { phase: string; ms: number } };
  };
};
type Gate = {
  mode: string;
  calls: number;
  denied: number;
  would_deny: number;
  overhead_us: { p50: number; p99: number };
  abi: { syscalls: number; by_class: Record<string, number> };
  policy: { sound: boolean; properties: { property: string; proven: boolean }[]; ms: number };
  coverage: { live: number; entries: number };
};
type Who = { status: { accounts: number; sessions: number; demo_login: boolean; breakglass_active: number; secret: string } };
type Devices = { devices: { verified: boolean; faults: number; failovers: number; state: string }[]; supervisor: { healthy: number; degraded: number } };
type Fleet = { simulated: boolean; counts: { total: number; streaming: number; degraded: number; offline: number; unassigned: number } };
type Prover = { healthy: boolean; failures: number; fail_closed: boolean };
type Consensus = { path: string; z3_status: string; safety_ok: boolean; certificate: { n: number; f: number; quorum: number; n_signatures: number } };
type Orchestrator = { invariant: string; attestation: { actions: number; integrity_ok: boolean } };
type Evidence = { detector: { id: string; promoted: boolean; roster: string[]; train_patients: number; judged_patients: number; shipped: { auroc: number; ece: number } } };

type Status = "ok" | "warn" | "err" | "idle";
type Component = {
  id: string;
  name: string;
  tag?: string;
  status: Status;
  lines: string[];
  about: string;
  facts: [string, string | number][];
  href?: string;
  hrefLabel?: string;
};
type Layer = { id: string; name: string; about: string; components: Component[] };

const DOT: Record<Status, string> = { ok: "#2fb67c", warn: "#eeb12a", err: "#e5484d", idle: "#5f666f" };
const pct = (x: number | undefined, d = 0) => (x == null ? "—" : `${(x * 100).toFixed(d)}%`);
const n = (x: number | undefined | null, d = 0) => (x == null ? "—" : Number(x).toFixed(d));

/* ------------------------------------------------------------------ */
/* The view                                                            */
/* ------------------------------------------------------------------ */
export function Architecture() {
  const proc = usePoll<ProcTree>("/aegis/proc", 5000);
  const gate = usePoll<Gate>("/aegis/sys/gate", 5000);
  const { caps, ready } = useCaps();
  const mayWho = caps.has("view:sessions");
  const who = usePoll<Who>(ready && mayWho ? "/aegis/sys/who" : null, 15000);
  const devices = usePoll<Devices>("/aegis/kernel/devices", 10000);
  const fleet = usePoll<Fleet>("/aegis/devices", 15000);
  const prover = usePoll<Prover>("/aegis/health/prover", 10000);
  const consensus = usePoll<Consensus>("/aegis/consensus", 20000);
  const orch = usePoll<Orchestrator>("/aegis/orchestrator", 20000);
  const evidence = usePoll<Evidence>("/aegis/lab/evidence");
  const health = useHealth();
  const { state } = usePaging();
  const [sel, setSel] = useState<string | null>(null);

  const layers: Layer[] = useMemo(() => {
    const t = proc.data?.tree ?? {};
    const st = state.stats;
    const open = activePages(state.pages);
    const recs = Object.values(state.recs).filter((r) => !r.acknowledged);
    const staff = Object.values(state.clinicians);
    const det = evidence.data?.detector;
    const g = gate.data;
    const sched = t["/proc/sched"];
    const ks = t["/proc/ksched"];
    const cage = t["/proc/cage"];
    const jr = t["/proc/journal"];
    const dv = devices.data?.devices ?? [];
    const verified = dv.filter((d) => d.verified).length;
    return [
      {
        id: "surfaces",
        name: "Surfaces",
        about: "Where people work. Every screen reads the system through the same gate.",
        components: [
          {
            id: "console",
            name: "Operator console",
            status: open.some((p) => p.status === "unassigned") ? "warn" : "ok",
            lines: [`${open.length} active · ${open.filter((p) => p.priority === 1).length} P1`, `${open.filter((p) => p.status === "review" || p.status === "unassigned").length} awaiting operator`],
            about: "The charge nurse's console: the building, the queue, the staff, the Sentinel and the paging policy.",
            facts: [["Active pages", open.length], ["Ringing", open.filter((p) => p.status === "paging").length], ["Awaiting operator", open.filter((p) => p.status === "review" || p.status === "unassigned").length]],
            href: "/operator/",
            hrefLabel: "Open the console",
          },
          {
            id: "pagers",
            name: "Pagers",
            status: "ok",
            lines: [`${staff.filter((c) => c.present).length} open on devices`, `${staff.filter((c) => c.pageable).length} pageable staff`],
            about: "Each clinician's pager. While one is open, pages for that clinician ring on it and the simulated responder stands down.",
            facts: [["Pageable staff", staff.filter((c) => c.pageable).length], ["Pagers open", staff.filter((c) => c.present).length], ["Being paged now", staff.filter((c) => c.status === "paging").length]],
            href: "/clinician/",
            hrefLabel: "Open the pager directory",
          },
          {
            id: "patients",
            name: "Patient records",
            status: "ok",
            lines: [`${st?.monitored_beds ?? "—"} monitored beds`, `${Object.values(state.monitor).filter((m) => m.watch).length} on sepsis watch`],
            about: "One patient at a time: the risk the unit acts on, vitals, the anatomical twin, organ systems, orders through the Z3 cage.",
            facts: [["Monitored beds", st?.monitored_beds ?? "—"], ["On sepsis watch", Object.values(state.monitor).filter((m) => m.watch).length]],
            href: "/patient/",
            hrefLabel: "Open patients",
          },
          {
            id: "evidence",
            name: "Evidence",
            status: det ? "ok" : "idle",
            lines: [det ? `AUROC ${n(det.shipped.auroc, 3)}` : "loading", det ? `${det.judged_patients.toLocaleString()} unseen patients` : ""],
            about: "The deployed detector read from its promotion manifest, with the operating point, abstention and the refusal record.",
            facts: det ? [["Detector", det.id], ["AUROC", n(det.shipped.auroc, 3)], ["ECE", n(det.shipped.ece, 4)], ["Trained on", det.train_patients.toLocaleString()], ["Judged on", det.judged_patients.toLocaleString()]] : [],
            href: "/lab/",
            hrefLabel: "Open evidence",
          },
        ],
      },
      {
        id: "gate",
        name: "Syscall gate",
        about: "Every request is a numbered syscall, authenticated and checked against a capability before its handler runs.",
        components: [
          {
            id: "gate",
            name: "Reference monitor",
            tag: g?.mode,
            status: g ? (g.denied > 0 ? "warn" : "ok") : "idle",
            lines: [g ? `${g.calls.toLocaleString()} calls · p99 ${n(g.overhead_us.p99)} µs` : "loading", g ? `${g.would_deny} would-deny · ${g.denied} denied` : ""],
            about: "One gate every call crosses. In audit mode it records what it would refuse; in enforce mode the refusals are real.",
            facts: g ? [["Mode", g.mode], ["Calls", g.calls], ["Denied", g.denied], ["Would deny", g.would_deny], ["Overhead p50", `${n(g.overhead_us.p50, 1)} µs`], ["Overhead p99", `${n(g.overhead_us.p99, 1)} µs`]] : [],
            href: "/os/?view=gate",
            hrefLabel: "Open the gate",
          },
          {
            id: "abi",
            name: "Syscall table",
            status: g ? (g.coverage.live === g.coverage.entries ? "ok" : "warn") : "idle",
            lines: [g ? `${g.abi.syscalls} syscalls` : "loading", g ? `${g.abi.by_class.read ?? 0} read · ${g.abi.by_class.write ?? 0} write · ${g.abi.by_class.egress ?? 0} egress` : ""],
            about: "Every HTTP route and socket is an entry with a number, a class and the capability it requires. A route without an entry fails the build.",
            facts: g ? [...Object.entries(g.abi.by_class).map(([k, v]) => [k, v] as [string, string | number]), ["Live routes covered", `${g.coverage.live}/${g.coverage.entries}`] as [string, string]] : [],
            href: "/os/?view=gate",
            hrefLabel: "Browse the table",
          },
          {
            id: "policy",
            name: "Access policy",
            tag: "Z3",
            status: g ? (g.policy.sound ? "ok" : "err") : "idle",
            lines: [g ? (g.policy.sound ? "proven sound" : "UNSOUND") : "loading", g ? `${g.policy.properties.filter((p) => p.proven).length}/${g.policy.properties.length} properties · ${n(g.policy.ms)} ms` : ""],
            about: "The role-to-capability policy is checked by Z3 against the whole syscall table: no escalation, guests reach no write, and so on.",
            facts: g ? g.policy.properties.map((p) => [p.property.replace(/_/g, " "), p.proven ? "proven" : "FAILED"] as [string, string]) : [],
          },
          {
            id: "sessions",
            name: "Sessions",
            tag: !mayWho ? "restricted" : undefined,
            status: who.data ? (who.data.status.secret.includes("random") ? "warn" : "ok") : "idle",
            lines: !mayWho
              ? ["requires view:sessions", "attending or chief"]
              : [who.data ? `${who.data.status.sessions} sessions · ${who.data.status.accounts} accounts` : "loading", who.data ? (who.data.status.secret.includes("random") ? "signing secret per boot" : "signing secret configured") : ""],
            about: "Signed, expiring sessions per user and privilege ring, with break-glass elevation that must be reviewed.",
            facts: who.data ? [["Accounts", who.data.status.accounts], ["Sessions", who.data.status.sessions], ["Demo sign-in", who.data.status.demo_login ? "on" : "off"], ["Break-glass active", who.data.status.breakglass_active], ["Signing secret", who.data.status.secret]] : [],
            href: "/os/?view=gate",
            hrefLabel: "See sessions",
          },
        ],
      },
      {
        id: "paging",
        name: "Paging",
        about: "Alerts from nurses and the sepsis detector, triaged, routed, certified and escalated.",
        components: [
          {
            id: "operator-agent",
            name: "Operator",
            status: st ? "ok" : "idle",
            lines: [st ? `${st.pages} pages · ${st.median_dispatch_ms ?? "—"} ms median` : "loading", st ? `${st.certificates} certified · ${st.blocked} held` : ""],
            about: "Runs each alert through triage, locate, route and certify, owns the escalation queue and the review fail-safe, and journals every step.",
            facts: st ? [["Pages since boot", st.pages], ["Median dispatch", `${st.median_dispatch_ms ?? "—"} ms`], ["Certified", st.certificates], ["Held by prover", st.blocked]] : [],
            href: "/operator/?view=pages",
            hrefLabel: "Open pages",
          },
          {
            id: "priority",
            name: "Priority Handler",
            status: "ok",
            lines: [`rules${t["/proc/llm"]?.available ? ` + ${t["/proc/llm"]!.primary_model}` : " only"}`, st ? `${st.by_source?.detector ?? 0} detector · ${(st.by_source?.nurse ?? 0) + (st.by_source?.voice ?? 0)} nurse` : ""],
            about: "P1–P4 from rules and red flags in milliseconds; a language model reviews afterwards and may only raise a priority.",
            facts: st ? Object.entries(st.by_priority).map(([k, v]) => [`P${k}`, v] as [string, number]) : [],
          },
          {
            id: "case",
            name: "Case Handler",
            status: st && (st.escalation_rate ?? 0) > 0.5 ? "warn" : "ok",
            lines: [st ? `${pct(st.escalation_rate ?? 0)} escalated` : "loading", st ? `${st.timeouts} timeouts · ${st.declines} declines` : ""],
            about: "Ranks every clinician with explicit points, then applies the guardrails as named steps.",
            facts: st ? [["Escalation rate", pct(st.escalation_rate ?? 0)], ["Timeouts", st.timeouts], ["Declines", st.declines]] : [],
            href: "/operator/?view=policy",
            hrefLabel: "Open the policy",
          },
          {
            id: "sentinel",
            name: "Sentinel",
            status: recs.some((r) => r.severity === "critical") ? "warn" : "ok",
            lines: [`${recs.length} active`, `${recs.filter((r) => r.severity === "critical").length} critical`],
            about: "Scans the whole paging system every 30 s for surges, coverage gaps, overload, escalation storms, sepsis clusters and bundle clocks. It never pages.",
            facts: recs.slice(0, 6).map((r) => [r.severity, r.title] as [string, string]),
            href: "/operator/?view=sentinel",
            hrefLabel: "Open the Sentinel",
          },
        ],
      },
      {
        id: "proof",
        name: "Decisions under proof",
        about: "No alert, protocol, order or note leaves without a certificate. A failed proof quarantines the output.",
        components: [
          {
            id: "cage",
            name: "Egress cage",
            tag: "Z3",
            status: cage ? (cage.quarantined > 0 ? "warn" : "ok") : "idle",
            lines: [cage ? `${cage.certified.toLocaleString()} certified · ${cage.blocked} blocked` : "loading", cage ? `pass rate ${pct(cage.pass_rate, 1)}` : ""],
            about: "The neurosymbolic cage at the edge of the kernel: every outbound alert, protocol, actuator command and automated note is proven first.",
            facts: cage ? [["Certified", cage.certified], ["Blocked", cage.blocked], ["Quarantined", cage.quarantined], ["Pass rate", pct(cage.pass_rate, 2)]] : [],
          },
          {
            id: "orchestrator",
            name: "Orchestrator",
            status: orch.data ? (orch.data.attestation.integrity_ok ? "ok" : "err") : "idle",
            lines: [orch.data ? `attestation ${orch.data.attestation.integrity_ok ? "intact" : "BROKEN"}` : "loading", orch.data ? `${orch.data.attestation.actions} governed actions` : ""],
            about: orch.data?.invariant ?? "One non-maleficence invariant over every layer, with a hash-chained attestation ledger.",
            facts: orch.data ? [["Attestation", orch.data.attestation.integrity_ok ? "intact" : "broken"], ["Governed actions", orch.data.attestation.actions]] : [],
          },
          {
            id: "consensus",
            name: "Consensus",
            tag: "BFT",
            status: consensus.data ? (consensus.data.safety_ok ? "ok" : "err") : "idle",
            lines: [consensus.data ? `quorum ${consensus.data.certificate.n_signatures}/${consensus.data.certificate.n}` : "loading", consensus.data ? `${consensus.data.path} · z3 ${consensus.data.z3_status}` : ""],
            about: "Specialist agents reach a Byzantine-fault-tolerant quorum on a plan (N = 3f + 1), with a physiologic-liveness fallback and a Z3 backstop.",
            facts: consensus.data ? [["Agents (N)", consensus.data.certificate.n], ["Faults tolerated (f)", consensus.data.certificate.f], ["Quorum", consensus.data.certificate.quorum], ["Signatures", consensus.data.certificate.n_signatures], ["Path", consensus.data.path], ["Z3", consensus.data.z3_status]] : [],
          },
          {
            id: "prover",
            name: "Prover",
            status: prover.data ? (prover.data.fail_closed ? "err" : prover.data.healthy ? "ok" : "warn") : "idle",
            lines: [prover.data ? (prover.data.fail_closed ? "FAIL-CLOSED" : prover.data.healthy ? "healthy" : "degraded") : "loading", prover.data ? `${prover.data.failures} failures` : ""],
            about: "A corrupted Z3 context cannot be repaired in-process; while unhealthy the cage fails closed and nothing is sent.",
            facts: prover.data ? [["Healthy", prover.data.healthy ? "yes" : "no"], ["Fail-closed", prover.data.fail_closed ? "yes" : "no"], ["Failures", prover.data.failures]] : [],
          },
        ],
      },
      {
        id: "runtime",
        name: "Clinical runtime",
        about: "Each bed's window becomes a calibrated risk, a body state and the evidence behind them.",
        components: [
          {
            id: "detector",
            name: "Sepsis detector",
            tag: det?.promoted ? "promoted" : undefined,
            status: det ? "ok" : "idle",
            lines: [det ? `AUROC ${n(det.shipped.auroc, 3)} · ECE ${n(det.shipped.ece, 4)}` : "loading", det ? `${det.roster.length}-model stacked ensemble` : ""],
            about: "A stacked ensemble trained on the patient label and calibrated without a staircase, scored on patients it never saw.",
            facts: det ? [["Artifact", det.id], ["Members", det.roster.join(", ")], ["AUROC", n(det.shipped.auroc, 3)], ["ECE", n(det.shipped.ece, 4)]] : [],
            href: "/lab/",
            hrefLabel: "Open evidence",
          },
          {
            id: "triage",
            name: "Live triage",
            status: t["/proc/triage"] ? (t["/proc/triage"]!.level === "CRITICAL" ? "warn" : "ok") : "idle",
            lines: [t["/proc/triage"] ? `${t["/proc/triage"]!.level} · fused ${n(t["/proc/triage"]!.fused_risk, 2)}` : "loading", t["/proc/triage"]?.note_certified ? "note certified" : ""],
            about: "The live monitored session's fused risk and triage level, as the kernel last certified them.",
            facts: t["/proc/triage"] ? [["Level", t["/proc/triage"]!.level], ["Fused risk", n(t["/proc/triage"]!.fused_risk, 3)], ["Note certified", t["/proc/triage"]!.note_certified ? "yes" : "no"]] : [],
            href: "/patient/",
            hrefLabel: "Open the patient",
          },
          {
            id: "resonance",
            name: "Resonance gate",
            status: t["/proc/resonance"] ? "ok" : "idle",
            lines: [t["/proc/resonance"]?.explain ?? "loading", t["/proc/resonance"] ? `resonance ${n(t["/proc/resonance"]!.resonance, 2)}` : ""],
            about: "Cross-modal agreement: a risk that only one pathway sees is held back as uncorroborated.",
            facts: t["/proc/resonance"] ? [["Resonance", n(t["/proc/resonance"]!.resonance, 3)], ["Gated risk", n(t["/proc/resonance"]!.gated_risk, 3)], ["Naive risk", n(t["/proc/resonance"]!.naive_risk, 3)], ["Alert certified", t["/proc/resonance"]!.alert_certified ? "yes" : "no"]] : [],
          },
          {
            id: "radar",
            name: "Early warning",
            status: t["/proc/radar"] ? (t["/proc/radar"]!.rising ? "warn" : "ok") : "idle",
            lines: [t["/proc/radar"] ? `EWS ${n(t["/proc/radar"]!.ews, 2)} · ${t["/proc/radar"]!.rising ? "rising" : "steady"}` : "loading", t["/proc/radar"] ? `${t["/proc/radar"]!.n} samples` : ""],
            about: "Critical-slowing-down indicators over the recent window, folded into scheduler priority.",
            facts: t["/proc/radar"] ? [["EWS", n(t["/proc/radar"]!.ews, 3)], ["Rising", t["/proc/radar"]!.rising ? "yes" : "no"], ["Warming", t["/proc/radar"]!.warming ? "yes" : "no"], ["Samples", t["/proc/radar"]!.n]] : [],
          },
          {
            id: "llm",
            name: "Language models",
            status: t["/proc/llm"] ? (t["/proc/llm"]!.available ? "ok" : "warn") : "idle",
            lines: [t["/proc/llm"]?.primary_model ?? "loading", t["/proc/llm"] ? `${t["/proc/llm"]!.budget_remaining} calls left · ${t["/proc/llm"]!.calls} used` : ""],
            about: "Multi-provider failover for triage review, written briefs and the grounded consult. Paging never waits on it.",
            facts: t["/proc/llm"] ? [["Preferred", t["/proc/llm"]!.prefer], ["Primary model", t["/proc/llm"]!.primary_model], ["Budget remaining", t["/proc/llm"]!.budget_remaining], ["Calls", t["/proc/llm"]!.calls]] : [],
          },
        ],
      },
      {
        id: "kernel",
        name: "Bio-kernel",
        about: "Verified drivers publish into a zero-copy memory fabric; a scheduler with Z3-proven invariants decides which beds get compute first.",
        components: [
          {
            id: "drivers",
            name: "Device drivers",
            tag: "eBPF",
            status: dv.length ? (verified === dv.length && dv.every((d) => d.state === "RUNNING") ? "ok" : "warn") : "idle",
            lines: [dv.length ? `${verified}/${dv.length} verified · ${dv.reduce((a, d) => a + d.faults, 0)} faults` : "loading", dv.length ? `${dv.reduce((a, d) => a + d.failovers, 0)} failovers` : ""],
            about: "/dev/bio drivers pass a static verifier (termination, memory safety, capabilities) before they run; a supervisor hot-swaps a faulting one.",
            facts: devices.data ? [["Drivers", dv.length], ["Verified", verified], ["Healthy slots", devices.data.supervisor.healthy], ["Degraded slots", devices.data.supervisor.degraded]] : [],
            href: "/os/?view=kernel",
            hrefLabel: "Open the kernel",
          },
          {
            id: "fabric",
            name: "Memory fabric",
            status: t["/proc/mem"] ? (t["/proc/mem"]!.pointer_stable ? "ok" : "warn") : "idle",
            lines: [t["/proc/mem"] ? `${pct(t["/proc/mem"]!.utilization, 1)} of ${(t["/proc/mem"]!.capacity_cells / 1e6).toFixed(2)}M cells` : "loading", t["/proc/mem"] ? `${(t["/proc/mem"]!.bytes / 1048576).toFixed(0)} MiB · pointer ${t["/proc/mem"]!.pointer_stable ? "stable" : "moved"}` : ""],
            about: "Per-signal ring buffers in one zero-copy arena; readers never copy a sample.",
            facts: t["/proc/mem"] ? [["Capacity (cells)", t["/proc/mem"]!.capacity_cells.toLocaleString()], ["Used (cells)", t["/proc/mem"]!.used_cells.toLocaleString()], ["Utilization", pct(t["/proc/mem"]!.utilization, 2)], ["Arena", `${(t["/proc/mem"]!.bytes / 1048576).toFixed(0)} MiB`]] : [],
          },
          {
            id: "scheduler",
            name: "Scheduler",
            tag: sched?.mode,
            status: sched ? "ok" : "idle",
            lines: [sched ? `owner ${sched.owner} · tick ${sched.tick}` : "loading", sched ? `${sched.lanes_used}/${sched.lane_budget} lanes · ${sched.tiers.CRITICAL ?? 0} critical` : ""],
            about: "A multi-tenant cognitive scheduler whose invariants are proven in Z3; early-warning foresight raises a bed's priority before it crashes.",
            facts: sched ? [["Mode", sched.mode], ["Owner", sched.owner], ["Beds", sched.procs], ["Critical", sched.tiers.CRITICAL ?? 0], ["Watch", sched.tiers.WATCH ?? 0], ["Stable", sched.tiers.STABLE ?? 0], ["Lanes", `${sched.lanes_used}/${sched.lane_budget}`], ["Interrupts", sched.interrupts]] : [],
            href: "/os/?view=kernel",
            hrefLabel: "Open the kernel",
          },
          {
            id: "procs",
            name: "Process table",
            status: ks ? "ok" : "idle",
            lines: [ks ? `${ks.processes} processes · ${ks.by_state.RUNNING ?? 0} running` : "loading", ks ? `${(ks.cpu_ms_total / 1000).toFixed(1)} s CPU · up ${fmtUptime(ks.uptime_s)}` : ""],
            about: "Kernel threads, services, per-bed monitors, drivers and sessions, with real CPU accounting.",
            facts: ks ? [...Object.entries(ks.by_state).map(([k, v]) => [k.toLowerCase(), v] as [string, string | number]), ["CPU total", `${(ks.cpu_ms_total / 1000).toFixed(2)} s`] as [string, string], ["Uptime", fmtUptime(ks.uptime_s)] as [string, string]] : [],
            href: "/os/?view=kernel",
            hrefLabel: "Open processes",
          },
          {
            id: "executor",
            name: "Executor",
            status: t["/proc/executor"] ? (t["/proc/executor"]!.violations > 0 ? "err" : "ok") : "idle",
            lines: [t["/proc/executor"] ? `${t["/proc/executor"]!.calls} calls · ${t["/proc/executor"]!.busy ? "busy" : "idle"}` : "loading", t["/proc/executor"] ? `max wait ${n(t["/proc/executor"]!.max_wait_ms, 1)} ms` : ""],
            about: "The single writer that owns the clinical arenas; every mutation is serialized through it.",
            facts: t["/proc/executor"] ? [["Calls", t["/proc/executor"]!.calls], ["Max wait", `${n(t["/proc/executor"]!.max_wait_ms, 2)} ms`], ["Max run", `${n(t["/proc/executor"]!.max_run_ms, 2)} ms`], ["Violations", t["/proc/executor"]!.violations]] : [],
          },
          {
            id: "services",
            name: "Services",
            tag: "aegisd",
            status: t["/proc/services"] ? (t["/proc/services"]!.quarantined > 0 ? "err" : "ok") : "idle",
            lines: [t["/proc/services"] ? `${t["/proc/services"]!.running}/${t["/proc/services"]!.n} running` : "loading", t["/proc/services"] ? `${t["/proc/services"]!.quarantined} quarantined` : ""],
            about: "Capability-confined clinical services under a supervisor that restarts or quarantines them.",
            facts: t["/proc/services"] ? [["Services", t["/proc/services"]!.n], ["Running", t["/proc/services"]!.running], ["Quarantined", t["/proc/services"]!.quarantined]] : [],
            href: "/os/?view=kernel",
            hrefLabel: "Open services",
          },
        ],
      },
      {
        id: "signals",
        name: "Signals",
        about: "Where the data comes from.",
        components: [
          {
            id: "live",
            name: "Live session",
            tag: health.source === "simulator" ? "simulated" : health.source ?? undefined,
            status: health.link === "online" ? "ok" : health.link === "offline" ? "err" : "warn",
            lines: [`source: ${health.source ?? "—"}`, "room 402 · wearable + camera path"],
            about: "The live monitored patient. With no camera or wearable connected, the simulator drives it and every screen says so.",
            facts: [["Source", health.source ?? "—"], ["Link", health.link]],
            href: "/patient/",
            hrefLabel: "Open the live patient",
          },
          {
            id: "replay",
            name: "Replayed records",
            status: "ok",
            lines: ["12 ICU beds · rooms 301–312", "PhysioNet / CinC 2019, de-identified"],
            about: "Twelve real de-identified ICU stays replayed hour by hour; the detector watches them and pages when they cross its line.",
            facts: [["Beds", 12], ["Rooms", "301–312"], ["Source", "PhysioNet/CinC 2019"]],
          },
          {
            id: "fleet",
            name: "Device fleet",
            tag: fleet.data?.simulated ? "simulated" : undefined,
            status: fleet.data ? (fleet.data.counts.offline > 0 ? "warn" : "ok") : "idle",
            lines: [fleet.data ? `${fleet.data.counts.streaming} streaming · ${fleet.data.counts.degraded} degraded` : "loading", fleet.data ? `${fleet.data.counts.offline} offline · ${fleet.data.counts.total} total` : ""],
            about: "Wearables, cameras and gateways: pairing, assignment to beds, calibration and signal quality.",
            facts: fleet.data ? Object.entries(fleet.data.counts).map(([k, v]) => [k, v] as [string, number]) : [],
          },
        ],
      },
      {
        id: "record",
        name: "Record",
        about: "Everything that happens is appended to a tamper-evident journal.",
        components: [
          {
            id: "journal",
            name: "Journal",
            status: jr ? (jr.chain_ok ? "ok" : "err") : "idle",
            lines: [jr ? `#${jr.seq.toLocaleString()} · head ${jr.head}` : "loading", jr ? `chain ${jr.chain_ok ? "verified" : "BROKEN"} · ${jr.verified_window} entries checked` : ""],
            about: "A hash-chained kernel log: each entry commits to the one before it, so an edited or deleted entry breaks the chain.",
            facts: jr ? [["Sequence", jr.seq], ["Head", jr.head], ["Chain", jr.chain_ok ? "verified" : "broken"], ["Verified window", jr.verified_window], ["Retained", jr.retained]] : [],
            href: "/os/?view=journal",
            hrefLabel: "Open the journal",
          },
          {
            id: "persistence",
            name: "Persistence",
            status: jr ? (jr.persisted.dropped > 0 ? "err" : jr.persisted.supabase_errors > 0 ? "warn" : "ok") : "idle",
            lines: [jr ? `${jr.persisted.sink} · ${jr.persisted.written.toLocaleString()} written` : "loading", jr ? `${jr.persisted.dropped} dropped · configured ${jr.persisted.configured}` : ""],
            about: "The journal is written in batches to Supabase, falling back to SQLite when the network is not there.",
            facts: jr ? [["Configured", jr.persisted.configured], ["Active sink", jr.persisted.sink], ["Written", jr.persisted.written], ["Dropped", jr.persisted.dropped], ["Supabase errors", jr.persisted.supabase_errors]] : [],
          },
          {
            id: "boot",
            name: "Boot",
            status: t["/proc/boot"] ? (t["/proc/boot"]!.failed.length ? "err" : "ok") : "idle",
            lines: [t["/proc/boot"] ? `${n(t["/proc/boot"]!.boot_seconds, 1)} s · ${t["/proc/boot"]!.phases} phases` : "loading", t["/proc/boot"] ? `slowest: ${t["/proc/boot"]!.slowest.phase} ${n(t["/proc/boot"]!.slowest.ms)} ms` : ""],
            about: "A measured boot: every phase timed and recorded, failures named.",
            facts: t["/proc/boot"] ? [["Boot time", `${n(t["/proc/boot"]!.boot_seconds, 2)} s`], ["Phases", t["/proc/boot"]!.phases], ["Failed", t["/proc/boot"]!.failed.length], ["Slowest", `${t["/proc/boot"]!.slowest.phase} (${n(t["/proc/boot"]!.slowest.ms)} ms)`]] : [],
            href: "/os/?view=kernel",
            hrefLabel: "Open the boot record",
          },
        ],
      },
    ];
  }, [proc.data, gate.data, who.data, mayWho, devices.data, fleet.data, prover.data, consensus.data, orch.data, evidence.data, health, state]);

  const all = layers.flatMap((l) => l.components.map((c) => ({ ...c, layer: l.name })));
  const selected = all.find((c) => c.id === sel) ?? null;
  const counts = { ok: all.filter((c) => c.status === "ok").length, warn: all.filter((c) => c.status === "warn").length, err: all.filter((c) => c.status === "err").length };
  const uptime = proc.data?.tree["/proc/ksched"]?.uptime_s;

  return (
    <div>
      <div className="mb-5 flex flex-wrap items-end gap-3">
        <div className="min-w-0 flex-1">
          <h1 className="text-[20px] font-semibold tracking-[-0.015em] text-ink">Architecture</h1>
          <p className="mt-1.5 max-w-3xl text-[13px] leading-relaxed text-muted">
            Every layer of AegisOS, live. Requests travel down through the syscall gate; telemetry travels up from the devices. Select any
            component to inspect it.
          </p>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <Tag tone="low" dot>
            {counts.ok} healthy
          </Tag>
          {counts.warn > 0 && (
            <Tag tone="med" dot>
              {counts.warn} attention
            </Tag>
          )}
          {counts.err > 0 && (
            <Tag tone="high" dot>
              {counts.err} failing
            </Tag>
          )}
          {uptime != null && <Tag>up {fmtUptime(uptime)}</Tag>}
        </div>
      </div>

      <div className="relative">
        {/* the bus: one spine every layer hangs off */}
        <span className="absolute bottom-6 left-[19px] top-6 hidden w-px overflow-hidden bg-line-bright md:block" aria-hidden>
          <span className="flow-down" />
          <span className="flow-up" />
        </span>
        <div className="space-y-2.5">
          {layers.map((l, i) => (
            <LayerBand key={l.id} layer={l} index={layers.length - i} selected={sel} onSelect={setSel} />
          ))}
        </div>
      </div>

      <Drawer open={!!selected} onClose={() => setSel(null)} width={440}>
        {selected && <Inspector c={selected} onClose={() => setSel(null)} />}
      </Drawer>
    </div>
  );
}

function fmtUptime(s: number): string {
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  const h = Math.floor(s / 3600);
  return `${h}h ${Math.floor((s % 3600) / 60)}m`;
}

function LayerBand({ layer, index, selected, onSelect }: { layer: Layer; index: number; selected: string | null; onSelect: (id: string) => void }) {
  const worst: Status = layer.components.some((c) => c.status === "err") ? "err" : layer.components.some((c) => c.status === "warn") ? "warn" : layer.components.every((c) => c.status === "idle") ? "idle" : "ok";
  return (
    <section className="relative grid gap-3 md:grid-cols-[220px_minmax(0,1fr)]">
      <div className="relative flex items-start gap-3 md:pr-2 md:pt-3">
        <span className="relative z-[1] mt-0.5 grid h-[22px] w-[40px] shrink-0 place-items-center rounded-[4px] border border-line-bright bg-bg font-mono text-[10.5px] text-dim">
          {String(index).padStart(2, "0")}
        </span>
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <span className="h-1.5 w-1.5 rounded-full" style={{ background: DOT[worst] }} />
            <h2 className="text-[13px] font-semibold text-ink">{layer.name}</h2>
          </div>
          <p className="mt-0.5 text-[11.5px] leading-snug text-dim">{layer.about}</p>
        </div>
      </div>
      <div className="grid gap-2 rounded-lg border border-line bg-s1 p-2 [grid-template-columns:repeat(auto-fill,minmax(210px,1fr))]">
        {layer.components.map((c) => (
          <button
            key={c.id}
            onClick={() => onSelect(c.id)}
            className={cx(
              "group min-w-0 rounded-md border px-3 py-2.5 text-left transition-colors",
              selected === c.id ? "border-accent/70 bg-accent-soft/40" : "border-line bg-s2/40 hover:border-line-bright hover:bg-s2",
            )}
          >
            <div className="flex items-center gap-2">
              <span className="h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: DOT[c.status] }} />
              <span className="min-w-0 truncate text-[12.5px] font-medium text-ink">{c.name}</span>
              {c.tag && <span className="ml-auto shrink-0 rounded-[3px] border border-line-bright px-1 font-mono text-[10px] text-dim">{c.tag}</span>}
            </div>
            <div className="mt-1.5 truncate font-mono text-[11.5px] text-muted">{c.lines[0]}</div>
            {c.lines[1] && <div className="truncate font-mono text-[11px] text-dim">{c.lines[1]}</div>}
          </button>
        ))}
      </div>
    </section>
  );
}

function Inspector({ c, onClose }: { c: Component & { layer: string }; onClose: () => void }) {
  return (
    <>
      <header className="flex shrink-0 items-start gap-3 border-b border-line px-4 py-3.5">
        <div className="min-w-0 flex-1">
          <div className="text-[11.5px] text-dim">{c.layer}</div>
          <div className="mt-0.5 flex items-center gap-2">
            <span className="h-2 w-2 rounded-full" style={{ background: DOT[c.status] }} />
            <h2 className="text-[16px] font-semibold text-ink">{c.name}</h2>
            {c.tag && <span className="rounded-[3px] border border-line-bright px-1 font-mono text-[10.5px] text-dim">{c.tag}</span>}
          </div>
        </div>
        <button onClick={onClose} className="grid h-7 w-7 place-items-center rounded-md text-dim transition-colors hover:bg-s3 hover:text-ink" aria-label="Close">
          <X size={14} />
        </button>
      </header>
      <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
        <p className="text-[13px] leading-relaxed text-muted">{c.about}</p>
        <div className="mt-4 rounded-md border border-line px-3 py-1.5">
          {c.facts.length === 0 ? (
            <div className="py-2 text-[12px] text-dim">Waiting for data…</div>
          ) : (
            c.facts.map(([k, v]) => <Kv key={k} k={k} v={<span className="font-mono text-[12px]">{String(v)}</span>} />)
          )}
        </div>
        <div className="mt-3 font-mono text-[11.5px] text-dim">{c.lines.filter(Boolean).join(" · ")}</div>
      </div>
      {c.href && (
        <footer className="shrink-0 border-t border-line px-4 py-3">
          <Link href={c.href} className="inline-flex h-8 items-center gap-1.5 rounded-md bg-ink px-3 text-[12.5px] font-medium text-bg transition-colors hover:bg-white">
            {c.hrefLabel ?? "Open"}
            <ArrowUpRight size={13} />
          </Link>
        </footer>
      )}
    </>
  );
}
