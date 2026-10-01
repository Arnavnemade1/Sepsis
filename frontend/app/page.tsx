"use client";

import { motion } from "motion/react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { useMemo, useState } from "react";

import { Avatar, PriorityChip } from "@/components/paging/bits";
import { Mark } from "@/components/shell";
import { Button, cx } from "@/components/ui";
import { api, writeSession, type Credential } from "@/lib/api";
import { homeFor } from "@/lib/ecosystem";
import { usePoll } from "@/lib/hooks";
import { STATUS_LABEL, usePaging } from "@/lib/paging";

type LoginOut = { status: string; token: string; credential: Credential; detail?: string };

const STEPS = [
  { k: "Alert", d: "A nurse types or says it in plain language, or the sepsis detector crosses its alarm line on a monitored bed." },
  { k: "Triage", d: "Rules catch the red flags in milliseconds. A language model reviews afterwards and may only raise the priority." },
  { k: "Route", d: "Specialty, unit, on-call, caseload and procedures weighed openly, then guardrails: no double-paging, no overload." },
  { k: "Verify", d: "Z3 proves the page satisfies the hospital's paging policy before it leaves. A violation holds it for the operator." },
  { k: "Escalate", d: "No answer or a decline, and the next-best clinician is paged. Accept, and an SBAR brief is waiting." },
];

const AGENTS = [
  { n: "Operator", d: "Runs the pipeline and the escalation queue; every step timed, every decision journalled." },
  { n: "Priority Handler", d: "P1–P4 from rules and red flags, refined by a model that can never lower a sent page." },
  { n: "Case Handler", d: "Ranks the staff with every factor visible, then applies the guardrails as explicit steps." },
  { n: "Sentinel", d: "Watches the whole system every 30 s for surges, coverage holes, overload and sepsis clusters." },
];

export default function Landing() {
  const router = useRouter();
  const { state, status } = usePaging();
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const health = usePoll<{ status: string }>("/aegis/health", 30_000);

  const signIn = async (username: string) => {
    setBusy(username);
    setErr(null);
    try {
      const r = await api<LoginOut>("/aegis/sys/login", { body: { username }, timeoutMs: 60_000 });
      if (r.status !== "ok") throw new Error(r.detail ?? "sign-in refused");
      writeSession({ token: r.token, credential: r.credential });
      router.push(homeFor(r.credential));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      setBusy(null);
    }
  };

  const staff = useMemo(() => Object.values(state.clinicians).filter((c) => c.pageable).sort((a, b) => a.specialty_label.localeCompare(b.specialty_label) || a.name.localeCompare(b.name)), [state.clinicians]);
  const openPages = Object.values(state.pages).filter((p) => ["review", "paging", "accepted", "held", "unassigned"].includes(p.status));
  const stats = state.stats;

  return (
    <div className="min-h-screen bg-[radial-gradient(90%_60%_at_50%_-10%,#ffffff_0%,#f6f6f3_60%)]">
      <header className="mx-auto flex max-w-[1180px] items-center gap-3 px-5 py-5">
        <Mark size={30} />
        <span className="text-[16px] font-semibold tracking-tight">AegisOS</span>
        <span className="ml-auto flex items-center gap-2 text-[12px] text-dim">
          <span className={cx("h-1.5 w-1.5 rounded-full", health.data ? "bg-low" : "bg-med")} />
          {health.data ? "System live" : "Waking the backend…"}
        </span>
      </header>

      <main className="mx-auto max-w-[1180px] px-5">
        <section className="grid items-center gap-10 pb-16 pt-10 lg:grid-cols-[1.15fr_1fr] lg:pt-16">
          <div>
            <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}>
              <div className="inline-flex items-center gap-2 rounded-full border border-line bg-s1 px-3 py-1 text-[12px] text-muted shadow-[var(--shadow-card)]">
                <span className="rounded-full bg-accent-soft px-1.5 font-medium text-accent-ink">New</span>
                The sepsis detector now pages the right clinician itself
              </div>
              <h1 className="serif mt-6 text-[56px] leading-[0.98] tracking-[-0.02em] text-ink lg:text-[76px]">
                Every alert reaches the right clinician. <span className="italic text-dim">In seconds.</span>
              </h1>
              <p className="mt-6 max-w-xl text-[17px] leading-relaxed text-muted">
                AegisOS is a paging operating system for hospital floors. Nurses raise alerts in plain language, a sepsis
                detector watches every monitored bed, and each page is triaged, routed, formally verified and escalated
                automatically, then briefed on arrival.
              </p>
              <div className="mt-8 flex flex-wrap items-center gap-3">
                <Button variant="primary" size="lg" disabled={!!busy} onClick={() => signIn("rn.patel")}>
                  {busy === "rn.patel" ? "Opening…" : "Open the operator console"}
                </Button>
                <Button variant="ghost" size="lg" onClick={() => router.push("/clinician/?id=dr.alvarez")}>
                  Carry a pager
                </Button>
              </div>
              {err && <p className="mt-3 text-[13px] text-high-ink">{err}</p>}
              <p className="mt-4 text-[12px] text-dim">Research and educational use only · not a medical device.</p>
            </motion.div>
          </div>

          <motion.div initial={{ opacity: 0, y: 14 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.7, delay: 0.1, ease: [0.16, 1, 0.3, 1] }} className="plate relative overflow-hidden p-5">
            <div className="flex items-center gap-2">
              <span className="font-mono text-[10.5px] uppercase tracking-[0.12em] text-dim">Right now at {state.ready ? "Aegis Memorial" : "the hospital"}</span>
              <span className="ml-auto inline-flex items-center gap-1.5 text-[11px] text-low-ink">
                <span className="live-dot h-1.5 w-1.5 rounded-full bg-low" style={{ color: "rgb(43 154 102 / .5)" }} />
                {status === "live" ? "live" : "connecting"}
              </span>
            </div>
            <div className="mt-4 grid grid-cols-3 gap-4">
              {[
                ["Open pages", openPages.length],
                ["Beds monitored", stats?.monitored_beds ?? "—"],
                ["Median dispatch", stats?.median_dispatch_ms != null ? `${stats.median_dispatch_ms} ms` : "—"],
              ].map(([k, v]) => (
                <div key={String(k)}>
                  <div className="text-[11px] text-dim">{k}</div>
                  <div className="num text-[26px] font-semibold tracking-tight text-ink">{v}</div>
                </div>
              ))}
            </div>
            <div className="mt-5 space-y-2">
              {openPages
                .sort((a, b) => a.priority - b.priority)
                .slice(0, 4)
                .map((p) => (
                  <div key={p.id} className="flex items-center gap-2.5 rounded-xl border border-line bg-s1 px-3 py-2.5">
                    <PriorityChip p={p.priority} size="sm" />
                    <span className="font-mono text-[11.5px] font-medium text-ink">{p.room}</span>
                    <span className="min-w-0 flex-1 truncate text-[12.5px] text-muted">{p.text}</span>
                    <span className="shrink-0 text-[11px] text-dim">{p.status === "paging" ? "ringing" : p.status}</span>
                  </div>
                ))}
              {state.ready && openPages.length === 0 && <div className="rounded-xl border border-dashed border-line-bright px-3 py-6 text-center text-[12.5px] text-dim">The queue is clear.</div>}
              {!state.ready && [0, 1, 2].map((i) => <div key={i} className="h-11 animate-pulse rounded-xl bg-s2" />)}
            </div>
          </motion.div>
        </section>

        <section className="border-t border-line py-14">
          <div className="font-mono text-[10.5px] uppercase tracking-[0.12em] text-dim">How a page moves</div>
          <div className="mt-6 grid gap-6 md:grid-cols-5">
            {STEPS.map((s, i) => (
              <motion.div key={s.k} initial={{ opacity: 0, y: 8 }} whileInView={{ opacity: 1, y: 0 }} viewport={{ once: true }} transition={{ delay: i * 0.06, duration: 0.5 }}>
                <div className="flex items-center gap-2">
                  <span className="grid h-6 w-6 place-items-center rounded-full bg-ink font-mono text-[11px] font-semibold text-white">{i + 1}</span>
                  <span className="text-[15px] font-semibold text-ink">{s.k}</span>
                </div>
                <p className="mt-2 text-[13px] leading-relaxed text-muted">{s.d}</p>
              </motion.div>
            ))}
          </div>
        </section>

        <section className="grid gap-10 border-t border-line py-14 lg:grid-cols-2">
          <div>
            <h2 className="serif text-[34px] leading-tight text-ink">Run the floor</h2>
            <p className="mt-2 text-[14px] text-muted">The charge nurse's console: the building, the queue, the staff and the policy.</p>
            <div className="mt-6 space-y-2.5">
              {[
                ["rn.patel", "Priya Patel, RN", "Charge nurse · runs the console"],
                ["dr.whitfield", "Dr. James Whitfield", "Chief of critical care · can change the paging policy"],
                ["guest", "Guest", "View only · change nothing"],
              ].map(([u, n, t]) => (
                <button key={u} disabled={!!busy} onClick={() => signIn(u)} className="plate flex w-full items-center gap-3 p-3.5 text-left transition hover:-translate-y-px hover:border-line-bright disabled:opacity-60">
                  <Avatar name={n} size={36} />
                  <span className="min-w-0 flex-1">
                    <span className="block text-[14px] font-medium text-ink">{n}</span>
                    <span className="block text-[12px] text-dim">{t}</span>
                  </span>
                  <span className="text-[12.5px] text-muted">{busy === u ? "Opening…" : "Operator console →"}</span>
                </button>
              ))}
            </div>
          </div>
          <div>
            <h2 className="serif text-[34px] leading-tight text-ink">Carry a pager</h2>
            <p className="mt-2 text-[14px] text-muted">Open any clinician's device. While it is open, pages for them ring here and nobody answers on their behalf.</p>
            <div className="mt-6 grid max-h-[340px] gap-2 overflow-y-auto pb-6 pr-1 [mask-image:linear-gradient(to_bottom,#000_82%,transparent)] sm:grid-cols-2">
              {staff.length === 0 && [0, 1, 2, 3].map((i) => <div key={i} className="h-14 animate-pulse rounded-xl bg-s2" />)}
              {staff.map((c) => (
                <Link key={c.id} href={`/clinician/?id=${c.id}`} className="plate flex items-center gap-3 p-3 transition hover:-translate-y-px hover:border-line-bright">
                  <Avatar name={c.name} status={c.status} size={32} present={c.present} />
                  <span className="min-w-0">
                    <span className="block truncate text-[13px] font-medium text-ink">{c.name}</span>
                    <span className="block truncate text-[11.5px] text-dim">{c.specialty_label} · {STATUS_LABEL[c.status]}</span>
                  </span>
                </Link>
              ))}
            </div>
          </div>
        </section>

        <section className="border-t border-line py-14">
          <div className="font-mono text-[10.5px] uppercase tracking-[0.12em] text-dim">Four agents, one queue</div>
          <div className="mt-6 grid gap-4 md:grid-cols-4">
            {AGENTS.map((a) => (
              <div key={a.n} className="plate p-4">
                <div className="text-[14px] font-semibold text-ink">{a.n}</div>
                <p className="mt-1.5 text-[12.5px] leading-relaxed text-muted">{a.d}</p>
              </div>
            ))}
          </div>
        </section>

        <footer className="border-t border-line py-10 text-[12px] leading-relaxed text-dim">
          AegisOS is a research system, not a medical device, and must not be used for clinical decisions. Aegis Memorial is a demo hospital:
          its twelve ICU beds replay real de-identified PhysioNet/CinC 2019 stays and room 402 is the live monitored session; every other patient
          and the nurse traffic are synthetic and labelled so. Clinicians with no pager open answer by simulation.
        </footer>
      </main>
    </div>
  );
}
