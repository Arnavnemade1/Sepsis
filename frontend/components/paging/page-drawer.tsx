"use client";

import Link from "next/link";
import { useState, type ReactNode } from "react";

import { Button, cx, Drawer } from "@/components/ui";
import {
  CATEGORY_LABEL,
  clock,
  PAGE_STATUS_LABEL,
  since,
  type Clinician,
  type Page,
} from "@/lib/paging";

import { Avatar, Countdown, P_RGB, PriorityChip, SourceTag, StatusText } from "./bits";
import { pageAction } from "./queue";

export function PageDrawer({ page, clinicians, invariants, now, canDispatch, onClose }: {
  page: Page | null;
  clinicians: Record<string, Clinician>;
  invariants: { id: string; text: string }[];
  now: number;
  canDispatch: boolean;
  onClose: () => void;
}) {
  return (
    <Drawer open={!!page} onClose={onClose} width={560}>
      {page && <Body page={page} clinicians={clinicians} invariants={invariants} now={now} canDispatch={canDispatch} onClose={onClose} />}
    </Drawer>
  );
}

function Section({ title, right, children }: { title: string; right?: ReactNode; children: ReactNode }) {
  return (
    <section className="border-t border-line-soft px-5 py-4">
      <div className="mb-2.5 flex items-center gap-2">
        <h3 className="font-mono text-[10.5px] font-medium uppercase tracking-[0.12em] text-dim">{title}</h3>
        {right && <div className="ml-auto">{right}</div>}
      </div>
      {children}
    </section>
  );
}

function Body({ page, clinicians, invariants, now, canDispatch, onClose }: {
  page: Page;
  clinicians: Record<string, Clinician>;
  invariants: { id: string; text: string }[];
  now: number;
  canDispatch: boolean;
  onClose: () => void;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [reassign, setReassign] = useState<string>("");
  const who = page.assignee ?? page.planned;
  const c = who ? clinicians[who] : null;
  const att = page.attempts[page.attempts.length - 1];
  const cert = page.certificate;
  const act = async (a: "approve" | "escalate" | "cancel" | "resolve", body: Record<string, unknown> = {}) => {
    setBusy(a);
    setErr(null);
    try {
      const r = await pageAction(page.id, a, body);
      if (r.status !== "ok") setErr(r.detail ?? `the page was ${r.status}`);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  const steps: { label: string; agent: string; value: ReactNode; ms?: number; state: "done" | "active" | "fail" | "wait" }[] = [
    { label: "Triage", agent: "Priority Handler", value: <>P{page.triage.rules_priority ?? page.priority}{page.triage.rules_priority ? ` → P${page.priority}` : ""} · {CATEGORY_LABEL[page.category]}</>, ms: page.timings.triage_ms, state: "done" },
    { label: "Locate", agent: "Operator", value: page.room ? `Room ${page.room} · ${page.zone_name}` : "room not identified", ms: page.timings.locate_ms, state: page.room ? "done" : "fail" },
    { label: "Route", agent: "Case Handler", value: page.decision.assignee ? clinicians[page.decision.assignee]?.name ?? page.decision.assignee : "no eligible clinician", ms: page.timings.route_ms, state: page.decision.assignee ? "done" : "fail" },
    { label: "Verify", agent: "Z3", value: cert ? (cert.certified ? "all invariants hold (unsat)" : `refused: ${cert.violations.join(", ")}`) : page.status === "review" ? "waits for review" : "—", ms: page.timings.certify_ms, state: cert ? (cert.certified ? "done" : "fail") : "wait" },
    { label: "Page", agent: "Queue", value: PAGE_STATUS_LABEL[page.status], state: page.status === "accepted" || page.status === "resolved" ? "done" : page.status === "paging" ? "active" : page.status === "cancelled" ? "wait" : "wait" },
  ];

  return (
    <>
      <header className="flex items-start gap-3 px-5 pb-4 pt-5">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <PriorityChip p={page.priority} size="lg" label />
            <span className="font-mono text-[12px] text-dim">{page.id}</span>
            <SourceTag source={page.source} synthetic={page.synthetic} />
          </div>
          <h2 className="mt-3 text-[19px] font-semibold leading-snug tracking-tight text-ink">
            {page.room ? `Room ${page.room}` : "Unknown room"}
            <span className="font-normal text-dim"> · {page.zone_name ?? "—"}</span>
          </h2>
          {page.patient?.name && (
            <div className="mt-0.5 text-[13px] text-muted">
              {page.patient.name}
              {page.patient.age ? `, ${page.patient.age}${page.patient.sex ?? ""}` : ""}
              {page.patient.synthetic ? " · synthetic demo record" : page.patient.monitored ? " · monitored, real replayed record" : ""}
            </div>
          )}
        </div>
        <button onClick={onClose} className="grid h-8 w-8 place-items-center rounded-lg text-dim hover:bg-s2 hover:text-ink" aria-label="Close">
          <svg width="14" height="14" viewBox="0 0 14 14"><path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>
        </button>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="px-5 pb-4">
          <blockquote className="rounded-xl border border-line bg-s2/60 px-4 py-3 text-[14px] leading-relaxed text-ink">
            “{page.text}”
            <div className="mt-1.5 text-[11.5px] text-dim">
              {page.reporter ?? page.source} · {since(page.created_at, now)}
            </div>
          </blockquote>
        </div>

        <Section title="Dispatch" right={page.timings.total_ms != null && <span className="num font-mono text-[11px] text-dim">{page.timings.total_ms.toFixed(1)} ms end to end</span>}>
          <ol className="relative space-y-3 pl-6">
            <span className="absolute bottom-2 left-[7px] top-2 w-px bg-line" />
            {steps.map((s) => (
              <li key={s.label} className="relative">
                <span
                  className={cx(
                    "absolute -left-6 top-0.5 grid h-[15px] w-[15px] place-items-center rounded-full border-2 bg-s1",
                    s.state === "done" && "border-low bg-low",
                    s.state === "active" && "border-st-paging",
                    s.state === "fail" && "border-high bg-high",
                    s.state === "wait" && "border-line-bright",
                  )}
                >
                  {s.state === "done" && <svg width="8" height="8" viewBox="0 0 8 8"><path d="M1.5 4.2 3.2 6 6.5 2" stroke="#fff" strokeWidth="1.5" fill="none" strokeLinecap="round" /></svg>}
                  {s.state === "active" && <span className="blink-soft h-1.5 w-1.5 rounded-full bg-st-paging" />}
                </span>
                <div className="flex items-baseline gap-2">
                  <span className="text-[13px] font-medium text-ink">{s.label}</span>
                  <span className="text-[11px] text-dim">{s.agent}</span>
                  {s.ms != null && <span className="num ml-auto font-mono text-[11px] text-dim">{s.ms < 1 ? "<1" : s.ms.toFixed(1)} ms</span>}
                </div>
                <div className="text-[12.5px] text-muted">{s.value}</div>
              </li>
            ))}
          </ol>
        </Section>

        <Section title="Triage" right={<span className="text-[11px] text-dim">{page.triage.source === "detector" ? "sepsis detector" : page.triage.source === "llm+rules" ? `rules + ${page.triage.model ?? "model"}` : "rules"}</span>}>
          {page.triage.red_flags.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-1.5">
              {page.triage.red_flags.map((f) => (
                <span key={f} className="rounded-md px-2 py-0.5 text-[11.5px] font-medium" style={{ background: `rgb(${P_RGB[1]} / .09)`, color: "#c32a2f" }}>
                  {f}
                </span>
              ))}
            </div>
          )}
          <p className="text-[12.5px] leading-relaxed text-muted">{page.triage.rationale}</p>
          {Object.keys(page.triage.vitals).length > 0 && (
            <div className="mt-2 flex flex-wrap gap-x-4 gap-y-1 font-mono text-[11.5px] text-muted">
              {Object.entries(page.triage.vitals).map(([k, v]) => (
                <span key={k}>
                  <span className="text-dim">{k.replace("_", " ")}</span> {typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(1)) : v}
                </span>
              ))}
            </div>
          )}
          {page.refining && <div className="shimmer mt-3 rounded-lg px-3 py-2 text-[12px] text-accent-ink">The model is reviewing this page. It may raise the priority; it cannot lower it.</div>}
          {page.refinement && (
            <div className={cx("mt-3 rounded-lg border px-3 py-2 text-[12px] leading-snug", page.refinement.outcome === "raised" ? "border-dys/30 bg-dys/5 text-dys-ink" : "border-line bg-s2/60 text-muted")}>
              <span className="font-medium">
                {page.refinement.outcome === "raised"
                  ? `Model raised P${page.refinement.from} → P${page.refinement.to}`
                  : page.refinement.outcome === "agrees"
                    ? "Model agrees"
                    : page.refinement.outcome === "kept"
                      ? "Model suggested lower; kept"
                      : "Model unavailable; rules stand"}
              </span>
              {page.refinement.note ? ` · ${page.refinement.note}` : ""}
            </div>
          )}
        </Section>

        <Section title="Routing" right={<span className="text-[11px] text-dim">Case Handler</span>}>
          {c ? (
            <div className="flex items-center gap-3 rounded-xl border border-line p-3">
              <Avatar name={c.name} status={c.status} size={36} present={c.present} />
              <div className="min-w-0 flex-1">
                <div className="text-[13.5px] font-medium text-ink">{c.name}</div>
                <div className="text-[12px] text-dim">
                  {c.specialty_label} · {c.title}
                </div>
                <StatusText status={c.status} />
              </div>
              {page.status === "paging" && att && <Countdown remaining={att.deadline - now} total={att.deadline - att.sent_at} priority={page.priority} size={40} />}
            </div>
          ) : (
            <div className="rounded-xl border border-dashed border-line-bright p-3 text-[12.5px] text-dys-ink">{page.review_reason ?? "No clinician assigned."}</div>
          )}
          <p className="mt-2 text-[12.5px] leading-relaxed text-muted">{page.decision.reasoning}</p>
          {page.decision.guardrails.length > 0 && (
            <ul className="mt-2 space-y-1">
              {page.decision.guardrails.map((g) => (
                <li key={g} className="flex gap-2 text-[12px] text-accent-ink">
                  <span className="mt-[5px] h-1.5 w-1.5 shrink-0 rounded-full bg-accent" />
                  {g}
                </li>
              ))}
            </ul>
          )}
          <div className="mt-3 space-y-1.5">
            {page.decision.candidates.slice(0, 6).map((cand, i) => {
              const max = Math.max(1, ...page.decision.candidates.map((x) => x.score));
              return (
                <div key={cand.id} className={cx("rounded-lg px-2.5 py-2", i === 0 && !cand.excluded ? "bg-accent-soft/70" : "bg-s2/50")}>
                  <div className="flex items-center gap-2">
                    <span className={cx("text-[12.5px] font-medium", cand.excluded ? "text-dim line-through decoration-faint" : "text-ink")}>{cand.name}</span>
                    <span className="text-[11px] text-dim">{cand.specialty_label}</span>
                    <span className="num ml-auto font-mono text-[11.5px] text-muted">{cand.excluded ? "—" : cand.score.toFixed(0)}</span>
                  </div>
                  {cand.excluded ? (
                    <div className="mt-0.5 text-[11.5px] text-dys-ink">Excluded: {cand.excluded}</div>
                  ) : (
                    <>
                      <div className="mt-1 h-1 overflow-hidden rounded-full bg-s3">
                        <div className="h-full rounded-full bg-accent/70" style={{ width: `${Math.max(4, (cand.score / max) * 100)}%` }} />
                      </div>
                      <div className="mt-1 flex flex-wrap gap-x-2 gap-y-0.5 text-[11px]">
                        {cand.factors.map((f) => (
                          <span key={f.label} className={f.points >= 0 ? "text-muted" : "text-dys-ink"}>
                            {f.points >= 0 ? "+" : ""}
                            {f.points} {f.label}
                          </span>
                        ))}
                      </div>
                    </>
                  )}
                </div>
              );
            })}
          </div>
        </Section>

        <Section
          title="Certificate"
          right={
            cert ? (
              <span className={cx("inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[11px] font-medium", cert.certified ? "bg-low/10 text-low-ink" : "bg-high/10 text-high-ink")}>
                {cert.certified ? "Proven · Z3 unsat" : `Refused · Z3 ${cert.z3_status}`}
              </span>
            ) : (
              <span className="text-[11px] text-dim">issued when the page is sent</span>
            )
          }
        >
          <ul className="space-y-1.5">
            {invariants.map((inv) => {
              const bad = cert?.violations.includes(inv.id) || page.blocked.includes(inv.id);
              return (
                <li key={inv.id} className="flex gap-2 text-[12px] leading-snug">
                  <span className={cx("mt-0.5 grid h-4 w-4 shrink-0 place-items-center rounded-full text-[9px] font-bold", !cert ? "bg-s3 text-dim" : bad ? "bg-high text-white" : "bg-low/15 text-low-ink")}>
                    {!cert ? "·" : bad ? "✕" : "✓"}
                  </span>
                  <span className={bad ? "text-high-ink" : "text-muted"}>{inv.text}</span>
                </li>
              );
            })}
          </ul>
          {cert && (
            <p className="mt-2 break-words font-mono text-[10.5px] leading-relaxed text-dim">
              {cert.proof} · {cert.ms} ms
            </p>
          )}
        </Section>

        {(page.attempts.length > 0 || page.escalation_log.length > 0) && (
          <Section title="Timeline">
            <ol className="space-y-2">
              <li className="flex gap-2 text-[12px]">
                <span className="num w-16 shrink-0 font-mono text-dim">{clock(page.created_at)}</span>
                <span className="text-muted">Alert raised by {page.reporter ?? page.source}</span>
              </li>
              {page.attempts.map((a, i) => (
                <li key={i} className="flex gap-2 text-[12px]">
                  <span className="num w-16 shrink-0 font-mono text-dim">{clock(a.sent_at)}</span>
                  <span className="text-muted">
                    Paged <span className="font-medium text-ink">{a.name}</span>
                    {a.outcome ? (
                      <>
                        {" "}
                        →{" "}
                        <span className={a.outcome === "accepted" ? "text-low-ink" : "text-dys-ink"}>
                          {a.outcome === "timeout" ? "no answer" : a.outcome}
                        </span>
                        {a.at ? ` after ${Math.round(a.at - a.sent_at)}s` : ""}
                        {a.reason ? ` (${a.reason})` : ""}
                      </>
                    ) : (
                      <span className="blink-soft text-accent-ink"> · ringing</span>
                    )}
                  </span>
                </li>
              ))}
            </ol>
          </Section>
        )}

        {page.co_notified.length > 0 && (
          <Section title="Also notified">
            {page.co_notified.map((n) => (
              <div key={n.clinician} className="flex items-center gap-2.5 text-[12.5px]">
                <Avatar name={n.name} size={22} />
                <span className="font-medium text-ink">{n.name}</span>
                <span className="text-dim">{n.why}</span>
              </div>
            ))}
          </Section>
        )}

        {page.sbar && (
          <Section title="Brief (SBAR)" right={<span className="text-[11px] text-dim">{page.sbar_refining ? "model writing…" : page.sbar.source === "llm" ? page.sbar.model ?? "model" : "template"} · {page.sbar.words} words</span>}>
            <dl className="space-y-2">
              {(["situation", "background", "assessment", "request"] as const).map((k) => (
                <div key={k} className="grid grid-cols-[22px_1fr] gap-2">
                  <dt className="grid h-5 w-5 place-items-center rounded bg-ink font-mono text-[10.5px] font-semibold text-white">{k[0].toUpperCase()}</dt>
                  <dd className="text-[12.5px] leading-relaxed text-ink">{page.sbar![k]}</dd>
                </div>
              ))}
            </dl>
          </Section>
        )}

        {page.bed && (
          <Section title="Patient">
            <Link href={`/patient/?id=${encodeURIComponent(page.bed)}`} className="flex items-center justify-between rounded-xl border border-line px-3 py-2.5 text-[13px] text-ink transition hover:border-line-bright hover:bg-s2">
              <span>
                Open {page.patient?.name ?? "the patient"}: twin, risk trend, orders
                {page.patient?.risk != null && <span className="ml-2 font-mono text-[11.5px] text-dim">risk {page.patient.risk.toFixed(2)}</span>}
              </span>
              <span aria-hidden>→</span>
            </Link>
          </Section>
        )}
      </div>

      {canDispatch && !["resolved", "cancelled"].includes(page.status) && (
        <footer className="border-t border-line bg-s2/60 px-5 py-3">
          {err && <div className="mb-2 text-[12px] text-high-ink">{err}</div>}
          <div className="flex flex-wrap items-center gap-2">
            {["review", "held", "unassigned"].includes(page.status) && (
              <>
                <select
                  value={reassign}
                  onChange={(e) => setReassign(e.target.value)}
                  className="h-8 min-w-0 flex-1 rounded-lg border border-line bg-s1 px-2 text-[12.5px] text-ink"
                  aria-label="Page someone else"
                >
                  <option value="">{page.planned ? `Planned: ${clinicians[page.planned]?.name ?? page.planned}` : "Best available"}</option>
                  {Object.values(clinicians)
                    .filter((x) => x.pageable && x.status !== "off_shift")
                    .map((x) => (
                      <option key={x.id} value={x.id}>
                        {x.name} · {x.specialty_label}
                      </option>
                    ))}
                </select>
                <Button variant="primary" disabled={!!busy} onClick={() => act("approve", reassign ? { clinician: reassign } : {})}>
                  {busy === "approve" ? "Paging…" : "Approve & page"}
                </Button>
              </>
            )}
            {page.status === "paging" && (
              <Button variant="ghost" disabled={!!busy} onClick={() => act("escalate")}>
                {busy === "escalate" ? "Escalating…" : "Escalate now"}
              </Button>
            )}
            {page.status === "accepted" && (
              <Button variant="ghost" disabled={!!busy} onClick={() => act("resolve")}>
                Mark resolved
              </Button>
            )}
            <Button variant="danger" disabled={!!busy} onClick={() => act("cancel")}>
              Cancel page
            </Button>
          </div>
        </footer>
      )}
    </>
  );
}
