"use client";

import Link from "next/link";
import { useState, type ReactNode } from "react";

import { OpenLink, ShieldCheck } from "@/components/icons";
import { Button, CloseIcon, cx, Drawer, IconButton, Tag } from "@/components/ui";
import {
  CATEGORY_LABEL,
  clock,
  PAGE_STATUS_LABEL,
  type Clinician,
  type Page,
} from "@/lib/paging";

import { SEV_RGB } from "@/lib/rooms";

import { Avatar, Countdown, P_RGB, PriorityChip, SourceTag, StatusText, TimerBar } from "./bits";
import { age, pageAction, statusLine } from "./queue";

export function PageDrawer({ page, clinicians, invariants, now, canDispatch, onClose }: {
  page: Page | null;
  clinicians: Record<string, Clinician>;
  invariants: { id: string; text: string }[];
  now: number;
  canDispatch: boolean;
  onClose: () => void;
}) {
  return (
    <Drawer open={!!page} onClose={onClose} width={500}>
      {page && <Body page={page} clinicians={clinicians} invariants={invariants} now={now} canDispatch={canDispatch} onClose={onClose} />}
    </Drawer>
  );
}

const VITAL_LABEL: Record<string, string> = {
  spo2: "SpO₂ %", hr: "Heart rate", rr: "Resp. rate", sbp: "Systolic", dbp: "Diastolic", map: "MAP", temp_c: "Temp °C",
  lactate: "Lactate", gcs: "GCS", glucose: "Glucose", nihss: "NIHSS", resp_rate: "Resp. rate", wbc: "WBC",
};

function Section({ title, right, children }: { title: string; right?: ReactNode; children: ReactNode }) {
  return (
    <section className="border-t border-line px-5 py-4">
      <div className="mb-3 flex items-center gap-2">
        <h3 className="text-sm font-semibold tracking-tight text-ink">{title}</h3>
        {right && <div className="ml-auto">{right}</div>}
      </div>
      {children}
    </section>
  );
}

const Check = ({ ok }: { ok: boolean }) =>
  ok ? (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden className="shrink-0 text-low">
      <circle cx="7" cy="7" r="6.25" fill="currentColor" fillOpacity=".14" stroke="currentColor" strokeOpacity=".5" />
      <path d="M4.3 7.2 6.1 9l3.6-4" stroke="currentColor" strokeWidth="1.5" fill="none" strokeLinecap="round" strokeLinejoin="round" />
    </svg>
  ) : (
    <svg width="14" height="14" viewBox="0 0 14 14" aria-hidden className="shrink-0 text-high">
      <circle cx="7" cy="7" r="6.25" fill="currentColor" fillOpacity=".16" stroke="currentColor" strokeOpacity=".55" />
      <path d="M4.8 4.8l4.4 4.4M9.2 4.8 4.8 9.2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
    </svg>
  );

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
  const st = statusLine(page, now);
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
    {
      label: "Triage",
      agent: "Priority Handler",
      value: `P${page.priority} · ${CATEGORY_LABEL[page.category] ?? page.category}${page.triage.rules_priority && page.triage.rules_priority !== page.priority ? ` (rules said P${page.triage.rules_priority})` : ""}`,
      ms: page.timings.triage_ms,
      state: "done",
    },
    { label: "Locate", agent: "Operator", value: page.room ? `Room ${page.room} · ${page.zone_name}` : "Room not identified", ms: page.timings.locate_ms, state: page.room ? "done" : "fail" },
    {
      label: "Route",
      agent: "Case Handler",
      value: page.decision.assignee ? clinicians[page.decision.assignee]?.name ?? page.decision.assignee : "No eligible clinician",
      ms: page.timings.route_ms,
      state: page.decision.assignee ? "done" : "fail",
    },
    {
      label: "Verify",
      agent: "Z3",
      value: cert ? (cert.certified ? "All invariants hold" : `Refused: ${cert.violations.join(", ")}`) : page.status === "review" ? "Waits for review" : "—",
      ms: page.timings.certify_ms,
      state: cert ? (cert.certified ? "done" : "fail") : "wait",
    },
    {
      label: "Page",
      agent: "Queue",
      value: PAGE_STATUS_LABEL[page.status],
      state: page.status === "accepted" || page.status === "resolved" ? "done" : page.status === "paging" ? "active" : "wait",
    },
  ];

  return (
    <>
      <header className="shrink-0 px-5 pb-4 pt-4">
        <div className="flex items-center gap-2">
          <PriorityChip p={page.priority} label />
          <span className="font-mono text-xs text-dim">{page.id}</span>
          <span className="text-faint">·</span>
          <SourceTag source={page.source} synthetic={page.synthetic} />
          <span className="text-faint">·</span>
          <span className="num text-xs text-dim">{age(page.created_at, now)} ago</span>
          <IconButton label="Close" onClick={onClose} className="ml-auto -mr-1">
            <CloseIcon />
          </IconButton>
        </div>
        <h2 className="mt-3 text-lg font-semibold tracking-tighter text-ink">
          {page.room ? `Room ${page.room}` : "Unknown room"}
          <span className="font-normal text-dim"> · {page.zone_name ?? "—"}</span>
        </h2>
        {page.patient?.name && (
          <div className="mt-0.5 text-sm text-muted">
            {page.patient.name}
            {page.patient.age ? `, ${page.patient.age}${page.patient.sex ?? ""}` : ""}
            <span className="text-dim">{page.patient.synthetic ? " · synthetic record" : page.patient.monitored ? " · monitored · replayed real record" : ""}</span>
          </div>
        )}
        <div className={cx("mt-2 text-xs", st.tone)}>{st.text}</div>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="px-5 pb-4">
          <div className="well px-4 py-3">
            <p className="text-sm leading-relaxed text-ink">{page.text}</p>
            <div className="num mt-2 text-xs text-dim">
              {page.reporter ?? page.source} · {clock(page.created_at)}
            </div>
          </div>
        </div>

        <Section title="Dispatch" right={page.timings.total_ms != null && <span className="num text-xs text-dim">{page.timings.total_ms.toFixed(1)} ms end to end</span>}>
          <div className="divide-y divide-line-soft">
            {steps.map((s) => (
              <div key={s.label} className="grid grid-cols-[16px_88px_minmax(0,1fr)_auto] items-center gap-3 py-2 first:pt-0 last:pb-0">
                {s.state === "done" ? (
                  <Check ok />
                ) : s.state === "fail" ? (
                  <Check ok={false} />
                ) : (
                  <span className={cx("mx-auto h-2 w-2 rounded-full", s.state === "active" ? "blink bg-st-paging" : "border border-line-bright")} />
                )}
                <div className="leading-tight">
                  <div className="text-sm font-medium text-ink">{s.label}</div>
                  <div className="text-2xs text-dim">{s.agent}</div>
                </div>
                <div className="truncate text-xs text-muted">{s.value}</div>
                <div className="num w-12 text-right text-xs text-dim">{s.ms != null ? (s.ms < 1 ? "<1 ms" : `${s.ms.toFixed(1)} ms`) : ""}</div>
              </div>
            ))}
          </div>
        </Section>

        <Section
          title="Triage"
          right={
            <span className="text-xs text-dim">
              {page.triage.source === "detector"
                ? "Sepsis detector"
                : page.triage.source === "monitor"
                  ? "Room monitor"
                  : page.triage.source === "llm+rules"
                    ? `Rules + ${page.triage.model ?? "model"}`
                    : "Rules"}
            </span>
          }
        >
          {page.triage.red_flags.length > 0 && (
            <div className="mb-2 flex flex-wrap gap-1.5">
              {page.triage.red_flags.map((f) => (
                <Tag key={f} tone="high">
                  {f}
                </Tag>
              ))}
            </div>
          )}
          <p className="text-sm leading-relaxed text-muted">{page.triage.rationale}</p>
          {page.monitor && page.monitor.findings.length > 0 && (
            <ul className="mt-3 divide-y divide-line-soft">
              {page.monitor.findings.map((f) => (
                <li key={f.key} className="flex items-start gap-2.5 py-2 first:pt-0 last:pb-0">
                  <span className="mt-[5px] h-1.5 w-1.5 shrink-0 rounded-full" style={{ background: `rgb(${SEV_RGB[f.severity]})` }} />
                  <div className="min-w-0 flex-1">
                    <div className="text-sm text-ink">{f.title}</div>
                    <div className="truncate text-xs text-dim">{f.source}</div>
                  </div>
                </li>
              ))}
            </ul>
          )}
          {Object.keys(page.triage.vitals).length > 0 && (
            <div className="mt-3 grid grid-cols-3 gap-x-4 gap-y-3 well px-4 py-3">
              {Object.entries(page.triage.vitals).map(([k, v]) => (
                <div key={k} className="min-w-0">
                  <div className="truncate text-xs text-dim">{VITAL_LABEL[k] ?? k.replace(/_/g, " ")}</div>
                  <div className="num mt-0.5 text-base font-medium text-ink">{typeof v === "number" ? (Number.isInteger(v) ? v : v.toFixed(1)) : v}</div>
                </div>
              ))}
            </div>
          )}
          {page.refining && <div className="working mt-2.5 rounded-md px-2.5 py-2 text-xs text-accent-ink">The model is reviewing this page. It may raise the priority; it cannot lower it.</div>}
          {page.refinement && (
            <div className={cx("mt-3 rounded-md px-3 py-2.5 text-xs leading-snug", page.refinement.outcome === "raised" ? "bg-dys/[0.1] text-dys-ink" : "well text-muted")}>
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

        <Section title="Routing" right={<span className="text-xs text-dim">Case Handler</span>}>
          {c ? (
            <div className="relative overflow-hidden well">
              <div className="flex items-center gap-3 px-4 py-3">
                <Avatar name={c.name} status={c.status} size={30} present={c.present} />
                <div className="min-w-0 flex-1">
                  <div className="text-sm font-medium text-ink">{c.name}</div>
                  <div className="truncate text-xs text-dim">
                    {c.specialty_label} · {c.title}
                  </div>
                </div>
                <div className="text-right">
                  <StatusText status={c.status} />
                  {page.status === "paging" && att && (
                    <div className="mt-0.5 text-xs text-dim">
                      escalates in <Countdown remaining={att.deadline - now} className="text-xs" />
                    </div>
                  )}
                </div>
              </div>
              {page.status === "paging" && att && <TimerBar remaining={att.deadline - now} total={att.deadline - att.sent_at} priority={page.priority} />}
            </div>
          ) : (
            <div className="rounded-md border border-dashed border-line-bright px-4 py-3 text-sm text-dys-ink">{page.review_reason ?? "No clinician assigned."}</div>
          )}
          <p className="mt-3 text-sm leading-relaxed text-muted">{page.decision.reasoning}</p>
          {page.decision.guardrails.length > 0 && (
            <ul className="mt-2.5 space-y-1.5">
              {page.decision.guardrails.map((g) => (
                <li key={g} className="flex gap-2 text-xs text-muted">
                  <ShieldCheck size={14} className="mt-px shrink-0 text-dim" />
                  {g}
                </li>
              ))}
            </ul>
          )}
          <div className="mt-4">
            <div className="grid grid-cols-[minmax(0,1fr)_64px] gap-2 border-b border-line pb-2 text-xs font-medium text-dim">
              <span>Candidate</span>
              <span className="text-right">Score</span>
            </div>
            {page.decision.candidates.slice(0, 6).map((cand, i) => {
              const max = Math.max(1, ...page.decision.candidates.map((x) => x.score));
              const top = i === 0 && !cand.excluded;
              return (
                <div key={cand.id} className={cx("-mx-2 rounded-md border-b border-line-soft px-2 py-2.5 last:border-b-0", top && "bg-accent/[0.08]")}>
                  <div className="grid grid-cols-[minmax(0,1fr)_64px] items-baseline gap-2">
                    <span className="truncate">
                      <span className={cx("text-sm font-medium", cand.excluded ? "text-dim line-through decoration-faint" : "text-ink")}>{cand.name}</span>
                      <span className="ml-1.5 text-xs text-dim">{cand.specialty_label}</span>
                    </span>
                    <span className="num text-right text-xs text-muted">{cand.excluded ? "—" : cand.score.toFixed(0)}</span>
                  </div>
                  {cand.excluded ? (
                    <div className="mt-0.5 text-xs text-dys-ink">Excluded: {cand.excluded}</div>
                  ) : (
                    <>
                      <div className="mt-1.5 h-[3px] overflow-hidden rounded-full bg-white/[0.06]">
                        <div className="h-full rounded-full bg-accent/80" style={{ width: `${Math.max(3, (cand.score / max) * 100)}%` }} />
                      </div>
                      <div className="mt-1 flex flex-wrap gap-x-2.5 gap-y-0.5 text-xs">
                        {cand.factors.map((f) => (
                          <span key={f.label} className={f.points >= 0 ? "text-dim" : "text-dys-ink"}>
                            <span className="num">{f.points >= 0 ? "+" : ""}{f.points}</span> {f.label}
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
          title="Policy certificate"
          right={
            cert ? (
              <Tag tone={cert.certified ? "low" : "high"}>{cert.certified ? "Proven · Z3 unsat" : `Refused · Z3 ${cert.z3_status}`}</Tag>
            ) : (
              <span className="text-xs text-dim">Issued when the page is sent</span>
            )
          }
        >
          <ul className="space-y-1.5">
            {invariants.map((inv, i) => {
              const bad = cert?.violations.includes(inv.id) || page.blocked.includes(inv.id);
              return (
                <li key={inv.id} className="flex gap-2 text-xs leading-snug">
                  {cert ? <span className="mt-px"><Check ok={!bad} /></span> : <span className="num mt-px w-3.5 shrink-0 text-center text-xs text-faint">{i + 1}</span>}
                  <span className={bad ? "text-high-ink" : "text-muted"}>{inv.text}</span>
                </li>
              );
            })}
          </ul>
          {cert && (
            <p className="mt-3 break-words well px-3 py-2.5 font-mono text-2xs leading-relaxed text-dim">
              {cert.proof} · {cert.ms} ms
            </p>
          )}
        </Section>

        {(page.attempts.length > 0 || page.escalation_log.length > 0) && (
          <Section title="Timeline">
            <ol className="space-y-2">
              <li className="grid grid-cols-[64px_minmax(0,1fr)] gap-2 text-xs">
                <span className="num text-xs text-dim">{clock(page.created_at)}</span>
                <span className="text-muted">Alert raised by {page.reporter ?? page.source}</span>
              </li>
              {page.attempts.map((a, i) => (
                <li key={i} className="grid grid-cols-[64px_minmax(0,1fr)] gap-2 text-xs">
                  <span className="num text-xs text-dim">{clock(a.sent_at)}</span>
                  <span className="text-muted">
                    Paged <span className="text-ink">{a.name}</span>
                    {a.outcome ? (
                      <>
                        {" → "}
                        <span className={a.outcome === "accepted" ? "text-low-ink" : "text-dys-ink"}>{a.outcome === "timeout" ? "no answer" : a.outcome}</span>
                        {a.at ? ` after ${Math.round(a.at - a.sent_at)} s` : ""}
                        {a.reason ? ` (${a.reason})` : ""}
                      </>
                    ) : (
                      <span className="blink text-accent-ink"> · ringing</span>
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
              <div key={n.clinician} className="flex items-center gap-2.5 text-sm">
                <Avatar name={n.name} size={20} />
                <span className="text-ink">{n.name}</span>
                <span className="truncate text-dim">{n.why}</span>
              </div>
            ))}
          </Section>
        )}

        {page.sbar && (
          <Section
            title="Brief (SBAR)"
            right={<span className="text-xs text-dim">{page.sbar_refining ? "model writing…" : page.sbar.source === "llm" ? page.sbar.model ?? "model" : "template"} · {page.sbar.words} words</span>}
          >
            <dl className="space-y-3">
              {(["situation", "background", "assessment", "request"] as const).map((k) => (
                <div key={k} className="grid grid-cols-[88px_minmax(0,1fr)] gap-3">
                  <dt className="pt-px text-xs font-medium capitalize text-dim">{k}</dt>
                  <dd className={cx("text-sm leading-relaxed", k === "request" ? "text-ink" : "text-ink/85")}>{page.sbar![k]}</dd>
                </div>
              ))}
            </dl>
          </Section>
        )}

        {page.room && (
          <Section title={page.bed ? "Room and patient" : "Room"}>
            <Link
              href={`/room/?id=${encodeURIComponent(page.room)}`}
              className="well mb-2 flex items-center justify-between px-4 py-2.5 text-sm text-ink transition-colors hover:bg-white/[0.04]"
            >
              <span>
                Open room {page.room}
                <span className="ml-2 text-xs text-dim">record, vitals, detections, timeline</span>
              </span>
              <OpenLink size={14} className="text-dim" aria-hidden />
            </Link>
            {page.bed && (
              <Link
                href={`/patient/?id=${encodeURIComponent(page.bed)}`}
                className="well flex items-center justify-between px-4 py-2.5 text-sm text-ink transition-colors hover:bg-white/[0.04]"
              >
                <span>
                  Open {page.patient?.name ?? "the patient"}&apos;s ICU record
                  {page.patient?.risk != null && <span className="num ml-2 text-xs text-dim">risk {page.patient.risk.toFixed(2)}</span>}
                </span>
                <OpenLink size={14} className="text-dim" aria-hidden />
              </Link>
            )}
          </Section>
        )}
        <div className="h-4" />
      </div>

      {canDispatch && !["resolved", "cancelled"].includes(page.status) && (
        <footer className="shrink-0 border-t border-line bg-s1 px-5 py-3.5">
          {err && <div className="mb-2 text-xs text-high-ink">{err}</div>}
          <div className="flex flex-wrap items-center gap-2">
            {["review", "held", "unassigned"].includes(page.status) && (
              <>
                <select
                  value={reassign}
                  onChange={(e) => setReassign(e.target.value)}
                  className="h-8 min-w-0 flex-1 field px-2.5 text-sm text-ink"
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
            <Button variant="danger" disabled={!!busy} onClick={() => act("cancel")} className="ml-auto">
              Cancel page
            </Button>
          </div>
        </footer>
      )}
    </>
  );
}

export { P_RGB };
