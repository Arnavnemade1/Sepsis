"use client";

import { useEffect, useRef, useState } from "react";

import { Button, cx, Empty, Meter, Pill, Plate, Skeleton, Sparkline, Stat } from "@/components/ui";
import { api } from "@/lib/api";
import { INDEX, pct, severityTone, tierTone, title, TONE_TEXT, VITALS, type Tone } from "@/lib/clinical";
import type { AgentOpinion, Decision, Snapshot } from "@/lib/types";

/* ------------------------------------------------------------------ */
/* Risk hero: the one number, what it is, and how sure we are about it */
/* ------------------------------------------------------------------ */
export function RiskHero({ snap }: { snap: Snapshot; recorded?: boolean }) {
  const r = snap.risk;
  const p = r.ml_probability;
  const tone = tierTone(r.level);
  const c = r.certainty as (Snapshot["risk"]["certainty"] & { unavailable?: boolean; reason?: string }) | undefined;
  const pop = r.population_risk;
  return (
    <div className="flex flex-wrap items-end gap-x-8 gap-y-4">
      <div>
        <div className="text-xs text-dim">Calibrated sepsis risk</div>
        <div className={cx("num text-4xl font-light leading-none tracking-tight", TONE_TEXT[tone === "low" ? "neutral" : tone])}>
          {p == null ? "—" : (p * 100).toFixed(1).split(".")[0]}
          <span className="text-2xl text-dim">{p == null ? "" : `.${(p * 100).toFixed(1).split(".")[1]}%`}</span>
        </div>
      </div>
      <div className="space-y-2 pb-1.5">
        <div className="flex items-center gap-2">
          <Pill tone={tone === "low" ? "neutral" : tone} dot>
            {title(r.level)} risk
          </Pill>
          {r.index != null && (
            <span
              className="num text-xs text-muted"
              title={`The ward risk index: this probability on the scale every room is read on. It pages at ${INDEX.page.toFixed(2)}, the detector's validated alarm point.`}
            >
              index {r.index.toFixed(2)} <span className="text-faint">/ pages at {INDEX.page.toFixed(2)}</span>
            </span>
          )}
          {r.model && (
            <span
              className="rounded-[4px] bg-white/[0.07] px-1.5 py-px font-mono text-2xs font-semibold text-muted"
              title={r.model === "A40" ? "Scored by A40, the all-variable detector (Architecture lists every model)" : "Scored by V7, the vital-sign detector (Architecture lists every model)"}
            >
              {r.model}
            </span>
          )}
        </div>
        <div className="text-xs text-muted">
          {c && !c.unavailable && c.lo != null && c.hi != null ? (
            <>
              Conformal band {pct(c.lo)}–{pct(c.hi)} at α={c.alpha} ·{" "}
              <span className="text-ink">{c.abstain ? "abstains" : c.state.replace(/_/g, " ")}</span>
            </>
          ) : (
            <span title={c?.reason}>No conformal claim for this score</span>
          )}
        </div>
      </div>
      <div className="flex gap-6 pb-1.5">
        <Stat label="SIRS" value={r.sirs_score} size="sm" tone={r.sirs_score >= 2 ? "med" : "neutral"} />
        <Stat label="qSOFA" value={r.qsofa_score} size="sm" tone={r.qsofa_score >= 2 ? "med" : "neutral"} />
        <Stat label="Signal" value={pct(r.signal_quality, 0)} size="sm" tone={r.signal_quality < 0.5 ? "med" : "neutral"} />
        {pop != null && Math.abs(pop - (p ?? 0)) > 0.0005 && (
          <Stat label="Before chart shift" value={pct(pop, 1)} size="sm" sub="the detector's own output" />
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Vitals                                                              */
/* ------------------------------------------------------------------ */
type Channel = {
  key: string;
  label?: string;
  value: number | null;
  unit?: string;
  normal_low?: number;
  normal_high?: number;
  status?: string;
  source?: string;
};

/** `source` overrides each channel's acquisition label: a replayed ICU record's values come
 *  from the record, whatever pathway the body-state model assigns the channel to. */
export function Vitals({ channels, trends, source }: { channels: Channel[]; trends: Record<string, (number | null)[]>; source?: string }) {
  const shown = channels.filter((c) => c.value != null && !["motion", "stress_vector", "diastolic_bp"].includes(c.key));
  if (!shown.length) return <Empty>No vitals on this feed.</Empty>;
  return (
    <div className="grid grid-cols-2 gap-x-4 gap-y-3">
      {shown.map((c) => {
        const out =
          c.status && c.status !== "NORMAL"
            ? true
            : (c.normal_low != null && c.value! < c.normal_low) || (c.normal_high != null && c.value! > c.normal_high);
        const tone: Tone = out ? "med" : "neutral";
        const digits = VITALS[c.key]?.d ?? (Math.abs(c.value!) < 10 ? 1 : 0);
        return (
          <div key={c.key} className="min-w-0">
            <div className="flex items-baseline justify-between gap-2">
              <span className="truncate text-xs text-dim">{c.label ?? title(c.key)}</span>
              {(source ?? c.source) && <span className="text-2xs text-faint">{source ?? c.source}</span>}
            </div>
            <div className="flex items-end justify-between gap-2">
              <span className={cx("num text-xl font-light leading-tight", TONE_TEXT[tone])}>
                {c.value!.toFixed(digits)}
                <span className="ml-1 text-xs text-dim">{c.unit}</span>
              </span>
              {trends[c.key] && <Sparkline data={trends[c.key]} tone={tone} w={64} h={20} />}
            </div>
            {c.normal_low != null && c.normal_high != null && (
              <div className="num text-2xs text-faint">
                normal {c.normal_low}–{c.normal_high}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Organ systems                                                       */
/* ------------------------------------------------------------------ */
export function Organs({ snap }: { snap: Snapshot }) {
  const systems = [...snap.body_state.organ_systems].sort(
    (a, b) => Number(b.monitored) - Number(a.monitored) || b.severity - a.severity,
  );
  return (
    <ul className="divide-y divide-line-soft">
      {systems.map((o) => {
        const tone = severityTone(o.severity, o.monitored);
        return (
          <li key={o.name} className="py-2">
            <div className="flex items-center gap-2">
              <span className={cx("text-sm", o.monitored ? "text-ink" : "text-dim")}>{o.name}</span>
              {!o.monitored && <span className="text-2xs text-faint">not measured</span>}
              {o.monitored && o.trend && o.trend !== "STABLE" && (
                <span className="text-2xs text-muted">{title(o.trend)}</span>
              )}
              <span className="num ml-auto text-xs text-muted">
                {o.monitored ? (o.sofa_points != null ? `SOFA ${o.sofa_points}` : pct(o.severity, 0)) : "—"}
              </span>
            </div>
            {o.monitored && (
              <>
                <Meter value={o.severity} tone={tone} className="mt-1.5" />
                {o.drivers.length > 0 && <div className="mt-1 truncate text-xs text-muted">{o.drivers.join(" · ")}</div>}
              </>
            )}
          </li>
        );
      })}
    </ul>
  );
}

/* ------------------------------------------------------------------ */
/* The decision, and why                                               */
/* ------------------------------------------------------------------ */
export function DecisionCard({ d }: { d: Decision | null }) {
  if (!d) return <Skeleton className="h-28" />;
  const urgent = d.urgency === "immediate" || d.urgency === "urgent";
  return (
    <div>
      <div className="flex items-center gap-3">
        <div className="text-lg font-medium tracking-tight">{title(d.action)}</div>
        <Pill tone={urgent ? "high" : d.urgency === "soon" ? "med" : "neutral"}>{title(d.urgency)}</Pill>
        {d.abstained && <Pill tone="med">Abstained</Pill>}
      </div>
      <div className="mt-1 text-xs text-dim">Route · {title(d.route)}</div>
      <ul className="mt-3 space-y-1.5 text-sm leading-snug text-muted">
        {d.because.map((b) => (
          <li key={b} className="flex gap-2">
            <span className="mt-[7px] h-1 w-1 shrink-0 rounded-full bg-faint" />
            <span>{b}</span>
          </li>
        ))}
      </ul>
      {d.limits?.length > 0 && (
        <details className="mt-3 text-xs">
          <summary className="cursor-pointer text-dim hover:text-ink">Limits of this recommendation</summary>
          <ul className="mt-2 space-y-1 text-muted">
            {d.limits.map((l) => (
              <li key={l}>{l}</li>
            ))}
          </ul>
        </details>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The specialist panel                                                */
/* ------------------------------------------------------------------ */
export function Reasoning({ snap }: { snap: Snapshot }) {
  const r = snap.reasoning as
    | { opinions: AgentOpinion[]; agreement?: number; dissent?: string[] | null; summary?: string }
    | null
    | undefined;
  if (!r?.opinions?.length) return <Empty>No specialist opinions on this feed.</Empty>;
  const ops = [...r.opinions].sort((a, b) => b.concern - a.concern);
  return (
    <div>
      {r.summary && <p className="mb-3 text-sm leading-snug text-muted">{r.summary}</p>}
      <ul className="space-y-2.5">
        {ops.map((o) => {
          const tone = severityTone(o.concern);
          return (
            <li key={o.agent}>
              <div className="flex items-baseline gap-2">
                <span className="text-sm text-ink">{o.agent}</span>
                <span className="truncate text-xs text-dim">{o.rationale}</span>
                <span className="num ml-auto text-xs text-muted">{pct(o.concern, 0)}</span>
              </div>
              <Meter value={o.concern} tone={tone} className="mt-1" />
              {o.recommendation && <div className="mt-1 text-xs text-muted">→ {o.recommendation}</div>}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Dr. Aegis — a grounded, read-mostly consult on this patient          */
/* ------------------------------------------------------------------ */
type Turn = { role: "user" | "model"; text: string; meta?: string };
type ConsultOut = { answer: string; source: string; model?: string | null; provider?: string | null; budget_remaining?: number };
type Brief = { headline?: string; unmeasured?: string[]; opening?: string };

export function Consult({ bed, name }: { bed: string | null; name: string }) {
  const [turns, setTurns] = useState<Turn[]>([]);
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [brief, setBrief] = useState<Brief | null>(null);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    setTurns([]);
    setBrief(null);
    const qs = bed ? `?bed=${encodeURIComponent(bed)}&opening=false` : "?opening=false";
    api<Brief>(`/aegis/doctor/brief${qs}`).then(setBrief).catch(() => {});
  }, [bed]);
  useEffect(() => {
    // scroll the consult pane only, never the page
    const pane = end.current?.parentElement;
    if (pane) pane.scrollTop = pane.scrollHeight;
  }, [turns]);

  const ask = async (text: string) => {
    const question = text.trim();
    if (!question || busy) return;
    setQ("");
    const history = turns.map((t) => ({ role: t.role, text: t.text }));
    setTurns((t) => [...t, { role: "user", text: question }]);
    setBusy(true);
    try {
      const out = await api<ConsultOut>("/aegis/doctor/consult", {
        body: { question, bed, history, agentic: false },
        timeoutMs: 90_000,
      });
      setTurns((t) => [
        ...t,
        { role: "model", text: out.answer, meta: out.source === "llm" ? out.model ?? "llm" : "offline template" },
      ]);
    } catch (e) {
      setTurns((t) => [...t, { role: "model", text: `Couldn't reach the consult: ${e instanceof Error ? e.message : e}`, meta: "error" }]);
    } finally {
      setBusy(false);
    }
  };

  const starters = ["What's driving this patient's risk?", "What aren't we measuring that would change your mind?", "Is the current plan safe?"];
  return (
    <div className="flex h-full min-h-[340px] flex-col">
      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto pr-1">
        {turns.length === 0 && (
          <div className="space-y-3">
            <p className="text-sm leading-relaxed text-muted">
              Dr. Aegis answers from {name}&rsquo;s live dossier only: the risk, the vitals, the organ systems and what the
              system is not measuring. It cannot order anything.
            </p>
            {brief?.unmeasured && brief.unmeasured.length > 0 && (
              <div className="well p-3.5">
                <div className="mb-1.5 text-xs text-dim">Not measured, so not known</div>
                <ul className="space-y-0.5 text-xs text-muted">
                  {brief.unmeasured.slice(0, 5).map((u) => (
                    <li key={u}>· {u}</li>
                  ))}
                </ul>
              </div>
            )}
            <div className="flex flex-wrap gap-2">
              {starters.map((s) => (
                <button key={s} onClick={() => ask(s)} className="rounded-full bg-white/[0.05] px-3 py-1.5 text-xs text-muted shadow-[inset_0_0_0_1px_rgb(255_255_255/.06)] transition-colors hover:bg-white/[0.08] hover:text-ink">
                  {s}
                </button>
              ))}
            </div>
          </div>
        )}
        {turns.map((t, i) => (
          <div key={i} className={cx("max-w-[92%] rounded-xl px-3.5 py-2.5 text-sm leading-relaxed", t.role === "user" ? "ml-auto bg-accent/[0.16] text-ink" : "bg-white/[0.04] text-ink")}>
            <div className="whitespace-pre-wrap">{t.text}</div>
            {t.meta && <div className="mt-1 text-2xs text-dim">{t.meta}</div>}
          </div>
        ))}
        {busy && <div className="text-xs text-dim">Dr. Aegis is reading the chart…</div>}
        <div ref={end} />
      </div>
      <form
        className="mt-3 flex gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          ask(q);
        }}
      >
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder={`Ask about ${name}…`}
          className="min-w-0 flex-1 field px-3 py-2 text-sm text-ink placeholder:text-faint"
        />
        <Button type="submit" variant="primary" disabled={busy || !q.trim()}>
          Ask
        </Button>
      </form>
    </div>
  );
}

export { Plate };
