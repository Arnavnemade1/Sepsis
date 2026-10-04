"use client";

import Link from "next/link";
import { Fragment, useMemo, useState } from "react";

import { Button, cx, Empty, Kv, MiniBars, MiniLine, Panel, Segmented, Switch, Tag } from "@/components/ui";
import { api } from "@/lib/api";
import {
  CATEGORY_LABEL,
  STATUS_LABEL,
  type Clinician,
  type Layout,
  type Monitor,
  type Page,
  type Rec,
  type Settings,
  type StaffStatus,
  type Stats,
} from "@/lib/paging";

import { Avatar, P_RGB, PriorityChip, SourceTag, StatusDot, StatusText } from "./bits";
import { age, statusLine } from "./queue";

/* ------------------------------------------------------------------ */
/* KPI strip                                                           */
/* ------------------------------------------------------------------ */
const BUCKET_S = 300;
const BUCKETS = 12;

/** The last hour in 5-minute buckets, oldest first, from every page the console holds. */
function useHourSeries(all: Record<string, Page>, now: number) {
  return useMemo(() => {
    const start = now - BUCKET_S * BUCKETS;
    const at = (t: number | null | undefined) => (t == null || t < start || t > now ? -1 : Math.min(BUCKETS - 1, Math.floor((t - start) / BUCKET_S)));
    const raised = Array(BUCKETS).fill(0);
    const p1 = Array(BUCKETS).fill(0);
    const esc = Array(BUCKETS).fill(0);
    const acc: number[][] = Array.from({ length: BUCKETS }, () => []);
    for (const p of Object.values(all)) {
      const b = at(p.created_at);
      if (b >= 0) {
        raised[b] += 1;
        if (p.priority === 1) p1[b] += 1;
      }
      for (const a of p.attempts.slice(1)) {
        const e = at(a.sent_at);
        if (e >= 0) esc[e] += 1;
      }
      if (p.accepted_at && p.priority <= 2) {
        const k = at(p.accepted_at);
        if (k >= 0) acc[k].push(p.accepted_at - p.created_at);
      }
    }
    const median = (xs: number[]) => (xs.length ? [...xs].sort((a, b) => a - b)[Math.floor(xs.length / 2)] : null);
    return { raised, p1, esc, accept: acc.map(median) };
    // bucket boundaries move once a minute, not every second
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [all, Math.floor(now / 60)]);
}

const plural = (n: number, word: string) => `${n} ${word}${n === 1 ? "" : "s"}`;

export function KpiStrip({ stats, pages, monitor, all, now }: { stats: Stats | null; pages: Page[]; monitor: Monitor; all?: Record<string, Page>; now?: number }) {
  const series = useHourSeries(all ?? {}, now ?? 0);
  const p1 = pages.filter((p) => p.priority === 1).length;
  const review = pages.filter((p) => p.status === "review" || p.status === "unassigned").length;
  const watch = Object.values(monitor).filter((m) => m.watch).length;
  const ack = stats?.median_ack_s ?? {};
  const fastAck = [ack["1"], ack["2"]].filter((x): x is number => x != null);
  const showSeries = !!all && now != null;
  const items: { k: string; v: string | number; sub: string; tone?: string; spark?: React.ReactNode }[] = [
    { k: "Active pages", v: pages.length, sub: `${pages.filter((p) => p.status === "paging").length} ringing`, spark: showSeries && <MiniBars data={series.raised} title="Pages raised per 5 min, last hour" /> },
    { k: "P1 active", v: p1, tone: p1 ? "text-high-ink" : undefined, sub: p1 ? "immediate response" : "none", spark: showSeries && <MiniBars data={series.p1} color="#e5484d" title="P1 pages per 5 min, last hour" /> },
    { k: "Awaiting operator", v: review, tone: review ? "text-dys-ink" : undefined, sub: "review or unassigned" },
    { k: "Time to accept", v: fastAck.length ? `${Math.round(Math.min(...fastAck))}s` : "—", sub: "median, P1/P2", spark: showSeries && <MiniLine data={series.accept} /> },
    { k: "Escalation rate", v: stats?.escalation_rate != null ? `${Math.round(stats.escalation_rate * 100)}%` : "—", sub: `${plural(stats?.timeouts ?? 0, "timeout")} · ${plural(stats?.declines ?? 0, "decline")}`, spark: showSeries && <MiniBars data={series.esc} color="#f5762e" title="Escalations per 5 min, last hour" /> },
    { k: "Monitored beds", v: stats?.monitored_beds ?? Object.keys(monitor).length, sub: watch ? `${watch} on sepsis watch` : "detector active" },
  ];
  return (
    <div className="grid shrink-0 grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-3 xl:grid-cols-6">
      {items.map((it) => (
        <div key={it.k} className="min-w-0 bg-s1 px-3.5 py-2.5">
          <div className="truncate text-[11px] text-dim">{it.k}</div>
          {/* the trend sits beside the number, so the line under it keeps the full width */}
          <div className="mt-0.5 flex items-end justify-between gap-2">
            <div className={cx("num text-[20px] font-medium leading-tight tracking-[-0.01em]", it.tone ?? "text-ink")}>{it.v}</div>
            {it.spark && <div className="mb-1 hidden shrink-0 opacity-90 sm:block">{it.spark}</div>}
          </div>
          <div className="truncate text-[11px] text-dim">{it.sub}</div>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Staff                                                               */
/* ------------------------------------------------------------------ */
export function StaffBoard({ clinicians, pages, layout, canDispatch, now }: {
  clinicians: Record<string, Clinician>;
  pages: Page[];
  layout: Layout | null;
  canDispatch: boolean;
  now: number;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const groups = useMemo(() => {
    const g: Record<string, Clinician[]> = {};
    for (const c of Object.values(clinicians)) (g[c.specialty_label] ??= []).push(c);
    return Object.entries(g).sort(([a], [b]) => a.localeCompare(b));
  }, [clinicians]);
  const load = (id: string) => pages.filter((p) => p.assignee === id && (p.status === "paging" || p.status === "accepted"));
  const setStatus = async (id: string, status: StaffStatus) => {
    setBusy(id);
    await api(`/aegis/paging/clinicians/${id}/status`, { body: { status } }).catch(() => {});
    setBusy(null);
  };
  const total = Object.values(clinicians);
  return (
    <Panel
      title="Staff on the roster"
      meta={`${total.filter((c) => c.status === "available").length} available · ${total.filter((c) => c.status === "paging").length} being paged · ${total.filter((c) => c.status === "in_procedure" || c.status === "on_case").length} busy · ${total.filter((c) => c.status === "off_shift").length} off shift`}
      flush
      className="shrink-0"
    >
      {/* a phone gets one row per clinician, grouped the same way */}
      <div className="md:hidden">
        {groups.map(([spec, staff]) => (
          <div key={spec}>
            <div className="border-b border-line bg-s2/70 px-3.5 py-1.5 text-[11.5px]">
              <span className="font-medium text-muted">{spec}</span>
              <span className="ml-2 text-dim">
                {staff.filter((c) => c.status === "available").length}/{staff.length} available
              </span>
            </div>
            {staff.map((c) => {
              const mine = load(c.id);
              const where = layout?.zones[c.zone]?.name ?? (c.zone === "remote" ? "Telehealth" : c.zone);
              const freeIn = c.busy_until && (c.status === "in_procedure" || c.status === "on_case") ? `free in ${Math.max(1, Math.round((c.busy_until - now) / 60))} min` : null;
              return (
                <div key={c.id} className="flex items-center gap-2.5 border-b border-line-soft px-3.5 py-2.5">
                  <Avatar name={c.name} size={24} present={c.present} />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span className="truncate text-[12.5px] text-ink">{c.name}</span>
                      {c.on_call && <Tag tone="neutral">on call</Tag>}
                    </div>
                    <div className="mt-0.5 flex flex-wrap items-center gap-x-2 text-[11.5px] text-dim">
                      <StatusText status={c.status} />
                      <span>{where}</span>
                      {mine.length > 0 && <span>{mine.length} active</span>}
                      {freeIn && <span className="font-mono">{freeIn}</span>}
                    </div>
                  </div>
                  {c.pageable && (
                    <Link href={`/clinician/?id=${c.id}`} className="shrink-0 rounded-md px-2 py-1 text-[11.5px] text-accent-ink transition-colors hover:bg-s3">
                      Pager ↗
                    </Link>
                  )}
                </div>
              );
            })}
          </div>
        ))}
      </div>
      <div className="hidden overflow-x-auto md:block">
        <table className="w-full min-w-[820px] text-[12.5px]">
          <thead>
            <tr className="border-b border-line text-left text-[11px] text-dim">
              <th className="px-3.5 py-2 font-medium">Clinician</th>
              <th className="px-3 py-2 font-medium">Status</th>
              <th className="px-3 py-2 font-medium">Location</th>
              <th className="px-3 py-2 font-medium">Active pages</th>
              <th className="px-3 py-2 font-medium">Free in</th>
              <th className="px-3 py-2 font-medium" />
            </tr>
          </thead>
          <tbody>
            {groups.map(([spec, staff]) => (
              <Fragment key={spec}>
                <tr className="border-b border-line bg-s2/70">
                  <td colSpan={6} className="px-3.5 py-1.5 text-[11.5px]">
                    <span className="font-medium text-muted">{spec}</span>
                    <span className="ml-2 text-dim">
                      {staff.filter((c) => c.status === "available").length}/{staff.length} available
                    </span>
                  </td>
                </tr>
                {staff.map((c) => {
                  const mine = load(c.id);
                  const where = layout?.zones[c.zone]?.name ?? (c.zone === "remote" ? "Telehealth" : c.zone);
                  const freeIn = c.busy_until && (c.status === "in_procedure" || c.status === "on_case") ? `${Math.max(1, Math.round((c.busy_until - now) / 60))} min` : "—";
                  return (
                    <tr key={c.id} className="border-b border-line-soft transition-colors last:border-b-0 hover:bg-s2">
                      <td className="px-3.5 py-2">
                        <div className="flex items-center gap-2.5">
                          <Avatar name={c.name} size={24} present={c.present} />
                          <div className="min-w-0">
                            <div className="flex items-center gap-1.5">
                              <span className="truncate text-ink">{c.name}</span>
                              {c.on_call && <Tag tone="neutral">on call</Tag>}
                              {!c.pageable && <Tag tone="neutral">operator</Tag>}
                            </div>
                            <div className="truncate text-[11px] text-dim">{c.title}</div>
                          </div>
                        </div>
                      </td>
                      <td className="px-3 py-2">
                        <StatusText status={c.status} />
                      </td>
                      <td className="px-3 py-2 text-muted">{where}</td>
                      <td className="num px-3 py-2 font-mono text-[12px] text-muted">{mine.length || "—"}</td>
                      <td className="num px-3 py-2 font-mono text-[12px] text-dim">{freeIn}</td>
                      <td className="px-3 py-2">
                        <div className="flex items-center justify-end gap-1.5">
                          {c.pageable && (
                            <Link href={`/clinician/?id=${c.id}`} className="rounded-md px-2 py-1 text-[11.5px] text-accent-ink transition-colors hover:bg-s3" title="Open this clinician's pager">
                              Pager ↗
                            </Link>
                          )}
                          {canDispatch && c.pageable && (
                            <select
                              disabled={busy === c.id}
                              value={c.status}
                              onChange={(e) => setStatus(c.id, e.target.value as StaffStatus)}
                              className="h-7 rounded-md border border-line-bright bg-s2 px-1.5 text-[11.5px] text-muted"
                              aria-label={`Status for ${c.name}`}
                            >
                              {/* "Being paged" is the system's to set, so it shows but cannot be chosen */}
                              {c.status === "paging" && (
                                <option value="paging" disabled>
                                  {STATUS_LABEL.paging}
                                </option>
                              )}
                              {(["available", "in_procedure", "on_case", "off_shift"] as StaffStatus[]).map((s) => (
                                <option key={s} value={s}>
                                  {STATUS_LABEL[s]}
                                </option>
                              ))}
                            </select>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </Panel>
  );
}

/* ------------------------------------------------------------------ */
/* Sentinel                                                            */
/* ------------------------------------------------------------------ */
const SEV: Record<Rec["severity"], { tone: "high" | "med" | "neutral"; label: string; bar: string }> = {
  critical: { tone: "high", label: "Critical", bar: "#e5484d" },
  warn: { tone: "med", label: "Warning", bar: "#eeb12a" },
  info: { tone: "neutral", label: "Info", bar: "#5f666f" },
};

export function Insights({ recs, stats, canDispatch, now }: { recs: Rec[]; stats: Stats | null; canDispatch: boolean; now: number }) {
  const ack = async (id: string) => api(`/aegis/paging/recs/${encodeURIComponent(id)}/ack`, { body: {} }).catch(() => {});
  const byP = stats?.by_priority ?? {};
  const maxP = Math.max(1, ...Object.values(byP));
  return (
    <div className="grid grid-cols-1 shrink-0 gap-3 xl:grid-cols-[minmax(0,1fr)_360px]">
      <Panel
        title="Sentinel"
        meta="Scans the whole paging system every 30 s for surges, coverage gaps, overload, escalation storms, sepsis clusters and bundle clocks. It recommends; it never pages."
        flush
      >
        {recs.length === 0 ? (
          <Empty>No systemic issues detected.</Empty>
        ) : (
          <ul>
            {recs.map((r) => {
              const sev = SEV[r.severity];
              return (
                <li key={r.id} className={cx("relative border-b border-line px-4 py-3 last:border-b-0", r.acknowledged && "opacity-55")}>
                  <span className="absolute inset-y-0 left-0 w-[2px]" style={{ background: sev.bar }} />
                  <div className="flex items-start gap-3">
                    <div className="min-w-0 flex-1">
                      <div className="flex flex-wrap items-center gap-2">
                        <Tag tone={sev.tone}>{sev.label}</Tag>
                        <span className="text-[13px] font-semibold text-ink">{r.title}</span>
                        <span className="text-[11px] text-dim">
                          {r.kind.replace(/_/g, " ")} · {age(r.created_at, now)}
                        </span>
                      </div>
                      <p className="mt-1.5 text-[12.5px] leading-relaxed text-muted">{r.detail}</p>
                      <p className="mt-1 text-[12.5px] leading-relaxed text-ink">
                        <span className="text-dim">Recommended: </span>
                        {r.action}
                      </p>
                      {r.evidence.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-1">
                          {r.evidence.map((e) => (
                            <span key={e} className="rounded-[3px] border border-line bg-s2 px-1.5 py-0.5 font-mono text-[10.5px] text-dim">
                              {e}
                            </span>
                          ))}
                        </div>
                      )}
                    </div>
                    {canDispatch && !r.acknowledged ? (
                      <Button size="sm" variant="ghost" onClick={() => ack(r.id)}>
                        Acknowledge
                      </Button>
                    ) : r.acknowledged ? (
                      <span className="text-[11px] text-dim">Acknowledged</span>
                    ) : null}
                  </div>
                </li>
              );
            })}
          </ul>
        )}
      </Panel>
      <div className="space-y-3">
        <Panel title="Pages by priority" meta="Since this server started">
          <div className="space-y-2">
            {[1, 2, 3, 4].map((p) => (
              <div key={p} className="grid grid-cols-[26px_minmax(0,1fr)_28px] items-center gap-2.5">
                <PriorityChip p={p} size="sm" />
                <div className="h-1 overflow-hidden rounded-[2px] bg-s3">
                  <div className="h-full" style={{ width: `${((byP[String(p)] ?? 0) / maxP) * 100}%`, background: `rgb(${P_RGB[p]})` }} />
                </div>
                <span className="num text-right font-mono text-[12px] text-muted">{byP[String(p)] ?? 0}</span>
              </div>
            ))}
          </div>
        </Panel>
        <Panel title="Dispatch performance">
          <Kv k="Median time to accept · P1" v={stats?.median_ack_s?.["1"] != null ? `${Math.round(stats.median_ack_s["1"]!)} s` : "—"} />
          <Kv k="Median time to accept · P2" v={stats?.median_ack_s?.["2"] != null ? `${Math.round(stats.median_ack_s["2"]!)} s` : "—"} />
          <Kv k="Median dispatch latency" v={stats?.median_dispatch_ms != null ? `${stats.median_dispatch_ms} ms` : "—"} />
          <Kv k="Certified by Z3" v={stats ? stats.certificates : "—"} />
          <Kv k="Held by the prover" v={stats ? stats.blocked : "—"} />
          <Kv k="Raised by the detector" v={stats?.by_source?.detector ?? 0} />
          <Kv k="Raised by nurses · typed / voice" v={stats ? `${stats.by_source?.nurse ?? 0} / ${stats.by_source?.voice ?? 0}` : "—"} />
          <p className="mt-2 border-t border-line pt-2 text-[11px] leading-snug text-dim">
            Demo traffic and simulated responders are synthetic. Detector pages come from replayed, de-identified ICU records.
          </p>
        </Panel>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Policy                                                              */
/* ------------------------------------------------------------------ */
const UNIT_LABEL: Record<string, string> = { icu: "Intensive care", ed: "Emergency", stepdown: "Step-down / cardiac", ward: "Wards", obs: "Observation", pacu: "Recovery (PACU)", or: "Operating rooms" };

export function PolicyPanel({ settings, invariants, canConfigure, stats }: {
  settings: Settings | null;
  invariants: { id: string; text: string }[];
  canConfigure: boolean;
  stats: Stats | null;
}) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  if (!settings) return <div className="h-64 animate-pulse rounded-lg bg-s1" />;
  const save = async (patch: Partial<Settings>) => {
    setBusy(true);
    setErr(null);
    try {
      const r = await api<{ status: string; detail?: string }>("/aegis/paging/settings", { body: { patch } });
      if (r.status !== "ok") setErr(r.detail ?? r.status);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="grid grid-cols-1 shrink-0 gap-3 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
      <div className="space-y-3">
        {!canConfigure && (
          <div className="flex items-start gap-2 rounded-lg border border-line bg-s1 px-3.5 py-2.5 text-[12.5px] text-muted">
            <Tag tone="neutral">Read only</Tag>
            <span>
              Changing the paging policy requires <span className="font-mono text-[11.5px] text-ink">page:configure</span> (chief of critical care).
            </span>
          </div>
        )}
        {err && <div className="rounded-lg border border-high/40 bg-high/10 px-3.5 py-2.5 text-[12.5px] text-high-ink">{err}</div>}
        <Panel
          title="Autonomy"
          meta={`Which pages are sent automatically and which wait for operator review. A P1 left in review sends itself after ${settings.review_failsafe_s["1"]} s.`}
          flush
        >
          <div className="overflow-x-auto">
            <table className="w-full min-w-[520px] text-[12.5px]">
              <thead>
                <tr className="border-b border-line text-left text-[11px] text-dim">
                  <th className="px-3.5 py-2 font-medium">Unit</th>
                  {[1, 2, 3, 4].map((p) => (
                    <th key={p} className="px-2 py-2 text-center font-medium">
                      <PriorityChip p={p} size="sm" />
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {Object.entries(settings.autonomy).map(([kind, row]) => (
                  <tr key={kind} className="border-b border-line-soft last:border-b-0">
                    <td className="px-3.5 py-1.5 text-ink">{UNIT_LABEL[kind] ?? kind}</td>
                    {row.map((mode, i) => (
                      <td key={i} className="px-2 py-1.5 text-center">
                        <button
                          disabled={!canConfigure || busy}
                          onClick={() => {
                            const next = [...row];
                            next[i] = mode === "review" ? "auto" : "review";
                            save({ autonomy: { [kind]: next } as Settings["autonomy"] });
                          }}
                          className={cx(
                            "h-6 w-[68px] rounded-[4px] border text-[11.5px] font-medium transition-colors disabled:cursor-default",
                            mode === "review" ? "border-[#5a4a1e] bg-med/15 text-med-ink" : "border-line bg-s2 text-dim enabled:hover:text-ink",
                          )}
                        >
                          {mode === "review" ? "Review" : "Auto"}
                        </button>
                      </td>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="flex items-center gap-3 border-t border-line px-3.5 py-2.5">
            <Switch checked={settings.detector_p1_review} disabled={!canConfigure || busy} onChange={(v) => save({ detector_p1_review: v })} label="Machine P1 review" />
            <span className="text-[12.5px] text-muted">A P1 raised by the sepsis detector or the room monitor always waits for a person first</span>
          </div>
          <div className="flex items-center gap-3 border-t border-line px-3.5 py-2.5">
            <Switch checked={settings.room_monitor_pages ?? true} disabled={!canConfigure || busy} onChange={(v) => save({ room_monitor_pages: v })} label="Room monitor pages" />
            <span className="text-[12.5px] text-muted">A ward room&apos;s detections (NEWS2, qSOFA, opioid and medication safety) raise pages; off, they are only shown</span>
          </div>
        </Panel>
        <Panel title="Guardrails" meta="Applied by the Case Handler as named steps on every routing decision">
          <div className="grid grid-cols-1 gap-x-6 sm:grid-cols-2">
            <Kv k="Ring time, P1 / P2 / P3 / P4" v={`${["1", "2", "3", "4"].map((p) => settings.ack_timeout_s[p]).join(" / ")} s`} />
            <Kv k="Consecutive-paging window" v={`${Math.round(settings.consecutive_window_s / 60)} min`} />
            <Kv k="P3/P4 workload limit" v={`${settings.workload_max_per_hour} / hour`} />
            <Kv k="Near-tie band" v={`${Math.round(settings.near_tie_pct * 100)}%`} />
            <Kv k="Hold for the right person" v={`up to ${Math.round(settings.hold_window_s / 60)} min`} />
            <Kv k="Escalations before the operator" v={settings.max_escalations} />
          </div>
        </Panel>
        <Panel title="Demonstration" meta="Behaviour of the demo hospital; both are labelled synthetic wherever they appear">
          <div className="space-y-2.5">
            <div className="flex items-center gap-3">
              <Switch checked={settings.simulated_responders} disabled={!canConfigure || busy} onChange={(v) => save({ simulated_responders: v })} label="Simulated responders" />
              <span className="text-[12.5px] text-muted">Clinicians without an open pager answer on their own</span>
            </div>
            <div className="flex items-center gap-3">
              <Switch checked={settings.demo_traffic} disabled={!canConfigure || busy} onChange={(v) => save({ demo_traffic: v })} label="Demo traffic" />
              <span className="text-[12.5px] text-muted">A synthetic nurse alert every few minutes, and simulated courses in the synthetic rooms</span>
            </div>
          </div>
        </Panel>
      </div>
      <Panel title="Invariants" meta="Checked by Z3 before any page is sent. unsat: the page cannot violate them and is sent. sat: it is held with the violated rule named. If the prover is unhealthy, nothing is sent." className="h-fit">
        <ol className="divide-y divide-line-soft">
          {invariants.map((inv, i) => (
            <li key={inv.id} className="grid grid-cols-[20px_minmax(0,1fr)] gap-2.5 py-2 first:pt-0 last:pb-0">
              <span className="num pt-px font-mono text-[11.5px] text-dim">{String(i + 1).padStart(2, "0")}</span>
              <div>
                <div className="text-[12.5px] leading-snug text-ink/90">{inv.text}</div>
                <div className="mt-0.5 font-mono text-[10.5px] text-faint">{inv.id}</div>
              </div>
            </li>
          ))}
        </ol>
        {stats && (
          <div className="mt-3 grid grid-cols-2 gap-px overflow-hidden rounded-md border border-line bg-line">
            <div className="bg-s1 px-3 py-2">
              <div className="text-[11px] text-dim">Certified since boot</div>
              <div className="num font-mono text-[16px] text-ink">{stats.certificates}</div>
            </div>
            <div className="bg-s1 px-3 py-2">
              <div className="text-[11px] text-dim">Held by the prover</div>
              <div className={cx("num font-mono text-[16px]", stats.blocked ? "text-high-ink" : "text-ink")}>{stats.blocked}</div>
            </div>
          </div>
        )}
      </Panel>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Pages                                                               */
/* ------------------------------------------------------------------ */
export function CasesTable({ pages, clinicians, now, onOpen }: {
  pages: Record<string, Page>;
  clinicians: Record<string, Clinician>;
  now: number;
  onOpen: (id: string) => void;
}) {
  const [filter, setFilter] = useState<"open" | "p1" | "detector" | "monitor" | "closed" | "all">("open");
  const rows = useMemo(() => {
    const all = Object.values(pages).sort((a, b) => b.created_at - a.created_at);
    const open = (p: Page) => !["resolved", "cancelled"].includes(p.status);
    if (filter === "open") return all.filter(open).sort((a, b) => a.priority - b.priority || b.created_at - a.created_at);
    if (filter === "p1") return all.filter((p) => p.priority === 1);
    if (filter === "detector") return all.filter((p) => p.source === "detector");
    if (filter === "monitor") return all.filter((p) => p.source === "monitor");
    if (filter === "closed") return all.filter((p) => !open(p));
    return all;
  }, [pages, filter]);
  return (
    <Panel
      title="Pages"
      right={
        <div className="flex items-center gap-3">
          <span className="num hidden font-mono text-[11.5px] text-dim sm:inline">{rows.length}</span>
          <Segmented
            size="sm"
            value={filter}
            onChange={setFilter}
            options={[
              { value: "open", label: "Active" },
              { value: "p1", label: "P1" },
              { value: "detector", label: "Detector" },
              { value: "monitor", label: "Monitor" },
              { value: "closed", label: "Closed" },
              { value: "all", label: "All" },
            ]}
          />
        </div>
      }
      flush
      className="shrink-0"
    >
      {rows.length === 0 ? (
        <Empty>No pages match this filter.</Empty>
      ) : (
        <>
        {/* a phone gets one card per page; the table needs ~900 px */}
        <ul className="divide-y divide-line-soft md:hidden">
          {rows.map((p) => {
            const c = p.assignee ? clinicians[p.assignee] : p.planned ? clinicians[p.planned] : null;
            const st = statusLine(p, now);
            return (
              <li key={p.id}>
                <button onClick={() => onOpen(p.id)} className="block w-full px-3.5 py-2.5 text-left transition-colors active:bg-s2">
                  <div className="flex items-center gap-2">
                    <PriorityChip p={p.priority} size="sm" />
                    <span className="font-mono font-semibold text-ink">{p.room ?? "—"}</span>
                    <span className="min-w-0 truncate text-[11.5px] text-dim">{p.zone_name}</span>
                    <span className="num ml-auto shrink-0 font-mono text-[11px] text-dim">{age(p.created_at, now)}</span>
                  </div>
                  <div className="mt-1 line-clamp-2 text-[12.5px] leading-snug text-ink/90">{p.text}</div>
                  <div className="mt-1.5 flex items-center gap-2 text-[11.5px]">
                    {c ? (
                      <span className="inline-flex min-w-0 items-center gap-1.5">
                        <StatusDot status={c.status} size={7} />
                        <span className="truncate text-muted">{c.name}</span>
                      </span>
                    ) : (
                      <span className="text-dim">Unassigned</span>
                    )}
                    <span className={cx("ml-auto shrink-0 truncate", st.tone)}>{st.text}</span>
                  </div>
                </button>
              </li>
            );
          })}
        </ul>
        <div className="hidden overflow-x-auto md:block">
          <table className="w-full min-w-[900px] text-[12.5px]">
            <thead>
              <tr className="border-b border-line text-left text-[11px] text-dim">
                <th className="w-10 px-3.5 py-2 font-medium" />
                <th className="px-2 py-2 font-medium">Page</th>
                <th className="px-2 py-2 font-medium">Room</th>
                <th className="px-2 py-2 font-medium">Alert</th>
                <th className="px-2 py-2 font-medium">Source</th>
                <th className="px-2 py-2 font-medium">Assigned</th>
                <th className="px-2 py-2 font-medium">Status</th>
                <th className="px-3.5 py-2 text-right font-medium">Age</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => {
                const c = p.assignee ? clinicians[p.assignee] : p.planned ? clinicians[p.planned] : null;
                const st = statusLine(p, now);
                return (
                  <tr key={p.id} onClick={() => onOpen(p.id)} className="cursor-pointer border-b border-line-soft transition-colors last:border-b-0 hover:bg-s2">
                    <td className="px-3.5 py-2">
                      <PriorityChip p={p.priority} size="sm" />
                    </td>
                    <td className="px-2 py-2 font-mono text-[11.5px] text-dim">{p.id}</td>
                    <td className="px-2 py-2">
                      <div className="font-mono font-medium text-ink">{p.room ?? "—"}</div>
                      <div className="text-[11px] text-dim">{p.zone_name}</div>
                    </td>
                    <td className="max-w-[380px] px-2 py-2">
                      <div className="truncate text-ink/90">{p.text}</div>
                      <div className="truncate text-[11px] text-dim">
                        {CATEGORY_LABEL[p.category]}
                        {p.patient?.name ? ` · ${p.patient.name}` : ""}
                      </div>
                    </td>
                    <td className="px-2 py-2">
                      <SourceTag source={p.source} synthetic={p.synthetic} />
                    </td>
                    <td className="px-2 py-2">
                      {c ? (
                        <span className="inline-flex items-center gap-2">
                          <StatusDot status={c.status} size={7} />
                          <span className="text-ink/90">{c.name}</span>
                        </span>
                      ) : (
                        <span className="text-dim">—</span>
                      )}
                    </td>
                    <td className="max-w-[220px] px-2 py-2">
                      <span className={cx("block truncate text-[12px]", st.tone)}>{st.text}</span>
                    </td>
                    <td className="num px-3.5 py-2 text-right font-mono text-[11.5px] text-dim">{age(p.created_at, now)}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        </>
      )}
    </Panel>
  );
}

export { StatusDot, StatusText };
