"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { Button, cx, Drawer, Empty, Segmented, Switch } from "@/components/ui";
import { api } from "@/lib/api";
import {
  CATEGORY_LABEL,
  PAGE_STATUS_LABEL,
  since,
  STATUS_LABEL,
  type Clinician,
  type Layout,
  type Monitor,
  type Occupant,
  type Page,
  type Rec,
  type Settings,
  type StaffStatus,
  type Stats,
} from "@/lib/paging";

import { Avatar, P_RGB, PriorityChip, SourceTag, StatusDot, StatusText } from "./bits";

/* ------------------------------------------------------------------ */
/* KPI strip                                                           */
/* ------------------------------------------------------------------ */
export function KpiStrip({ stats, pages, monitor }: { stats: Stats | null; pages: Page[]; monitor: Monitor }) {
  const p1 = pages.filter((p) => p.priority === 1).length;
  const review = pages.filter((p) => p.status === "review" || p.status === "unassigned").length;
  const watch = Object.values(monitor).filter((m) => m.watch).length;
  const ack = stats?.median_ack_s ?? {};
  const fastAck = [ack["1"], ack["2"]].filter((x): x is number => x != null);
  const items = [
    { k: "Open pages", v: pages.length, sub: `${pages.filter((p) => p.status === "paging").length} ringing` },
    { k: "P1 open", v: p1, tone: p1 ? "text-high-ink" : undefined, sub: p1 ? "immediate" : "none" },
    { k: "Need you", v: review, tone: review ? "text-dys-ink" : undefined, sub: "review or unassigned" },
    { k: "Time to accept", v: fastAck.length ? `${Math.round(Math.min(...fastAck))}s` : "—", sub: "median, P1/P2" },
    { k: "Escalation rate", v: stats?.escalation_rate != null ? `${Math.round(stats.escalation_rate * 100)}%` : "—", sub: `${stats?.timeouts ?? 0} timeouts · ${stats?.declines ?? 0} declines` },
    { k: "Monitored beds", v: stats?.monitored_beds ?? Object.keys(monitor).length, sub: watch ? `${watch} on sepsis watch` : "detector live" },
  ];
  return (
    <div className="grid grid-cols-3 divide-x divide-line-soft rounded-xl border border-line bg-s1 shadow-[var(--shadow-card)] xl:grid-cols-6">
      {items.map((it) => (
        <div key={it.k} className="px-4 py-2.5">
          <div className="text-[11px] text-dim">{it.k}</div>
          <div className={cx("num text-[20px] font-semibold leading-tight tracking-tight", it.tone ?? "text-ink")}>{it.v}</div>
          <div className="truncate text-[11px] text-dim">{it.sub}</div>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Room drawer                                                         */
/* ------------------------------------------------------------------ */
type RoomDetail = {
  room: string;
  zone_name: string;
  floor: string;
  patient: null | Record<string, unknown>;
  pages: Page[];
  alerts: { id: string; text: string; created_at: number; status: string; priority: number | null }[];
  episode: null | { open: boolean };
};

export function RoomDrawer({ room, onClose, now, onRaise, onOpenPage }: {
  room: string | null;
  onClose: () => void;
  now: number;
  onRaise: (room: string) => void;
  onOpenPage: (id: string) => void;
}) {
  const [d, setD] = useState<RoomDetail | null>(null);
  useEffect(() => {
    if (!room) return;
    setD(null);
    let alive = true;
    const load = () => api<RoomDetail>(`/aegis/paging/rooms/${room}`).then((r) => alive && setD(r)).catch(() => {});
    load();
    const t = setInterval(load, 4000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, [room]);
  const p = (d?.patient ?? null) as null | Record<string, any>; // eslint-disable-line @typescript-eslint/no-explicit-any
  return (
    <Drawer open={!!room} onClose={onClose} width={440}>
      <header className="flex items-start gap-3 px-5 pb-3 pt-5">
        <div className="min-w-0 flex-1">
          <div className="font-mono text-[11px] uppercase tracking-[0.12em] text-dim">{d ? `Floor ${d.floor} · ${d.zone_name}` : "Room"}</div>
          <h2 className="serif mt-1 text-[30px] leading-none text-ink">Room {room}</h2>
        </div>
        <button onClick={onClose} className="grid h-8 w-8 place-items-center rounded-lg text-dim hover:bg-s2 hover:text-ink" aria-label="Close">
          <svg width="14" height="14" viewBox="0 0 14 14"><path d="M3 3l8 8M11 3l-8 8" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>
        </button>
      </header>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto px-5 pb-5">
        {!d ? (
          <div className="h-40 animate-pulse rounded-xl bg-s2" />
        ) : !p ? (
          <Empty>Unoccupied.</Empty>
        ) : (
          <div className="rounded-xl border border-line p-4">
            <div className="flex items-baseline gap-2">
              <span className="text-[16px] font-semibold text-ink">{p.name}</span>
              <span className="text-[13px] text-dim">
                {p.age ? `${p.age}${p.sex ?? ""}` : ""} {p.mrn ? `· ${p.mrn}` : ""}
              </span>
            </div>
            <div className="mt-0.5 text-[12.5px] text-muted">{p.diagnosis ?? (p.monitored ? "Continuously monitored" : "")}</div>
            <div className={cx("mt-2 inline-flex rounded-md px-2 py-0.5 text-[11px] font-medium", p.synthetic ? "bg-s2 text-dim" : "bg-accent-soft text-accent-ink")}>
              {p.synthetic ? "Synthetic demo record" : String(p.source ?? "real record")}
            </div>
            {p.monitored && (
              <div className="mt-3 grid grid-cols-3 gap-3 border-t border-line-soft pt-3">
                <div>
                  <div className="text-[11px] text-dim">Sepsis risk</div>
                  <div className={cx("num text-[20px] font-semibold", (p.risk ?? 0) >= 0.7 ? "text-high-ink" : (p.risk ?? 0) >= 0.6 ? "text-med-ink" : "text-ink")}>
                    {p.risk != null ? Number(p.risk).toFixed(2) : "—"}
                  </div>
                </div>
                <div>
                  <div className="text-[11px] text-dim">MAP</div>
                  <div className="num text-[20px] font-semibold text-ink">{p.vitals?.map != null ? Math.round(p.vitals.map) : "—"}</div>
                </div>
                <div>
                  <div className="text-[11px] text-dim">SpO₂</div>
                  <div className="num text-[20px] font-semibold text-ink">{p.vitals?.spo2 != null ? Math.round(p.vitals.spo2) : "—"}</div>
                </div>
                {Array.isArray(p.drivers) && p.drivers.length > 0 && (
                  <ul className="col-span-3 space-y-0.5 text-[12px] text-muted">
                    {p.drivers.slice(0, 3).map((x: string) => (
                      <li key={x}>· {x}</li>
                    ))}
                  </ul>
                )}
              </div>
            )}
            {!p.monitored && (
              <div className="mt-3 space-y-1 border-t border-line-soft pt-3 text-[12px] text-muted">
                {p.history?.length > 0 && <div><span className="text-dim">History </span>{p.history.join(", ")}</div>}
                {p.medications?.length > 0 && <div><span className="text-dim">Meds </span>{p.medications.join(", ")}</div>}
                <div><span className="text-dim">Allergies </span>{p.allergies?.length ? p.allergies.join(", ") : "NKDA"}</div>
                {p.code_status && <div><span className="text-dim">Code status </span>{p.code_status}</div>}
              </div>
            )}
            {p.bed && (
              <Link href={`/patient/?id=${encodeURIComponent(p.bed)}`} className="mt-3 flex items-center justify-between rounded-lg bg-s2 px-3 py-2 text-[12.5px] font-medium text-ink hover:bg-s3">
                Open the patient: twin, trend, orders <span aria-hidden>→</span>
              </Link>
            )}
          </div>
        )}
        <Button variant="primary" className="w-full" onClick={() => room && onRaise(room)}>
          Raise an alert for this room
        </Button>
        {d && d.pages.length > 0 && (
          <div>
            <div className="mb-2 font-mono text-[10.5px] uppercase tracking-[0.12em] text-dim">Pages</div>
            <div className="space-y-1.5">
              {[...d.pages].reverse().map((pg) => (
                <button key={pg.id} onClick={() => onOpenPage(pg.id)} className="flex w-full items-center gap-2 rounded-lg border border-line px-3 py-2 text-left hover:bg-s2">
                  <PriorityChip p={pg.priority} size="sm" />
                  <span className="min-w-0 flex-1 truncate text-[12.5px] text-ink">{pg.text}</span>
                  <span className="shrink-0 text-[11px] text-dim">{PAGE_STATUS_LABEL[pg.status]}</span>
                </button>
              ))}
            </div>
          </div>
        )}
        {d && d.alerts.length > 0 && (
          <div>
            <div className="mb-2 font-mono text-[10.5px] uppercase tracking-[0.12em] text-dim">History</div>
            <ul className="space-y-1.5 text-[12px]">
              {[...d.alerts].reverse().map((a) => (
                <li key={a.id} className="flex gap-2 text-muted">
                  <span className="shrink-0 text-dim">{since(a.created_at, now)}</span>
                  <span className="truncate">{a.text}</span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>
    </Drawer>
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
  return (
    <div className="grid gap-4 md:grid-cols-2 2xl:grid-cols-3">
      {groups.map(([spec, staff]) => (
        <div key={spec} className="plate p-4">
          <div className="mb-3 flex items-center gap-2">
            <span className="text-[13px] font-semibold text-ink">{spec}</span>
            <span className="text-[11.5px] text-dim">
              {staff.filter((c) => c.status === "available").length}/{staff.length} available
            </span>
          </div>
          <div className="space-y-2">
            {staff.map((c) => {
              const mine = load(c.id);
              const where = layout?.zones[c.zone]?.name ?? (c.zone === "remote" ? "Telehealth" : c.zone);
              return (
                <div key={c.id} className="flex items-center gap-3 rounded-lg px-1 py-1">
                  <Avatar name={c.name} status={c.status} size={32} present={c.present} />
                  <div className="min-w-0 flex-1">
                    <div className="flex items-center gap-1.5">
                      <span className="truncate text-[13px] font-medium text-ink">{c.name}</span>
                      {c.on_call && <span className="rounded bg-accent-soft px-1 text-[10px] font-medium text-accent-ink">on call</span>}
                      {!c.pageable && <span className="rounded bg-s2 px-1 text-[10px] text-dim">operator</span>}
                    </div>
                    <div className="truncate text-[11.5px] text-dim">
                      {STATUS_LABEL[c.status]}
                      {c.busy_until && (c.status === "in_procedure" || c.status === "on_case") ? ` · free in ~${Math.max(1, Math.round((c.busy_until - now) / 60))} min` : ""} · {where}
                      {mine.length ? ` · ${mine.length} page${mine.length > 1 ? "s" : ""}` : ""}
                    </div>
                  </div>
                  {c.pageable && (
                    <Link href={`/clinician/?id=${c.id}`} className="rounded-md px-2 py-1 text-[11.5px] text-accent-ink hover:bg-accent-soft" title="Open this clinician's pager">
                      Pager
                    </Link>
                  )}
                  {canDispatch && c.pageable && (
                    <select
                      disabled={busy === c.id}
                      value={c.status}
                      onChange={(e) => setStatus(c.id, e.target.value as StaffStatus)}
                      className="h-7 rounded-md border border-line bg-s1 px-1.5 text-[11.5px] text-muted"
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
              );
            })}
          </div>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Insights: the Sentinel + how dispatch is doing                      */
/* ------------------------------------------------------------------ */
export function Insights({ recs, stats, canDispatch, now }: { recs: Rec[]; stats: Stats | null; canDispatch: boolean; now: number }) {
  const ack = async (id: string) => api(`/aegis/paging/recs/${encodeURIComponent(id)}/ack`, { body: {} }).catch(() => {});
  const sev = { critical: "border-high/30 bg-high/[.04]", warn: "border-dys/25 bg-dys/[.035]", info: "border-line bg-s1" };
  const byP = stats?.by_priority ?? {};
  const maxP = Math.max(1, ...Object.values(byP));
  return (
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
      <div>
        <div className="mb-3 flex items-baseline gap-2">
          <h3 className="serif text-[24px] text-ink">Sentinel</h3>
          <span className="text-[12.5px] text-dim">Watches the whole paging system every 30 seconds. It never pages anyone.</span>
        </div>
        {recs.length === 0 ? (
          <div className="plate">
            <Empty>No systemic risk right now: no surges, coverage holes, overload or clusters.</Empty>
          </div>
        ) : (
          <div className="space-y-3">
            {recs.map((r) => (
              <div key={r.id} className={cx("rounded-xl border p-4 transition", sev[r.severity], r.acknowledged && "opacity-60")}>
                <div className="flex items-start gap-3">
                  <span className={cx("mt-1 h-2 w-2 shrink-0 rounded-full", r.severity === "critical" ? "bg-high" : r.severity === "warn" ? "bg-dys" : "bg-accent")} />
                  <div className="min-w-0 flex-1">
                    <div className="text-[14px] font-semibold text-ink">{r.title}</div>
                    <p className="mt-1 text-[12.5px] leading-relaxed text-muted">{r.detail}</p>
                    <p className="mt-2 text-[12.5px] font-medium text-ink">→ {r.action}</p>
                    {r.evidence.length > 0 && (
                      <div className="mt-2 flex flex-wrap gap-1.5">
                        {r.evidence.map((e) => (
                          <span key={e} className="rounded bg-s2 px-1.5 py-0.5 font-mono text-[10.5px] text-dim">
                            {e}
                          </span>
                        ))}
                      </div>
                    )}
                    <div className="mt-2 text-[11px] text-dim">
                      {r.kind.replace("_", " ")} · since {since(r.created_at, now)}
                    </div>
                  </div>
                  {canDispatch && !r.acknowledged && (
                    <Button size="sm" variant="ghost" onClick={() => ack(r.id)}>
                      Acknowledge
                    </Button>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
      <div className="space-y-4">
        <div className="plate p-4">
          <div className="mb-3 text-[13px] font-semibold text-ink">Pages by priority</div>
          <div className="space-y-2">
            {[1, 2, 3, 4].map((p) => (
              <div key={p} className="flex items-center gap-3">
                <PriorityChip p={p} size="sm" />
                <div className="h-2 flex-1 overflow-hidden rounded-full bg-s2">
                  <div className="h-full rounded-full" style={{ width: `${((byP[String(p)] ?? 0) / maxP) * 100}%`, background: `rgb(${P_RGB[p]})` }} />
                </div>
                <span className="num w-8 text-right font-mono text-[12px] text-muted">{byP[String(p)] ?? 0}</span>
              </div>
            ))}
          </div>
        </div>
        <div className="plate grid grid-cols-2 gap-4 p-4">
          {[
            ["Median time to accept · P1", stats?.median_ack_s?.["1"] != null ? `${Math.round(stats.median_ack_s["1"]!)}s` : "—"],
            ["Median time to accept · P2", stats?.median_ack_s?.["2"] != null ? `${Math.round(stats.median_ack_s["2"]!)}s` : "—"],
            ["Median dispatch", stats?.median_dispatch_ms != null ? `${stats.median_dispatch_ms} ms` : "—"],
            ["Pages certified by Z3", stats ? `${stats.certificates}${stats.blocked ? ` · ${stats.blocked} held` : ""}` : "—"],
            ["From the detector", stats?.by_source?.detector ?? 0],
            ["From nurses (typed / voice)", stats ? `${stats.by_source?.nurse ?? 0} / ${stats.by_source?.voice ?? 0}` : "—"],
          ].map(([k, v]) => (
            <div key={String(k)}>
              <div className="text-[11px] text-dim">{k}</div>
              <div className="num text-[18px] font-semibold text-ink">{v}</div>
            </div>
          ))}
          <p className="col-span-2 text-[11px] leading-snug text-faint">
            Since this server started. Demo traffic and simulated responders are synthetic; detector pages come from replayed real ICU records.
          </p>
        </div>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Policy                                                              */
/* ------------------------------------------------------------------ */
const UNIT_LABEL: Record<string, string> = { icu: "ICU", ed: "Emergency", stepdown: "Step-down / cardiac", ward: "Wards", obs: "Observation", pacu: "Recovery (PACU)", or: "Operating rooms" };

export function PolicyPanel({ settings, invariants, canConfigure, stats }: {
  settings: Settings | null;
  invariants: { id: string; text: string }[];
  canConfigure: boolean;
  stats: Stats | null;
}) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  if (!settings) return <div className="h-64 animate-pulse rounded-xl bg-s2" />;
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
    <div className="grid gap-5 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
      <div className="space-y-5">
        {!canConfigure && (
          <div className="rounded-xl border border-line bg-s2/60 px-4 py-3 text-[12.5px] text-muted">
            Read-only. Changing the paging policy needs <span className="font-mono text-[11.5px]">page:configure</span> (the chief of critical care). It is governance, like the safety policy.
          </div>
        )}
        {err && <div className="rounded-lg bg-high/8 px-3 py-2 text-[12.5px] text-high-ink">{err}</div>}
        <div className="plate overflow-hidden">
          <div className="border-b border-line-soft px-4 py-3">
            <div className="text-[13px] font-semibold text-ink">Autonomy</div>
            <div className="text-[12px] text-dim">Which pages go out on their own, and which the operator confirms first. Review never blocks care: an unreviewed P1 sends itself after {settings.review_failsafe_s["1"]}s.</div>
          </div>
          <table className="w-full text-[12.5px]">
            <thead>
              <tr className="text-left text-[11px] text-dim">
                <th className="px-4 py-2 font-normal">Unit</th>
                {[1, 2, 3, 4].map((p) => (
                  <th key={p} className="px-2 py-2 text-center font-normal">
                    <PriorityChip p={p} size="sm" />
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {Object.entries(settings.autonomy).map(([kind, row]) => (
                <tr key={kind} className="border-t border-line-soft">
                  <td className="px-4 py-2 text-ink">{UNIT_LABEL[kind] ?? kind}</td>
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
                          "h-7 w-[72px] rounded-md text-[11.5px] font-medium transition disabled:cursor-default",
                          mode === "review" ? "bg-ink text-white" : "bg-s2 text-muted hover:bg-s3",
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
          <div className="flex items-center gap-3 border-t border-line-soft px-4 py-3">
            <Switch checked={settings.detector_p1_review} disabled={!canConfigure || busy} onChange={(v) => save({ detector_p1_review: v })} label="Machine-raised P1 review" />
            <span className="text-[12.5px] text-muted">A P1 raised by the sepsis detector always gets a human look first</span>
          </div>
        </div>
        <div className="plate p-4">
          <div className="mb-3 text-[13px] font-semibold text-ink">Guardrails</div>
          <div className="grid gap-3 sm:grid-cols-2">
            {[
              ["Ring time before escalation", ["1", "2", "3", "4"].map((p) => `P${p} ${settings.ack_timeout_s[p]}s`).join(" · ")],
              ["Consecutive-paging window", `${Math.round(settings.consecutive_window_s / 60)} min`],
              ["P3/P4 workload limit", `${settings.workload_max_per_hour} pages / hour`],
              ["Near-tie band", `${Math.round(settings.near_tie_pct * 100)}%`],
              ["Hold for the right person", `up to ${Math.round(settings.hold_window_s / 60)} min`],
              ["Escalations before the operator", `${settings.max_escalations}`],
            ].map(([k, v]) => (
              <div key={k} className="rounded-lg bg-s2/60 px-3 py-2">
                <div className="text-[11px] text-dim">{k}</div>
                <div className="num text-[13px] font-medium text-ink">{v}</div>
              </div>
            ))}
          </div>
          <div className="mt-4 space-y-2 border-t border-line-soft pt-3">
            <div className="flex items-center gap-3">
              <Switch checked={settings.simulated_responders} disabled={!canConfigure || busy} onChange={(v) => save({ simulated_responders: v })} label="Simulated responders" />
              <span className="text-[12.5px] text-muted">Demo: staff with no pager open answer on their own</span>
            </div>
            <div className="flex items-center gap-3">
              <Switch checked={settings.demo_traffic} disabled={!canConfigure || busy} onChange={(v) => save({ demo_traffic: v })} label="Demo traffic" />
              <span className="text-[12.5px] text-muted">Demo: a synthetic nurse alert every couple of minutes</span>
            </div>
          </div>
        </div>
      </div>
      <div className="plate h-fit p-5">
        <div className="flex items-baseline gap-2">
          <h3 className="serif text-[24px] text-ink">Proven on every page</h3>
        </div>
        <p className="mt-1 text-[12.5px] leading-relaxed text-muted">
          Before a page leaves, Z3 is asked whether the decision could violate any of these. <span className="font-medium text-ink">unsat</span> means it cannot, and the page is sent. Anything else holds it for the operator with the broken rule named. If the prover itself is unhealthy, nothing is sent.
        </p>
        <ol className="mt-4 space-y-2.5">
          {invariants.map((inv, i) => (
            <li key={inv.id} className="flex gap-3 text-[12.5px] leading-snug">
              <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-low/12 font-mono text-[10px] font-semibold text-low-ink">{i + 1}</span>
              <span className="text-muted">{inv.text}</span>
            </li>
          ))}
        </ol>
        {stats && (
          <div className="mt-4 rounded-lg bg-s2/60 px-3 py-2 text-[12px] text-muted">
            <span className="num font-semibold text-ink">{stats.certificates}</span> pages certified since boot
            {stats.blocked ? <>, <span className="num font-semibold text-high-ink">{stats.blocked}</span> held by the prover</> : ", none refused"}.
          </div>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Cases                                                               */
/* ------------------------------------------------------------------ */
export function CasesTable({ pages, clinicians, now, onOpen }: {
  pages: Record<string, Page>;
  clinicians: Record<string, Clinician>;
  now: number;
  onOpen: (id: string) => void;
}) {
  const [filter, setFilter] = useState<"open" | "p1" | "detector" | "closed">("open");
  const rows = useMemo(() => {
    const all = Object.values(pages).sort((a, b) => b.created_at - a.created_at);
    const open = (p: Page) => !["resolved", "cancelled"].includes(p.status);
    if (filter === "open") return all.filter(open).sort((a, b) => a.priority - b.priority || b.created_at - a.created_at);
    if (filter === "p1") return all.filter((p) => p.priority === 1);
    if (filter === "detector") return all.filter((p) => p.source === "detector");
    return all.filter((p) => !open(p));
  }, [pages, filter]);
  return (
    <div className="plate shrink-0 overflow-hidden">
      <div className="flex items-center gap-3 border-b border-line-soft px-4 py-3">
        <Segmented
          value={filter}
          onChange={setFilter}
          options={[
            { value: "open", label: "Open" },
            { value: "p1", label: "P1" },
            { value: "detector", label: "Detector" },
            { value: "closed", label: "Closed" },
          ]}
        />
        <span className="ml-auto text-[12px] text-dim">{rows.length} pages</span>
      </div>
      {rows.length === 0 ? (
        <Empty>Nothing here.</Empty>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full min-w-[860px] text-[12.5px]">
            <thead>
              <tr className="text-left text-[11px] text-dim">
                {["", "Room", "Alert", "Source", "Assigned", "Status", "Age", ""].map((h, i) => (
                  <th key={i} className="px-3 py-2 font-normal">
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {rows.map((p) => {
                const c = p.assignee ? clinicians[p.assignee] : p.planned ? clinicians[p.planned] : null;
                return (
                  <tr key={p.id} onClick={() => onOpen(p.id)} className="cursor-pointer border-t border-line-soft hover:bg-s2/60">
                    <td className="px-3 py-2.5"><PriorityChip p={p.priority} size="sm" /></td>
                    <td className="px-3 py-2.5">
                      <div className="font-mono font-medium text-ink">{p.room ?? "—"}</div>
                      <div className="text-[11px] text-dim">{p.zone_name}</div>
                    </td>
                    <td className="max-w-[360px] px-3 py-2.5">
                      <div className="truncate text-ink">{p.text}</div>
                      <div className="text-[11px] text-dim">{CATEGORY_LABEL[p.category]}{p.patient?.name ? ` · ${p.patient.name}` : ""}</div>
                    </td>
                    <td className="px-3 py-2.5"><SourceTag source={p.source} synthetic={p.synthetic} /></td>
                    <td className="px-3 py-2.5">
                      {c ? (
                        <span className="inline-flex items-center gap-2">
                          <Avatar name={c.name} size={20} status={c.status} />
                          <span className="text-ink">{c.name}</span>
                        </span>
                      ) : (
                        <span className="text-dim">—</span>
                      )}
                    </td>
                    <td className="px-3 py-2.5">
                      <span className={cx("text-[12px]", p.status === "unassigned" ? "text-high-ink" : p.status === "review" ? "text-dys-ink" : "text-muted")}>{PAGE_STATUS_LABEL[p.status]}</span>
                      {p.escalations > 0 && <span className="ml-1.5 text-[11px] text-dim">×{p.escalations}</span>}
                    </td>
                    <td className="num px-3 py-2.5 font-mono text-[11.5px] text-dim">{since(p.created_at, now)}</td>
                    <td className="px-3 py-2.5 text-dim">→</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}

export { StatusDot, StatusText };
