"use client";

import Link from "next/link";
import { Fragment, useEffect, useMemo, useState } from "react";

import { Button, CloseIcon, cx, Drawer, Empty, IconButton, Kv, Panel, Segmented, Switch, Tag } from "@/components/ui";
import { api } from "@/lib/api";
import {
  CATEGORY_LABEL,
  clock,
  PAGE_STATUS_LABEL,
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
export function KpiStrip({ stats, pages, monitor }: { stats: Stats | null; pages: Page[]; monitor: Monitor }) {
  const p1 = pages.filter((p) => p.priority === 1).length;
  const review = pages.filter((p) => p.status === "review" || p.status === "unassigned").length;
  const watch = Object.values(monitor).filter((m) => m.watch).length;
  const ack = stats?.median_ack_s ?? {};
  const fastAck = [ack["1"], ack["2"]].filter((x): x is number => x != null);
  const items: { k: string; v: string | number; sub: string; tone?: string }[] = [
    { k: "Active pages", v: pages.length, sub: `${pages.filter((p) => p.status === "paging").length} ringing` },
    { k: "P1 active", v: p1, tone: p1 ? "text-high-ink" : undefined, sub: p1 ? "immediate response" : "none" },
    { k: "Awaiting operator", v: review, tone: review ? "text-dys-ink" : undefined, sub: "review or unassigned" },
    { k: "Time to accept", v: fastAck.length ? `${Math.round(Math.min(...fastAck))}s` : "—", sub: "median, P1/P2" },
    { k: "Escalation rate", v: stats?.escalation_rate != null ? `${Math.round(stats.escalation_rate * 100)}%` : "—", sub: `${stats?.timeouts ?? 0} timeouts · ${stats?.declines ?? 0} declines` },
    { k: "Monitored beds", v: stats?.monitored_beds ?? Object.keys(monitor).length, sub: watch ? `${watch} on sepsis watch` : "detector active" },
  ];
  return (
    <div className="grid shrink-0 grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-3 xl:grid-cols-6">
      {items.map((it) => (
        <div key={it.k} className="min-w-0 bg-s1 px-3.5 py-2.5">
          <div className="truncate text-[11px] text-dim">{it.k}</div>
          <div className={cx("num mt-0.5 text-[20px] font-medium leading-tight tracking-[-0.01em]", it.tone ?? "text-ink")}>{it.v}</div>
          <div className="truncate text-[11px] text-dim">{it.sub}</div>
        </div>
      ))}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Room inspector                                                      */
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
      <header className="flex shrink-0 items-start gap-3 border-b border-line px-4 py-3.5">
        <div className="min-w-0 flex-1">
          <div className="text-[11.5px] text-dim">{d ? `Floor ${d.floor} · ${d.zone_name}` : "Room"}</div>
          <h2 className="mt-0.5 font-mono text-[18px] font-semibold text-ink">Room {room}</h2>
        </div>
        <IconButton label="Close" onClick={onClose}>
          <CloseIcon />
        </IconButton>
      </header>
      <div className="min-h-0 flex-1 space-y-4 overflow-y-auto p-4">
        {!d ? (
          <div className="h-40 animate-pulse rounded-md bg-s2" />
        ) : !p ? (
          <Empty>Unoccupied.</Empty>
        ) : (
          <div className="rounded-md border border-line">
            <div className="px-3 py-2.5">
              <div className="flex items-baseline gap-2">
                <span className="text-[14px] font-semibold text-ink">{p.name}</span>
                <span className="text-[12px] text-dim">{p.age ? `${p.age}${p.sex ?? ""}` : ""}</span>
                {p.mrn && <span className="font-mono text-[11px] text-faint">{p.mrn}</span>}
                <span className="ml-auto">
                  <Tag tone={p.synthetic ? "neutral" : "low"}>{p.synthetic ? "Synthetic record" : "Replayed real record"}</Tag>
                </span>
              </div>
              <div className="mt-0.5 text-[12px] text-muted">{p.diagnosis ?? (p.monitored ? "Continuously monitored" : "")}</div>
            </div>
            {p.monitored && (
              <div className="grid grid-cols-3 gap-px border-t border-line bg-line">
                {[
                  ["Sepsis risk", p.risk != null ? Number(p.risk).toFixed(2) : "—", (p.risk ?? 0) >= 0.7 ? "text-high-ink" : (p.risk ?? 0) >= 0.6 ? "text-med-ink" : "text-ink"],
                  ["MAP", p.vitals?.map != null ? Math.round(p.vitals.map) : "—", "text-ink"],
                  ["SpO₂", p.vitals?.spo2 != null ? Math.round(p.vitals.spo2) : "—", "text-ink"],
                ].map(([k, v, tone]) => (
                  <div key={String(k)} className="bg-s1 px-3 py-2">
                    <div className="text-[11px] text-dim">{k}</div>
                    <div className={cx("num font-mono text-[16px] font-medium", String(tone))}>{v}</div>
                  </div>
                ))}
              </div>
            )}
            {p.monitored && Array.isArray(p.drivers) && p.drivers.length > 0 && (
              <ul className="space-y-0.5 border-t border-line px-3 py-2 text-[12px] text-muted">
                {p.drivers.slice(0, 3).map((x: string) => (
                  <li key={x}>{x}</li>
                ))}
              </ul>
            )}
            {!p.monitored && (
              <div className="space-y-0.5 border-t border-line px-3 py-2">
                {p.history?.length > 0 && <Kv k="History" v={<span className="whitespace-normal">{p.history.join(", ")}</span>} />}
                {p.medications?.length > 0 && <Kv k="Medications" v={<span className="whitespace-normal">{p.medications.join(", ")}</span>} />}
                <Kv k="Allergies" v={p.allergies?.length ? p.allergies.join(", ") : "NKDA"} />
                {p.code_status && <Kv k="Code status" v={p.code_status} />}
              </div>
            )}
            {p.bed && (
              <Link href={`/patient/?id=${encodeURIComponent(p.bed)}`} className="flex items-center justify-between border-t border-line px-3 py-2 text-[12.5px] text-ink transition-colors hover:bg-s2">
                Open patient record <span className="text-dim">↗</span>
              </Link>
            )}
          </div>
        )}
        <Button variant="primary" className="w-full" onClick={() => room && onRaise(room)}>
          Raise an alert for this room
        </Button>
        {d && d.pages.length > 0 && (
          <div>
            <div className="mb-1.5 text-[11.5px] font-medium text-dim">Pages</div>
            <div className="overflow-hidden rounded-md border border-line">
              {[...d.pages].reverse().map((pg) => (
                <button key={pg.id} onClick={() => onOpenPage(pg.id)} className="flex w-full items-center gap-2 border-b border-line-soft px-3 py-2 text-left transition-colors last:border-b-0 hover:bg-s2">
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
            <div className="mb-1.5 text-[11.5px] font-medium text-dim">History</div>
            <ul className="space-y-1 text-[12px]">
              {[...d.alerts].reverse().map((a) => (
                <li key={a.id} className="grid grid-cols-[44px_minmax(0,1fr)] gap-2">
                  <span className="num font-mono text-[11px] text-dim">{clock(a.created_at, false)}</span>
                  <span className="truncate text-muted">{a.text}</span>
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
  const total = Object.values(clinicians);
  return (
    <Panel
      title="Staff on the roster"
      meta={`${total.filter((c) => c.status === "available").length} available · ${total.filter((c) => c.status === "paging").length} being paged · ${total.filter((c) => c.status === "in_procedure" || c.status === "on_case").length} busy · ${total.filter((c) => c.status === "off_shift").length} off shift`}
      flush
      className="shrink-0"
    >
      <div className="overflow-x-auto">
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
    <div className="grid shrink-0 gap-3 xl:grid-cols-[minmax(0,1fr)_360px]">
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
    <div className="grid shrink-0 gap-3 xl:grid-cols-[minmax(0,1.2fr)_minmax(0,1fr)]">
      <div className="space-y-3">
        {!canConfigure && (
          <div className="flex items-center gap-2 rounded-lg border border-line bg-s1 px-3.5 py-2.5 text-[12.5px] text-muted">
            <Tag tone="neutral">Read only</Tag>
            Changing the paging policy requires <span className="font-mono text-[11.5px] text-ink">page:configure</span> (chief of critical care).
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
            <Switch checked={settings.detector_p1_review} disabled={!canConfigure || busy} onChange={(v) => save({ detector_p1_review: v })} label="Detector P1 review" />
            <span className="text-[12.5px] text-muted">A P1 raised by the sepsis detector always waits for a person first</span>
          </div>
        </Panel>
        <Panel title="Guardrails" meta="Applied by the Case Handler as named steps on every routing decision">
          <div className="grid gap-x-6 sm:grid-cols-2">
            <Kv k="Ring time before escalation" v={["1", "2", "3", "4"].map((p) => `P${p} ${settings.ack_timeout_s[p]}s`).join(" · ")} />
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
              <span className="text-[12.5px] text-muted">A synthetic nurse alert every couple of minutes</span>
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
  const [filter, setFilter] = useState<"open" | "p1" | "detector" | "closed" | "all">("open");
  const rows = useMemo(() => {
    const all = Object.values(pages).sort((a, b) => b.created_at - a.created_at);
    const open = (p: Page) => !["resolved", "cancelled"].includes(p.status);
    if (filter === "open") return all.filter(open).sort((a, b) => a.priority - b.priority || b.created_at - a.created_at);
    if (filter === "p1") return all.filter((p) => p.priority === 1);
    if (filter === "detector") return all.filter((p) => p.source === "detector");
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
        <div className="overflow-x-auto">
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
      )}
    </Panel>
  );
}

export { StatusDot, StatusText };
