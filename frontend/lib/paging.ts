"use client";

import { useEffect, useMemo, useReducer, useRef, useState } from "react";

import { api, wsProtocols, wsUrl } from "./api";
import type { RoomStatus } from "./rooms";

/* ------------------------------------------------------------------ */
/* Contracts (backend/aegis/paging)                                    */
/* ------------------------------------------------------------------ */
export type Priority = 1 | 2 | 3 | 4;
export type PageStatus = "new" | "review" | "held" | "paging" | "accepted" | "resolved" | "cancelled" | "unassigned";
export type StaffStatus = "available" | "paging" | "in_procedure" | "on_case" | "off_shift";
export type Source = "nurse" | "voice" | "detector" | "monitor" | "demo";

export type Layout = {
  name: string;
  note: string;
  plate: { w: number; d: number };
  floors: { id: string; label: string; index: number; zones: string[] }[];
  zones: Record<string, { id: string; name: string; kind: string; floor: string; rect: number[]; rooms: string[]; critical: boolean }>;
  rooms: Record<string, { id: string; zone: string; floor: string; rect: number[] }>;
  specialties: Record<string, string>;
};

export type Clinician = {
  id: string;
  name: string;
  role: string;
  specialty: string;
  specialty_label: string;
  home_zone: string;
  zone: string;
  title: string;
  on_call: boolean;
  shift: string;
  status: StaffStatus;
  busy_until: number | null;
  pageable: boolean;
  present: boolean;
};

export type Attempt = {
  clinician: string;
  name: string;
  specialty: string;
  sent_at: number;
  deadline: number;
  outcome: "accepted" | "declined" | "timeout" | "escalated" | "cancelled" | "rerouted" | "off_shift" | null;
  at: number | null;
  reason: string | null;
};

export type Candidate = {
  id: string;
  name: string;
  specialty: string;
  specialty_label: string;
  status: StaffStatus;
  zone: string;
  score: number;
  factors: { label: string; points: number }[];
  excluded: string | null;
  caseload: number;
  pages_last_hour: number;
};

export type Page = {
  id: string;
  alert_id: string;
  priority: Priority;
  category: string;
  room: string | null;
  zone: string | null;
  floor: string | null;
  zone_name: string | null;
  bed: string | null;
  patient: null | { name?: string; age?: number; sex?: string; mrn?: string; diagnosis?: string; monitored?: boolean; synthetic?: boolean; source?: string; risk?: number; bed?: string | null; live?: boolean };
  text: string;
  source: Source;
  reporter: string | null;
  synthetic: boolean;
  status: PageStatus;
  assignee: string | null;
  planned: string | null;
  backups: string[];
  attempts: Attempt[];
  escalations: number;
  escalation_log: { at: number; why: string; to: string | null; name?: string }[];
  created_at: number;
  accepted_at: number | null;
  resolved_at: number | null;
  hold_until: number | null;
  review_deadline: number | null;
  requires_review: boolean;
  review_reason: string | null;
  operator_approved: boolean;
  failsafe: boolean;
  blocked: string[];
  triage: {
    priority: number;
    category: string;
    symptoms: string[];
    vitals: Record<string, number>;
    red_flags: string[];
    rationale: string;
    source: string;
    model: string | null;
    latency_ms: number;
    rules_priority?: number;
  };
  decision: {
    mode: string;
    assignee: string | null;
    backups: string[];
    candidates: Candidate[];
    guardrails: string[];
    primary: string[];
    acceptable: string[];
    co_notify: string[];
    reasoning: string;
    previous_reasoning?: string;
  };
  certificate: null | { certified: boolean; z3_status: string; violations: string[]; checked: string[]; proof: string; ms: number; at: number; clinician: string };
  co_notified: { clinician: string; name: string; specialty: string; why: string }[];
  sbar: null | { situation: string; background: string; assessment: string; request: string; text: string; words: number; source: string; model: string | null };
  sbar_refining?: boolean;
  refining?: boolean;
  refinement?: { outcome: "raised" | "agrees" | "kept" | "unavailable"; note: string; from?: number; to?: number; model?: string | null; suggested?: number };
  timings: { triage_ms: number; locate_ms: number; route_ms: number; certify_ms?: number; total_ms?: number };
  /** set when the room monitor raised (or added findings to) this page */
  monitor?: { findings: { key: string; detector: string; title: string; severity: "info" | "watch" | "urgent" | "critical"; source: string }[] };
};

export type FeedItem = {
  id: string;
  page_id: string | null;
  kind: "alert" | "note";
  text: string;
  priority: Priority | null;
  category: string;
  room: string | null;
  zone: string | null;
  floor: string | null;
  patient: string | null;
  source: Source;
  reporter: string | null;
  created_at: number;
  status: string;
  synthetic: boolean;
  history: boolean;
};

export type Rec = {
  id: string;
  kind: string;
  severity: "info" | "warn" | "critical";
  title: string;
  detail: string;
  action: string;
  evidence: string[];
  created_at: number;
  updated_at: number;
  acknowledged: boolean;
  active: boolean;
};

/** Per monitored room. `risk` is the ward risk index (see INDEX in lib/clinical.ts);
 *  `probability` is the detector's calibrated P(sepsis) it was mapped from. */
export type Monitor = Record<
  string,
  { risk: number | null; probability?: number | null; tier: string | null; watch: boolean; episode: boolean; bed: string | null; live: boolean }
>;
export type Occupant = { name: string; age: number | null; sex: string | null; diagnosis: string; synthetic: boolean; monitored: boolean; bed?: string | null; live?: boolean };

export type Stats = {
  since: number;
  pages: number;
  by_priority: Record<string, number>;
  active: number;
  review: number;
  by_source: Record<string, number>;
  median_ack_s: Record<string, number | null>;
  escalation_rate: number | null;
  timeouts: number;
  declines: number;
  certificates: number;
  blocked: number;
  median_dispatch_ms: number | null;
  monitored_beds: number;
  uptime_s: number;
};

export type Settings = {
  autonomy: Record<string, ("review" | "auto")[]>;
  detector_p1_review: boolean;
  ack_timeout_s: Record<string, number>;
  review_failsafe_s: Record<string, number>;
  consecutive_window_s: number;
  workload_max_per_hour: number;
  near_tie_pct: number;
  hold_window_s: number;
  max_escalations: number;
  simulated_responders: boolean;
  demo_traffic: boolean;
  room_monitor_pages?: boolean;
};

export type Notify = { to: string | null; page: string; priority: number; room: string | null; text: string; kind: "page" | "fyi" | "sbar" | "upgrade"; ts: number };

export type PagingState = {
  ready: boolean;
  pages: Record<string, Page>;
  alerts: FeedItem[];
  clinicians: Record<string, Clinician>;
  recs: Record<string, Rec>;
  monitor: Monitor;
  occupancy: Record<string, Occupant>;
  /** every occupied room's unit status from the room monitor */
  rooms: Record<string, RoomStatus>;
  settings: Settings | null;
  stats: Stats | null;
  invariants: { id: string; text: string }[];
  notifications: Notify[];
  offsetMs: number;            // server clock - local clock
};

type Snapshot = {
  now: number;
  settings: Settings;
  clinicians: Clinician[];
  pages: Page[];
  alerts: FeedItem[];
  recs: Rec[];
  monitor: Monitor;
  occupancy: Record<string, Occupant>;
  rooms?: Record<string, RoomStatus>;
  stats: Stats;
  invariants: { id: string; text: string }[];
};

type Event = { type: string; data: unknown; ts: number };
type Action = { kind: "snapshot"; snap: Snapshot } | { kind: "events"; events: Event[] };

const empty: PagingState = {
  ready: false, pages: {}, alerts: [], clinicians: {}, recs: {}, monitor: {}, occupancy: {}, rooms: {},
  settings: null, stats: null, invariants: [], notifications: [], offsetMs: 0,
};

function upsertAlert(list: FeedItem[], a: FeedItem): FeedItem[] {
  const i = list.findIndex((x) => x.id === a.id);
  const next = i >= 0 ? list.map((x, j) => (j === i ? a : x)) : [...list, a];
  return next.slice(-150);
}

function reduce(state: PagingState, action: Action): PagingState {
  if (action.kind === "snapshot") {
    const s = action.snap;
    return {
      ...state,
      ready: true,
      pages: Object.fromEntries(s.pages.map((p) => [p.id, p])),
      alerts: s.alerts,
      clinicians: Object.fromEntries(s.clinicians.map((c) => [c.id, c])),
      recs: Object.fromEntries(s.recs.map((r) => [r.id, r])),
      monitor: s.monitor,
      occupancy: s.occupancy,
      rooms: s.rooms ?? {},
      settings: s.settings,
      stats: s.stats,
      invariants: s.invariants,
      offsetMs: s.now * 1000 - Date.now(),
    };
  }
  let next = state;
  for (const e of action.events) {
    switch (e.type) {
      case "page": {
        const p = e.data as Page;
        next = { ...next, pages: { ...next.pages, [p.id]: p } };
        break;
      }
      case "alert":
        next = { ...next, alerts: upsertAlert(next.alerts, e.data as FeedItem) };
        break;
      case "clinician": {
        const c = e.data as Clinician;
        next = { ...next, clinicians: { ...next.clinicians, [c.id]: c } };
        break;
      }
      case "rec": {
        const r = e.data as Rec;
        const recs = { ...next.recs };
        if (r.active) recs[r.id] = r;
        else delete recs[r.id];
        next = { ...next, recs };
        break;
      }
      case "monitor":
        next = { ...next, monitor: e.data as Monitor };
        break;
      case "rooms":
        next = { ...next, rooms: { ...next.rooms, ...(e.data as Record<string, RoomStatus>) } };
        break;
      case "settings":
        next = { ...next, settings: e.data as Settings };
        break;
      case "stats":
        next = { ...next, stats: e.data as Stats };
        break;
      case "notify":
        next = { ...next, notifications: [...next.notifications, { ...(e.data as Omit<Notify, "ts">), ts: e.ts }].slice(-40) };
        break;
    }
  }
  return next;
}

/**
 * The live paging state. One WebSocket: a snapshot on connect, then batches of
 * events once a second. A clinician device passes its id so the server knows a
 * person is holding it (the demo's simulated responders stand down for them).
 */
export function usePaging(opts: { clinician?: string | null; token?: string | null } = {}) {
  const [state, dispatch] = useReducer(reduce, empty);
  const [status, setStatus] = useState<"connecting" | "live" | "reconnecting">("connecting");
  // bumps on every successful (re)connect: a reconnect may mean the backend restarted
  // and forgot every session, so consumers re-check theirs when it changes
  const [epoch, setEpoch] = useState(0);
  const { clinician, token } = opts;

  useEffect(() => {
    let ws: WebSocket | null = null;
    let closed = false;
    let backoff = 1000;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let ping: ReturnType<typeof setInterval> | undefined;
    const path = "/aegis/paging/stream" + (clinician ? `?clinician=${encodeURIComponent(clinician)}` : "");
    const open = () => {
      setStatus((s) => (s === "live" ? "reconnecting" : s));
      try {
        ws = new WebSocket(wsUrl(path), wsProtocols(token === undefined ? undefined : token));
      } catch {
        schedule();
        return;
      }
      ws.onopen = () => {
        backoff = 1000;
        setStatus("live");
        setEpoch((e) => e + 1);
        ping = setInterval(() => ws?.readyState === 1 && ws.send("ping"), 20_000);
      };
      ws.onmessage = (ev) => {
        if (ev.data === "pong") return;
        try {
          const m = JSON.parse(ev.data);
          if (m.type === "snapshot") dispatch({ kind: "snapshot", snap: m.data });
          else if (m.type === "batch") dispatch({ kind: "events", events: m.events });
        } catch {}
      };
      ws.onclose = () => {
        clearInterval(ping);
        if (!closed) schedule();
      };
      ws.onerror = () => ws?.close();
    };
    const schedule = () => {
      setStatus("reconnecting");
      timer = setTimeout(open, backoff);
      backoff = Math.min(backoff * 2, 15_000);
    };
    // hidden tabs close the socket (Render free tier); a device keeps it, it is a pager
    const onVis = () => {
      if (clinician) return;
      if (document.hidden) {
        closed = true;
        clearTimeout(timer);
        clearInterval(ping);
        ws?.close();
      } else if (closed) {
        closed = false;
        backoff = 1000;
        open();
      }
    };
    document.addEventListener("visibilitychange", onVis);
    open();
    return () => {
      closed = true;
      document.removeEventListener("visibilitychange", onVis);
      clearTimeout(timer);
      clearInterval(ping);
      ws?.close();
    };
  }, [clinician, token]);

  return { state, status, epoch };
}

/** The building, fetched once per session. */
let layoutPromise: Promise<Layout> | null = null;
export function useLayout(): Layout | null {
  const [layout, setLayout] = useState<Layout | null>(null);
  useEffect(() => {
    let alive = true;
    const load = () => {
      layoutPromise ??= api<Layout>("/aegis/paging/hospital", { timeoutMs: 60_000 });
      layoutPromise.then((l) => alive && setLayout(l)).catch(() => {
        layoutPromise = null;
        if (alive) setTimeout(load, 4000);
      });
    };
    load();
    return () => {
      alive = false;
    };
  }, []);
  return layout;
}

/** A clock that ticks every second on the server's time base. */
export function useServerNow(offsetMs: number, everyMs = 1000): number {
  const [now, setNow] = useState(() => (Date.now() + offsetMs) / 1000);
  const off = useRef(offsetMs);
  off.current = offsetMs;
  useEffect(() => {
    const t = setInterval(() => setNow((Date.now() + off.current) / 1000), everyMs);
    return () => clearInterval(t);
  }, [everyMs]);
  return now;
}

/* ------------------------------------------------------------------ */
/* Presentation helpers                                                */
/* ------------------------------------------------------------------ */
export const PRIORITY_LABEL: Record<number, string> = { 1: "Immediate", 2: "Urgent", 3: "Soon", 4: "Routine" };
export const STATUS_LABEL: Record<StaffStatus, string> = {
  available: "Available", paging: "Being paged", in_procedure: "In procedure", on_case: "On a case", off_shift: "Off shift",
};
export const PAGE_STATUS_LABEL: Record<PageStatus, string> = {
  new: "New", review: "Awaiting review", held: "Queued", paging: "Ringing", accepted: "Accepted",
  resolved: "Resolved", cancelled: "Cancelled", unassigned: "Needs an owner",
};
/** Why the certificate refused a page, in a few words (policy.INVARIANTS has the full text). */
export const BLOCK_LABEL: Record<string, string> = {
  on_shift: "off shift", consecutive: "paged for another case", workload: "over the hourly limit",
  specialty: "not this service", autonomy: "needs review", p1_timeout: "timeout policy", prover_unavailable: "prover down",
};
export const SOURCE_LABEL: Record<Source, string> = { nurse: "Nurse", voice: "Voice", detector: "Detector", monitor: "Room monitor", demo: "Demo" };
export const CATEGORY_LABEL: Record<string, string> = {
  code: "Code", sepsis: "Sepsis", cardiac: "Cardiac", neuro: "Neuro", respiratory: "Respiratory",
  surgical: "Surgical", medication: "Medication", trauma: "Trauma", general: "General",
};

export function activePages(pages: Record<string, Page>): Page[] {
  const order: Record<string, number> = { review: 0, unassigned: 0, paging: 1, held: 2, accepted: 3 };
  return Object.values(pages)
    .filter((p) => p.status in order)
    .sort((a, b) => a.priority - b.priority || (order[a.status] - order[b.status]) || a.created_at - b.created_at);
}

export function mmss(sec: number): string {
  const s = Math.max(0, Math.round(sec));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}

export function since(ts: number | null | undefined, now: number): string {
  if (!ts) return "—";
  const s = Math.max(0, now - ts);
  if (s < 45) return "just now";
  if (s < 3600) return `${Math.round(s / 60)} min ago`;
  const h = Math.floor(s / 3600);
  return `${h} hr${h > 1 ? "s" : ""} ago`;
}

/** Wall-clock time on the 24-hour clock wards use: 14:07:32. */
export function clock(ts: number, seconds = true): string {
  return new Date(ts * 1000).toLocaleTimeString("en-GB", { hour: "2-digit", minute: "2-digit", ...(seconds ? { second: "2-digit" } : {}), hour12: false });
}

/** How a colleague is referred to in a sentence: "Dr. Alvarez", "Luis". */
export function shortName(name: string): string {
  const base = name.split(",")[0].trim();
  const parts = base.split(/\s+/);
  return base.startsWith("Dr.") ? `Dr. ${parts.at(-1)}` : parts[0];
}

export function initials(name: string): string {
  return name.replace(/^Dr\.\s*/, "").split(/[\s,]+/).filter(Boolean).slice(0, 2).map((p) => p[0]).join("").toUpperCase();
}

export function useRoomPages(pages: Record<string, Page>) {
  return useMemo(() => {
    const by: Record<string, Page> = {};
    for (const p of Object.values(pages)) {
      if (!p.room || !["review", "held", "paging", "accepted", "unassigned"].includes(p.status)) continue;
      const cur = by[p.room];
      if (!cur || p.priority < cur.priority) by[p.room] = p;
    }
    return by;
  }, [pages]);
}
