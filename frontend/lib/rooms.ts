"use client";

import { usePoll } from "./hooks";

/* ------------------------------------------------------------------ */
/* Contracts (backend/aegis/rooms)                                     */
/* ------------------------------------------------------------------ */
export type Severity = "info" | "watch" | "urgent" | "critical";
export type RoomKind = "synthetic" | "replay" | "live";

/** What the paging stream carries for every occupied room (snapshot + "rooms" events). */
export type RoomStatus = { sev: Severity | null; title: string | null; news2: number | null; band: string | null; n: number; episode: boolean };

export type Evidence = { label: string; value: string | number | null; unit: string; points?: number; at?: number };

export type Finding = {
  key: string;
  detector: string;
  title: string;
  severity: Severity;
  category: string;
  evidence: Evidence[];
  context: string[];
  response: string;
  source: string;
  score: number | null;
  page: number | null;
  page_category: string;
  static: boolean;
  first_seen?: number;
  last_seen?: number;
  resolved_at?: number;
  peak?: Severity;
  history?: [number, Severity][];
  open?: boolean;
};

export type News2 = {
  score: number;
  band: "low" | "low-medium" | "medium" | "high";
  complete: boolean;
  missing: string[];
  scale: 1 | 2;
  parts: { label: string; value: string | number; unit: string; points: number }[];
};
export type Scores = { news2?: News2; shock_index?: number; qsofa?: number };

export type Vitals = {
  t?: number;
  hr?: number;
  sbp?: number;
  dbp?: number;
  map?: number;
  rr?: number;
  spo2?: number;
  temp_c?: number;
  o2_lpm?: number;
  avpu?: "A" | "C" | "V" | "P" | "U";
  chronic_confusion?: boolean;
};

export type Lab = { t: number; v: number; baseline?: boolean };

export type Order = {
  name: string;
  generic?: string;
  dose?: string;
  route?: string;
  freq?: string;
  classes: string[];
  prn?: boolean;
  infusion?: boolean;
  renal?: string;
  qt?: boolean;
  kind: "drug" | "order";
  status: "active" | "held";
  started_at?: number;
  last_given_at?: number | null;
  note?: string;
  detail?: string;
  simulated?: boolean;
};

export type RoomRecord = {
  room: string;
  kind: RoomKind;
  provenance: { synthetic: boolean; label: string; detail: string };
  patient: { name: string | null; age: number | null; sex: string | null; mrn: string | null; weight_kg?: number; bed?: string | null };
  admission: { reason?: string; admitted_at?: number; day?: number; icu_hour?: number; unit?: string; hospital_admit_h?: number | null };
  problems: { name: string; status: string; since: string; note?: string }[];
  orders: Order[];
  home_meds: { name: string; dose: string; freq: string }[];
  allergies: { substance: string; reaction: string; severity: string; kind: string; recorded?: string }[];
  code_status: string | null;
  encounters: { at: number; setting: string; reason: string; los_days: number | null; vitals: Record<string, number>; note: string | null }[];
  baseline: null | { hr: number; sbp: number; dbp: number; rr: number; spo2: number; temp_c: number; mental: string; n: number; source: string };
  spo2_target: null | { lo: number; hi: number; scale: number; why: string };
  planned: string[];
  labs: Record<string, Lab[]>;
  mobility: string | null;
  flags: string[];
  linked: { source: string | null; label: string | null; imported_at: number | null; summary: string | null }[];
  missing?: string[];
};

export type TimelineEvent = {
  t: number;
  kind: "finding" | "escalated" | "resolved" | "page" | "lab" | "course" | "order" | "obs" | "episode" | "alert" | "record";
  severity: Severity;
  text: string;
  key?: string;
  detector?: string;
  page?: string | null;
  priority?: number | null;
  simulated?: boolean;
  lab?: string;
};

export type Simulation = { kind: string; label: string; started_at: number; phase: "developing" | "established" | "recovering"; progress: number; page: string | null; observed: boolean };

export type UnitPage = { id: string; priority: number; status: string; text: string; source: string; created_at: number; assignee: string | null; category: string; active: boolean };

export type Unit = {
  room: string;
  zone: string;
  zone_name: string;
  floor: string;
  critical_care: boolean;
  occupied: boolean;
  kind?: RoomKind;
  now?: number;
  clock?: "wall" | "icu_hours";
  record?: RoomRecord;
  vitals?: Vitals | null;
  series?: Vitals[];
  labs?: Record<string, Lab[]>;
  scores?: Scores;
  news2_history?: { t: number; score: number }[];
  findings?: Finding[];
  resolved?: Finding[];
  faults?: string[];
  episode?: { open: boolean; page?: string; priority?: number } | null;
  med_episode?: { open: boolean; page?: string; priority?: number } | null;
  pages?: UnitPage[];
  timeline?: TimelineEvent[];
  care_team?: { id: string; name: string; role: string; status: string; specialty: string }[];
  devices?: { name: string; detail: string; status: string }[];
  simulation?: Simulation | null;
  monitoring?: { kind: RoomKind; sample_s: number | null; eval_period_s: number; pages_from: string; detectors: string[] };
  risk?: number | null;
};

export type BoardRoom = {
  room: string;
  zone: string;
  zone_name: string;
  floor: string;
  kind: RoomKind | null;
  name?: string | null;
  age?: number | null;
  sex?: string | null;
  mrn?: string | null;
  bed?: string | null;
  diagnosis?: string | null;
  code_status?: string | null;
  allergies?: string[];
  synthetic?: boolean;
  scores?: Scores;
  vitals?: Vitals;
  vitals_at?: number | null;
  clock?: "wall" | "icu_hours";
  findings?: { key: string; detector: string; title: string; severity: Severity; category: string }[];
  counts?: { critical: number; urgent: number; watch: number };
  worst?: Severity | null;
  trend?: number[];
  episode?: { open: boolean; page: string | null } | null;
  simulation?: Simulation | null;
  risk?: number | null;
};

export type Board = {
  rooms: BoardRoom[];
  evaluated_at: number;
  eval_period_s: number;
  eval_ms_p50: number | null;
  counters: { evaluations: number; findings_opened: number; pages_opened: number; pages_raised: number; pages_linked: number; scenarios: number };
  detectors: string[];
};

/* ------------------------------------------------------------------ */
/* Severity: the same four hues as priority (P1 red … P4 grey)          */
/* ------------------------------------------------------------------ */
export const SEV_RANK: Record<Severity, number> = { info: 0, watch: 1, urgent: 2, critical: 3 };
export const SEV_RGB: Record<Severity, string> = { critical: "229 72 77", urgent: "245 118 46", watch: "238 177 42", info: "120 129 141" };
export const SEV_INK: Record<Severity, string> = { critical: "#ff6b70", urgent: "#ff9357", watch: "#f4c55a", info: "#a8afb8" };
export const SEV_LABEL: Record<Severity, string> = { critical: "Critical", urgent: "Urgent", watch: "Watch", info: "Info" };
export const BAND_SEV: Record<string, Severity> = { high: "critical", medium: "urgent", "low-medium": "watch", low: "info" };
export const BAND_LABEL: Record<string, string> = { high: "High", medium: "Medium", "low-medium": "Low–medium", low: "Low" };
export const KIND_LABEL: Record<RoomKind, string> = { synthetic: "Synthetic record", replay: "Replayed real record", live: "Live session" };
/** NEWS2 points on one parameter → the hue its reading carries (0 is neutral). */
export const POINT_SEV: Severity[] = ["info", "watch", "urgent", "critical"];

export const sevRank = (s: Severity | null | undefined) => (s ? SEV_RANK[s] : -1);
export const worse = (a: Severity | null | undefined, b: Severity | null | undefined): Severity | null =>
  sevRank(a) >= sevRank(b) ? a ?? null : b ?? null;

/* ------------------------------------------------------------------ */
/* Data                                                                */
/* ------------------------------------------------------------------ */
export function useBoard(everyMs = 5000) {
  return usePoll<Board>("/aegis/rooms", everyMs);
}

export function useUnit(room: string | null, everyMs = 4000) {
  return usePoll<Unit>(room ? `/aegis/rooms/${encodeURIComponent(room)}` : null, everyMs);
}

/* ------------------------------------------------------------------ */
/* Formatting                                                          */
/* ------------------------------------------------------------------ */
export function since(t: number | null | undefined, now: number): string {
  if (t == null) return "—";
  const s = Math.max(0, now - t);
  if (s < 60) return `${Math.round(s)} s`;
  if (s < 3600) return `${Math.floor(s / 60)} min`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ${Math.floor((s % 3600) / 60)} min`;
  return `${Math.floor(s / 86400)} d`;
}

/** A reading as written: integers bare, others to at most two decimals ("0.96", "18.4", never "1.0" for 0.96). */
export function fmtVal(v: string | number | null | undefined, digits = 2): string {
  if (v == null || v === "") return "—";
  if (typeof v === "number") return Number.isInteger(v) ? String(v) : String(Number(v.toFixed(digits)));
  return v;
}

export const LAB_LABEL: Record<string, [string, string]> = {
  creatinine: ["Creatinine", "mg/dL"], wbc: ["WBC", "×10⁹/L"], hemoglobin: ["Haemoglobin", "g/dL"], platelets: ["Platelets", "×10⁹/L"],
  potassium: ["Potassium", "mmol/L"], glucose: ["Glucose", "mg/dL"], lactate: ["Lactate", "mmol/L"], anc: ["ANC", "/µL"],
  bun: ["BUN", "mg/dL"], bilirubin: ["Bilirubin", "mg/dL"], fio2: ["FiO₂", ""], ph: ["pH", ""], paco2: ["PaCO₂", "mmHg"],
  hco3: ["HCO₃", "mmol/L"], troponin: ["Troponin I", "ng/mL"], ptt: ["PTT", "s"], fibrinogen: ["Fibrinogen", "mg/dL"],
};

/** Reference ranges for flagging a lab (adult, general). [low, high]; null = no bound. */
export const LAB_RANGE: Record<string, [number | null, number | null]> = {
  creatinine: [null, 1.2], wbc: [4, 12], hemoglobin: [12, null], platelets: [150, null], potassium: [3.5, 5.2],
  glucose: [70, 180], lactate: [null, 2], anc: [1500, null], bun: [null, 25], bilirubin: [null, 1.2], troponin: [null, 0.04],
  ph: [7.35, 7.45], paco2: [35, 45], hco3: [22, 28],
};

export function labFlag(name: string, v: number): "low" | "high" | null {
  const r = LAB_RANGE[name];
  if (!r) return null;
  if (r[0] != null && v < r[0]) return "low";
  if (r[1] != null && v > r[1]) return "high";
  return null;
}
