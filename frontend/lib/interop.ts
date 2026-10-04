/** The integration plane's API shapes (backend/aegis/interop) and small helpers. */
import type { Tone } from "./clinical";

export type Conversion = { factor: number; offset: number; label: string };

export type Check = { name: string; status: "pass" | "fail" | "unsure" | "skip"; stat: number | null; limit: number | null; n: number; detail: string };

export type Fit = { concept: string; conversion: Conversion; status: string; score: number | null; n: number; median: number | null; checks: Check[] };

export type Candidate = {
  proposal: { concept: string; score: number; proposer: string; rationale: string };
  concept: string;
  label: string;
  decision: string;
  declared: Conversion | null;
  conversion: Conversion | null;
  inference: { decision: string; declared: Fit | null; best: Fit | null; alternatives: Fit[] };
  joint: Check[];
  viable: boolean;
};

export type MappingRow = {
  id: string;
  interface: string;
  code: string;
  system: string;
  text: string;
  unit_written: string;
  concept: string | null;
  conversion: Conversion | null;
  unit_source: string;
  status: "learning" | "proposing" | "review" | "adopted" | "quarantined" | "rejected" | "unmapped";
  proposer: string | null;
  lead?: { concept: string; proposer: string; decision: string } | null;
  reason: string;
  version: number;
  created_at: number;
  updated_at: number;
  adopted_at: number | null;
  adopted_by: string | null;
  counts: { seen: number; delivered: number; withheld: number; backfilled: number; retracted: number; non_numeric: number };
  n: number;
  fault: {
    at: number;
    cause: string;
    checks: Check[];
    withdrawn: number;
    suggested: Conversion | null;
    by_identity?: boolean;
    swap?: { with: string; code: string | null; concept: string | null; label: string } | null;
    first_suspect_t: number | null;
  } | null;
  checks_run: number;
};

export type MappingDetail = MappingRow & {
  candidates: Candidate[];
  evidence: {
    at?: number;
    onboarding?: Candidate | null;
    live?: { at: number; n: number; checks: Check[]; invariants: Check[]; liveness: { longest_identical_run: number; limit: number } | null } | null;
    human?: { fit: Fit; against_evidence: boolean };
  } | null;
  history: { at: number; from: string | null; to: string; by: string; why: string }[];
  samples: { t: number; raw: number; unit: string; room: string | null; delivered: boolean }[];
  concept_info: { key: string; label: string; loinc: string; loinc_display: string; unit: string; kind: string; window: [number, number]; scale: string; band: number } | null;
  reference: { key: string; n: number; source: string; loc: number; spread: number; tail: [number, number]; quantiles: Record<string, number> } | null;
};

export type Iface = {
  id: string;
  name: string;
  standard: string;
  direction: string;
  transport: string;
  sending_app: string;
  facility: string;
  kind: string;
  simulated: boolean;
  status: "live" | "degraded" | "idle" | "pending";
  note: string;
  last_message_at: number | null;
  stats: Record<string, number>;
  per_min: number;
  series: number[];
  latency_us: { p50: number | null; p99: number | null };
};

export type MsgSummary = {
  id: string;
  at: number;
  interface: string | null;
  interface_name?: string | null;
  type: string | null;
  control_id?: string | null;
  ack: "AA" | "AE" | "AR";
  summary: string;
  latency_us: number;
  bytes?: number;
  duplicate?: boolean;
  error?: { code: number; text: string; where: string; message: string } | null;
};

export type Outcome = { set_id: string; code: string; text: string; system: string; value: string; unit: string; flag: string; status: string; concept: string | null; canonical: number | null; mapping: string | null };

export type MsgDetail = MsgSummary & { raw: string; outcomes: Outcome[]; transport: string; version: string | null; ack_text: string };

export type Overview = {
  now: number;
  since: number;
  settings: { adoption: string; unknown_senders: string; drift_monitor: boolean; quarantine: boolean; model_proposer: boolean; simulated_feeds: boolean };
  interfaces: Iface[];
  mappings: { total: number; by_status: Record<string, number>; by_interface: Record<string, Record<string, number>> };
  review: MappingRow[];
  quarantined: MappingRow[];
  census: number;
  counters: Record<string, number>;
  model: { available: boolean; queue: number; busy: boolean; calls: { at: number; codes: string[]; model: string | null; answered: string[]; ms: number }[] };
  reference: { built?: string; fit_digest?: string; training_patients?: number; concepts: number };
  vocabulary: number;
  invariants: number;
  recent: MsgSummary[];
};

export type SimStatus = {
  patients: { mrn: string; room: string; name: string; dx: string; scenario: { kind: string; label: string; phase: string; progress: number } | null; discharge_in_s: number }[];
  faults: Record<string, { label: string; since: number }>;
  drills: { kind: string; label: string; system: string }[];
  sent: number;
  refused: number;
};

export type Webhooks = {
  subscriptions: { id: string; url: string; events: string[]; created_by: string; created_at: number; active: boolean; description: string; delivered: number; failed: number; secret: string }[];
  queued: number;
  counters: Record<string, number>;
  event_types: string[];
  recent: { id: string; sub: string; event: string; attempts: number; status: string; last_code: number | null; last_error: string; created_at: number; delivered_at: number | null; latency_ms: number | null; data: unknown }[];
  dead: Webhooks["recent"];
  echo: { at: number; verified: boolean; id: string; type: string; data: unknown }[];
  cds: { at: number; service: string; hook: string; patient: string; room: string | null; cards: number; ms: number }[];
};

export type CdsCard = {
  uuid: string;
  summary: string;
  indicator: "info" | "warning" | "critical";
  source: { label: string };
  detail?: string;
  links?: { label: string; url: string; type: string }[];
  suggestions?: { label: string; uuid: string; isRecommended?: boolean; actions: { type: string; description: string; resourceId?: string[] }[] }[];
};

export const STATUS_TONE: Record<string, Tone> = {
  adopted: "low",
  quarantined: "high",
  review: "med",
  proposing: "neutral",
  learning: "neutral",
  unmapped: "nodata",
  rejected: "nodata",
  live: "low",
  degraded: "dys",
  idle: "nodata",
  pending: "neutral",
};

export const STATUS_LABEL: Record<string, string> = {
  adopted: "Adopted",
  quarantined: "Quarantined",
  review: "Needs review",
  proposing: "Asking the model",
  learning: "Learning",
  unmapped: "Not in vocabulary",
  rejected: "Rejected",
  live: "Live",
  degraded: "Degraded",
  idle: "Idle",
  pending: "Pending",
};

export const ACK_TONE: Record<string, Tone> = { AA: "low", AE: "med", AR: "high" };

export const CHECK_TONE: Record<string, Tone> = { pass: "low", fail: "high", unsure: "med", skip: "nodata" };

export function ago(t: number | null | undefined, now = Date.now() / 1000): string {
  if (!t) return "never";
  const s = Math.max(0, now - t);
  if (s < 5) return "now";
  if (s < 60) return `${Math.round(s)}s ago`;
  if (s < 3600) return `${Math.round(s / 60)}m ago`;
  if (s < 86400) return `${(s / 3600).toFixed(1)}h ago`;
  return `${Math.round(s / 86400)}d ago`;
}

export function proposerLabel(p: string | null | undefined): string {
  if (!p) return "—";
  if (p.startsWith("model:")) return `Language model (${p.slice(6)})`;
  if (p.startsWith("human:")) return `By hand (${p.slice(6)})`;
  return { loinc: "LOINC code", mdc: "IEEE 11073 code", lexical: "Lexical rules" }[p] ?? p;
}

export function fmtNum(v: number | null | undefined, digits = 3): string {
  if (v == null || Number.isNaN(v)) return "—";
  const a = Math.abs(v);
  if (a >= 1000) return v.toFixed(0);
  if (a >= 100) return v.toFixed(1);
  if (a >= 1) return v.toFixed(2);
  return v.toPrecision(Math.max(1, digits - 1));
}
