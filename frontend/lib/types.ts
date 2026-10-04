/**
 * The backend contracts this app reads. Hand-written from live payloads and the
 * schema in backend/aegis/stream.py (SNAPSHOT_SCHEMA_VERSION 1.5). Only fields
 * the UI actually renders are typed; everything else stays `unknown` so a field
 * the backend drops shows up as a type error here, not a silent blank.
 */

export type Tier = "LOW" | "MEDIUM" | "HIGH";
export type WardTier = "STABLE" | "WATCH" | "CRITICAL";

export type OrganSystem = {
  name: string;
  region: string;
  monitored: boolean;
  status: string;
  severity: number;
  drivers: string[];
  sofa_points?: number | null;
  trend?: string;
  projected_severity?: number | null;
  eta_critical_min?: number | null;
};

export type BodyRegion = { severity: number; monitored: boolean; systems: string[]; status: string };

export type BodyState = {
  overall_risk: Tier;
  sepsis_probability: number | null;
  organ_systems: OrganSystem[];
  regions: Record<string, BodyRegion>;
  channels?: { key: string; label?: string; value: number | null; unit?: string }[];
  tracking?: { foci?: string[] };
};

export type Certainty = {
  state: string;
  abstain: boolean;
  lo: number | null;
  hi: number | null;
  alpha: number;
  set?: string[];
};

export type AgentOpinion = {
  agent: string;
  concern: number;
  confidence: number;
  rationale: string;
  evidence: string[];
  recommendation: string | null;
};

export type Snapshot = {
  schema_version: string;
  timestamp: number;
  source: string;
  patient: {
    id: string;
    name: string;
    age: number | null;
    gender: string | null;
    clinical_history: Record<string, unknown> | null;
  };
  risk: {
    level: Tier;
    /** the detector's calibrated P(sepsis label) — for the live session and every ward bed */
    ml_probability: number | null;
    /** the same on the ward risk index (lib/clinical.ts INDEX); the tier is read off it */
    index?: number | null;
    /** the named model that scored it: A40 (all-variable) or V7 (vital-sign); /aegis/models */
    model?: string | null;
    /** the probability before an imported chart's prior shift, when one applies */
    population_risk?: number | null;
    sirs_score: number;
    qsofa_score: number;
    signal_quality: number;
    certainty?: Certainty;
  };
  vitals: Record<string, number | null>;
  detection: Record<string, unknown> & { sepsis_probability?: number };
  triggered_rules: string[];
  body_state: BodyState;
  reasoning?: {
    opinions: AgentOpinion[];
    consensus?: Record<string, unknown>;
    escalation?: Record<string, unknown> | null;
  } | null;
  history: {
    t: number[];
    channels: Record<string, (number | null)[]>;
    risk: Tier[];
    probability: (number | null)[];
  };
  protocol: null | {
    fluid_rate?: number;
    vasopressor_dose?: number;
    antibiotic?: string;
    antibiotic_code?: number;
  };
  verification: null | { ok?: boolean; message?: string; proof?: string; z3_status?: string };
  narrative: string | null;
  narrative_source?: string | null;
};

export type RosterPatient = {
  id: string;
  name: string;
  kind: "live" | "cohort";
  septic: boolean | null;
  history?: Record<string, unknown>;
};

export type WardBed = {
  bed_id: string;
  name: string;
  fused_risk: number;
  mrn: string;
  age: number;
  gender: string;
  icu_hour: number;
  los_total: number;
  will_be_septic: boolean;
  onset_hour: number | null;
  hours_to_onset: number | null;
  drivers: string[];
  primary_system: string;
  components: Record<string, number>;
  enterprise: Record<string, number>;
  edge: Record<string, number>;
  population_risk?: number;
  deviation?: unknown;
};

export type TopologyNode = {
  node_id: string;
  name: string;
  p: number;
  slope: number;
  foresight: number;
  tier: WardTier;
  lanes: number;
  cpu: number;
  pin: number;
};

export type SafetyRecord = {
  ok: boolean;
  egress: string;
  z3_status: string;
  message: string;
  proof: string;
  ts: number;
  verifier: string;
};

export type SchedEvent = {
  tick: number;
  type: string;
  owner: string | null;
  reason: string;
  p: number;
  slope: number;
  ts: number;
};

export type Cluster = {
  topology: TopologyNode[];
  beds: WardBed[];
  events: SchedEvent[];
  kstat: {
    tick: number;
    mode: string;
    owner: string | null;
    procs: number;
    tiers: Record<WardTier, number>;
    lane_budget: number;
    lanes_used: number;
    utilization: number;
    interrupts: number;
    anticipating: number;
  };
  safety: { certified: number; blocked: number; quarantined: number; pass_rate: number; recent: SafetyRecord[] };
  syscalls: { invoked: number; allowed: number; denied: number; recent: unknown[]; active_interventions: Record<string, unknown> };
  bundle: null | {
    bed_id: string;
    name: string;
    fused_risk: number;
    severity_name: string;
    risk_level: Tier;
    primary_system: string;
    alert: { certified: boolean; z3: string; msg: string };
    protocol: {
      fluid_rate: number;
      vasopressor_dose: number;
      antibiotic: string;
      certified: boolean;
      z3: string;
      msg: string;
      proof: string;
      violations: string[];
    };
  };
  services: { running: number; n: number; services: { name: string; state: string; healthy: boolean }[] };
  devbio: { mounted: number; latency: { p50_us: number; p95_us: number } };
  data_mode: string;
  ts: number;
};

export type Decision = {
  action: string;
  route: string;
  urgency: string;
  because: string[];
  abstained: boolean;
  certainty_state: string;
  limits: string[];
  risk_level: Tier;
  ml_probability: number | null;
  tradeoff?: { threshold: number };
};

export type Timeline = {
  history: { hour: number; probability: number | null; source: string; lo: number | null; hi: number | null }[];
  forecast: { hour: number; probability: number; lo?: number; hi?: number }[];
  alarm_threshold: number;
  medium_threshold: number;
  uncertain_lo: number;
  uncertain_hi: number;
  trajectory: string;
  trend_per_hour: number;
  breach_in_hours: number | null;
  refused?: string | null;
  limits: string[];
  provenance: Record<string, unknown>;
};

export type Account = {
  uid: number;
  username: string;
  display: string;
  role: import("./api").Role;
  ring: number;
  title: string;
  kind: string;
  demo: boolean;
  passphrase: boolean;
};
