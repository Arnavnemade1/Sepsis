import {
  Bed,
  BellRinging,
  Cpu,
  Flask,
  Heartbeat,
  Journal,
  Layers,
  Overview,
  Pager,
  Pill,
  Plugs,
  Sentinel,
  ShieldCheck,
  Sliders,
  Staff,
  Terminal,
  type Icon,
} from "./icons";

export type NavKey =
  | "overview" | "pages" | "staff" | "sentinel" | "policy"
  | "rooms" | "patients" | "pharmacy"
  | "architecture" | "kernel" | "journal" | "gate" | "shell" | "evidence" | "integrations"
  | "pagers";

export type NavItem = { key: NavKey; label: string; href: string; icon: Icon; hint: string };
export const NAV_GROUPS: { label: string; items: NavItem[] }[] = [
  {
    label: "Operations",
    items: [
      { key: "overview", label: "Overview", href: "/operator/", icon: Overview, hint: "Live floor plan, active pages, event log" },
      { key: "pages", label: "Pages", href: "/operator/?view=pages", icon: BellRinging, hint: "Every page, filterable" },
      { key: "staff", label: "Staff", href: "/operator/?view=staff", icon: Staff, hint: "Roster, status and load" },
      { key: "sentinel", label: "Sentinel", href: "/operator/?view=sentinel", icon: Sentinel, hint: "System-wide recommendations" },
      { key: "policy", label: "Policy", href: "/operator/?view=policy", icon: Sliders, hint: "Autonomy, guardrails, invariants" },
    ],
  },
  {
    label: "Clinical",
    items: [
      { key: "rooms", label: "Rooms", href: "/room/", icon: Bed, hint: "Every room as a unit: record, vitals, detections" },
      { key: "patients", label: "Patients", href: "/patient/", icon: Heartbeat, hint: "ICU census, twin, risk trend, orders" },
      { key: "pharmacy", label: "Pharmacy", href: "/pharmacy/", icon: Pill, hint: "Treatment paths, interactions" },
    ],
  },
  {
    label: "Platform",
    items: [
      { key: "architecture", label: "Architecture", href: "/os/", icon: Layers, hint: "Every layer of the OS, live" },
      { key: "integrations", label: "Integrations", href: "/integrations/", icon: Plugs, hint: "HL7 / FHIR feeds, mappings earned on evidence, CDS Hooks, webhooks" },
      { key: "kernel", label: "Kernel", href: "/os/?view=kernel", icon: Cpu, hint: "Processes, scheduler, drivers, services" },
      { key: "journal", label: "Journal", href: "/os/?view=journal", icon: Journal, hint: "Hash-chained kernel log" },
      { key: "gate", label: "Syscall gate", href: "/os/?view=gate", icon: ShieldCheck, hint: "ABI, capabilities, sessions" },
      { key: "shell", label: "Shell", href: "/os/?view=shell", icon: Terminal, hint: "Commands as syscalls" },
      { key: "evidence", label: "Evidence", href: "/lab/", icon: Flask, hint: "The detector and its validation" },
    ],
  },
  {
    label: "Devices",
    items: [{ key: "pagers", label: "Pagers", href: "/clinician/", icon: Pager, hint: "Every clinician's pager" }],
  },
];
export const NAV_ITEMS: NavItem[] = NAV_GROUPS.flatMap((g) => g.items);
export const GROUP_OF: Record<string, string> = Object.fromEntries(NAV_GROUPS.flatMap((g) => g.items.map((i) => [i.key, g.label])));


export const SWITCH_USERS = [
  { u: "rn.patel", n: "Priya Patel, RN", r: "Charge nurse" },
  { u: "dr.whitfield", n: "Dr. James Whitfield", r: "Chief of critical care" },
  { u: "guest", n: "Guest", r: "View only" },
];

export function initialsOf(name: string) {
  return name.replace(/^Dr\.\s*/, "").split(/[\s,]+/).filter(Boolean).slice(0, 2).map((p) => p[0]).join("").toUpperCase();
}

