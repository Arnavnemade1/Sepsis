import {
  BedDouble,
  BellRing,
  Cpu,
  FlaskConical,
  HeartPulse,
  LayoutDashboard,
  Network,
  Pill,
  Radar,
  ScrollText,
  ShieldCheck,
  SlidersHorizontal,
  Smartphone,
  SquareTerminal,
  Users,
  type LucideIcon,
} from "lucide-react";

export type NavKey =
  | "overview" | "pages" | "staff" | "sentinel" | "policy"
  | "rooms" | "patients" | "pharmacy"
  | "architecture" | "kernel" | "journal" | "gate" | "shell" | "evidence"
  | "pagers";

export type NavItem = { key: NavKey; label: string; href: string; icon: LucideIcon; hint: string };
export const NAV_GROUPS: { label: string; items: NavItem[] }[] = [
  {
    label: "Operations",
    items: [
      { key: "overview", label: "Overview", href: "/operator/", icon: LayoutDashboard, hint: "Live floor plan, active pages, event log" },
      { key: "pages", label: "Pages", href: "/operator/?view=pages", icon: BellRing, hint: "Every page, filterable" },
      { key: "staff", label: "Staff", href: "/operator/?view=staff", icon: Users, hint: "Roster, status and load" },
      { key: "sentinel", label: "Sentinel", href: "/operator/?view=sentinel", icon: Radar, hint: "System-wide recommendations" },
      { key: "policy", label: "Policy", href: "/operator/?view=policy", icon: SlidersHorizontal, hint: "Autonomy, guardrails, invariants" },
    ],
  },
  {
    label: "Clinical",
    items: [
      { key: "rooms", label: "Rooms", href: "/room/", icon: BedDouble, hint: "Every room as a unit: record, vitals, detections" },
      { key: "patients", label: "Patients", href: "/patient/", icon: HeartPulse, hint: "ICU census, twin, risk trend, orders" },
      { key: "pharmacy", label: "Pharmacy", href: "/pharmacy/", icon: Pill, hint: "Treatment paths, interactions" },
    ],
  },
  {
    label: "Platform",
    items: [
      { key: "architecture", label: "Architecture", href: "/os/", icon: Network, hint: "Every layer of the OS, live" },
      { key: "kernel", label: "Kernel", href: "/os/?view=kernel", icon: Cpu, hint: "Processes, scheduler, drivers, services" },
      { key: "journal", label: "Journal", href: "/os/?view=journal", icon: ScrollText, hint: "Hash-chained kernel log" },
      { key: "gate", label: "Syscall gate", href: "/os/?view=gate", icon: ShieldCheck, hint: "ABI, capabilities, sessions" },
      { key: "shell", label: "Shell", href: "/os/?view=shell", icon: SquareTerminal, hint: "Commands as syscalls" },
      { key: "evidence", label: "Evidence", href: "/lab/", icon: FlaskConical, hint: "The detector and its validation" },
    ],
  },
  {
    label: "Devices",
    items: [{ key: "pagers", label: "Pagers", href: "/clinician/", icon: Smartphone, hint: "Every clinician's pager" }],
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

