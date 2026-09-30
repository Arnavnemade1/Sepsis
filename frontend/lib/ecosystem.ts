import type { Role } from "./api";

/**
 * The ecosystem: five workspaces, each a different job in the same ICU, all
 * reading one running system. A role lands in its home workspace and can open
 * any other it is allowed to see; the backend gate is still the authority on
 * what a role may DO (see backend/aegis/rbac.py) — this table only decides
 * where people start and what the nav offers.
 */
export type Workspace = {
  key: "command" | "bed" | "pharmacy" | "lab" | "os";
  href: string;
  label: string;
  who: string;
  blurb: string;
  roles: Role[] | "all";
};

export const WORKSPACES: Workspace[] = [
  {
    key: "command",
    href: "/command/",
    label: "Command",
    who: "Charge nurse · chief",
    blurb: "The whole unit at once: every bed ranked by calibrated risk, the scheduler's attention, and every alert the prover certified.",
    roles: "all",
  },
  {
    key: "bed",
    href: "/bed/",
    label: "Bedside",
    who: "Nurse · attending · consult",
    blurb: "One patient: the anatomical twin, vitals, the risk timeline with its uncertainty band, the decision and why, and Dr. Aegis on call.",
    roles: "all",
  },
  {
    key: "pharmacy",
    href: "/pharmacy/",
    label: "Pharmacy",
    who: "Clinical pharmacist · attending",
    blurb: "Drug-interaction screening over 143k edges compiled to Z3, and the pharmacokinetic twin that projects each treatment option.",
    roles: ["pharmacist", "attending", "chief", "guest"],
  },
  {
    key: "lab",
    href: "/lab/",
    label: "Lab",
    who: "Research · chief",
    blurb: "The evidence behind every number: validation, calibration, federation, scenario tests, and training a detector from the OS.",
    roles: "all",
  },
  {
    key: "os",
    href: "/os/",
    label: "System",
    who: "Chief · operators",
    blurb: "The operating system underneath: processes, the journal, boot, services, verified drivers, and the syscall gate.",
    roles: "all",
  },
];

export const HOME: Record<Role, Workspace["key"]> = {
  guest: "command",
  nurse: "command",
  chief: "command",
  attending: "bed",
  remote_specialist: "bed",
  pharmacist: "pharmacy",
};

export const ROLE_LABEL: Record<Role, string> = {
  guest: "Guest",
  nurse: "Nurse",
  remote_specialist: "Remote consult",
  pharmacist: "Pharmacist",
  attending: "Attending",
  chief: "Chief",
};

export function workspaceFor(key: Workspace["key"]): Workspace {
  return WORKSPACES.find((w) => w.key === key)!;
}
