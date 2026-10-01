import type { Credential, Role } from "./api";

/**
 * Where each person starts. The product is two surfaces: the operator console
 * (the charge nurse runs the floor; the chief governs it) and each clinician's
 * pager. Everything else — a patient's twin, pharmacy, the lab, the kernel — is
 * a drill-down reached from those two.
 */
export const OPERATORS = new Set(["rn.patel"]);

export function homeFor(c: Pick<Credential, "username" | "role">): string {
  if (c.role === "guest" || c.role === "chief" || OPERATORS.has(c.username)) return "/operator/";
  return `/clinician/?id=${encodeURIComponent(c.username)}`;
}

export const ROLE_LABEL: Record<Role, string> = {
  guest: "Guest",
  nurse: "Nurse",
  remote_specialist: "Remote consult",
  pharmacist: "Pharmacist",
  attending: "Attending",
  chief: "Chief",
};
