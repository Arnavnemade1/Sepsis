"""Case Handler: rank the staff for a case and choose who gets the page.

Scoring is deterministic and every term is recorded, so the operator sees the
exact reasons (specialty, proximity, status, caseload) rather than a black box.
Guardrails are applied as explicit steps on top of the score and each one that
changes the outcome is written into the decision:

  * consecutive paging  — nobody is paged twice within the window for different cases
  * workload            — P3/P4 skip anyone above the hourly limit
  * zone escalation     — P1/P2 prefer staff already on the unit
  * off-shift           — excluded, except as a P1 last resort
  * near-tie balancing  — when the top three are within 10%, the lightest caseload wins
  * time-queued         — a P3/P4 with nobody free can wait for the right person
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from backend.aegis.paging.hospital import (REMOTE_ZONE, ROOMS, SPECIALTY_LABEL, ZONES, Clinician,
                                           floor_distance)

# the physician who owns a patient on each kind of unit
_ZONE_PHYSICIAN = {"icu": "critical_care", "ed": "emergency", "pacu": "anesthesia", "or": "surgery",
                   "stepdown": "hospitalist", "ward": "hospitalist", "obs": "hospitalist"}
# categories where a remote consultant can genuinely help without being at the bedside
_REMOTE_OK = {"medication", "sepsis"}


def zone_physician(zone_id: Optional[str]) -> str:
    kind = ZONES[zone_id].kind if zone_id in ZONES else "ward"
    return _ZONE_PHYSICIAN.get(kind, "hospitalist")


def specialties_for(category: str, zone_id: Optional[str]) -> Tuple[List[str], List[str]]:
    """(primary, acceptable) specialties for a case."""
    zp = zone_physician(zone_id)
    table = {
        "code": (["rapid_response", "critical_care"], ["emergency", "anesthesia", zp]),
        "sepsis": ([zp], ["critical_care", "rapid_response", "hospitalist", "emergency", "infectious_disease"]),
        "cardiac": (["cardiology"], ["critical_care", "hospitalist", "emergency", zp]),
        "neuro": (["neurology"], ["critical_care", "emergency", "hospitalist"]),
        "respiratory": (["critical_care", "rapid_response"], ["anesthesia", "hospitalist", "emergency"]),
        "surgical": (["surgery"], ["hospitalist", "emergency", zp]),
        "medication": (["pharmacy"], ["hospitalist", zp]),
        "trauma": (["emergency" if zp == "emergency" else "surgery"], ["surgery", "emergency", "hospitalist"]),
        "general": ([zp], ["hospitalist", "rapid_response", "emergency", "critical_care"]),
    }
    primary, acceptable = table.get(category, table["general"])
    acceptable = [s for s in dict.fromkeys(acceptable) if s not in primary]
    return primary, acceptable


def co_notify_for(category: str, priority: int) -> List[str]:
    """Who else is told (not paged) — sepsis activates the Hour-1 bundle team."""
    if category == "sepsis" and priority <= 2:
        return ["pharmacy"]            # antibiotics within the hour start with the pharmacist
    if category == "code":
        return ["pharmacy", "critical_care"]
    return []


@dataclass
class Candidate:
    id: str
    name: str
    specialty: str
    status: str
    zone: str
    score: float
    factors: List[Tuple[str, float]] = field(default_factory=list)
    excluded: Optional[str] = None
    caseload: int = 0
    pages_last_hour: int = 0
    on_shift: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["factors"] = [{"label": a, "points": round(b, 1)} for a, b in self.factors]
        d["score"] = round(self.score, 1)
        d["specialty_label"] = SPECIALTY_LABEL.get(self.specialty, self.specialty)
        return d


@dataclass
class RouteDecision:
    mode: str                               # dispatch | hold | unassigned (review is applied by the engine)
    assignee: Optional[str]
    backups: List[str]
    candidates: List[Candidate]
    guardrails: List[str]
    primary: List[str]
    acceptable: List[str]
    co_notify: List[str]
    hold_until: Optional[float] = None
    reasoning: str = ""
    # facts the policy certificate needs
    on_shift_alternative: bool = True
    consecutive_alternative: bool = True

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["candidates"] = [c.to_dict() for c in self.candidates]
        return d


def _caseload(c: Clinician, load: Dict[str, Dict[str, Any]]) -> Tuple[int, int, Optional[float], Optional[str]]:
    l = load.get(c.id, {})
    return l.get("active", 0), l.get("last_hour", 0), l.get("last_paged_at"), l.get("last_case")


def rank(priority: int, category: str, room: Optional[str], roster: Dict[str, Clinician],
         load: Dict[str, Dict[str, Any]], settings: Dict[str, Any], case_id: str,
         now: Optional[float] = None, exclude: Optional[List[str]] = None) -> RouteDecision:
    now = now or time.time()
    zone = ROOMS[room].zone if room in ROOMS else None
    zone_floor = ZONES[zone].floor if zone else None
    primary, acceptable = specialties_for(category, zone)
    relevant = set(primary) | set(acceptable)
    exclude = set(exclude or [])
    window = settings["consecutive_window_s"]
    cands: List[Candidate] = []
    for c in roster.values():
        if not c.pageable:
            continue
        active, last_hour, last_paged_at, last_case = _caseload(c, load)
        cand = Candidate(c.id, c.name, c.specialty, c.status, c.zone, 0.0, [],
                         caseload=active, pages_last_hour=last_hour, on_shift=c.status != "off_shift")
        if c.id in exclude:
            cand.excluded = "already paged for this case"
        elif c.specialty in primary:
            cand.factors.append((f"{SPECIALTY_LABEL[c.specialty]} is the right service", 100))
        elif c.specialty in acceptable:
            cand.factors.append((f"{SPECIALTY_LABEL[c.specialty]} can take it", 60))
        else:
            cand.excluded = f"{SPECIALTY_LABEL.get(c.specialty, c.specialty)} does not cover {category}"
        if cand.excluded is None:
            # proximity
            if c.zone == REMOTE_ZONE:
                pts = 5 if category in _REMOTE_OK else -30
                cand.factors.append(("remote (telehealth)", pts))
            elif zone and c.zone == zone:
                cand.factors.append(("already on this unit", 25))
                if priority <= 2:
                    cand.factors.append(("P1/P2 zone escalation", 20))
            elif zone and c.zone in ZONES and ZONES[c.zone].floor == zone_floor:
                cand.factors.append(("same floor", 15))
            elif zone:
                d = floor_distance(c.zone, zone)
                cand.factors.append((f"{d} floor{'s' if d != 1 else ''} away", max(0, 10 - 3 * d)))
            if c.on_call:
                cand.factors.append(("on call", 10))
            # status
            status_pts = {"available": 0, "paging": -15, "on_case": -20, "in_procedure": -45}.get(c.status)
            if c.status == "off_shift":
                cand.excluded = "off shift"
            elif status_pts:
                label = {"paging": "has an unanswered page", "on_case": "on another case",
                         "in_procedure": "in a procedure"}[c.status]
                cand.factors.append((label, status_pts))
            if active:
                cand.factors.append((f"{active} active page{'s' if active > 1 else ''}", -8 * active))
            if last_hour:
                cand.factors.append((f"{last_hour} page{'s' if last_hour > 1 else ''} this hour", -3 * last_hour))
            # guardrails that exclude
            if cand.excluded is None and last_paged_at and now - last_paged_at < window and last_case != case_id:
                cand.excluded = f"paged {int((now - last_paged_at) // 60)} min ago for another case"
            if cand.excluded is None and priority >= 3 and last_hour > settings["workload_max_per_hour"]:
                cand.excluded = f"{last_hour} pages this hour (limit {settings['workload_max_per_hour']} for P3/P4)"
        cand.score = sum(p for _, p in cand.factors)
        cands.append(cand)

    eligible = sorted([c for c in cands if c.excluded is None], key=lambda c: -c.score)
    guardrails: List[str] = []
    if any(c.excluded and c.excluded.startswith("paged ") for c in cands):
        guardrails.append("Consecutive-paging protection removed someone paged minutes ago for another case.")
    if any(c.excluded and "limit" in c.excluded for c in cands):
        guardrails.append("Workload limit skipped someone over the hourly P3/P4 limit.")
    on_shift_alt = any(c.on_shift for c in eligible)
    consecutive_alt = bool(eligible)

    # P1 last resort: nobody eligible -> reconsider the consecutive-paging and off-shift exclusions
    if not eligible and priority == 1:
        fallback = [c for c in cands if c.excluded and (c.excluded.startswith("paged ") or c.excluded == "off shift")
                    and (c.specialty in primary or c.specialty in acceptable)]
        for c in fallback:
            c.factors.append((f"P1 last resort ({c.excluded})", -60 if c.excluded == "off shift" else -25))
            c.score = sum(p for _, p in c.factors)
            c.excluded = None
        eligible = sorted(fallback, key=lambda c: -c.score)
        if eligible:
            guardrails.append("P1 last resort: nobody eligible, so the consecutive/off-shift exclusions were lifted.")

    if not eligible:
        return RouteDecision("unassigned", None, [], _top(cands, relevant), guardrails, primary, acceptable,
                             co_notify_for(category, priority),
                             reasoning="No clinician who covers this case is eligible. The operator must assign it.",
                             on_shift_alternative=False, consecutive_alternative=False)

    # near-tie load distribution: top three within the tie band -> lightest caseload
    top = eligible[0]
    band = [c for c in eligible[:3] if c.score >= top.score * (1 - settings["near_tie_pct"]) and top.score > 0]
    if len(band) > 1:
        light = min(band, key=lambda c: (c.caseload + c.pages_last_hour, -c.score))
        if light.id != top.id:
            guardrails.append(f"Near-tie ({len(band)} within {int(settings['near_tie_pct'] * 100)}%): "
                              f"{light.name} has the lighter caseload, so gets it over {top.name}.")
            eligible.remove(light)
            eligible.insert(0, light)
            top = light

    # A P3/P4 never interrupts a procedure. If someone who covers the case is free,
    # it goes to them; if nobody is, it waits (time-queued) for the first person
    # out of their procedure rather than pulling anyone from a sterile field.
    mode, hold_until = "dispatch", None
    if priority >= 3 and top.status == "in_procedure":
        free = [c for c in eligible if c.status == "available"]
        if free:
            alt = free[0]
            guardrails.append(f"Not interrupting {top.name} mid-procedure for a P{priority}: "
                              f"{alt.name} is free and covers it.")
            eligible.remove(alt)
            eligible.insert(0, alt)
            top = alt
        else:
            busy = [c for c in eligible if c.status == "in_procedure" and roster[c.id].busy_until]
            soonest = min(busy, key=lambda c: roster[c.id].busy_until) if busy else None
            if soonest and roster[soonest.id].busy_until - now <= settings["hold_window_s"]:
                until = roster[soonest.id].busy_until
                mode, hold_until = "hold", until
                if soonest.id != top.id:
                    eligible.remove(soonest)
                    eligible.insert(0, soonest)
                    top = soonest
                guardrails.append(f"Time-queued: nobody is free; {top.name} is out of a procedure in "
                                  f"{max(1, int((until - now) // 60))} min, so this P{priority} waits for them.")

    backups = [c.id for c in eligible[1:4]]
    why = ", ".join(a for a, p in top.factors if p > 0)
    reasoning = f"{top.name} ({SPECIALTY_LABEL[top.specialty]}): {why or 'best available'}."
    return RouteDecision(mode, top.id, backups, _top(cands, relevant), guardrails, primary, acceptable,
                         co_notify_for(category, priority), hold_until, reasoning,
                         on_shift_alternative=on_shift_alt, consecutive_alternative=consecutive_alt)


def _top(cands: List[Candidate], relevant: set, n: int = 8) -> List[Candidate]:
    """Eligible first by score, then the excluded staff who COULD have taken the case
    (their exclusion is the interesting part: consecutive paging, workload, shift)."""
    ok = sorted([c for c in cands if c.excluded is None], key=lambda c: -c.score)
    no = sorted([c for c in cands if c.excluded is not None and c.specialty in relevant], key=lambda c: -c.score)
    return (ok + no)[:n]
