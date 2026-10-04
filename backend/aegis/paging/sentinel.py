"""Sentinel: the agent that watches the paging system instead of patients.

It runs on a 30-second timer, never pages anyone, and surfaces what no single
page shows: a surge of alerts on one unit, a specialty with nobody available, one
clinician absorbing the whole floor, pages that keep timing out — and two that
exist because this is a sepsis system: several detector alerts on one unit
(a possible infection cluster), and a sepsis page accepted without the Hour-1
bundle started.

A recommendation is keyed by its pattern (``zone:5e``, ``overload:dr.reyes``), so
a pattern that persists updates one card instead of stacking duplicates, and a
card clears itself when its condition does.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Dict, List, Optional

from backend.aegis.paging.hospital import SPECIALTY_LABEL, ZONES, Clinician

ZONE_WINDOW_S = 20 * 60
ZONE_THRESHOLD = 3
OVERLOAD_PER_HOUR = 5
OVERLOAD_ACTIVE = 3
STORM_WINDOW_S = 15 * 60
STORM_THRESHOLD = 3
CLUSTER_WINDOW_S = 6 * 3600
CLUSTER_THRESHOLD = 2
BUNDLE_CLOCK_S = 60 * 60
WATCHED_SPECIALTIES = ("critical_care", "cardiology", "neurology", "surgery", "emergency", "hospitalist", "pharmacy")


@dataclass
class Rec:
    id: str
    kind: str                  # zone_surge | coverage_hole | overload | escalation_storm | sepsis_cluster | bundle_clock
    severity: str              # info | warn | critical
    title: str
    detail: str
    action: str
    evidence: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    acknowledged: bool = False
    active: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class Sentinel:
    def __init__(self) -> None:
        self.recs: Dict[str, Rec] = {}
        self.last_scan = 0.0
        # other planes' system-wide findings, folded into each scan (the integration engine's
        # quarantined feeds). Called under the paging OS's lock: it must not call back into it.
        self.external: Optional[Callable[[float], List[Rec]]] = None

    def scan(self, now: float, alerts: List[Dict[str, Any]], pages: List[Dict[str, Any]],
             roster: Dict[str, Clinician], load: Dict[str, Dict[str, Any]],
             bundle_started: Optional[Callable[[Optional[str]], Optional[bool]]] = None) -> List[Rec]:
        """Return the recommendations that changed (new, updated or cleared)."""
        self.last_scan = now
        found: Dict[str, Rec] = {}

        # 1 · a surge of alerts on one unit. Alerts only (not the detector's notes), and
        # across at least two rooms: one patient alarming again is that patient's
        # episode, not evidence that the unit is short
        by_zone: Dict[str, List[Dict[str, Any]]] = {}
        for a in alerts:
            if a.get("kind", "alert") == "alert" and a.get("zone") and now - a["created_at"] <= ZONE_WINDOW_S \
                    and not a.get("history"):
                by_zone.setdefault(a["zone"], []).append(a)
        for z, items in by_zone.items():
            if len(items) >= ZONE_THRESHOLD and len({a.get("room") for a in items}) >= 2:
                name = ZONES[z].name if z in ZONES else z
                p1 = sum(1 for a in items if a.get("priority") == 1)
                found[f"zone:{z}"] = Rec(
                    f"zone:{z}", "zone_surge", "critical" if p1 >= 2 else "warn",
                    f"{len(items)} alerts on {name} in {ZONE_WINDOW_S // 60} min",
                    f"{p1} of them P1. Pages on one unit this close together usually mean the unit is short, not unlucky.",
                    f"Send the rapid response nurse to {name} and ask the charge nurse there for a huddle.",
                    [f"{a['id']} P{a['priority']} {a.get('room') or ''}".strip() for a in items[:5]])

        # 2 · a specialty with nobody available on shift
        for spec in WATCHED_SPECIALTIES:
            staff = [c for c in roster.values() if c.specialty == spec and c.pageable]
            if not staff:
                continue
            free = [c for c in staff if c.status == "available"]
            on_shift = [c for c in staff if c.status != "off_shift"]
            if not free:
                soonest = min((c.busy_until for c in on_shift if c.busy_until), default=None)
                eta = f" First free in {max(1, int((soonest - now) // 60))} min." if soonest and soonest > now else ""
                found[f"coverage:{spec}"] = Rec(
                    f"coverage:{spec}", "coverage_hole", "warn" if on_shift else "critical",
                    f"No {SPECIALTY_LABEL[spec].lower()} available",
                    f"{len(on_shift)} on shift, all busy." + eta if on_shift else "Nobody from this service is on shift.",
                    "Line up a backup (on-call from home, telehealth, or a covering service) before the next page needs one.",
                    [f"{c.name}: {c.status.replace('_', ' ')}" for c in staff])

        # 3 · one clinician taking the whole floor
        for cid, l in load.items():
            c = roster.get(cid)
            if not c:
                continue
            if l.get("last_hour", 0) >= OVERLOAD_PER_HOUR or l.get("active", 0) >= OVERLOAD_ACTIVE:
                found[f"overload:{cid}"] = Rec(
                    f"overload:{cid}", "overload", "warn",
                    f"{c.name} is overloaded",
                    f"{l.get('last_hour', 0)} pages this hour, {l.get('active', 0)} open.",
                    f"Route new P3/P4 pages away from {c.name.split(',')[0]} and hand one open case to a colleague.",
                    [])

        # 4 · pages that keep timing out
        timeouts = [(p, at) for p in pages for at in p.get("attempts", [])
                    if at.get("outcome") == "timeout" and now - (at.get("at") or 0) <= STORM_WINDOW_S]
        if len(timeouts) >= STORM_THRESHOLD:
            found["storm"] = Rec(
                "storm", "escalation_storm", "critical",
                f"{len(timeouts)} pages timed out in {STORM_WINDOW_S // 60} min",
                "Pages are escalating because nobody answers, not because anyone declined.",
                "Check pager coverage and device battery for the staff involved; call rather than page P1s.",
                [f"{p['id']} → {at.get('name', at.get('clinician'))}" for p, at in timeouts[:5]])

        # 5 · sepsis alerts clustering on one unit (the detector's own pages)
        det_by_zone: Dict[str, List[Dict[str, Any]]] = {}
        for a in alerts:
            if a.get("source") == "detector" and a.get("zone") and now - a["created_at"] <= CLUSTER_WINDOW_S:
                det_by_zone.setdefault(a["zone"], []).append(a)
        for z, items in det_by_zone.items():
            rooms = sorted({a.get("room") for a in items if a.get("room")})
            if len(rooms) >= CLUSTER_THRESHOLD:
                name = ZONES[z].name if z in ZONES else z
                found[f"cluster:{z}"] = Rec(
                    f"cluster:{z}", "sepsis_cluster", "warn",
                    f"Sepsis alerts on {len(rooms)} beds in {name}",
                    "Several patients on one unit crossing the sepsis alarm within hours can be coincidence or a shared "
                    "source. The detector cannot tell which.",
                    "Tell infection prevention; review lines, shared equipment and cultures across these beds.",
                    [f"room {r}" for r in rooms])

        # 6 · a sepsis page accepted but the bundle not started
        for p in pages:
            if p.get("category") == "sepsis" and p.get("priority", 4) <= 2 and p.get("status") == "accepted":
                since = now - (p.get("accepted_at") or now)
                started = bundle_started(p.get("bed")) if bundle_started else None
                if since >= BUNDLE_CLOCK_S * 0.5 and started is False:
                    left = max(0, int((BUNDLE_CLOCK_S - since) // 60))
                    found[f"bundle:{p['id']}"] = Rec(
                        f"bundle:{p['id']}", "bundle_clock", "critical" if left <= 10 else "warn",
                        f"Hour-1 bundle not started for room {p.get('room')}",
                        f"Sepsis page accepted {int(since // 60)} min ago; no certified antibiotic order yet. "
                        f"{left} min left in the hour.",
                        "Prompt the accepting clinician; pharmacy can prepare the antibiotic now.",
                        [p["id"]])

        if self.external is not None:
            try:
                for rec in self.external(now):
                    found[rec.id] = rec
            except Exception:
                pass
        changed: List[Rec] = []
        for key, rec in found.items():
            old = self.recs.get(key)
            if old and old.active:
                rec.created_at, rec.acknowledged = old.created_at, old.acknowledged
                rec.updated_at = now
                if (old.title, old.detail, old.severity) != (rec.title, rec.detail, rec.severity):
                    changed.append(rec)
            else:
                rec.created_at = rec.updated_at = now
                changed.append(rec)
            self.recs[key] = rec
        for key, rec in list(self.recs.items()):
            if key not in found and rec.active:
                rec.active, rec.updated_at = False, now
                changed.append(rec)
            if not rec.active and now - rec.updated_at > 1800:
                del self.recs[key]
        return changed

    def acknowledge(self, rec_id: str) -> Optional[Rec]:
        rec = self.recs.get(rec_id)
        if rec:
            rec.acknowledged = True
        return rec

    def active(self) -> List[Dict[str, Any]]:
        order = {"critical": 0, "warn": 1, "info": 2}
        return [r.to_dict() for r in sorted((r for r in self.recs.values() if r.active),
                                            key=lambda r: (order.get(r.severity, 3), -r.updated_at))]
