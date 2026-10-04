"""CDS Hooks 2.0: AegisOS's findings inside the EHR, at the moment a clinician acts.

CDS Hooks is how decision support plugs into an EHR's workflow without the clinician
opening another application: the EHR calls a service when a chart is opened
(``patient-view``) or an order is about to be signed (``order-sign``), and shows the
cards it returns. Two services:

* ``aegis-patient-view``: the room's sepsis risk against its detector's own alarm
  line, the active detections with their evidence, and, from the integration plane,
  any of this patient's feeds that are quarantined ("creatinine withheld: lab feed
  failed its evidence"). A clinician who sees a normal creatinine in the EHR should
  know that AegisOS stopped trusting the feed it came from.
* ``aegis-order-sign``: the draft medication orders screened against the patient's
  other medications through the pharmacopoeia's screen (DDInter plus the curated
  CYP450 / QT / renal layer) and against the room's recorded allergies; a critical
  card carries a suggestion to remove the draft order.

Cards follow the spec: ``summary`` (≤ 140 characters), ``indicator`` (info / warning /
critical), ``source``, ``detail`` in markdown, ``suggestions`` with ``actions``, and
``links`` back into AegisOS. Feedback (accepted / overridden, with the reason) is
journalled.

Research and educational use only: these cards are decision support from a system that
is not a medical device, and they say so.
"""
from __future__ import annotations

import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional

SOURCE = {"label": "AegisOS (research use; not a medical device)", "topic": {"code": "aegis", "display": "AegisOS"}}

_SEVERITY = {"moderate": 1, "major": 2}      # what reaches a card; minor interactions do not

SERVICES = [
    {"hook": "patient-view", "id": "aegis-patient-view", "title": "AegisOS room status",
     "description": "Sepsis risk against the detector's own alarm line, active detections in the patient's room, "
                    "and any of the patient's data feeds AegisOS has quarantined.",
     "prefetch": {"patient": "Patient/{{context.patientId}}"}},
    {"hook": "order-sign", "id": "aegis-order-sign", "title": "AegisOS medication safety",
     "description": "Draft medication orders screened for interactions (DDInter plus a curated CYP450, QT and "
                    "renal layer) and against the patient's recorded allergies.",
     "prefetch": {"patient": "Patient/{{context.patientId}}"}},
]


def _card(summary: str, indicator: str, detail: str = "", links: Optional[List[Dict[str, str]]] = None,
          suggestions: Optional[List[Dict[str, Any]]] = None, override: bool = False) -> Dict[str, Any]:
    c: Dict[str, Any] = {"uuid": str(uuid.uuid4()), "summary": summary[:140], "indicator": indicator,
                         "source": SOURCE, "detail": detail}
    if links:
        c["links"] = links
    if suggestions:
        c["suggestions"] = suggestions
        c["selectionBehavior"] = "at-most-one"
    if override:
        c["overrideReasons"] = [{"code": "clinical-judgment", "display": "Clinical judgment, risk accepted"},
                                {"code": "already-aware", "display": "Already aware; monitoring"}]
    return c


def _drugs_from(draft: Any) -> List[Dict[str, str]]:
    """(id, name) of each MedicationRequest in ``context.draftOrders`` (a Bundle)."""
    out = []
    entries = (draft or {}).get("entry") or [] if isinstance(draft, dict) else []
    for e in entries:
        r = (e or {}).get("resource") or {}
        if r.get("resourceType") != "MedicationRequest":
            continue
        cc = r.get("medicationCodeableConcept") or {}
        name = cc.get("text") or next((c.get("display") for c in cc.get("coding") or [] if c.get("display")), None)
        if name:
            m = re.match(r"\s*([A-Za-z][A-Za-z\-]+)", name)
            out.append({"id": r.get("id") or "", "name": name, "generic": (m.group(1) if m else name).lower()})
    return out


class CDSHooks:
    def __init__(self, room_for: Callable[[str], Optional[str]], room_unit: Callable[[str], Optional[Dict[str, Any]]],
                 feed: Callable[[str], Optional[Dict[str, Any]]], screen: Optional[Callable[..., Any]] = None,
                 app_url: str = "", journal: Optional[Callable[..., None]] = None) -> None:
        self.room_for = room_for          # patientId (FHIR id or MRN) -> room
        self.room_unit = room_unit
        self.feed = feed
        self.screen = screen
        self.app_url = app_url.rstrip("/")
        self.journal = journal
        self.calls: List[Dict[str, Any]] = []

    def discovery(self) -> Dict[str, Any]:
        return {"services": SERVICES}

    def _link(self, room: str) -> List[Dict[str, str]]:
        if not self.app_url:
            return []
        return [{"label": f"Open room {room} in AegisOS", "url": f"{self.app_url}/room/?id={room}", "type": "absolute"}]

    def call(self, service: str, req: Dict[str, Any]) -> Dict[str, Any]:
        t0 = time.perf_counter()
        ctx = (req or {}).get("context") or {}
        pid = str(ctx.get("patientId") or "")
        room = self.room_for(pid) if pid else None
        if service == "aegis-patient-view":
            cards = self._patient_view(room, pid)
        elif service == "aegis-order-sign":
            cards = self._order_sign(room, pid, ctx.get("draftOrders"))
        else:
            return {"cards": [], "error": f"no service {service!r}"}
        self.calls.append({"at": time.time(), "service": service, "hook": req.get("hook"), "patient": pid,
                           "room": room, "cards": len(cards), "ms": round((time.perf_counter() - t0) * 1000, 2),
                           "hookInstance": req.get("hookInstance")})
        del self.calls[:-100]
        return {"cards": cards}

    # ------------------------------------------------------------------ #
    def _patient_view(self, room: Optional[str], pid: str) -> List[Dict[str, Any]]:
        if room is None:
            return [_card("AegisOS does not monitor this patient", "info",
                          f"No occupied AegisOS room matches patient `{pid}`.")]
        u = self.room_unit(room) or {}
        if not u.get("occupied"):
            return [_card(f"Room {room} is empty in AegisOS", "info")]
        cards: List[Dict[str, Any]] = []
        name = (u["record"].get("patient") or {}).get("name") or "this patient"
        idx, prob = u.get("risk"), u.get("probability")
        if idx is not None and prob is not None:
            ind = "critical" if idx >= 0.70 else "warning" if idx >= 0.35 else "info"
            cards.append(_card(f"Sepsis risk {idx:.2f} on AegisOS's index ({'above' if idx >= 0.70 else 'below'} "
                               f"its 0.70 alarm line)", ind,
                               f"Calibrated probability **{prob:.3f}** from AegisOS's sepsis detector for {name} "
                               f"(room {room}). The index is pinned so 0.70 is this detector's own validated alarm "
                               f"point. Research use only; not a medical device.", self._link(room)))
        for f in (u.get("findings") or [])[:4]:
            if f.get("severity") not in ("critical", "urgent", "watch"):
                continue
            ev = "; ".join(f"{e['label']} {e['value']}{(' ' + e['unit']) if e.get('unit') else ''}"
                           for e in (f.get("evidence") or [])[:4])
            cards.append(_card(f"{f['title']} ({f['severity']}) in room {room}",
                               "critical" if f["severity"] == "critical" else "warning",
                               f"**{f['title']}**: {ev}.\n\n{f.get('response') or ''}\n\n"
                               f"_Source: {f.get('source') or f.get('detector')}_", self._link(room)))
        feed = self.feed(room) or {}
        bad = [c for c in feed.get("channels") or [] if c.get("status") == "quarantined"]
        for c in bad[:3]:
            cards.append(_card(f"{c['label']} values withheld: the {c.get('interface_name') or 'interface'} feed "
                               f"failed its evidence", "warning",
                               f"AegisOS stopped trusting the feed that sends **{c['label']}** "
                               f"(local code `{c['code']}`): {c.get('fault') or 'its live evidence failed'}. "
                               f"A value shown elsewhere may come from that feed. New values are withheld until "
                               f"the mapping is reviewed.", self._link(room)))
        if not cards:
            cards.append(_card(f"AegisOS: nothing active in room {room}", "info",
                               "No detection above information level, and every feed for this patient passes "
                               "its evidence.", self._link(room)))
        return cards

    def _order_sign(self, room: Optional[str], pid: str, draft: Any) -> List[Dict[str, Any]]:
        drugs = _drugs_from(draft)
        if not drugs:
            return []
        u = (self.room_unit(room) or {}) if room else {}
        rec = u.get("record") or {}
        active = [o["name"] for o in rec.get("orders") or [] if o.get("status") == "active" and o.get("kind") == "drug"]
        cards: List[Dict[str, Any]] = []
        allergies = [a.get("substance", "").lower() for a in rec.get("allergies") or []]
        for d in drugs:
            dn = d["name"].lower()
            hit = next((a for a in allergies if a and (a in dn or dn in a or
                                                       (a == "penicillin" and any(x in dn for x in (
                                                           "cillin", "piperacillin", "amoxi")))
                                                       or (a == "sulfa" and "sulfa" in dn))), None)
            if hit:
                cards.append(_card(f"{d['name']}: recorded allergy to {hit}", "critical",
                                   f"The patient's record (room {room}) lists an allergy to **{hit}**.",
                                   self._link(room) if room else None,
                                   [{"label": f"Remove the {d['name']} order", "uuid": str(uuid.uuid4()),
                                     "isRecommended": True,
                                     "actions": [{"type": "delete", "description": f"Remove {d['name']}",
                                                  "resourceId": [f"MedicationRequest/{d['id']}"]}]}],
                                   override=True))
        if self.screen is not None:
            names = [d["generic"] for d in drugs] + [a.lower() for a in active]
            try:
                found = self.screen(names, {"room": room}) or []
            except Exception:
                found = []
            drafted = {d["generic"]: d for d in drugs}
            # one card per set of drugs: the knowledge layers report the same pair separately
            # (DDInter "moderate", the curated QT layer "major"), and two cards of different
            # severity for one hazard read as two hazards
            groups: Dict[frozenset, Dict[str, Any]] = {}
            for h in found:
                hd = h.to_dict() if hasattr(h, "to_dict") else (h if isinstance(h, dict) else {})
                sev = str(hd.get("severity") or "").lower()
                if sev not in _SEVERITY:
                    continue
                pair = [x.strip() for x in str(hd.get("a") or "").split(",") if x.strip()] + \
                    ([str(hd["b"])] if hd.get("b") else [])
                mine = [p for p in pair if p in drafted]
                if not mine:
                    continue                     # an interaction between drugs already given: not this order's
                g = groups.setdefault(frozenset(pair), {"pair": pair, "sev": sev, "mine": mine, "why": []})
                if _SEVERITY[sev] > _SEVERITY[g["sev"]]:
                    g["sev"] = sev
                g["why"].append((_SEVERITY[sev], f"{hd.get('mechanism') or 'documented interaction'} "
                                                 f"({hd.get('source')}, {sev})"))
            # a pair inside a larger group that is at least as severe is the same hazard seen pairwise
            for k in sorted(groups, key=len):
                big = next((b for b in groups if k < b and _SEVERITY[groups[b]["sev"]] >= _SEVERITY[groups[k]["sev"]]),
                           None)
                if big is not None:
                    small = groups.pop(k)
                    groups[big]["why"] += [(r, f"{' + '.join(small['pair'])}: {t}") for r, t in small["why"]]
            for g in sorted(groups.values(), key=lambda g: -_SEVERITY[g["sev"]]):
                ind = "critical" if g["sev"] == "major" else "warning"
                draft_d = drafted[g["mine"][0]]
                cards.append(_card(f"{' + '.join(g['pair'])}: {g['sev']} interaction", ind,
                                   f"{'; '.join(dict.fromkeys(t for _, t in sorted(g['why'], key=lambda w: -w[0])))}. "
                                   f"Screened against the loaded interaction "
                                   f"knowledge base, exhaustive over what is loaded, not over every drug.",
                                   self._link(room) if room else None,
                                   [{"label": f"Remove the {draft_d['name']} order", "uuid": str(uuid.uuid4()),
                                     "actions": [{"type": "delete", "description": f"Remove {draft_d['name']}",
                                                  "resourceId": [f"MedicationRequest/{draft_d['id']}"]}]}]
                                   if ind == "critical" else None, override=ind == "critical"))
        return cards

    def feedback(self, service: str, body: Dict[str, Any], who: str) -> Dict[str, Any]:
        items = (body or {}).get("feedback") or []
        for f in items:
            if self.journal:
                self.journal("cds.feedback", service=service, card=f.get("card"), outcome=f.get("outcome"),
                             override=(f.get("overrideReason") or {}).get("reason", {}).get("code")
                             if isinstance(f.get("overrideReason"), dict) else None, by=who)
        return {"status": "ok", "received": len(items)}
