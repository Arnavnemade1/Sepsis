"""The room monitor: every occupied room watched as one unit, continuously.

Owned by the paging OS and ticked from its clock. Every few seconds it:

1. **samples** each synthetic room's feed (``physiology``), and reads the ICU rooms'
   real replayed stay and the live session from the paging OS's monitored beds;
2. **runs the detector library** over each room with that room's record
   (``detectors.run_all``), keeping each finding's onset, persistence and history;
3. **writes the room's timeline** (findings opening, escalating and resolving, labs
   resulting, pages, simulated courses);
4. **escalates**: a finding that calls for a page, and has persisted, raises one
   through the paging OS (triaged, routed, Z3-certified like any other), or raises an
   open page in the room if it is more urgent than the page already there. One room,
   one page: the monitor never pages a room twice for the same episode.

Paging policy by kind of room. Synthetic ward rooms are paged by the monitor. The ICU
replay rooms and the live session are paged by the sepsis detector, as before; the
monitor shows its findings there but does not page, so nothing is paged twice.

The monitor never places or changes an order. The one exception is inside the demo
world: when a simulated pharmacist resolves an allergy page, the synthetic order is
marked held, as a pharmacist would, and the timeline says it was simulated.
"""
from __future__ import annotations

import hashlib
import random
import statistics
import time
from collections import deque
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from backend.aegis.paging.hospital import ICU_ROOMS, LIVE_ROOM, ROOMS, ZONES
from backend.aegis.paging.triage import Triage
from backend.aegis.rooms import detectors as D
from backend.aegis.rooms.physiology import SAMPLE_S, SELF_LIMIT_S, WEIGHTS, Physiology, fitting
from backend.aegis.rooms.records import icu_record, interface_record, live_record, synthetic_record

EVAL_S = 3.0                      # the monitor's pass: rooms whose inputs changed are re-evaluated
REFRESH_S = 15.0                  # and every room at least this often
SERIES_N = 360                    # one hour of synthetic samples (10 s apart)
TIMELINE_N = 200
PERSIST_S = {1: 9.0, 2: 20.0}     # a P1 finding must hold 9 s before it pages, a P2 20 s
URGENT_HOLD_S = 90.0              # a P3 from an urgent-band score must hold a minute and a half
CLEAR_EVALS = 5                   # an episode closes after 5 clear evaluations
SCENARIO_GAP_S = (150.0, 300.0)   # demo: a new deterioration every few minutes
MAX_SCENARIOS = 2
TREATED_S = 75.0                  # a page accepted this long ago means care has begun
_CATEGORY_SPECIFICITY = {"general": 0, "medication": 1}   # prefer a specific category at equal priority
_ICU_CHANNELS = ("hr", "sbp", "map", "rr", "spo2", "temp_c")
_LAB_NAME = {"wbc": "WBC", "anc": "ANC", "glucose": "Glucose", "lactate": "Lactate", "potassium": "Potassium",
             "hemoglobin": "Haemoglobin", "creatinine": "Creatinine", "platelets": "Platelets",
             # keys an interface feed can deliver (backend/aegis/interop/vocab.py)
             "sodium": "Sodium", "chloride": "Chloride", "hco3": "HCO₃", "bun": "BUN", "calcium": "Calcium",
             "ica": "Ionized Ca", "magnesium": "Magnesium", "phosphate": "Phosphate", "hct": "Haematocrit",
             "bilirubin": "Bilirubin", "bilirubin_direct": "Direct bilirubin", "ast": "AST", "alp": "ALP",
             "troponin": "Troponin I", "ptt": "aPTT", "inr": "INR", "fibrinogen": "Fibrinogen", "ph": "pH",
             "paco2": "PaCO₂", "sao2": "SaO₂", "base_excess": "Base excess", "fio2": "FiO₂"}


def _seed(*parts: Any) -> int:
    return int(hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()[:12], 16)


def _active() -> Tuple[str, ...]:
    from backend.aegis.paging.engine import ACTIVE
    return ACTIVE


class RoomMonitor:
    def __init__(self, engine: Any, seed: int = 2026) -> None:
        self.engine = engine
        self.rng = random.Random(seed)
        now = engine.now()
        self.started_at = now
        self.records: Dict[str, Dict[str, Any]] = {}
        self.phys: Dict[str, Physiology] = {}
        self.series: Dict[str, Deque[Dict[str, Any]]] = {}
        self.next_sample: Dict[str, float] = {}
        self.findings: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self.resolved: Dict[str, Deque[Dict[str, Any]]] = {}
        self.timeline: Dict[str, Deque[Dict[str, Any]]] = {}
        self.episodes: Dict[str, Dict[str, Any]] = {}
        self.scores: Dict[str, Dict[str, Any]] = {}
        self.body: Dict[str, Dict[str, Any]] = {}          # organ systems for the anatomical twin
        self.news_hist: Dict[str, Deque[Tuple[float, int]]] = {}
        self.faults: Dict[str, List[str]] = {}
        self.status: Dict[str, Dict[str, Any]] = {}
        self.last_eval = 0.0
        self.next_scenario_at = now + self.rng.uniform(70, 130)
        self.eval_ms: Deque[float] = deque(maxlen=200)
        self.counters = {"evaluations": 0, "findings_opened": 0, "pages_opened": 0, "pages_raised": 0,
                         "pages_linked": 0, "scenarios": 0}
        # hooks the server sets (left None when the paging OS runs standalone, as in the verify suites)
        self.icu_patient: Optional[Callable[[str], Any]] = None          # bed id -> the replayed CinC patient
        self.imported_history: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None
        self.live_history: Optional[Callable[[], Optional[Dict[str, Any]]]] = None
        self.screen: Optional[Callable[..., List[Any]]] = None           # the pharmacopoeia's interaction screen
        self.feed_status: Optional[Callable[[str], Optional[Dict[str, Any]]]] = None   # integration engine's view
        # rooms admitted over HL7 (backend/aegis/interop): vitals and results arrive through mappings
        self.interfaced: Dict[str, Dict[str, Any]] = {}
        self._icu_cache: Dict[str, Tuple[Any, Dict[str, Any]]] = {}
        self._evaluated: Dict[str, Tuple[float, Any]] = {}
        self._last_pass: Dict[str, float] = {}
        for room, p in engine.synthetic.items():
            self._admit(room, p, now)

    # ------------------------------------------------------------------ #
    # occupants
    # ------------------------------------------------------------------ #
    def _admit(self, room: str, p: Dict[str, Any], now: float) -> None:
        rec = synthetic_record(p, now)
        self.records[room] = rec
        ph = Physiology(rec, _seed("phys", room, p["mrn"]), now - SAMPLE_S * 90)
        self.phys[room] = ph
        s: Deque[Dict[str, Any]] = deque(maxlen=SERIES_N)
        # fifteen minutes of history, so a room is never blank when it is first opened
        for k in range(90):
            t = now - SAMPLE_S * (90 - k)
            s.append({"t": t, **ph.sample(t)})
        self.series[room] = s
        self.next_sample[room] = now + (_seed("phase", room) % 1000) / 1000 * SAMPLE_S
        self.timeline[room] = deque(maxlen=TIMELINE_N)
        self.findings[room] = {}
        self.resolved[room] = deque(maxlen=30)
        self.news_hist[room] = deque(maxlen=120)

    def _room_kind(self, room: str) -> Optional[str]:
        if room in self.interfaced:
            return "interface"
        if room in self.records and self.records[room]["kind"] == "synthetic":
            return "synthetic"
        m = self.engine.monitored.get(room)
        if m:
            return "live" if m.get("live") else "replay"
        return None

    def _ensure_monitored(self, room: str) -> Optional[Dict[str, Any]]:
        """Build (or refresh) the record of an ICU or live room from the paging OS's monitored beds."""
        m = self.engine.monitored.get(room)
        if not m:
            return None
        if m.get("live"):
            hist = None
            try:
                hist = self.live_history() if self.live_history else None
            except Exception:
                hist = None
            rec = live_record(room, m, hist)
        else:
            bed = m.get("bed")
            hour = int(m.get("icu_hour") or 0)
            imp = None
            try:
                imp = self.imported_history(bed) if (self.imported_history and bed) else None
            except Exception:
                imp = None
            key = (bed, hour, bool(imp), (imp or {}).get("imported"))
            cached = self._icu_cache.get(room)
            if cached and cached[0] == key:
                rec = cached[1]
            else:
                pt = None
                try:
                    pt = self.icu_patient(bed) if (self.icu_patient and bed) else None
                except Exception:
                    pt = None
                rec = icu_record(room, m, pt, imp)
                self._icu_cache[room] = (key, rec)
        if room not in self.timeline:
            self.timeline[room] = deque(maxlen=TIMELINE_N)
            self.findings[room] = {}
            self.resolved[room] = deque(maxlen=30)
            self.news_hist[room] = deque(maxlen=120)
            self.series[room] = deque(maxlen=SERIES_N)
        self.records[room] = rec
        return rec

    # ------------------------------------------------------------------ #
    # the clock
    # ------------------------------------------------------------------ #
    def tick(self, now: float) -> None:
        """Called from the paging OS's tick, under its lock."""
        for room, ph in self.phys.items():
            if now >= self.next_sample.get(room, 0):
                self.next_sample[room] = now + SAMPLE_S
                v = ph.sample(now)
                self.series[room].append({"t": now, **v})
                for lab, val in ph.new_labs:
                    self._event(room, now, "lab", "info", f"{_LAB_NAME.get(lab, lab)} resulted: {val:g}", lab=lab)
        if now - self.last_eval < EVAL_S:
            return
        self.last_eval = now
        t0 = time.perf_counter()
        for room in list(self.engine.monitored):
            if self._ensure_monitored(room) is not None:
                self._feed_monitored(room, now)
        changed: Dict[str, Dict[str, Any]] = {}
        for room in self._occupied():
            if not self._due(room, now):
                continue
            st = self._evaluate(room, now)
            if st != self.status.get(room):
                changed[room] = st
                self.status[room] = st
        self._scenarios(now)
        self.counters["evaluations"] += 1
        self.eval_ms.append((time.perf_counter() - t0) * 1000)
        if changed:
            self.engine.emit("rooms", changed)

    def _due(self, room: str, now: float) -> bool:
        """Re-run a room's detectors when its inputs changed (a new sample, a new replay hour) and at
        least every REFRESH_S regardless, for the detectors that read the clock (a stale result, a
        monitor that stopped sending). The live room runs every pass."""
        series = self.series.get(room)
        mark = (series[-1]["t"] if series else None, len(self.records.get(room, {}).get("orders") or []))
        m = self.engine.monitored.get(room)
        if m and m.get("live"):
            return True
        last = self._evaluated.get(room)
        if last is None or last[1] != mark or now - last[0] >= REFRESH_S:
            self._evaluated[room] = (now, mark)
            return True
        # unchanged: keep its findings current so they are not read as resolved
        for st in (self.findings.get(room) or {}).values():
            if st["last_seen"] == last[0] or st["last_seen"] == self._last_pass.get(room):
                st["last_seen"] = now
        self._last_pass[room] = now
        return False

    def _occupied(self) -> List[str]:
        return sorted(set(self.phys) | set(self.engine.monitored) | set(self.interfaced))

    def _feed_monitored(self, room: str, now: float) -> None:
        m = self.engine.monitored.get(room) or {}
        s = self.series[room]
        if m.get("live"):
            v = m.get("vitals") or {}
            row = {"t": now, "hr": v.get("heart_rate"), "map": v.get("map"), "spo2": v.get("spo2"),
                   "rr": v.get("respiratory_rate"), "temp_c": v.get("skin_temp")}
            if any(row[k] is not None for k in ("hr", "map", "spo2", "rr")):
                s.append({k: (round(x, 1) if isinstance(x, float) else x) for k, x in row.items()})
        else:
            rec = self.records.get(room) or {}
            rows = rec.get("replay_vitals") or []
            s.clear()
            s.extend(rows[-SERIES_N:])

    # ------------------------------------------------------------------ #
    # detection
    # ------------------------------------------------------------------ #
    def _ctx(self, room: str, now: float) -> D.Ctx:
        rec = self.records[room]
        kind = rec["kind"]
        series = list(self.series.get(room) or [])
        if kind == "synthetic":
            recent = series[-3:]
            v: Dict[str, Any] = {}
            for k in ("hr", "sbp", "dbp", "map", "rr", "spo2", "temp_c"):
                xs = [r[k] for r in recent if r.get(k) is not None]
                if xs:
                    v[k] = statistics.median(xs)     # artefact rejection: the median of the last 30 s
            if series:
                last = series[-1]
                v["o2_lpm"] = last.get("o2_lpm", 0.0)
                v["avpu"] = last.get("avpu", "A")
                v["chronic_confusion"] = last.get("chronic_confusion", False)
            labs = _merge_labs(rec.get("labs") or {}, self.phys[room].labs)
            return D.Ctx(room, kind, rec, v, series[-1]["t"] if series else None, series, labs, now,
                         flags=frozenset(rec.get("flags") or []))
        if kind == "interface":
            v = {}
            for k in ("hr", "sbp", "dbp", "map", "rr", "spo2", "temp_c"):
                xs = [r[k] for r in series[-6:] if r.get(k) is not None][-3:]
                if xs:
                    v[k] = statistics.median(xs)     # the median of the last three readings, as for a synthetic feed
            last_o2 = next((r["o2_lpm"] for r in reversed(series[-6:]) if r.get("o2_lpm") is not None), None)
            if last_o2 is not None:
                v["o2_lpm"] = last_o2
            labs = _merge_labs(rec.get("labs") or {}, self.interfaced[room]["labs"])
            return D.Ctx(room, kind, rec, v, series[-1]["t"] if series else None, series, labs, now,
                         flags=frozenset(rec.get("flags") or []))
        if kind == "replay":
            v = {}
            for k in _ICU_CHANNELS:
                for r in reversed(series[-6:]):
                    if r.get(k) is not None:
                        v[k] = r[k]
                        break
            m = self.engine.monitored.get(room) or {}
            hour = float(m.get("icu_hour") or (series[-1]["t"] if series else 0))
            ml = {"risk": m.get("risk"), "probability": m.get("probability"), "drivers": m.get("drivers") or []}
            return D.Ctx(room, kind, rec, v, hour, series, rec.get("labs") or {}, hour, clock="icu_hours", ml=ml,
                         flags=frozenset(rec.get("flags") or []))
        m = self.engine.monitored.get(room) or {}
        v = {k: x for k, x in (series[-1] if series else {}).items() if k != "t" and x is not None}
        return D.Ctx(room, kind, rec, v, series[-1]["t"] if series else None, series, rec.get("labs") or {}, now,
                     ml={"risk": m.get("risk"), "probability": m.get("probability"), "drivers": m.get("drivers") or []},
                     flags=frozenset(rec.get("flags") or []))

    def _evaluate(self, room: str, now: float) -> Dict[str, Any]:
        rec = self.records.get(room)
        if rec is None:
            return {}
        c = self._ctx(room, now)
        found, faults = D.run_all(c, self.screen)
        self.faults[room] = faults
        cur = self.findings.setdefault(room, {})
        seen = set()
        for f in found:
            d = f.to_dict()
            seen.add(f.key)
            st = cur.get(f.key)
            if st is None:
                st = cur[f.key] = {"finding": d, "first_seen": now, "last_seen": now, "open": False, "peak": "info",
                                   "pending": 0, "below": 0, "opened_at": None, "page_since": None,
                                   "history": [(now, d["severity"])]}
            elif d["severity"] != st["finding"]["severity"]:
                st["history"].append((now, d["severity"]))
                del st["history"][:-20]
            if d.get("page") is None:
                st["page_since"] = None
            elif st["page_since"] is None:
                st["page_since"] = now
            st.update(finding=d, last_seen=now)
            self._track(room, st, d["severity"], now)
        for key in [k for k in cur if k not in seen]:
            st = cur[key]
            st["page_since"] = None
            self._track(room, st, None, now)
            if not st["open"]:
                cur.pop(key)
        # headline scores, whether or not they produced a finding
        sc: Dict[str, Any] = {}
        if rec["kind"] != "replay" and c.v:
            tgt = rec.get("spo2_target")
            n = D.news2_score(c.v, bool(tgt and tgt.get("scale") == 2), bool(c.v.get("chronic_confusion")))
            sc["news2"] = {"score": n["score"], "band": n["band"], "complete": n["complete"], "missing": n["missing"],
                           "scale": 2 if tgt and tgt.get("scale") == 2 else 1,
                           "parts": [{"label": a, "value": b, "unit": u, "points": p} for a, b, u, p in n["parts"]]}
            hist = self.news_hist.setdefault(room, deque(maxlen=120))
            if not hist or now - hist[-1][0] >= 30:
                hist.append((now, n["score"]))
        if c.v.get("hr") and c.v.get("sbp"):
            sc["shock_index"] = round(c.v["hr"] / c.v["sbp"], 2)
        rr, sbp, avpu = c.v.get("rr"), c.v.get("sbp"), c.v.get("avpu")
        if rr is not None or sbp is not None:
            altered = avpu in ("V", "P", "U") or (avpu == "C" and not c.v.get("chronic_confusion"))
            sc["qsofa"] = int(rr is not None and rr >= 22) + int(sbp is not None and sbp <= 100) + int(bool(altered))
        self.scores[room] = sc
        self.body[room] = body_state(rec, c, [st["finding"] for st in cur.values() if st["last_seen"] == now], sc)
        if rec["kind"] in ("synthetic", "interface"):
            self._escalate(room, rec, c, now)
        return self._status_of(room)

    def _track(self, room: str, st: Dict[str, Any], sev: Optional[str], now: float) -> None:
        """Debounced finding episodes, so a score flickering at a threshold writes one line, not twenty.

        A finding opens after two consecutive evaluations above information, escalates when it
        exceeds the worst it has been in this episode, and resolves after five evaluations back
        at information or absent."""
        rank = D.SEVERITY_RANK.get(sev, -1) if sev else -1
        f = st["finding"]
        if rank >= 1:
            st["below"] = 0
            if not st["open"]:
                st["pending"] += 1
                if st["pending"] >= 2:
                    st.update(open=True, peak=sev, opened_at=now, pending=0, opened=dict(f))
                    self.counters["findings_opened"] += 1
                    self._event(room, now, "finding", sev, f["title"], key=f["key"], detector=f["detector"])
            elif rank > D.SEVERITY_RANK[st["peak"]]:
                st["peak"] = sev
                self._event(room, now, "escalated", sev, f["title"], key=f["key"], detector=f["detector"])
            if st["open"]:
                st["worst"] = dict(f) if rank >= D.SEVERITY_RANK.get((st.get("worst") or {}).get("severity"), -1) else st["worst"]
        else:
            st["pending"] = 0
            if st["open"]:
                st["below"] += 1
                if st["below"] >= CLEAR_EVALS:
                    st["open"] = False
                    shown = st.get("worst") or f
                    self.resolved[room].appendleft({"finding": shown, "first_seen": st["opened_at"],
                                                    "resolved_at": now, "peak": st["peak"]})
                    self._event(room, now, "resolved", "info", f"Resolved: {shown['title']}", key=f["key"],
                                detector=f["detector"])
                    st["worst"] = None

    def _status_of(self, room: str) -> Dict[str, Any]:
        cur = self.findings.get(room) or {}
        act = [x["finding"] for x in cur.values() if x["finding"]["severity"] != "info" and x["last_seen"] == self.last_eval]
        worst = max(act, key=lambda f: D.SEVERITY_RANK[f["severity"]], default=None)
        news = (self.scores.get(room) or {}).get("news2")
        ep = self.episodes.get(room) or {}
        med = self.episodes.get(f"{room}:med") or {}
        return {"sev": worst["severity"] if worst else None, "title": worst["title"] if worst else None,
                "news2": news["score"] if news else None, "band": news["band"] if news else None,
                "n": len(act), "episode": bool(ep.get("open") or med.get("open"))}

    # ------------------------------------------------------------------ #
    # escalation: the monitor as an alert source
    # ------------------------------------------------------------------ #
    def _escalate(self, room: str, rec: Dict[str, Any], c: D.Ctx, now: float) -> None:
        """Two tracks per room, each with its own episode: the patient's physiology (the unit's
        physician or rapid response) and the medication record (pharmacy). An allergy conflict
        found while a sepsis page is open is a different problem for a different person."""
        if not self.engine.settings.get("room_monitor_pages", True):
            return
        due: Dict[str, List[Dict[str, Any]]] = {"phys": [], "med": []}
        for st in self.findings[room].values():
            f = st["finding"]
            if not f.get("page") or st["page_since"] is None or st["last_seen"] != now:
                continue
            hold = PERSIST_S.get(f["page"], URGENT_HOLD_S) if not f.get("static") else PERSIST_S[1]
            if now - st["page_since"] >= hold:
                due["med" if f["page_category"] == "medication" else "phys"].append(f)
        for track, paging in due.items():
            self._episode(room, track, rec, c, paging, now)

    def _episode(self, room: str, track: str, rec: Dict[str, Any], c: D.Ctx, paging: List[Dict[str, Any]],
                 now: float) -> None:
        eng = self.engine
        key = room if track == "phys" else f"{room}:med"
        ep = self.episodes.get(key) or {"open": False, "clear": 0, "track": track}
        active = _active()
        if not paging:
            if ep.get("open"):
                ep["clear"] = ep.get("clear", 0) + 1
                pg = eng.pages.get(ep.get("page")) if ep.get("page") else None
                if ep["clear"] >= CLEAR_EVALS and not (pg and pg["status"] in active):
                    ep["open"] = False
                    self._event(room, now, "episode", "info",
                                "Medication episode closed" if track == "med" else "Monitor episode closed: nothing calls for a page")
            self.episodes[key] = ep
            return
        ep["clear"] = 0
        best = min(paging, key=lambda f: (f["page"], -_CATEGORY_SPECIFICITY.get(f["page_category"], 2),
                                          -D.SEVERITY_RANK[f["severity"]]))
        page = eng.pages.get(ep.get("page")) if ep.get("page") else None
        if ep.get("open") and page is not None:
            if page["status"] in active:
                # what the monitor learns while the page is open is written on it
                mon = page.setdefault("monitor", {"findings": []})
                known = {f["key"] for f in mon["findings"]}
                for f in paging:
                    if f["key"] not in known:
                        mon["findings"].append({"key": f["key"], "detector": f["detector"], "title": f["title"],
                                                "severity": f["severity"], "source": f["source"]})
                        page["escalation_log"].append({"at": now, "why": f"room monitor: also {f['title']}",
                                                       "to": page.get("assignee")})
                        eng.emit("page", page)
                if best["page"] < page["priority"]:
                    self._raise(page, best, room, now)
            elif best["page"] < (ep.get("priority") or 4):
                self._open(room, rec, c, best, paging, now, ep)
            self.episodes[key] = ep
            return
        # an open page for this room on the same track (a nurse's medication question is the
        # pharmacy track; anything else is the physiology track) takes the findings instead
        other = [p for p in eng.pages.values() if p.get("room") == room and p["status"] in active
                 and (p.get("category") == "medication") == (track == "med")]
        if other:
            p = min(other, key=lambda p: p["priority"])
            first = ep.get("page") != p["id"]
            ep.update(open=True, page=p["id"], priority=p["priority"], linked=True, opened_at=now, clear=0)
            ph = self.phys.get(room)
            if track == "phys" and ph and ph.scenario and ph.scenario.page is None and ph.scenario.recover_t0 is None:
                ph.scenario.page = p["id"]        # whoever is on that page is now caring for this course
            if first:
                self.counters["pages_linked"] += 1
                self._event(room, now, "page", "info", f"Findings attached to {p['id']} (P{p['priority']}), already open for this room",
                            page=p["id"])
            if best["page"] < p["priority"]:
                self._raise(p, best, room, now)
                ep["priority"] = best["page"]
            self.episodes[key] = ep
            return
        self._open(room, rec, c, best, paging, now, ep)
        self.episodes[key] = ep

    def _open(self, room: str, rec: Dict[str, Any], c: D.Ctx, best: Dict[str, Any], paging: List[Dict[str, Any]],
              now: float, ep: Dict[str, Any]) -> None:
        eng = self.engine
        pt = rec["patient"]
        vit = {k: float(c.v[k]) for k in ("hr", "sbp", "rr", "spo2", "temp_c", "map") if c.v.get(k) is not None}
        others = [f["title"] for f in paging if f is not best][:2]
        allergy = ", ".join(a["substance"] for a in rec.get("allergies") or []) or None
        code = rec.get("code_status")
        text = (f"Room {room} · {pt['name']}: {best['title']}."
                + (f" Also: {'; '.join(others)}." if others else "")
                + (f" Allergies: {allergy}." if allergy and best["page_category"] in ("medication", "sepsis") else "")
                + (f" Code status {code}." if code and code != "Full code" else ""))
        q = (self.scores.get(room) or {}).get("qsofa", 0)
        infection = bool(c.has("infection", "neutropenia") or (c.v.get("temp_c") or 37) >= 38.3)
        shock = bool((c.v.get("map") or 99) < 65 or (c.v.get("sbp") or 200) < 90)
        tri = Triage(best["page"], best["page_category"] if best["page_category"] in (
            "sepsis", "cardiac", "respiratory", "medication", "surgical", "neuro") else "general",
            room, [f["title"] for f in paging], vit, [f"{f['detector']}: {f['title']}" for f in paging],
            f"Room monitor: {best['title']}. {best.get('response', '')}", "monitor", None,
            {"qsofa": q, "infection_signs": infection, "shock_signs": shock,
             "altered": c.v.get("avpu") in ("C", "V", "P", "U") and not c.v.get("chronic_confusion")})
        page = eng._open_case(tri, text, "monitor", "AegisOS room monitor",
                              synthetic=bool((rec.get("provenance") or {}).get("synthetic", True)))
        page["monitor"] = {"findings": [{"key": f["key"], "detector": f["detector"], "title": f["title"],
                                         "severity": f["severity"], "source": f["source"]} for f in paging]}
        ep.update(open=True, page=page["id"], priority=best["page"], linked=False, opened_at=now, clear=0,
                  key=best["key"])
        self.counters["pages_opened"] += 1
        self._event(room, now, "page", best["severity"], f"Paged P{best['page']}: {best['title']}", page=page["id"])
        ph = self.phys.get(room)
        if ep.get("track", "phys") == "phys" and ph and ph.scenario and not ph.scenario.page:
            ph.scenario.page = page["id"]

    def _raise(self, page: Dict[str, Any], best: Dict[str, Any], room: str, now: float) -> None:
        """Raise an open page to the monitor's priority (never lower it)."""
        eng = self.engine
        old = page["priority"]
        page["priority"] = best["page"]
        page["escalation_log"].append({"at": now, "why": f"raised P{old} → P{best['page']} by the room monitor: {best['title']}",
                                       "to": page.get("assignee")})
        eng._set_status(page, page["status"])
        eng.emit("page", page)
        if page.get("assignee"):
            eng.emit("notify", {"to": page["assignee"], "page": page["id"], "priority": best["page"], "room": room,
                                "text": f"Raised to P{best['page']}: {best['title']}", "kind": "upgrade"})
        self.counters["pages_raised"] += 1
        self._event(room, now, "page", best["severity"], f"Raised {page['id']} to P{best['page']}: {best['title']}",
                    page=page["id"])

    # ------------------------------------------------------------------ #
    # the demo's simulated courses
    # ------------------------------------------------------------------ #
    def _scenarios(self, now: float) -> None:
        eng = self.engine
        active = _active()
        for room, ph in self.phys.items():
            s = ph.scenario
            if not s or s.recover_t0 is not None:
                continue
            if s.page is None:
                # the page for THIS course: the first raised in the room since it began (an earlier
                # episode's page belongs to an earlier course)
                later = [p for p in eng.pages.values() if p.get("room") == room and p["created_at"] >= s.t0 - 1]
                if later:
                    s.page = min(later, key=lambda p: p["created_at"])["id"]
            page = eng.pages.get(s.page) if s.page else None
            if page is not None:
                treated = (page["status"] == "accepted" and page.get("accepted_at") and now - page["accepted_at"] >= TREATED_S)
                if treated or page["status"] in ("resolved", "cancelled"):
                    ph.recover(now)
                    self._event(room, now, "course", "info",
                                f"Improving after the page was handled ({page['id']})", simulated=True)
                    if s.kind == "sepsis":
                        self._order(room, "piperacillin-tazobactam", now, "antibiotics broadened for sepsis")
            elif now - s.t0 >= SELF_LIMIT_S:
                ph.recover(now)
                self._event(room, now, "course", "info", "Settling without a page", simulated=True)
        # an allergy page the (simulated) pharmacist has resolved: the order is held, as they would
        for key, ep in self.episodes.items():
            room = key.split(":")[0]
            page = eng.pages.get(ep.get("page")) if ep.get("page") else None
            if not page or page.get("category") != "medication" or page["status"] != "resolved" or ep.get("held"):
                continue
            rec = self.records.get(room)
            if not rec or rec["kind"] != "synthetic":
                continue
            for st in list(self.findings.get(room, {}).values()):
                key = st["finding"]["key"]
                if key.startswith("med:allergy:"):
                    name = key.split(":", 2)[2]
                    for o in rec["orders"]:
                        if o["name"] == name and o["status"] == "active":
                            o["status"] = "held"
                            o["note"] = "held pending prescriber review (simulated pharmacist)"
                            self._event(room, now, "order", "info", f"{name} held by pharmacy review (simulated)",
                                        page=page["id"], simulated=True)
            ep["held"] = True
        if not eng.settings.get("demo_traffic") or now < self.next_scenario_at:
            return
        self.next_scenario_at = now + self.rng.uniform(*SCENARIO_GAP_S)
        running = sum(1 for ph in self.phys.values() if ph.scenario and ph.scenario.recover_t0 is None)
        if running >= MAX_SCENARIOS:
            return
        busy = {p.get("room") for p in eng.pages.values() if p["status"] in active}
        pool = [(room, fitting(ph.flags)) for room, ph in self.phys.items()
                if ph.scenario is None and now >= ph.cooldown_until and room not in busy]
        pool = [(r, k) for r, k in pool if k]
        if not pool:
            return
        pairs = [(r, k) for r, kinds in pool for k in kinds]
        if self.counters["scenarios"] == 0:
            # the first course of a session shows the whole loop: sepsis in a patient whose record
            # carries a penicillin allergy, so the broadened antibiotic meets the allergy check
            allergic = [(r, k) for r, k in pairs if k == "sepsis"
                        and any(a["substance"] == "penicillin" for a in self.records[r].get("allergies") or [])]
            pairs = allergic or pairs
        room, kind = self.rng.choices(pairs, weights=[WEIGHTS.get(k, 1.0) for _r, k in pairs])[0]
        s = self.phys[room].start(kind, now)
        self.counters["scenarios"] += 1
        self._event(room, now, "course", "info", f"Simulated course begins: {s.label.lower()}", simulated=True)

    def _order(self, room: str, name: str, now: float, why: str) -> None:
        """A simulated clinician's new order in a synthetic room. The detectors read it like any other."""
        from backend.aegis.rooms.formulary import FORMULARY
        rec = self.records.get(room)
        d = FORMULARY.get(name)
        if not rec or rec["kind"] != "synthetic" or d is None:
            return
        if any(o["name"] == name and o["status"] == "active" for o in rec["orders"]):
            return
        rec["orders"].append({"name": d.name, "generic": d.generic, "dose": d.dose, "route": d.route, "freq": d.freq,
                              "classes": sorted(d.classes), "prn": d.prn, "infusion": d.infusion, "renal": d.renal,
                              "qt": d.qt, "kind": "drug", "status": "active", "started_at": now, "last_given_at": None,
                              "note": f"{why} (simulated order)", "simulated": True})
        self._event(room, now, "order", "info", f"New order: {d.name} {d.dose} {d.route} {d.freq} ({why}, simulated)",
                    simulated=True)

    # ------------------------------------------------------------------ #
    # interface-fed rooms (called by the paging OS under its lock)
    # ------------------------------------------------------------------ #
    def admit_interface(self, room: str, patient: Dict[str, Any], now: float) -> None:
        rec = interface_record(room, patient, now)
        fresh = room not in self.interfaced
        self.records[room] = rec
        if fresh:
            self.interfaced[room] = {"labs": {}, "since": now, "withheld": deque(maxlen=40),
                                     "interface": patient.get("interface_name"), "last_withheld": {}}
            self.series[room] = deque(maxlen=SERIES_N)
            self.timeline[room] = deque(maxlen=TIMELINE_N)
            self.findings[room] = {}
            self.resolved[room] = deque(maxlen=30)
            self.news_hist[room] = deque(maxlen=120)
            for k in (room, f"{room}:med"):
                self.episodes.pop(k, None)
            self.scores.pop(room, None)
            self.body.pop(room, None)
            self.status.pop(room, None)
            self._evaluated.pop(room, None)
            self._event(room, now, "record", "info", f"Admitted by {patient.get('interface_name') or 'the ADT feed'} "
                                                      f"(HL7 ADT^A01): {rec['patient']['name']}, {rec['admission']['reason']}")
        else:
            self._event(room, now, "record", "info", "Record updated by the ADT feed")

    def discharge_interface(self, room: str, why: str, now: float) -> None:
        if room not in self.interfaced:
            return
        name = (self.records.get(room) or {}).get("patient", {}).get("name")
        self.interfaced.pop(room, None)
        for d in (self.records, self.series, self.findings, self.scores, self.body, self.news_hist, self.faults,
                  self._evaluated, self._last_pass):
            d.pop(room, None)
        for k in (room, f"{room}:med"):
            self.episodes.pop(k, None)
        self.status.pop(room, None)
        self.resolved.pop(room, None)
        self._event(room, now, "record", "info", f"{name or 'Patient'} {why} (HL7 ADT)")
        self.engine.emit("rooms", {room: {}})

    def observe_interface(self, room: str, t: float, vitals: Dict[str, float],
                          labs: Dict[str, List[Tuple[str, float]]], meta: Dict[str, Any]) -> None:
        st = self.interfaced.get(room)
        if st is None:
            return
        if vitals:
            s = self.series[room]
            if s and abs(s[-1]["t"] - t) < 0.5:
                s[-1].update(vitals)
            else:
                s.append({"t": t, **vitals})
        shown = []
        for key, items in labs.items():
            lst = st["labs"].setdefault(key, [])
            for oid, v in items:
                lst.append({"t": t, "v": v, "id": oid})
                shown.append(f"{_LAB_NAME.get(key, key.replace('_', ' ').capitalize())} {v:g}")
            lst.sort(key=lambda r: r["t"])
            del lst[:-24]
        if shown:      # one line per message, not one per value
            self._event(room, t, "lab", "info", f"{len(shown)} result{'s' if len(shown) != 1 else ''} from "
                                                f"{meta.get('name') or 'the interface'}: " + ", ".join(shown[:6])
                                                + (" …" if len(shown) > 6 else ""))

    def backfill_interface(self, room: str, key: str, rows: List[Tuple[str, float, float]], note: str) -> None:
        st = self.interfaced.get(room)
        if st is None or not rows:
            return
        lst = st["labs"].setdefault(key, [])
        have = {r.get("id") for r in lst}
        for oid, v, t in rows:
            if oid not in have:
                lst.append({"t": t, "v": v, "id": oid})
        lst.sort(key=lambda r: r["t"])
        del lst[:-24]
        self._event(room, self.engine.now(), "lab", "info", f"{len(rows)} earlier result(s) delivered: {note}", lab=key)

    def withheld_interface(self, room: str, t: float, items: List[Dict[str, Any]], meta: Dict[str, Any]) -> None:
        st = self.interfaced.get(room)
        if st is None:
            return
        now = self.engine.now()
        for it in items:
            st["withheld"].append({"t": t, **it, "interface": meta.get("name")})
            last = st["last_withheld"].get(it.get("concept"))
            if last is None or now - last > 120:          # one line per value kind every two minutes
                st["last_withheld"][it.get("concept")] = now
                self._event(room, t, "interface", "watch", f"{it.get('label')} withheld ({it.get('why')}): the value "
                                                          f"is not shown or used until the feed is reviewed")

    def retract_interface(self, room: str, ids: List[str], vitals_since: Dict[str, float]) -> None:
        st = self.interfaced.get(room)
        if st is None:
            return
        gone = set(ids)
        n = 0
        for key, lst in st["labs"].items():
            keep = [r for r in lst if r.get("id") not in gone]
            n += len(lst) - len(keep)
            st["labs"][key] = keep
        for ch, since in (vitals_since or {}).items():
            for r in self.series.get(room) or []:
                if r["t"] >= since and ch in r:
                    r.pop(ch, None)
                    n += 1
        self._evaluated.pop(room, None)            # re-read the room without them at the next pass
        if n:
            self._event(room, self.engine.now(), "interface", "watch", f"{n} value(s) withdrawn from this room's record")

    def event_interface(self, room: str, kind: str, severity: str, text: str) -> None:
        if room in self.timeline:
            self._event(room, self.engine.now(), kind, severity, text)

    def report(self, room: Optional[str], vitals: Dict[str, Any], text: str, page: Optional[str] = None) -> None:
        """A nurse reported values for a room: they are a measurement, and the room's record takes them.

        In a synthetic room the feed moves to the reported values; in a replayed ICU room the real data
        is never altered, and the report is only written to the timeline."""
        if not room:
            return
        now = self.engine.now()
        ph = self.phys.get(room)
        v = dict(vitals or {})
        if "confus" in (text or "").lower():
            v["confused"] = True
        shown = ", ".join(f"{k} {x:g}" for k, x in vitals.items() if isinstance(x, (int, float)))
        if ph is not None and any(k in v for k in ("hr", "rr", "spo2", "temp_c", "sbp", "glucose", "lactate", "confused")):
            ph.observe(v, now).page = page
            self._event(room, now, "obs", "info", f"Nurse report: {shown or text[:80]}")
        elif room in self.timeline:
            self._event(room, now, "obs", "info", f"Nurse report: {shown or text[:80]}")

    # ------------------------------------------------------------------ #
    # timeline
    # ------------------------------------------------------------------ #
    def _event(self, room: str, t: float, kind: str, severity: str, text: str, **ref: Any) -> None:
        tl = self.timeline.setdefault(room, deque(maxlen=TIMELINE_N))
        tl.append({"t": t, "kind": kind, "severity": severity, "text": text, **ref})

    # ------------------------------------------------------------------ #
    # views
    # ------------------------------------------------------------------ #
    def board(self) -> Dict[str, Any]:
        """Every room's unit at a glance (the watch board and the floor plan)."""
        out = []
        occupied = set(self._occupied())
        for room in sorted(ROOMS):
            z = ZONES[ROOMS[room].zone]
            rec = self.records.get(room) if room in occupied else None
            row: Dict[str, Any] = {"room": room, "zone": z.id, "zone_name": z.name, "floor": z.floor, "kind": None}
            if rec is None:
                out.append(row)
                continue
            series = list(self.series.get(room) or [])
            last = series[-1] if series else {}
            cur = self.findings.get(room) or {}
            act = [x["finding"] for x in cur.values() if x["finding"]["severity"] != "info" and x["last_seen"] == self.last_eval]
            act.sort(key=lambda f: -D.SEVERITY_RANK[f["severity"]])
            counts = {s: sum(1 for f in act if f["severity"] == s) for s in ("critical", "urgent", "watch")}
            ep = self.episodes.get(room) or {}
            ph = self.phys.get(room)
            pt = rec["patient"]
            row.update({
                "kind": rec["kind"], "name": pt.get("name"), "age": pt.get("age"), "sex": pt.get("sex"),
                "mrn": pt.get("mrn"), "bed": pt.get("bed"), "diagnosis": (rec.get("admission") or {}).get("reason"),
                "code_status": rec.get("code_status"), "allergies": [a["substance"] for a in rec.get("allergies") or []],
                "synthetic": rec["provenance"]["synthetic"], "scores": self.scores.get(room) or {},
                "vitals": {k: last.get(k) for k in ("hr", "sbp", "map", "rr", "spo2", "temp_c", "o2_lpm", "avpu") if k in last},
                "vitals_at": last.get("t"), "clock": "icu_hours" if rec["kind"] == "replay" else "wall",
                "findings": [{"key": f["key"], "detector": f["detector"], "title": f["title"], "severity": f["severity"],
                              "category": f["category"]} for f in act[:4]],
                "counts": counts, "worst": act[0]["severity"] if act else None,
                "trend": [s for _t, s in list(self.news_hist.get(room) or [])[-40:]],
                "episode": {"open": bool(ep.get("open")), "page": ep.get("page")} if ep else None,
                "simulation": ph.scenario.to_dict(self.engine.now()) if ph and ph.scenario else None,
                "risk": (self.engine.monitored.get(room) or {}).get("risk"),
                "probability": (self.engine.monitored.get(room) or {}).get("probability"),
            })
            out.append(row)
        ms = sorted(self.eval_ms)
        return {"rooms": out, "evaluated_at": self.last_eval, "eval_period_s": EVAL_S,
                "eval_ms_p50": round(ms[len(ms) // 2], 2) if ms else None, "counters": dict(self.counters),
                "detectors": [n for n, _ in D.DETECTORS]}

    def unit(self, room: str) -> Optional[Dict[str, Any]]:
        if room not in ROOMS:
            return None
        eng = self.engine
        z = ZONES[ROOMS[room].zone]
        base = {"room": room, "zone": z.id, "zone_name": z.name, "floor": z.floor, "critical_care": z.kind == "icu"}
        rec = self.records.get(room) if room in self._occupied() else None
        if rec is None:
            return {**base, "occupied": False}
        now = eng.now()
        cur = self.findings.get(room) or {}
        findings = []
        for st in sorted(cur.values(), key=lambda s: (-D.SEVERITY_RANK[s["finding"]["severity"]], s["first_seen"])):
            if st["last_seen"] != self.last_eval:
                continue                  # clearing: shown in the timeline, not as active
            findings.append({**st["finding"], "first_seen": st["opened_at"] or st["first_seen"], "last_seen": st["last_seen"],
                             "peak": st["peak"], "history": st["history"], "open": st["open"]})
        resolved = [{**r["finding"], "first_seen": r["first_seen"], "resolved_at": r["resolved_at"], "peak": r["peak"]}
                    for r in list(self.resolved.get(room) or [])[:12]]
        active = _active()
        pages = sorted([p for p in eng.pages.values() if p.get("room") == room], key=lambda p: -p["created_at"])[:12]
        pages_out = [{"id": p["id"], "priority": p["priority"], "status": p["status"], "text": p["text"],
                      "source": p["source"], "created_at": p["created_at"], "assignee": p.get("assignee"),
                      "category": p.get("category"), "active": p["status"] in active} for p in pages]
        series = list(self.series.get(room) or [])
        ph = self.phys.get(room)
        labs = _merge_labs(rec.get("labs") or {}, ph.labs if ph else {})
        ist = self.interfaced.get(room)
        if ist is not None:
            labs = _merge_labs(labs, {k: [{"t": r["t"], "v": r["v"]} for r in v] for k, v in ist["labs"].items()})
        timeline = list(self.timeline.get(room) or [])
        for a in eng.alerts:
            if a.get("room") == room and a.get("source") != "monitor":
                timeline.append({"t": a["created_at"], "kind": "alert", "severity": "info",
                                 "text": f"{(a.get('source') or 'nurse').capitalize()} alert: {a['text']}",
                                 "page": a.get("page_id"), "priority": a.get("priority")})
        if rec["kind"] == "synthetic":
            adm = rec["admission"]["admitted_at"]
            timeline.append({"t": adm, "kind": "record", "severity": "info", "text": f"Admitted: {rec['admission']['reason']}"})
            for o in rec.get("orders") or []:
                if o.get("started_at"):
                    timeline.append({"t": o["started_at"], "kind": "order", "severity": "info",
                                     "text": f"Order started: {o['name']} {o.get('dose', '')} {o.get('route', '')} {o.get('freq', '')}".strip()})
        timeline.sort(key=lambda e: -e["t"])
        ms_hist = [{"t": t, "score": s} for t, s in self.news_hist.get(room) or []]
        return {
            **base, "occupied": True, "kind": rec["kind"], "now": now, "clock": "icu_hours" if rec["kind"] == "replay" else "wall",
            "record": rec, "vitals": series[-1] if series else None, "series": series[-SERIES_N:],
            "labs": labs, "scores": self.scores.get(room) or {}, "news2_history": ms_hist,
            "findings": findings, "resolved": resolved, "faults": self.faults.get(room) or [],
            "episode": self.episodes.get(room), "med_episode": self.episodes.get(f"{room}:med"),
            "pages": pages_out, "timeline": timeline[:120],
            "care_team": self._care_team(room, pages), "devices": self._devices(room, rec, series, now),
            "simulation": ph.scenario.to_dict(now) if ph and ph.scenario else None,
            "monitoring": {"kind": rec["kind"], "sample_s": SAMPLE_S if rec["kind"] == "synthetic" else None,
                           "eval_period_s": EVAL_S,
                           "pages_from": "room monitor" if rec["kind"] in ("synthetic", "interface") else "sepsis detector",
                           "detectors": [n for n, _ in D.DETECTORS]},
            "feed": self._feed(room) if ist is not None else None,
            "withheld": list(ist["withheld"])[-12:] if ist is not None else [],
            "risk": (eng.monitored.get(room) or {}).get("risk"),
            "probability": (eng.monitored.get(room) or {}).get("probability"),
            "body": self.body.get(room),
        }

    def _care_team(self, room: str, pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        from backend.aegis.paging.router import zone_physician
        eng = self.engine
        zone = ROOMS[room].zone
        spec = zone_physician(zone)
        team: List[Dict[str, Any]] = []
        docs = [c for c in eng.roster.values() if c.specialty == spec]
        doc = next((c for c in docs if c.home_zone == zone), None) or next(iter(docs), None)
        if doc:
            team.append({"id": doc.id, "name": doc.name, "role": "Attending of record", "status": doc.status,
                         "specialty": doc.specialty})
        charge = eng.roster.get("rn.patel")
        if charge:
            team.append({"id": charge.id, "name": charge.name, "role": "Charge nurse", "status": charge.status,
                         "specialty": charge.specialty})
        seen = {t["id"] for t in team}
        for p in pages:
            cid = p.get("assignee")
            if cid and cid in eng.roster and cid not in seen and p["status"] in _active():
                c = eng.roster[cid]
                seen.add(cid)
                team.append({"id": c.id, "name": c.name, "role": f"On {p['id']}", "status": c.status,
                             "specialty": c.specialty})
        return team

    def _feed(self, room: str) -> Optional[Dict[str, Any]]:
        try:
            return self.feed_status(room) if self.feed_status else None
        except Exception:
            return None

    def _devices(self, room: str, rec: Dict[str, Any], series: List[Dict[str, Any]], now: float) -> List[Dict[str, Any]]:
        kind = rec["kind"]
        if kind == "interface":
            feed = self._feed(room) or {}
            by_iface: Dict[str, List[Dict[str, Any]]] = {}
            for ch in feed.get("channels") or []:
                by_iface.setdefault(ch["interface"], []).append(ch)
            out = []
            for iface, chans in by_iface.items():
                q = [c["label"] for c in chans if c["status"] == "quarantined"]
                last = max((c["last_t"] or 0) for c in chans)
                age = now - last
                out.append({"name": chans[0].get("interface_name") or iface,
                            "detail": f"HL7 · {sum(1 for c in chans if c['status'] == 'adopted')} of {len(chans)} codes "
                                      f"mapped" + (f" · quarantined: {', '.join(q)}" if q else ""),
                            "status": "quarantined" if q else ("streaming" if age < 120 else
                                                               "connected" if age < 1800 else "no signal")})
            return out or [{"name": "Interface feed", "detail": "waiting for the first results", "status": "no signal"}]
        if kind == "replay":
            return [{"name": "Bedside monitor", "detail": "PhysioNet/CinC 2019 replay feed (HR, SpO₂, BP, RR, temperature)",
                     "status": "streaming"},
                    {"name": "Laboratory interface", "detail": "Results as the stay recorded them", "status": "streaming"}]
        if kind == "live":
            m = self.engine.monitored.get(room) or {}
            return [{"name": "Live session", "detail": m.get("source") or "camera, wearable or simulator", "status": "streaming"}]
        z = ZONES[ROOMS[room].zone].kind
        last = series[-1]["t"] if series else None
        ok = last is not None and now - last < 90
        devs = [{"name": "Telemetry" if z == "stepdown" else "Bedside monitor" if z in ("ed", "pacu") else "Wearable vitals patch",
                 "detail": "Simulated feed · a set every 10 s", "status": "streaming" if ok else "no signal"}]
        for o in rec.get("orders") or []:
            if o.get("infusion") and o.get("status") == "active":
                devs.append({"name": "PCA pump" if o.get("freq") == "PCA" else "Infusion pump",
                             "detail": f"{o['name']} · {o['dose']}", "status": "running"})
        if (series[-1].get("o2_lpm") if series else 0):
            devs.append({"name": "Oxygen", "detail": f"Nasal cannula {series[-1]['o2_lpm']:g} L/min", "status": "running"})
        return devs


# --------------------------------------------------------------------------- #
# The body: every organ system scored from what this room measures and finds
# --------------------------------------------------------------------------- #
SYSTEMS = {"heart": "Cardiovascular", "lungs": "Respiratory", "brain": "Neurological", "kidneys": "Renal",
           "liver": "Hepatic", "metabolic": "Metabolic", "immune": "Inflammatory", "blood": "Haematological"}
_FINDING_REGIONS = {
    "shock_index": ("heart",), "map_floor": ("heart",), "af_rvr": ("heart",), "baseline:sbp": ("heart",),
    "baseline:hr": ("heart",), "bleeding": ("blood", "heart"), "opioid_rd": ("lungs", "brain"),
    "baseline:rr": ("lungs",), "baseline:spo2": ("lungs",), "aki": ("kidneys",), "glucose": ("metabolic",),
    "lactate": ("metabolic",), "potassium": ("metabolic",), "neutropenic_fever": ("immune",), "sirs": ("immune",),
    "qsofa": ("immune",), "sepsis_model": ("immune",),
}
_SEV_VALUE = {"critical": 0.9, "urgent": 0.62, "watch": 0.35, "info": 0.0}
_POINT_VALUE = {0: 0.0, 1: 0.12, 2: 0.35, 3: 0.62}
_NEWS_PART_REGION = {"Respiration rate": "lungs", "SpO₂": "lungs", "Air or oxygen": "lungs", "Systolic BP": "heart",
                     "Pulse": "heart", "Consciousness": "brain", "Temperature": "immune"}
_TREND_REGION = {"Pulse": "heart", "Systolic": "heart", "Respiratory rate": "lungs", "SpO₂": "lungs",
                 "Temperature": "immune"}
# where a documented infection sits on the body (the twin's suspected-focus marker)
_INFECTION_SITE = (("pneumonia", "Pneumonia", "lungs"), ("pyelonephritis", "Pyelonephritis", "kidneys"),
                   ("urinary tract", "Urinary tract", "kidneys"), ("colectomy", "Abdominal (post-operative)", "abdomen"),
                   ("cholecystectomy", "Abdominal (post-operative)", "abdomen"), ("cellulitis", "Cellulitis, left leg", None),
                   ("febrile neutropenia", "Source unknown (neutropenic)", "systemic"))


def _status(sev: float) -> str:
    return "CRITICAL" if sev >= 0.75 else "DYSFUNCTION" if sev >= 0.45 else "STRESS" if sev >= 0.15 else "NORMAL"


def _num(v: Any) -> Optional[float]:
    return float(v) if isinstance(v, (int, float)) else None


def body_state(rec: Dict[str, Any], c: D.Ctx, found: List[Dict[str, Any]], scores: Dict[str, Any]) -> Dict[str, Any]:
    """Organ systems for the anatomical twin, from this room's own detections and measurements.

    A system is MONITORED only if something in the room measures it (a vital, a lab, a
    consciousness assessment); otherwise the twin shows it as no data, never as healthy.
    Its severity is the worst of the findings that implicate it, the NEWS2 points of its
    parameters, the SOFA points of its labs, and plain lab thresholds."""
    v, labs = c.v, c.labs
    last = {k: (xs[-1]["v"] if xs else None) for k, xs in labs.items()}
    sev = {r: 0.0 for r in SYSTEMS}
    why: Dict[str, List[Dict[str, Any]]] = {r: [] for r in SYSTEMS}

    def bump(region: str, value: float, key: Optional[str] = None, title: Optional[str] = None,
             severity: Optional[str] = None) -> None:
        if value > sev[region]:
            sev[region] = value
        if key and not any(w["key"] == key for w in why[region]):
            why[region].append({"key": key, "title": title, "severity": severity})

    for f in found:
        if f["severity"] == "info" and f["key"] not in ("news2", "sofa"):
            continue
        for region in _FINDING_REGIONS.get(f["key"], ()):
            bump(region, _SEV_VALUE[f["severity"]], f["key"], f["title"], f["severity"])
        if f["key"] == "news2":
            for e in f["evidence"]:
                region = next((r for lbl, r in _NEWS_PART_REGION.items() if str(e["label"]).startswith(lbl)), None)
                if region and e.get("points"):
                    bump(region, _POINT_VALUE.get(int(e["points"]), 0.62),
                         "news2" if e["points"] >= 2 else None, f["title"], f["severity"] if e["points"] >= 2 else None)
        if f["key"] == "sofa":
            for e in f["evidence"]:
                lbl, pts = str(e["label"]), int(e.get("points") or 0)
                region = "blood" if lbl.startswith("Coagulation") else "liver" if lbl.startswith("Liver") else \
                    "kidneys" if lbl.startswith("Renal") else "heart" if lbl.startswith("Cardio") else None
                if region and pts:
                    bump(region, {1: 0.2, 2: 0.45, 3: 0.7, 4: 0.9}[min(4, pts)], "sofa", f["title"], f["severity"])
        if f["key"] == "trend":
            # a worsening slope is a reason to look, not a dysfunction: it can lift a system
            # to stress, never past it, while the values themselves are still in range
            for e in f["evidence"]:
                region = _TREND_REGION.get(str(e["label"]))
                if region:
                    bump(region, min(0.3, _SEV_VALUE[f["severity"]]), "trend", f["title"], f["severity"])
    plt, hb, bili = _num(last.get("platelets")), _num(last.get("hemoglobin")), _num(last.get("bilirubin"))
    if plt is not None:
        bump("blood", 0.8 if plt < 50 else 0.45 if plt < 100 else 0.22 if plt < 150 else 0.0)
    if hb is not None and hb < 8:
        bump("blood", 0.45)
    if bili is not None:
        bump("liver", 0.75 if bili >= 6 else 0.45 if bili >= 2 else 0.22 if bili >= 1.2 else 0.0)

    hr, sbp, dbp, mp = _num(v.get("hr")), _num(v.get("sbp")), _num(v.get("dbp")), _num(v.get("map"))
    rr, spo2, temp, o2 = _num(v.get("rr")), _num(v.get("spo2")), _num(v.get("temp_c")), v.get("o2_lpm")
    # the inflammatory response on its own numbers, so a WBC of 20 never reads "normal" just
    # because no sepsis-context finding happens to be open (SIRS bands, as for the labs above)
    wbc = _num(last.get("wbc"))
    if wbc is not None:
        bump("immune", 0.45 if (wbc >= 20 or wbc < 2) else 0.25 if (wbc > 12 or wbc < 4) else 0.0)
    if temp is not None:
        bump("immune", 0.45 if (temp >= 39.5 or temp < 35) else 0.25 if (temp >= 38.3 or temp < 36) else 0.0)
    avpu = v.get("avpu")

    def fmt(x: Optional[float], nd: int = 0) -> str:
        return "—" if x is None else (f"{x:.{nd}f}" if nd else f"{round(x)}")

    read: Dict[str, Optional[str]] = {r: None for r in SYSTEMS}
    if hr is not None or sbp is not None or mp is not None:
        bp = f"BP {fmt(sbp)}/{fmt(dbp)}" if sbp is not None and dbp is not None else f"MAP {fmt(mp)}" if mp is not None else None
        si = scores.get("shock_index")
        read["heart"] = " · ".join(x for x in (f"HR {fmt(hr)}" if hr is not None else None, bp,
                                              f"SI {si:.2f}" if si and si >= 0.9 else None) if x)
    if rr is not None or spo2 is not None:
        on = f" on {o2:g} L" if o2 else " on air" if o2 == 0 else ""
        read["lungs"] = " · ".join(x for x in (f"RR {fmt(rr)}" if rr is not None else None,
                                              f"SpO₂ {fmt(spo2)}%{on}" if spo2 is not None else None) if x)
    if avpu is not None:
        read["brain"] = {"A": "Alert", "C": "Confused at baseline" if v.get("chronic_confusion") else "New confusion",
                         "V": "Responds to voice only", "P": "Responds to pain only", "U": "Unresponsive"}[avpu]
    if last.get("creatinine") is not None:
        base = next((x["v"] for x in labs["creatinine"] if x.get("baseline")), None)
        read["kidneys"] = f"Creatinine {last['creatinine']:g}" + (f" (baseline {base:g})" if base else "")
    if bili is not None:
        read["liver"] = f"Bilirubin {bili:g}"
    met = [f"{n} {last[k]:g}" for k, n in (("lactate", "Lactate"), ("glucose", "Glucose"), ("potassium", "K"))
           if last.get(k) is not None]
    if met:
        read["metabolic"] = " · ".join(met)
    if temp is not None or last.get("wbc") is not None:
        read["immune"] = " · ".join(x for x in (f"{temp:.1f} °C" if temp is not None else None,
                                               f"WBC {last['wbc']:g}" if last.get("wbc") is not None else None) if x)
    if plt is not None or hb is not None:
        read["blood"] = " · ".join(x for x in (f"Platelets {plt:g}" if plt is not None else None,
                                              f"Hb {hb:g}" if hb is not None else None) if x)

    regions = {}
    for r, label in SYSTEMS.items():
        mon = read[r] is not None
        sv = round(sev[r], 2) if mon else 0.0
        regions[r] = {"label": label, "monitored": mon, "severity": sv, "status": _status(sv) if mon else "NO_DATA",
                      "reading": read[r], "findings": why[r] if mon else []}
    infection = None
    if any(f["category"] == "sepsis" and f["severity"] in ("urgent", "critical") for f in found) or \
            any(f["key"] == "sirs" and f["severity"] == "watch" for f in found):
        dx = ((rec.get("admission") or {}).get("reason") or "").lower()
        site = next(((label, region) for kw, label, region in _INFECTION_SITE if kw in dx), None)
        if site:
            infection = {"site": site[0], "region": site[1], "source": "the admitting diagnosis"}
    return {"regions": regions, "infection": infection,
            "channels": [{"key": "heart_rate", "value": hr}, {"key": "respiratory_rate", "value": rr},
                         {"key": "map", "value": mp}, {"key": "spo2", "value": spo2}],
            "vitals": {"heart_rate": hr, "respiratory_rate": rr, "map": mp, "spo2": spo2, "temp_c": temp,
                       "lactate": last.get("lactate"), "wbc": last.get("wbc")}}


def _merge_labs(a: Dict[str, List[Dict[str, float]]], b: Dict[str, List[Dict[str, float]]]) -> Dict[str, List[Dict[str, float]]]:
    out = {k: list(v) for k, v in a.items()}
    for k, rows in b.items():
        out.setdefault(k, []).extend(rows)
        out[k].sort(key=lambda r: r["t"])
    return out
