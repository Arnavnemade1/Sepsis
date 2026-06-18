"""HDC associative bio-memory — hyperdimensional case-based reasoning.

This promotes Hyperdimensional Computing from a single benchmarked classifier to
a first-class reasoning substrate for the live system. Every ICU patient-hour is
ENCODED into one 10,000-D bipolar hypervector (the VSA record encoding from
`backend.core.hdc`: bind channel-id ⊗ value-level, then bundle). A labelled bank
of these vectors is the system's *associative memory of real precedent*.

At inference we encode the patient's current bio-state and retrieve, by cosine
similarity, the most similar HISTORICAL states — and report what happened to
them (did they become septic, and how many hours later). This is exactly what
HDC is good at (holographic distributed memory + similarity), and it adds
something the tree/logistic detectors cannot: an *interpretable, case-based*
second opinion grounded in real patients — "this state resembles N prior states,
M of which were septic within the next few hours."

HONESTY: this is a complementary evidence channel, not a claimed accuracy win.
Its standalone discriminative power is measured patient-level, leakage-free, and
reported as-is (see verify_hdc_memory.py). Retrieval at serve time excludes the
querying patient's own rows so it can never trivially match itself.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.core.hdc import HDCEngine

# cohort channels the cohort actually scores on, with encoding ranges
CHANNELS = ["hr", "temp_c", "resp_rate", "spo2", "map", "wbc", "lactate"]
RANGES: Dict[str, Tuple[float, float]] = {
    "hr": (40.0, 200.0), "temp_c": (34.0, 42.0), "resp_rate": (8.0, 40.0),
    "spo2": (70.0, 100.0), "map": (45.0, 130.0), "wbc": (1.0, 40.0), "lactate": (0.0, 12.0),
}
# deterioration TREND per channel (Δ over a short window) — sepsis is a trajectory,
# not just a level — and interpretable clinical composites.
TREND_RANGES: Dict[str, Tuple[float, float]] = {
    "hr": (-25, 25), "temp_c": (-2, 2), "resp_rate": (-10, 10),
    "spo2": (-8, 8), "map": (-25, 25), "wbc": (-8, 8), "lactate": (-4, 4),
}
COMPOSITE_RANGES: Dict[str, Tuple[float, float]] = {
    "shock_index": (0.4, 2.5), "sirs": (0.0, 4.0), "qsofa": (0.0, 3.0),
}
ALL_RANGES = {**RANGES, **{ch + "__tr": r for ch, r in TREND_RANGES.items()}, **COMPOSITE_RANGES}
TREND_WIN = 4
# a "septic-soon" precedent = septic-labelled OR within this many hours of onset
SOON_H = 6


def enrich_series(states: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Per hour, add channel trends (Δ vs a short trailing window) + clinical
    composites (shock index, SIRS, qSOFA) so the hypervector carries the
    deterioration pattern, not just the instantaneous level."""
    out = []
    for hr, s in enumerate(states):
        e = dict(s)
        j = max(0, hr - TREND_WIN)
        for ch in CHANNELS:
            past = [states[t][ch] for t in range(j, hr) if states[t].get(ch) is not None]
            if past and s.get(ch) is not None:
                e[ch + "__tr"] = float(s[ch]) - float(np.mean(past))
        hrv, mapv = s.get("hr"), s.get("map")
        if hrv and mapv:
            e["shock_index"] = float(hrv) / max(float(mapv), 1.0)
        e["sirs"] = float(sum([
            (s.get("hr", 0) or 0) > 90,
            (s.get("temp_c", 37) or 37) > 38 or (s.get("temp_c", 37) or 37) < 36,
            (s.get("resp_rate", 0) or 0) > 20,
            (s.get("wbc", 8) or 8) > 12 or (s.get("wbc", 8) or 8) < 4,
        ]))
        e["qsofa"] = float(sum([
            (s.get("resp_rate", 0) or 0) >= 22,
            (s.get("map", 100) or 100) <= 65,
        ]))
        out.append(e)
    return out


class HDCBioMemory:
    def __init__(self, dim: int = 10000, seed: int = 42) -> None:
        self.eng = HDCEngine(dimensions=dim)
        self.eng.ranges.update(ALL_RANGES)              # let encode_value handle every feature
        rng = np.random.default_rng(seed + 7)
        self.feats = list(ALL_RANGES.keys())
        self.ids = {f: rng.choice([-1, 1], size=dim).astype(np.int8) for f in self.feats}
        self.mem: Optional[np.ndarray] = None           # (N, D) int8
        self.meta: List[Dict[str, Any]] = []            # per-row {label, tto, pid, hour}

    # ---- encoding ---- #
    def encode(self, state: Dict[str, float]) -> np.ndarray:
        binds = []
        for f in self.feats:
            v = state.get(f)
            if v is None or (isinstance(v, float) and v != v):
                continue
            lvl = self.eng.encode_value(f, float(v))
            binds.append(self.eng.bind(self.ids[f], lvl))
        return self.eng.bundle(binds) if binds else self.eng.random_vector()

    # ---- memory bank ---- #
    def build(self, rows: List[Tuple[Dict[str, float], int, Optional[int], str]]) -> "HDCBioMemory":
        """rows: (state, hour_label, time_to_onset|None, patient_id)."""
        vecs, meta = [], []
        for state, label, tto, pid in rows:
            vecs.append(self.encode(state))
            meta.append({"label": int(label), "tto": tto, "pid": pid})
        self.mem = np.stack(vecs).astype(np.int8) if vecs else None
        self._M = self.mem.astype(np.float32) if self.mem is not None else None   # fast matmul
        self.meta = meta
        return self

    @staticmethod
    def rows_from_patient(p) -> List[Tuple[Dict[str, float], int, Optional[int], str]]:
        """Extract (state, label, tto, pid) for every hour of a cohort RealPatient."""
        labels = p.record.labels
        onset = p.onset_hour
        raw = [{ch: float(p._ff[hr, k]) for k, ch in enumerate(CHANNELS)} for hr in range(p.los_total)]
        enriched = enrich_series(raw)
        out = []
        for hr in range(p.los_total):
            lab = int(labels[hr]) if hr < len(labels) else 0
            tto = (onset - hr) if (onset is not None) else None
            out.append((enriched[hr], lab, tto, p.bed_id))
        return out

    # ---- retrieval ---- #
    def query(self, state: Dict[str, float], k: int = 16, exclude_pid: Optional[str] = None) -> Dict[str, Any]:
        """Return the k most similar historical states + a similarity-weighted
        case-based resemblance score (soft k-NN P(septic-soon))."""
        if self.mem is None or not len(self.meta):
            return {"resemblance": 0.0, "k": 0, "n_septic": 0, "lead_h": None, "top": []}
        q = np.ascontiguousarray(self.encode(state), dtype=np.float32)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = (self._M @ q) / float(self.eng.D)
        order = np.argsort(-sims)
        nbrs: List[Tuple[float, Dict[str, Any]]] = []
        for i in order:
            m = self.meta[int(i)]
            if exclude_pid is not None and m["pid"] == exclude_pid:
                continue
            soon = int(m["label"] == 1 or (m["tto"] is not None and 0 <= m["tto"] <= SOON_H))
            nbrs.append((float(sims[i]), {**m, "soon": soon}))
            if len(nbrs) >= k:
                break
        if not nbrs:
            return {"resemblance": 0.0, "k": 0, "n_septic": 0, "lead_h": None, "top": []}
        # similarity-weighted vote (shifted so only positive correlation contributes)
        w = np.array([max(s, 0.0) + 1e-6 for s, _ in nbrs])
        soon = np.array([m["soon"] for _, m in nbrs])
        resemblance = float((w * soon).sum() / w.sum())
        n_septic = int(soon.sum())
        leads = [m["tto"] for _, m in nbrs if m["soon"] and m["tto"] is not None and m["tto"] >= 0]
        lead_h = float(np.median(leads)) if leads else None
        return {
            "resemblance": round(resemblance, 3),
            "k": len(nbrs), "n_septic": n_septic,
            "lead_h": lead_h, "top_sim": round(nbrs[0][0], 3),
        }
