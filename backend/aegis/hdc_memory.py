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


def _enrich_at(states: List[Dict[str, float]], hr: int) -> Dict[str, float]:
    """Hour ``hr`` of :func:`enrich_series`: its levels plus trends against the trailing
    window and the clinical composites."""
    s = states[hr]
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
    return e


def enrich_series(states: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Per hour, add channel trends (Δ vs a short trailing window) + clinical
    composites (shock index, SIRS, qSOFA) so the hypervector carries the
    deterioration pattern, not just the instantaneous level."""
    return [_enrich_at(states, hr) for hr in range(len(states))]


def enrich_last(states: List[Dict[str, float]]) -> Dict[str, float]:
    """``enrich_series(states)[-1]`` without enriching the hours before it (the live
    frame needs only the current hour; the series cost 5x that on every recompute)."""
    return _enrich_at(states, len(states) - 1) if states else {}


#: states encoded per block by :meth:`HDCBioMemory.encode_many` — bounds the int16
#: accumulator at CHUNK x D x 2 bytes (~10 MB) however large the batch
CHUNK = 512


class HDCBioMemory:
    def __init__(self, dim: int = 10000, seed: int = 42) -> None:
        self.eng = HDCEngine(dimensions=dim)
        self.eng.ranges.update(ALL_RANGES)              # let encode_value handle every feature
        rng = np.random.default_rng(seed + 7)
        self.feats = list(ALL_RANGES.keys())
        self.ids = {f: rng.choice([-1, 1], size=dim).astype(np.int8) for f in self.feats}
        self.mem: Optional[np.ndarray] = None           # (N, D) int8
        self.meta: List[Dict[str, Any]] = []            # per-row {label, tto, pid, hour}
        self._T: Optional[np.ndarray] = None            # (F, L, D) int8 bind tables, lazy

    # ---- encoding ---- #
    def tables(self) -> np.ndarray:
        """``T[f, l] = bind(id_f, level_l)`` for every feature and level, built once.

        Encoding a state used to re-derive each bind per call (a 10,000-D multiply per
        feature per state). There are only F x L distinct binds (17 x 100), so they are
        a 17 MB lookup table and an encode is a gather, a sum and a sign."""
        if self._T is None:
            lv = self.eng.level_vectors
            self._T = np.stack([(self.ids[f][None, :] * lv).astype(np.int8) for f in self.feats])
        return self._T

    def levels(self, states: List[Dict[str, float]],
               feats: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray]:
        """``(level index, present)`` per state and feature, exactly as
        ``HDCEngine.encode_value`` computes it (clip, scale, truncate). Missing and NaN
        values are absent, as :meth:`encode` skips them."""
        feats = self.feats if feats is None else feats
        B, F = len(states), len(feats)
        vals = np.zeros((B, F))
        present = np.zeros((B, F), dtype=bool)
        odd = []                                         # present but non-Python-float NaN
        for i, st in enumerate(states):
            for j, f in enumerate(feats):
                v = st.get(f)
                if v is None or (isinstance(v, float) and v != v):
                    continue
                x = float(v)
                present[i, j] = True
                vals[i, j] = x
                if x != x:
                    odd.append((i, j, f, x))
        lo = np.array([self.eng.ranges[f][0] for f in feats], dtype=np.float64)
        hi = np.array([self.eng.ranges[f][1] for f in feats], dtype=np.float64)
        frac = (np.maximum(lo, np.minimum(hi, vals)) - lo) / (hi - lo)
        idx = np.zeros((B, F), dtype=np.int64)
        ok = present & np.isfinite(frac)
        idx[ok] = (frac[ok] * (self.eng.num_levels - 1)).astype(np.int64)
        for i, j, f, x in odd:                           # the scalar path's own answer
            idx[i, j] = self._level_index(f, x)
        return idx, present

    def _level_index(self, f: str, value: float) -> int:
        """``HDCEngine.encode_value``'s index arithmetic, verbatim (Python ``min``/``max``
        send a NaN to the top level, which the vectorised clip would not)."""
        val_min, val_max = self.eng.ranges[f]
        clipped_val = max(val_min, min(val_max, value))
        fraction = (clipped_val - val_min) / (val_max - val_min)
        return int(fraction * (self.eng.num_levels - 1))

    def encode_many(self, states: List[Dict[str, float]],
                    feats: Optional[List[str]] = None) -> np.ndarray:
        """``(B, D)`` int8 — row ``i`` bit-identical to ``encode(states[i])`` (or, with
        ``feats``, to bundling only those features). Ties break on the engine's fixed
        pattern and a state with no usable feature draws ``random_vector()`` in order,
        both exactly as the scalar path does."""
        feats = self.feats if feats is None else feats
        idx, present = self.levels(states, feats)
        out = self.bundle_levels(idx, present, feats)
        for i in np.flatnonzero(~present.any(axis=1)):   # same RNG draws, same order
            out[i] = self.eng.random_vector()
        return out

    def bundle_levels(self, idx: np.ndarray, present: np.ndarray,
                      feats: Optional[List[str]] = None) -> np.ndarray:
        """``sign(sum of the present binds)`` per row, ties on the engine's fixed
        pattern. A row with nothing present comes back as the tie pattern; callers
        decide what an empty state means (``encode`` draws a random vector, a
        resonance modality is absent)."""
        feats = self.feats if feats is None else feats
        T = self.tables()
        fidx = [self.feats.index(f) for f in feats]
        B, D = idx.shape[0], self.eng.D
        out = np.empty((B, D), dtype=np.int8)
        tie = self.eng._tie_break
        for a in range(0, B, CHUNK):
            b = min(a + CHUNK, B)
            acc = np.zeros((b - a, D), dtype=np.int16)
            for j, fj in enumerate(fidx):
                pres = present[a:b, j]
                cnt = int(pres.sum())
                if cnt == b - a:
                    acc += T[fj, idx[a:b, j]]
                elif 2 * cnt > b - a:                # mostly present: masked add, no scatter
                    acc += T[fj, idx[a:b, j]] * pres[:, None]
                elif cnt:
                    rows = np.flatnonzero(pres)
                    acc[rows] += T[fj, idx[a + rows, j]]
            acc = np.where(acc == 0, tie[None, :], acc)
            out[a:b] = np.sign(acc)
        return out

    def encode(self, state: Dict[str, float]) -> np.ndarray:
        return self.encode_many([state])[0]

    def encode_reference(self, state: Dict[str, float]) -> np.ndarray:
        """The original per-feature encoder: the specification :meth:`encode_many` is
        verified bit-for-bit against."""
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
        meta = [{"label": int(label), "tto": tto, "pid": pid} for _, label, tto, pid in rows]
        self.mem = self.encode_many([r[0] for r in rows]) if rows else None
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
    def _rank(self, sims: np.ndarray, k: int, exclude_pid: Optional[str]):
        """Indices of the best candidates, most-similar first, without a full sort."""
        n = sims.shape[0]
        window = min(n, max(4 * k, k + 64))
        if window >= n:
            return np.argsort(-sims)
        idx = np.argpartition(-sims, window - 1)[:window]
        idx = idx[np.argsort(-sims[idx])]
        if exclude_pid is None:
            return idx
        survivors = sum(1 for i in idx if self.meta[int(i)]["pid"] != exclude_pid)
        if survivors >= k:
            return idx
        return np.argsort(-sims)          # exact fallback; correctness over speed

    def query(self, state: Dict[str, float], k: int = 16, exclude_pid: Optional[str] = None) -> Dict[str, Any]:
        """Return the k most similar historical states + a similarity-weighted
        case-based resemblance score (soft k-NN P(septic-soon))."""
        if self.mem is None or not len(self.meta):
            return self._empty()
        q = np.ascontiguousarray(self.encode(state), dtype=np.float32)
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            sims = (self._M @ q) / float(self.eng.D)
        # Rank without sorting the whole store. This asks for the 16 nearest cases and
        # used to pay a full ``argsort`` over every stored patient-hour to get them —
        # O(N log N) per bed per tick, against an N that grows with every bed-hour the
        # OS observes. At 100 beds that single line was 83% of the tick. argpartition
        # is O(N) and only the small candidate window is then sorted.
        #
        # The window is oversized because ``exclude_pid`` may discard many of the top
        # hits (a patient's own recent hours are its nearest neighbours by construction),
        # and it falls back to the exact full ordering if the window somehow cannot
        # supply k survivors — so this is a speedup, not an approximation.
        return self._finish(sims, k, exclude_pid)

    def query_batch(self, states: List[Dict[str, float]], k: int = 16,
                    exclude_pids: Optional[List[Optional[str]]] = None
                    ) -> List[Dict[str, Any]]:
        """Answer the whole ward's queries in one matrix multiply.

        Every bed scores against the SAME case matrix on the same tick, so B separate
        ``(N,D)@(D,)`` matvecs are B passes over the same N*D floats — memory-bound,
        cache-hostile, and single-threaded. One ``(N,D)@(D,B)`` GEMM reads the matrix
        once and hands BLAS a shape it can actually use.

        This is the difference between an OS that runs a ward and one that runs a bed:
        the per-bed query cost grows with the case store, and the case store grows with
        the number of beds, so the naive loop is quadratic in ward size. Results are
        identical to calling :meth:`query` per bed — same similarities, same neighbours.
        """
        n = len(states)
        if self.mem is None or not len(self.meta) or n == 0:
            return [self._empty() for _ in range(n)]
        if exclude_pids is None:
            exclude_pids = [None] * n
        Q = np.empty((self.eng.D, n), dtype=np.float32)
        Q[:] = self.encode_many(states).T
        with np.errstate(divide="ignore", over="ignore", invalid="ignore"):
            S = (self._M @ Q) / float(self.eng.D)        # (N, B) — one GEMM
        return [self._finish(np.ascontiguousarray(S[:, j]), k, exclude_pids[j])
                for j in range(n)]

    @staticmethod
    def _empty() -> Dict[str, Any]:
        return {"resemblance": 0.0, "k": 0, "n_septic": 0, "lead_h": None, "top": []}

    def _finish(self, sims: np.ndarray, k: int, exclude_pid: Optional[str]) -> Dict[str, Any]:
        """Turn one column of similarities into the case-based verdict."""
        order = self._rank(sims, k, exclude_pid)
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
            return self._empty()
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
