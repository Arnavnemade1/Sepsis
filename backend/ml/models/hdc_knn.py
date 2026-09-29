"""HDC case retrieval as a detector — the precedent memory turned into a stack member.

Every earlier attempt made hyperdimensional computing the *classifier* (archetype
prototypes, HDC features into a booster) and lost to gradient boosting on raw ranking.
Stacking does not pay for strength, it pays for disagreement (``docs/FRONTIER_REPORT.md``:
a weaker XGBoost that rank-correlated 0.68 with the ensemble added AUROC where re-aiming
the strong members added nothing). So this asks HDC to be what the live case memory
already is — an instance-based retriever, the least tree-like model in the project — and
measures whether its disagreement is worth anything to the ensemble.

Each patient-hour is encoded as a hypervector; its score is the similarity-weighted vote
of its ``k`` nearest hours in a bank drawn from OTHER patients (the training folds):

    flat        record encoding: bind(id_f, level(x_f)) for every feature, bundled, sign
    segmented   one segment of the hypervector per vital channel (+ one for context),
                integer bundles kept (no sign): fewer features superposed per segment and
                no 1-bit quantisation — the two losses diagnosed in the old HDC detector

Levels are data-driven (training-fold quantiles), so every level carries mass. The vote
reads either the hour label or the patient label (ever septic) of the neighbours — the
latter is the supervision this project found to be its one real lever.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

CHANNEL_PREFIXES = ("HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp")


def _level_table(rng, levels: int, dim: int) -> np.ndarray:
    """(levels, dim) int8: level 0 and level L-1 independent, each step flips the next
    slice of the positions where they differ — similarity falls linearly with distance."""
    lo = rng.choice(np.array([-1, 1], dtype=np.int8), size=dim)
    hi = rng.choice(np.array([-1, 1], dtype=np.int8), size=dim)
    diff = rng.permutation(np.flatnonzero(lo != hi))
    table = np.empty((levels, dim), dtype=np.int8)
    table[0] = lo
    for i in range(1, levels):
        table[i] = table[i - 1]
        a = int((i - 1) * len(diff) / (levels - 1))
        b = int(i * len(diff) / (levels - 1))
        table[i, diff[a:b]] = hi[diff[a:b]]
    return table


def segments_of(feature_names: List[str]) -> List[int]:
    """Segment id per feature: its vital channel, or the shared context segment."""
    out = []
    for n in feature_names:
        head = n.split("__", 1)[0]
        out.append(CHANNEL_PREFIXES.index(head) if head in CHANNEL_PREFIXES else len(CHANNEL_PREFIXES))
    return out


class HDCKNN:
    def __init__(self, feature_names: List[str], dim: int = 4096, levels: int = 32, k: int = 64,
                 per_patient: int = 8, encoding: str = "flat", seed: int = 0, chunk: int = 1024):
        if encoding not in ("flat", "segmented"):
            raise ValueError(encoding)
        self.names, self.dim, self.levels, self.k = list(feature_names), dim, levels, k
        self.per_patient, self.encoding, self.seed, self.chunk = per_patient, encoding, seed, chunk
        rng = np.random.default_rng(seed)
        F = len(self.names)
        self.seg = np.array(segments_of(self.names))
        n_seg = int(self.seg.max()) + 1
        if encoding == "segmented":
            self.seg_dim = dim // n_seg
            self.dim = self.seg_dim * n_seg
            width = self.seg_dim
        else:
            width = dim
        level = _level_table(rng, levels, width)
        ids = rng.choice(np.array([-1, 1], dtype=np.int8), size=(F, width))
        self.T = (ids[:, None, :] * level[None, :, :]).astype(np.int8)   # (F, L, width) bound table
        self.tie = rng.choice(np.array([-1, 1], dtype=np.int8), size=self.dim)
        self.edges: Optional[List[np.ndarray]] = None

    # ------------------------------------------------------------------ #
    def _levels(self, X: np.ndarray) -> np.ndarray:
        L = np.empty(X.shape, dtype=np.int16)
        for j, e in enumerate(self.edges):
            L[:, j] = np.searchsorted(e, X[:, j], side="right")
        return L

    def encode(self, X: np.ndarray) -> np.ndarray:
        """(n, dim) float32 unit vectors."""
        lv = self._levels(X)
        n, F = lv.shape
        out = np.empty((n, self.dim), dtype=np.float32)
        for a in range(0, n, 8192):
            b = min(a + 8192, n)
            if self.encoding == "flat":
                acc = np.zeros((b - a, self.dim), dtype=np.int16)
                for f in range(F):
                    acc += self.T[f, lv[a:b, f]]
                acc = np.where(acc == 0, self.tie[None, :], acc)
                v = np.sign(acc).astype(np.float32)
            else:
                acc = np.zeros((b - a, self.dim), dtype=np.int16)
                for f in range(F):
                    s = int(self.seg[f]) * self.seg_dim
                    acc[:, s:s + self.seg_dim] += self.T[f, lv[a:b, f]]
                v = acc.astype(np.float32)
            nrm = np.linalg.norm(v, axis=1, keepdims=True)
            out[a:b] = v / np.maximum(nrm, 1e-6)
        return out

    # ------------------------------------------------------------------ #
    def fit(self, X: np.ndarray, y: np.ndarray, groups: np.ndarray) -> "HDCKNN":
        """Bank = up to ``per_patient`` evenly spaced hours per training patient, plus as
        many from each septic patient's positive hours. Both label views are kept, so one
        retrieval answers the hour vote and the patient vote."""
        rng = np.random.default_rng(self.seed)
        sub = rng.choice(len(X), size=min(len(X), 200_000), replace=False)
        qs = np.linspace(0, 1, self.levels + 1)[1:-1]
        self.edges = [np.unique(np.quantile(X[sub, j], qs)) for j in range(X.shape[1])]
        order = np.lexsort((np.arange(len(groups)), groups))
        g = groups[order]
        starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        ends = np.r_[starts[1:], len(g)]
        pick, plab = [], []
        for a, b in zip(starts, ends):
            rows = order[a:b]
            ever = bool(y[rows].any())
            take = rows[np.unique(np.linspace(0, len(rows) - 1, min(self.per_patient, len(rows))).astype(int))]
            if ever:
                pos = rows[y[rows] == 1]
                take = np.unique(np.r_[take, pos[np.unique(np.linspace(0, len(pos) - 1,
                                                                      min(self.per_patient, len(pos))).astype(int))]])
            pick.append(take)
            plab.append(np.full(len(take), float(ever)))
        self.bank_rows = np.concatenate(pick)
        self.bank = self.encode(X[self.bank_rows])
        self.lab = {"hour": y[self.bank_rows].astype(np.float32),
                    "patient": np.concatenate(plab).astype(np.float32)}
        return self

    def predict_votes(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Similarity-weighted vote of the k nearest bank hours, for each label view."""
        out = {t: np.empty(len(X), dtype=np.float64) for t in self.lab}
        bankT = np.ascontiguousarray(self.bank.T)
        for a in range(0, len(X), self.chunk):
            Q = self.encode(X[a:a + self.chunk])
            with np.errstate(all="ignore"):
                S = Q @ bankT
            idx = np.argpartition(-S, self.k - 1, axis=1)[:, :self.k]
            w = np.maximum(np.take_along_axis(S, idx, axis=1), 0.0) + 1e-6
            for t, lab in self.lab.items():
                out[t][a:a + len(Q)] = (w * lab[idx]).sum(1) / w.sum(1)
        return out
