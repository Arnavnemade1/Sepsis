"""Forest-VSA — tree-leaf topology as a representation, and whether it buys anything.

THE PROPOSAL BEING TESTED
-------------------------
The incumbent stack trains its boosters and its HDC channel in isolation and blends
only their final scalars. The proposal: take the *decision topology* — which leaf of
each tree a patient-hour lands in — and use it as the representation itself. In the
"Forest-VSA" form, each tree id ``T_i`` and leaf id ``L_{i,k}`` gets a random bipolar
hypervector, the active pair is bound, all trees are bundled, and the sign of the sum
is a single 10,000-D descriptor supporting associative recall.

WHY THIS MODULE TESTS TWO THINGS AND NOT ONE
--------------------------------------------
The Forest-VSA hypervector is a **random projection of the exact leaf one-hot
vector**. Binding-and-bundling is a lossy compression of the same event set: the
leaves a row activates. It therefore cannot carry information the one-hot does not,
and the honest experiment is a sandwich —

    leaf-onehot + linear      an exact, lossless read of the same topology
    leaf-VSA    + prototypes  the proposed lossy read

If the lossless version does not beat the incumbent, the lossy one cannot, and any
apparent win by the VSA form is fold noise. Measuring only the VSA form would leave
that unanswerable, which is how a representation with no information gain gets
shipped as an architecture.

Both are fitted strictly inside the caller's training fold: the donor booster, the
codebook, the prototypes and the linear head all see ``tr`` only.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np


def leaf_matrix(model, X: np.ndarray) -> np.ndarray:
    """(n_rows, n_trees) leaf index per tree, for whichever booster was handed in.

    XGBoost is the donor of choice because ``xgb@utility`` is the strongest single
    learner in this project's frontier sweep, so its partitions are the ones most
    worth reading. sklearn's HistGradientBoosting exposes no public ``apply``, which
    is why the donor is a parameter rather than an assumption.
    """
    # `.apply` is the accessor on BOTH the XGBoost sklearn wrapper and sklearn's own
    # forests. (`predict(..., pred_leaf=True)` is the low-level Booster API and is
    # NOT accepted by XGBClassifier.predict.)
    if hasattr(model, "apply"):
        return np.asarray(model.apply(X), dtype=np.int64).reshape(len(X), -1)
    if hasattr(model, "get_booster"):
        import xgboost as xgb
        b = model.get_booster()
        return np.asarray(b.predict(xgb.DMatrix(X), pred_leaf=True),
                          dtype=np.int64).reshape(len(X), -1)
    raise TypeError(f"{type(model).__name__} exposes no leaf-index accessor")


# --------------------------------------------------------------------------- #
@dataclass
class LeafOneHot:
    """The lossless reference: leaf topology as exact sparse indicators.

    This is the classical GBDT+linear construction. It is included not as a
    candidate to ship but as the ceiling the VSA form is measured against — the most
    information the topology can possibly carry.
    """
    n_trees_: int = 0
    offsets_: Optional[np.ndarray] = None
    width_: int = 0

    def fit(self, leaves: np.ndarray) -> "LeafOneHot":
        self.n_trees_ = leaves.shape[1]
        # Leaf ids are per-tree; offsetting makes them globally unique columns.
        # +1 because ids are 0-based and a tree may not use its full id range.
        maxima = leaves.max(axis=0) + 1
        self.offsets_ = np.r_[0, np.cumsum(maxima)[:-1]].astype(np.int64)
        self.width_ = int(maxima.sum())
        return self

    def transform(self, leaves: np.ndarray):
        from scipy.sparse import csr_matrix
        n, t = leaves.shape
        # Unseen-in-training leaf ids are clipped into range rather than dropped: a
        # dropped row would silently become an all-zero vector that the linear head
        # reads as "average patient", which is a confident answer to a question the
        # model was never asked.
        cols = np.clip(leaves, 0, None) + self.offsets_[None, :]
        cols = np.clip(cols, 0, self.width_ - 1).ravel()
        rows = np.repeat(np.arange(n), t)
        return csr_matrix((np.ones(n * t, dtype=np.float32), (rows, cols)),
                          shape=(n, self.width_))


# --------------------------------------------------------------------------- #
@dataclass
class ForestVSA:
    """The proposal: bind tree-id ⊗ leaf-id, bundle, sign; score by prototype cosine.

    ``dimensions`` is the hypervector width. ``seed`` fixes the codebook so a rerun
    reproduces the same projection — an unfixed codebook would make every fold a
    different model and the comparison meaningless.
    """
    dimensions: int = 10000
    seed: int = 42
    codebook_: Optional[np.ndarray] = None        # (n_trees, max_leaf, D) is too big;
    tree_vecs_: Optional[np.ndarray] = None       # (n_trees, D) int8
    leaf_vecs_: Optional[np.ndarray] = None       # (max_leaf, D) int8, shared across trees
    proto_pos_: Optional[np.ndarray] = None
    proto_neg_: Optional[np.ndarray] = None

    def _build(self, n_trees: int, max_leaf: int) -> None:
        rng = np.random.default_rng(self.seed)
        # One shared leaf codebook, bound with a per-tree vector. Independent
        # codebooks per tree would need n_trees × max_leaf × D bytes — 600×64×10k is
        # 384 MB — while binding with a tree vector already makes the same leaf id in
        # two different trees near-orthogonal, which is the property that matters.
        self.tree_vecs_ = rng.choice(np.array([-1, 1], dtype=np.int8),
                                     size=(n_trees, self.dimensions))
        self.leaf_vecs_ = rng.choice(np.array([-1, 1], dtype=np.int8),
                                     size=(max_leaf, self.dimensions))

    def encode(self, leaves: np.ndarray) -> np.ndarray:
        """(n_rows, D) bipolar forest hypervectors."""
        n, t = leaves.shape
        acc = np.zeros((n, self.dimensions), dtype=np.int16)
        lv = np.clip(leaves, 0, self.leaf_vecs_.shape[0] - 1)
        for i in range(t):
            acc += (self.leaf_vecs_[lv[:, i]] * self.tree_vecs_[i]).astype(np.int16)
        out = np.where(acc >= 0, 1, -1).astype(np.int8)
        return out

    def fit(self, leaves: np.ndarray, y: np.ndarray,
            sample_weight: Optional[np.ndarray] = None) -> "ForestVSA":
        self._build(leaves.shape[1], int(leaves.max()) + 1)
        Hv = self.encode(leaves).astype(np.float32)
        y = np.asarray(y).astype(int)
        w = np.ones(len(y), dtype=np.float32) if sample_weight is None \
            else np.asarray(sample_weight, dtype=np.float32)
        # Class prototypes are weighted means, not sums: at 2% prevalence an unweighted
        # sum makes the negative prototype the dataset mean and the positive one noise.
        for lab, name in ((1, "proto_pos_"), (0, "proto_neg_")):
            m = y == lab
            p = (Hv[m] * w[m, None]).sum(0) / max(w[m].sum(), 1e-9) if m.any() \
                else np.zeros(self.dimensions, dtype=np.float32)
            setattr(self, name, p / (np.linalg.norm(p) + 1e-9))
        return self

    def score(self, leaves: np.ndarray) -> np.ndarray:
        """Signed resonance: cos(H, septic prototype) − cos(H, non-septic prototype)."""
        Hv = self.encode(leaves).astype(np.float32)
        Hv /= (np.linalg.norm(Hv, axis=1, keepdims=True) + 1e-9)
        return Hv @ self.proto_pos_ - Hv @ self.proto_neg_
