"""Multi-Hospital Federated Intelligence Network (MEGA-UPGRADE Layer 1).

The cross-hospital generalization gap is the project's most honest weakness: a
detector trained at one site degrades at another (CinC-2019 setA→setB is exactly
that — two real source hospitals). Federated learning is the correct fix, and
unlike most "AI" claims it is **falsifiable**: train across sites *without sharing
records* and measure whether cross-site AUROC actually recovers toward the
centralized (data-pooled) ceiling and above a single-site local model.

What this module is, stated precisely (the integrity rules):

  * **The federated model is logistic regression.** FedAvg averages *model
    weights*; that only makes sense for a differentiable model. Decision-tree
    ensembles (the production GBM) have no averageable weights, so the FL detector
    is an L2 logistic model with real gradients/weight-deltas. This is the correct
    substrate, not a shortcut — and it is stated, not hidden.

  * **No raw records ever leave a node.** A node emits only a weight-delta vector +
    sample count + an HMAC signature. There is no code path by which X or y leaves
    a :class:`HospitalNode`. (This is data-minimization, NOT differential privacy —
    weight deltas can still leak; DP/secure-aggregation is out of scope and not
    claimed.)

  * **"Byzantine-fault-tolerant" = signed updates + robust aggregation.** We do NOT
    claim Z3 "proves" a gradient is benign (it cannot). Updates are HMAC-signed and
    verified (a forged/unsigned update is dropped), and the server aggregates with a
    robust rule (coordinate-wise median / trimmed-mean / Krum) that provably
    tolerates up to f malicious nodes — demonstrated by injecting a poisoner and
    showing the global model survives.

  * **Standardization is a shared preprocessing spec** (population feature
    mean/std), not patient data. Held constant across local/central/federated so the
    comparison isolates the *training scheme*, which is the whole point.
"""
from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np


# --------------------------------------------------------------------------- #
# Logistic-regression core (the averageable model).
# --------------------------------------------------------------------------- #
def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Clip first so neither branch's exp overflows (np.where evaluates both).
    z = np.clip(np.nan_to_num(z, nan=0.0, posinf=30.0, neginf=-30.0), -30.0, 30.0)
    return np.where(z >= 0, 1.0 / (1.0 + np.exp(-z)),
                    np.exp(z) / (1.0 + np.exp(z)))


@dataclass
class LogRegModel:
    """L2-regularized logistic regression with an explicit weight vector.

    weights[0] is the bias; weights[1:] multiply the (already standardized)
    features. Kept as a flat float64 vector so it averages trivially across nodes.
    """
    weights: np.ndarray

    @staticmethod
    def zeros(n_features: int) -> "LogRegModel":
        return LogRegModel(np.zeros(n_features + 1, dtype=np.float64))

    def proba(self, X: np.ndarray) -> np.ndarray:
        w = np.nan_to_num(self.weights, posinf=1e6, neginf=-1e6)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            return _sigmoid(w[0] + X @ w[1:])

    def copy(self) -> "LogRegModel":
        return LogRegModel(self.weights.copy())


def _grad(w: np.ndarray, X: np.ndarray, y: np.ndarray,
          pos_weight: float, l2: float) -> np.ndarray:
    """Class-weighted logistic gradient (positives up-weighted for imbalance)."""
    z = w[0] + X @ w[1:]
    p = _sigmoid(z)
    sw = np.where(y == 1, pos_weight, 1.0)
    err = sw * (p - y)
    g = np.empty_like(w)
    g[0] = err.mean()
    g[1:] = (X.T @ err) / len(y) + l2 * w[1:]
    return g


def train_local(w0: np.ndarray, X: np.ndarray, y: np.ndarray,
                epochs: int = 60, lr: float = 0.5, l2: float = 1e-3,
                pos_weight: Optional[float] = None) -> np.ndarray:
    """Full-batch gradient descent from w0 on one node's data. Returns new weights."""
    w = w0.astype(np.float64).copy()
    if pos_weight is None:
        pos = max(1, int(y.sum()))
        pos_weight = float(len(y) - pos) / pos          # balance to ~50/50 influence
        pos_weight = float(np.clip(pos_weight, 1.0, 50.0))
    # Overflow here only occurs when starting from adversarially-exploded weights
    # (the poisoning path); robust aggregation is what defends against it.
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        for _ in range(epochs):
            w -= lr * _grad(w, X, y, pos_weight, l2)
    return w


# --------------------------------------------------------------------------- #
# Signed weight-delta update (the only thing that crosses the wire).
# --------------------------------------------------------------------------- #
@dataclass
class Update:
    node_id: str
    delta: np.ndarray            # w_local - w_global  (NO patient data, ever)
    n_samples: int
    signature: str

    def carries_patient_data(self) -> bool:
        """Structural privacy assertion: an Update is a weight vector + scalar."""
        return any(isinstance(getattr(self, f, None), np.ndarray) and
                   getattr(self, f).ndim > 1 for f in ("delta",))


def _sign(key: bytes, delta: np.ndarray, n: int) -> str:
    h = hmac.new(key, digestmod=hashlib.sha256)
    h.update(np.ascontiguousarray(delta, dtype=np.float64).tobytes())
    h.update(str(int(n)).encode())
    return h.hexdigest()


# --------------------------------------------------------------------------- #
# A hospital node — holds its own data + key; emits only signed deltas.
# --------------------------------------------------------------------------- #
@dataclass
class HospitalNode:
    node_id: str
    X: np.ndarray
    y: np.ndarray
    key: bytes
    byzantine: bool = False
    attack: str = "sign_flip"         # how a malicious node poisons its update

    def local_update(self, global_w: np.ndarray, epochs: int = 60,
                     lr: float = 0.5) -> Update:
        if self.byzantine:
            delta = self._poison(global_w)
        else:
            w_local = train_local(global_w, self.X, self.y, epochs=epochs, lr=lr)
            delta = w_local - global_w
        return Update(self.node_id, delta, len(self.y),
                      _sign(self.key, delta, len(self.y)))

    def _poison(self, global_w: np.ndarray) -> np.ndarray:
        """A malicious node tries to wreck the global model. It still signs with its
        own key (it is an authenticated-but-faulty participant — the realistic threat
        robust aggregation defends against)."""
        honest = train_local(global_w, self.X, self.y, epochs=20, lr=0.5) - global_w
        if self.attack == "scaled":
            return honest * -25.0
        return -honest * 10.0          # sign-flip + amplify (gradient ascent)


# --------------------------------------------------------------------------- #
# Robust aggregation rules — tolerate up to f Byzantine nodes.
# --------------------------------------------------------------------------- #
def agg_fedavg(deltas: List[np.ndarray], weights: List[float]) -> np.ndarray:
    W = np.array(weights, dtype=np.float64); W /= W.sum()
    return np.average(np.stack(deltas), axis=0, weights=W)


def agg_median(deltas: List[np.ndarray], weights: List[float]) -> np.ndarray:
    return np.median(np.stack(deltas), axis=0)


def agg_trimmed_mean(deltas: List[np.ndarray], weights: List[float],
                     trim: int = 1) -> np.ndarray:
    M = np.sort(np.stack(deltas), axis=0)
    if 2 * trim < len(deltas):
        M = M[trim:len(deltas) - trim]
    return M.mean(axis=0)


def agg_krum(deltas: List[np.ndarray], weights: List[float], f: int = 1) -> np.ndarray:
    """Krum: pick the update closest (sum of sq-dist to its n-f-2 nearest peers)
    to the honest majority — a single Byzantine update can't win."""
    n = len(deltas)
    D = np.stack(deltas)
    m = max(1, n - f - 2)
    scores = []
    for i in range(n):
        d = np.sum((D - D[i]) ** 2, axis=1)
        d.sort()
        scores.append(d[1:m + 1].sum())
    return D[int(np.argmin(scores))]


AGGREGATORS: Dict[str, Callable] = {
    "fedavg": agg_fedavg, "median": agg_median,
    "trimmed_mean": agg_trimmed_mean, "krum": agg_krum,
}


# --------------------------------------------------------------------------- #
# The federated server — verifies signatures, aggregates, never sees data.
# --------------------------------------------------------------------------- #
@dataclass
class FederatedServer:
    n_features: int
    keys: Dict[str, bytes]                       # node_id -> shared HMAC key
    method: str = "fedavg"
    server_lr: float = 1.0
    rejected: int = 0
    accepted: int = 0
    audit: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        self.global_model = LogRegModel.zeros(self.n_features)

    def _verify(self, u: Update) -> bool:
        key = self.keys.get(u.node_id)
        if key is None:
            return False
        good = hmac.compare_digest(u.signature, _sign(key, u.delta, u.n_samples))
        self.audit.append({"node": u.node_id, "accepted": good, "n": u.n_samples})
        return good

    def step(self, updates: List[Update]) -> None:
        valid = [u for u in updates if self._verify(u)]
        self.rejected += len(updates) - len(valid)
        self.accepted += len(valid)
        if not valid:
            return
        deltas = [u.delta for u in valid]
        weights = [float(u.n_samples) for u in valid]
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            agg = AGGREGATORS[self.method](deltas, weights)
            self.global_model.weights += self.server_lr * agg

    def run(self, nodes: List[HospitalNode], rounds: int = 12,
            epochs: int = 60, lr: float = 0.5) -> LogRegModel:
        for _ in range(rounds):
            gw = self.global_model.weights
            self.step([n.local_update(gw, epochs=epochs, lr=lr) for n in nodes])
        return self.global_model


# --------------------------------------------------------------------------- #
# Experiment plumbing — real CinC setA/setB as distinct hospitals.
# --------------------------------------------------------------------------- #
@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(X: np.ndarray) -> "Standardizer":
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd < 1e-6] = 1.0
        return Standardizer(mu, sd)

    def apply(self, X: np.ndarray) -> np.ndarray:
        # Clip standardized values to tame heavy-tailed physiologic outliers.
        return np.clip((X - self.mean) / self.std, -10.0, 10.0)


def load_site_matrix(set_name: str, limit: int, cfg=None):
    """Feature matrix + patient groups for one real hospital (cached)."""
    from backend.ml.data.cinc2019 import load_patients
    from backend.ml.data.features import build_dataset_cached, FeatureConfig
    cfg = cfg or FeatureConfig.wearable_only()
    ds = build_dataset_cached(
        lambda: load_patients(sets=[set_name], limit=limit),
        cfg, cache_key=f"fed_{set_name}_{limit}")
    return ds


def _patient_split(groups: np.ndarray, frac: float, seed: int):
    """Leakage-free split: whole patients go to train or test, never split."""
    rng = np.random.default_rng(seed)
    pts = np.unique(groups)
    rng.shuffle(pts)
    k = int(len(pts) * frac)
    train_pts = set(pts[:k].tolist())
    mask = np.array([g in train_pts for g in groups])
    return mask, ~mask


def make_nodes(shards: List[Tuple[str, np.ndarray, np.ndarray]],
               seed: int = 0) -> List[HospitalNode]:
    rng = np.random.default_rng(seed)
    return [HospitalNode(nid, X, y, key=rng.bytes(16)) for nid, X, y in shards]


# --------------------------------------------------------------------------- #
# The headline experiment: does federation recover cross-hospital AUROC?
# --------------------------------------------------------------------------- #
def run_experiment(limit: int = 500, n_nodes: int = 4, rounds: int = 12,
                   epochs: int = 60, seed: int = 7) -> Dict[str, Any]:
    """Real CinC setA (federation hospitals) vs setB (unseen external hospital).

    Compares, on the SAME shared standardizer, three training schemes evaluated
    in-distribution (setA held-out) and cross-site (setB, never trained on):
      local-only  — one hospital, no collaboration
      centralized — all setA data pooled (privacy-violating upper bound)
      federated   — setA hospitals collaborate via signed weight-deltas only
    """
    from backend.ml.eval.metrics import auroc

    A = load_site_matrix("training_setA", limit)
    B = load_site_matrix("training_setB", limit)

    tr_mask, te_mask = _patient_split(A.groups, frac=0.8, seed=seed)
    std = Standardizer.fit(A.X[tr_mask])
    XA_tr, yA_tr, gA_tr = std.apply(A.X[tr_mask]), A.y[tr_mask], A.groups[tr_mask]
    XA_te, yA_te = std.apply(A.X[te_mask]), A.y[te_mask]
    XB, yB = std.apply(B.X), B.y                      # external hospital (unseen)
    d = XA_tr.shape[1]

    # carve setA-train into n_nodes hospitals (patient-grouped, no leakage)
    rng = np.random.default_rng(seed)
    pts = np.unique(gA_tr); rng.shuffle(pts)
    buckets = np.array_split(pts, n_nodes)
    shards = []
    for i, b in enumerate(buckets):
        m = np.isin(gA_tr, b)
        shards.append((f"hospital_{i}", XA_tr[m], yA_tr[m]))

    def ev(model):
        return {"setA_internal": round(auroc(yA_te, model.proba(XA_te)), 4),
                "setB_cross_site": round(auroc(yB, model.proba(XB)), 4)}

    # local-only: each hospital alone (mean over hospitals)
    local_eval = []
    for _, Xs, ys in shards:
        w = train_local(np.zeros(d + 1), Xs, ys, epochs=epochs * rounds // 2)
        local_eval.append(ev(LogRegModel(w)))
    local = {k: round(float(np.mean([e[k] for e in local_eval])), 4) for k in local_eval[0]}

    # centralized: pooled setA (data shared)
    w_c = train_local(np.zeros(d + 1), XA_tr, yA_tr, epochs=epochs * rounds // 2)
    central = ev(LogRegModel(w_c))

    # federated: hospitals collaborate, only signed weight-deltas move
    nodes = make_nodes(shards, seed=seed)
    server = FederatedServer(d, keys={n.node_id: n.key for n in nodes}, method="fedavg")
    gmodel = server.run(nodes, rounds=rounds, epochs=epochs)
    fed = ev(gmodel)

    gap_local = round(local["setA_internal"] - local["setB_cross_site"], 4)
    gap_fed = round(fed["setA_internal"] - fed["setB_cross_site"], 4)
    return {
        "n_hospitals": n_nodes, "rounds": rounds, "limit_per_set": limit,
        "patients": {"setA": int(len(np.unique(A.groups))), "setB": int(len(np.unique(B.groups)))},
        "local_only": local, "centralized": central, "federated": fed,
        "cross_site_gap": {"local_only": gap_local, "federated": gap_fed,
                           "narrowed_by": round(gap_local - gap_fed, 4)},
        "headline": {
            "local_setB": local["setB_cross_site"],
            "federated_setB": fed["setB_cross_site"],
            "centralized_setB": central["setB_cross_site"],
            "federated_recovers": round(fed["setB_cross_site"] - local["setB_cross_site"], 4),
        },
    }


def run_poisoning(limit: int = 500, n_nodes: int = 5, rounds: int = 12,
                  epochs: int = 60, seed: int = 7) -> Dict[str, Any]:
    """One hospital turns Byzantine. FedAvg (mean) is corruptible; robust
    aggregation (Krum/median) tolerates it. Measured on setB."""
    from backend.ml.eval.metrics import auroc

    A = load_site_matrix("training_setA", limit)
    B = load_site_matrix("training_setB", limit)
    tr_mask, _ = _patient_split(A.groups, frac=0.8, seed=seed)
    std = Standardizer.fit(A.X[tr_mask])
    XA, yA, gA = std.apply(A.X[tr_mask]), A.y[tr_mask], A.groups[tr_mask]
    XB, yB = std.apply(B.X), B.y
    d = XA.shape[1]

    rng = np.random.default_rng(seed)
    pts = np.unique(gA); rng.shuffle(pts)
    shards = [(f"hospital_{i}", XA[np.isin(gA, b)], yA[np.isin(gA, b)])
              for i, b in enumerate(np.array_split(pts, n_nodes))]

    def federated(method, byz):
        nodes = make_nodes(shards, seed=seed)
        if byz:
            nodes[0].byzantine = True
        srv = FederatedServer(d, keys={n.node_id: n.key for n in nodes}, method=method)
        m = srv.run(nodes, rounds=rounds, epochs=epochs)
        return round(auroc(yB, m.proba(XB)), 4)

    return {
        "clean_fedavg": federated("fedavg", byz=False),
        "poisoned_fedavg": federated("fedavg", byz=True),
        "poisoned_krum": federated("krum", byz=True),
        "poisoned_median": federated("median", byz=True),
    }


def _demo() -> None:
    res = run_experiment(limit=400, n_nodes=4, rounds=10, epochs=50)
    print(f"hospitals={res['n_hospitals']} patients={res['patients']}")
    for scheme in ("local_only", "centralized", "federated"):
        r = res[scheme]
        print(f"  {scheme:12} setA(internal)={r['setA_internal']}  setB(cross-site)={r['setB_cross_site']}")
    print("  cross-site gap:", res["cross_site_gap"])
    print("  poisoning:", run_poisoning(limit=400, n_nodes=5, rounds=10, epochs=50))


if __name__ == "__main__":
    _demo()
