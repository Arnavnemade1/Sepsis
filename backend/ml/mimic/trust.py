"""The trust layer — a held-out number is only as good as the process that produced it.

Every mechanism here answers a way a locked-test number goes bad even when every line of
modelling code is right.

HASH CHAIN. Pre-registrations, lock accesses, confirmations and Ladder submissions are
entries in one append-only JSONL chain. Each entry commits to the previous entry's hash
(SHA-256 of a canonical encoding) and carries a fingerprint of the harness code — the
source of every module that computes a number, plus the git commit. Editing, deleting,
inserting or reordering any past entry breaks the chain from that entry on, and
``Chain.verify`` names the first bad one; ``append`` refuses to extend a broken chain. So
a pre-registration is provably older than the result that confirms it and bound to the
code that produced it (committing the chain to git anchors it in time — the idea of
Irving & Holden, F1000Research 2016, for trial protocols, applied inside a harness).

POWER-GATED PRE-REGISTRATION. Before the lock can be spent, the harness measures on the
dev folds how precisely the lock could answer the claim (a subject-bootstrap standard
error, scaled to the number of locked subjects) and REFUSES a registration it cannot
answer: a comparative claim whose minimum important difference is smaller than the lock's
minimum detectable effect, or an absolute claim the lock would confirm with less than 80%
probability. An underpowered test cannot confirm anything, and a futile one burns the lock.

LADDER LOCK. A single-use lock allows one look; a lock consulted repeatedly becomes a dev
set, because every look leaks information and adaptive choices overfit it. Between the two
is the Ladder (Blum & Hardt, ICML 2015): each exploratory submission is scored on the lock,
but a new number is RELEASED only when it beats the best released number by more than a
step eta — set from the lock's own sampling error — and then only rounded to eta; otherwise
the previous best is repeated. The budget is fixed, and the raw score is never written
anywhere. ``verify_mimic`` runs the classic boosting attack: against a naive holdout, an
attacker who only submits noise climbs to AUROC ~0.6; against the Ladder it cannot.

A pre-registered confirmatory test is spent once and released in full (with its bound);
the Ladder is for everything else.
"""
from __future__ import annotations

import glob
import hashlib
import json
import os
import subprocess
import time
from dataclasses import dataclass
from statistics import NormalDist
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

from backend.ml import config
from backend.ml.genesis.harness import Ranked

OUT_DIR = os.path.join(config.ARTIFACTS_DIR, "mimic")
CHAIN_PATH = os.path.join(OUT_DIR, "lock_ledger.jsonl")
ROOT_HASH = "0" * 64
#: every module whose code can change a number the chain vouches for
HARNESS_GLOBS = ("backend/ml/mimic/*.py", "backend/ml/genesis/features.py",
                 "backend/ml/genesis/learners.py", "backend/ml/genesis/harness.py",
                 "backend/ml/genesis/stack.py", "backend/ml/genesis/network.py",
                 "backend/ml/genesis/rawstore.py")
LADDER_BUDGET = 8
ETA_MIN = 0.005
_Z = NormalDist()


# --------------------------------------------------------------------------- #
# The chain
# --------------------------------------------------------------------------- #
def _plain(o):
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON-serialisable: {type(o).__name__}")


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=_plain, allow_nan=True)


def _digest(entry: Dict) -> str:
    return hashlib.sha256(canonical({k: v for k, v in entry.items() if k != "hash"}).encode()).hexdigest()


class Chain:
    """Append-only, hash-linked JSONL ledger."""

    def __init__(self, path: str = CHAIN_PATH):
        self.path = path

    def _lines(self) -> List[str]:
        if not os.path.exists(self.path):
            return []
        with open(self.path) as fh:
            return [ln for ln in fh.read().split("\n") if ln.strip()]

    def entries(self) -> List[Dict]:
        return [json.loads(ln) for ln in self._lines()]

    def verify(self) -> Tuple[bool, int, Optional[int]]:
        """``(intact, n_entries, first_bad_seq)``."""
        prev = ROOT_HASH
        lines = self._lines()
        for i, ln in enumerate(lines):
            try:
                e = json.loads(ln)
            except json.JSONDecodeError:
                return False, len(lines), i
            if e.get("seq") != i or e.get("prev") != prev or _digest(e) != e.get("hash"):
                return False, len(lines), i
            prev = e["hash"]
        return True, len(lines), None

    def head(self) -> str:
        es = self.entries()
        return es[-1]["hash"] if es else ROOT_HASH

    def append(self, kind: str, body: Dict) -> Dict:
        ok, n, bad = self.verify()
        if not ok:
            raise RuntimeError(f"ledger chain {self.path} is broken at entry {bad}; refusing to extend it")
        entry = {"seq": n, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "kind": kind,
                 "body": json.loads(canonical(body)), "prev": self.head()}
        entry["hash"] = _digest(entry)
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a") as fh:
            fh.write(canonical(entry) + "\n")
        return entry

    def of_kind(self, kind: str) -> List[Dict]:
        return [e for e in self.entries() if e["kind"] == kind]


def code_fingerprint(root: str = config.REPO_ROOT) -> Dict:
    """SHA-256 over the harness source (path + bytes of every module that computes a
    number), plus the git commit and whether those files differ from it."""
    files = sorted({f for g in HARNESS_GLOBS for f in glob.glob(os.path.join(root, g))})
    h = hashlib.sha256()
    for f in files:
        h.update(os.path.relpath(f, root).encode() + b"\0")
        with open(f, "rb") as fh:
            h.update(fh.read() + b"\0")
    git, dirty = None, None
    try:
        rel = [os.path.relpath(f, root) for f in files]
        git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True,
                             text=True, timeout=5).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "status", "--porcelain", "--", *rel], cwd=root,
                                    capture_output=True, text=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {"sha256": h.hexdigest(), "files": len(files), "git": git, "dirty": dirty}


# --------------------------------------------------------------------------- #
# Estimates
# --------------------------------------------------------------------------- #
def _subject_draws(subjects: np.ndarray, n_boot: int, seed: int) -> Iterator[np.ndarray]:
    codes, inv = np.unique(subjects, return_inverse=True)
    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        yield np.bincount(rng.integers(0, len(codes), len(codes)), minlength=len(codes))[inv]


def _rows(led, p, q, locked: bool) -> np.ndarray:
    f = led.folds[led.store.seg]
    m = led.eval_mask & ((f < 0) if locked else (f >= 0)) & np.isfinite(p)
    if q is not None:
        m &= np.isfinite(q)
    return np.flatnonzero(m)


def estimate(led, p: np.ndarray, q: Optional[np.ndarray] = None, locked: bool = False,
             n_boot: int = 300, seed: int = 0) -> Dict:
    """AUROC of ``p`` (or AUROC(p) - AUROC(q)) on the dev or locked rows, with its
    subject-bootstrap draws (design-weighted)."""
    rows = _rows(led, p, q, locked)
    y, w = led.y[rows], led.design_w[rows]
    subj = led.store.subject[led.store.seg[rows]]
    Rp = Ranked(p[rows])
    Rq = Ranked(q[rows]) if q is not None else None

    def val(ww):
        return Rp.auc(y, ww) - (Rq.auc(y, ww) if Rq is not None else 0.0)
    draws = np.array([val(w * m) for m in _subject_draws(subj, n_boot, seed)])
    draws = draws[np.isfinite(draws)]
    return {"value": val(w), "se": float(np.std(draws, ddof=1)) if len(draws) > 1 else float("nan"),
            "draws": draws, "n_subjects": int(len(np.unique(subj))), "n_rows": int(len(rows))}


def power_plan(led, p: np.ndarray, q: Optional[np.ndarray] = None, bar: float = 0.0,
               alpha: float = 0.05, power: float = 0.8, n_boot: int = 300, seed: int = 0) -> Dict:
    """Could the lock answer this claim? The dev-fold SE, scaled to the locked subjects
    (SE ~ 1/sqrt(subjects)), gives the lock's precision.

    absolute (``q is None``): H1 AUROC > ``bar``; expected power = P(lower bound > bar)
        at the dev estimate. Refused below ``power`` (a futile test burns the lock).
    comparative: H1 AUROC(p) > AUROC(q), ``bar`` = minimum important difference.
        MDE = (z_{1-alpha} + z_power) * SE_lock. Refused when bar < MDE (underpowered)."""
    est = estimate(led, p, q, locked=False, n_boot=n_boot, seed=seed)
    n_lock = int(len(np.unique(led.store.subject[led.folds < 0])))
    se_lock = est["se"] * np.sqrt(est["n_subjects"] / max(n_lock, 1))
    za, zb = _Z.inv_cdf(1 - alpha), _Z.inv_cdf(power)
    out = {"mode": "absolute" if q is None else "comparative", "bar": float(bar), "alpha": alpha,
           "power_required": power, "dev_value": est["value"], "se_dev": est["se"],
           "se_lock": float(se_lock), "n_dev_subjects": est["n_subjects"], "n_lock_subjects": n_lock}
    if not np.isfinite(se_lock) or se_lock <= 0:
        return out | {"admissible": False, "reason": "no usable standard error on the dev folds"}
    if q is None:
        pw = _Z.cdf((est["value"] - bar) / se_lock - za)
        ok = pw >= power
        return out | {"expected_power": pw, "admissible": bool(ok),
                      "reason": "" if ok else (f"futile: at the dev estimate {est['value']:.3f} the lock "
                                               f"(SE {se_lock:.3f}) confirms AUROC > {bar} with "
                                               f"probability {pw:.2f} < {power}")}
    mde = (za + zb) * se_lock
    ok = bar >= mde
    return out | {"mde": float(mde), "expected_power": _Z.cdf(est["value"] / se_lock - za),
                  "admissible": bool(ok),
                  "reason": "" if ok else (f"underpowered: the lock's minimum detectable difference is "
                                           f"{mde:.3f} (SE {se_lock:.3f}, {n_lock} subjects); a claimed "
                                           f"difference of {bar} cannot be confirmed")}


# --------------------------------------------------------------------------- #
# Pre-registration and confirmation
# --------------------------------------------------------------------------- #
def lock_scores(led, cand, purpose: str, chain: Optional[Chain] = None) -> np.ndarray:
    """Fit ``cand`` on every dev row and score the locked rows (NaN elsewhere). The access
    is chained BEFORE any locked row is read."""
    lock = led.lock_rows(f"{purpose}: {cand.name}", chain)
    cols = cand.cols(led)
    dev = np.flatnonzero((led.folds[led.store.seg] >= 0) & led.train_mask)
    predictor = cand.fit(led, cols, dev, -1)
    p = np.full(led.store.n_rows, np.nan)
    p[lock] = predictor(led, lock)
    return p


def preregister(led, name: str, hypothesis: str, p: np.ndarray, q: Optional[np.ndarray] = None,
                bar: float = 0.0, spec: Optional[Dict] = None, chain: Optional[Chain] = None,
                **plan_kw) -> Dict:
    """Chain a pre-registration — refused if the lock could not answer it."""
    from backend.ml.mimic.ledger import ONSET_MIN, lock_fingerprint
    plan = power_plan(led, p, q, bar, **plan_kw)
    if not plan["admissible"]:
        raise PermissionError(f"pre-registration refused — {plan['reason']}")
    return (chain or Chain()).append("preregister", {
        "name": name, "hypothesis": hypothesis, "rule": led.rule, "anchor": led.anchor,
        "cohort": f"early-warning (onset >= {ONSET_MIN:g} h)", "metric": "auroc", "plan": plan,
        "spec": spec or {}, "code": code_fingerprint(), "lock": lock_fingerprint(led.store, led.folds)})


def confirm(led, prereg_hash: str, p_lock: np.ndarray, q_lock: Optional[np.ndarray] = None,
            chain: Optional[Chain] = None, deviation: str = "", n_boot: int = 1000,
            seed: int = 0) -> Dict:
    """The confirmatory test of one pre-registration: spent once, released in full."""
    from backend.ml.mimic.ledger import lock_fingerprint
    chain = chain or Chain()
    ok, _, bad = chain.verify()
    if not ok:
        raise RuntimeError(f"ledger chain broken at entry {bad}")
    reg = [e for e in chain.of_kind("preregister") if e["hash"] == prereg_hash]
    if not reg:
        raise PermissionError("no pre-registration with that hash in the chain")
    reg = reg[0]["body"]
    if any(e["body"]["prereg"] == prereg_hash for e in chain.of_kind("confirm")):
        raise PermissionError("this pre-registration has already been confirmed — a confirmatory test is spent once")
    if lock_fingerprint(led.store, led.folds) != reg["lock"]:
        raise PermissionError("the locked subjects differ from the pre-registered lock")
    code = code_fingerprint()
    if code["sha256"] != reg["code"]["sha256"]:
        if not deviation:
            raise PermissionError("the harness code changed after pre-registration; confirm with "
                                  "deviation='<reason>' to record the deviation in the chain")
        chain.append("deviation", {"prereg": prereg_hash, "reason": deviation, "code": code})
    plan = reg["plan"]
    est = estimate(led, p_lock, q_lock, locked=True, n_boot=n_boot, seed=seed)
    lower = float(np.percentile(est["draws"], 100 * plan["alpha"])) if len(est["draws"]) else float("nan")
    target = plan["bar"] if plan["mode"] == "absolute" else 0.0
    body = {"prereg": prereg_hash, "name": reg["name"], "value": est["value"], "lower": lower,
            "alpha": plan["alpha"], "target": target, "passed": bool(lower > target),
            "n_subjects": est["n_subjects"], "n_rows": est["n_rows"], "code": code["sha256"]}
    chain.append("confirm", body)
    return body


# --------------------------------------------------------------------------- #
# The Ladder
# --------------------------------------------------------------------------- #
@dataclass
class LadderCore:
    """Blum & Hardt's Ladder for a higher-is-better score, without I/O."""
    eta: float
    budget: int = LADDER_BUDGET
    best: Optional[float] = None
    used: int = 0

    def submit(self, value: float) -> Dict:
        if self.used >= self.budget:
            raise PermissionError(f"the Ladder's budget of {self.budget} submissions is spent")
        self.used += 1
        if self.best is None or value > self.best + self.eta:
            self.best = self.eta * round(value / self.eta)
            return {"released": self.best, "improved": True, "remaining": self.budget - self.used}
        return {"released": self.best, "improved": False, "remaining": self.budget - self.used}


class Ladder:
    """The Ladder over the locked subjects; its state lives only in the chain."""

    def __init__(self, chain: Optional[Chain] = None, budget: int = LADDER_BUDGET):
        self.chain = chain or Chain()
        self.budget = budget

    def core(self) -> Optional[LadderCore]:
        es = self.chain.of_kind("ladder")
        if not es:
            return None
        released = [e["body"]["released"] for e in es]
        return LadderCore(eta=es[0]["body"]["eta"], budget=self.budget, best=max(released), used=len(es))

    def submit(self, led, name: str, p_lock: np.ndarray, seed: int = 0) -> Dict:
        from backend.ml.mimic.ledger import lock_fingerprint
        core = self.core()
        est = estimate(led, p_lock, locked=True, n_boot=200 if core is None else 0, seed=seed)
        if core is None:
            core = LadderCore(eta=float(max(ETA_MIN, est["se"])) if np.isfinite(est["se"]) else ETA_MIN,
                              budget=self.budget)
        out = core.submit(est["value"])
        # the released number only — the raw lock score is never written
        self.chain.append("ladder", {"name": name, "released": out["released"], "improved": out["improved"],
                                     "eta": core.eta, "n": core.used, "budget": self.budget,
                                     "code": code_fingerprint()["sha256"],
                                     "lock": lock_fingerprint(led.store, led.folds)})
        return out | {"eta": core.eta}


def summary(chain: Optional[Chain] = None) -> Dict:
    chain = chain or Chain()
    ok, n, bad = chain.verify()
    kinds: Dict[str, int] = {}
    for e in chain.entries() if ok else []:
        kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    return {"path": os.path.relpath(chain.path, config.REPO_ROOT), "intact": ok, "entries": n,
            "first_bad": bad, "head": chain.head() if ok else None, "by_kind": kinds}
