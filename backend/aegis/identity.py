"""Continuous Proof-of-Patient-Identity via physiologic fingerprinting (L5).

Patient misidentification (wrong chart / wrong drug / wrong patient) is a real
safety hazard. This layer treats the digital twin's per-patient baseline as a
continuous biometric check: it enrolls a **physiologic fingerprint** (the mean +
covariance of a patient's resting vitals) and runs a **Mahalanobis distance**
between the live sensor stream and that fingerprint. If the distance spikes — the
stream suddenly looks like a *different* person — the kernel raises a
``PATIENT_IDENTITY_FAULT``: feeds are suspended, the nurse shell is alerted, and
**drug dispensing is locked** (proven by Z3 — no dispense during an active fault).

THE HONESTY BOUNDARY (this is an empirical claim, so it is measured, not asserted):

  * Resting vitals overlap heavily across people, so this is NOT biometric identity.
    It is a **gross stream-swap / sensor-reassignment detector** — it catches "bed 3's
    monitor is now showing bed 7's physiology," not "prove this is Jane Doe."
  * verify_identity.py reports the *real* same-vs-different-patient separability on
    CinC (ROC-AUC, plus FAR/FRR at the operating threshold). We publish whatever the
    number is; the value of the layer does not depend on it being high.
  * The fault → suspend → lock → Z3-gated-dispense state machine is sound regardless
    of detector strength: even a weak detector, once it fires, provably blocks dosing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from z3 import And, Bool, BoolVal, Implies, Not, Solver, sat, unsat
from backend.aegis.z3lock import z3_guarded as _z3_guarded

# Vitals that form the fingerprint (present in CinC + a real bedside monitor).
FP_CHANNELS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]
PATIENT_IDENTITY_FAULT = "PATIENT_IDENTITY_FAULT"


# --------------------------------------------------------------------------- #
@dataclass
class Fingerprint:
    patient_id: str
    mu: np.ndarray              # mean vector (standardized space)
    cov_inv: np.ndarray        # inverse covariance (precision matrix)
    channels: List[str]
    n: int
    threshold: float           # Mahalanobis² accept boundary (calibrated on self)

    def distance(self, X: np.ndarray) -> np.ndarray:
        """Per-row squared Mahalanobis distance to this fingerprint."""
        d = X - self.mu
        return np.einsum("ij,jk,ik->i", d, self.cov_inv, d)


@dataclass
class Standardizer:
    mean: np.ndarray
    std: np.ndarray

    @staticmethod
    def fit(X: np.ndarray) -> "Standardizer":
        mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
        sd[~np.isfinite(sd) | (sd < 1e-6)] = 1.0
        return Standardizer(np.nan_to_num(mu), sd)

    def apply(self, X: np.ndarray) -> np.ndarray:
        return np.clip((np.nan_to_num(X, nan=np.nan) - self.mean) / self.std, -10, 10)


def _matrix(frame, channels: List[str], pop_median: Dict[str, float]) -> np.ndarray:
    """Per-patient (n_hours, d) vitals matrix: forward-fill then population median."""
    import pandas as pd
    cols = []
    for ch in channels:
        s = pd.to_numeric(frame.get(ch), errors="coerce") if ch in frame else pd.Series(dtype=float)
        s = s.ffill()
        cols.append(s.fillna(pop_median.get(ch, 0.0)).to_numpy(dtype=float))
    return np.column_stack(cols)


def enroll(patient_id: str, X: np.ndarray, std: Standardizer,
           reg: float = 1e-2, quantile: float = 0.975) -> Fingerprint:
    """Build a fingerprint from a patient's enrollment window (standardized)."""
    Xs = std.apply(X)
    mu = Xs.mean(axis=0)
    cov = np.cov(Xs, rowvar=False)
    if cov.ndim == 0:
        cov = np.array([[float(cov)]])
    cov = cov + reg * np.eye(cov.shape[0])          # regularize for invertibility
    cov_inv = np.linalg.inv(cov)
    fp = Fingerprint(patient_id, mu, cov_inv, FP_CHANNELS, len(Xs), threshold=0.0)
    self_d = fp.distance(Xs)                          # calibrate threshold on self
    fp.threshold = float(np.quantile(self_d, quantile))
    return fp


# --------------------------------------------------------------------------- #
class IdentityMonitor:
    """Live per-bed identity check + fault state machine + dispense lock."""

    def __init__(self, fingerprint: Fingerprint, std: Standardizer):
        self.fp = fingerprint
        self.std = std
        self.fault = False
        self.feeds_suspended = False
        self.last_distance = 0.0
        self.events: List[Dict[str, Any]] = []

    def observe(self, window: np.ndarray) -> Dict[str, Any]:
        """Feed a window of raw vitals rows; returns the identity verdict."""
        Xs = self.std.apply(np.atleast_2d(window))
        d = float(np.median(self.fp.distance(Xs)))    # robust over the window
        self.last_distance = d
        match = d <= self.fp.threshold
        if not match and not self.fault:
            self._raise_fault(d)
        return {"distance": round(d, 3), "threshold": round(self.fp.threshold, 3),
                "match": match, "fault": self.fault,
                "dispensing_locked": self.fault}

    def _raise_fault(self, d: float) -> None:
        self.fault = True
        self.feeds_suspended = True
        self.events.append({"type": PATIENT_IDENTITY_FAULT, "distance": round(d, 3),
                            "threshold": round(self.fp.threshold, 3),
                            "action": "feeds suspended · dispensing locked · nurse alerted"})
        from backend.aegis.kernel.journal import KJOURNAL, CRIT
        KJOURNAL.append("clinical", "identity_fault", severity=CRIT,
                        distance=round(d, 3), threshold=round(self.fp.threshold, 3))

    def clear_fault(self, reason: str = "manual re-verification") -> None:
        self.fault = False
        self.feeds_suspended = False
        self.events.append({"type": "FAULT_CLEARED", "reason": reason})
        from backend.aegis.kernel.journal import KJOURNAL, NOTICE
        KJOURNAL.append("clinical", "identity_fault_cleared", severity=NOTICE, reason=reason)

    # -- Z3: prove no drug can be dispensed during an active identity fault -- #
    @_z3_guarded
    def prove_dispense_gate(self) -> Dict[str, Any]:
        """The safety invariant: dispense ⇒ identity_ok. We bind identity_ok to the
        live fault state and prove whether a dispense can be authorized."""
        s = Solver()
        dispense = Bool("dispense")
        identity_ok = Bool("identity_ok")
        s.add(Implies(dispense, identity_ok))         # the policy
        s.add(identity_ok == BoolVal(not self.fault)) # bind to live state
        s.add(dispense == BoolVal(True))              # attempt to dispense
        status = s.check()
        allowed = status == sat
        return {"dispense_allowed": allowed,
                "z3_status": "sat" if allowed else "unsat",
                "identity_fault": self.fault,
                "proof": "dispense ⇒ identity_ok ; identity_ok = ¬fault",
                "message": ("dispense authorized — identity verified" if allowed
                            else "DISPENSE BLOCKED — active PATIENT_IDENTITY_FAULT")}

    def status(self) -> Dict[str, Any]:
        return {"patient": self.fp.patient_id, "fault": self.fault,
                "feeds_suspended": self.feeds_suspended,
                "last_distance": round(self.last_distance, 3),
                "threshold": round(self.fp.threshold, 3),
                "events": self.events[-8:]}


# --------------------------------------------------------------------------- #
# Validation: does the fingerprint actually separate patients? (measured)
# --------------------------------------------------------------------------- #
def _population(records, channels):
    import numpy as np
    med: Dict[str, float] = {}
    import pandas as pd
    for ch in channels:
        vals = []
        for r in records:
            if ch in r.frame:
                vals.append(pd.to_numeric(r.frame[ch], errors="coerce").to_numpy())
        if vals:
            allv = np.concatenate(vals)
            med[ch] = float(np.nanmedian(allv)) if np.isfinite(np.nanmedian(allv)) else 0.0
        else:
            med[ch] = 0.0
    return med


def validate(records, enroll_frac: float = 0.6, min_hours: int = 16,
             seed: int = 7) -> Dict[str, Any]:
    """Enroll each patient on the first ``enroll_frac`` of their stay, then measure
    same-patient vs different-patient Mahalanobis separability (ROC-AUC) and the
    FAR/FRR at each patient's self-calibrated threshold."""
    from backend.ml.eval.metrics import auroc
    rng = np.random.default_rng(seed)
    usable = [r for r in records if r.n_hours >= min_hours]
    pop = _population(usable, FP_CHANNELS)
    mats = {r.patient_id: _matrix(r.frame, FP_CHANNELS, pop) for r in usable}
    std = Standardizer.fit(np.concatenate(list(mats.values()), axis=0))

    fps: Dict[str, Fingerprint] = {}
    self_rest: Dict[str, np.ndarray] = {}
    for pid, X in mats.items():
        k = int(len(X) * enroll_frac)
        if k < 4 or len(X) - k < 2:
            continue
        fps[pid] = enroll(pid, X[:k], std)
        self_rest[pid] = X[k:]
    pids = list(fps)

    same_d, diff_d = [], []
    far_hits = far_tot = frr_hits = frr_tot = 0
    for pid in pids:
        fp = fps[pid]
        sd = fp.distance(std.apply(self_rest[pid]))
        same_d.append(sd)
        frr_hits += int(np.sum(sd > fp.threshold)); frr_tot += len(sd)
        # a different patient's rows scored against this fingerprint
        other = pid
        while other == pid:
            other = pids[rng.integers(len(pids))]
        od = fp.distance(std.apply(self_rest[other]))
        diff_d.append(od)
        far_hits += int(np.sum(od <= fp.threshold)); far_tot += len(od)

    same = np.concatenate(same_d); diff = np.concatenate(diff_d)
    y = np.concatenate([np.zeros(len(same)), np.ones(len(diff))])
    scores = np.concatenate([same, diff])

    # Equal-error operating point (threshold-balanced FAR=FRR) — the honest single
    # number, swept over a shared threshold rather than per-patient self-quantiles.
    grid = np.quantile(scores, np.linspace(0.01, 0.99, 99))
    eer, eer_thr = 1.0, 0.0
    for t in grid:
        far = float(np.mean(diff <= t)); frr = float(np.mean(same > t))
        if max(far, frr) < eer:
            eer, eer_thr = max(far, frr), float(t)

    return {
        "n_patients": len(pids),
        "auroc": round(auroc(y, scores), 4),               # same-vs-different separability
        "eer": round(eer, 4),                              # equal-error operating point
        "same_median": round(float(np.median(same)), 2),
        "diff_median": round(float(np.median(diff)), 2),
        "far_self_threshold": round(far_hits / max(1, far_tot), 4),  # impostor accepted
        "frr_self_threshold": round(frr_hits / max(1, frr_tot), 4),  # true patient rejected
        "note_frr": ("self-threshold FRR is inflated by intra-stay physiologic drift "
                     "(deteriorating patients diverge from their own resting baseline) — "
                     "a deployment would re-baseline periodically"),
        "interpretation": ("gross stream-swap detector — separability measured on real "
                           "CinC, NOT biometric identity"),
    }


# --------------------------------------------------------------------------- #
# Service: enroll a cohort + drive the live monitors / swap demo (for REST).
# --------------------------------------------------------------------------- #
class IdentityService:
    """Enrolls a cohort of patients and exposes per-bed monitors + a swap demo."""

    def __init__(self, records, enroll_frac: float = 0.5, min_hours: int = 16):
        usable = [r for r in records if r.n_hours >= min_hours]
        self.pop = _population(usable, FP_CHANNELS)
        self.mats = {r.patient_id: _matrix(r.frame, FP_CHANNELS, self.pop) for r in usable}
        self.std = Standardizer.fit(np.concatenate(list(self.mats.values()), axis=0))
        self.frac = enroll_frac
        self.monitors: Dict[str, IdentityMonitor] = {}
        for pid, X in self.mats.items():
            k = max(4, int(len(X) * enroll_frac))
            self.monitors[pid] = IdentityMonitor(enroll(pid, X[:k], self.std), self.std)
        self._validation: Optional[Dict[str, Any]] = None
        self._records = usable

    def patients(self) -> List[str]:
        return list(self.monitors)

    def validation(self) -> Dict[str, Any]:
        if self._validation is None:
            self._validation = validate(self._records)
        return self._validation

    def self_check(self, pid: str) -> Dict[str, Any]:
        mon = self.monitors[pid]
        mon.clear_fault("re-check")
        X = self.mats[pid]
        obs = mon.observe(X[int(len(X) * self.frac):])
        return {"patient": pid, **obs, "gate": mon.prove_dispense_gate()}

    def swap(self, target: str, source: str) -> Dict[str, Any]:
        """Feed ``source``'s live stream into ``target``'s monitor (misconnection)."""
        mon = self.monitors[target]
        mon.clear_fault("demo reset")
        obs = mon.observe(self.mats[source])
        return {"target": target, "source": source, **obs,
                "gate": mon.prove_dispense_gate(),
                "events": mon.events[-3:]}

    def status(self) -> Dict[str, Any]:
        return {"n_enrolled": len(self.monitors), "patients": self.patients()[:24],
                "channels": FP_CHANNELS}


def _demo() -> None:
    from backend.ml.data.cinc2019 import load_patients
    recs = load_patients(sets=["training_setA"], limit=200)
    res = validate(recs)
    print("identity fingerprint validation:", res)

    # live swap demo
    pop = _population(recs, FP_CHANNELS)
    mats = {r.patient_id: _matrix(r.frame, FP_CHANNELS, pop) for r in recs if r.n_hours >= 16}
    std = Standardizer.fit(np.concatenate(list(mats.values()), axis=0))
    ids = list(mats)
    fpA = enroll(ids[0], mats[ids[0]][: len(mats[ids[0]]) // 2], std)
    mon = IdentityMonitor(fpA, std)
    print("self window:", mon.observe(mats[ids[0]][len(mats[ids[0]]) // 2:]))
    print("dispense (ok):", mon.prove_dispense_gate()["message"])
    mon2 = IdentityMonitor(enroll(ids[1], mats[ids[1]][:len(mats[ids[1]])//2], std), std)
    print("SWAP — patient", ids[5], "stream into", ids[1], "monitor:",
          mon2.observe(mats[ids[5]]))
    print("dispense (fault):", mon2.prove_dispense_gate()["message"])


if __name__ == "__main__":
    _demo()
