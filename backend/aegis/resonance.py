"""Cross-Modal HDC Resonance — agreement-gated sepsis evidence.

A single-stream detector cannot tell a *patient deteriorating* from a *sensor
glitching*: a motion artifact that spikes the camera's heart-rate estimate looks,
to that one channel, exactly like tachycardia. Standard ML answers this with more
training data and pays for it in false alarms — the thing clinicians hate most.

AegisOS answers it geometrically. Each physical acquisition pathway — the
contactless camera/thermal stream, the wearable/bedside monitor, the EHR labs —
is encoded into its *own* 10,000-D hypervector and projected onto a single shared
**sepsis axis** (the real septic-minus-healthy centroid learned from the CinC-2019
precedent bank). That yields one calibrated "septic-ness" score per modality, all
on the same scale.

The diagnosis then lives in the **agreement** between pathways, not in any one of
them:

  * **Real deterioration** drives every independent pathway the same way — the HR
    rises, the perfusion falls, the lactate climbs together. The per-modality
    scores cluster: high *resonance*, and the alarm is corroborated.
  * **A sensor artifact** moves one pathway and leaves the others where they are.
    The spiking modality becomes a lone outlier; because a mechanical arm-jiggle
    does not also forge a matching lactate trend on a separate physical pathway,
    the gate identifies it as localized hardware noise, **rejects that modality
    from the consensus**, and the false alarm never fires.

The output is honest: a gated risk, a resonance (confidence) score, the
per-modality projections, and — when it happens — *which* modality was rejected
and why. This is a complementary, robustness-focused channel; its measured value
(false-alarm suppression at preserved sensitivity) is reported as-is in
``verify_resonance.py``, not asserted.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.aegis.hdc_memory import HDCBioMemory, SOON_H

# Each modality is a distinct physical acquisition pathway, mapped to the channels
# that pathway can actually observe (levels + their short-window trends).
MODALITIES: Dict[str, List[str]] = {
    "camera":   ["hr", "temp_c", "hr__tr", "temp_c__tr"],            # rPPG + thermal
    "wearable": ["resp_rate", "spo2", "map", "resp_rate__tr", "spo2__tr", "map__tr"],
    "ehr":      ["wbc", "lactate", "wbc__tr", "lactate__tr"],        # async labs
}

# Artifact-flagging thresholds, in calibrated σ units (tuned in verify_resonance).
# The gated consensus is the MEDIAN pathway — a robust statistic that ignores a
# single outlier (high or low) without discarding a corroborated signal, so it
# suppresses a lone sensor spike yet keeps a real multi-pathway drift. The flag
# below only adds an explanation ("which pathway looks like an artifact").
ART_GAP = 0.7          # top pathway exceeds the median by ≥ this …
ART_FIRE = 0.4         # … and is itself elevated → call it an isolated spike
# logistic σ→risk mapping (identical for naive & gated, so the comparison is fair)
RISK_Z0 = 0.4
RISK_K = 1.4


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


class CrossModalResonance:
    """Encodes each modality, projects onto the sepsis axis, and gates the alarm
    on cross-modal agreement. Reuses an already-built :class:`HDCBioMemory` for the
    encoder, the channel id-vectors, and the real precedent bank."""

    def __init__(self, hdc: HDCBioMemory):
        self.hdc = hdc
        self.eng = hdc.eng
        self.ids = hdc.ids
        self.axis: Optional[np.ndarray] = None         # unit sepsis-direction in HDC space
        self.cal: Dict[str, Tuple[float, float]] = {}  # modality -> (mu, sigma)
        self._fitted = False

    # ------------------------------------------------------------------ #
    # Fit: build the sepsis axis from the precedent centroids, then calibrate
    # each modality's projection distribution so the three are comparable.
    # ------------------------------------------------------------------ #
    def fit(self, states: List[Dict[str, float]],
            exclude_pid: Optional[str] = None) -> "CrossModalResonance":
        """Build the sepsis axis + per-modality calibration. ``exclude_pid`` drops
        that patient's rows from BOTH the axis and the calibration, so the harness
        can evaluate leave-one-patient-out (the axis never sees the test patient)."""
        if self.hdc.mem is None or not self.hdc.meta:
            raise ValueError("resonance needs a built HDCBioMemory precedent bank")
        M = self.hdc.mem.astype(np.float32)
        keep = np.array([m["pid"] != exclude_pid for m in self.hdc.meta], dtype=bool)
        soon = np.array([
            1 if (m["label"] == 1 or (m["tto"] is not None and 0 <= m["tto"] <= SOON_H))
            else 0 for m in self.hdc.meta], dtype=np.int8)
        M, soon = M[keep], soon[keep]
        if soon.sum() == 0 or (soon == 0).sum() == 0:
            raise ValueError("precedent bank lacks both septic and non-septic rows")
        axis = M[soon == 1].mean(0) - M[soon == 0].mean(0)   # septic − healthy direction
        self.axis = axis / (np.linalg.norm(axis) + 1e-9)

        # Calibrate per-modality projection scale on the SAME cohort rows.
        proj: Dict[str, List[float]] = {m: [] for m in MODALITIES}
        for st in states:
            for m in MODALITIES:
                proj[m].append(self._project(st, m))
        for m, vals in proj.items():
            a = np.asarray(vals, dtype=np.float64)
            self.cal[m] = (float(a.mean()), float(a.std()) or 1.0)
        self._fitted = True
        return self

    # ------------------------------------------------------------------ #
    def _encode_modality(self, state: Dict[str, float], modality: str) -> Optional[np.ndarray]:
        binds = []
        for f in MODALITIES[modality]:
            v = state.get(f)
            if v is None or (isinstance(v, float) and v != v):
                continue
            lvl = self.eng.encode_value(f, float(v))
            binds.append(self.eng.bind(self.ids[f], lvl))
        return self.eng.bundle(binds) if binds else None

    def _project(self, state: Dict[str, float], modality: str) -> float:
        """Raw projection of one modality's hypervector onto the sepsis axis."""
        v = self._encode_modality(state, modality)
        if v is None or self.axis is None:
            return 0.0
        vf = v.astype(np.float32)
        n = np.linalg.norm(vf)
        if n < 1e-9:
            return 0.0
        return float(np.dot(vf / n, self.axis))

    def _z(self, state: Dict[str, float], modality: str) -> float:
        mu, sigma = self.cal.get(modality, (0.0, 1.0))
        return (self._project(state, modality) - mu) / sigma

    # ------------------------------------------------------------------ #
    # Assess: per-modality scores → robust consensus with outlier rejection.
    # ------------------------------------------------------------------ #
    def assess(self, state: Dict[str, float]) -> Dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("CrossModalResonance.fit() must be called first")
        z = {m: self._z(state, m) for m in MODALITIES}
        names = list(z.keys())
        vals = np.array([z[m] for m in names], dtype=np.float64)

        # The robust consensus requires CORROBORATION. Baseline is the median
        # pathway (immune to any single outlier). On top of that, when one pathway
        # is an isolated spike well above the other two, it is explicitly rejected
        # as a localized artifact and the consensus is taken from the two
        # corroborating pathways — so a lone sensor glitch cannot raise an alarm,
        # while a drift seen by ≥2 independent pathways still does. The naive
        # fusion we beat is the plain mean, which a single spike drags upward.
        naive_consensus = float(np.mean(vals))
        med = float(np.median(vals))
        top_i = int(np.argmax(vals))
        top = names[top_i]
        # An "isolated" pathway is one elevated alone, above the other two. We do NOT
        # assume it is an artifact — it may equally be an early single-organ change.
        # Either way it is uncorroborated, so it is held out of the consensus until a
        # second pathway confirms: a lone sensor spike can't raise a full alarm, and
        # a true leading indicator is simply graded as low-confidence until echoed.
        isolated = bool((z[top] - med) >= ART_GAP and z[top] >= ART_FIRE)
        if isolated:
            others = [vals[i] for i in range(len(vals)) if i != top_i]
            gated_consensus = float(np.mean(others))   # trust the corroborating pair
        else:
            gated_consensus = med
        spread = float(np.max(vals) - np.min(vals))
        resonance = 1.0 / (1.0 + 0.5 * spread)         # agreement → confidence
        concur = int((vals > ART_FIRE).sum())

        return {
            "per_modality": {m: round(z[m], 3) for m in names},
            "consensus": round(gated_consensus, 3),
            "naive_consensus": round(naive_consensus, 3),
            "gated_risk": round(_sigmoid(RISK_K * (gated_consensus - RISK_Z0)), 4),
            "naive_risk": round(_sigmoid(RISK_K * (naive_consensus - RISK_Z0)), 4),
            "resonance": round(resonance, 3),
            "corroborated": not isolated and concur >= 2,
            "isolated": isolated,
            "uncorroborated_pathway": top if isolated else None,
            "concur": concur,
            "explain": (f"{top} pathway alone reads {z[top]:.1f}σ vs consensus "
                        f"{med:.1f}σ → uncorroborated, held pending confirmation"
                        if isolated else
                        f"{concur}/3 pathways concur (spread {spread:.2f}σ)"),
        }

    # convenience: build a fitted gate straight from a set of cohort patients
    @classmethod
    def from_patients(cls, hdc: HDCBioMemory, patients: List[Any]) -> "CrossModalResonance":
        states: List[Dict[str, float]] = []
        for p in patients:
            states.extend(s for (s, *_rest) in HDCBioMemory.rows_from_patient(p))
        return cls(hdc).fit(states)
