"""Comprehensive inventory of camera-observable signs of sepsis.

The goal here is completeness: enumerate *every* sign of sepsis a camera can
plausibly observe, give each its own detector, confidence, severity, and
calibrated evidence, and tag it honestly as validated / proxy / experimental.
Together they form a structured, explainable "what the camera can see" panel that
also feeds the sequential monitor as extra evidence channels.

Sign categories (Sepsis-3 + bedside examination, restricted to what optics see):
  • Cardiovascular   — tachycardia, HRV collapse
  • Respiratory      — tachypnea, laboured/irregular breathing
  • Perfusion/skin   — poor perfusion, pallor, mottling, cyanosis, flush,
                       jaundice, diaphoresis
  • Autonomic        — rigors / micro-tremor, pupillary instability
  • Neurologic (AMS) — reduced responsiveness, gaze instability, drowsiness,
                       hypomobility  (altered mental status = a qSOFA criterion)

Validation honesty: HR/RR/HRV are signal-validated (see signal_processing + the
BIDMC PPG-vs-ECG study). The colour-, expression-, and motion-derived signs are
clinically motivated PROXIES/EXPERIMENTAL and are labelled as such — they
contribute *weighted* evidence, never an unearned certainty.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

import numpy as np


@dataclass
class SignReading:
    name: str
    category: str
    present: bool
    observable: bool                 # was the required signal available?
    severity: float                  # 0..1
    confidence: float                # 0..1 (gated by signal quality)
    value: Optional[float]
    evidence: str
    grade: str                       # "validated" | "proxy" | "experimental"
    log_odds: float                  # weighted evidence contribution
    core: bool = False               # already counted by CameraSepsisDetector?

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "category": self.category, "present": self.present,
                "observable": self.observable, "severity": round(self.severity, 3),
                "confidence": round(self.confidence, 3),
                "value": (round(self.value, 3) if self.value is not None else None),
                "evidence": self.evidence, "grade": self.grade,
                "log_odds": round(self.log_odds, 4)}


def _clip(x, lo=0.0, hi=1.0):
    return float(max(lo, min(hi, x)))


# --------------------------------------------------------------------------- #
# Skin-colour feature extraction (drives the integument signs)
# --------------------------------------------------------------------------- #
def skin_color_features(region_rgb: Dict[str, np.ndarray]) -> Dict[str, float]:
    """Derive colour-based skin signs from mean RGB per face region.

    region_rgb: {region_name: (3,) mean RGB}. Returns normalized indices in 0..1
    where higher = more abnormal in the sign's direction.
    """
    regs = [np.asarray(v, dtype=np.float64) for v in region_rgb.values() if v is not None and np.size(v) == 3]
    if not regs:
        return {}
    arr = np.stack(regs)
    mean = arr.mean(axis=0)
    s = mean.sum() or 1.0
    r, g, b = mean / s
    feats = {
        "redness": float(r),                              # flush / fever
        "pallor": float(_clip((0.36 - r) / 0.12)),        # low red fraction => pale
        "yellowness": float(_clip(((r + g) / 2 - b - 0.04) / 0.12)),   # jaundice
        "mottling": float(_clip(np.std(arr / arr.sum(axis=1, keepdims=True), axis=0).mean() / 0.05)),
        "brightness": float(mean.mean() / 255.0),
    }
    lips = region_rgb.get("lips")
    if lips is not None and np.size(lips) == 3:
        lp = np.asarray(lips, dtype=np.float64); ls = lp.sum() or 1.0
        feats["cyanosis"] = float(_clip((lp[2] / ls - 0.34) / 0.12))   # blue lip fraction
    return feats


# --------------------------------------------------------------------------- #
# Sign detectors — each takes the flat feature dict + a quality, returns a Sign
# --------------------------------------------------------------------------- #
def _sig(name, cat, grade, present, observable, severity, q, value, evidence, w, core=False):
    sev = _clip(severity)
    conf = _clip(q) * (1.0 if observable else 0.0)
    return SignReading(name, cat, bool(present and observable), bool(observable), sev, conf,
                       value, evidence, grade, w * sev * conf, core)


def _has(f, k):
    v = f.get(k)
    return v is not None and not (isinstance(v, float) and np.isnan(v))


def assess_signs(f: Dict[str, float], quality: float = 1.0) -> List[SignReading]:
    """Score the full observable-sign inventory from a flat feature dict."""
    q = _clip(quality)
    hr_base = f.get("hr_baseline", 75.0) or 75.0
    sd_base = f.get("hrv_sdnn_baseline", 50.0) or 50.0
    out: List[SignReading] = []

    # --- Cardiovascular (core vitals — already in CameraSepsisDetector) ---- #
    if _has(f, "heart_rate"):
        sev = _clip((f["heart_rate"] - hr_base * 1.12) / (hr_base * 0.5))
        out.append(_sig("Tachycardia", "cardiovascular", "validated", sev > 0.05, True, sev, q,
                        f["heart_rate"], f"HR {f['heart_rate']:.0f} bpm vs baseline {hr_base:.0f}", 0.40, core=True))
    if _has(f, "hrv_sdnn"):
        sev = _clip((0.8 - f["hrv_sdnn"] / max(sd_base, 1e-6)) / 0.45)
        out.append(_sig("HRV collapse", "cardiovascular", "validated", sev > 0.05, True, sev, q,
                        f["hrv_sdnn"], f"SDNN {f['hrv_sdnn']:.0f} ms (HeRO autonomic signature)", 0.55, core=True))

    # --- Respiratory ------------------------------------------------------- #
    if _has(f, "respiratory_rate"):
        sev = _clip((f["respiratory_rate"] - 20.0) / 10.0)
        out.append(_sig("Tachypnea", "respiratory", "validated", sev > 0.05, True, sev, q,
                        f["respiratory_rate"], f"RR {f['respiratory_rate']:.0f}/min", 0.35, core=True))
    if _has(f, "resp_irregularity"):
        sev = _clip(f["resp_irregularity"])
        out.append(_sig("Laboured/irregular breathing", "respiratory", "experimental", sev > 0.3, True, sev,
                        q * 0.7, sev, "irregular respiratory envelope", 0.18))

    # --- Perfusion / integument (skin) ------------------------------------- #
    if _has(f, "perfusion_index"):
        sev = _clip((0.6 - f["perfusion_index"]) / 0.4)
        out.append(_sig("Poor peripheral perfusion", "perfusion", "proxy", sev > 0.1, True, sev, q,
                        f["perfusion_index"], "reduced pulsatile perfusion (possible shunting)", 0.40, core=True))
    if _has(f, "pallor"):
        out.append(_sig("Pallor", "perfusion", "proxy", f["pallor"] > 0.3, True, f["pallor"], q * 0.8,
                        f["pallor"], "reduced facial redness", 0.20))
    if _has(f, "mottling"):
        out.append(_sig("Skin mottling", "perfusion", "experimental", f["mottling"] > 0.4, True, f["mottling"],
                        q * 0.6, f["mottling"], "spatial skin-colour heterogeneity", 0.18))
    if _has(f, "cyanosis"):
        out.append(_sig("Cyanosis (lips)", "perfusion", "proxy", f["cyanosis"] > 0.3, True, f["cyanosis"],
                        q * 0.7, f["cyanosis"], "bluish lip discolouration (low SpO₂ proxy)", 0.30))
    if _has(f, "redness") and _has(f, "skin_warmth"):
        sev = _clip((f["redness"] - 0.37) / 0.06) * _clip(f.get("skin_warmth", 0))
        out.append(_sig("Flushing / pyrexia", "integument", "proxy", sev > 0.3, True, sev, q * 0.7,
                        sev, "facial flushing with warmth", 0.18))
    if _has(f, "yellowness"):
        out.append(_sig("Jaundice", "integument", "experimental", f["yellowness"] > 0.4, True, f["yellowness"],
                        q * 0.5, f["yellowness"], "yellow skin/scleral tint (hepatic)", 0.15))
    if _has(f, "diaphoresis"):
        out.append(_sig("Diaphoresis", "autonomic", "experimental", f["diaphoresis"] > 0.4, True, f["diaphoresis"],
                        q * 0.6, f["diaphoresis"], "skin sheen / specular highlights (sweating)", 0.15))

    # --- Autonomic --------------------------------------------------------- #
    if _has(f, "tremor_power"):
        out.append(_sig("Rigors / micro-tremor", "autonomic", "experimental", f["tremor_power"] > 0.45, True,
                        f["tremor_power"], q * 0.7, f["tremor_power"], "4–12 Hz facial micro-tremor", 0.18))
    if _has(f, "pupil_instability"):
        sev = _clip(f["pupil_instability"] / 0.15)
        out.append(_sig("Pupillary instability", "autonomic", "experimental", sev > 0.4, True, sev, q * 0.6,
                        f["pupil_instability"], "pupil-diameter variability (hippus)", 0.15))

    # --- Neurologic / altered mental status (qSOFA criterion) -------------- #
    if _has(f, "responsiveness"):
        sev = _clip(1.0 - f["responsiveness"])
        out.append(_sig("Reduced responsiveness", "neurologic", "experimental", sev > 0.4, True, sev, q * 0.6,
                        f["responsiveness"], "low facial activity / response", 0.30))
    if _has(f, "eye_openness"):
        sev = _clip((0.55 - f["eye_openness"]) / 0.3)
        out.append(_sig("Drowsiness / ptosis", "neurologic", "experimental", sev > 0.4, True, sev, q * 0.6,
                        f["eye_openness"], "reduced eye openness", 0.20))
    if _has(f, "movement_index"):
        sev = _clip((0.3 - f["movement_index"]) / 0.3)
        out.append(_sig("Lethargy / hypomobility", "neurologic", "experimental", sev > 0.4, True, sev, q * 0.5,
                        f["movement_index"], "reduced spontaneous movement", 0.18))
    return out


@dataclass
class SignInventory:
    signs: List[SignReading]

    @property
    def added_log_odds(self) -> float:
        """Evidence from signs NOT already counted by CameraSepsisDetector
        (avoids double-counting the core vitals)."""
        return float(sum(s.log_odds for s in self.signs if not s.core))

    @property
    def present(self) -> List[SignReading]:
        return [s for s in self.signs if s.present]

    def by_category(self) -> Dict[str, List[Dict[str, Any]]]:
        cats: Dict[str, List[Dict[str, Any]]] = {}
        for s in self.signs:
            cats.setdefault(s.category, []).append(s.to_dict())
        return cats

    def to_dict(self) -> Dict[str, Any]:
        return {
            "signs": [s.to_dict() for s in self.signs],
            "present_count": len(self.present),
            "observable_count": sum(1 for s in self.signs if s.observable),
            "added_log_odds": round(self.added_log_odds, 4),
            "by_category": self.by_category(),
        }


def build_inventory(features: Dict[str, float], quality: float = 1.0) -> SignInventory:
    return SignInventory(assess_signs(features, quality))
