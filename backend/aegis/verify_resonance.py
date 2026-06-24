"""Measured validation of the Cross-Modal Resonance gate — on REAL CinC-2019 data.

The claim under test is narrow and falsifiable: *at equal sensitivity, gating the
alarm on cross-modal agreement raises far fewer false alarms when one sensor is
corrupted by a motion artifact.*

Methodology (leave-one-patient-out, matched operating point — the rigorous way):
  1. For both detectors — naive fusion (mean of the three pathway scores) and the
     resonance gate (robust median) — pick the threshold that yields the SAME
     sensitivity (70%) on clean true-septic hours. Comparing false-alarm rates at
     equal sensitivity is a proper ROC operating-point comparison, not cherry-pick.
  2. A · inject a realistic rPPG lock-failure (the camera HR estimate jumps to a
     wildly wrong value) on clean non-septic hours and measure the false-alarm
     rate of each detector at its matched threshold.
  3. C · inject the same artifact during real sepsis and confirm the gate stays
     sensitive (the wearable + EHR pathways still carry the signal).

The sepsis axis never sees the test patient. Numbers are printed as-is; the
script fails if the gate does not beat naive fusion.

    python -m backend.aegis.verify_resonance
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np

from backend.aegis.cohort import build_cohort
from backend.aegis.hdc_memory import HDCBioMemory, SOON_H
from backend.aegis.resonance import CrossModalResonance

TARGET_SENS = 0.70         # matched operating sensitivity on clean septic hours
CLEAN_GAP = 12             # a "clean non-septic" hour is >this many h from any onset


def _is_soon(label: int, tto) -> bool:
    return label == 1 or (tto is not None and 0 <= tto <= SOON_H)


def _inject_camera_artifact(state: Dict[str, float], rng) -> Dict[str, float]:
    """A realistic rPPG lock failure: motion throws the camera HR estimate to a
    wildly wrong value (and its trend with it). The wearable and labs pathways —
    separate physical hardware — are untouched, exactly as in reality."""
    s = dict(state)
    bad_hr = float(rng.uniform(150.0, 195.0))
    s["hr__tr"] = bad_hr - s.get("hr", 80.0)
    s["hr"] = bad_hr
    s["temp_c"] = s.get("temp_c", 37.0) + float(rng.uniform(-0.3, 0.6))
    return s


def _thr_for_sens(scores: np.ndarray, sens: float) -> float:
    """Threshold that detects ``sens`` fraction of these (septic) scores."""
    return float(np.quantile(scores, 1.0 - sens))


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


def main() -> int:
    rng = np.random.default_rng(7)
    patients = build_cohort(n=12, seed=7)
    if not patients:
        print("CinC-2019 corpus unavailable — cannot run the real-data validation.")
        return 1

    rows = [r for p in patients for r in HDCBioMemory.rows_from_patient(p)]
    hdc = HDCBioMemory().build(rows)
    states_by_pid: Dict[str, List[Tuple[Dict[str, float], int, Any]]] = {}
    for st, label, tto, pid in rows:
        states_by_pid.setdefault(pid, []).append((st, label, tto))

    # pooled scores across the LOPO folds
    cs_naive, cs_gated = [], []          # clean septic-soon (for the operating point)
    n0_naive, n0_gated = [], []          # clean non-septic, NO artifact (baseline FA)
    n1_naive, n1_gated = [], []          # clean non-septic, WITH artifact
    as_gated = []                        # septic + artifact (robustness)
    localized = 0

    _hr("LEAVE-ONE-PATIENT-OUT EVALUATION  ·  12 real CinC-2019 patients")
    for held in patients:
        pid = held.bed_id
        train = [r[0] for p in patients if p.bed_id != pid
                 for r in HDCBioMemory.rows_from_patient(p)]
        gate = CrossModalResonance(hdc).fit(train, exclude_pid=pid)

        for st, label, tto in states_by_pid[pid]:
            soon = _is_soon(label, tto)
            far = (tto is None) or (tto > CLEAN_GAP)
            if soon:
                a = gate.assess(st)
                cs_naive.append(a["naive_consensus"]); cs_gated.append(a["consensus"])
                c = gate.assess(_inject_camera_artifact(st, rng))
                as_gated.append(c["consensus"])
            elif far:
                a0 = gate.assess(st)
                n0_naive.append(a0["naive_consensus"]); n0_gated.append(a0["consensus"])
                a1 = gate.assess(_inject_camera_artifact(st, rng))
                n1_naive.append(a1["naive_consensus"]); n1_gated.append(a1["consensus"])
                localized += int(a1["isolated"] and a1["uncorroborated_pathway"] == "camera")

    cs_naive, cs_gated = np.array(cs_naive), np.array(cs_gated)
    n0_naive, n0_gated = np.array(n0_naive), np.array(n0_gated)
    n1_naive, n1_gated = np.array(n1_naive), np.array(n1_gated)
    as_gated = np.array(as_gated)

    # matched operating point: same clean sensitivity for both detectors
    t_naive = _thr_for_sens(cs_naive, TARGET_SENS)
    t_gated = _thr_for_sens(cs_gated, TARGET_SENS)
    sens_naive = float((cs_naive > t_naive).mean())
    sens_gated = float((cs_gated > t_gated).mean())
    # baseline vs artifact false-alarm rates, and the ARTIFACT-INDUCED delta
    fa0_naive, fa1_naive = float((n0_naive > t_naive).mean()), float((n1_naive > t_naive).mean())
    fa0_gated, fa1_gated = float((n0_gated > t_gated).mean()), float((n1_gated > t_gated).mean())
    ind_naive, ind_gated = fa1_naive - fa0_naive, fa1_gated - fa0_gated
    rob_gated = float((as_gated > t_gated).mean())
    loc_rate = localized / max(1, len(n1_naive))

    _hr("MATCHED OPERATING POINT (both detectors fixed to ~70% clean sensitivity)")
    print(f"  clean septic hours n={len(cs_naive)} · "
          f"naive sens {sens_naive*100:.0f}% · gated sens {sens_gated*100:.0f}%")

    _hr("A · ARTIFACT-INDUCED FALSE ALARMS (the extra alarms the artifact causes)")
    print(f"  clean non-septic hours n={len(n0_naive)} · same threshold w/ and w/o artifact")
    print(f"  naive fusion (mean)    :  {fa0_naive*100:4.1f}% → {fa1_naive*100:4.1f}%   "
          f"(+{ind_naive*100:.1f} pts caused by the artifact)")
    print(f"  resonance gate (median):  {fa0_gated*100:4.1f}% → {fa1_gated*100:4.1f}%   "
          f"(+{ind_gated*100:.1f} pts caused by the artifact)")
    removed = 1 - (ind_gated / ind_naive) if ind_naive > 0 else 1.0
    print(f"  → the gate removes {removed*100:.0f}% of the artifact-induced false alarms")
    print(f"  artifact localized to the camera pathway: {localized}/{len(n1_naive)} "
          f"({loc_rate*100:.0f}%)")

    _hr("C · ROBUSTNESS — gate sensitivity holds with the artifact during real sepsis")
    print(f"  septic hours n={len(as_gated)} · gate detects {rob_gated*100:.0f}% "
          f"(clean was {sens_gated*100:.0f}%) — the wearable + EHR carry it")

    _hr("VERDICT")
    checks = [
        ("matched sensitivity (~70% both)",        abs(sens_naive - sens_gated) <= 0.06),
        ("gate removes >=80% of induced FAs",      ind_naive > 0 and ind_gated <= 0.20 * ind_naive),
        ("gate barely reacts to artifact (<=2.5pts)", ind_gated <= 0.025),
        ("gate sensitivity holds under artifact",  abs(rob_gated - sens_gated) <= 0.12),
    ]
    failures = 0
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1
    print("\n  RESONANCE GATE VALIDATED ✓" if failures == 0
          else f"\n  {failures} CHECK(S) FAILED — tune thresholds in resonance.py")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
