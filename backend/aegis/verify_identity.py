"""Verify harness for physiologic identity fingerprinting — claims asserted by measurement.

  1. SEPARABILITY — on real CinC, same-vs-different-patient Mahalanobis is genuinely
                    separable (ROC-AUC reported, with a balanced equal-error point).
                    Reported honestly: this is a stream-swap detector, not biometrics.
  2. SWAP DETECT  — a patient's own later window matches their fingerprint; feeding a
                    DIFFERENT patient's stream spikes the distance and raises
                    PATIENT_IDENTITY_FAULT (feeds suspended).
  3. FAULT → LOCK — Z3 proves no drug can be dispensed during an active fault (unsat),
                    and dispensing resumes once the fault is cleared (sat).
  4. HONESTY      — the validation payload carries AUROC + EER + self-threshold
                    FAR/FRR + the intra-stay-drift caveat.

    python -m backend.aegis.verify_identity
"""
from __future__ import annotations

import numpy as np

from backend.aegis.identity import (FP_CHANNELS, IdentityMonitor, Standardizer,
                                    enroll, validate, _matrix, _population,
                                    PATIENT_IDENTITY_FAULT)
from backend.ml.data.cinc2019 import load_patients


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


def main() -> int:
    failures = 0
    checks = []
    recs = load_patients(sets=["training_setA"], limit=180)

    # ---- 1. SEPARABILITY ------------------------------------------------- #
    _hr("1. SEPARABILITY (real CinC — measured, not asserted)")
    v = validate(recs)
    print(f"    n={v['n_patients']} AUROC={v['auroc']} EER={v['eer']}")
    print(f"    same-median={v['same_median']}  diff-median={v['diff_median']}")
    print(f"    self-threshold FAR={v['far_self_threshold']} FRR={v['frr_self_threshold']}")
    checks.append(("same-vs-different-patient AUROC is clearly separable (>0.8)",
                   v["auroc"] > 0.8))
    checks.append(("different-patient distance >> same-patient distance",
                   v["diff_median"] > 2 * v["same_median"]))
    checks.append(("equal-error operating point is usable (<0.3)", v["eer"] < 0.3))
    checks.append(("intra-stay-drift caveat is disclosed", "drift" in v["note_frr"]))

    # ---- build live monitors -------------------------------------------- #
    usable = [r for r in recs if r.n_hours >= 16]
    pop = _population(usable, FP_CHANNELS)
    mats = {r.patient_id: _matrix(r.frame, FP_CHANNELS, pop) for r in usable}
    std = Standardizer.fit(np.concatenate(list(mats.values()), axis=0))
    ids = list(mats)
    a, b = ids[0], ids[1]
    halfA = len(mats[a]) // 2
    monA = IdentityMonitor(enroll(a, mats[a][:halfA], std), std)

    # ---- 2. SWAP DETECTION ---------------------------------------------- #
    _hr("2. SWAP DETECTION")
    self_obs = monA.observe(mats[a][halfA:])
    print(f"    self window:  distance={self_obs['distance']} thr={self_obs['threshold']} "
          f"match={self_obs['match']} fault={self_obs['fault']}")
    checks.append(("a patient's own later window does NOT raise a fault",
                   self_obs["fault"] is False))
    # now feed a different patient's stream into A's monitor
    swap_obs = monA.observe(mats[b])
    print(f"    swapped {b} → {a}'s monitor: distance={swap_obs['distance']} "
          f"fault={swap_obs['fault']} feeds_suspended={monA.feeds_suspended}")
    checks.append(("a different patient's stream spikes the Mahalanobis distance",
                   swap_obs["distance"] > self_obs["threshold"]))
    checks.append(("PATIENT_IDENTITY_FAULT is raised + feeds suspended",
                   monA.fault and monA.feeds_suspended
                   and any(e["type"] == PATIENT_IDENTITY_FAULT for e in monA.events)))

    # ---- 3. FAULT → LOCK (Z3) ------------------------------------------- #
    _hr("3. DISPENSE LOCK (Z3-gated)")
    gate_fault = monA.prove_dispense_gate()
    print(f"    during fault: {gate_fault['message']} (z3 {gate_fault['z3_status']})")
    checks.append(("Z3 proves dispensing is BLOCKED during a fault",
                   gate_fault["dispense_allowed"] is False and gate_fault["z3_status"] == "unsat"))
    monA.clear_fault("re-verified")
    gate_ok = monA.prove_dispense_gate()
    print(f"    after clear: {gate_ok['message']} (z3 {gate_ok['z3_status']})")
    checks.append(("dispensing resumes once identity is re-verified",
                   gate_ok["dispense_allowed"] is True and gate_ok["z3_status"] == "sat"))

    # a verified patient (a matching observation, no fault) authorizes dispensing
    halfB = len(mats[b]) // 2
    monB = IdentityMonitor(enroll(b, mats[b][:halfB], std), std)
    obsB = monB.observe(mats[b][:halfB])          # baseline-consistent window → match
    checks.append(("a verified (matching) observation raises no fault",
                   obsB["fault"] is False and obsB["match"] is True))
    checks.append(("a verified patient can be dispensed to",
                   monB.prove_dispense_gate()["dispense_allowed"] is True))

    # ---- tally ----------------------------------------------------------- #
    _hr("VERIFYING THE GUARANTEE")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1

    _hr("RESULT")
    print("  ALL CHECKS PASSED ✓  — a continuous, measured identity check that "
          "provably locks dosing the moment a patient's stream stops matching."
          if failures == 0 else f"  {failures} CHECK(S) FAILED ✗")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
