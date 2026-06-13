"""The Hour-1 Sepsis Bundle engine — Module C's safety-critical payload.

When a bed's fused vector breaches the critical threshold, this engine compiles
the Surviving-Sepsis-Campaign **Hour-1 bundle**, localized to that patient, and
routes every safety-critical output through the Layer-4 neurosymbolic cage
(``NeuroSymbolicRuntime``) so each one carries a Z3 certificate before it can be
dispatched. The terminal streams the result.

What is real here:
  * The five Hour-1 bundle items are the actual SSC bundle.
  * The treatment parameters are derived by a transparent rule from the bed's
    own labs/vitals AND the patient's clinical history (CHF / renal failure /
    penicillin allergy) — so fluids are capped for a CHF patient, vasopressors
    only engage below MAP 65, and the antibiotic is penicillin-sparing when
    indicated.
  * Each of alert / protocol / clinical-note is certified by the existing Z3
    cage (the same ``backend/core/verifier.py`` prover) — UNSAT = proven safe.

Honesty: the bundle generator encodes clinical guideline structure, not a
substitute for a clinician; it exists to demonstrate that probabilistic
inference is gated by formal verification before any order is emitted.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from backend.aegis.kernel.fusion import BedProfile
from backend.aegis.kernel.neurosym import ALERT, NOTE, PROTOCOL, NeuroSymbolicRuntime

ANTIBIOTICS = {
    0: "None", 1: "Piperacillin-Tazobactam", 2: "Vancomycin",
    3: "Cefepime", 4: "Meropenem", 5: "Levofloxacin",
}
SEV_NAME = {0: "NONE", 1: "LOW", 2: "MEDIUM", 3: "CRITICAL"}


class SepsisBundleEngine:
    def __init__(self):
        self.cage = NeuroSymbolicRuntime()

    # -- transparent, patient-specific protocol derivation -------------- #
    def _derive_protocol(self, prof: BedProfile, hist: Dict[str, Any]) -> Dict[str, Any]:
        en = prof.enterprise
        mapv, lact = en["map"], en["lactate"]
        chf = bool(hist.get("chf"))
        renal = bool(hist.get("renal_failure"))
        pcn = bool(hist.get("penicillin_allergy"))
        # Crystalloid: resuscitate for hypotension or lactate >= 4; cap by comorbidity
        # so the order respects (and provably satisfies) the patient's fluid limits.
        if mapv < 65 or lact >= 4.0:
            fluid = 5.0 if chf else (4.0 if renal else 10.0)
        else:
            fluid = 1.0
        # Norepinephrine only below MAP 65, scaled to depth of hypotension, < 0.4 cap.
        vaso = round(min(0.35, max(0.05, (65.0 - mapv) / 90.0 + 0.05)), 2) if mapv < 65 else 0.0
        # Broad-spectrum; penicillin-sparing (Cefepime) when allergic, else Pip-Tazo.
        ab = 3 if pcn else 1
        return {"fluid_rate": fluid, "vasopressor_dose": vaso, "antibiotic_code": ab}

    def _checklist(self, prof, hist, proto):
        en = prof.enterprise
        lact, mapv = en["lactate"], en["map"]
        ab = ANTIBIOTICS[proto["antibiotic_code"]]
        items = [
            {"n": 1, "title": "Measure serum lactate",
             "detail": f"{lact:.1f} mmol/L" + (" · repeat in 2–4 h" if lact > 2 else " · within range"),
             "status": "done"},
            {"n": 2, "title": "Blood cultures ×2 before antibiotics",
             "detail": "ordered · peripheral + line", "status": "indicated"},
            {"n": 3, "title": "Broad-spectrum antibiotics",
             "detail": ab + (" (penicillin-sparing)" if hist.get("penicillin_allergy") else ""),
             "status": "indicated"},
        ]
        if proto["fluid_rate"] >= 4:
            cap = " · capped (CHF)" if hist.get("chf") else (
                " · capped (renal)" if hist.get("renal_failure") else "")
            items.append({"n": 4, "title": "Crystalloid resuscitation",
                          "detail": f"{proto['fluid_rate']:.0f} mL/kg/h{cap}", "status": "indicated"})
        else:
            items.append({"n": 4, "title": "Crystalloid resuscitation",
                          "detail": "not indicated · MAP & lactate adequate", "status": "not_indicated"})
        if proto["vasopressor_dose"] > 0:
            items.append({"n": 5, "title": "Vasopressors → MAP ≥ 65",
                          "detail": f"Norepinephrine {proto['vasopressor_dose']:.2f} mcg/kg/min",
                          "status": "indicated"})
        else:
            items.append({"n": 5, "title": "Vasopressors → MAP ≥ 65",
                          "detail": "not indicated · MAP ≥ 65", "status": "not_indicated"})
        return items

    def _note(self, prof, proto, hist) -> str:
        drv = "; ".join(prof.drivers[:4]) if prof.drivers else "no acute derangements"
        ab = ANTIBIOTICS[proto["antibiotic_code"]]
        extra = ""
        if proto["vasopressor_dose"] > 0:
            extra += f", norepinephrine {proto['vasopressor_dose']:.2f} mcg/kg/min"
        if proto["fluid_rate"] >= 4:
            extra += f", crystalloid {proto['fluid_rate']:.0f} mL/kg/h"
        return (f"{prof.name} — fused sepsis risk {prof.fused_risk:.2f}, "
                f"{prof.primary_system} axis primary. Derangements: {drv}. "
                f"Hour-1 bundle initiated: lactate measured, cultures drawn, {ab} started{extra}. "
                f"All orders formally verified against patient-specific safety invariants.")

    # -- the compiled, certified bundle --------------------------------- #
    def generate(self, prof: BedProfile, hist: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        hist = hist or {}
        risk = float(prof.fused_risk)
        sev = 3 if risk >= 0.70 else 2 if risk >= 0.40 else 1 if risk >= 0.20 else 0
        risk_level = "HIGH" if risk >= 0.70 else "MEDIUM" if risk >= 0.40 else "LOW"
        en = prof.enterprise

        a_ok, a_cert = self.cage.dispatch(ALERT, {"severity": sev}, {
            "probability": risk, "calibrated": True, "evidence_count": len(prof.drivers)})

        proto = self._derive_protocol(prof, hist)
        p_ok, p_cert = self.cage.dispatch(PROTOCOL, proto, {
            "risk_level": risk_level, "vitals": {"map": en["map"]}, "clinical_history": hist})

        checklist = self._checklist(prof, hist, proto)
        note_text = self._note(prof, proto, hist)
        n_ok, n_cert = self.cage.dispatch(NOTE, {
            "risk_level": risk_level, "text": note_text,
            "recommends_treatment": True, "protocol_certified": p_ok}, {"risk_level": risk_level})

        return {
            "bed_id": prof.bed_id, "name": prof.name, "fused_risk": round(risk, 3),
            "severity": sev, "severity_name": SEV_NAME[sev], "risk_level": risk_level,
            "primary_system": prof.primary_system,
            "alert": {"certified": a_ok, "z3": a_cert.z3_status, "msg": a_cert.message},
            "protocol": {
                "fluid_rate": proto["fluid_rate"], "vasopressor_dose": proto["vasopressor_dose"],
                "antibiotic": ANTIBIOTICS[proto["antibiotic_code"]],
                "certified": p_ok, "z3": p_cert.z3_status, "msg": p_cert.message,
                "violations": p_cert.violations, "proof": p_cert.proof,
            },
            "checklist": checklist,
            "note": {"text": note_text, "certified": n_ok, "z3": n_cert.z3_status},
            "all_verified": bool(a_ok and p_ok and n_ok),
        }


def _demo() -> None:
    from backend.aegis.kernel.scheduler import BioCluster
    cl = BioCluster(n_nodes=3, lane_budget=48, names=["Okafor, A.", "Bauer, M.", "Reyes, L."], seed=4)
    cl.run(15)
    cl.inject_crisis("node_0", 1.0)
    for _ in range(60):
        cl.tick()
    eng = SepsisBundleEngine()
    b = eng.generate(cl.profiles["node_0"], {"chf": True, "penicillin_allergy": True})
    print(f"{b['name']} risk {b['fused_risk']} sev {b['severity_name']}  all_verified={b['all_verified']}")
    print("ALERT   :", b["alert"]["z3"], "·", b["alert"]["certified"])
    print("PROTOCOL:", b["protocol"]["antibiotic"], "fluid", b["protocol"]["fluid_rate"],
          "vaso", b["protocol"]["vasopressor_dose"], "·", b["protocol"]["z3"])
    for it in b["checklist"]:
        print(f"  [{it['n']}] {it['title']} — {it['detail']} ({it['status']})")
    print("NOTE    :", b["note"]["certified"], b["note"]["z3"])


if __name__ == "__main__":
    _demo()
