"""Verification for the per-patient medical-history intake + clinician synthesis.

Locks the load-bearing behaviour:
  * a free-text history specialises the monitor (seeds foci + watches) deterministically
    with NO LLM attached;
  * the records-source registry exposes an honest stub (never claims a live chart);
  * a clinician synthesis is logged AND its monitoring intent applied, while any
    treatment/order intent is flagged and NEVER auto-executed (the safety boundary).
"""
from backend.aegis.patient_terminal import PatientTerminal
from backend.aegis.patient_records import (
    parse_history, RecordsRegistry, MockHospitalSource, ManualSource,
)


def _check(name, cond):
    print(f"      [{'PASS' if cond else 'FAIL'}] {name}")
    if not cond:
        raise SystemExit(f"verification failed: {name}")


def main():
    print("=" * 66)
    print("AegisOS — medical-history intake + clinician synthesis verification")
    print("=" * 66)

    pt = PatientTerminal(llm_client=None)   # deterministic path only

    print("[1] Deterministic history parse seeds foci + watches (no LLM)")
    txt = ("PMH: chronic kidney disease stage 3b, congestive heart failure EF 30%, "
           "COPD on home oxygen, immunocompromised on chemotherapy. "
           "Medications: furosemide, metformin, insulin. Allergy: penicillin.")
    hist, directive = parse_history("bed_A", txt, llm_client=None)
    foci = set(directive["focus"])
    _check("CKD → Renal focus", "Renal" in foci)
    _check("CHF → Cardiovascular focus", "Cardiovascular" in foci)
    _check("COPD → Respiratory focus", "Respiratory" in foci)
    _check("chemo/immunocompromised → Immune focus", "Immune" in foci)
    _check("seed watches produced", len(directive["watches"]) >= 2)
    _check("comorbidities captured", len(hist.comorbidities) >= 3)

    print("[2] import_history applies the specialisation to the profile")
    out = pt.import_history("bed_A", txt, source="manual")
    prof = out["profile"]
    _check("profile now focuses the seeded systems",
           {"Renal", "Cardiovascular", "Respiratory", "Immune"}.issubset(set(prof["foci"])))
    _check("watches active on the profile", prof["n_watches"] >= 2)
    _check("history retrievable", pt.get_history("bed_A") is not None)
    _check("priorities recorded as notes or empty", isinstance(prof["notes"], list))

    print("[3] Records registry is honest about stubs")
    reg = RecordsRegistry()
    ids = {s["id"] for s in reg.list_sources()}
    _check("manual + mock_hospital registered", {"manual", "mock_hospital"} <= ids)
    _check("mock hospital flagged NOT live (stub)",
           reg.get("mock_hospital").live is False)
    _check("manual source is live (local paste path)", reg.get("manual").live is True)

    print("[4] link_source pulls the stub + specialises (septic vs stable differ)")
    link = pt.link_source("bed_S", "mock_hospital", patient={"septic": True})
    _check("link ok", link.get("ok") is True)
    _check("stub link carries live=False", link["source"]["live"] is False)
    _check("septic demo seeds multiple foci", len(link["profile"]["foci"]) >= 3)
    link2 = pt.link_source("bed_T", "mock_hospital", patient={"septic": False})
    _check("stable demo differs from septic demo",
           set(link["profile"]["foci"]) != set(link2["profile"]["foci"]))
    _check("unknown source refused",
           pt.link_source("bed_T", "nope")["ok"] is False)

    print("[5] Clinician synthesis: monitoring applied, treatment NEVER executed")
    s1 = pt.add_synthesis("bed_A", "Keep a close eye on her kidneys and alert if MAP under 60.")
    _check("monitoring synthesis applied", len(s1["applied"]) >= 1)
    _check("no false treatment flag on a pure watch", s1["order_intent"] is False)
    s2 = pt.add_synthesis("bed_A", "Start norepinephrine 5 mcg/min and give 2L fluids stat.")
    _check("treatment intent detected", s2["order_intent"] is True)
    _check("treatment reply redirects to the certified gate",
           "treat" in s2["reply"].lower() and "not" in s2["reply"].lower())
    _check("both syntheses logged as annotations", len(pt.get_annotations("bed_A")) == 2)
    _check("annotations are observation-only",
           all(a["monitoring_only"] for a in pt.get_annotations("bed_A")))

    print("[6] Empty / junk input is safe")
    e = pt.import_history("bed_Z", "   ")
    _check("empty history seeds nothing", not e["profile"]["foci"] and e["profile"]["n_watches"] == 0)
    j, jd = parse_history("bed_Z", "the weather is nice today", llm_client=None)
    _check("irrelevant text seeds no foci/watches", not jd["focus"] and not jd["watches"])

    print("-" * 66)
    print("MEDICAL-HISTORY INTAKE + SYNTHESIS VERIFIED ✓  "
          "(specialise · honest stub · safe order boundary)")
    print("=" * 66)


if __name__ == "__main__":
    main()
