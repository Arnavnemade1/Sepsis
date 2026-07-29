"""Verification of the treatment-projection REST contract.

Phases 1 and 2 proved the pharmacology. This proves the wiring — specifically the
things that would let a correct engine still show a clinician the wrong thing:

  * **Projecting an order nobody placed.** The twin must distinguish "this is what
    is running in the patient" from "this is what would be certified if you asked".
    Rendering the second and labelling it the first is the failure mode that
    matters, so the endpoint's ``administered`` flag is checked on both sides of a
    real ``treat`` syscall.
  * **A stale cache.** Projections are cached on the inputs that determine them.
    If the cache key misses a covariate, a patient in renal failure gets served the
    healthy patient's curve. Every covariate is therefore checked to actually move
    the cached result.
  * **A projection that quietly mutates the system.** A projection is a READ. It
    must not treat, escalate, or touch the scheduler — the Z3 cage stays the only
    thing that can change a patient's care.
  * **Provenance surviving the trip.** Citations, evidence tiers and the honesty
    statement have to arrive at the UI, because that is where they are read.

Runs against the FastAPI app in-process via TestClient — no server needed.

    python -m backend.aegis.verify_pk_api
"""
from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Tuple

PASS, FAIL = "PASS", "FAIL"
_results: List[Tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((name, bool(ok), detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""))
    return bool(ok)


def main() -> int:
    from fastapi.testclient import TestClient
    from backend.aegis import server as srv

    client = TestClient(srv.app)
    beds = list(getattr(srv.cluster, "profiles", {}))
    if not beds:
        print("no cohort beds available — cannot verify")
        return 1
    bed = beds[0]

    # ---------------------------------------------------------------------- #
    print("\n=== 1. THE PROVENANCE SURFACE ===")
    r = client.get("/aegis/pk/drugs")
    check("GET /aegis/pk/drugs responds", r.status_code == 200, f"HTTP {r.status_code}")
    d = r.json()
    check("every drug ships its PK citation",
          all(x["citation"] and len(x["citation"]) > 15 for x in d["drugs"]),
          f"{len(d['drugs'])} drugs, all cited")
    check("every drug states its own projected timeframe",
          all(x["horizon_h"] > 0 for x in d["drugs"]),
          " · ".join(f"{x['label'].split()[0]} {x['horizon_h']:.0f}h"
                     for x in d["drugs"]))
    check("the adverse-event catalogue travels with its evidence tiers",
          all(e["evidence"] in ("ANCHORED", "LABEL_RATE", "MECHANISTIC")
              for e in d["adverse_events"]),
          f"{len(d['adverse_events'])} events")
    check("every adverse event maps to something renderable on the body",
          all(e["mesh"] for e in d["adverse_events"]),
          "no event is invisible to the twin")
    check("the physiology source is disclosed", "Brown" in d["provenance"]["physiology"],
          d["provenance"]["physiology"][:52] + "…")

    # ---------------------------------------------------------------------- #
    print("\n=== 2. A BED PROJECTS ITS OWN ORDER ===")
    r = client.get(f"/aegis/pk/bed/{bed}")
    p = r.json()
    check(f"GET /aegis/pk/bed/{bed} responds ok", p.get("status") == "ok",
          f"bed {p.get('bed_name')}")
    check("arms carry per-organ series of equal length to the time axis",
          all(len(s) == len(a["times_h"])
              for a in p["arms"] for s in a["organs"].values()),
          f"{len(p['arms'])} arms × {len(p['arms'][0]['organs'])} compartments "
          f"× {len(p['arms'][0]['times_h'])} samples")
    check("each arm declares its unit and kind",
          all(a["unit"] and a["kind"] in ("concentration", "volume")
              for a in p["arms"]),
          " · ".join(f"{a['label'].split()[0]} [{a['kind']}, {a['unit']}]"
                     for a in p["arms"]))
    check("timeline events are ordered and inside the horizon",
          all(all(0 <= e["t_h"] <= a["horizon_h"] + 1e-9 for e in a["events"])
              and [e["t_h"] for e in a["events"]] ==
              sorted(e["t_h"] for e in a["events"]) for a in p["arms"]),
          "arrival/peak/infusion markers well formed")
    check("the honesty statement survives to the UI payload",
          all("not a therapeutic" in a["honesty"] or "Population" in a["honesty"]
              for a in p["arms"]),
          "stated per arm")

    # ---------------------------------------------------------------------- #
    print("\n=== 3. ADMINISTERED vs MERELY CERTIFIED ===")
    print("      (the twin must never imply a drug is running when none was ordered)")
    fresh = [b for b in beds if not getattr(
        srv.syscalls.interventions.get(b), "treated", False)]
    tb = fresh[0] if fresh else bed
    before = client.get(f"/aegis/pk/bed/{tb}").json()
    check("an untreated bed is labelled as a hypothetical",
          before.get("administered") is False and "not yet ordered" in before["source"],
          f"source: {before['source']}")

    t = client.post("/aegis/syscall",
                    json={"syscall": "treat", "bed": tb, "args": {}}).json()
    if t.get("allowed"):
        check("a certified treat returns a projection handle",
              t.get("projection", {}).get("href") == f"/aegis/pk/bed/{tb}",
              t["projection"]["href"])
        after = client.get(f"/aegis/pk/bed/{tb}").json()
        check("after the order, the same bed is labelled as administered",
              after.get("administered") is True and "administered" in after["source"],
              f"source: {after['source']}")
        iv = srv.syscalls.interventions[tb].protocol
        check("the projection is of what was ACTUALLY administered",
              all(abs(float(after["protocol"][k]) - float(iv[k])) < 1e-9
                  for k in ("fluid_rate", "vasopressor_dose", "antibiotic_code")),
              f"{iv}")
    else:
        check("treat was denied by the cage — projection handle withheld",
              "projection" not in t, f"z3 {t.get('z3_status')}: {t.get('message','')[:50]}")

    # ---------------------------------------------------------------------- #
    print("\n=== 4. THE CACHE KEY COVERS EVERY COVARIATE THAT MOVES A CURVE ===")
    print("      (a key that misses one serves a different patient's projection)")
    base_body = {"protocol": {"fluid_rate": 10.0, "vasopressor_dose": 0.1,
                              "antibiotic_code": 2},
                 "patient": {"weight_kg": 78.0, "egfr": 90.0, "map_mmhg": 75.0,
                             "hepatic_factor": 1.0, "cardiac_output_factor": 1.0},
                 "adverse": False}

    def project(body: Dict[str, Any]) -> Dict[str, Any]:
        r = client.post("/aegis/pk/project", json=body).json()
        arm = next(a for a in r["arms"] if a["drug_class"] == "antibiotic")
        return {"digest": hashlib.sha256(
                    json.dumps(r["arms"], sort_keys=True).encode()).hexdigest()[:12],
                "auc": arm["metrics"]["auc"],
                "kidney": arm["metrics"]["organ_auc"]["kidney"]}

    # The probe is a digest of the WHOLE payload, not one metric. A covariate can
    # legitimately leave plasma AUC untouched while changing everything else —
    # hypotension redistributes flow without altering the clearance of a
    # low-extraction drug — and a cache-key test that only watched plasma AUC
    # would then be reading pharmacology as though it were a cache hit.
    ref = project(base_body)
    for field, value, why in (("egfr", 25.0, "renal clearance"),
                              ("weight_kg", 110.0, "body size, volumes, cardiac output"),
                              ("map_mmhg", 45.0, "shock flow redistribution"),
                              ("hepatic_factor", 0.4, "hepatic clearance"),
                              ("cardiac_output_factor", 1.5, "delivery rate")):
        body = {**base_body, "patient": {**base_body["patient"], field: value}}
        got = project(body)
        check(f"changing {field:22s} yields a different projection",
              got["digest"] != ref["digest"],
              f"payload {ref['digest']} → {got['digest']} ({why})")

    # And the two covariates whose effect is easy to misread, checked where the
    # effect actually lives rather than where it happens not to be.
    shocked = project({**base_body,
                       "patient": {**base_body["patient"], "map_mmhg": 45.0}})
    check("hypotension lowers RENAL exposure even at unchanged plasma AUC",
          shocked["kidney"] < ref["kidney"]
          and abs(shocked["auc"] - ref["auc"]) / ref["auc"] < 0.01,
          f"kidney AUC {ref['kidney']:.0f} → {shocked['kidney']:.0f} while plasma "
          f"AUC holds at {ref['auc']:.0f} (clearance is flow-independent here)")
    fixed_dose = {**base_body, "protocol": {**base_body["protocol"],
                                            "antibiotic_code": 3}}   # cefepime, 2 g
    heavy = project({**fixed_dose,
                     "patient": {**base_body["patient"], "weight_kg": 110.0}})
    light = project(fixed_dose)
    check("weight moves exposure for a FIXED dose (it cannot for a mg/kg dose)",
          heavy["auc"] < light["auc"] * 0.85,
          f"cefepime 2 g: 78 kg → {light['auc']:.0f}, 110 kg → {heavy['auc']:.0f} "
          f"mg/L·h; vancomycin's mg/kg dose is weight-invariant by construction")
    body = {**base_body, "protocol": {**base_body["protocol"], "antibiotic_code": 4}}
    other = project(body)
    check("changing the ordered drug changes the projection",
          other["digest"] != ref["digest"],
          f"vancomycin AUC {ref['auc']:.1f} → meropenem {other['auc']:.1f}")
    check("an identical request is served from cache, byte-identical",
          client.post("/aegis/pk/project", json=base_body).json()["arms"][0]["plasma"]
          == client.post("/aegis/pk/project", json=base_body).json()["arms"][0]["plasma"],
          "same inputs → same curve")

    # ---------------------------------------------------------------------- #
    print("\n=== 5. A PROJECTION IS A READ — IT MUTATES NOTHING ===")
    untreated = [b for b in beds
                 if not getattr(srv.syscalls.interventions.get(b), "treated", False)]
    probe = untreated[0] if untreated else beds[-1]
    n_audit = len(srv.syscalls.audit)
    treated_before = {b: getattr(srv.syscalls.interventions.get(b), "treated", False)
                      for b in beds}
    pins_before = {b: getattr(srv.syscalls.interventions.get(b), "pin", 1.0)
                   for b in beds}
    for _ in range(3):
        client.get(f"/aegis/pk/bed/{probe}")
        client.post("/aegis/pk/project", json=base_body)
    check("projecting writes nothing to the syscall audit log",
          len(srv.syscalls.audit) == n_audit, f"log still {n_audit} entries")
    check("projecting does not mark any bed as treated",
          all(getattr(srv.syscalls.interventions.get(b), "treated", False)
              == treated_before[b] for b in beds),
          "intervention state unchanged across the wing")
    check("projecting does not move any scheduler pin",
          all(getattr(srv.syscalls.interventions.get(b), "pin", 1.0)
              == pins_before[b] for b in beds),
          "no priority side effects")

    # ---------------------------------------------------------------------- #
    print("\n=== 6. ADVERSE EVENTS ARRIVE RENDER-READY ===")
    r = client.post("/aegis/pk/project", json={
        "protocol": {"fluid_rate": 10.0, "vasopressor_dose": 0.2,
                     "antibiotic_code": 2},
        "patient": {"weight_kg": 78.0, "egfr": 30.0, "map_mmhg": 55.0},
        "adverse": True}).json()
    ae = r["adverse"]
    check("the side-effect projection is attached", ae["n_events"] > 0,
          f"{ae['n_events']} events, top: {ae['events'][0]['label']}")
    check("every event carries organ, onset window and interval",
          all(e["mesh"] and e["onset_p10_h"] <= e["onset_p50_h"] <= e["onset_p90_h"]
              and e["ci_low"] <= e["probability"] <= e["ci_high"]
              for e in ae["events"]),
          "the twin can place and time every one of them")
    check("the evidence mix is reported alongside the numbers",
          "fraction_anchored" in ae["evidence_mix"],
          f"{ae['evidence_mix']['anchored']} anchored · "
          f"{ae['evidence_mix']['label_rate']} label-rate · "
          f"{ae['evidence_mix']['mechanistic']} mechanistic")
    check("the payload says it never gates an order",
          "never gates an order" in ae["honesty"], "Z3 cage remains the only gate")
    check("adverse=false omits the side-effect projection",
          "adverse" not in client.post(
              "/aegis/pk/project",
              json={**base_body, "adverse": False}).json(),
          "opt-out honoured (it is the expensive half)")

    # ---------------------------------------------------------------------- #
    print("\n=== 7. BAD INPUT FAILS CLEANLY ===")
    r = client.get("/aegis/pk/bed/bed_does_not_exist").json()
    check("an unknown bed returns an error, not a projection",
          r.get("status") == "error" and "arms" not in r, r.get("detail", ""))
    r = client.post("/aegis/pk/project",
                    json={"bed": "nope", "adverse": False}).json()
    check("an unknown bed on POST returns an error too",
          r.get("status") == "error", r.get("detail", ""))
    r = client.post("/aegis/pk/project", json={
        "protocol": {"fluid_rate": 0.0, "vasopressor_dose": 0.0,
                     "antibiotic_code": 0},
        "patient": {"weight_kg": 78.0}, "adverse": True}).json()
    check("an empty order projects nothing rather than inventing an arm",
          r["arms"] == [], "no drugs ordered → no drug rendered")

    # ---------------------------------------------------------------------- #
    n_pass = sum(1 for _, ok, _ in _results if ok)
    n_tot = len(_results)
    print("\n" + "=" * 72)
    print(f"  TREATMENT-PROJECTION API: {n_pass}/{n_tot} checks passed")
    if n_pass != n_tot:
        print("  FAILED:")
        for name, ok, detail in _results:
            if not ok:
                print(f"    · {name} — {detail}")
    print("=" * 72 + "\n")
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
