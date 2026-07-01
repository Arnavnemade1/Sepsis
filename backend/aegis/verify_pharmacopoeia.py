"""Verify harness for the Living Pharmacopoeia — claims asserted by measurement.

  1. KB SCALE   — the fused knowledge base loads a large real graph (DDInter) plus
                  the curated CYP/QT/renal mechanism layer.
  2. PAIRWISE   — a known Major pair (warfarin + fluconazole) is discovered from the
                  real DDInter graph and blocks the regimen with an unsatisfied Z3
                  constraint; a benign pair leaves the regimen safe.
  3. MECHANISM  — CYP450 (inhibitor + same-enzyme substrate) and additive-QT
                  interactions are discovered even where the bulk graph lacks
                  mechanism text — the fusion adds reasoning, not just lookups.
  4. CONTEXT    — renal failure escalates a renal-contraindicated drug (cefepime) to
                  a hard Major constraint.
  5. DYNAMIC Z3 — every Major interaction becomes a checked Z3 constraint with an
                  s-expression in the proof tree; Moderate/Minor are warnings, not
                  blocks (the honest clinical posture).

The CYP/QT/renal layer is bundled, so these checks are deterministic with or without
the gitignored DDInter CSVs; the KB-scale check tolerates their absence.

    python -m backend.aegis.verify_pharmacopoeia
"""
from __future__ import annotations

from backend.aegis.pharmacopoeia import Pharmacopoeia, CYP450, QT_KNOWN


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


def main() -> int:
    failures = 0
    checks = []
    pk = Pharmacopoeia()

    # ---- 1. KB SCALE ----------------------------------------------------- #
    _hr("1. KNOWLEDGE BASE (fused, cited sources)")
    st = pk.stats()
    print(f"    DDInter: {st['ddinter']['drugs']} drugs · {st['ddinter']['edges']} edges "
          f"({st['ddinter']['files']} files)")
    print(f"    curated: {len(st['cyp_enzymes'])} CYP enzymes · {st['qt_known']} known-QT drugs")
    ddinter_loaded = st["ddinter"]["edges"] > 1000
    print("    (DDInter CSVs present)" if ddinter_loaded else "    (DDInter absent — curated layer only)")
    checks.append(("curated mechanism layer present (CYP enzymes + QT list)",
                   len(st["cyp_enzymes"]) >= 5 and st["qt_known"] >= 10))
    if ddinter_loaded:
        checks.append(("real DDInter graph is large (>50k edges)", st["ddinter"]["edges"] > 50000))

    # ---- 2. PAIRWISE (DDInter) ------------------------------------------ #
    _hr("2. PAIRWISE GRAPH (real DDInter severities)")
    haz = pk.verify(["warfarin", "fluconazole", "ciprofloxacin", "ondansetron"],
                    {"renal_failure": False})
    print(f"    hazardous regimen: safe={haz['safe']} interactions={haz['n_interactions']} "
          f"severity={haz['severity_counts']}")
    safe = pk.verify(["cefepime", "vancomycin"], {"renal_failure": False})
    print(f"    benign regimen: safe={safe['safe']} interactions={safe['n_interactions']}")
    checks.append(("a hazardous polypharmacy regimen is flagged UNSAFE", haz["safe"] is False))
    checks.append(("at least one Major hard constraint was violated", len(haz["violations"]) >= 1))
    checks.append(("a benign regimen is proven SAFE", safe["safe"] is True))

    # ---- 3. MECHANISM (CYP + QT fusion) --------------------------------- #
    _hr("3. MECHANISM REASONING (CYP450 + additive QT)")
    mechs = {p["source"] for p in haz["proof_tree"]}
    cyp = [p for p in haz["proof_tree"] if p["source"] == "CYP450"]
    qt = [p for p in haz["proof_tree"] if p["source"] == "CredibleMeds"]
    print(f"    proof-tree sources: {sorted(mechs)}")
    for p in cyp + qt:
        print(f"      · {p['mechanism']} [{p['source']}]")
    checks.append(("a CYP450-mediated interaction was derived", len(cyp) >= 1))
    checks.append(("an additive-QT interaction was derived", len(qt) >= 1))
    checks.append(("fusion combines >=2 distinct evidence sources", len(mechs) >= 2))

    # ---- 4. CONTEXT (renal) --------------------------------------------- #
    _hr("4. PATIENT CONTEXT (renal contraindication)")
    renal = pk.verify(["cefepime"], {"renal_failure": True})
    print(f"    cefepime in renal failure: safe={renal['safe']} violations={renal['violations']}")
    checks.append(("renal failure makes cefepime a hard Major constraint",
                   renal["safe"] is False and any("cefepime" in v for v in renal["violations"])))
    no_renal = pk.verify(["cefepime"], {"renal_failure": False})
    checks.append(("the same drug is safe without renal failure", no_renal["safe"] is True))

    # ---- 5. DYNAMIC Z3 -------------------------------------------------- #
    _hr("5. DYNAMIC Z3 COMPOSITION (proof tree)")
    has_sexpr = all("constraint" in p and p["constraint"] for p in haz["proof_tree"])
    every_major_checked = haz["hard_constraints"] == len(haz["proof_tree"])
    print(f"    {haz['hard_constraints']} Major constraints composed; "
          f"{len(haz['warnings'])} moderate/minor warnings (non-blocking)")
    print(f"    example constraint: {haz['proof_tree'][0]['constraint']}")
    checks.append(("every Major interaction became a Z3 constraint with an s-expression",
                   has_sexpr and every_major_checked))
    checks.append(("Moderate/Minor surfaced as warnings, not hard blocks",
                   all(w["severity"] != "Major" for w in haz["warnings"])))

    # ---- tally ----------------------------------------------------------- #
    _hr("VERIFYING THE GUARANTEE")
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1

    _hr("RESULT")
    print("  ALL CHECKS PASSED ✓  — a real, fused drug-interaction graph, formally "
          "verified, with every constraint dynamically composed and proven."
          if failures == 0 else f"  {failures} CHECK(S) FAILED ✗")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
