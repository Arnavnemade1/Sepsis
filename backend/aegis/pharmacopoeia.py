"""The Living Pharmacopoeia — a drug-interaction graph engine that composes Z3
constraints dynamically (MEGA-UPGRADE Layer 4).

The cage's hand-written rules check a handful of invariants. This subsystem loads a
**real, large drug-interaction knowledge base**, traverses it for the patient's
actual drug list, turns every discovered interaction into a **dynamically-composed
Z3 constraint**, and emits a **proof tree** showing why the combined regimen is (or
is not) globally safe.

DATA PROVENANCE — fused from open sources, each cited (the honesty crux):

  * **DDInter** (ddinter.scbdd.com, CC BY-NC-SA) — the pairwise interaction graph:
    ~143k undirected drug-drug edges with a severity Level (Major/Moderate/Minor)
    across 1.8k drugs. Downloaded into ``data/ddinter/`` (gitignored, like the CinC
    corpus). The bulk export carries *severity but not mechanism prose*, so:
  * **CYP450 / Flockhart classification** (public reference) — enzyme substrate /
    inhibitor / inducer tables, used to derive *mechanistic* interactions
    (inhibitor + substrate of the same enzyme → exposure interaction).
  * **CredibleMeds-style QT categories** (public clinical knowledge) — additive
    QT-prolongation risk when ≥2 QT-active drugs are co-administered.
  * **Curated renal/hepatic dosing + nephrotoxic-synergy** flags (standard
    references) — e.g. vancomycin + piperacillin-tazobactam AKI; cefepime
    neurotoxicity in renal failure.

WHAT IS AND ISN'T CLAIMED:

  * It is **exhaustive and formal over the loaded knowledge base** — every relevant
    edge becomes a checked Z3 constraint, with a proof. It is NOT "the entire FDA
    dataset" (no single clean open such dataset exists) and NOT "safer than a
    pharmacist" — it cannot reason about interactions absent from the KB.
  * Hard (blocking) constraints = **Major** pairwise interactions, additive QT with a
    *known*-risk drug, and renal-contraindicated dosing. **Moderate/Minor** are
    surfaced as managed warnings, not forbidden — the honest clinical posture.
  * If ``data/ddinter/`` is absent (fresh clone), the engine still runs on the
    bundled curated CYP/QT/renal layer; the pairwise graph is simply smaller.
"""
from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from z3 import And, Bool, BoolVal, Implies, Not, Or, Solver, sat, unsat

SOURCES = {
    "DDInter": "ddinter.scbdd.com — open DDI database (CC BY-NC-SA)",
    "CYP450": "Flockhart CYP450 Drug Interaction Table (Indiana University, public reference)",
    "CredibleMeds": "CredibleMeds QTdrugs categories (public clinical reference)",
    "curated": "standard references (renal/hepatic dosing, nephrotoxic synergy)",
}
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
_DDINTER_DIR = os.path.join(_REPO_ROOT, "data", "ddinter")
_SEV_RANK = {"Minor": 1, "Moderate": 2, "Major": 3}


def normalize(name: str) -> str:
    return name.strip().lower()


# --------------------------------------------------------------------------- #
# Curated mechanism layer (public references).
# --------------------------------------------------------------------------- #
CYP450: Dict[str, Dict[str, Set[str]]] = {
    "CYP3A4": {
        "inhibitor": {"fluconazole", "ketoconazole", "voriconazole", "clarithromycin",
                      "erythromycin", "ritonavir", "amiodarone", "diltiazem", "verapamil"},
        "inducer": {"rifampin", "rifampicin", "phenytoin", "carbamazepine", "phenobarbital"},
        "substrate": {"midazolam", "simvastatin", "atorvastatin", "tacrolimus",
                      "cyclosporine", "fentanyl", "quetiapine", "apixaban", "rivaroxaban"},
    },
    "CYP2C9": {
        "inhibitor": {"fluconazole", "amiodarone", "metronidazole", "sulfamethoxazole"},
        "inducer": {"rifampin", "carbamazepine"},
        "substrate": {"warfarin", "phenytoin", "glipizide", "glimepiride", "losartan", "celecoxib"},
    },
    "CYP1A2": {
        "inhibitor": {"ciprofloxacin", "fluvoxamine"},
        "inducer": set(),
        "substrate": {"theophylline", "clozapine", "olanzapine", "tizanidine", "caffeine"},
    },
    "CYP2D6": {
        "inhibitor": {"fluoxetine", "paroxetine", "bupropion", "quinidine", "amiodarone"},
        "inducer": set(),
        "substrate": {"metoprolol", "codeine", "tramadol", "haloperidol", "risperidone"},
    },
    "CYP2C19": {
        "inhibitor": {"omeprazole", "fluconazole", "fluvoxamine"},
        "inducer": set(),
        "substrate": {"clopidogrel", "citalopram", "escitalopram", "diazepam"},
    },
}
# QT-prolongation risk (CredibleMeds-style categories)
QT_KNOWN = {"amiodarone", "sotalol", "haloperidol", "methadone", "ondansetron",
            "azithromycin", "moxifloxacin", "levofloxacin", "erythromycin",
            "clarithromycin", "droperidol", "citalopram", "quinidine", "dofetilide",
            "chlorpromazine", "thioridazine", "pimozide", "procainamide"}
QT_POSSIBLE = {"ciprofloxacin", "fluconazole", "olanzapine", "clozapine", "norfloxacin"}
# Renal dose-adjusted / contraindicated drugs
RENAL_ADJUST = {"vancomycin", "gentamicin", "tobramycin", "amikacin", "cefepime",
                "ceftazidime", "meropenem", "piperacillin", "fluconazole", "levofloxacin",
                "ciprofloxacin", "acyclovir", "colistin", "daptomycin", "metformin"}
RENAL_CONTRA = {"cefepime"}        # cefepime neurotoxicity in unadjusted renal failure
HEPATIC_ADJUST = {"acetaminophen", "rifampin", "isoniazid", "fluconazole", "voriconazole"}
# Curated nephrotoxic synergy (escalate to Major even if DDInter says Moderate)
NEPHROTOXIC_SYNERGY = {frozenset({"vancomycin", "piperacillin"}),
                       frozenset({"vancomycin", "gentamicin"}),
                       frozenset({"vancomycin", "tobramycin"}),
                       frozenset({"gentamicin", "furosemide"})}

# antibiotic_code (from the protocol synthesizer) -> drug name
DRUG_CODE = {0: None, 1: "cefepime", 2: "piperacillin", 3: "vancomycin",
             4: "meropenem", 5: "ceftriaxone", 6: "ciprofloxacin"}


@dataclass
class Interaction:
    a: str
    b: Optional[str]
    severity: str               # Major | Moderate | Minor
    mechanism: str
    source: str

    def to_dict(self) -> Dict[str, Any]:
        return {"a": self.a, "b": self.b, "severity": self.severity,
                "mechanism": self.mechanism, "source": self.source}


# --------------------------------------------------------------------------- #
def load_ddinter(directory: str = _DDINTER_DIR) -> Tuple[Dict[str, Dict[str, str]], Dict[str, Any]]:
    """Load the DDInter pairwise graph (highest severity kept per pair)."""
    graph: Dict[str, Dict[str, str]] = defaultdict(dict)
    files = sorted(glob.glob(os.path.join(directory, "ddinter_*.csv")))
    for f in files:
        with open(f, newline="") as fh:
            for row in csv.DictReader(fh):
                lv = row.get("Level", "").strip()
                if lv not in _SEV_RANK:
                    continue
                a, b = normalize(row["Drug_A"]), normalize(row["Drug_B"])
                if not a or not b or a == b:
                    continue
                if _SEV_RANK.get(graph[a].get(b, ""), 0) < _SEV_RANK[lv]:
                    graph[a][b] = lv
                    graph[b][a] = lv
    n_edges = sum(len(v) for v in graph.values()) // 2
    return graph, {"drugs": len(graph), "edges": n_edges, "files": len(files)}


# --------------------------------------------------------------------------- #
class Pharmacopoeia:
    """Drug-interaction knowledge base + dynamic Z3 safety verifier."""

    def __init__(self, ddinter_dir: str = _DDINTER_DIR):
        self.graph, self.ddinter_stats = load_ddinter(ddinter_dir)

    # -- graph traversal: discover every relevant interaction ----------- #
    def screen(self, drugs: List[str], context: Optional[Dict[str, Any]] = None
               ) -> List[Interaction]:
        context = context or {}
        ds = [normalize(d) for d in drugs if d]
        found: List[Interaction] = []
        seen: Set[Tuple] = set()

        # 1) pairwise DDInter edges
        for i in range(len(ds)):
            for j in range(i + 1, len(ds)):
                a, b = ds[i], ds[j]
                lv = self.graph.get(a, {}).get(b)
                pair = frozenset({a, b})
                if pair in NEPHROTOXIC_SYNERGY:
                    found.append(Interaction(a, b, "Major",
                                 "additive nephrotoxicity (AKI risk)", "curated"))
                    seen.add(("pair", pair))
                elif lv:
                    found.append(Interaction(a, b, lv, "documented interaction", "DDInter"))
                    seen.add(("pair", pair))

        # 2) CYP450-mediated (inhibitor/inducer + substrate of same enzyme)
        for enz, roles in CYP450.items():
            subs = roles["substrate"]
            for a in ds:
                for b in ds:
                    if a == b:
                        continue
                    if a in roles["inhibitor"] and b in subs:
                        found.append(Interaction(a, b, "Major",
                            f"{enz} inhibition — {a} raises {b} exposure", "CYP450"))
                    elif a in roles["inducer"] and b in subs:
                        found.append(Interaction(a, b, "Moderate",
                            f"{enz} induction — {a} lowers {b} efficacy", "CYP450"))

        # 3) additive QT prolongation
        qt = [d for d in ds if d in QT_KNOWN or d in QT_POSSIBLE]
        if len(qt) >= 2:
            sev = "Major" if any(d in QT_KNOWN for d in qt) else "Moderate"
            found.append(Interaction(", ".join(qt), None, sev,
                         "additive QT-interval prolongation", "CredibleMeds"))

        # 4) renal dosing / contraindication
        if context.get("renal_failure"):
            for d in ds:
                if d in RENAL_CONTRA:
                    found.append(Interaction(d, None, "Major",
                                 "neurotoxicity in renal failure — contraindicated unadjusted", "curated"))
                elif d in RENAL_ADJUST:
                    found.append(Interaction(d, None, "Moderate",
                                 "renal dose adjustment required", "curated"))
        return found

    # -- dynamic Z3 composition + proof tree ---------------------------- #
    def verify(self, drugs: List[str], context: Optional[Dict[str, Any]] = None
               ) -> Dict[str, Any]:
        """Compose a Z3 constraint per discovered interaction and prove the regimen
        is globally safe (no Major hazard). Returns a proof tree."""
        context = context or {}
        ds = [normalize(d) for d in drugs if d]
        interactions = self.screen(ds, context)
        present = {d: Bool(f"rx_{d.replace(' ', '_').replace('-', '_')}") for d in ds}
        binds = [present[d] == BoolVal(True) for d in ds]

        proof: List[Dict[str, Any]] = []
        warnings: List[Dict[str, Any]] = []
        for ix in interactions:
            if ix.severity != "Major":
                warnings.append(ix.to_dict())
                continue
            # the hard safety constraint this interaction imposes
            if ix.b is None:
                # single-drug or set hazard (QT set / renal contra): forbid the
                # hazardous combination of the implicated drugs being all present.
                members = [m.strip() for m in ix.a.split(",")] if "," in ix.a else [ix.a]
                lits = [present[m] for m in members if m in present]
                constraint = Not(And(*lits)) if len(lits) > 1 else Not(lits[0])
                label = ix.a
            else:
                constraint = Not(And(present[ix.a], present[ix.b]))
                label = f"{ix.a} + {ix.b}"
            s = Solver()
            s.add(*binds)
            s.add(Not(constraint))
            status = s.check()
            satisfied = status == unsat       # unsat ⇒ constraint holds for this regimen
            proof.append({
                "interaction": label, "severity": ix.severity, "mechanism": ix.mechanism,
                "source": ix.source, "constraint": constraint.sexpr(),
                "z3_status": "unsat" if satisfied else "sat", "satisfied": satisfied,
            })

        n_major = len(proof)
        violations = [p for p in proof if not p["satisfied"]]
        sev_counts = {"Major": sum(1 for i in interactions if i.severity == "Major"),
                      "Moderate": sum(1 for i in interactions if i.severity == "Moderate"),
                      "Minor": sum(1 for i in interactions if i.severity == "Minor")}
        return {
            "safe": len(violations) == 0,
            "drugs": ds,
            "n_interactions": len(interactions),
            "severity_counts": sev_counts,
            "hard_constraints": n_major,
            "violations": [v["interaction"] for v in violations],
            "proof_tree": proof,
            "warnings": warnings,
            "kb": {"ddinter": self.ddinter_stats, "sources": SOURCES,
                   "cyp_enzymes": list(CYP450), "qt_known": len(QT_KNOWN)},
        }

    # -- protocol integration ------------------------------------------- #
    def drugs_from_protocol(self, protocol: Dict[str, Any],
                            current_meds: Optional[List[str]] = None) -> List[str]:
        drugs = list(current_meds or [])
        code = int(protocol.get("antibiotic_code", 0) or 0)
        if DRUG_CODE.get(code):
            drugs.append(DRUG_CODE[code])
        return drugs

    def stats(self) -> Dict[str, Any]:
        return {"ddinter": self.ddinter_stats, "sources": SOURCES,
                "cyp_enzymes": list(CYP450), "qt_known": len(QT_KNOWN),
                "qt_possible": len(QT_POSSIBLE), "renal_adjust": len(RENAL_ADJUST)}


# --------------------------------------------------------------------------- #
def _demo() -> None:
    pk = Pharmacopoeia()
    print("KB:", pk.stats()["ddinter"])
    print("\n-- HAZARDOUS regimen (warfarin + fluconazole + ciprofloxacin + ondansetron) --")
    res = pk.verify(["warfarin", "fluconazole", "ciprofloxacin", "ondansetron"],
                    {"renal_failure": False})
    print(f"safe={res['safe']} interactions={res['n_interactions']} "
          f"severity={res['severity_counts']} violations={res['violations']}")
    for p in res["proof_tree"]:
        print(f"  [{'OK ' if p['satisfied'] else 'HIT'}] {p['interaction']:32} "
              f"{p['mechanism']} ({p['source']})")
    print("\n-- SAFE regimen (cefepime + vancomycin, no renal failure) --")
    res2 = pk.verify(["cefepime", "vancomycin"], {"renal_failure": False})
    print(f"safe={res2['safe']} interactions={res2['n_interactions']} "
          f"severity={res2['severity_counts']}")


if __name__ == "__main__":
    _demo()
