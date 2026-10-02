"""The ward formulary: what each order is, and the curated safety rules over it.

Every drug the demo hospital prescribes is listed with a standard adult dose, its
pharmacological classes and the properties the safety detectors need (renal
clearance, QT risk, sedation). The rules below are the ward-relevant subset of
standard references, written out so each one names its source. They complement
``backend.aegis.pharmacopoeia`` (DDInter where the corpus is installed, CYP450,
CredibleMeds), which the medication detector also consults.

This is decision support for a demo hospital, not a pharmacist: a rule that is
not written here, or an interaction absent from the knowledge base, is not found.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional


@dataclass(frozen=True)
class Drug:
    name: str                    # as ordered ("hydromorphone PCA")
    generic: str                 # for the interaction knowledge base ("hydromorphone")
    dose: str
    route: str
    freq: str
    classes: FrozenSet[str] = field(default_factory=frozenset)
    prn: bool = False
    infusion: bool = False       # runs on a pump: the pump is a device in the room
    renal: str = ""              # renal-dosing note when clearance falls
    renal_below: float = 0.0     # the creatinine clearance (mL/min) below which that note applies
    qt: bool = False             # CredibleMeds known risk of torsades


def _d(name: str, generic: str, dose: str, route: str, freq: str, classes: str, **kw) -> Drug:
    return Drug(name, generic, dose, route, freq, frozenset(classes.split()), **kw)


FORMULARY: Dict[str, Drug] = {d.name: d for d in [
    _d("ceftriaxone", "ceftriaxone", "1 g", "IV", "q24h", "antibiotic cephalosporin"),
    _d("azithromycin", "azithromycin", "500 mg", "IV", "q24h", "antibiotic macrolide", qt=True),
    _d("albuterol", "albuterol", "2.5 mg", "neb", "q4h", "bronchodilator", prn=True),
    _d("cefazolin", "cefazolin", "2 g", "IV", "q8h", "antibiotic cephalosporin", renal="extend the interval below CrCl 35", renal_below=35),
    _d("metformin", "metformin", "500 mg", "PO", "BID", "biguanide", renal="contraindicated below eGFR 30; hold for iodinated contrast", renal_below=30),
    _d("insulin glargine", "insulin glargine", "18 units", "SC", "nightly", "insulin"),
    _d("amlodipine", "amlodipine", "5 mg", "PO", "daily", "ccb_dhp antihypertensive"),
    _d("acetaminophen", "acetaminophen", "1 g", "PO", "q6h", "analgesic"),
    _d("enoxaparin", "enoxaparin", "40 mg", "SC", "daily", "anticoagulant lmwh", renal="30 mg daily below CrCl 30", renal_below=30),
    _d("ondansetron", "ondansetron", "4 mg", "IV", "q8h", "antiemetic", prn=True, qt=True),
    _d("hydromorphone PCA", "hydromorphone", "0.2 mg per demand, 10 min lockout", "IV", "PCA", "opioid sedating", infusion=True),
    _d("piperacillin-tazobactam", "piperacillin", "4.5 g", "IV", "q6h", "antibiotic penicillin", renal="reduce below CrCl 40", renal_below=40),
    _d("aspirin", "aspirin", "81 mg", "PO", "daily", "antiplatelet"),
    _d("heparin infusion", "heparin", "12 units/kg/h (ACS nomogram)", "IV", "continuous", "anticoagulant ufh", infusion=True),
    _d("atorvastatin", "atorvastatin", "80 mg", "PO", "nightly", "statin"),
    _d("metoprolol", "metoprolol", "25 mg", "PO", "BID", "beta_blocker antihypertensive"),
    _d("diltiazem", "diltiazem", "30 mg", "PO", "q6h", "ccb_nondhp antihypertensive"),
    _d("apixaban", "apixaban", "5 mg", "PO", "BID", "anticoagulant doac"),
    _d("furosemide", "furosemide", "40 mg", "IV", "BID", "loop_diuretic sulfonamide_nonantibiotic"),
    _d("furosemide infusion", "furosemide", "10 mg/h", "IV", "continuous", "loop_diuretic sulfonamide_nonantibiotic", infusion=True),
    _d("carvedilol", "carvedilol", "6.25 mg", "PO", "BID", "beta_blocker antihypertensive"),
    _d("spironolactone", "spironolactone", "25 mg", "PO", "daily", "k_sparing", renal="avoid below eGFR 30 (hyperkalaemia)", renal_below=30),
    _d("labetalol", "labetalol", "10 mg if SBP > 180", "IV", "q4h", "beta_blocker antihypertensive", prn=True),
    _d("cefepime", "cefepime", "2 g", "IV", "q8h", "antibiotic cephalosporin", renal="neurotoxicity if not reduced below CrCl 60", renal_below=60),
    _d("filgrastim", "filgrastim", "5 mcg/kg", "SC", "daily", "g_csf"),
    _d("IV fluids", "lactated ringer", "100 mL/h", "IV", "continuous", "fluid", infusion=True),
    _d("morphine", "morphine", "2–4 mg", "IV", "q3h", "opioid sedating", prn=True, renal="active metabolite accumulates below CrCl 30", renal_below=30),
    _d("nitroglycerin", "nitroglycerin", "0.4 mg for chest pain", "SL", "q5min ×3", "nitrate", prn=True),
    _d("insulin infusion", "insulin regular", "0.1 units/kg/h (DKA protocol)", "IV", "continuous", "insulin", infusion=True),
    _d("potassium chloride", "potassium chloride", "10 mEq/h", "IV", "continuous", "potassium", infusion=True),
    _d("levetiracetam", "levetiracetam", "500 mg", "IV", "BID", "antiepileptic", renal="reduce below CrCl 50", renal_below=50),
    _d("lorazepam PRN", "lorazepam", "2 mg for seizure > 5 min", "IV", "PRN", "benzodiazepine sedating", prn=True),
]}

# Orders in the demo cases that are not drugs.
NON_DRUG_ORDERS = {"NG decompression": "NG tube to low intermittent suction"}


def lookup(name: str) -> Optional[Drug]:
    return FORMULARY.get(name)


# --------------------------------------------------------------------------- #
# Allergies: which classes an allergy covers, and how far it reaches.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AllergyRule:
    substance: str
    direct: FrozenSet[str]               # classes the allergy contraindicates
    cross: Dict[str, str]                # class -> the honest cross-reactivity note
    kind: str = "allergy"                # allergy | intolerance | procedure


ALLERGY_RULES: Dict[str, AllergyRule] = {
    "penicillin": AllergyRule(
        "penicillin", frozenset({"penicillin"}),
        {"cephalosporin": "Cross-reactivity with cephalosporins is about 1–2% (dissimilar side chains lower it further); "
                          "most guidelines allow use with monitoring unless the reaction was anaphylaxis."}),
    "sulfa": AllergyRule(
        "sulfonamide antibiotics", frozenset({"sulfonamide_antibiotic"}),
        {"sulfonamide_nonantibiotic": "Non-antibiotic sulfonamides (furosemide) show no meaningful cross-reactivity "
                                      "with a sulfonamide-antibiotic allergy."}),
    "vancomycin (red man)": AllergyRule(
        "vancomycin infusion reaction", frozenset(), {}, kind="intolerance"),
    "iodinated contrast": AllergyRule(
        "iodinated contrast", frozenset(), {}, kind="procedure"),
}

# Procedures that need iodinated contrast (a contrast allergy needs a premedication plan).
CONTRAST_PROCEDURES = ("catheterization", "ct with contrast", "angiography")


# --------------------------------------------------------------------------- #
# Curated pair rules. Each: (class or drug A, class or drug B) -> rule.
# A rule may carry a context exception: a diagnosis that makes the combination the
# expected regimen (heparin + aspirin in ACS), which lowers it to information.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PairRule:
    a: str
    b: str
    severity: str                       # info | watch | urgent | critical
    title: str
    why: str
    source: str
    expected_in: tuple = ()             # diagnosis keywords that make it the standard regimen


PAIR_RULES: List[PairRule] = [
    PairRule("anticoagulant", "anticoagulant", "urgent", "Two anticoagulants active",
             "Therapeutic duplication: two anticoagulants at once multiplies bleeding risk.",
             "ISMP high-alert medication list; CHEST antithrombotic guidance"),
    PairRule("anticoagulant", "antiplatelet", "watch", "Anticoagulant with an antiplatelet",
             "Combined antithrombotic therapy raises major-bleeding risk.",
             "CHEST antithrombotic guidance", expected_in=("nstemi", "acs", "rule out acs")),
    PairRule("beta_blocker", "ccb_nondhp", "watch", "Beta-blocker with diltiazem",
             "Additive AV-nodal blockade: bradycardia, heart block, hypotension.",
             "AHA/ACC/HRS 2023 atrial fibrillation guideline"),
    PairRule("opioid", "benzodiazepine", "watch", "Opioid with a benzodiazepine",
             "Additive respiratory depression and sedation.",
             "FDA boxed warning, 2016"),
    PairRule("k_sparing", "potassium", "watch", "Potassium-sparing diuretic with potassium",
             "Additive hyperkalaemia risk, worse with renal impairment.",
             "KDIGO; product labelling"),
]


def creatinine_clearance(age: float, weight_kg: float, creatinine: float, female: bool) -> float:
    """Cockcroft–Gault (1976), mL/min."""
    if creatinine <= 0:
        return 999.0
    return (140 - age) * weight_kg / (72 * creatinine) * (0.85 if female else 1.0)


def matches(token: str, drug: Drug) -> bool:
    return token == drug.name or token == drug.generic or token in drug.classes
