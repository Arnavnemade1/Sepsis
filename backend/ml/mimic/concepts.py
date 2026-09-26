"""The MIMIC-IV variable catalog.

Every itemid below was checked against MIMIC's own dictionaries (``d_items``,
``d_labitems``); ``verify_mimic.py`` re-checks each label against the pattern recorded
here, so a wrong itemid fails the build instead of silently measuring something else.

Names follow CinC 2019 wherever the concept exists there (``HR``, ``Lactate``,
``Bilirubin_total`` ...), so the GENESIS feature engine and its composites run unchanged;
variables CinC never had (``GCS``, ``UrineOutput``, the vasopressors, ``PaO2``, ``INR`` ...)
extend it.

TIERS — what a model is allowed to read (the Epic-Sepsis-Model lesson):

    P  physiology   vitals, labs, GCS, urine output, CVP — measurements of the patient —
                    and what was known about them on arrival (admission source, service,
                    comorbidities from PRIOR admissions)
    T  treatment    vasopressors, ventilation and its settings, fluids, blood products,
                    sedation, dialysis — clinicians' response to the patient
                    (vasopressors also define SOFA's cardiovascular score)
    S  suspicion    cultures and antibiotics (orders and IV administrations) — the orders
                    that DEFINE Sepsis-3's suspected-infection time; a model reading them
                    partly detects the label

A tier-P model is the honest physiology number; T and S are reported beside it, never
instead of it.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

TIER_ORDER = {"P": 0, "T": 1, "S": 2}


@dataclass(frozen=True)
class Concept:
    name: str
    kind: str                       # chart | lab | lab_text | uo | vaso | vent | amount | sedation | dialysis | culture | abx | abx_admin
    itemids: Tuple[int, ...] = ()
    tier: str = "P"
    lo: float = -np.inf             # plausibility window (values outside are dropped)
    hi: float = np.inf
    label_pattern: str = ""         # regex the dictionary label must match (verify_mimic)
    unit: str = ""
    #: for results charted as text (urinalysis): text -> ordinal value; anything else dropped
    text_map: Tuple[Tuple[str, float], ...] = ()


def _c(name, kind, itemids, tier="P", lo=-np.inf, hi=np.inf, pat="", unit="", text_map=()):
    return Concept(name, kind, tuple(itemids), tier, lo, hi, pat, unit, tuple(text_map))


# --------------------------------------------------------------------------- #
# chartevents — vitals, FiO2, GCS
# --------------------------------------------------------------------------- #
CHART: List[Concept] = [
    _c("HR", "chart", [220045], lo=0, hi=300, pat=r"^Heart Rate$", unit="bpm"),
    _c("SBP", "chart", [220050, 220179], lo=0, hi=400, pat=r"systolic", unit="mmHg"),
    _c("DBP", "chart", [220051, 220180], lo=0, hi=300, pat=r"diastolic", unit="mmHg"),
    _c("MAP", "chart", [220052, 220181], lo=0, hi=300, pat=r"mean", unit="mmHg"),
    _c("Resp", "chart", [220210, 224690], lo=0, hi=80, pat=r"^Respiratory Rate", unit="insp/min"),
    _c("O2Sat", "chart", [220277], lo=0, hi=100, pat=r"O2 saturation pulseoxymetry", unit="%"),
    _c("TempF", "chart", [223761], lo=77, hi=113, pat=r"Temperature Fahrenheit", unit="F"),
    _c("TempC", "chart", [223762], lo=25, hi=45, pat=r"Temperature Celsius", unit="C"),
    _c("EtCO2", "chart", [228640], lo=0, hi=100, pat=r"^EtCO2$", unit="mmHg"),
    _c("FiO2", "chart", [223835], lo=0.21, hi=100, pat=r"Inspired O2 Fraction", unit="%"),
    _c("GCS_eye", "chart", [220739], lo=1, hi=4, pat=r"GCS - Eye"),
    _c("GCS_verbal", "chart", [223900], lo=0, hi=5, pat=r"GCS - Verbal"),
    _c("GCS_motor", "chart", [223901], lo=1, hi=6, pat=r"GCS - Motor"),
]

# --------------------------------------------------------------------------- #
# labevents
# --------------------------------------------------------------------------- #
LAB: List[Concept] = [
    _c("Lactate", "lab", [50813, 52442], lo=0, hi=50, pat=r"^Lactate$", unit="mmol/L"),
    _c("WBC", "lab", [51300, 51301], lo=0, hi=1000, pat=r"WBC|White Blood Cells", unit="K/uL"),
    _c("Platelets", "lab", [51265], lo=0, hi=2000, pat=r"Platelet Count", unit="K/uL"),
    _c("Creatinine", "lab", [50912], lo=0, hi=50, pat=r"^Creatinine$", unit="mg/dL"),
    _c("Bilirubin_total", "lab", [50885], lo=0, hi=100, pat=r"Bilirubin, Total", unit="mg/dL"),
    _c("BUN", "lab", [51006], lo=0, hi=300, pat=r"Urea Nitrogen", unit="mg/dL"),
    _c("HCO3", "lab", [50882], lo=0, hi=70, pat=r"Bicarbonate", unit="mEq/L"),
    _c("pH", "lab", [50820], lo=6.5, hi=8.0, pat=r"^pH$"),
    _c("PaCO2", "lab", [50818], lo=5, hi=200, pat=r"^pCO2$", unit="mmHg"),
    _c("PaO2", "lab", [50821], lo=10, hi=800, pat=r"^pO2$", unit="mmHg"),
    _c("SaO2", "lab", [50817], lo=0, hi=100, pat=r"Oxygen Saturation", unit="%"),
    _c("BaseExcess", "lab", [50802], lo=-40, hi=40, pat=r"Base Excess", unit="mEq/L"),
    _c("Glucose", "lab", [50931, 50809], lo=0, hi=2000, pat=r"^Glucose$", unit="mg/dL"),
    _c("Potassium", "lab", [50971, 50822], lo=0, hi=15, pat=r"Potassium", unit="mEq/L"),
    _c("Sodium", "lab", [50983, 50824], lo=80, hi=200, pat=r"Sodium", unit="mEq/L"),
    _c("Chloride", "lab", [50902, 50806], lo=50, hi=200, pat=r"Chloride", unit="mEq/L"),
    _c("Hgb", "lab", [51222, 50811], lo=0, hi=30, pat=r"Hemoglobin", unit="g/dL"),
    _c("Hct", "lab", [51221, 50810], lo=0, hi=80, pat=r"Hematocrit", unit="%"),
    _c("INR", "lab", [51237], lo=0, hi=30, pat=r"INR"),
    _c("PTT", "lab", [51275], lo=0, hi=200, pat=r"^PTT$", unit="sec"),
    _c("Albumin", "lab", [50862], lo=0, hi=10, pat=r"^Albumin$", unit="g/dL"),
    _c("AST", "lab", [50878], lo=0, hi=20000, pat=r"AST", unit="IU/L"),
    _c("ALT", "lab", [50861], lo=0, hi=20000, pat=r"ALT", unit="IU/L"),
    _c("Alkalinephos", "lab", [50863], lo=0, hi=5000, pat=r"Alkaline Phosphatase", unit="IU/L"),
    _c("TroponinT", "lab", [51003], lo=0, hi=100, pat=r"Troponin T", unit="ng/mL"),
    _c("Magnesium", "lab", [50960], lo=0, hi=10, pat=r"Magnesium", unit="mg/dL"),
    _c("Calcium", "lab", [50893], lo=0, hi=20, pat=r"Calcium, Total", unit="mg/dL"),
    _c("Phosphate", "lab", [50970], lo=0, hi=30, pat=r"Phosphate", unit="mg/dL"),
    _c("Fibrinogen", "lab", [51214], lo=0, hi=2000, pat=r"Fibrinogen", unit="mg/dL"),
    _c("Bands", "lab", [51144], lo=0, hi=100, pat=r"^Bands$", unit="%"),
]
#: v3: inflammation, the differential and organ injury beyond SOFA. Left out on purpose:
#: D-dimer (itemids charted in two unit systems, ng/mL and ng/mL FEU) and one absolute
#: lymphocyte itemid in #/uL beside the K/uL one — mixing units would be noise
LAB_V3: List[Concept] = [
    _c("CRP", "lab", [50889], lo=0, hi=1000, pat=r"^C-Reactive Protein$", unit="mg/L"),
    _c("Neutrophils", "lab", [51256], lo=0, hi=100, pat=r"^Neutrophils$", unit="%"),
    _c("Lymphocytes", "lab", [51244], lo=0, hi=100, pat=r"^Lymphocytes$", unit="%"),
    _c("ImmGran", "lab", [52135], lo=0, hi=100, pat=r"^Immature Granulocytes$", unit="%"),
    _c("Eosinophils", "lab", [51200], lo=0, hi=100, pat=r"^Eosinophils$", unit="%"),
    _c("AbsNeutrophils", "lab", [52075], lo=0, hi=500, pat=r"^Absolute Neutrophil Count$", unit="K/uL"),
    _c("AbsLymphocytes", "lab", [51133], lo=0, hi=500, pat=r"^Absolute Lymphocyte Count$", unit="K/uL"),
    _c("AnionGap", "lab", [50868, 52500], lo=0, hi=80, pat=r"^Anion Gap$", unit="mEq/L"),
    _c("Ferritin", "lab", [50924], lo=0, hi=200000, pat=r"^Ferritin$", unit="ng/mL"),
    _c("LDH", "lab", [50954], lo=0, hi=50000, pat=r"^Lactate Dehydrogenase", unit="IU/L"),
    _c("CK", "lab", [50910], lo=0, hi=500000, pat=r"^Creatine Kinase \(CK\)$", unit="IU/L"),
    _c("Lipase", "lab", [50956], lo=0, hi=50000, pat=r"^Lipase$", unit="IU/L"),
    _c("Ammonia", "lab", [50866], lo=0, hi=2000, pat=r"^Ammonia$", unit="umol/L"),
    _c("NTproBNP", "lab", [50963], lo=0, hi=300000, pat=r"^NTproBNP$", unit="pg/mL"),
]
LAB += LAB_V3

#: urinalysis — charted as text (a urinary source of infection); ordinal-coded
LAB_TEXT: List[Concept] = [
    _c("UrineLeuk", "lab_text", [51486], lo=0, hi=3, pat=r"^Leukocytes$",
       text_map=(("NEG", 0.0), ("TR", 0.5), ("SM", 1.0), ("MOD", 2.0), ("LG", 3.0))),
    _c("UrineNitrite", "lab_text", [51487], lo=0, hi=1, pat=r"^Nitrite$",
       text_map=(("NEG", 0.0), ("POS", 1.0))),
]

# --------------------------------------------------------------------------- #
# chartevents (v3) — central venous pressure (P); ventilator settings (T)
# --------------------------------------------------------------------------- #
HEMO: List[Concept] = [
    _c("CVP", "chart", [220074], lo=-10, hi=60, pat=r"^Central Venous Pressure$", unit="mmHg"),
]
VENT_SET: List[Concept] = [
    _c("PEEP", "chart", [220339, 224700], "T", 0, 40, r"PEEP", "cmH2O"),
    _c("TidalVolume", "chart", [224685], "T", 0, 2500, r"^Tidal Volume \(observed\)$", "mL"),
    _c("MinuteVolume", "chart", [224687], "T", 0, 60, r"^Minute Volume$", "L/min"),
    _c("PeakPressure", "chart", [224695], "T", 0, 120, r"^Peak Insp. Pressure$", "cmH2O"),
]
#: every chartevents concept the binning reads
CHART_ALL: List[Concept] = []    # filled below, after CHART

# --------------------------------------------------------------------------- #
# outputevents — urine output (mimic-code urine_output concept; GU irrigant IN subtracts)
# --------------------------------------------------------------------------- #
UO_ITEMS: Tuple[int, ...] = (226559, 226560, 226561, 226584, 226563, 226564, 226565,
                             226567, 226557, 226558, 227489)
UO_IRRIGANT_IN = 227488
UO = _c("UrineOutput", "uo", list(UO_ITEMS) + [UO_IRRIGANT_IN], lo=-5000, hi=5000,
        pat=r"Foley|Void|Condom|Ileoconduit|Suprapubic|Nephrostomy|Straight Cath|Ureteral|GU Irrigant",
        unit="mL")

# --------------------------------------------------------------------------- #
# inputevents — vasopressor infusion rates (tier T)
# --------------------------------------------------------------------------- #
VASO: List[Concept] = [
    _c("Norepinephrine", "vaso", [221906], "T", 0, 10, r"^Norepinephrine$", "mcg/kg/min"),
    _c("Epinephrine", "vaso", [221289], "T", 0, 10, r"^Epinephrine$", "mcg/kg/min"),
    _c("Dopamine", "vaso", [221662], "T", 0, 100, r"^Dopamine$", "mcg/kg/min"),
    _c("Dobutamine", "vaso", [221653], "T", 0, 100, r"^Dobutamine$", "mcg/kg/min"),
    _c("Phenylephrine", "vaso", [221749], "T", 0, 50, r"^Phenylephrine", "mcg/kg/min"),
    _c("Vasopressin", "vaso", [222315], "T", 0, 20, r"^Vasopressin$", "units/hour"),
]

# --------------------------------------------------------------------------- #
# procedureevents — ventilation (tier T)
# --------------------------------------------------------------------------- #
VENT: List[Concept] = [
    _c("VentInvasive", "vent", [225792], "T", pat=r"^Invasive Ventilation$"),
    _c("VentNIV", "vent", [225794], "T", pat=r"^Non-invasive Ventilation$"),
]

# --------------------------------------------------------------------------- #
# inputevents / procedureevents (v3) — the rest of the treatment (tier T)
# --------------------------------------------------------------------------- #
#: volume STARTED in the hour (mL): resuscitation fluids and blood products
TREAT_AMOUNT: List[Concept] = [
    _c("Crystalloid", "amount", [225158, 225828], "T", 0, 10000, r"^(NaCl 0.9%|LR)$", "mL"),
    _c("BloodProducts", "amount", [225168, 225170, 220970], "T", 0, 5000,
       r"^(Packed Red Blood Cells|Platelets|Fresh Frozen Plasma)$", "mL"),
]
#: 1 for every hour an administration interval covers
TREAT_INTERVAL: List[Concept] = [
    _c("Sedation", "sedation", [222168, 221668, 221744, 225942, 225150, 221712, 221385], "T",
       pat=r"^(Propofol|Midazolam|Fentanyl|Dexmedetomidine|Ketamine|Lorazepam)"),
    _c("Dialysis", "dialysis", [225802, 225441, 225805], "T",
       pat=r"^(Dialysis - CRRT|Hemodialysis|Peritoneal Dialysis)$"),
]

# --------------------------------------------------------------------------- #
# suspicion of infection (tier S): microbiologyevents + prescriptions
# --------------------------------------------------------------------------- #
CULTURE = _c("Culture", "culture", [], tier="S")
ANTIBIOTIC = _c("Antibiotic", "abx", [], tier="S")
#: IV antibiotics as ADMINISTERED at the bedside (inputevents), beside the pharmacy orders
#: the Sepsis-3 label reads (prescriptions): an input channel, never part of the label
ABX_ADMIN = _c("AbxAdmin", "abx_admin", [], tier="S")
ABX_ADMIN_CATEGORY = "08-Antibiotics (IV)"

#: Drug-name substrings of systemic antibiotics (after the MIT-LCP mimic-code
#: ``antibiotic`` concept, plus newer agents). Matched case-insensitively.
ABX_NAMES: Tuple[str, ...] = (
    "adoxa", "ala-tet", "alodox", "amikacin", "amikin", "amoxicill", "ampicillin", "augmentin",
    "avelox", "avidoxy", "avycaz", "azactam", "azithromycin", "aztreonam", "axetil", "bactocill",
    "bactrim", "bethkis", "biaxin", "bicillin", "cayston", "cefazolin", "cedax", "cefoxitin",
    "ceftazidime", "cefaclor", "cefadroxil", "cefdinir", "cefditoren", "cefepime", "cefiderocol",
    "cefotan", "cefotetan", "cefotaxime", "ceftaroline", "cefpodoxime", "cefpirome", "cefprozil",
    "ceftibuten", "ceftin", "ceftolozane", "ceftriaxone", "cefuroxime", "cephalexin",
    "cephalothin", "cephapririn", "chloramphenicol", "cipro", "claforan", "clarithromycin",
    "cleocin", "clindamycin", "colistin", "cubicin", "dalbavancin", "daptomycin", "dicloxacillin",
    "dificid", "doryx", "doxycycline", "duricef", "dynacin", "ertapenem", "ery-tab", "eryped",
    "eryc", "erythrocin", "erythromycin", "factive", "flagyl", "fortaz", "fosfomycin",
    "furadantin", "garamycin", "gentamicin", "imipenem", "kanamycin", "keflex", "kefzol", "ketek",
    "levaquin", "levofloxacin", "lincocin", "linezolid", "macrobid", "macrodantin", "maxipime",
    "mefoxin", "meropenem", "methicillin", "metronidazole", "minocin", "minocycline", "monodox",
    "monurol", "morgidox", "moxatag", "moxifloxacin", "myrac", "nafcillin", "neomycin",
    "nitrofurantoin", "norfloxacin", "noroxin", "ocudox", "ofloxacin", "omnicef", "oracea",
    "oraxyl", "oritavancin", "oxacillin", "pc pen vk", "pce dispertab", "penicillin",
    "periostat", "piperacillin", "polymyxin", "primsol", "proquin", "raniclor", "rifadin",
    "rifampin", "rocephin", "smz-tmp", "septra", "solodyn", "spectracef", "streptomycin",
    "sulfadiazine", "sulfamethoxazole", "sulfameth", "sulfatrim", "sulfisoxazole", "suprax",
    "synercid", "tazicef", "tazobactam", "tedizolid", "telavancin", "tetracycline", "tigecycline",
    "timentin", "tobramycin", "trimethoprim", "unasyn", "vancocin", "vancomycin", "vantin",
    "vibativ", "vibra-tabs", "vibramycin", "zerbaxa", "zinacef", "zithromax", "zosyn", "zyvox",
)
#: Non-systemic routes (eyes, ears, skin) and preparations that are not treatment of an
#: infection being suspected — excluded exactly as mimic-code does, plus the eye/irrigation
#: routes that appear in MIMIC-IV.
ABX_EXCLUDE_ROUTES = frozenset({"OU", "OS", "OD", "AU", "AS", "AD", "TP", "BOTH EYES",
                                "LEFT EYE", "RIGHT EYE", "IRR", "IR", "BOTH EARS",
                                "LEFT EAR", "RIGHT EAR", "EX", "TOPICAL"})
ABX_EXCLUDE_WORDS: Tuple[str, ...] = ("cream", "desensitization", "ophth", "oint", "gel",
                                      "irrigation", "enema")

#: Hard cap on the hourly grid: 14 days, the same horizon CinC 2019 used — applied to
#: EVERY stay, septic or not, so record length cannot carry the label.
MAX_HOURS = 336


CHART_ALL[:] = CHART + HEMO + VENT_SET


def all_concepts() -> List[Concept]:
    return CHART_ALL + LAB + LAB_TEXT + [UO] + VASO + VENT + TREAT_AMOUNT + TREAT_INTERVAL + \
        [CULTURE, ANTIBIOTIC, ABX_ADMIN]


def is_antibiotic(drug: str, route: Optional[str]) -> bool:
    d = str(drug).lower()
    r = str(route).upper().strip() if route is not None else ""
    if r in ABX_EXCLUDE_ROUTES:
        return False
    if any(w in d for w in ABX_EXCLUDE_WORDS):
        return False
    return any(n in d for n in ABX_NAMES)
