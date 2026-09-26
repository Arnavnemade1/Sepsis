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
    #: for results charted as text (urinalysis, the bedside exam): text -> ordinal value;
    #: matched after trimming and upper-casing; anything else is dropped
    text_map: Tuple[Tuple[str, float], ...] = ()
    #: how several readings in one hour combine: the last one, or the WORST (max) — for an
    #: exam charted on both sides (capillary refill L / R) or per lobe, the worst finding
    agg: str = "last"


def _c(name, kind, itemids, tier="P", lo=-np.inf, hi=np.inf, pat="", unit="", text_map=(), agg="last"):
    return Concept(name, kind, tuple(itemids), tier, lo, hi, pat, unit, tuple(text_map), agg)


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
# --------------------------------------------------------------------------- #
# chartevents (v4) — THE BEDSIDE EXAM, charted by nurses as text and read by almost no
# sepsis model: perfusion (capillary refill, mottling, cool clammy skin, weak pulses — the
# clinical signs of septic shock), infection sources (adventitious lung sounds, productive
# cough, cloudy urine, an acute abdomen) and encephalopathy (delirium screen, RASS, sluggish
# pupils). Ordinal-coded; the worst of several readings in an hour (both sides, every lobe).
# --------------------------------------------------------------------------- #
_W = "max"
EXAM: List[Concept] = [
    _c("CapRefill", "chart_text", [223951, 224308], lo=0, hi=1, pat=r"^Capillary Refill [LR]$", agg=_W,
       text_map=(("NORMAL <3 SECONDS", 0.0), ("ABNORMAL >3 SECONDS", 1.0))),
    _c("SkinMottled", "chart_text", [224028], lo=0, hi=3, pat=r"^Skin Color$", agg=_W,
       text_map=(("NORMAL FOR RACE", 0.0), ("FLUSHED", 0.0), ("JAUNDICED", 0.0), ("PALE", 1.0),
                 ("ASHEN", 2.0), ("DUSKY", 2.0), ("MOTTLED", 3.0), ("CYANOTIC", 3.0))),
    _c("SkinCool", "chart_text", [224027], lo=0, hi=2, pat=r"^Skin Temperature$", agg=_W,
       text_map=(("WARM", 0.0), ("HOT", 0.0), ("COOL", 1.0), ("COLD", 2.0))),
    _c("SkinClammy", "chart_text", [224860], lo=0, hi=1, pat=r"^Skin Condition$", agg=_W,
       text_map=(("DRY", 0.0), ("MOIST", 0.0), ("DIAPHORETIC", 1.0), ("CLAMMY", 1.0))),
    _c("PulseWeak", "chart_text", [223934, 223943, 223935, 223947, 223936, 223948], lo=0, hi=3,
       pat=r"(Dorsal PedPulse|PostTib\. Pulses|Radial Pulse) [LR]$", agg=_W,
       text_map=(("STRONG/PALPABLE", 0.0), ("EASILY PALPABLE", 0.0), ("WEAK PALPABLE", 1.0),
                 ("DIFFICULT TO PALPATE", 2.0), ("DOPPLER", 2.0), ("ABSENT", 3.0))),
    _c("LungSounds", "chart_text", [223986, 223987, 223988, 223989], lo=0, hi=2, pat=r"^[RL][UML]L Lung Sounds$",
       agg=_W, text_map=(("CLEAR", 0.0), ("DIMINISHED", 0.5), ("EXP WHEEZE", 0.5), ("INSP WHEEZE", 0.5),
                         ("INSP/EXP WHEEZE", 0.5), ("INS/EXP WHEEZE", 0.5), ("RHONCHI", 1.0), ("ABSENT", 1.0),
                         ("CRACKLES", 2.0), ("BRONCHIAL", 2.0), ("TUBULAR", 2.0))),
    _c("CoughProductive", "chart_text", [223992], lo=0, hi=1, pat=r"^Cough Type$", agg=_W,
       text_map=(("NON-PRODUCTIVE", 0.0), ("NON-PRODUCTIVE/CONGESTED", 0.5), ("CONGESTED", 0.5),
                 ("PRODUCTIVE", 1.0), ("PRODUCTIVE/CONGESTED", 1.0))),
    _c("UrineCloudy", "chart_text", [224876], lo=0, hi=1, pat=r"^Urine Appearance$", agg=_W,
       text_map=(("CLEAR", 0.0), ("CLOTS", 0.5), ("SEDIMENT", 1.0), ("CLOUDY", 1.0), ("SLUDGE", 1.0))),
    _c("AbdomenAcute", "chart_text", [224003], lo=0, hi=2, pat=r"^Abdominal Assessment$", agg=_W,
       text_map=(("SOFT", 0.0), ("NON-DISTENDED", 0.0), ("OBESE", 0.0), ("SOFT DISTENDED", 1.0), ("FIRM", 1.0),
                 ("ASCITES", 1.0), ("TENDER", 2.0), ("FIRM DISTENDED", 2.0))),
    _c("Delirium", "chart_text", [228332], lo=0, hi=1, pat=r"^Delirium assessment$", agg=_W,
       text_map=(("NEGATIVE", 0.0), ("POSITIVE", 1.0))),
    _c("PupilSluggish", "chart_text", [227121, 227288], lo=0, hi=2, pat=r"^Pupil Response (Right|Left)$", agg=_W,
       text_map=(("BRISK", 0.0), ("SLUGGISH", 1.0), ("NON-REACTIVE", 2.0))),
    _c("RASS", "chart", [228096], lo=-5, hi=4, pat=r"^Richmond-RAS Scale$"),
]
#: vitals whose within-hour EXTREME is kept beside the last value: 12.5% of demo MAP-hours
#: have several readings, and 205 of 2,169 hypotensive hours (9.5%) dip below 65 and recover
#: before the hour's last reading — invisible to a last-value grid. mimic-code's SOFA uses
#: the hour's lowest MAP too.
EXTREMES: Dict[str, str] = {"MAP": "min", "SBP": "min", "O2Sat": "min", "HR": "max", "Resp": "max", "Temp": "max"}
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


CHART_ALL[:] = CHART + HEMO + VENT_SET + EXAM

# --------------------------------------------------------------------------- #
# suspicion detail (v4, tier S): which workup, which drugs
# --------------------------------------------------------------------------- #
#: microbiology specimen groups; routine MRSA screens and serology / immunology are NOT an
#: acute infection workup, though the Sepsis-3 label (like mimic-code) counts every specimen
SPECIMENS: Dict[str, Tuple[str, ...]] = {
    "CultureBlood": ("BLOOD CULTURE",),
    "CultureUrine": ("URINE",),
    "CultureResp": ("SPUTUM", "BRONCHOALVEOLAR LAVAGE", "BRONCHIAL WASHINGS", "PLEURAL FLUID",
                    "TRACHEAL ASPIRATE"),
    "CultureRoutine": ("MRSA SCREEN", "SEROLOGY/BLOOD", "IMMUNOLOGY (CMV)", "BLOOD (CMV AB)"),
}
#: antibiotic spectrum from the drug name: anti-MRSA, and broad gram-negative / anti-pseudomonal
ABX_MRSA = ("vancomycin", "linezolid", "zyvox", "daptomycin", "cubicin", "ceftaroline", "tedizolid",
            "telavancin", "dalbavancin", "oritavancin")
ABX_BROAD = ("piperacillin", "zosyn", "tazobactam", "cefepime", "maxipime", "ceftazidime", "fortaz",
             "tazicef", "meropenem", "imipenem", "ertapenem", "aztreonam", "azactam", "cefiderocol",
             "ceftolozane", "zerbaxa", "avycaz", "tobramycin", "amikacin", "colistin", "polymyxin")


def abx_classes(drug: str) -> Tuple[bool, bool]:
    """(anti-MRSA, broad anti-pseudomonal) for one systemic antibiotic order. Oral vancomycin
    treats C. difficile, not MRSA bacteraemia — its route decides, in ``sources``."""
    d = str(drug).lower()
    return any(n in d for n in ABX_MRSA), any(n in d for n in ABX_BROAD)


def specimen_group(spec: str) -> str:
    s = str(spec).upper().strip()
    for name, specs in SPECIMENS.items():
        if s in specs:
            return name
    return "CultureOther"


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
