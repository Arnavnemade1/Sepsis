"""The canonical vocabulary: what AegisOS means by "creatinine", and how to get there.

Every inbound observation, from an HL7 feed or a FHIR server, ends up as one of these
concepts in one canonical unit, because that is what the detectors were built on:
the room monitor's 19 detectors read its own keys (``hr``, ``creatinine``, ``anc``...),
and the ICU detector was trained on the PhysioNet/CinC 2019 columns in their units.

Each concept carries:

* ``loinc``: the LOINC code a standards-speaking sender would use, so a feed that
  already codes in LOINC maps with no guessing (the mapping is still verified);
* ``unit``: the canonical unit token (``units.norm``) and the conversions that
  depend on THIS analyte (molar mass, valence, urea vs urea nitrogen);
* ``window``: the range a value can physiologically take in an adult. Not the normal
  range: a creatinine of 9 is possible, a creatinine of 90 is not. Fixed a priori from
  physiology and stated here, never fitted to data, so a dataset that is itself wrong
  cannot quietly widen it (see ``docs/INTEROP.md`` on the CinC calcium column);
* ``scale`` and ``band``: how far a hospital's population may legitimately sit from
  the reference (log units for skewed analytes; absolute units for temperature, pH,
  saturations and base excess). A unit error is a multiplicative jump (×2.43 for
  magnesium, ×18 for glucose, ×1000 for counts), larger than any population shift
  these bands allow;
* ``aliases`` and ``mdc``: what the lexical proposer matches against (local lab
  mnemonics, IEEE 11073 reference ids from monitor gateways).
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from backend.aegis.interop.units import IDENTITY, Conversion, generic, norm

REFERENCE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.json")


@dataclass(frozen=True)
class Concept:
    key: str
    label: str
    loinc: str
    loinc_display: str
    unit: str
    kind: str                                   # vital | lab | gas
    window: Tuple[float, float]
    room_key: Optional[str] = None              # vitals channel or labs key in a room
    cinc: Optional[str] = None                  # PhysioNet/CinC 2019 column
    scale: str = "log"                          # log | linear | none
    band: float = 0.47                          # allowed population shift on that scale
    molar: Tuple[Tuple[str, float, float], ...] = ()      # (unit token, factor, offset) analyte-specific
    variants: Tuple[Tuple[str, float, float], ...] = ()   # (label, factor, offset): analyte confusions
    aliases: Tuple[str, ...] = ()
    mdc: Tuple[str, ...] = ()

    def conversion(self, unit_token: str) -> Optional[Conversion]:
        """From a unit token to this concept's canonical unit, or None if no rule exists."""
        if unit_token == self.unit:
            return IDENTITY
        for tok, f, o in self.molar:
            if tok == unit_token:
                return Conversion(f, o, f"{unit_token} → {self.unit}")
        return generic(unit_token, self.unit)

    def candidate_conversions(self) -> List[Conversion]:
        """Every conversion the unit-inference search tries for this concept: the
        identity, each unit this analyte is known to be reported in, the analyte
        confusions, and decade shifts for a feed whose unit label cannot be trusted."""
        out: List[Conversion] = [IDENTITY]
        seen = {(1.0, 0.0)}

        def add(c: Optional[Conversion]) -> None:
            if c is None:
                return
            k = (round(c.factor, 9), round(c.offset, 6))
            if k not in seen:
                seen.add(k)
                out.append(c)

        for tok, f, o in self.molar:
            add(Conversion(f, o, f"{tok} → {self.unit}"))
        from backend.aegis.interop.units import DIMENSION_OF, DIMENSIONS
        dim = DIMENSION_OF.get(self.unit)
        for tok in DIMENSIONS.get(dim, ()):
            if tok != self.unit and not any(t == tok for t, _f, _o in self.molar):
                add(generic(tok, self.unit))
        for label, f, o in self.variants:
            add(Conversion(f, o, label))
        if self.scale == "log":
            for k in (-3, -2, -1, 1, 2, 3):
                add(Conversion(10.0 ** k, 0.0, f"decade shift ×10^{k}"))
        return out

    def to_dict(self) -> Dict[str, object]:
        return {"key": self.key, "label": self.label, "loinc": self.loinc, "loinc_display": self.loinc_display,
                "unit": self.unit, "kind": self.kind, "window": list(self.window), "room_key": self.room_key,
                "cinc": self.cinc, "scale": self.scale, "band": self.band}


V, L, G = "vital", "lab", "gas"
CONCEPTS: List[Concept] = [
    # ---- vital signs (monitor gateways) --------------------------------------------- #
    Concept("hr", "Heart rate", "8867-4", "Heart rate", "/min", V, (20, 300), "hr", "HR", band=0.30,
            aliases=("heart rate", "hr", "pulse", "pulse rate", "hr ecg", "ecg hr", "apical pulse", "pr"),
            mdc=("MDC_ECG_HEART_RATE", "MDC_PULS_RATE", "MDC_PULS_OXIM_PULS_RATE", "MDC_BLD_PULS_RATE_INV")),
    Concept("spo2", "SpO₂", "59408-5", "Oxygen saturation in Arterial blood by Pulse oximetry", "%", V, (40, 100),
            "spo2", "O2Sat", scale="linear", band=6.0,
            aliases=("spo2", "pulse ox", "pulse oximetry", "o2 sat", "sao2 pulse", "oxygen saturation", "sat", "pox"),
            mdc=("MDC_PULS_OXIM_SAT_O2", "MDC_SAT_O2")),
    Concept("temp", "Temperature", "8310-5", "Body temperature", "Cel", V, (25, 45), "temp_c", "Temp",
            scale="linear", band=1.5,
            aliases=("temperature", "temp", "body temp", "tympanic temp", "oral temp", "core temp", "tmp", "t"),
            mdc=("MDC_TEMP", "MDC_TEMP_BODY", "MDC_TEMP_ORAL", "MDC_TEMP_TYMP", "MDC_TEMP_CORE")),
    Concept("sbp", "Systolic BP", "8480-6", "Systolic blood pressure", "mm[Hg]", V, (40, 300), "sbp", "SBP", band=0.30,
            aliases=("systolic", "sbp", "nibp sys", "bp systolic", "art sys", "abp sys", "systolic bp", "sys"),
            mdc=("MDC_PRESS_BLD_NONINV_SYS", "MDC_PRESS_BLD_ART_SYS", "MDC_PRESS_BLD_ART_ABP_SYS", "MDC_PRESS_BLD_SYS")),
    Concept("dbp", "Diastolic BP", "8462-4", "Diastolic blood pressure", "mm[Hg]", V, (15, 200), "dbp", "DBP", band=0.30,
            aliases=("diastolic", "dbp", "nibp dia", "bp diastolic", "art dia", "abp dia", "diastolic bp", "dia"),
            mdc=("MDC_PRESS_BLD_NONINV_DIA", "MDC_PRESS_BLD_ART_DIA", "MDC_PRESS_BLD_ART_ABP_DIA", "MDC_PRESS_BLD_DIA")),
    Concept("map", "Mean arterial pressure", "8478-0", "Mean blood pressure", "mm[Hg]", V, (25, 250), "map", "MAP",
            band=0.30,
            aliases=("mean arterial pressure", "map", "nibp mean", "mean bp", "art mean", "abp mean", "mean"),
            mdc=("MDC_PRESS_BLD_NONINV_MEAN", "MDC_PRESS_BLD_ART_MEAN", "MDC_PRESS_BLD_ART_ABP_MEAN", "MDC_PRESS_BLD_MEAN")),
    Concept("rr", "Respiratory rate", "9279-1", "Respiratory rate", "/min", V, (2, 80), "rr", "Resp", band=0.30,
            aliases=("respiratory rate", "rr", "resp", "resp rate", "respirations", "breath rate", "awrr"),
            mdc=("MDC_RESP_RATE", "MDC_TTHOR_RESP_RATE", "MDC_AWAY_RESP_RATE", "MDC_CO2_RESP_RATE")),
    Concept("etco2", "End-tidal CO₂", "19889-5", "Carbon dioxide [Partial pressure] in Exhaled gas --at end expiration",
            "mm[Hg]", V, (3, 120), "etco2", "EtCO2", band=0.40,
            aliases=("etco2", "end tidal co2", "end-tidal co2", "petco2", "capnography"),
            mdc=("MDC_AWAY_CO2_ET", "MDC_CONC_AWAY_CO2_ET")),
    Concept("fio2", "FiO₂", "3150-0", "Inhaled oxygen concentration", "1", G, (0.21, 1.0), "fio2", "FiO2", band=0.60,
            aliases=("fio2", "fraction inspired oxygen", "inspired o2", "o2 concentration", "fio2 vent"),
            mdc=("MDC_CONC_AWAY_O2_INSP", "MDC_VENT_CONC_AWAY_O2_INSP")),
    Concept("o2_flow", "O₂ flow", "3151-8", "Inhaled oxygen flow rate", "L/min", V, (0, 70), "o2_lpm", None,
            scale="none", band=0.0,
            aliases=("o2 flow", "oxygen flow", "o2 lpm", "liter flow", "o2 l/min", "nasal cannula flow")),
    # ---- haematology ------------------------------------------------------------------ #
    Concept("wbc", "White cells", "6690-2", "Leukocytes [#/volume] in Blood by Automated count", "10*3/uL", L,
            (0.05, 500), "wbc", "WBC",
            aliases=("wbc", "white blood cells", "white count", "leukocytes", "wbc count", "leuk", "white cell count")),
    Concept("anc", "Neutrophils (absolute)", "751-8", "Neutrophils [#/volume] in Blood by Automated count", "/uL", L,
            (0, 300000), "anc", None, band=1.6,
            aliases=("anc", "absolute neutrophils", "neutrophils abs", "neut abs", "abs neut", "neutrophil count", "agc")),
    Concept("hemoglobin", "Haemoglobin", "718-7", "Hemoglobin [Mass/volume] in Blood", "g/dL", L, (2, 25),
            "hemoglobin", "Hgb", band=0.30, molar=(("mmol/L", 1.611, 0.0),),
            aliases=("hemoglobin", "haemoglobin", "hgb", "hb", "hemoglobin blood")),
    Concept("hematocrit", "Haematocrit", "4544-3", "Hematocrit [Volume Fraction] of Blood by Automated count", "%", L,
            (6, 75), "hct", "Hct", band=0.30,
            aliases=("hematocrit", "haematocrit", "hct", "pcv", "packed cell volume")),
    Concept("platelets", "Platelets", "777-3", "Platelets [#/volume] in Blood by Automated count", "10*3/uL", L,
            (1, 2500), "platelets", "Platelets", band=0.60,
            aliases=("platelets", "plt", "platelet count", "thrombocytes", "plts")),
    # ---- chemistry --------------------------------------------------------------------- #
    Concept("glucose", "Glucose", "2345-7", "Glucose [Mass/volume] in Serum or Plasma", "mg/dL", L, (10, 1500),
            "glucose", "Glucose", molar=(("mmol/L", 18.016, 0.0),),
            aliases=("glucose", "glu", "gluc", "blood sugar", "bg", "glucose serum", "glucose poc", "accucheck")),
    Concept("lactate", "Lactate", "2524-7", "Lactate [Moles/volume] in Serum or Plasma", "mmol/L", L, (0.1, 30),
            "lactate", "Lactate", band=0.90, molar=(("mg/dL", 1 / 9.008, 0.0), ("meq/L", 1.0, 0.0)),
            aliases=("lactate", "lactic acid", "lac", "lact", "lactate venous", "lactate arterial")),
    Concept("potassium", "Potassium", "2823-3", "Potassium [Moles/volume] in Serum or Plasma", "mmol/L", L, (1.0, 10.0),
            "potassium", "Potassium", band=0.20, aliases=("potassium", "k", "k+", "pot", "kalium", "potassium serum")),
    Concept("sodium", "Sodium", "2951-2", "Sodium [Moles/volume] in Serum or Plasma", "mmol/L", L, (95, 190),
            "sodium", None, band=0.10, aliases=("sodium", "na", "na+", "natrium", "sodium serum")),
    Concept("chloride", "Chloride", "2075-0", "Chloride [Moles/volume] in Serum or Plasma", "mmol/L", L, (60, 150),
            "chloride", "Chloride", band=0.10, aliases=("chloride", "cl", "cl-", "chlor")),
    Concept("hco3", "Bicarbonate", "1963-8", "Bicarbonate [Moles/volume] in Serum or Plasma", "mmol/L", L, (2, 60),
            "hco3", "HCO3", band=0.30,
            aliases=("bicarbonate", "hco3", "bicarb", "co2 total", "tco2", "total co2", "carbon dioxide")),
    Concept("bun", "Urea nitrogen", "3094-0", "Urea nitrogen [Mass/volume] in Serum or Plasma", "mg/dL", L, (1, 300),
            "bun", "BUN", band=0.60, molar=(("mmol/L", 2.801, 0.0),),
            variants=(("urea mass → urea nitrogen", 28.0 / 60.0, 0.0),),
            aliases=("bun", "urea nitrogen", "blood urea nitrogen", "urea", "un")),
    Concept("creatinine", "Creatinine", "2160-0", "Creatinine [Mass/volume] in Serum or Plasma", "mg/dL", L, (0.1, 25),
            "creatinine", "Creatinine", band=0.60, molar=(("umol/L", 1 / 88.42, 0.0), ("mmol/L", 1000 / 88.42, 0.0)),
            aliases=("creatinine", "creat", "cr", "crea", "creatinine serum", "scr")),
    Concept("calcium", "Calcium (total)", "17861-6", "Calcium [Mass/volume] in Serum or Plasma", "mg/dL", L,
            (3.5, 18), "calcium", "Calcium", band=0.20, molar=(("mmol/L", 4.008, 0.0), ("meq/L", 2.004, 0.0)),
            aliases=("calcium", "ca", "calcium total", "total calcium", "ca total")),
    Concept("ionized_calcium", "Calcium (ionized)", "1994-3", "Calcium.ionized [Moles/volume] in Blood", "mmol/L", L,
            (0.4, 3.0), "ica", None, band=0.20, molar=(("mg/dL", 1 / 4.008, 0.0), ("meq/L", 0.5, 0.0)),
            aliases=("ionized calcium", "ionised calcium", "ica", "ca ionized", "ca++", "ca2+", "free calcium")),
    Concept("magnesium", "Magnesium", "19123-9", "Magnesium [Mass/volume] in Serum or Plasma", "mg/dL", L, (0.4, 10),
            "magnesium", "Magnesium", band=0.30, molar=(("mmol/L", 2.431, 0.0), ("meq/L", 1.2155, 0.0)),
            aliases=("magnesium", "mg", "mag", "magnesium serum")),
    Concept("phosphate", "Phosphate", "2777-1", "Phosphate [Mass/volume] in Serum or Plasma", "mg/dL", L, (0.3, 20),
            "phosphate", "Phosphate", molar=(("mmol/L", 3.097, 0.0),),
            aliases=("phosphate", "phos", "po4", "phosphorus", "inorganic phosphate")),
    Concept("bilirubin", "Bilirubin (total)", "1975-2", "Bilirubin.total [Mass/volume] in Serum or Plasma", "mg/dL", L,
            (0.05, 60), "bilirubin", "Bilirubin_total", band=0.90, molar=(("umol/L", 1 / 17.104, 0.0),),
            aliases=("bilirubin total", "total bilirubin", "tbili", "t bili", "bili total", "bilirubin")),
    Concept("bilirubin_direct", "Bilirubin (direct)", "1968-7", "Bilirubin.direct [Mass/volume] in Serum or Plasma",
            "mg/dL", L, (0.0, 40), "bilirubin_direct", "Bilirubin_direct", band=0.90,
            molar=(("umol/L", 1 / 17.104, 0.0),),
            aliases=("bilirubin direct", "direct bilirubin", "dbili", "d bili", "conjugated bilirubin")),
    Concept("ast", "AST", "1920-8", "Aspartate aminotransferase [Enzymatic activity/volume] in Serum or Plasma", "U/L", L,
            (3, 30000), "ast", "AST", band=0.90, aliases=("ast", "sgot", "aspartate aminotransferase", "got")),
    Concept("alp", "Alkaline phosphatase", "6768-6", "Alkaline phosphatase [Enzymatic activity/volume] in Serum or Plasma",
            "U/L", L, (5, 5000), "alp", "Alkalinephos", band=0.90,
            aliases=("alkaline phosphatase", "alk phos", "alp", "alkp", "alk phosphatase")),
    Concept("troponin", "Troponin I", "10839-9", "Troponin I.cardiac [Mass/volume] in Serum or Plasma", "ng/mL", L,
            (0, 1000), "troponin", "TroponinI", band=1.6,
            aliases=("troponin i", "troponin", "tni", "ctni", "trop i", "trop", "hs troponin", "hstni")),
    Concept("ptt", "aPTT", "3173-2", "aPTT in Blood by Coagulation assay", "s", L, (10, 250), "ptt", "PTT",
            aliases=("ptt", "aptt", "partial thromboplastin time", "activated ptt")),
    Concept("inr", "INR", "6301-6", "INR in Platelet poor plasma by Coagulation assay", "1", L, (0.5, 20), "inr", None,
            aliases=("inr", "international normalized ratio", "pt inr", "protime inr")),
    Concept("fibrinogen", "Fibrinogen", "3255-7", "Fibrinogen [Mass/volume] in Platelet poor plasma by Coagulation assay",
            "mg/dL", L, (20, 2000), "fibrinogen", "Fibrinogen", band=0.60, molar=(("umol/L", 34.0, 0.0),),
            aliases=("fibrinogen", "fib", "fibrinogen level")),
    # ---- blood gas ---------------------------------------------------------------------- #
    Concept("ph", "pH", "2744-1", "pH of Arterial blood", "[pH]", G, (6.5, 8.0), "ph", "pH", scale="linear", band=0.15,
            aliases=("ph", "ph arterial", "arterial ph", "abg ph", "blood ph")),
    Concept("paco2", "PaCO₂", "2019-8", "Carbon dioxide [Partial pressure] in Arterial blood", "mm[Hg]", G, (8, 150),
            "paco2", "PaCO2", band=0.35,
            aliases=("paco2", "pco2", "pco2 arterial", "arterial pco2", "abg pco2", "carbon dioxide partial pressure")),
    Concept("sao2", "SaO₂", "2708-6", "Oxygen saturation in Arterial blood", "%", G, (30, 100), "sao2", "SaO2",
            scale="linear", band=8.0,
            aliases=("sao2", "arterial o2 sat", "o2 sat arterial", "abg sao2", "oxygen saturation arterial")),
    Concept("base_excess", "Base excess", "1925-7", "Base excess in Arterial blood by calculation", "mmol/L", G,
            (-35, 35), "base_excess", "BaseExcess", scale="linear", band=6.0,
            aliases=("base excess", "be", "base exc", "abe", "sbe", "base deficit")),
]
BY_KEY: Dict[str, Concept] = {c.key: c for c in CONCEPTS}
BY_LOINC: Dict[str, Concept] = {c.loinc: c for c in CONCEPTS}
BY_CINC: Dict[str, Concept] = {c.cinc: c for c in CONCEPTS if c.cinc}


# --------------------------------------------------------------------------- #
# Reference distributions (built by backend/ml/scripts/interop_reference.py)
# --------------------------------------------------------------------------- #
@dataclass
class Reference:
    key: str
    n: int
    source: str
    quantiles: Dict[float, float]
    loc: float                       # median on the concept's scale
    spread: float                    # robust sd on the concept's scale ((Q75 - Q25) / 1.349)
    tail: Tuple[float, float]        # Q0.5% and Q99.5% (raw units)

    def to_dict(self) -> Dict[str, object]:
        return {"key": self.key, "n": self.n, "source": self.source, "loc": self.loc, "spread": self.spread,
                "tail": list(self.tail), "quantiles": {str(k): v for k, v in sorted(self.quantiles.items())}}


def transform(c: Concept, v: float) -> Optional[float]:
    """A value on the concept's comparison scale (log for skewed analytes)."""
    if c.scale == "log":
        return math.log(v) if v > 0 else None
    return v


_REFS: Optional[Dict[str, Reference]] = None
_REF_META: Dict[str, object] = {}


def references() -> Dict[str, Reference]:
    global _REFS, _REF_META
    if _REFS is None:
        refs: Dict[str, Reference] = {}
        try:
            with open(REFERENCE_PATH) as fh:
                data = json.load(fh)
            _REF_META = data.get("meta") or {}
            for k, r in (data.get("concepts") or {}).items():
                refs[k] = Reference(k, int(r["n"]), r["source"], {float(p): float(v) for p, v in r["quantiles"].items()},
                                    float(r["loc"]), float(r["spread"]), (float(r["tail"][0]), float(r["tail"][1])))
        except (OSError, ValueError, KeyError):
            refs = {}
        _REFS = refs
    return _REFS


def reference_meta() -> Dict[str, object]:
    references()
    return dict(_REF_META)


def concept(key: str) -> Optional[Concept]:
    return BY_KEY.get(key)


UNITLESS_BY_CONVENTION = ("ph", "inr")     # a feed that writes no unit for these means the canonical one


def resolve_unit(c: Concept, written: str) -> Tuple[str, Optional[Conversion]]:
    """(token, conversion to canonical) for a unit as written in the message."""
    tok = norm(written)
    if not tok:
        if c.key in UNITLESS_BY_CONVENTION:
            return c.unit, IDENTITY
        return tok, None
    return tok, c.conversion(tok)
