"""Units: what a feed writes in OBX-6, and how to get from it to AegisOS's canonical unit.

Hospitals do not agree on units, and they do not agree on how to WRITE the units they
agree on. Creatinine arrives as ``mg/dL``, ``MG/DL``, ``mg/dl`` or ``umol/L``; a white
count as ``K/uL``, ``10*3/uL``, ``x10^9/L`` or ``×10⁹/L``. This module does two things:

* ``norm(unit)`` reduces a written unit to one token (UCUM where UCUM has one, e.g.
  ``10*3/uL``, ``[degF]``, ``mm[Hg]``), or ``""`` when the field is empty and
  ``?<text>`` when the string is not recognised. It never guesses a dimension.
* ``generic(src, dst)`` converts between tokens of the same dimension when no molar mass
  is involved (kPa→mmHg, °F→°C, fraction→%, cells/µL→10³/µL, g/L→g/dL...). Conversions
  that DO depend on the analyte (glucose mmol/L→mg/dL is ×18.016, creatinine µmol/L→mg/dL
  is ÷88.42) live with the analyte in ``vocab``.

A conversion is affine, ``canonical = value × factor + offset`` (the offset is non-zero
only for temperature).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# Canonical tokens this module knows. Grouped by dimension; generic() converts within one.
DIMENSIONS: Dict[str, Tuple[str, ...]] = {
    "rate": ("/min",),
    "fraction": ("%", "1"),
    "temperature": ("Cel", "[degF]", "K"),
    "pressure": ("mm[Hg]", "kPa"),
    "count": ("10*3/uL", "/uL", "10*9/L"),
    "mass_conc": ("g/dL", "g/L", "mg/dL", "mg/L", "ng/mL", "ng/L"),
    "molar_conc": ("mmol/L", "umol/L", "meq/L"),
    "enzyme": ("U/L", "ukat/L"),
    "time": ("s",),
    "flow": ("L/min", "mL/min"),
    "pH": ("[pH]",),
}
DIMENSION_OF = {tok: dim for dim, toks in DIMENSIONS.items() for tok in toks}

_SUPERSCRIPT = str.maketrans({"⁰": "0", "¹": "1", "²": "2", "³": "3", "⁴": "4", "⁵": "5", "⁶": "6", "⁷": "7",
                              "⁸": "8", "⁹": "9", "µ": "u", "μ": "u", "×": "x", "°": "deg"})

# written form (lower-case, spaces removed, after _SUPERSCRIPT) -> token
_ALIASES: Dict[str, str] = {}


def _alias(token: str, *forms: str) -> None:
    for f in forms:
        _ALIASES[f] = token


_alias("/min", "/min", "1/min", "bpm", "beats/min", "{beats}/min", "breaths/min", "{breaths}/min", "br/min",
       "rpm", "permin", "/minute", "min-1", "{beat}/min", "{breath}/min")
_alias("%", "%", "percent", "pct", "%sat", "%o2", "{%}")
_alias("1", "1", "fraction", "frac", "ratio", "{ratio}", "l/l", "decimal")
_alias("Cel", "cel", "c", "degc", "dgc", "celsius", "degreesc", "centigrade")
_alias("[degF]", "[degf]", "f", "degf", "dgf", "fahrenheit", "degreesf")
_alias("K", "k(kelvin)", "kelvin")
_alias("mm[Hg]", "mm[hg]", "mmhg", "mm/hg", "torr", "mmofhg")
_alias("kPa", "kpa")
_alias("10*3/uL", "10*3/ul", "10^3/ul", "x10*3/ul", "x10^3/ul", "x103/ul", "10e3/ul", "x10e3/ul", "k/ul", "k/mcl",
       "thou/ul", "thou/mcl", "10*3/mm3", "10^3/mm3", "x10^3/mm3", "k/mm3", "10*3/mcl", "10^3/mcl", "103/ul",
       "thousand/ul", "x1000/ul")
_alias("/uL", "/ul", "cells/ul", "/mcl", "cells/mcl", "/mm3", "cells/mm3", "{cells}/ul", "#/ul", "cell/ul")
_alias("10*9/L", "10*9/l", "10^9/l", "x10*9/l", "x10^9/l", "x109/l", "10e9/l", "x10e9/l", "giga/l", "gpt/l",
       "109/l", "10**9/l")
_alias("g/dL", "g/dl", "gm/dl", "gms/dl", "g%", "gm%", "gram/dl", "g/100ml")
_alias("g/L", "g/l", "gm/l", "gram/l")
_alias("mg/dL", "mg/dl", "mg%", "mgs/dl", "mg/100ml", "milligram/dl")
_alias("mg/L", "mg/l")
_alias("ng/mL", "ng/ml", "ug/l", "mcg/l", "ng/cc")
_alias("ng/L", "ng/l", "pg/ml")
_alias("mmol/L", "mmol/l", "mm/l", "mmol", "millimol/l", "mmo/l", "mmol/litre")
_alias("umol/L", "umol/l", "micromol/l", "mcmol/l", "umo/l")
_alias("meq/L", "meq/l", "mequiv/l", "meq")
_alias("U/L", "u/l", "iu/l", "units/l", "unit/l", "[iu]/l", "{units}/l")
_alias("ukat/L", "ukat/l", "microkat/l")
_alias("s", "s", "sec", "secs", "second", "seconds", "sek")
_alias("L/min", "l/min", "lpm", "lit/min", "liters/min", "litres/min", "l/m")
_alias("mL/min", "ml/min")
_alias("[pH]", "[ph]", "ph", "phunits", "units(ph)")


def norm(unit: Optional[str]) -> str:
    """One token for a written unit; ``""`` for empty; ``?<cleaned>`` when unknown.

    One case is decided by case: ``G/L`` written in capitals is the European cell-count
    unit (giga per litre = 10⁹/L), while ``g/L`` is grams per litre. Everything else is
    case-insensitive."""
    if unit is None:
        return ""
    raw = unit.strip()
    if not raw:
        return ""
    if raw in ("G/L", "G/l", "Gpt/L"):
        return "10*9/L"
    s = raw.translate(_SUPERSCRIPT).lower().replace(" ", "")
    s = s.replace("**", "*").replace("deg.", "deg")
    if s in _ALIASES:
        return _ALIASES[s]
    # x10(9)/L, 10(3)/uL
    s2 = re.sub(r"\((\d)\)", r"^\1", s)
    if s2 in _ALIASES:
        return _ALIASES[s2]
    return "?" + raw


def dimension(token: str) -> Optional[str]:
    return DIMENSION_OF.get(token)


@dataclass(frozen=True)
class Conversion:
    """canonical = value × factor + offset."""
    factor: float = 1.0
    offset: float = 0.0
    label: str = "identity"

    def apply(self, v: float) -> float:
        return v * self.factor + self.offset

    def invert(self, v: float) -> float:
        return (v - self.offset) / self.factor

    @property
    def identity(self) -> bool:
        return self.factor == 1.0 and self.offset == 0.0

    def to_dict(self) -> Dict[str, object]:
        return {"factor": self.factor, "offset": self.offset, "label": self.label}


IDENTITY = Conversion()

# token -> (factor, offset) into the FIRST token of its dimension's base, per dimension
_TO_BASE: Dict[str, Tuple[float, float]] = {
    "/min": (1, 0),
    "%": (1, 0), "1": (100, 0),
    "Cel": (1, 0), "[degF]": (5 / 9, -32 * 5 / 9), "K": (1, -273.15),
    "mm[Hg]": (1, 0), "kPa": (7.50062, 0),
    "10*3/uL": (1, 0), "/uL": (0.001, 0), "10*9/L": (1, 0),
    "g/dL": (1, 0), "g/L": (0.1, 0), "mg/dL": (0.001, 0), "mg/L": (0.0001, 0), "ng/mL": (1e-7, 0), "ng/L": (1e-10, 0),
    "mmol/L": (1, 0), "umol/L": (0.001, 0), "meq/L": (1, 0),   # meq = mmol only for monovalent ions (vocab overrides)
    "U/L": (1, 0), "ukat/L": (60, 0),
    "s": (1, 0),
    "L/min": (1, 0), "mL/min": (0.001, 0),
    "[pH]": (1, 0),
}


def generic(src: str, dst: str) -> Optional[Conversion]:
    """A conversion between two tokens of one dimension that needs no molar mass, or None."""
    if src == dst:
        return IDENTITY
    d1, d2 = DIMENSION_OF.get(src), DIMENSION_OF.get(dst)
    if d1 is None or d1 != d2:
        return None
    a1, b1 = _TO_BASE[src]
    a2, b2 = _TO_BASE[dst]
    # base = v*a1 + b1 ; dst = (base - b2) / a2
    factor = a1 / a2
    offset = (b1 - b2) / a2
    return Conversion(round(factor, 12), round(offset, 9), f"{src} → {dst}")
