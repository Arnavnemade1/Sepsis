"""Evidence: does the data agree with what a mapping claims it is?

A mapping says "local code CREA, in umol/L, is creatinine, divide by 88.42". That is a
claim about data, and like every claim in AegisOS it is checked against the data
before anything acts on it. Whoever made the claim (a lexical matcher, a language
model, an integration analyst) cannot adopt it on their say-so. This module does the
checking, and the same checks keep running on live traffic after adoption.

Per mapping, over a window of converted values:

* **plausibility**: values outside the concept's physiological window (``vocab``,
  fixed a priori). A creatinine of 132 is not a creatinine in mg/dL.
* **location**: the window's median, on the concept's scale, against the reference
  median, with a 99.9% interval. FAIL when the whole interval sits outside the
  population band; PASS when the median sits inside it; UNSURE otherwise. The
  asymmetry is deliberate: a small sample can never refuse a mapping, and adoption
  also needs plausibility and tail mass to pass. Unit errors are multiplicative jumps
  bigger than any band, so they land in FAIL; a different hospital's patients land
  in PASS.
* **tail mass**: the fraction beyond the reference's 0.5–99.5% range. This catches
  a MIXTURE that barely moves the median: a feed sending ionized calcium (mmol/L) and
  total calcium (mg/dL) under one code.

**Values cluster by patient.** A small census is not a large sample: twelve values from
four patients are four draws from the population, not twelve. Given each value's room,
location and tail count patients (n_eff = distinct rooms): the location interval is over
the patients' medians, using the reference's spread until there are ``N_MIN`` patients to
estimate their own, and the tail is a patient-weighted fraction tested against
Binomial(n_eff). Two febrile-neutropenic patients among four then make the white count
UNSURE (not enough patients to refuse), where pooling the values refused it. A unit
error that leaves the physiological window is refused as fast as before (plausibility
stays per value); one that stays inside it needs more patients (docs/INTEROP.md measures
how many). Nothing is adopted meanwhile: an unsure mapping keeps learning.

Across mappings, over co-observed samples (one message, one blood gas):

* **physiological identities** (``INVARIANTS``): SBP > MAP > DBP; Hct/Hgb between 2.3
  and 3.8 (an MCHC of 26–43 g/dL); Henderson–Hasselbalch; SpO₂ against SaO₂; direct ≤
  total bilirubin; and the rest. These catch what no single-variable check can see:
  two fields swapped whose marginal distributions overlap (systolic and mean pressure),
  and they need a handful of samples where a distribution needs dozens.

**Unit inference.** When the declared unit fails, every conversion the analyte is
known to come in is tried (``Concept.candidate_conversions``). If exactly one passes,
the evidence names the fix ("these values are µmol/L"). The fix is a proposal too: a
person accepts it.

Every threshold here is fixed in advance (``docs/INTEROP.md``); the reference data
only supplies the reference distribution. The test statistics are exact binomial
tails and an order-statistic interval, with nothing fitted.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from backend.aegis.interop.units import Conversion
from backend.aegis.interop.vocab import BY_KEY, Concept, Reference, references, transform

ALPHA = 1e-4                 # a FAIL must be this improbable under a correct mapping
N_MIN = 12                   # values before a distribution check may PASS
N_MIN_INV = 5                # co-observed samples before an identity check may PASS
P_IMPLAUSIBLE = 0.02         # tolerated implausible fraction under a correct mapping
P_TAIL = 0.08                # tolerated fraction beyond the reference's 0.5–99.5% range
Z_CI = 3.29                  # 99.9% two-sided


def binom_sf(k: int, n: int, p: float) -> float:
    """P(X >= k) for X ~ Binomial(n, p), exactly, in log space."""
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0:
        return 0.0
    if p >= 1:
        return 1.0
    lp, lq = math.log(p), math.log1p(-p)
    terms = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
             for i in range(k, n + 1)]
    m = max(terms)
    return min(1.0, math.exp(m) * sum(math.exp(t - m) for t in terms))


@dataclass
class Check:
    name: str
    status: str                       # pass | fail | unsure | skip
    stat: Optional[float]
    limit: Optional[float]
    n: int
    detail: str

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "status": self.status,
                "stat": None if self.stat is None else round(self.stat, 4),
                "limit": None if self.limit is None else round(self.limit, 4), "n": self.n, "detail": self.detail}


@dataclass
class Fit:
    concept: str
    conversion: Conversion
    checks: List[Check]
    status: str                       # pass | fail | unsure | insufficient
    score: float                      # lower is a better fit (for ranking candidates)
    n: int
    median: Optional[float] = None    # converted, raw units
    n_eff: Optional[int] = None       # independent units behind the location and tail tests (patients)

    def check(self, name: str) -> Optional[Check]:
        return next((c for c in self.checks if c.name == name), None)

    def to_dict(self) -> Dict[str, object]:
        return {"concept": self.concept, "conversion": self.conversion.to_dict(), "status": self.status,
                "score": None if math.isinf(self.score) else round(self.score, 3), "n": self.n,
                "n_eff": self.n if self.n_eff is None else self.n_eff,
                "median": None if self.median is None else round(self.median, 4),
                "checks": [c.to_dict() for c in self.checks]}


def _robust_sd(xs: Sequence[float]) -> float:
    med = statistics.median(xs)
    mad = statistics.median(abs(x - med) for x in xs)
    return 1.4826 * mad


def _by_cluster(pairs: Sequence[Tuple[float, object]]) -> Dict[object, List[float]]:
    out: Dict[object, List[float]] = {}
    for x, k in pairs:
        out.setdefault(k, []).append(x)
    return out


def assess(c: Concept, values: Sequence[float], conv: Conversion,
           ref: Optional[Reference] = None, n_min: int = N_MIN,
           clusters: Optional[Sequence[object]] = None) -> Fit:
    """The single-variable evidence for ``values`` (as received) read through ``conv``.

    ``clusters`` gives, per value, the unit it came from (the room): values from one patient
    are not independent draws from a population, so the location and tail tests count
    patients, not values (n_eff = distinct clusters). Without it every value is its own unit
    and the tests are exactly the per-value ones."""
    ref = ref if ref is not None else references().get(c.key)
    keep = [i for i, v in enumerate(values) if v is not None and v == v]
    xs = [conv.apply(values[i]) for i in keep]
    cl: List[object] = [clusters[i] for i in keep] if clusters is not None else list(range(len(xs)))
    n = len(xs)
    checks: List[Check] = []
    if n == 0:
        return Fit(c.key, conv, [Check("data", "unsure", None, None, 0, "no values")], "insufficient", math.inf, 0,
                   n_eff=0)
    n_eff = len(set(cl))
    lo, hi = c.window
    # plausibility: per value (a body cannot produce these, however its values cluster)
    k = sum(1 for x in xs if x < lo or x > hi)
    p = binom_sf(k, n, P_IMPLAUSIBLE)
    st = "fail" if p < ALPHA else ("pass" if n >= n_min else "unsure")
    checks.append(Check("plausibility", st, k / n, P_IMPLAUSIBLE, n,
                        f"{k} of {n} outside the physiological window {lo:g}–{hi:g} {c.unit}"))
    med = statistics.median(xs)
    loc_ratio = math.inf
    tail_frac = None
    who = f" from {n_eff} patient{'s' if n_eff != 1 else ''}" if clusters is not None else ""
    if c.scale != "none" and ref is not None:
        tpairs = [(t, g) for t, g in ((transform(c, x), g) for x, g in zip(xs, cl)) if t is not None]
        ts = [t for t, _g in tpairs]
        if len(ts) >= 3:
            if clusters is None:
                m = statistics.median(ts)
                s = max(_robust_sd(ts), 0.25 * ref.spread, 1e-6)
                h = Z_CI * 1.2533 * s / math.sqrt(len(ts))
            else:
                # the median of the patients' medians, with an interval over PATIENTS. Below
                # N_MIN patients their spread cannot be estimated, so the reference's own spread
                # (which includes within-patient variation, so it is wide enough) stands in for it
                cm = [statistics.median(v) for v in _by_cluster(tpairs).values()]
                m = statistics.median(cm)
                s_obs = _robust_sd(cm) if len(cm) >= 2 else 0.0
                s = max(s_obs, (0.25 if len(cm) >= N_MIN else 1.0) * ref.spread, 1e-6)
                h = Z_CI * 1.2533 * s / math.sqrt(len(cm))
            d = m - ref.loc
            loc_ratio = abs(d) / c.band if c.band > 0 else math.inf
            # deliberately asymmetric: REFUSING needs the whole 99.9% interval outside the band
            # (a small sample cannot refuse); ADOPTING needs the point estimate inside it, with
            # plausibility and tail passing beside it. Unit errors land far outside any band.
            if abs(d) - h > c.band:
                lst = "fail"
            elif abs(d) <= c.band and len(ts) >= n_min:
                lst = "pass"
            else:
                lst = "unsure"
            scale = "×%.2f" % math.exp(d) if c.scale == "log" else "%+.2f %s" % (d, c.unit)
            shown = (math.exp(m) if c.scale == "log" else m) if clusters is not None else med
            checks.append(Check("location", lst, d, c.band, len(ts) if clusters is None else n_eff,
                                ("median" if clusters is None else "median of patient medians")
                                + f" {shown:.4g}{who} vs reference {_ref_median(c, ref):.4g} ({scale}; band "
                                + (f"×{math.exp(c.band):.2f}" if c.scale == "log" else f"±{c.band:g}") + ")"
                                + ("; too few patients to refuse" if lst == "unsure" and clusters is not None
                                   and abs(d) > c.band else "")))
        else:
            checks.append(Check("location", "unsure", None, c.band, len(ts), "too few positive values"))
        if c.scale == "linear" and len(ts) >= n_min:
            # on a linear scale a wrong factor can leave the median in place (base excess centred
            # on zero read x0.001 is still centred on zero) while crushing or inflating the spread;
            # log-scale concepts need no such check, since a factor only moves their location
            sd_obs = _robust_sd(ts)
            if sd_obs > 0:
                ratio = sd_obs / max(ref.spread, 1e-9)
                checks.append(Check("spread", "fail" if (ratio < 0.1 or ratio > 10) else "pass", ratio, None, len(ts),
                                    f"spread {sd_obs:.3g} vs reference {ref.spread:.3g} (×{ratio:.2f}; "
                                    f"allowed ×0.1–×10)"))
        beyond = [(1.0 if (x < ref.tail[0] or x > ref.tail[1]) else 0.0, g) for x, g in zip(xs, cl)]
        kt = int(sum(b for b, _g in beyond))
        if clusters is None:
            tail_frac, k_eff = kt / n, kt
        else:
            # each patient contributes the fraction of their values in the tail; the count of
            # patients is the binomial's n. A fraction in [0, 1] is a contraction of a coin with
            # the same mean, so this is the worst case for any within-patient correlation
            fr = [sum(v) / len(v) for v in _by_cluster(beyond).values()]
            tail_frac = sum(fr) / len(fr)
            k_eff = int(math.floor(tail_frac * n_eff + 1e-9))
        pt = binom_sf(k_eff, n_eff, P_TAIL)
        tst = "fail" if pt < ALPHA else ("pass" if n >= n_min else "unsure")
        checks.append(Check("tail", tst, tail_frac, P_TAIL, n_eff,
                            f"{kt} of {n}{who} beyond the reference 0.5–99.5% range {ref.tail[0]:.4g}–{ref.tail[1]:.4g}"
                            + (f" (a patient-weighted {tail_frac:.0%})" if clusters is not None else "")))
    elif ref is None and c.scale != "none":
        checks.append(Check("location", "skip", None, None, n, "no reference distribution for this concept"))
    statuses = [ch.status for ch in checks]
    if "fail" in statuses:
        status = "fail"
    elif n < n_min:
        status = "insufficient"
    elif all(s in ("pass", "skip") for s in statuses):
        status = "pass"
    else:
        status = "unsure"
    score = (loc_ratio if not math.isinf(loc_ratio) else 50.0) + 25.0 * max(0.0, k / n - P_IMPLAUSIBLE) \
        + 10.0 * max(0.0, (tail_frac or 0.0) - P_TAIL)
    return Fit(c.key, conv, checks, status, score, n, med, n_eff=n_eff)


def _ref_median(c: Concept, ref: Reference) -> float:
    return math.exp(ref.loc) if c.scale == "log" else ref.loc


@dataclass
class Inference:
    decision: str                     # declared | inferred | ambiguous | none | insufficient
    declared: Optional[Fit]
    best: Optional[Fit]
    passing: List[Fit] = field(default_factory=list)
    ranked: List[Fit] = field(default_factory=list)

    @property
    def conversion(self) -> Optional[Conversion]:
        if self.decision == "declared" and self.declared is not None:
            return self.declared.conversion
        if self.decision == "inferred" and self.best is not None:
            return self.best.conversion
        return None

    def to_dict(self) -> Dict[str, object]:
        return {"decision": self.decision,
                "declared": self.declared.to_dict() if self.declared else None,
                "best": self.best.to_dict() if self.best else None,
                "alternatives": [f.to_dict() for f in self.ranked[:4]]}


def infer(c: Concept, values: Sequence[float], declared: Optional[Conversion],
          ref: Optional[Reference] = None, n_min: int = N_MIN,
          clusters: Optional[Sequence[object]] = None) -> Inference:
    """Which reading of ``values`` does the data support for concept ``c``?

    The declared unit wins whenever it passes. Otherwise exactly one passing candidate
    is an inferred fix; several passing candidates are ambiguous (a person decides)."""
    ref = ref if ref is not None else references().get(c.key)
    d_fit = assess(c, values, declared, ref, n_min, clusters) if declared is not None else None
    n = d_fit.n if d_fit else sum(1 for v in values if v is not None and v == v)
    if n < n_min:
        return Inference("insufficient", d_fit, d_fit)
    fits = [assess(c, values, cv, ref, n_min, clusters) for cv in c.candidate_conversions()]
    if d_fit is not None and all(not (f.conversion.factor == d_fit.conversion.factor
                                      and f.conversion.offset == d_fit.conversion.offset) for f in fits):
        fits.append(d_fit)
    fits.sort(key=lambda f: f.score)
    passing = [f for f in fits if f.status == "pass"]
    if d_fit is not None and d_fit.status == "pass":
        # the written unit is confirmed only if the data can tell it apart from every other unit
        # this analyte is reported in: magnesium in mEq/L and in mg/dL differ by x1.22, inside its
        # population band, so values that fit one fit the other and a person must say which
        rivals = [f for f in passing if not f.conversion.label.startswith("decade shift")
                  and not (abs(f.conversion.factor - d_fit.conversion.factor) < 1e-9
                           and abs(f.conversion.offset - d_fit.conversion.offset) < 1e-9)]
        if rivals:
            return Inference("ambiguous", d_fit, d_fit, passing, fits)
        return Inference("declared", d_fit, d_fit, passing, fits)
    if d_fit is not None and d_fit.status == "unsure" and not any(
            f.status == "pass" and f.conversion is not d_fit.conversion for f in fits):
        return Inference("insufficient", d_fit, d_fit, passing, fits)
    # a named unit beats a decade guess: ÷88.42 (µmol/L) and ÷100 can both fit a creatinine,
    # and only one of them is a unit anyone reports creatinine in
    named = [f for f in passing if not f.conversion.label.startswith("decade shift")]
    pool = named or passing
    if len(pool) == 1:
        return Inference("inferred", d_fit, pool[0], passing, fits)
    if len(pool) > 1:
        return Inference("ambiguous", d_fit, pool[0], passing, fits)
    return Inference("none", d_fit, fits[0] if fits else None, passing, fits)


# --------------------------------------------------------------------------- #
# Physiological identities across co-observed samples
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Invariant:
    name: str
    concepts: Tuple[str, ...]
    violated: Callable[[Dict[str, float]], Optional[bool]]
    p0: float
    statement: str
    catches: str

    def to_dict(self) -> Dict[str, object]:
        return {"name": self.name, "concepts": list(self.concepts), "p0": self.p0,
                "statement": self.statement, "catches": self.catches}


def _hh(s: Dict[str, float]) -> Optional[bool]:
    ph, pco2, hco3 = s["ph"], s["paco2"], s["hco3"]
    if pco2 <= 0 or hco3 <= 0:
        return True
    return abs(ph - (6.1 + math.log10(hco3 / (0.0307 * pco2)))) > 0.12


INVARIANTS: List[Invariant] = [
    Invariant("bp_order", ("sbp", "map", "dbp"),
              lambda s: not (s["sbp"] > s["map"] > s["dbp"]), 0.02,
              "systolic > mean > diastolic",
              "swapped blood-pressure fields, whose ranges overlap too much for a distribution to tell apart"),
    Invariant("map_estimate", ("sbp", "dbp", "map"),
              lambda s: abs(s["map"] - (s["sbp"] + 2 * s["dbp"]) / 3) > 20, 0.02,
              "MAP within 20 mmHg of (SBP + 2·DBP) / 3",
              "a mean pressure field carrying something else (pulse pressure, another site)"),
    Invariant("hct_hgb", ("hematocrit", "hemoglobin"),
              lambda s: not (2.3 <= s["hematocrit"] / max(s["hemoglobin"], 1e-6) <= 3.8), 0.02,
              "Hct / Hgb between 2.3 and 3.8 (MCHC 26–43 g/dL)",
              "haemoglobin and haematocrit swapped, or haemoglobin in g/L or mmol/L"),
    Invariant("henderson_hasselbalch", ("ph", "paco2", "hco3"), _hh, 0.03,
              "pH within 0.12 of 6.1 + log₁₀(HCO₃ / (0.0307 · PaCO₂))",
              "PaCO₂ and bicarbonate swapped, or PaCO₂ in kPa read as mmHg"),
    Invariant("spo2_sao2", ("spo2", "sao2"),
              lambda s: abs(s["spo2"] - s["sao2"]) > 15, 0.02,
              "pulse oximetry within 15 points of arterial saturation (ISO 80601-2-61: Arms ≤ 4%)",
              "a venous or mixed-venous saturation recorded as arterial"),
    Invariant("bun_creatinine", ("bun", "creatinine"),
              lambda s: not (2 <= s["bun"] / max(s["creatinine"], 1e-6) <= 100), 0.02,
              "BUN / creatinine between 2 and 100",
              "creatinine in µmol/L read as mg/dL, or the two fields swapped"),
    Invariant("bilirubin_fraction", ("bilirubin_direct", "bilirubin"),
              lambda s: s["bilirubin_direct"] > s["bilirubin"] * 1.05 + 0.1, 0.01,
              "direct bilirubin ≤ total bilirubin",
              "direct and total bilirubin swapped"),
    Invariant("anc_wbc", ("anc", "wbc"),
              lambda s: s["anc"] / 1000.0 > s["wbc"] * 1.02 + 0.1, 0.01,
              "absolute neutrophils ≤ white cells",
              "an ANC sent in cells/µL under a 10³/µL label (read a thousand times too high)"),
    Invariant("anion_gap", ("sodium", "chloride", "hco3"),
              lambda s: not (0 <= s["sodium"] - s["chloride"] - s["hco3"] <= 35), 0.02,
              "anion gap Na − (Cl + HCO₃) between 0 and 35",
              "sodium carrying another analyte, or one of the three in the wrong unit (a Cl/HCO₃ swap "
              "leaves the gap unchanged; the per-variable checks catch that one)"),
    Invariant("van_slyke", ("base_excess", "hco3", "ph"),
              lambda s: abs(s["base_excess"] - 0.93 * (s["hco3"] - 24.4 + 14.8 * (s["ph"] - 7.4))) > 6, 0.02,
              "base excess within 6 of 0.93 · (HCO₃ − 24.4 + 14.8 · (pH − 7.4))",
              "base excess carrying bicarbonate, or the sign convention flipped (base deficit)"),
]
INVARIANT_BY_NAME = {i.name: i for i in INVARIANTS}


def invariants_for(keys: Sequence[str]) -> List[Invariant]:
    ks = set(keys)
    return [inv for inv in INVARIANTS if set(inv.concepts) <= ks]


def check_invariant(inv: Invariant, samples: Sequence[Dict[str, float]], n_min: int = N_MIN_INV) -> Check:
    """Violations of one identity over co-observed samples, against its a-priori rate."""
    tuples = [s for s in samples if all(k in s and s[k] is not None for k in inv.concepts)]
    n = len(tuples)
    if n == 0:
        return Check(f"invariant:{inv.name}", "skip", None, inv.p0, 0, "no co-observed samples")
    k = 0
    for s in tuples:
        try:
            if inv.violated(s):
                k += 1
        except (ZeroDivisionError, ValueError, KeyError):
            k += 1
    p = binom_sf(k, n, inv.p0)
    st = "fail" if (p < ALPHA and n >= 3) else ("pass" if n >= n_min else "unsure")
    return Check(f"invariant:{inv.name}", st, k / n, inv.p0, n,
                 f"{inv.statement}: violated in {k} of {n} co-observed samples")


LN2 = math.log(2.0)


def feed_baseline(c: Concept, canonical: Sequence[float]) -> Optional[Dict[str, float]]:
    """The feed's own distribution when its mapping was adopted: what live evidence
    compares against, so a dialysis unit's creatinine is judged against itself."""
    ts = [t for t in (transform(c, x) for x in canonical) if t is not None]
    if len(ts) < 3 or c.scale == "none":
        return None
    return {"loc": statistics.median(ts), "spread": max(_robust_sd(ts), 1e-6), "n": len(ts)}


def jump_tolerance(c: Concept) -> float:
    """How far a feed's median may move from its own baseline before it is a fault: twice
    the population band, and at least a doubling on a log scale. Clinical deterioration
    of a ward moves a median far less than that; the unit errors it must catch move it
    more (×2.43 magnesium, ×9 lactate, ×18 glucose, ×88 creatinine, ×1000 counts)."""
    return max(2 * c.band, LN2) if c.scale == "log" else 2 * c.band


def drift_checks(c: Concept, recent: Sequence[float], window: Sequence[float],
                 baseline: Optional[Dict[str, float]]) -> List[Check]:
    """Live evidence for an adopted mapping, on CANONICAL values. Only what stays true
    however sick the patients get: values a body cannot produce, and a jump of unit-error
    size against the feed's own baseline. (The identities run separately, across fields.)"""
    out: List[Check] = []
    lo, hi = c.window
    for name, xs in (("recent", list(recent)), ("window", list(window))):
        n = len(xs)
        if not n:
            continue
        k = sum(1 for x in xs if x < lo or x > hi)
        p = binom_sf(k, n, P_IMPLAUSIBLE)
        out.append(Check(f"plausibility:{name}", "fail" if p < ALPHA else "pass", k / n, P_IMPLAUSIBLE, n,
                         f"{k} of the last {n} outside the physiological window {lo:g}–{hi:g} {c.unit}"))
    if baseline is not None and c.scale != "none":
        tol = jump_tolerance(c)
        ref = references().get(c.key)
        for name, xs in (("recent", list(recent)), ("window", list(window))):
            ts = [t for t in (transform(c, x) for x in xs) if t is not None]
            if len(ts) < 8:
                continue
            m = statistics.median(ts)
            sd = max(_robust_sd(ts), 0.25 * baseline["spread"], 1e-6)
            h = Z_CI * 1.2533 * sd / math.sqrt(len(ts))
            d = m - baseline["loc"]
            # a unit error moves a feed away from its own baseline AND out of the population
            # band; a feed whose baseline happened to be unrepresentative (a dozen high early
            # bilirubins) moves back INTO the population, which is not a fault
            outside = ref is None or abs(m - ref.loc) > c.band
            shown = ("×%.2f" % math.exp(d)) if c.scale == "log" else ("%+.2f %s" % (d, c.unit))
            out.append(Check(f"jump:{name}", "fail" if (abs(d) - h > tol and outside) else "pass", d, tol, len(ts),
                             f"median of the last {len(ts)} moved {shown} from this feed's baseline (limit "
                             + (f"×{math.exp(tol):.2f}" if c.scale == "log" else f"±{tol:g} {c.unit}") + ")"
                             + ("" if outside else ", back inside the population band")))
    return out


# --------------------------------------------------------------------------- #
# Sequential identity monitoring (Page's CUSUM)
# --------------------------------------------------------------------------- #
CUSUM_P1 = 0.5          # what a swap or mis-mapping looks like: the identity fails half the time or more
CUSUM_H = 13.0          # simulated under each identity's a-priori violation rate: ~4 million clean rows
                        # between false alarms in the worst case (true rate = allowance); a swap needs 5


def cusum_identity(inv: "Invariant", samples: Sequence[Dict[str, float]], h: float = CUSUM_H) -> Check:
    """Page's CUSUM over co-observed samples in arrival order. Each violation adds
    log(p1/p0), each clean sample adds log((1-p1)/(1-p0)); the statistic never goes below
    zero, so a burst that has passed decays and only evidence that persists alarms. This
    replaces repeated overlapping binomial windows, whose many correlated tests turned
    chance bursts at the clean rate into quarantines (registered run, H4)."""
    lp, lq = math.log(CUSUM_P1 / inv.p0), math.log((1 - CUSUM_P1) / (1 - inv.p0))
    S, n, k = 0.0, 0, 0
    for s in samples:
        if not all(key in s and s[key] is not None for key in inv.concepts):
            continue
        n += 1
        try:
            v = bool(inv.violated(s))
        except (ZeroDivisionError, ValueError, KeyError):
            v = True
        k += v
        S = max(0.0, S + (lp if v else lq))
    st = "fail" if S >= h else ("pass" if n >= N_MIN_INV else ("skip" if n == 0 else "unsure"))
    return Check(f"invariant:{inv.name}", st, S, h, n,
                 f"{inv.statement}: violated in {k} of {n} co-observed samples (CUSUM {S:.1f} of {h:g})")


def stuck_run(values: Sequence[float], ceiling: Optional[float] = None) -> int:
    """The longest run of identical consecutive readings, ignoring readings at an
    instrument ceiling (an SpO₂ of 100 repeats legitimately)."""
    best = run = 0
    prev = None
    for v in values:
        if ceiling is not None and v >= ceiling:
            prev, run = None, 0
            continue
        run = run + 1 if (prev is not None and v == prev) else 1
        prev = v
        best = max(best, run)
    return best


def concept_or_raise(key: str) -> Concept:
    c = BY_KEY.get(key)
    if c is None:
        raise KeyError(key)
    return c
