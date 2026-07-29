"""Adverse-event projection — what the drug might do to this patient, and how likely.

Phase 2 of the treatment twin. The PBPK layer says where the drug goes; this layer
says what that exposure risks. Ask it to run the side effects of a certified order
and it returns a ranked list of adverse events, each with a probability, a
confidence interval, the organ it lands in, and the hours after the order when it
would appear — everything the 3-D twin needs to make a side effect *visible* on
the body at the moment it would occur.

HOW A PROBABILITY IS PRODUCED
-----------------------------
Three ingredients, and the code keeps them separate on purpose:

  1. **A published base incidence.** Every event starts from a rate reported in the
     drug's label or in a clinical study — the population frequency at a normal
     dose in a typical patient.
  2. **An exposure multiplier.** The projected exposure from the PBPK model (organ
     AUC, plasma trough, interstitial volume) shifts that rate up or down relative
     to the exposure the published rate was observed at. Doubling the exposure a
     kidney sees does not leave nephrotoxicity risk where the label put it.
  3. **Patient modifiers.** Renal impairment, hypotension, age, and drug–drug
     interactions drawn from the existing Living Pharmacopoeia (vancomycin +
     piperacillin-tazobactam is the textbook nephrotoxic pair, and it is in the
     curated layer already).

The result is a hazard, and the hazard is run through a Monte Carlo rather than
reported as a point estimate. That is not decoration: population PK carries real
inter-individual variability (clearance and volume vary ~30% CV between patients),
so the honest output is a *distribution* over exposure and therefore a probability
with an interval, not a single confident number.

THE EVIDENCE FIREWALL (the part that matters)
---------------------------------------------
It would be easy to invent a dose-response curve for every adverse event and have
the output look identical to one built on evidence. So every event declares what
its exposure-response is actually founded on, and the tier travels with the number
all the way to the UI:

  * ``ANCHORED``    — the exposure-response threshold itself is published (e.g.
                      vancomycin AUC24 > 600 mg·h/L and AKI; cefepime trough
                      > 20 mg/L and neurotoxicity). Cited on the event.
  * ``LABEL_RATE``  — the base incidence is published, but no quantitative
                      exposure-response exists, so exposure is applied as a mild
                      monotone amplifier only.
  * ``MECHANISTIC`` — no published rate for this exposure pattern; the event is
                      derived from an accepted mechanism. Reported, flagged, and
                      never presented as a measured frequency.

A projection that is mostly MECHANISTIC is a projection to be sceptical of, and
the summary says so out loud instead of hiding it in an average.

WHAT IS NOT CLAIMED
-------------------
This does not predict whether *this* patient will have *this* reaction. It is a
population-rate model, conditioned on projected exposure, with no outcome data in
this system to fit or validate against — there is no adverse-event ground truth in
the CinC corpus. It is a structured way to ask "what is this order risking, and
where would it show up", not a validated forecast, and it never gates or modifies
an order: the Z3 safety cage remains the only thing that can do that.

    python -m backend.aegis.adverse          # demo
    python -m backend.aegis.verify_adverse   # the falsifiable checks
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.aegis.pharmacokinetics import (ANTIBIOTIC_CODE_TO_DRUG, DRUGS,
                                            ORGAN_MESH, PBPKModel, PatientState,
                                            Regimen)

# Evidence tiers — see the module docstring. Ordered worst → best.
MECHANISTIC, LABEL_RATE, ANCHORED = "MECHANISTIC", "LABEL_RATE", "ANCHORED"
EVIDENCE_RANK = {MECHANISTIC: 0, LABEL_RATE: 1, ANCHORED: 2}

# Severity ladder. Used for ranking and for how loudly the twin renders an event.
MILD, MODERATE, SEVERE, CRITICAL = "mild", "moderate", "severe", "life-threatening"
SEVERITY_RANK = {MILD: 0, MODERATE: 1, SEVERE: 2, CRITICAL: 3}

# Population PK variability: how much clearance and volume differ between patients
# with identical covariates. ~30% CV is the usual figure from critically ill
# population-PK studies, and it is the reason a probability gets an interval.
CV_CLEARANCE = 0.32
CV_VOLUME = 0.25
VARIABILITY_SOURCE = ("inter-individual variability ~30% CV on CL and Vd — Roberts "
                      "et al., Clin Infect Dis 58:1072 (2014), DALI study")


@dataclass(frozen=True)
class AdverseEvent:
    """One thing a drug can do, and what determines how likely it is here."""
    key: str
    label: str
    drug: str
    organ: str                       # compartment key → mesh via ORGAN_MESH
    system: str                      # renal | neuro | cardiac | hepatic | ...
    severity: str
    base_incidence: float            # published population rate at normal exposure
    evidence: str                    # MECHANISTIC | LABEL_RATE | ANCHORED
    # Exposure response: which projected quantity drives it, the exposure at which
    # the base incidence was observed, and how steeply risk rises above that.
    exposure_metric: str = ""        # "" = exposure-independent (idiosyncratic)
    # The exposure at which ``base_incidence`` was observed. Either a *published*
    # threshold (ANCHORED events — vancomycin AUC24, cefepime trough), or, where no
    # such number exists, computed from the model itself by projecting the drug's
    # standard regimen into a reference patient. Deriving it is strictly better
    # than typing one in: a published incidence is by definition the rate at the
    # standard dose in a typical patient, so that is exactly the exposure the rate
    # must be pinned to — and a hand-picked constant silently shifts every
    # probability that depends on it.
    reference_exposure: float = 0.0
    reference_from_model: bool = False
    exposure_exponent: float = 1.0   # risk ∝ (exposure / reference) ** exponent
    max_multiplier: float = 8.0      # hazard amplification is capped, not unbounded
    onset_h: Tuple[float, float, float] = (1.0, 6.0, 24.0)   # min, mode, max
    reversible: bool = True
    monitoring: str = ""
    citation: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {"key": self.key, "label": self.label, "drug": self.drug,
                "organ": self.organ, "mesh": ORGAN_MESH.get(self.organ, ""),
                "system": self.system, "severity": self.severity,
                "base_incidence": self.base_incidence, "evidence": self.evidence,
                "exposure_metric": self.exposure_metric,
                "reference_exposure": self.reference_exposure,
                "onset_h": list(self.onset_h), "reversible": self.reversible,
                "monitoring": self.monitoring, "citation": self.citation}


# --------------------------------------------------------------------------- #
# The catalogue. Base rates from labels and clinical studies; exposure anchors
# only where a quantitative exposure-response is actually published.
# --------------------------------------------------------------------------- #
CATALOGUE: List[AdverseEvent] = [
    # ---- vancomycin -------------------------------------------------------- #
    AdverseEvent(
        key="vanc_aki", label="Acute kidney injury", drug="vancomycin",
        organ="kidney", system="renal", severity=SEVERE,
        base_incidence=0.12, evidence=ANCHORED,
        exposure_metric="auc24", reference_exposure=500.0, exposure_exponent=2.2,
        onset_h=(24.0, 72.0, 168.0), reversible=True,
        monitoring="daily creatinine; AUC-guided dosing, target 400–600 mg·h/L",
        citation="Rybak et al., Am J Health Syst Pharm 77:835 (2020) — AKI risk "
                 "rises steeply above AUC24 600 mg·h/L"),
    AdverseEvent(
        key="vanc_redman", label="Vancomycin infusion reaction (red man)",
        drug="vancomycin", organ="skin", system="dermatologic", severity=MILD,
        base_incidence=0.09, evidence=LABEL_RATE,
        exposure_metric="cmax", reference_from_model=True, exposure_exponent=1.4,
        onset_h=(0.1, 0.4, 1.5), reversible=True,
        monitoring="slow the infusion to ≥60 min; antihistamine",
        citation="Vancocin FDA label — rate-related histamine release"),
    AdverseEvent(
        key="vanc_ototox", label="Ototoxicity", drug="vancomycin",
        organ="brain", system="neuro", severity=MODERATE,
        base_incidence=0.02, evidence=MECHANISTIC,
        exposure_metric="auc24", reference_exposure=500.0, exposure_exponent=1.6,
        onset_h=(48.0, 120.0, 336.0), reversible=False,
        monitoring="audiometry if prolonged therapy",
        citation="mechanism accepted; no quantitative exposure-response published"),

    # ---- piperacillin-tazobactam ------------------------------------------- #
    AdverseEvent(
        key="pip_aki", label="Acute kidney injury", drug="piperacillin-tazobactam",
        organ="kidney", system="renal", severity=SEVERE,
        base_incidence=0.08, evidence=LABEL_RATE,
        exposure_metric="organ_auc:kidney", reference_from_model=True,
        exposure_exponent=1.5, onset_h=(24.0, 72.0, 168.0), reversible=True,
        monitoring="daily creatinine; markedly higher combined with vancomycin",
        citation="Zosyn FDA label; Luther et al., Crit Care Med 46:12 (2018)"),
    AdverseEvent(
        key="pip_cdiff", label="C. difficile colitis", drug="piperacillin-tazobactam",
        organ="intestine", system="gastrointestinal", severity=MODERATE,
        base_incidence=0.03, evidence=LABEL_RATE,
        exposure_metric="organ_auc:intestine", reference_from_model=True,
        exposure_exponent=0.9, onset_h=(48.0, 168.0, 720.0), reversible=True,
        monitoring="stool PCR if diarrhoea develops",
        citation="Zosyn FDA label — broad-spectrum flora disruption"),
    AdverseEvent(
        key="pip_hypersens", label="Hypersensitivity reaction",
        drug="piperacillin-tazobactam", organ="skin", system="immune",
        severity=MODERATE, base_incidence=0.04, evidence=LABEL_RATE,
        exposure_metric="", onset_h=(0.2, 2.0, 48.0), reversible=True,
        monitoring="beta-lactam allergy history; observe first dose",
        citation="Zosyn FDA label — idiosyncratic, not exposure-dependent"),

    # ---- cefepime ---------------------------------------------------------- #
    AdverseEvent(
        key="cefe_neurotox", label="Cefepime neurotoxicity (encephalopathy)",
        drug="cefepime", organ="brain", system="neuro", severity=SEVERE,
        base_incidence=0.05, evidence=ANCHORED,
        exposure_metric="ctrough", reference_exposure=20.0, exposure_exponent=2.6,
        onset_h=(24.0, 96.0, 240.0), reversible=True,
        monitoring="mental status; this is the reason cefepime needs renal dosing",
        citation="Huwyler et al., Clin Microbiol Infect 23:454 (2017) — neurotoxicity "
                 "risk rises sharply above a trough of 20 mg/L"),
    AdverseEvent(
        key="cefe_seizure", label="Seizure", drug="cefepime", organ="brain",
        system="neuro", severity=CRITICAL, base_incidence=0.01, evidence=ANCHORED,
        exposure_metric="ctrough", reference_exposure=35.0, exposure_exponent=3.0,
        onset_h=(24.0, 96.0, 240.0), reversible=True,
        monitoring="continuous EEG if unexplained obtundation",
        citation="Maxipime FDA label; Payne et al., Crit Care 21:276 (2017)"),

    # ---- meropenem --------------------------------------------------------- #
    AdverseEvent(
        key="mero_seizure", label="Seizure", drug="meropenem", organ="brain",
        system="neuro", severity=CRITICAL, base_incidence=0.005, evidence=LABEL_RATE,
        exposure_metric="ctrough", reference_exposure=16.0, exposure_exponent=2.0,
        onset_h=(12.0, 72.0, 240.0), reversible=True,
        monitoring="lower seizure risk than imipenem; still dose-adjust in renal failure",
        citation="Merrem FDA label"),
    AdverseEvent(
        key="mero_cdiff", label="C. difficile colitis", drug="meropenem",
        organ="intestine", system="gastrointestinal", severity=MODERATE,
        base_incidence=0.025, evidence=LABEL_RATE,
        exposure_metric="organ_auc:intestine", reference_from_model=True,
        exposure_exponent=0.9, onset_h=(48.0, 168.0, 720.0), reversible=True,
        monitoring="stool PCR if diarrhoea develops",
        citation="Merrem FDA label"),

    # ---- levofloxacin ------------------------------------------------------ #
    AdverseEvent(
        key="levo_qt", label="QT prolongation / torsades risk", drug="levofloxacin",
        organ="heart", system="cardiac", severity=CRITICAL,
        base_incidence=0.015,
        # NOT anchored. CredibleMeds classifies levofloxacin as a known-risk QT
        # drug and the label reports the incidence, but neither publishes a
        # concentration threshold the way the vancomycin AUC and cefepime trough
        # literature does. Claiming ANCHORED here would have meant inventing a
        # dose-response and presenting it with the same authority as a measured
        # one — the firewall check refused it, which is the point of the check.
        evidence=LABEL_RATE,
        exposure_metric="cmax", reference_from_model=True, exposure_exponent=1.8,
        onset_h=(1.0, 4.0, 24.0), reversible=True,
        monitoring="ECG QTc; additive with other QT-active drugs",
        citation="Levaquin FDA label; CredibleMeds — known-risk QT drug "
                 "(categorical, no published concentration threshold)"),
    AdverseEvent(
        key="levo_tendon", label="Tendinopathy / rupture", drug="levofloxacin",
        organ="muscle", system="musculoskeletal", severity=MODERATE,
        base_incidence=0.004, evidence=LABEL_RATE, exposure_metric="auc24",
        reference_from_model=True, exposure_exponent=1.0,
        onset_h=(48.0, 336.0, 1440.0), reversible=False,
        monitoring="boxed warning; higher with corticosteroids and age >60",
        citation="Levaquin FDA label — boxed warning"),
    AdverseEvent(
        key="levo_delirium", label="Delirium / CNS effects", drug="levofloxacin",
        organ="brain", system="neuro", severity=MODERATE,
        base_incidence=0.02, evidence=LABEL_RATE, exposure_metric="organ_auc:brain",
        reference_from_model=True, exposure_exponent=1.3,
        onset_h=(6.0, 36.0, 168.0), reversible=True,
        monitoring="mental status; higher in the elderly",
        citation="Levaquin FDA label"),

    # ---- norepinephrine ---------------------------------------------------- #
    AdverseEvent(
        key="ne_arrhythmia", label="Tachyarrhythmia", drug="norepinephrine",
        organ="heart", system="cardiac", severity=SEVERE,
        base_incidence=0.10, evidence=LABEL_RATE,
        exposure_metric="organ_auc:heart", reference_from_model=True,
        exposure_exponent=1.3, onset_h=(0.1, 1.0, 12.0), reversible=True,
        monitoring="continuous ECG; risk scales with dose",
        citation="Levophed FDA label; Sakr et al., Crit Care Med 34:589 (2006)"),
    AdverseEvent(
        key="ne_ischaemia", label="Peripheral / digital ischaemia",
        drug="norepinephrine", organ="skin", system="vascular", severity=SEVERE,
        base_incidence=0.04, evidence=LABEL_RATE,
        exposure_metric="css", reference_from_model=True, exposure_exponent=1.8,
        onset_h=(2.0, 12.0, 72.0), reversible=False,
        monitoring="hourly extremity perfusion check",
        citation="Levophed FDA label — vasoconstriction is the mechanism of action"),
    AdverseEvent(
        key="ne_splanchnic", label="Splanchnic hypoperfusion / gut ischaemia",
        drug="norepinephrine", organ="intestine", system="gastrointestinal",
        severity=SEVERE, base_incidence=0.03, evidence=MECHANISTIC,
        exposure_metric="css", reference_from_model=True, exposure_exponent=1.6,
        onset_h=(2.0, 12.0, 72.0), reversible=True,
        monitoring="lactate trend; abdominal exam",
        citation="mechanism accepted; no quantitative exposure-response published"),
    AdverseEvent(
        key="ne_extravasation", label="Extravasation tissue necrosis",
        drug="norepinephrine", organ="skin", system="dermatologic", severity=SEVERE,
        base_incidence=0.02, evidence=MECHANISTIC, exposure_metric="",
        onset_h=(0.5, 6.0, 48.0), reversible=False,
        monitoring="central line preferred; inspect the site hourly",
        citation="mechanism accepted; incidence depends on access, not exposure"),

    # ---- crystalloid ------------------------------------------------------- #
    AdverseEvent(
        key="fluid_pulm_oedema", label="Pulmonary oedema", drug="crystalloid",
        organ="lung", system="respiratory", severity=SEVERE,
        base_incidence=0.07, evidence=LABEL_RATE,
        exposure_metric="organ_auc:lung", reference_from_model=True,
        exposure_exponent=1.6, onset_h=(1.0, 6.0, 24.0), reversible=True,
        monitoring="SpO₂, work of breathing; far higher in CHF",
        citation="Marik & Bellomo, Br J Anaesth 116:339 (2016) — fluid overload harm"),
    AdverseEvent(
        key="fluid_overload", label="Interstitial oedema / positive fluid balance",
        drug="crystalloid", organ="skin", system="vascular", severity=MODERATE,
        base_incidence=0.25, evidence=LABEL_RATE,
        exposure_metric="interstitial_ml", reference_from_model=True,
        exposure_exponent=1.2, onset_h=(1.0, 8.0, 48.0), reversible=True,
        monitoring="daily weights, cumulative balance",
        citation="Chowdhury et al., Ann Surg 256:18 (2012)"),
    AdverseEvent(
        key="fluid_hyperchloraemia", label="Hyperchloraemic acidosis",
        drug="crystalloid", organ="kidney", system="metabolic", severity=MILD,
        base_incidence=0.10, evidence=MECHANISTIC,
        exposure_metric="infused_ml", reference_exposure=2000.0,
        exposure_exponent=1.2, onset_h=(2.0, 8.0, 24.0), reversible=True,
        monitoring="chloride and bicarbonate; balanced solutions reduce it",
        citation="mechanism accepted; magnitude depends on the fluid chosen"),
]

# Patient-covariate hazard modifiers, applied per system.
#
# These are SUSCEPTIBILITY factors only — the patient being more easily harmed by
# the same exposure. They must not restate anything the PBPK model already does.
# A patient with eGFR 38 receiving vancomycin is at higher risk for two distinct
# reasons: the drug accumulates (more exposure) *and* an already-injured kidney
# tolerates it worse (more susceptibility). The first is computed by the PK layer
# and arrives here inside ``exposure``; only the second belongs in this table.
# Sizing these at full clinical odds-ratios would count the renal pathway twice
# and was doing exactly that — it put vancomycin AKI at 55%.
COVARIATE_MODIFIERS = {
    "renal": [("egfr_below_45", 1.4, "injured kidney tolerates the same exposure worse"),
              ("hypotension", 1.3, "hypoperfusion compounds tubular injury")],
    "neuro": [("egfr_below_45", 1.3, "blood-brain barrier and CNS susceptibility in uraemia"),
              ("age_over_70", 1.5, "age-related CNS susceptibility")],
    "cardiac": [("hypotension", 1.4, "haemodynamic instability")],
    "respiratory": [("chf", 3.0, "cardiac failure — poor tolerance of volume")],
    "vascular": [("chf", 2.0, "cardiac failure — poor tolerance of volume"),
                 ("hypotension", 1.3, "vasoconstricted peripheral bed")],
    "musculoskeletal": [("age_over_70", 2.0, "age is the dominant risk factor")],
    "gastrointestinal": [("hypotension", 1.4, "splanchnic hypoperfusion")],
}

# Drug-pair hazard multipliers, keyed to the same curated interactions the Living
# Pharmacopoeia already screens for — so the two layers cannot disagree.
INTERACTION_MODIFIERS = [
    (frozenset({"vancomycin", "piperacillin-tazobactam"}), "renal", 2.2,
     "additive nephrotoxicity — Luther et al., Crit Care Med 46:12 (2018)"),
]


# The "typical patient at the standard dose" that model-derived reference
# exposures are computed in. Normal renal function, no shock — the population a
# published incidence rate is drawn from.
REFERENCE_PATIENT = PatientState(weight_kg=78.0, egfr=90.0, map_mmhg=75.0)
_REFERENCE_CACHE: Dict[str, Dict[str, Any]] = {}


def reference_projection(drug: str) -> Dict[str, Any]:
    """PK projection of ``drug``'s standard regimen in the reference patient."""
    cached = _REFERENCE_CACHE.get(drug)
    if cached is None:
        model = PBPKModel(REFERENCE_PATIENT)
        if drug == "crystalloid":
            cached = model.project_fluid().to_dict()
        else:
            cached = model.project(drug).to_dict()
        _REFERENCE_CACHE[drug] = cached
    return cached


@dataclass
class EventProjection:
    """One adverse event, projected onto this patient and this order."""
    event: AdverseEvent
    probability: float
    ci_low: float
    ci_high: float
    baseline: float                  # the published rate, before this patient
    exposure: float
    reference: float                 # the exposure the base rate is pinned to
    exposure_ratio: float            # projected / reference exposure
    multiplier: float
    modifiers: List[Dict[str, Any]]
    onset_p50_h: float
    onset_p10_h: float
    onset_p90_h: float

    def to_dict(self) -> Dict[str, Any]:
        d = self.event.to_dict()
        d.update({
            "probability": round(self.probability, 5),
            "ci_low": round(self.ci_low, 5), "ci_high": round(self.ci_high, 5),
            "baseline": round(self.baseline, 5),
            "exposure": round(self.exposure, 4),
            "reference": round(self.reference, 4),
            "exposure_ratio": round(self.exposure_ratio, 4),
            "multiplier": round(self.multiplier, 4),
            "modifiers": self.modifiers,
            "onset_p50_h": round(self.onset_p50_h, 2),
            "onset_p10_h": round(self.onset_p10_h, 2),
            "onset_p90_h": round(self.onset_p90_h, 2),
            "risk_delta": round(self.probability - self.baseline, 5),
        })
        return d


class AdverseEventEngine:
    """Turns a projected exposure into ranked, interval-bearing event risks."""

    def __init__(self, catalogue: Optional[List[AdverseEvent]] = None,
                 n_samples: int = 20000, seed: int = 7):
        self.catalogue = catalogue if catalogue is not None else CATALOGUE
        self.n_samples = int(n_samples)
        self.seed = int(seed)

    # ------------------------------------------------------------------ #
    def _exposure(self, metric: str, projection: Dict[str, Any]) -> Optional[float]:
        """Pull the driving exposure out of a PBPK projection dict."""
        if not metric:
            return None
        m = projection.get("metrics", {})
        if metric.startswith("organ_auc:"):
            return float(m.get("organ_auc", {}).get(metric.split(":", 1)[1], 0.0))
        if metric == "ctrough":
            # Trough = the concentration just before the next dose is due, which is
            # the number the neurotoxicity literature is anchored on. Falls back to
            # the end-of-horizon value for a single-dose projection.
            series = projection.get("plasma", [])
            times = projection.get("times_h", [])
            spec = DRUGS.get(projection.get("drug", ""))
            if series and times and spec and not spec.regimen.continuous:
                iv = spec.regimen.interval_h
                idx = int(np.argmin(np.abs(np.array(times) - iv))) if iv < times[-1] \
                    else len(series) - 1
                return float(series[idx])
            return float(series[-1]) if series else 0.0
        if metric == "interstitial_ml":
            return float(m.get("interstitial_ml_end", 0.0))
        if metric == "infused_ml":
            return float(m.get("infused_ml", 0.0))
        return float(m.get(metric, 0.0))

    def _modifiers(self, ev: AdverseEvent, patient: PatientState,
                   history: Dict[str, Any], drugs: List[str]
                   ) -> Tuple[float, List[Dict[str, Any]]]:
        flags = {
            "egfr_below_45": patient.egfr < 45.0,
            "hypotension": patient.map_mmhg < 65.0,
            "age_over_70": float(history.get("age", 60) or 60) > 70,
            "chf": bool(history.get("chf")),
        }
        mult, applied = 1.0, []
        for flag, factor, why in COVARIATE_MODIFIERS.get(ev.system, []):
            if flags.get(flag):
                mult *= factor
                applied.append({"kind": "covariate", "flag": flag,
                                "factor": factor, "why": why})
        drugset = {d for d in drugs}
        for pair, system, factor, why in INTERACTION_MODIFIERS:
            if system == ev.system and pair <= drugset:
                mult *= factor
                applied.append({"kind": "interaction",
                                "flag": " + ".join(sorted(pair)),
                                "factor": factor, "why": why})
        return mult, applied

    # ------------------------------------------------------------------ #
    def project(self, projections: List[Dict[str, Any]], patient: PatientState,
                history: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Run the side-effect projection for a set of PBPK arms.

        ``projections`` are ``PKProjection.to_dict()`` results — one per drug arm
        of the order. Returns ranked events plus a summary the UI can lead with.
        """
        history = history or {}
        rng = np.random.default_rng(self.seed)
        by_drug = {p["drug"]: p for p in projections}
        drugs = list(by_drug)
        out: List[EventProjection] = []

        for ev in self.catalogue:
            proj = by_drug.get(ev.drug)
            if proj is None:
                continue                      # this drug was not ordered
            exposure = self._exposure(ev.exposure_metric, proj)
            cov_mult, applied = self._modifiers(ev, patient, history, drugs)
            reference = ev.reference_exposure
            if ev.reference_from_model and ev.exposure_metric:
                reference = self._exposure(ev.exposure_metric,
                                           reference_projection(ev.drug)) or 0.0

            if exposure is None or reference <= 0:
                # Idiosyncratic events (hypersensitivity, extravasation) do not
                # scale with exposure — pretending otherwise would be invention.
                ratio = 1.0
                samples = np.full(self.n_samples, ev.base_incidence * cov_mult)
                exposure = 0.0
            else:
                ratio = exposure / reference
                # Inter-individual PK variability: the same order lands on a
                # distribution of exposures, not one number. Clearance varies
                # log-normally, and exposure moves inversely with clearance.
                sigma = CV_CLEARANCE
                cl_var = rng.lognormal(mean=-0.5 * sigma ** 2, sigma=sigma,
                                       size=self.n_samples)
                sampled_ratio = ratio / cl_var
                k = ev.exposure_exponent
                # Anchoring correction. Risk is a convex function of exposure, so
                # by Jensen the MEAN risk over the variability distribution exceeds
                # the risk at mean exposure — for k=2.2 that alone inflates every
                # probability by ~40%, and the published base incidence would no
                # longer be recovered at reference exposure. E[(1/L)^k] is
                # analytic for a log-normal, so divide it out: the mean lands
                # exactly on base_incidence·ratio^k, and the spread still carries
                # the real variability into the confidence interval.
                jensen = math.exp(sigma ** 2 * k * (k + 1.0) / 2.0)
                amp = np.clip(sampled_ratio ** k / jensen, 0.0, ev.max_multiplier)
                samples = ev.base_incidence * amp * cov_mult
            samples = np.clip(samples, 0.0, 0.95)

            p = float(samples.mean())
            lo, hi = (float(np.percentile(samples, 2.5)),
                      float(np.percentile(samples, 97.5)))
            # Onset: a triangular distribution over the published window.
            lo_h, mode_h, hi_h = ev.onset_h
            onset = rng.triangular(lo_h, mode_h, hi_h, size=min(self.n_samples, 4000))
            out.append(EventProjection(
                event=ev, probability=p, ci_low=lo, ci_high=hi,
                baseline=ev.base_incidence, exposure=exposure,
                reference=reference, exposure_ratio=ratio,
                multiplier=(p / ev.base_incidence) if ev.base_incidence else 1.0,
                modifiers=applied,
                onset_p50_h=float(np.percentile(onset, 50)),
                onset_p10_h=float(np.percentile(onset, 10)),
                onset_p90_h=float(np.percentile(onset, 90))))

        # Rank by expected harm: probability weighted by severity, worst first.
        out.sort(key=lambda e: -(e.probability * (1 + SEVERITY_RANK[e.event.severity])))
        return self._summarise(out, patient, history, drugs)

    # ------------------------------------------------------------------ #
    def _summarise(self, events: List[EventProjection], patient: PatientState,
                   history: Dict[str, Any], drugs: List[str]) -> Dict[str, Any]:
        serious = [e for e in events
                   if SEVERITY_RANK[e.event.severity] >= SEVERITY_RANK[SEVERE]]
        # P(at least one serious event), treating events as independent — which is
        # an assumption, and is labelled as one rather than buried.
        p_none = 1.0
        for e in serious:
            p_none *= (1.0 - e.probability)
        tiers = {t: sum(1 for e in events if e.event.evidence == t)
                 for t in (ANCHORED, LABEL_RATE, MECHANISTIC)}
        n = max(len(events), 1)
        return {
            "events": [e.to_dict() for e in events],
            "n_events": len(events),
            "drugs": drugs,
            "patient": patient.to_dict(),
            "summary": {
                "p_any_serious": round(1.0 - p_none, 4),
                "n_serious": len(serious),
                "top": events[0].event.label if events else None,
                "top_probability": round(events[0].probability, 4) if events else 0.0,
                "independence_assumption": (
                    "P(any serious) treats events as independent; correlated "
                    "toxicity (a failing kidney raises several at once) would make "
                    "the true joint risk lower than this figure"),
            },
            "evidence_mix": {
                "anchored": tiers[ANCHORED], "label_rate": tiers[LABEL_RATE],
                "mechanistic": tiers[MECHANISTIC],
                "fraction_anchored": round(tiers[ANCHORED] / n, 3),
                "note": ("ANCHORED = published exposure-response; LABEL_RATE = "
                         "published incidence, exposure applied as a monotone "
                         "amplifier; MECHANISTIC = accepted mechanism, no published "
                         "rate — treat those as directional, not quantitative"),
            },
            "variability": VARIABILITY_SOURCE,
            "honesty": (
                "Population adverse-event rates conditioned on projected exposure. "
                "This does not predict whether this patient will have this reaction, "
                "and there is no adverse-event ground truth in this system to "
                "validate it against. It never gates an order — the Z3 safety cage "
                "remains the only thing that can."),
        }


# --------------------------------------------------------------------------- #
def project_order(protocol: Dict[str, Any], patient: PatientState,
                  history: Optional[Dict[str, Any]] = None,
                  n_samples: int = 20000) -> Dict[str, Any]:
    """Convenience: PK-project a certified order, then run its side effects."""
    from backend.aegis.pharmacokinetics import project_protocol
    pkout = project_protocol(protocol, patient)
    eng = AdverseEventEngine(n_samples=n_samples)
    ae = eng.project(pkout["arms"], patient, history)
    ae["pk"] = pkout
    return ae


def _demo() -> None:
    pt = PatientState(weight_kg=78.0, egfr=38.0, map_mmhg=56.0)
    hist = {"age": 74, "chf": False}
    proto = {"fluid_rate": 10.0, "vasopressor_dose": 0.15, "antibiotic_code": 2}
    res = project_order(proto, pt, hist)
    s = res["summary"]
    print(f"order: vancomycin + norepinephrine 0.15 + crystalloid 10 mL/kg/h")
    print(f"patient: eGFR {pt.egfr:.0f}, MAP {pt.map_mmhg:.0f}, age {hist['age']}\n")
    print(f"{'event':42s} {'p':>7s} {'95% CI':>15s} {'base':>7s} {'×':>6s} {'onset':>8s}")
    for e in res["events"][:10]:
        ci = f"{e['ci_low']*100:.1f}–{e['ci_high']*100:.1f}%"
        print(f"{e['label'][:40]:42s} {e['probability']*100:6.1f}% {ci:>15s} "
              f"{e['baseline']*100:6.1f}% {e['multiplier']:5.1f}× "
              f"{e['onset_p50_h']:6.1f} h   [{e['evidence']}]")
    print(f"\nP(any serious event) = {s['p_any_serious']*100:.1f}% "
          f"over {s['n_serious']} serious events")
    em = res["evidence_mix"]
    print(f"evidence: {em['anchored']} anchored · {em['label_rate']} label-rate · "
          f"{em['mechanistic']} mechanistic")


if __name__ == "__main__":
    _demo()
