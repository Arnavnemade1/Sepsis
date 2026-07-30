"""The Pharmacokinetic Twin — where an ordered drug actually *goes*, organ by organ.

Every other layer of this OS answers "should we treat?". This one answers the
question that comes immediately after the order is certified: **where does the
drug go, how fast, how much of it reaches each organ, and for how long?** It is
the engine behind the treatment-projection twin: a second 3D body in which the
drug itself is rendered, filling each organ at its own real rate.

WHAT THIS IS
------------
A **perfusion-limited physiologically-based pharmacokinetic (PBPK) model** —
the standard structure used in clinical pharmacology. The body is a closed
circulatory loop of organ compartments, each fed by its own real blood flow:

    dA_t/dt = Q_t · (C_art − C_t/Kp_t) − CLint_t · C_t/Kp_t

with correct anatomy, not a cartoon: the gut, spleen and pancreas drain into the
**portal vein**, so drug reaching the liver is the sum of hepatic-artery flow and
splanchnic outflow (real first-pass geometry). The lungs receive the *entire*
cardiac output, which is why they light up first in the render.

PARAMETER PROVENANCE (the honesty crux — every number is sourced)
-----------------------------------------------------------------
  * **Physiology** — organ blood-flow fractions of cardiac output, organ volumes
    as fractions of body weight, and the CO = 12.5·BW^0.75 L/h allometric scale:
    Brown et al., *Toxicology and Industrial Health* 13(4):407–484 (1997),
    "Physiological parameter values for physiologically based pharmacokinetic
    models" — the reference table this entire field uses.
  * **Drug parameters** — volume of distribution, total clearance, terminal
    half-life, and renal-excretion fraction are taken per drug from its FDA label
    and published population-PK studies in critically ill patients (cited inline
    on each ``DrugSpec``).
  * **Shock redistribution** — under hypotension, flow is shunted away from the
    splanchnic, renal, muscle and skin beds while cerebral and coronary flow are
    autoregulated; standard septic-shock physiology, applied as a *redistribution*
    of cardiac output (fractions are renormalised, so no flow is invented).

SELF-CONSISTENCY, NOT GUESSWORK
-------------------------------
Two structural choices keep this from being a pile of tuned constants:

  1. **Tissue partition coefficients are scaled, not asserted.** Each drug carries
     a *relative* tissue-affinity shape (lipophilic drugs into fat, beta-lactams
     largely extracellular, almost nothing across an intact blood–brain barrier).
     That shape is then scaled by a single factor so the model's steady-state
     volume of distribution equals the drug's *published* Vd. The composite Vd is
     therefore not a free parameter — it is pinned to the literature.
  2. **Organ clearances are back-solved from published total clearance.** For a
     well-stirred organ, CL_organ = Q·CLint/(Q+CLint); that relation is inverted
     to hit a target, and then the targets are *calibrated against the assembled
     body* — because the sum of per-organ clearances is not the whole-body
     clearance. The liver draws most of its inflow from the portal vein, which the
     gut, spleen and pancreas have already stripped of drug, so it extracts from a
     diluted supply. For a low-extraction antibiotic that gap is negligible; for
     norepinephrine, whose clearance approaches cardiac output, it is ~20%. The
     verification suite catches it by measuring AUC = dose/CL and the steady-state
     plateau off the simulated curves themselves.

  A third property falls out of the same discipline: the three published numbers
  for each drug must satisfy t½ = ln2·Vd/CL. That is a definition, not a fit, so
  the suite asserts it across the whole library — it caught two transcription
  errors on the first run.

WHAT IS *NOT* CLAIMED
---------------------
This is a literature-parameterised population model, **not** a therapeutic-drug-
monitoring prediction for an individual patient. There are no measured serum
levels in this system to fit against, so the honest reading of every curve is
"where a typical patient with these covariates would put this drug", not "where
*this* patient's drug is". It projects *exposure*, and exposure is what the
adverse-event layer then reasons over. Concentrations are plasma-equivalent.

    python -m backend.aegis.pharmacokinetics        # demo
    python -m backend.aegis.verify_pharmacokinetics # the falsifiable checks
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.linalg import expm

# --------------------------------------------------------------------------- #
# Physiology — Brown et al. 1997, Tables 5 (volumes) & 8 (blood flows).
# --------------------------------------------------------------------------- #
PHYSIOLOGY_SOURCE = ("Brown RP et al., Toxicol Ind Health 13(4):407-484 (1997) — "
                     "physiological parameter values for PBPK models")

# Fraction of cardiac output perfusing each compartment.
FLOW_FRACTION: Dict[str, float] = {
    "lung":      1.000,   # the entire cardiac output crosses the pulmonary bed
    "liver_art": 0.065,   # hepatic artery only; portal inflow is routed below
    "intestine": 0.146,
    "spleen":    0.030,
    "pancreas":  0.010,
    "kidney":    0.175,
    "brain":     0.114,
    "heart":     0.040,
    "muscle":    0.170,
    "skin":      0.058,
    "adipose":   0.052,
    "rest":      0.140,
}
# Organs whose venous blood drains into the portal vein, not straight to the heart.
PORTAL_ORGANS = ("intestine", "spleen", "pancreas")

# Compartment volume as a fraction of body weight (L/kg, tissue density ~1).
VOLUME_FRACTION: Dict[str, float] = {
    "lung": 0.0076, "liver": 0.0257, "intestine": 0.0171, "spleen": 0.0026,
    "pancreas": 0.0010, "kidney": 0.0044, "brain": 0.0200, "heart": 0.0047,
    "muscle": 0.4000, "skin": 0.0370, "adipose": 0.2140, "rest": 0.1500,
}
BLOOD_FRACTION = 0.079            # total blood volume, fraction of body weight
ARTERIAL_SHARE = 0.30             # arterial share of blood volume (venous = 0.70)
CO_ALLOMETRIC = 12.5              # cardiac output = 12.5 · BW^0.75  (L/h)

# The tissue compartments carried through the ODE (liver is fed portally).
TISSUES: Tuple[str, ...] = ("liver", "intestine", "spleen", "pancreas", "kidney",
                            "brain", "heart", "muscle", "skin", "adipose", "rest")

# How each compartment maps onto an organ mesh in the 3-D treatment twin. The
# renderer needs this to know which body part to fill; compartments with no
# visible mesh (muscle/skin/adipose/rest) are still simulated — they are where
# most of the drug actually sits — but render as the body shell wash.
ORGAN_MESH: Dict[str, str] = {
    "lung": "lung", "heart": "heart", "liver": "liver", "kidney": "kidney",
    "spleen": "spleen", "pancreas": "pancreas", "intestine": "intestine",
    "brain": "brain", "arterial": "vessels", "venous": "vessels",
    # No mesh of their own — but they must still map to something, or an adverse
    # event that lands in skin or muscle (extravasation necrosis, tendinopathy,
    # interstitial oedema) would have nowhere to appear on the body. They render
    # as a wash over the whole body shell, which is anatomically what they are.
    "skin": "shell", "muscle": "shell", "adipose": "shell", "rest": "shell",
}


# --------------------------------------------------------------------------- #
# Drug library — every entry cited.
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Regimen:
    """A real order: dose, how it is given, and how often."""
    dose_mg: float = 0.0              # per administration (absolute mg)
    dose_per_kg: bool = False         # if True, dose_mg is mg/kg
    infusion_min: float = 30.0        # 0 = IV push
    interval_h: float = 8.0
    n_doses: int = 1
    continuous: bool = False          # infusion runs for the whole horizon
    rate_mcg_kg_min: float = 0.0      # continuous-infusion rate (vasopressors)
    label: str = ""


@dataclass(frozen=True)
class DrugSpec:
    key: str
    label: str
    drug_class: str                    # antibiotic | vasopressor | fluid
    vd_l_per_kg: float                 # published steady-state volume of distribution
    cl_l_per_h_per_kg: float           # published total body clearance
    t_half_h: float                    # published terminal half-life (cross-check)
    f_renal: float                     # fraction of clearance that is renal
    f_hepatic: float = 0.0
    f_diffuse: float = 0.0             # ubiquitous enzymatic clearance (COMT/MAO)
    protein_binding: float = 0.0
    kp_rel: Dict[str, float] = field(default_factory=dict)   # relative tissue affinity
    regimen: Regimen = field(default_factory=Regimen)
    horizon_h: float = 8.0             # the drug's own projected timeframe
    mic_mg_l: Optional[float] = None   # susceptibility breakpoint, for %fT>MIC
    # Antibiotics live at mg/L; a catecholamine infusion lives three orders of
    # magnitude lower, so each drug reports in the unit clinicians actually use.
    conc_unit: str = "mg/L"
    conc_scale: float = 1.0            # multiply model mg/L to reach conc_unit
    target_note: str = ""
    citation: str = ""

    def normalized_clearance_split(self) -> Tuple[float, float, float]:
        tot = self.f_renal + self.f_hepatic + self.f_diffuse
        if tot <= 0:
            return 1.0, 0.0, 0.0
        return self.f_renal / tot, self.f_hepatic / tot, self.f_diffuse / tot


# Relative tissue-affinity shapes. These are *shapes*, not absolute Kp values —
# each is rescaled so the composite Vd matches the published figure (see module
# docstring). Beta-lactams and vancomycin are hydrophilic and largely confined to
# extracellular water: high in the well-perfused kidney, poor across an intact
# blood-brain barrier, negligible in fat.
_KP_HYDROPHILIC = {
    "liver": 0.70, "intestine": 0.55, "spleen": 0.60, "pancreas": 0.55,
    "kidney": 1.20, "brain": 0.08, "heart": 0.55, "muscle": 0.45,
    "skin": 0.45, "adipose": 0.12, "rest": 0.40, "lung": 0.60,
}
# Fluoroquinolones distribute far more widely into tissue (Vd > 1 L/kg).
_KP_LIPOPHILIC = {
    "liver": 2.20, "intestine": 1.80, "spleen": 2.00, "pancreas": 1.60,
    "kidney": 3.00, "brain": 0.55, "heart": 1.60, "muscle": 1.30,
    "skin": 1.30, "adipose": 0.90, "rest": 1.20, "lung": 3.50,
}
# Catecholamines: rapid extraneuronal uptake, high in heart and lung.
_KP_CATECHOLAMINE = {
    "liver": 0.80, "intestine": 0.60, "spleen": 0.60, "pancreas": 0.60,
    "kidney": 0.90, "brain": 0.05, "heart": 1.60, "muscle": 0.50,
    "skin": 0.50, "adipose": 0.20, "rest": 0.50, "lung": 1.80,
}

DRUGS: Dict[str, DrugSpec] = {
    "piperacillin-tazobactam": DrugSpec(
        key="piperacillin-tazobactam", label="Piperacillin-Tazobactam",
        drug_class="antibiotic",
        vd_l_per_kg=0.24, cl_l_per_h_per_kg=0.166, t_half_h=1.0,
        f_renal=0.80, f_hepatic=0.20, protein_binding=0.30, kp_rel=_KP_HYDROPHILIC,
        # 4.5 g of the combination product; the 4 g piperacillin component is what
        # the concentration curve tracks (tazobactam is the beta-lactamase inhibitor).
        regimen=Regimen(dose_mg=4000, infusion_min=30, interval_h=6, n_doses=2,
                        label="4.5 g IV q6h (30-min infusion) · piperacillin 4 g"),
        horizon_h=12.0, mic_mg_l=16.0,
        target_note="target ≥50% fT>MIC (Pseudomonas breakpoint 16 mg/L)",
        citation="Zosyn FDA label; Felton et al., AAC 2014 (critically ill pop-PK)"),

    "cefepime": DrugSpec(
        key="cefepime", label="Cefepime", drug_class="antibiotic",
        vd_l_per_kg=0.25, cl_l_per_h_per_kg=0.087, t_half_h=2.0,
        f_renal=0.85, f_hepatic=0.15, protein_binding=0.20, kp_rel=_KP_HYDROPHILIC,
        regimen=Regimen(dose_mg=2000, infusion_min=30, interval_h=8, n_doses=2,
                        label="2 g IV q8h (30-min infusion)"),
        horizon_h=16.0, mic_mg_l=8.0,
        target_note="target ≥60% fT>MIC; accumulates in renal failure (neurotoxicity)",
        citation="Maxipime FDA label; Tam et al., AAC 2003"),

    "vancomycin": DrugSpec(
        key="vancomycin", label="Vancomycin", drug_class="antibiotic",
        vd_l_per_kg=0.70, cl_l_per_h_per_kg=0.0693, t_half_h=7.0,
        f_renal=0.90, f_hepatic=0.10, protein_binding=0.50, kp_rel=_KP_HYDROPHILIC,
        regimen=Regimen(dose_mg=15.0, dose_per_kg=True, infusion_min=90, interval_h=12,
                        n_doses=2, label="15 mg/kg IV q12h (90-min infusion)"),
        horizon_h=24.0,
        target_note="target AUC24 400–600 mg·h/L; >600 raises AKI risk",
        citation="Rybak et al., Am J Health Syst Pharm 2020 (consensus guideline)"),

    "meropenem": DrugSpec(
        key="meropenem", label="Meropenem", drug_class="antibiotic",
        vd_l_per_kg=0.31, cl_l_per_h_per_kg=0.215, t_half_h=1.0,
        f_renal=0.70, f_hepatic=0.30, protein_binding=0.02, kp_rel=_KP_HYDROPHILIC,
        regimen=Regimen(dose_mg=1000, infusion_min=30, interval_h=8, n_doses=2,
                        label="1 g IV q8h (30-min infusion)"),
        horizon_h=16.0, mic_mg_l=2.0,
        target_note="target ≥40% fT>MIC (Enterobacterales breakpoint 2 mg/L)",
        citation="Merrem FDA label; Li et al., J Clin Pharmacol 2006"),

    "levofloxacin": DrugSpec(
        key="levofloxacin", label="Levofloxacin", drug_class="antibiotic",
        vd_l_per_kg=1.25, cl_l_per_h_per_kg=0.133, t_half_h=6.5,
        f_renal=0.87, f_hepatic=0.13, protein_binding=0.30, kp_rel=_KP_LIPOPHILIC,
        regimen=Regimen(dose_mg=750, infusion_min=90, interval_h=24, n_doses=1,
                        label="750 mg IV q24h (90-min infusion)"),
        horizon_h=24.0, mic_mg_l=1.0,
        target_note="target AUC/MIC ≥ 87; QT-prolonging",
        citation="Levaquin FDA label; Preston et al., JAMA 1998"),

    "norepinephrine": DrugSpec(
        key="norepinephrine", label="Norepinephrine", drug_class="vasopressor",
        vd_l_per_kg=0.12, cl_l_per_h_per_kg=2.66, t_half_h=0.031,
        f_renal=0.05, f_hepatic=0.25, f_diffuse=0.70, protein_binding=0.25,
        kp_rel=_KP_CATECHOLAMINE,
        regimen=Regimen(continuous=True, rate_mcg_kg_min=0.10,
                        label="continuous IV infusion, titrated to MAP ≥ 65"),
        horizon_h=2.0, conc_unit="µg/L", conc_scale=1000.0,
        target_note="plasma half-life ~2 min — steady state in ~10 min, off in ~10 min",
        citation="Beloeil et al., Intensive Care Med 2005; Levophed FDA label"),
}

# The bundle engine's antibiotic codes → this library.
ANTIBIOTIC_CODE_TO_DRUG: Dict[int, Optional[str]] = {
    0: None, 1: "piperacillin-tazobactam", 2: "vancomycin", 3: "cefepime",
    4: "meropenem", 5: "levofloxacin",
}

# --------------------------------------------------------------------------- #
# Crystalloid — a different kind of "drug", so a different (and correct) model.
# --------------------------------------------------------------------------- #
FLUID_SOURCE = ("Hahn RG, Anesthesiology 113:470-481 (2010) — volume kinetics of "
                "infusion fluids; Chowdhury et al., Ann Surg 256:18-24 (2012)")


@dataclass
class PatientState:
    """The covariates that actually move a PK curve."""
    weight_kg: float = 78.0
    egfr: float = 90.0                 # mL/min/1.73m² — scales renal clearance
    hepatic_factor: float = 1.0        # 1.0 = normal; <1 = impaired
    map_mmhg: float = 75.0             # drives shock flow redistribution
    cardiac_output_factor: float = 1.0 # sepsis is often hyperdynamic
    albumin_g_dl: float = 3.5
    # Multiplier on the volume of distribution. Resuscitation fluid does not just
    # sit alongside a drug — crystalloid distributes into extracellular water,
    # which IS the distribution volume of a hydrophilic antibiotic, so loading a
    # patient dilutes the same dose into a larger space. This is the mechanism
    # behind the well-documented beta-lactam under-exposure seen after aggressive
    # fluid resuscitation in sepsis (Roberts et al., Clin Infect Dis 58:1072, the
    # DALI study). Default 1.0 = no load.
    vd_expansion: float = 1.0

    def shock_index(self) -> float:
        """0 at MAP ≥ 65, saturating to 1 by MAP 40."""
        if self.map_mmhg >= 65.0:
            return 0.0
        return float(min(1.0, (65.0 - self.map_mmhg) / 25.0))

    def to_dict(self) -> Dict[str, Any]:
        return {"weight_kg": round(self.weight_kg, 1), "egfr": round(self.egfr, 1),
                "hepatic_factor": round(self.hepatic_factor, 2),
                "map_mmhg": round(self.map_mmhg, 1),
                "cardiac_output_factor": round(self.cardiac_output_factor, 2),
                "vd_expansion": round(self.vd_expansion, 3),
                "shock_index": round(self.shock_index(), 3)}


# --------------------------------------------------------------------------- #
# The projection result.
# --------------------------------------------------------------------------- #
@dataclass
class PKProjection:
    """Everything the treatment twin needs to render one drug's journey."""
    drug: str
    label: str
    drug_class: str
    kind: str                          # "concentration" | "volume"
    unit: str
    times_h: List[float]
    plasma: List[float]                # arterial (what a blood draw would show)
    organs: Dict[str, List[float]]     # compartment → series, same length as times
    peak: Dict[str, float]             # compartment → max over the horizon
    metrics: Dict[str, Any]
    regimen: str
    horizon_h: float
    patient: Dict[str, Any]
    provenance: Dict[str, str]
    events: List[Dict[str, Any]] = field(default_factory=list)
    honesty: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "drug": self.drug, "label": self.label, "drug_class": self.drug_class,
            "kind": self.kind, "unit": self.unit, "times_h": self.times_h,
            "plasma": self.plasma, "organs": self.organs, "peak": self.peak,
            "organ_mesh": {k: v for k, v in ORGAN_MESH.items() if k in self.organs
                           or k in ("arterial", "venous")},
            "metrics": self.metrics, "regimen": self.regimen,
            "horizon_h": self.horizon_h, "patient": self.patient,
            "provenance": self.provenance, "events": self.events,
            "honesty": self.honesty,
        }


class PBPKModel:
    """A closed-loop, perfusion-limited PBPK body for one patient."""

    def __init__(self, patient: Optional[PatientState] = None):
        self.patient = patient or PatientState()
        bw = self.patient.weight_kg
        # Cardiac output, allometric (Brown 1997), modulated by the patient's state.
        self.co = CO_ALLOMETRIC * (bw ** 0.75) * float(self.patient.cardiac_output_factor)
        self.v_blood = BLOOD_FRACTION * bw
        self.v_art = self.v_blood * ARTERIAL_SHARE
        self.v_ven = self.v_blood * (1.0 - ARTERIAL_SHARE)
        self.v_lung = VOLUME_FRACTION["lung"] * bw
        self.v: Dict[str, float] = {t: VOLUME_FRACTION[t] * bw for t in TISSUES}
        self.q: Dict[str, float] = self._flows()

    # ---------------------------------------------------------------- flows --
    def _flows(self) -> Dict[str, float]:
        """Organ blood flows (L/h), redistributed under shock.

        Hypotension shunts flow away from the splanchnic, renal, muscle and skin
        beds; cerebral and coronary flow are autoregulated and preserved. This is
        a *redistribution* — the fractions are renormalised, so cardiac output is
        conserved and no flow is conjured out of nothing.
        """
        shock = self.patient.shock_index()
        frac = {k: v for k, v in FLOW_FRACTION.items() if k != "lung"}
        if shock > 0:
            for organ in ("intestine", "spleen", "pancreas", "liver_art"):
                frac[organ] *= (1.0 - 0.45 * shock)
            frac["kidney"] *= (1.0 - 0.50 * shock)
            for organ in ("muscle", "skin", "adipose"):
                frac[organ] *= (1.0 - 0.55 * shock)
            for organ in ("brain", "heart"):
                frac[organ] *= (1.0 + 0.10 * shock)
        total = sum(frac.values())
        q = {k: self.co * v / total for k, v in frac.items()}
        q["lung"] = self.co
        return q

    # ------------------------------------------------------- drug parameters --
    def _partition(self, spec: DrugSpec) -> Dict[str, float]:
        """Scale the drug's relative tissue-affinity shape so the model's
        steady-state Vd equals the published Vd. The composite volume of
        distribution is pinned to the literature, not tuned."""
        bw = self.patient.weight_kg
        # Fluid loading expands the space a hydrophilic drug distributes into. A
        # lipophilic drug already has a Vd many times extracellular water, so the
        # same litre of crystalloid barely moves it — the expansion is scaled by
        # how extracellular the drug is, rather than applied flat.
        exp = float(self.patient.vd_expansion)
        if exp != 1.0:
            # How much of this drug's distribution volume behaves like extracellular
            # water — that is the fraction fluid loading can dilute. The first
            # version was a linear ramp hitting ZERO at Vd 0.7 L/kg, which is
            # vancomycin's exact value: a hydrophilic glycopeptide with well
            # documented Vd expansion in sepsis got no coupling at all. A reciprocal
            # keeps it monotone and non-zero across the whole library — beta-lactams
            # (~0.25 L/kg) fully extracellular, vancomycin most of the way, a
            # lipophilic fluoroquinolone (1.25 L/kg) largely tissue-bound.
            extracellular = max(0.15, min(1.0, 0.55 / max(spec.vd_l_per_kg, 1e-6)))
            exp = 1.0 + (exp - 1.0) * extracellular
        vd_target = spec.vd_l_per_kg * bw * exp
        shape_volume = sum(self.v[t] * spec.kp_rel.get(t, 0.5) for t in TISSUES)
        shape_volume += self.v_lung * spec.kp_rel.get("lung", 0.5)
        scale = (vd_target - self.v_blood) / shape_volume
        scale = max(scale, 1e-3)       # a drug confined to blood still needs Kp>0
        kp = {t: max(1e-3, spec.kp_rel.get(t, 0.5) * scale) for t in TISSUES}
        kp["lung"] = max(1e-3, spec.kp_rel.get("lung", 0.5) * scale)
        return kp

    # -------------------------------------------------------- system matrix --
    def _system_matrix(self, kp: Dict[str, float], clint: Dict[str, float],
                       accumulators: bool = False):
        """Assemble the linear system. One definition of the body, used by both
        the simulation and the clearance calibration — they cannot drift apart.

        Returns ``(M, index, i_auc, i_elim, dim)``. With ``accumulators`` the
        matrix carries two extra rows that integrate arterial AUC and cumulative
        eliminated drug alongside the state, so no separate quadrature is needed.
        """
        n = 3 + len(TISSUES)                        # venous, arterial, lung, tissues
        dim = n + 2 if accumulators else n
        i_auc, i_elim = n, n + 1
        q_liver_art = self.q["liver_art"]
        q_portal = {o: self.q[o] for o in PORTAL_ORGANS}
        q_liver_total = q_liver_art + sum(q_portal.values())

        vol = np.array([self.v_ven, self.v_art, self.v_lung]
                       + [self.v[t] for t in TISSUES])
        kpv = np.array([1.0, 1.0, kp["lung"]] + [kp[t] for t in TISSUES])
        w = 1.0 / (vol * kpv)                       # amount -> outflow concentration
        clv = np.array([0.0, 0.0, 0.0] + [clint[t] for t in TISSUES])
        qv = np.array([0.0, 0.0, 0.0]
                      + [(q_liver_total if t == "liver" else self.q[t]) for t in TISSUES])
        ti = {t: 3 + k for k, t in enumerate(TISSUES)}

        M = np.zeros((dim, dim))
        # Venous pool collects everything that does NOT drain portally.
        for t in TISSUES:
            if t in PORTAL_ORGANS:
                continue                            # drains into the liver instead
            M[0, ti[t]] += qv[ti[t]] * w[ti[t]]
        M[0, 0] -= self.co * w[0]
        # Venous -> lung -> arterial: the entire cardiac output crosses the lung.
        M[2, 0] += self.co * w[0]
        M[2, 2] -= self.co * w[2]
        M[1, 2] += self.co * w[2]
        M[1, 1] -= self.co * w[1]
        # Tissues, fed arterially; the liver is fed by hepatic artery + portal vein.
        for t in TISSUES:
            j = ti[t]
            if t == "liver":
                M[j, 1] += q_liver_art * w[1]
                for o in PORTAL_ORGANS:
                    M[j, ti[o]] += q_portal[o] * w[ti[o]]
            else:
                M[j, 1] += qv[j] * w[1]
            M[j, j] -= (qv[j] + clv[j]) * w[j]
        if accumulators:
            M[i_auc, 1] = w[1]
            for t in TISSUES:
                M[i_elim, ti[t]] = clv[ti[t]] * w[ti[t]]
        return M, ti, i_auc, i_elim, dim

    def _systemic_clearance(self, kp: Dict[str, float],
                            clint: Dict[str, float]) -> float:
        """The clearance the assembled body ACTUALLY achieves, in L/h.

        At steady state under a constant input u, ``M·x + b = 0``, so
        CL = u / C_arterial. This is not the same as the sum of the per-organ
        clearances: the liver is fed by the portal vein as well as the hepatic
        artery, and portal blood has already been stripped of drug by the gut,
        spleen and pancreas. The liver therefore extracts from a *diluted* inflow,
        and summing well-stirred organ clearances overstates the whole-body value.
        For a low-extraction antibiotic the gap is fractions of a percent; for
        norepinephrine, whose clearance approaches cardiac output, it is ~20%.
        """
        n = 3 + len(TISSUES)
        M, _ti, _a, _e, _d = self._system_matrix(kp, clint, accumulators=False)
        b = np.zeros(n)
        b[0] = 1.0                                  # 1 mg/h into the venous pool
        x = np.linalg.solve(M, -b)
        c_art = x[1] / self.v_art
        return 1.0 / max(c_art, 1e-12)

    def _intrinsic_clearances(self, spec: DrugSpec,
                              kp: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """Back-solve per-organ intrinsic clearance so whole-body clearance
        reproduces the drug's published CL.

        For a well-stirred organ CL_org = Q·CLint/(Q+CLint); inverting gives
        CLint = Q·CL_org/(Q−CL_org). A target organ clearance at or above organ
        blood flow is flow-limited and pinned just below it (extraction ≈ 1).

        Every elimination route acting on the *same* organ is summed into one
        target clearance BEFORE the inversion. Inverting each route separately and
        adding the results would be wrong: the well-stirred relation is non-linear,
        so CL(a) + CL(b) < CL(a+b), and whole-body clearance would silently come
        out below the published value. That mattered most for norepinephrine, whose
        clearance approaches cardiac output — the steady-state check catches it.
        """
        f_ren, f_hep, f_dif = spec.normalized_clearance_split()
        cl_total = spec.cl_l_per_h_per_kg * self.patient.weight_kg
        # Renal clearance tracks glomerular filtration; hepatic tracks liver function.
        cl_ren = cl_total * f_ren * min(1.6, max(0.05, self.patient.egfr / 90.0))
        cl_hep = cl_total * f_hep * max(0.05, self.patient.hepatic_factor)
        cl_dif = cl_total * f_dif

        q_liver = self.q["liver_art"] + sum(self.q[o] for o in PORTAL_ORGANS)

        def flow(t: str) -> float:
            return q_liver if t == "liver" else self.q[t]

        # Step 1 — accumulate every route's target ORGAN clearance.
        target: Dict[str, float] = {t: 0.0 for t in TISSUES}
        target["kidney"] += cl_ren
        target["liver"] += cl_hep
        if cl_dif > 0:
            # Ubiquitous enzymatic clearance (COMT/MAO for catecholamines) is
            # apportioned across tissues in proportion to their blood flow.
            q_sum = sum(flow(t) for t in TISSUES)
            for t in TISSUES:
                target[t] += cl_dif * flow(t) / q_sum

        # Step 2 — one inversion per organ, on the summed target.
        def invert(q: float, cl_org: float) -> float:
            cl_org = min(cl_org, 0.95 * q)
            return q * cl_org / max(q - cl_org, 1e-9)

        clint = {t: invert(flow(t), target[t]) for t in TISSUES}
        if kp is None:
            return clint

        # Step 3 — calibrate against the assembled body. Summed organ clearances
        # are only an approximation of whole-body clearance (see
        # ``_systemic_clearance``), so the per-organ targets are rescaled until the
        # system the simulation will actually integrate delivers the published CL.
        # A few fixed-point iterations converge to machine precision.
        want = sum(target.values())
        if want <= 0:
            return clint
        for _ in range(24):
            got = self._systemic_clearance(kp, clint)
            if abs(got - want) <= 1e-10 * want:
                break
            target = {t: min(v * (want / got), 0.98 * flow(t))
                      for t, v in target.items()}
            clint = {t: invert(flow(t), target[t]) for t in TISSUES}
        return clint

    # ------------------------------------------------------------ simulation --
    def project(self, drug: str, regimen: Optional[Regimen] = None,
                horizon_h: Optional[float] = None,
                n_points: int = 121) -> PKProjection:
        """Integrate the drug through the body and return the render-ready series."""
        if drug == "crystalloid":
            return self.project_fluid(regimen, horizon_h, n_points)
        spec = DRUGS.get(drug)
        if spec is None:
            raise KeyError(f"unknown drug {drug!r} — library: {', '.join(sorted(DRUGS))}")
        reg = regimen or spec.regimen
        horizon = float(horizon_h or spec.horizon_h)
        bw = self.patient.weight_kg

        kp = self._partition(spec)
        clint = self._intrinsic_clearances(spec, kp)
        q_liver_art = self.q["liver_art"]
        q_portal = {o: self.q[o] for o in PORTAL_ORGANS}
        q_liver_total = q_liver_art + sum(q_portal.values())

        # Dosing: build the infusion-rate schedule (mg/h into the venous pool).
        infusions: List[Tuple[float, float, float]] = []   # (start_h, end_h, mg/h)
        if reg.continuous:
            rate_mg_h = reg.rate_mcg_kg_min * bw * 60.0 / 1000.0
            infusions.append((0.0, horizon, rate_mg_h))
            dose_total = rate_mg_h * horizon
        else:
            dose = reg.dose_mg * bw if reg.dose_per_kg else reg.dose_mg
            dur = max(reg.infusion_min, 1.0) / 60.0
            dose_total = 0.0
            for i in range(max(1, reg.n_doses)):
                t0 = i * reg.interval_h
                if t0 >= horizon:
                    break
                infusions.append((t0, t0 + dur, dose / dur))
                dose_total += dose

        # ---- exact linear solution ------------------------------------------ #
        # This whole body is a LINEAR system: every term is a flow or a clearance
        # acting on a compartment amount, and the dose enters as an input. So it is
        # solved exactly with a matrix exponential rather than stepped with Euler —
        # which is both faster and removes integration error as a source of doubt.
        # Mass balance then holds to machine precision, and the verification suite
        # checks exactly that.
        #
        # State vector: [venous, arterial, lung, *TISSUES, cumulative AUC,
        #                cumulative eliminated]. The last two are accumulator rows
        #                integrated by the same exponential — no separate quadrature.
        M, ti, i_auc, i_elim, dim = self._system_matrix(kp, clint, accumulators=True)
        names = ["venous", "arterial", "lung", *TISSUES]

        # Piecewise-constant dosing: step the exact solution between breakpoints.
        sample_at = np.linspace(0.0, horizon, n_points)
        breaks = {0.0, horizon}
        for (s0, e0, _r) in infusions:
            breaks.add(min(max(s0, 0.0), horizon))
            breaks.add(min(max(e0, 0.0), horizon))
        grid = np.array(sorted(breaks | {float(t) for t in sample_at}))

        def rate_in(t_h: float) -> float:
            return sum(r for (s0, e0, r) in infusions if s0 <= t_h < e0 - 1e-12)

        def propagator(rate: float, h: float) -> np.ndarray:
            """exp of the augmented [[M, b], [0, 0]] block — handles a singular M.

            The scaling and squaring is done here rather than left to ``expm``.
            This system is stiff — the pancreas turns over ~10⁴ times an hour while
            adipose barely moves — so ‖M·h‖ can reach several hundred, well outside
            the range where a single Padé evaluation is comfortable. Halving until
            the norm is ≤1 and squaring back is the same algorithm run where the
            arithmetic is well conditioned; squaring a matrix exponential is exact,
            so nothing is approximated away.

            The warning filter is unrelated to any of that: SciPy 1.13's
            ``pick_pade_structure`` emits spurious divide-by-zero / overflow
            warnings for matrices of this size in this BLAS build — it fires
            identically on an all-zero 17×17 matrix. The returned exponential is
            correct (the suite checks it against an independent fine-step
            integration); the warning is noise, and is silenced here and only here.
            Because that filter would also mask a *genuine* overflow, the result is
            explicitly checked for finiteness before it is allowed out.
            """
            aug = np.zeros((dim + 1, dim + 1))
            aug[:dim, :dim] = M
            aug[0, dim] = rate                 # the infusion enters the venous pool
            aug *= h
            nrm = float(np.abs(aug).sum(axis=1).max())
            k = int(max(0, math.ceil(math.log2(nrm)))) if nrm > 1.0 else 0
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                E = expm(aug / (2.0 ** k))
                for _ in range(k):
                    E = E @ E
            if not np.isfinite(E).all():
                raise FloatingPointError(
                    f"PBPK propagator diverged (‖M·h‖={nrm:.3g}, squarings={k}) — "
                    "this is a real numerical failure, not the SciPy warning noise")
            return E

        prop_cache: Dict[Tuple[float, float], np.ndarray] = {}
        x = np.zeros(dim)
        out_plasma: List[float] = []
        out: Dict[str, List[float]] = {k: [] for k in names}
        s_idx = 0
        infused = 0.0

        for gi, t_h in enumerate(grid):
            while s_idx < n_points and t_h >= sample_at[s_idx] - 1e-9:
                out["arterial"].append(x[1] / self.v_art)
                out["venous"].append(x[0] / self.v_ven)
                out["lung"].append(x[2] / self.v_lung)
                for t in TISSUES:
                    out[t].append(x[ti[t]] / self.v[t])     # tissue concentration
                out_plasma.append(x[1] / self.v_art)
                s_idx += 1
            if gi + 1 >= len(grid):
                break
            h = float(grid[gi + 1] - t_h)
            if h <= 0:
                continue
            rate = rate_in(float(t_h))
            key = (round(rate, 9), round(h, 12))
            E = prop_cache.get(key)
            if E is None:
                E = propagator(rate, h)
                prop_cache[key] = E
            x = E[:dim, :dim] @ x + E[:dim, dim]
            infused += rate * h

        eliminated = float(x[i_elim])
        cum_auc = float(x[i_auc])
        a_ven, a_art, a_lung = float(x[0]), float(x[1]), float(x[2])
        a = {t: float(x[ti[t]]) for t in TISSUES}
        dose_total = infused if reg.continuous else dose_total

        # Everything leaving this function is in the drug's own clinical unit.
        us = spec.conc_scale
        organs = {k: [round(x * us, 6) for x in v] for k, v in out.items()}
        peak = {k: round(max(v) * us, 6) for k, v in out.items()}

        # ---- metrics ---------------------------------------------------------- #
        arr = np.array(out["arterial"]) * us
        tmax = float(sample_at[int(np.argmax(arr))]) if len(arr) else 0.0
        # A continuous infusion has no meaningful Cmax — it plateaus, and the
        # clinically useful number is how fast it gets there (and how fast it goes
        # away when you turn it off). Reporting argmax of a flat line would be noise.
        t_ss = None
        if reg.continuous and len(arr) and arr.max() > 0:
            idx = int(np.argmax(arr >= 0.90 * arr[-1]))
            t_ss = float(sample_at[idx])
        in_body = a_ven + a_art + a_lung + sum(a.values())
        balance_err = abs(dose_total - (in_body + eliminated)) / max(dose_total, 1e-9)
        cl_eff = spec.cl_l_per_h_per_kg * bw * (
            spec.normalized_clearance_split()[0] * min(1.6, max(0.05, self.patient.egfr / 90.0))
            + spec.normalized_clearance_split()[1] * max(0.05, self.patient.hepatic_factor)
            + spec.normalized_clearance_split()[2])

        metrics: Dict[str, Any] = {
            "unit": spec.conc_unit,
            "cmax": round(float(arr.max()) if len(arr) else 0.0, 4),
            "tmax_h": round(tmax, 3),
            "auc": round(cum_auc * us, 3),
            "auc24": round(cum_auc * us * 24.0 / horizon, 3),
            "dose_total_mg": round(dose_total, 1),
            "clearance_l_h": round(cl_eff, 2),
            "vd_l": round(spec.vd_l_per_kg * bw, 2),
            "t_half_published_h": spec.t_half_h,
            "mass_balance_error": round(float(balance_err), 8),
            "eliminated_mg": round(eliminated, 1),
            "remaining_mg": round(in_body, 1),
            "cardiac_output_l_h": round(self.co, 1),
            "continuous": bool(reg.continuous),
        }
        if t_ss is not None:
            metrics["t_to_90pct_steady_state_h"] = round(t_ss, 4)
            metrics["css"] = round(float(arr[-1]), 4)
        if spec.mic_mg_l:
            # %fT>MIC is a *free*-drug criterion, and MIC is always quoted in mg/L —
            # so this comparison runs on the unscaled model concentration.
            free = 1.0 - spec.protein_binding
            unscaled = np.array(out["arterial"])
            above = float(np.mean(unscaled * free >= spec.mic_mg_l)) if len(unscaled) else 0.0
            metrics["mic_mg_l"] = spec.mic_mg_l
            metrics["pct_ft_above_mic"] = round(100.0 * above, 1)
        # Per-organ exposure drives the adverse-event layer downstream.
        metrics["organ_auc"] = {k: round(float(np.trapezoid(np.array(v) * us, sample_at)), 4)
                                for k, v in out.items()}

        events = self._events(spec, reg, sample_at, out, infusions)

        return PKProjection(
            drug=spec.key, label=spec.label, drug_class=spec.drug_class,
            kind="concentration", unit=spec.conc_unit,
            times_h=[round(float(t), 4) for t in sample_at],
            plasma=[round(x * us, 6) for x in out_plasma], organs=organs, peak=peak,
            metrics=metrics, regimen=reg.label or spec.regimen.label,
            horizon_h=horizon, patient=self.patient.to_dict(),
            provenance={"physiology": PHYSIOLOGY_SOURCE, "drug": spec.citation,
                        "target": spec.target_note},
            events=events,
            honesty=("Literature-parameterised population PBPK projection — not a "
                     "therapeutic-drug-monitoring prediction for this individual. "
                     "No measured serum levels exist in this system to fit against."),
        )

    # ---------------------------------------------------------------- events --
    def _events(self, spec: DrugSpec, reg: Regimen, times: np.ndarray,
                out: Dict[str, List[float]],
                infusions: List[Tuple[float, float, float]]) -> List[Dict[str, Any]]:
        """Landmarks the renderer annotates on the timeline."""
        ev: List[Dict[str, Any]] = []
        for i, (s, e, _r) in enumerate(infusions[:6]):
            ev.append({"t_h": round(s, 3), "kind": "infusion_start",
                       "label": ("infusion begins" if i == 0 else f"dose {i+1}")})
            if not reg.continuous:
                ev.append({"t_h": round(e, 3), "kind": "infusion_end",
                           "label": "infusion complete"})
        arr = np.array(out["arterial"])
        if len(arr) and arr.max() > 0:
            if reg.continuous:
                idx = int(np.argmax(arr >= 0.90 * arr[-1]))
                ev.append({"t_h": round(float(times[idx]), 3), "kind": "steady_state",
                           "label": "90% of steady state"})
            else:
                ev.append({"t_h": round(float(times[int(np.argmax(arr))]), 3),
                           "kind": "cmax", "label": "peak plasma concentration"})
        # First arrival in each organ = 5% of that organ's own peak.
        for key in ("lung", "liver", "kidney", "brain"):
            v = np.array(out.get(key, []))
            if not len(v) or v.max() <= 0:
                continue
            idx = int(np.argmax(v >= 0.05 * v.max()))
            ev.append({"t_h": round(float(times[idx]), 3), "kind": "arrival",
                       "organ": key, "label": f"reaches {key}"})
        return sorted(ev, key=lambda e: e["t_h"])

    # ----------------------------------------------------------------- fluid --
    def project_fluid(self, regimen: Optional[Regimen] = None,
                      horizon_h: Optional[float] = None,
                      n_points: int = 121) -> PKProjection:
        """Crystalloid resuscitation, as **volume kinetics** — not concentration.

        A litre of Ringer's has no Cmax; what matters is where the volume ends up.
        Hahn's two-compartment volume-kinetic model tracks expansion of the central
        (plasma) space, redistribution into the interstitium at rate k_t, **return
        from the interstitium by lymphatic drainage at k_b**, and renal elimination
        at k_r. Only ~20–25% of an infused crystalloid remains intravascular at one
        hour — the rest is the tissue oedema the renderer shows pooling into the
        organs, and the mechanism behind the pulmonary-oedema adverse event
        downstream.

        The return limb is not decoration: without it the interstitium is a one-way
        sink and infused volume simply disappears, which is exactly what the fluid
        mass-balance check refuses to accept. With it, infused = plasma +
        interstitium + excreted holds identically at every step.
        """
        bw = self.patient.weight_kg
        reg = regimen or Regimen(rate_mcg_kg_min=0.0, continuous=True,
                                 label="crystalloid 10 mL/kg/h × 1 h")
        horizon = float(horizon_h or 4.0)
        # rate_mcg_kg_min is reused as mL/kg/h for the fluid arm (see callers).
        ml_per_kg_h = reg.rate_mcg_kg_min or 10.0
        infuse_h = min(1.0, horizon)
        rate_ml_h = ml_per_kg_h * bw

        v_c0 = 0.043 * bw * 1000.0           # central (plasma) volume, mL
        v_t0 = 3.0 * v_c0                    # interstitial space ≈ 3× plasma volume
        k_t = 130.0 * 60.0                   # plasma → interstitium, mL/h (Hahn 2010)
        k_b = k_t                            # lymphatic return, mL/h (equal rate
        #                                      constants ⇒ equilibrium at equal
        #                                      fractional expansion, i.e. ~75% of
        #                                      retained volume ends up interstitial)
        # Renal elimination of a fluid load. Hahn's k_r spans ~10–110 mL/min; the
        # high end is an awake, euvolaemic volunteer. A hypotensive septic patient
        # sits at the low end — which is precisely why resuscitation fluid
        # accumulates instead of being excreted — so the base rate is the low end
        # and is scaled by the patient's own filtration.
        k_r = 20.0 * 60.0 * min(1.5, max(0.05, self.patient.egfr / 90.0))

        n_steps = 4000
        dt = horizon / n_steps
        vc, vt, out_ml = 0.0, 0.0, 0.0       # expansions above baseline, mL
        sample_at = np.linspace(0.0, horizon, n_points)
        s_idx = 0
        series_c: List[float] = []
        series_t: List[float] = []
        infused_total = 0.0
        for step in range(n_steps + 1):
            t_h = step * dt
            while s_idx < n_points and t_h >= sample_at[s_idx] - 1e-12:
                series_c.append(vc)
                series_t.append(vt)
                s_idx += 1
            if step == n_steps:
                break
            r_in = rate_ml_h if t_h < infuse_h else 0.0
            frac_c = vc / v_c0                 # fractional plasma expansion
            frac_t = vt / v_t0                 # fractional interstitial expansion
            d_vc = r_in - k_t * frac_c + k_b * frac_t - k_r * frac_c
            d_vt = k_t * frac_c - k_b * frac_t
            vc = max(0.0, vc + d_vc * dt)
            vt = max(0.0, vt + d_vt * dt)
            out_ml += k_r * frac_c * dt
            infused_total += r_in * dt

        # The interstitial volume distributes across tissues in proportion to each
        # organ's own interstitial space (Starling equilibrium, first order).
        interstitial_share = {t: self.v[t] for t in TISSUES}
        interstitial_share["lung"] = self.v_lung
        tot_share = sum(interstitial_share.values())
        organs: Dict[str, List[float]] = {}
        for k, share in interstitial_share.items():
            f = share / tot_share
            organs[k] = [round(x * f, 4) for x in series_t]
        organs["arterial"] = [round(x * ARTERIAL_SHARE, 4) for x in series_c]
        organs["venous"] = [round(x * (1.0 - ARTERIAL_SHARE), 4) for x in series_c]

        peak_c = max(series_c) if series_c else 0.0

        def retention_at(t_h: float) -> float:
            if horizon < t_h:
                return 0.0
            i = int(np.argmin(np.abs(sample_at - t_h)))
            return series_c[i] / max(infused_total, 1e-9)

        # Two different numbers that both get called "retention". At the moment the
        # bolus finishes, most of it is still intravascular. The widely quoted
        # 20–25% figure is measured an hour AFTER the infusion ends, once
        # redistribution has run — so both are reported, distinctly named.
        retained_eoi = retention_at(infuse_h)
        retained_1h_post = retention_at(infuse_h + 1.0)

        metrics = {
            "infused_ml": round(infused_total, 0),
            "peak_plasma_expansion_ml": round(peak_c, 0),
            "plasma_retention_end_of_infusion": round(retained_eoi, 3),
            "plasma_retention_1h_post": round(retained_1h_post, 3),
            "interstitial_ml_end": round(series_t[-1] if series_t else 0.0, 0),
            "excreted_ml": round(out_ml, 0),
            "organ_auc": {k: round(float(np.trapezoid(np.array(v), sample_at)), 2)
                          for k, v in organs.items()},
        }
        return PKProjection(
            drug="crystalloid", label="Crystalloid (balanced)", drug_class="fluid",
            kind="volume", unit="mL",
            times_h=[round(float(t), 4) for t in sample_at],
            plasma=[round(x, 4) for x in series_c], organs=organs,
            peak={k: round(max(v), 4) for k, v in organs.items()},
            metrics=metrics, regimen=reg.label, horizon_h=horizon,
            patient=self.patient.to_dict(),
            provenance={"physiology": PHYSIOLOGY_SOURCE, "drug": FLUID_SOURCE,
                        "target": "only ~20–25% of infused crystalloid is still "
                                  "intravascular at 1 h"},
            events=[{"t_h": 0.0, "kind": "infusion_start", "label": "bolus begins"},
                    {"t_h": round(infuse_h, 2), "kind": "infusion_end",
                     "label": "bolus complete"}],
            honesty=("Population volume kinetics (Hahn). Distribution across organs "
                     "is proportional to interstitial volume — an equilibrium "
                     "assumption, not a measured per-organ oedema map."),
        )


# --------------------------------------------------------------------------- #
# Protocol-level projection — what the OS actually ordered.
# --------------------------------------------------------------------------- #
def patient_from_profile(prof: Any, hist: Optional[Dict[str, Any]] = None) -> PatientState:
    """Derive PK covariates from a live bed profile + imported history."""
    hist = hist or {}
    en = getattr(prof, "enterprise", {}) or {}
    mapv = float(en.get("map", 75.0) or 75.0)
    hr = float(en.get("heart_rate", en.get("hr", 90.0)) or 90.0)
    egfr = 90.0
    if hist.get("renal_failure"):
        egfr = 25.0
    elif float(en.get("creatinine", 1.0) or 1.0) > 1.5:
        egfr = 45.0
    hep = 0.6 if hist.get("cirrhosis") or hist.get("hepatic_failure") else 1.0
    # Sepsis is typically hyperdynamic: tachycardia raises cardiac output.
    co_factor = float(min(1.6, max(0.7, 0.85 + (hr - 80.0) / 200.0)))
    return PatientState(weight_kg=float(hist.get("weight_kg", 78.0)), egfr=egfr,
                        hepatic_factor=hep, map_mmhg=mapv,
                        cardiac_output_factor=co_factor)


def project_protocol(protocol: Dict[str, Any], patient: PatientState,
                     n_points: int = 121) -> Dict[str, Any]:
    """Project every arm of a certified Hour-1 order.

    ``protocol`` is the bundle engine's dict: ``fluid_rate`` (mL/kg/h),
    ``vasopressor_dose`` (mcg/kg/min norepinephrine), ``antibiotic_code``.
    """
    model = PBPKModel(patient)
    arms: List[Dict[str, Any]] = []

    code = int(protocol.get("antibiotic_code", 0) or 0)
    drug = ANTIBIOTIC_CODE_TO_DRUG.get(code)
    if drug:
        arms.append(model.project(drug, n_points=n_points).to_dict())

    vaso = float(protocol.get("vasopressor_dose", 0.0) or 0.0)
    if vaso > 0:
        spec = DRUGS["norepinephrine"]
        reg = Regimen(continuous=True, rate_mcg_kg_min=vaso,
                      label=f"norepinephrine {vaso:.2f} mcg/kg/min, titrated to MAP ≥ 65")
        arms.append(model.project("norepinephrine", regimen=reg,
                                  horizon_h=spec.horizon_h, n_points=n_points).to_dict())

    fluid = float(protocol.get("fluid_rate", 0.0) or 0.0)
    if fluid > 0:
        reg = Regimen(continuous=True, rate_mcg_kg_min=fluid,
                      label=f"crystalloid {fluid:.0f} mL/kg/h × 1 h")
        arms.append(model.project_fluid(regimen=reg, n_points=n_points).to_dict())

    return {
        "arms": arms,
        "patient": patient.to_dict(),
        "horizon_h": max([a["horizon_h"] for a in arms], default=0.0),
        "protocol": {k: protocol.get(k) for k in
                     ("fluid_rate", "vasopressor_dose", "antibiotic_code")},
        "honesty": ("Each arm is projected independently; the model does not "
                    "simulate pharmacodynamic interaction between the arms."),
    }


# --------------------------------------------------------------------------- #
def _demo() -> None:
    pt = PatientState(weight_kg=78, egfr=90, map_mmhg=58)
    model = PBPKModel(pt)
    print(f"cardiac output {model.co:.0f} L/h · shock index {pt.shock_index():.2f}\n")
    for key in ("piperacillin-tazobactam", "vancomycin", "norepinephrine"):
        p = model.project(key)
        m = p.metrics
        print(f"{p.label:26s} {p.regimen}")
        print(f"  Cmax {m['cmax']:8.2f} {p.unit} @ {m['tmax_h']*60:5.1f} min · "
              f"AUC {m['auc']:8.2f} {p.unit}·h · CL {m['clearance_l_h']:.1f} L/h")
        if "pct_ft_above_mic" in m:
            print(f"  %fT>MIC {m['pct_ft_above_mic']:.0f}%  (MIC {m['mic_mg_l']} mg/L)")
        peaks = sorted(((k, v) for k, v in p.peak.items()
                        if k in ("lung", "liver", "kidney", "heart", "brain", "spleen")),
                       key=lambda kv: -kv[1])
        print("  organ peaks: " + " · ".join(f"{k} {v:.1f}" for k, v in peaks))
        print(f"  mass-balance error {m['mass_balance_error']:.2e}\n")
    f = PBPKModel(pt).project_fluid(Regimen(continuous=True, rate_mcg_kg_min=10.0,
                                            label="10 mL/kg/h × 1 h"))
    print(f"{f.label:26s} {f.regimen}")
    print(f"  infused {f.metrics['infused_ml']:.0f} mL · plasma retention @1h "
          f"{f.metrics['plasma_retention_1h_post']*100:.0f}% · interstitial "
          f"{f.metrics['interstitial_ml_end']:.0f} mL")


if __name__ == "__main__":
    _demo()
