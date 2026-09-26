"""What was known about a patient on ICU arrival — where they came from, which service
admitted them, and their comorbidities from PRIOR hospital admissions.

On CinC 2019 the between-patient part of the AUROC ("is this somebody who becomes septic?")
carried 90% of the pooled number (docs/GENESIS.md), and the only per-patient context was age,
sex and unit. MIMIC-IV records far more: the admission source (emergency department,
transfer, referral, procedure), the admitting service, and every diagnosis coded in the
patient's earlier admissions.

LEAKAGE. A hospital admission's diagnoses are coded at DISCHARGE — for the current admission
that is the future, and would include "sepsis" itself. Only admissions discharged before the
current one began are read; ``verify_mimic`` plants a sepsis code in the current admission
and asserts no feature moves.

Comorbidity groups: the Charlson index in the Quan et al. (Med Care 2005) ICD-9-CM and
ICD-10 codings, plus prior sepsis (ICD-10 A40 / A41 / R65.2; ICD-9 038, 995.91, 995.92,
785.52). MIMIC stores codes without the dot, so every entry is a code prefix.
"""
from __future__ import annotations

import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SOURCES: Dict[str, Tuple[str, ...]] = {
    "ED": ("EMERGENCY ROOM",),
    "Transfer": ("TRANSFER FROM HOSPITAL", "TRANSFER FROM SKILLED NURSING FACILITY",
                 "INTERNAL TRANSFER TO OR FROM PSYCH"),
    "Referral": ("PHYSICIAN REFERRAL", "CLINIC REFERRAL", "WALK-IN/SELF REFERRAL"),
    "Procedure": ("PROCEDURE SITE", "PACU", "AMBULATORY SURGERY TRANSFER"),
}
SRC_COLS = [f"Src_{k}" for k in SOURCES] + ["Src_Other"]

SERVICES: Dict[str, Tuple[str, ...]] = {
    "MED": ("MED", "OMED"), "CMED": ("CMED",), "NMED": ("NMED",),
    "SURG": ("SURG", "VSURG", "PSURG", "GYN", "ORTHO", "ENT", "EYE", "DENT", "GU", "OBS"),
    "CSURG": ("CSURG",), "NSURG": ("NSURG",), "TRAUMA": ("TRAUM", "TSURG"),
}
SVC_COLS = [f"Svc_{k}" for k in SERVICES] + ["Svc_Other"]

CHARLSON10 = {
    "mi": "I21 I22 I252",
    "chf": "I099 I110 I130 I132 I255 I420 I425-I429 I43 I50 P290",
    "pvd": "I70 I71 I731 I738 I739 I771 I790 I792 K551 K558 K559 Z958 Z959",
    "cvd": "G45 G46 H340 I60-I69",
    "dementia": "F00-F03 F051 G30 G311",
    "copd": "I278 I279 J40-J47 J60-J67 J684 J701 J703",
    "rheum": "M05 M06 M315 M32-M34 M351 M353 M360",
    "pud": "K25-K28",
    "mild_liver": "B18 K700-K703 K709 K713-K715 K717 K73 K74 K760 K762-K764 K768 K769 Z944",
    "diabetes": "E100 E101 E106 E108 E109 E110 E111 E116 E118 E119 E120 E121 E126 E128 E129 "
                "E130 E131 E136 E138 E139 E140 E141 E146 E148 E149",
    "diabetes_cx": "E102-E105 E107 E112-E115 E117 E122-E125 E127 E132-E135 E137 E142-E145 E147",
    "plegia": "G041 G114 G801 G802 G81 G82 G830-G834 G839",
    "renal": "I120 I131 N032-N037 N052-N057 N18 N19 N250 Z490-Z492 Z940 Z992",
    "cancer": "C00-C26 C30-C34 C37-C41 C43 C45-C58 C60-C76 C81-C85 C88 C90-C97",
    "severe_liver": "I850 I859 I864 I982 K704 K711 K721 K729 K765 K766 K767",
    "metastatic": "C77-C80",
    "hiv": "B20-B22 B24",
    "sepsis": "A40 A41 R652",
}
CHARLSON9 = {
    "mi": "410 412",
    "chf": "39891 40201 40211 40291 40401 40403 40411 40413 40491 40493 4254-4259 428",
    "pvd": "0930 4373 440 441 4431-4439 4471 5571 5579 V434",
    "cvd": "36234 430-438",
    "dementia": "290 2941 3312",
    "copd": "4168 4169 490-505 5064 5081 5088",
    "rheum": "4465 7100-7104 7140-7142 7148 725",
    "pud": "531-534",
    "mild_liver": "07022 07023 07032 07033 07044 07054 0706 0709 570 571 5733 5734 5738 5739 V427",
    "diabetes": "2500-2503 2508 2509",
    "diabetes_cx": "2504-2507",
    "plegia": "3341 342 343 3440-3446 3449",
    "renal": "40301 40311 40391 40402 40403 40412 40413 40492 40493 582 5830-5837 585 586 5880 "
             "V420 V451 V56",
    "cancer": "140-172 174-194 1950-1958 200-208 2386",
    "severe_liver": "4560-4562 5722-5728",
    "metastatic": "196-199",
    "hiv": "042-044",
    "sepsis": "038 99591 99592 78552",
}
GROUPS = list(CHARLSON10)
HX_COLS = [f"Hx_{g}" for g in GROUPS] + ["Hx_prior_adm", "Hx_prior_adm_1y", "Hx_days_since_discharge"]


def expand(spec: str) -> Tuple[str, ...]:
    """'I425-I429 I43' -> ('I425', 'I426', 'I427', 'I428', 'I429', 'I43')."""
    out: List[str] = []
    for tok in spec.split():
        if "-" not in tok:
            out.append(tok)
            continue
        a, b = tok.split("-")
        ma, mb = re.fullmatch(r"([A-Z]*)(\d+)", a), re.fullmatch(r"([A-Z]*)(\d+)", b)
        if not ma or not mb or ma.group(1) != mb.group(1) or len(ma.group(2)) != len(mb.group(2)):
            raise ValueError(f"bad code range {tok!r}")
        w = len(ma.group(2))
        out += [f"{ma.group(1)}{i:0{w}d}" for i in range(int(ma.group(2)), int(mb.group(2)) + 1)]
    return tuple(out)


_RX = {(v, g): re.compile("^(" + "|".join(expand(spec)) + ")")
       for v, table in ((10, CHARLSON10), (9, CHARLSON9)) for g, spec in table.items()}


def code_groups(codes: pd.Series, version: pd.Series) -> pd.DataFrame:
    """One boolean column per group for each (code, version) row."""
    c = codes.astype(str).str.strip().str.upper().str.replace(".", "", regex=False)
    v = version.astype(int).to_numpy()
    out = {}
    for g in GROUPS:
        hit = np.zeros(len(c), dtype=bool)
        for ver in (9, 10):
            m = v == ver
            if m.any():
                hit[m] = c[m].str.match(_RX[(ver, g)]).to_numpy()
        out[g] = hit
    return pd.DataFrame(out, index=codes.index)


def admission_context(stays: pd.DataFrame, services: pd.DataFrame) -> pd.DataFrame:
    """Per stay: admission-source and admitting-service one-hots. ``stays`` needs stay_id,
    hadm_id, intime, admission_location; the service is the latest one assigned at or before
    ICU admission (the admission's first service if none precedes it)."""
    loc = stays.admission_location.fillna("").astype(str).str.upper().str.strip()
    out = pd.DataFrame({"stay_id": stays.stay_id.to_numpy()})
    other = np.ones(len(stays), dtype=bool)
    for k, names in SOURCES.items():
        hit = loc.isin(names).to_numpy()
        out[f"Src_{k}"] = hit.astype(float)
        other &= ~hit
    out["Src_Other"] = other.astype(float)
    sv = services.merge(stays[["stay_id", "hadm_id", "intime"]], on="hadm_id")
    sv = sv.sort_values(["stay_id", "transfertime"])
    before = sv[sv.transfertime <= sv.intime].groupby("stay_id").curr_service.last()
    first = sv.groupby("stay_id").curr_service.first()
    svc = out.stay_id.map(before).fillna(out.stay_id.map(first)).fillna("").astype(str).str.upper()
    other = np.ones(len(stays), dtype=bool)
    for k, names in SERVICES.items():
        hit = svc.isin(names).to_numpy()
        out[f"Svc_{k}"] = hit.astype(float)
        other &= ~hit
    out["Svc_Other"] = other.astype(float)
    return out


def prior_history(stays: pd.DataFrame, admissions: pd.DataFrame, diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Per stay: comorbidity groups from admissions DISCHARGED before this admission began,
    how many such admissions (ever / last year), and days since the last discharge."""
    cur = stays[["stay_id", "subject_id", "admittime"]].copy()
    adm = admissions[["subject_id", "hadm_id", "admittime", "dischtime"]].rename(
        columns={"hadm_id": "prior_hadm", "admittime": "p_admit", "dischtime": "p_disch"})
    prior = cur.merge(adm, on="subject_id")
    prior = prior[prior.p_disch <= prior.admittime]
    out = pd.DataFrame({"stay_id": cur.stay_id.to_numpy()})
    n = prior.groupby("stay_id").size()
    recent = prior[prior.p_disch >= prior.admittime - pd.Timedelta(days=365)].groupby("stay_id").size()
    grp = prior.groupby("stay_id")
    last = ((grp.admittime.first() - grp.p_disch.max()).dt.total_seconds() / 86400.0) \
        if len(prior) else pd.Series(dtype=float)
    dx = prior[["stay_id", "prior_hadm"]].merge(
        diagnoses[["hadm_id", "icd_code", "icd_version"]].rename(columns={"hadm_id": "prior_hadm"}), on="prior_hadm")
    if len(dx):
        flags = code_groups(dx.icd_code, dx.icd_version)
        flags["stay_id"] = dx.stay_id.to_numpy()
        per = flags.groupby("stay_id").any()
    else:
        per = pd.DataFrame(columns=GROUPS)
    for g in GROUPS:
        flag = per[g].astype(float) if g in per else pd.Series(dtype=float)
        out[f"Hx_{g}"] = out.stay_id.map(flag).fillna(0.0).to_numpy(float)
    out["Hx_prior_adm"] = out.stay_id.map(n).fillna(0).astype(float)
    out["Hx_prior_adm_1y"] = out.stay_id.map(recent).fillna(0).astype(float)
    out["Hx_days_since_discharge"] = out.stay_id.map(last).astype(float)          # NaN: no prior admission
    return out
