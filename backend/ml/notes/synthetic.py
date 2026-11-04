"""Invented notes with planted traps, laid out like the release. No real patient, no real text.

Two patients. Patient 1 has an earlier admission (A1, ended a week before) and the current one
(A2, ICU stay S1 from hour 0 to hour 48). Patient 2 has one admission and one stay, S2.

Traps, each tied to a check in ``verify_notes``:

  r_late      film taken at hour 2, read stored at hour 9           -> hour 9, never 2
  r_nostore   no storetime                                          -> dropped
  r_after     stored after the stay ended                           -> dropped
  r_ed        stored 10 h before ICU admission (the ED film)        -> hour 0
  r_old       stored 30 h before ICU admission                      -> dropped (past the lookback)
  r_neg       "No pneumonia. Pleural effusion has resolved."        -> pneumonia 0, effusion 0
  r_hedge     "Possible right lower lobe consolidation."            -> pneumonia 0.5
  r_ind       indication asks about fever; impression is clean      -> ind_infection 1, pneumonia 0
  r_two       two reads stored in the same hour                     -> max per finding, count 2
  r_other     patient 2's report                                    -> stay S2 only

  d_prior     A1's summary, written before S1                       -> history of S1
  d_current   A2's summary, written at discharge                    -> never read for S1
"""
from __future__ import annotations

import gzip
import os
from typing import Dict

import pandas as pd

T0 = pd.Timestamp("2180-03-01 08:00")


def h(x: float) -> pd.Timestamp:
    return T0 + pd.Timedelta(hours=x)


def world() -> Dict[str, pd.DataFrame]:
    stays = pd.DataFrame({"stay_id": [1001, 2001], "subject_id": [1, 2], "hadm_id": [12, 21],
                          "intime": [h(0), h(0)], "outtime": [h(48), h(30)]})
    admissions = pd.DataFrame({"hadm_id": [11, 12, 21], "subject_id": [1, 1, 2],
                               "admittime": [h(-24 * 14), h(-6), h(-2)], "dischtime": [h(-24 * 7), h(24 * 6), h(24 * 3)]})
    R = [
        ("r_late", 1, 12, 2, 9, "FINDINGS: Dense right lower lobe consolidation."),
        ("r_nostore", 1, 12, 3, None, "IMPRESSION: Large left pleural effusion."),
        ("r_after", 1, 12, 47, 50, "IMPRESSION: Pneumoperitoneum, concerning for perforation."),
        ("r_ed", 1, 12, -11, -10, "IMPRESSION: Cholecystitis."),
        ("r_old", 1, 11, -31, -30, "IMPRESSION: Cellulitis of the left leg."),
        ("r_neg", 1, 12, 13.5, 14.2, "IMPRESSION: No pneumonia. Pleural effusion has resolved."),
        ("r_hedge", 1, 12, 20, 21.5, "IMPRESSION: Possible right lower lobe consolidation."),
        ("r_ind", 1, 12, 30, 31, "INDICATION: Fever, evaluate for pneumonia.\nIMPRESSION: No acute cardiopulmonary process."),
        ("r_two_a", 1, 12, 35, 35.1, "IMPRESSION: Pulmonary edema."),
        ("r_two_b", 1, 12, 35, 35.8, "IMPRESSION: Bibasilar atelectasis. Possible abscess."),
        ("r_other", 2, 21, 5, 6, "IMPRESSION: Hydronephrosis with perinephric stranding."),
    ]
    radiology = pd.DataFrame([{"note_id": n, "subject_id": s, "hadm_id": a, "charttime": h(c),
                               "storetime": h(t) if t is not None else pd.NaT, "text": x} for n, s, a, c, t, x in R])
    D = [
        ("d_prior", 1, 11, -24 * 7, "Discharge diagnosis: urosepsis with pyelonephritis. Prior abscess drained."),
        ("d_current", 1, 12, 24 * 6, "Hospital course: septic shock from pneumonia with empyema."),
    ]
    discharge = pd.DataFrame([{"note_id": n, "subject_id": s, "hadm_id": a, "charttime": h(c),
                               "storetime": h(c + 1), "text": x} for n, s, a, c, x in D])
    return {"stays": stays, "admissions": admissions, "radiology": radiology, "discharge": discharge}


def write_release(root: str) -> str:
    """The notes as ``<root>/note/{radiology,discharge}.csv.gz``, the release's layout."""
    w = world()
    os.makedirs(os.path.join(root, "note"), exist_ok=True)
    for t in ("radiology", "discharge"):
        with gzip.open(os.path.join(root, "note", f"{t}.csv.gz"), "wt") as fh:
            w[t].to_csv(fh, index=False, date_format="%Y-%m-%d %H:%M:%S")   # as the release writes them
    return root
