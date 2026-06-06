"""Central configuration: paths, channel definitions, and label semantics.

Keeping these in one place means the loader, feature builder, every model, and
the evaluation harness all agree on exactly which signals exist and what the
label means. Disagreement here is the classic source of silent data leakage.
"""
from __future__ import annotations

import os
from typing import Dict, List

# --------------------------------------------------------------------------- #
# Filesystem layout
# --------------------------------------------------------------------------- #
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(REPO_ROOT, "data", "cinc2019")
ARTIFACTS_DIR = os.path.join(REPO_ROOT, "artifacts")

# PhysioNet/CinC 2019 lives as one .psv per patient under two training sets.
CINC_BASE_URL = "https://physionet.org/files/challenge-2019/1.0.0/training"
CINC_SETS = ("training_setA", "training_setB")

# --------------------------------------------------------------------------- #
# Channel taxonomy
# --------------------------------------------------------------------------- #
# Raw CinC column -> the internal vital name used by the existing engine/twin.
# We are deliberately explicit about clinical caveats in the comments because
# the wearable analogue is not always physiologically identical to the ICU
# measurement the model is trained on.
CINC_TO_INTERNAL: Dict[str, str] = {
    "HR": "heart_rate",          # bpm — directly comparable to a PPG wrist HR
    "O2Sat": "spo2",             # % — pulse oximetry, comparable to wrist SpO2
    "Temp": "skin_temp",         # CinC is CORE temp (°C); a wrist sensor reads
                                 # skin temp. Mapped for continuity but flagged.
    "SBP": "systolic_bp",        # mmHg — cuff BP; cuffless wearable BP is noisier
    "MAP": "map",                # mmHg
    "DBP": "diastolic_bp",       # mmHg
    "Resp": "respiratory_rate",  # breaths/min — derivable from PPG/accelerometer
    "Lactate": "sweat_lactate",  # CinC is BLOOD lactate (mmol/L); the project's
                                 # "sweat_lactate" is a future-sensor proxy.
    "WBC": "wbc",                # k/uL — lab only, sparsely sampled
}

# Channels a wrist wearable can plausibly stream in real time. This is the
# honest "wearable-only" feature set the deployed product would actually have.
WEARABLE_CHANNELS: List[str] = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp",
]

# Sparsely-sampled labs. Including them defines the "lab-augmented" feature set,
# which we report separately so the wearable-only ceiling is never overstated.
LAB_CHANNELS: List[str] = [
    "Lactate", "WBC", "Creatinine", "Platelets", "BUN",
    "HCO3", "pH", "PaCO2", "Bilirubin_total", "Glucose",
]

# Static per-patient context.
DEMOGRAPHIC_CHANNELS: List[str] = ["Age", "Gender", "HospAdmTime", "ICULOS"]

# Full ordered column header of every CinC .psv (used to validate files).
CINC_COLUMNS: List[str] = [
    "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2",
    "BaseExcess", "HCO3", "FiO2", "pH", "PaCO2", "SaO2", "AST", "BUN",
    "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct",
    "Glucose", "Lactate", "Magnesium", "Phosphate", "Potassium",
    "Bilirubin_total", "TroponinI", "Hct", "Hgb", "PTT", "WBC", "Fibrinogen",
    "Platelets", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
    "SepsisLabel",
]

LABEL_COLUMN = "SepsisLabel"

# --------------------------------------------------------------------------- #
# Label semantics (this is the crux of an honest evaluation)
# --------------------------------------------------------------------------- #
# In CinC 2019, for a patient who becomes septic at hour t_sepsis, SepsisLabel
# is 1 for every hour t >= t_sepsis - 6. That 6-hour pull-forward is the entire
# point of the challenge: reward EARLY prediction. We preserve that label as-is
# for benchmarking, and separately expose the true onset hour so we can measure
# real early-warning lead time.
SEPSIS_LABEL_LEAD_HOURS = 6
