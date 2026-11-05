"""Will the full run fit this laptop? Estimated before a single feature is built.

Full MIMIC-IV is ~94k ICU stays and ~6-8M hourly rows. Built naively — every stay, 1,000+
features, three tiers of boosters with their products, two twins, the harness — the run
needs more disk than this machine has free, more memory than it has, and days of CPU. The
planner reads the extracted store (stays, hours, how many ever become septic), the feature
count (built exactly on a handful of stays), the machine (free disk, RAM) and a time budget,
and walks from the richest plan to the leanest until one fits — or refuses, naming what
does not fit, before hours are spent.

Cost models, calibrated where this repo has measurements:
  * one LightGBM fit ~ ``SEC_PER_ROW_COL`` x training rows x columns, measured with the FULL
    preset itself (learning rate 0.02, 31 leaves, 30% features, 70% bagging) on real CinC:
    995k rows x 510 columns in 102.6 s, 452 trees + patience, ingestion included. The first
    constant (1.8e-7) came from the untuned all-variable model, which stopped at ~160
    trees; the slower tuned boosting grows ~3x the trees and lands within 12% of it;
  * ``CONSTRUCT_SHARE`` of a fit is binning the data (16.5 of those 102.6 s); a head that
    reuses its fold's dataset (``learners.shared_datasets``: who / soft / distill beside
    direct) skips it;
  * the record twin ~ ``TWIN_SEC_PER_ROW_STEP`` per (row x optimiser step) at d = 128
    (extrapolated from ~17 ms per 32-stay step at d = 64 on the demo), the physiology twin 2x;
  * float16 features on disk; LightGBM's binned copy ~1 byte per training cell, plus its
    bin-boundary sample — up to ``SAMPLE_ROWS`` rows x columns x ~10 bytes (the float16
    gather and LightGBM's float64 columns; ~2 GB at 1,000 columns); the twin's inputs ~220
    float32 per row.
These are estimates to plan with, not promises — the report records the plan and the actual
time beside it.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import asdict, dataclass, replace
from typing import Dict, List, Tuple

import numpy as np

from backend.ml import config

SEC_PER_ROW_COL = 2.0e-7
#: the thread count both time constants were measured at (GENESIS_THREADS on the laptop). A
#: machine with fewer threads is slower in proportion, which overstates a many-core machine's
#: time a little (scaling isn't linear) and never understates a small one's.
CALIBRATION_THREADS = 6
CONSTRUCT_SHARE = 0.16
SAMPLE_ROWS = 200_000                 # LightGBM's bin_construct_sample_cnt
TWIN_SEC_PER_ROW_STEP = 1.2e-5
TWIN_INPUTS_PER_ROW = 220
HARNESS_HOURS = 0.75                  # audits, cross-examination, transplant (sampled), lock plans
#: Memory a run holds but rarely touches: every model's out-of-fold scores, masks, the Python
#: heap. Measured on the first full MIMIC-IV run, during the P tier: 10.1 GB of anonymous memory
#: (5.4 resident + 4.7 in swap, with almost no swap traffic) at 2,364,579 kept rows, of which the
#: hot estimate below accounts for 5.26 GB. The rest, 4.84 GB, is ``COLD_BASE_GB`` plus
#: ``COLD_BYTES_PER_ROW`` per kept row. The old planner had only the 2 GB base, and undercounted
#: by about 40%.
COLD_BASE_GB = 2.0
COLD_BYTES_PER_ROW = 1200
#: of free swap, what cold memory may use. Swap only ever holds cold memory: a hot set that
#: doesn't fit in RAM is refused whatever the swap, because training against swap thrashes.
SWAP_FRACTION = 0.8


@dataclass
class Budget:
    disk_gb: float
    ram_gb: float
    hours: float
    swap_gb: float = 0.0               # for cold memory only; 0 means everything must fit in RAM

    @staticmethod
    def detect(hours: float = 12.0, disk_reserve_gb: float = 6.0, ram_fraction: float = None,
               path: str = config.ARTIFACTS_DIR) -> "Budget":
        if ram_fraction is None:       # a dedicated machine runs nothing else (deploy/physionet/run.sh)
            ram_fraction = float(os.environ.get("MIMIC_RAM_FRACTION", "0.65"))
        free = shutil.disk_usage(path).free / 1e9
        ram = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
        return Budget(disk_gb=free - disk_reserve_gb, ram_gb=ram * ram_fraction, hours=hours,
                      swap_gb=free_swap_gb() * SWAP_FRACTION)


def free_swap_gb() -> float:
    """Free swap on Linux (``/proc/meminfo``). Anywhere else, 0: macOS grows swap on demand on
    the same disk as the run, so it isn't counted on."""
    try:
        with open("/proc/meminfo") as fh:
            for line in fh:
                if line.startswith("SwapFree:"):
                    return int(line.split()[1]) * 1024 / 1e9
    except OSError:
        pass
    return 0.0


#: from the richest plan to the leanest; each overrides fields of the scale
STAGE_SETS: List[Tuple[str, Dict]] = [
    ("full", {}),
    ("no tier T, one record-twin seed", {"tiers": ("P", "S"), "twin_seeds": (0,)}),
    ("lean: no soft label, fewer tuning trials", {"tiers": ("P", "S"), "twin_seeds": (0,), "soft": False,
                                                  "tune_trials": 8}),
    ("lean, no distilled student", {"tiers": ("P", "S"), "twin_seeds": (0,), "soft": False, "distill": False,
                                    "tune_trials": 8}),
    ("core: direct boosters + twin", {"tiers": ("P", "S"), "twin_seeds": (0,), "soft": False, "distill": False,
                                      "products": False, "tune_trials": 6, "phys_seeds": (0,)}),
]
KEEP_RATES = (0.5, 0.25, 0.15, 0.10, 0.05)


def threads() -> int:
    return int(os.environ.get("GENESIS_THREADS", str(CALIBRATION_THREADS)))


def estimate(store, n_features: int, scale, n_threads: int = None) -> Dict:
    """Rows, disk, memory and hours for ``scale`` on ``store`` (before design sampling)."""
    slow = CALIBRATION_THREADS / max(n_threads or threads(), 1)
    ever = np.zeros(store.n_patients, dtype=bool)
    for col in [c for c in store.onsets.columns if c.startswith("t_sepsis_") and c.endswith("_min")]:
        ever |= store.onsets[col].notna().to_numpy()
    lens = store.lens.astype(float)
    if scale.trim_septic:                                 # extract.trim_septic, estimated exactly
        prim = store.onsets["t_sepsis_measured_min"].notna().to_numpy()
        cols_t = [c for c in store.onsets.columns if c.startswith("t_sepsis_")]
        latest = np.nanmax(store.onsets[cols_t].to_numpy(dtype=float), axis=1)
        cut = prim & np.isfinite(latest)
        lens[cut] = np.minimum(lens[cut], np.ceil(latest[cut]) + 8)
    rows_ever = float(lens[ever].sum())
    rows_never = float(lens[~ever].sum())
    kept = rows_ever + scale.keep_rate * rows_never
    dev_rows = 0.8 * kept
    train_rows = 0.8 * dev_rows                          # four of the five dev folds
    cols = 0.8 * n_features                               # a tier sees most columns
    fit_s = slow * SEC_PER_ROW_COL * train_rows * cols
    folds = 5
    per_tier = 3 if scale.products else 1
    n_fits = folds * (per_tier * len(scale.tiers) + 1 + 1 + (1 if scale.soft else 0)
                      + (1 if getattr(scale, "distill", False) else 0))   # + strict + clinician
    n_fits += 3 * folds * min(1.0, (scale.learn_stays or store.n_patients) / max(store.n_patients, 1))
    n_fits += 1                                           # temporal validation
    if getattr(scale, "distill", False):
        n_fits += 10 * 0.6                                # pairwise teachers, each on 3 of 5 dev folds
    if scale.info_ledger:
        n_fits += 7 * folds
    # heads that bin the same fold as a tier's direct head reuse its dataset
    shared = folds * ((len(scale.tiers) if scale.products else 0)
                      + ((int(scale.soft) + int(getattr(scale, "distill", False))) if "P" in scale.tiers else 0))
    tune_s = slow * scale.tune_trials * len(scale.tune_val_folds) * SEC_PER_ROW_COL * \
        min(scale.tune_rows, train_rows) * cols
    booster_h = ((n_fits - CONSTRUCT_SHARE * shared) * fit_s + tune_s) / 3600.0
    twin_h = 0.0
    if scale.twin:
        stays_train = 0.64 * (float(ever.sum()) + scale.keep_rate * float((~ever).sum()))
        avg_len = kept / max(float(ever.sum()) + scale.keep_rate * float((~ever).sum()), 1.0)
        steps = max(scale.twin_epochs * stays_train / 32.0 * 0.6, scale.twin_min_steps)   # early stopping ~60%
        step_s = slow * TWIN_SEC_PER_ROW_STEP * 32 * avg_len * (scale.twin_d_model / 128.0)
        twin_h = folds * steps * step_s * (len(scale.twin_seeds) + 2 * len(scale.phys_seeds)) / 3600.0
    feat_gb = kept * n_features * 2 / 1e9
    # two phases, never together: extraction holds the whole grid plus a page of query
    # results; modelling holds the sampled grid plus LightGBM's binned copy OR the twin's inputs.
    # The planner runs on an extracted store, so only modelling is still ahead of it: ``choose``
    # judges that, and extraction is reported for a machine planning ahead.
    extract_gb = store.n_rows * len(store.columns) * 4 / 1e9 + 2.0
    grid_gb = kept * len(store.columns) * 4 / 1e9
    lgb_gb = (train_rows * cols * 1.2 + min(SAMPLE_ROWS, train_rows) * cols * 10) / 1e9
    twin_gb = kept * TWIN_INPUTS_PER_ROW * 4 / 1e9 if scale.twin else 0.0
    hot_gb = grid_gb + max(lgb_gb, twin_gb)                  # touched by every fit: must be RAM
    cold_gb = COLD_BASE_GB + kept * COLD_BYTES_PER_ROW / 1e9     # held, rarely touched: may swap
    model_gb = hot_gb + cold_gb
    ram_gb = max(extract_gb, model_gb)
    return {"stays": int(store.n_patients), "hours_all": int(store.n_rows), "rows_kept": int(kept),
            "features": int(n_features), "keep_rate": scale.keep_rate, "feature_disk_gb": feat_gb,
            "ram_peak_gb": ram_gb, "model_ram_gb": model_gb, "hot_ram_gb": hot_gb, "cold_ram_gb": cold_gb,
            "extract_ram_gb": extract_gb,
            "threads": n_threads or threads(), "booster_fits": int(round(n_fits)), "seconds_per_fit": fit_s,
            "booster_hours": booster_h, "twin_hours": twin_h, "hours": booster_h + twin_h + HARNESS_HOURS}


def fits_memory(est: Dict, budget: Budget) -> bool:
    """The hot set in RAM, and everything in RAM plus the swap allowed for cold memory."""
    return est["hot_ram_gb"] <= budget.ram_gb and est["model_ram_gb"] <= budget.ram_gb + budget.swap_gb


def choose(store, n_features: int, scale, budget: Budget) -> Tuple[object, Dict]:
    """The richest plan that fits ``budget``; raises (naming what does not fit) otherwise."""
    tried = []
    keeps = [scale.keep_rate] + [k for k in KEEP_RATES if k < scale.keep_rate]
    order = [(k, STAGE_SETS[0]) for k in keeps] if getattr(scale, "prefer_full_stages", False) else []
    order += [(k, st) for k in keeps for st in STAGE_SETS if (k, st) not in order]
    for keep, (label, over) in order:
        sc = replace(scale, keep_rate=keep, **over)
        est = estimate(store, n_features, sc)
        fits = {"disk": est["feature_disk_gb"] <= budget.disk_gb, "ram": fits_memory(est, budget),
                "time": est["hours"] <= budget.hours}
        tried.append({"plan": label, "keep_rate": keep, **{k: round(v, 2) if isinstance(v, float) else v
                                                             for k, v in est.items()}, "fits": fits})
        if all(fits.values()):
            return sc, {"chosen": label, "estimate": est, "budget": asdict(budget), "tried": tried}
    worst = tried[-1]
    raise RuntimeError(f"no plan fits this machine: leanest plan needs {worst['feature_disk_gb']:.1f} GB disk "
                       f"(have {budget.disk_gb:.1f}), {worst['hot_ram_gb']:.1f} GB hot + {worst['cold_ram_gb']:.1f} GB cold RAM "
                       f"(budget {budget.ram_gb:.1f} RAM + {budget.swap_gb:.1f} swap), "
                       f"{worst['hours']:.1f} h (budget {budget.hours:.1f})")
