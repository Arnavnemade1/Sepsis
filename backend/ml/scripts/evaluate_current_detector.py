"""Measure the current hand-tuned detector on real CinC 2019 data.

    python -m backend.ml.scripts.evaluate_current_detector --limit 2400

This prints the honest baseline the rest of the project must beat.
"""
from __future__ import annotations

import argparse

from backend.ml.data.cinc2019 import load_patients, dataset_summary
from backend.ml.models.heuristic_adapter import score_cohort
from backend.ml.eval import metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="max patients to score")
    args = ap.parse_args()

    print("Loading patients from local CinC 2019 cache...")
    records = load_patients(limit=args.limit)
    if not records:
        raise SystemExit("No data found. Run: python -m backend.ml.data.cinc2019 --download")
    print("Dataset:", dataset_summary(records))

    print("\nScoring with the EXISTING heuristic detector (HDC + twin + neurosymbolic)...")
    p, y, groups, hour_index, onset_hour = score_cohort(records)

    print("\n================ BASELINE: current heuristic detector ================")
    report = metrics.evaluate(y, p, groups=groups, hour_index=hour_index, onset_hour=onset_hour)
    print(report.pretty())
    print("=====================================================================")
    print("This is the number to beat with the data-grounded HDC model.")


if __name__ == "__main__":
    main()
