"""Render ``artifacts/genesis/locked_results.json`` as the markdown table in docs/GENESIS.md.

    python -m backend.ml.genesis.report

Numbers in the write-up are generated from the artifact, never retyped."""
from __future__ import annotations

import json
import os

from backend.ml.genesis.harness import RESULTS_DIR

ROWS = [
    ("shipped_stack_served", "shipped wearable stack (the ~0.77 detector), refit on dev, served output"),
    ("shipped_stack", "shipped wearable stack, pre-isotonic meta score"),
    ("wearable_lgbm", "single LightGBM, 7 wearable vitals"),
    ("direct_lgbm", "single LightGBM, all variables, tuned"),
    ("network_alone", "Genesis Network alone (3 seeds)"),
    ("genesis_trees", "who × when × tuned direct (trees only)"),
    ("genesis", "**GENESIS** (trees × network)"),
]


def table(path: str = os.path.join(RESULTS_DIR, "locked_results.json")) -> str:
    d = json.load(open(path))
    R = d["results"]
    out = ["| detector | AUROC | hour-matched | between | within | AUPRC | util* | det@FA15 |",
           "|---|---|---|---|---|---|---|---|"]
    for key, label in ROWS:
        if key not in R:
            continue
        s = R[key]["score"]
        out.append(f"| {label} | {s['auroc']:.4f} | {s['auroc_hourmatched']:.4f} | "
                   f"{s['between']:.4f} | {s['within_septic']:.4f} | {s['auprc']:.4f} | "
                   f"{s['utility_best']:+.3f} | {s['detect_far15']:.3f} |")
    lines = ["", "Paired patient bootstrap on the locked patients (2,000 draws):", ""]
    for key, label in ROWS:
        for k, cmp in (R.get(key, {}) or {}).items():
            if k.startswith("vs_") and cmp:
                lines.append(f"* {label} {k.replace('_', ' ')}: ΔAUROC **{cmp['d_auroc']:+.4f}** "
                             f"[{cmp['auroc_ci'][0]:+.4f}, {cmp['auroc_ci'][1]:+.4f}]; "
                             f"Δbetween {cmp['d_between']:+.4f}, Δwithin {cmp['d_within']:+.4f}")
    p = d.get("p19", {})
    if p:
        lines += ["", f"P19-style patient task on the locked patients: AUROC {p['p19_auroc']:.4f} "
                      f"({p['n_patients']} patients, {p['n_septic']} septic); without the explicit "
                      f"record-length features {p['p19_auroc_without_length_features']:.4f}."]
    head = (f"Locked set: {d['n_locked_patients']:,} patients, {d['n_locked_rows']:,} hours, "
            f"fingerprint `{d['lock_fingerprint']}`, design `{d['spec']['version']}` "
            f"frozen {d['spec']['frozen']}.")
    return "\n".join([head, ""] + out + lines)


if __name__ == "__main__":  # pragma: no cover
    print(table())
