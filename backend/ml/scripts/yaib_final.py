"""Run all 25 runs of YAIB's protocol with the configuration a sweep chose.

Reads the sweep's ``selection.json`` (the choice, the registered plan and the rule are
all in it), builds the same ``bench run`` call the sweep made for that configuration
without ``--runs`` and ``--validation-only``, and replaces this process with it. The
checkpoint directory is the sweep's, so its development fits are resumed.

Nothing here can change the choice: the configuration is whatever ``selection.json``
names, and a selection that was made with anything but validation scores is refused.

An ENSEMBLE selection (``selection["ensemble"]``, written by Forge's YAIB lane) names its
members, each with its own registered plan, and the weights frozen on validation. Each
member's 25 runs are scored the same way (resuming its development fits), then
``yaib_ensemble.combine`` applies the frozen weights run by run, and the combined test
predictions are scored once and written where the paired comparisons read them.

    python -m backend.ml.scripts.yaib_final <cohort> --selection /sweep/selection.json \
        --work /lake/yaib-work/sweep --feature-store /lake/yaib-work/feature-store \
        --predictions /lake/yaib-work/ours --out /output/final.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from backend.ml.scripts import yaib_sweep as S


def build(selection: Dict[str, Any], cohort: str, work: str, store: str, predictions: str, out: str,
          threads: int = 2) -> List[str]:
    plan = selection["plan"]
    if "validation AUROC only" not in selection.get("note", ""):
        raise ValueError("this selection does not declare that it read validation scores only")
    recomputed = S.select(plan["reference"], {n: r["val"] for n, r in selection["stage2"].items()},
                          plan["stage2_runs"].split(","), float(plan["min_gain"]))
    if recomputed["chosen"] != selection["chosen"]:
        raise ValueError("the recorded choice is not what the registered rule gives on the recorded scores")
    ref = plan["reference"]
    if selection["chosen"] == ref["name"] and ref.get("feature_set", plan["feature_set"]) != plan["feature_set"]:
        # the reference is a model already scored on another feature set: it keeps its place
        # and its 25 runs stand; running its settings on this plan's features would be neither
        raise ValueError(f"the reference ({ref['name']}, {ref['feature_set']} features) keeps its place; "
                         "there is no new configuration to run")
    return S.command(cohort, plan, selection["chosen"], None, work, store, out, threads, predictions=predictions)


def build_ensemble(selection: Dict[str, Any], cohort: str, work: str, store: str, predictions: str, out: str,
                   threads: int = 2) -> List[Dict[str, Any]]:
    """One scored ``bench run`` per member, each from its own registered plan."""
    if "validation AUROC only" not in selection.get("note", ""):
        raise ValueError("this selection does not declare that it read validation scores only")
    ens = selection["ensemble"]
    weights, plans = ens["weights"], ens["plans"]
    if not weights or set(weights) - set(plans):
        raise ValueError("an ensemble member has no registered plan")
    if abs(sum(weights.values()) - 1) > 1e-9 or min(weights.values()) <= 0:
        raise ValueError("ensemble weights must be positive and sum to 1")
    runs = selection["plan"]["stage2_runs"].split(",")
    ref = selection["plan"]["reference"]["val_auroc"]
    lead = sum(ens["cross_fitted"][r] - ref[r] for r in runs) / len(runs)
    if not lead > float(selection["plan"]["min_gain"]):
        raise ValueError("the recorded ensemble does not clear the registered minimum gain on validation")
    jobs = []
    for name in sorted(weights):
        member_out = os.path.join(os.path.dirname(out) or ".", f"member-{name}.json")
        member_pred = os.path.join(predictions, "members", name)
        jobs.append({"name": name, "out": member_out, "predictions": member_pred,
                     "cmd": S.command(cohort, plans[name], name, None, work, store, member_out, threads,
                                      predictions=member_pred)})
    return jobs


def run_ensemble(selection: Dict[str, Any], jobs: List[Dict[str, Any]], predictions: str, out: str) -> Dict[str, Any]:
    import numpy as np
    import pandas as pd
    from sklearn.metrics import average_precision_score, roc_auc_score

    from backend.ml.scripts import yaib_ensemble as E
    from backend.ml.yaib.bench import yaib_summary
    for job in jobs:
        if not Path(job["out"]).exists():
            subprocess.run(job["cmd"], check=True)
    weights = selection["ensemble"]["weights"]
    files = E.combine({j["name"]: j["predictions"] for j in jobs}, weights, predictions)
    runs = []
    for path in files:
        frame = pd.read_parquet(path)
        y, p = frame.label.to_numpy(float), frame.p.to_numpy(float)
        rep, fold = (int(v) for v in re.match(r"ours_r(\d+)_f(\d+)$", Path(path).stem).groups())
        runs.append({"repetition": rep, "fold": fold,
                     "auroc": float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else math.nan,
                     "auprc": float(average_precision_score(y, p)) if y.sum() > 0 else math.nan})
    au = [r["auroc"] for r in runs]
    pr = [r["auprc"] for r in runs]
    res = {"schema": "yaib_ensemble_final/1", "weights": weights, "members": [j["name"] for j in jobs],
           "runs": runs, "yaib_metric": {**yaib_summary("auroc", np.array(au)),
                                         **yaib_summary("auprc", np.array(pr)), "n_runs": len(runs)}}
    tmp = Path(out).with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, sort_keys=True))
    os.replace(tmp, out)
    return res


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cohort")
    ap.add_argument("--selection", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--feature-store", required=True)
    ap.add_argument("--predictions", required=True, help="per-run test predictions (rows: inside the lake)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--print", action="store_true", help="print the command and stop")
    a = ap.parse_args(argv)
    selection = json.loads(Path(a.selection).read_text())
    if "ensemble" in selection:
        jobs = build_ensemble(selection, a.cohort, a.work, a.feature_store, a.predictions, a.out, a.threads)
        if a.print:
            print(json.dumps([j["cmd"] for j in jobs]))
            return 0
        os.umask(0o077)
        res = run_ensemble(selection, jobs, a.predictions, a.out)
        m = res["yaib_metric"]
        print(f"ensemble AUROC {m['auroc_mean']:.4f}  AUPRC {m['auprc_mean']:.4f} over {m['n_runs']} runs")
        return 0
    cmd = build(json.loads(Path(a.selection).read_text()), a.cohort, a.work, a.feature_store,
                a.predictions, a.out, a.threads)
    if a.print:
        print(json.dumps(cmd))
        return 0
    os.umask(0o077)
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    raise SystemExit(main())
