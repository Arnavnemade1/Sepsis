"""The paired YAIB bootstrap of ``bench compare``, fast, with progress, reading gzipped runs.

``backend.ml.yaib.bench.compare`` calls sklearn's weighted ``roc_auc_score`` twice per run
per draw: 25 runs x 1,000 draws x 2 on ~1.28M rows each took 2 h 41 min on the eICU worker
(2026-11-21), single-threaded and silent. This computes the same statistic:

* the same pairing (``bench._join``: sorted position, refused unless ids and labels agree),
* the same random stream (one ``default_rng(seed)``, repetitions in order, the same
  ``integers`` call per draw), so the draws are the same multiset weights,
* the same weighted AUROC: the area under the weighted ROC curve, ties counted half
  (what sklearn's trapezoid over distinct thresholds gives),

but sorts each run's scores once and then needs only a gather and two grouped sums per
draw. ``verify_yaib_paired_fast`` checks it against ``bench.compare`` to 1e-12 and shows
the check fails for a tie-blind variant.

It lives in ``scripts/`` because editing ``backend/ml/yaib/*.py`` changes every fit identity.

python -m backend.ml.scripts.yaib_paired_fast --ours <dir> --yaib <log dir> --out paired.json
"""
from __future__ import annotations

import glob
import os
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from backend.ml.yaib.bench import _join, read_yaib_predictions

PRED_NAMES = ("pred_indicators.csv", "pred_indicators.csv.gz")


class _Sorted:
    """One run's labels and one model's scores, sorted once by score (descending)."""

    def __init__(self, y: np.ndarray, p: np.ndarray, codes: np.ndarray):
        order = np.argsort(-p, kind="mergesort")
        ps = p[order]
        self.y = y[order].astype(float)
        self.codes = codes[order]
        self.starts = np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]])  # first row of each tie group

    def auc(self, m: np.ndarray) -> float:
        w = m[self.codes]
        gp = np.add.reduceat(w * self.y, self.starts)
        gn = np.add.reduceat(w, self.starts) - gp
        P, N = gp.sum(), gn.sum()
        before = np.cumsum(gp) - gp  # positive weight scored strictly higher than this group
        return float(np.dot(gn, before + 0.5 * gp) / (P * N))


def compare(ours: Dict[tuple, pd.DataFrame], theirs: Dict[tuple, pd.DataFrame], n_boot: int = 1000,
            seed: int = 0, progress=None) -> Dict[str, Any]:
    """Same result as ``bench.compare`` (ours - baseline, stays bootstrapped per repetition)."""
    keys = sorted(set(ours) & set(theirs))
    if not keys or set(ours) != set(theirs):
        raise ValueError(f"runs differ: ours {sorted(ours)}, baseline {sorted(theirs)}")
    by_rep: Dict[int, List[pd.DataFrame]] = {}
    for k in keys:
        by_rep.setdefault(k[0], []).append(_join(ours[k], theirs[k]))
    rng = np.random.default_rng(seed)
    point, draws = [], np.zeros(n_boot)
    t0 = time.time()
    for r, frames in sorted(by_rep.items()):
        stays = np.unique(frames[0]["stay_id"].to_numpy())
        runs = []
        for f in frames:
            y = f["label_ours"].to_numpy()
            c = np.searchsorted(stays, f["stay_id"].to_numpy())
            runs.append((_Sorted(y, f["p_ours"].to_numpy(), c), _Sorted(y, f["p_base"].to_numpy(), c),
                         c, y > 0))
        point.append(np.mean([roc_auc_score(f["label_ours"], f["p_ours"]) - roc_auc_score(f["label_ours"], f["p_base"])
                              for f in frames]))
        for b in range(n_boot):
            m = np.bincount(rng.integers(0, len(stays), len(stays)), minlength=len(stays)).astype(float)
            d = []
            for a, base, c, pos in runs:
                w = m[c]
                # bench skips a run whose drawn stays hold no rows or a single class
                if w.sum() == 0 or not (w[pos].sum() > 0 and w[~pos].sum() > 0):
                    continue
                d.append(a.auc(m) - base.auc(m))
            draws[b] += np.mean(d) if d else np.nan
        if progress:
            progress(f"repetition {r}: {len(frames)} runs x {n_boot} draws, {time.time() - t0:.0f}s elapsed")
    draws /= len(by_rep)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    delta = float(np.mean(point))
    return {"auroc_delta": delta, "ci_low": float(lo), "ci_high": float(hi), "repetitions": len(by_rep),
            "runs": len(keys), "better": bool(lo > 0), "worse": bool(hi < 0)}


def load_runs(ours_dir: str, yaib_root: str) -> tuple:
    """``bench.load_runs``, also reading ``pred_indicators.csv.gz``; a run saved both ways is refused."""
    ours, theirs = {}, {}
    for f in glob.glob(os.path.join(ours_dir, "ours_r*_f*.parquet")):
        r, k = os.path.basename(f)[len("ours_"):-len(".parquet")].split("_")
        ours[(int(r[1:]), int(k[1:]))] = pd.read_parquet(f)
    for name in PRED_NAMES:
        for f in glob.glob(os.path.join(yaib_root, "**", name), recursive=True):
            parts = os.path.normpath(f).split(os.sep)
            rep = next((p for p in parts if p.startswith("repetition_")), None)
            fold = next((p for p in parts if p.startswith("fold_")), None)
            if rep and fold:
                key = (int(rep.split("_")[1]), int(fold.split("_")[1]))
                if key in theirs:
                    raise ValueError(f"run {key} saved twice (.csv and .csv.gz) under {yaib_root}")
                theirs[key] = read_yaib_predictions(f)  # np.loadtxt opens .gz itself
    return ours, theirs


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    from backend.ml.jsonio import json_safe

    ap = argparse.ArgumentParser(description="Paired AUROC vs YAIB's saved predictions (fast, aggregates only).")
    ap.add_argument("--ours", required=True, help="directory of ours_r*_f*.parquet")
    ap.add_argument("--yaib", required=True, help="YAIB log directory of one baseline model")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=1000)
    a = ap.parse_args(argv)
    say = lambda s: print(s, file=sys.stderr, flush=True)  # noqa: E731
    ours, theirs = load_runs(a.ours, a.yaib)
    say(f"loaded {len(ours)} of our runs, {len(theirs)} baseline runs")
    res = compare(ours, theirs, a.n_boot, progress=say)
    print(f"AUROC ours - baseline {res['auroc_delta']:+.4f} [{res['ci_low']:+.4f}, {res['ci_high']:+.4f}]")
    with open(a.out, "w") as fh:
        json.dump(json_safe(res), fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
