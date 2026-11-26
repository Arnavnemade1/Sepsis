"""Small tree ensembles, chosen on validation predictions only.

``select`` reads the validation predictions that ``bench run --save-validation`` wrote for
each member on the development runs, and chooses weights by greedy ensemble selection with
replacement (Caruana et al.) on logits: start from the best single member, add whichever
member (again, if it helps) raises validation AUROC most, keep the best prefix, never more
than ``max_members`` different members. The weights reported for each run were chosen on the
OTHER runs only (leave one run out), so a run's ensemble AUROC is cross-fitted, not
in-sample; the weights an ensemble is frozen with are chosen on all development runs pooled.

``combine`` applies frozen weights to the members' saved predictions run by run (the 25-run
test predictions ``yaib_final`` writes), checking that every member scored the same rows with
the same labels, and writes them where the paired comparisons read them.

``profile`` gives the evidence an agent may see: each member's validation AUROC by hours
since admission and by stay length, and how correlated the members' logits are.

Outputs are aggregates; the predictions read and written stay in the lake.

    python -m backend.ml.scripts.yaib_ensemble select  --member a=/v/a --member b=/v/b --runs 0:0,0:1 --out sel.json
    python -m backend.ml.scripts.yaib_ensemble combine --member a=/p/a --member b=/p/b --selection sel.json --out-dir /p
    python -m backend.ml.scripts.yaib_ensemble profile --member a=/v/a --runs 0:0,0:1 --out profile.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS = 1e-7
HOUR_BANDS = ((0, 6), (6, 24), (24, 72), (72, np.inf))
LENGTH_BANDS = ((0, 24), (24, 72), (72, 168), (168, np.inf))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, float), EPS, 1 - EPS)
    return np.log(p) - np.log1p(-p)


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def _run_file(directory: str, run: str, prefix: str) -> str:
    rep, fold = run.split(":")
    return os.path.join(directory, f"{prefix}_r{int(rep)}_f{int(fold)}.parquet")


def aligned(frames: Dict[str, pd.DataFrame]) -> Tuple[pd.DataFrame, np.ndarray]:
    """The members' rows (stay, hour, label) and their logits as columns. Members must have
    scored the same rows, in the same order, with the same labels: anything else is refused."""
    names = list(frames)
    first = frames[names[0]]
    for n in names[1:]:
        f = frames[n]
        if len(f) != len(first) or not (
            np.array_equal(f.stay_id.to_numpy(), first.stay_id.to_numpy())
            and np.array_equal(f.hour.to_numpy(), first.hour.to_numpy())
            and np.array_equal(f.label.to_numpy(), first.label.to_numpy())
        ):
            raise ValueError(f"{n} did not score the same rows as {names[0]}")
    L = np.column_stack([logit(frames[n].p.to_numpy()) for n in names])
    return first[["stay_id", "hour", "label"]].reset_index(drop=True), L


def caruana(L: np.ndarray, y: np.ndarray, max_members: int = 3, steps: int = 12) -> np.ndarray:
    """Weights (summing to 1) from greedy selection with replacement by AUROC of the mean
    logit; at most ``max_members`` different members; the best prefix is kept (ties: the
    shorter one)."""
    m = L.shape[1]
    single = [roc_auc_score(y, L[:, j]) for j in range(m)]
    counts = np.zeros(m, int)
    counts[int(np.argmax(single))] = 1
    total = L[:, int(np.argmax(single))].copy()
    best, best_counts = max(single), counts.copy()
    for _ in range(steps - 1):
        k = counts.sum() + 1
        allowed = [j for j in range(m) if counts[j] > 0 or (counts > 0).sum() < max_members]
        scores = [(roc_auc_score(y, (total + L[:, j]) / k), j) for j in allowed]
        score, j = max(scores)
        counts[j] += 1
        total += L[:, j]
        if score > best + 1e-12:
            best, best_counts = score, counts.copy()
    return best_counts / best_counts.sum()


def _members(spec: Sequence[str]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in spec:
        name, _, directory = item.partition("=")
        if not name or not directory or name in out:
            raise ValueError(f"--member must be name=DIR, each name once: {item!r}")
        out[name] = directory
    if len(out) < 1:
        raise ValueError("at least one member")
    return out


def select(members: Dict[str, str], runs: List[str], max_members: int = 3, steps: int = 12) -> dict:
    names = list(members)
    data = {}
    for r in runs:
        rows, L = aligned({n: pd.read_parquet(_run_file(d, r, "val")) for n, d in members.items()})
        data[r] = (rows.label.to_numpy(float), L)
    singles = {r: {n: float(roc_auc_score(y, L[:, j])) for j, n in enumerate(names)} for r, (y, L) in data.items()}
    weights_by_run, cross = {}, {}
    for r in runs:
        others = [o for o in runs if o != r]
        if not others:
            raise ValueError("cross-fitting needs at least two development runs")
        y = np.concatenate([data[o][0] for o in others])
        L = np.vstack([data[o][1] for o in others])
        w = caruana(L, y, max_members, steps)
        weights_by_run[r] = {n: float(x) for n, x in zip(names, w) if x > 0}
        cross[r] = float(roc_auc_score(data[r][0], data[r][1] @ w))
    y_all = np.concatenate([data[r][0] for r in runs])
    L_all = np.vstack([data[r][1] for r in runs])
    w = caruana(L_all, y_all, max_members, steps)
    corr = {}
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            corr[f"{names[a]}|{names[b]}"] = round(float(np.mean(
                [np.corrcoef(data[r][1][:, a], data[r][1][:, b])[0, 1] for r in runs])), 4)
    return {"schema": "yaib_ensemble_selection/1", "members": names, "runs": runs,
            "max_members": max_members, "steps": steps,
            "weights": {n: float(x) for n, x in zip(names, w) if x > 0},
            "weights_by_run": weights_by_run, "cross_fitted": cross, "singles": singles,
            "correlation": corr,
            "note": "weights chosen on validation predictions only; each run's AUROC used weights chosen on the other runs"}


def combine(members: Dict[str, str], weights: Dict[str, float], out_dir: str, prefix: str = "ours") -> List[str]:
    """Frozen weights applied to every run the members all scored; returns the files written."""
    if not weights or set(weights) - set(members):
        raise ValueError("the weights name members that were not given")
    if abs(sum(weights.values()) - 1) > 1e-9 or min(weights.values()) < 0:
        raise ValueError("weights must be non-negative and sum to 1")
    names = list(weights)
    pattern = re.compile(rf"^{prefix}_r(\d+)_f(\d+)\.parquet$")
    found = {n: {f"{m.group(1)}:{m.group(2)}" for f in glob.glob(os.path.join(members[n], f"{prefix}_r*_f*.parquet"))
                 if (m := pattern.match(os.path.basename(f)))} for n in names}
    runs = found[names[0]]
    for n in names[1:]:
        if found[n] != runs:
            raise ValueError(f"{n} scored different runs from {names[0]}")
    if not runs:
        raise ValueError("no run that every member scored")
    os.makedirs(out_dir, mode=0o700, exist_ok=True)
    written = []
    w = np.array([weights[n] for n in names])
    for r in sorted(runs, key=lambda x: tuple(int(v) for v in x.split(":"))):
        rows, L = aligned({n: pd.read_parquet(_run_file(members[n], r, prefix)) for n in names})
        path = _run_file(out_dir, r, prefix)
        tmp = path + ".tmp"
        rows.assign(p=sigmoid(L @ w)).to_parquet(tmp, index=False)
        os.replace(tmp, path)
        written.append(path)
    return written


def _band_auroc(y: np.ndarray, s: np.ndarray, key: np.ndarray, bands) -> Dict[str, float]:
    out = {}
    for lo, hi in bands:
        m = (key >= lo) & (key < hi)
        if m.sum() and 0 < y[m].sum() < m.sum():
            out[f"{lo}-{hi}"] = round(float(roc_auc_score(y[m], s[m])), 4)
    return out


def profile(members: Dict[str, str], runs: List[str]) -> dict:
    """Validation AUROC by hours since admission and by stay length, per member and run,
    and the members' mean logit correlation: aggregates for the agents."""
    out = {"schema": "yaib_ensemble_profile/1", "runs": runs, "by_hour": {}, "by_stay_length": {}}
    corr: Dict[str, List[float]] = {}
    names = list(members)
    for r in runs:
        rows, L = aligned({n: pd.read_parquet(_run_file(d, r, "val")) for n, d in members.items()})
        y = rows.label.to_numpy(float)
        hour = rows.hour.to_numpy(float)
        length = rows.groupby("stay_id").hour.transform("max").to_numpy(float) + 1
        for j, n in enumerate(names):
            out["by_hour"].setdefault(n, {})[r] = _band_auroc(y, L[:, j], hour, HOUR_BANDS)
            out["by_stay_length"].setdefault(n, {})[r] = _band_auroc(y, L[:, j], length, LENGTH_BANDS)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                corr.setdefault(f"{names[a]}|{names[b]}", []).append(float(np.corrcoef(L[:, a], L[:, b])[0, 1]))
    out["correlation"] = {k: round(float(np.mean(v)), 4) for k, v in corr.items()}
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("--member", action="append", required=True, help="name=DIR of val_r*_f*.parquet")
    s.add_argument("--runs", default="0:0,0:1")
    s.add_argument("--max-members", type=int, default=3)
    s.add_argument("--steps", type=int, default=12)
    s.add_argument("--out", required=True)
    c = sub.add_parser("combine")
    c.add_argument("--member", action="append", required=True, help="name=DIR of ours_r*_f*.parquet")
    c.add_argument("--selection", required=True)
    c.add_argument("--out-dir", required=True)
    p = sub.add_parser("profile")
    p.add_argument("--member", action="append", required=True)
    p.add_argument("--runs", default="0:0,0:1")
    p.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    os.umask(0o077)
    members = _members(a.member)
    if a.cmd == "select":
        res = select(members, a.runs.split(","), a.max_members, a.steps)
    elif a.cmd == "combine":
        sel = json.loads(Path(a.selection).read_text())
        res = {"written": len(combine(members, sel["weights"], a.out_dir)), "weights": sel["weights"]}
    else:
        res = profile(members, a.runs.split(","))
    tmp = Path(a.out if a.cmd != "combine" else os.path.join(a.out_dir, "combine.json")).with_suffix(".tmp")
    tmp.write_text(json.dumps(res, indent=1, sort_keys=True))
    os.replace(tmp, tmp.with_suffix(".json"))
    print(json.dumps({k: res[k] for k in res if k in ("weights", "cross_fitted", "written")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
