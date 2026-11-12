"""Choose a learner configuration on a YAIB cohort by VALIDATION AUROC only.

The plan (``artifacts/yaib/*_sweep_prereg.json``) names every candidate and the rule
before anything runs. Each candidate is fitted in a fresh process (LightGBM and Arrow
keep allocator state between fits in one process), with ``--validation-only``, so no
test metric of a candidate is ever written, and through one shared feature store, so
the cohort's matrix is built once for the whole sweep.

  stage 1   every candidate on the plan's first development run
  stage 2   the best ``top`` of them on the second run as well
  choice    the highest mean validation AUROC among configurations with every run;
            a tie, or a lead inside ``min_gain``, keeps the reference

The chosen configuration's checkpoint directory already holds its development fits, so
the full 25 runs resume from them instead of repeating them.

    python -m backend.ml.scripts.yaib_sweep <cohort> --plan <plan.json> \\
        --work /lake/yaib-work/sweep --feature-store /lake/yaib-work/feature-store --out /output

Outputs are aggregates: one validation-only result per candidate and stage,
``state.json`` while it runs, ``selection.json`` at the end. Never a row.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

FLAGS = {"learning_rate": "--learning-rate", "num_leaves": "--num-leaves",
         "min_child_samples": "--min-child-samples", "feature_fraction": "--feature-fraction",
         "bagging_fraction": "--bagging-fraction", "lambda_l2": "--lambda-l2",
         "max_rounds": "--max-rounds", "early_stopping": "--early-stopping"}


def command(cohort: str, plan: Dict[str, Any], name: str, runs: str, work: str, store: str,
            out: str, threads: int) -> List[str]:
    """The exact ``bench run`` call of one candidate. The final 25-run call is this one
    without ``--runs`` and ``--validation-only``."""
    cfg = dict(plan["configs"][name])
    unknown = set(cfg) - set(FLAGS) - {"unbalanced"}
    if unknown:
        raise ValueError(f"{name}: unknown settings {sorted(unknown)}")
    cmd = [sys.executable, "-m", "backend.ml.yaib.bench", "run", cohort,
           "--feature-set", plan["feature_set"], "--seed", str(plan["seed"]), "--threads", str(threads),
           "--runs", runs, "--validation-only", "--resume-dir", os.path.join(work, name),
           "--feature-store", store, "--out", out]
    for key in sorted(cfg):
        if key == "unbalanced":
            if cfg[key]:
                cmd.append("--unbalanced")
        else:
            cmd += [FLAGS[key], str(cfg[key])]
    return cmd


def validation(path: Path) -> Optional[Dict[str, Any]]:
    """{run: validation AUROC}, rounds and seconds of a finished candidate, or None.
    A result that carries a test metric is refused: selection must not be able to see one."""
    if not path.exists():
        return None
    res = json.loads(path.read_text())
    if not res.get("validation_only") or "yaib_metric" in res or any("auroc" in r for r in res["runs"]):
        raise ValueError(f"{path.name} carries test metrics; a sweep reads validation only")
    return {"val": {f"{r['repetition']}:{r['fold']}": r["val_auroc"] for r in res["runs"]},
            "rounds": {f"{r['repetition']}:{r['fold']}": r["rounds"] for r in res["runs"]},
            "seconds": res["seconds"]}


def select(reference: Dict[str, Any], candidates: Dict[str, Dict[str, float]], runs: List[str],
           min_gain: float) -> Dict[str, Any]:
    """The registered rule. ``candidates`` maps a name to {run: validation AUROC}; only
    configurations with every run in ``runs`` compete. The reference keeps its place
    unless a candidate's mean leads it by more than ``min_gain``."""
    def mean(v):
        return sum(v[r] for r in runs) / len(runs)
    table = [{"name": reference["name"], "mean": mean(reference["val_auroc"]), "reference": True}]
    for name, val in candidates.items():
        if all(r in val for r in runs):
            table.append({"name": name, "mean": mean(val), "reference": False})
    best = max(table[1:], key=lambda t: t["mean"], default=None)
    chosen = table[0]
    if best is not None and best["mean"] > table[0]["mean"] + min_gain:
        chosen = best
    return {"chosen": chosen["name"], "chosen_mean_val_auroc": chosen["mean"],
            "lead_over_reference": chosen["mean"] - table[0]["mean"],
            "compared": sorted(table, key=lambda t: -t["mean"]), "runs": runs, "min_gain": min_gain}


def write(path: Path, value: Dict[str, Any]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, indent=1, sort_keys=True))
    os.replace(tmp, path)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cohort")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--work", required=True, help="private checkpoint root (inside the lake)")
    ap.add_argument("--feature-store", required=True, help="private shared feature store (inside the lake)")
    ap.add_argument("--out", required=True, help="directory for aggregate results")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--fit-timeout-hours", type=float, default=6.0)
    ap.add_argument("--max-hours", type=float, default=30.0, help="start no new candidate after this long")
    a = ap.parse_args(argv)

    os.umask(0o077)
    plan = json.loads(Path(a.plan).read_text())
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    Path(a.work).mkdir(parents=True, exist_ok=True, mode=0o700)
    first, both = plan["stage1_runs"], plan["stage2_runs"]
    runs = both.split(",")
    t0 = time.time()
    state: Dict[str, Any] = {"status": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                             "done": [], "failed": {}}

    def fit(name: str, stage: str, which: str) -> Optional[Dict[str, Any]]:
        path = out / f"{stage}-{name}.json"
        done = validation(path)
        if done is not None:
            return done
        if time.time() - t0 > a.max_hours * 3600:
            state["failed"][f"{stage}-{name}"] = "not started: sweep time budget spent"
            return None
        state.update(current=f"{stage}-{name}")
        write(out / "state.json", state)
        try:
            # a traceback may quote values, so the child's output stays with the checkpoints
            with open(os.path.join(a.work, f"{stage}-{name}.log"), "w") as log:
                subprocess.run(command(a.cohort, plan, name, which, a.work, a.feature_store, str(path), a.threads),
                               check=True, timeout=a.fit_timeout_hours * 3600, stdout=log, stderr=log)
        except subprocess.TimeoutExpired:
            state["failed"][f"{stage}-{name}"] = "timeout"
        except subprocess.CalledProcessError as error:
            state["failed"][f"{stage}-{name}"] = f"exit {error.returncode}"
        done = validation(path)
        if done is not None:
            state["done"].append(f"{stage}-{name}")
        write(out / "state.json", state)
        return done

    stage1 = {name: fit(name, "stage1", first) for name in plan["configs"]}
    ranked = sorted((n for n, r in stage1.items() if r), key=lambda n: -stage1[n]["val"][first])
    stage2 = {name: fit(name, "stage2", both) for name in ranked[:int(plan["stage2_top"])]}
    candidates = {n: r["val"] for n, r in stage2.items() if r}
    result = select(plan["reference"], candidates, runs, float(plan["min_gain"]))
    result.update(
        plan=plan, stage1={n: r for n, r in stage1.items() if r}, stage2={n: r for n, r in stage2.items() if r},
        failed=state["failed"], seconds=round(time.time() - t0, 1),
        chosen_settings=plan["configs"].get(result["chosen"], plan["reference"].get("settings", {})),
        note="validation AUROC only; no test metric of any candidate was computed into these files")
    write(out / "selection.json", result)
    state.update(status="finished", current=None, chosen=result["chosen"])
    write(out / "state.json", state)
    print(json.dumps({"chosen": result["chosen"], "mean_val_auroc": result["chosen_mean_val_auroc"],
                      "lead_over_reference": result["lead_over_reference"], "failed": state["failed"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
