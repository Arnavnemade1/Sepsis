"""Run all 25 runs of YAIB's protocol with the configuration a sweep chose.

Reads the sweep's ``selection.json`` (the choice, the registered plan and the rule are
all in it), builds the same ``bench run`` call the sweep made for that configuration
without ``--runs`` and ``--validation-only``, and replaces this process with it. The
checkpoint directory is the sweep's, so its development fits are resumed.

Nothing here can change the choice: the configuration is whatever ``selection.json``
names, and a selection that was made with anything but validation scores is refused.

    python -m backend.ml.scripts.yaib_final <cohort> --selection /sweep/selection.json \
        --work /lake/yaib-work/sweep --feature-store /lake/yaib-work/feature-store \
        --predictions /lake/yaib-work/ours --out /output/final.json
"""
from __future__ import annotations

import argparse
import json
import os
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
    return S.command(cohort, plan, selection["chosen"], None, work, store, out, threads, predictions=predictions)


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
    cmd = build(json.loads(Path(a.selection).read_text()), a.cohort, a.work, a.feature_store,
                a.predictions, a.out, a.threads)
    if a.print:
        print(json.dumps(cmd))
        return 0
    os.umask(0o077)
    os.execv(cmd[0], cmd)


if __name__ == "__main__":
    raise SystemExit(main())
