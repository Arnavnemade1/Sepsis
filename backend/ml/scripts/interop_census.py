"""EXPLORATORY (post hoc): onboarding evidence on a small census, pooled vs counted by patient.

The registered benchmark (``interop_bench.py``) streams 8,062 patients interleaved by
hour, so a code's first 12 values come from about 12 different patients and pooling the
values is harmless. A real interface often starts with a handful of patients. This
measures what the onboarding decision does then, with the location and tail tests either
pooling the values (``pooled``, the gate as registered) or counting patients
(``clustered``, the post-hoc fix):

* a census of K holdout patients (K = 1, 2, 4, 8, 12; and 48 patients with one value each,
  the registered stream's regime, where the two rules must agree);
* each patient's values for the column in time order, the census interleaved by hour, so a
  code collects values the way an interface engine sees them;
* the engine's decision rule (``engine._propose``) at 12 values and every 4 after, up to 48:
  adopt (``declared``), refuse (the written unit fails), hold for a person (anything else
  that is not ``insufficient``), or keep learning (``insufficient``: pooled until 36 values,
  clustered until 36 values AND 12 patients);
* four scenarios per concept: the correct mapping on a random census; the correct mapping
  on a clinically skewed census (half of it from the patients whose own median is in the
  lowest decile, the way two febrile-neutropenic patients skew a white count); and every
  value x10 or /88.42 under the canonical label (unit faults).

The holdout is the registered one (fingerprint checked), but the census design and the
rule are post hoc: read every number as exploratory. Writes
``artifacts/interop/interop_census.json``.

    python -m backend.ml.scripts.interop_census            # ~5 min, one job at a time
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from backend.ml import config
from backend.aegis.interop import evidence as E
from backend.aegis.interop import vocab as V
from backend.ml.scripts.interop_bench import PREREG, load

OUT = os.path.join(config.ARTIFACTS_DIR, "interop", "interop_census.json")
CHECKPOINTS = list(range(12, 49, 4))
KS = [1, 2, 4, 8, 12, 48]
SCENARIOS = {"correct": None, "correct_skewed": None, "x10": 10.0, "div88": 1 / 88.42}
RULES = ("pooled", "clustered")


def decide(c: V.Concept, vals: List[float], rooms: List[int], declared, rule: str) -> Tuple[str, int, int]:
    """The engine's onboarding outcome for one census: (outcome, values seen, patients seen)."""
    n = 0
    for n in CHECKPOINTS:
        if n > len(vals):
            n = len(vals)
            break
        inf = E.infer(c, vals[:n], declared, clusters=rooms[:n] if rule == "clustered" else None)
        k = len(set(rooms[:n]))
        if inf.decision == "declared":
            return "adopted", n, k
        if inf.decision == "insufficient":
            if n < 36 or (rule == "clustered" and k < E.N_MIN):
                continue
            return "held", n, k
        if inf.declared is not None and inf.declared.status == "fail":
            return "refused", n, k
        return "held", n, k
    return "learning", n, len(set(rooms[:n]))


def census_stream(per_patient: Dict[int, List[Tuple[int, float]]], pts: List[int], one_each: bool) -> Tuple[List[float], List[int]]:
    rows = []
    for p in pts:
        seq = per_patient[p][:1] if one_each else per_patient[p]
        rows += [(h, p, v) for h, v in seq]
    rows.sort(key=lambda r: (r[0], r[1]))
    rows = rows[:CHECKPOINTS[-1]]
    return [r[2] for r in rows], [r[1] for r in rows]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=40)
    ap.add_argument("--concepts", default="", help="comma-separated concept keys (default: every CinC concept)")
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    t0 = time.time()
    with open(PREREG) as fh:
        prereg = json.load(fh)
    sub, fp = load(prereg)
    columns = list(sub.columns)
    seg, hour = sub.seg, sub.hour
    keys = [k for k in args.concepts.split(",") if k]
    concepts = [c for c in V.CONCEPTS if c.cinc in columns and c.key in V.references() and c.scale != "none"
                and (not keys or c.key in keys)]
    print(f"holdout {sub.n_patients:,} patients (fingerprint {fp}); {len(concepts)} concepts; {args.reps} reps per cell")
    res: Dict[str, Any] = {"note": "EXPLORATORY post-hoc census study; see the module docstring", "fingerprint": fp,
                           "reps": args.reps, "checkpoints": CHECKPOINTS, "ks": KS, "concepts": {}}
    for c in concepts:
        tc = time.time()
        v = sub.X[:, columns.index(c.cinc)]
        obs = np.flatnonzero(~np.isnan(v))
        per: Dict[int, List[Tuple[int, float]]] = {}
        for r in obs[np.lexsort((hour[obs], seg[obs]))]:
            per.setdefault(int(seg[r]), []).append((int(hour[r]), float(v[r])))
        pts = np.array(sorted(per))
        if len(pts) < 200:
            continue
        meds = np.array([np.median([x for _h, x in per[p]]) for p in pts])
        low = pts[meds <= np.quantile(meds, 0.10)]
        _tok, declared = V.resolve_unit(c, c.unit)
        rng = np.random.default_rng(sum(map(ord, c.key)))            # hash() is salted per process
        cell: Dict[str, Any] = {}
        for K in KS:
            one_each = K == 48
            for scen, factor in SCENARIOS.items():
                outs = {r: [] for r in RULES}
                done = 0
                tries = 0
                while done < args.reps and tries < args.reps * 40:
                    tries += 1
                    if scen == "correct_skewed" and not one_each:
                        nl = max(1, math.ceil(K / 2))
                        chosen = list(rng.choice(low, nl, replace=False)) + list(rng.choice(pts, K - nl, replace=False))
                    else:
                        chosen = list(rng.choice(pts, K, replace=False))
                    if len(set(chosen)) < K:
                        continue
                    vals, rooms = census_stream(per, [int(p) for p in chosen], one_each)
                    if len(vals) < CHECKPOINTS[0]:
                        continue
                    if factor is not None:
                        vals = [x * factor for x in vals]
                    for rule in RULES:
                        outs[rule].append(decide(c, vals, rooms, declared, rule))
                    done += 1
                summary = {}
                for rule in RULES:
                    o = outs[rule]
                    cnt = Counter(x[0] for x in o)
                    nn = [x[1] for x in o if x[0] in ("refused", "held", "adopted")]
                    refused_n = [x[1] for x in o if x[0] == "refused"]
                    refused_k = [x[2] for x in o if x[0] == "refused"]
                    summary[rule] = {"runs": len(o), **{k: cnt.get(k, 0) for k in ("adopted", "refused", "held", "learning")},
                                     "median_values_to_decision": float(np.median(nn)) if nn else None,
                                     "median_values_to_refusal": float(np.median(refused_n)) if refused_n else None,
                                     "median_patients_to_refusal": float(np.median(refused_k)) if refused_k else None,
                                     "refused_at_first_judgement": sum(1 for x in o if x[0] == "refused" and x[1] == CHECKPOINTS[0]),
                                     # caught = out of learning and not adopted: refused, or held for a person
                                     "caught_at_first_judgement": sum(1 for x in o if x[0] in ("refused", "held")
                                                                      and x[1] == CHECKPOINTS[0]),
                                     "median_values_to_caught": (float(np.median([x[1] for x in o if x[0] in ("refused", "held")]))
                                                                 if any(x[0] in ("refused", "held") for x in o) else None)}
                cell[f"K{K}/{scen}"] = summary
        res["concepts"][c.key] = cell
        print(f"  {c.key:16s} {len(pts):5d} pts  [{time.time() - tc:.0f}s]", flush=True)

    # across concepts: per K and scenario, totals
    agg: Dict[str, Any] = {}
    for K in KS:
        for scen in SCENARIOS:
            key = f"K{K}/{scen}"
            agg[key] = {}
            for rule in RULES:
                rows = [res["concepts"][ck][key][rule] for ck in res["concepts"] if key in res["concepts"][ck]]
                tot = {k: sum(r[k] for r in rows) for k in ("runs", "adopted", "refused", "held", "learning",
                                                           "refused_at_first_judgement", "caught_at_first_judgement")}
                vtr = [r["median_values_to_refusal"] for r in rows if r["median_values_to_refusal"] is not None]
                tot["median_of_concept_median_values_to_refusal"] = float(np.median(vtr)) if vtr else None
                agg[key][rule] = tot
    res["aggregate"] = agg
    # where the rules disagree on a fault: which concepts lose speed
    slower = {}
    for ck, cell in res["concepts"].items():
        for K in KS:
            for scen in ("x10", "div88"):
                p, q = cell[f"K{K}/{scen}"]["pooled"], cell[f"K{K}/{scen}"]["clustered"]
                if (q["caught_at_first_judgement"] < p["caught_at_first_judgement"]
                        or q["refused"] + q["held"] < p["refused"] + p["held"] or q["adopted"] > p["adopted"]):
                    slower.setdefault(f"K{K}/{scen}", []).append(
                        {"concept": ck, **{r: {k: x[k] for k in ("refused", "held", "caught_at_first_judgement", "adopted",
                                                                  "learning", "median_values_to_caught")}
                                           for r, x in (("pooled", p), ("clustered", q))}})
    res["clustered_slower_on_faults"] = slower
    res["seconds"] = round(time.time() - t0)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(f"{'cell':22s} {'rule':10s} {'runs':>5s} {'adopt':>6s} {'refuse':>6s} {'held':>5s} {'learn':>5s} {'ref@12':>6s} {'caught@12':>9s}")
    for key, d in agg.items():
        for rule, t in d.items():
            print(f"{key:22s} {rule:10s} {t['runs']:5d} {t['adopted']:6d} {t['refused']:6d} {t['held']:5d} "
                  f"{t['learning']:5d} {t['refused_at_first_judgement']:6d} {t['caught_at_first_judgement']:9d}")
    print(f"wrote {args.out} [{res['seconds']}s]")


if __name__ == "__main__":
    main()
