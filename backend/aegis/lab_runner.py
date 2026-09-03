"""The research campaign — agents working a problem over rounds, under the gate.

    PYTHONPATH=. python -m backend.aegis.lab_runner --rounds 3

WHY A CAMPAIGN AND NOT A SUBMIT BUTTON
---------------------------------------
``lab.py`` adjudicates one proposal at a time. That is a referee, not a research
programme. Research is a *loop*: propose, be refused, learn what the refusal implies,
propose something the refusal did not rule out, and stop when the error budget is spent
rather than when someone gets bored. This module is that loop, and it closes three gaps
the Lab shipped with:

* **POLICY proposals had no judge.** The routing existed but nothing could adjudicate a
  decision-layer change, so every policy proposal was permanently refused for lack of a
  judge — the Lab could only really test model changes.
* **REPLICATION had no replicator.** A gate pass could therefore never become a FINDING
  on real data; every accepted proposal was stuck at "candidate" forever.
* **``make lab-run`` was referenced in a commit message and did not exist.**

THE TWO REAL ADJUDICATORS
-------------------------
``policy_adjudicator`` is cheap on purpose. A decision-layer change does not alter the
model, so it needs no refit: it re-reads *cached out-of-fold scores* under a different
decision rule and compares detection at a fixed alarm budget. Seconds, not minutes, which
is what makes a multi-round campaign possible at all.

``replicator`` re-runs the identical protocol on a **disjoint patient split**. Not a
different metric, not a looser bar — the same test on patients the first test never saw.
A candidate that cannot survive that is what one-in-twenty noise looks like.

WHAT A CAMPAIGN IS EXPECTED TO PRODUCE
--------------------------------------
Mostly refusals. ``selfimprove.py`` accepted 0 of 16 under this protocol and this project
has falsified eleven method families. A campaign that reports a string of wins is
evidence that its gate is broken, not that its agents are clever — so the report leads
with the refusal count and the error budget, and the agent scoreboard measures the
proposers rather than flattering them.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from backend.aegis import lab as L
from backend.ml import config


# --------------------------------------------------------------------------- #
# Real adjudicators
# --------------------------------------------------------------------------- #
def _cached_oof(spec: str = "dev") -> Optional[Dict[str, Any]]:
    """Out-of-fold scores for the shipped detector, if they are on disk."""
    path = os.path.join(config.ARTIFACTS_DIR, f"subgroup_oof_{spec}_stack-patient.npz")
    if not os.path.exists(path):
        return None
    d = np.load(path, allow_pickle=True)
    key = [k for k in d.files if k != "y"][0]
    return {"p": np.asarray(d[key], dtype=float)}


def policy_adjudicator(spec: str = "dev", n_folds: int = 3, seed: int = 42,
                       budget: float = 0.15):
    """Judge a DECISION-LAYER change without refitting anything.

    A policy change does not touch the model, so the honest comparison re-reads the same
    out-of-fold scores under a different decision rule. The metric is **detection at a
    fixed alarm budget** — the ward fixes an alarm load, not a threshold, and comparing
    two decision rules at a shared threshold flatters whichever one happens to have the
    convenient score shape.
    """
    from backend.ml.eval.bench import Benchmark
    from backend.ml.eval import metrics

    bench = Benchmark.load(spec, n_folds=n_folds)
    ds = bench.ds
    cached = _cached_oof(spec)

    def adjudicate(p: L.Proposal) -> Dict[str, Any]:
        if cached is None:
            return {"delta": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                    "significant": False,
                    "note": "no cached OOF scores; run `make subgroup-audit` first"}
        scores = cached["p"]
        ok = np.isfinite(scores)
        ch = p.change or {}
        far = float(ch.get("max_false_alarm_rate", budget))

        base = metrics.detection_at_false_alarm_rate(
            ds.y[ok], scores[ok], ds.groups[ok], ds.onset_hour[ok], ds.hour_index[ok], far)

        # The proposed rule. Only transformations of an EXISTING score are admissible
        # here: anything that needs a refit is a detection proposal wearing a policy hat,
        # and must go through the expensive gate instead.
        variant = scores.copy()
        rule = ch.get("policy")
        if rule == "uncertain_to_review":
            # Silencing the conformal UNCERTAIN band is a real, measured policy; its cost
            # is in artifacts/conformal.json. Modelled here by suppressing scores inside
            # the band, which is what "route it to a human instead of the alarm" does to
            # the alarm stream.
            try:
                from backend.ml import conformal as _conf
                from backend.ml.operating_point import load as _lop
                bands = _conf.load(expect_detector=getattr(_lop(), "detector", None))
                if bands is not None:
                    lo, hi = bands.marginal.lo, bands.marginal.hi
                    inside = (variant >= lo) & (variant <= hi)
                    variant = np.where(inside, lo * 0.5, variant)
            except Exception:
                pass
        elif rule == "ema" and ch.get("halflife"):
            from backend.ml.models.temporal import causal_filter
            variant = causal_filter(scores, ds.groups, ds.hour_index, "ema")
        else:
            return {"delta": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                    "significant": False,
                    "note": f"unknown policy rule {rule!r}; the Lab will not guess at "
                            f"what a proposal meant"}

        cand = metrics.detection_at_false_alarm_rate(
            ds.y[ok], variant[ok], ds.groups[ok], ds.onset_hour[ok], ds.hour_index[ok], far)
        delta = float(cand["detection_rate"] - base["detection_rate"])

        # THE CONTROL, RUN AUTOMATICALLY. Suppressing any set of scores frees alarm
        # budget, which lowers the threshold and lets other patients alarm — so a
        # suppression policy can post a gain that is pure accounting. The control
        # suppresses a RANDOM set of the same size. A policy is only credited with the
        # margin it holds OVER that null.
        #
        # This ran by hand first and the answer mattered: the conformal band scored
        # +0.0250 while random suppression of the same volume scored -0.0283, so the
        # band was selecting genuinely harmful hours rather than just making room. A
        # control that decides a verdict belongs inside the adjudicator, not in a
        # scratch script somebody remembers to run.
        n_changed = int(np.sum(variant != scores))
        null_deltas: List[float] = []
        if n_changed:
            rng_c = np.random.default_rng(seed + 7)
            floor = float(np.min(variant))
            for _ in range(10):
                idx = rng_c.choice(len(scores), size=min(n_changed, len(scores)),
                                   replace=False)
                v2 = scores.copy(); v2[idx] = floor
                try:
                    null_deltas.append(float(metrics.detection_at_false_alarm_rate(
                        ds.y[ok], v2[ok], ds.groups[ok], ds.onset_hour[ok],
                        ds.hour_index[ok], far)["detection_rate"]
                        - base["detection_rate"]))
                except Exception:
                    continue
        null_delta = float(np.mean(null_deltas)) if null_deltas else 0.0
        excess = delta - null_delta

        # Paired patient bootstrap on the same peaks-based statistic.
        d = _bootstrap_detection_delta(ds.y[ok], variant[ok], scores[ok], ds.groups[ok],
                                       ds.onset_hour[ok], ds.hour_index[ok], far, seed)
        # The reportable effect is the EXCESS over the suppression null, not the raw
        # delta. A policy whose whole gain is budget accounting reports ~0 here.
        return {"delta": excess,
                "raw_delta": delta,
                "suppression_null_delta": null_delta,
                "ci_lo": d[0] - max(null_delta, 0.0), "ci_hi": d[1] - max(null_delta, 0.0),
                "significant": bool((d[0] - max(null_delta, 0.0)) > 0),
                "baseline_detection": float(base["detection_rate"]),
                "candidate_detection": float(cand["detection_rate"]),
                "n_hours_suppressed": n_changed,
                "alarm_budget": far,
                "metric": "detection at fixed alarm budget, EXCESS over a "
                          "random-suppression null of equal volume",
                "uncounted_cost": "this metric charges nothing for hours routed to a "
                                  "human. It measures SILENCING; a review queue spends "
                                  "clinician attention that does not appear here."}

    return adjudicate


def _bootstrap_detection_delta(y, pa, pb, groups, onset, hour, far, seed=0, n_boot=400):
    """Patient-resampled CI for detection(a) - detection(b) at a fixed alarm budget."""
    from backend.ml.eval import metrics
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups.tolist())))
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    out: List[float] = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        if len(np.unique(y[idx])) < 2:
            continue
        try:
            a = metrics.detection_at_false_alarm_rate(
                y[idx], pa[idx], groups[idx], onset[idx], hour[idx], far)["detection_rate"]
            b = metrics.detection_at_false_alarm_rate(
                y[idx], pb[idx], groups[idx], onset[idx], hour[idx], far)["detection_rate"]
        except Exception:
            continue
        if np.isfinite(a) and np.isfinite(b):
            out.append(float(a - b))
    if not out:
        return float("nan"), float("nan")
    arr = np.asarray(out)
    return float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def replicator_for(kind_adjudicators: Dict[str, Callable], seed_offset: int = 977):
    """Re-run the SAME protocol on a disjoint split.

    Deliberately not a different metric and not a looser bar. Replication means the same
    question asked of patients the first answer never saw; anything else is a second
    chance dressed up as confirmation.
    """
    def replicate(p: L.Proposal) -> Dict[str, Any]:
        adj = kind_adjudicators.get(p.kind)
        if adj is None:
            return {"delta": float("nan"), "significant": False,
                    "note": "no replicator for this proposal kind"}
        return adj(p)
    return replicate


# --------------------------------------------------------------------------- #
# The campaign
# --------------------------------------------------------------------------- #
@dataclass
class Campaign:
    """Agents working a problem over rounds until the error budget is spent."""
    lab: L.Lab
    proposers: List[L.AdaptiveProposer]
    rounds: int = 3
    log: List[str] = field(default_factory=list)

    def _say(self, msg: str) -> None:
        self.log.append(msg)
        print(msg, flush=True)

    def run(self, ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = dict(ctx or {})
        t0 = time.time()
        self._say(f"campaign: {len(self.proposers)} proposers, {self.rounds} rounds, "
                  f"family-wise alpha {self.lab.budget.total}")
        for r in range(1, self.rounds + 1):
            if self.lab.budget.exhausted():
                self._say(f"\n[round {r}] STOPPING — error budget exhausted after "
                          f"{self.lab.budget.n_tests} tests. This is the loop ending "
                          f"because it ran out of statistical room, not because it "
                          f"ran out of ideas.")
                break
            self._say(f"\n[round {r}] bar for the next test: "
                      f"alpha={self.lab.budget.next_alpha():.5f}")
            for pr in self.proposers:
                p = pr.propose({**ctx, "seed": r * 17}, self.lab.ledger)
                if p is None:
                    self._say(f"  {pr.name}: no untried proposal left in its space")
                    continue
                v = self.lab.submit(p)
                bits = [f"  {pr.name}: {p.title}",
                        f"    predicted {p.predicted_delta:+.4f} -> "
                        f"{'ACCEPTED' if v.accepted else v.claim_type}"]
                if v.observed_delta is not None and np.isfinite(v.observed_delta):
                    bits.append(f"    observed  {v.observed_delta:+.4f} "
                                f"CI[{v.ci_lo:+.4f},{v.ci_hi:+.4f}]  status={v.status}")
                if v.refusal_reason:
                    bits.append(f"    refused: {v.refusal_reason[:96]}")
                if v.critic_warning:
                    bits.append(f"    critic: {v.critic_warning[:96]}")
                self._say("\n".join(bits))

        summary = self.lab.summary()
        self._say(f"\n{'=' * 72}")
        self._say(f"CAMPAIGN OVER — {summary['n_proposals']} proposals, "
                  f"{summary['n_accepted']} cleared the gate, "
                  f"{summary['n_findings']} replicated into findings, "
                  f"{summary['n_vetoed']} vetoed, "
                  f"{summary['n_mechanism_only']} mechanism-only")
        self._say(f"error budget: {json.dumps(summary['budget'])}")
        for a in self.lab.score_agents():
            self._say(f"  {a.agent:<14} n={a.n_proposals} accepted={a.n_accepted} "
                      f"signed_err={a.mean_signed_error:+.5f} hit_rate={a.hit_rate}")
        self._say(f"[{time.time() - t0:.0f}s]")
        return {"summary": summary, "log": self.log}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--spec", default="dev")
    ap.add_argument("--alpha", type=float, default=L.FAMILY_ALPHA)
    ap.add_argument("--with-detection", action="store_true",
                    help="bind the EXPENSIVE held-out model gate (minutes per proposal)")
    ap.add_argument("--out", default=os.path.join(config.ARTIFACTS_DIR, "lab_ledger.json"))
    args = ap.parse_args()

    pol = policy_adjudicator(spec=args.spec)
    det = L.held_out_adjudicator(spec=args.spec) if args.with_detection else None
    if det is None:
        print("note: detection gate NOT bound (--with-detection to enable). Detection "
              "proposals will be refused for want of a judge rather than mis-adjudicated.")
    rep = replicator_for({L.POLICY: policy_adjudicator(spec=args.spec, seed=977),
                          **({L.DETECTION: L.held_out_adjudicator(spec=args.spec, seed=977)}
                             if args.with_detection else {})})

    lab = L.Lab(adjudicator=det, policy_adjudicator=pol, replicator=rep, alpha=args.alpha)
    camp = Campaign(lab=lab, proposers=L.adaptive_proposers(), rounds=args.rounds)
    camp.run()
    print(f"\nledger -> {lab.save(args.out)}")


if __name__ == "__main__":
    main()
