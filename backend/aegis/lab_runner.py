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
from backend.aegis.lab_registry import LabRegistry
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
                       budget: float = 0.15, arm: str = "gate"):
    """Judge a DECISION-LAYER change without refitting anything.

    A policy change does not touch the model, so the honest comparison re-reads the same
    out-of-fold scores under a different decision rule. The metric is **detection at a
    fixed alarm budget** — the ward fixes an alarm load, not a threshold, and comparing
    two decision rules at a shared threshold flatters whichever one happens to have the
    convenient score shape.

    ``arm`` SELECTS A DISJOINT PATIENT POOL, AND IT HAD TO
    ------------------------------------------------------
    This function did not split the cohort at all. Every call scored the same 8000
    patients against the same cached scores, so the campaign's "replication" — built by
    calling it again at ``seed=977`` — re-ran the identical computation. Measured: the
    raw delta came back **bit-identical at +0.025037** on both calls, and only the
    suppression null's RNG moved the reported number, by 0.002.

    A policy candidate therefore replicated with probability ~1. The conformal-band
    policy would have been published as a FINDING at +0.0543 "replicated +0.0523" on the
    strength of a test that was the same test twice. The gate and replication arms now
    draw from ``three_way_split``, whose pools are disjoint by construction.
    """
    from backend.ml.eval.bench import Benchmark
    from backend.ml.eval import metrics

    bench = Benchmark.load(spec, n_folds=n_folds)
    ds = bench.ds
    cached = _cached_oof(spec)
    _, gate_idx, rep_idx = L.three_way_split(ds.groups, ds.y, seed=42)
    pool = rep_idx if arm == "replication" else gate_idx
    in_pool = np.zeros(len(ds.y), dtype=bool)
    in_pool[pool] = True

    def adjudicate(p: L.Proposal, alpha: float = 0.05) -> Dict[str, Any]:
        """``alpha`` is the bar the Lab charged this test, and it is APPLIED here.

        Before this parameter existed the Lab spent a geometric alpha, printed it in the
        verdict's limits, and then took a fixed 95% interval regardless.
        """
        if cached is None:
            return {"delta": float("nan"), "ci_lo": float("nan"), "ci_hi": float("nan"),
                    "significant": False,
                    "note": "no cached OOF scores; run `make subgroup-audit` first"}
        scores = cached["p"]
        # The arm's pool, intersected with rows that actually carry a score. Before this
        # the mask was `np.isfinite(scores)` alone — every patient, every time.
        ok = np.isfinite(scores) & in_pool
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
        # The suppression null is only a valid control for a policy that SUPPRESSES.
        # A smoothing filter changes almost every score without silencing any, so
        # counting "scores changed" and then suppressing that many at random compares a
        # filter against near-total suppression — which scored -0.0788 and handed EMA a
        # fabricated +0.0714 "significant" excess, larger than the real band finding.
        # The control is applied only where it is meaningful.
        # Detect suppression STRUCTURALLY: a suppression policy drives the hours it
        # touches to one constant floor. A first attempt compared the variant against
        # `min(scores) * 1.5`, which failed to recognise the conformal-band policy at all
        # — so while fixing the EMA over-crediting I silently switched the control OFF
        # for the one case that needed it, and the band finding was briefly reported with
        # no control behind it. Testing the shape of the change cannot make that mistake.
        changed = variant != scores
        n_ch = int(changed.sum())
        is_suppression = bool(
            n_ch > 0
            and np.all(variant[changed] < scores[changed])
            and np.allclose(variant[changed], variant[changed][0], rtol=0, atol=1e-12))
        n_changed = n_ch if is_suppression else 0
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
        # The null is ESTIMATED, not known, so its sampling error must be carried into
        # the interval. Subtracting a point estimate as if it were exact produced a
        # verdict of +0.0197 with CI [+0.0197, +0.0197] — a zero-width bootstrap
        # interval, which cannot happen for a statistic that actually varies. All the
        # uncertainty had been silently discarded along with the null's spread.
        null_se = (float(np.std(null_deltas, ddof=1)) / max(len(null_deltas), 1) ** 0.5
                   if len(null_deltas) > 1 else 0.0)
        excess = delta - null_delta
        control = ("random-suppression null of equal volume" if n_changed
                   else "none applicable — this policy does not suppress, so the raw "
                        "delta is reported and no null is subtracted")

        # Paired patient bootstrap on the same peaks-based statistic.
        d = _bootstrap_detection_delta(ds.y[ok], variant[ok], scores[ok], ds.groups[ok],
                                       ds.onset_hour[ok], ds.hour_index[ok], far, seed,
                                       alpha=alpha)
        # The reportable effect is the EXCESS over the suppression null, not the raw
        # delta. A policy whose whole gain is budget accounting reports ~0 here.
        return {"delta": excess,
                "alpha": float(alpha),
                "raw_delta": delta,
                "suppression_null_delta": null_delta,
                # The CI must describe the SAME quantity as the point estimate. An
                # earlier version shifted the bootstrap by max(null_delta, 0) as a
                # "conservative" hack while the point estimate used the full
                # null_delta — so with a negative null the two disagreed and a verdict
                # came back as delta +0.0197 with CI [+0.0000, +0.0000], a point
                # estimate outside its own interval. Both are shifted identically now.
                # Widen by the null's standard error, in quadrature with the raw
                # interval's own half-width, so the reported interval covers both
                # sources of uncertainty rather than only the one that was convenient.
                "ci_lo": _widen(d[0] - null_delta, d[1] - null_delta, null_se)[0],
                "ci_hi": _widen(d[0] - null_delta, d[1] - null_delta, null_se)[1],
                "null_standard_error": round(null_se, 6),
                "significant": bool(
                    _widen(d[0] - null_delta, d[1] - null_delta, null_se)[0] > 0
                    and _widen(d[0] - null_delta, d[1] - null_delta, null_se)[1]
                        - _widen(d[0] - null_delta, d[1] - null_delta, null_se)[0] > 1e-9),
                "arm": arm,
                "n_score_patients": int(len(set(ds.groups[ok].tolist()))),
                "patient_ids_hash": L._pool_hash(ds.groups[ok]),
                "baseline_detection": float(base["detection_rate"]),
                "candidate_detection": float(cand["detection_rate"]),
                "n_hours_suppressed": n_changed,
                "alarm_budget": far,
                "control": control,
                "metric": ("detection at fixed alarm budget"
                           + (", EXCESS over a random-suppression null of equal volume"
                              if n_changed else "")),
                "uncounted_cost": "this metric charges nothing for hours routed to a "
                                  "human. It measures SILENCING; a review queue spends "
                                  "clinician attention that does not appear here."}

    return adjudicate


def _widen(lo: float, hi: float, extra_se: float) -> tuple:
    """Widen an interval by an independent standard error, added in quadrature.

    A degenerate (zero-width) interval is the signature of discarded uncertainty, and a
    verdict must never be called significant off one — so a result whose interval is
    still a point after widening is refused by the caller rather than believed.
    """
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return lo, hi
    half = max((hi - lo) / 2.0, 0.0)
    mid = (hi + lo) / 2.0
    new_half = float(np.sqrt(half * half + (1.96 * extra_se) ** 2))
    return mid - new_half, mid + new_half


def _bootstrap_detection_delta(y, pa, pb, groups, onset, hour, far, seed=0, n_boot=400,
                               alpha=0.05):
    """Patient-resampled CI for detection(a) - detection(b) at a fixed alarm budget.

    ``alpha`` sets the interval's width. It was hardcoded to the 2.5/97.5 quantiles, so
    a test the Lab charged at alpha=0.00625 still got a 95% interval.
    """
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
    q = float(alpha) / 2.0
    return float(np.quantile(arr, q)), float(np.quantile(arr, 1.0 - q))


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
class PrincipalInvestigator:
    """The agent that runs the research programme.

    WHAT IT CONTROLS, WHICH IS EVERYTHING EXCEPT THE VERDICT
    --------------------------------------------------------
    The campaign used to round-robin a fixed proposer list, which is a schedule, not a
    research programme. The PI owns the decisions a scientist actually makes:

    * **the agenda** — which line of inquiry gets the next test, ranked by what the
      ledger says has yielded so far rather than by list order;
    * **budget allocation** — the family-wise alpha is finite, so spending it on a line
      that has produced nothing is a real cost;
    * **when to stop** — exhausted budget, or every line dry;
    * **what the campaign concluded** — written from the ledger, refusals included.

    What it cannot do is decide whether a result counts. That belongs to the gate, and
    the separation is not a restriction on the agents — it is the only reason their
    output is worth reading. An agent that could rule on its own work would be back to
    the failure mode ``selfimprove.py`` was built to prevent.

    HOW IT RANKS A LINE OF INQUIRY
    ------------------------------
    Yield per test, with an optimism term for lines never tried. A line that has been
    tested and refused repeatedly drops; a line that has never been tested is worth one
    look. This is a bandit-style allocation over research directions, and the point is
    that it is *visible* — ``agenda()`` returns the ranking and the reason, so a reader
    can disagree with the PI rather than take its word.
    """

    name = "PrincipalInvestigator"

    def __init__(self, proposers: Sequence[L.AdaptiveProposer]):
        self.proposers = list(proposers)

    def _yield_of(self, agent: str, ledger: Sequence[L.Verdict]) -> Dict[str, Any]:
        mine = [v for v in ledger if v.proposal.agent == agent]
        tested = [v for v in mine if v.observed_delta is not None
                  and np.isfinite(v.observed_delta)]
        best = max((v.observed_delta for v in tested), default=None)
        return {"n": len(mine), "n_tested": len(tested),
                "n_accepted": sum(1 for v in mine if v.accepted),
                "best_delta": best,
                "untestable": bool(mine) and not tested}

    def agenda(self, ledger: Sequence[L.Verdict]) -> List[Dict[str, Any]]:
        """Rank the lines of inquiry, with the reason attached to each."""
        out = []
        for pr in self.proposers:
            y = self._yield_of(pr.name, ledger)
            if y["n"] == 0:
                score, why = 1.0, "never tried — worth one look"
            elif y["untestable"]:
                score, why = -1.0, ("no adjudicator can judge this line; spending budget "
                                    "on it buys nothing")
            elif y["n_accepted"]:
                score, why = 0.9, f"has produced {y['n_accepted']} accepted result(s)"
            elif y["best_delta"] is not None and y["best_delta"] > 0:
                score = 0.5 + float(y["best_delta"])
                why = f"refused so far but trending positive (best {y['best_delta']:+.4f})"
            else:
                score = 0.2 - 0.05 * y["n_tested"]
                why = f"{y['n_tested']} test(s), nothing positive — deprioritised"
            out.append({"line": pr.name, "prior": pr.prior, "score": round(score, 4),
                        "why": why, **y})
        return sorted(out, key=lambda r: -r["score"])

    def next_move(self, ledger: Sequence[L.Verdict], ctx: Dict[str, Any]
                  ) -> Optional[tuple]:
        """Pick the line to spend the next test on, and get a proposal from it."""
        for row in self.agenda(ledger):
            if row["score"] < 0:
                continue
            pr = next(p for p in self.proposers if p.name == row["line"])
            prop = pr.propose(ctx, ledger)
            if prop is not None:
                return prop, row
        return None

    def conclude(self, lab: L.Lab) -> List[str]:
        """What the campaign found, written from the ledger including the refusals."""
        s = lab.summary()
        out = [f"{s['n_proposals']} proposals: {s['n_accepted']} cleared the gate, "
               f"{s['n_findings']} replicated, {s['n_vetoed']} vetoed, "
               f"{s['n_mechanism_only']} mechanism-only."]
        findings = [v for v in lab.ledger if v.status == "finding"]
        for v in findings:
            out.append(f"FINDING — {v.proposal.title}: {v.observed_delta:+.4f} "
                       f"CI[{v.ci_lo:+.4f},{v.ci_hi:+.4f}], replicated "
                       f"{v.replication_delta:+.4f}. Proposed by {v.proposal.agent}, "
                       f"which predicted {v.proposal.predicted_delta:+.4f}.")
        dead = [r for r in self.agenda(lab.ledger) if r["score"] < 0]
        for r in dead:
            out.append(f"UNTESTABLE — the {r['line']} line: {r['why']}. Its refusals say "
                       f"nothing about the science, only that no judge was bound.")
        if not findings:
            out.append("No finding. That is the expected outcome under this protocol and "
                       "is reported as a result, not as a failure to try.")
        b = lab.budget
        out.append(f"Budget: {b.spent:.5f} of {b.total} over {b.n_tests} tests"
                   + (" — exhausted." if b.exhausted() else "."))
        return out


@dataclass
class Campaign:
    """Agents working a problem over rounds until the error budget is spent."""
    lab: L.Lab
    proposers: List[L.AdaptiveProposer]
    rounds: int = 3
    log: List[str] = field(default_factory=list)
    pi: "PrincipalInvestigator" = None

    def __post_init__(self):
        if self.pi is None:
            self.pi = PrincipalInvestigator(self.proposers)

    def _say(self, msg: str) -> None:
        self.log.append(msg)
        print(msg, flush=True)

    def run(self, ctx: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = dict(ctx or {})
        t0 = time.time()
        # PRE-REGISTER the test count, before the first proposal is generated. The
        # ceiling is rounds x proposers and it is known here, so the budget can be spent
        # flat instead of geometrically — 8 tests at alpha=0.00625 rather than 4 tests
        # front-loaded at 0.025 and then a bar under the bootstrap's resolution. The
        # budget refuses a plan declared after any alpha has been spent, which is what
        # keeps this a pre-registration rather than a knob.
        planned = max(1, self.rounds * len(self.proposers))
        try:
            self.lab.budget.plan(planned)
        except ValueError as exc:
            self._say(f"note: keeping the open-ended geometric schedule ({exc})")
        b = self.lab.budget
        self._say(f"campaign: {len(self.proposers)} proposers, {self.rounds} rounds, "
                  f"family-wise alpha {b.total} over a PRE-DECLARED {b.planned_tests} "
                  f"tests ({b.schedule()}), so every test faces "
                  f"alpha={b.next_alpha():.5f}")
        for r in range(1, self.rounds + 1):
            if self.lab.budget.exhausted():
                self._say(f"\n[round {r}] STOPPING — error budget exhausted after "
                          f"{self.lab.budget.n_tests} tests. This is the loop ending "
                          f"because it ran out of statistical room, not because it "
                          f"ran out of ideas.")
                break
            self._say(f"\n[round {r}] bar for the next test: "
                      f"alpha={self.lab.budget.next_alpha():.5f}")
            agenda = self.pi.agenda(self.lab.ledger)
            self._say("  PI agenda: " + " | ".join(
                f"{a['line']}({a['score']:+.2f})" for a in agenda))
            for _ in range(len(self.proposers)):
                move = self.pi.next_move(self.lab.ledger,
                                         {**ctx, "seed": int(ctx.get("seed", 0)) + r * 17})
                if move is None:
                    self._say("  PI: every line is dry or unjudgeable — stopping the round")
                    break
                p, row = move
                self._say(f"  PI picks {row['line']} — {row['why']}")
                v = self.lab.submit(p)
                bits = [f"    {p.agent}: {p.title}",
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
        self._say("\nPI CONCLUSIONS:")
        for line in self.pi.conclude(self.lab):
            self._say(f"  · {line}")
        self._say("")
        for a in self.lab.score_agents():
            self._say(f"  {a.agent:<14} n={a.n_proposals} accepted={a.n_accepted} "
                      f"signed_err={a.mean_signed_error:+.5f} hit_rate={a.hit_rate}")
        self._say(f"[{time.time() - t0:.0f}s]")
        return {"summary": summary, "log": self.log}


def _write_partition(spec: str, path: str) -> Optional[Dict[str, Any]]:
    """Measure the search/gate/replication partition and persist what it is.

    The console used to state the pool sizes and the zero intersection as literal text.
    That is exactly the kind of claim that goes stale silently — the split could change
    and the page would keep asserting the old numbers. Measured here, served from disk.
    """
    try:
        from backend.ml.eval.bench import Benchmark
        ds = Benchmark.load(spec, n_folds=2).ds
        srch, gate, rep = L.three_way_split(ds.groups, ds.y, 42)
        S = set(ds.groups[srch].tolist())
        G = set(ds.groups[gate].tolist())
        R = set(ds.groups[rep].tolist())
        out = {
            "spec": spec, "seed": 42,
            "measured": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "pools": {
                "search": {"n_patients": len(S), "septic_rate": round(float(ds.y[srch].mean()), 5),
                           "role": "every model is fitted here, and only here"},
                "gate": {"n_patients": len(G), "septic_rate": round(float(ds.y[gate].mean()), 5),
                         "role": "scored once per proposal; can accept a candidate",
                         "fingerprint": L._pool_hash(ds.groups[gate])},
                "replication": {"n_patients": len(R), "septic_rate": round(float(ds.y[rep].mean()), 5),
                                "role": "disjoint by construction; the only route to a finding",
                                "fingerprint": L._pool_hash(ds.groups[rep])},
            },
            "intersections": {"gate_replication": len(G & R),
                              "search_gate": len(S & G),
                              "search_replication": len(S & R)},
            "n_patients_total": len(S | G | R),
            "disjoint": bool(not (G & R) and not (S & G) and not (S & R)),
        }
        with open(path, "w") as fh:
            json.dump(out, fh, indent=2, allow_nan=False)
        print(f"partition: search {len(S)} | gate {len(G)} | replication {len(R)}, "
              f"pairwise intersection {len(G & R)}/{len(S & G)}/{len(S & R)} -> {path}")
        return out
    except Exception as exc:
        print(f"note: could not measure the partition ({exc})")
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=None,
                    help="proposal-generation seed. Defaults to one derived from the "
                         "number of campaigns already on the record, so a new campaign "
                         "explores new hypotheses instead of re-asking the last one's "
                         "questions and hitting the dedup refusal on every proposal.")
    ap.add_argument("--spec", default="dev")
    ap.add_argument("--alpha", type=float, default=L.FAMILY_ALPHA)
    ap.add_argument("--with-detection", action="store_true",
                    help="bind the EXPENSIVE held-out model gate (minutes per proposal)")
    ap.add_argument("--out", default=os.path.join(config.ARTIFACTS_DIR, "lab_ledger.json"))
    ap.add_argument("--registry",
                    default=os.path.join(config.ARTIFACTS_DIR, "lab_registry.json"))
    args = ap.parse_args()

    pol = policy_adjudicator(spec=args.spec)
    det = L.held_out_adjudicator(spec=args.spec) if args.with_detection else None
    if det is None:
        print("note: detection gate NOT bound (--with-detection to enable). Detection "
              "proposals will be refused for want of a judge rather than mis-adjudicated.")
    # The replication arm scores a DISJOINT pool from the same partition. It used to be
    # built by re-seeding at 977, which drew an overlapping sample for detection (25.1%
    # shared) and no new sample at all for policy (the same 8000 patients twice).
    rep = replicator_for({
        L.POLICY: policy_adjudicator(spec=args.spec, arm="replication"),
        **({L.DETECTION: L.held_out_adjudicator(spec=args.spec, arm="replication")}
           if args.with_detection else {})})

    # The registry is the Lab's memory. Loading it here is what makes a campaign part of
    # a research programme rather than an isolated run that overwrites its predecessor.
    reg = LabRegistry.load(args.registry)
    prog = reg.programme()
    if prog["n_campaigns"]:
        print(f"registry: {prog['n_campaigns']} prior campaign(s), "
              f"{prog['n_hypotheses']} hypotheses already tested, "
              f"{prog['expected_false_findings']:.4f} of alpha spent across the "
              f"programme ({prog['findings_vs_expected_false']})")
        ok, bad = reg.verify_chain()
        if not ok:
            print(f"WARNING: registry hash chain BROKEN at seq {bad}. A verdict has been "
                  f"edited or removed since it was recorded.")

    # Record the partition the campaign actually ran on, so the console can show real
    # pool sizes instead of numbers typed into a page by hand. Disjointness is the claim
    # a finding rests on; it should be served from the split, not asserted in markup.
    _write_partition(args.spec, os.path.join(
        os.path.dirname(args.registry) or config.ARTIFACTS_DIR, "lab_partition.json"))

    lab = L.Lab(adjudicator=det, policy_adjudicator=pol, replicator=rep,
                alpha=args.alpha, registry=reg)
    camp = Campaign(lab=lab, proposers=L.adaptive_proposers(), rounds=args.rounds)
    seed = args.seed if args.seed is not None else 1009 * (prog["n_campaigns"] + 1)
    print(f"proposal seed {seed} (campaign {prog['n_campaigns'] + 1} of the programme)")
    camp.run({"seed": seed})

    cid = reg.record_campaign(lab.ledger, lab.budget,
                              notes={"spec": args.spec,
                                     "detection_gate": bool(det is not None)})
    reg.save()
    after = reg.programme()
    print(f"\ncampaign {cid} recorded. Programme now: {after['n_campaigns']} campaigns, "
          f"{after['n_tested']} tests, {after['expected_false_findings']:.4f} expected "
          f"false findings, {after['n_findings']} finding(s).")
    print(f"registry -> {reg.path}")
    print(f"ledger   -> {lab.save(args.out)}")


if __name__ == "__main__":
    main()
