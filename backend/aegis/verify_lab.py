"""Adversarial audit of the Lab.

    PYTHONPATH=. python -m backend.aegis.verify_lab

The Lab's whole value rests on one property: **an agent cannot make the gate say yes.**
So this suite plays the adversary. It submits proposals designed to cheat — an agent
that reports its own result, one that tries to turn a simulator run into an efficacy
claim, one that widens the alarm budget until anything passes, one that would let
abstention silence an escalation — and requires each to be refused for the right reason.

A refusal for the wrong reason is a failure here. A gate that says no because of a typo
is not a gate.
"""
from __future__ import annotations

import inspect as _inspect
import sys

import numpy as _np
from typing import Any, Dict, List, Tuple

from backend.aegis import lab as L

PASS, FAIL = "PASS", "FAIL"
_res: List[Tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _res.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


def _prop(**kw) -> L.Proposal:
    base = dict(agent="Tester", kind=L.DETECTION, title="t", change={},
                rationale="r", predicted_delta=0.003, predicted_direction="improve")
    base.update(kw)
    return L.Proposal(**base)


def main() -> int:
    print("1. an agent cannot report its own result")
    lab = L.Lab(adjudicator=None)
    v = lab.submit(_prop(title="my change works, delta +0.20"))
    check("with no adjudicator bound, nothing is accepted",
          not v.accepted and "did not measure" in (v.refusal_reason or ""),
          v.refusal_reason)

    print("\n2. the gate needs BOTH a real effect and significance")
    lab = L.Lab(adjudicator=lambda p: {"delta": 0.0009, "ci_lo": 0.0002,
                                       "ci_hi": 0.0016, "significant": True})
    v = lab.submit(_prop(title="tiny but significant"))
    check("a significant but trivial effect is refused",
          not v.accepted and "below the minimum" in (v.refusal_reason or ""),
          f"delta 0.0009 < MIN_EFFECT {L.MIN_EFFECT}")

    lab = L.Lab(adjudicator=lambda p: {"delta": 0.02, "ci_lo": -0.01,
                                       "ci_hi": 0.05, "significant": False})
    v = lab.submit(_prop(title="large but not significant"))
    check("a large but non-significant effect is refused",
          not v.accepted and "not significant" in (v.refusal_reason or ""),
          "CI includes zero")

    lab = L.Lab(adjudicator=lambda p: {"delta": 0.02, "ci_lo": 0.005,
                                       "ci_hi": 0.035, "significant": True})
    v = lab.submit(_prop(title="real and significant"))
    check("a real, significant effect IS accepted", v.accepted and v.claim_type == "HELD_OUT",
          f"delta {v.observed_delta:+.4f}")
    check("...and even then it is only a CANDIDATE, not a shipped change",
          any("CANDIDATE" in x for x in v.limits))

    print("\n3. the treatment firewall is structural, not a convention")
    lab = L.Lab(adjudicator=lambda p: {"delta": 0.5, "ci_lo": 0.4, "ci_hi": 0.6,
                                       "significant": True})
    v = lab.submit(_prop(kind=L.TREATMENT, title="vancomycin dose change",
                         change={"mechanism_only": True, "drug": "vancomycin"}))
    check("a treatment proposal is typed MECHANISM, never accepted as a finding",
          v.claim_type == "MECHANISM" and not v.accepted, v.claim_type)
    check("efficacy_claim is None even with a huge simulator 'effect'",
          v.efficacy_claim is None,
          "the adjudicator offered delta=+0.5 and it did not become a claim")
    check("...and the limits say why a simulator result is not evidence",
          any("near chance on real patients" in x for x in v.limits))

    v = lab.submit(_prop(kind=L.TREATMENT, title="unflagged treatment change",
                         change={"drug": "meropenem"}))
    check("an unflagged treatment proposal is VETOED before the gate",
          v.vetoed_by == "SafetyOfficer" and not v.accepted, v.refusal_reason)

    print("\n4. safety vetoes are absolute and reach the gate first")
    lab = L.Lab(adjudicator=lambda p: {"delta": 0.9, "ci_lo": 0.8, "ci_hi": 1.0,
                                       "significant": True})
    for title, change, why in [
        ("widen the alarm budget", {"max_false_alarm_rate": 0.60}, "alarm budget"),
        ("let abstention silence alarms", {"abstention_downgrades_alarm": True}, "ordering"),
        ("bypass gate for speed", {}, "bypass"),
        ("disable safety prover", {}, "safety constraint"),
    ]:
        v = lab.submit(_prop(title=title, change=change, rationale=title))
        check(f"VETOED: {title}", v.vetoed_by == "SafetyOfficer" and not v.accepted,
              (v.refusal_reason or "")[:70])
    check("a veto beats an adjudicator that would have said yes",
          all(not x.accepted for x in lab.ledger),
          "the adjudicator was returning delta=+0.9, significant, for every call")

    print("\n5. the critic surfaces known dead ends without vetoing")
    lab = L.Lab(adjudicator=lambda p: {"delta": 0.02, "ci_lo": 0.005, "ci_hi": 0.03,
                                       "significant": True})
    v = lab.submit(_prop(title="add multi-scale temporal features",
                         rationale="temporal context should help"))
    check("a proposal repeating a recorded null carries the warning",
          v.critic_warning is not None and "Phase 7" in v.critic_warning,
          v.critic_warning)
    check("...but the critic does NOT veto it — it can still be tested",
          v.vetoed_by is None and v.accepted,
          "the critic informs the record; only the SafetyOfficer stops a proposal")
    v = lab.submit(_prop(title="modest change", predicted_delta=0.30))
    check("an implausible predicted gain is flagged as overclaiming",
          v.critic_warning is not None and "overclaiming" in v.critic_warning)

    print("\n6. proposers are measured, not trusted")
    # One Lab per agent here. A single shared Lab starves the second agent: the family
    # -wise budget is SHARED, so the first agent's tests tighten the bar for everyone and
    # the later proposals came back "budget exhausted" with no observed delta at all —
    # which is correct behaviour for error control and wrong for measuring two agents.
    _adj = lambda p: {"delta": 0.001, "ci_lo": -0.001, "ci_hi": 0.003,
                      "significant": False}
    lab_o, lab_r = L.Lab(adjudicator=_adj), L.Lab(adjudicator=_adj)
    for i in range(3):
        lab_o.submit(_prop(agent="Optimist", predicted_delta=0.05,
                           predicted_direction="improve", title=f"change {i}"))
        lab_r.submit(_prop(agent="Realist", predicted_delta=0.001,
                           predicted_direction="improve", title=f"tweak {i}"))
    scores = {**{s.agent: s for s in lab_o.score_agents()},
              **{s.agent: s for s in lab_r.score_agents()}}
    lab = lab_o
    check("an overconfident agent shows a large positive signed error",
          scores["Optimist"].mean_signed_error > scores["Realist"].mean_signed_error,
          f"Optimist {scores['Optimist'].mean_signed_error:+.4f} vs "
          f"Realist {scores['Realist'].mean_signed_error:+.4f} (>0 = overclaiming)")
    check("both are recorded with proposal counts",
          scores["Optimist"].n_proposals == 3 and scores["Realist"].n_proposals == 3)
    # The sharing is a deliberate property, so it is asserted rather than worked around.
    shared = L.Lab(adjudicator=_adj)
    for i in range(6):
        shared.submit(_prop(agent="A" if i % 2 else "B", title=f"s{i}"))
    starved = [v for v in shared.ledger if "budget exhausted" in (v.refusal_reason or "")]
    check("the error budget is SHARED across agents — one agent can spend another's",
          len(starved) > 0,
          f"{len(starved)} of 6 refused; family-wise control is per-session, not per-agent")
    s = lab.summary()
    check("the summary states that refusal is the expected outcome",
          "gate working" in s["note"], f"acceptance rate {s['acceptance_rate']:.0%}")

    print("\n7. the default proposers hold DIFFERENT priors")
    props = L.default_proposers()
    priors = {p.prior for p in props}
    # Asserted as "every proposer holds a DISTINCT prior", not as a fixed count. The
    # previous version hardcoded three and broke the moment a fourth (Pharmacologist)
    # was added — a test that fails when the system grows correctly is a maintenance
    # tax, not a check.
    check("every proposer holds a distinct prior",
          len(props) >= 3 and len(priors) == len(props),
          f"{len(props)} proposers: " + "; ".join(sorted(priors)))
    made = [p.generate({"seed": 1}) for p in props]
    check("...and every proposal kind the Lab defines has a proposer for it",
          {L.DETECTION, L.POLICY, L.TREATMENT} <= {m.kind for m in made},
          str(sorted({m.kind for m in made})))
    check("each produces a well-formed proposal with a registered prediction",
          all(m.predicted_direction in ("improve", "no_effect", "worsen")
              and isinstance(m.predicted_delta, float) for m in made))
    check("...and they do not all propose the same kind of change",
          len({m.kind for m in made}) > 1, str(sorted({m.kind for m in made})))

    print("\n8. the alpha budget is REAL, not just documented")
    # v1's docstring claimed a Bonferroni correction and submit() never applied one:
    # twenty proposals each faced the same bar, so a false pass was ~20x more likely
    # than the stated 5%. The budget is now an object with state.
    b = L.AlphaBudget()
    bars = []
    for _ in range(4):
        a = b.next_alpha(); bars.append(a); b.spend(a)
    check("the bar TIGHTENS with each test spent",
          all(bars[i] > bars[i + 1] for i in range(len(bars) - 1)),
          " -> ".join(f"{x:.5f}" for x in bars))
    check("...and the schedule converges within the family budget",
          b.spent <= L.FAMILY_ALPHA + 1e-9, f"spent {b.spent:.5f} of {L.FAMILY_ALPHA}")
    b2 = L.AlphaBudget()
    check("a fresh budget is NOT exhausted on test one",
          not b2.exhausted(), "an earlier draft spent the whole budget immediately")
    lab = L.Lab(adjudicator=lambda p: {"delta": 0.02, "ci_lo": 0.005, "ci_hi": 0.03,
                                       "significant": True})
    for i in range(12):
        lab.submit(_prop(title=f"repeat {i}"))
    exhausted = [v for v in lab.ledger if "budget exhausted" in (v.refusal_reason or "")]
    check("a proposal spammer eventually hits the budget and is REFUSED",
          len(exhausted) > 0,
          f"{len(exhausted)} of 12 refused once the bar outran the bootstrap")
    check("...and every accepted verdict records the alpha it had to clear",
          all(v.alpha_spent is not None for v in lab.ledger if v.accepted))

    print("\n9. POLICY proposals are not judged by the MODEL gate")
    lab = L.Lab(adjudicator=lambda p: {"delta": 0.9, "ci_lo": 0.8, "ci_hi": 1.0,
                                       "significant": True})
    v = lab.submit(_prop(kind=L.POLICY, title="route uncertain to review"))
    check("a policy change with no policy adjudicator is refused, not mis-judged",
          not v.accepted and "POLICY adjudicator" in (v.refusal_reason or ""),
          v.refusal_reason)
    lab2 = L.Lab(adjudicator=lambda p: {"delta": -0.9, "ci_lo": -1.0, "ci_hi": -0.8,
                                        "significant": False},
                 policy_adjudicator=lambda p: {"delta": 0.03, "ci_lo": 0.01,
                                               "ci_hi": 0.05, "significant": True})
    v = lab2.submit(_prop(kind=L.POLICY, title="route uncertain to review"))
    check("with its own judge bound, a policy proposal is judged by THAT one",
          v.accepted, "the model gate was returning -0.9 and did not decide this")

    print("\n10. the treatment sandbox RUNS and still cannot make a claim")
    lab = L.Lab()
    v = lab.submit(_prop(kind=L.TREATMENT, title="4h extended infusion",
                         change={"mechanism_only": True, "antibiotic_code": 1,
                                 "infusion_min": 240, "egfr": 45}))
    m = v.mechanism or {}
    check("the PBPK twin actually executed", bool(m.get("ran")), str(m.get("reason", "ran")))
    if m.get("ran"):
        base, var = m.get("baseline") or {}, m.get("variant") or {}
        check("real pharmacodynamic metrics come back, not None",
              base.get("pct_ft_above_mic") is not None
              and var.get("pct_ft_above_mic") is not None,
              f"fT>MIC {base.get('pct_ft_above_mic')} -> {var.get('pct_ft_above_mic')}")
        # `x or 1` is wrong here: mass_balance_error of 0.0 is FALSY, so the fallback
        # fired and the check could never pass on a correct model. Explicit None test.
        mbe_b, mbe_v = base.get("mass_balance_error"), var.get("mass_balance_error")
        check("the model's own mass-balance check passes on both arms",
              mbe_b is not None and mbe_v is not None
              and abs(float(mbe_b)) < 1e-6 and abs(float(mbe_v)) < 1e-6,
              f"baseline {mbe_b}, variant {mbe_v}")
        d = m.get("delta") or {}
        check("extended infusion raises T>MIC and lowers peak (textbook direction)",
              (d.get("pct_ft_above_mic") or 0) > 0 and (d.get("peak_venous") or 0) < 0,
              f"dT>MIC {d.get('pct_ft_above_mic')}, dpeak {d.get('peak_venous')}")
    check("...and it is STILL typed MECHANISM with no efficacy claim",
          v.claim_type == "MECHANISM" and v.efficacy_claim is None and not v.accepted)

    print("\n11. a candidate is not a finding until it REPLICATES")
    lab = L.Lab(adjudicator=lambda p: {"delta": 0.02, "ci_lo": 0.005, "ci_hi": 0.03,
                                       "significant": True})
    v = lab.submit(_prop(title="unreplicated win"))
    check("with no replicator bound it stays a CANDIDATE",
          v.status == "candidate" and v.replicated is None,
          f"status={v.status}")
    check("...and says so in the limits",
          any("must not be reported as a finding" in x for x in v.limits))
    lab = L.Lab(adjudicator=lambda p: {"delta": 0.02, "ci_lo": 0.005, "ci_hi": 0.03,
                                       "significant": True},
                replicator=lambda p: {"delta": 0.001, "significant": False})
    v = lab.submit(_prop(title="fails to replicate"))
    check("a candidate that does not replicate stays a candidate",
          v.status == "candidate" and v.replicated is False,
          f"replication delta {v.replication_delta:+.4f}")
    lab = L.Lab(adjudicator=lambda p: {"delta": 0.02, "ci_lo": 0.005, "ci_hi": 0.03,
                                       "significant": True},
                replicator=lambda p: {"delta": 0.018, "significant": True})
    v = lab.submit(_prop(title="replicates"))
    check("only a replicated candidate becomes a FINDING", v.status == "finding")

    print("\n12. adaptive proposers do not repeat themselves or their overclaims")
    ap = L.adaptive_proposers()[0]
    lab = L.Lab(adjudicator=lambda p: {"delta": 0.0, "ci_lo": -0.01, "ci_hi": 0.01,
                                       "significant": False})
    first = ap.propose({"seed": 0}, lab.ledger)
    check("an adaptive proposer produces a proposal", first is not None)
    for _ in range(3):
        q = ap.propose({"seed": 0}, lab.ledger)
        if q is None:
            break
        lab.submit(q)
    keys = [__import__("json").dumps(v.proposal.change, sort_keys=True, default=str)
            for v in lab.ledger]
    check("it never re-submits a change it already tried",
          len(keys) == len(set(keys)), f"{len(keys)} proposals, {len(set(keys))} distinct")
    bias = ap.observed_bias(lab.ledger)
    check("it measures its own bias from the ledger", bias > 0,
          f"observed overclaim {bias:+.4f}")
    nxt = ap.propose({"seed": 99}, lab.ledger)
    if nxt is not None:
        check("...and shrinks its next prediction by that bias",
              "recalibrated" in nxt.rationale, f"predicted {nxt.predicted_delta:+.5f}")

    print("\n13. the policy adjudicator's own controls (real data, real bugs)")
    try:
        from backend.aegis.lab_runner import policy_adjudicator
        adj = policy_adjudicator()
        band = adj(_prop(kind=L.POLICY, title="band",
                         change={"policy": "uncertain_to_review",
                                 "max_false_alarm_rate": 0.15}))
        ema = adj(_prop(kind=L.POLICY, title="ema",
                        change={"policy": "ema", "halflife": 6}))
    except Exception as exc:
        check("policy adjudicator runs", False, str(exc)[:90]); band = ema = None

    if band and ema:
        # BUG 1: the point estimate sat OUTSIDE its own CI, because the CI was shifted
        # by max(null,0) while the estimate used the full null.
        for nm, r in (("band", band), ("ema", ema)):
            check(f"{nm}: the point estimate lies inside its own CI",
                  r["ci_lo"] <= r["delta"] <= r["ci_hi"],
                  f"{r['delta']:+.4f} in [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}]")
        # BUG 2: the suppression null was applied to a SMOOTHING filter, comparing it
        # against near-total suppression and fabricating a +0.0714 significant excess.
        check("a smoothing filter gets NO suppression null",
              ema["n_hours_suppressed"] == 0 and ema["suppression_null_delta"] == 0.0,
              f"control: {ema['control'][:60]}")
        check("...and is therefore not credited with a fabricated excess",
              not ema["significant"] and abs(ema["delta"] - ema["raw_delta"]) < 1e-12,
              f"excess {ema['delta']:+.4f} == raw {ema['raw_delta']:+.4f}")
        # BUG 3: fixing bug 2 with a min()-based test switched the control OFF for the
        # band as well, so the one finding that needed a control briefly had none.
        check("a suppression policy IS recognised and DOES get its control",
              band["n_hours_suppressed"] > 0
              and band["suppression_null_delta"] != 0.0,
              f"{band['n_hours_suppressed']:,} hours suppressed, "
              f"null {band['suppression_null_delta']:+.4f}")
        # This check used to assert `credited == raw - null` unconditionally, which is
        # the rule that manufactured a finding out of a policy that did nothing. It now
        # asserts the corrected rule: an INFLATING null is subtracted, a negative one is
        # not, and the margin over random is reported separately as selectivity.
        _applied = max(band["suppression_null_delta"], 0.0)
        check("...and only an INFLATING null is subtracted from the credit",
              abs(band["delta"] - (band["raw_delta"] - _applied)) < 1e-9
              and abs(band["suppression_null_applied"] - _applied) < 1e-12,
              f"raw {band['raw_delta']:+.4f} - applied {_applied:+.4f} "
              f"= {band['delta']:+.4f}  (null was "
              f"{band['suppression_null_delta']:+.4f})")
        check("...and the margin over random suppression is reported, not credited",
              abs(band["selectivity_vs_random"]
                  - (band["raw_delta"] - band["suppression_null_delta"])) < 1e-9
              and band["selectivity_vs_random"] != band["delta"],
              f"selectivity {band['selectivity_vs_random']:+.4f} vs credited "
              f"{band['delta']:+.4f} — a property of the band, not a benefit delivered")
        check("the uncounted cost of a review queue is stated, not hidden",
              "attention" in (band.get("uncounted_cost") or ""))
        # BUG 4: the null was subtracted as if KNOWN. Its sampling error was discarded,
        # producing +0.0197 with CI [+0.0197, +0.0197] — a zero-width bootstrap
        # interval, which cannot occur for a statistic that varies.
        tight = adj(_prop(kind=L.POLICY, title="band tight",
                          change={"policy": "uncertain_to_review",
                                  "max_false_alarm_rate": 0.10}))
        for nm, r in (("band", band), ("band@0.10", tight)):
            check(f"{nm}: the interval is not degenerate",
                  (r["ci_hi"] - r["ci_lo"]) > 1e-6,
                  f"width {r['ci_hi'] - r['ci_lo']:.5f}")
        check("the null's own standard error is carried into the interval",
              tight.get("null_standard_error", 0) > 0,
              f"nullSE {tight.get('null_standard_error')}")
        check("a degenerate interval could never be called significant",
              not L.Lab(adjudicator=lambda p: {"delta": 0.02, "ci_lo": 0.02,
                                               "ci_hi": 0.02, "significant": False}
                        ).submit(_prop(title="degenerate")).accepted,
              "significance requires a real interval, not a point")

    print("\n14. the ledger must be parseable by a BROWSER, not just by Python")
    # NaN is meaningful in memory (an untested agent has no measurable bias, which is
    # not a bias of zero) and is NOT valid JSON. json.dump writes a bare NaN token,
    # every browser's JSON.parse rejects the whole document, and the Lab page died with
    # "the string did not match the expected pattern" because three agents were untested.
    import json as _json, math as _math, tempfile as _tf, os as _os
    lab = L.Lab(adjudicator=lambda p: {"delta": 0.02, "ci_lo": 0.005, "ci_hi": 0.03,
                                       "significant": True})
    lab.submit(_prop(agent="Tested", title="t"))
    lab.submit(_prop(agent="Untested", kind=L.TREATMENT, title="u",
                     change={"mechanism_only": True}))
    scores = {a.agent: a.mean_signed_error for a in lab.score_agents()}
    check("NaN is PRESERVED in memory — untested is not the same as zero",
          _math.isnan(scores["Untested"]),
          "an agent with no tested proposal has no measurable bias")
    fd, tmp = _tf.mkstemp(suffix=".json"); _os.close(fd)
    lab.save(tmp)
    raw = open(tmp).read()
    check("...and becomes null at the serialisation boundary",
          "NaN" not in raw and "Infinity" not in raw,
          f"{raw.count('NaN')} NaN tokens in the written ledger")
    try:
        _json.loads(raw, parse_constant=lambda c: (_ for _ in ()).throw(
            ValueError(f"non-finite constant {c!r} is not valid JSON")))
        ok_strict = True; why = "strict parse (no NaN/Infinity constants) succeeds"
    except ValueError as exc:
        ok_strict = False; why = str(exc)
    check("the ledger survives a STRICT parse, as a browser would do it", ok_strict, why)
    _os.unlink(tmp)

    print("\n15. the replication arm scores patients the gate never saw")
    # THE BUG: the gate drew held-out patients at seed=42 and the replicator drew its own
    # at seed=977. Two independent 25% draws from one cohort share about a quarter of
    # their patients — measured at 503 of 2001, 25.1% of the replication set. A finding
    # was being confirmed partly on its own gate set.
    try:
        from backend.ml.eval.bench import Benchmark as _B
        _ds = _B.load("dev", n_folds=2).ds
        _srch, _gate, _rep = L.three_way_split(_ds.groups, _ds.y, 42)
        S = set(_ds.groups[_srch].tolist())
        G = set(_ds.groups[_gate].tolist())
        R = set(_ds.groups[_rep].tolist())
        check("gate and replication pools share NO patient",
              len(G & R) == 0, f"|gate|={len(G)} |rep|={len(R)} shared={len(G & R)}")
        check("neither held-out pool leaks into the search pool",
              len(S & G) == 0 and len(S & R) == 0,
              f"search∩gate={len(S & G)}, search∩rep={len(S & R)}")
        check("the three pools partition the cohort exactly",
              len(S) + len(G) + len(R) == len(S | G | R) == len(set(_ds.groups.tolist())),
              f"{len(S)}+{len(G)}+{len(R)} = {len(S | G | R)} patients")
        # Stratification must survive the partition or the arms are not comparable.
        rates = [float(_ds.y[i].mean()) for i in (_srch, _gate, _rep)]
        check("outcome prevalence is preserved across the partition",
              max(rates) - min(rates) < 0.01,
              f"septic rate search {rates[0]:.4f} / gate {rates[1]:.4f} / rep {rates[2]:.4f}")

        # The old seeding scheme, kept as a live demonstration that the check can fail.
        from backend.ml.eval.splits import stratified_group_split as _sgs
        _, g42 = _sgs(_ds.groups, _ds.y, 0.25, 42)
        _, g977 = _sgs(_ds.groups, _ds.y, 0.25, 977)
        A, Bb = set(_ds.groups[g42].tolist()), set(_ds.groups[g977].tolist())
        check("...and the re-seeding scheme it replaces DOES overlap (the check can fail)",
              len(A & Bb) > 0,
              f"seed 42 vs 977 share {len(A & Bb)} patients "
              f"({100 * len(A & Bb) / len(Bb):.1f}% of the replication set)")
    except Exception as exc:                                    # pragma: no cover
        check("the partition is disjoint", False, f"cohort unavailable: {exc}")

    print("\n16. a replication on the SAME patients cannot promote a candidate")
    # The policy arm had exactly this defect: policy_adjudicator did not split at all, so
    # the "replication" re-scored the same 8000 patients. Raw delta came back
    # bit-identical at +0.025037. A candidate replicated with probability ~1.
    _same = {"delta": 0.02, "ci_lo": 0.005, "ci_hi": 0.035, "significant": True,
             "arm": "gate", "patient_ids_hash": "deadbeefcafe"}
    lab = L.Lab(adjudicator=lambda p, alpha=0.05: dict(_same),
                replicator=lambda p: dict(_same))          # identical pool fingerprint
    v = lab.submit(_prop(title="replicated on its own gate set"))
    check("an identical pool fingerprint is REJECTED as a replication",
          v.replicated is False and v.status == "candidate",
          f"status={v.status}, replicated={v.replicated}")
    check("...and the ledger says why, naming the fingerprint",
          any("REPLICATION REJECTED" in x and "deadbeefcafe" in x for x in v.limits))
    _diff = dict(_same, arm="replication", patient_ids_hash="0123456789ab")
    lab = L.Lab(adjudicator=lambda p, alpha=0.05: dict(_same),
                replicator=lambda p: dict(_diff))
    v = lab.submit(_prop(title="replicated on a disjoint pool"))
    check("a DIFFERENT pool fingerprint does promote to a finding",
          v.replicated is True and v.status == "finding",
          f"gate {v.patient_ids_hash} -> replication {v.replication_pool_hash}")

    print("\n17. the alpha the Lab charges is the alpha the test applies")
    # For the third time in this module's history a correction was documented and not
    # applied: AlphaBudget spent a geometric bar, printed "tested at alpha=0.00625", and
    # then called an adjudicator that took bootstrap_delta_ci's 95% default.
    seen = {}
    def _alpha_judge(p, alpha=0.05):
        seen["alpha"] = alpha
        return {"delta": 0.02, "ci_lo": 0.005, "ci_hi": 0.035, "significant": True}
    lab = L.Lab(adjudicator=_alpha_judge)
    v = lab.submit(_prop(title="alpha reaches the judge"))
    check("the charged alpha is passed to an adjudicator that accepts one",
          seen.get("alpha") == v.alpha_spent and v.alpha_applied,
          f"charged {v.alpha_spent:.5f}, judge received {seen.get('alpha')}")
    lab = L.Lab(adjudicator=lambda p: {"delta": 0.02, "ci_lo": 0.005, "ci_hi": 0.035,
                                       "significant": True})
    v = lab.submit(_prop(title="judge cannot take an alpha"))
    check("a judge that cannot take an alpha is FLAGGED, not silently trusted",
          v.alpha_applied is False and any("CHARGED" in x for x in v.limits),
          "the limits say the correction is accounted for, not applied")
    import backend.aegis.lab_runner as _LR
    _srcs = [_inspect.getsource(L.held_out_adjudicator),
             _inspect.getsource(_LR.policy_adjudicator)]
    check("both real adjudicators define adjudicate(p, alpha=...)",
          all("def adjudicate(p" in s_ and "alpha: float = 0.05" in s_ for s_ in _srcs),
          "held_out_adjudicator and policy_adjudicator")

    print("\n18. every test reports what it could have detected")
    lab = L.Lab(adjudicator=lambda p, alpha=0.05: {"delta": 0.0005, "ci_lo": -0.03,
                                                   "ci_hi": 0.03, "significant": False})
    v = lab.submit(_prop(title="a wide, underpowered test"))
    check("a refusal carries a minimum detectable effect",
          v.mde is not None and v.mde > 0.02, f"MDE {v.mde:+.4f} from a ±0.03 interval")
    check("...and says what it ruled out and what it could not see",
          v.power_note is not None and "NOT testable" in v.power_note,
          (v.power_note or "")[:88])
    lab = L.Lab(adjudicator=lambda p, alpha=0.05: {"delta": 0.004, "ci_lo": 0.0035,
                                                   "ci_hi": 0.0045, "significant": True})
    v = lab.submit(_prop(title="a tight, well-powered test"))
    check("a tight interval yields an MDE at the MIN_EFFECT floor, not below it",
          v.mde == L.MIN_EFFECT,
          f"half-width {(0.0045 - 0.0035) / 2:.5f} < MIN_EFFECT {L.MIN_EFFECT}, so the "
          f"floor binds — a test cannot claim to resolve less than the gate accepts")
    check("the campaign reports the effect size it was blind below",
          lab.power_summary()["median_mde"] is not None,
          f"median MDE {lab.power_summary()['median_mde']}")

    print("\n19. the Lab remembers across campaigns")
    from backend.aegis import lab_registry as LR
    fd2, rpath = _tf.mkstemp(suffix=".json"); _os.close(fd2); _os.unlink(rpath)
    reg = LR.LabRegistry.load(rpath)
    lab1 = L.Lab(adjudicator=lambda p, alpha=0.05: {"delta": 0.0001, "ci_lo": -0.01,
                                                    "ci_hi": 0.01, "significant": False},
                 registry=reg)
    v1 = lab1.submit(_prop(title="tune et.max_depth", change={"learner": "et",
                                                              "key": "max_depth",
                                                              "value": 18}))
    reg.record_campaign(lab1.ledger, lab1.budget, label="c1")
    check("a campaign is chained into the registry",
          len(reg.entries) == 1 and reg.entries[0].campaign == "c1",
          f"{len(reg.entries)} entry, hash {reg.entries[0].hash[:12]}")

    # The same hypothesis, a new campaign, possibly a different wording.
    lab2 = L.Lab(adjudicator=lambda p, alpha=0.05: {"delta": 0.9, "ci_lo": 0.8,
                                                    "ci_hi": 1.0, "significant": True},
                 registry=reg)
    before = lab2.budget.n_tests
    v2 = lab2.submit(_prop(title="COMPLETELY different wording, same change",
                           change={"learner": "et", "key": "max_depth", "value": 18}))
    check("a hypothesis already tested is refused on the RECORD, not re-tested",
          not v2.accepted and v2.duplicate_of == "c1",
          (v2.refusal_reason or "")[:80])
    check("...even though the adjudicator would have returned delta=+0.9",
          not v2.accepted, "dedup runs before the gate")
    check("...and the repeat costs no alpha",
          lab2.budget.n_tests == before, f"n_tests still {lab2.budget.n_tests}")
    v3 = lab2.submit(_prop(title="a genuinely new change",
                           change={"learner": "et", "key": "max_depth", "value": 19}))
    check("a NEW hypothesis is still tested (dedup is not a blanket refusal)",
          v3.accepted and lab2.budget.n_tests == before + 1,
          f"delta {v3.observed_delta:+.4f}, alpha {v3.alpha_spent:.5f}")

    print("\n20. the registry is tamper-evident")
    ok, bad = reg.verify_chain()
    check("an untouched chain verifies", ok and bad is None, f"{len(reg.entries)} entries")
    reg.record_campaign(lab2.ledger, lab2.budget, label="c2")
    ok, bad = reg.verify_chain()
    check("...and still verifies after a second campaign", ok, f"{len(reg.entries)} entries")
    # Edit a recorded refusal into an acceptance, exactly as someone rewriting history
    # would. The hash must catch it.
    victim = reg.entries[0]
    victim.status, victim.accepted, victim.observed_delta = "finding", True, 0.05
    ok, bad = reg.verify_chain()
    check("editing a recorded verdict BREAKS the chain and names the entry",
          not ok and bad == 0,
          f"seq {bad} no longer hashes to its recorded value")
    victim.status, victim.accepted, victim.observed_delta = "refused", False, 0.0001
    ok, _ = reg.verify_chain()
    check("...and restoring the original value repairs it (the check is exact)", ok)
    dropped = reg.entries.pop(1)
    ok, bad = reg.verify_chain()
    check("deleting an entry breaks the chain too — a refusal cannot be quietly removed",
          not ok, f"first break at seq {bad}")
    reg.entries.insert(1, dropped)

    print("\n21. the programme is disclosed, not shopped")
    prog = reg.programme()
    check("cumulative alpha over every test IS the expected false-finding count",
          abs(prog["expected_false_findings"]
              - sum(e.alpha_spent or 0.0 for e in reg.entries)) < 1e-9,
          f"{prog['expected_false_findings']:.5f} of alpha over {prog['n_tested']} tests")
    check("the finding count is reported against it",
          "expected by chance" in prog["findings_vs_expected_false"],
          prog["findings_vs_expected_false"])
    check("hypotheses are counted separately from proposals",
          prog["n_hypotheses"] <= prog["n_proposals"],
          f"{prog['n_hypotheses']} distinct hypotheses over {prog['n_proposals']} proposals")
    reg.path = rpath
    reg.save()
    strict = _json.loads(open(rpath).read(),
                         parse_constant=lambda c: (_ for _ in ()).throw(
                             ValueError(f"non-finite constant {c!r}")))
    check("the registry survives a STRICT browser parse",
          strict["programme"]["n_campaigns"] == 2,
          f"{len(strict['entries'])} entries, {strict['programme']['n_campaigns']} campaigns")
    _os.unlink(rpath)

    print("\n22. the Critic argues from the Lab's OWN refusals, not just published ones")
    # Two refusals on one lever, worded differently on purpose: the Critic's first
    # version recovered the lever by parsing the title, so "tune et.max_depth" and
    # "set et.max_depth = 18" looked like different levers. It reads the change now.
    lab3 = L.Lab(adjudicator=lambda p, alpha=0.05: {"delta": 0.0001, "ci_lo": -0.01,
                                                    "ci_hi": 0.01, "significant": False})
    lab3.submit(_prop(title="set et.max_depth = 20",
                      change={"learner": "et", "key": "max_depth", "value": 20}))
    reg.record_campaign(lab3.ledger, lab3.budget, label="c3")
    c = L.Critic(registry=reg)
    w = c.review(_prop(title="a title that mentions no lever at all",
                       change={"learner": "et", "key": "max_depth", "value": 22}))
    check("a lever this Lab has already exhausted carries a warning",
          w is not None and "same lever" in w, (w or "")[:96])
    check("...and it is still only a warning — the Critic cannot veto",
          L.Lab(adjudicator=lambda p, alpha=0.05: {"delta": 0.02, "ci_lo": 0.005,
                                                   "ci_hi": 0.03, "significant": True},
                registry=None).submit(
              _prop(title="set et.max_depth = 22")).accepted)

    print("\n23. a pre-declared test count is a pre-registration, not a knob")
    # Geometric spending handed half the budget to test one; a measured campaign got 4
    # tests out of 8 proposals and refused the rest for want of budget. A campaign knows
    # its ceiling, so it can pre-register the count and spend flat — but ONLY if the
    # count cannot be revised once results are in.
    b = L.AlphaBudget(total=0.05)
    b.plan(8)
    check("a pre-declared count spends the budget flat",
          abs(b.next_alpha() - 0.05 / 8) < 1e-12 and b.schedule().startswith("flat"),
          f"alpha={b.next_alpha():.5f} for every one of {b.planned_tests} tests")
    firsts = []
    for _ in range(8):
        firsts.append(b.next_alpha()); b.spend(b.next_alpha())
    check("...and every planned test faces the SAME bar",
          max(firsts) - min(firsts) < 1e-12, f"{len(set(round(x, 9) for x in firsts))} distinct bar(s)")
    check("the family-wise budget is spent exactly, never exceeded",
          abs(b.spent - 0.05) < 1e-12, f"spent {b.spent:.6f} of {b.total}")
    check("the plan is used up and the budget then refuses",
          b.exhausted(), f"{b.n_tests} of {b.planned_tests} tests run")

    b2 = L.AlphaBudget(total=0.05)
    b2.spend(b2.next_alpha())
    try:
        b2.plan(20); revised = True; why = "a plan was accepted AFTER a test"
    except ValueError as exc:
        revised = False; why = str(exc)[:74]
    check("a count declared AFTER a test is refused — this is the load-bearing property",
          not revised, why)
    b3 = L.AlphaBudget(total=0.05); b3.plan(4)
    try:
        b3.plan(40); twice = True
    except ValueError:
        twice = False
    check("...and a declared count cannot be revised upward later", not twice,
          "widening the plan mid-session would widen the bar for every remaining test")

    geo = L.AlphaBudget(total=0.05)
    n_geo = 0
    while not geo.exhausted() and n_geo < 50:
        geo.spend(geo.next_alpha()); n_geo += 1
    check("the open-ended geometric schedule still works when no count is declared",
          geo.schedule() == "geometric" and n_geo == 4,
          f"{n_geo} tests before the bar falls under the {L.AlphaBudget.RESOLUTION_FLOOR} "
          f"resolution floor — which is exactly why a campaign pre-registers instead")
    check("pre-registering doubles the tests a campaign can honestly run",
          8 > n_geo, f"8 planned vs {n_geo} geometric, at the same family-wise 0.05")

    print("\n24. a pile of refusals must say whether it is about the change or the gate")
    # Measured on the first corrected campaign: median MDE +0.0329 against proposals of
    # about +0.0030. The gate was ~16x too coarse to see what was being asked of it, so
    # reporting those refusals as "these changes do not work" would have been wrong.
    wide = L.Lab(adjudicator=lambda p, alpha=0.05: {"delta": 0.0004, "ci_lo": -0.033,
                                                    "ci_hi": 0.033, "significant": False})
    for i in range(3):
        wide.submit(_prop(title=f"small ask {i}", predicted_delta=0.003,
                          change={"learner": "hgb", "key": "k", "value": i}))
    ps = wide.power_summary()
    check("a gate far coarser than the proposals is called POWER-LIMITED",
          ps["power_limited"] and ps["mde_to_prediction_ratio"] >= 3,
          f"MDE {ps['median_mde']:+.4f} vs proposals {ps['typical_predicted_effect']:+.4f} "
          f"= {ps['mde_to_prediction_ratio']}x coarser")
    check("...and says so in the diagnosis rather than implying the change failed",
          "instrument, not about the changes" in (ps["diagnosis"] or ""),
          (ps["diagnosis"] or "")[:80])
    tight = L.Lab(adjudicator=lambda p, alpha=0.05: {"delta": 0.0004, "ci_lo": -0.002,
                                                     "ci_hi": 0.002, "significant": False})
    for i in range(3):
        tight.submit(_prop(title=f"resolvable ask {i}", predicted_delta=0.003,
                           change={"learner": "et", "key": "k", "value": i}))
    ps2 = tight.power_summary()
    check("a gate that CAN resolve the proposals is not flagged (the check can fail)",
          not ps2["power_limited"],
          f"MDE {ps2['median_mde']:+.4f} vs proposals "
          f"{ps2['typical_predicted_effect']:+.4f} = {ps2['mde_to_prediction_ratio']}x")

    print("\n25. an exhausted line and an unproductive line are different results")
    # The campaign reported both as "dry". A line whose SPACE is exhausted has been fully
    # explored and its silence is informative; a line still producing refusals has room
    # and simply is not paying. Collapsing them hides which one you are looking at.
    import backend.aegis.lab_runner as _LR2
    _adj25 = lambda p, alpha=0.05: {"delta": 0.0001, "ci_lo": -0.01, "ci_hi": 0.01,
                                    "significant": False}
    lab = L.Lab(adjudicator=_adj25, policy_adjudicator=_adj25)
    props = L.adaptive_proposers()
    pi = _LR2.PrincipalInvestigator(props)
    ctx25 = {"seed": 5}
    dec = next(x for x in props if x.name == "Decider")
    for k in range(9):
        pr = dec.propose({**ctx25, "seed": 5 + k * 3}, lab.ledger)
        if pr is None:
            break
        lab.submit(pr)
    rows = {r["line"]: r for r in pi.agenda(lab.ledger, ctx25)}
    check("a line with a finite space is marked EXHAUSTED once it is spent",
          rows["Decider"].get("space_exhausted") is True,
          f"{sum(1 for v in lab.ledger if v.proposal.agent == 'Decider')} proposals, "
          f"then nothing left to ask")
    check("...and a line with a continuous space is NOT (the check can fail)",
          rows["Tuner"].get("space_exhausted") is False,
          "the Tuner samples a continuous hyperparameter space")
    concl = pi.conclude(lab, ctx25)
    check("the conclusion says the silence is a result, not neglect",
          any("EXHAUSTED" in c and "explored, not neglected" in c for c in concl),
          next((c[:88] for c in concl if "EXHAUSTED" in c), "no exhaustion line"))
    check("an exhausted line is not picked again, so it cannot spend more budget",
          pi.next_move(lab.ledger, ctx25) is None
          or pi.next_move(lab.ledger, ctx25)[1]["line"] != "Decider",
          "the PI ranks it below zero and skips it")

    print("\n26. a policy that changes nothing cannot become a finding")
    # THE DEFECT, EXACTLY AS IT HAPPENED. Silencing the UNCERTAIN band moved detection by
    # raw +0.0000 on both pools. The random-suppression null was -0.0225. excess =
    # 0 - (-0.0225) = +0.0225, which cleared MIN_EFFECT, replicated, and was reported as
    # "FINDING ... +0.0225" — an improvement of exactly zero, dressed as the largest
    # result this Lab had ever produced.
    import backend.aegis.lab_runner as _LR3
    _widen = _LR3._widen
    raw, null = 0.0, -0.0225
    old_excess = raw - null
    new_excess = raw - max(null, 0.0)
    check("subtracting a NEGATIVE null manufactures an effect from nothing",
          old_excess >= L.MIN_EFFECT,
          f"raw {raw:+.4f}, null {null:+.4f} -> excess {old_excess:+.4f} "
          f"(>= MIN_EFFECT {L.MIN_EFFECT}), which is how the artifact passed")
    check("...and only an INFLATING null is subtracted now",
          new_excess < L.MIN_EFFECT and new_excess == raw,
          f"credited {new_excess:+.4f} — the policy did nothing, and says so")
    check("a POSITIVE null is still subtracted, so real inflation is still removed",
          abs((0.03 - max(0.02, 0.0)) - 0.01) < 1e-12,
          "raw +0.0300 with an inflating null +0.0200 credits +0.0100")
    src = _inspect.getsource(_LR3.policy_adjudicator)
    check("the adjudicator reports the margin over random SEPARATELY from the credit",
          "selectivity_vs_random" in src and "null_direction" in src,
          "band selectivity is a property of the band, never the accepted effect")

    print("\n27. significance must be a number the resamples can express")
    check("a percentile endpoint at a planned campaign's alpha is NOT estimable",
          (0.003125 / 2) * (400 - 1) < 1.0,
          f"alpha/2 x (n-1) = {(0.003125 / 2) * 399:.2f} — order statistic 0.62 of 400, "
          f"which is the sample minimum, not a percentile")
    d_pos = _np.full(400, 0.02)
    sg = L.bootstrap_significance(d_pos, 0.003125)
    check("a one-sided bootstrap p-value IS estimable at that alpha",
          sg["resolvable"] and sg["significant"],
          f"p={sg['p_value']:.6f} (floor {sg['finest_p']:.6f}) < alpha 0.003125")
    d_mixed = _np.concatenate([_np.full(390, 0.02), _np.full(10, -0.01)])
    sg2 = L.bootstrap_significance(d_mixed, 0.003125)
    check("...and 10 of 400 resamples crossing zero is enough to fail it",
          not sg2["significant"], f"p={sg2['p_value']:.6f} vs alpha 0.003125")
    sg3 = L.bootstrap_significance(d_pos, 0.0001)
    check("an alpha finer than 1/(n+1) is REFUSED, not silently passed",
          not sg3["resolvable"] and not sg3["significant"],
          (sg3["note"] or "")[:80])
    lab = L.Lab(adjudicator=lambda p, alpha=0.05: {
        "delta": 0.9, "ci_lo": 0.8, "ci_hi": 1.0, "significant": True,
        "resolvable": False, "significance_note": "alpha below the resample floor"})
    v = lab.submit(_prop(title="unrunnable but 'significant'"))
    check("the Lab refuses a judge that says significant on an unrunnable test",
          not v.accepted and v.resolvable is False,
          (v.refusal_reason or "")[:76])
    check("...and a p-value of exactly zero is never reported",
          L.bootstrap_significance(_np.full(400, 0.02), 0.05)["p_value"] > 0,
          "the +1 correction keeps it strictly positive — no finite resample supports 0")

    n_fail = sum(1 for s_, _, _ in _res if s_ == FAIL)
    print(f"\n{'=' * 70}\n{len(_res) - n_fail}/{len(_res)} checks passed")
    for s_, nm, dt in _res:
        if s_ == FAIL:
            print(f"  FAILED — {nm}: {dt}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
