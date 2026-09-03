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

import sys
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
        check("...and the credited effect is the EXCESS over that null, not the raw delta",
              abs(band["delta"] - (band["raw_delta"] - band["suppression_null_delta"])) < 1e-9,
              f"raw {band['raw_delta']:+.4f} - null {band['suppression_null_delta']:+.4f} "
              f"= {band['delta']:+.4f}")
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

    n_fail = sum(1 for s_, _, _ in _res if s_ == FAIL)
    print(f"\n{'=' * 70}\n{len(_res) - n_fail}/{len(_res)} checks passed")
    for s_, nm, dt in _res:
        if s_ == FAIL:
            print(f"  FAILED — {nm}: {dt}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
