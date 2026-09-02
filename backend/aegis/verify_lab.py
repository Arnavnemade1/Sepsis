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
    lab = L.Lab(adjudicator=lambda p: {"delta": 0.001, "ci_lo": -0.001, "ci_hi": 0.003,
                                       "significant": False})
    for i in range(4):
        lab.submit(_prop(agent="Optimist", predicted_delta=0.05,
                         predicted_direction="improve", title=f"change {i}"))
    for i in range(4):
        lab.submit(_prop(agent="Realist", predicted_delta=0.001,
                         predicted_direction="improve", title=f"tweak {i}"))
    scores = {s.agent: s for s in lab.score_agents()}
    check("an overconfident agent shows a large positive signed error",
          scores["Optimist"].mean_signed_error > scores["Realist"].mean_signed_error,
          f"Optimist {scores['Optimist'].mean_signed_error:+.4f} vs "
          f"Realist {scores['Realist'].mean_signed_error:+.4f} (>0 = overclaiming)")
    check("both are recorded with proposal counts",
          scores["Optimist"].n_proposals == 4 and scores["Realist"].n_proposals == 4)
    s = lab.summary()
    check("the summary states that refusal is the expected outcome",
          "gate working" in s["note"], f"acceptance rate {s['acceptance_rate']:.0%}")

    print("\n7. the default proposers hold DIFFERENT priors")
    props = L.default_proposers()
    priors = {p.prior for p in props}
    check("three proposers with three distinct priors", len(props) == 3 and len(priors) == 3,
          "; ".join(sorted(priors)))
    made = [p.generate({"seed": 1}) for p in props]
    check("each produces a well-formed proposal with a registered prediction",
          all(m.predicted_direction in ("improve", "no_effect", "worsen")
              and isinstance(m.predicted_delta, float) for m in made))
    check("...and they do not all propose the same kind of change",
          len({m.kind for m in made}) > 1, str(sorted({m.kind for m in made})))

    n_fail = sum(1 for s_, _, _ in _res if s_ == FAIL)
    print(f"\n{'=' * 70}\n{len(_res) - n_fail}/{len(_res)} checks passed")
    for s_, nm, dt in _res:
        if s_ == FAIL:
            print(f"  FAILED — {nm}: {dt}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
