"""Adversarial audit of the clinical disposition layer.

    PYTHONPATH=. python -m backend.aegis.verify_disposition

The safety property this layer must not violate is an ORDERING one: uncertainty may add
a demand on a person, and may never remove one. A layer that could silence a HIGH
escalation because the model felt unsure would be strictly worse than having no
uncertainty layer at all, and it would fail in exactly the direction that hurts patients.

So every path that could downgrade is planted here and required not to.
"""
from __future__ import annotations

import sys
from typing import List, Tuple

from backend.aegis import disposition as D

PASS, FAIL = "PASS", "FAIL"
_res: List[Tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _res.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


UNCERTAIN = {"state": "uncertain", "abstain": True}
SEPTIC = {"state": "septic", "abstain": False}
NOT_SEPTIC = {"state": "not_septic", "abstain": False}
CONFLICTING = {"state": "conflicting", "abstain": True}


def main() -> int:
    print("1. abstention may never DOWNGRADE an escalation")
    d = D.decide("HIGH", 0.9, certainty=UNCERTAIN)
    check("HIGH + model abstains still ALARMs", d.action == D.ALARM,
          f"action={d.action}, abstained={d.abstained}")
    check("...and the disposition says why rather than hiding it",
          any("never downgrade" in b or "may never downgrade" in b for b in d.because),
          "the reason is carried in `because`")

    d = D.decide("LOW", 0.02, certainty=UNCERTAIN, override="qSOFA>=2")
    check("hard rule + model abstains still ALARMs", d.action == D.ALARM,
          f"action={d.action}")
    check("...and abstained is FALSE on a rule-driven escalation",
          d.abstained is False,
          "an abstention badge on a decision the model did not make would misdescribe it")

    print("\n1b. a SIRS CRITERION is not a hard override (regression)")
    # Found live: the REST route passed the first entry of `triggered_rules` as the
    # override, and that list carries descriptive SIRS criteria like
    # "SIRS: Tachycardia (98.0 bpm)". Every mildly abnormal patient therefore ALARMed
    # and REVIEW became unreachable. core.py's definition is qSOFA >= 2 or shock.
    d = D.decide("MEDIUM", 0.0241, certainty=UNCERTAIN, override=None)
    check("uncertain + only SIRS criteria present -> REVIEW, not ALARM",
          d.action == D.REVIEW,
          f"action={d.action} (a SIRS criterion must never reach `override`)")
    d = D.decide("MEDIUM", 0.0241, certainty=UNCERTAIN, override="qSOFA>=2")
    check("uncertain + a REAL hard override -> ALARM", d.action == D.ALARM,
          f"action={d.action}")

    print("\n2. uncertainty ADDS demand where the model alone would have decided")
    low_conf = D.decide("MEDIUM", 0.04, certainty=NOT_SEPTIC)
    low_unc = D.decide("MEDIUM", 0.04, certainty=UNCERTAIN)
    check("MEDIUM + uncertain escalates above MEDIUM + confident-negative",
          D.escalates_over(low_unc, low_conf),
          f"{low_conf.action} -> {low_unc.action}")
    check("uncertain routes to a PERSON, not to the alarm and not to silence",
          low_unc.action == D.REVIEW and "review" in low_unc.route.lower(),
          f"route={low_unc.route}")

    print("\n3. the empty set is a data-quality signal, not a clinical one")
    d = D.decide("MEDIUM", 0.04, certainty=CONFLICTING)
    check("conflicting routes to the feed, not the bedside",
          d.action == D.REVIEW and "quality" in d.route.lower(), f"route={d.route}")

    print("\n4. absent conformal claim is reported, not defaulted to confidence")
    d = D.decide("MEDIUM", 0.04, certainty=None)
    check("no certainty -> falls back to the tier and says so",
          d.abstained is False and any("no conformal claim" in b for b in d.because),
          f"action={d.action}")
    d2 = D.decide("MEDIUM", 0.04,
                  certainty={"unavailable": True, "state": "no_claim",
                             "reason": "cohort bed score, not the calibrated detector"})
    check("explicit unavailable is handled the same way",
          d2.action == D.MONITOR and any("no conformal claim" in b for b in d2.because))

    print("\n5. confident states behave")
    check("confident septic ALARMs", D.decide("MEDIUM", 0.06, certainty=SEPTIC).action == D.ALARM)
    check("confident negative at LOW does nothing",
          D.decide("LOW", 0.001, certainty=NOT_SEPTIC).action == D.NO_ACTION)
    check("confident negative at MEDIUM is still watched, not dismissed",
          D.decide("MEDIUM", 0.04, certainty=NOT_SEPTIC).action == D.MONITOR)

    print("\n6. every disposition carries its cost and its limits")
    d = D.decide("MEDIUM", 0.04, certainty=UNCERTAIN)
    check("limits are attached and mention the cross-site collapse",
          len(d.limits) >= 3 and any("collaps" in x for x in d.limits),
          f"{len(d.limits)} limits")
    check("REVIEW is priced as spending attention, not sold as free",
          any("attention" in x for x in d.limits))
    if d.tradeoff and d.tradeoff.get("policies"):
        pol = d.tradeoff["policies"]
        cut = d.tradeoff.get("false_alarm_cut_if_silenced")
        check("the measured conversion trade-off is attached",
              "UNCERTAIN silenced" in pol and "UNCERTAIN escalated" in pol,
              f"false-alarm cut if silenced: {cut}")
        esc = pol.get("UNCERTAIN escalated", {}).get("false_alarm")
        check("...and it shows why escalating everything uncertain is not an option",
              esc is not None and esc > 0.5, f"escalated false-alarm rate {esc}")
    else:
        check("the measured conversion trade-off is attached", False,
              "conformal artefact missing or detector mismatch — run `make conformal`")

    n_fail = sum(1 for s, _, _ in _res if s == FAIL)
    print(f"\n{'=' * 66}\n{len(_res) - n_fail}/{len(_res)} checks passed")
    for s, nm, dt in _res:
        if s == FAIL:
            print(f"  FAILED — {nm}: {dt}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
