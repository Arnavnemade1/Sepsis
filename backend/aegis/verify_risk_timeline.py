"""Adversarial audit of the risk timeline.

    PYTHONPATH=. python -m backend.aegis.verify_risk_timeline

Every check here corresponds to a defect that was actually shipped on the old chart, so
each one is a regression test rather than a hypothetical. The old bugs, in order: a
threshold on the wrong scale, a forecast that did not start where the history ended, a
"confidence" that was one minus a band width, and a trajectory badge computed from a
different series than the projection printed beside it.
"""
from __future__ import annotations

import sys
from typing import List, Tuple

from backend.aegis import risk_timeline as RT

PASS, FAIL = "PASS", "FAIL"
_res: List[Tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    _res.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f" — {detail}" if detail else ""), flush=True)
    return ok


def main() -> int:
    print("1. the forecast starts where the history ends (the visible jump)")
    hist = [0.004, 0.006, 0.009, 0.013, 0.018, 0.024]
    tl = RT.build(hist, horizon=8)
    jump = abs(tl.forecast[0].probability - (hist[-1] + tl.trend_per_hour))
    check("first forecast point is anchored at now + one step of trend",
          jump < 1e-6,
          f"forecast[0]={tl.forecast[0].probability:.5f}, now={hist[-1]:.5f}, "
          f"trend={tl.trend_per_hour:+.5f}")
    gap = abs(tl.forecast[0].probability - hist[-1])
    check("...and the discontinuity at `now` is bounded by one step, not arbitrary",
          gap <= abs(tl.trend_per_hour) + 1e-9, f"gap={gap:.5f}")

    print("\n2. the alarm line is the SHIPPED operating point, not a hardcoded 0.70")
    check("alarm threshold is read from the operating point artifact",
          tl.alarm_threshold is not None and tl.alarm_threshold < 0.5,
          f"alarm={tl.alarm_threshold} (the old chart drew 0.70 on a different scale)")
    check("provenance names where the alarm came from",
          "operating_point" in (tl.provenance.get("alarm_source") or ""),
          tl.provenance.get("alarm_source"))
    check("provenance names the detector the series belongs to",
          bool(tl.provenance.get("detector")), str(tl.provenance.get("detector")))

    print("\n3. no fabricated confidence")
    d = tl.as_dict()
    check("the payload exposes NO 'confidence' field",
          "confidence" not in d,
          "the old chart showed `1 - mean band width` as 'forecast confidence 82%'")
    check("the limits say breach_in_hours is not a probability",
          any("NOT a probability" in x for x in tl.limits))

    print("\n4. the badge and the projection cannot contradict each other")
    rising = RT.build([0.004, 0.008, 0.013, 0.019, 0.026, 0.034], horizon=8)
    falling = RT.build([0.090, 0.070, 0.055, 0.040, 0.030, 0.022], horizon=8)
    check("a rising series reads deteriorating", rising.trajectory == "deteriorating",
          f"{rising.trajectory}, breach_in={rising.breach_in_hours}")
    check("a falling series reads improving", falling.trajectory == "improving",
          f"{falling.trajectory}, breach_in={falling.breach_in_hours}")
    check("an improving series NEVER projects a future breach",
          falling.breach_in_hours in (None, 0.0),
          "this is the 'IMPROVING - projected critical in ~1h' bug; build() asserts it")
    flat = RT.build([0.02, 0.021, 0.0205, 0.0202, 0.0203, 0.0201], horizon=8)
    check("a flat series reads stable", flat.trajectory == "stable", flat.trajectory)

    print("\n5. the conformal UNCERTAIN band is drawn, and the alarm sits inside it")
    if tl.uncertain_lo is not None:
        check("uncertain band is present and ordered",
              tl.uncertain_lo < tl.uncertain_hi,
              f"[{tl.uncertain_lo:.4f}, {tl.uncertain_hi:.4f}]")
        inside = tl.uncertain_lo <= (tl.alarm_threshold or 0) <= tl.uncertain_hi
        check("the alarm threshold sits INSIDE the band — shown, not hidden",
              inside,
              f"alarm {tl.alarm_threshold} in [{tl.uncertain_lo:.4f},{tl.uncertain_hi:.4f}]"
              if inside else "alarm is outside the band on this artifact")
    else:
        check("uncertain band is present", False, "conformal artifact missing")

    print("\n6. it refuses rather than extrapolating from noise")
    thin = RT.build([0.01, 0.02], horizon=8)
    check("fewer than MIN_HISTORY points -> refused, no forecast drawn",
          thin.refused is not None and not thin.forecast, thin.refused)
    check("...and the trajectory says so instead of guessing",
          thin.trajectory == "insufficient history", thin.trajectory)

    print("\n7. the forecast band widens with the horizon and stays ordered")
    ok_order = all(pt.lo <= pt.probability <= pt.hi for pt in rising.forecast)
    widens = (rising.forecast[-1].hi - rising.forecast[-1].lo) >= \
             (rising.forecast[0].hi - rising.forecast[0].lo)
    check("lo <= mean <= hi at every horizon", ok_order)
    check("band widens with horizon", widens,
          f"{rising.forecast[0].hi-rising.forecast[0].lo:.5f} -> "
          f"{rising.forecast[-1].hi-rising.forecast[-1].lo:.5f}")
    check("probabilities stay in [0,1]",
          all(0.0 <= pt.probability <= 1.0 for pt in rising.forecast))

    n_fail = sum(1 for s, _, _ in _res if s == FAIL)
    print(f"\n{'=' * 68}\n{len(_res) - n_fail}/{len(_res)} checks passed")
    for s, nm, dt in _res:
        if s == FAIL:
            print(f"  FAILED — {nm}: {dt}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
