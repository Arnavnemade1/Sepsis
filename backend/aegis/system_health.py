"""System health — how well this OS actually detects sepsis, and where it does not.

Every other surface in AegisOS shows the system working. This one is where it is
allowed to look bad, because a validation page that can only report good news is
not a validation page.

WHAT IT READS
-------------
The cross-hospital evaluation artefact (``artifacts/rigorous_eval.json``): train
on one hospital, test on the other, both directions, bootstrap confidence
intervals. That protocol is the point — a model tuned and tested on one site
flatters itself, and the gap between internal and external performance is the
single biggest reason clinical ML fails when deployed. Nothing here is computed
live from the twelve-patient demo cohort; twelve stays cannot support an AUROC.

STRENGTHS AND GAPS ARE DERIVED, NOT WRITTEN
-------------------------------------------
Same discipline as the treatment layer. A gap appears because a number crossed a
threshold, and the number travels with it: "false-alarm rate 34% — one in three
stable patients still triggers an alarm" is a claim about this evaluation that
changes when the evaluation changes. The alternative — a hand-written limitations
list — says the same thing forever and is therefore worth nothing.

The thresholds are deliberately unkind. AUROC below 0.75 is called out as modest
discrimination even though this model is at 0.72, because it IS modest, and a
health page that grades on a curve is the thing being guarded against.

    python -m backend.aegis.system_health
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_ART = os.path.join(_ROOT, "artifacts")

# Production model. The stacked ensemble is what actually serves; gbm is kept as
# the honest baseline it has to beat.
PRODUCTION = "stacked"
BASELINE = "gbm"

# Computed once per report and read by the findings pass, which needs it but is
# called before the payload is assembled.
_gen_cache: Dict[str, Any] = {}


def _load(name: str) -> Optional[Dict[str, Any]]:
    try:
        with open(os.path.join(_ART, name)) as fh:
            return json.load(fh)
    except Exception:
        return None


def _mean(vals: List[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _pooled(rig: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Average a model's metrics across both train/test directions.

    Both directions are reported individually too — pooling is for the headline,
    and a single number that hides a large asymmetry between sites would be the
    wrong headline, so the spread is carried alongside it.
    """
    out: Dict[str, Any] = {"directions": []}
    aurocs, eces, utils, dets, fars, leads = [], [], [], [], [], []
    lo_s, hi_s = [], []
    for key, blk in (rig.get("directions") or {}).items():
        m = (blk.get("models") or {}).get(model)
        if not m:
            continue
        ew = m.get("early_warning") or {}
        # The artefact stores the bootstrap interval as a [lo, hi] pair; accept a
        # dict too so a future format change degrades rather than crashes.
        ci = m.get("auroc_ci")
        lo = hi = None
        if isinstance(ci, (list, tuple)) and len(ci) == 2:
            lo, hi = ci
        elif isinstance(ci, dict):
            lo, hi = ci.get("lo", ci.get("low")), ci.get("hi", ci.get("high"))
        aurocs.append(float(m["auroc"]))
        eces.append(float(m.get("ece", 0)))
        utils.append(float(m.get("normalized_utility", 0)))
        dets.append(float(ew.get("detection_rate", 0)))
        fars.append(float(ew.get("false_alarm_rate", 0)))
        leads.append(float(ew.get("median_lead_time_hours", 0)))
        if lo is not None:
            lo_s.append(float(lo))
        if hi is not None:
            hi_s.append(float(hi))
        out["directions"].append({
            "key": key, "train": blk.get("train"), "test": blk.get("test"),
            "auroc": round(float(m["auroc"]), 4),
            "detection_rate": round(float(ew.get("detection_rate", 0)), 4),
            "false_alarm_rate": round(float(ew.get("false_alarm_rate", 0)), 4),
            "median_lead_hours": ew.get("median_lead_time_hours"),
            "septic": ew.get("septic_patients"), "n": ew.get("nonseptic_patients"),
        })
    if not aurocs:
        return {}
    out.update({
        "auroc": round(_mean(aurocs), 4),
        "auroc_lo": round(_mean(lo_s), 4) if lo_s else None,
        "auroc_hi": round(_mean(hi_s), 4) if hi_s else None,
        "auroc_spread": round(max(aurocs) - min(aurocs), 4),
        "ece": round(_mean(eces), 4),
        "utility": round(_mean(utils), 4),
        "detection_rate": round(_mean(dets), 4),
        "false_alarm_rate": round(_mean(fars), 4),
        "median_lead_hours": round(_mean(leads), 1),
    })
    return out


def _findings(prod: Dict[str, Any], base: Dict[str, Any],
              op: Optional[Dict[str, Any]]) -> Tuple[List[Dict], List[Dict]]:
    """What this system does well, and where it falls short — from the numbers."""
    good: List[Dict[str, str]] = []
    bad: List[Dict[str, str]] = []

    auroc = prod.get("auroc", 0.0)
    # Discrimination, graded strictly. 0.72 is a real signal and a modest one, and
    # saying only the first half would be the flattering version.
    if auroc >= 0.75:
        good.append({"title": f"AUROC {auroc:.3f} held across hospitals",
                     "detail": "trained on one site, tested on the other — this is "
                               "external validation, not a held-out split"})
    else:
        bad.append({"title": f"Discrimination is modest — AUROC {auroc:.3f}",
                    "detail": "a useful signal but not a strong one; well short of "
                              "the ≥0.80 a screening tool would want, and this is "
                              "the number to improve"})
    if base and prod.get("auroc", 0) > base.get("auroc", 0) + 0.01:
        d = prod["auroc"] - base["auroc"]
        good.append({"title": f"Beats its own baseline by {d:+.3f} AUROC",
                     "detail": f"stacked ensemble {prod['auroc']:.3f} vs gradient "
                               f"boosting {base['auroc']:.3f} on the same "
                               "cross-hospital protocol"})
    spread = prod.get("auroc_spread", 0)
    if spread > 0.05:
        bad.append({"title": f"Performance differs by site — {spread:.3f} AUROC "
                             "between directions",
                    "detail": "the model does not transfer equally in both "
                              "directions, so a new hospital may land at the "
                              "lower end"})
    else:
        good.append({"title": "Transfers symmetrically between sites",
                     "detail": f"only {spread:.3f} AUROC separates the two "
                               "train/test directions"})

    ece = prod.get("ece")
    if ece is not None and ece <= 0.02:
        good.append({"title": f"Calibrated — ECE {ece:.3f}",
                     "detail": "a stated probability means what it says, which is "
                               "what makes the alarm threshold meaningful"})
    elif ece is not None:
        bad.append({"title": f"Probabilities are miscalibrated — ECE {ece:.3f}",
                    "detail": "the number attached to a risk cannot be taken at "
                              "face value"})

    far = prod.get("false_alarm_rate")
    if far is not None and far > 0.20:
        bad.append({"title": f"False-alarm rate {far*100:.0f}%",
                    "detail": f"roughly one in {max(2, round(1/far))} stable "
                              "patients still triggers an alarm at this operating "
                              "point — the main barrier to bedside use"})
    det = prod.get("detection_rate")
    if det is not None and det < 0.60:
        bad.append({"title": f"Catches {det*100:.0f}% of septic stays",
                    "detail": "a substantial minority are still missed; raising "
                              "this costs false alarms, which is the trade this "
                              "operating point is making"})
    lead = prod.get("median_lead_hours")
    if lead and lead >= 12:
        good.append({"title": f"Median {lead:.0f} h of warning before onset",
                     "detail": "on the stays it does catch — time enough for the "
                               "Hour-1 bundle to matter"})
    if base and prod.get("false_alarm_rate", 1) < base.get("false_alarm_rate", 1) - 0.05:
        good.append({"title": "Fewer false alarms than the baseline",
                     "detail": f"{prod['false_alarm_rate']*100:.0f}% vs "
                               f"{base['false_alarm_rate']*100:.0f}% — the "
                               "ensemble buys specificity, not just AUROC"})
    gen = _gen_cache.get("g")
    if gen and gen["gap"] > 0.05:
        bad.append({"title": f"Loses {gen['gap']:.3f} AUROC moving to a new hospital",
                    "detail": f"{gen['internal_auroc']:.3f} inside the training site "
                              f"against {gen['external_auroc']:.3f} across sites — "
                              "the internal figure is the one a paper would quote, "
                              "and it is not the one a new ICU would get"})
    if op and op.get("oof_false_alarm_rate") is not None:
        good.append({"title": "Alarm thresholds derived from data, not chosen",
                     "detail": f"tuned out-of-fold to hold false alarms under "
                               f"{op.get('max_false_alarm_rate', 0.15)*100:.0f}% "
                               f"(achieved {op['oof_false_alarm_rate']*100:.1f}%)"})
    return good, bad


def _generalization(rig: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Internal split vs external hospital — the number that actually predicts
    whether a clinical model survives deployment.

    A model tuned and tested inside one site flatters itself. The distance between
    that figure and the cross-hospital one is the validation gap, and it is the
    single most common reason a published AUROC does not reproduce at the bedside.
    Showing both side by side is the least flattering and most useful thing this
    page does — the internal number is the one a paper would quote.
    """
    comp = _load("compare_wearable.json")
    if not comp:
        return None
    internal = comp.get(BASELINE) or {}
    ext = _pooled(rig, BASELINE)
    if not internal.get("auroc") or not ext.get("auroc"):
        return None
    gap = internal["auroc"] - ext["auroc"]
    return {
        "internal_auroc": round(float(internal["auroc"]), 4),
        "external_auroc": ext["auroc"],
        "gap": round(gap, 4),
        "model": BASELINE,
        "note": ("measured on the same model, same features — the only difference "
                 "is whether the test patients came from the training hospital"),
    }


def _ladder(rig: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every model on the SAME cross-hospital protocol, worst to best.

    This is what "where it improves" looks like when there is no time series to
    plot: the ladder the production model had to climb, each rung measured the
    same way.
    """
    # SAME PROTOCOL ONLY. The hyperdimensional encoder was measured on an internal
    # split (0.696) and the gradient-boosting baseline across hospitals (0.661);
    # putting both in one ranked bar chart would show HDC winning, which is not a
    # thing this evaluation says. Two rungs measured identically beat three that
    # cannot be compared — a ladder whose rungs mean different things is a false
    # comparison however carefully the scope is labelled underneath.
    out: List[Dict[str, Any]] = []
    for key, label in ((BASELINE, "Gradient boosting"),
                       (PRODUCTION, "Stacked ensemble")):
        p = _pooled(rig, key)
        if p.get("auroc"):
            out.append({"name": label, "key": key, "auroc": p["auroc"],
                        "scope": "cross-hospital, both directions",
                        "note": ("the baseline it has to beat" if key == BASELINE
                                 else "in production"),
                        "production": key == PRODUCTION})
    return out


def _alarm_policy() -> Optional[Dict[str, Any]]:
    """The alarm-policy search result. This exists so the false-alarm gap on this
    page is not just a complaint: it records that four policy families were
    searched against a locked holdout and none beat a plain threshold, which
    redirects the reader from "tune the alarm" to "the score needs to separate
    better"."""
    d = _load("alarm_policy.json")
    if not d:
        return None
    v = d.get("verdict") or {}
    return {
        "shipped": (d.get("shipped_policy") or {}).get("describe"),
        "challenger": v.get("challenger"),
        "promoted": bool(v.get("promoted")),
        "reason": v.get("reason"),
        "champion_far": v.get("champion_far"),
        "challenger_far": v.get("challenger_far"),
        "champion_detection": v.get("champion_detection"),
        "challenger_detection": v.get("challenger_detection"),
        "ci": [v.get("ci_low"), v.get("ci_high")],
        "families": ["plain threshold", "K consecutive hours",
                     "leaky exceedance integrator", "per-patient z-score"],
        "verdict": ("No alarm policy beat a plain threshold. At a fixed detection "
                    "rate the false-alarm rate is set by how well the score "
                    "separates, and no post-processing recovers separation the "
                    "score does not have."),
    }


def _refusal_class(reason: str, accepted: bool) -> str:
    """Group a refusal by WHY, so sixteen rejections read as a pattern rather than
    sixteen separate disappointments. The three classes are meaningfully different:
    a trial that lost outright is the search exploring, a trial that gained but
    failed the significance test is the gate doing its job, and a trial that was
    simply not the round's best never got tested at all."""
    if accepted:
        return "accepted"
    r = (reason or "").lower()
    if "won the round" in r:
        return "lost-round"
    if "not the round" in r:
        return "untested"
    if "p=" in r or "significan" in r:
        return "not-significant"
    if "min_gain" in r or "<=" in r:
        return "no-gain"
    return "refused"


def _transcript(led: Dict[str, Any], trials: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Per-trial record for the UI: what was changed, what it scored, what happened.

    The ledger stores each trial's full resulting config but older runs pre-date the
    mutation diff, so it is derived here when absent — the search never promoted
    anything, which means the champion never moved and every trial is a mutation of
    the baseline. Ordered worst-delta last so the near-misses read first.
    """
    try:
        from backend.ml.selfimprove import config_diff
    except Exception:
        config_diff = None            # type: ignore[assignment]
    base = led.get("champion_config") or led.get("baseline_config") or {}
    out: List[Dict[str, Any]] = []
    for t in trials:
        mut = t.get("mutation")
        if not mut and config_diff is not None and t.get("config"):
            try:
                mut = config_diff(base, t["config"])
            except Exception:
                mut = []
        out.append({
            "id": t.get("trial_id"),
            "round": t.get("round_idx"),
            "mutation": mut or ["(no change recorded)"],
            "auroc": t.get("mean_auroc"),
            "delta": t.get("delta_vs_champion"),
            "p": t.get("p_value"),
            "seconds": t.get("seconds"),
            "accepted": bool(t.get("accepted")),
            "outcome": _refusal_class(t.get("reason", ""), bool(t.get("accepted"))),
            "reason": t.get("reason") or "",
        })
    out.sort(key=lambda r: (r["delta"] is None, -(r["delta"] or 0)))
    return out


def _improvement() -> Optional[Dict[str, Any]]:
    """The record of attempts to make it better — including the failures.

    A page that only showed successful improvements would imply the model has been
    getting steadily better. It has not: the last automated search ran sixteen
    mutations against a locked gate and promoted none of them. That is the honest
    state, and the refusal is the machinery working rather than a gap in it.
    """
    led = _load("self_improve_ledger.json")
    if not led:
        return None
    trials = led.get("trials") or []
    accepted = [t for t in trials if t.get("accepted")]
    return {
        "trials": len(trials),
        "accepted": len(accepted),
        "transcript": _transcript(led, trials),
        "promoted": bool(led.get("promoted")),
        "baseline_auroc": round(float(led.get("baseline_search_auroc", 0)), 4),
        "champion_auroc": round(float(led.get("champion_search_auroc", 0)), 4),
        "verdict": led.get("verdict"),
    }


def report() -> Dict[str, Any]:
    """The system-health payload."""
    rig = _load("rigorous_eval.json")
    op = _load("operating_point.json")
    if not rig:
        return {"status": "unavailable",
                "note": "No cross-hospital evaluation artefact found. Run "
                        "`make rigorous` to generate artifacts/rigorous_eval.json."}

    prod = _pooled(rig, PRODUCTION)
    base = _pooled(rig, BASELINE)
    _gen_cache["g"] = _generalization(rig)
    good, bad = _findings(prod, base, op)
    meta = rig.get("meta") or {}

    auroc = prod.get("auroc", 0.0)
    grade = ("healthy" if auroc >= 0.75 else
             "fair" if auroc >= 0.65 else "weak")
    return {
        "status": "ok",
        "grade": grade,
        "headline": {
            "auroc": prod.get("auroc"),
            "auroc_lo": prod.get("auroc_lo"),
            "auroc_hi": prod.get("auroc_hi"),
            "baseline_auroc": base.get("auroc"),
            "delta": (round(prod["auroc"] - base["auroc"], 4)
                      if prod.get("auroc") and base.get("auroc") else None),
            "model": PRODUCTION, "baseline": BASELINE,
        },
        "metrics": {
            "detection_rate": prod.get("detection_rate"),
            "false_alarm_rate": prod.get("false_alarm_rate"),
            "median_lead_hours": prod.get("median_lead_hours"),
            "ece": prod.get("ece"), "utility": prod.get("utility"),
        },
        "directions": prod.get("directions", []),
        "ladder": _ladder(rig),
        "generalization": _generalization(rig),
        "improvement": _improvement(),
        "alarm_policy": _alarm_policy(),
        "strengths": good,
        "gaps": bad,
        "evaluation": {
            "protocol": meta.get("protocol"),
            "generated": meta.get("generated"),
            "bootstrap": meta.get("boot"),
            "sets": {k: meta.get(k) for k in ("setA", "setB") if meta.get(k)},
        },
        "honesty": (
            "Cross-hospital external validation: trained on one site, tested on the "
            "other, both directions, with bootstrap intervals. It is the honest "
            "protocol and it is also why the numbers are lower than a held-out "
            "split would show. Nothing here is measured on the demo cohort — twelve "
            "stays cannot support an AUROC."),
    }


def _demo() -> None:
    r = report()
    if r["status"] != "ok":
        print(r["note"])
        return
    h = r["headline"]
    print(f"grade {r['grade']} · AUROC {h['auroc']} "
          f"[{h['auroc_lo']}–{h['auroc_hi']}] vs baseline {h['baseline_auroc']} "
          f"({h['delta']:+.3f})\n")
    for d in r["directions"]:
        print(f"  {d['train']} → {d['test']}: AUROC {d['auroc']} · "
              f"detect {d['detection_rate']*100:.0f}% · FA {d['false_alarm_rate']*100:.0f}%")
    print("\nSTRENGTHS")
    for g in r["strengths"]:
        print(f"  + {g['title']}\n      {g['detail']}")
    print("\nGAPS")
    for g in r["gaps"]:
        print(f"  − {g['title']}\n      {g['detail']}")


if __name__ == "__main__":
    _demo()
