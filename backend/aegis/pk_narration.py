"""Narration — what is happening in the body right now, and why.

The treatment twin shows a drug moving through a patient. Someone who already
knows pharmacokinetics can read that; someone who does not sees a body changing
colour. This layer turns the projection into a short sequence of BEATS — timed
explanations that surface while the simulation plays, each one anchored to a
moment the model actually produced.

EVERY BEAT IS GENERATED FROM THE CURVE
--------------------------------------
A beat is not a caption written next to a drug name. Its time comes from the
projection (when the lungs actually cross 5% of their peak, when plasma actually
peaks, when the trough actually falls), and its text quotes the number at that
moment. Change the patient's kidneys and the elimination beat changes with them;
give a different dose and the peak beat moves and re-reads. That is the same
discipline the upsides/downsides use, for the same reason: text that cannot be
wrong is text that is not connected to anything.

WHAT THE BEATS ARE FOR
----------------------
They explain the mechanism the picture is already showing — why the lungs light
first, why the kidney runs hottest, why muscle lags, where the drug goes when it
leaves. A clinician can turn them off; someone learning the system can leave them
on. They never introduce a claim the projection did not make.

    python -m backend.aegis.pk_narration        # demo
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

# Beat kinds drive the accent the UI gives each card.
ARRIVAL, PEAK, DISTRIBUTION, ELIMINATION, TARGET, WARN = (
    "arrival", "peak", "distribution", "elimination", "target", "warn")

ORGAN_NAME = {
    "lung": "lungs", "heart": "heart", "liver": "liver", "kidney": "kidneys",
    "spleen": "spleen", "pancreas": "pancreas", "intestine": "gut",
    "brain": "brain", "muscle": "muscle", "skin": "skin", "adipose": "fat",
    "rest": "other tissue",
}


def _at(series: List[float], times: List[float], t: float) -> float:
    if not series:
        return 0.0
    if t <= times[0]:
        return series[0]
    if t >= times[-1]:
        return series[-1]
    step = (times[-1] - times[0]) / (len(times) - 1)
    i = min(len(series) - 2, max(0, int((t - times[0]) / step)))
    w = (t - times[0]) / step - i
    return series[i] * (1 - w) + series[i + 1] * w


def _fmt(v: float) -> str:
    a = abs(v)
    if a == 0:
        return "0"
    if a < 0.001:
        return "<0.001"
    if a < 1:
        return f"{v:.3f}"
    if a < 10:
        return f"{v:.2f}"
    if a < 1000:
        return f"{v:.1f}"
    return f"{v:,.0f}"


def _clock(h: float) -> str:
    if h < 0.1:
        total_s = int(round(h * 3600))
        return f"{total_s // 60}:{total_s % 60:02d} min"   # carry, or 4:60 appears
    hh = int(h)
    mm = int(round((h - hh) * 60))
    if mm == 60:
        hh, mm = hh + 1, 0
    return f"{hh}h{mm:02d}"


def _first_crossing(series: List[float], times: List[float], frac: float) -> Optional[float]:
    if not series:
        return None
    peak = max(series)
    if peak <= 0:
        return None
    thresh = peak * frac
    for i, v in enumerate(series):
        if v >= thresh:
            return times[i]
    return None


def narrate(arm: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build the beat sequence for one projected drug arm."""
    times = arm.get("times_h") or []
    organs = arm.get("organs") or {}
    m = arm.get("metrics") or {}
    unit = arm.get("unit", "")
    label = arm.get("label", "the drug")
    horizon = float(arm.get("horizon_h", 0) or 0)
    kind = arm.get("kind", "concentration")
    beats: List[Dict[str, Any]] = []

    def add(t, title, body, k, organ=None):
        if t is None or not (0 <= t <= horizon + 1e-9):
            return
        beats.append({"t_h": round(float(t), 4), "title": title, "body": body,
                      "kind": k, "organ": organ, "at": _clock(float(t))})

    # ---- crystalloid is a volume story, not a concentration one -------------- #
    if kind == "volume":
        infused = m.get("infused_ml", 0)
        ret = m.get("plasma_retention_1h_post")
        inter = m.get("interstitial_ml_end", 0)
        add(0.0, "Infusion begins",
            f"{_fmt(infused)} mL of crystalloid entering the circulation. It expands "
            "the plasma first — the compartment that carries blood pressure.",
            ARRIVAL)
        peak_t = _first_crossing(organs.get("arterial", []), times, 0.99)
        add(peak_t, "Plasma expansion peaks",
            f"Peak intravascular expansion {_fmt(m.get('peak_plasma_expansion_ml', 0))} mL. "
            "This is the part of the bolus doing haemodynamic work.", PEAK)
        if ret is not None:
            add(min(horizon, 2.0), "Most of it has already left the vessels",
                f"Only {ret*100:.0f}% is still intravascular an hour after the bolus; "
                f"{_fmt(inter)} mL has moved into tissue. That redistribution is why "
                "volume has to be reassessed rather than assumed.", DISTRIBUTION)
        return _space(beats, horizon)

    # ---- concentration drugs ------------------------------------------------- #
    art = organs.get("arterial", [])
    continuous = bool(m.get("continuous"))

    add(0.0, "Infusion begins",
        f"{label} entering a peripheral vein. It must cross the right heart and the "
        "lungs before any other organ can see it.", ARRIVAL)

    lung_t = _first_crossing(organs.get("lung", []), times, 0.05)
    add(lung_t, "First pass — the lungs",
        "The lungs receive the ENTIRE cardiac output, so they are always the first "
        f"organ the drug reaches — here at {_clock(lung_t) if lung_t else '—'}, before "
        "anything downstream. This ordering is physiology, not animation.",
        ARRIVAL, "lung")

    if continuous:
        t90 = m.get("t_to_90pct_steady_state_h")
        css = m.get("css")
        add(t90, "Steady state reached",
            f"Plasma has levelled at {_fmt(css)} {unit}. A continuous infusion holds "
            "this plateau: rate in equals clearance out. Turning it off empties just "
            "as fast, which is why a short-half-life pressor is titratable.",
            PEAK)
    else:
        add(m.get("tmax_h"), "Peak plasma concentration",
            f"Cmax {_fmt(m.get('cmax'))} {unit}. Every organ is now loading from "
            "arterial blood at its own perfusion rate — this is the moment the "
            "gradient into tissue is steepest.", PEAK)

    # The organ that ends up carrying the most, with the reason it does.
    oa = m.get("organ_auc") or {}
    tissue = {k: v for k, v in oa.items()
              if k not in ("arterial", "venous") and k in ORGAN_NAME}
    if tissue:
        top = max(tissue, key=lambda k: tissue[k])
        top_t = _first_crossing(organs.get(top, []), times, 0.95)
        why = {
            "kidney": "it is both the best-perfused organ per gram and the route this "
                      "drug leaves by",
            "liver": "it receives the hepatic artery AND the portal vein draining the "
                     "whole gut",
            "lung": "it takes the entire cardiac output",
        }.get(top, "of its blood flow relative to its mass")
        add(top_t, f"The {ORGAN_NAME[top]} carry the most",
            f"{_fmt(_at(organs.get(top, []), times, top_t or 0))} {unit} — the highest "
            f"tissue exposure of any compartment, because {why}.",
            DISTRIBUTION, top)

    brain = organs.get("brain", [])
    if brain and art:
        b_peak, a_peak = max(brain), max(art)
        if a_peak > 0 and b_peak / a_peak < 0.25:
            add(m.get("tmax_h"), "The brain stays largely protected",
                f"Only {_fmt(b_peak)} {unit} against {_fmt(a_peak)} in plasma — about "
                f"{100*b_peak/a_peak:.0f}%. The blood-brain barrier excludes most of "
                "this drug, which is why CNS effects need either high exposure or a "
                "broken barrier.", DISTRIBUTION, "brain")

    # Muscle is most of the body by mass and always lags — the clearest lesson in
    # the whole render, and the easiest to miss because it has no mesh.
    mus_t = _first_crossing(organs.get("muscle", []), times, 0.95)
    if mus_t is not None and lung_t is not None and mus_t > lung_t * 3:
        add(mus_t, "Muscle fills last",
            "Muscle is the largest compartment by mass but has low flow per litre of "
            f"tissue, so it lags the lungs by {_clock(mus_t - lung_t)}. Most of the "
            "drug in the body ends up here, not in the organs you can see.",
            DISTRIBUTION, "muscle")

    pct = m.get("pct_ft_above_mic")
    if pct is not None:
        good = pct >= 50.0
        add(min(horizon, (m.get("tmax_h") or 0) + horizon * 0.25),
            "Time above MIC" + ("" if good else " — short"),
            f"Free drug stays above the {m.get('mic_mg_l')} mg/L breakpoint for "
            f"{pct:.0f}% of the interval. For a beta-lactam this — not the peak — is "
            "what kills bacteria.", TARGET if good else WARN)

    if not continuous and horizon > 0:
        tail = _at(art, times, horizon)
        add(horizon * 0.98, "Falling toward the trough",
            f"Plasma is down to {_fmt(tail)} {unit}. Clearance is "
            f"{_fmt(m.get('clearance_l_h'))} L/h, so {_fmt(m.get('eliminated_mg'))} mg "
            "of the dose has already left the body — mostly renally.", ELIMINATION)

    return _space(beats, horizon)


def _space(beats: List[Dict[str, Any]], horizon: float) -> List[Dict[str, Any]]:
    """Order the beats. Deliberately does NOT shift them in time.

    Several beats legitimately share an instant — at the peak, the plasma maximum,
    the hottest organ and the barrier-excluded brain are all true at once. The
    obvious fix is to nudge them apart so they present one at a time, and the first
    version did exactly that. It was wrong: on a 16 h horizon the gap needed to
    make a card readable is most of an hour, which fired the lung first-pass beat
    fifty minutes after the lungs actually lit up. A card that describes a moment
    the body is no longer in is worse than two cards at once.

    So the times stay true and the UI stacks whatever overlaps. Presentation is a
    presentation problem; it does not get to move the physiology."""
    if not beats or horizon <= 0:
        return beats
    out = sorted(beats, key=lambda b: (b["t_h"], b["title"]))
    return [b for b in out if b["t_h"] <= horizon + 1e-9]


def narrate_arms(arms: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
    """Beats keyed by drug, so the UI can switch arms without refetching."""
    return {a["drug"]: narrate(a) for a in arms}


def _demo() -> None:
    from backend.aegis.pharmacokinetics import PBPKModel, PatientState
    model = PBPKModel(PatientState(weight_kg=78, egfr=45, map_mmhg=62))
    for key in ("cefepime", "norepinephrine"):
        p = model.project(key).to_dict()
        print(f"\n=== {p['label']} ===")
        for b in narrate(p):
            print(f"  [{b['at']:>9s}] {b['title']}")
            print(f"              {b['body'][:96]}")


if __name__ == "__main__":
    _demo()
