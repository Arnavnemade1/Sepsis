"""Self-improving ALARM POLICY search, with a gate that refuses by default.

The detector produces a probability per patient-hour. Turning that stream into a
bedside alarm is a separate decision — and until now that decision was a single
scalar threshold, chosen once and never searched. This module makes the alarm
policy itself a searchable object and puts it behind the same evidence bar the
model search uses ([[self-improvement-loop]]): a locked holdout the search never
sees, paired bootstrap over PATIENTS, and promotion only when the confidence
interval on the improvement excludes zero.

WHAT THE SEARCH FOUND (re-measured 2026-09-12, CinC 2019, wearable channels)
---------------------------------------------------------------------------
Measured at the SHIPPED operating point — high = 0.03 on detector
``patient:hgb+hgb2+et+lr+xgb@utility`` — over 3,004 patients, half searched and
half locked away. Four first-alarm families, at or below the shipped policy's own
false-alarm rate on the half it constrains (32.7%):

    plain threshold (shipped)        FA 32.7%   detection 62.9%
    K consecutive hours above thr    FA 20.8%   detection 51.7%   (best k = 2; the
                                                                   grid searches k >= 2
                                                                   because k = 1 IS
                                                                   the plain family)
    leaky exceedance integrator      FA 30.6%   detection 59.4%
    per-patient z against own stay   FA 31.5%   detection 46.2%

None of them beats a plain threshold, and the gate refused the winner on the
locked half: detection delta 0.000, 95% CI [+0.000, +0.000]. Every alternative
buys a lower false-alarm rate with a larger loss of detection — movement along
the ROC curve, not a better curve.

THE PREVIOUS NUMBERS ARE SUPERSEDED, NOT DELETED. The 2026-08-03 run reported
FA 10.6% / detection 38.1% for the shipped policy. That was measured at
``plain(0.0725)`` on roster ``[hgb, hgb2, et, lr]`` — an operating point and a
model this system no longer runs — by a script that hardcoded both. The
conclusion survived re-measurement; its numbers did not, and could not, because
the shipped threshold more than halved. A claim whose fit is not named is a claim
that cannot be checked, so the artifact now carries the operating point and the
fit identity beside the verdict.

EPISODE ECONOMICS (new in this run)
-----------------------------------
The study above measures only the FIRST alarm. Replaying the same locked patients
through the live AlertEpisodeController, the shipped renotify(4h) schedule costs
1.05 non-septic alert episodes per patient-day and 3.43 repeat notifications per
septic episode, with a median 2.0 h to clear.

The re-notification interval trades interruptions against silence, monotonically:

    renotify(1h)      burden 2.78 /patient-day    unannounced HIGH  0.0 h/episode
    renotify(4h)      burden 1.68                                  13.1
    renotify(never)   burden 1.42                                  17.2

So the burden endpoint can always be "won" by never interrupting again — which is
why a burden reduction is refused unless unannounced HIGH hours also fail to rise.
On the locked half that refusal fired: burden 1.60 -> 1.31 (95% CI [-0.341,
-0.240], which excludes zero) but unannounced HIGH hours 11.1 -> 14.5. REFUSED.
The quiet was the whole of the saving.

Cross-hospital threshold transfer behaves the same way. The shipped threshold
sits at the 93rd percentile of the home score distribution; matching that
PERCENTILE at the new site rather than the raw probability moves the operating
point but does not improve it:

    A -> B   FA  7.6% -> 16.7%    detection 32.7% -> 41.2%
    B -> A   FA 31.8% -> 22.0%    detection 52.2% -> 43.7%

Roughly one point of false alarm per point of detection, in both directions.
That is movement ALONG the ROC curve, not a better curve.

The conclusion is worth stating plainly because it redirects effort: at a given
detection rate the false-alarm rate is set by how well the score separates
septic from non-septic patients, and no post-processing of that score recovers
separation the score does not have. The 36% cross-hospital false-alarm rate is
the AUROC 0.716 being honest, not a policy that needs tuning. Reducing it
requires a better-separating signal — which is the data-limited problem this
project has already documented — or accepting less detection.

(The cross-hospital transfer figures in the paragraph above are from the 2026-08-03
run and have NOT been re-measured at the current operating point; only the
same-cohort search was re-run. What did change is that the home-cohort false-alarm
rate is now 32.2% at the shipped threshold, because the operating point moved down
to 0.03 — so the alarm burden this project once described as a cross-hospital
problem is now substantially a same-hospital one.)

WHAT MUST NOT BE RE-PROPOSED
----------------------------
Everything below transforms the SCORE before thresholding. All of it has been
measured on this data and all of it lost; re-running any of it is spending the
evidence budget on a question already answered:

    EMA / causal smoothing      docs/FRONTIER_REPORT.md — utility +0.294 vs +0.339 raw,
                                and permanently dedup-blocked in the Lab registry
                                (change_key for {"policy":"ema","halflife":6})
    running max (cummax)        utility +0.304, ECE 0.0145
    decaying log-odds (hazard)  utility -0.011 — the worst decision layer measured here
    50/50 blend                 utility +0.323; an AUROC crumb for the metric that counts
    persistent K>=2 / integrator / per-patient z   the four families above

THE EPISODE ENDPOINT IS GAMEABLE, AND IS GUARDED. Interruptions per patient-day is
monotone in the re-notification interval: "never re-notify" minimises it by
construction, which would be a tautology dressed as a finding. So a burden win is
refused unless unannounced HIGH hours per septic episode also fail to rise. Episode
metrics are an additional operational outcome, never a way to hide a missed patient.

The genuinely open object is the EPISODE layer — how often a person is interrupted
AFTER the first alarm. That leaves the score, the threshold and the first crossing
untouched, so first-alarm detection and false-alarm rate are invariant BY
CONSTRUCTION (asserted, not measured, in verify_alarm_policy). See EpisodePolicy.

LEAD TIME IS NOT REPORTED HERE. A random score reproduces this detector's median
lead at every matched detection rate (docs/LEAD_TIME_NULL.md), so no lead quantity
is written to the artifact. ``PolicyResult.median_lead_hours`` survives as an
internal field only because the first-alarm MECHANICS test asserts against it.

This module is kept because the SEARCH is the durable asset: any new policy idea
can be added as a family and tested against the same locked gate, and the gate
will refuse it unless it genuinely wins.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from backend.aegis.alert_episodes import AlertEpisodeController
from backend.ml.jsonio import json_safe

ARTIFACT = Path("artifacts/alarm_policy.json")


# --------------------------------------------------------------------------- #
# patient view
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PatientSeries:
    """One admission: scores in time order, their hours, and the onset hour.

    onset < 0 marks a non-septic stay. Keeping the series pre-sorted per patient
    is what makes a policy sweep cheap enough to search thousands of candidates.
    """
    scores: np.ndarray
    hours: np.ndarray
    onset: int

    @property
    def septic(self) -> bool:
        return self.onset >= 0


def to_series(scores: np.ndarray, groups: np.ndarray, hours: np.ndarray,
              onset: np.ndarray) -> List[PatientSeries]:
    out: List[PatientSeries] = []
    for g in sorted(set(groups.tolist())):
        m = groups == g
        order = np.argsort(hours[m])
        out.append(PatientSeries(np.asarray(scores)[m][order],
                                 np.asarray(hours)[m][order],
                                 int(np.asarray(onset)[m][0])))
    return out


# --------------------------------------------------------------------------- #
# policy families — each returns the hour an alarm fires, or None
# --------------------------------------------------------------------------- #
FireFn = Callable[[np.ndarray, np.ndarray], Optional[float]]


def plain(threshold: float) -> FireFn:
    """The shipped policy: first hour at or above a fixed probability."""
    def fire(scores, hours):
        idx = np.nonzero(scores >= threshold)[0]
        return float(hours[idx[0]]) if idx.size else None
    return fire


def persistent(threshold: float, k: int) -> FireFn:
    """Require k CONSECUTIVE hours above threshold before alarming."""
    def fire(scores, hours):
        run = 0
        for i, s in enumerate(scores):
            run = run + 1 if s >= threshold else 0
            if run >= k:
                return float(hours[i])
        return None
    return fire


def integrator(threshold: float, decay: float, trigger: float) -> FireFn:
    """Leaky accumulation of exceedance — magnitude AND duration, not a binary run.

    s <- decay*s + max(0, p - threshold); alarm when s crosses trigger. This is the
    mechanism the camera monitor relies on, which is why it was worth testing here.
    """
    def fire(scores, hours):
        s = 0.0
        for i, p in enumerate(scores):
            s = decay * s + max(0.0, float(p) - threshold)
            if s >= trigger:
                return float(hours[i])
        return None
    return fire


def per_patient(warmup: int, z: float, floor: float) -> FireFn:
    """Score against this admission's OWN history. Causal: only uses past hours."""
    def fire(scores, hours):
        for i in range(warmup, len(scores)):
            hist = scores[:i]
            sd = float(hist.std())
            if sd < 1e-9:
                sd = 1e-9
            if (float(scores[i]) - float(hist.mean())) / sd >= z and scores[i] >= floor:
                return float(hours[i])
        return None
    return fire


@dataclass(frozen=True)
class Policy:
    family: str
    params: Dict[str, float]

    def build(self) -> FireFn:
        f = {"plain": plain, "persistent": persistent,
             "integrator": integrator, "per_patient": per_patient}[self.family]
        return f(**self.params)   # type: ignore[arg-type]

    def describe(self) -> str:
        ps = ", ".join(f"{k}={v:g}" for k, v in self.params.items())
        return f"{self.family}({ps})"


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #
@dataclass
class PolicyResult:
    false_alarm_rate: float
    detection_rate: float
    median_lead_hours: float
    n_septic: int
    n_nonseptic: int
    fired_septic: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))
    fired_nonseptic: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


def evaluate(policy: Policy, patients: Sequence[PatientSeries]) -> PolicyResult:
    """Per-patient alarm outcome. A non-septic stay counts as a false alarm if the
    policy ever fires; a septic stay counts as detected only if it fires at or
    before the labelled onset."""
    fire = policy.build()
    det_flags, fa_flags, leads = [], [], []
    for p in patients:
        t = fire(p.scores, p.hours)
        if p.septic:
            hit = t is not None and t <= p.onset
            det_flags.append(1.0 if hit else 0.0)
            if hit:
                leads.append(p.onset - t)
        else:
            fa_flags.append(1.0 if t is not None else 0.0)
    det = np.asarray(det_flags, dtype=float)
    fa = np.asarray(fa_flags, dtype=float)
    return PolicyResult(
        false_alarm_rate=float(fa.mean()) if fa.size else 0.0,
        detection_rate=float(det.mean()) if det.size else 0.0,
        median_lead_hours=float(np.median(leads)) if leads else 0.0,
        n_septic=int(det.size), n_nonseptic=int(fa.size),
        fired_septic=det, fired_nonseptic=fa)



# --------------------------------------------------------------------------- #
# episode economics — the OPERATIONAL layer, measured with the RUNTIME controller
# --------------------------------------------------------------------------- #
#: Human labels for the artifact, so the console and the website render the field
#: that was actually searched instead of a list re-declared in three other files.
FAMILY_LABELS = {
    "plain": "plain threshold",
    "persistent": "K consecutive hours",
    "integrator": "leaky exceedance integrator",
    "per_patient": "per-patient z-score",
    "renotify": "re-notification schedule (episode layer)",
}


@dataclass(frozen=True)
class EpisodePolicy:
    """A notification schedule, evaluated on top of a FIXED alarm threshold.

    This is a different object from a :class:`Policy`. A Policy decides when the
    first alarm fires; an EpisodePolicy decides how often a person is interrupted
    after it. Keeping them apart is what makes the safety guard structural rather
    than statistical: nothing here can move the first crossing, so detection and
    false-alarm rate are unchanged by construction.
    """
    high_renotify_seconds: Optional[float] = 4 * 3600.0

    def describe(self) -> str:
        h = self.high_renotify_seconds
        return "renotify(%s)" % ("never" if h is None else "%gh" % (h / 3600.0))


def tier_of(score: float, high: float, medium: float) -> str:
    """The product's own tier mapping. MEDIUM is a pre-alarm below HIGH."""
    return "HIGH" if score >= high else ("MEDIUM" if score >= medium else "LOW")


def replay_episodes(p: PatientSeries, high: float, medium: float,
                    policy: EpisodePolicy) -> List[Dict[str, object]]:
    """Drive the LIVE controller across one admission and return its events.

    The controller is imported, never reimplemented. An offline study that models
    the notification layer with its own copy of the rules measures an object the
    product does not have — and the product's rules include one guarantee that must
    survive any schedule searched here: an escalation is never suppressed.
    """
    ctl = AlertEpisodeController(high_renotify_seconds=policy.high_renotify_seconds)
    out: List[Dict[str, object]] = []
    for s, h in zip(p.scores, p.hours):
        level = tier_of(float(s), high, medium)
        ev = ctl.update(level, float(h) * 3600.0)
        out.append({"state": ev.state, "notify": bool(ev.notify), "level": level,
                    "elapsed": float(ev.elapsed_seconds)})
    return out


@dataclass
class EpisodeResult:
    """Operational burden, in the units a clinician is actually charged in."""
    nonseptic_episodes_per_patient_day: float
    repeat_notifications_per_septic_episode: float
    escalation_recall: float
    median_time_to_clear_h: float
    total_notifications: int
    n_septic: int
    n_nonseptic: int
    #: SEPTIC stays only: HIGH-tier hours that pass with no notification, per opened
    #: episode. This is the COST side of a quieter schedule, and it exists because the
    #: burden endpoint is mechanically monotone in the re-notification interval — a
    #: schedule can always "win" on interruptions by never interrupting again. That is
    #: the same failure as the retracted lead-time claim: a number that measures the
    #: policy's own geometry rather than anything about the patient. Re-notification
    #: exists so an unresolved HIGH does not disappear; this counts how long it does.
    silent_high_hours_per_septic_episode: float = float("nan")
    #: Per NON-SEPTIC patient: notifications per patient-day. This is the vector the
    #: burden bootstrap resamples — the pooled headline rate above is not a mean over
    #: patients, so it is not a legal unit for a patient-level interval.
    burden_per_patient: np.ndarray = field(repr=False, default_factory=lambda: np.array([]))


def evaluate_episodes(policy: EpisodePolicy, patients: Sequence[PatientSeries],
                      high: float, medium: float) -> EpisodeResult:
    """Replay every admission through the runtime controller and tally the burden."""
    ns_notifies: List[float] = []
    ns_days: List[float] = []
    ns_episodes = opened_septic = renotify_septic = 0
    escalated = escalatable = 0
    silent_high = 0
    clears: List[float] = []
    total_notifications = 0
    n_septic = n_nonseptic = 0

    for p in patients:
        ev = replay_episodes(p, high, medium, policy)
        notifies = sum(1 for e in ev if e["notify"])
        total_notifications += notifies
        clears.extend(float(e["elapsed"]) / 3600.0 for e in ev if e["state"] == "RESOLVED")
        # escalation recall: MEDIUM -> HIGH inside an already-open episode. The
        # controller emits ESCALATED exactly then, so this is 1.0 by construction —
        # it is a conformance tripwire on the replay, not a discovery.
        for i in range(1, len(ev)):
            if ev[i]["level"] == "HIGH" and ev[i - 1]["level"] == "MEDIUM" \
                    and ev[i]["state"] != "OPENED":
                escalatable += 1
        escalated += sum(1 for e in ev if e["state"] == "ESCALATED")
        if p.septic:
            n_septic += 1
            opened_septic += sum(1 for e in ev if e["state"] == "OPENED")
            renotify_septic += sum(1 for e in ev if e["state"] == "RENOTIFY")
            silent_high += sum(1 for e in ev
                               if e["level"] == "HIGH" and not e["notify"])
        else:
            n_nonseptic += 1
            days = max(len(p.scores), 1) / 24.0
            ns_days.append(days)
            ns_notifies.append(float(notifies))
            ns_episodes += sum(1 for e in ev if e["state"] == "OPENED")

    nan = float("nan")
    pooled = (ns_episodes / sum(ns_days)) if ns_days else nan
    per_patient = (np.asarray(ns_notifies, dtype=float) / np.asarray(ns_days, dtype=float)
                   if ns_days else np.array([]))
    return EpisodeResult(
        nonseptic_episodes_per_patient_day=float(pooled),
        repeat_notifications_per_septic_episode=(
            float(renotify_septic / opened_septic) if opened_septic else nan),
        escalation_recall=float(escalated / escalatable) if escalatable else nan,
        median_time_to_clear_h=float(np.median(clears)) if clears else nan,
        total_notifications=int(total_notifications),
        n_septic=n_septic, n_nonseptic=n_nonseptic,
        silent_high_hours_per_septic_episode=(
            float(silent_high / opened_septic) if opened_septic else nan),
        burden_per_patient=per_patient)


@dataclass
class BurdenVerdict:
    promoted: bool
    reason: str
    champion: str
    challenger: str
    champion_burden: float
    challenger_burden: float
    delta_burden: float
    ci_low: float
    ci_high: float
    detection_delta: float
    false_alarm_delta: float
    champion_silent_high: float
    challenger_silent_high: float
    n_boot: int


def gate_burden(champion: EpisodePolicy, challenger: EpisodePolicy,
                locked: Sequence[PatientSeries], high: float, medium: float,
                n_boot: int = 2000, seed: int = 0) -> BurdenVerdict:
    """Promote a schedule only on a burden reduction measured on locked patients.

    The existing :func:`gate` resamples SEPTIC patients, because it adjudicates
    detection. An interruption-burden endpoint lives entirely on the NON-SEPTIC
    side, which that bootstrap never touches — so a burden claim gated by it would
    be an interval about the wrong patients. This resamples non-septic patients for
    the burden delta and keeps first-alarm detection as a non-inferiority guard.
    """
    c_ep = evaluate_episodes(champion, locked, high, medium)
    x_ep = evaluate_episodes(challenger, locked, high, medium)
    # The first-alarm layer is identical for both by construction; measuring it is
    # how we find out if that ever stops being true.
    fire = Policy("plain", {"threshold": high})
    c_fa = evaluate(fire, locked)
    x_fa = evaluate(fire, locked)

    rng = np.random.default_rng(seed)
    n = c_ep.burden_per_patient.size
    if n == 0 or x_ep.burden_per_patient.size != n:
        return BurdenVerdict(False, "refused: no non-septic patients to measure burden on",
                             champion.describe(), challenger.describe(),
                             float("nan"), float("nan"), float("nan"),
                             float("nan"), float("nan"), 0.0, 0.0,
                             c_ep.silent_high_hours_per_septic_episode,
                             x_ep.silent_high_hours_per_septic_episode, n_boot)
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        i = rng.integers(0, n, n)
        deltas[b] = x_ep.burden_per_patient[i].mean() - c_ep.burden_per_patient[i].mean()
    lo, hi = (float(v) for v in np.percentile(deltas, [2.5, 97.5]))
    d = float(x_ep.burden_per_patient.mean() - c_ep.burden_per_patient.mean())
    det_d = x_fa.detection_rate - c_fa.detection_rate
    far_d = x_fa.false_alarm_rate - c_fa.false_alarm_rate

    # A quieter schedule must not buy its quiet by leaving an unresolved HIGH
    # unannounced for longer. Without this the endpoint is a tautology: burden is
    # monotone in the re-notification interval, so "never re-notify" wins every time
    # while being exactly the change a clinician would refuse.
    c_sil = c_ep.silent_high_hours_per_septic_episode
    x_sil = x_ep.silent_high_hours_per_septic_episode
    silent_worse = (np.isfinite(c_sil) and np.isfinite(x_sil) and x_sil > c_sil + 1e-9)

    if det_d < -1e-12:
        ok, why = False, "refused: first-alarm detection fell (%.4f)" % det_d
    elif far_d > 1e-12:
        ok, why = False, "refused: first-alarm false-alarm rate rose (%.4f)" % far_d
    elif silent_worse:
        ok, why = False, ("refused: burden falls only by going quiet — unannounced HIGH "
                          "hours per septic episode rise %.2f -> %.2f" % (c_sil, x_sil))
    elif hi >= 0.0:
        ok, why = False, ("refused: burden reduction is not separable from zero "
                          "(95%% CI %+.4f to %+.4f)" % (lo, hi))
    else:
        ok, why = True, ("promoted: burden reduction 95%% CI %+.4f to %+.4f excludes zero"
                         % (lo, hi))
    return BurdenVerdict(ok, why, champion.describe(), challenger.describe(),
                         float(c_ep.burden_per_patient.mean()),
                         float(x_ep.burden_per_patient.mean()),
                         d, lo, hi, float(det_d), float(far_d),
                         float(c_sil), float(x_sil), n_boot)


def episode_grid() -> List[EpisodePolicy]:
    """Only the re-notification interval is searched.

    It is the one schedule parameter ``AlertEpisodeController`` already exposes.
    Cooldown, resolve-hysteresis and episode-gap would each require a change to the
    live controller first; searching them here would measure an object the product
    does not have, which is the failure this module's docstring warns about.
    """
    return [EpisodePolicy(h) for h in
            (3600.0, 2 * 3600.0, 4 * 3600.0, 8 * 3600.0, 12 * 3600.0, None)]


# --------------------------------------------------------------------------- #
# search
# --------------------------------------------------------------------------- #
def default_grid(scores: np.ndarray) -> List[Policy]:
    """Candidate policies, with thresholds drawn from the score distribution itself
    so the grid is meaningful whatever the model's calibration happens to be."""
    qs = np.unique(np.quantile(scores, np.linspace(0.75, 0.999, 60)))
    out: List[Policy] = [Policy("plain", {"threshold": float(t)}) for t in qs]
    for t in qs[::4]:
        for k in (2, 3, 4, 6, 8):
            out.append(Policy("persistent", {"threshold": float(t), "k": int(k)}))
        for decay in (0.80, 0.90, 0.95, 1.0):
            for trig in (0.02, 0.05, 0.1, 0.2, 0.4, 0.8, 1.5):
                out.append(Policy("integrator", {"threshold": float(t),
                                                 "decay": float(decay),
                                                 "trigger": float(trig)}))
    for w in (6, 12, 24):
        for z in (2.0, 2.5, 3.0, 4.0, 5.0):
            for fl in (0.0, 0.02, 0.04, 0.06):
                out.append(Policy("per_patient", {"warmup": int(w), "z": float(z),
                                                  "floor": float(fl)}))
    return out


def search(patients: Sequence[PatientSeries], max_false_alarm_rate: float,
           grid: Optional[Iterable[Policy]] = None
           ) -> Tuple[Optional[Policy], Optional[PolicyResult], List[Tuple[Policy, PolicyResult]]]:
    """Best detection subject to a false-alarm ceiling. Returns the winner and the
    full evaluated field, because the field is the evidence that the winner won."""
    scores = np.concatenate([p.scores for p in patients])
    cands = list(grid) if grid is not None else default_grid(scores)
    scored: List[Tuple[Policy, PolicyResult]] = []
    for pol in cands:
        r = evaluate(pol, patients)
        scored.append((pol, r))
    ok = [(p, r) for p, r in scored if r.false_alarm_rate <= max_false_alarm_rate + 1e-12]
    if not ok:
        return None, None, scored
    best = max(ok, key=lambda pr: (pr[1].detection_rate, -pr[1].false_alarm_rate))
    return best[0], best[1], scored


# --------------------------------------------------------------------------- #
# the gate
# --------------------------------------------------------------------------- #
@dataclass
class Verdict:
    promoted: bool
    reason: str
    champion: str
    challenger: str
    champion_detection: float
    challenger_detection: float
    champion_far: float
    challenger_far: float
    delta_detection: float
    ci_low: float
    ci_high: float
    n_boot: int


def gate(champion: Policy, challenger: Policy, locked: Sequence[PatientSeries],
         n_boot: int = 2000, seed: int = 0) -> Verdict:
    """Promote only on evidence, measured on patients the search never saw.

    Paired bootstrap over PATIENTS (not hours) — hours within an admission are
    heavily correlated, so resampling them would manufacture significance. Both
    policies are re-scored on every identical resample, so the interval is on the
    DIFFERENCE and the patient mix cancels.

    Two conditions, both required: the challenger may not raise the false-alarm
    rate, and the 95% interval on the detection gain must exclude zero. Anything
    else is refused, including a challenger that merely looks better.
    """
    c_res = evaluate(champion, locked)
    x_res = evaluate(challenger, locked)
    rng = np.random.default_rng(seed)

    n_s, n_n = c_res.n_septic, c_res.n_nonseptic
    deltas = np.empty(n_boot)
    for b in range(n_boot):
        si = rng.integers(0, n_s, n_s)
        deltas[b] = x_res.fired_septic[si].mean() - c_res.fired_septic[si].mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    d = x_res.detection_rate - c_res.detection_rate

    if x_res.false_alarm_rate > c_res.false_alarm_rate + 1e-12:
        ok, why = False, "refused: challenger raises the false-alarm rate"
    elif lo <= 0.0:
        ok, why = False, ("refused: detection gain is not separable from zero "
                          f"(95% CI {lo:+.3f} to {hi:+.3f})")
    else:
        ok, why = True, f"promoted: detection gain 95% CI {lo:+.3f} to {hi:+.3f} excludes zero"

    return Verdict(promoted=ok, reason=why,
                   champion=champion.describe(), challenger=challenger.describe(),
                   champion_detection=c_res.detection_rate,
                   challenger_detection=x_res.detection_rate,
                   champion_far=c_res.false_alarm_rate,
                   challenger_far=x_res.false_alarm_rate,
                   delta_detection=d, ci_low=float(lo), ci_high=float(hi),
                   n_boot=n_boot)


def save(verdict: Verdict, shipped: Policy, path: Path = ARTIFACT, *,
         families: Optional[Sequence[str]] = None,
         verdict_note: Optional[str] = None,
         operating_point: Optional[Dict[str, object]] = None,
         fit: Optional[Dict[str, object]] = None,
         episode_semantics: Optional[Dict[str, object]] = None,
         episode_metrics: Optional[Dict[str, object]] = None,
         burden_verdict: Optional[Dict[str, object]] = None,
         candidate_field: Optional[Sequence[Dict[str, object]]] = None) -> None:
    """Write the verdict, the field that produced it, and the fit it belongs to.

    Everything goes through ``json_safe``: an empty episode set makes a median NaN,
    ``json.dump`` writes that as bare ``NaN``, and the website's sync script parses
    this file with a strict parser that rejects it. That exact failure has already
    happened once in this repo, to a different artifact.

    ``shipped_policy`` and ``verdict`` keep their historic shape — the live console
    and the public site read those two objects by name.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "shipped_policy": {"family": shipped.family, "params": shipped.params,
                           "describe": shipped.describe()},
        "verdict": verdict.__dict__,
        "families": list(families) if families else None,
        "verdict_note": verdict_note,
        "operating_point": operating_point,
        "fit": fit,
        "episode_semantics": episode_semantics,
        "episode_metrics": episode_metrics,
        "burden_verdict": burden_verdict,
        "candidate_field": list(candidate_field) if candidate_field else None,
        "lead_time": {"reported": False,
                      "reason": "retracted — a random score reproduces it; "
                                "see docs/LEAD_TIME_NULL.md"},
    }
    path.write_text(json.dumps(json_safe(payload), indent=2))
