"""Self-improving ALARM POLICY search, with a gate that refuses by default.

The detector produces a probability per patient-hour. Turning that stream into a
bedside alarm is a separate decision — and until now that decision was a single
scalar threshold, chosen once and never searched. This module makes the alarm
policy itself a searchable object and puts it behind the same evidence bar the
model search uses ([[self-improvement-loop]]): a locked holdout the search never
sees, paired bootstrap over PATIENTS, and promotion only when the confidence
interval on the improvement excludes zero.

WHAT THE SEARCH FOUND (2026-08-03, CinC 2019, wearable channels)
---------------------------------------------------------------
Four policy families were searched at or below the shipped false-alarm rate:

    plain threshold (shipped)        FA 10.6%   detection 38.1%
    K consecutive hours above thr    FA 10.6%   detection 38.1%   (best K = 1)
    leaky exceedance integrator      FA 10.6%   detection 38.1%   (collapsed to
                                                                   a plain threshold)
    per-patient z against own stay   FA  9.0%   detection 32.8%

None of them beats a plain threshold. The integrator result is the informative
one: the search was free to pick any decay and trigger level and it chose
decay = 1.0 with a trigger so small that the policy reduces to "first hour above
a threshold". Given the freedom to be cleverer, the optimum was to not be.

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


def save(verdict: Verdict, shipped: Policy, path: Path = ARTIFACT) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "shipped_policy": {"family": shipped.family, "params": shipped.params,
                           "describe": shipped.describe()},
        "verdict": verdict.__dict__,
    }, indent=2))
