"""Recursive self-improvement for the sepsis detector — a search loop that tunes
its own champion, and its own step size, under a protocol built to catch itself
lying.

WHAT "RECURSIVE" MEANS HERE (and what it does not)
--------------------------------------------------
Each round mutates the *current champion* — the winner of the previous round is the
seed of the next, so the search distribution walks toward its own best result rather
than sampling a fixed grid. A second, slower recursion sits on top: the mutation
step size is itself adapted from the loop's own hit-rate (shrink after a failed
round, re-expand after a win), so the search tunes the search. That is a
self-adaptive (1+lambda) evolution strategy.

It is NOT a model that rewrites its own code, and it cannot conjure signal that is
not in the data. The honest prior from this project's own history (docs/PHASE7_REPORT.md,
backend/ml/README.md) is that sepsis detection here is DATA-limited, not model-limited:
within-site AUROC plateaus ~0.77, cross-hospital ~0.69, and adding labs buys +0.02.
A tuning loop should therefore be *expected* to find little or nothing. This module is
built so that "nothing" is a first-class, reportable outcome — not something the search
can paper over.

WHY THE PROTOCOL LOOKS PARANOID
-------------------------------
A self-improvement loop's failure mode is not "fails to improve", it is
"improves its own scoreboard". Three mechanisms exist purely to prevent that:

1. **A locked gate set.** Patients are split off BEFORE the search starts and are
   never scored, never selected on, and never fitted during any round. The champion
   meets them exactly once, at the end. Search-set gains are self-reported; the gate
   number is the only one that has any claim on reality.

2. **Paired folds + a corrected significance test.** Every candidate is scored on the
   *identical* patient-grouped folds as the champion, so candidate-vs-champion is a
   paired comparison (removes fold-difficulty variance, the dominant noise term at
   this sample size). Because each round takes the best of ``population`` challengers
   and then tests it, the winner's p-value is optimistically biased by that
   max-selection; the acceptance threshold is Bonferroni-corrected by the population
   size to compensate. A candidate must clear BOTH a minimum effect size and the
   corrected test — beating the champion on the mean alone is not enough.

3. **A ledger of everything.** Every trial is recorded with its config, per-fold
   scores, delta, p-value, and the reason it was accepted or rejected. Rejected
   trials are the majority of the scientific content and are kept verbatim; a run
   that accepts nothing still produces a full, publishable record.

Patient-grouping is enforced throughout (rows from one patient never straddle a
split) because without it every number here becomes a fantasy — see
backend/ml/eval/splits.py.

Usage: see backend/ml/scripts/self_improve.py.
"""
from __future__ import annotations

import copy
import json
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from backend.ml.eval import metrics
from backend.ml.eval.splits import stratified_group_folds, stratified_group_split
from backend.ml.models.stacked import (
    DEFAULT_PARAMS,
    DEFAULT_ROSTER,
    StackedEnsembleDetector,
)

Config = Dict[str, object]


# --------------------------------------------------------------------------- #
# Search space
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Knob:
    """One tunable hyperparameter and the bounds the search may not leave.

    ``kind`` picks the mutation geometry: ``log`` knobs (learning rates, regularisation)
    are perturbed multiplicatively because their effect is scale-free; ``int``/``float``
    knobs are perturbed additively relative to their range. Bounds are not cosmetic —
    they keep the searched model shippable (e.g. ExtraTrees depth is capped because an
    unbounded fit serialises to ~300 MB, see backend/ml/models/stacked.py).
    """
    learner: str
    key: str
    lo: float
    hi: float
    kind: str  # "log" | "float" | "int"


SEARCH_SPACE: Tuple[Knob, ...] = (
    Knob("hgb", "learning_rate", 0.01, 0.30, "log"),
    Knob("hgb", "max_iter", 100, 600, "int"),
    Knob("hgb", "max_leaf_nodes", 15, 127, "int"),
    Knob("hgb", "l2_regularization", 0.01, 10.0, "log"),
    Knob("hgb2", "learning_rate", 0.02, 0.40, "log"),
    Knob("hgb2", "max_iter", 100, 500, "int"),
    Knob("hgb2", "max_leaf_nodes", 7, 63, "int"),
    Knob("hgb2", "max_depth", 2, 8, "int"),
    Knob("hgb2", "l2_regularization", 0.01, 5.0, "log"),
    Knob("hgb2", "max_features", 0.3, 1.0, "float"),
    Knob("et", "n_estimators", 60, 300, "int"),
    Knob("et", "max_depth", 5, 20, "int"),
    Knob("et", "min_samples_leaf", 10, 120, "int"),
    Knob("lr", "C", 0.01, 10.0, "log"),
    Knob("meta", "C", 0.05, 10.0, "log"),
)


def default_config() -> Config:
    """The incumbent: the shipped stack's roster and hyperparameters."""
    return {"roster": list(DEFAULT_ROSTER),
            "params": copy.deepcopy(DEFAULT_PARAMS)}


def _clip(v: float, k: Knob) -> float:
    v = min(max(v, k.lo), k.hi)
    return int(round(v)) if k.kind == "int" else float(v)


def config_diff(parent: Config, child: Config) -> List[str]:
    """What actually changed, in words.

    A trial recorded only its full resulting config, so reading the ledger told you
    the state a challenger ended in but never the edit that produced it — and with
    three knobs moving out of a dozen, the difference is most of the information.
    An accepted win you cannot attribute to a knob is not a finding, and a refusal
    you cannot attribute to one teaches nothing about where the search has been.
    """
    out: List[str] = []
    pr, cr = list(parent.get("roster", [])), list(child.get("roster", []))
    for learner in cr:
        if learner not in pr:
            out.append(f"roster +{learner}")
    for learner in pr:
        if learner not in cr:
            out.append(f"roster \u2212{learner}")
    pp: Dict[str, Dict[str, float]] = parent.get("params", {})   # type: ignore[assignment]
    cp: Dict[str, Dict[str, float]] = child.get("params", {})    # type: ignore[assignment]
    for learner in sorted(set(pp) | set(cp)):
        a, b = pp.get(learner, {}), cp.get(learner, {})
        for key in sorted(set(a) | set(b)):
            va, vb = a.get(key), b.get(key)
            if va is None or vb is None or va == vb:
                continue
            if isinstance(va, float) or isinstance(vb, float):
                if abs(float(va) - float(vb)) < 1e-12:
                    continue
                out.append(f"{learner}.{key} {float(va):.4g} \u2192 {float(vb):.4g}")
            else:
                out.append(f"{learner}.{key} {va} \u2192 {vb}")
    return out


def mutate(cfg: Config, scale: float, rng: random.Random,
           n_knobs: int = 3, roster_prob: float = 0.15) -> Config:
    """Perturb a config to produce a challenger.

    ``scale`` is the step size the loop adapts from its own hit-rate. Only ``n_knobs``
    knobs move at a time: full-config randomisation would make an accepted win
    uninterpretable (you would not know which knob earned it) and wastes the paired
    test's power on noise.

    A roster edit (dropping/restoring a base learner) fires with low probability and is
    kept structurally legal: at least two learners must survive, or there is nothing
    left to stack.
    """
    out = copy.deepcopy(cfg)
    params: Dict[str, Dict[str, float]] = out["params"]  # type: ignore[assignment]
    roster: List[str] = out["roster"]  # type: ignore[assignment]

    # only mutate knobs whose learner is actually in the roster (plus the meta-learner,
    # which is always present) — tuning a dropped learner is a wasted evaluation.
    live = [k for k in SEARCH_SPACE if k.learner in roster or k.learner == "meta"]
    for k in rng.sample(live, min(n_knobs, len(live))):
        cur = float(params.get(k.learner, {}).get(k.key, DEFAULT_PARAMS[k.learner][k.key]))
        if k.kind == "log":
            nxt = cur * float(np.exp(rng.gauss(0.0, scale)))
        else:
            nxt = cur + rng.gauss(0.0, scale) * (k.hi - k.lo) * 0.5
        params.setdefault(k.learner, {})[k.key] = _clip(nxt, k)

    if rng.random() < roster_prob:
        pool = list(DEFAULT_ROSTER)
        pick = rng.choice(pool)
        if pick in roster and len(roster) > 2:
            roster.remove(pick)
        elif pick not in roster:
            roster.append(pick)
        out["roster"] = [r for r in DEFAULT_ROSTER if r in roster]  # canonical order
    return out


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def paired_t_one_sided(deltas: np.ndarray) -> float:
    """One-sided paired t-test p-value for H1: mean(delta) > 0.

    Falls back to a conservative p=1.0 when the deltas carry no usable variance
    (identical scores, or a single fold) rather than reporting a spurious p=0.
    """
    d = np.asarray(deltas, dtype=float)
    n = d.size
    if n < 2:
        return 1.0
    sd = float(d.std(ddof=1))
    if sd <= 1e-12:
        return 1.0 if d.mean() <= 0 else 0.0
    t = float(d.mean()) / (sd / np.sqrt(n))
    try:
        from scipy import stats
        return float(stats.t.sf(t, df=n - 1))
    except Exception:                                    # pragma: no cover
        # normal approximation if scipy is unavailable (conservative enough here)
        from math import erfc, sqrt
        return float(0.5 * erfc(t / sqrt(2.0)))


def bootstrap_delta_ci(y: np.ndarray, pa: np.ndarray, pb: np.ndarray,
                       groups: np.ndarray, n_boot: int = 400, seed: int = 0,
                       alpha: float = 0.05,
                       return_deltas: bool = False) -> Tuple[float, ...]:
    """Patient-level bootstrap CI for AUROC(b) - AUROC(a).

    Resamples PATIENTS, not rows: rows within a patient are correlated, so a row
    bootstrap would understate the interval and manufacture false confidence — the
    same reason the splits are patient-grouped.
    """
    rng = np.random.default_rng(seed)
    uniq = np.array(sorted(set(groups.tolist())))
    idx_by_g = metrics._rows_by_patient(groups, uniq)
    deltas: List[float] = []
    for _ in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[g] for g in pick])
        yy = y[idx]
        if len(np.unique(yy)) < 2:
            continue
        deltas.append(metrics.auroc(yy, pb[idx]) - metrics.auroc(yy, pa[idx]))
    if not deltas:
        if return_deltas:
            return float("nan"), float("nan"), float("nan"), np.empty(0)
        return float("nan"), float("nan"), float("nan")
    arr = np.array(deltas)
    out = (float(arr.mean()),
           float(np.quantile(arr, alpha / 2)),
           float(np.quantile(arr, 1 - alpha / 2)))
    # The resample deltas themselves, for callers that need a p-value rather than an
    # interval endpoint. At a tight alpha the endpoint is an extreme order statistic and
    # stops being estimable — alpha/2 = 0.0016 against 400 resamples asks for order
    # statistic 0.6, which is the sample minimum wearing a percentile's name. A one-sided
    # bootstrap p-value is resolvable to 1/(n+1) and does not have that failure mode.
    return (out + (arr,)) if return_deltas else out


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
@dataclass
class Trial:
    trial_id: int
    round_idx: int
    config: Config
    fold_scores: List[float]
    mean_auroc: float
    delta_vs_champion: float
    p_value: float
    accepted: bool
    reason: str
    seconds: float
    mutation: List[str] = field(default_factory=list)   # what this trial changed


@dataclass
class SelfImproveResult:
    baseline_config: Config
    champion_config: Config
    baseline_search_auroc: float
    champion_search_auroc: float
    trials: List[Trial] = field(default_factory=list)
    gate: Dict[str, object] = field(default_factory=dict)
    promoted: bool = False
    verdict: str = ""

    def to_json(self) -> Dict[str, object]:
        d = asdict(self)
        d["trials"] = [asdict(t) if not isinstance(t, dict) else t for t in self.trials]
        return d


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class RecursiveSelfImprover:
    """Self-adaptive (1+lambda) search over the stacked detector's configuration.

    The data is cut once into a search set and a locked gate set; every round then
    (a) mutates the current champion ``population`` times, (b) scores each challenger
    on the champion's own folds, (c) promotes the best challenger only if it clears a
    minimum effect size AND a Bonferroni-corrected paired t-test, and (d) adapts the
    step size from whether the round hit or missed. The champion faces the gate set
    once, at the end.
    """

    def __init__(self,
                 feats: pd.DataFrame,
                 y: np.ndarray,
                 groups: np.ndarray,
                 hour_index: Optional[np.ndarray] = None,
                 onset_hour: Optional[np.ndarray] = None,
                 channels: Optional[Sequence[str]] = None,
                 *,
                 rounds: int = 4,
                 population: int = 4,
                 folds: int = 3,
                 gate_fraction: float = 0.2,
                 min_gain: float = 0.005,
                 alpha: float = 0.05,
                 init_scale: float = 0.35,
                 min_scale: float = 0.05,
                 use_hdc: bool = False,
                 seed: int = 42,
                 log: Optional[Callable[[str], None]] = None):
        self.feats, self.y, self.groups = feats, np.asarray(y).astype(int), np.asarray(groups)
        self.hour_index, self.onset_hour = hour_index, onset_hour
        self.channels = list(channels or [])
        self.rounds, self.population, self.folds_n = rounds, population, folds
        self.min_gain, self.alpha = min_gain, alpha
        self.scale, self.min_scale = init_scale, min_scale
        self.use_hdc, self.seed = use_hdc, seed
        self.log = log or (lambda m: print(m, flush=True))
        self.rng = random.Random(seed)

        # --- the locked gate: carved out BEFORE anything is searched ---------- #
        s_idx, g_idx = stratified_group_split(self.groups, self.y, gate_fraction, seed)
        self.search_idx, self.gate_idx = s_idx, g_idx
        assert not (set(self.groups[s_idx].tolist()) & set(self.groups[g_idx].tolist())), \
            "gate/search patient overlap — the gate would be contaminated"

        sy, sg = self.y[s_idx], self.groups[s_idx]
        # Precompute folds ONCE so every candidate is scored on identical splits;
        # that pairing is what makes candidate-vs-champion deltas comparable.
        self.folds = [(a, b) for a, b in stratified_group_folds(sg, sy, folds, seed)]
        self._trials: List[Trial] = []
        self._next_id = 0

    # ------------------------------------------------------------------ #
    def _fit_predict(self, cfg: Config, tr: np.ndarray, te: np.ndarray,
                     feats: pd.DataFrame, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
        m = StackedEnsembleDetector(
            channels=list(self.channels), use_hdc=self.use_hdc, seed=self.seed,
            roster=list(cfg["roster"]), params=copy.deepcopy(cfg["params"]),  # type: ignore[arg-type]
        )
        m.fit(feats.iloc[tr], y[tr], groups=groups[tr])
        return m.predict_proba(feats.iloc[te])

    def _cv_scores(self, cfg: Config) -> np.ndarray:
        sf = self.feats.iloc[self.search_idx].reset_index(drop=True)
        sy, sg = self.y[self.search_idx], self.groups[self.search_idx]
        out = []
        for tr, te in self.folds:
            p = self._fit_predict(cfg, tr, te, sf, sy, sg)
            out.append(metrics.auroc(sy[te], p))
        return np.array(out, dtype=float)

    # ------------------------------------------------------------------ #
    def run(self) -> SelfImproveResult:
        t0 = time.time()
        base_cfg = default_config()
        self.log("Scoring the incumbent (shipped) configuration on the search folds...")
        champ_scores = self._cv_scores(base_cfg)
        champ_cfg = copy.deepcopy(base_cfg)
        base_mean = float(champ_scores.mean())
        self.log(f"  incumbent search AUROC {base_mean:.4f} "
                 f"(folds {np.round(champ_scores, 4).tolist()})")

        # Bonferroni: each round tests the BEST of `population` challengers, so the
        # winner's p-value is max-selection-biased. Correcting by the population size
        # keeps the per-round false-accept rate near `alpha`.
        alpha_c = self.alpha / max(self.population, 1)
        self.log(f"Acceptance: delta > {self.min_gain} AND paired p < {alpha_c:.4f} "
                 f"(alpha {self.alpha} / {self.population} challengers)")

        for r in range(1, self.rounds + 1):
            self.log(f"\n--- round {r}/{self.rounds} (step scale {self.scale:.3f}) ---")
            round_trials: List[Tuple[Trial, np.ndarray]] = []
            for _ in range(self.population):
                cfg = mutate(champ_cfg, self.scale, self.rng)
                ts = time.time()
                sc = self._cv_scores(cfg)
                d = sc - champ_scores                      # paired, same folds
                p = paired_t_one_sided(d)
                tr = Trial(trial_id=self._next_id, round_idx=r, config=cfg,
                           fold_scores=[float(x) for x in sc], mean_auroc=float(sc.mean()),
                           delta_vs_champion=float(d.mean()), p_value=float(p),
                           accepted=False, reason="", seconds=round(time.time() - ts, 1),
                           mutation=config_diff(champ_cfg, cfg))
                self._next_id += 1
                round_trials.append((tr, sc))
                self.log(f"  trial {tr.trial_id}: AUROC {tr.mean_auroc:.4f} "
                         f"(delta {tr.delta_vs_champion:+.4f}, p={p:.3f}, {tr.seconds}s)")

            # Judge EVERY challenger, not just the one with the biggest delta.
            #
            # The old rule picked the round's leader by delta and then tested that
            # one candidate's p-value. Those two orderings disagree: a smaller but
            # far more consistent gain can carry much stronger evidence than a large
            # noisy one. In the first real run, trial 5 (delta +0.0156, p=0.21) was
            # selected over trial 1 (delta +0.0037, p=0.07) — the strongest evidence
            # in the round was discarded for having a smaller mean, and 12 of 16
            # evaluations produced no verdict at all despite being fully paid for.
            #
            # Every trial is already scored on the same folds, so testing all of them
            # costs nothing extra. alpha_c is Bonferroni over the population, which is
            # exactly the correction that testing `population` candidates requires —
            # it was already being applied to the single selected one, so the bar is
            # unchanged. Among those clearing BOTH bars, the largest delta wins.
            for tr, _ in round_trials:
                if tr.delta_vs_champion <= self.min_gain:
                    tr.reason = (f"delta {tr.delta_vs_champion:+.4f} <= min_gain "
                                 f"{self.min_gain} (effect too small to be worth a change)")
                elif tr.p_value >= alpha_c:
                    tr.reason = (f"delta {tr.delta_vs_champion:+.4f} but p={tr.p_value:.3f} "
                                 f">= corrected alpha {alpha_c:.4f} (not distinguishable "
                                 f"from fold noise)")
                else:
                    tr.accepted = True
                    tr.reason = (f"delta {tr.delta_vs_champion:+.4f} > {self.min_gain} "
                                 f"and p={tr.p_value:.3f} < {alpha_c:.4f}")

            passing = [(tr, sc) for tr, sc in round_trials if tr.accepted]
            if passing:
                best_tr, best_sc = max(passing, key=lambda x: x[0].delta_vs_champion)
                # only one config can seed the next round; the rest are recorded as
                # having cleared the bar but lost the round, which is a real outcome
                # and not the same as never having been tested.
                for tr, _ in passing:
                    if tr is not best_tr:
                        tr.accepted = False
                        tr.reason = (f"cleared the bar ({tr.reason}) but a stronger "
                                     f"challenger won the round")
            else:
                best_tr, best_sc = max(round_trials, key=lambda x: x[0].delta_vs_champion)

            for tr, _ in round_trials:
                self._trials.append(tr)

            if best_tr.accepted:
                # RECURSION: the winner becomes the seed the next round mutates.
                champ_cfg, champ_scores = copy.deepcopy(best_tr.config), best_sc
                self.scale = min(self.scale * 1.15, 1.0)   # a hit -> explore a little wider
                self.log(f"  ACCEPTED trial {best_tr.trial_id}: {best_tr.reason}")
            else:
                # A miss means the step is likely overshooting the local optimum;
                # contract it so later rounds probe finer. This is the loop adapting
                # its own search — the second-order recursion.
                self.scale = max(self.scale * 0.7, self.min_scale)
                self.log(f"  rejected: {best_tr.reason}")

        champ_mean = float(champ_scores.mean())
        res = SelfImproveResult(
            baseline_config=base_cfg, champion_config=champ_cfg,
            baseline_search_auroc=base_mean, champion_search_auroc=champ_mean,
            trials=self._trials)
        self.log(f"\nSearch done in {time.time()-t0:.0f}s. "
                 f"Search-set AUROC {base_mean:.4f} -> {champ_mean:.4f} "
                 f"({champ_mean-base_mean:+.4f}) — self-reported, NOT yet trustworthy.")
        self._gate(res)
        return res

    # ------------------------------------------------------------------ #
    def _gate(self, res: SelfImproveResult) -> None:
        """Confront the champion with the locked gate patients — once.

        Everything before this point selected on the search folds, so the search-set
        gain is contaminated by selection. These patients were never scored, never
        fitted, never chosen on; this delta is the only honest one.
        """
        same = res.champion_config == res.baseline_config
        sf = self.feats.iloc[self.search_idx].reset_index(drop=True)
        sy, sg = self.y[self.search_idx], self.groups[self.search_idx]
        gf = self.feats.iloc[self.gate_idx].reset_index(drop=True)
        gy, gg = self.y[self.gate_idx], self.groups[self.gate_idx]

        if same:
            res.gate = {"skipped": True,
                        "why": "search accepted nothing; champion IS the incumbent, "
                               "so there is no change to gate"}
            res.promoted = False
            res.verdict = ("NO CHANGE — the search could not beat the shipped configuration. "
                           "Consistent with this project's data-limited prior.")
            self.log("\n=== GATE ===\n" + res.verdict)
            return

        self.log("\n=== GATE (locked patients, never searched) ===")
        self.log("Refitting incumbent and champion on the full search set...")

        # Both models are fitted on the SAME search patients and predict the SAME gate
        # patients, so the only difference between them is the configuration.
        combined = pd.concat([sf, gf], ignore_index=True)
        cy, cg = np.concatenate([sy, gy]), np.concatenate([sg, gg])
        tr_idx = np.arange(len(sy))
        te_idx = np.arange(len(sy), len(sy) + len(gy))
        p_base = self._fit_predict(res.baseline_config, tr_idx, te_idx, combined, cy, cg)
        p_champ = self._fit_predict(res.champion_config, tr_idx, te_idx, combined, cy, cg)

        a_base, a_champ = metrics.auroc(gy, p_base), metrics.auroc(gy, p_champ)
        mean_d, lo, hi = bootstrap_delta_ci(gy, p_base, p_champ, gg, seed=self.seed)
        res.gate = {
            "skipped": False,
            "gate_patients": int(len(set(gg.tolist()))),
            "gate_rows": int(len(gy)),
            "incumbent_auroc": a_base,
            "champion_auroc": a_champ,
            "delta": float(a_champ - a_base),
            "bootstrap_mean_delta": mean_d,
            "bootstrap_ci95": [lo, hi],
            "incumbent_auprc": metrics.auprc(gy, p_base),
            "champion_auprc": metrics.auprc(gy, p_champ),
        }
        self.log(f"  incumbent gate AUROC {a_base:.4f} | champion {a_champ:.4f} "
                 f"| delta {a_champ - a_base:+.4f}")
        self.log(f"  bootstrap delta {mean_d:+.4f} (95% CI {lo:+.4f}..{hi:+.4f})")

        # Promote only on a real effect whose CI excludes zero. A CI straddling zero
        # means the gain is indistinguishable from patient sampling — exactly the
        # illusion this whole protocol exists to catch.
        gain = a_champ - a_base
        if gain <= self.min_gain:
            res.promoted = False
            res.verdict = (f"REJECTED — gate gain {gain:+.4f} <= min_gain {self.min_gain}. "
                           f"The search-set gain "
                           f"({res.champion_search_auroc - res.baseline_search_auroc:+.4f}) "
                           f"did NOT survive on locked patients: it was selection noise.")
        elif not (lo > 0):
            res.promoted = False
            res.verdict = (f"REJECTED — gate gain {gain:+.4f} but the 95% bootstrap CI "
                           f"({lo:+.4f}..{hi:+.4f}) includes zero, so it is not "
                           f"distinguishable from patient sampling.")
        else:
            res.promoted = True
            res.verdict = (f"CONFIRMED — gate gain {gain:+.4f} with 95% CI "
                           f"({lo:+.4f}..{hi:+.4f}) excluding zero. Survives on patients "
                           f"the search never touched.")
        self.log("  " + res.verdict)


# --------------------------------------------------------------------------- #
def save_ledger(res: SelfImproveResult, path: str) -> None:
    """Write the full run — accepted AND rejected trials — as JSON.

    The rejects are the point: they are the record of what was tried and failed, which
    is what makes a no-gain outcome a result rather than a silence.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as fh:
        json.dump(res.to_json(), fh, indent=2, default=str)

