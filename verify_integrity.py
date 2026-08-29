"""Adversarial integrity audit — verify the CLAIMS, not the code.

`verify_full_system.py` answers "does the pipeline run and produce sensible output".
This suite answers a harder and, for this project, more important question:

    when we feed the honesty machinery a known lie, does it catch it?

Everything this project stands on is a *negative* claim — "no leakage", "no admissible
gain", "the operating point is derived, not fitted to the test". Negative claims are not
provable by a green functional test; they are only ever falsifiable. So each check here
INJECTS a specific pathology and asserts the machinery reacts to it. A check passes only
if the guard fires on the planted problem — a suite of tripwires, not smoke tests.

Design rules that make this trustworthy rather than theatre:
  * Real cached CinC data (a small patient set, kept fast), never synthetic signal —
    except where the test is deliberately about statistics on constructed vectors.
  * Every expensive model fit is patient-grouped; a check that leaked would itself be a
    lie, which is the one thing this file may not be.
  * Each check is guarded so one failure does not abort the audit, and prints WHAT it
    planted and WHAT it expected to happen — the reasoning is legible, not a bare assert.

Run:  python verify_integrity.py
"""
from __future__ import annotations

import copy
import json
import os
import sys
import traceback

import numpy as np
import pandas as pd

from backend.ml.data.cinc2019 import load_patients
from backend.ml.data.features import FeatureConfig, build_dataset_cached
from backend.ml.eval import metrics
from backend.ml.eval.splits import stratified_group_split, stratified_group_folds
from backend.ml.models.stacked import StackedEnsembleDetector
from backend.ml import operating_point as opmod
from backend.ml.selfimprove import (
    paired_t_one_sided,
    bootstrap_delta_ci,
    RecursiveSelfImprover,
    default_config,
    mutate,
)

LIMIT = 600           # small enough to run in a couple of minutes, real patients
SEED = 42


def _ok(msg): print(f"      [PASS] {msg}")
def _detail(msg): print(f"             {msg}")


# --------------------------------------------------------------------------- #
# Shared fixture: one honest fit, reused across the model-based checks.
# --------------------------------------------------------------------------- #
class Fixture:
    def __init__(self):
        cfg = FeatureConfig.wearable_only()
        ds = build_dataset_cached(lambda: load_patients(limit=LIMIT), cfg,
                                  f"wearable_limit{LIMIT}")
        self.cfg = cfg
        self.feats = pd.DataFrame(ds.X, columns=ds.feature_names)
        self.y, self.groups = ds.y, ds.groups
        self.hour, self.onset = ds.hour_index, ds.onset_hour

        tr, te = stratified_group_split(self.groups, self.y, 0.25, SEED)
        self.tr, self.te = tr, te
        self.ftr = self.feats.iloc[tr].reset_index(drop=True)
        self.ytr, self.gtr = self.y[tr], self.groups[tr]
        self.htr, self.otr = self.hour[tr], self.onset[tr]

        # The detector this audit reasons about must be the one that SHIPPED, read out
        # of the manifest rather than written down a second time here. This used to be
        # hard-coded to the stack's constructor defaults, and check [6] — the one that
        # asserts the shipped threshold still beats the old hand-set guess — was
        # therefore evaluating that threshold on a DIFFERENT model's scores. It passed
        # only for as long as the defaults happened to equal what was deployed, and it
        # failed the moment they diverged. That is the same undetectable detector/
        # evidence mismatch this file exists to catch, one level up: an audit that
        # re-declares the configuration cannot audit a change to it.
        with open(os.path.join("artifacts", "detector_manifest.json")) as fh:
            _promoted = json.load(fh)["configs"]["candidate"]
        self.promoted = _promoted
        _mk = lambda: StackedEnsembleDetector(
            channels=list(cfg.channels), use_hdc=False,
            objective=_promoted.get("objective", "balanced"),
            roster=list(_promoted.get("roster") or []) or None)
        _fit_kw = dict(hour_index=None, onset_hour=None)

        # honest OOF over train + a deployed fit + held-out predictions, computed once.
        self.oof = np.zeros(len(self.ytr))
        for a, b in stratified_group_folds(self.gtr, self.ytr, 3, SEED):
            m = _mk().fit(self.ftr.iloc[a], self.ytr[a], groups=self.gtr[a],
                          hour_index=self.htr[a], onset_hour=self.otr[a])
            self.oof[b] = m.predict_proba(self.ftr.iloc[b])
        self.model = _mk().fit(self.ftr, self.ytr, groups=self.gtr,
                               hour_index=self.htr, onset_hour=self.otr)
        self.p_insample = self.model.predict_proba(self.ftr)          # over-sharp
        self.p_te = self.model.predict_proba(self.feats.iloc[te])
        self.yte = self.y[te]


# --------------------------------------------------------------------------- #
# 1. Leakage tripwire — destroy the signal, demand the score collapse.
# --------------------------------------------------------------------------- #
def check_label_shuffle_collapse(fx: Fixture) -> None:
    print("[1] Leakage tripwire — patient-grouped CV on SHUFFLED labels must collapse to chance")
    rng = np.random.default_rng(SEED)
    y_shuf = fx.ytr.copy()
    rng.shuffle(y_shuf)                                   # break every real relationship
    aurocs = []
    for a, b in stratified_group_folds(fx.gtr, y_shuf, 3, SEED):
        if len(np.unique(y_shuf[a])) < 2 or len(np.unique(y_shuf[b])) < 2:
            continue
        m = StackedEnsembleDetector(channels=list(fx.cfg.channels), use_hdc=False).fit(
            fx.ftr.iloc[a], y_shuf[a], groups=fx.gtr[a])
        aurocs.append(metrics.auroc(y_shuf[b], m.predict_proba(fx.ftr.iloc[b])))
    auc = float(np.mean(aurocs))
    _detail(f"planted: labels permuted (signal destroyed); observed CV AUROC = {auc:.3f}")
    assert 0.42 <= auc <= 0.58, (
        f"AUROC {auc:.3f} on shuffled labels is NOT chance — the model is memorising "
        f"something label-correlated that survives permutation, i.e. LEAKAGE")
    _ok(f"shuffled-label AUROC {auc:.3f} ~ 0.5: no leakage path detected")


# --------------------------------------------------------------------------- #
# 2. OOF necessity — prove the in-sample shortcut is a trap, not a convenience.
# --------------------------------------------------------------------------- #
def check_oof_necessity(fx: Fixture) -> None:
    print("[2] OOF necessity — in-sample predictions must be measurably over-sharp vs OOF")
    auc_in = metrics.auroc(fx.ytr, fx.p_insample)
    auc_oof = metrics.auroc(fx.ytr, fx.oof)
    u_in = metrics.normalized_utility(
        fx.ytr, (fx.p_insample >= 0.05).astype(int), fx.gtr, fx.htr)
    u_oof = metrics.normalized_utility(
        fx.ytr, (fx.oof >= 0.05).astype(int), fx.gtr, fx.htr)
    _detail(f"in-sample AUROC {auc_in:.3f} / utility {u_in:+.3f}  vs  "
            f"OOF AUROC {auc_oof:.3f} / utility {u_oof:+.3f}")
    assert auc_in - auc_oof > 0.03, (
        f"in-sample and OOF AUROC are within {auc_in-auc_oof:.3f} — if they barely "
        f"differ, either the model is underfit or the OOF split is contaminated; the "
        f"whole reason operating_point.py forbids in-sample thresholds is that this gap "
        f"is large")
    _ok(f"in-sample over-sharp by {auc_in-auc_oof:.3f} AUROC — OOF discipline is load-bearing, "
        f"not cosmetic")


# --------------------------------------------------------------------------- #
# 3. Gate rejects noise — a delta that is pure fold-difficulty must NOT promote.
# --------------------------------------------------------------------------- #
def check_gate_rejects_noise(_fx: Fixture) -> None:
    print("[3] Statistical gate — a zero-mean noise delta must be rejected")
    rng = np.random.default_rng(SEED)
    # a champion and a challenger that are statistically identical: their per-fold
    # difference is mean-zero noise. The gate must not mistake luck for a gain.
    deltas = rng.normal(0.0, 0.01, size=5)
    deltas -= deltas.mean()                              # exactly zero mean: pure noise
    p = paired_t_one_sided(deltas)
    _detail(f"planted: 5 folds of zero-mean delta; paired one-sided p = {p:.3f}")
    assert p > 0.20, f"p={p:.3f} on zero-mean noise would wave through a non-existent gain"
    # and a genuine, consistent improvement MUST clear it — the gate has to discriminate,
    # not just reject everything.
    real = np.array([0.02, 0.025, 0.018, 0.03, 0.022])
    p_real = paired_t_one_sided(real)
    _detail(f"planted: 5 folds of consistent +0.02 gain; paired p = {p_real:.4f}")
    assert p_real < 0.01, f"p={p_real:.4f} — the gate is so strict it rejects a real gain too"
    _ok(f"noise p={p:.3f} rejected, real p={p_real:.4f} accepted: the gate discriminates")


# --------------------------------------------------------------------------- #
# 4. Bootstrap CI honesty — identical models must yield a CI that includes zero.
# --------------------------------------------------------------------------- #
def check_bootstrap_ci_honesty(fx: Fixture) -> None:
    print("[4] Promotion CI — two IDENTICAL predictors must produce a CI straddling zero")
    mean_d, lo, hi = bootstrap_delta_ci(
        fx.yte, fx.p_te, fx.p_te.copy(), fx.groups[fx.te], n_boot=300, seed=SEED)
    _detail(f"planted: champion == incumbent; bootstrap delta {mean_d:+.4f} "
            f"CI [{lo:+.4f}, {hi:+.4f}]")
    assert lo <= 0 <= hi and abs(mean_d) < 1e-6, (
        f"identical predictors gave a CI that excludes zero ({lo:+.4f}..{hi:+.4f}) — the "
        f"promotion gate would fire on a model that changed nothing")
    # a genuinely better vector (nudge the positives up) must exclude zero.
    better = fx.p_te.copy()
    better[fx.yte == 1] = np.clip(better[fx.yte == 1] + 0.15, 0, 1)
    md2, lo2, hi2 = bootstrap_delta_ci(
        fx.yte, fx.p_te, better, fx.groups[fx.te], n_boot=300, seed=SEED)
    _detail(f"planted: positives boosted +0.15; bootstrap delta {md2:+.4f} "
            f"CI [{lo2:+.4f}, {hi2:+.4f}]")
    assert lo2 > 0, f"a real improvement's CI ({lo2:+.4f}..{hi2:+.4f}) failed to exclude zero"
    _ok(f"null CI straddles 0, real CI [{lo2:+.4f},{hi2:+.4f}] excludes 0: promotion gate is sound")


# --------------------------------------------------------------------------- #
# 5. Locked-gate integrity — the self-improver's gate patients are truly untouched.
# --------------------------------------------------------------------------- #
def check_locked_gate_disjoint(fx: Fixture) -> None:
    print("[5] Locked gate — search and gate patient sets must be provably disjoint")
    imp = RecursiveSelfImprover(
        fx.feats, fx.y, fx.groups, hour_index=fx.hour, onset_hour=fx.onset,
        channels=list(fx.cfg.channels), rounds=1, population=1, folds=3,
        gate_fraction=0.2, seed=SEED, log=lambda *_: None)
    search_pts = set(fx.groups[imp.search_idx].tolist())
    gate_pts = set(fx.groups[imp.gate_idx].tolist())
    overlap = search_pts & gate_pts
    # and no fold inside the search set may touch the gate either
    fold_pts = set()
    for a, b in imp.folds:
        fold_pts |= set(fx.groups[imp.search_idx][a].tolist())
        fold_pts |= set(fx.groups[imp.search_idx][b].tolist())
    _detail(f"{len(search_pts)} search patients, {len(gate_pts)} gate patients, "
            f"{len(overlap)} shared")
    assert not overlap, f"{len(overlap)} patients are in BOTH search and gate — gate contaminated"
    assert not (fold_pts & gate_pts), "a CV fold reaches into the gate patients"
    _ok(f"gate is disjoint from every search fold: no patient can leak search→gate")


# --------------------------------------------------------------------------- #
# 6. Shipped operating point — the artifact on disk must still be valid & in-range.
# --------------------------------------------------------------------------- #
def check_shipped_operating_point(fx: Fixture) -> None:
    print("[6] Shipped operating point — the on-disk artifact must beat the guess AND stay in range")
    op = opmod.load()
    assert op is not None, "artifacts/operating_point.json missing — AegisCore would fall back to guesses"
    _detail(f"loaded HIGH={op.high:.4f} MEDIUM={op.medium:.4f} "
            f"(p95={op.prob_p95:.3f} p99={op.prob_p99:.3f} max={op.prob_max:.3f})")
    # structural sanity
    assert 0 < op.medium < op.high, "MEDIUM must sit below HIGH (pre-alarm invariant)"
    # THE regression guard against the original bug. The shipped 0.12 sat at the 98.3rd
    # percentile of the model's output — i.e. ABOVE p95 (0.086), out in the sparse tail
    # where a calibrated 2%-prevalence detector almost never emits, and right at the edge
    # of the cliff where detection collapses. A valid alarm must sit at or below p95, in
    # the range where the output density actually lives. (Note: 0.12 is BELOW p99=0.140,
    # so a p99 guard would NOT have caught this bug — p95 is the honest tripwire.)
    assert op.high <= op.prob_p95, (
        f"HIGH {op.high:.4f} is above the model's p95 {op.prob_p95:.4f} — it sits in the "
        f"sparse output tail, the exact calibration-cliff failure (the old 0.12 lived at "
        f"the 98.3rd percentile) that this derivation exists to prevent")
    # THE second regression guard. The p95 rule above only compares the threshold to a
    # p95 recorded in the same file, so it is satisfied by any internally consistent
    # artifact — INCLUDING one whose evidence was measured on a different detector than
    # the one it governs. Nothing in the artifact used to say which detector that was,
    # so the mismatch was not merely undetected, it was undetectABLE: the one script
    # that could have installed such a point (derive_operating_point.py) defaulted to a
    # roster and objective that no longer matched the promoted one, and would have
    # written a perfectly self-consistent file. A threshold is a reading of one score
    # distribution; this asserts it is a reading of the RIGHT one.
    with open(os.path.join("artifacts", "detector_manifest.json")) as fh:
        promoted = json.load(fh)["configs"]["candidate"]
    expect = f"{promoted['objective']}:{'+'.join(promoted['roster'])}"
    _detail(f"artifact detector '{op.detector}' vs promoted '{expect}'")
    assert op.detector == expect, (
        f"the operating point was derived on detector '{op.detector}' but the promoted "
        f"detector is '{expect}' — every number in operating_point.json describes a model "
        f"that is not the one deployed, so the alarm threshold is unvalidated")
    # THE third regression guard, and the subtlest. This comparison used to be made
    # against ``fx`` — a stack REFITTED inside this audit on 600 patients. That is the
    # cross-fit threshold read the rest of the project forbids in writing: promote_
    # frontier.py records that holding a threshold fixed and varying only the fit size
    # moves the false-alarm rate 0.085 / 0.178 / 0.209 / 0.165 across fits on 750 /
    # 1500 / 2250 / 3000 patients — a 2.5x swing with no monotone trend to correct for.
    # An absolute threshold evaluated on a different fit than the one it was derived on
    # measures nothing, and this check passed for as long as it did only because the
    # proxy happened to resemble what shipped.
    #
    # So it now scores the ARTIFACT ON DISK — the model actually deployed — on the
    # holdout patients of the split it was promoted under, which are disjoint from its
    # training set by construction.
    import joblib
    from backend.ml.eval.bench import _load_cached
    art = os.path.join("artifacts", "gbm_wearable.joblib")
    assert os.path.exists(art), f"{art} missing — nothing is deployed to audit"
    shipped = joblib.load(art)
    ds8 = _load_cached("wearable_limit8000")
    s_tr, s_te = stratified_group_split(ds8.groups, ds8.y, 0.25, SEED)
    held = pd.DataFrame(ds8.X, columns=ds8.feature_names).iloc[s_te]
    p_held = shipped.predict_proba(held)
    _detail(f"scoring the DEPLOYED artifact on {len(set(ds8.groups[s_te].tolist())):,} "
            f"holdout patients it never trained on")
    u_new = metrics.normalized_utility(
        ds8.y[s_te], (p_held >= op.high).astype(int), ds8.groups[s_te], ds8.hour_index[s_te])
    u_old = metrics.normalized_utility(
        ds8.y[s_te], (p_held >= 0.12).astype(int), ds8.groups[s_te], ds8.hour_index[s_te])
    _detail(f"held-out utility: derived {u_new:+.4f}  vs  shipped-guess 0.12 {u_old:+.4f}")
    assert u_new >= u_old, (
        f"derived point {u_new:+.4f} no longer beats the old 0.12 guess {u_old:+.4f} on "
        f"held-out data — the artifact may be stale")
    _ok(f"artifact valid, in-range (<= p95), and beats the guess by {u_new-u_old:+.4f} utility")


# --------------------------------------------------------------------------- #
# 7. Threshold monotonicity — raising the alarm can only reduce detection.
# --------------------------------------------------------------------------- #
def check_detection_monotonic(fx: Fixture) -> None:
    print("[7] Monotonicity — detection must be non-increasing as the threshold rises")
    grid = [0.02, 0.05, 0.08, 0.12, 0.16, 0.20]
    det = [metrics.early_warning(fx.yte, fx.p_te, fx.groups[fx.te], fx.onset[fx.te],
                                 fx.hour[fx.te], t)["detection_rate"] for t in grid]
    _detail("thr  " + "  ".join(f"{t:.2f}" for t in grid))
    _detail("det  " + "  ".join(f"{d:.2f}" for d in det))
    for i in range(1, len(det)):
        assert det[i] <= det[i - 1] + 1e-9, (
            f"detection rose from {det[i-1]:.3f} to {det[i]:.3f} as threshold increased "
            f"{grid[i-1]}->{grid[i]} — the alarm logic is not monotone")
    _ok("detection decreases monotonically with threshold: alarm semantics are consistent")


# --------------------------------------------------------------------------- #
# 8. Mutation legality — the search can never emit an unshippable config.
# --------------------------------------------------------------------------- #
def check_mutation_stays_legal(_fx: Fixture) -> None:
    print("[8] Search safety — 2000 recursive mutations must never leave a shippable config")
    import random
    from backend.ml.selfimprove import SEARCH_SPACE
    bounds = {(k.learner, k.key): (k.lo, k.hi) for k in SEARCH_SPACE}
    rng = random.Random(SEED)
    cfg = default_config()
    for _ in range(2000):
        cfg = mutate(cfg, 0.6, rng)               # deliberately large step
        assert len(cfg["roster"]) >= 2, "roster underflowed below 2 base learners"
        for ln, d in cfg["params"].items():
            for key, v in d.items():
                if (ln, key) in bounds:
                    lo, hi = bounds[(ln, key)]
                    assert lo - 1e-9 <= v <= hi + 1e-9, f"{ln}.{key}={v} escaped [{lo},{hi}]"
    _detail(f"final roster {cfg['roster']}; all knobs within bounds after 2000 steps")
    _ok("no reachable mutation produces an out-of-bounds or degenerate model")


# --------------------------------------------------------------------------- #
# 9. Subgroup-audit claims — a published disparity must survive its own arithmetic.
# --------------------------------------------------------------------------- #
def check_subgroup_claims(_fx: "Fixture") -> None:
    """The lie this catches is a *fabricated fairness finding*.

    Subgroup results are the easiest numbers in the project to overstate: split a
    cohort eight ways, test eight metrics, and something will look damning by
    chance. So this guard does not re-run the audit — it interrogates the claim
    file the audit publishes, the way a reviewer would, and asserts the internal
    arithmetic of every claim holds:

      * a gap marked significant must have an ADJUSTED interval excluding zero,
      * the correction must actually be Bonferroni over the pre-specified family,
      * a post-hoc selected comparison must never be marked significant,
      * an underpowered cell must never appear on either side of a claim.

    Then it plants the lie — flips one straddling-zero gap to significant — and
    asserts the same guard rejects it.
    """
    print("[9] Subgroup claims — a published disparity must survive its own arithmetic")
    path = os.path.join("artifacts", "subgroup_audit.json")
    if not os.path.exists(path):
        _detail(f"planted: nothing — {path} absent; run `make subgroup-audit` first")
        _ok("skipped (no published subgroup claims to audit)")
        return
    with open(path) as fh:
        rep = json.load(fh)

    def violations(r):
        """Every way a claim in this report contradicts its own numbers."""
        bad = []
        small = {c["name"] for c in r["cells"] if c["metrics"].get("underpowered")}
        fam = [g for g in r["gaps"] if not g.get("selected")]
        expect_alpha = 0.05 / max(len(fam), 1)
        for g in r["gaps"]:
            if g.get("significant_adj"):
                if not (g["ci_lo_adj"] > 0 or g["ci_hi_adj"] < 0):
                    bad.append(f"{g['metric']} {g['a']}-{g['b']}: significant but "
                               f"adjusted CI [{g['ci_lo_adj']:+.4f},{g['ci_hi_adj']:+.4f}] "
                               f"straddles zero")
                if g.get("selected"):
                    bad.append(f"{g['metric']} {g['a']}-{g['b']}: post-hoc selected cell "
                               f"claimed as significant")
                if g["a"] in small or g["b"] in small:
                    bad.append(f"{g['metric']} {g['a']}-{g['b']}: claim rests on an "
                               f"underpowered cell")
            if abs(g.get("alpha_adj", expect_alpha) - expect_alpha) > 1e-12:
                bad.append(f"{g['metric']} {g['a']}-{g['b']}: alpha {g.get('alpha_adj')} "
                           f"!= 0.05/{len(fam)}")
        return bad

    found = violations(rep)
    n_sig = sum(1 for g in rep["gaps"] if g.get("significant_adj"))
    _detail(f"planted: nothing yet — auditing {len(rep['gaps'])} published gaps "
            f"({n_sig} claimed significant) for self-consistency")
    assert not found, "the published subgroup report contradicts itself: " + "; ".join(found)

    # Now the lie: promote a gap whose interval straddles zero.
    forged = copy.deepcopy(rep)
    victim = next((g for g in forged["gaps"]
                   if not g.get("selected") and g["ci_lo_adj"] <= 0 <= g["ci_hi_adj"]), None)
    assert victim is not None, ("no straddling gap available to forge — cannot prove the "
                               "guard fires, so this check would be vacuous")
    victim["significant_adj"] = True
    caught = violations(forged)
    _detail(f"planted: '{victim['metric']} {victim['a']} - {victim['b']}' "
            f"[{victim['ci_lo_adj']:+.4f},{victim['ci_hi_adj']:+.4f}] forged to significant")
    assert caught, ("a gap whose adjusted interval includes zero was accepted as a "
                    "significant disparity — the fairness claims are unguarded")
    _ok(f"{len(rep['gaps'])} published gaps self-consistent; forged claim rejected "
        f"({caught[0].split(':')[-1].strip()})")


# --------------------------------------------------------------------------- #
CHECKS = [
    ("model", check_label_shuffle_collapse),
    ("model", check_oof_necessity),
    ("pure",  check_gate_rejects_noise),
    ("model", check_bootstrap_ci_honesty),
    ("model", check_locked_gate_disjoint),
    ("model", check_shipped_operating_point),
    ("model", check_detection_monotonic),
    ("pure",  check_mutation_stays_legal),
    ("pure",  check_subgroup_claims),
]


def main() -> int:
    print("=" * 74)
    print(" ADVERSARIAL INTEGRITY AUDIT — inject a known lie, verify the guard fires")
    print("=" * 74)
    # build the shared fixture only if a model-based check needs it
    fx = None
    if any(kind == "model" for kind, _ in CHECKS):
        print("\nPreparing fixture (real CinC, patient-grouped OOF + held-out fit)...")
        fx = Fixture()
        print(f"  {len(fx.y)} rows / {len(set(fx.groups.tolist()))} patients / "
              f"prevalence {fx.y.mean():.4f}\n")

    failed = 0
    for _kind, fn in CHECKS:
        try:
            fn(fx)
        except Exception as e:                            # noqa: BLE001
            failed += 1
            print(f"      [FAIL] {e}")
            if not isinstance(e, AssertionError):
                traceback.print_exc()
        print("-" * 74)

    total = len(CHECKS)
    if failed:
        print(f"INTEGRITY AUDIT: {total - failed}/{total} guards fired correctly, "
              f"{failed} FAILED.")
        return 1
    print(f"INTEGRITY AUDIT PASSED — all {total} guards caught their planted pathology.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
