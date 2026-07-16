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

        # honest OOF over train + a deployed fit + held-out predictions, computed once.
        self.oof = np.zeros(len(self.ytr))
        for a, b in stratified_group_folds(self.gtr, self.ytr, 3, SEED):
            m = StackedEnsembleDetector(channels=list(cfg.channels), use_hdc=False).fit(
                self.ftr.iloc[a], self.ytr[a], groups=self.gtr[a])
            self.oof[b] = m.predict_proba(self.ftr.iloc[b])
        self.model = StackedEnsembleDetector(channels=list(cfg.channels), use_hdc=False).fit(
            self.ftr, self.ytr, groups=self.gtr)
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
    # and it must still beat the old hand-set 0.12 on THIS held-out fold
    u_new = metrics.normalized_utility(
        fx.yte, (fx.p_te >= op.high).astype(int), fx.groups[fx.te], fx.hour[fx.te])
    u_old = metrics.normalized_utility(
        fx.yte, (fx.p_te >= 0.12).astype(int), fx.groups[fx.te], fx.hour[fx.te])
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
CHECKS = [
    ("model", check_label_shuffle_collapse),
    ("model", check_oof_necessity),
    ("pure",  check_gate_rejects_noise),
    ("model", check_bootstrap_ci_honesty),
    ("model", check_locked_gate_disjoint),
    ("model", check_shipped_operating_point),
    ("model", check_detection_monotonic),
    ("pure",  check_mutation_stays_legal),
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
