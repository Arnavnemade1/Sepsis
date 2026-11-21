"""The fast paired bootstrap must return what ``bench compare`` returns, and the check must bite.

python -m backend.ml.scripts.verify_yaib_paired_fast
"""
from __future__ import annotations

import os
import tempfile
import time
import unittest

import numpy as np
import pandas as pd

from backend.ml.scripts import yaib_paired_fast as FAST
from backend.ml.yaib import bench


def _runs(seed=1, reps=2, folds=3, stays=60, ties=True):
    """Synthetic paired runs: shuffled YAIB row order, rounded (tied) scores, rare positives."""
    rng = np.random.default_rng(seed)
    ours, theirs = {}, {}
    for r in range(reps):
        for f in range(folds):
            ids = np.repeat(np.arange(stays) + 1000 * r, rng.integers(3, 12, stays))
            hour = np.concatenate([np.arange(n) for n in np.bincount(np.searchsorted(np.unique(ids), ids))])
            y = (rng.random(len(ids)) < 0.08).astype(float)
            po = np.clip(0.3 * y + rng.random(len(ids)) * 0.7, 0, 1)
            pb = np.clip(0.15 * y + rng.random(len(ids)) * 0.85, 0, 1)
            if ties:
                po, pb = np.round(po, 2), np.round(pb, 1)
            o = pd.DataFrame({"stay_id": ids, "hour": hour.astype(float), "label": y, "p": po})
            t = o.assign(p=pb).sort_values(["stay_id", "hour"]).reset_index(drop=True)
            # YAIB writes its ids unsorted beside sorted labels/probs: shuffle only the id columns
            perm = rng.permutation(len(t))
            t[["stay_id", "hour"]] = t[["stay_id", "hour"]].to_numpy()[perm]
            ours[(r, f)], theirs[(r, f)] = o.sample(frac=1, random_state=f).reset_index(drop=True), t
    return ours, theirs


class SameAsBench(unittest.TestCase):
    def assert_same(self, a, b):
        for k in ("auroc_delta", "ci_low", "ci_high"):
            self.assertAlmostEqual(a[k], b[k], places=12, msg=k)
        for k in ("repetitions", "runs", "better", "worse"):
            self.assertEqual(a[k], b[k], msg=k)

    def test_matches_bench_with_ties(self):
        ours, theirs = _runs()
        self.assert_same(FAST.compare(ours, theirs, 200, seed=3), bench.compare(ours, theirs, 200, seed=3))

    def test_matches_bench_without_ties(self):
        ours, theirs = _runs(seed=7, ties=False)
        self.assert_same(FAST.compare(ours, theirs, 100), bench.compare(ours, theirs, 100))

    def test_matches_bench_when_draws_skip_runs(self):
        # tiny folds with one positive: many draws hold a single class and are skipped
        ours, theirs = _runs(seed=11, reps=1, folds=2, stays=6)
        self.assert_same(FAST.compare(ours, theirs, 300), bench.compare(ours, theirs, 300))

    def test_the_check_fails_for_a_tie_blind_auc(self):
        """Vacuous-check guard: counting ties as wins (not halves) must be caught."""
        ours, theirs = _runs(seed=5)
        real = FAST._Sorted.auc

        def tie_blind(self, m):
            w = m[self.codes]
            gp = np.add.reduceat(w * self.y, self.starts)
            gn = np.add.reduceat(w, self.starts) - gp
            return float(np.dot(gn, np.cumsum(gp)) / (gp.sum() * gn.sum()))

        FAST._Sorted.auc = tie_blind
        try:
            bad = FAST.compare(ours, theirs, 100)
        finally:
            FAST._Sorted.auc = real
        good = bench.compare(ours, theirs, 100)
        self.assertGreater(abs(bad["ci_low"] - good["ci_low"]) + abs(bad["ci_high"] - good["ci_high"]), 1e-6)

    def test_refuses_mismatched_run_sets(self):
        ours, theirs = _runs()
        theirs.pop((0, 1))
        with self.assertRaises(ValueError):
            FAST.compare(ours, theirs, 10)

    def test_is_faster_than_bench(self):
        ours, theirs = _runs(seed=2, reps=1, folds=2, stays=4000)
        t = time.time(); FAST.compare(ours, theirs, 20); fast = time.time() - t
        t = time.time(); bench.compare(ours, theirs, 20); slow = time.time() - t
        self.assertLess(fast, slow / 2, f"fast {fast:.2f}s vs bench {slow:.2f}s")


class ReadsGzip(unittest.TestCase):
    def _write(self, root, ours, theirs, gz_keys=()):
        od = os.path.join(root, "ours"); os.makedirs(od)
        for (r, f), o in ours.items():
            o.to_parquet(os.path.join(od, f"ours_r{r}_f{f}.parquet"))
            d = os.path.join(root, "yaib", "x", f"repetition_{r}", f"fold_{f}"); os.makedirs(d)
            t = theirs[(r, f)]
            a = np.c_[t["stay_id"], t["hour"], t["label"], 1 - t["p"], t["p"]]
            name = "pred_indicators.csv.gz" if (r, f) in gz_keys else "pred_indicators.csv"
            np.savetxt(os.path.join(d, name), a, delimiter=",")
        return od, os.path.join(root, "yaib")

    def test_gzipped_runs_give_the_same_result(self):
        ours, theirs = _runs(seed=4, reps=1, folds=3)
        with tempfile.TemporaryDirectory() as plain, tempfile.TemporaryDirectory() as mixed:
            a = FAST.compare(*FAST.load_runs(*self._write(plain, ours, theirs)), 50)
            b = FAST.compare(*FAST.load_runs(*self._write(mixed, ours, theirs, {(0, 0), (0, 2)})), 50)
        self.assertEqual(a, b)

    def test_a_run_saved_both_ways_is_refused(self):
        ours, theirs = _runs(seed=4, reps=1, folds=1)
        with tempfile.TemporaryDirectory() as root:
            od, yd = self._write(root, ours, theirs)
            d = os.path.join(yd, "x", "repetition_0", "fold_0")
            np.savetxt(os.path.join(d, "pred_indicators.csv.gz"), np.loadtxt(os.path.join(d, "pred_indicators.csv"), delimiter=","), delimiter=",")
            with self.assertRaises(ValueError):
                FAST.load_runs(od, yd)


if __name__ == "__main__":
    unittest.main(verbosity=2)
