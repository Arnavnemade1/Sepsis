"""A nine-hour run must survive a crash. Every expensive fit is kept the moment it finishes.

The full run is ~73 booster fits and two twins, in one process that used to write its report
only at the very end. A sleeping laptop or a killed instance at hour eight cost all eight hours.
Inside ``active``, ``learners.fit_lgb`` and ``ProcessTwin.fit`` answer from disk when they have
already made this exact fit, and a re-run walks back to where it stopped.

**The key is what the fit reads**, never a name: rows, target, weights, columns, parameters,
seed, a content sample of the feature matrix (a counterfactual world trains on the same rows
with edited values, so names and rows alone would wrongly reuse a model) and a salt binding the
harness code and the store. Change any of them and the fit simply runs again. A cached model
predicts bit for bit what the fresh one did (``verify_physionet``).

A cache that cannot be written is reported and the run continues: a checkpoint must never be
the thing that kills the run it protects.
"""
from __future__ import annotations

import contextlib
import hashlib
import os
import pickle
import tempfile
from dataclasses import asdict
from typing import Dict, Optional

import numpy as np

from backend.ml.genesis import learners as L
from backend.ml.mimic import twin as TW

SAMPLE_ROWS = 2000


def _feed(h, x) -> None:
    if x is None:
        h.update(b"\0none")
    elif isinstance(x, np.ndarray):
        h.update(str((x.dtype.str, x.shape)).encode())
        h.update(np.ascontiguousarray(x).tobytes())
    else:
        h.update(repr(x).encode())
    h.update(b"\0")


def _sample(F, rows: np.ndarray, cols: Optional[np.ndarray]) -> np.ndarray:
    pick = rows[np.linspace(0, len(rows) - 1, min(SAMPLE_ROWS, len(rows))).astype(np.int64)] if len(rows) else rows
    out = np.asarray(F[pick])
    return out if cols is None else np.take(out, cols, axis=1)


class Memo:
    def __init__(self, directory: str, salt: str):
        self.dir, self.salt = directory, salt
        self.hits = self.misses = self.failures = 0
        os.makedirs(directory, exist_ok=True)

    def key(self, kind: str, *parts) -> str:
        h = hashlib.sha256(f"{kind}|{self.salt}".encode())
        for p in parts:
            _feed(h, p)
        return h.hexdigest()

    def load(self, key: str):
        path = os.path.join(self.dir, key + ".pkl")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "rb") as fh:
                return pickle.load(fh)
        except Exception as err:                 # torn by a crash: a miss, not a failure
            print(f"  checkpoint {key[:12]} unreadable ({type(err).__name__}); refitting", flush=True)
            return None

    def save(self, key: str, obj) -> None:
        try:
            fd, tmp = tempfile.mkstemp(dir=self.dir, suffix=".tmp")
            with os.fdopen(fd, "wb") as fh:
                pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
                fh.flush()
                os.fsync(fh.fileno())
            os.replace(tmp, os.path.join(self.dir, key + ".pkl"))
        except Exception as err:                 # e.g. a full disk: warn, keep the run alive
            self.failures += 1
            print(f"  WARNING: checkpoint not written ({type(err).__name__}: {err})", flush=True)


def memo_fit_lgb(memo: Memo, fit_lgb):
    def fit(F, rows, target, seg, weight=None, params=None, rounds=3000, valid_frac=0.1, early_stop=150,
            cols=None, seed=0, strat=None, fixed_rounds=None, verbose=False, feval_metric="auc"):
        import lightgbm as lgb
        rows = np.asarray(rows)
        key = memo.key("lgb", _sample(F, rows, cols), getattr(F, "shape", None), rows, np.asarray(target),
                       np.asarray(seg), None if weight is None else np.asarray(weight),
                       sorted((params or {}).items()), rounds, valid_frac, early_stop,
                       None if cols is None else np.asarray(cols), seed,
                       None if strat is None else np.asarray(strat), fixed_rounds, feval_metric)
        kept = memo.load(key)
        if kept is not None:
            memo.hits += 1
            return L.LGBFit(lgb.Booster(model_str=kept["model"]), kept["best_iter"], kept["cols"], kept["calib"])
        memo.misses += 1
        out = fit_lgb(F, rows, target, seg, weight=weight, params=params, rounds=rounds, valid_frac=valid_frac,
                      early_stop=early_stop, cols=cols, seed=seed, strat=strat, fixed_rounds=fixed_rounds,
                      verbose=verbose, feval_metric=feval_metric)
        memo.save(key, {"model": out.booster.model_to_string(num_iteration=-1), "best_iter": out.best_iter,
                        "cols": out.cols, "calib": out.calib})
        return out
    return fit


def memo_twin_fit(memo: Memo, twin_fit):
    def fit(self, led, T, train_stays, verbose: bool = False):
        s = led.store
        key = memo.key("twin", sorted(asdict(self.cfg).items()), np.asarray(train_stays),
                       _sample(s.X, np.arange(s.n_rows), None), np.asarray(s.y))
        kept = memo.load(key)
        if kept is not None:
            import torch
            memo.hits += 1
            weights = kept.pop("net_")
            self.__dict__.update(kept)
            xv, xp = self._inputs(led)
            self.net_ = TW._build(xv.shape[1], xp.shape[1], self.cfg).to(TW._device(self.cfg.device))
            self.net_.load_state_dict({k: torch.from_numpy(v) for k, v in weights.items()})
            self.net_.eval()
            return self
        memo.misses += 1
        twin_fit(self, led, T, train_stays, verbose)
        # the network's class is local to _build, so the weights are what can be kept
        memo.save(key, dict(self.__dict__, net_={k: v.detach().cpu().numpy() for k, v in self.net_.state_dict().items()}))
        return self
    return fit


@contextlib.contextmanager
def active(directory: Optional[str], salt: str):
    """Everything inside resumes from ``directory``. ``None`` turns the whole thing off."""
    if not directory:
        yield None
        return
    memo = Memo(directory, salt)
    keep_fit, keep_twin = L.fit_lgb, TW.ProcessTwin.fit
    L.fit_lgb = memo_fit_lgb(memo, keep_fit)
    TW.ProcessTwin.fit = memo_twin_fit(memo, keep_twin)
    try:
        yield memo
    finally:
        L.fit_lgb, TW.ProcessTwin.fit = keep_fit, keep_twin
        print(f"checkpoints: {memo.hits} reused, {memo.misses} fitted, {memo.failures} not written", flush=True)


def stats(memo: Optional[Memo]) -> Dict:
    return {} if memo is None else {"hits": memo.hits, "misses": memo.misses, "failures": memo.failures}
