"""Adversarial audit of the GENESIS line — every claim it makes, attacked.

Run: ``make verify-genesis`` (or ``.venv/bin/python verify_genesis.py``).

Each check plants the failure it guards against and asserts the guard fires, because a
check that cannot fail is not evidence (see the project's vacuous-check history):

 1. CAUSALITY — rewrite every patient's future with noise and truncate it; no feature
    at or before the cut may move. Then plant a backward-fill feature and assert this
    same test catches it (the test can fail).
 2. TEMPORAL SUMMARIES — the stage-2 risk summaries obey the same causality test.
 3. FAST AUROC — equals sklearn exactly, with heavy ties and patient weights.
 4. DECOMPOSITION — pooled AUROC is exactly the negative-share-weighted mean of the
    between-patient and within-septic AUROCs.
 5. LABEL IDENTITY — the 6-hour horizon target IS the challenge label, bit for bit;
    the timing target never trains on a never-septic patient; an unrecorded future
    always carries weight 0.
 6. LOCKED SET — disjoint from every dev fold, stable fingerprint, stratified, and the
    harness refuses to score a locked patient as a dev prediction.
 7. LEAK TRIPWIRE — permute outcomes between patients; a real fit on the real
    features must collapse to chance. Then leak the label into a feature and assert
    the same fit does NOT stay at chance (the tripwire can trip).
 8. NETWORK — causal under a rewritten future; the Sepsis-3 head is an AND gate.
 9. SELF-SUPERVISED PRETEXT — a hidden value stays hidden: blanked and flagged, no
    composite hands it back (checked with the guard off too), targets only where measured.
"""
from __future__ import annotations

import sys

import numpy as np

from backend.ml.genesis import features as feat
from backend.ml.genesis import harness as H
from backend.ml.genesis import rawstore
from backend.ml.genesis import stack as S
from backend.ml.genesis import targets as T

_results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    _results.append((name, bool(ok), detail))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""), flush=True)


def _perturb_future(raw: rawstore.RawStore, rng: np.random.Generator, truncate: bool):
    """Return (new_raw, keep_mask) where every patient's rows after a random cut are
    replaced by noise (or removed). ``keep_mask`` marks the untouched rows."""
    X = raw.X.copy()
    y = raw.y.copy()
    keep = np.ones(raw.n_rows, dtype=bool)
    lens = raw.lens.copy()
    rows_keep = []
    for k in range(raw.n_patients):
        a, n = int(raw.starts[k]), int(raw.lens[k])
        cut = int(rng.integers(0, n))                # rows a..a+cut are the "past"
        fut = slice(a + cut + 1, a + n)
        keep[fut] = False
        if truncate:
            rows_keep.append(np.arange(a, a + cut + 1))
            lens[k] = cut + 1
        else:
            Z = X[fut]
            noise = rng.normal(0, 5, size=Z.shape).astype(np.float32)
            Z = np.where(rng.random(Z.shape) < 0.5, np.nan, np.nan_to_num(Z, nan=1.0) * 3 + noise)
            X[fut] = Z
    if truncate:
        r = np.concatenate(rows_keep)
        return rawstore.RawStore(X=X[r], y=y[r], lens=lens, pid=raw.pid, site=raw.site,
                                 columns=raw.columns), keep
    return rawstore.RawStore(X=X, y=y, lens=lens, pid=raw.pid, site=raw.site,
                             columns=raw.columns), keep


def _same(a: np.ndarray, b: np.ndarray) -> bool:
    return bool(np.array_equal(np.isnan(a), np.isnan(b)) and
                np.array_equal(np.nan_to_num(a, nan=0.0), np.nan_to_num(b, nan=0.0)))


def check_causality(raw_all: rawstore.RawStore) -> None:
    print("\n1. causality of every feature column")
    rng = np.random.default_rng(0)
    raw = raw_all.subset(rng.choice(raw_all.n_patients, 400, replace=False))
    F0, names = feat.build(raw)
    F0 = np.array(F0)
    for truncate in (False, True):
        new, keep = _perturb_future(raw, rng, truncate)
        F1, _ = feat.build(new)
        F1 = np.array(F1)
        past0 = F0[keep]
        past1 = F1 if truncate else F1[keep]
        bad = [n for j, n in enumerate(names) if not _same(past0[:, j], past1[:, j])]
        check(f"no past feature moves when the future is {'truncated' if truncate else 'rewritten'}",
              not bad, f"{len(names)} columns, {int(keep.sum())} past rows"
              + (f"; MOVED: {bad[:5]}" if bad else ""))
    B0, bn = feat.intent_block(raw)
    for truncate in (False, True):
        new, keep = _perturb_future(raw, rng, truncate)
        B1, _ = feat.intent_block(new)
        past1 = B1 if truncate else B1[keep]
        # recompute the reference against THIS draw's keep mask
        bad = [n for j, n in enumerate(bn) if not _same(B0[keep][:, j], past1[:, j])]
        check(f"no intent feature moves when the future is {'truncated' if truncate else 'rewritten'}",
              not bad, f"{len(bn)} columns" + (f"; MOVED: {bad[:5]}" if bad else ""))
    # the test can fail: a backward-filled (non-causal) column must be caught
    def leaky(r):
        F, n = feat.build(r)
        F = np.array(F)
        s = r.col("Lactate").astype(np.float64)
        import pandas as pd
        bf = pd.Series(s).groupby(r.seg).bfill().to_numpy()
        return np.column_stack([F, bf]), n + ["Lactate__bfill"]
    Fa, na = leaky(raw)
    new, keep = _perturb_future(raw, rng, False)
    Fb, _ = leaky(new)
    moved = [n for j, n in enumerate(na) if not _same(Fa[keep][:, j], Fb[keep][:, j])]
    check("a planted backward-fill column is caught by the same test",
          moved == ["Lactate__bfill"], f"flagged {moved}")


def check_temporal(c: H.Cohort) -> None:
    print("\n2. causality of stage-2 temporal summaries")
    rng = np.random.default_rng(1)
    z = rng.normal(size=len(c.y))
    base = S.temporal_summaries(c, z, "z")
    cut_hour = rng.integers(0, 50, size=c.raw.n_patients)[c.seg]
    fut = c.hour > cut_hour
    z2 = z.copy()
    z2[fut] = rng.normal(5, 3, size=fut.sum())
    new = S.temporal_summaries(c, z2, "z")
    moved = [k for k in base if not _same(base[k][~fut], new[k][~fut])]
    check("no stage-2 summary of the past moves when future risk is rewritten", not moved,
          f"{len(base)} summaries" + (f"; MOVED {moved}" if moved else ""))
    changed = [k for k in base if not _same(base[k][fut], new[k][fut])]
    check("…and the rewrite does reach the future rows (non-vacuous)", len(changed) == len(base))


def check_auroc() -> None:
    print("\n3. fast weighted AUROC")
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(2)
    worst = 0.0
    for _ in range(20):
        n = 5000
        y = (rng.random(n) < 0.05).astype(int)
        s = np.round(rng.normal(size=n) + y, 1)
        w = rng.integers(0, 3, size=n).astype(float)
        worst = max(worst, abs(H.Ranked(s).auc(y, w) - roc_auc_score(y, s, sample_weight=w)))
    check("equals sklearn with ties and weights (20 draws)", worst < 1e-12, f"max |diff| {worst:.1e}")


def check_platt() -> None:
    print("\n3b. Platt recalibration (fold-scale alignment)")
    from backend.ml.genesis.learners import platt
    rng = np.random.default_rng(6)
    m = rng.normal(size=20000)
    t = (rng.random(20000) < 1 / (1 + np.exp(-(2 * m - 3)))).astype(float)
    a, b = platt(m, t)
    check("recovers a planted map sigmoid(2m - 3)", abs(a - 2) < 0.1 and abs(b + 3) < 0.1,
          f"a={a:.3f} b={b:.3f}")
    # REGRESSION: raw Newton diverged to a = -1.3e9 on near-separable, extreme margins
    # (boosters with ~500 trees) and turned two folds of a learning curve upside down.
    m2 = np.r_[rng.normal(-12, 3, 5000), rng.normal(6, 3, 60)]
    t2 = np.r_[np.zeros(5000), np.ones(60)]
    a2, b2 = platt(m2, t2)
    check("extreme, near-separable margins give a finite INCREASING map", np.isfinite(a2) and
          np.isfinite(b2) and 0 < a2 < 1e3, f"a={a2:.3f} b={b2:.3f}")
    a3, b3 = platt(-m, t)
    check("an anti-correlated margin falls back to the identity (never reverses a ranking)",
          (a3, b3) == (1.0, 0.0), f"a={a3} b={b3}")


def check_decomposition(c: H.Cohort) -> None:
    print("\n4. pooled = weighted mean of between and within-septic")
    rng = np.random.default_rng(3)
    p = rng.random(len(c.y)) + 0.3 * c.y + 0.2 * c.plabel[c.seg]
    s = H.score(c, p, with_utility=False)
    share = s["neg_share_septic"]
    recon = (1 - share) * s["between"] + share * s["within_septic"]
    check("identity holds exactly", abs(recon - s["auroc"]) < 1e-12,
          f"pooled {s['auroc']:.6f} vs {recon:.6f}; septic share of negatives {share:.4f}")


def check_targets(c: H.Cohort) -> None:
    print("\n5. targets")
    h6, _ = T.horizon(c, 6)
    check("horizon(6) is the challenge label bit for bit", np.array_equal(h6, c.y.astype(float)))
    h5, _ = T.horizon(c, 5)
    check("…and horizon(5) is not (the identity is non-vacuous)", not np.array_equal(h5, c.y.astype(float)))
    _, w = T.timing(c)
    check("timing target never trains on a never-septic patient",
          w[c.plabel[c.seg] == 0].max() == 0 and w[c.plabel[c.seg] == 1].min() == 1)
    ev = np.zeros(len(c.y))
    tgt, w = T.future_any(c, ev + 1, 6)
    ends = (c.raw.starts + c.raw.lens - 1)[c.seg]
    check("an unrecorded future carries weight 0",
          np.all(w[np.arange(len(w)) + 6 > ends] == 0) and np.all(w[np.arange(len(w)) + 6 <= ends] == 1))
    lo = c.raw.starts[c.seg]
    idx = np.arange(len(ev))
    mark = np.zeros(len(ev))
    mark[lo] = 1                                   # event only at each stay's first row
    tgt, w = T.future_any(c, mark, 6)
    check("future_any never looks backwards or across stays",
          tgt[w > 0].max() == 0, "an event at hour 0 is never 'in the future' of any row")


def check_lock(c: H.Cohort) -> None:
    print("\n6. locked test set")
    locked = c.folds < 0
    ok = True
    for k in range(c.n_folds):
        tr, te = c.fold_rows(k)
        ok &= not locked[c.seg[tr]].any() and not locked[c.seg[te]].any()
    check("locked patients are in no dev fold", ok, f"{int(locked.sum())} locked patients")
    fp = H.lock_fingerprint(c)
    check("fingerprint is the one recorded when the lock was made", fp == "77e793853667b28e", fp)
    d = abs(c.plabel[locked].mean() - c.plabel[~locked].mean())
    check("septic share matches dev within 0.2 pp", d < 0.002, f"|diff| {d:.4f}")
    # plant: a fold whose test rows smuggle in locked patients must be refused
    orig = c.fold_rows
    locked_rows = np.flatnonzero(locked[c.seg])[:10]

    def smuggle(k):
        tr, te = orig(k)
        return tr, np.r_[te, locked_rows]

    c.fold_rows = smuggle
    refused = False
    try:
        H.run(c, "planted_lock_leak", lambda cc, tr, te: np.zeros(len(te)),
              folds=[0], verbose=False, save=False)
    except RuntimeError:
        refused = True
    finally:
        del c.fold_rows
    check("the harness refuses to score a locked patient as dev", refused)
    ok_clean = True
    try:
        H.run(c, "clean_fold", lambda cc, tr, te: np.random.default_rng(0).random(len(te)),
              folds=[0], verbose=False, save=False)
    except RuntimeError:
        ok_clean = False
    check("…and accepts a clean fold (the guard is not trigger-happy)", ok_clean)


def check_tripwire(c: H.Cohort) -> None:
    print("\n7. leak tripwire (real features, real fit)")
    from backend.ml.genesis import learners as L
    rng = np.random.default_rng(4)
    pts = np.flatnonzero(c.folds >= 0)
    pts = rng.choice(pts, 3000, replace=False)
    rows = c.rows_of(pts)
    seg = c.seg[rows]
    half = rng.random(c.raw.n_patients) < 0.5
    tr, te = rows[half[seg]], rows[~half[seg]]
    # permute whole outcome trajectories between patients: labels keep their shape,
    # lose their patient
    perm_pl = rng.permutation(c.plabel)
    fake = perm_pl[c.seg].astype(float)
    cols = np.arange(min(200, c.F.shape[1]))
    fit = L.fit_lgb(c.F, tr, fake[tr], c.seg[tr], cols=cols, fixed_rounds=150,
                    params={"num_leaves": 15})
    a = H.auroc(fake[te], fit.predict(c.F, te))
    check("permuted outcomes collapse to chance", abs(a - 0.5) < 0.05, f"AUROC {a:.3f}")
    leak = np.column_stack([np.asarray(c.F[:, :5]), fake + rng.normal(0, 0.5, len(fake))]).astype(np.float32)
    fit2 = L.fit_lgb(leak, tr, fake[tr], c.seg[tr], fixed_rounds=150, params={"num_leaves": 15})
    a2 = H.auroc(fake[te], fit2.predict(leak, te))
    check("…and a planted label leak does NOT (the tripwire can trip)", a2 > 0.75, f"AUROC {a2:.3f}")


def check_network() -> None:
    print("\n8. Genesis Network causality (every head, every hour)")
    try:
        import torch
    except Exception:  # pragma: no cover
        check("torch available for the network check", False, "torch missing")
        return
    from backend.ml.genesis import network as N
    cfg = N.NetConfig(device="cpu")
    torch.manual_seed(0)
    net = N._build(20, cfg, len(N.SUSPICION_AUX), len(N.ORGAN_AUX)).eval()
    rng = np.random.default_rng(5)
    x = torch.from_numpy(rng.normal(size=(3, 40, 20)).astype(np.float32))
    cut = 17
    x2 = x.clone()
    x2[:, cut + 1:] = torch.from_numpy(rng.normal(3, 2, size=(3, 40 - cut - 1, 20)).astype(np.float32))
    with torch.no_grad():
        a, b = net(x), net(x2)
    past_same = all(torch.equal(u[:, :cut + 1], v[:, :cut + 1]) for u, v in zip(a, b))
    future_moves = all(not torch.equal(u[:, cut + 1:], v[:, cut + 1:]) for u, v in zip(a, b))
    check("no network output at or before the cut moves when the future is rewritten", past_same,
          f"{len(a)} output tensors")
    check("…and the rewrite reaches every output after the cut (non-vacuous)", future_moves)
    # the AND gate really is a product of the two latent halves
    main, s3, _, _, a_s, a_o, _, _ = a
    prod = torch.sigmoid(a_s) * torch.sigmoid(a_o)
    check("the Sepsis-3 head equals sigmoid(susp) * sigmoid(organ)",
          torch.allclose(torch.exp(s3), prod, atol=1e-6))


def check_ssl_masking(c: H.Cohort) -> None:
    print("\n9. self-supervised pretext: a hidden value stays hidden")
    import torch
    from backend.ml.genesis import network as N
    X, _ = N.input_matrix(c)
    gn = N.GenesisNetwork(N.NetConfig(device="cpu", mask_channels=True))
    pts = np.flatnonzero(c.folds >= 0)[:300]
    gn.std_ = N.Standardiser.fit(X, c.rows_of(pts))
    dev = torch.device("cpu")
    x0, _, _, _, valid = gn._tensors(c, X, {}, pts, dev)                   # no masking
    x1, ys, _, _, _ = gn._tensors(c, X, {}, pts, dev, mask_rng=np.random.default_rng(7),
                                  with_world=False)
    x0, x1 = x0.numpy(), x1.numpy()
    n_in = X.shape[1]
    M = x1[..., n_in:n_in + N.N_CH] > 0.5
    frac = M[valid].mean()
    check("the pretext hides a non-trivial share of values", 0.05 < frac < 0.30,
          f"{frac:.3f} of (channel, hour) entries hidden")
    vals0, vals1 = x0[..., N.VALUE_COLS], x1[..., N.VALUE_COLS]
    check("hidden values are blanked, visible ones untouched",
          np.all(vals1[M] == 0) and np.array_equal(vals1[~M], vals0[~M]))
    base = N.N_CH * 4

    def leaks(xm, mask):
        bad = []
        for name, chans in N.DERIVED.items():
            col = base + N.CONTEXT.index(name)
            dep = np.zeros(mask.shape[:2], dtype=bool)
            for ch in chans:
                dep |= mask[..., N.DYNAMIC.index(ch)]
            moved = dep & valid & (x0[..., col] != 0)
            if np.any(xm[..., col][moved] == x0[..., col][moved]):
                bad.append(name)
        return bad

    bad = leaks(x1, M)
    check("no composite reveals a hidden channel (shock index would hand back HR)", not bad,
          f"leaking: {bad}" if bad else f"{len(N.DERIVED)} composites guarded")
    t, m = ys["mvr_t"].numpy(), ys["mvr_m"].numpy() > 0.5
    now = x0[..., N.NOW_COLS]
    check("targets exist only where a hidden value was actually measured",
          np.all(M[m]) and m.sum() > 0 and np.allclose(t[m], vals0[m]),
          f"{int(m.sum())} targets")
    # the leak check can fail: switch the composite guard off and it must fire
    saved = dict(N.DERIVED)
    N.DERIVED.clear()
    try:
        x2, _, _, _, _ = gn._tensors(c, X, {}, pts, dev, mask_rng=np.random.default_rng(7),
                                     with_world=False)
    finally:
        N.DERIVED.update(saved)
    M2 = x2.numpy()[..., n_in:n_in + N.N_CH] > 0.5
    check("…and with the guard switched off the same check catches the leak",
          len(leaks(x2.numpy(), M2)) > 0)


def main() -> int:
    raw = rawstore.load()
    check_causality(raw)
    c = H.Cohort.load()
    check_temporal(c)
    check_auroc()
    check_platt()
    check_decomposition(c)
    check_targets(c)
    check_lock(c)
    check_tripwire(c)
    check_network()
    check_ssl_masking(c)
    n_ok = sum(ok for _, ok, _ in _results)
    print(f"\nverify_genesis: {n_ok}/{len(_results)} passed")
    return 0 if n_ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
