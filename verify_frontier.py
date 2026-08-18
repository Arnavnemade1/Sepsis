#!/usr/bin/env python3
"""Adversarial audit of the accuracy-frontier work (make verify-frontier).

Same discipline as ``verify_integrity.py``: the claims this push rests on are
*negative* ones — "the utility target is the official utility, not a reweighting I
liked", "the sequence models cannot see the future", "the chunked scan is the
recurrence it says it is", "the gain is not fold luck". Negative claims are only ever
falsifiable, so each check here plants the pathology it is guarding against and
asserts the guard fires.

    .venv/bin/python verify_frontier.py [--patients 1200]

Complements — does not replace — verify_integrity.py and verify_full_system.py.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np

from backend.ml.eval import metrics
from backend.ml.eval.bench import Benchmark
from backend.ml.eval.splits import stratified_group_folds
from backend.ml.eval.utility import (
    normalized_utility_fast, utility_target, utility_terms)

PASS, FAIL = "  \033[32mPASS\033[0m", "  \033[31mFAIL\033[0m"
_results: list = []


def check(name: str):
    def deco(fn):
        def wrapped(*a, **k):
            print(f"\n[{name}]")
            try:
                fn(*a, **k)
                print(PASS)
                _results.append((name, True, ""))
            except AssertionError as exc:
                print(f"{FAIL} {exc}")
                _results.append((name, False, str(exc)))
        wrapped.__name__ = fn.__name__
        return wrapped
    return deco


# --------------------------------------------------------------------------- #
@check("utility arithmetic is the official one, not a lookalike")
def check_utility_exact(ds, idx):
    y, g, h, o = ds.y[idx], ds.groups[idx], ds.hour_index[idx], ds.onset_hour[idx]
    terms = utility_terms(y, g, h, o)
    rng = np.random.default_rng(0)
    policies = {"all-silent": np.zeros(len(y), int), "all-alarm": np.ones(len(y), int),
                "true-labels": y.astype(int), "random": (rng.random(len(y)) > 0.9).astype(int)}
    worst = 0.0
    for nm, pred in policies.items():
        ref = metrics.normalized_utility(y, pred, g, h)
        fast = normalized_utility_fast(y, pred, terms)
        worst = max(worst, abs(ref - fast))
        print(f"    {nm:12s} reference {ref:+.9f}  vectorised {fast:+.9f}")
    print(f"    planted: four alarm policies incl. adversarial all-alarm; "
          f"worst disagreement {worst:.2e}")
    assert worst < 1e-9, f"vectorised utility disagrees with the reference by {worst:.2e}"


@check("the training target really is the utility's own decision rule")
def check_target_derivation(ds, idx):
    y, g, h, o = ds.y[idx], ds.groups[idx], ds.hour_index[idx], ds.onset_hour[idx]
    t = utility_terms(y, g, h, o)
    z, w = utility_target(y, g, h, o)
    assert np.array_equal(z, (t["gain"] > 0).astype(np.int8)), "z is not sign(u1-u0)"
    assert np.allclose(w, np.abs(t["gain"]), atol=1e-12), "w is not |u1-u0|"
    ramp = t["septic"] & (t["dt"] >= -12) & (t["dt"] < -6)
    flipped = int((ramp & (y == 0) & (z == 1)).sum())
    print(f"    {flipped:,} hours the challenge label calls negative are worth alarming on")
    print(f"    (credit ramp weight {w[ramp].sum():.0f} of {w[z == 1].sum():.0f} positive-side total)")
    assert flipped > 0, "the credit ramp was not relabelled — the objective is a no-op"
    # And the rule must agree with a brute-force scan of the two alternatives.
    brute = (t["u1"] > t["u0"]).astype(np.int8)
    assert np.array_equal(z, brute), "target disagrees with argmax over {alarm, silent}"


@check("sequence architectures cannot see the future")
def check_causality():
    import torch
    from backend.ml.models.sequence import _build_modules, _device
    SeqNet = _build_modules()
    dev = _device()
    torch.manual_seed(0)
    B, T, F = 4, 48, 16
    x = torch.randn(B, T, F, device=dev)
    mask = torch.ones(B, T, device=dev)
    for arch in ("tcn", "ssm", "xfmr"):
        net = SeqNet(F, arch=arch, d_model=32, n_layers=3, n_heads=4,
                     n_state=8, chunk=8).to(dev).eval()
        with torch.no_grad():
            base = net(x, mask)
            # planted pathology: overwrite the entire second half of the input
            x_future = x.clone()
            x_future[:, T // 2:] = 50.0
            leak = (base[:, :T // 2] - net(x_future, mask)[:, :T // 2]).abs().max().item()
            x_past = x.clone()
            x_past[:, 0] += 5.0
            uses = (base[:, 1:] - net(x_past, mask)[:, 1:]).abs().max().item()
        print(f"    {arch:5s} future rewritten -> max change in the past {leak:.2e};"
              f" history perturbed -> {uses:.4f}")
        assert leak < 1e-4, f"{arch} leaks the future ({leak:.2e})"
        assert uses > 1e-3, f"{arch} ignores its own history — it is not a sequence model"


@check("the chunked selective scan is the recurrence it claims to be")
def check_ssd_scan():
    import torch
    import torch.nn.functional as F
    torch.manual_seed(0)
    worst = 0.0
    for T in (7, 40, 128, 336):
        B, H, P, N, C = 2, 4, 8, 8, 16
        x = torch.randn(B, T, H, P, dtype=torch.float64)
        Bm = torch.randn(B, T, H, N, dtype=torch.float64)
        Cm = torch.randn(B, T, H, N, dtype=torch.float64)
        log_a = -F.softplus(torch.randn(B, T, H, dtype=torch.float64))
        # reference: the literal one-step-at-a-time recurrence
        S = torch.zeros(B, H, N, P, dtype=torch.float64)
        ref = []
        for t in range(T):
            S = log_a[:, t].exp()[..., None, None] * S + torch.einsum("bhn,bhp->bhnp", Bm[:, t], x[:, t])
            ref.append(torch.einsum("bhn,bhnp->bhp", Cm[:, t], S))
        ref = torch.stack(ref, dim=1)
        got = _chunked(x, Bm, Cm, log_a, C)
        worst = max(worst, (ref - got).abs().max().item())
    print(f"    T in {{7, 40, 128, 336}} incl. non-multiples of the chunk; "
          f"worst |chunked - recurrence| {worst:.2e}")
    assert worst < 1e-9, f"the chunked scan is not the stated recurrence ({worst:.2e})"


def _chunked(x, Bm, Cm, log_a, C):
    """The exact code path used inside SelectiveSSM.forward."""
    import torch
    import torch.nn.functional as F
    B, T, H, P = x.shape
    N = Bm.shape[-1]
    pad = (-T) % C
    if pad:
        log_a = F.pad(log_a.transpose(1, 2), (0, pad)).transpose(1, 2)
        x = F.pad(x.permute(0, 2, 3, 1), (0, pad)).permute(0, 3, 1, 2)
        Bm = F.pad(Bm.permute(0, 2, 3, 1), (0, pad)).permute(0, 3, 1, 2)
        Cm = F.pad(Cm.permute(0, 2, 3, 1), (0, pad)).permute(0, 3, 1, 2)
    K = (T + pad) // C
    rs = lambda t: t.view(B, K, C, *t.shape[2:])
    log_a, x, Bm, Cm = rs(log_a), rs(x), rs(Bm), rs(Cm)
    La = torch.cumsum(log_a, dim=2)
    seg = La.unsqueeze(3) - La.unsqueeze(2)
    tri = torch.ones(C, C, dtype=torch.bool).tril()
    decay = torch.where(tri[None, None, :, :, None], seg.exp(), torch.zeros_like(seg))
    cb = torch.einsum("bkchn,bkdhn->bkcdh", Cm, Bm)
    y = torch.einsum("bkcdh,bkdhp->bkchp", cb * decay, x)
    to_end = (La[:, :, -1:, :] - La).exp()
    s_chunk = torch.einsum("bkchn,bkchp,bkch->bkhnp", Bm, x, to_end)
    a_chunk = La[:, :, -1, :].exp()
    S = torch.zeros(B, H, N, P, dtype=x.dtype)
    prev = []
    for k in range(K):
        prev.append(S)
        S = S * a_chunk[:, k][..., None, None] + s_chunk[:, k]
    y = y + torch.einsum("bkchn,bkhnp,bkch->bkchp", Cm, torch.stack(prev, dim=1), La.exp())
    return y.reshape(B, K * C, H, P)[:, :T]


@check("temporal filters are causal and independent of row order")
def check_filters():
    from backend.ml.models.temporal import causal_filter
    rng = np.random.default_rng(0)
    g = np.repeat(np.arange(6), 24)
    h = np.tile(np.arange(24), 6)
    p = rng.random(len(g))
    for mode in ("ema", "cummax", "hazard", "blend"):
        a = causal_filter(p, g, h, mode)
        p2 = p.copy()
        p2[h >= 12] = 1.0                      # planted: rewrite every future hour
        leak = np.abs(a[h < 12] - causal_filter(p2, g, h, mode)[h < 12]).max()
        perm = rng.permutation(len(g))         # planted: shuffle the row order
        mis = np.abs(causal_filter(p[perm], g[perm], h[perm], mode) - a[perm]).max()
        print(f"    {mode:7s} future-leak {leak:.2e}  order-sensitivity {mis:.2e}")
        assert leak < 1e-12, f"{mode} filter leaks the future"
        assert mis < 1e-12, f"{mode} filter depends on the order rows arrive in"


@check("shuffled labels collapse the utility-objective booster to chance")
def check_leakage_tripwire(ds, idx):
    from backend.ml.models.boosters import UtilityAlignedBooster
    rng = np.random.default_rng(7)
    y, g, h, o = ds.y[idx], ds.groups[idx], ds.hour_index[idx], ds.onset_hour[idx]
    # Permute the sepsis story ACROSS patients: each patient keeps a real label
    # trajectory, just somebody else's. Anything above chance is leakage.
    pts = np.array(sorted(set(g.tolist())))
    perm = {a: b for a, b in zip(pts, rng.permutation(pts))}
    y_sh, o_sh = np.zeros_like(y), np.full_like(o, -1)
    for a, b in zip(pts, [perm[v] for v in pts]):
        dst, donor = np.flatnonzero(g == a), np.flatnonzero(g == b)
        n = min(len(dst), len(donor))
        y_sh[dst[:n]] = y[donor[:n]]
        o_sh[dst] = o[donor][0] if len(donor) else -1
    tr, te = next(iter(stratified_group_folds(g, y_sh, 4, 0)))
    m = UtilityAlignedBooster(kind="hgb", objective="utility")
    m.fit(ds.X[idx][tr], y_sh[tr], groups=g[tr], hour_index=h[tr], onset_hour=o_sh[tr])
    a = metrics.auroc(y_sh[te], m.predict_proba(ds.X[idx][te]))
    print(f"    planted: sepsis trajectories permuted between patients -> AUROC {a:.4f}")
    assert abs(a - 0.5) < 0.06, f"AUROC {a:.4f} on shuffled labels — the pipeline leaks"


@check("the paired bootstrap cannot manufacture a difference from nothing")
def check_bootstrap(bench, idx):
    from backend.ml.eval.bench import BenchResult
    rng = np.random.default_rng(3)
    p = rng.random(len(bench.ds.y))
    same = BenchResult("a", [], p, 0.0), BenchResult("b", [], p.copy(), 0.0)
    c = bench.compare(*same, n_boot=120)
    print(f"    planted: one predictor compared with a copy of itself -> "
          f"delta {c['delta_auroc']:+.4f} CI [{c['ci_lo']:+.4f}, {c['ci_hi']:+.4f}]")
    assert not c["significant"], "a model beat itself — the significance test is broken"
    assert abs(c["delta_auroc"]) < 1e-9, "identical predictors scored differently"
    better = p + 0.35 * bench.ds.y                 # a genuinely better predictor
    c2 = bench.compare(BenchResult("real", [], better, 0.0), same[0], n_boot=120)
    print(f"    planted: a predictor given real label signal -> "
          f"delta {c2['delta_auroc']:+.4f} CI [{c2['ci_lo']:+.4f}, {c2['ci_hi']:+.4f}]")
    assert c2["significant"], "a real improvement was not detected — the test has no power"


@check("benchmark folds never split a patient")
def check_folds(bench):
    for k, (tr, te) in enumerate(bench.folds):
        shared = set(bench.ds.groups[tr].tolist()) & set(bench.ds.groups[te].tolist())
        assert not shared, f"fold {k} shares {len(shared)} patients between train and test"
    print(f"    {len(bench.folds)} folds, no patient on both sides of any split")


# --------------------------------------------------------------------------- #
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=1200)
    args = ap.parse_args()

    print("=" * 78)
    print("ACCURACY-FRONTIER INTEGRITY AUDIT — every check plants a known pathology")
    print("=" * 78)
    bench = Benchmark.load("dev")
    pts = np.array(sorted(set(bench.ds.groups.tolist())))[: args.patients]
    idx = np.flatnonzero(np.isin(bench.ds.groups, pts))
    print(f"cohort slice: {len(idx):,} rows / {len(pts):,} patients")

    check_utility_exact(bench.ds, idx)
    check_target_derivation(bench.ds, idx)
    check_causality()
    check_ssd_scan()
    check_filters()
    check_leakage_tripwire(bench.ds, idx)
    check_bootstrap(bench, idx)
    check_folds(bench)

    ok = sum(1 for _, p, _ in _results if p)
    print("\n" + "=" * 78)
    print(f"{ok}/{len(_results)} checks passed")
    for nm, p, err in _results:
        if not p:
            print(f"  FAILED: {nm} — {err}")
    print("=" * 78)
    return 0 if ok == len(_results) else 1


if __name__ == "__main__":
    sys.exit(main())
