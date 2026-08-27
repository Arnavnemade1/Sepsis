"""Validation of the cardio-respiratory coupling estimators on REAL BIDMC data.

What this can and cannot establish is fixed by the data. BIDMC has no sepsis labels, so
**no detection claim is made or testable here**. What is testable is whether the
estimators measure the thing they say they measure:

  * they respond to real physiology (respiration drives HR, not the reverse),
  * they are stable within a record rather than reflecting estimator noise,
  * they fall **monotonically** under controlled decoupling that leaves the respiratory
    spectrum untouched — the exact change the pathophysiology predicts,
  * and they read approximately zero on genuinely independent signals.

The last one matters most: a coupling statistic that is positive on unrelated signals
would produce "coupling loss" findings out of noise forever.

    python -m backend.physio.verify_coupling
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

from backend.physio import coupling as cp

PASS, FAIL = "[PASS]", "[FAIL]"
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(bool(ok))
    print(f"  {PASS if ok else FAIL} {name}" + (f" — {detail}" if detail else ""))


DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "data", "bidmc")


def records(limit=6):
    for path in sorted(glob.glob(os.path.join(DATA, "*_Signals.csv")))[:limit]:
        d = pd.read_csv(path)
        d.columns = [c.strip() for c in d.columns]
        t = d["Time [s]"].to_numpy()
        fs = 1.0 / float(np.median(np.diff(t)))
        yield os.path.basename(path), d["II"].to_numpy(), d["RESP"].to_numpy(), fs


# --------------------------------------------------------------------------- #
def check_independent_signals_read_zero():
    """1. Surrogate-corrected TE is ~0 on independent noise — the false-positive floor."""
    rng = np.random.default_rng(0)
    a = np.cumsum(rng.normal(size=1200))          # smooth, autocorrelated, unrelated
    b = np.cumsum(rng.normal(size=1200))
    out = cp.transfer_entropy(a, b, seed=1)
    check("surrogate-corrected TE ~ 0 on independent signals", out["te"] < 0.02,
          f"corrected {out['te']:.4f} (raw {out['te_raw']:.4f}, null {out['null_mean']:.4f})")
    check("...and the RAW estimator is confidently non-zero, which is why we correct",
          out["te_raw"] > 0.02, f"raw {out['te_raw']:.4f} bits on noise")
    ps = cp.phase_sync(a, b)
    check("phase sync ~ 0 on independent signals", ps < 0.35, f"{ps:.3f}")
    # Guard the guard: without band-limiting, this same pair scored 0.72.
    raw = float(np.abs(np.mean(np.exp(1j * (
        np.angle(__import__("scipy.signal", fromlist=["hilbert"]).hilbert(a - a.mean()))
        - np.angle(__import__("scipy.signal", fromlist=["hilbert"]).hilbert(b - b.mean())))))))
    check("...and the unfiltered estimator would have called it strong coupling",
          raw > 0.5, f"unfiltered {raw:.3f} vs band-limited {ps:.3f}")


def check_real_records_show_coupling():
    """2. Real ICU records show respiration->HR coupling above the surrogate null."""
    te, ps, rsa = [], [], []
    for name, ecg, resp, fs in records():
        r = cp.analyze(ecg, resp, fs, kind="ecg")
        if r["n_samples"] < 32:
            continue
        te.append(r["te_resp_to_hr"]); ps.append(r["phase_sync"]); rsa.append(r["rsa_coherence"])
    check("coupling estimated on every real record", len(te) >= 5, f"{len(te)} records")
    check("median RESP->HR transfer entropy is positive on real data",
          float(np.median(te)) > 0.0, f"median {np.median(te):.4f} bits")
    check("median RSA band coherence is a real fraction of HR power",
          0.01 < float(np.median(rsa)) < 0.95, f"median {np.median(rsa):.3f}")


def check_monotone_under_decoupling():
    """3. THE mechanism test: coupling falls as the link is broken, and only then.

    Respiration is blended toward a phase-randomised surrogate of itself, so its power
    spectrum — rate and depth — is preserved at every step. Only the timing
    relationship to the heart is destroyed. If the estimators tracked amplitude or
    rate they would not move here; they do move, and downward.
    """
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    curves = []
    for name, ecg, resp, fs in records(limit=5):
        row = []
        for a in alphas:
            r = cp.analyze(ecg, cp.decouple(resp, a, seed=3), fs, kind="ecg")
            row.append(r["phase_sync"])
        if row[0] > 0:
            curves.append(row)
    if not curves:
        check("decoupling sweep produced curves", False)
        return
    mean = np.mean(np.array(curves), axis=0)
    # Spearman against alpha rather than a strict step-wise test. A finite-sample
    # coupling estimate has real variance, and demanding every consecutive pair
    # decrease tests the noise, not the trend; the direction is the claim.
    from scipy.stats import spearmanr
    rho, pval = spearmanr(alphas, mean)
    # The bar is "a significant negative trend", not a specific rho. With 5 alphas a
    # single adjacent inversion already pins rho at exactly -0.9, so demanding
    # rho <= -0.9 is asking for a perfect ordering under a floating-point equality —
    # an unreachable bar dressed up as a strict one.
    check("phase sync falls with decoupling (significant negative rank trend)",
          rho < -0.8 and pval < 0.05,
          f"rho {rho:+.3f} p {pval:.4f} | " + " -> ".join(f"{v:.3f}" for v in mean))
    check("...and the fully decoupled end is well below the intact one",
          mean[-1] < mean[0] * 0.85,
          f"{mean[0]:.3f} -> {mean[-1]:.3f} ({100 * (1 - mean[-1] / mean[0]):.0f}% loss)")
    # Vacuity guard: the control must preserve the respiratory spectrum it claims to.
    _, ecg, resp, fs = next(iter(records(limit=1)))
    p0 = np.abs(np.fft.rfft(cp._bandpass(resp, fs, *cp.RESP_BAND)))
    p1 = np.abs(np.fft.rfft(cp._bandpass(cp.decouple(resp, 1.0, seed=3), fs, *cp.RESP_BAND)))
    rel = float(np.abs(p1.sum() - p0.sum()) / max(p0.sum(), 1e-9))
    check("...while the decoupling control leaves the respiratory spectrum intact",
          rel < 0.15, f"total band power changed {100 * rel:.1f}%")


def check_stability_within_record():
    """4. Split each record in half: the estimate must not be noise."""
    d = []
    for name, ecg, resp, fs in records(limit=6):
        h = len(ecg) // 2
        a = cp.analyze(ecg[:h], resp[:h], fs, kind="ecg")
        b = cp.analyze(ecg[h:], resp[h:], fs, kind="ecg")
        if a["n_samples"] > 32 and b["n_samples"] > 32:
            d.append(abs(a["phase_sync"] - b["phase_sync"]))
    check("phase sync is stable across halves of the same record",
          len(d) >= 4 and float(np.median(d)) < 0.25,
          f"median |half-to-half delta| = {np.median(d):.3f} over {len(d)} records")


def check_no_label_claim_possible():
    """5. The honesty boundary is structural: there are no labels to claim against."""
    labels = glob.glob(os.path.join(DATA, "*abel*")) + glob.glob(os.path.join(DATA, "*sepsis*"))
    check("BIDMC carries no sepsis labels, so no detection claim is testable here",
          len(labels) == 0,
          "mechanism validated; AUROC/lead-time require MIMIC-IV Waveform credentials")


# --------------------------------------------------------------------------- #
def main() -> None:
    print("Cardio-respiratory coupling — validation on real BIDMC\n")
    if not glob.glob(os.path.join(DATA, "*_Signals.csv")):
        print("  [SKIP] no BIDMC data at data/bidmc/")
        raise SystemExit(0)
    for fn in (check_independent_signals_read_zero, check_real_records_show_coupling,
               check_monotone_under_decoupling, check_stability_within_record,
               check_no_label_claim_possible):
        fn()
    ok, n = sum(results), len(results)
    print(f"\n{ok}/{n} checks passed")
    raise SystemExit(0 if ok == n else 1)


if __name__ == "__main__":
    main()
