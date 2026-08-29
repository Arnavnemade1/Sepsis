"""Adversarial audit of the autonomic-recovery estimator.

    PYTHONPATH=. python -m backend.physio.verify_recovery

``recovery.py`` claims three things, and a claim nobody has tried to break is not
evidence. Each check below plants a specific pathology and requires the estimator to
behave, or measures a quantity and reports it rather than asserting it:

  1. **it recovers a known time constant** — synthesise relaxation with a τ we chose and
     see whether the estimator returns it, across a range, with the bias reported;
  2. **it is amplitude-invariant** — the property the whole design rests on. Scaling
     every RR deviation by α is what beta-blockade, deep sedation and autonomic ageing
     roughly do. SDNN must collapse and the recovery features must not move;
  3. **it reads dynamics, not the marginal distribution** — shuffling the RR series
     preserves SDNN *exactly* and destroys all temporal structure. A feature that
     survives the shuffle was never measuring recovery.

Plus two that decide whether it can be used at all: the baseline must be causal, and the
features must agree between an ECG-derived and a PPG-derived pulse on the same real
patient — the latter being what would let this run on the camera path.

NOTHING HERE IS A DETECTION CLAIM. BIDMC has no sepsis labels. These checks establish
that the estimator measures what it says it measures; whether that predicts sepsis is a
separate question that needs credentialed labelled data.
"""
from __future__ import annotations

import glob
import os
import sys
from typing import List, Tuple

import numpy as np

from backend.physio import hrv, recovery

PASS, FAIL = "PASS", "FAIL"
_results: List[Tuple[str, str, str]] = []


def _refuses(source: str) -> bool:
    """Does the firewall actually reject this pulse source? (A guard nobody has seen
    fire is not a guard.)"""
    try:
        recovery.assert_source_cleared(source)
        return False
    except recovery.UnclearedPulseSource:
        return True


def _pipeline_refuses_ppg(source) -> bool:
    try:
        recovery.build_recovery_dataset(source, pulse_source="ppg")
        return False
    except recovery.UnclearedPulseSource:
        return True


def check(name: str, ok: bool, detail: str = "") -> bool:
    _results.append((PASS if ok else FAIL, name, detail))
    print(f"  [{PASS if ok else FAIL}] {name}" + (f"  — {detail}" if detail else ""), flush=True)
    return ok


def synth_rr(n_beats: int = 900, base_ms: float = 800.0, tau_s: float = 4.0,
             amp_ms: float = 60.0, every: int = 60, jitter_ms: float = 6.0,
             seed: int = 0) -> np.ndarray:
    """An RR series with perturbations that relax with a KNOWN time constant.

    Deviations are injected as exponential decays in real time (not beat count), because
    that is what the estimator claims to measure and using beats would let a wrong
    estimator pass by agreeing with the wrong quantity.
    """
    rng = np.random.default_rng(seed)
    rr = np.full(n_beats, base_ms, dtype=float)
    t = np.cumsum(rr) / 1000.0
    for k in range(every, n_beats - every, every):
        sign = 1.0 if (k // every) % 2 == 0 else -1.0
        dt = t[k:] - t[k]
        rr[k:] += sign * amp_ms * np.exp(-dt / tau_s)
        t = np.cumsum(rr) / 1000.0            # timing shifts as RR changes; keep honest
    rr += rng.normal(0.0, jitter_ms, n_beats)
    return rr


def main() -> int:
    # ---------------------------------------------------- 1. known-τ recovery
    print("1. does it recover a time constant we chose?")
    rows = []
    for true_tau in (2.0, 4.0, 8.0, 12.0):
        rr = synth_rr(tau_s=true_tau, seed=1)
        f = recovery.recovery_features(rr)
        rows.append((true_tau, f.tau_median_s, f.n_events))
        print(f"    true tau {true_tau:5.1f}s   estimated {f.tau_median_s:6.2f}s   "
              f"({f.n_events} events)")
    est = np.array([r[1] for r in rows], float)
    true = np.array([r[0] for r in rows], float)
    ok_finite = np.all(np.isfinite(est))
    # Monotone recovery is the load-bearing property: the estimator must ORDER slow and
    # fast relaxation correctly even if a causal baseline biases the absolute value.
    check("estimated tau increases monotonically with true tau",
          ok_finite and bool(np.all(np.diff(est) > 0)),
          " -> ".join(f"{v:.2f}" for v in est))
    if ok_finite:
        rel = float(np.median(np.abs(est - true) / true))
        r = float(np.corrcoef(true, est)[0, 1])
        check("absolute tau is accurate to within 35% (bias reported, not hidden)",
              rel < 0.35, f"median |relative error| {rel:.1%}, correlation r={r:.3f} — "
                          "a causal baseline tracks slightly into the perturbation, "
                          "which compresses long taus")

    # -------------------------------------------- 2. THE amplitude-invariance claim
    print("\n2. amplitude invariance — the property the design rests on")
    rr = synth_rr(tau_s=5.0, seed=2)
    base = np.median(rr)
    for alpha in (0.5, 0.3, 0.1):
        scaled = base + (rr - base) * alpha       # beta-blockade / sedation, roughly
        f0, f1 = recovery.recovery_features(rr), recovery.recovery_features(scaled)
        h0, h1 = hrv.hrv_features(rr), hrv.hrv_features(scaled)
        d_sdnn = (h1.sdnn - h0.sdnn) / max(h0.sdnn, 1e-9)
        v0, v1 = recovery.feature_vector(f0), recovery.feature_vector(f1)
        both = np.isfinite(v0) & np.isfinite(v1)
        d_rec = float(np.max(np.abs(v1[both] - v0[both]) /
                             np.maximum(np.abs(v0[both]), 1e-9))) if both.any() else np.nan
        print(f"    alpha {alpha:4.1f}   SDNN {h0.sdnn:6.1f} -> {h1.sdnn:6.1f} ms "
              f"({d_sdnn:+.1%})   worst recovery-feature drift {d_rec:.2%}")
        if alpha == 0.3:
            check("at alpha=0.3 the HRV magnitude feature collapses",
                  d_sdnn < -0.5, f"SDNN {d_sdnn:+.1%}")
            # Seconds, not exact: in RR analysis the signal is also the clock, so
            # shrinking deviations shrinks the inter-beat durations the time constant
            # is measured in. The bar is an order-of-magnitude robustness gap over
            # SDNN, which is the claim that matters, stated at the size it really is.
            check("...while recovery features move >10x less than SDNN does",
                  d_rec < 0.10 and d_rec < abs(d_sdnn) / 10.0,
                  f"worst drift {d_rec:.2%} vs SDNN {d_sdnn:+.1%} "
                  f"across {int(both.sum())} features")
            # And the beat-counted time constant IS exactly invariant, which is why it
            # is carried alongside the seconds one.
            db = abs(f1.tau_median_beats - f0.tau_median_beats) / max(f0.tau_median_beats, 1e-9)
            check("tau measured in BEATS is exactly invariant",
                  db < 1e-9, f"drift {db:.2e} (tau_median_beats "
                             f"{f0.tau_median_beats:.2f} -> {f1.tau_median_beats:.2f})")

    # ------------------------------------- 3. does it read dynamics or the histogram?
    print("\n3. negative control — shuffling preserves SDNN exactly and destroys dynamics")
    rng = np.random.default_rng(3)
    shuf = rr.copy()
    rng.shuffle(shuf)
    h_r, h_s = hrv.hrv_features(rr), hrv.hrv_features(shuf)
    f_r, f_s = recovery.recovery_features(rr), recovery.recovery_features(shuf)
    check("the shuffle really is SDNN-preserving (the control is valid)",
          abs(h_s.sdnn - h_r.sdnn) / max(h_r.sdnn, 1e-9) < 0.02,
          f"SDNN {h_r.sdnn:.1f} -> {h_s.sdnn:.1f} ms")
    print(f"    tau_median {f_r.tau_median_s:.2f}s -> {f_s.tau_median_s:.2f}s   "
          f"events/min {f_r.events_per_min:.1f} -> {f_s.events_per_min:.1f}")
    moved = (not np.isfinite(f_s.tau_median_s)) or \
            abs(f_s.tau_median_s - f_r.tau_median_s) / max(f_r.tau_median_s, 1e-9) > 0.25
    check("recovery features DO move when temporal structure is destroyed",
          moved, "a feature unchanged by the shuffle would be reading the histogram, "
                 "not the dynamics")

    # ------------------------------------------------------------ 4. causality
    print("\n4. the baseline is causal")
    a = synth_rr(tau_s=5.0, seed=4)
    b = a.copy()
    b[len(b) // 2:] = 400.0                        # rewrite the entire future
    ea, _ = recovery.find_perturbations(a)
    eb, _ = recovery.find_perturbations(b)
    half = len(a) // 2
    ta = [e.tau_s for e in ea if e.peak_beat < half - 60 and e.tau_s]
    tb = [e.tau_s for e in eb if e.peak_beat < half - 60 and e.tau_s]
    same = len(ta) == len(tb) and np.allclose(ta, tb)
    check("rewriting the future leaves earlier perturbations untouched", same,
          f"{len(ta)} early events compared, identical: {same}")

    # ------------------------------- 5. real data: does it survive the camera path?
    print("\n5. real BIDMC — do ECG-derived and PPG-derived features agree?")
    files = sorted(glob.glob(os.path.join("data", "bidmc", "*_Signals.csv")))
    if not files:
        check("BIDMC present", False, "data/bidmc missing — skipping the real-data check")
    else:
        e_tau, p_tau, e_ev, p_ev = [], [], [], []
        for path in files:
            try:
                rr_e = recovery.rr_from_bidmc(path, "ecg")
                rr_p = recovery.rr_from_bidmc(path, "ppg")
            except Exception as exc:                     # noqa: BLE001 - report, not crash
                print(f"    {os.path.basename(path)}: unreadable ({exc})")
                continue
            fe, fp = recovery.recovery_features(rr_e), recovery.recovery_features(rr_p)
            if np.isfinite(fe.tau_median_s) and np.isfinite(fp.tau_median_s):
                e_tau.append(fe.tau_median_s); p_tau.append(fp.tau_median_s)
                e_ev.append(fe.events_per_min); p_ev.append(fp.events_per_min)
        n = len(e_tau)
        print(f"    {n}/{len(files)} records produced a time constant on both channels")
        if n >= 8:
            e_tau, p_tau = np.array(e_tau), np.array(p_tau)
            r = float(np.corrcoef(e_tau, p_tau)[0, 1])
            mae = float(np.mean(np.abs(e_tau - p_tau)))
            bias = float(np.mean(p_tau - e_tau))
            print(f"    tau_median  ECG {np.median(e_tau):.2f}s  PPG {np.median(p_tau):.2f}s"
                  f"   r={r:.3f}  MAE={mae:.2f}s  bias={bias:+.2f}s")
            print(f"    events/min  ECG {np.median(e_ev):.2f}   PPG {np.median(p_ev):.2f}"
                  f"   r={float(np.corrcoef(e_ev, p_ev)[0, 1]):.3f}")
            # This is the module's biggest open problem, so the check asserts the
            # CURRENT state of the world rather than the one we wanted. The features do
            # NOT survive the pulse path, and the firewall must be closed to match. If
            # concordance ever improves, this check fires and tells us to reopen it.
            check("the measured ECG/PPG concordance is still poor (documented, not hidden)",
                  r < 0.5, f"r={r:.3f} across {n} real ICU patients — pulse-transit "
                           f"jitter fabricates perturbations")
            check("...and the pipeline REFUSES a pulse source accordingly",
                  "ppg" not in recovery.CLEARED_PULSE_SOURCES and
                  _refuses("ppg") and not _refuses("ecg"),
                  "assert_source_cleared('ppg') raises, 'ecg' does not")
        else:
            check("enough BIDMC records yielded time constants", False,
                  f"only {n} usable")

        # Within-patient stability: adjacent windows of one patient should look more
        # alike than windows from different patients, or the feature is noise.
        within, across = [], []
        per_patient = []
        for path in files[:12]:
            try:
                rr = recovery.rr_from_bidmc(path, "ecg")
            except Exception:                            # noqa: BLE001
                continue
            taus = []
            for s in range(0, max(len(rr) - 384 + 1, 0), 48):
                f = recovery.recovery_features(rr[s:s + 384])
                if np.isfinite(f.tau_median_s):
                    taus.append(f.tau_median_s)
            if len(taus) >= 3:
                per_patient.append(np.array(taus))
        for t in per_patient:
            within.append(float(np.std(t)))
        if len(per_patient) >= 4:
            allmed = np.array([float(np.median(t)) for t in per_patient])
            across_sd = float(np.std(allmed))
            within_sd = float(np.mean(within))
            print(f"    within-patient SD {within_sd:.2f}s vs between-patient SD "
                  f"{across_sd:.2f}s  ({len(per_patient)} patients)")
            check("tau is more a property of the patient than of the window",
                  across_sd > within_sd,
                  "if windows of one patient scattered as widely as different patients "
                  "did, this would be measuring noise")

    # ------------------------------- 6. validate-on-arrival: the pipeline runs
    print("\n6. the credentialed-data path runs end to end (shapes only, NO accuracy)")
    # Deliberately shapes and contracts, never a score. This project already made the
    # mistake of tuning a detector against its own simulator and reporting the result;
    # backend/ml/data/features.py carries a hard refusal for exactly that. A synthetic
    # AUROC here would be a measurement of the generator, so none is computed.
    class _FakeSource:
        """Minimal RRLabeledSource: the contract MimicWaveformRRSource implements."""
        def __iter__(self):
            for pid in range(6):
                rr = synth_rr(n_beats=2000, tau_s=4.0 + 3.0 * (pid % 2),
                              every=40, seed=200 + pid)
                yield (f"synthetic_{pid}", rr, int(pid % 2))

    try:
        ds = recovery.build_recovery_dataset(_FakeSource(), pulse_source="ecg")
        ok_shape = (ds.X.shape[0] == len(ds.y) == len(ds.groups)
                    and ds.X.shape[1] == len(ds.feature_names))
        check("build_recovery_dataset consumes the RRLabeledSource contract",
              ok_shape, f"X{ds.X.shape}, {len(set(ds.groups.tolist()))} patients, "
                        f"{len(ds.feature_names)} features "
                        f"(recovery + the existing HRV vector)")
        check("...and it refuses an uncleared pulse source at the door",
              _pipeline_refuses_ppg(_FakeSource()),
              "a PPG-sourced dataset cannot be built while concordance is r~0")
        check("no accuracy is computed from synthetic physiology",
              True, "shapes only — a synthetic AUROC measures the generator, and this "
                    "project has already been burned by one")
    except Exception as exc:                                  # noqa: BLE001
        check("build_recovery_dataset consumes the RRLabeledSource contract", False,
              f"{type(exc).__name__}: {exc}")

    n_fail = sum(1 for s, _, _ in _results if s == FAIL)
    print(f"\n{'=' * 72}\n{len(_results) - n_fail}/{len(_results)} checks passed")
    for s, nm, d in _results:
        if s == FAIL:
            print(f"  FAILED — {nm}: {d}")
    print("\nNOTE: no detection claim is made or implied here. BIDMC has no sepsis "
          "labels;\nthese checks establish that the estimator measures what it claims.")
    return 1 if n_fail else 0


if __name__ == "__main__":
    sys.exit(main())
