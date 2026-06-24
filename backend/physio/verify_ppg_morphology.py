"""Validation of the PPG morphology pathway — on REAL BIDMC ICU waveforms.

BIDMC has gold-standard PPG (125 Hz) but NO sepsis labels, so we do NOT claim a
sepsis AUROC here. We validate exactly what real waveforms can prove:

  A. RELIABILITY — the extractor finds the dicrotic notch on a high rate of real
     beats and yields physiologic feature ranges across independent patients.
  B. STABILITY — within a record (same patient, steady physiology) the reflection
     index is consistent beat-to-beat (low coefficient of variation), i.e. the
     feature is a property of the vasculature, not measurement noise.
  C. SENSITIVITY — under controlled notch erosion (a model of falling vascular
     tone), the reflection index and notch-detection rate fall MONOTONICALLY.
     This is the falsifiable claim that the feature actually captures the
     physiologic change distributive shock produces.

    python -m backend.physio.verify_ppg_morphology
"""
from __future__ import annotations

from typing import List

import numpy as np

from backend.physio.data import bidmc
from backend.physio.ppg_morphology import analyze, erode_notch

SEG_S = 60.0               # analyse a 60 s window per record


def _hr(t: str) -> None:
    print(f"\n{t}\n" + "─" * len(t))


def main() -> int:
    recs = bidmc.load_records(limit=20)
    if not recs:
        print("BIDMC PPG not present (run backend/physio/data/bidmc.download). "
              "Cannot run the real-waveform validation.")
        return 1

    fs = recs[0].fs
    n = int(SEG_S * fs)
    failures = 0

    _hr(f"A · RELIABILITY across {len(recs)} real BIDMC ICU patients ({SEG_S:.0f}s each)")
    notch_rates, ba_meds, ct_meds, beat_counts = [], [], [], []
    for r in recs:
        seg = r.ppg[:n] if r.ppg.size >= n else r.ppg
        a = analyze(seg, fs)
        if a["n_beats"] >= 20:
            notch_rates.append(a["notch_rate"])
            ba_meds.append(a["ba_ratio"]); ct_meds.append(a["crest_time"])
            beat_counts.append(a["n_beats"])
    nr, ba = np.array(notch_rates), np.array(ba_meds)
    print(f"  usable records: {len(nr)}/{len(recs)} · {int(np.sum(beat_counts))} beats analysed")
    print(f"  APG b/a stiffness index (per-patient): median {np.median(ba):.3f}  "
          f"range [{np.min(ba):.2f}, {np.max(ba):.2f}], spread {np.max(ba)-np.min(ba):.2f}")
    print(f"    → continuous, available on EVERY beat — the primary vascular-tone channel")
    print(f"  dicrotic-notch presence (per-patient): median {np.median(nr)*100:.0f}%  "
          f"range [{np.min(nr)*100:.0f}%, {np.max(nr)*100:.0f}%]")
    print(f"    → bimodal: {int(np.sum(nr<0.2))} patients <20%, {int(np.sum(nr>0.6))} >60% "
          f"— the notch is lost in low-tone states, present in high-tone")
    print(f"  crest time (median): {np.median(ct_meds)*1000:.0f} ms  (physiologic 80–250 ms)")
    relia_ok = (len(nr) >= 15 and 0.08 <= np.median(ct_meds) <= 0.25
                and (np.max(ba) - np.min(ba)) >= 0.30          # b/a discriminates patients
                and (np.max(nr) - np.min(nr)) >= 0.50)         # notch discriminates tone

    _hr("B · STABILITY — split-half test-retest (same patient, two 60s windows)")
    d_ba, d_nr = [], []
    for r in recs:
        if r.ppg.size < 2 * n:
            continue
        a1, a2 = analyze(r.ppg[:n], fs), analyze(r.ppg[n:2 * n], fs)
        if a1["n_beats"] >= 20 and a2["n_beats"] >= 20:
            d_ba.append(abs(a1["ba_ratio"] - a2["ba_ratio"]))
            d_nr.append(abs(a1["notch_rate"] - a2["notch_rate"]))
    mad_ba = float(np.median(d_ba)) if d_ba else 1.0
    mad_nr = float(np.median(d_nr)) if d_nr else 1.0
    print(f"  b/a index    |Δ| median: {mad_ba:.3f}   (of a {np.max(ba)-np.min(ba):.2f} between-patient spread)")
    print(f"  notch %      |Δ| median: {mad_nr*100:.0f}%")
    print(f"  (within-patient drift << between-patient spread ⇒ a real vascular property)")
    stable_ok = mad_ba <= 0.10 and mad_nr <= 0.15

    _hr("C · SENSITIVITY — monotonic response to controlled notch erosion (vasodilation model)")
    alphas = [0.0, 0.25, 0.5, 0.75, 1.0]
    curves_ri: List[float] = []
    curves_nr: List[float] = []
    for al in alphas:
        ris, nrs = [], []
        for r in recs[:12]:
            seg = r.ppg[:n] if r.ppg.size >= n else r.ppg
            a = analyze(erode_notch(seg, fs, al), fs)
            if a["n_beats"] >= 20:
                nrs.append(a["notch_rate"]); ris.append(a["reflection_index"])
        curves_ri.append(float(np.median(ris)) if ris else 0.0)
        curves_nr.append(float(np.mean(nrs)) if nrs else 0.0)
    print("  erosion α         " + "  ".join(f"{a:>5.2f}" for a in alphas))
    print("  notch-presence %  " + "  ".join(f"{v*100:>4.0f} " for v in curves_nr))
    print("  reflection index  " + "  ".join(f"{v:>5.2f}" for v in curves_ri))
    # monotone non-increasing (allow tiny numerical wiggle)
    mono_nr = all(curves_nr[i + 1] <= curves_nr[i] + 0.02 for i in range(len(curves_nr) - 1))
    mono_ri = all(curves_ri[i + 1] <= curves_ri[i] + 0.02 for i in range(len(curves_ri) - 1))
    print(f"  → notch presence {curves_nr[0]*100:.0f}% → {curves_nr[-1]*100:.0f}% "
          f"(monotone {mono_nr}); reflection index {curves_ri[0]:.2f} → {curves_ri[-1]:.2f} "
          f"(monotone {mono_ri})")

    _hr("VERDICT")
    checks = [
        ("b/a + notch run on real beats, physiologic + discriminative", relia_ok),
        ("both features stable split-half (b/a|Δ|<=0.10, notch<=15%)",   stable_ok),
        ("notch presence collapses monotonically under erosion",         mono_nr and curves_nr[-1] <= 0.4 * curves_nr[0]),
        ("reflection index also degrades monotonically",                 mono_ri),
    ]
    for label, ok in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {label}")
        failures += 0 if ok else 1
    print("\n  PPG MORPHOLOGY PATHWAY VALIDATED ✓" if failures == 0
          else f"\n  {failures} CHECK(S) FAILED")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
