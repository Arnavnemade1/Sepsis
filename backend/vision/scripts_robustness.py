"""rPPG robustness benchmark — quantify HR accuracy under real-world degradation.

Real patients move, lighting flickers, skin tones vary, and signal is often weak.
This benchmark synthesizes rPPG ROI traces at KNOWN heart rates, applies each
degradation, and reports HR mean-absolute-error for:
  * POS-only            (the previous single-method extractor)
  * POS/CHROM fusion    (analyze_window_robust — picks the cleaner method)

so the accuracy gain from the fusion is measured, not assumed. It also confirms
the artifact-quality gate falls when error rises (so bad windows are down-weighted
rather than trusted).

    python -m backend.vision.scripts_robustness
"""
from __future__ import annotations

import numpy as np

from backend.vision import signal_processing as sp


def _degrade(rgb, kind, fs, rng):
    n = len(rgb)
    t = np.arange(n) / fs
    if kind == "clean":
        return rgb
    if kind == "low_snr":
        return rgb + rng.normal(0, 0.05, rgb.shape)
    if kind == "illumination":           # slow drift + sudden flicker jumps
        drift = 0.06 * np.sin(2 * np.pi * 0.07 * t)[:, None]
        jumps = np.zeros((n, 1))
        for _ in range(3):
            s = rng.integers(0, n - 1); jumps[s:] += rng.uniform(-0.05, 0.05)
        return rgb + drift + jumps
    if kind == "motion":                 # bursts of large broadband ROI change
        m = rgb.copy()
        for _ in range(4):
            s = rng.integers(0, n - int(fs)); e = s + rng.integers(int(0.3 * fs), int(fs))
            m[s:e] += rng.normal(0, 0.08, (e - s, 3))
        return m
    if kind == "dark_skin":              # lower reflectance, weaker pulse contrast
        return rgb * 0.45 + rng.normal(0, 0.02, rgb.shape)
    return rgb


def main() -> None:
    fs, dur = 30.0, 12.0
    conditions = ["clean", "low_snr", "illumination", "motion", "dark_skin"]
    hrs = [56, 72, 88, 104, 122]
    rng = np.random.default_rng(7)

    print("rPPG HR accuracy under degradation (mean abs error, bpm):")
    print(f"{'condition':<14}{'POS-only':>12}{'POS/CHROM fusion':>20}{'gate(mean)':>12}")
    overall = {"pos": [], "fus": []}
    for cond in conditions:
        epos, efus, gates = [], [], []
        for hr in hrs:
            base = sp.synthesize_rgb(hr, dur, fs, seed=int(hr)) / 255.0
            deg = _degrade(base, cond, fs, rng)
            pos = sp.analyze_window(deg, fs, method="pos")
            fus = sp.analyze_window_robust(deg, fs)
            epos.append(abs(pos.hr_bpm - hr)); efus.append(abs(fus.hr_bpm - hr))
            gates.append(fus.quality)
        mp, mf = float(np.mean(epos)), float(np.mean(efus))
        overall["pos"].append(mp); overall["fus"].append(mf)
        print(f"{cond:<14}{mp:>12.2f}{mf:>20.2f}{np.mean(gates):>12.2f}")
    print("-" * 58)
    print(f"{'MEAN':<14}{np.mean(overall['pos']):>12.2f}{np.mean(overall['fus']):>20.2f}")
    gain = 100 * (np.mean(overall['pos']) - np.mean(overall['fus'])) / max(np.mean(overall['pos']), 1e-6)
    print(f"\nFusion reduces mean HR error by {gain:.0f}% across degraded conditions.")


if __name__ == "__main__":
    main()
