"""Verification for the trajectory forecaster.

Proves the forecast is real and honest (not an oracle, but a calibrated predictor):

  1. A clearly deteriorating recent window (rising HR/lactate/resp, falling MAP)
     yields trajectory "deteriorating" and a projected future breach.
  2. A stable healthy window yields "stable"/"improving" and no projected breach.
  3. The uncertainty band is ordered (risk_lo ≤ risk_mean ≤ risk_hi) and widens
     with the horizon — honest uncertainty.
  4. End-to-end on real CinC patients: BEFORE labelled onset, deteriorating
     septic patients are forecast to breach — i.e. the forecast adds lead time on
     real data. Reported as measured, not asserted to be perfect.

    .venv/bin/python -m backend.aegis.verify_forecast
"""
from __future__ import annotations

from backend.aegis.cohort import CHANNELS, CohortCluster
from backend.aegis.forecast import SCORED, TrajectoryForecaster


def _ok(msg): print(f"      [PASS] {msg}")


def _window(spec):
    """Build a 12-hour window dict from per-channel (start, end) ramps."""
    n = 12
    return {ch: [spec[ch][0] + (spec[ch][1] - spec[ch][0]) * i / (n - 1) for i in range(n)]
            for ch in SCORED}


def test_deteriorating() -> None:
    print("[1] deteriorating window -> 'deteriorating' + projected breach")
    w = _window({"hr": (82, 98), "temp_c": (37.3, 38.0), "resp_rate": (16, 22),
                 "spo2": (98, 95), "map": (90, 76), "wbc": (9, 12), "lactate": (1.2, 2.2)})
    f = TrajectoryForecaster(8).forecast(w, start_hour=40)
    assert f["trajectory"] == "deteriorating", f["trajectory"]
    assert f["projected_onset_in"] is not None
    _ok(f"trajectory={f['trajectory']}, projected critical in ~{f['projected_onset_in']}h, "
        f"conf {f['confidence']}")


def test_stable() -> None:
    print("[2] stable healthy window -> no projected breach")
    w = _window({"hr": (74, 76), "temp_c": (36.9, 37.0), "resp_rate": (14, 15),
                 "spo2": (98, 98), "map": (88, 90), "wbc": (8, 8), "lactate": (1.0, 1.0)})
    f = TrajectoryForecaster(8).forecast(w, start_hour=20)
    assert f["trajectory"] in ("stable", "improving")
    assert f["projected_onset_in"] is None
    assert f["current_risk"] < 0.4
    _ok(f"trajectory={f['trajectory']}, current risk {f['current_risk']}, no breach projected")


def test_band_ordering() -> None:
    print("[3] uncertainty band ordered + widening")
    w = _window({"hr": (85, 100), "temp_c": (37.4, 38.2), "resp_rate": (17, 23),
                 "spo2": (97, 94), "map": (86, 74), "wbc": (9, 13), "lactate": (1.4, 2.6)})
    # a perfect ramp has zero residuals, hence a zero-width band that "widens" 0 -> 0;
    # measured vitals are noisy, so the window is too
    for ch, amp in (("hr", 4.0), ("map", 5.0), ("resp_rate", 1.5), ("lactate", 0.25)):
        w[ch] = [v + (amp if i % 2 else -amp) for i, v in enumerate(w[ch])]
    f = TrajectoryForecaster(8).forecast(w, start_hour=30)
    for h in range(f["horizon"]):
        assert f["risk_lo"][h] <= f["risk_mean"][h] + 1e-9 <= f["risk_hi"][h] + 1e-9
    widths = [f["risk_hi"][h] - f["risk_lo"][h] for h in range(f["horizon"])]
    assert widths[0] > 0.01 and widths[-1] > widths[0], widths
    _ok(f"lo≤mean≤hi for all horizons; band widens {widths[0]:.2f} -> {widths[-1]:.2f}")


def test_band_two_sided() -> None:
    print("[3b] a two-sided channel: 'better' never crosses into the other abnormality")
    # WBC drifting down to ~4.4 with noise: pushing "better" further DOWN would be
    # leukopenia (a SIRS criterion) and score worse than the mean — the band would
    # then exclude its own mean. Better moves toward normal and stops at its centre.
    w = _window({"hr": (70, 69), "temp_c": (39.0, 38.7), "resp_rate": (19, 18),
                 "spo2": (99, 99), "map": (72, 68), "wbc": (6.5, 4.4), "lactate": (3.6, 3.4)})
    w["wbc"] = [v + (0.9 if i % 2 else -0.9) for i, v in enumerate(w["wbc"])]
    f = TrajectoryForecaster(8).forecast(w, start_hour=13)
    for h in range(f["horizon"]):
        assert f["risk_lo"][h] <= f["risk_mean"][h] + 1e-9 <= f["risk_hi"][h] + 1e-9, \
            (h, f["risk_lo"][h], f["risk_mean"][h], f["risk_hi"][h])
    _ok("lo≤mean≤hi with WBC near the leukopenia line")


def test_band_real(cl: CohortCluster) -> None:
    print("[3c] the band is ordered on every hour of every replayed patient")
    fc, n, bad = TrajectoryForecaster(8), 0, []
    for p in cl.patients:
        for hi in range(2, p._ff.shape[0] + 1):
            w = {ch: [float(p._ff[t, k]) for t in range(max(0, hi - 12), hi)] for k, ch in enumerate(CHANNELS)}
            f = fc.forecast(w, start_hour=hi - 1)
            n += 1
            if any(not (f["risk_lo"][h] <= f["risk_mean"][h] + 1e-9 <= f["risk_hi"][h] + 1e-9)
                   for h in range(f["horizon"])):
                bad.append((p.bed_id, hi - 1))
    assert not bad, f"{len(bad)}/{n} forecasts have a band that excludes its mean, e.g. {bad[:3]}"
    _ok(f"{n} forecasts, every band contains its mean")


def test_real_lead(cl: CohortCluster) -> None:
    print("[4] real CinC patients — forecast adds lead time before onset")
    early = 0
    septic = [p for p in cl.patients if p.is_septic and p.onset_hour and p.onset_hour > 8]
    for p in septic:
        # forecast from a pre-onset window a few hours before labelled onset
        hr = max(8, p.onset_hour - 4)
        wlo = max(0, hr - 12)
        w = {ch: [float(p._ff[t, k]) for t in range(wlo, hr)]
             for k, ch in enumerate(CHANNELS)}
        f = TrajectoryForecaster(8).forecast(w, start_hour=hr)
        if f["trajectory"] == "deteriorating" or f["projected_onset_in"] is not None or f["current_risk"] >= 0.5:
            early += 1
    assert early >= 1
    _ok(f"{early}/{len(septic)} septic patients flagged deteriorating/breaching "
        f"~4 h before labelled onset (real data)")


def main() -> None:
    print("=" * 64)
    print("AegisOS — trajectory forecaster verification")
    print("=" * 64)
    test_deteriorating()
    test_stable()
    test_band_ordering()
    test_band_two_sided()
    cl = CohortCluster(n=12, lane_budget=160, seed=7)
    test_band_real(cl)
    test_real_lead(cl)
    print("-" * 64)
    print("ALL FORECAST CHECKS PASSED")


if __name__ == "__main__":
    main()
