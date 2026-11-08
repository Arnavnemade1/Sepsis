"""Planted-failure checks on the system replay (backend/ml/system_replay.py).

A replay that measures its own copy of the paging rules, credits pages it should
not, or lets a silent system "win" would publish a false system-level claim. Each
check below plants one of those failures, or drives the live engine beside the
replay, and asserts the difference shows:

  [1] conformance   the live PagingOS and the replay, fed the same readings, open
                    the same pages at the same priorities with the same upgrades
  [2] arms          bare pages every upward crossing; the system pages once per
                    episode, re-pages on shock only after the page is handled
  [3] crediting     a page too early is false, a page after onset is late (not
                    false, not a catch), a page in the window is a catch
  [4] lead time     a stay observed 2 h before onset cannot count toward
                    "warned >= 3 h" in either direction
  [5] verdict       a quieter system that loses catches is refused on sensitivity;
                    a real reduction in repeat pages on noise is accepted
  [6] release       small cells are withheld; no subject id or row leaves
  [7] adapters      shock flags carry forward causally; within-stay sampling is
                    refused; rows round-trip through the lake file

    python -m backend.ml.scripts.verify_system_replay
"""
from __future__ import annotations

import json
import math
import os
import tempfile
from types import SimpleNamespace

import numpy as np

from backend.aegis.paging import detector_episode as DE
from backend.ml import system_replay as SR

PASS, FAIL = "[PASS]", "[FAIL]"
results = []


def check(name: str, ok: bool, detail: str = "") -> None:
    results.append(bool(ok))
    print(f"  {PASS if ok else FAIL} {name}" + (f" — {detail}" if detail else ""))


def stay(index, onset=math.nan, shock=None, subject=1, start=0, weight=1.0, cert=None) -> SR.Stay:
    """A stay given directly on the index scale (p is a placeholder; the arms take index)."""
    n = len(index)
    return SR.Stay(subject, np.arange(start, start + n, dtype=float), np.full(n, 0.5), onset,
                   None if shock is None else np.asarray(shock, bool),
                   None if cert is None else np.asarray(cert, object), weight)


# --------------------------------------------------------------------------- #
def conformance() -> None:
    print("[1] conformance: the live engine and the replay make the same pages")
    from backend.aegis.paging.engine import PagingOS
    from backend.aegis.paging.hospital import ICU_ROOMS

    class Clock:
        t = 1_800_000_000.0

        def __call__(self):
            return self.t

    clk = Clock()
    os_ = PagingOS(clock=clk, seed=5, demo_history=False)
    os_.sync_sbar = True
    os_.settings["simulated_responders"] = False   # nobody answers: every page stays active
    os_.settings["demo_traffic"] = False
    room = ICU_ROOMS[0]
    # hour: 0 quiet, 1 page, 2-3 above/watch, 4-6 three lows -> close, 7 recross while the
    # page is still being handled (no page), 8 shock -> upgrade it, 9-11 lows -> close,
    # the page is then handled (resolved), 12 recross -> a NEW page, 13 shock -> upgrade,
    # 14 a dip, 15 back up (still the same episode: one low is not three)
    idx = [0.50, 0.75, 0.72, 0.65, 0.55, 0.55, 0.55, 0.80, 0.81, 0.50, 0.50, 0.50, 0.85, 0.86, 0.55, 0.90]
    shk = [False] * 8 + [True] + [False] * 4 + [True] + [False] * 2
    resolve_before = 12
    page_active_h = 10.5          # the hour-1 page counts as handled from hour 12 on, as in the engine
    for h, (r, s) in enumerate(zip(idx, shk)):
        clk.t += 3600
        if h == resolve_before:
            for p in os_.pages.values():
                if p["room"] == room and p["source"] == "detector":
                    p["status"] = "resolved"
        b = {"name": "Replay Patient", "enterprise": {"map": 60.0} if s else {"map": 80.0}, "drivers": []}
        os_._episode(room, "bed_0", r, b, clk.t)
    live = sorted((p for p in os_.pages.values() if p["room"] == room and p["source"] == "detector"),
                  key=lambda p: p["created_at"])
    live_upgrades = sum(1 for p in live for e in p["escalation_log"] if "upgraded" in e.get("why", ""))
    rep = SR.replay_system(stay(idx, shock=shk), np.array(idx), page_active_h=page_active_h)
    check("same number of pages", len(live) == len(rep.pages) == 2, f"live {len(live)}, replay {len(rep.pages)}")
    check("same final priorities", [p["priority"] for p in live] == [p.priority for p in rep.pages],
          f"live {[p['priority'] for p in live]}, replay {[p.priority for p in rep.pages]}")
    check("same number of upgrades", live_upgrades == rep.upgrades == 2,
          f"live {live_upgrades}, replay {rep.upgrades}")

    # and the shared rule really is the engine's: change it, and the replay changes with it
    dip = [0.80, 0.55, 0.80]
    real = DE.CLEAR_READINGS
    try:
        DE.CLEAR_READINGS = 1
        broken = SR.replay_system(stay(dip), np.array(dip), page_active_h=0.5)
    finally:
        DE.CLEAR_READINGS = real
    intact = SR.replay_system(stay(dip), np.array(dip), page_active_h=0.5)
    check("the replay reads the engine's own constant (a planted change moves it)",
          (len(intact.pages), len(broken.pages)) == (1, 2), f"{len(intact.pages)} -> {len(broken.pages)} pages")


def arms() -> None:
    print("[2] arms: bare crossings vs system episodes")
    noise = [0.65, 0.72, 0.68, 0.71, 0.69, 0.73, 0.66, 0.74]      # hovers at the line
    bare = SR.replay_bare(stay(noise), np.array(noise))
    sysm = SR.replay_system(stay(noise), np.array(noise), page_active_h=4)
    check("bare pages every upward crossing", len(bare.pages) == 4, f"{len(bare.pages)}")
    check("the system pages that hovering patient once", len(sysm.pages) == 1, f"{len(sysm.pages)}")

    # shock while the page is being handled: an upgrade, not a second page
    up = [0.8, 0.8, 0.8]
    s1 = SR.replay_system(stay(up, shock=[False, True, True]), np.array(up), page_active_h=4)
    check("shock during an active page upgrades it", (len(s1.pages), s1.upgrades) == (1, 1),
          f"pages {len(s1.pages)}, upgrades {s1.upgrades}")
    # the same shock once the page is no longer being handled: a new P1 page
    long = [0.8] * 6
    s2 = SR.replay_system(stay(long, shock=[False] * 5 + [True]), np.array(long), page_active_h=2)
    check("shock after the page was handled re-pages at P1",
          [p.priority for p in s2.pages] == [2, 1], f"{[p.priority for p in s2.pages]}")
    # recross after close while the page is still active: written on it, no page
    rc = [0.8, 0.5, 0.5, 0.5, 0.8]
    s3 = SR.replay_system(stay(rc), np.array(rc), page_active_h=10)
    s4 = SR.replay_system(stay(rc), np.array(rc), page_active_h=2)
    check("a recrossing pages again only once the old page is handled",
          (len(s3.pages), len(s4.pages)) == (1, 2), f"active {len(s3.pages)}, handled {len(s4.pages)}")


def crediting() -> None:
    print("[3] crediting: early, in-window and late pages")
    cfg = SR.ReplayConfig(0.1, 0.2, lead_window_h=6, page_active_h=1)
    s = stay([0.0] * 30, onset=20.0)
    sp = SR.StayPages([SR.Page(2.0, 2), SR.Page(18.0, 2), SR.Page(25.0, 2)])
    t = SR.tally([s], [sp], cfg)
    check("page 18 h early with a 6 h window is false", t.false_pages[0] == 1, f"{t.false_pages[0]}")
    check("page 2 h before onset is the catch", (t.caught[0], t.credited[0]) == (1, 1))
    check("page after onset is late, not false", t.late[0] == 1)
    check("warning is measured from the first credited page", t.warning_h[0] == 2.0, f"{t.warning_h[0]}")
    t0 = SR.tally([s], [SR.StayPages([SR.Page(25.0, 2)])], cfg)
    check("a page only after onset is not a catch", t0.caught[0] == 0)


def lead_time() -> None:
    print("[4] lead time has a fixed denominator")
    cfg = SR.ReplayConfig(0.1, 0.2, lead_window_h=48, page_active_h=1)
    short = SR.Stay(1, np.arange(10, 13, dtype=float), np.full(3, .5), onset=12.0)      # 2 h observed
    longs = SR.Stay(2, np.arange(0, 13, dtype=float), np.full(13, .5), onset=12.0)      # 12 h observed
    t = SR.tally([short, longs], [SR.StayPages([SR.Page(10.0, 2)]), SR.StayPages([SR.Page(11.0, 2)])], cfg)
    m = SR.metrics(t)
    check("a stay seen 2 h pre-onset is outside warned >= 3 h", m["warned_ge_3h"] == 0.0,
          f"{m['warned_ge_3h']} (1 eligible stay, warned 1 h)")
    check("both stays count toward warned >= 1 h", m["warned_ge_1h"] == 1.0, f"{m['warned_ge_1h']}")


def cohort(n=400, seed=0, frac=0.15):
    """Probabilities with noise that hovers at the line on non-septic stays and a ramp
    before onset on septic ones."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        T = int(rng.integers(30, 80))
        septic = rng.random() < frac
        onset = float(int(T * 0.7)) if septic else math.nan
        # log-odds: a per-stay level plus AR(1) noise, so some never-septic stays wander up
        # to the line and hover there (the repeat-page case), and septic stays ramp before onset
        z = np.empty(T)
        z[0] = rng.normal(0, 1)
        for h in range(1, T):
            z[h] = 0.8 * z[h - 1] + rng.normal(0, 0.6)
        lo = rng.normal(-3.5, 0.8) + z
        if septic:
            lo = lo + np.clip((np.arange(T) - onset + 12) / 12, 0, 1) * 3.0
        out.append(SR.Stay(i // 2, np.arange(T, dtype=float), 1 / (1 + np.exp(-lo)), onset))
    return out


def verdicts() -> None:
    print("[5] verdict: quiet is not the same as better")
    stays = cohort()
    cfg = SR.ReplayConfig.from_alert_rate(np.concatenate([s.p for s in stays]), 0.05, 0.15)
    paged = SR.replay(stays, cfg)
    t_bare = SR.tally(stays, paged["bare"], cfg)
    t_sys = SR.tally(stays, paged["system"], cfg)
    v = SR.verdict(SR.paired_bootstrap(stays, t_bare, t_sys, n_boot=300))
    check("episodes on hovering noise are accepted", v["system_better"] is True, v["reason"])

    # plant: a 'system' that keeps only the first page and silently drops a third of septic stays
    rng = np.random.default_rng(1)
    silent = []
    for s, sp in zip(stays, paged["bare"]):
        keep = sp.pages[:1]
        if s.septic and rng.random() < 0.35:
            keep = []
        silent.append(SR.StayPages(keep))
    t_sil = SR.tally(stays, silent, cfg)
    boot = SR.paired_bootstrap(stays, t_bare, t_sil, n_boot=300)
    v2 = SR.verdict(boot)
    check("a system that loses catches is refused on sensitivity",
          v2["system_better"] is False and v2["fewer_false_pages"] and not v2["sensitivity_noninferior"],
          v2["reason"])
    check("the interval covers its own point estimate",
          boot["sensitivity"]["ci_low"] <= boot["sensitivity"]["delta"] <= boot["sensitivity"]["ci_high"])


def release() -> None:
    print("[6] release: aggregates only")
    stays = cohort(n=300, seed=3)
    stays[0] = SR.Stay(987654321, stays[0].hours, stays[0].p, stays[0].onset)
    cfg = SR.ReplayConfig.from_alert_rate(np.concatenate([s.p for s in stays]))
    res = SR.study(stays, cfg, n_boot=100, page_active_grid=(1.0, 4.0))
    blob = json.dumps(res, default=str)
    check("no subject id in the release", "987654321" not in blob)
    check("the release is small (no per-row arrays)", len(blob) < 12_000, f"{len(blob)} bytes")
    sys_pages = [res["page_active_sensitivity"][k]["pages_per_100_patient_days"] for k in ("1h", "4h")]
    check("longer page handling never adds pages", sys_pages[1] <= sys_pages[0] + 1e-12, f"{sys_pages}")

    few = [s for s in stays if not s.septic][:40] + [s for s in stays if s.septic][:5]
    r2 = SR.study(few, cfg, n_boot=50, page_active_grid=(4.0,))
    check("five septic stays: the septic count is written <10", r2["cohort"]["septic_stays"] == "<10")
    check("... the rates are withheld", r2["arms"]["system"]["sensitivity"] is None)
    check("... and there is no verdict", r2["verdict"]["system_better"] is None)


def adapters() -> None:
    print("[7] adapters")
    # two stays: rows in stay/hour order; MAP 60 at hour 2 of stay 0, lactate 5 at hour 0 of stay 1
    nan = np.nan
    X = np.array([[80, nan], [nan, nan], [60, nan], [nan, nan], [nan, 5.0], [nan, nan], [nan, 1.0]], float)
    store = SimpleNamespace(columns=["MAP", "Lactate"], X=X, seg=np.array([0, 0, 0, 0, 1, 1, 1]),
                            hour=np.array([0, 1, 2, 3, 0, 1, 2]), subject=np.array([11, 12]))
    rows = np.arange(7)
    sh = SR.shock_flags(store, rows)
    check("MAP < 65 carries forward within the stay", list(sh[:4]) == [False, False, True, True], f"{sh[:4]}")
    check("lactate >= 4 holds until a new value clears it", list(sh[4:]) == [True, True, False], f"{sh[4:]}")
    check("a value never leaks into the next stay", not sh[4] or X[4, 1] >= 4)
    X2 = X.copy()
    X2[3, 0] = 50
    check("changing a later hour leaves earlier flags alone",
          list(SR.shock_flags(SimpleNamespace(**{**vars(store), "X": X2}), rows)[:3]) == list(sh[:3]))

    led = SimpleNamespace(store=store, eval_mask=np.ones(7, bool), folds=np.array([0, 1]),
                          onset=np.array([2.0, nan]), design_w=np.ones(7))
    p = np.linspace(0.1, 0.7, 7)
    st = SR.stays_from_ledger(led, p)
    check("the ledger gives one stay per stay, in hour order",
          [len(s.hours) for s in st] == [4, 3] and st[0].septic and not st[1].septic)
    led.design_w = np.array([1, 1, 4, 1, 4, 4, 4], float)
    try:
        SR.stays_from_ledger(led, p)
        check("weights that vary inside a stay are refused", False)
    except ValueError:
        check("weights that vary inside a stay are refused", True)

    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "rows.npz")
        stays = cohort(n=20, seed=5)
        SR.export_rows(f, stays)
        back = SR.load_rows(f)
        check("rows round-trip through the lake file",
              len(back) == len(stays) and all(np.array_equal(a.p, b.p) and
                                              (a.onset == b.onset or (math.isnan(a.onset) and math.isnan(b.onset)))
                                              for a, b in zip(stays, back)))
        out = os.path.join(d, "agg.json")
        SR.main([f, "--out", out, "--n-boot", "50"])
        check("the CLI writes an aggregate file", json.load(open(out))["schema"] == "system_replay/1")


def main() -> int:
    for fn in (conformance, arms, crediting, lead_time, verdicts, release, adapters):
        fn()
    n_ok = sum(results)
    print("-" * 72)
    print(f"{n_ok}/{len(results)} system-replay checks passed")
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
