"""System replay: what a hospital would experience from the whole paging path.

Every number this project reports so far is a MODEL number (AUROC, detection at a
fixed alert rate). A hospital does not run a model; it runs a paging system. This
module replays held-out ICU stays hour by hour through the paging OS's own detector
logic and counts what reaches a person:

    pages per 100 patient-days        the alert load the unit carries
    sensitivity                       septic stays paged inside the lead window
    false pages per caught case       the alert-fatigue price of each catch
    warned >= k h                     early warning, over stays observed >= k h pre-onset
    P1 share, pages while uncertain   what the pages are, when they fire

Two arms see the SAME scores, so any difference is the system, not the model:

    bare      page every time the risk index crosses up through the alarm line:
              a model behind a threshold with nothing else around it
    system    ``backend.aegis.paging.detector_episode.step``, imported, not copied:
              hysteresis (close only after three readings under the clear line),
              one patient one page while the page is being handled, P2 -> P1 on
              Sepsis-3 shock criteria

WHAT THIS CANNOT MEASURE
------------------------
Routing (who is paged), acknowledgement and escalation run on a synthetic roster.
MIMIC records no pager, so "the right clinician was paged" has no ground truth here
and is not reported. Nobody acknowledges a page in a replay either: a page counts as
"being handled" for ``page_active_h`` hours after it goes out. That is an
assumption, stated in every result, and ``sensitivity`` to it is part of the output.

THE BAR, FIXED BEFORE ANY HELD-OUT RUN
--------------------------------------
``verdict`` calls the system better than the bare threshold only if, on a paired
patient bootstrap over subjects, BOTH hold:

  * false pages per caught case falls: the 95% interval of (system - bare) is < 0;
  * sensitivity is non-inferior: the interval's lower end is >= -NONINFERIORITY.

Fewer pages is mechanically easy (never page again), so the sensitivity condition
is what stops a quiet system from "winning". A miss is reported as a miss.

LEAD TIME, CAREFULLY
--------------------
A median "hours of warning" mostly measures how long stays are before onset (the
retracted lead-time claim, docs and memory: lead-time null). So warning is reported
as a rate with a fixed denominator: of the septic stays observed at least k hours
before onset, the share paged at least k hours early. The median is kept, labelled.

OUTPUT IS AGGREGATE ONLY
------------------------
``release`` returns counts and rates, never a row, a stay id or an hour. Any count
under ``MIN_CELL`` is written as "<10", and a rate whose denominator is under it is
withheld, following the PhysioNet small-cell rule the egress guard enforces.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from backend.aegis.paging import detector_episode as DE
from backend.aegis.paging.triage import DETECTOR_ALARM
from backend.aegis.ward_risk import IndexMap

MIN_CELL = 10
NONINFERIORITY = 0.02           # sensitivity may fall by at most 2 points (CI lower end)
WARN_HOURS = (1, 3, 6)


# --------------------------------------------------------------------------- #
# inputs
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stay:
    """One held-out ICU stay, in hour order.

    ``p`` is the detector's calibrated probability for each hour, computed from hours
    <= that hour only. ``onset`` is the onset hour, NaN for a stay that never became
    septic. ``shock`` marks hours where Sepsis-3 shock criteria hold (MAP < 65 or
    lactate >= 4, last value carried forward); without it every page is P2.
    ``certainty`` is the conformal state per hour ("septic", "not_septic",
    "uncertain", "conflicting"), if a band was fitted. ``weight`` is the stay's
    design weight (1 / sampling probability).
    """
    subject: int
    hours: np.ndarray
    p: np.ndarray
    onset: float = math.nan
    shock: Optional[np.ndarray] = None
    certainty: Optional[np.ndarray] = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        n = len(self.hours)
        if n == 0 or len(self.p) != n:
            raise ValueError("a stay needs at least one hour and one score per hour")
        if np.any(np.diff(self.hours) <= 0):
            raise ValueError("hours must be strictly increasing")
        for name in ("shock", "certainty"):
            a = getattr(self, name)
            if a is not None and len(a) != n:
                raise ValueError(f"{name} must have one value per hour")
        if not np.all(np.isfinite(self.p)):
            raise ValueError("every replayed hour needs a finite score")
        if not (self.weight > 0):
            raise ValueError("weight must be positive")

    @property
    def septic(self) -> bool:
        return bool(np.isfinite(self.onset))


@dataclass(frozen=True)
class ReplayConfig:
    """The operating point (calibrated probabilities) and the replay's assumptions."""
    p_medium: float
    p_high: float
    #: a page earlier than this before onset is not credited as a catch
    lead_window_h: float = 48.0
    #: a replayed page counts as "being handled" for this long (nobody acks in a replay)
    page_active_h: float = 4.0
    detector: str = "replay"

    def index_map(self) -> IndexMap:
        return IndexMap.pinned(self.p_medium, self.p_high, self.detector)

    @staticmethod
    def from_alert_rate(p: np.ndarray, high_rate: float = 0.05, medium_rate: float = 0.15,
                        **kw: Any) -> "ReplayConfig":
        """Lines at fixed shares of replayed hours: HIGH where ``high_rate`` of hours sit
        above it. Label-free (a quantile of every hour, septic or not), so choosing the
        line on the evaluation stays reads nothing about who became septic. Only the
        ordering of scores matters to a replay (the index map is monotone), so this works
        for any score that is a probability; pass logits through a sigmoid first."""
        p = np.asarray(p, dtype=float)
        p = p[np.isfinite(p)]
        if not (0 < high_rate < medium_rate < 1):
            raise ValueError("need 0 < high_rate < medium_rate < 1")
        if len(p) == 0 or p.min() < 0 or p.max() > 1:
            raise ValueError("scores must be probabilities in [0, 1]")
        hi, med = np.quantile(p, [1 - high_rate, 1 - medium_rate])
        if not (0 < med < hi < 1):
            raise ValueError("scores too tied to place two distinct lines")
        return ReplayConfig(float(med), float(hi), **kw)


# --------------------------------------------------------------------------- #
# the two arms
# --------------------------------------------------------------------------- #
@dataclass
class Page:
    hour: float
    priority: int
    uncertain: bool = False


@dataclass
class StayPages:
    pages: List[Page] = field(default_factory=list)
    upgrades: int = 0


def _priority(stay: Stay, i: int) -> int:
    """The detector triage's priority: P1 on shock criteria, else P2."""
    return 1 if (stay.shock is not None and bool(stay.shock[i])) else 2


def _uncertain(stay: Stay, i: int) -> bool:
    return stay.certainty is not None and str(stay.certainty[i]) == "uncertain"


def replay_bare(stay: Stay, index: np.ndarray) -> StayPages:
    """A threshold with nothing around it: a page on every upward crossing."""
    out = StayPages()
    above = index >= DETECTOR_ALARM
    for i in np.flatnonzero(above & ~np.r_[False, above[:-1]]):
        out.pages.append(Page(float(stay.hours[i]), _priority(stay, i), _uncertain(stay, i)))
    return out


def replay_system(stay: Stay, index: np.ndarray, page_active_h: float) -> StayPages:
    """The paging OS's detector episodes, driven by the product's own ``step``."""
    out = StayPages()
    ep: Optional[Dict[str, Any]] = None
    live: Dict[str, Any] = {"hour": None, "priority": None}     # the page being handled

    for i, (h, r) in enumerate(zip(stay.hours, index)):
        h = float(h)
        active = live["hour"] is not None and h - live["hour"] < page_active_h
        pr = _priority(stay, i)
        ep, action = DE.step(ep, float(r), pr,
                             lambda: live["priority"] if active else None,
                             lambda: active)
        if action in (DE.PAGE, DE.REPAGE):
            out.pages.append(Page(h, pr, _uncertain(stay, i)))
            live.update(hour=h, priority=pr)
        elif action in (DE.UPGRADE, DE.RECROSS_UPGRADE):
            # the page being handled is always the last one sent; it is raised in place
            out.upgrades += 1
            live["priority"] = min(live["priority"], pr)
            out.pages[-1].priority = live["priority"]
    return out


ARMS = ("bare", "system")


def replay(stays: Sequence[Stay], cfg: ReplayConfig) -> Dict[str, List[StayPages]]:
    imap = cfg.index_map()
    out: Dict[str, List[StayPages]] = {a: [] for a in ARMS}
    for s in stays:
        idx = imap.index_array(s.p)
        out["bare"].append(replay_bare(s, idx))
        out["system"].append(replay_system(s, idx, cfg.page_active_h))
    return out


# --------------------------------------------------------------------------- #
# per-stay tallies -> metrics
# --------------------------------------------------------------------------- #
@dataclass
class Tally:
    """Per-stay vectors, so every metric is a ratio of weighted sums (bootstrappable)."""
    w: np.ndarray
    septic: np.ndarray
    days: np.ndarray
    pages: np.ndarray
    caught: np.ndarray
    false_pages: np.ndarray
    credited: np.ndarray
    late: np.ndarray
    p1: np.ndarray
    uncertain: np.ndarray
    upgrades: np.ndarray
    warning_h: np.ndarray          # first credited page -> onset; NaN if not caught
    observed_pre_h: np.ndarray     # first replayed hour -> onset; NaN if not septic


def tally(stays: Sequence[Stay], paged: Sequence[StayPages], cfg: ReplayConfig) -> Tally:
    n = len(stays)
    z = lambda: np.zeros(n)                                              # noqa: E731
    t = Tally(w=np.array([s.weight for s in stays], float), septic=np.array([s.septic for s in stays]),
              days=np.array([len(s.hours) / 24.0 for s in stays]), pages=z(), caught=z(), false_pages=z(),
              credited=z(), late=z(), p1=z(), uncertain=z(), upgrades=z(),
              warning_h=np.full(n, np.nan), observed_pre_h=np.full(n, np.nan))
    for k, (s, sp) in enumerate(zip(stays, paged)):
        hrs = np.array([p.hour for p in sp.pages])
        t.pages[k] = len(hrs)
        t.p1[k] = sum(p.priority == 1 for p in sp.pages)
        t.uncertain[k] = sum(p.uncertain for p in sp.pages)
        t.upgrades[k] = sp.upgrades
        if not s.septic:
            t.false_pages[k] = len(hrs)
            continue
        t.observed_pre_h[k] = s.onset - float(s.hours[0])
        in_window = (hrs >= s.onset - cfg.lead_window_h) & (hrs <= s.onset)
        t.credited[k] = in_window.sum()
        t.late[k] = (hrs > s.onset).sum()
        t.false_pages[k] = (hrs < s.onset - cfg.lead_window_h).sum()
        if in_window.any():
            t.caught[k] = 1.0
            t.warning_h[k] = s.onset - hrs[in_window].min()
    return t


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den > 0 else math.nan


def metrics(t: Tally, m: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Weighted metrics; ``m`` multiplies the weights (bootstrap resample counts)."""
    w = t.w if m is None else t.w * m
    sw = lambda a: float(np.sum(w * a))                                  # noqa: E731
    sep = t.septic.astype(float)
    caught = sw(t.caught)
    out = {
        "pages_per_100_patient_days": 100.0 * _ratio(sw(t.pages), sw(t.days)),
        "sensitivity": _ratio(caught, sw(sep)),
        "false_pages_per_caught_case": _ratio(sw(t.false_pages), caught),
        "pages_per_caught_case": _ratio(sw(t.pages), caught),
        "credited_share_of_pages": _ratio(sw(t.credited), sw(t.pages)),
        "late_share_of_pages": _ratio(sw(t.late), sw(t.pages)),
        "p1_share_of_pages": _ratio(sw(t.p1), sw(t.pages)),
        "uncertain_share_of_pages": _ratio(sw(t.uncertain), sw(t.pages)),
        "upgrades_per_100_patient_days": 100.0 * _ratio(sw(t.upgrades), sw(t.days)),
    }
    for k in WARN_HOURS:
        eligible = sep * (np.nan_to_num(t.observed_pre_h, nan=-1.0) >= k)
        warned = eligible * (np.nan_to_num(t.warning_h, nan=-1.0) >= k)
        out[f"warned_ge_{k}h"] = _ratio(sw(warned), sw(eligible))
    return out


def _median_warning(t: Tally) -> float:
    c = t.caught > 0
    return float(np.median(t.warning_h[c])) if c.any() else math.nan


# --------------------------------------------------------------------------- #
# paired bootstrap over subjects, and the pre-registered verdict
# --------------------------------------------------------------------------- #
DELTAS = ("false_pages_per_caught_case", "sensitivity", "pages_per_100_patient_days", "warned_ge_3h")


def paired_bootstrap(stays: Sequence[Stay], a: Tally, b: Tally, n_boot: int = 1000,
                     seed: int = 0) -> Dict[str, Dict[str, float]]:
    """(b - a) for each delta, resampling SUBJECTS so a patient's stays move together."""
    subj, inv = np.unique([s.subject for s in stays], return_inverse=True)
    rng = np.random.default_rng(seed)
    draws = {k: np.empty(n_boot) for k in DELTAS}
    for i in range(n_boot):
        m = np.bincount(rng.integers(0, len(subj), len(subj)), minlength=len(subj))[inv].astype(float)
        ma, mb = metrics(a, m), metrics(b, m)
        for k in DELTAS:
            draws[k][i] = mb[k] - ma[k]
    point_a, point_b = metrics(a), metrics(b)
    out = {}
    for k in DELTAS:
        d = draws[k][np.isfinite(draws[k])]
        lo, hi = (np.percentile(d, [2.5, 97.5]) if len(d) else (math.nan, math.nan))
        out[k] = {"delta": point_b[k] - point_a[k], "ci_low": float(lo), "ci_high": float(hi),
                  "n_boot_finite": int(len(d))}
    return out


def verdict(boot: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
    fp, sens = boot["false_pages_per_caught_case"], boot["sensitivity"]
    fewer = bool(fp["ci_high"] < 0)
    noninferior = bool(sens["ci_low"] >= -NONINFERIORITY)
    if fewer and noninferior:
        reason = "fewer false pages per caught case, and sensitivity non-inferior"
    elif not fewer:
        reason = "the interval for false pages per caught case does not exclude zero"
    else:
        reason = f"sensitivity may fall by more than {NONINFERIORITY:.2f} (CI lower end {sens['ci_low']:+.3f})"
    return {"system_better": fewer and noninferior, "fewer_false_pages": fewer,
            "sensitivity_noninferior": noninferior, "noninferiority_margin": NONINFERIORITY,
            "reason": reason}


# --------------------------------------------------------------------------- #
# the whole study, released as aggregates only
# --------------------------------------------------------------------------- #
def _cell(n: float) -> Any:
    n = int(round(n))
    return n if n >= MIN_CELL else f"<{MIN_CELL}"


def study(stays: Sequence[Stay], cfg: ReplayConfig, n_boot: int = 1000, seed: int = 0,
          page_active_grid: Sequence[float] = (1.0, 4.0, 12.0)) -> Dict[str, Any]:
    """Run both arms, bootstrap the difference, judge it, and release aggregates."""
    if not stays:
        raise ValueError("no stays to replay")
    paged = replay(stays, cfg)
    t = {a: tally(stays, paged[a], cfg) for a in ARMS}
    boot = paired_bootstrap(stays, t["bare"], t["system"], n_boot, seed)
    n_septic = sum(s.septic for s in stays)
    small = n_septic < MIN_CELL or (len(stays) - n_septic) < MIN_CELL

    # the page-handling assumption, varied: the system arm only (bare does not use it)
    sens = {}
    for h in page_active_grid:
        c2 = ReplayConfig(cfg.p_medium, cfg.p_high, cfg.lead_window_h, h, cfg.detector)
        imap = c2.index_map()
        ts = tally(stays, [replay_system(s, imap.index_array(s.p), h) for s in stays], c2)
        sens[f"{h:g}h"] = _withhold(metrics(ts), small)

    return {
        "schema": "system_replay/1",
        "config": asdict(cfg) | {"alarm_index": DETECTOR_ALARM, "clear_readings": DE.CLEAR_READINGS},
        "cohort": {"stays": _cell(len(stays)), "septic_stays": _cell(n_septic),
                   "subjects": _cell(len({s.subject for s in stays})),
                   "patient_days": _cell(sum(len(s.hours) for s in stays) / 24.0)},
        "arms": {a: _withhold(metrics(t[a]) | {"median_warning_h_caught_only": _median_warning(t[a]),
                                                "pages": _cell(float(np.sum(t[a].pages)))}, small)
                 for a in ARMS},
        "system_minus_bare": boot,
        "verdict": verdict(boot) if not small else {"system_better": None,
                                                    "reason": f"fewer than {MIN_CELL} stays in a class"},
        "page_active_sensitivity": sens,
        "cannot_measure": ["routing correctness (no pager ground truth in MIMIC)",
                           "acknowledgement and escalation timing (synthetic roster)"],
    }


def _withhold(d: Dict[str, Any], small: bool) -> Dict[str, Any]:
    if not small:
        return d
    return {k: (v if isinstance(v, str) else None) for k, v in d.items()} | {
        "withheld": f"fewer than {MIN_CELL} stays in a class"}


# --------------------------------------------------------------------------- #
# adapters
# --------------------------------------------------------------------------- #
def shock_flags(store, rows: np.ndarray) -> np.ndarray:
    """Sepsis-3 shock criteria per row (MAP < 65 or lactate >= 4), last value carried
    forward within the stay, from hours <= the row only. ``rows`` must be in stay and
    hour order (as ``stays_from_ledger`` passes them)."""
    cols = list(store.columns)
    for need in ("MAP", "Lactate"):
        if need not in cols:
            raise KeyError(f"store has no {need!r} column")
    seg = store.seg[rows]
    out = np.zeros(len(rows), dtype=bool)
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    ends = np.r_[starts[1:], len(rows)]
    for name, bad in (("MAP", lambda v: v < 65), ("Lactate", lambda v: v >= 4)):
        v = np.asarray(store.X[rows, cols.index(name)], dtype=float)
        for a, b in zip(starts, ends):
            seg_v = v[a:b]
            idx = np.where(np.isfinite(seg_v), np.arange(b - a), -1)
            np.maximum.accumulate(idx, out=idx)
            filled = np.where(idx >= 0, seg_v[np.maximum(idx, 0)], np.nan)
            with np.errstate(invalid="ignore"):
                out[a:b] |= bad(filled)
    return out


def stays_from_ledger(led, p: np.ndarray, rows: Optional[np.ndarray] = None,
                      shock: bool = True, certainty: Optional[np.ndarray] = None) -> List[Stay]:
    """Held-out stays from a MIMIC ``Ledger`` and an out-of-fold score vector.

    Default rows are the ledger's own evaluation rows on dev folds (septic stays
    scored up to onset + 3 h; onsets in the first hours excluded), the rows every
    reported AUROC uses. Refuses a design that sampled rows inside a stay: an
    episode replay needs each stay's whole series.
    """
    s = led.store
    if rows is None:
        rows = np.flatnonzero(led.eval_mask & np.isfinite(p) & (led.folds[s.seg] >= 0))
    rows = rows[np.lexsort((s.hour[rows], s.seg[rows]))]
    seg = s.seg[rows]
    starts = np.flatnonzero(np.r_[True, seg[1:] != seg[:-1]])
    ends = np.r_[starts[1:], len(rows)]
    sh = shock_flags(s, rows) if shock else None
    subj = s.subject
    out: List[Stay] = []
    for a, b in zip(starts, ends):
        r = rows[a:b]
        w = led.design_w[r]
        if not np.all(w == w[0]):
            raise ValueError("design weights vary inside a stay: rows were sampled within stays, "
                             "and a replay needs whole series")
        out.append(Stay(subject=int(subj[seg[a]]), hours=s.hour[r].astype(float), p=p[r].astype(float),
                        onset=float(led.onset[seg[a]]), shock=None if sh is None else sh[a:b],
                        certainty=None if certainty is None else certainty[r], weight=float(w[0])))
    return out


ROW_FIELDS = ("stay", "subject", "hour", "p", "onset", "weight")


def export_rows(path: str, stays: Sequence[Stay]) -> None:
    """Write stays as flat per-row arrays (``.npz``). These are patient rows: write them
    only inside the lake, never into the repo or a release. ``study`` is what leaves."""
    n = [len(s.hours) for s in stays]
    arr = {"stay": np.repeat(np.arange(len(stays)), n),
           "subject": np.repeat([s.subject for s in stays], n),
           "hour": np.concatenate([s.hours for s in stays]),
           "p": np.concatenate([s.p for s in stays]),
           "onset": np.repeat([s.onset for s in stays], n),
           "weight": np.repeat([s.weight for s in stays], n)}
    if all(s.shock is not None for s in stays):
        arr["shock"] = np.concatenate([s.shock for s in stays])
    np.savez_compressed(path, **arr)


def load_rows(path: str) -> List[Stay]:
    """Inverse of ``export_rows``: rows sorted by (stay, hour), one Stay per stay."""
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    missing = [k for k in ROW_FIELDS if k not in d]
    if missing:
        raise KeyError(f"rows file lacks {missing}")
    order = np.lexsort((d["hour"], d["stay"]))
    d = {k: v[order] for k, v in d.items()}
    starts = np.flatnonzero(np.r_[True, d["stay"][1:] != d["stay"][:-1]])
    ends = np.r_[starts[1:], len(order)]
    out = []
    for a, b in zip(starts, ends):
        for k in ("subject", "onset", "weight"):
            v = d[k][a:b]
            same = bool(np.all(v == v[0])) or (v.dtype.kind == "f" and bool(np.all(np.isnan(v))))
            if not same:
                raise ValueError(f"{k} varies inside a stay")
        out.append(Stay(int(d["subject"][a]), d["hour"][a:b].astype(float), d["p"][a:b].astype(float),
                        float(d["onset"][a]), d["shock"][a:b].astype(bool) if "shock" in d else None,
                        None, float(d["weight"][a])))
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse
    import json

    from backend.ml.jsonio import json_safe

    ap = argparse.ArgumentParser(description="Replay held-out stays through the paging OS's "
                                             "detector logic; write aggregates only.")
    ap.add_argument("rows", help="per-row .npz from export_rows (inside the lake)")
    ap.add_argument("--out", required=True, help="aggregate JSON to write")
    ap.add_argument("--p-high", type=float, help="HIGH line, calibrated probability")
    ap.add_argument("--p-medium", type=float, help="MEDIUM line, calibrated probability")
    ap.add_argument("--high-rate", type=float, default=0.05,
                    help="without --p-high: HIGH at this share of hours (label-free)")
    ap.add_argument("--medium-rate", type=float, default=0.15)
    ap.add_argument("--lead-window-h", type=float, default=48.0)
    ap.add_argument("--page-active-h", type=float, default=4.0)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    stays = load_rows(a.rows)
    kw = dict(lead_window_h=a.lead_window_h, page_active_h=a.page_active_h)
    if (a.p_high is None) != (a.p_medium is None):
        ap.error("--p-high and --p-medium go together")
    cfg = ReplayConfig(a.p_medium, a.p_high, **kw) if a.p_high is not None else \
        ReplayConfig.from_alert_rate(np.concatenate([s.p for s in stays]), a.high_rate, a.medium_rate, **kw)
    res = study(stays, cfg, n_boot=a.n_boot, seed=a.seed)
    with open(a.out, "w") as fh:
        json.dump(json_safe(res), fh, indent=1)
    v = res["verdict"]
    print(f"system better: {v['system_better']} ({v['reason']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
