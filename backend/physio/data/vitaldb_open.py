"""VitalDB (OPEN ACCESS) — the confound-controlled retest of the recovery hypothesis.

WHY THERE IS A SECOND DATASET
-----------------------------
``mimic3_open.py`` ran the recovery-time-constant hypothesis against circulatory
decompensation and returned a clean null (``docs/RECOVERY_NULL.md``): τ added
−0.071 / −0.022 / −0.066 over HRV magnitude at 15/30/60-minute leads, and *every* feature
family including plain heart rate sat at chance. Two explanations were left standing, and
they call for different things:

1. **Underpowered** — 29 dual-label patients, confidence half-widths ±0.12.
2. **The outcome was dirty** — a large share of ICU hypotensive episodes are
   *iatrogenic*: a sedation bolus, a position change, dialysis initiation. Those are
   unpredictable from prior autonomic state by construction, and MIMIC-III's open
   waveform database carries no medication chart with which to remove them.

VitalDB fixes both, which is why it is here rather than a bigger MIMIC pull.

WHAT THIS DATASET HAS THAT THE LAST ONE DID NOT
------------------------------------------------
* **Scale.** 3,702 cases carry both ``SNUADC/ECG_II`` and ``Solar8000/ART_MBP`` against
  29 usable MIMIC patients, and a whole three-hour case loads in about three seconds from
  AWS open data rather than fifteen to thirty seconds per 75-minute window.
* **Beat timing at 500 Hz** rather than 125 Hz. RR precision goes from 8 ms to 2 ms, and
  τ is estimated from RR deviations, so this is a direct improvement in the measurement
  rather than a convenience.
* **The intervention, as a continuous signal.** ``Orchestra/PPF20_CE`` and
  ``Orchestra/RFTN20_CE`` are modelled *effect-site concentrations* of propofol and
  remifentanil — anaesthetic depth, sampled through the case, on 1,914 of those cases.
  A fall in blood pressure that follows the anaesthetist deepening the anaesthetic is a
  pharmacology event, not a physiology event, and here it can be identified and removed.
  That is exactly the confound the MIMIC null could not exclude.

THE HONEST COST OF THE TRADE
-----------------------------
These are surgical patients under general anaesthesia, and anaesthesia *suppresses*
autonomic function — the very control loop τ measures. So this cohort is cleaner and far
larger, and it is also further from the ICU sepsis patient the project cares about. A
null here is weaker evidence against τ-for-sepsis than a null on ICU data would be, and a
positive here does not establish τ-for-sepsis either. It tests the mechanism — does
autonomic recovery slow before circulatory decompensation, when the drug that could
explain the decompensation is measured and excluded — on the best data reachable without
credentialing. Anything written from this module states that boundary.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.physio import hrv
from backend.physio.data import netguard

# The layer that was missing when this module hung at 1541 of 1914 cases.
netguard.install_timeouts()

#: Track names. ECG is the predictor channel; ART_MBP is the outcome channel, from a
#: separate transducer, which keeps the label non-circular exactly as in mimic3_open.
T_ECG = "SNUADC/ECG_II"
T_MBP = "Solar8000/ART_MBP"
T_PPF_CE = "Orchestra/PPF20_CE"      # propofol effect-site concentration
T_RFTN_CE = "Orchestra/RFTN20_CE"    # remifentanil effect-site concentration

ECG_FS = 500.0
NUM_INTERVAL_S = 2.0

#: Outcome definition, identical to the MIMIC-III arm so the two are comparable.
MAP_THRESHOLD_MMHG = 65.0
EVENT_MIN_MINUTES = 5.0
STABLE_MIN_MINUTES = 10.0
STABLE_MAP_MMHG = 75.0

#: Physiologic bounds on the arterial mean. Outside these the line is disconnected,
#: zeroed or being flushed — plumbing, not patient.
ABP_MIN_MMHG = 20.0
ABP_MAX_MMHG = 300.0

#: The confound gate. An event is called ANAESTHETIC-DRIVEN, and excluded, when either
#: effect-site concentration rose by more than this fraction of its own level over the
#: window before the event. A relative test rather than an absolute one because the
#: concentrations run on quite different scales (propofol µg/mL, remifentanil ng/mL) and
#: because what matters is a *change in depth*, not a depth.
CE_RISE_FRACTION = 0.15
CE_LOOKBACK_MIN = 10.0


_retry = netguard.retry


# --------------------------------------------------------------------------- #
# Cohort selection
# --------------------------------------------------------------------------- #
def cohort(require_ce: bool = True) -> List[int]:
    """Case ids carrying the tracks this experiment needs.

    ``require_ce`` restricts to cases that also record both anaesthetic effect-site
    concentrations. That is the scientifically preferable cohort — it is the subset on
    which the intervention confound can actually be removed — at the cost of roughly half
    the cases.
    """
    import pandas as pd
    tr = _retry(lambda: pd.read_csv("https://api.vitaldb.net/trks"))
    if tr is None:
        return []
    need = {T_ECG, T_MBP} | ({T_PPF_CE, T_RFTN_CE} if require_ce else set())
    have = tr.groupby("caseid")["tname"].apply(set)
    return sorted(int(c) for c, s in have.items() if need <= s)


# --------------------------------------------------------------------------- #
# Discovery: outcomes and the drug that might explain them
# --------------------------------------------------------------------------- #
@dataclass
class CaseOutcome:
    caseid: int
    t_s: np.ndarray                  # seconds from case start, regular 2 s grid
    map_mmhg: np.ndarray             # NaN where the arterial line is not believable
    ppf_ce: np.ndarray
    rftn_ce: np.ndarray
    event_times_s: List[float]       # spontaneous decompensations
    drug_driven_s: List[float]       # decompensations explained by deepening anaesthesia
    hours: float


def _valid_map(x: np.ndarray) -> np.ndarray:
    return np.where(np.isfinite(x) & (x > ABP_MIN_MMHG) & (x < ABP_MAX_MMHG), x, np.nan)


def _bridge_short_gaps(m: np.ndarray, dt_s: float, max_gap_s: float = 180.0) -> np.ndarray:
    """Interpolate dropouts shorter than ``max_gap_s``; leave longer ones NaN.

    Carried over from the MIMIC arm, where compacting the series instead fragmented
    genuine half-hour episodes into sub-5-minute runs and discarded 3,673 candidate
    events while zero survived.
    """
    x = np.asarray(m, dtype=np.float64).copy()
    good = np.isfinite(x)
    if good.sum() < 2:
        return x
    max_n = int(max_gap_s / max(dt_s, 1e-9))
    idx = np.flatnonzero(good)
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_n + 1:
            x[a + 1:b] = np.linspace(x[a], x[b], b - a + 1)[1:-1]
    return x


def _ce_rose(ce: np.ndarray, i: int, need: int) -> bool:
    """Did this effect-site concentration climb materially before index ``i``?"""
    a = max(0, i - need)
    seg = ce[a:i + 1]
    seg = seg[np.isfinite(seg)]
    if seg.size < 3:
        return False
    base = float(np.median(seg[:max(len(seg) // 3, 1)]))
    peak = float(np.max(seg))
    if base <= 0:
        return peak > 0
    return (peak - base) / base > CE_RISE_FRACTION


def find_events(t_s: np.ndarray, map_mmhg: np.ndarray, ppf: np.ndarray,
                rftn: np.ndarray) -> Tuple[List[float], List[float]]:
    """Split decompensations into spontaneous and anaesthetic-driven.

    The definition of the event itself is byte-for-byte the MIMIC one — MAP below 65
    sustained five minutes, preceded by a contiguous ten-minute run whose *median* is at
    or above 75 — so the two arms of this investigation answer the same question. What is
    new is the second return value: the events a rising anaesthetic concentration can
    explain, which are reported rather than silently dropped, because how many there are
    is itself the answer to why the MIMIC arm may have failed.
    """
    m = np.asarray(map_mmhg, dtype=np.float64)
    known = np.isfinite(m)
    low = known & (m < MAP_THRESHOLD_MMHG)
    dt = float(np.median(np.diff(t_s))) if len(t_s) > 1 else NUM_INTERVAL_S
    need_low = max(int(EVENT_MIN_MINUTES * 60.0 / dt), 2)
    need_stable = max(int(STABLE_MIN_MINUTES * 60.0 / dt), 2)
    need_ce = max(int(CE_LOOKBACK_MIN * 60.0 / dt), 3)
    gap = ~known

    spontaneous: List[float] = []
    drug: List[float] = []
    i, n = 0, len(low)
    while i < n:
        if not low[i]:
            i += 1
            continue
        j = i
        while j < n and low[j] and not gap[j]:
            j += 1
        if j - i >= need_low:
            a = max(0, i - need_stable)
            prior = m[a:i]
            enough = np.isfinite(prior).sum() >= 0.8 * need_stable
            if enough and float(np.nanmedian(prior)) >= STABLE_MAP_MMHG:
                if _ce_rose(ppf, i, need_ce) or _ce_rose(rftn, i, need_ce):
                    drug.append(float(t_s[i]))
                else:
                    spontaneous.append(float(t_s[i]))
        i = max(j, i + 1)
    return spontaneous, drug


def discover(caseid: int) -> Optional[CaseOutcome]:
    """Load one case's numerics and locate its decompensations."""
    import vitaldb
    arr = _retry(lambda: vitaldb.load_case(
        caseid, [T_MBP, T_PPF_CE, T_RFTN_CE], NUM_INTERVAL_S))
    if arr is None or arr.ndim != 2 or arr.shape[0] < 60:
        return None
    t = np.arange(arr.shape[0], dtype=np.float64) * NUM_INTERVAL_S
    mp = _bridge_short_gaps(_valid_map(arr[:, 0]), NUM_INTERVAL_S)
    if np.isfinite(mp).sum() < 60:
        return None
    ppf = arr[:, 1] if arr.shape[1] > 1 else np.full(len(t), np.nan)
    rftn = arr[:, 2] if arr.shape[1] > 2 else np.full(len(t), np.nan)
    spont, drug = find_events(t, mp, ppf, rftn)
    return CaseOutcome(caseid=caseid, t_s=t, map_mmhg=mp, ppf_ce=ppf, rftn_ce=rftn,
                       event_times_s=spont, drug_driven_s=drug,
                       hours=float(t[-1] / 3600.0) if len(t) else 0.0)


def control_times(out: CaseOutcome, n_controls: int = 3, guard_min: float = 30.0,
                  seed: int = 0) -> List[float]:
    """Stable moments far from any decompensation, and with the anaesthetic steady.

    Controls must be matched on the confound as well as on the patient: a control drawn
    while the anaesthetist was deepening the anaesthetic would differ from an event
    window in drug trajectory rather than in physiology, which would let the model win by
    reading the pharmacology.
    """
    rng = np.random.default_rng(seed)
    guard = guard_min * 60.0
    dt = NUM_INTERVAL_S
    need_ce = max(int(CE_LOOKBACK_MIN * 60.0 / dt), 3)
    all_events = list(out.event_times_s) + list(out.drug_driven_s)
    lo = float(out.t_s[0]) + 20.0 * 60.0        # skip induction
    hi = float(out.t_s[-1]) - 5.0 * 60.0
    if hi <= lo:
        return []
    picked: List[float] = []
    for _ in range(300):
        t = float(rng.uniform(lo, hi))
        if any(abs(t - e) < guard for e in all_events):
            continue
        i = int(t / dt)
        w = (out.t_s >= t - STABLE_MIN_MINUTES * 60.0) & (out.t_s <= t)
        seg = out.map_mmhg[w]
        if np.isfinite(seg).sum() < 0.8 * (STABLE_MIN_MINUTES * 60.0 / dt):
            continue
        if float(np.nanmedian(seg)) < STABLE_MAP_MMHG:
            continue
        if _ce_rose(out.ppf_ce, i, need_ce) or _ce_rose(out.rftn_ce, i, need_ce):
            continue
        if all(abs(t - c) > guard for c in picked):
            picked.append(t)
        if len(picked) >= n_controls:
            break
    return picked


# --------------------------------------------------------------------------- #
# Retrieval: beat timing for one window
# --------------------------------------------------------------------------- #
def fetch_rr(caseid: int, start_s: float, end_s: float
             ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """RR intervals (ms) and their times for one window of a case.

    ``vitaldb.load_case`` returns whole tracks, so the window is sliced after loading.
    A three-hour 500 Hz ECG is ~3 s over the wire, which makes that acceptable here in a
    way it never was for MIMIC's per-window HTTP range reads.
    """
    import vitaldb
    arr = _retry(lambda: vitaldb.load_case(caseid, [T_ECG], 1.0 / ECG_FS))
    if arr is None or arr.size == 0:
        return None
    ecg = arr[:, 0].astype(np.float64)
    a = int(max(start_s, 0.0) * ECG_FS)
    b = int(min(end_s * ECG_FS, len(ecg)))
    if b - a < ECG_FS * 60:
        return None
    seg = ecg[a:b]
    if np.isfinite(seg).mean() < 0.8:
        return None
    peaks = hrv.detect_ecg_rpeaks(np.where(np.isfinite(seg), seg, 0.0), ECG_FS)
    if len(peaks) < 200:
        return None
    rr = hrv.rr_intervals_ms(peaks, ECG_FS)
    rr_t = start_s + peaks[1:] / ECG_FS
    return rr, rr_t


def rhythm_ok(rr_ms: np.ndarray) -> Tuple[bool, str]:
    """Reject irregularly-irregular rhythms, where a recovery constant is meaningless.

    Same rule as the MIMIC arm: in atrial fibrillation the RR series has no operating
    point to return to, so a 'recovery from a deviation' is not a physiologic quantity.
    """
    nn = hrv.clean_nn(np.asarray(rr_ms, dtype=np.float64))
    if len(nn) < 60:
        return False, "too few clean beats"
    d = np.abs(np.diff(nn))
    pnn50 = float((d > 50.0).mean())
    if pnn50 > 0.5:
        return False, f"pnn50 {pnn50:.0%} — irregularly irregular"
    return True, ""
