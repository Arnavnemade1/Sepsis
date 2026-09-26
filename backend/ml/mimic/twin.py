"""The Sepsis-3 Process Twin v2 — a temporal model built against what the harness found.

v1 made the two halves of Sepsis-3 OBSERVED targets: on CinC the Genesis Network's AND gate
``sigmoid(a_susp) * sigmoid(a_organ)`` was a prior over two latent factors, and MIMIC-IV
records both for every patient. v2 keeps that gate and adds four things, each aimed at a
weakness the harness (docs/MIMIC.md §4) exposes:

1. RAW SEQUENCE, TWO STREAMS. Inputs come straight from the hourly grid, not from the
   782-column tree feature matrix: per channel its latest value (the PHYSIOLOGY stream)
   and whether and how long ago it was measured (the PROCESS stream), fused through a
   learned gate. A recurrent model should learn its own temporal summaries, and the grid
   is what lets it train on counterfactual patients built on the fly.

2. A PHYSIOLOGY TWIN, TRAINED ON COUNTERFACTUAL PATIENTS. The twin is fitted in one of two
   regimes. ``regime="record"`` reads records as they were: the FULL score, free to use
   every signal, the care process included. ``regime="transplanted"`` never sees a record
   as it was: every batch is rebuilt on random donors' measurement schedules with
   state-matched successor values (the harness's Care-Process Transplant), fresh donors each
   time. Its predictions (``mode="physiology"``, averaged over ``k_donors`` donor
   schedules) form the PHYSIOLOGY score, and the calibrated logit difference between the
   two twins is a PER-ALERT PROCESS ATTRIBUTION: how much of this alert comes from how the
   team is measuring the patient rather than from the patient. Two regimes rather than one
   network doing both, because it was measured: shown both kinds of record, one network
   gives up signal the process legitimately carries (planted world, d = 128: AUROC 0.90 ->
   0.76), and a learned "this record is transplanted" flag did not win it back (0.76 vs
   0.75). Donors come from the TRAINING cohort's schedules and every draw is keyed by
   hashing (stay, hour, channel), so the physiology score is causal and patient-local:
   it faces the admission audit like any other model.

3. DISCRETE-TIME HAZARDS. A censoring-aware hazard head over eight horizon bins out to
   24 h. Every at-risk hour of every stay contributes; a discharge censors, never counts
   as survival it did not observe. The per-bin Bernoulli terms ARE the discrete-time
   survival likelihood. The cumulative incidence ``F(t, H) = P(onset within H hours)`` is
   monotone in H by construction; ``F(t, 6)`` is the natural early-warning score, and the
   curve is what the clock-matched horizon lens asks for.

4. DEFINITION-COMPOSED ORGAN OUTLOOK. Per organ, a three-way forecast of its measured SOFA
   rise six hours ahead (0 / 1 / >= 2 points). The probability that the TOTAL rise reaches
   2 — the organ half of Sepsis-3 — is composed from them exactly, given the hidden state
   (P(R <= 1) = prod p0 + sum_o p1_o prod_{o' != o} p0_o'), and trained on the observed
   total, so the organ forecasts must add up. Every alert can then be read in the
   definition's own language: which organs, how likely, and whether suspicion is coming.

Also kept from v1: the who / when / direct heads, suspicion within 24 h, per-organ rises
within 12 h, and a world model of the physiology stream (next value of every channel).
Causal by construction: a unidirectional GRU, per-timestep layer norm, inputs that look
back only. ``verify_mimic`` §7 rewrites the future and asserts no past output moves; §20
attacks every v2 mechanism with a planted failure.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import numpy as np

from backend.ml.genesis.learners import platt
from backend.ml.genesis.network import Standardiser, _bce, _bce_logp, _device
from backend.ml.mimic import concepts as K
from backend.ml.mimic import counterfactual as CF
from backend.ml.mimic import extract as E
from backend.ml.mimic import sepsis3 as S3

ORGANS = ["resp", "coag", "liver", "cardio", "cns", "renal"]
MAIN = ["direct", "who", "when"]
AUX = ["susp6", "susp24", "organ"] + [f"rise12_{o}" for o in ORGANS]
HAZ_EDGES = (0, 1, 2, 3, 4, 6, 8, 12, 24)          # hazard bins, hours ahead: (0,1], (1,2], ...
N_BINS = len(HAZ_EDGES) - 1
OUTLOOK_H = 6
DENSE = frozenset(E.DENSE_CH)                     # dense channels: 0 = off, never missing
HEADS = ("direct", "who", "when", "s3", "susp6", "organ", "hazard6", "outlook")


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
def input_channels(store: E.MimicStore, tier: str, drop: FrozenSet[str] = frozenset()) -> List[str]:
    return [c for c in E.DYNAMIC if K.TIER_ORDER[store.tiers[c]] <= K.TIER_ORDER[tier] and c not in drop]


#: the arrival context (source, service, comorbidities) is the TREES' job, not the twin's:
#: on a planted world the 33 context columns let the physiology twin's early stopping lock onto
#: between-patient signal before it learned a within-stay heart-rate pattern (0.609 with
#: them, 0.757 without, three seeds each) — the stack combines trees and twin anyway
TWIN_STATIC = [c for c in E.STATIC if c not in E.CONTEXT]


def sequence_inputs(store: E.MimicStore, tier: str = "P", drop: FrozenSet[str] = frozenset(),
                    context: bool = False) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """``(values, process, value_names, process_names)``, one row per grid hour.

    Values: each channel's latest reading (NaN until first measured; standardised to the
    median later) and the static context. Process: measured this hour, log-hours since
    measured, ever measured. Everything looks back only, inside the stay."""
    N = store.n_rows
    idx = np.arange(N)
    rs = store.row_start
    vals, procs, vn, pn = [], [], [], []
    for c in input_channels(store, tier, drop):
        v = store.col(c).astype(np.float64)
        if c in DENSE:
            vals.append(np.nan_to_num(v))
            vn.append(c)
            continue
        obs = np.isfinite(v)
        last = np.maximum.accumulate(np.where(obs, idx, -1))
        last[last < rs] = -1
        has = last >= 0
        vals.append(np.where(has, v[np.maximum(last, 0)], np.nan))
        vn.append(c)
        procs += [obs.astype(np.float64), np.log1p(np.where(has, idx - last, 96)), has.astype(np.float64)]
        pn += [f"{c}__now", f"{c}__logage", f"{c}__ever"]
    for c in (E.STATIC if context else TWIN_STATIC):
        v = store.col(c).astype(np.float64)
        if c in ("ICULOS", "HospAdmTime"):
            v = np.sign(v) * np.log1p(np.abs(v))
        vals.append(v)
        vn.append(c)
    return (np.stack(vals, axis=1).astype(np.float32), np.stack(procs, axis=1).astype(np.float32), vn, pn)


# --------------------------------------------------------------------------- #
# Targets
# --------------------------------------------------------------------------- #
def process_targets(led) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """(target, mask) per head, on every row of the store."""
    s = led.store
    N = s.n_rows
    rs, hour, seg = s.row_start, s.hour, s.seg
    end = (s.starts + s.lens - 1)[seg] - rs          # last hour of the row's stay
    T: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    septic = np.isfinite(led.onset[seg])
    m = led.train_mask.astype(np.float32)
    T["direct"] = (led.y.astype(np.float32), m)
    T["who"] = (septic.astype(np.float32), m)
    T["when"] = (led.y.astype(np.float32), m * septic)
    fsi = s.onsets.first_si.to_numpy()[seg]
    for k, name in ((6, "susp6"), (24, "susp24")):
        happened = np.isfinite(fsi) & (fsi <= hour + k)
        observed = happened | (hour + k <= end)
        T[name] = (happened.astype(np.float32), observed.astype(np.float32))
    col = lambda nm: s.col(nm).astype(np.float64)
    parts = S3.component_points_now(col, rs, hour)                 # (N, 6) carried forward
    rise_parts = S3.measured_rise_parts(parts, rs, 48)             # (N, 6) vs a measured baseline
    hit = rise_parts.sum(axis=1) >= 2
    idx = np.arange(N)
    past = _window_any(hit, rs, 48, 0, end, hour)
    fut = _window_any(hit, rs, 0, 6, end, hour)
    T["organ"] = ((past | fut).astype(np.float32), (past | (hour + 6 <= end)).astype(np.float32))
    for c, o in enumerate(ORGANS):
        up = np.zeros(N, dtype=bool)
        for lag in range(1, 13):
            dst = idx + lag
            ok = dst - rs <= end
            nxt = np.full(N, np.nan)
            nxt[ok] = parts[dst[ok], c]
            up |= np.isfinite(nxt) & np.isfinite(parts[:, c]) & (nxt >= parts[:, c] + 1)
        T[f"rise12_{o}"] = (up.astype(np.float32), (hour + 12 <= end).astype(np.float32))

    # discrete-time hazards: at risk = never septic, or before the onset hour
    orow = np.floor(led.onset[seg])
    at_risk = (~septic | (hour < orow)) & led.train_mask
    e = np.where(septic, orow - hour, np.inf)                      # hours until onset
    ahead = end - hour                                             # future hours observed
    hy = np.zeros((N, N_BINS), dtype=np.float32)
    hm = np.zeros((N, N_BINS), dtype=np.float32)
    for b in range(N_BINS):
        lo, hi = HAZ_EDGES[b], HAZ_EDGES[b + 1]
        event = at_risk & (e > lo) & (e <= hi)
        survived = at_risk & (e > hi) & (ahead >= hi)
        hy[:, b], hm[:, b] = event, event | survived
    T["hazard"] = (hy, hm)
    T["onset6"] = ((at_risk & (e <= 6)).astype(np.float32),
                   (at_risk & ((e <= 6) | (ahead >= 6))).astype(np.float32))

    # organ outlook: each organ's measured rise OUTLOOK_H hours ahead (0 / 1 / >= 2)
    ok = hour + OUTLOOK_H <= end
    cat = np.zeros((N, len(ORGANS)), dtype=np.int64)
    fut_rows = idx[ok] + OUTLOOK_H
    cat[ok] = np.minimum(np.round(rise_parts[fut_rows]), 2).astype(np.int64)
    total = np.zeros(N, dtype=np.float32)
    total[ok] = rise_parts[fut_rows].sum(axis=1) >= 2
    T["outlook"] = (cat, ok.astype(np.float32))
    T["outlook_total"] = (total, ok.astype(np.float32))
    return T


def _window_any(flag: np.ndarray, rs: np.ndarray, back: int, fwd: int, end: np.ndarray,
                hour: np.ndarray) -> np.ndarray:
    n = len(flag)
    idx = np.arange(n)
    out = np.zeros(n, dtype=bool)
    for lag in range(-fwd, back + 1):
        src = idx - lag
        ok = (src >= rs) & (src - rs <= end) & (src >= 0) & (src < n)
        v = np.zeros(n, dtype=bool)
        v[ok] = flag[src[ok]]
        out |= v
    return out


# --------------------------------------------------------------------------- #
# The composed quantities (numpy references; the network uses the torch twins below)
# --------------------------------------------------------------------------- #
def outlook_p_ge2(p: np.ndarray) -> np.ndarray:
    """P(total rise >= 2) from per-organ category probabilities ``p`` (..., 6, 3), assuming
    the organs are independent given the hidden state: 1 - [prod p0 + sum_o p1_o prod p0_o']."""
    p0, p1 = p[..., 0], p[..., 1]
    le1 = np.prod(p0, axis=-1) + sum(p1[..., o] * np.prod(np.delete(p0, o, axis=-1), axis=-1)
                                     for o in range(p.shape[-2]))
    return 1.0 - le1


def cumulative_incidence(h: np.ndarray, horizon: int) -> np.ndarray:
    """F(t, H) = 1 - prod over bins ending by H of (1 - hazard)."""
    k = sum(1 for b in range(N_BINS) if HAZ_EDGES[b + 1] <= horizon)
    return 1.0 - np.prod(1.0 - h[..., :k], axis=-1)


def _t_outlook_log_le1(ol):
    """log P(R <= 1) from outlook logits (..., 6, 3), in log space."""
    import torch
    lo = torch.log_softmax(ol, dim=-1)
    lp0, lp1 = lo[..., 0], lo[..., 1]
    ratio = torch.cat([torch.zeros_like(lp0[..., :1]), lp1 - lp0], dim=-1)
    return lp0.sum(-1) + torch.logsumexp(ratio, dim=-1)


def _t_hazard_margin(hz, horizon: int):
    """logit F(t, horizon) from hazard logits."""
    import torch
    import torch.nn.functional as Fn
    k = sum(1 for b in range(N_BINS) if HAZ_EDGES[b + 1] <= horizon)
    S = torch.clamp(Fn.logsigmoid(-hz[..., :k]).sum(-1), max=-1e-7)          # log(1 - F)
    return torch.log(-torch.expm1(S)) - S


def _t_logp_margin(logp):
    """logit of a probability given as log p."""
    import torch
    logp = torch.clamp(logp, max=-1e-7)
    return logp - torch.log(-torch.expm1(logp))


# --------------------------------------------------------------------------- #
@dataclass
class TwinConfig:
    d_model: int = 128
    layers: int = 2
    dropout: float = 0.1
    lr: float = 2e-3
    weight_decay: float = 1e-5
    batch: int = 32
    #: training length is counted in OPTIMISER STEPS, not epochs: with 90 stays and batches
    #: of 32 an "epoch" is 3 steps, and v1's 8 epochs on the demo were 24 steps — a network
    #: that had barely started learning (a planted signal the booster finds at 0.85 was
    #: reached at 0.58). Validation runs every ``eval_steps`` steps (or every epoch on a
    #: large cohort), early stopping counts evaluations.
    epochs: int = 25
    min_steps: int = 400
    eval_steps: int = 25
    patience: int = 5
    pos_weight: float = 3.0
    w_main: Tuple[float, float, float] = (1.0, 0.5, 0.5)
    w_s3: float = 0.5
    w_factor: float = 0.5          # the gate factors' own ground truth
    w_aux: float = 0.2
    w_hazard: float = 0.5
    w_outlook: float = 0.3
    w_wm: float = 0.1
    tier: str = "P"
    strict: bool = False           # drop the ledger's covert channels (``infoflow``)
    context: bool = False          # read the arrival context too (``TWIN_STATIC`` explains the default)
    #: "record": train on records as they were (the full score); "transplanted": train only
    #: on records rebuilt on random donors' schedules (the physiology score); "both": both
    #: at once (kept for the ablation that showed it costs the full score)
    regime: str = "record"
    k_donors: int = 4              # donor schedules averaged by the physiology score
    device: str = "cpu"
    seed: int = 0


def _build(d_val: int, d_proc: int, cfg: TwinConfig):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    class ProcessTwinNet(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg.d_model
            self.val = nn.Sequential(nn.Linear(d_val, d), nn.GELU())
            self.proc = nn.Sequential(nn.Linear(d_proc, d), nn.GELU())
            self.gate = nn.Parameter(torch.zeros(d))             # sigmoid(0) = 0.5
            self.norm = nn.LayerNorm(d)
            self.drop = nn.Dropout(cfg.dropout)
            self.gru = nn.GRU(d, d, num_layers=cfg.layers, batch_first=True,
                              dropout=cfg.dropout if cfg.layers > 1 else 0.0)
            self.main = nn.Linear(d, len(MAIN))
            self.susp = nn.Linear(d, 1)
            self.organ = nn.Linear(d, 1)
            self.aux = nn.Linear(d, len(AUX) - 2)       # susp24 + the 6 per-organ rises
            self.hazard = nn.Linear(d, N_BINS)
            self.outlook = nn.Linear(d, len(ORGANS) * 3)
            self.wm = nn.Linear(d, d_val)

        def forward(self, xv, xp):
            z = self.val(xv) + torch.sigmoid(self.gate) * self.proc(xp)
            h, _ = self.gru(self.drop(self.norm(z)))
            a_s, a_o = self.susp(h).squeeze(-1), self.organ(h).squeeze(-1)
            return {"main": self.main(h), "a_s": a_s, "a_o": a_o,
                    "s3": Fn.logsigmoid(a_s) + Fn.logsigmoid(a_o), "aux": self.aux(h),
                    "hazard": self.hazard(h), "outlook": self.outlook(h).view(*h.shape[:2], len(ORGANS), 3),
                    "wm": self.wm(h)}

    return ProcessTwinNet()


def _rows(store: E.MimicStore, stays: np.ndarray) -> np.ndarray:
    return np.concatenate([np.arange(store.starts[k], store.starts[k] + store.lens[k]) for k in stays]) \
        if len(stays) else np.zeros(0, np.int64)


@dataclass
class ProcessTwin:
    cfg: TwinConfig = field(default_factory=TwinConfig)
    std_v_: Optional[Standardiser] = None
    std_p_: Optional[Standardiser] = None
    net_: object = None
    calib_: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    calib_phys_: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    bank_: Optional[CF.DonorBank] = None
    ref_: Optional[CF.Reference] = None
    drop_: FrozenSet[str] = frozenset()
    best_epoch_: int = 0             # the optimiser step of the kept weights

    # ---------------- inputs and batches ---------------- #
    def _inputs(self, led_or_view) -> Tuple[np.ndarray, np.ndarray]:
        key = ("twin_seq", self.cfg.tier, self.drop_, self.cfg.context)
        cache = getattr(led_or_view, "cache", None)
        if cache is not None and key in cache:
            return cache[key]
        xv, xp, _, _ = sequence_inputs(led_or_view.store, self.cfg.tier, self.drop_, self.cfg.context)
        if cache is not None:
            cache[key] = (xv, xp)
        return xv, xp

    def _pad(self, store: E.MimicStore, stays: np.ndarray, Xv: np.ndarray, Xp: np.ndarray, dev,
             T: Optional[Dict] = None, row_map: Optional[np.ndarray] = None):
        """Padded (B, T, .) tensors for ``stays`` of ``store``. ``row_map`` maps the padded rows
        to the rows of the TARGET arrays (default: the same rows)."""
        import torch
        starts, lens = store.starts[stays], store.lens[stays]
        B, Tm = len(stays), int(lens.max())
        rows = np.full((B, Tm), -1, dtype=np.int64)
        for b, (a, n) in enumerate(zip(starts, lens)):
            rows[b, :n] = np.arange(a, a + n)
        valid = rows >= 0
        flat = rows[valid]
        xv = np.zeros((B, Tm, Xv.shape[1]), np.float32)
        xp = np.zeros((B, Tm, Xp.shape[1]), np.float32)
        xv[valid], xp[valid] = self.std_v_(Xv[flat]), self.std_p_(Xp[flat])
        ys = {}
        if T is not None:
            tflat = flat if row_map is None else row_map[flat]
            for k, (tgt, msk) in T.items():
                yy = np.zeros((B, Tm) + tgt.shape[1:], tgt.dtype)
                mm = np.zeros((B, Tm) + msk.shape[1:], np.float32)
                yy[valid], mm[valid] = tgt[tflat], msk[tflat]
                ys[k] = (torch.from_numpy(yy).to(dev), torch.from_numpy(mm).to(dev))
            nxt = np.zeros_like(xv)
            nm = np.zeros((B, Tm, 1), np.float32)
            nxt[:, :-1] = xv[:, 1:]
            nm[:, :-1, 0] = valid[:, 1:] & valid[:, :-1]
            ys["_wm"] = (torch.from_numpy(nxt).to(dev), torch.from_numpy(nm).to(dev))
        return torch.from_numpy(xv).to(dev), torch.from_numpy(xp).to(dev), ys, rows, valid

    # ---------------- loss ---------------- #
    def _loss(self, out, ys, pw):
        import torch
        import torch.nn.functional as Fn
        c = self.cfg
        main = out["main"]
        L = 0.0
        for i, (k, w) in enumerate(zip(MAIN, c.w_main)):
            L = L + w * _bce(main[..., i], *ys[k], **({"pos_weight": pw} if k == "direct" else {}))
        L = L + c.w_s3 * _bce_logp(out["s3"], *ys["direct"])
        L = L + c.w_factor * (_bce(out["a_s"], *ys["susp6"]) + _bce(out["a_o"], *ys["organ"]))
        for i, k in enumerate(AUX[1:2] + AUX[3:]):
            L = L + c.w_aux * _bce(out["aux"][..., i], *ys[k])
        L = L + c.w_hazard * _bce(out["hazard"], *ys["hazard"])
        oy, om = ys["outlook"]
        ce = Fn.cross_entropy(out["outlook"].reshape(-1, 3), oy.reshape(-1), reduction="none").view(oy.shape)
        L = L + c.w_outlook * (ce.sum(-1) * om).sum() / om.sum().clamp(min=1.0)
        tot, tm = ys["outlook_total"]
        l1 = torch.clamp(_t_outlook_log_le1(out["outlook"]), max=-1e-6)
        lp = torch.log(-torch.expm1(l1))
        L = L + c.w_outlook * (-(tot * lp + (1 - tot) * l1) * tm).sum() / tm.sum().clamp(min=1.0)
        t, m = ys["_wm"]
        hub = Fn.smooth_l1_loss(out["wm"], t, reduction="none")
        L = L + c.w_wm * (hub * m).sum() / (m.sum() * out["wm"].shape[-1]).clamp(min=1.0)
        return L

    # ---------------- fit ---------------- #
    def fit(self, led, T, train_stays: np.ndarray, verbose: bool = False) -> "ProcessTwin":
        import torch
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        dev = _device(cfg.device)
        rng = np.random.default_rng(cfg.seed)
        s = led.store
        self.drop_ = frozenset(led.covert) if cfg.strict else frozenset()
        Xv, Xp = self._inputs(led)
        # early-stopping slice: 10% of the training SUBJECTS
        subj = s.subject[train_stays]
        us = np.unique(subj)
        val_subj = rng.choice(us, max(1, int(0.1 * len(us))), replace=False)
        is_val = np.isin(subj, val_subj)
        fit_st, val_st = train_stays[~is_val], train_stays[is_val]
        fit_rows = _rows(s, fit_st)
        self.std_v_ = Standardiser.fit(Xv, fit_rows, seed=cfg.seed)
        self.std_p_ = Standardiser.fit(Xp, fit_rows, seed=cfg.seed)
        # the counterfactual machinery comes from the TRAINING stays only
        self.bank_ = CF.DonorBank.from_store(s, fit_st)
        self.ref_ = CF.reference_from(s, fit_st, seed=cfg.seed)
        net = _build(Xv.shape[1], Xp.shape[1], cfg).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
        pw = torch.tensor(cfg.pos_weight, device=dev)
        order = fit_st[np.argsort(s.lens[fit_st], kind="mergesort")]
        groups = [order[i:i + cfg.batch] for i in range(0, len(order), cfg.batch)]
        every = max(len(groups), cfg.eval_steps)
        max_steps = max(cfg.epochs * len(groups), cfg.min_steps)
        best, best_state, bad, step, stop = -1.0, None, 0, 0, False
        t0 = time.time()
        while step < max_steps and not stop:
            for gi in rng.permutation(len(groups)):
                net.train()
                g = groups[gi]
                loss = 0.0
                if cfg.regime in ("record", "both"):
                    xv, xp, ys, _, _ = self._pad(s, g, Xv, Xp, dev, T)
                    loss = loss + self._loss(net(xv, xp), ys, pw)
                if cfg.regime in ("transplanted", "both"):
                    donors = rng.integers(0, len(self.bank_), len(g))
                    cst = CF.transplanted_store(s, g, donors, self.bank_, self.ref_, seed=int(rng.integers(1 << 30)))
                    cv, cp, _, _ = sequence_inputs(cst, cfg.tier, self.drop_, cfg.context)
                    xv2, xp2, ys2, _, _ = self._pad(cst, np.arange(cst.n_patients), cv, cp, dev, T,
                                                    row_map=_rows(s, g))
                    loss = loss + self._loss(net(xv2, xp2), ys2, pw)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                step += 1
                if step % every and step < max_steps:
                    continue
                self.net_ = net
                a = self._val_auroc(led, T, val_st)
                if verbose:
                    print(f"      twin step {step}: val AUROC {a:.4f} [{time.time()-t0:.0f}s]", flush=True)
                sched.step(a)
                if a > best + 1e-4:
                    best, bad, self.best_epoch_ = a, 0, step
                    best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                else:
                    bad += 1
                    if bad >= cfg.patience:
                        stop = True
                if stop or step >= max_steps:
                    break
        net.load_state_dict(best_state)
        self.net_ = net
        self._calibrate(led, T, val_st)
        return self

    def _val_auroc(self, led, T, val_st) -> float:
        """Validation in the regime the twin is FOR: the physiology twin is judged on
        transplanted records, the full twin on records as they were."""
        from backend.ml.genesis.harness import auroc
        mode = "physiology" if self.cfg.regime == "transplanted" else "full"
        pv = self.predict(led, val_st, ("direct",), calibrated=False, mode=mode)["direct"]
        vr = _rows(led.store, val_st)
        ok = T["direct"][1][vr] > 0
        yv = T["direct"][0][vr][ok]
        return auroc(yv, pv[ok]) if 0 < yv.sum() < len(yv) else 0.5

    def _calibrate(self, led, T, val_st) -> None:
        target = {"direct": "direct", "who": "who", "when": "when", "s3": "direct", "susp6": "susp6",
                  "organ": "organ", "hazard6": "onset6", "outlook": "outlook_total"}
        vr = _rows(led.store, val_st)
        for mode, maps, heads in (("full", self.calib_, tuple(target)),
                                  ("physiology", self.calib_phys_, ("s3", "direct"))):
            pv = self.predict(led, val_st, heads, calibrated=False, mode=mode)
            for h in heads:
                t, m = T[target[h]]
                k = m[vr] > 0
                if k.sum() > 10 and 0 < t[vr][k].sum() < k.sum():
                    # design weights: under a subsampled cohort the map must target the FULL
                    # cohort's scale, the same one the tree heads it is multiplied with use
                    maps[h] = platt(pv[h][k], t[vr][k], led.design_w[vr][k])

    # ---------------- predict ---------------- #
    def _forward(self, store: E.MimicStore, Xv: np.ndarray, Xp: np.ndarray, stays: np.ndarray,
                 heads: Sequence[str], organs: bool = False) -> Dict[str, np.ndarray]:
        """Margins per head for every row of ``stays`` of ``store`` (in ``_rows`` order)."""
        import torch
        dev = _device(self.cfg.device)
        net = self.net_
        net.eval()
        rows_all = _rows(store, stays)
        where = np.full(store.n_rows, -1, dtype=np.int64)
        where[rows_all] = np.arange(len(rows_all))
        out = {h: np.empty(len(rows_all)) for h in heads}
        if organs:
            out["organ_p1plus"] = np.empty((len(rows_all), len(ORGANS)))
        with torch.no_grad():
            for i in range(0, len(stays), 64):
                grp = stays[i:i + 64]
                xv, xp, _, rows, valid = self._pad(store, grp, Xv, Xp, dev)
                o = net(xv, xp)
                l1 = _t_outlook_log_le1(o["outlook"])
                vals = {"direct": o["main"][..., 0], "who": o["main"][..., 1], "when": o["main"][..., 2],
                        "s3": _t_logp_margin(o["s3"]), "susp6": o["a_s"], "organ": o["a_o"],
                        "hazard6": _t_hazard_margin(o["hazard"], 6),
                        "outlook": torch.log(-torch.expm1(torch.clamp(l1, max=-1e-7))) - l1}
                dest = where[rows[valid]]
                for h in heads:
                    out[h][dest] = vals[h].cpu().numpy()[valid]
                if organs:
                    p1 = 1.0 - torch.softmax(o["outlook"], dim=-1)[..., 0]
                    out["organ_p1plus"][dest] = p1.cpu().numpy()[valid]
        return out

    def predict(self, led_or_view, stays: np.ndarray, heads: Sequence[str] = HEADS, calibrated: bool = True,
                mode: str = "full") -> Dict[str, np.ndarray]:
        """Margins per head for every row of ``stays`` (in ``_rows`` order). ``mode="full"``
        reads the record as it was; ``mode="physiology"`` averages the margins over
        ``k_donors`` transplanted copies (training-cohort donor schedules, hash-keyed draws)."""
        store = led_or_view.store
        stays = np.asarray(stays, dtype=np.int64)
        if mode == "full":
            Xv, Xp = self._inputs(led_or_view)
            out = self._forward(store, Xv, Xp, stays, heads)
        else:
            out = {h: 0.0 for h in heads}
            pid = store.pid[stays].astype(np.int64)
            for k in range(self.cfg.k_donors):
                donors = np.minimum((CF.hash_uniform(self.cfg.seed, 7919 + k, pid) * len(self.bank_)).astype(np.int64),
                                    len(self.bank_) - 1)
                cst = CF.transplanted_store(store, stays, donors, self.bank_, self.ref_, seed=104729 + k)
                cv, cp, _, _ = sequence_inputs(cst, self.cfg.tier, self.drop_, self.cfg.context)
                o = self._forward(cst, cv, cp, np.arange(cst.n_patients), heads)
                for h in heads:
                    out[h] = out[h] + o[h] / self.cfg.k_donors
        if calibrated:
            maps = self.calib_ if mode == "full" else self.calib_phys_
            for h in heads:
                if h in maps:
                    a, b = maps[h]
                    out[h] = a * out[h] + b
        return out

    def explain(self, led_or_view, stays: np.ndarray, physiology: Optional["ProcessTwin"] = None
                ) -> Dict[str, np.ndarray]:
        """Every hour of ``stays`` in the definition's own language: suspicion coming, each
        organ's chance of a measured rise in 6 h, the composed organ half, the Sepsis-3
        gate, the hazard curve's 6-h risk — and how much of the alert is the care process:
        this (record) twin's calibrated margin minus the ``physiology`` twin's."""
        Xv, Xp = self._inputs(led_or_view)
        store = led_or_view.store
        stays = np.asarray(stays, dtype=np.int64)
        raw = self._forward(store, Xv, Xp, stays, ("s3", "susp6", "outlook", "hazard6"), organs=True)
        full = self.predict(led_or_view, stays, ("s3",))["s3"]
        phys = (physiology or self).predict(led_or_view, stays, ("s3",), mode="physiology")["s3"]
        sig = lambda z: 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
        return {"p_sepsis3": sig(full), "p_sepsis3_physiology": sig(phys), "process_attribution": full - phys,
                "p_suspicion_6h": sig(raw["susp6"]), "p_organ_rise_6h": raw["organ_p1plus"],
                "p_total_rise_ge2_6h": sig(raw["outlook"]), "p_onset_6h": sig(raw["hazard6"])}

    def process_gate(self) -> float:
        import torch
        return float(torch.sigmoid(self.net_.gate.detach()).mean())
