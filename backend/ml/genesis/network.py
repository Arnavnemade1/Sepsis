"""The Genesis Network — a causal sequence model whose output layer is Sepsis-3.

Sepsis-3 is a conjunction: suspected infection AND an acute organ-dysfunction rise.
Neither half is recorded in CinC 2019 — there are no cultures, no antibiotics — but
each half leaves observable traces:

* suspicion leaves a trace in **clinician behaviour**: the work-up widens, a lactate
  and a blood gas are drawn, many labs are ordered at once;
* organ dysfunction leaves a trace in **the labs and vitals that feed SOFA**.

The network reads a stay hour by hour with a GRU (strictly causal: the hidden state
at ``t`` is a function of hours ``<= t``) and is trained on three families of targets
at once:

``direct / who / when``  the challenge label, the patient label, and the label inside
                         septic stays (the same factorisation the tree heads use);
``aux``                  the observable traces of each half — lactate / gas / new-test /
                         lab-burst in the next hours (suspicion), SOFA-proxy rise,
                         hypotension, fever (organ dysfunction and infection);
``s3``                   the STRUCTURED head: two latent factors, ``a_susp`` and
                         ``a_organ``, combined as ``P = sigmoid(a_susp) * sigmoid(a_organ)``
                         — an AND gate, the definition's shape as an architectural prior.
                         Each factor is also tied to its own auxiliaries (``a_susp``
                         predicts the suspicion traces, ``a_organ`` the organ traces), so
                         the gate's two inputs mean what their names say.

Every head's output is a margin that the GENESIS stage-2 meta-learner can use.

SELF-SUPERVISED PRETRAINING (``GenesisNetwork.pretrain``). Before any sepsis label is
seen, the encoder can be trained on two pretext tasks that need no labels and respect
causality: (1) block-masked value reconstruction — hide a channel's values for 1-6
consecutive hours and reconstruct the measured ones from everything else the network can
see at that hour; (2) the world-model forecasts. Hiding a channel also hides every
composite computed from it (shock index = HR / SBP would otherwise hand back a hidden HR
exactly, and the pretext would teach arithmetic instead of physiology). Fine-tuning then
runs exactly as training from scratch, starting from the pretrained weights.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.ml.genesis.harness import Cohort
from backend.ml.genesis.rawstore import DYNAMIC

#: Auxiliary traces, split by which half of Sepsis-3 they evidence.
SUSPICION_AUX = ["fut_lactate6", "fut_gas6", "fut_newlab6", "fut_burst6"]
ORGAN_AUX = ["fut_sofa2_24", "fut_hypot6", "fut_fever12"]
MAIN = ["direct", "who", "when"]


# --------------------------------------------------------------------------- #
# Inputs
# --------------------------------------------------------------------------- #
CONTEXT = ["Age", "Gender", "unit_sicu", "HospAdmTime", "ICULOS", "hour",
           "sofa_proxy", "sofa_rise24", "sirs", "qsofa_proxy", "shock_index",
           "proc__labs6", "proc__new24", "proc__draws24"]

#: Composite inputs and the channels they are computed from. When pretraining hides a
#: channel, these are hidden with it — otherwise the pretext leaks its own answer.
DERIVED = {
    "sofa_proxy": ["Platelets", "Bilirubin_total", "Creatinine", "MAP", "FiO2", "SaO2", "O2Sat"],
    "sofa_rise24": ["Platelets", "Bilirubin_total", "Creatinine", "MAP", "FiO2", "SaO2", "O2Sat"],
    "sirs": ["Temp", "HR", "Resp", "PaCO2", "WBC"],
    "qsofa_proxy": ["Resp", "SBP"],
    "shock_index": ["HR", "SBP"],
}
N_CH = len(DYNAMIC)
VALUE_COLS = np.arange(N_CH) * 4        # value / now / logage / ever per channel
NOW_COLS = VALUE_COLS + 1


def input_matrix(c: Cohort) -> Tuple[np.ndarray, List[str]]:
    """Compact per-hour input: for each of the 34 channels its last value, whether it
    was measured this hour, log-hours since measured and whether it ever was; plus
    context and a few composites. Values are left raw here; ``Standardiser`` fits
    robust scales on training rows only."""
    ctxs = CONTEXT
    need = [n for ch in DYNAMIC for n in (ch, f"{ch}__age")] + ctxs
    block = c.cols(need)
    get = {n: block[:, j] for j, n in enumerate(need)}
    cols, names = [], []
    for ch in DYNAMIC:
        v = get[ch]
        age = get[f"{ch}__age"]
        ever = np.isfinite(v)
        cols += [v, (age == 0).astype(np.float32), np.log1p(np.nan_to_num(age, nan=96.0)),
                 ever.astype(np.float32)]
        names += [ch, f"{ch}__now", f"{ch}__logage", f"{ch}__ever"]
    for ctx in ctxs:
        v = np.array(get[ctx], dtype=np.float32)
        if ctx == "HospAdmTime":
            v = np.sign(v) * np.log1p(np.abs(v))
        if ctx in ("ICULOS", "hour"):
            v = np.log1p(np.maximum(v, 0))
        cols.append(v)
        names.append(ctx)
    return np.stack(cols, axis=1), names


def world_targets(c: Cohort, horizon: int = 6) -> Dict[str, np.ndarray]:
    """Self-supervised targets: for every hour and every one of the 34 channels, the
    NEXT observed value within ``horizon`` hours (NaN if none) and whether there is one.

    68 targets per hour, on every hour of every stay — the densest supervision the data
    holds, and none of it is the sepsis label. The loss is masked to hours whose next
    ``horizon`` hours are recorded (``weight``): a truncated future is not evidence of
    "not measured", and learning where records end would teach the network the CinC
    truncation rather than physiology."""
    raw = c.raw
    N = raw.n_rows
    idx = np.arange(N, dtype=np.int64)
    end = (raw.starts + raw.lens - 1)[c.seg]
    big = np.iinfo(np.int64).max
    V = np.full((N, len(DYNAMIC)), np.nan, dtype=np.float32)
    E = np.zeros((N, len(DYNAMIC)), dtype=np.float32)
    for j, ch in enumerate(DYNAMIC):
        v = raw.col(ch)
        obs = np.isfinite(v)
        arr = np.where(obs, idx, big)
        nxt_ge = np.minimum.accumulate(arr[::-1])[::-1]          # first obs at >= i
        nxt = np.r_[nxt_ge[1:], big]                             # first obs at > i
        ok = (nxt <= np.minimum(idx + horizon, end))
        V[ok, j] = v[nxt[ok]]
        E[ok, j] = 1.0
    return {"value": V, "event": E, "weight": (idx + horizon <= end).astype(np.float32)}


@dataclass
class Standardiser:
    med: np.ndarray
    scale: np.ndarray

    @staticmethod
    def fit(X: np.ndarray, rows: np.ndarray, sample: int = 200_000, seed: int = 0) -> "Standardiser":
        rng = np.random.default_rng(seed)
        r = rows if len(rows) <= sample else rng.choice(rows, sample, replace=False)
        Z = X[r]
        med = np.nanmedian(Z, axis=0)
        q75, q25 = np.nanpercentile(Z, 75, axis=0), np.nanpercentile(Z, 25, axis=0)
        scale = np.where((q75 - q25) > 1e-6, q75 - q25, np.nanstd(Z, axis=0))
        scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0)
        return Standardiser(np.nan_to_num(med), scale)

    def __call__(self, Z: np.ndarray) -> np.ndarray:
        out = (Z - self.med) / self.scale
        return np.clip(np.nan_to_num(out, nan=0.0), -6, 6).astype(np.float32)


# --------------------------------------------------------------------------- #
@dataclass
class NetConfig:
    d_model: int = 128
    layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch: int = 32
    epochs: int = 12
    patience: int = 2
    pos_weight: float = 3.0
    w_who: float = 0.5
    w_when: float = 0.5
    w_s3: float = 0.5
    w_aux: float = 0.2
    #: world-model weight: forecasting every channel's next value and next measurement
    w_wm: float = 0.1
    wm_horizon: int = 6
    #: halve the learning rate when validation AUROC stops improving
    plateau_decay: bool = True
    #: self-supervised pretraining (``GenesisNetwork.pretrain``): 34 mask-indicator
    #: inputs, the fraction of (channel, hour) values hidden, block length, loss weights
    mask_channels: bool = False
    pretrain_epochs: int = 8
    mvr_rate: float = 0.15
    mvr_max_len: int = 6
    w_mvr: float = 1.0
    w_wm_pre: float = 1.0
    device: str = "auto"
    seed: int = 0


def _device(pref: str):
    import torch
    if pref != "auto":
        return torch.device(pref)
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def _build(d_in: int, cfg: NetConfig, n_susp: int, n_org: int, n_ch: int = len(DYNAMIC)):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    class GenesisNet(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg.d_model
            self.inp = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(cfg.dropout))
            self.gru = nn.GRU(d, d, num_layers=cfg.layers, batch_first=True,
                              dropout=cfg.dropout if cfg.layers > 1 else 0.0)
            self.main = nn.Linear(d, len(MAIN))
            # the two latent halves of Sepsis-3, each read out by its own traces
            self.susp = nn.Linear(d, 1)
            self.org = nn.Linear(d, 1)
            self.susp_aux = nn.Linear(d + 1, n_susp)
            self.org_aux = nn.Linear(d + 1, n_org)
            # physiological world model: next value (regression) + next measurement
            self.wm = nn.Linear(d, 2 * n_ch)
            # masked-value reconstruction (pretraining pretext): the hidden values NOW
            self.mvr = nn.Linear(d, n_ch)

        def forward(self, x):
            h, _ = self.gru(self.inp(x))
            a_s, a_o = self.susp(h), self.org(h)
            s3_logp = Fn.logsigmoid(a_s) + Fn.logsigmoid(a_o)        # log P(AND)
            aux_s = self.susp_aux(torch.cat([h, a_s], -1))
            aux_o = self.org_aux(torch.cat([h, a_o], -1))
            return (self.main(h), s3_logp.squeeze(-1), aux_s, aux_o, a_s.squeeze(-1),
                    a_o.squeeze(-1), self.wm(h), self.mvr(h))

    return GenesisNet()


def _bce_logp(logp, y, mask):
    """BCE for a model that outputs log P directly (the AND gate)."""
    import torch
    logp = torch.clamp(logp, max=-1e-6)
    log1mp = torch.log(-torch.expm1(logp))
    loss = -(y * logp + (1 - y) * log1mp)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


def _bce(logit, y, mask, pos_weight=None):
    import torch.nn.functional as Fn
    loss = Fn.binary_cross_entropy_with_logits(logit, y, reduction="none", pos_weight=pos_weight)
    return (loss * mask).sum() / mask.sum().clamp(min=1.0)


class _Batches:
    """Patients bucketed by length so a batch pads little."""

    def __init__(self, c: Cohort, patients: np.ndarray, batch: int, shuffle: bool, seed: int):
        self.c, self.batch, self.shuffle = c, batch, shuffle
        self.rng = np.random.default_rng(seed)
        order = patients[np.argsort(c.raw.lens[patients], kind="mergesort")]
        self.groups = [order[i:i + batch] for i in range(0, len(order), batch)]

    def __iter__(self):
        idx = self.rng.permutation(len(self.groups)) if self.shuffle else range(len(self.groups))
        for i in idx:
            yield self.groups[i]


@dataclass
class GenesisNetwork:
    cfg: NetConfig = field(default_factory=NetConfig)
    std_: Optional[Standardiser] = None
    net_: object = None
    best_epoch_: int = 0
    history_: List[Dict[str, float]] = field(default_factory=list)
    #: Platt map per head, fitted on the patient-held-out early-stopping slice. Five fold
    #: networks otherwise emit margins on five scales (pos_weight alone shifts ``direct``
    #: by ~log 3), and a product of experts pooled across folds would inherit that.
    calib_: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # ------------------------------------------------------------------ #
    def _tensors(self, c: Cohort, X: np.ndarray, T: Dict[str, Tuple[np.ndarray, np.ndarray]],
                 pts: np.ndarray, dev, mask_rng: Optional[np.random.Generator] = None,
                 with_world: Optional[bool] = None):
        """Batch tensors. With ``mask_rng`` (pretraining) value blocks are hidden and the
        reconstruction targets are returned in ``ys["mvr_t"] / ys["mvr_m"]``."""
        import torch
        starts, lens = c.raw.starts[pts], c.raw.lens[pts]
        B, Tm = len(pts), int(lens.max())
        rows = np.full((B, Tm), -1, dtype=np.int64)
        for b, (a, n) in enumerate(zip(starts, lens)):
            rows[b, :n] = np.arange(a, a + n)
        valid = rows >= 0
        flat = rows[valid]
        n_mask = N_CH if getattr(self.cfg, "mask_channels", False) else 0
        x = np.zeros((B, Tm, X.shape[1] + n_mask), dtype=np.float32)
        raw_block = X[flat]
        x[valid, :X.shape[1]] = self.std_(raw_block)
        ys, ms = {}, {}
        if mask_rng is not None:
            now = np.zeros((B, Tm, N_CH), dtype=bool)
            now[valid] = raw_block[:, NOW_COLS] > 0.5
            M, tv = self._mask_values(x, valid, now, mask_rng, X.shape[1])
            ys["mvr_t"] = torch.from_numpy(tv).to(dev)
            ys["mvr_m"] = torch.from_numpy((M & now).astype(np.float32)).to(dev)
        W = getattr(self, "_world", None)
        if with_world is None:
            with_world = bool(len(T))
        if W is not None and with_world:
            n_ch = W["value"].shape[1]
            vcols = np.arange(n_ch) * 4                     # channel value columns of X
            vv = np.zeros((B, Tm, n_ch), dtype=np.float32)
            vm = np.zeros((B, Tm, n_ch), dtype=np.float32)
            ee = np.zeros((B, Tm, n_ch), dtype=np.float32)
            em = np.zeros((B, Tm, 1), dtype=np.float32)
            raw_v = W["value"][flat]
            zv = (raw_v - self.std_.med[vcols]) / self.std_.scale[vcols]
            fin = np.isfinite(zv)
            vv[valid] = np.clip(np.nan_to_num(zv), -6, 6)
            vm[valid] = fin * W["weight"][flat][:, None]
            ee[valid] = W["event"][flat]
            em[valid] = W["weight"][flat][:, None]
            for k, a in (("wm_v", vv), ("wm_vm", vm), ("wm_e", ee), ("wm_em", em)):
                ys[k] = torch.from_numpy(a).to(dev)
        for k, (tgt, w) in T.items():
            yy = np.zeros((B, Tm), dtype=np.float32)
            mm = np.zeros((B, Tm), dtype=np.float32)
            yy[valid] = tgt[flat]
            mm[valid] = (w[flat] > 0)
            ys[k] = torch.from_numpy(yy).to(dev)
            ms[k] = torch.from_numpy(mm).to(dev)
        return torch.from_numpy(x).to(dev), ys, ms, rows, valid

    def _mask_values(self, x: np.ndarray, valid: np.ndarray, now: np.ndarray,
                     rng: np.random.Generator, n_in: int):
        """Hide blocks of 1..``mvr_max_len`` hours per channel, IN PLACE on ``x``.

        Returns the boolean mask (B, T, 34) and the standardised true values (targets are
        read only where the channel was actually measured that hour). Every composite
        derived from a hidden channel is hidden at the same hour."""
        cfg = self.cfg
        B, Tm, _ = x.shape
        L = int(cfg.mvr_max_len)
        p_start = float(cfg.mvr_rate) / ((1 + L) / 2.0)
        starts = (rng.random((B, Tm, N_CH)) < p_start) & valid[..., None]
        length = rng.integers(1, L + 1, size=(B, Tm, N_CH))
        M = np.zeros((B, Tm, N_CH), dtype=bool)
        for l in range(L):
            M[:, l:, :] |= (starts & (length > l))[:, :Tm - l, :]
        M &= valid[..., None]
        tv = x[..., VALUE_COLS].copy()
        x[..., VALUE_COLS] = np.where(M, 0.0, x[..., VALUE_COLS])
        base = N_CH * 4
        for name, chans in DERIVED.items():
            col = base + CONTEXT.index(name)
            dep = np.zeros((B, Tm), dtype=bool)
            for ch in chans:
                dep |= M[..., DYNAMIC.index(ch)]
            x[..., col] = np.where(dep, 0.0, x[..., col])
        x[..., n_in:n_in + N_CH] = M.astype(np.float32)
        return M, tv

    def _wm_loss(self, wm, ys):
        import torch.nn.functional as Fn
        n_ch = ys["wm_v"].shape[-1]
        pv, pe = wm[..., :n_ch], wm[..., n_ch:]
        hub = Fn.smooth_l1_loss(pv, ys["wm_v"], reduction="none")
        lv = (hub * ys["wm_vm"]).sum() / ys["wm_vm"].sum().clamp(min=1.0)
        be = Fn.binary_cross_entropy_with_logits(pe, ys["wm_e"], reduction="none")
        m = ys["wm_em"].expand_as(be)
        le = (be * m).sum() / m.sum().clamp(min=1.0)
        return lv + le

    def _mvr_loss(self, mvr, ys):
        import torch.nn.functional as Fn
        hub = Fn.smooth_l1_loss(mvr, ys["mvr_t"], reduction="none")
        return (hub * ys["mvr_m"]).sum() / ys["mvr_m"].sum().clamp(min=1.0)

    def _loss(self, out, ys, ms, pw):
        import torch
        import torch.nn.functional as Fn
        main, s3, aux_s, aux_o, _, _, wm, _ = out
        cfg = self.cfg
        L = _bce(main[..., 0], ys["direct"], ms["direct"], pw)
        L = L + cfg.w_who * _bce(main[..., 1], ys["who"], ms["who"])
        L = L + cfg.w_when * _bce(main[..., 2], ys["when"], ms["when"])
        L = L + cfg.w_s3 * _bce_logp(s3, ys["direct"], ms["direct"])
        for j, k in enumerate(SUSPICION_AUX):
            L = L + cfg.w_aux * _bce(aux_s[..., j], ys[k], ms[k])
        for j, k in enumerate(ORGAN_AUX):
            L = L + cfg.w_aux * _bce(aux_o[..., j], ys[k], ms[k])
        if cfg.w_wm > 0 and "wm_v" in ys:
            L = L + cfg.w_wm * self._wm_loss(wm, ys)
        return L

    def pretrain(self, c: Cohort, X: np.ndarray, pts: np.ndarray,
                 world: Optional[Dict[str, np.ndarray]] = None,
                 verbose: bool = True) -> "GenesisNetwork":
        """Self-supervised pretraining on ``pts`` — inputs only, no sepsis label.

        Loss = w_mvr * masked-value reconstruction + w_wm_pre * world-model forecasts.
        ``pretext_`` records, per epoch, the reconstruction loss on a held-out slice next
        to two trivial baselines (predict the median; carry the last visible value
        forward) — the pretext has been learned only if the model beats both."""
        import torch
        cfg = self.cfg
        if not cfg.mask_channels:
            raise ValueError("pretraining needs mask_channels=True")
        torch.manual_seed(cfg.seed)
        dev = _device(cfg.device)
        self._world = world
        rng = np.random.default_rng(cfg.seed + 1000)
        mon = np.sort(rng.choice(pts, max(1, int(0.05 * len(pts))), replace=False))
        fit_pts = np.setdiff1d(pts, mon)
        self.std_ = Standardiser.fit(X, c.rows_of(fit_pts), seed=cfg.seed)
        net = _build(X.shape[1] + N_CH, cfg, len(SUSPICION_AUX), len(ORGAN_AUX)).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.pretext_ = []
        for ep in range(cfg.pretrain_epochs):
            t0 = time.time()
            net.train()
            tot, nb = 0.0, 0
            for grp in _Batches(c, fit_pts, cfg.batch, True, cfg.seed + 100 + ep):
                x, ys, _, _, _ = self._tensors(c, X, {}, grp, dev, mask_rng=rng,
                                               with_world=world is not None)
                out = net(x)
                loss = cfg.w_mvr * self._mvr_loss(out[7], ys)
                if world is not None:
                    loss = loss + cfg.w_wm_pre * self._wm_loss(out[6], ys)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach().cpu())
                nb += 1
            rep_ = self._pretext_report(c, X, mon, net, dev)
            rep_.update({"epoch": ep, "loss": tot / max(nb, 1), "sec": time.time() - t0})
            self.pretext_.append(rep_)
            if verbose:
                print(f"      pretrain epoch {ep}: loss {rep_['loss']:.4f}  held-out recon "
                      f"{rep_['model']:.4f} vs median {rep_['median']:.4f} / carry-forward "
                      f"{rep_['carry']:.4f}  [{rep_['sec']:.0f}s]", flush=True)
        self.pre_state_ = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        self.net_ = net
        return self

    def _pretext_report(self, c: Cohort, X: np.ndarray, mon: np.ndarray, net, dev) -> Dict[str, float]:
        """Held-out reconstruction loss (fixed masking seed) vs two trivial baselines."""
        import torch
        import torch.nn.functional as Fn
        rng = np.random.default_rng(12345)
        net.eval()
        acc = {"model": 0.0, "median": 0.0, "carry": 0.0, "n": 0.0}
        with torch.no_grad():
            for grp in _Batches(c, mon, 64, False, 0):
                x, ys, _, _, valid = self._tensors(c, X, {}, grp, dev, mask_rng=rng, with_world=False)
                pred = net(x)[7]
                t, m = ys["mvr_t"], ys["mvr_m"]
                # carry-forward: the most recent VISIBLE (unmasked) value of the channel
                xv = x[..., VALUE_COLS].cpu().numpy()
                hidden = x[..., X.shape[1]:X.shape[1] + N_CH].cpu().numpy() > 0.5
                carry = np.zeros_like(xv)
                last = np.zeros(xv[:, 0].shape, dtype=np.float32)
                for ti in range(xv.shape[1]):
                    last = np.where(hidden[:, ti], last, xv[:, ti])
                    carry[:, ti] = last
                carry = torch.from_numpy(carry).to(dev)
                for k, p_ in (("model", pred), ("median", torch.zeros_like(t)), ("carry", carry)):
                    acc[k] += float((Fn.smooth_l1_loss(p_, t, reduction="none") * m).sum())
                acc["n"] += float(m.sum())
        n = max(acc.pop("n"), 1.0)
        return {k: v / n for k, v in acc.items()}

    def fit(self, c: Cohort, X: np.ndarray, T: Dict[str, Tuple[np.ndarray, np.ndarray]],
            train_pts: np.ndarray, verbose: bool = True,
            world: Optional[Dict[str, np.ndarray]] = None,
            init: Optional["GenesisNetwork"] = None) -> "GenesisNetwork":
        """Supervised training. ``init`` — a pretrained network — supplies the starting
        weights AND the input scaling it was pretrained with; everything else (schedule,
        early stopping, heads, calibration) is identical to training from scratch."""
        import torch
        cfg = self.cfg
        self._world = world if cfg.w_wm > 0 else None
        torch.manual_seed(cfg.seed)
        dev = _device(cfg.device)
        rng = np.random.default_rng(cfg.seed)
        # patient-grouped early-stopping slice from the TRAINING patients only
        sep = c.plabel[train_pts] == 1
        val = np.concatenate([rng.choice(train_pts[sep], max(1, int(0.1 * sep.sum())), replace=False),
                              rng.choice(train_pts[~sep], max(1, int(0.1 * (~sep).sum())), replace=False)])
        fit_pts = np.setdiff1d(train_pts, val)
        rows_fit = c.rows_of(fit_pts)
        n_in = X.shape[1] + (N_CH if cfg.mask_channels else 0)
        net = _build(n_in, cfg, len(SUSPICION_AUX), len(ORGAN_AUX)).to(dev)
        if init is not None:
            self.std_ = init.std_
            net.load_state_dict(init.pre_state_)
        else:
            self.std_ = Standardiser.fit(X, rows_fit, seed=cfg.seed)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        pw = torch.tensor(cfg.pos_weight, device=dev)
        sched = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
                 if cfg.plateau_decay else None)
        best, best_state, bad = -1.0, None, 0
        for ep in range(cfg.epochs):
            t0 = time.time()
            net.train()
            tot, nb = 0.0, 0
            for pts in _Batches(c, fit_pts, cfg.batch, True, cfg.seed + ep):
                x, ys, ms, _, _ = self._tensors(c, X, T, pts, dev)
                loss = self._loss(net(x), ys, ms, pw)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                tot += float(loss.detach().cpu())
                nb += 1
            self.net_ = net
            pv = self.predict(c, X, val, heads=("direct",), calibrated=False)["direct"]
            vrows = c.rows_of(val)
            from backend.ml.genesis.harness import auroc
            a = auroc(c.y[vrows], pv)
            self.history_.append({"epoch": ep, "loss": tot / max(nb, 1), "val_auroc": a,
                                  "sec": time.time() - t0})
            if verbose:
                print(f"      epoch {ep}: loss {tot/max(nb,1):.4f}  val AUROC {a:.4f}  "
                      f"[{time.time()-t0:.0f}s]", flush=True)
            if sched is not None:
                sched.step(a)
            if a > best + 1e-4:
                best, bad, self.best_epoch_ = a, 0, ep
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
        net.load_state_dict(best_state)
        self.net_ = net
        from backend.ml.genesis.learners import platt
        heads = ("direct", "who", "when", "s3")
        pv = self.predict(c, X, val, heads=heads, calibrated=False)
        vrows = c.rows_of(val)
        septic = (c.plabel[c.seg[vrows]] == 1).astype(float)
        tgt = {"direct": (c.y[vrows], None), "who": (septic, None),
               "when": (c.y[vrows], septic), "s3": (c.y[vrows], None)}
        for h in heads:
            t, w = tgt[h]
            keep = slice(None) if w is None else w > 0
            self.calib_[h] = platt(pv[h][keep], np.asarray(t, float)[keep])
        return self

    def predict(self, c: Cohort, X: np.ndarray, pts: np.ndarray,
                heads: Sequence[str] = ("direct", "who", "when", "s3", "a_susp", "a_organ"),
                calibrated: bool = True) -> Dict[str, np.ndarray]:
        """Margins per head for every row of ``pts``, in ``c.rows_of(pts)`` order
        (Platt-calibrated where a map was fitted, unless ``calibrated=False``)."""
        import torch
        dev = _device(self.cfg.device)
        net = self.net_
        net.eval()
        rows_all = c.rows_of(pts)
        out = {h: np.empty(len(rows_all)) for h in heads}
        where = np.full(len(c.y), -1, dtype=np.int64)
        where[rows_all] = np.arange(len(rows_all))
        empty = {}
        with torch.no_grad():
            for grp in _Batches(c, np.asarray(pts), 64, False, 0):
                x, _, _, rows, valid = self._tensors(c, X, empty, grp, dev)
                main, s3, aux_s, aux_o, a_s, a_o, _, _ = net(x)
                dest = where[rows[valid]]
                vals = {"direct": main[..., 0], "who": main[..., 1], "when": main[..., 2],
                        "s3": s3 - torch.log(-torch.expm1(torch.clamp(s3, max=-1e-6))),
                        "a_susp": a_s, "a_organ": a_o}
                for h in heads:
                    out[h][dest] = vals[h].detach().cpu().numpy()[valid]
        if calibrated:
            for h in heads:
                if h in self.calib_:
                    a, b = self.calib_[h]
                    out[h] = a * out[h] + b
        return out
