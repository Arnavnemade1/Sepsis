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
def input_matrix(c: Cohort) -> Tuple[np.ndarray, List[str]]:
    """Compact per-hour input: for each of the 34 channels its last value, whether it
    was measured this hour, log-hours since measured and whether it ever was; plus
    context and a few composites. Values are left raw here; ``Standardiser`` fits
    robust scales on training rows only."""
    ctxs = ["Age", "Gender", "unit_sicu", "HospAdmTime", "ICULOS", "hour",
            "sofa_proxy", "sofa_rise24", "sirs", "qsofa_proxy", "shock_index",
            "proc__labs6", "proc__new24", "proc__draws24"]
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

        def forward(self, x):
            h, _ = self.gru(self.inp(x))
            a_s, a_o = self.susp(h), self.org(h)
            s3_logp = Fn.logsigmoid(a_s) + Fn.logsigmoid(a_o)        # log P(AND)
            aux_s = self.susp_aux(torch.cat([h, a_s], -1))
            aux_o = self.org_aux(torch.cat([h, a_o], -1))
            return (self.main(h), s3_logp.squeeze(-1), aux_s, aux_o, a_s.squeeze(-1),
                    a_o.squeeze(-1), self.wm(h))

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
                 pts: np.ndarray, dev):
        import torch
        starts, lens = c.raw.starts[pts], c.raw.lens[pts]
        B, Tm = len(pts), int(lens.max())
        rows = np.full((B, Tm), -1, dtype=np.int64)
        for b, (a, n) in enumerate(zip(starts, lens)):
            rows[b, :n] = np.arange(a, a + n)
        valid = rows >= 0
        flat = rows[valid]
        x = np.zeros((B, Tm, X.shape[1]), dtype=np.float32)
        x[valid] = self.std_(X[flat])
        ys, ms = {}, {}
        W = getattr(self, "_world", None)
        if W is not None and len(T):
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

    def _loss(self, out, ys, ms, pw):
        import torch
        import torch.nn.functional as Fn
        main, s3, aux_s, aux_o, _, _, wm = out
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
            n_ch = ys["wm_v"].shape[-1]
            pv, pe = wm[..., :n_ch], wm[..., n_ch:]
            hub = Fn.smooth_l1_loss(pv, ys["wm_v"], reduction="none")
            lv = (hub * ys["wm_vm"]).sum() / ys["wm_vm"].sum().clamp(min=1.0)
            be = Fn.binary_cross_entropy_with_logits(pe, ys["wm_e"], reduction="none")
            m = ys["wm_em"].expand_as(be)
            le = (be * m).sum() / m.sum().clamp(min=1.0)
            L = L + cfg.w_wm * (lv + le)
        return L

    def fit(self, c: Cohort, X: np.ndarray, T: Dict[str, Tuple[np.ndarray, np.ndarray]],
            train_pts: np.ndarray, verbose: bool = True,
            world: Optional[Dict[str, np.ndarray]] = None) -> "GenesisNetwork":
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
        self.std_ = Standardiser.fit(X, rows_fit, seed=cfg.seed)
        net = _build(X.shape[1], cfg, len(SUSPICION_AUX), len(ORGAN_AUX)).to(dev)
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
                main, s3, aux_s, aux_o, a_s, a_o, _ = net(x)
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
