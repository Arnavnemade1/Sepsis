"""The Sepsis-3 Process Twin — a temporal model whose gate factors are OBSERVED.

On CinC the Genesis Network's AND gate (``sigmoid(a_susp) * sigmoid(a_organ)``) was a
structural prior: neither half of Sepsis-3 was recorded, so its two factors were latent.
MIMIC-IV records both halves for every patient — when clinicians began to suspect
infection (culture + antibiotic pairs) and how every organ system's SOFA score moved —
so here each factor is trained on its own ground truth:

    a_susp   "suspicion of infection has begun, or begins within 6 h"   (every stay, septic
             or not: suspicion without organ failure is common and is exactly what the
             factor must learn to separate)
    a_organ  "SOFA has risen >= 2 above a MEASURED baseline in [t - 48 h, t + 6 h]"
             (the ``measured`` rule of ``sepsis3`` — no rise from data arrival)
    s3       sigmoid(a_susp) * sigmoid(a_organ), trained on the challenge label

plus dense process supervision on every hour: per-organ rises within 12 h (6 heads),
suspicion within 24 h, the GENESIS who / when / direct heads, and the world model (next
value of every input channel). Inputs are tier-P by default: the suspicion record is a
TARGET here, never an input — the model learns to anticipate clinicians' suspicion from
physiology instead of reading their orders (the Epic Sepsis Model trap).

Causal by construction (a unidirectional GRU over hours <= t); ``verify_mimic`` rewrites
the future and asserts no past output moves.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend.ml.genesis.learners import platt
from backend.ml.genesis.network import Standardiser, _bce, _bce_logp, _device
from backend.ml.mimic import extract as E
from backend.ml.mimic import sepsis3 as S3

ORGANS = ["resp", "coag", "liver", "cardio", "cns", "renal"]
MAIN = ["direct", "who", "when"]
AUX = ["susp6", "susp24", "organ"] + [f"rise12_{o}" for o in ORGANS]


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
    # measured rise at each hour vs the running minimum of the preceding 48 h
    base = np.full_like(parts, np.nan)
    idx = np.arange(N)
    for lag in range(49):
        src = idx - lag
        ok = src >= rs
        v = np.full_like(parts, np.nan)
        v[ok] = parts[src[ok]]
        base = np.fmin(base, v)
    rise = np.nansum(np.where(np.isfinite(parts) & np.isfinite(base), parts - base, 0.0), axis=1)
    hit = rise >= 2
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


def twin_inputs(led, tier: str) -> Tuple[np.ndarray, List[str]]:
    """Per-hour inputs from the ledger's feature matrix, restricted to ``tier``: for every
    allowed channel its last value, measured-now, log-hours since measured, ever measured;
    plus the static columns."""
    allowed = [c for c in E.DYNAMIC if E.K.TIER_ORDER[led.store.tiers[c]] <= E.K.TIER_ORDER[tier]]
    j = {n: i for i, n in enumerate(led.names)}
    cols, names = [], []
    for ch in allowed:
        v = led.F[:, j[ch]]
        age_name = f"{ch}__age"
        if age_name in j:
            age = led.F[:, j[age_name]]
            cols += [v, (age == 0).astype(np.float32), np.log1p(np.nan_to_num(age, nan=96.0)),
                     np.isfinite(v).astype(np.float32)]
            names += [ch, f"{ch}__now", f"{ch}__logage", f"{ch}__ever"]
        else:
            cols.append(v)
            names.append(ch)
    for st in E.STATIC:
        v = led.F[:, j[st]].astype(np.float32)
        if st in ("ICULOS", "HospAdmTime"):
            v = np.sign(v) * np.log1p(np.abs(v))
        cols.append(v)
        names.append(st)
    return np.stack(cols, axis=1).astype(np.float32), names


# --------------------------------------------------------------------------- #
@dataclass
class TwinConfig:
    d_model: int = 128
    layers: int = 2
    dropout: float = 0.1
    lr: float = 1e-3
    weight_decay: float = 1e-5
    batch: int = 32
    epochs: int = 25
    patience: int = 4
    pos_weight: float = 3.0
    w_main: Tuple[float, float, float] = (1.0, 0.5, 0.5)
    w_s3: float = 0.5
    w_factor: float = 0.5          # the gate factors' own ground truth
    w_aux: float = 0.2
    device: str = "cpu"
    seed: int = 0


def _build(d_in: int, cfg: TwinConfig):
    import torch
    import torch.nn as nn
    import torch.nn.functional as Fn

    class ProcessTwinNet(nn.Module):
        def __init__(self):
            super().__init__()
            d = cfg.d_model
            self.inp = nn.Sequential(nn.Linear(d_in, d), nn.GELU(), nn.Dropout(cfg.dropout))
            self.gru = nn.GRU(d, d, num_layers=cfg.layers, batch_first=True,
                              dropout=cfg.dropout if cfg.layers > 1 else 0.0)
            self.main = nn.Linear(d, len(MAIN))
            self.susp = nn.Linear(d, 1)
            self.organ = nn.Linear(d, 1)
            self.aux = nn.Linear(d, len(AUX) - 2)       # susp24 + the 6 per-organ rises
            self.wm = nn.Linear(d, d_in)

        def forward(self, x):
            h, _ = self.gru(self.inp(x))
            a_s, a_o = self.susp(h).squeeze(-1), self.organ(h).squeeze(-1)
            s3 = Fn.logsigmoid(a_s) + Fn.logsigmoid(a_o)
            return self.main(h), a_s, a_o, s3, self.aux(h), self.wm(h)

    return ProcessTwinNet()


@dataclass
class ProcessTwin:
    cfg: TwinConfig = field(default_factory=TwinConfig)
    std_: Optional[Standardiser] = None
    net_: object = None
    calib_: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    best_epoch_: int = 0

    def _batch(self, led, X, T, stays, dev):
        import torch
        s = led.store
        starts, lens = s.starts[stays], s.lens[stays]
        B, Tm = len(stays), int(lens.max())
        rows = np.full((B, Tm), -1, dtype=np.int64)
        for b, (a, n) in enumerate(zip(starts, lens)):
            rows[b, :n] = np.arange(a, a + n)
        valid = rows >= 0
        flat = rows[valid]
        x = np.zeros((B, Tm, X.shape[1]), dtype=np.float32)
        x[valid] = self.std_(X[flat])
        ys = {}
        for k, (tgt, msk) in T.items():
            yy = np.zeros((B, Tm), np.float32)
            mm = np.zeros((B, Tm), np.float32)
            yy[valid], mm[valid] = tgt[flat], msk[flat]
            ys[k] = (torch.from_numpy(yy).to(dev), torch.from_numpy(mm).to(dev))
        if T:
            nxt = np.zeros((B, Tm, X.shape[1]), np.float32)
            nm = np.zeros((B, Tm, 1), np.float32)
            nxt[:, :-1] = x[:, 1:]
            nm[:, :-1, 0] = (valid[:, 1:] & valid[:, :-1])
            ys["_wm"] = (torch.from_numpy(nxt).to(dev), torch.from_numpy(nm).to(dev))
        return torch.from_numpy(x).to(dev), ys, rows, valid

    def _loss(self, out, ys, pw):
        import torch.nn.functional as Fn
        main, a_s, a_o, s3, aux, wm = out
        c = self.cfg
        L = 0.0
        for i, (k, w) in enumerate(zip(MAIN, c.w_main)):
            L = L + w * _bce(main[..., i], *ys[k], **({"pos_weight": pw} if k == "direct" else {}))
        L = L + c.w_s3 * _bce_logp(s3, *ys["direct"])
        L = L + c.w_factor * (_bce(a_s, *ys["susp6"]) + _bce(a_o, *ys["organ"]))
        for i, k in enumerate(AUX[1:2] + AUX[3:]):
            L = L + c.w_aux * _bce(aux[..., i], *ys[k])
        t, m = ys["_wm"]
        hub = Fn.smooth_l1_loss(wm, t, reduction="none")
        L = L + c.w_aux * (hub * m).sum() / (m.sum() * wm.shape[-1]).clamp(min=1.0)
        return L

    def fit(self, led, X: np.ndarray, T, train_stays: np.ndarray, verbose: bool = False) -> "ProcessTwin":
        import torch
        cfg = self.cfg
        torch.manual_seed(cfg.seed)
        dev = _device(cfg.device)
        rng = np.random.default_rng(cfg.seed)
        s = led.store
        # early-stopping slice: 10% of the training SUBJECTS
        subj = s.subject[train_stays]
        us = np.unique(subj)
        val_subj = rng.choice(us, max(1, int(0.1 * len(us))), replace=False)
        is_val = np.isin(subj, val_subj)
        fit_st, val_st = train_stays[~is_val], train_stays[is_val]
        fit_rows = np.concatenate([np.arange(s.starts[k], s.starts[k] + s.lens[k]) for k in fit_st])
        self.std_ = Standardiser.fit(X, fit_rows, seed=cfg.seed)
        net = _build(X.shape[1], cfg).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=1)
        pw = torch.tensor(cfg.pos_weight, device=dev)
        order = fit_st[np.argsort(s.lens[fit_st], kind="mergesort")]
        groups = [order[i:i + cfg.batch] for i in range(0, len(order), cfg.batch)]
        best, best_state, bad = -1.0, None, 0
        for ep in range(cfg.epochs):
            t0 = time.time()
            net.train()
            for gi in rng.permutation(len(groups)):
                x, ys, _, _ = self._batch(led, X, T, groups[gi], dev)
                loss = self._loss(net(x), ys, pw)
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            self.net_ = net
            pv = self.predict(led, X, val_st, ("direct",), calibrated=False)["direct"]
            vr = self._rows(led, val_st)
            ok = T["direct"][1][vr] > 0
            from backend.ml.genesis.harness import auroc
            yv = T["direct"][0][vr][ok]
            a = auroc(yv, pv[ok]) if 0 < yv.sum() < len(yv) else 0.5
            if verbose:
                print(f"      twin epoch {ep}: val AUROC {a:.4f} [{time.time()-t0:.0f}s]", flush=True)
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
        heads = ("direct", "who", "when", "s3", "susp6", "organ")
        pv = self.predict(led, X, val_st, heads, calibrated=False)
        vr = self._rows(led, val_st)
        tmap = {"direct": "direct", "who": "who", "when": "when", "s3": "direct",
                "susp6": "susp6", "organ": "organ"}
        for h in heads:
            t, m = T[tmap[h]]
            k = m[vr] > 0
            if k.sum() > 10 and 0 < t[vr][k].sum() < k.sum():
                # design weights: under a subsampled cohort the map must target the FULL
                # cohort's scale, the same one the tree heads it is multiplied with use
                self.calib_[h] = platt(pv[h][k], t[vr][k], led.design_w[vr][k])
        return self

    @staticmethod
    def _rows(led, stays: np.ndarray) -> np.ndarray:
        s = led.store
        return np.concatenate([np.arange(s.starts[k], s.starts[k] + s.lens[k]) for k in stays])

    def predict(self, led, X: np.ndarray, stays: np.ndarray,
                heads: Sequence[str] = ("direct", "who", "when", "s3", "susp6", "organ"),
                calibrated: bool = True) -> Dict[str, np.ndarray]:
        """Margins per head for every row of ``stays`` (in ``_rows(stays)`` order)."""
        import torch
        dev = _device(self.cfg.device)
        net = self.net_
        net.eval()
        rows_all = self._rows(led, stays)
        where = np.full(led.store.n_rows, -1, dtype=np.int64)
        where[rows_all] = np.arange(len(rows_all))
        out = {h: np.empty(len(rows_all)) for h in heads}
        with torch.no_grad():
            for i in range(0, len(stays), 64):
                grp = stays[i:i + 64]
                x, _, rows, valid = self._batch(led, X, {}, grp, dev)
                main, a_s, a_o, s3, aux, _ = net(x)
                vals = {"direct": main[..., 0], "who": main[..., 1], "when": main[..., 2],
                        "s3": s3 - torch.log(-torch.expm1(torch.clamp(s3, max=-1e-6))),
                        "susp6": a_s, "organ": a_o}
                dest = where[rows[valid]]
                for h in heads:
                    out[h][dest] = vals[h].cpu().numpy()[valid]
        if calibrated:
            for h in heads:
                if h in self.calib_:
                    a, b = self.calib_[h]
                    out[h] = a * out[h] + b
        return out
