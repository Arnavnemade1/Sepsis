"""Causal deep sequence detectors — the first model here that is not row-wise.

Every detector this project has shipped (heuristic, HDC, GBM, the stacked ensemble)
scores a patient-hour from a **flat feature vector**. Temporal context reaches those
models only through hand-authored aggregates — rolling means, slopes, expanding
z-scores — chosen in advance by a human. Phase 7 found that adding more such
aggregates (185 features instead of 94) bought nothing, which is evidence that the
limit is not *how many* summaries you write down but that a fixed summary is being
written down at all.

This module changes the inductive bias instead of the feature count: the model reads
a patient's entire causal prefix as a sequence and learns its own temporal
representation. Three architectures, one trainer, so the comparison isolates the
architecture:

``tcn``
    Dilated causal convolutions (Bai et al. 2018). Dilations 1,2,4,…  give a
    receptive field of ~127 hours with a handful of layers, and causality is
    structural — the left-padding makes a peek at hour ``t+1`` impossible.

``ssm``
    A selective state-space layer in the Mamba-2 / SSD form (Dao & Gu 2024):
    per-head input-dependent decay ``a_t`` with state ``S_t = a_t S_{t-1} + B_t x_tᵀ``
    and readout ``y_t = C_tᵀ S_t``. Because the decay is a function of the input, the
    model can hold a deterioration in memory for days and then flush it when the
    patient stabilises — a data-dependent forgetting that a fixed rolling window
    cannot express. Computed by the chunked scan (parallel inside a chunk, recurrent
    across chunks), which keeps it exact and O(T).

``xfmr``
    A causal transformer: every hour attends to every earlier hour directly, with no
    decay prior at all. The opposite extreme from a fixed window, and the control
    that says whether the SSM's decay structure is doing real work.

All three are trained with the utility-aligned objective from
:mod:`backend.ml.eval.utility` by default, so the sequence models and the boosters
are optimising the same thing and any difference between them is architectural.
Outputs are isotonic-calibrated against the challenge label, matching every other
detector in the project.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from backend.ml.eval.utility import utility_target
from backend.ml.models.calibration import ProbabilityCalibrator
from backend.ml.models.hdc_detector import _grouped_holdout


def _device(prefer: str = "auto"):
    import torch
    if prefer != "auto":
        return torch.device(prefer)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# --------------------------------------------------------------------------- #
# Sequence assembly
# --------------------------------------------------------------------------- #
@dataclass
class PatientSeq:
    rows: np.ndarray        # original row indices, in hour order
    x: np.ndarray           # (T, F) features
    y: np.ndarray           # (T,) challenge label
    z: np.ndarray           # (T,) utility-implied target
    w: np.ndarray           # (T,) utility stake


def build_sequences(X, y, groups, hour_index, z, w, idx) -> List[PatientSeq]:
    """Regroup selected rows into per-patient, hour-ordered sequences.

    Only ``idx`` rows are used, so a fold's sequences never contain a patient hour
    from outside that fold — the grouped-CV guarantee is preserved at the sequence
    level, not just the row level.
    """
    g = groups[idx]
    order = np.lexsort((hour_index[idx], g))
    idx_sorted = idx[order]
    g_sorted = g[order]
    bounds = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1], True])
    out: List[PatientSeq] = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        rows = idx_sorted[a:b]
        out.append(PatientSeq(rows=rows, x=X[rows], y=y[rows], z=z[rows], w=w[rows]))
    return out


def _batches(seqs: List[PatientSeq], batch_size: int, shuffle: bool, rng) -> List[List[PatientSeq]]:
    """Length-bucketed batches: sort by stay length, chunk, then shuffle the chunks.

    Stays run from a few hours to ~336, so uniform batching would pad most sequences
    to many times their real length and spend the majority of the compute on padding.
    """
    order = np.argsort([len(s.y) for s in seqs], kind="stable")
    chunks = [[seqs[i] for i in order[k:k + batch_size]]
              for k in range(0, len(order), batch_size)]
    if shuffle:
        rng.shuffle(chunks)
    return chunks


def _pad(batch: List[PatientSeq], device):
    import torch
    T = max(len(s.y) for s in batch)
    F = batch[0].x.shape[1]
    n = len(batch)
    x = np.zeros((n, T, F), dtype=np.float32)
    z = np.zeros((n, T), dtype=np.float32)
    w = np.zeros((n, T), dtype=np.float32)
    m = np.zeros((n, T), dtype=np.float32)
    for i, s in enumerate(batch):
        t = len(s.y)
        x[i, :t] = s.x
        z[i, :t] = s.z
        w[i, :t] = s.w
        m[i, :t] = 1.0
    tt = lambda a: torch.from_numpy(a).to(device)
    return tt(x), tt(z), tt(w), tt(m)


# --------------------------------------------------------------------------- #
# Architectures
# --------------------------------------------------------------------------- #
def _build_modules():
    """Defined inside a function so importing this module never requires torch."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    class CausalConvBlock(nn.Module):
        def __init__(self, d, dilation, dropout):
            super().__init__()
            self.pad = (3 - 1) * dilation           # left-pad only => strictly causal
            self.conv1 = nn.Conv1d(d, 2 * d, 3, dilation=dilation)
            self.conv2 = nn.Conv1d(d, d, 1)
            self.norm = nn.LayerNorm(d)
            self.drop = nn.Dropout(dropout)

        def forward(self, h):                        # h: (B, T, D)
            r = h
            h = self.norm(h).transpose(1, 2)
            h = F.pad(h, (self.pad, 0))
            a, b = self.conv1(h).chunk(2, dim=1)
            h = self.conv2(a * torch.sigmoid(b))     # gated activation unit
            return r + self.drop(h.transpose(1, 2))

    class SelectiveSSM(nn.Module):
        """Mamba-2 style SSD layer with input-dependent scalar decay per head."""

        def __init__(self, d, n_state=16, n_heads=8, chunk=16, dropout=0.1):
            super().__init__()
            assert d % n_heads == 0
            self.h, self.p, self.n, self.c = n_heads, d // n_heads, n_state, chunk
            self.norm = nn.LayerNorm(d)
            self.in_proj = nn.Linear(d, 2 * d)                    # value + gate
            self.bc_proj = nn.Linear(d, 2 * n_heads * n_state)    # selective B, C
            self.a_proj = nn.Linear(d, n_heads)                   # selective decay
            self.dt_bias = nn.Parameter(torch.zeros(n_heads))
            self.out = nn.Linear(d, d)
            self.drop = nn.Dropout(dropout)

        def forward(self, h, mask):
            B, T, D = h.shape
            C = self.c
            pad = (-T) % C
            r = h
            h = self.norm(h)
            v, gate = self.in_proj(h).chunk(2, dim=-1)
            bc = self.bc_proj(h).view(B, T, 2, self.h, self.n)
            Bm, Cm = bc[:, :, 0], bc[:, :, 1]
            # decay in (0,1): softplus(dt) is a positive timestep, a = exp(-dt).
            dt = F.softplus(self.a_proj(h) + self.dt_bias)        # (B,T,H)
            log_a = -dt * mask.unsqueeze(-1)                      # padding => no decay
            x = v.view(B, T, self.h, self.p) * mask[..., None, None]

            if pad:
                log_a = F.pad(log_a.transpose(1, 2), (0, pad)).transpose(1, 2)
                x = F.pad(x.permute(0, 2, 3, 1), (0, pad)).permute(0, 3, 1, 2)
                Bm = F.pad(Bm.permute(0, 2, 3, 1), (0, pad)).permute(0, 3, 1, 2)
                Cm = F.pad(Cm.permute(0, 2, 3, 1), (0, pad)).permute(0, 3, 1, 2)
            K = (T + pad) // C
            rs = lambda t: t.view(B, K, C, *t.shape[2:])
            log_a, x, Bm, Cm = rs(log_a), rs(x), rs(Bm), rs(Cm)

            La = torch.cumsum(log_a, dim=2)                        # (B,K,C,H)
            # --- inside a chunk: exact semiseparable mask, no exp of a positive --- #
            seg = La.unsqueeze(3) - La.unsqueeze(2)                # (B,K,C,C,H)
            tri = torch.ones(C, C, device=h.device, dtype=torch.bool).tril()
            decay = torch.where(tri[None, None, :, :, None], seg.exp(), torch.zeros_like(seg))
            cb = torch.einsum("bkchn,bkdhn->bkcdh", Cm, Bm)
            y = torch.einsum("bkcdh,bkdhp->bkchp", cb * decay, x)
            # --- across chunks: a short recurrence over K (T/16) steps ----------- #
            to_end = (La[:, :, -1:, :] - La).exp()                 # (B,K,C,H)
            s_chunk = torch.einsum("bkchn,bkchp,bkch->bkhnp", Bm, x, to_end)
            a_chunk = La[:, :, -1, :].exp()                        # (B,K,H)
            S = torch.zeros(B, self.h, self.n, self.p, device=h.device, dtype=h.dtype)
            prev = []
            for k in range(K):
                prev.append(S)
                S = S * a_chunk[:, k][..., None, None] + s_chunk[:, k]
            S_prev = torch.stack(prev, dim=1)                      # (B,K,H,N,P)
            y = y + torch.einsum("bkchn,bkhnp,bkch->bkchp", Cm, S_prev, La.exp())

            y = y.reshape(B, K * C, D)[:, :T]
            return r + self.drop(self.out(y * torch.sigmoid(gate)))

    class CausalAttention(nn.Module):
        def __init__(self, d, n_heads=8, dropout=0.1):
            super().__init__()
            self.norm = nn.LayerNorm(d)
            self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
            self.ff = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 2 * d),
                                    nn.GELU(), nn.Linear(2 * d, d))
            self.drop = nn.Dropout(dropout)

        def forward(self, h, mask):
            T = h.shape[1]
            causal = torch.ones(T, T, device=h.device, dtype=torch.bool).triu(1)
            q = self.norm(h)
            a, _ = self.attn(q, q, q, attn_mask=causal,
                             key_padding_mask=(mask < 0.5), need_weights=False)
            h = h + self.drop(torch.nan_to_num(a))
            return h + self.drop(self.ff(h))

    class SeqNet(nn.Module):
        def __init__(self, n_features, arch="tcn", d_model=128, n_layers=4,
                     dropout=0.1, n_state=16, n_heads=8, chunk=16):
            super().__init__()
            self.arch = arch
            self.inp = nn.Sequential(nn.Linear(n_features, d_model), nn.GELU())
            if arch == "tcn":
                self.blocks = nn.ModuleList(
                    [CausalConvBlock(d_model, 2 ** i, dropout) for i in range(n_layers)])
            elif arch == "ssm":
                self.blocks = nn.ModuleList(
                    [SelectiveSSM(d_model, n_state, n_heads, chunk, dropout)
                     for _ in range(n_layers)])
            elif arch == "xfmr":
                self.blocks = nn.ModuleList(
                    [CausalAttention(d_model, n_heads, dropout) for _ in range(n_layers)])
            else:
                raise ValueError(f"unknown arch {arch!r}")
            self.head = nn.Sequential(nn.LayerNorm(d_model), nn.Linear(d_model, 1))

        def forward(self, x, mask):
            h = self.inp(x)
            if self.arch == "xfmr":                      # position via a simple ramp
                T = h.shape[1]
                pos = torch.arange(T, device=h.device, dtype=h.dtype)[None, :, None]
                h = h + 0.02 * pos
            for blk in self.blocks:
                h = blk(h, mask) if self.arch != "tcn" else blk(h)
                h = h * mask.unsqueeze(-1)
            return self.head(h).squeeze(-1)

    return SeqNet


# --------------------------------------------------------------------------- #
@dataclass
class DeepSequenceDetector:
    """Trains a causal sequence net and serves calibrated per-hour probabilities."""

    arch: str = "tcn"
    objective: str = "utility"          # "utility" | "balanced"
    d_model: int = 128
    n_layers: int = 4
    dropout: float = 0.1
    n_state: int = 16
    n_heads: int = 8
    chunk: int = 16
    epochs: int = 24
    batch_size: int = 64
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 5
    val_fraction: float = 0.15
    seed: int = 42
    device: str = "auto"
    verbose: bool = False

    mean_: Optional[np.ndarray] = None
    std_: Optional[np.ndarray] = None
    net_: Optional[object] = None
    calibrator_: Optional[ProbabilityCalibrator] = None
    history_: List[Dict[str, float]] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def _targets(self, y, groups, hour_index, onset_hour):
        if self.objective == "balanced":
            y = np.asarray(y).astype(int)
            pos, neg = max(int((y == 1).sum()), 1), max(int((y == 0).sum()), 1)
            return y.astype(np.float32), np.where(y == 1, neg / pos, 1.0).astype(np.float32)
        z, w = utility_target(y, groups, hour_index, onset_hour)
        return z.astype(np.float32), np.maximum(w, 1e-4).astype(np.float32)

    def _standardise(self, X):
        Xs = (X - self.mean_) / self.std_
        return np.clip(np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0), -10, 10).astype(np.float32)

    # ------------------------------------------------------------------ #
    def fit(self, X, y, groups, hour_index, onset_hour) -> "DeepSequenceDetector":
        import torch
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        dev = _device(self.device)
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y).astype(int)

        z, w = self._targets(y, groups, hour_index, onset_hour)
        # Patient-grouped: an inner slice for early stopping, another for calibration.
        fit_idx, cal_idx = _grouped_holdout(y, groups, 0.15, self.seed)
        tr_idx, val_idx = _grouped_holdout(y[fit_idx], groups[fit_idx],
                                           self.val_fraction, self.seed + 1)
        tr_idx, val_idx = fit_idx[tr_idx], fit_idx[val_idx]

        self.mean_ = X[tr_idx].mean(axis=0)
        self.std_ = X[tr_idx].std(axis=0)
        self.std_[self.std_ < 1e-6] = 1.0
        Xs = self._standardise(X)

        mk = lambda idx: build_sequences(Xs, y, groups, hour_index, z, w, idx)
        tr_seqs, val_seqs = mk(tr_idx), mk(val_idx)

        SeqNet = _build_modules()
        net = SeqNet(X.shape[1], arch=self.arch, d_model=self.d_model,
                     n_layers=self.n_layers, dropout=self.dropout,
                     n_state=self.n_state, n_heads=self.n_heads, chunk=self.chunk).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        n_steps = max(1, self.epochs * math.ceil(len(tr_seqs) / self.batch_size))
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=self.lr, total_steps=n_steps,
                                                    pct_start=0.25)
        bce = torch.nn.BCEWithLogitsLoss(reduction="none")

        best_auroc, best_state, stale = -np.inf, None, 0
        for ep in range(self.epochs):
            net.train()
            tot, seen = 0.0, 0.0
            for batch in _batches(tr_seqs, self.batch_size, True, rng):
                xb, zb, wb, mb = _pad(batch, dev)
                logits = net(xb, mb)
                loss_el = bce(logits, zb) * wb * mb
                loss = loss_el.sum() / (wb * mb).sum().clamp_min(1.0)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
                sched.step()
                tot += float(loss.detach()) * len(batch)
                seen += len(batch)
            from backend.ml.eval.metrics import auroc as _auroc
            va = self._score_seqs(net, val_seqs, dev)
            a = _auroc(va["y"], va["p"])
            self.history_.append({"epoch": ep, "loss": tot / max(seen, 1), "val_auroc": a})
            if self.verbose:
                print(f"      ep{ep:02d} loss {tot/max(seen,1):.4f}  val AUROC {a:.4f}", flush=True)
            if a > best_auroc + 1e-4:
                best_auroc, stale = a, 0
                best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            else:
                stale += 1
                if stale >= self.patience:
                    break
        if best_state is not None:
            net.load_state_dict(best_state)
        net.eval()
        self.net_ = net

        raw_cal = self.predict_raw(X, groups, hour_index, cal_idx, standardised=False)
        self.calibrator_ = ProbabilityCalibrator().fit(raw_cal, y[cal_idx])
        return self

    # ------------------------------------------------------------------ #
    def _score_seqs(self, net, seqs, dev) -> Dict[str, np.ndarray]:
        import torch
        ps, ys = [], []
        rng = np.random.default_rng(0)
        with torch.no_grad():
            for batch in _batches(seqs, self.batch_size, False, rng):
                xb, zb, wb, mb = _pad(batch, dev)
                out = torch.sigmoid(net(xb, mb)).cpu().numpy()
                for i, s in enumerate(batch):
                    ps.append(out[i, :len(s.y)])
                    ys.append(s.y)
        return {"p": np.concatenate(ps), "y": np.concatenate(ys)}

    def predict_raw(self, X, groups, hour_index, idx, standardised: bool = True) -> np.ndarray:
        import torch
        dev = _device(self.device)
        Xs = X if standardised else self._standardise(np.asarray(X, dtype=np.float32))
        dummy = np.zeros(len(Xs), dtype=np.float32)
        seqs = build_sequences(Xs, dummy.astype(int), groups, hour_index, dummy, dummy, idx)
        out = np.zeros(len(idx), dtype=float)
        pos = {int(r): i for i, r in enumerate(idx)}
        rng = np.random.default_rng(0)
        with torch.no_grad():
            for batch in _batches(seqs, self.batch_size, False, rng):
                xb, _, _, mb = _pad(batch, dev)
                p = torch.sigmoid(self.net_(xb, mb)).cpu().numpy()
                for i, s in enumerate(batch):
                    for j, r in enumerate(s.rows):
                        out[pos[int(r)]] = p[i, j]
        return out

    def predict_proba(self, X, groups, hour_index, idx=None) -> np.ndarray:
        idx = np.arange(len(X)) if idx is None else np.asarray(idx)
        raw = self.predict_raw(np.asarray(X, dtype=np.float32), groups, hour_index,
                               idx, standardised=False)
        return self.calibrator_.transform(raw) if self.calibrator_ is not None else raw

    # ------------------------------------------------------------------ #
    def save(self, path: str) -> None:
        import joblib
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(self, path)
