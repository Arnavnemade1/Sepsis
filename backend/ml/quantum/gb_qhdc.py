"""Gradient-Boosted Quantum Hyperdimensional Computing (GB-QHDC).

A unified classical/quantum detector: a classical gradient-boosting envelope whose
weak learner is a Quantum Hyperdimensional circuit (Cumbo et al. 2026 primitives,
simulated exactly on statevectors — see qhdc.py). Instead of a decision tree fixing
the ensemble's residuals, each round fits a QHDC *matched template*:

    round m:
      1. classical driver computes pseudo-residuals  r_i = y_i − σ(F_{m-1}(x_i))
      2. project x_i → N angles → encode as an N-qubit hypervector |ψ(x_i; θ)>
      3. build the class template by RESIDUAL-WEIGHTED LCU bundling
             |c_m> = normalise( Σ_i r_i |ψ(x_i)> )                      (quantum-native,
         which classical sign-bundling cannot represent — see verify_qhdc §5)
      4. weak-learner score  h_m(x_i) = Re<ψ(x_i)|c_m>                  (Hadamard test)
      5. SPSA tunes the encoder angles θ so h_m aligns with r (noise-robust, dodges
         barren plateaus); a 1-D line search sets the step γ_m
      6. F_m = F_{m-1} + ν · γ_m · h_m

Honesty: this is an EXACT statevector simulation (amplitudes), tractable for N ≤ ~12
qubits (D = 2^N virtual dimensions). It is a research framework, benchmarked head-to-
head against the GBM and classical HDC in scripts/qhdc_ablation.py — no accuracy claim
is made here that the ablation does not earn. See backend/ml/README.md.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

Array = np.ndarray


# --------------------------------------------------------------------------- #
# Batched statevector ops (process B rows at once — the encoder is the hot loop).
# --------------------------------------------------------------------------- #
def _apply_1q_batch(states: Array, gates: Array, q: int, n: int) -> Array:
    """Apply a 1-qubit gate to qubit q across a batch. states (B, 2^n); gates either
    (2,2) shared or (B,2,2) per-row. Returns (B, 2^n)."""
    B = states.shape[0]
    L, R = 1 << q, 1 << (n - q - 1)
    st = states.reshape(B, L, 2, R)
    if gates.ndim == 2:
        st = np.einsum("ab,ilbr->ilar", gates, st, optimize=True)
    else:
        st = np.einsum("iab,ilbr->ilar", gates, st, optimize=True)
    return st.reshape(B, -1)


def _cnot_batch(states: Array, ctrl: int, tgt: int, n: int) -> Array:
    B = states.shape[0]
    st = states.reshape([B] + [2] * n)
    idx = [slice(None)] * (n + 1)
    idx[ctrl + 1] = 1
    block = st[tuple(idx)]                       # ctrl axis removed; batch axis at 0
    axis = (tgt + 1) if tgt < ctrl else tgt      # +1 for batch, −1 for removed ctrl
    st[tuple(idx)] = np.flip(block, axis=axis)
    return st.reshape(B, -1)


def _ry_batch(angles: Array) -> Array:
    c, s = np.cos(angles / 2.0), np.sin(angles / 2.0)
    g = np.zeros((len(angles), 2, 2), dtype=np.complex128)
    g[:, 0, 0] = c; g[:, 0, 1] = -s; g[:, 1, 0] = s; g[:, 1, 1] = c
    return g


def _rz(theta: float) -> Array:
    return np.array([[np.exp(-1j * theta / 2), 0], [0, np.exp(1j * theta / 2)]], np.complex128)


def _ry(theta: float) -> Array:
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], np.complex128)


def encode_batch(Xp: Array, n_qubits: int, thetas: Array, layers: int) -> Array:
    """Batched variational encoder with DATA RE-UPLOADING. Xp (B, n_qubits*layers)
    in [0,1]; returns (B, 2^n) unit statevectors. Each layer injects a FRESH slice of
    ``n_qubits`` projected features (re-uploading), so N qubits absorb N·layers
    features — the standard way a small register encodes many inputs. Data RY(π·f) is
    interleaved with trainable RZ·RY and a CNOT ring (entanglement)."""
    B = Xp.shape[0]
    st = np.zeros((B, 1 << n_qubits), dtype=np.complex128)
    st[:, 0] = 1.0
    th = thetas.reshape(layers, n_qubits, 2)
    for L in range(layers):
        base = L * n_qubits
        for q in range(n_qubits):
            col = base + q
            fq = Xp[:, col] if col < Xp.shape[1] else np.zeros(B)
            st = _apply_1q_batch(st, _ry_batch(np.pi * fq), q, n_qubits)
            st = _apply_1q_batch(st, _rz(th[L, q, 0]), q, n_qubits)
            st = _apply_1q_batch(st, _ry(th[L, q, 1]), q, n_qubits)
        for q in range(n_qubits):
            st = _cnot_batch(st, q, (q + 1) % n_qubits, n_qubits)
    st /= np.linalg.norm(st, axis=1, keepdims=True) + 1e-12
    return st


# --------------------------------------------------------------------------- #
@dataclass
class GBQHDC:
    n_qubits: int = 8
    n_rounds: int = 15
    learning_rate: float = 0.3
    layers: int = 1
    spsa_steps: int = 8
    spsa_a: float = 0.25
    spsa_c: float = 0.15
    subsample: int = 4000          # rows used to build each round's template (bounded)
    proj_mode: str = "supervised"  # "supervised" (top |corr(feature,y)|) or "random"
    seed: int = 42

    # learned state
    proj_: Optional[Array] = None          # (n_features, n_qubits) random projection
    lo_: Optional[Array] = None
    hi_: Optional[Array] = None
    quantiles_: Optional[Array] = None     # (Q, k) per-angle CDF edges for uniform spread
    F0_: float = 0.0
    thetas_: List[Array] = field(default_factory=list)
    templates_: List[Array] = field(default_factory=list)
    gammas_: List[float] = field(default_factory=list)

    # ------------------------------------------------------------------ #
    def _project(self, X: Array) -> Array:
        Z = X @ self.proj_
        if self.quantiles_ is not None:
            # Rank/quantile transform → uniform [0,1] per angle, so encoded states
            # SPREAD out (decorrelated) instead of clustering. Fixes the near-constant
            # quantum kernel (concentrated states) that otherwise stalls learning.
            Q = self.quantiles_.shape[0]
            out = np.empty_like(Z)
            for c in range(Z.shape[1]):
                out[:, c] = np.interp(Z[:, c], self.quantiles_[:, c],
                                      np.linspace(0.0, 1.0, Q))
            return np.clip(out, 0.0, 1.0)
        Z = (Z - self.lo_) / (self.hi_ - self.lo_ + 1e-9)
        return np.clip(Z, 0.0, 1.0)

    def _template(self, states: Array, resid: Array) -> Array:
        """Residual-weighted LCU bundle: |c> = normalise(Σ r_i |ψ_i>)."""
        c = (resid[:, None] * states).sum(axis=0)
        n = np.linalg.norm(c)
        return c / n if n > 0 else c

    def _scores(self, states: Array, template: Array) -> Array:
        """Hadamard-test overlaps Re<ψ_i|c> for a batch."""
        return np.real(states @ np.conj(template))

    def _round_objective(self, Xp: Array, resid: Array, theta: Array) -> Tuple[float, Array, Array]:
        st = encode_batch(Xp, self.n_qubits, theta, self.layers)
        c = self._template(st, resid)
        s = self._scores(st, c)
        # objective: maximise correlation of scores with residuals
        obj = float(np.corrcoef(s, resid)[0, 1]) if s.std() > 1e-9 else 0.0
        return (obj if np.isfinite(obj) else 0.0), c, s

    def _spsa(self, Xp: Array, resid: Array, theta0: Array) -> Array:
        """Maximise round objective over encoder angles via SPSA (2 evals/step)."""
        rng = np.random.default_rng(self.seed + len(self.thetas_))
        theta = theta0.copy()
        for t in range(self.spsa_steps):
            ak = self.spsa_a / (t + 1 + 10) ** 0.602
            ck = self.spsa_c / (t + 1) ** 0.101
            delta = rng.choice([-1.0, 1.0], size=theta.shape)
            o_plus = self._round_objective(Xp, resid, theta + ck * delta)[0]
            o_minus = self._round_objective(Xp, resid, theta - ck * delta)[0]
            ghat = (o_plus - o_minus) / (2.0 * ck) * delta       # ascent
            theta = theta + ak * ghat
        return theta

    # ------------------------------------------------------------------ #
    def fit(self, X: Array, y: Array, groups: Optional[Array] = None) -> "GBQHDC":
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        rng = np.random.default_rng(self.seed)
        nf = X.shape[1]
        k = self.n_qubits * self.layers
        if self.proj_mode == "supervised":
            # Leakage-free (TRAIN-only) supervised projection: select the k features
            # most correlated with the label, each its own angle. Feeds the quantum
            # encoder real discriminative signal (random projection washed it out).
            Xc = X - X.mean(0)
            yc = y - y.mean()
            denom = (np.linalg.norm(Xc, axis=0) * (np.linalg.norm(yc) + 1e-9)) + 1e-9
            corr = np.abs((Xc * yc[:, None]).sum(0) / denom)
            top = np.argsort(-corr)[:k]
            self.proj_ = np.zeros((nf, k))
            self.proj_[top, np.arange(k)] = 1.0
        else:
            self.proj_ = rng.standard_normal((nf, k)) / np.sqrt(nf)
        Zraw = X @ self.proj_
        self.lo_ = np.quantile(Zraw, 0.02, axis=0)
        self.hi_ = np.quantile(Zraw, 0.98, axis=0)
        # per-angle empirical CDF edges (TRAIN only) for the uniform-spread transform
        Q = 64
        self.quantiles_ = np.quantile(Zraw, np.linspace(0.0, 1.0, Q), axis=0)
        Xp_all = self._project(X)

        p = float(np.clip(y.mean(), 1e-4, 1 - 1e-4))
        self.F0_ = np.log(p / (1 - p))
        F = np.full(len(y), self.F0_)

        theta_dim = self.layers * self.n_qubits * 2
        theta = np.zeros(theta_dim)
        for m in range(self.n_rounds):
            resid = y - _sigmoid(F)                       # log-loss pseudo-residual
            # bounded, class-balanced subsample to build this round's template
            idx = _balanced_subsample(y, self.subsample, rng)
            Xp, r_sub = Xp_all[idx], resid[idx]
            theta = self._spsa(Xp, r_sub, theta) if self.spsa_steps > 0 else theta
            st = encode_batch(Xp, self.n_qubits, theta, self.layers)
            template = self._template(st, r_sub)
            # full-data scores + 1-D line search for the boosting step
            s_all = self._scores(encode_batch(Xp_all, self.n_qubits, theta, self.layers), template)
            gamma = _line_search(y, F, s_all)
            F = F + self.learning_rate * gamma * s_all
            self.thetas_.append(theta.copy())
            self.templates_.append(template)
            self.gammas_.append(gamma)
        return self

    def decision_function(self, X: Array) -> Array:
        X = np.asarray(X, dtype=np.float64)
        Xp = self._project(X)
        F = np.full(len(X), self.F0_)
        for theta, template, gamma in zip(self.thetas_, self.templates_, self.gammas_):
            st = encode_batch(Xp, self.n_qubits, theta, self.layers)
            F = F + self.learning_rate * gamma * self._scores(st, template)
        return F

    def predict_proba(self, X: Array) -> Array:
        return _sigmoid(self.decision_function(X))


# --------------------------------------------------------------------------- #
def _sigmoid(z: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def _balanced_subsample(y: Array, k: int, rng) -> Array:
    pos = np.flatnonzero(y == 1); neg = np.flatnonzero(y == 0)
    if len(pos) + len(neg) <= k:
        return np.arange(len(y))
    n_pos = min(len(pos), k // 2)
    n_neg = min(len(neg), k - n_pos)
    return np.concatenate([rng.choice(pos, n_pos, replace=False),
                           rng.choice(neg, n_neg, replace=False)])


def _line_search(y: Array, F: Array, s: Array) -> float:
    """1-D Newton step for the optimal γ minimising log-loss of F + γ·s."""
    p = _sigmoid(F)
    grad = float(np.sum((p - y) * s))
    hess = float(np.sum(p * (1 - p) * s * s)) + 1e-9
    return float(np.clip(-grad / hess, -10.0, 10.0))
