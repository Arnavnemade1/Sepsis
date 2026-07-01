"""Quantum Hyperdimensional Computing (QHDC) — exact statevector simulation.

Implements the QHDC primitives of Cumbo, Blankenberg et al., "Quantum
Hyperdimensional Computing: a foundational paradigm for quantum neuromorphic
architectures", npj Unconventional Computing (2026) [arXiv:2511.12664], as a
dependency-free, exact numpy statevector simulator. For N ≤ ~14 qubits the full
2^N complex amplitude vector fits in memory, so we simulate the *exact* output of
the quantum algorithm — no framework, no approximation, every operation auditable.

Why this matters for sepsis detection (the honest scientific thesis):
  Classical HDC bundles k hypervectors by majority sign, ``sign(Σ v_k)``. When many
  vectors are superposed the per-coordinate votes COLLIDE and individual components
  wash out — the information bottleneck we measured (standalone HDC caps ~0.68 AUROC
  vs the tree benchmark ~0.74; see backend/ml/README.md). A QHDC register instead
  holds a hypervector as the amplitudes of an N-qubit state (D = 2^N dimensions) and
  bundles via a Linear Combination of Unitaries (LCU): the *coherent, amplitude-
  preserving* sum ``Σ α_k |ψ_k⟩`` renormalised — no majority-vote collision. This
  module lets us measure exactly where that lossless bundling separates from the
  classical collision limit (the Dimensional-Capacity Curve).

Scope / honesty: we simulate the MATHEMATICAL RESULT the LCU+OAA circuit produces
(the normalised coherent sum) and the Hadamard/SWAP-test overlap it measures — this
is the standard, faithful statevector simulation of the algorithm, and it is exact
for the amplitudes. We do NOT model shot noise or NISQ gate error here (a separate,
optional noise model can be layered on; see `noisy_overlap`). "500× faster" in the
source paper is a *throughput* claim on quantum-native ops, not an accuracy claim —
this file makes no accuracy claim on its own; see the ablation in
`backend/ml/scripts/qhdc_ablation.py`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np

Array = np.ndarray


# --------------------------------------------------------------------------- #
# Low-level statevector helpers (exact, complex128).
# --------------------------------------------------------------------------- #
def _normalize(state: Array) -> Array:
    n = np.linalg.norm(state)
    return state / n if n > 0 else state


def random_hypervector_state(n_qubits: int, rng: np.random.Generator) -> Array:
    """A random unit hypervector as a normalised complex statevector (D=2^N)."""
    d = 1 << n_qubits
    v = rng.standard_normal(d) + 1j * rng.standard_normal(d)
    return _normalize(v.astype(np.complex128))


# --------------------------------------------------------------------------- #
# Encoding: real feature vector -> N-qubit hypervector state.
# --------------------------------------------------------------------------- #
def angle_encode(features: Array, n_qubits: int) -> Array:
    """Product-state angle encoding: each qubit i carries RY(π·f_i)|0>, giving a
    separable 2^N-D hypervector. Features are the first N components (padded/clipped
    to [0,1]); this is the resource-cheap encoding (no entanglement) and the baseline
    the variational encoder improves on."""
    f = np.clip(np.asarray(features, dtype=np.float64).ravel()[:n_qubits], 0.0, 1.0)
    if f.size < n_qubits:
        f = np.concatenate([f, np.zeros(n_qubits - f.size)])
    state = np.array([1.0 + 0j])
    for fi in f:
        a = np.cos(np.pi * fi / 2.0)
        b = np.sin(np.pi * fi / 2.0)
        state = np.kron(state, np.array([a, b], dtype=np.complex128))
    return state  # already unit norm (product of unit qubits)


def _ry(theta: float) -> Array:
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def _rz(theta: float) -> Array:
    return np.array([[np.exp(-1j * theta / 2.0), 0.0],
                     [0.0, np.exp(1j * theta / 2.0)]], dtype=np.complex128)


def _apply_1q(state: Array, gate: Array, q: int, n: int) -> Array:
    st = state.reshape([2] * n)
    st = np.tensordot(gate, st, axes=([1], [q]))
    st = np.moveaxis(st, 0, q)
    return st.reshape(-1)


def _apply_cnot(state: Array, ctrl: int, tgt: int, n: int) -> Array:
    st = state.reshape([2] * n)
    idx = [slice(None)] * n
    idx[ctrl] = 1
    sub = st[tuple(idx)]
    sub = np.flip(sub, axis=tgt if tgt < ctrl else tgt - 1)
    st[tuple(idx)] = sub
    return st.reshape(-1)


def variational_encode(features: Array, n_qubits: int, thetas: Array,
                       layers: int = 1) -> Array:
    """Parameterised (variational) encoder: interleave data-dependent RY(π·f) with
    trainable RZ(θ)·RY(θ) rotations and a ring of CNOTs (entanglement). This is the
    QHDC weak learner's tunable feature map — SPSA optimises ``thetas`` so the
    encoded state's overlap with the class prototype tracks the boosting residual.

    thetas shape: (layers, n_qubits, 2).  Returns a unit statevector (D=2^N)."""
    f = np.asarray(features, dtype=np.float64).ravel()
    fx = np.zeros(n_qubits)
    fx[:min(n_qubits, f.size)] = f[:n_qubits]
    state = np.zeros(1 << n_qubits, dtype=np.complex128)
    state[0] = 1.0
    th = np.asarray(thetas, dtype=np.float64).reshape(layers, n_qubits, 2)
    for L in range(layers):
        for q in range(n_qubits):
            state = _apply_1q(state, _ry(np.pi * fx[q]), q, n_qubits)      # data
            state = _apply_1q(state, _rz(th[L, q, 0]), q, n_qubits)        # trainable
            state = _apply_1q(state, _ry(th[L, q, 1]), q, n_qubits)
        for q in range(n_qubits):                                          # entangle (ring)
            state = _apply_cnot(state, q, (q + 1) % n_qubits, n_qubits)
    return _normalize(state)


# --------------------------------------------------------------------------- #
# Binding — phase-oracle (diagonal unitary). bind(a,b) ~ elementwise product.
# --------------------------------------------------------------------------- #
def bind(a: Array, b: Array) -> Array:
    """Binding via a phase oracle: multiply amplitudes of |a> by the phases of |b>.
    Invertible (unbind with the conjugate), similarity-preserving-under-unbind — the
    VSA binding law, realised as a diagonal unitary (Cumbo et al., §binding)."""
    phase = np.exp(1j * np.angle(b))
    return _normalize(a * phase)


def unbind(bound: Array, b: Array) -> Array:
    return _normalize(bound * np.exp(-1j * np.angle(b)))


# --------------------------------------------------------------------------- #
# Bundling — Linear Combination of Unitaries (LCU): lossless coherent sum.
# --------------------------------------------------------------------------- #
def bundle_lcu(states: Sequence[Array], weights: Optional[Sequence[float]] = None
               ) -> Array:
    """Quantum bundling: the normalised coherent sum ``Σ_k α_k |ψ_k>`` that an
    LCU+Oblivious-Amplitude-Amplification circuit prepares. Unlike classical
    ``sign(Σ v_k)`` this preserves every amplitude — no majority-vote collision.

    We return the exact ideal LCU output state (the normalised weighted sum), which
    is what the circuit yields on success; OAA makes that success deterministic. This
    is the amplitude-exact simulation of the published primitive."""
    S = np.asarray(states, dtype=np.complex128)
    if weights is None:
        w = np.ones(len(S))
    else:
        w = np.asarray(weights, dtype=np.float64)
    acc = (w[:, None] * S).sum(axis=0)
    return _normalize(acc)


def bundle_classical_sign(states: Sequence[Array]) -> Array:
    """The CLASSICAL HDC bundle for contrast: majority sign of the real parts,
    renormalised. This is the collision-prone operation QHDC replaces — used by the
    Dimensional-Capacity Curve to show where quantum bundling pulls ahead."""
    S = np.asarray(states, dtype=np.complex128)
    summed = S.real.sum(axis=0)
    signed = np.sign(summed)
    signed[signed == 0] = 1.0
    return _normalize(signed.astype(np.complex128))


# --------------------------------------------------------------------------- #
# Similarity — Hadamard test (Re<a|b>) and SWAP test (|<a|b>|^2).
# --------------------------------------------------------------------------- #
def hadamard_overlap(a: Array, b: Array) -> float:
    """Real part of the inner product Re<a|b>, the quantity a Hadamard test yields
    (P0 - P1 of the ancilla). This is the QHDC 'cosine similarity'."""
    return float(np.real(np.vdot(a, b)))


def swap_overlap(a: Array, b: Array) -> float:
    """Fidelity |<a|b>|^2 — the SWAP-test estimate (ancilla P0 = (1+|<a|b>|^2)/2)."""
    ip = np.vdot(a, b)
    return float((ip * np.conj(ip)).real)


def noisy_overlap(a: Array, b: Array, shots: int, rng: np.random.Generator,
                  depolarize: float = 0.0) -> float:
    """Hadamard overlap under finite sampling (+ optional depolarising noise) — the
    honest NISQ-era estimate. P0 = (1 + Re<a|b>)/2; sample ``shots`` Bernoulli
    outcomes. ``depolarize`` shrinks the visibility toward 0.5 (white noise)."""
    val = hadamard_overlap(a, b)
    p0 = 0.5 * (1.0 + val)
    p0 = 0.5 + (1.0 - depolarize) * (p0 - 0.5)
    k = rng.binomial(shots, min(max(p0, 0.0), 1.0))
    return 2.0 * (k / shots) - 1.0


# --------------------------------------------------------------------------- #
# A convenience object bundling the register config.
# --------------------------------------------------------------------------- #
@dataclass
class QHDCRegister:
    """An N-qubit QHDC register: D = 2^N virtual hyperdimensions."""
    n_qubits: int

    @property
    def dimension(self) -> int:
        return 1 << self.n_qubits

    def encode(self, features: Array) -> Array:
        return angle_encode(features, self.n_qubits)

    def bind(self, a: Array, b: Array) -> Array:
        return bind(a, b)

    def bundle(self, states: Sequence[Array], weights=None) -> Array:
        return bundle_lcu(states, weights)

    def similarity(self, a: Array, b: Array) -> float:
        return hadamard_overlap(a, b)
