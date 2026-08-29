"""Physiologic Coordination — detecting sepsis from the *structure* between vitals.

THE OBSERVATION THAT MOTIVATES THIS MODULE
------------------------------------------
Every feature this project has ever built (``backend/ml/data/features.py``) is a
**marginal** statistic: the value of one channel, its slope, its rolling spread,
its deviation from that patient's own history. The only cross-channel terms are a
shock index and two threshold counts. Phase 7 added 91 more such features and
bought nothing, and the frontier pass then found the remaining lever was in the
*objective*, not the feature count. Both results are consistent with one reading:
**the marginal information in hourly vitals is saturated.**

But sepsis is not a marginal phenomenon. Its bedside signs are statements about
*relationships between* channels, and none of them are expressible in the space above:

  * fever **without** the tachycardia that should accompany it (relative bradycardia),
  * tachypnea **without** hypoxia (respiratory drive answering acidosis, not O₂),
  * a heart rate that no longer tracks blood pressure — the baroreflex coupling
    inverting as distributive shock replaces a normal pressure–flow relation,
  * a narrowing pulse pressure while the mean arterial pressure is still "normal".

Godin & Buchman (Crit Care Med 1996; 24:1107) proposed exactly this as the
pathogenesis of multiple organ dysfunction: healthy organs behave as coupled
biological oscillators, and systemic inflammation **uncouples** them. That
hypothesis is one of the most cited conceptual frameworks in critical care and is
almost always operationalised on high-rate waveforms (beat-to-beat variability),
which is why it has stayed out of reach of hourly EHR pipelines like this one.
Fairchild et al. (Pediatr Res 2017; PMC5309159) did reach it in neonates — HR–SpO₂
cross-correlation was their single best predictor of sepsis — but at 2-second
resolution in the NICU.

This module asks the question that sits in the gap: **how much of it survives at
one sample per hour, in adults, at cohort scale?**

WHAT IS ACTUALLY NEW HERE
-------------------------
Not "cross-correlation of vitals" — that exists (above). The contributions are:

1. **The marginal/structural split.** For a Gaussian normative model, the pointwise
   log-likelihood of the joint minus the summed pointwise log-likelihoods of the
   marginals is an exactly computable quantity::

       C(x) = ½[ rᵀ Σ⁻¹ r − Σᵢ rᵢ²/Σᵢᵢ ]        (+ a constant in |Σ|)

   This is the *pointwise total correlation* — the local analogue of multi-information
   (Ince 2017, arXiv:1702.01591; PNAS 2023 for its use on brain activity). It is the
   surprise that exists **only** in the dependency structure: every marginal term has
   been subtracted off. It is, by construction, orthogonal to the entire incumbent
   feature space, because every incumbent feature is a function of the marginals.
   To our knowledge this decomposition has not been applied to sepsis.

2. **Multi-timescale structural divergence.** Rather than a static correlation, the
   patient carries two exponentially-weighted covariance estimates — fast (8h) and
   slow (48h) — and the detector reads the KL divergence between them, and between
   each and the population norm. "This patient's short-term coordination has diverged
   from their own long-term coordination" is a causal, per-patient, unit-free signal
   that needs no population threshold to be meaningful.

3. **Scale-invariance by construction.** Every feature here is a function of
   correlation matrices and Mahalanobis forms, so the arbitrary per-channel scale
   constants below cancel out exactly. Nothing in the patient-local state is fitted
   to a cohort: it is a deterministic causal function of that patient's own stay.
   Only the normative Σ₀ is fitted, and only on training patients.

HOW IT CAN FAIL, AND HOW THAT IS CHECKED
----------------------------------------
The obvious way this could be a lie: hourly vitals are forward-filled and sparsely
measured, so two channels can look "coupled" merely because they were sampled in the
same nursing round. Three defences, none of them assertions:

  * **Innovations.** The δ-plane uses first differences, so an unmeasured channel
    contributes an exact zero rather than a spurious repeated level.
  * **Incrementality.** The number that is reported is the gain *over* a detector
    that already holds every ``__measured``, ``__age`` and ``n_measured_this_hour``
    feature. A measurement-pattern artifact is redundant with those and cannot
    produce an incremental gain against them.
  * **Surrogates.** ``surrogate_uncouple`` builds a control that preserves every
    marginal distribution and every measurement pattern exactly, and destroys only
    the cross-channel alignment. If this detector is reading structure, it must lose
    its signal on that control; if it does not, it was never reading structure.

The honest null result is available and would be reported: if C(x) adds nothing over
the incumbent, that is a measurement of how much relational information one sample
per hour can carry, which is a fact worth having either way.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

#: The channels a bedside monitor / wearable actually streams. Same seven the rest
#: of the project calls ``WEARABLE_CHANNELS``; duplicated as an explicit ordering
#: because every matrix in this module is indexed by position in this list.
COORD_CHANNELS: List[str] = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp"]

#: Fixed per-channel scales, in the natural units of each vital. These are NOT
#: fitted — they are round textbook population spreads, hard-coded so that no
#: cohort statistic can leak into the patient-local state. Every feature this
#: module emits is scale-invariant (correlations and Mahalanobis forms), so these
#: constants cancel exactly; they exist only to keep the covariance recursions
#: numerically comfortable. ``test_scale_invariance`` proves the cancellation.
CHANNEL_SCALE: Dict[str, float] = {
    "HR": 15.0, "O2Sat": 3.0, "Temp": 0.7, "SBP": 20.0,
    "MAP": 15.0, "DBP": 12.0, "Resp": 5.0,
}

#: Physiologically named pairs whose coupling has a bedside meaning. These get
#: their own columns so a clinician can be told *which* relationship broke, rather
#: than only that some scalar divergence rose.
NAMED_PAIRS: List[Tuple[str, str, str]] = [
    ("HR", "MAP", "baroreflex"),        # rate vs perfusion pressure
    ("HR", "SBP", "pressure_rate"),     # the shock-index relationship, as a coupling
    ("HR", "Temp", "chronotropic"),     # fever without tachycardia = relative bradycardia
    ("HR", "Resp", "cardioresp"),       # cardiorespiratory coupling
    ("O2Sat", "Resp", "hypoxic_drive"), # tachypnea WITHOUT hypoxia is the sepsis pattern
    ("MAP", "Resp", "perfusion_drive"),
    ("SBP", "DBP", "pulse_pressure"),   # vascular tone / arterial compliance
]

_EPS = 1e-6


@dataclass
class CoordinationConfig:
    """Timescales of the coordination state. All halflives are in hours."""
    channels: List[str] = field(default_factory=lambda: list(COORD_CHANNELS))
    mean_halflife: float = 24.0    # personal operating point (what is "normal for them")
    fast_halflife: float = 8.0     # current coordination
    slow_halflife: float = 48.0    # the patient's own established coordination
    ridge: float = 5e-2            # shrinkage toward I for the covariance estimates
    use_innovations: bool = True   # also build the δ-plane (first differences)

    @property
    def k(self) -> int:
        return len(self.channels)


# --------------------------------------------------------------------------- #
# Patient-local state: causal, unfitted, and computed once for a whole cohort
# --------------------------------------------------------------------------- #
@dataclass
class CoordinationState:
    """The per-row coordination state of every patient in a cohort.

    ``r`` and ``d`` are the level-residual and innovation vectors; ``S_fast`` /
    ``S_slow`` are their exponentially-weighted covariance estimates. Every array
    is indexed by dataset row, and every value at row *t* uses only rows at or
    before *t* **of the same patient** — the causality audit in
    ``verify_coordination.py`` enforces this by rewriting the future and requiring
    that nothing earlier moves.
    """
    r: np.ndarray                  # (n, k)   level residual vs personal EW mean
    d: np.ndarray                  # (n, k)   innovation (first difference)
    S_fast: np.ndarray             # (n, k, k)
    S_slow: np.ndarray             # (n, k, k)
    D_fast: np.ndarray             # (n, k, k) innovation-plane fast covariance
    channels: List[str]


def build_state(X: np.ndarray, feature_names: Sequence[str], groups: np.ndarray,
                cfg: Optional[CoordinationConfig] = None) -> CoordinationState:
    """Compute the causal coordination state from a built feature matrix.

    Reads the forward-filled channel columns straight out of the project's existing
    feature matrix, so this needs no second pass over the raw PSV files and inherits
    that builder's causality guarantees for the values themselves.
    """
    cfg = cfg or CoordinationConfig()
    name_ix = {n: i for i, n in enumerate(feature_names)}
    missing = [c for c in cfg.channels if c not in name_ix]
    if missing:
        raise KeyError(f"coordination needs raw channel columns {missing} in the feature matrix")

    k = cfg.k
    scale = np.array([CHANNEL_SCALE[c] for c in cfg.channels], dtype=np.float64)
    u = X[:, [name_ix[c] for c in cfg.channels]].astype(np.float64) / scale

    codes, _ = pd.factorize(groups)                       # patient id -> dense code
    order = np.argsort(codes, kind="stable")              # rows grouped by patient
    inv = np.argsort(order, kind="stable")                # and back again
    u_s, g_s = u[order], codes[order]
    starts = np.flatnonzero(np.r_[True, g_s[1:] != g_s[:-1]])

    df = pd.DataFrame(u_s)
    grouped = df.groupby(g_s, sort=False)

    # Personal operating point, shifted one hour so the residual at t is measured
    # against a mean that does not contain t. Rows at the very start of a stay have
    # no history: their residual is zero, which is the honest "no information yet".
    mu = grouped.ewm(halflife=cfg.mean_halflife, adjust=True).mean().to_numpy(np.float64)
    mu_shift = np.roll(mu, 1, axis=0)
    mu_shift[starts] = u_s[starts]
    r_s = u_s - mu_shift

    # Innovations: an unmeasured channel is forward-filled, so its difference is
    # exactly zero and it contributes nothing rather than a fabricated co-movement.
    d_s = np.diff(u_s, axis=0, prepend=u_s[:1])
    d_s[starts] = 0.0

    S_fast_s = _ew_covariance(r_s, g_s, cfg.fast_halflife, k)
    S_slow_s = _ew_covariance(r_s, g_s, cfg.slow_halflife, k)
    D_fast_s = (_ew_covariance(d_s, g_s, cfg.fast_halflife, k)
                if cfg.use_innovations else np.zeros_like(S_fast_s))

    return CoordinationState(
        r=r_s[inv], d=d_s[inv], S_fast=S_fast_s[inv], S_slow=S_slow_s[inv],
        D_fast=D_fast_s[inv], channels=list(cfg.channels))


def _ew_covariance(v: np.ndarray, gcodes: np.ndarray, halflife: float, k: int) -> np.ndarray:
    """Per-patient exponentially-weighted covariance of ``v``, one matrix per row.

    Computed as the EW mean of the k(k+1)/2 unique outer-product terms — a single
    vectorised pandas pass over all pairs at once, rather than a Python loop stepping
    a matrix recursion 310k times.
    """
    n = len(v)
    iu, ju = np.triu_indices(k)
    prod = pd.DataFrame(v[:, iu] * v[:, ju])
    ew = (prod.groupby(gcodes, sort=False)
              .ewm(halflife=halflife, adjust=True).mean().to_numpy(np.float64))
    S = np.empty((n, k, k), dtype=np.float64)
    S[:, iu, ju] = ew
    S[:, ju, iu] = ew
    return S


# --------------------------------------------------------------------------- #
# Population-normative structure — the only fitted object in the module
# --------------------------------------------------------------------------- #
@dataclass
class NormativeStructure:
    """Σ₀: what coordinated physiology looks like, estimated on training patients.

    Fitted on the rows of patients who never became septic, so it is a model of
    *normality* rather than of the discriminative boundary. That matters at 2%
    hour-prevalence: the normative class is the one with 300k examples.
    """
    sigma: np.ndarray              # (k, k)
    prec: np.ndarray               # Σ₀⁻¹
    diag: np.ndarray               # (k,) marginal variances
    logdet: float
    corr: np.ndarray               # normative CORRELATION matrix (magnitude-free)
    corr_prec: np.ndarray          # its inverse — the normative precision of pattern
    corr_logdet: float
    channels: List[str]

    @staticmethod
    def fit(r: np.ndarray, healthy_rows: np.ndarray, channels: List[str],
            ridge: float = 5e-2, r_struct: Optional[np.ndarray] = None,
            ) -> "NormativeStructure":
        """``r_struct`` is the per-patient-normalised residual the structure plane
        reads. The normative correlation must be estimated in that same space —
        estimating it on raw-scale residuals and applying it to normalised ones
        would silently mix the two planes back together."""
        v = r[healthy_rows]
        k = v.shape[1]
        sigma = np.cov(v, rowvar=False)
        # Shrink toward a scaled identity: the estimate must stay invertible even
        # when a fold's normative rows are few or a channel is near-constant.
        sigma = (1.0 - ridge) * sigma + ridge * np.eye(k) * np.trace(sigma) / k
        sign, logdet = np.linalg.slogdet(sigma)
        if sign <= 0:
            raise ValueError("normative covariance is not positive definite")
        vs = v if r_struct is None else r_struct[healthy_rows]
        sig_s = np.cov(vs, rowvar=False)
        sig_s = (1.0 - ridge) * sig_s + ridge * np.eye(k) * np.trace(sig_s) / k
        d = np.sqrt(np.diag(sig_s))
        corr = sig_s / np.outer(d, d)
        _, corr_logdet = np.linalg.slogdet(corr)
        return NormativeStructure(sigma=sigma, prec=np.linalg.inv(sigma),
                                  diag=np.diag(sigma).copy(), logdet=float(logdet),
                                  corr=corr, corr_prec=np.linalg.inv(corr),
                                  corr_logdet=float(corr_logdet),
                                  channels=list(channels))


def pointwise_coordination_surprisal(r: np.ndarray, norm: NormativeStructure) -> Dict[str, np.ndarray]:
    """The joint/marginal split of the pointwise Gaussian surprisal.

    Returns the joint Mahalanobis form, the marginal-only form, and their difference —
    the pointwise total correlation. The difference is the whole point: a row can be
    entirely unremarkable channel by channel and still be highly improbable as a
    *combination*, and only the difference sees that. Positive means the observed
    combination is less likely than independence would predict ("these values do not
    belong together"); negative means the deviations move in their usual concert.
    """
    joint = np.einsum("ni,ij,nj->n", r, norm.prec, r)
    marg = np.sum(r * r / norm.diag, axis=1)
    return {"joint": joint, "marginal": marg, "surprisal": joint - marg}


def _kl_gaussian(A: np.ndarray, B_inv: np.ndarray, B_logdet: float,
                 ridge: float) -> np.ndarray:
    """KL( N(0,A) ‖ N(0,B) ) for a batch of A, in nats.

    = ½[ tr(B⁻¹A) − k + ln|B| − ln|A| ]. A is ridged first: early in a stay the
    exponentially-weighted A has an effective sample size below k and is singular by
    construction, and a raw log-determinant there would be −∞ rather than "unknown".
    """
    k = A.shape[-1]
    scale = np.trace(A, axis1=-2, axis2=-1)[:, None, None] / k
    A = (1.0 - ridge) * A + ridge * np.maximum(scale, _EPS) * np.eye(k)
    tr = np.einsum("ij,nji->n", B_inv, A)
    sign, logdet_a = np.linalg.slogdet(A)
    logdet_a = np.where(sign > 0, logdet_a, -30.0)
    return 0.5 * (tr - k + B_logdet - logdet_a)


def _batched_corr(S: np.ndarray) -> np.ndarray:
    """Correlation matrices from a batch of covariances (scale-invariance lives here).

    A degenerate channel — zero spread so far in this stay — is given a unit diagonal
    and zero off-diagonals rather than being divided by a small constant. Flooring the
    variance would make the result depend on the arbitrary units of the input, which is
    precisely the property this function exists to remove.
    """
    d = np.sqrt(np.maximum(np.diagonal(S, axis1=-2, axis2=-1), 0.0))
    ok = d > 0.0
    inv = np.where(ok, 1.0 / np.where(ok, d, 1.0), 0.0)
    C = S * inv[:, :, None] * inv[:, None, :]
    k = S.shape[-1]
    eye = np.eye(k, dtype=bool)
    dead = ~ok
    if dead.any():
        C[np.broadcast_to(dead[:, :, None] | dead[:, None, :], C.shape)] = 0.0
        C[np.broadcast_to(eye, C.shape) & np.broadcast_to(dead[:, :, None], C.shape)] = 1.0
    return C


def _spectral(S: np.ndarray, prefix: str, chunk: int = 40000) -> Dict[str, np.ndarray]:
    """Network descriptors of a batch of correlation matrices.

    ``lambda1`` is how much of the physiology has collapsed onto a single mode
    (integration); ``eig_entropy`` is the spread of the spectrum, which is the
    natural numeric reading of Buchman's "uncoupling" — a fragmented system has its
    variance spread across many independent modes instead of a few coordinated ones.
    """
    n, k = S.shape[0], S.shape[-1]
    lam1 = np.empty(n); ent = np.empty(n); logdet = np.empty(n)
    for a in range(0, n, chunk):
        b = min(a + chunk, n)
        C = _batched_corr(S[a:b])
        C = 0.5 * (C + np.swapaxes(C, -1, -2))            # symmetrise against fp drift
        w = np.linalg.eigvalsh(C)
        w = np.maximum(w, _EPS)
        p = w / w.sum(axis=1, keepdims=True)
        lam1[a:b] = w[:, -1] / k
        ent[a:b] = -(p * np.log(p)).sum(axis=1) / np.log(k)
        logdet[a:b] = np.log(w).sum(axis=1)
    off = ~np.eye(k, dtype=bool)
    mean_abs = np.abs(_batched_corr(S)[:, off]).mean(axis=1)
    return {f"{prefix}_lambda1": lam1, f"{prefix}_eig_entropy": ent,
            f"{prefix}_logdet": logdet, f"{prefix}_mean_abs_corr": mean_abs}


# --------------------------------------------------------------------------- #
# The feature plane
# --------------------------------------------------------------------------- #
def _normalise_plane(v: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Divide a residual vector by the per-channel spread the patient is currently
    showing, so what remains is pure *pattern*: which way the channels leaned
    relative to each other, with the size of the lean divided out."""
    # An ABSOLUTE floor here would silently break the scale-invariance this module
    # claims: rescaling every channel moves the spreads across a fixed threshold and
    # changes which rows get floored. (The audit caught exactly that — max |delta|
    # 14.8 under a 3.7x rescale.) So the guard is a strict positivity test, which is
    # itself scale-free: at the first hour of a stay the exponentially-weighted
    # covariance is exactly zero, and those rows are defined to have no deviation yet
    # rather than an arbitrary large one. The clip is on the already-normalised value,
    # so it is scale-free too.
    sd = np.sqrt(np.maximum(np.diagonal(S, axis1=-2, axis2=-1), 0.0))
    out = np.where(sd > 0.0, v / np.where(sd > 0.0, sd, 1.0), 0.0)
    return np.clip(np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0), -8.0, 8.0)


def coordination_features(state: CoordinationState, norm: NormativeStructure,
                          cfg: Optional[CoordinationConfig] = None,
                          plane: str = "structure",
                          ) -> Tuple[np.ndarray, List[str]]:
    """Turn the causal state + fitted normative structure into a feature matrix.

    ``plane`` is the methodological crux of this module, not a tuning knob:

    ``"structure"``
        Every matrix is reduced to a **correlation** matrix and every residual is
        divided by the spread the patient is currently showing, before anything is
        computed. All magnitude information — how abnormal, how variable, how fast —
        is therefore gone, and what remains can only describe *which way the channels
        leaned relative to one another*. This is the plane whose signal cannot be a
        restatement of the incumbent, because the incumbent is magnitudes.
    ``"magnitude"``
        The complement: the same quantities computed on raw covariances, which do
        carry variance and severity. Kept so the split can be *measured* rather than
        asserted — if "structure" scores like "magnitude", the separation failed.
    ``"both"``
        The union, for a detector that is allowed everything.

    Contains no channel level, no slope, no measurement flag and no demographic in
    any plane.
    """
    cfg = cfg or CoordinationConfig()
    if plane not in ("structure", "magnitude", "both"):
        raise ValueError(f"unknown plane {plane!r}")
    cols: Dict[str, np.ndarray] = {}
    want_struct = plane in ("structure", "both")
    want_mag = plane in ("magnitude", "both")

    Cf, Cs = _batched_corr(state.S_fast), _batched_corr(state.S_slow)
    Cd = _batched_corr(state.D_fast)

    # --- 1. pointwise surprisal ---------------------------------------------- #
    if want_struct:
        # Magnitude-free form. The residual is normalised by the patient's own
        # current spread and read against the normative *correlation*, so the
        # marginal term collapses to Σᵢr̃ᵢ² and the difference is exactly the
        # off-diagonal contribution: r̃ᵀ(P₀ − I)r̃, the pointwise total correlation
        # with every scale of every channel divided out.
        rt = _normalise_plane(state.r, state.S_slow)   # already clipped, scale-free
        joint = np.einsum("ni,ij,nj->n", rt, norm.corr_prec, rt)
        marg = np.sum(rt * rt, axis=1)
        cols["coordS_surprisal"] = joint - marg
        cols["coordS_surprisal_share"] = (joint - marg) / (np.abs(joint) + 1.0)
    if want_mag:
        lev = pointwise_coordination_surprisal(state.r, norm)
        cols["coordM_mah_joint"] = lev["joint"]
        cols["coordM_mah_marginal"] = lev["marginal"]
        cols["coordM_surprisal"] = lev["surprisal"]

    # --- 2. structural divergence across timescales -------------------------- #
    if want_struct:
        # KL between *correlation* matrices: a patient whose vitals merely became
        # more variable moves this by exactly zero, because scaling a covariance
        # leaves its correlation matrix untouched. That is the whole reason this
        # plane exists — ``coord_kl_self`` on raw covariances was the strongest
        # single feature in the first screen, and could not be distinguished from
        # "this patient got more unstable", which the incumbent already knows.
        cols["coordS_kl_fast_norm"] = _kl_gaussian(Cf, norm.corr_prec, norm.corr_logdet, cfg.ridge)
        cols["coordS_kl_slow_norm"] = _kl_gaussian(Cs, norm.corr_prec, norm.corr_logdet, cfg.ridge)
        cols["coordS_kl_self"] = _kl_batch_pairs(Cf, _ridge_batch(Cs, cfg.ridge), cfg.ridge)
        cols["coordS_kl_inno_norm"] = _kl_gaussian(Cd, norm.corr_prec, norm.corr_logdet, cfg.ridge)
    if want_mag:
        klf = _kl_gaussian(state.S_fast, norm.prec, norm.logdet, cfg.ridge)
        kls = _kl_gaussian(state.S_slow, norm.prec, norm.logdet, cfg.ridge)
        cols["coordM_kl_fast_norm"] = klf
        cols["coordM_kl_slow_norm"] = kls
        cols["coordM_kl_self"] = _kl_batch_pairs(
            state.S_fast, _ridge_batch(state.S_slow, cfg.ridge), cfg.ridge)
        cols["coordM_kl_delta"] = klf - kls

    # --- 3. network descriptors ---------------------------------------------- #
    # ``_spectral`` already correlation-normalises internally, so these are
    # magnitude-free and belong to the structure plane by construction. The one
    # exception is logdet of the raw covariance, which is a generalised variance.
    if want_struct:
        fast_spec = _spectral(state.S_fast, "coordS_fast")
        slow_spec = _spectral(state.S_slow, "coordS_slow")
        inno_spec = _spectral(state.D_fast, "coordS_inno")
        cols.update(fast_spec); cols.update(slow_spec); cols.update(inno_spec)
        for key in ("lambda1", "eig_entropy", "logdet", "mean_abs_corr"):
            cols[f"coordS_dspec_{key}"] = (fast_spec[f"coordS_fast_{key}"]
                                           - slow_spec[f"coordS_slow_{key}"])
    if want_mag:
        for tag, S in (("fast", state.S_fast), ("slow", state.S_slow), ("inno", state.D_fast)):
            d = np.maximum(np.diagonal(S, axis1=-2, axis2=-1), _EPS)
            cols[f"coordM_{tag}_genvar"] = np.log(d).sum(axis=1)

    # --- 4. named bedside couplings ------------------------------------------ #
    # Correlations are magnitude-free, so every named pair is structural.
    if want_struct:
        ix = {c: i for i, c in enumerate(state.channels)}
        for a, b, label in NAMED_PAIRS:
            if a not in ix or b not in ix:
                continue
            i, j = ix[a], ix[b]
            cols[f"coordS_pair_{label}"] = Cf[:, i, j]
            cols[f"coordS_pair_{label}_drift"] = Cf[:, i, j] - Cs[:, i, j]
            if cfg.use_innovations:
                cols[f"coordS_inno_{label}"] = Cd[:, i, j]

    names = list(cols)
    M = np.column_stack([np.nan_to_num(cols[n], nan=0.0, posinf=0.0, neginf=0.0)
                         for n in names]).astype(np.float32)
    return M, names


def conditional_residual_features(state: CoordinationState, norm: NormativeStructure,
                                  cfg: Optional[CoordinationConfig] = None,
                                  ) -> Tuple[np.ndarray, List[str]]:
    """Each vital's deviation from what *the other vitals* predict it should be.

    This is the well-powered way to ask the relational question. The per-patient
    correlation matrices used by :func:`coordination_features` need 21 parameters and
    an 8-hour exponential window supplies about 11 effective samples to estimate them
    with, so those estimates are dominated by noise — which is exactly what the
    surrogate control showed, since noise is unchanged by destroying alignment.

    The fix is to put the correlation structure where the data is. Σ₀ is estimated
    once from ~200k normative patient-hours, and the *patient* only has to supply
    their position relative to it. For a Gaussian, the conditional expectation of one
    channel given all the others has a closed form in the precision matrix
    P = Σ₀⁻¹::

        E[xᵢ | x₋ᵢ] = μᵢ − (1/Pᵢᵢ) Σⱼ≠ᵢ Pᵢⱼ (xⱼ − μⱼ)
        zᵢ = (xᵢ − E[xᵢ | x₋ᵢ]) · √Pᵢᵢ   =   (P r)ᵢ / √Pᵢᵢ

    ``zᵢ`` is one number per channel per hour, each estimated against a matrix fitted
    on 200k rows, and each with a bedside sentence attached: *"the heart rate is 2.4
    standard deviations below what this patient's blood pressure, temperature and
    respiratory rate say it should be."* That sentence is relative bradycardia — a
    textbook sepsis sign that no marginal feature in this project can express, because
    the heart rate is not abnormal; it is abnormal **for the company it is keeping**.

    Note what is and is not new. The Gaussian conditional is elementary statistics.
    Using it as the detection representation for sepsis — every channel scored against
    the other channels rather than against its own history or a population range — is
    the part that is not standard, and the part that is measured below.
    """
    cfg = cfg or CoordinationConfig()
    P = norm.corr_prec
    rt = np.clip(_normalise_plane(state.r, state.S_slow), -8.0, 8.0)
    # einsum rather than ``rt @ P``: numpy 2.0 on Apple's Accelerate BLAS raises
    # spurious divide-by-zero/overflow warnings from matmul on this shape. The two
    # agree bit for bit (checked), but the false warnings would mask a real one.
    z = np.einsum("ni,ij->nj", rt, P) / np.sqrt(np.maximum(np.diag(P), _EPS))

    cols: Dict[str, np.ndarray] = {}
    for i, ch in enumerate(state.channels):
        cols[f"cond_z_{ch}"] = z[:, i]
    absz = np.abs(z)
    cols["cond_z_absmax"] = absz.max(axis=1)
    cols["cond_z_abssum"] = absz.sum(axis=1)
    # How much of the hour's total abnormality is *relational*: the same deviations
    # read against the other channels rather than against the channel's own scale.
    cols["cond_z_vs_marginal"] = absz.sum(axis=1) - np.abs(rt).sum(axis=1)
    cols["cond_n_discordant"] = (absz > 2.0).sum(axis=1).astype(np.float64)
    return (np.column_stack([np.nan_to_num(cols[n], nan=0.0, posinf=0.0, neginf=0.0)
                             for n in cols]).astype(np.float32),
            list(cols))


def temporal_context(M: np.ndarray, names: Sequence[str], groups: np.ndarray,
                     halflife: float = 8.0) -> Tuple[np.ndarray, List[str]]:
    """Add per-patient causal EW level and one-hour change to every column.

    A discordance that has held for twelve hours means something different from one
    that appeared in the last hour, and a single-row feature cannot say which it is.
    """
    codes, _ = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    inv = np.argsort(order, kind="stable")
    g_s = codes[order]
    starts = np.flatnonzero(np.r_[True, g_s[1:] != g_s[:-1]])
    df = pd.DataFrame(M[order].astype(np.float64))
    ew = df.groupby(g_s, sort=False).ewm(halflife=halflife, adjust=True).mean().to_numpy()
    d1 = np.diff(M[order].astype(np.float64), axis=0, prepend=M[order][:1])
    d1[starts] = 0.0
    out = np.hstack([M, ew[inv].astype(np.float32), d1[inv].astype(np.float32)])
    return out, (list(names) + [f"{n}__ew" for n in names] + [f"{n}__d1" for n in names])


def _ridge_batch(S: np.ndarray, ridge: float) -> np.ndarray:
    k = S.shape[-1]
    scale = np.trace(S, axis1=-2, axis2=-1)[:, None, None] / k
    return (1.0 - ridge) * S + ridge * np.maximum(scale, _EPS) * np.eye(k)


def _kl_batch_pairs(A: np.ndarray, B: np.ndarray, ridge: float,
                    chunk: int = 40000) -> np.ndarray:
    """KL(N(0,Aₙ) ‖ N(0,Bₙ)) where both sides vary per row."""
    n, k = A.shape[0], A.shape[-1]
    out = np.empty(n)
    for a0 in range(0, n, chunk):
        b0 = min(a0 + chunk, n)
        A_c = _ridge_batch(A[a0:b0], ridge)
        B_c = B[a0:b0]
        B_inv = np.linalg.inv(B_c)
        tr = np.einsum("nij,nji->n", B_inv, A_c)
        sa, lda = np.linalg.slogdet(A_c)
        sb, ldb = np.linalg.slogdet(B_c)
        lda = np.where(sa > 0, lda, -30.0)
        ldb = np.where(sb > 0, ldb, -30.0)
        out[a0:b0] = 0.5 * (tr - k + ldb - lda)
    return out


# --------------------------------------------------------------------------- #
# The surrogate control
# --------------------------------------------------------------------------- #
def surrogate_uncouple(X: np.ndarray, feature_names: Sequence[str], groups: np.ndarray,
                       channels: Optional[List[str]] = None, max_lag: int = 6,
                       seed: int = 0, mode: str = "independent") -> np.ndarray:
    """A cohort with identical marginals and identical sampling, and no coupling.

    Each channel after the first is **delayed** by its own random lag, drawn once per
    patient per channel, with the stay's first value held during the gap. Every
    channel keeps its own trajectory, its own distribution and its own relationship
    to the clock; only the *alignment between* channels is broken.

    ``mode`` selects the arm of a **matched pair**, and the pair is the point:

    ``"independent"``
        each channel gets its own lag — alignment destroyed, staleness added.
    ``"common"``
        every channel gets the *same* lag — alignment preserved exactly, the same
        staleness added.

    Delaying a channel does two things at once: it desynchronises it from the others,
    and it makes it older. Only the first is the thing under test. Running both arms
    at a matched lag cancels the staleness, so the gap between them is the value of
    inter-channel alignment and nothing else. A single-arm surrogate cannot separate
    those and would charge decoupling for a cost that was really staleness.

    A note on a control that did not work, because it is the reason this one is
    written the way it is. The first version used a circular ``np.roll``, which is
    the textbook surrogate for stationary signals. It is wrong here: wrapping moves a
    septic patient's late, deteriorated hours to the *front* of their stay, so every
    hour of that stay inherits evidence of the outcome. Both the coordination
    detector and the incumbent scored **higher** on that surrogate than on the real
    cohort — a control that manufactures signal rather than removing it. A delay
    only ever moves information forward in time, so it cannot leak an outcome
    backwards, and a detector that scores the same on it was not reading alignment.
    """
    channels = channels or list(COORD_CHANNELS)
    name_ix = {n: i for i, n in enumerate(feature_names)}
    out = X.copy()
    rng = np.random.default_rng(seed)
    codes, _ = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    g_s = codes[order]
    starts = np.flatnonzero(np.r_[True, g_s[1:] != g_s[:-1]])
    bounds = np.r_[starts, len(g_s)]

    if mode not in ("independent", "common"):
        raise ValueError(f"unknown surrogate mode {mode!r}")
    if max_lag <= 0:
        return out                          # identity arm, for the pipeline sanity check

    # Draw the lags first so both arms consume the same randomness in the same order:
    # the common arm reuses each patient's first channel lag for every channel, so the
    # two arms are matched patient by patient rather than only in expectation.
    n_pat = len(bounds) - 1
    lags = rng.integers(1, max_lag + 1, size=(n_pat, len(channels)))
    if mode == "common":
        lags[:] = lags[:, :1]

    for ci, ch in enumerate(channels):
        if ch not in name_ix:
            continue
        col = name_ix[ch]
        vals = out[order, col]
        for pi, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
            span = b - a
            if span < 3:
                continue
            lag = int(min(lags[pi, ci], span - 1))
            seg = vals[a:b]
            vals[a:b] = np.concatenate([np.full(lag, seg[0]), seg[:-lag]])
        out[order, col] = vals
    return out


# --------------------------------------------------------------------------- #
# The organ-system layer — the hypothesis tested at the level it was stated
# --------------------------------------------------------------------------- #
#
# The first pass of this module tested inter-channel coupling on seven bedside
# vitals: HR, O2Sat, Temp, SBP, MAP, DBP, Resp. Six of those seven are outputs of a
# *single* organ system — the cardiopulmonary one — so finding them coupled, and
# finding that coupling uninformative, says very little about a hypothesis whose
# subject is coupling **between organ systems**. Buchman's claim is that sepsis
# uncouples the heart from the kidney from the liver, not the systolic pressure from
# the diastolic pressure.
#
# This layer tests the claim as stated. Each organ system is reduced to one causal
# dysfunction score, and the coordination machinery above is then applied to the
# seven-dimensional *organ* vector instead of the raw channel vector. Two properties
# make this the better-powered experiment as well as the more faithful one: organ
# dysfunction evolves over days rather than minutes, so a 48-hour window contains
# many effective samples rather than eleven; and the dimension is set by physiology
# rather than by how many ways a blood pressure can be written down.
#
# It is also the definition of the disease. Sepsis-3 *is* multi-organ dysfunction
# secondary to infection, so a detector that measures whether organ systems are
# decompensating together is measuring the diagnostic criterion directly.

#: organ system -> (channel, direction). +1 means high values are the dysfunction;
#: -1 means low values are; 0 means deviation in *either* direction is (fever and
#: hypothermia are both sepsis, and so are leukocytosis and leukopenia).
ORGAN_SYSTEMS: Dict[str, List[Tuple[str, int]]] = {
    "cardiovascular": [("MAP", -1), ("SBP", -1), ("DBP", -1), ("HR", +1)],
    "respiratory":    [("Resp", +1), ("O2Sat", -1), ("PaCO2", 0)],
    "renal":          [("Creatinine", +1), ("BUN", +1)],
    "hepatic":        [("Bilirubin_total", +1)],
    "haematologic":   [("Platelets", -1), ("WBC", 0)],
    "metabolic":      [("Lactate", +1), ("HCO3", -1), ("pH", -1), ("Glucose", 0)],
    "thermoregulatory": [("Temp", 0)],
}

#: Fixed spreads for the lab channels, same rationale as ``CHANNEL_SCALE``: round
#: textbook numbers, never fitted, and they cancel out of every emitted feature.
LAB_SCALE: Dict[str, float] = {
    "Lactate": 1.5, "WBC": 6.0, "Creatinine": 1.0, "Platelets": 100.0,
    "BUN": 15.0, "HCO3": 4.0, "pH": 0.08, "PaCO2": 8.0,
    "Bilirubin_total": 1.5, "Glucose": 45.0,
}


def organ_dysfunction(X: np.ndarray, feature_names: Sequence[str], groups: np.ndarray,
                      baseline_halflife: float = 48.0,
                      ) -> Tuple[np.ndarray, List[str]]:
    """One causal dysfunction score per organ system per patient-hour.

    A channel contributes ``direction × (value − that patient's own slow baseline)``
    in units of a fixed channel spread, so the score reads "sigmas into the bad
    direction, for this patient". Channels with direction 0 contribute ``|deviation|``
    because for those the disease lies at both ends. An organ's score is the mean over
    whichever of its channels the patient actually has, so a stay missing a bilirubin
    scores its hepatic system on nothing rather than on an imputed normal.
    """
    name_ix = {n: i for i, n in enumerate(feature_names)}
    scales = {**CHANNEL_SCALE, **LAB_SCALE}
    codes, _ = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    inv = np.argsort(order, kind="stable")
    g_s = codes[order]
    starts = np.flatnonzero(np.r_[True, g_s[1:] != g_s[:-1]])

    organs = [o for o, chs in ORGAN_SYSTEMS.items()
              if any(c in name_ix for c, _ in chs)]
    out = np.zeros((len(groups), len(organs)), dtype=np.float64)

    for oi, organ in enumerate(organs):
        parts = []
        for ch, direction in ORGAN_SYSTEMS[organ]:
            if ch not in name_ix:
                continue
            v = X[order, name_ix[ch]].astype(np.float64) / scales.get(ch, 1.0)
            base = (pd.Series(v).groupby(g_s, sort=False)
                    .ewm(halflife=baseline_halflife, adjust=True).mean().to_numpy())
            base = np.roll(base, 1); base[starts] = v[starts]
            dev = v - base
            parts.append(np.abs(dev) if direction == 0 else direction * dev)
        out[:, oi] = np.mean(parts, axis=0)[inv] if parts else 0.0
    return out.astype(np.float64), organs


def organ_state(O: np.ndarray, groups: np.ndarray,
                cfg: Optional[CoordinationConfig] = None) -> CoordinationState:
    """Run the coordination recursions on the organ vector rather than channels."""
    cfg = cfg or CoordinationConfig()
    k = O.shape[1]
    codes, _ = pd.factorize(groups)
    order = np.argsort(codes, kind="stable")
    inv = np.argsort(order, kind="stable")
    o_s, g_s = O[order], codes[order]
    starts = np.flatnonzero(np.r_[True, g_s[1:] != g_s[:-1]])

    mu = (pd.DataFrame(o_s).groupby(g_s, sort=False)
          .ewm(halflife=cfg.mean_halflife, adjust=True).mean().to_numpy(np.float64))
    mu_shift = np.roll(mu, 1, axis=0); mu_shift[starts] = o_s[starts]
    r_s = o_s - mu_shift
    d_s = np.diff(o_s, axis=0, prepend=o_s[:1]); d_s[starts] = 0.0

    return CoordinationState(
        r=r_s[inv], d=d_s[inv],
        S_fast=_ew_covariance(r_s, g_s, cfg.fast_halflife, k)[inv],
        S_slow=_ew_covariance(r_s, g_s, cfg.slow_halflife, k)[inv],
        D_fast=_ew_covariance(d_s, g_s, cfg.fast_halflife, k)[inv],
        channels=[f"organ_{i}" for i in range(k)])


def organ_synchrony_features(O: np.ndarray, organs: List[str], groups: np.ndarray,
                             cfg: Optional[CoordinationConfig] = None,
                             ) -> Tuple[np.ndarray, List[str]]:
    """Marginal and structural readings of the organ-dysfunction vector.

    The split is kept as strictly as in the channel layer, because it is the whole
    experiment. ``organM_*`` is the burden — how many systems are failing and how
    badly — which is essentially a SOFA count and which the lab-augmented incumbent
    can already assemble. ``organS_*`` is **synchrony**: whether the systems are
    moving together. Only the second is the hypothesis. A chronic dialysis patient
    with a bad kidney and a bad liver has high burden and no synchrony; a septic
    patient is supposed to show systems turning down *at the same time*.
    """
    cfg = cfg or CoordinationConfig()
    st = organ_state(O, groups, cfg)
    cols: Dict[str, np.ndarray] = {}

    # --- marginal: the burden -------------------------------------------------
    cols["organM_mean"] = O.mean(axis=1)
    cols["organM_max"] = O.max(axis=1)
    cols["organM_n_failing"] = (O > 1.0).sum(axis=1).astype(np.float64)
    cols["organM_spread"] = O.std(axis=1)

    # --- structural: the synchrony -------------------------------------------
    Cf, Cs = _batched_corr(st.S_fast), _batched_corr(st.S_slow)
    k = O.shape[1]
    off = ~np.eye(k, dtype=bool)
    # Mean pairwise correlation of the organ trajectories: the single number that
    # says "these systems are moving as one". This is the synchrony index.
    cols["organS_synchrony"] = Cf[:, off].mean(axis=1)
    cols["organS_synchrony_slow"] = Cs[:, off].mean(axis=1)
    cols["organS_synchrony_drift"] = cols["organS_synchrony"] - cols["organS_synchrony_slow"]
    spec = _spectral(st.S_fast, "organS")
    cols.update(spec)
    # Concordance: of the systems currently deviating, do they all lean the same way?
    sign = np.sign(st.r)
    cols["organS_concordance"] = np.abs(sign.sum(axis=1)) / max(k, 1)
    cols["organS_codeviation"] = (np.abs(st.r) > 0.5).sum(axis=1).astype(np.float64)

    names = list(cols)
    M = np.column_stack([np.nan_to_num(cols[n], nan=0.0, posinf=0.0, neginf=0.0)
                         for n in names]).astype(np.float32)
    return M, names
