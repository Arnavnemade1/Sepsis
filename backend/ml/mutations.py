"""The mutation catalogue — one entry per real failure mode this repo has suffered.

Every mutation here corresponds to a mistake that was actually made in this project and
cost something. They are not mutation *operators* chosen for ease of implementation;
they are the specific ways a scientific claim has been observed to go wrong here.

Reading order, roughly by what each has cost:

    fit-identity        3 false findings in one investigation
    causality           would have posted a lead time made of hindsight
    exchangeability     an interval several times too narrow
    class-conditioning  the whole conformal construction going degenerate at 2%
    naive-statistic     an estimator that reads positive on pure noise
    label-leak          a deployment method silently becoming an oracle
    floor-removal       a quantile estimated from two patients, with a guarantee on it
"""
from __future__ import annotations

import numpy as np

from backend.ml.falsify import mutation

CONF = "backend.ml.scripts.verify_conformal"
SHIFT = "backend.ml.scripts.verify_shift"
MR = "backend.ml.scripts.verify_minirocket"
OP = "backend.ml.scripts.verify_operating_point"
FID = "backend.ml.scripts.verify_fitid"


# --------------------------------------------------------------------------- #
# causality — a transform that reads forward in time
# --------------------------------------------------------------------------- #
@mutation("causality", MR, ["a future spike moves NO earlier hour"],
          note="the textbook centred MiniRocket kernel; reads (K-1)*d/2 hours ahead")
def centred_convolution(mod):
    import backend.ml.models.minirocket as M
    K = M.KERNEL_LEN

    def conv(x, w, d):
        T = x.shape[0]
        y = np.zeros(T, dtype=np.float32)
        half = (K - 1) // 2 * d
        for j in range(K):
            sh = (K - 1 - j) * d - half            # negative shift => future tap
            y += w[j] * x[np.clip(np.arange(T) - sh, 0, T - 1)]
        return y
    M.causal_conv = conv
    mod.causal_conv = conv


@mutation("causality", MR, ["a future spike moves NO earlier hour"],
          note="whole-stay PPV pooling: every hour sees the entire admission")
def whole_series_ppv(mod):
    import backend.ml.models.minirocket as M

    def ppv(ind, window):
        return np.full(ind.shape[0], float(ind.mean()), dtype=np.float32)
    M.trailing_ppv = ppv
    mod.trailing_ppv = ppv


# --------------------------------------------------------------------------- #
# exchangeability — the unit of resampling
# --------------------------------------------------------------------------- #
@mutation("exchangeability", MR, ["rewriting one patient leaves every other patient"],
          note="treat the whole cohort as one series; stays bleed into each other")
def no_patient_split(mod):
    import backend.ml.models.minirocket as M

    def bounds(groups, hour_index):
        order = np.lexsort((hour_index, groups))
        return order, np.array([0, len(groups)])
    M.CausalMiniRocket._patient_bounds = staticmethod(bounds)


@mutation("exchangeability", MR, ["a shuffled row order yields identical per-row"],
          note="drop the hour sort; 'causal' over arrival order is meaningless")
def no_hour_sort(mod):
    import backend.ml.models.minirocket as M

    def bounds(groups, hour_index):
        order = np.argsort(groups, kind="stable")
        g = groups[order]
        starts = np.flatnonzero(np.r_[True, g[1:] != g[:-1]])
        return order, np.r_[starts, len(g)]
    M.CausalMiniRocket._patient_bounds = staticmethod(bounds)


@mutation("exchangeability", CONF, ["row-split calibration overstates coverage"],
          note="split rows, not patients — the leak that flatters every coverage number")
def conformal_row_split(mod):
    mod._cal_test_by_row = mod._cal_test_by_patient


@mutation("exchangeability", CONF, ["macro (per-patient) coverage differs from micro"],
          note="report macro as micro, hiding within-stay correlation entirely")
def macro_equals_micro(mod):
    import backend.ml.conformal as cf
    orig = cf._cov

    def patched(covered, y, groups):
        d = orig(covered, y, groups)
        return dict(d, macro=d["micro"], macro_pos=d["micro_pos"], macro_neg=d["micro_neg"])
    cf._cov = patched
    mod.cf = cf


# --------------------------------------------------------------------------- #
# class-conditioning — the fix that keeps conformal alive at 2% prevalence
# --------------------------------------------------------------------------- #
@mutation("class-conditioning", CONF, ["...while class-conditional does admit it"],
          note="pool both classes into one quantile; label 1 never enters a set again")
def pooled_quantile(mod):
    import backend.ml.conformal as cf

    def pooled(stratum, y, p, groups, alpha, fallback, mh, mp, w=None):
        s_true = np.where(y == 1, 1.0 - p, p)
        q = cf.conformal_quantile(s_true, alpha)
        return cf.CellBand(stratum=stratum, lo=1.0 - q, hi=q, q_pos=q, q_neg=q,
                           n_hours_pos=int((y == 1).sum()), n_hours_neg=int((y == 0).sum()),
                           n_patients_pos=0, n_patients_neg=0,
                           fallback_pos=False, fallback_neg=False)
    cf._cell_band = pooled
    mod.cf = cf


# --------------------------------------------------------------------------- #
# naive-statistic — a corrected estimator replaced by its uncorrected form
# --------------------------------------------------------------------------- #
@mutation("naive-statistic", CONF, ["(n+1) correction reaches nominal"],
          note="the textbook empirical quantile; under-covers by ~1/n forever")
def plain_empirical_quantile(mod):
    import backend.ml.conformal as cf
    cf.conformal_quantile = lambda s, a: float(np.quantile(np.asarray(s, float), 1 - a))
    mod.cf = cf


@mutation("naive-statistic", SHIFT, ["EM recovers the target prior"],
          note="anchor EM to the label prior; a 3.6% calibration gap collapses it to 0")
def em_mis_anchored(mod):
    import backend.ml.shift as sh
    orig = sh.estimate_prior_em
    sh.estimate_prior_em = lambda p_tgt, pi_src, p_src=None, **k: orig(p_tgt, pi_src, None, **k)
    mod.shift = sh


@mutation("naive-statistic", SHIFT, ["...and lopsided weights move it"],
          note="a weighted quantile that ignores its weights. NOTE the target check: "
               "'uniform weights reproduce the empirical quantile' is SATISFIED by this "
               "mutation, since ignoring the weights is exactly what uniform weights do. "
               "The engine reported this as a survivor and the fault was in the mutation "
               "spec, not the suite — which is the audit working on its own author.")
def weighted_quantile_ignores_weights(mod):
    import backend.ml.shift as sh
    sh.weighted_quantile = lambda v, w, q: float(
        np.quantile(np.asarray(v, float), q, method="lower"))
    mod.shift = sh


@mutation("naive-statistic", SHIFT, ["BBSE recovers it too"],
          note="BBSE that just echoes the source prior back")
def bbse_echoes_source(mod):
    import backend.ml.shift as sh
    sh.estimate_prior_bbse = lambda y, ps, pt, threshold: float(
        np.mean(np.asarray(y).astype(int)))
    mod.shift = sh


@mutation("naive-statistic", SHIFT, ["...while changing how many rows cross"],
          note="prior correction as the identity map — moves nothing at all")
def prior_correction_identity(mod):
    import backend.ml.shift as sh
    sh.adjust_for_prior = lambda p, a, b: np.asarray(p, dtype=float)
    mod.shift = sh


# --------------------------------------------------------------------------- #
# label-leak — a label-free estimator handed the answers
# --------------------------------------------------------------------------- #
@mutation("label-leak", SHIFT, ["...and the true target prior is recorded for scoring only"],
          note="_safe_prior invents a prevalence from all-NaN labels")
def safe_prior_ignores_nan(mod):
    import backend.ml.shift as sh
    sh._safe_prior = lambda y: 0.02 if y is not None else float("nan")
    mod.shift = sh


@mutation("label-leak", SHIFT, ["pure label shift is nearly invisible to a domain"],
          note="a domain classifier that always reports shift, so the diagnosis is fixed")
def domain_auc_always_high(mod):
    import backend.ml.shift as sh
    orig = sh.covariate_weights
    sh.covariate_weights = lambda Xs, Xt, **k: (orig(Xs, Xt, **k)[0], 0.95)
    mod.shift = sh


# --------------------------------------------------------------------------- #
# floor-removal — a size guard disabled, leaving a guarantee on nothing
# --------------------------------------------------------------------------- #
@mutation("floor-removal", CONF, ["undersized cell falls back to marginal"],
          note="no fallback: a cell of 60 correlated hours fits its own tail quantile")
def no_small_cell_fallback(mod):
    import backend.ml.conformal as cf
    orig = cf._cell_band
    cf._cell_band = lambda st, y, p, g, a, fb, mh, mp, w=None: orig(st, y, p, g, a, None, mh, mp, w)
    mod.cf = cf


@mutation("floor-removal", MR, ["estimate_bytes matches the real output size"],
          note="a memory guard that lies; the 11.6 GB transform ships unnoticed")
def estimate_bytes_lies(mod):
    import backend.ml.models.minirocket as M
    M.CausalMiniRocket.estimate_bytes = lambda self, n: 1


# --------------------------------------------------------------------------- #
# fit-identity — the family that cost three false findings
# --------------------------------------------------------------------------- #
@mutation("fit-identity", CONF, ["save/load round-trips bands, spec and detector identity"],
          note="drop the detector stamp; bands become readable against any fit")
def conformal_forgets_detector(mod):
    import backend.ml.conformal as cf
    orig = cf.ConformalBands.to_json

    def patched(self):
        d = orig(self)
        d["detector"] = "unknown"
        return d
    cf.ConformalBands.to_json = patched
    mod.cf = cf


@mutation("fit-identity", OP, ["the detector whose scores these are is recorded"],
          note="the operating point forgets which model its threshold was measured on")
def operating_point_forgets_detector(mod):
    import backend.ml.operating_point as op
    orig = op.derive

    def patched(*a, **k):
        k["detector"] = "unknown"
        return orig(*a, **k)
    op.derive = patched
    mod.opmod = op


@mutation("fit-identity", OP, ["the in-range rule pulls it back"],
          note="disable the in-range rule: the threshold wanders into the sparse tail")
def operating_point_no_in_range(mod):
    import backend.ml.operating_point as op
    orig = op.derive

    def patched(*a, **k):
        k["require_in_range"] = False
        return orig(*a, **k)
    op.derive = patched
    mod.opmod = op


@mutation("fit-identity", OP, ["setting it on the upper bound breaches strictly less"],
          note="drop the bootstrap margin; the cap holds only on the patients that set it")
def operating_point_no_margin(mod):
    import backend.ml.operating_point as op
    orig = op.derive

    def patched(*a, **k):
        k["far_margin"] = False
        return orig(*a, **k)
    op.derive = patched
    mod.opmod = op


# --------------------------------------------------------------------------- #
# fit-identity, continued — the guard built BECAUSE of the three false findings
# is itself audited, because a provenance check that cannot fail is the worst
# possible thing to have: it would license exactly the comparisons it exists to stop.
# --------------------------------------------------------------------------- #
@mutation("fit-identity", FID,
          ["an absent fingerprint is refused, not assumed compatible"],
          note="treat a MISSING fingerprint as compatible — the permissive rule that "
               "would have waved through all three of the original mismatches. TARGET "
               "NOTE: this is NOT caught by the three-false-findings check, which "
               "compares real fingerprints and is therefore unaffected. The engine "
               "reported that as a survivor and the fault was the mutation spec naming "
               "the wrong defender — the second time the audit corrected its author, "
               "and the argument for naming a specific check rather than accepting "
               "'something failed'.")
def unknown_reads_as_compatible(mod):
    import backend.ml.fitid as F
    orig = F.require_same

    def lax(a, b, what="these artifacts", strict=True):
        if a is None or b is None:
            return
        return orig(a, b, what, strict)
    F.require_same = lax
    mod.require_same = lax


@mutation("fit-identity", FID,
          ["all three of this session's false findings are now refused"],
          note="a digest that ignores the spec, so dev-vs-setA compares as identical")
def digest_ignores_spec(mod):
    import backend.ml.fitid as F
    F.FIELDS = tuple(f for f in F.FIELDS if f != "spec")
    mod.fitid = F


@mutation("fit-identity", FID, ["identical fits produce identical digests"],
          note="a non-deterministic digest — provenance that cannot be compared at all")
def digest_is_random(mod):
    import backend.ml.fitid as F
    import uuid
    F.FitId.digest = lambda self: uuid.uuid4().hex[:12]
    mod.fitid = F


@mutation("fit-identity", FID, ["bands cut on the old fit are refused for the refit"],
          note="the configuration-only key the data lever exposed: the same roster refitted "
               "on four times the patients reads as the same detector, so the old fit's "
               "bands and threshold are accepted for the new one")
def key_ignores_fit(mod):
    import backend.ml.operating_point as O
    O.detector_key = (lambda roster, objective, fit=None, calibration="isotonic":
                      "%s:%s" % (objective, "+".join(roster)))


@mutation("fit-identity", FID, ["the same fit read through another calibrator is another detector"],
          note="a key blind to the calibrator: the centred and the isotonic map of one fit "
               "share a threshold and bands, though their scores differ")
def key_ignores_calibration(mod):
    import backend.ml.operating_point as O
    orig = O.detector_key
    O.detector_key = lambda roster, objective, fit=None, calibration="isotonic": orig(roster, objective, fit)


@mutation("fit-identity", FID, ["changes with one patient, the source, or the split seed"],
          note="a digest of the training set's SIZE, not its members: swapping patients "
               "for others is invisible")
def digest_counts_patients(mod):
    import hashlib
    import backend.ml.fitid as F
    F.training_digest = lambda g, source, seed: hashlib.sha256(
        f"{source}|{seed}|{len({str(x) for x in g})}".encode()).hexdigest()[:12]
    mod.fitid = F


@mutation("fit-identity", FID, ["serving flags a model file from another fit than its threshold"],
          note="serving that trusts whatever model file is on disk: a restored backup or "
               "an in-place retrain is read at another fit's cut points")
def serving_trusts_model_file(mod):
    import backend.aegis.core as C
    C.AegisCore._fit_mismatch = lambda self: None


# ═══════════════════════════════════════════════════════════════════════════ #
# EXTENSION — the remaining suites that share the check contract.
#
# Cost matters here in a way it did not for the ML suites: verify_subgroup takes
# ~130s per run and verify_alarm_policy ~35s, so each mutation against them costs
# that much again. They are tagged SLOW and excluded from the default audit; the
# fast eleven cover 392 additional checks for a handful of seconds each.
# ═══════════════════════════════════════════════════════════════════════════ #
COUP = "backend.physio.verify_coupling"
SELFIMP = "backend.ml.scripts.verify_selfimprove"
Z3S = "backend.aegis.verify_z3_safety"
AGENT = "backend.aegis.verify_agent"
DOCTOR = "backend.llm.verify_doctor"
SUBGRP = "backend.ml.scripts.verify_subgroup"
ALARM = "backend.ml.scripts.verify_alarm_policy"

#: Suites whose single run costs more than the whole rest of the audit combined.
SLOW_SUITES = {SUBGRP, ALARM}


# --------------------------------------------------------------------------- #
# naive-statistic — the coupling estimators, whose whole validity is a correction
# --------------------------------------------------------------------------- #
@mutation("naive-statistic", COUP, ["surrogate-corrected TE ~ 0 on independent signals"],
          note="drop the surrogate null: binned TE reads 0.025 bits on pure noise, so "
               "every pair of unrelated signals becomes a coupling finding")
def te_without_surrogate(mod):
    import backend.physio.coupling as C
    orig = C.transfer_entropy

    def raw(src, dst, bins=4, lag=1, n_surrogate=12, seed=0):
        r = orig(src, dst, bins, lag, n_surrogate, seed)
        return dict(r, te=r["te_raw"])            # uncorrected
    C.transfer_entropy = raw
    mod.cp = C


@mutation("naive-statistic", COUP, ["phase sync ~ 0 on independent signals"],
          note="skip the respiratory band-pass: Hilbert phase is only meaningful for a "
               "narrowband signal, and unfiltered this scored 0.716 on two independent "
               "random walks — higher than real coupled physiology")
def phase_sync_unfiltered(mod):
    import backend.physio.coupling as C
    from scipy.signal import hilbert
    import numpy as _np

    def ps(a, b, n=1, m=1, fs=C.HR_GRID_HZ, band=C.RESP_BAND):
        if len(a) < 16 or len(b) < 16:
            return 0.0
        k = min(len(a), len(b))
        pa = _np.angle(hilbert(_np.asarray(a[:k], float) - _np.mean(a[:k])))
        pb = _np.angle(hilbert(_np.asarray(b[:k], float) - _np.mean(b[:k])))
        return float(_np.abs(_np.mean(_np.exp(1j * (n * pa - m * pb)))))
    C.phase_sync = ps
    mod.cp = C


@mutation("causality", COUP, ["...while the decoupling control leaves the respiratory"],
          note="a decoupling control that also destroys the respiratory spectrum, so a "
               "monotone response would prove the estimator tracks amplitude, not timing")
def decouple_destroys_spectrum(mod):
    import backend.physio.coupling as C
    import numpy as _np
    C.decouple = lambda resp, alpha, seed=0: _np.asarray(resp, float) * (1.0 - alpha)
    mod.cp = C


# --------------------------------------------------------------------------- #
# floor-removal — the self-improvement acceptance rule, i.e. the refusal itself
# --------------------------------------------------------------------------- #
@mutation("floor-removal", SELFIMP, ["a significant but trivial gain is refused (min_gain)"],
          note="weaken the effect-size floor in the rule THE SUITE TESTS",
          targets="suite")
def selfimprove_local_no_min_gain(mod):
    orig = mod.apply_rule
    mod.apply_rule = lambda trials, alpha_c, min_gain: orig(trials, alpha_c, -1.0)


@mutation("floor-removal", SELFIMP, ["a large but noisy gain is refused (alpha)"],
          note="weaken the significance bar in the rule THE SUITE TESTS",
          targets="suite")
def selfimprove_local_no_alpha(mod):
    orig = mod.apply_rule
    mod.apply_rule = lambda trials, alpha_c, min_gain: orig(trials, 1.0, min_gain)


@mutation("fit-identity", SELFIMP, ["a significant but trivial gain is refused (min_gain)"],
          note="THE FINDING. Weaken the SHIPPED acceptance rule — RecursiveSelfImprover's "
               "own min_gain and alpha — and nothing notices, because verify_selfimprove "
               "never references RecursiveSelfImprover at all. It defines a local "
               "apply_rule() whose docstring says 'the shipped rule' and tests that. The "
               "0-of-16 refusal is the crown-jewel evidence for this project's refusal "
               "architecture, and it is verified against a re-implementation in the test "
               "file rather than against the code that produced it. This mutation is "
               "EXPECTED TO SURVIVE; the survival is the result.")
def selfimprove_shipped_rule_drifts(mod):
    import backend.ml.selfimprove as S
    imp = getattr(S, "RecursiveSelfImprover", None)
    if imp is None:
        raise RuntimeError("no RecursiveSelfImprover to weaken")
    orig_init = imp.__init__

    def loose(self, *a, **k):
        orig_init(self, *a, **k)
        self.min_gain = -1.0                       # accept any delta
        self.alpha = 1.0                           # accept any p-value
    imp.__init__ = loose
    mod.selfimprove = S


# --------------------------------------------------------------------------- #
# concurrency — the Z3 safety cage. A lock that is not held is not a cage.
# --------------------------------------------------------------------------- #
@mutation("concurrency", Z3S, ["no public function builds Z3 terms without the lock"],
          note="strip the guard from a public prover entry point; Z3 contexts are not "
               "thread-safe and a corrupted one previously returned garbage rather "
               "than raising")
def z3_unguarded_entrypoint(mod):
    src = getattr(mod, "_unguarded_public", None)
    if src is None:
        raise RuntimeError("suite does not expose _unguarded_public")
    mod._unguarded_public = lambda path: ["verify_protocol"]   # report a real breach


@mutation("concurrency", Z3S, ["no public function builds Z3 terms without the lock"],
          note="report a SECOND unguarded entry point. The earlier version of this "
               "mutation swapped ProverUnavailable for RuntimeError, which broke the "
               "suite's own exception handling and produced an ERROR rather than a "
               "FAIL — correctly scored as a survivor, since a crash is not evidence "
               "that a claim is defended.")
def z3_second_unguarded_entrypoint(mod):
    if not hasattr(mod, "_unguarded_public"):
        raise RuntimeError("suite does not expose _unguarded_public")
    mod._unguarded_public = lambda path: ["safety_model", "verify_protocol"]


# --------------------------------------------------------------------------- #
# confinement — the agent plane's read-only boundary
# --------------------------------------------------------------------------- #
@mutation("label-leak", AGENT, ["gets only"],
          note="index a POST-only route as readable. The read plane's guarantee is that "
               "every route reachable through os_read is a GET; listing a write route "
               "there is exactly how a read tool acquires a mutation.")
def agent_write_route_indexed_as_read(mod):
    import backend.aegis.server as srv
    rw = getattr(srv, "_READ_ROUTES", None)
    ww = getattr(srv, "_WRITE_ROUTES", None)
    if not isinstance(rw, dict) or not ww:
        raise RuntimeError("route indexes not available")
    victim = next((p for p in ww if p not in rw), None)
    if victim is None:
        raise RuntimeError("no write-only route to smuggle in")
    rw[victim] = ww[victim]
    mod.srv = srv


# ═══════════════════════════════════════════════════════════════════════════ #
# THE KERNEL (AegisOS 1.0). Each mutation below is a failure this OS either
# actually shipped (process-global writer ownership refused the boot warm-up's own
# writes for a whole phase; the unlocked journal verify reported concurrent
# appends as tampering) or is the precise way its central claims would be false:
# a gate that lets everything through, a sign-out that does not end the session,
# an emergency elevation that lets a nurse prescribe, a prover that is always
# green, a process table that accepts a kill it never delivers.
# ═══════════════════════════════════════════════════════════════════════════ #
K_GATE = "backend.aegis.kernel.verify_gate"
K_USERS = "backend.aegis.kernel.verify_users"
K_POLICY = "backend.aegis.kernel.verify_policy"
K_JOURNAL = "backend.aegis.kernel.verify_journal"
K_PROC = "backend.aegis.kernel.verify_proc"
K_EXEC = "backend.aegis.kernel.verify_executor"
K_BOOT = "backend.aegis.kernel.verify_boot"


@mutation("access-control", K_GATE, ["a guest write is refused with 403 EPERM",
                                     "the refused handler never ran"],
          note="the reference monitor decides 'allow' for every call")
def gate_allows_everything(mod):
    import backend.aegis.kernel.gate as G

    def decide(self, cred, entry):
        return G.Decision(True, True, entry, cred)
    G.Gate.decide = decide


@mutation("access-control", K_USERS, ["after sign-out the same token no longer works"],
          note="sign-out drops the session but forgets the revocation: the still-valid "
               "signature re-materialises it on the next call, as if after a restart")
def logout_forgets_revocation(mod):
    import backend.aegis.kernel.users as U
    orig = U.UserDB._end

    def _end(self, sid, reason):
        out = orig(self, sid, reason)
        self.revoked.discard(sid)
        return out
    U.UserDB._end = _end


@mutation("access-control", K_USERS, ["still cannot prescribe, change policy or install software"],
          note="break-glass adds order:treat — an emergency lets a nurse prescribe")
def breakglass_confers_prescribing(mod):
    import backend.aegis.kernel.users as U
    U.elevated_caps = lambda role: frozenset(U.GRANTS.get(role, frozenset())
                                             | U.BREAKGLASS_CAPS | {"order:treat"})


@mutation("access-control", K_POLICY, ["a guest granted pkg:install is caught"],
          note="the prover reports every property proven whatever the tables say")
def policy_proof_always_green(mod):
    import backend.aegis.kernel.policy as P
    orig = P.prove_kernel_policy

    def fake(*a, **k):
        r = orig(*a, **k)
        for prop in r["properties"]:
            prop["proven"] = True
            prop.pop("counterexample", None)
        r["sound"] = True
        return r
    P.prove_kernel_policy = fake


@mutation("tamper-evidence", K_JOURNAL, ["an altered entry breaks the chain"],
          note="verify() reports the window intact without recomputing a hash")
def journal_verify_blind(mod):
    import backend.aegis.kernel.journal as J

    def verify(self):
        ents, head = self._snapshot()
        return {"ok": True, "window": len(ents),
                "from_seq": ents[0].seq if ents else None,
                "to_seq": ents[-1].seq if ents else None}
    J.Journal.verify = verify


@mutation("lifecycle", K_PROC, ["a patient's monitor refuses SIGSTOP"],
          note="every process claims to honour every signal — a kill accepted and "
               "never delivered")
def signals_never_refused(mod):
    import backend.aegis.kernel.proc as P
    P.Process.supported_signals = lambda self: P.SIGNALS


@mutation("concurrency", K_EXEC, ["a private arena is still writable from another thread"],
          note="the shipped Phase 0 design: one writer thread for the whole process, so "
               "the boot warm-up's writes into its OWN private cluster were refused")
def ownership_process_global(mod):
    import threading
    import backend.aegis.kernel.executor as E
    import backend.aegis.kernel.memfabric as M
    bound = {"ident": None}
    orig_own = E.KernelExecutor.own
    orig_push = M.LocklessRing.push

    def own(self, arena):
        bound["ident"] = self._ident
        return orig_own(self, arena)

    def push(self, t_us, row):
        if bound["ident"] is not None and threading.get_ident() != bound["ident"]:
            E.refuse_write("ring write (%s)" % self.name)
        saved, self._arena.writer_ident = self._arena.writer_ident, None
        try:
            return orig_push(self, t_us, row)
        finally:
            self._arena.writer_ident = saved
    E.KernelExecutor.own = own
    M.LocklessRing.push = push


@mutation("lifecycle", K_BOOT, ["a failing phase propagates"],
          note="the boot recorder notes a failure and swallows it — a broken deploy "
               "becomes a silently degraded one")
def boot_swallows_failures(mod):
    import contextlib
    import time as _time
    import backend.aegis.kernel.boot as B

    @contextlib.contextmanager
    def phase(self, name, summary=""):
        p = B.Phase(name=name, summary=summary, started_at=_time.time())
        self.phases.append(p)
        try:
            yield p
        except BaseException as exc:                 # recorded, and swallowed
            p.ok, p.error = False, str(exc)
    B.BootSequencer.phase = phase


K_VFS = "backend.aegis.kernel.verify_vfs"
K_STREAM = "backend.aegis.kernel.verify_osstream"


@mutation("access-control", K_VFS, ["a guest cannot read a view:audit file"],
          note="the filesystem forgets to check a node's read capability")
def vfs_read_ignores_capability(mod):
    import backend.aegis.kernel.vfs as V

    class Everyone:
        def __init__(self, cred):
            self.__dict__.update(cred.__dict__)

        def has(self, cap):
            return True
    orig = V.VFS.read

    def read(self, cred, path, cwd="/"):
        return orig(self, Everyone(cred), path, cwd)
    V.VFS.read = read


@mutation("access-control", K_VFS, ["another user cannot read it"],
          note="a home directory that does not check its owner")
def home_ignores_owner(mod):
    import backend.aegis.kernel.vfs as V
    V.HomeFS._check_owner = lambda self, user, cred, write: None


@mutation("policy", K_VFS, ["a self-contradictory policy is refused"],
          note="the consistency check always reports consistent — a policy no order "
               "could satisfy becomes the law")
def policy_consistency_skipped(mod):
    import backend.aegis.kernel.policyfile as P
    P.PolicyStore.consistency = lambda self, rules: {"consistent": True, "profiles_checked": 0}


@mutation("policy", K_VFS, ["a Ciprofloxacin ban does NOT catch Levofloxacin"],
          note="the browser's old codebook translation: Levofloxacin proven as "
               "Ciprofloxacin, so a rule about one drug silently governs another")
def levofloxacin_substituted(mod):
    import backend.aegis.kernel.policyfile as P
    from backend.aegis import clinical_dsl as dsl
    P.CAGE_TO_DSL[5] = dsl.ENUMS["antibiotic"]["Ciprofloxacin"]


@mutation("policy", K_VFS, ["a policy forbidding the bundle's own drug makes it NOT all_verified"],
          note="the bundle carries the policy certificate but all_verified ignores it — "
               "a file that is shown and not enforced")
def bundle_ignores_policy(mod):
    import backend.aegis.kernel.bundle as B
    orig = B.SepsisBundleEngine.generate

    def generate(self, prof, hist=None):
        out = orig(self, prof, hist)
        out["all_verified"] = bool(out["alert"]["certified"] and out["protocol"]["certified"]
                                   and out["note"]["certified"])
        return out
    B.SepsisBundleEngine.generate = generate


@mutation("access-control", K_STREAM, ["a guest's journal feed excludes auth, syscall and clinical"],
          note="the stream sends every facility to every viewer")
def stream_journal_unfiltered(mod):
    import backend.aegis.kernel.osstream as O
    O.SENSITIVE_FACILITIES = frozenset()


# --------------------------------------------------------------------------- #
# The integration plane (backend/aegis/interop). Each of these was a real defect found
# while building it, or the plane's central safety claim: no proposer, human or model,
# decides on its own what the detectors read.
# --------------------------------------------------------------------------- #
INTEROP = "backend.aegis.interop.verify_interop"


@mutation("integration", INTEROP, ["ICU and live rooms are still never free"],
          note="occupancy read from the monitored-bed table alone: at boot, before the ward has ticked, an HL7 "
               "admission landed in ICU replay room 312 (found running it)")
def icu_rooms_free_at_boot(mod):
    import backend.aegis.paging.engine as P

    def free(self, room):
        r = P.ROOMS.get(room)
        if r is None or P.ZONES[r.zone].kind in ("or", "imaging", "support"):
            return False
        return room not in self.synthetic and room not in self.monitored and room not in self.interfaced
    P.PagingOS.interface_room_free = free


@mutation("integration", INTEROP, ["a withheld haemoglobin is left out of the Hct/Hgb identity"],
          note="identities read a partner's WITHHELD values: a haemoglobin the partner no longer vouched for "
               "convicted the haematocrit beside it (a false quarantine, found running the drills)")
def withheld_partner_convicts(mod):
    import backend.aegis.interop.engine as En
    orig = En.Integration._joint_rows

    def rows(self, m, concept, conv, adopted_only=False, since_seq=0):
        saved = [(g, g["held"]) for g in self.groups.get(m.interface, ())]
        for g, _h in saved:
            g["held"] = set()
        try:
            return orig(self, m, concept, conv, adopted_only, since_seq)
        finally:
            for g, h in saved:
                g["held"] = h
    En.Integration._joint_rows = rows


@mutation("integration", INTEROP, ["passes the feed-relative test"],
          note="live evidence judged against the GLOBAL reference with the population band (the rejected "
               "design): a ward of septic patients cuts its own heart-rate feed when it matters most")
def drift_against_global_reference(mod):
    import backend.aegis.interop.evidence as Ev
    from backend.aegis.interop.units import IDENTITY
    Ev.drift_checks = lambda c, recent, window, baseline: list(Ev.assess(c, list(recent), IDENTITY).checks)


@mutation("integration", INTEROP, ["beats the ÷100 decade guess"],
          note="decade guesses compete with named units: a creatinine in µmol/L read as ambiguous (÷88.42 vs "
               "÷100), so a relabelled feed was quarantined with no fix (found running the drills)")
def decade_guess_competes(mod):
    import backend.aegis.interop.vocab as Vo
    from backend.aegis.interop.units import Conversion
    orig = Vo.Concept.candidate_conversions

    def cands(self):
        return [Conversion(c.factor, c.offset, c.label.replace("decade shift ", "")) for c in orig(self)]
    Vo.Concept.candidate_conversions = cands


@mutation("integration", INTEROP, ["the model's WRONG answer in the right unit"],
          note="a proposal adopted on its proposer's say-so: the language model decides what every detector "
               "reads, and its wrong analyte flows into rooms")
def proposals_adopt_unchecked(mod):
    import backend.aegis.interop.evidence as Ev
    orig = Ev.infer

    def infer(c, values, declared, ref=None, n_min=Ev.N_MIN):
        inf = orig(c, values, declared, ref, n_min)
        if inf.decision != "declared" and inf.ranked:
            top = inf.ranked[0]
            return Ev.Inference("declared", top, top, [top], inf.ranked)
        return inf
    Ev.infer = infer


@mutation("integration", INTEROP, ["is not processed twice"],
          note="no idempotence: an interface engine resending after a lost ACK duplicates every result")
def resends_reprocessed(mod):
    import backend.aegis.interop.engine as En
    orig_init = En.Integration.__init__

    class Forgetful(En.OrderedDict):
        def get(self, k, d=None):
            return None

    def init(self, *a, **kw):
        orig_init(self, *a, **kw)
        self._acks = Forgetful()
    En.Integration.__init__ = init


@mutation("integration", INTEROP, ["drill glucose_units: GLU quarantined"],
          note="impossible values withheld one at a time but the feed never judged: it stays 'adopted' while "
               "sending nothing usable, and nobody is told")
def never_quarantine(mod):
    import backend.aegis.interop.engine as En
    En.Integration._quarantine = lambda self, group, now, effects, cause, checks, since_t=None: None


# the post-hoc fixes the registered interop benchmark motivated (docs/INTEROP.md)
@mutation("integration", INTEROP, ["cannot be the proposed concept refuses it"],
          note="a written unit that cannot be the concept treated as no unit: heart-rate values written '/min' "
               "adopt as mean arterial pressure (the defect the benchmark's smoke run found; H5 6.9% unsafe)")
def unit_mismatch_as_missing(mod):
    import backend.aegis.interop.vocab as Vo
    from backend.aegis.interop.units import norm
    orig = Vo.resolve_unit

    def resolve(c, written):
        tok, conv = orig(c, written)
        return ("", None) if (tok and conv is None) else (tok, conv)
    Vo.resolve_unit = resolve
    import backend.aegis.interop.engine as En
    En.resolve_unit = resolve


@mutation("integration", INTEROP, ["a feed with no unit waits for a person"],
          note="a feed that writes no unit adopted on whatever fits: potassium treated as unitless by convention")
def unitless_auto_adopts(mod):
    import backend.aegis.interop.vocab as Vo
    Vo.UNITLESS_BY_CONVENTION = ("ph", "inr", "potassium")


@mutation("integration", INTEROP, ["moving back INTO the population band"],
          note="the jump test without the population condition: a dozen high early bilirubins made the baseline, "
               "and the feed returning to normal was quarantined (registered run, H4)")
def jump_ignores_population(mod):
    import backend.aegis.interop.evidence as Ev
    Ev.references = lambda: {}


@mutation("integration", INTEROP, ["CUSUM waits"],
          note="live identities judged by overlapping binomial windows again: chance bursts at the clean rate "
               "quarantine blood pressure (registered run, H4: hospital A 2.71 per 100k)")
def cusum_is_windowed(mod):
    import backend.aegis.interop.evidence as Ev
    Ev.cusum_identity = lambda inv, samples, h=13.0: Ev.check_invariant(inv, list(samples)[-12:])


@mutation("integration", INTEROP, ["withholds exactly those two"],
          note="no swap explanation: an identity failure withholds every field in it, and a partial quarantine "
               "blinds Henderson-Hasselbalch (registered run: 6,821 PaCO2 values delivered)")
def swap_unexplained(mod):
    import backend.aegis.interop.engine as En
    En.Integration._explain_swap = lambda self, m, failing: None


@mutation("integration", INTEROP, ["mg/dL values labelled mEq/L (x1.22) are ambiguous"],
          note="a written unit confirmed even when another unit the analyte comes in fits the same values: "
               "magnesium mg/dL labelled mEq/L adopted (registered run, H5: 3.4% unsafe)")
def units_never_ambiguous(mod):
    import backend.aegis.interop.evidence as Ev
    orig = Ev.infer

    def infer(c, values, declared, ref=None, n_min=Ev.N_MIN):
        inf = orig(c, values, declared, ref, n_min)
        if inf.decision == "ambiguous" and inf.declared is not None and inf.declared.status == "pass":
            return Ev.Inference("declared", inf.declared, inf.declared, inf.passing, inf.ranked)
        return inf
    Ev.infer = infer


@mutation("integration", INTEROP, ["base excess (linear, centred on zero) is confirmed"],
          note="no spread check on linear scales: base excess read x0.001 keeps its median at zero, so it "
               "rivals the right reading and a correct feed is held as ambiguous")
def spread_unchecked(mod):
    import backend.aegis.interop.evidence as Ev
    orig = Ev.assess

    def assess(c, values, conv, ref=None, n_min=Ev.N_MIN):
        f = orig(c, values, conv, ref, n_min)
        if any(ch.name == "spread" and ch.status == "fail" for ch in f.checks):
            f.checks = [ch for ch in f.checks if ch.name != "spread"]
            if all(ch.status in ("pass", "skip") for ch in f.checks) and f.n >= n_min:
                f.status = "pass"
        return f
    Ev.assess = assess


@mutation("integration", INTEROP, ["ONE card at the higher severity"],
          note="a card per knowledge-layer report: DDInter (moderate) and the curated QT layer (major) each "
               "carded the same levofloxacin + azithromycin pair, two severities for one hazard (found live in "
               "the CDS Hooks try-out)")
def cds_card_per_source(mod):
    import itertools
    import backend.aegis.interop.cdshooks as C
    n = itertools.count()
    C.frozenset = lambda xs: frozenset([*xs, f"#{next(n)}"])      # no two reports ever share a group


@mutation("integration", INTEROP, ["no room still lists creatinine as withheld"],
          note="the fix delivered the withheld values but the room kept listing them as withheld (found live: "
               "the Feeds panel still said creatinine was withheld after the chief accepted the fix)")
def release_keeps_withheld(mod):
    import backend.aegis.rooms.monitor as Mo
    Mo.RoomMonitor.released_interface = lambda self, room, concept, interface: None


@mutation("integration", INTEROP, ["a failed model call (quota, timeout) is no answer"],
          note="a failed model call recorded as the model declining: every code in the batch left the review "
               "queue as 'unmapped' (found while tracing why B-Hb was unmapped in one boot and adopted in the next)")
def model_failure_is_decline(mod):
    import backend.aegis.interop.engine as En
    orig = En.ask_model

    def ask(llm, items, max_batch=12, outcomes=None):
        props, model_id = orig(llm, items, max_batch, outcomes)
        if outcomes is not None:
            for it in items:
                if it["code"] not in props:
                    outcomes.setdefault(it["code"], "declined")
        return props, model_id
    En.ask_model = ask
