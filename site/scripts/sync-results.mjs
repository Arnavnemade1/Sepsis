#!/usr/bin/env node
/**
 * artifacts/*.json  →  site/data/results.json
 *
 * WHY THIS IS NOT A `prebuild` HOOK
 * ---------------------------------
 * Vercel's Root Directory sandbox is explicit: a project cannot read files
 * outside its root, and `..` is not allowed. With Root Directory = site/, the
 * path ../artifacts simply does not exist at build time. So this is an explicit
 * LOCAL task, and site/data/results.json is COMMITTED — that file is the build's
 * only data input. The script exits 0 with a warning when ../artifacts is
 * unreachable, so it is harmless if it ever runs in CI.
 *
 * A CI job re-runs this and fails on a dirty `git diff site/data/results.json`,
 * which makes stale numbers a red check rather than a quiet lie on the research
 * page.
 *
 * Every number that reaches the website passes through here, so this file is
 * also the provenance record: each block below names the artifact it came from.
 *
 * Run: npm run sync:results
 */
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const HERE = dirname(fileURLToPath(import.meta.url));
const ART = resolve(HERE, "../../artifacts");
const OUT = resolve(HERE, "../data/results.json");

if (!existsSync(ART)) {
  console.warn(
    "[sync-results] ../artifacts not reachable — keeping the committed data/results.json",
  );
  process.exit(0);
}

const read = (name) => JSON.parse(readFileSync(join(ART, `${name}.json`), "utf8"));

/** Round to `n` decimals. Full float precision is what makes rigorous_eval.json
 *  107 KB; 4-5 dp is more than the charts or the tables can resolve. */
const r = (v, n = 4) =>
  typeof v === "number" && Number.isFinite(v) ? Number(v.toFixed(n)) : v;
const arr = (a, n = 4) => (a ?? []).map((v) => r(v, n));

/** Evenly downsample a monotone curve, always keeping both endpoints.
 *  These curves are drawn a few hundred CSS pixels wide, so points beyond that
 *  are bytes in the page bundle that no one can see. Endpoints are kept because
 *  an ROC that does not reach (1,1) reads as a broken chart, not a sparse one. */
const thin = (a, keep) => {
  if (!a || a.length <= keep) return a ?? [];
  const out = [];
  for (let i = 0; i < keep; i++) out.push(a[Math.round((i * (a.length - 1)) / (keep - 1))]);
  return out;
};

// ─── rigorous_eval.json — the judge-facing cross-hospital evaluation ─────────
const rig = read("rigorous_eval");
const sub = read("subgroup_audit");
const scale = read("scale");
const transfer = read("transfer");
const transferStack = read("transfer_stack");
const conf = read("conformal");
const confExt = read("conformal_external");
const fals = read("falsification");

const model = (m) => ({
  auroc: r(m.auroc),
  aurocCi: arr(m.auroc_ci),
  auprc: r(m.auprc),
  auprcCi: arr(m.auprc_ci),
  brier: r(m.brier, 5),
  ece: r(m.ece, 5),
  prevalence: r(m.prevalence, 5),
  utility: r(m.normalized_utility),
  // Sensitivity at fixed specificity — the clinically legible operating points.
  // NOTE the source stores an OBJECT per level ({sensitivity, specificity,
  // threshold}), not a bare number. Assigning it straight through silently
  // produced NaN everywhere it was formatted as a percentage.
  //
  // `achieved` is kept because the requested specificity is a floor the sweep
  // aims at, not one it always hits — the 0.85 level actually lands near 0.81 —
  // and a page that prints "at 85% specificity" should be able to check itself.
  sensAtSpec: Object.fromEntries(
    ["0.85", "0.90", "0.95"].map((lvl) => {
      const op = m.operating_points?.[`sens@spec>=${lvl}`];
      return [
        lvl,
        {
          sensitivity: r(op?.sensitivity),
          achieved: r(op?.specificity),
          threshold: r(op?.threshold, 5),
        },
      ];
    }),
  ),
  earlyWarning: {
    detectionRate: r(m.early_warning?.detection_rate),
    medianLeadHours: r(m.early_warning?.median_lead_time_hours, 1),
    meanLeadHours: r(m.early_warning?.mean_lead_time_hours, 1),
    falseAlarmRate: r(m.early_warning?.false_alarm_rate),
    septicPatients: m.early_warning?.septic_patients,
    nonsepticPatients: m.early_warning?.nonseptic_patients,
  },
  // Curves are downsampled to what a chart this size can actually render. The
  // ROC/PR pairs across two directions and several models were 22 KB of the
  // bundle — more than every headline number on the site combined.
  // 88 -> 64 points. These curves are drawn a few hundred CSS pixels wide, so the
  // difference is invisible; what it buys is headroom under the 60 KB refusal ceiling
  // this script enforces on itself. The four (direction x model) cells were 18 KB of
  // a 32 KB payload, and pretty-printing nearly doubles that on disk.
  roc: { fpr: thin(arr(m.roc?.fpr), 64), tpr: thin(arr(m.roc?.tpr), 64) },
  pr: { recall: thin(arr(m.pr?.recall), 64), precision: thin(arr(m.pr?.precision, 5), 64) },
  reliability: {
    meanPred: arr(m.reliability?.mean_pred, 5),
    obsFreq: arr(m.reliability?.obs_freq, 5),
    count: m.reliability?.count ?? [],
  },
  decisionCurve: {
    thresholds: arr(m.decision_curve?.thresholds, 5),
    model: arr(m.decision_curve?.model, 5),
    treatAll: arr(m.decision_curve?.treat_all, 5),
    treatNone: arr(m.decision_curve?.treat_none, 5),
  },
  alarmBurden: {
    thresholds: arr(m.alarm_burden?.thresholds, 5),
    detectionRate: arr(m.alarm_burden?.detection_rate),
    falseAlarmRate: arr(m.alarm_burden?.false_alarm_rate),
    falseAlertsPer100h: arr(m.alarm_burden?.false_alerts_per_100h, 2),
  },
});

const direction = (d) => ({
  train: d.train,
  test: d.test,
  models: { gbm: model(d.models.gbm), stacked: model(d.models.stacked) },
});

// ─── the pooled headline, computed the same way backend/aegis/system_health.py
//     does it, so the site and GET /aegis/health/system cannot disagree ───────
const dirs = ["A_to_B", "B_to_A"];
const mean = (xs) => xs.reduce((a, b) => a + b, 0) / xs.length;
const pooled = (which) => {
  const ms = dirs.map((k) => rig.directions[k].models[which]);
  return {
    auroc: r(mean(ms.map((m) => m.auroc))),
    ece: r(mean(ms.map((m) => m.ece)), 5),
    utility: r(mean(ms.map((m) => m.normalized_utility))),
    detectionRate: r(mean(ms.map((m) => m.early_warning.detection_rate))),
    falseAlarmRate: r(mean(ms.map((m) => m.early_warning.false_alarm_rate))),
    medianLeadHours: r(mean(ms.map((m) => m.early_warning.median_lead_time_hours)), 1),
    // how much the two directions disagree — small means it transfers symmetrically
    spread: r(Math.abs(ms[0].auroc - ms[1].auroc)),
  };
};

// ─── compare_wearable.json — the INTERNAL split, for the generalization gap ──
const cmp = read("compare_wearable");

// ─── operating_point.json — the signature finding ────────────────────────────
const op = read("operating_point");

// ─── alarm_policy.json — the search that refused to promote ──────────────────
const alarm = read("alarm_policy");

// ─── self_improve_ledger.json — 16 trials, 0 accepted ────────────────────────
const si = read("self_improve_ledger");

// ─── qhdc_ablation.json — the honest quantum negative result ─────────────────
const q = read("qhdc_ablation");
const pair = (p) => ({ mean: r(p?.[0]), sd: r(p?.[1]) });

// ─── external_validation_wearable.json — the earlier Phase 7 protocol ────────
const ext = read("external_validation_wearable");

// ─── frontier_summary.json — the accuracy-frontier push ──────────────────────
// WITHIN-SITE cross-validation, not cross-hospital. It is kept in its own block,
// and labelled as such everywhere it surfaces, because the headline above is the
// cross-hospital number and silently mixing protocols is exactly the move this
// page exists to refuse. The production detector changed AFTER rigorous_eval.json
// was generated, so the two describe different models — also stated on the page.
const fr = existsSync(join(ART, "frontier_summary.json")) ? read("frontier_summary") : null;

const out = {
  // provenance travels with the numbers — the research page prints it
  generatedFrom: "artifacts/*.json",
  syncedAt: new Date().toISOString().slice(0, 10),

  meta: {
    protocol: rig.meta.protocol,
    featureSet: rig.meta.feature_set,
    bootstrap: rig.meta.boot,
    generated: rig.meta.generated,
    setA: {
      rows: rig.meta.setA.rows,
      patients: rig.meta.setA.patients,
      prevalence: r(rig.meta.setA.prevalence, 5),
    },
    setB: {
      rows: rig.meta.setB.rows,
      patients: rig.meta.setB.patients,
      prevalence: r(rig.meta.setB.prevalence, 5),
    },
  },

  directions: { AtoB: direction(rig.directions.A_to_B), BtoA: direction(rig.directions.B_to_A) },

  headline: {
    production: pooled("stacked"),
    baseline: pooled("gbm"),
    delta: r(pooled("stacked").auroc - pooled("gbm").auroc),
  },

  /** internal (same-site test split) vs external (other hospital), same model,
   *  same features. The difference IS the deployment honesty number.
   *
   *  DISCRIMINATION ONLY — deliberately no ECE here. compare_wearable.json's
   *  internal ECE (0.382) is the UNCALIBRATED model; putting it beside the
   *  calibrated cross-site ECE would imply calibration improves off-site, which
   *  is nonsense. The calibration story is its own block below. */
  generalization: {
    internalAuroc: r(cmp.gbm.auroc),
    externalAuroc: pooled("gbm").auroc,
    gap: r(cmp.gbm.auroc - pooled("gbm").auroc),
    internalNote: "single-site held-out test split, 40,335 patients",
    externalNote: "trained on one hospital, tested on the other, 6,000 + 6,000 patients",
  },

  /** Two facts, from two committed artifacts, kept apart on purpose.
   *
   *  The README also quotes a calibrated INTERNAL ECE of 0.0008, but that number
   *  lives only in artifacts/gbm_full_calibrated.log, which .gitignore excludes
   *  (`artifacts/*.log`). It is therefore not sourceable from a fresh clone and
   *  is not charted here — the site only prints numbers it can re-derive. */
  calibration: {
    uncalibratedEce: r(cmp.gbm.ece),
    uncalibratedNote: "GBM before post-hoc calibration, internal split (compare_wearable.json)",
    calibratedEce: pooled("stacked").ece,
    calibratedNote: "production model, cross-hospital, mean of both directions (rigorous_eval.json)",
    foldReduction: Math.round(cmp.gbm.ece / pooled("stacked").ece),
    claim:
      "Post-hoc calibration is what makes the probability mean what it says — and it survives the move to a different hospital.",
  },

  operatingPoint: {
    high: op.high,
    medium: op.medium,
    objective: op.objective,
    maxFalseAlarmRate: op.max_false_alarm_rate,
    oof: {
      utility: r(op.oof_utility),
      detectionRate: r(op.oof_detection_rate),
      falseAlarmRate: r(op.oof_false_alarm_rate),
      medianLeadHours: r(op.oof_median_lead_hours, 1),
    },
    probP95: r(op.prob_p95, 5),
    probP99: r(op.prob_p99, 5),
    nPatients: op.n_patients,
    /** Held-out gate: patients never used to derive the threshold. AUROC is
     *  IDENTICAL across all three rows — the entire difference is where the
     *  decision boundary was placed. From backend/ml/README.md.
     *
     *  FROZEN. These rows are a historical study of ONE detector, so
     *  `derivedThreshold` is recorded here rather than read from `op.high` at
     *  render time. They were the same number until the detector changed and the
     *  operating point was re-derived (0.0725 -> 0.055); a chart that looked the
     *  row up by the live value then found nothing and crashed the page. A frozen
     *  table must carry its own key. */
    heldOut: {
      derivedThreshold: 0.0725,
      auroc: 0.7334,
      rows: [
        { label: "shipped hand-set", threshold: 0.12, utility: 0.2946, detection: 0.352, far: 0.093, lead: 50 },
        { label: "derived, FAR ≤ 15%", threshold: 0.0725, utility: 0.3356, detection: 0.451, far: 0.192, lead: 52 },
        { label: "utility-optimal", threshold: 0.025, utility: 0.367, detection: 0.527, far: 0.348, lead: 40 },
      ],
      shippedPercentile: 98.3,
      fracAboveShipped: 0.0172,
    },
  },

  alarmPolicy: {
    shipped: alarm.shipped_policy.describe,
    challenger: alarm.verdict.challenger,
    promoted: alarm.verdict.promoted,
    reason: alarm.verdict.reason,
    deltaDetection: r(alarm.verdict.delta_detection),
    ci: [r(alarm.verdict.ci_low), r(alarm.verdict.ci_high)],
    nBoot: alarm.verdict.n_boot,
    familiesSearched: ["plain threshold", "K consecutive hours", "leaky exceedance integrator", "per-patient z-score"],
  },

  selfImprove: {
    baselineAuroc: r(si.baseline_search_auroc),
    championAuroc: r(si.champion_search_auroc),
    promoted: si.promoted,
    verdict: si.verdict,
    nTrials: si.trials.length,
    nAccepted: si.trials.filter((t) => t.accepted).length,
    // the instructive rejections: biggest apparent gain, and the near-misses
    trials: si.trials
      .map((t) => ({
        id: t.trial_id,
        round: t.round_idx,
        auroc: r(t.mean_auroc),
        delta: r(t.delta_vs_champion),
        p: r(t.p_value, 4),
        accepted: t.accepted,
        reason: t.reason,
      }))
      .sort((a, b) => b.delta - a.delta),
  },

  qhdc: {
    ablation: Object.fromEntries(
      Object.entries(q.ablation).map(([name, m]) => [
        name,
        { auroc: pair(m.auroc), auprc: pair(m.auprc), utility: pair(m.normalized_utility) },
      ]),
    ),
    sweep: Object.entries(q.sweep).map(([qubits, s]) => ({
      qubits: Number(qubits),
      dim: s.dim,
      auroc: pair(s.auroc),
      seconds: r(s.sec, 1),
    })),
  },

  /** The accuracy-frontier push: what the detector's training objective was aimed
   *  at, why re-aiming the whole ensemble did nothing, and the one member that
   *  earned its place. Null until `python -m backend.ml.scripts.frontier_summary`
   *  has been run, so the page must render without it. */
  frontier: fr && {
    protocol: fr.protocol,
    rows: fr.rows,
    patients: fr.patients,
    folds: fr.folds,
    bootstrap: fr.bootstrap,
    baseline: fr.baseline,
    production: fr.production,
    crossHospitalConfirmed: fr.crossHospitalConfirmed,
    // Only the fields the page prints. data/results.json is imported into the page
    // bundle and is already near its size ceiling, so passing a summary through
    // wholesale costs every visitor bytes for numbers nobody renders.
    candidates: Object.fromEntries(
      Object.entries(fr.candidates).map(([k, c]) => [
        k,
        {
          auroc: c.auroc,
          aurocSd: c.aurocSd,
          auprc: c.auprc,
          falseAlarm: c.false_alarm,
          detectAtFar15: c.detect_at_far15,
        },
      ]),
    ),
    comparisons: Object.fromEntries(
      Object.entries(fr.comparisons).map(([k, c]) => [
        k,
        {
          deltaAuroc: c.deltaAuroc,
          ci: c.ci,
          significant: c.significant,
          rankCorr: c.rankCorr,
          // The budget sweep is the evidence that a one-threshold win was an
          // artifact, so it is kept only for the candidate that produced it.
          detectionAtBudget:
            k === "stack-utility"
              ? Object.fromEntries(
                  Object.entries(c.detectionAtBudget ?? {}).map(([b, v]) => [
                    b,
                    { delta: v.delta_detection, ci: [v.ci_lo, v.ci_hi], significant: v.significant },
                  ]),
                )
              : undefined,
        },
      ]),
    ),
    promotion: {
      created: fr.promotion?.created,
      gates: fr.promotion?.gates,
      promoted: fr.promotion?.promoted,
      holdoutAurocDelta: fr.promotion?.holdoutAurocDelta,
      holdoutAuprcDelta: fr.promotion?.holdoutAuprcDelta,
    },
  },

  /** Subgroup fairness + calibration audit — who the shipped detector serves worse.
   *  Only the cells and the gaps that carry a claim; the full artifact (every cell's
   *  interval, every gap's adjusted CI) stays in artifacts/subgroup_audit.json. */
  subgroup: {
    threshold: sub.threshold.value,
    thresholdPolicy: sub.threshold.policy,
    patients: sub.cohort.patients,
    familySize: sub.multiplicity.family_size,
    alphaAdjusted: sub.multiplicity.alpha_adjusted,
    worstCell: sub.worst_cell,
    cells: sub.cells
      .filter((c) => c.kind !== "age\u00d7sex" || c.name === sub.worst_cell)
      .map((c) => ({
        name: c.name,
        kind: c.kind,
        patients: c.metrics.n_patients,
        septic: c.metrics.n_septic_patients,
        auroc: r(c.metrics.auroc),
        aurocCi: [r(c.ci.auroc.lo), r(c.ci.auroc.hi)],
        calibration: r(c.metrics.calibration_ratio),
        detection: r(c.metrics.detection_rate),
        falseAlarm: r(c.metrics.false_alarm_rate),
        falseAlarmCi: [r(c.ci.false_alarm_rate.lo), r(c.ci.false_alarm_rate.hi)],
        leadHours: c.metrics.median_lead_hours,
        underpowered: Boolean(c.metrics.underpowered),
      })),
    /** Gaps that survived the Bonferroni correction — the only ones the page claims. */
    significant: sub.gaps
      .filter((g) => g.significant_adj)
      .map((g) => ({
        metric: g.metric, a: g.a, b: g.b,
        delta: r(g.delta, 4), ci: [r(g.ci_lo, 4), r(g.ci_hi, 4)],
      })),
    /** The worst cell, flagged for what it is: chosen after seeing the results. */
    exploratory: sub.gaps
      .filter((g) => g.selected && (g.metric === "false_alarm_rate" || g.metric === "auroc"))
      .map((g) => ({
        metric: g.metric, a: g.a, b: g.b,
        delta: r(g.delta, 4), ci: [r(g.ci_lo, 4), r(g.ci_hi, 4)],
      })),
  },

  /** Measured capacity of the runtime — the real CohortCluster.tick() over real
   *  patients, median of repeated runs with the spread beside it. */
  scale: {
    host: scale.host,
    note: scale.note,
    tickBudgetMs: 600,
    levels: scale.levels.map((l) => ({
      beds: l.beds_built,
      p50: r(l.tick_ms.p50, 1),
      p99: r(l.tick_ms.p99, 1),
      spreadPct: l.p50_spread ? r(l.p50_spread.rel_range * 100, 0) : null,
      msPerBed: r(l.ms_per_bed_tick, 2),
      peakRssMb: r(l.peak_rss_mb, 0),
      repeats: l.repeats ?? 1,
      topPhase: Object.keys(l.phases)[0],
      topPhaseShare: r(Object.values(l.phases)[0].share_of_tick, 3),
    })),
  },

  /** Conformal abstention (docs/CONFORMAL.md). The alarm's third state: a per-hour
   *  prediction SET, where {0,1} is an explicit refusal to decide. */
  conformal: {
    alpha: conf.alpha,
    nominal: r(1 - conf.alpha, 3),
    detector: conf.detector,
    lockedCovPos: r(conf.coverage.locked.overall.micro_pos),
    lockedCovNeg: r(conf.coverage.locked.overall.micro_neg),
    lockedMacroPos: r(conf.coverage.locked.overall.macro_pos),
    abstainRate: r(conf.coverage.locked.abstain_rate, 3),
    miscoverageRatio: r(conf.coverage.gate.miscoverage_ratio, 2),
    overrunLimit: conf.coverage.gate.overrun_limit,
    /** What abstention does to the shipped alarm — bracketed BOTH ways, because
     *  what a ward does with a refusal is a values judgement, not a modelling one. */
    conversion: conf.coverage.conversion.policies.map((p) => ({
      policy: p.policy, utility: r(p.utility), detection: r(p.detection, 3),
      falseAlarm: r(p.false_alarm, 3), leadHours: r(p.lead_h, 1),
    })),
    /** CORRECTED. The previously published 0.929 -> 0.739 "collapse" was a two-fit
     *  bug in derive_conformal.py --external: bands calibrated on a model fit on 60%
     *  of setA, applied to scores from a model fit on 100% of it. Single fit: */
    external: {
      withinCovPos: r(confExt.within_setA.overall.micro_pos),
      acrossCovPos: r(confExt.across_setB.overall.micro_pos),
      delta: r(confExt.delta_micro_pos),
      corrected: true,
      supersedes: "0.929 -> 0.739 (two-fit artifact)",
    },
  },

  /** Label-free deployment transfer (docs/TRANSFER.md). A decisive NEGATIVE: the
   *  remedies a new hospital could apply without labels are harmful or inert, and
   *  the gap they were meant to close was largely our own threshold/fit bug. */
  transfer: {
    limitPerSite: transfer.limit_per_site,
    directions: transfer.directions.map((d) => ({
      direction: d.direction,
      domainAuc: r(d.shift.domain_auc, 3),
      priorRatioTrue: r(d.shift.prior_ratio_true, 2),
      priorRatioEm: r(d.shift.prior_ratio_em, 2),
      unadapted: r(d.locked.unadapted.utility),
      priorCorrected: r(d.locked.prior_corrected.utility),
      rateMatched: r(d.locked.rate_matched.utility),
      oracle: r(d.locked.oracle.utility),
      /** The reachable gap: what re-deriving the threshold buys EVEN WITH labels. */
      reachableGap: r(d.locked.oracle.utility - d.locked.unadapted.utility),
      priorDelta: r(d.prior_corrected.delta),
      priorCi: [r(d.prior_corrected.ci[0]), r(d.prior_corrected.ci[1])],
      priorSignificant: d.prior_corrected.significant,
    })),
    /** Conformal coverage transfer under both detectors — plain transferred bands
     *  land within 0.002 of bands calibrated on the target's own LABELS. */
    coverage: [
      ...transfer.directions.map((d) => ({
        detector: transfer.detector, direction: d.direction,
        plain: r(d.conformal["plain (transferred)"].cov_pos),
        weighted: r(d.conformal.weighted.cov_pos),
        oracle: r(d.conformal.oracle.cov_pos),
      })),
      ...transferStack.directions.map((d) => ({
        detector: transferStack.detector, direction: d.direction,
        plain: r(d.conformal["plain (transferred)"].cov_pos),
        weighted: r(d.conformal.weighted.cov_pos),
        oracle: r(d.conformal.oracle.cov_pos),
      })),
    ],
  },

  /** Hyperdimensional computing, reported as it is ACTUALLY used.
   *  Sources: artifacts/qhdc_ablation.json (discrimination) and artifacts/scale.json
   *  (runtime share, real CohortCluster.tick over real patients). HDC is NOT the
   *  detector — the shipped roster is hgb+hgb2+et+lr+xgb@utility with use_hdc=False. */
  hdc: {
    inProductionDetector: false,
    shippedRoster: read("detector_manifest").configs.candidate.roster,
    ablation: Object.entries(q.ablation).map(([name, m]) => ({
      name, auroc: r(m.auroc[0]), aurocSd: r(m.auroc[1]),
      utility: r(m.normalized_utility[0]),
    })),
    /** Where the tick actually goes at the measured ceiling. */
    runtime: (() => {
      const lv = scale.levels[scale.levels.length - 1];
      return {
        beds: lv.beds_built,
        phases: Object.entries(lv.phases).map(([k, v]) => ({
          phase: k, share: r(v.share_of_tick, 3), msPerTick: r(v.ms_per_tick, 1),
        })),
      };
    })(),
  },

  /** Architectures proposed, built, benchmarked and NOT adopted. Provenance:
   *  backend/ml/README.md "Architectures proposed, tested, and not adopted", from
   *  artifacts/minirocket_bench.log and artifacts/forest_vsa_bench.log (the frontier
   *  harness writes tables to stdout, not JSON — the log IS the artifact). */
  negatives: {
    protocol: "dev spec, 3 patient-grouped folds, 310,061 hours / 8,000 patients",
    rows: [
      { name: "xgb-utility (paired baseline)", auroc: 0.7503, aurocSd: 0.0098,
        auprc: 0.0774, utility: 0.3381, detectAtFar15: 0.497, seconds: 22,
        verdict: "baseline" },
      { name: "+ causal MiniRocket (336 feats)", auroc: 0.7495, aurocSd: 0.0088,
        auprc: 0.0728, utility: 0.3246, detectAtFar15: 0.468, seconds: 162,
        verdict: "not adopted" },
    ],
    minirocket: {
      delta: 0.0012, ci: [-0.0090, 0.0129], significant: false,
      standardBankGb: 11.6, shippedFeatures: 336, shippedGb: 0.39,
      checks: 17, mutationAudited: 5,
    },
    forestVsa: { losslessCeiling: 0.7146, donorSameFold: 0.7384 },
  },

  /** Falsification coverage (docs/FALSIFICATION.md). Not code-coverage mutation
   *  testing: this perturbs the SCIENTIFIC invariant and asserts a NAMED check fails.
   *  A surviving mutant is not weak coverage — it is a published claim nothing is
   *  defending, which is why `survivors` is carried and the percentage is not the
   *  point. */
  falsification: {
    generated: fals.generated,
    mutations: fals.n_mutations,
    killed: fals.n_killed,
    coverage: r(fals.coverage, 3),
    suites: Object.entries(fals.suites).map(([k, v]) => ({
      suite: k.split(".").pop(), checks: v.checks,
      mutations: v.mutations, killed: v.killed,
    })),
    families: Object.entries(fals.families).map(([k, v]) => ({
      family: k, n: v.n, killed: v.killed,
    })),
    survivors: fals.survivors.map((x) => ({
      family: x.family, suite: x.suite.split(".").pop(), name: x.name,
      missed: x.missed, reason: x.error || (x.caught_by.length ? "wrong check" : "caught nothing"),
    })),
  },

  /** The earlier Phase 7 protocol. Kept because README and PHASE7_REPORT quote
   *  it (mean AUROC 0.689) — the page labels which protocol each number is from,
   *  since the project's own system_health.py refuses to mix them. */
  phase7: {
    AtoB: { auroc: r(ext.A_to_B.auroc), ece: r(ext.A_to_B.ece, 5) },
    BtoA: { auroc: r(ext.B_to_A.auroc), ece: r(ext.B_to_A.ece, 5) },
    meanAuroc: r((ext.A_to_B.auroc + ext.B_to_A.auroc) / 2),
  },
};

mkdirSync(dirname(OUT), { recursive: true });
writeFileSync(OUT, JSON.stringify(out, null, 1) + "\n");

const kb = (readFileSync(OUT).length / 1024).toFixed(1);
console.log(`wrote data/results.json — ${kb} KB`);
if (Number(kb) > 60) {
  console.error(`\nrefusing: ${kb} KB is too large to import into a page bundle. Prune harder.`);
  process.exit(1);
}
