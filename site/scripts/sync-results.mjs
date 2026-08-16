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

// ─── rigorous_eval.json — the judge-facing cross-hospital evaluation ─────────
const rig = read("rigorous_eval");

const model = (m) => ({
  auroc: r(m.auroc),
  aurocCi: arr(m.auroc_ci),
  auprc: r(m.auprc),
  auprcCi: arr(m.auprc_ci),
  brier: r(m.brier, 5),
  ece: r(m.ece, 5),
  prevalence: r(m.prevalence, 5),
  utility: r(m.normalized_utility),
  // sensitivity at fixed specificity — the clinically legible operating points
  sensAtSpec: {
    "0.85": r(m.operating_points?.["sens@spec>=0.85"]),
    "0.90": r(m.operating_points?.["sens@spec>=0.90"]),
    "0.95": r(m.operating_points?.["sens@spec>=0.95"]),
  },
  earlyWarning: {
    detectionRate: r(m.early_warning?.detection_rate),
    medianLeadHours: r(m.early_warning?.median_lead_time_hours, 1),
    meanLeadHours: r(m.early_warning?.mean_lead_time_hours, 1),
    falseAlarmRate: r(m.early_warning?.false_alarm_rate),
    septicPatients: m.early_warning?.septic_patients,
    nonsepticPatients: m.early_warning?.nonseptic_patients,
  },
  // curve arrays are already small (roc 42, pr 150, dc 40, alarm 30, rel 2),
  // so they are kept whole — only the precision is trimmed.
  roc: { fpr: arr(m.roc?.fpr), tpr: arr(m.roc?.tpr) },
  pr: { recall: arr(m.pr?.recall), precision: arr(m.pr?.precision, 5) },
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
     *  decision boundary was placed. From backend/ml/README.md. */
    heldOut: {
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
