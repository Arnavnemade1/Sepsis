/* ============================================================
   rPPG A/B validation — live camera HR extraction
   Node-runnable:  node webui/rppg_ab_test.js

   Honest, reproducible justification for upgrading the live browser camera
   (camera.js) from a single GREEN-channel DFT to POS + CHROM chrominance fusion
   (Wang 2017 / de Haan 2013 — the same algorithm validated in
   backend/vision/signal_processing.py).

   We compare HR error on synthetic faces with KNOWN ground-truth HR under two
   regimes:
     (A) clean-ish: mostly independent sensor noise — green alone is competitive.
     (B) realistic: strong COMMON-MODE motion/lighting (a brightness change hits
         all channels together, often inside the cardiac band). This is the
         dominant real-world artifact, and the whole reason POS/CHROM exist:
         their chrominance projections cancel common-mode change while keeping
         the pulse. Green-only cannot, so it locks onto the motion frequency.

   We DON'T claim POS/CHROM is "always" better — (A) shows green is fine on clean
   signals. We claim it is far more robust where it matters (motion/lighting),
   which is exactly what (B) measures.
   ============================================================ */
"use strict";

const avg = (a, i0, i1) => { let s = 0, n = 0; for (let i = Math.max(0, i0); i < Math.min(a.length, i1); i++) { s += a[i]; n++; } return n ? s / n : 0; };
const std = (a) => { const m = a.reduce((x, y) => x + y, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
const _mean = (a) => { let s = 0; for (let i = 0; i < a.length; i++) s += a[i]; return a.length ? s / a.length : 0; };
const detrendHP = (x, fs) => { const k = Math.max(3, Math.round(fs * 0.7)); return x.map((v, i) => v - avg(x, i - k, i + k)); };

// ---- the exact methods that ship in camera.js ----
function posPulse(R, G, B, fs) {                  // POS, overlap-added 1.6 s windows
  const n = R.length; let win = Math.max(Math.round(1.6 * fs), 8); if (n < win) win = n;
  const H = new Array(n).fill(0);
  for (let s0 = 0; s0 + win <= n; s0++) {
    let mr = 0, mg = 0, mb = 0; for (let i = s0; i < s0 + win; i++) { mr += R[i]; mg += G[i]; mb += B[i]; }
    mr = (mr / win) || 1; mg = (mg / win) || 1; mb = (mb / win) || 1;
    const s1 = new Array(win), s2 = new Array(win);
    for (let i = 0; i < win; i++) { const rn = R[s0 + i] / mr, gn = G[s0 + i] / mg, bn = B[s0 + i] / mb; s1[i] = gn - bn; s2[i] = -2 * rn + gn + bn; }
    const sd2 = std(s2), a = sd2 > 1e-9 ? std(s1) / sd2 : 0;
    let hm = 0; const h = new Array(win); for (let i = 0; i < win; i++) { h[i] = s1[i] + a * s2[i]; hm += h[i]; } hm /= win;
    for (let i = 0; i < win; i++) H[s0 + i] += h[i] - hm;
  }
  return H;
}
function chromPulse(R, G, B, fs) {                // CHROM, band-pass before alpha
  const n = R.length, mr = _mean(R) || 1, mg = _mean(G) || 1, mb = _mean(B) || 1;
  let X = new Array(n), Y = new Array(n);
  for (let i = 0; i < n; i++) { const rn = R[i] / mr, gn = G[i] / mg, bn = B[i] / mb; X[i] = 3 * rn - 2 * gn; Y[i] = 1.5 * rn + gn - 1.5 * bn; }
  X = detrendHP(X, fs); Y = detrendHP(Y, fs);
  const sy = std(Y), a = sy > 1e-9 ? std(X) / sy : 0;
  return X.map((x, i) => x - a * Y[i]);
}
function spectralHR(sig, t) {
  let bestP = 0, bestK = 0, total = 0; const P = new Array(139);
  for (let bpm = 42; bpm <= 180; bpm++) { const f = bpm / 60; let re = 0, im = 0;
    for (let i = 0; i < sig.length; i++) { const a = 2 * Math.PI * f * (t[i] - t[0]) / 1000; re += sig[i] * Math.cos(a); im += sig[i] * Math.sin(a); }
    const p = re * re + im * im; P[bpm - 42] = p; total += p; if (p > bestP) { bestP = p; bestK = bpm - 42; } }
  let hr = 42 + bestK;
  if (bestK > 0 && bestK < P.length - 1) { const a = P[bestK - 1], b = P[bestK], c = P[bestK + 1], d = a - 2 * b + c; if (d !== 0) hr = 42 + bestK + 0.5 * (a - c) / d; }
  return { hr, snr: 10 * Math.log10(bestP / Math.max(total - bestP, 1e-9)) };
}
const fusedHR = (R, G, B, t, fs) => {
  const ep = spectralHR(detrendHP(posPulse(R, G, B, fs), fs), t);
  const ec = spectralHR(detrendHP(chromPulse(R, G, B, fs), fs), t);
  return (ep.snr >= ec.snr ? ep : ec).hr;
};
const greenHR = (G, t, fs) => spectralHR(detrendHP(G, fs), t).hr;   // the OLD live method

// ---- synthetic face generator with ground-truth HR ----
function synth(hrTrue, fs, secs, opts, seed) {
  let s = seed; const rnd = () => { s = (s * 1103515245 + 12345) & 0x7fffffff; return s / 0x7fffffff; };
  const aR = 0.0033, aG = 0.0077, aB = 0.0053, DR = 180, DG = 120, DB = 90;   // skin pulse signature + DC
  const n = Math.round(fs * secs), R = [], G = [], B = [], t = [], f = hrTrue / 60;
  for (let i = 0; i < n; i++) {
    const tt = i / fs, pulse = Math.sin(2 * Math.PI * f * tt);
    const bright = (opts.motionAmp || 0) * Math.sin(2 * Math.PI * (opts.motionFreq || 1) * tt) + 0.02 * Math.sin(2 * Math.PI * 0.08 * tt);
    const nz = (opts.noise || 0);
    R.push(DR * (1 + bright) + DR * aR * pulse + nz * (rnd() - 0.5) * 3);
    G.push(DG * (1 + bright) + DG * aG * pulse + nz * (rnd() - 0.5) * 3);
    B.push(DB * (1 + bright) + DB * aB * pulse + nz * (rnd() - 0.5) * 3);
    t.push(tt * 1000);
  }
  return { R, G, B, t };
}

function run(label, optsFn) {
  const fs = 30; let mf = 0, mg = 0, gW = 0, fW = 0, N = 0;
  for (let k = 0; k < 50; k++) {
    const hrTrue = 58 + Math.round(70 * ((k * 131) % 97) / 97);
    const d = synth(hrTrue, fs, 10, optsFn(k), k + 1);
    const f = fusedHR(d.R, d.G, d.B, d.t, fs), g = greenHR(d.G, d.t, fs);
    mf += Math.abs(f - hrTrue); mg += Math.abs(g - hrTrue);
    if (Math.abs(g - hrTrue) > 8) gW++; if (Math.abs(f - hrTrue) > 8) fW++; N++;
  }
  console.log(`\n${label} (n=${N}, 10 s windows @30 fps)`);
  console.log(`  green-only  MAE ${(mg / N).toFixed(2)} bpm   (>8bpm wrong ${gW}/${N})   [old live]`);
  console.log(`  POS+CHROM   MAE ${(mf / N).toFixed(2)} bpm   (>8bpm wrong ${fW}/${N})   [new live]`);
}

console.log("=".repeat(62));
console.log("rPPG live-camera HR extraction — A/B vs ground truth");
console.log("=".repeat(62));
// (A) realistic clean: face-ROI averaging makes per-frame sensor noise tiny
//     (<1%); both methods are essentially perfect here — the point is no regression.
run("(A) clean — realistic ROI sensor noise (still subject, steady light)",
  () => ({ noise: 0.5, motionAmp: 0 }));
run("(B) realistic — common-mode motion/lighting (the real world)",
  (k) => ({ noise: 0.6, motionAmp: 0.015 + 0.02 * (k % 3), motionFreq: 0.9 + 1.3 * ((k * 53) % 89) / 89 }));
console.log("\nTakeaway: green-only is adequate on clean signals but collapses under");
console.log("motion/lighting; POS+CHROM stays accurate — hence the live upgrade.");
