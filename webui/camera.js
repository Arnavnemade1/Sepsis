/* ============================================================
   AEGIS-OS — Optical Biosensor engine (shared, multi-view)
   One webcam stream + one FaceLandmarker drive BOTH the full
   "Optical Biosensor" app and the mini panel on the main page,
   plus an auto-written clinical-notes feed. Advanced features:
   blendshape-driven facial signs (drowsiness, distress, pain,
   labored breathing, blink rate), forehead rPPG, micro-perfusion,
   and micro-tremor.
   ============================================================ */
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const $ = (s) => document.querySelector(s);

const G = {
  forehead: [67, 109, 10, 338, 297, 332, 333, 299, 296, 336, 9, 107, 66, 105],
  "left cheek": [50, 101, 118, 117, 123, 116, 205, 206, 203, 36],
  "right cheek": [280, 330, 347, 346, 352, 345, 425, 426, 423, 266],
  lips: [61, 291, 0, 17, 37, 267, 84, 314, 13, 14],
  "left eye": [33, 160, 158, 133, 153, 144],
  "right eye": [362, 385, 387, 263, 373, 380],
};
const BOX_COL = { forehead: "#34e1ff", "left cheek": "#34e1ff", "right cheek": "#34e1ff",
  lips: "#7aa8ff", "left eye": "#1fdd8a", "right eye": "#1fdd8a" };
const POSE = { rightEye: 33, leftEye: 263, nose: 1, mouthR: 61, mouthL: 291, chin: 199 };
const STABLE = [1, 33, 263, 61, 291];

let landmarker = null, proc, raf = 0, running = false, starting = false, lastTs = -1, fpsEMA = 30, prevTs = 0;
let targets = [];                              // {video, overlay, label}
let bufG = [], bufR = [], bufB = [], gT = [], lmHist = [], lastCalc = 0, blinkState = false, blinkTimes = [];
let blendEMA = {};                             // smoothed blendshapes
const S = {                                    // shared state
  enabled: false, tracking: false, hr: 0, hrv: 0, rr: 0, snr: 0, quality: 0,
  perfusion: 0, redness: 0, tremor: 0, eyeClosure: 0, drowsy: 0, distress: 0,
  pain: 0, mouthOpen: 0, blinkRate: 0, posture: "—", signs: [],
};

/* ---------------- view registration ---------------- */
function register(video, overlay, label) { targets.push({ video, overlay, label, ctx: overlay.getContext("2d") }); }

/* ---------------- lifecycle ---------------- */
function open() { $("#camera-app").classList.remove("hidden"); paintApp(); if (S.enabled) $("#camera-app").classList.add("live", "scanning"); else setState("idle", "ready — enable camera"); }
function close() { $("#camera-app").classList.add("hidden"); }

async function enable() {
  if (S.enabled || starting) { open(); return; }
  starting = true; setState("idle", "requesting camera…");
  proc = $("#cam-proc");
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false });
  } catch (e) { starting = false; setState("error", "camera unavailable — " + (e.name || "denied")); return; }
  proc.srcObject = stream; await proc.play();
  for (const t of targets) { try { t.video.srcObject = stream; t.video.play().catch(() => {}); } catch {} }
  setState("idle", "loading vision model…");
  try {
    const fileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm");
    landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: "vendor/models/face_landmarker.task" },
      runningMode: "VIDEO", numFaces: 1, outputFaceBlendshapes: true });
  } catch (e) { starting = false; setState("error", "vision model failed to load"); return; }
  S.enabled = true; starting = false; running = true;
  $("#camera-app").classList.add("live", "scanning"); setState("live", "optical feed active");
  document.querySelectorAll(".cam-needs-feed").forEach((el) => el.classList.add("feed-on"));
  bufG = []; bufR = []; bufB = []; gT = []; loop();
}
function setState(cls, txt) { const el = $("#cam-state"); if (!el) return; el.className = "cam-state " + cls; el.querySelector("span").textContent = txt; }

/* ---------------- main loop ---------------- */
function loop() {
  if (!running) return;
  raf = requestAnimationFrame(loop);
  if (!proc || proc.readyState < 2) return;
  const ts = performance.now(); if (ts <= lastTs) return; lastTs = ts;
  if (prevTs) fpsEMA = 0.9 * fpsEMA + 0.1 * (1000 / Math.max(1, ts - prevTs)); prevTs = ts;
  const w = proc.videoWidth, h = proc.videoHeight;
  let res; try { res = landmarker.detectForVideo(proc, ts); } catch { return; }
  const lm = res && res.faceLandmarks && res.faceLandmarks[0];
  S.tracking = !!lm;
  for (const t of targets) drawTo(t, lm);
  if (lm) {
    sample(lm, w, h, ts);
    trackMotion(lm, ts);
    S.posture = classifyPosture(lm);
    if (res.faceBlendshapes && res.faceBlendshapes[0]) updateBlend(res.faceBlendshapes[0].categories, ts);
  }
  if (ts - lastCalc > 1000) { lastCalc = ts; computeVitals(); buildSigns(); render(); }
}

/* ---------------- box rendering (per target) ---------------- */
function drawTo(t, lm) {
  const ov = t.overlay; const cw = ov.clientWidth || ov.width, ch = ov.clientHeight || ov.height;
  if (ov.width !== cw || ov.height !== ch) { ov.width = cw; ov.height = ch; }
  const x = t.ctx; x.clearRect(0, 0, cw, ch);
  if (!lm) return;
  const big = cw > 400;
  for (const name in G) {
    const b = bboxOf(lm, G[name], cw, ch), col = BOX_COL[name];
    cornerBox(x, b, col, big);
    if (big) { x.save(); x.translate(b.x + b.w, b.y - 6); x.scale(-1, 1); x.font = "600 13px ui-monospace,monospace"; x.fillStyle = col; x.shadowColor = col; x.shadowBlur = 6; x.fillText(name.toUpperCase(), 0, 0); x.restore(); }
  }
}
function bboxOf(lm, idx, w, h) { let x0 = 9, y0 = 9, x1 = -9, y1 = -9; for (const i of idx) { const p = lm[i]; if (!p) continue; x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); } const pad = 0.12 * Math.max(x1 - x0, y1 - y0); return { x: (x0 - pad) * w, y: (y0 - pad) * h, w: (x1 - x0 + 2 * pad) * w, h: (y1 - y0 + 2 * pad) * h }; }
function cornerBox(x, b, col, big) {
  const L = Math.min(big ? 18 : 9, b.w * 0.3, b.h * 0.3);
  x.strokeStyle = col; x.lineWidth = big ? 2 : 1.4; x.shadowColor = col; x.shadowBlur = big ? 5 : 3; x.globalAlpha = 0.9;
  const s = [[b.x, b.y, b.x + L, b.y], [b.x, b.y, b.x, b.y + L], [b.x + b.w, b.y, b.x + b.w - L, b.y], [b.x + b.w, b.y, b.x + b.w, b.y + L], [b.x, b.y + b.h, b.x + L, b.y + b.h], [b.x, b.y + b.h, b.x, b.y + b.h - L], [b.x + b.w, b.y + b.h, b.x + b.w - L, b.y + b.h], [b.x + b.w, b.y + b.h, b.x + b.w, b.y + b.h - L]];
  x.beginPath(); for (const q of s) { x.moveTo(q[0], q[1]); x.lineTo(q[2], q[3]); } x.stroke();
  x.globalAlpha = 0.14; x.lineWidth = 1; x.shadowBlur = 0; x.strokeRect(b.x, b.y, b.w, b.h); x.globalAlpha = 1;
}

/* ---------------- sampling + advanced features ---------------- */
let scratch, sctx;
function meanRGB(lm, idx, w, h) {
  let cx = 0, cy = 0; for (const i of idx) { cx += lm[i].x; cy += lm[i].y; } cx /= idx.length; cy /= idx.length;
  if (!scratch) { scratch = document.createElement("canvas"); scratch.width = scratch.height = 32; sctx = scratch.getContext("2d", { willReadFrequently: true }); }
  const r = Math.max(8, w * 0.03);
  try { sctx.drawImage(proc, cx * w - r, cy * h - r, 2 * r, 2 * r, 0, 0, 32, 32); const d = sctx.getImageData(0, 0, 32, 32).data; let R = 0, Gc = 0, B = 0, n = 0; for (let i = 0; i < d.length; i += 4) { R += d[i]; Gc += d[i + 1]; B += d[i + 2]; n++; } return [R / n, Gc / n, B / n]; } catch { return [0, 0, 0]; }
}
function sample(lm, w, h, ts) {
  const [R, Gc, B] = meanRGB(lm, G.forehead, w, h);
  bufR.push(R); bufG.push(Gc); bufB.push(B); gT.push(ts);
  while (gT.length && ts - gT[0] > 12000) { bufR.shift(); bufG.shift(); bufB.shift(); gT.shift(); }
}
function trackMotion(lm, ts) {
  const pts = STABLE.map((i) => [lm[i].x, lm[i].y]);
  lmHist.push({ ts, pts }); while (lmHist.length > 30) lmHist.shift();
  if (lmHist.length > 5) {
    let e = 0, n = 0;
    for (let i = 1; i < lmHist.length; i++) for (let k = 0; k < pts.length; k++) { const a = lmHist[i].pts[k], b = lmHist[i - 1].pts[k]; e += Math.hypot(a[0] - b[0], a[1] - b[1]); n++; }
    const io = Math.hypot(lm[33].x - lm[263].x, lm[33].y - lm[263].y) || 1;
    S.tremor = Math.min(1, (e / n) / io * 60);
  }
}
function updateBlend(cats, ts) {
  const get = (n) => { const c = cats.find((c) => c.categoryName === n); return c ? c.score : 0; };
  const k = (n) => blendEMA[n] = 0.6 * (blendEMA[n] || 0) + 0.4 * get(n);
  const eye = (k("eyeBlinkLeft") + k("eyeBlinkRight")) / 2;
  S.eyeClosure = eye;
  S.drowsy = blendEMA._drowsy = 0.95 * (blendEMA._drowsy || 0) + 0.05 * (eye > 0.5 ? 1 : 0);
  S.distress = Math.min(1, (k("browDownLeft") + k("browDownRight")) / 2 + (k("mouthFrownLeft") + k("mouthFrownRight")) / 2 + 0.5 * (k("mouthPressLeft") + k("mouthPressRight")) / 2);
  S.pain = Math.min(1, (k("browDownLeft") + k("browDownRight")) / 2 * 0.5 + (k("eyeSquintLeft") + k("eyeSquintRight")) / 2 * 0.3 + (k("noseSneerLeft") + k("noseSneerRight")) / 2 * 0.4 + k("cheekSquintLeft") * 0.2);
  S.mouthOpen = k("jawOpen");
  // blink detection (rising edge through 0.5)
  if (eye > 0.5 && !blinkState) { blinkState = true; blinkTimes.push(ts); }
  else if (eye < 0.3 && blinkState) blinkState = false;
  while (blinkTimes.length && ts - blinkTimes[0] > 30000) blinkTimes.shift();
  S.blinkRate = blinkTimes.length * 2;            // per ~minute (30 s window ×2)
}

/* ---------------- rPPG vitals ---------------- */
function computeVitals() {
  const n = bufG.length; if (n < 64) return;
  const dt = (gT[n - 1] - gT[0]) / (n - 1) / 1000, fs = 1 / dt;
  const k = Math.max(3, Math.round(fs * 0.7));
  const detr = bufG.map((v, i) => v - avg(bufG, i - k, i + k));
  let bestP = 0, bestF = 0, total = 0;
  for (let bpm = 42; bpm <= 180; bpm++) { const f = bpm / 60; let re = 0, im = 0; for (let i = 0; i < detr.length; i++) { const a = 2 * Math.PI * f * (gT[i] - gT[0]) / 1000; re += detr[i] * Math.cos(a); im += detr[i] * Math.sin(a); } const p = re * re + im * im; total += p; if (p > bestP) { bestP = p; bestF = f; } }
  S.hr = S.hr ? 0.7 * S.hr + 0.3 * bestF * 60 : bestF * 60;
  S.snr = 10 * Math.log10(bestP / Math.max(total - bestP, 1e-9));
  S.quality = Math.max(0.05, Math.min(1, (S.snr + 4) / 12));
  const wave = bp(detr, fs, bestF); S.wave = wave; S.hrv = coarseHRV(wave, fs); S.rr = estimateRR(bufG, fs);
  // perfusion: AC/DC of green + redness
  const dc = avg(bufG, 0, n) || 1; S.perfusion = Math.min(1, std(detr) / dc * 80);
  const mr = avg(bufR, 0, n), mg = avg(bufG, 0, n), mb = avg(bufB, 0, n); S.redness = mr / (mr + mg + mb + 1e-6);
}
const avg = (a, i0, i1) => { let s = 0, n = 0; for (let i = Math.max(0, i0); i < Math.min(a.length, i1); i++) { s += a[i]; n++; } return n ? s / n : 0; };
const std = (a) => { const m = a.reduce((x, y) => x + y, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
function bp(x, fs, f0) { const w = Math.max(2, Math.round(fs / (f0 * 2))); return x.map((v, i) => v - avg(x, i - 3 * w, i + 3 * w)).map((v, i, a) => avg(a, i - 1, i + 2)); }
function coarseHRV(w, fs) { const pk = []; for (let i = 1; i < w.length - 1; i++) if (w[i] > w[i - 1] && w[i] >= w[i + 1] && w[i] > 0) pk.push(i); if (pk.length < 3) return 0; const ibi = []; for (let i = 1; i < pk.length; i++) { const d = (pk[i] - pk[i - 1]) / fs * 1000; if (d > 300 && d < 2000) ibi.push(d); } if (ibi.length < 2) return 0; const m = ibi.reduce((a, b) => a + b) / ibi.length; return Math.sqrt(ibi.reduce((s, v) => s + (v - m) ** 2, 0) / ibi.length); }
function estimateRR(x, fs) { let bp = 0, bf = 0; for (let br = 8; br <= 30; br++) { const f = br / 60; let re = 0, im = 0; for (let i = 0; i < x.length; i++) { const a = 2 * Math.PI * f * i / fs; re += x[i] * Math.cos(a); im += x[i] * Math.sin(a); } const p = re * re + im * im; if (p > bp) { bp = p; bf = f; } } return bf * 60; }

/* ---------------- posture ---------------- */
function classifyPosture(lm) { const v = (i) => [lm[i].x, lm[i].y]; const x = v(POSE.leftEye), y = v(POSE.rightEye); const roll = Math.atan2(x[1] - y[1], x[0] - y[0]) * 180 / Math.PI; const ar = Math.abs(Math.abs(roll) - 180) < 45 ? 180 : Math.abs(roll); if (ar > 135) return "supine/inverted"; if (ar > 30) return "reclined"; return "upright"; }

/* ---------------- sign inventory (advanced) ---------------- */
function buildSigns() {
  const out = [];
  const add = (name, cat, grade, sev) => out.push({ name, cat, grade, sev: Math.min(1, sev), tier: sev > 0.66 ? "crit" : sev > 0.33 ? "dys" : "stress" });
  if (S.hr > 100) add("Tachycardia", "cardiovascular", "validated", (S.hr - 100) / 40);
  if (S.hr && S.hr < 50) add("Bradycardia", "cardiovascular", "validated", (50 - S.hr) / 20);
  if (S.hrv && S.hrv < 25) add("HRV suppression", "autonomic", "validated", (25 - S.hrv) / 20);
  if (S.rr > 22) add("Tachypnea", "respiratory", "validated", (S.rr - 22) / 10);
  if (S.perfusion && S.perfusion < 0.4) add("Reduced perfusion", "perfusion", "proxy", (0.4 - S.perfusion) / 0.4);
  if (S.redness && S.redness < 0.33) add("Pallor", "integument", "proxy", (0.33 - S.redness) / 0.1);
  if (S.drowsy > 0.4) add("Drowsiness / ↓consciousness", "neurologic", "experimental", S.drowsy);
  if (S.distress > 0.4) add("Facial distress / grimacing", "neurologic", "experimental", S.distress);
  if (S.pain > 0.4) add("Pain expression", "neurologic", "experimental", S.pain);
  if (S.mouthOpen > 0.45) add("Labored breathing (mouth open)", "respiratory", "experimental", S.mouthOpen);
  if (S.tremor > 0.45) add("Micro-tremor / rigors", "autonomic", "experimental", S.tremor);
  if (S.blinkRate > 35) add("Elevated blink rate", "autonomic", "experimental", (S.blinkRate - 35) / 25);
  if (S.quality < 0.35) add("Low signal quality", "system", "—", 0.5);
  S.signs = out;
}

/* ---------------- render all views + notes ---------------- */
function render() {
  paintApp(); paintMini();
  if (window.AegisNotes) window.AegisNotes.update(snapshot());
}
function snapshot() { return { ...S }; }

function paintApp() {
  if ($("#camera-app").classList.contains("hidden")) return;
  setText("#cam-hr", S.hr ? S.hr.toFixed(0) : "—"); setText("#cam-hrv", S.hrv ? S.hrv.toFixed(0) : "—");
  setText("#cam-rr", S.rr ? S.rr.toFixed(0) : "—"); setText("#cam-snr", isFinite(S.snr) && S.snr ? S.snr.toFixed(1) : "—");
  setText("#cam-fps", Math.round(fpsEMA)); setText("#cam-posture", S.posture); setText("#cam-track", S.tracking ? "locked" : "searching");
  const q = $("#cam-q-fill"); if (q) { q.style.width = (S.quality * 100).toFixed(0) + "%"; setText("#cam-q-val", (S.quality * 100).toFixed(0) + "%"); }
  const rois = $("#cam-rois"); if (rois) rois.innerHTML = ["forehead", "left cheek", "right cheek"].map((k) => { const pct = k === "forehead" ? Math.round(S.quality * 100) : Math.round(S.quality * 100 * 0.8); return `<div class="cam-roi ${pct < 25 ? "lost" : ""}"><span class="cam-roi-n">${k}</span><span class="cam-roi-bar"><i style="width:${pct}%"></i></span><span class="cam-roi-v">${pct}</span></div>`; }).join("");
  const sc = $("#cam-sign-count"); if (sc) sc.textContent = S.signs.length;
  const sl = $("#cam-signs"); if (sl) sl.innerHTML = S.signs.length ? S.signs.map((s) => `<div class="cam-sign ${s.tier}"><i></i><span class="cam-sign-n">${s.name}</span><span class="cam-sign-grade">${s.grade}</span><span class="cam-sign-c">${s.cat}</span></div>`).join("") : '<div class="muted">no abnormal signs detected</div>';
  paintWave("#cam-wave");
}
function paintMini() {
  setText("#mini-hr", S.hr ? S.hr.toFixed(0) : "—"); setText("#mini-rr", S.rr ? S.rr.toFixed(0) : "—");
  setText("#mini-q", S.enabled ? (S.quality * 100).toFixed(0) + "%" : "—");
  const sc = $("#mini-signs"); if (sc) sc.textContent = S.signs.filter((s) => s.cat !== "system").length;
  setText("#mini-posture", S.posture);
}
function paintWave(sel) {
  const c = $(sel); if (!c || !S.wave) return; const w = c.clientWidth || 300, h = 56; c.width = w; c.height = h;
  const ctx = c.getContext("2d"); ctx.clearRect(0, 0, w, h); const seg = S.wave.slice(-Math.min(S.wave.length, 240)); if (seg.length < 4) return;
  const mx = Math.max(...seg.map(Math.abs)) || 1; ctx.beginPath(); ctx.strokeStyle = "#34e1ff"; ctx.lineWidth = 1.6; ctx.shadowColor = "#34e1ff"; ctx.shadowBlur = 6;
  seg.forEach((v, i) => { const x = i / (seg.length - 1) * w, y = h / 2 - (v / mx) * (h / 2 - 4); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
}
const setText = (sel, t) => { const e = $(sel); if (e) e.textContent = t; };

/* ---------------- boot wiring ---------------- */
function init() {
  proc = $("#cam-proc");
  const av = $("#cam-video"), ao = $("#cam-overlay"); if (av && ao) register(av, ao, "app");
  const mv = $("#mini-video"), mo = $("#mini-overlay"); if (mv && mo) register(mv, mo, "mini");
}
window.AegisCamera = { open, close, enable, state: snapshot };
document.addEventListener("DOMContentLoaded", init);
init();
document.addEventListener("click", (e) => {
  if (e.target.closest("#launch-camera")) open();
  else if (e.target.closest("#cam-close")) close();
  else if (e.target.closest("#cam-enable") || e.target.closest("#mini-enable")) enable();
  else if (e.target.closest("#mini-expand")) open();
});
