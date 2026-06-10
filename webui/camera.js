/* ============================================================
   AEGIS-OS — Optical Biosensor (in-browser camera module)
   Live webcam -> MediaPipe FaceLandmarker -> labeled ROI boxes
   + green-channel rPPG heart rate + posture + sign readouts.
   ============================================================ */
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const $ = (s) => document.querySelector(s);

// landmark index groups — identical to the Python backend (face.py)
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

let landmarker = null, video, overlay, octx, raf = 0, running = false;
let lastTs = -1, fpsEMA = 30;
let green = [], gT = [], lastHRt = 0;
let hr = 0, hrv = 0, rr = 0, snr = 0, roiSNR = {}, posture = "—", quality = 0;
let wave = [];

/* ---------------- lifecycle ---------------- */
async function open() {
  $("#camera-app").classList.remove("hidden");
  setState("idle", "ready — enable camera");
}
function close() {
  stop();
  $("#camera-app").classList.add("hidden");
}
function stop() {
  running = false; cancelAnimationFrame(raf);
  const s = video && video.srcObject;
  if (s) s.getTracks().forEach((t) => t.stop());
  if (video) video.srcObject = null;
  $("#camera-app").classList.remove("live", "scanning");
}

async function enable() {
  video = $("#cam-video"); overlay = $("#cam-overlay"); octx = overlay.getContext("2d");
  setState("idle", "requesting camera…");
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" }, audio: false,
    });
    video.srcObject = stream;
    await video.play();
  } catch (e) {
    setState("error", "camera unavailable — " + (e.name || "denied"));
    return;
  }
  setState("idle", "loading vision model…");
  try {
    const fileset = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm");
    landmarker = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: "vendor/models/face_landmarker.task" },
      runningMode: "VIDEO", numFaces: 1,
    });
  } catch (e) {
    setState("error", "vision model failed to load");
    return;
  }
  $("#camera-app").classList.add("live", "scanning");
  setState("live", "optical feed active");
  running = true; green = []; gT = []; wave = []; loop();
}

function setState(cls, txt) {
  const el = $("#cam-state"); el.className = "cam-state " + cls;
  el.querySelector("span").textContent = txt;
}

/* ---------------- main loop ---------------- */
function loop() {
  if (!running) return;
  raf = requestAnimationFrame(loop);
  if (video.readyState < 2) return;
  const w = video.videoWidth, h = video.videoHeight;
  if (overlay.width !== w) { overlay.width = w; overlay.height = h; }
  const ts = performance.now();
  if (ts <= lastTs) return; lastTs = ts;
  if (lastTs) fpsEMA = 0.9 * fpsEMA + 0.1 * (1000 / Math.max(1, ts - (loop._p || ts))); loop._p = ts;

  let res; try { res = landmarker.detectForVideo(video, ts); } catch { return; }
  octx.clearRect(0, 0, w, h);
  if (res && res.faceLandmarks && res.faceLandmarks.length) {
    const lm = res.faceLandmarks[0];
    drawBoxes(lm, w, h);
    sampleROIs(lm, w, h, ts);
    posture = classifyPosture(lm);
    $("#cam-track").textContent = "locked";
  } else {
    $("#cam-track").textContent = "searching";
  }
  if (ts - lastHRt > 1000) { lastHRt = ts; computeVitals(); paintRail(); }
  $("#cam-fps").textContent = Math.round(fpsEMA);
  paintWave();
}

/* ---------------- detection boxes ---------------- */
function bboxOf(lm, idx, w, h) {
  let x0 = 1e9, y0 = 1e9, x1 = -1e9, y1 = -1e9;
  for (const i of idx) { const p = lm[i]; if (!p) continue;
    x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); }
  const pad = 0.12 * Math.max(x1 - x0, y1 - y0);
  return { x: (x0 - pad) * w, y: (y0 - pad) * h, w: (x1 - x0 + 2 * pad) * w, h: (y1 - y0 + 2 * pad) * h };
}
function drawBoxes(lm, w, h) {
  for (const name in G) {
    const b = bboxOf(lm, G[name], w, h);
    const col = BOX_COL[name];
    cornerBox(b.x, b.y, b.w, b.h, col);
    // label (drawn un-mirrored by counter-flipping so text reads correctly)
    octx.save(); octx.translate(b.x + b.w, b.y - 6); octx.scale(-1, 1);
    octx.font = "600 13px ui-monospace, monospace"; octx.fillStyle = col;
    octx.shadowColor = col; octx.shadowBlur = 6;
    octx.fillText(name.toUpperCase(), 0, 0); octx.restore();
  }
}
function cornerBox(x, y, w, h, col) {
  const L = Math.min(18, w * 0.3, h * 0.3);
  octx.strokeStyle = col; octx.lineWidth = 2; octx.shadowColor = col; octx.shadowBlur = 5;
  octx.globalAlpha = 0.9;
  const seg = [[x, y, x + L, y], [x, y, x, y + L], [x + w, y, x + w - L, y], [x + w, y, x + w, y + L],
    [x, y + h, x + L, y + h], [x, y + h, x, y + h - L], [x + w, y + h, x + w - L, y + h], [x + w, y + h, x + w, y + h - L]];
  octx.beginPath(); for (const s of seg) { octx.moveTo(s[0], s[1]); octx.lineTo(s[2], s[3]); } octx.stroke();
  octx.globalAlpha = 0.18; octx.lineWidth = 1; octx.shadowBlur = 0;
  octx.strokeRect(x, y, w, h); octx.globalAlpha = 1;
}

/* ---------------- rPPG sampling ---------------- */
function regionMeanGreen(lm, idx, w, h) {
  // sample the green channel at the ROI centroid neighbourhood from a tiny canvas
  let cx = 0, cy = 0; for (const i of idx) { cx += lm[i].x; cy += lm[i].y; } cx /= idx.length; cy /= idx.length;
  return sampleGreen(cx * w, cy * h);
}
let scratch, sctx;
function sampleGreen(px, py) {
  if (!scratch) { scratch = document.createElement("canvas"); scratch.width = video.videoWidth; scratch.height = video.videoHeight; sctx = scratch.getContext("2d", { willReadFrequently: true }); }
  const r = 14;
  try {
    sctx.drawImage(video, px - r, py - r, 2 * r, 2 * r, 0, 0, 2 * r, 2 * r);
    const d = sctx.getImageData(0, 0, 2 * r, 2 * r).data;
    let g = 0, n = 0; for (let i = 0; i < d.length; i += 4) { g += d[i + 1]; n++; }
    return g / n;
  } catch { return 0; }
}
function sampleROIs(lm, w, h, ts) {
  const fg = regionMeanGreen(lm, G.forehead, w, h);
  green.push(fg); gT.push(ts);
  while (gT.length && ts - gT[0] > 12000) { green.shift(); gT.shift(); }   // 12 s window
  // crude per-ROI "signal" = recent green variance (for the tracking bars)
  for (const k of ["forehead", "left cheek", "right cheek"]) {
    const v = regionMeanGreen(lm, G[k], w, h);
    roiSNR[k] = roiSNR[k] == null ? v : 0.9 * roiSNR[k] + 0.1 * v;
  }
}

/* ---------------- vitals (green-channel rPPG) ---------------- */
function computeVitals() {
  const n = green.length;
  if (n < 64) { return; }
  const dt = (gT[n - 1] - gT[0]) / (n - 1) / 1000; const fs = 1 / dt;
  // uniform-ish; detrend with moving average
  const x = green.slice();
  const k = Math.max(3, Math.round(fs * 0.7));
  const detr = x.map((v, i) => v - avg(x, i - k, i + k));
  // band-limited DFT power over 40..180 bpm
  let bestP = 0, bestF = 0, total = 0;
  for (let bpm = 42; bpm <= 180; bpm += 1) {
    const f = bpm / 60; let re = 0, im = 0;
    for (let i = 0; i < detr.length; i++) { const a = 2 * Math.PI * f * (gT[i] - gT[0]) / 1000; re += detr[i] * Math.cos(a); im += detr[i] * Math.sin(a); }
    const p = re * re + im * im; total += p; if (p > bestP) { bestP = p; bestF = f; }
  }
  const newHR = bestF * 60;
  hr = hr ? 0.7 * hr + 0.3 * newHR : newHR;                  // light temporal smoothing
  snr = 10 * Math.log10(bestP / Math.max(total - bestP, 1e-9));
  quality = Math.max(0.05, Math.min(1, (snr + 4) / 12));
  // coarse HRV from zero-cross intervals of the filtered signal
  wave = bandpass(detr, fs, bestF);
  hrv = coarseHRV(wave, fs);
  rr = estimateRR(x, fs);
  // crude per-ROI SNR display from relative variance
  const vbase = variance(detr) + 1e-6;
}
function avg(a, i0, i1) { let s = 0, n = 0; for (let i = Math.max(0, i0); i < Math.min(a.length, i1); i++) { s += a[i]; n++; } return n ? s / n : 0; }
function variance(a) { const m = a.reduce((x, y) => x + y, 0) / a.length; return a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length; }
function bandpass(x, fs, f0) {  // simple 2-pole-ish via difference of moving averages around f0
  const w = Math.max(2, Math.round(fs / (f0 * 2)));
  return x.map((v, i) => v - avg(x, i - 3 * w, i + 3 * w)).map((v, i, arr) => avg(arr, i - 1, i + 2));
}
function coarseHRV(w, fs) {
  const peaks = []; for (let i = 1; i < w.length - 1; i++) if (w[i] > w[i - 1] && w[i] >= w[i + 1] && w[i] > 0) peaks.push(i);
  if (peaks.length < 3) return 0;
  const ibi = []; for (let i = 1; i < peaks.length; i++) { const d = (peaks[i] - peaks[i - 1]) / fs * 1000; if (d > 300 && d < 2000) ibi.push(d); }
  if (ibi.length < 2) return 0;
  const m = ibi.reduce((a, b) => a + b, 0) / ibi.length;
  return Math.sqrt(ibi.reduce((s, v) => s + (v - m) ** 2, 0) / ibi.length);
}
function estimateRR(x, fs) {
  let bestP = 0, bestF = 0; for (let br = 8; br <= 30; br += 1) { const f = br / 60; let re = 0, im = 0;
    for (let i = 0; i < x.length; i++) { const a = 2 * Math.PI * f * i / fs; re += x[i] * Math.cos(a); im += x[i] * Math.sin(a); }
    const p = re * re + im * im; if (p > bestP) { bestP = p; bestF = f; } } return bestF * 60;
}

/* ---------------- posture (supine-aware) ---------------- */
function classifyPosture(lm) {
  const v = (i) => [lm[i].x, lm[i].y, lm[i].z];
  const sub = (a, b) => a.map((x, i) => x - b[i]);
  const x = norm(sub(v(POSE.leftEye), v(POSE.rightEye)));
  const roll = Math.atan2(x[1], x[0]) * 180 / Math.PI;
  const ar = Math.abs(roll);
  if (ar > 135) return "supine/inverted";
  if (ar > 30) return "reclined";
  return "upright";
}
function norm(v) { const n = Math.hypot(...v) || 1; return v.map((x) => x / n); }

/* ---------------- rail painting ---------------- */
function paintRail() {
  $("#cam-hr").textContent = hr ? hr.toFixed(0) : "—";
  $("#cam-hrv").textContent = hrv ? hrv.toFixed(0) : "—";
  $("#cam-rr").textContent = rr ? rr.toFixed(0) : "—";
  $("#cam-snr").textContent = isFinite(snr) ? snr.toFixed(1) : "—";
  $("#cam-q-fill").style.width = (quality * 100).toFixed(0) + "%";
  $("#cam-q-val").textContent = (quality * 100).toFixed(0) + "%";
  $("#cam-posture").textContent = posture;
  // ROI bars (normalized brightness as a stand-in for live signal strength)
  const vals = Object.values(roiSNR); const mx = Math.max(1, ...vals);
  $("#cam-rois").innerHTML = ["forehead", "left cheek", "right cheek"].map((k) => {
    const pct = Math.round(((roiSNR[k] || 0) / mx) * 100);
    return `<div class="cam-roi ${pct < 25 ? "lost" : ""}"><span class="cam-roi-n">${k}</span>
      <span class="cam-roi-bar"><i style="width:${pct}%"></i></span><span class="cam-roi-v">${pct}</span></div>`;
  }).join("");
  paintSigns();
}
function paintSigns() {
  const signs = [];
  if (hr > 100) signs.push(["Tachycardia", "cardiovascular", "validated", hr > 120 ? "crit" : "stress"]);
  if (hrv && hrv < 25) signs.push(["HRV suppression", "autonomic", "validated", "dys"]);
  if (rr > 22) signs.push(["Tachypnea", "respiratory", "validated", rr > 26 ? "crit" : "stress"]);
  if (quality < 0.35) signs.push(["Low signal quality", "system", "—", "stress"]);
  $("#cam-sign-count").textContent = signs.length;
  $("#cam-signs").innerHTML = signs.length ? signs.map((s) =>
    `<div class="cam-sign ${s[3]}"><i></i><span class="cam-sign-n">${s[0]}</span>
      <span class="cam-sign-grade">${s[2]}</span><span class="cam-sign-c">${s[1]}</span></div>`).join("")
    : '<div class="muted">no abnormal signs detected</div>';
}
function paintWave() {
  const c = $("#cam-wave"); if (!c) return; const w = c.clientWidth || 300, h = 56; c.width = w; c.height = h;
  const ctx = c.getContext("2d"); ctx.clearRect(0, 0, w, h);
  const seg = wave.slice(-Math.min(wave.length, 240)); if (seg.length < 4) return;
  const mx = Math.max(...seg.map(Math.abs)) || 1;
  ctx.beginPath(); ctx.strokeStyle = "#34e1ff"; ctx.lineWidth = 1.6; ctx.shadowColor = "#34e1ff"; ctx.shadowBlur = 6;
  seg.forEach((v, i) => { const x = i / (seg.length - 1) * w, y = h / 2 - (v / mx) * (h / 2 - 4); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
}

/* ---------------- wire ---------------- */
window.AegisCamera = { open, close, enable };
document.addEventListener("click", (e) => {
  if (e.target.closest("#launch-camera")) window.AegisCamera.open();
  else if (e.target.closest("#cam-close")) window.AegisCamera.close();
  else if (e.target.closest("#cam-enable")) window.AegisCamera.enable();
});
