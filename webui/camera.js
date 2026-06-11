/* ============================================================
   AEGIS-OS — Optical Biosensor engine (shared, multi-view)
   One webcam stream drives THREE MediaPipe models (face 478-pt,
   body pose 33-pt, hands 21-pt×2) to observe as many sepsis-
   relevant signs as a camera physically can — across the face,
   torso, shoulders, neck and hands.

   PRECISION CORE: signs are NOT raised on absolute thresholds.
   Each signal is scored as a deviation from the subject's OWN
   adaptive baseline (learned during a warm-up + from stable
   frames), quality-weighted, then passed through a per-sign
   leaky integrator + hysteresis band so a momentary frown or a
   single noisy frame can never raise an alert — only sustained,
   real physiological change does. (Mirrors backend/vision/
   camera_detector.py + monitor.py.)
   ============================================================ */
import { FaceLandmarker, PoseLandmarker, HandLandmarker, FilesetResolver }
  from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18";

const $ = (s) => document.querySelector(s);
const clip01 = (x) => x < 0 ? 0 : x > 1 ? 1 : x;
/* linear ramp: value at/below lo -> 0, at/above hi -> 1 */
const ramp = (v, lo, hi) => clip01((v - lo) / (hi - lo || 1e-6));

/* ---- face ROIs (normalized landmark groups) ---- */
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

/* ---- body pose landmark indices (MediaPipe Pose) ---- */
const PL = { nose: 0, lSh: 11, rSh: 12, lEl: 13, rEl: 14, lWr: 15, rWr: 16,
  lHip: 23, rHip: 24, lKnee: 25, rKnee: 26 };
const POSE_MOTION = [0, 11, 12, 13, 14, 15, 16];   // landmarks tracked for agitation
/* pose skeleton bones for the overlay */
const BONES = [[11, 12], [11, 13], [13, 15], [12, 14], [14, 16], [11, 23], [12, 24],
  [23, 24], [23, 25], [25, 27], [24, 26], [26, 28]];
/* hand connections for the overlay */
const HAND_BONES = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12], [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

const POSE_INTERVAL = 120, HAND_INTERVAL = 180;   // ms — throttle vs face to protect rPPG fps

let faceLm = null, poseLm = null, handLm = null, proc;
let raf = 0, running = false, starting = false, lastTs = -1, fpsEMA = 30, prevTs = 0;
let lastPose = 0, lastHand = 0, lastPoseLm = null, lastHandLms = [];
let targets = [];
let bufG = [], bufR = [], bufB = [], gT = [], lmHist = [], lastCalc = 0, blinkState = false, blinkTimes = [];
let blendEMA = {};
let poseRefPts = null, shBuf = [], poseMotionBuf = [], handCenterBuf = [];

const S = {                                    // shared observable state
  enabled: false, tracking: false, bodyTracking: false, handTracking: false,
  hr: 0, hrv: 0, rr: 0, snr: 0, quality: 0,
  perfusion: 0, redness: 0, faceBlue: 0, tremor: 0,
  eyeClosure: 0, drowsy: 0, distress: 0, pain: 0, mouthOpen: 0, blinkRate: 0,
  posture: "—", rrBody: 0, respEffort: 0, shExc: 0, accessory: 0,
  agit: 0, headGap: 0, forwardLean: 0,
  handJit: 0, handBlue: 0, handMottle: 0, handLum: 0, neckBlue: 0,
  signs: [], emerging: [], risk: 0, calibrating: true, calibProgress: 0,
};
let riskHist = [];

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
  setState("idle", "loading vision models…");
  let fileset;
  try {
    fileset = await FilesetResolver.forVisionTasks("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.18/wasm");
    faceLm = await FaceLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: "vendor/models/face_landmarker.task" },
      runningMode: "VIDEO", numFaces: 1, outputFaceBlendshapes: true });
  } catch (e) { starting = false; setState("error", "face model failed to load"); return; }
  // pose + hands are enhancements — never let a load failure block the core rPPG feed
  setState("idle", "loading body model…");
  try {
    poseLm = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: "vendor/models/pose_landmarker_lite.task" },
      runningMode: "VIDEO", numPoses: 1, minPoseDetectionConfidence: 0.5 });
  } catch (e) { poseLm = null; }
  setState("idle", "loading hand model…");
  try {
    handLm = await HandLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: "vendor/models/hand_landmarker.task" },
      runningMode: "VIDEO", numHands: 2, minHandDetectionConfidence: 0.5 });
  } catch (e) { handLm = null; }
  S.enabled = true; starting = false; running = true;
  Detector.reset();
  $("#camera-app").classList.add("live", "scanning"); setState("live", "calibrating to subject…");
  document.querySelectorAll(".cam-needs-feed").forEach((el) => el.classList.add("feed-on"));
  bufG = []; bufR = []; bufB = []; gT = []; shBuf = []; poseMotionBuf = []; handCenterBuf = []; loop();
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

  // FACE — every frame (rPPG needs the framerate)
  let fres; try { fres = faceLm.detectForVideo(proc, ts); } catch { return; }
  const lm = fres && fres.faceLandmarks && fres.faceLandmarks[0];
  S.tracking = !!lm;
  if (lm) {
    sample(lm, w, h, ts);
    trackMotion(lm, ts);
    S.posture = classifyFacePosture(lm);
    if (fres.faceBlendshapes && fres.faceBlendshapes[0]) updateBlend(fres.faceBlendshapes[0].categories, ts);
  }

  // BODY POSE + HANDS — throttled, and automatically backed off if fps dips so the
  // rPPG framerate (which the HR estimate depends on) is always protected.
  const slow = fpsEMA < 14;
  if (poseLm && ts - lastPose > (slow ? 220 : POSE_INTERVAL)) {
    lastPose = ts;
    try { handlePose(poseLm.detectForVideo(proc, ts), ts, w, h); } catch {}
  }
  if (handLm && ts - lastHand > (slow ? 330 : HAND_INTERVAL)) {
    lastHand = ts;
    try { handleHands(handLm.detectForVideo(proc, ts), ts, w, h); } catch {}
  }

  for (const t of targets) drawTo(t, lm);
  if (ts - lastCalc > 1000) { lastCalc = ts; computeVitals(); computeBody(); runDetector(); render(); }
}

/* ---------------- BODY POSE handling ---------------- */
function handlePose(res, ts, w, h) {
  const lm = res && res.landmarks && res.landmarks[0];
  lastPoseLm = lm || null;
  const vis = lm && (lm[PL.lSh]?.visibility ?? 0) > 0.5 && (lm[PL.rSh]?.visibility ?? 0) > 0.5;
  S.bodyTracking = !!vis;
  if (!vis) return;
  const lS = lm[PL.lSh], rS = lm[PL.rSh], nose = lm[PL.nose];
  const shY = (lS.y + rS.y) / 2, shX = (lS.x + rS.x) / 2;
  const shW = Math.hypot(lS.x - rS.x, lS.y - rS.y) || 0.2;
  shBuf.push({ ts, y: shY, w: shW }); while (shBuf.length && ts - shBuf[0].ts > 14000) shBuf.shift();

  // whole-body agitation: mean per-landmark displacement, normalized by shoulder width
  const pts = POSE_MOTION.map((i) => [lm[i].x, lm[i].y]);
  if (poseRefPts) {
    let e = 0; for (let k = 0; k < pts.length; k++) e += Math.hypot(pts[k][0] - poseRefPts[k][0], pts[k][1] - poseRefPts[k][1]);
    poseMotionBuf.push({ ts, m: e / pts.length / shW }); while (poseMotionBuf.length > 40) poseMotionBuf.shift();
  }
  poseRefPts = pts;

  // head/neck vertical tone: gap nose→shoulders normalized (shrinks when head drops)
  S.headGap = (shY - nose.y) / shW;
  // forward lean (tripod proxy) from relative depth
  const shZ = ((lS.z ?? 0) + (rS.z ?? 0)) / 2;
  S.forwardLean = clip01(((shZ) - (nose.z ?? 0)) / 0.25);

  // neck skin colour (just above the shoulder midline)
  const neck = patchStats(shX, shY - shW * 0.45, w, h, 0.05);
  if (neck) S.neckBlue = 0.7 * (S.neckBlue || neck.blue) + 0.3 * neck.blue;
}

/* ---------------- HAND handling ---------------- */
function handleHands(res, ts, w, h) {
  const hands = (res && res.landmarks) || [];
  lastHandLms = hands;
  S.handTracking = hands.length > 0;
  if (!hands.length) return;
  const hand = hands[0];
  // hand size for normalization (wrist -> middle MCP)
  const hsz = Math.hypot(hand[0].x - hand[9].x, hand[0].y - hand[9].y) || 0.1;
  const cx = (hand[0].x + hand[5].x + hand[17].x) / 3, cy = (hand[0].y + hand[5].y + hand[17].y) / 3;
  handCenterBuf.push({ ts, x: cx, y: cy, s: hsz }); while (handCenterBuf.length > 24) handCenterBuf.shift();
  // back-of-hand skin colour (between wrist and knuckles)
  const bx = (hand[0].x + hand[9].x) / 2, by = (hand[0].y + hand[9].y) / 2;
  const c = patchStats(bx, by, w, h, 0.045);
  if (c) {
    S.handBlue = 0.7 * (S.handBlue || c.blue) + 0.3 * c.blue;
    S.handMottle = 0.7 * (S.handMottle || c.mottle) + 0.3 * c.mottle;
    S.handLum = 0.7 * (S.handLum || c.lum) + 0.3 * c.lum;
  }
}

/* ---------------- box / skeleton rendering (per target) ---------------- */
function drawTo(t, lm) {
  const ov = t.overlay; const cw = ov.clientWidth || ov.width, ch = ov.clientHeight || ov.height;
  if (ov.width !== cw || ov.height !== ch) { ov.width = cw; ov.height = ch; }
  const x = t.ctx; x.clearRect(0, 0, cw, ch);
  const big = cw > 400;
  if (big) drawScan(x, cw, ch);
  // body skeleton + hands only on the large app view (cheap to skip on the mini)
  if (big && lastPoseLm) drawSkeleton(x, lastPoseLm, cw, ch);
  if (big && lastHandLms.length) drawHands(x, lastHandLms, cw, ch);
  if (!lm) return;
  for (const name in G) {
    const b = bboxOf(lm, G[name], cw, ch), col = BOX_COL[name];
    cornerBox(x, b, col, big);
    if (big) { x.save(); x.translate(b.x + b.w, b.y - 6); x.scale(-1, 1); x.font = "600 13px ui-monospace,monospace"; x.fillStyle = col; x.shadowColor = col; x.shadowBlur = 6; x.fillText(name.toUpperCase(), 0, 0); x.restore(); }
  }
}
function drawScan(x, w, h) {
  const y = ((performance.now() % 2600) / 2600) * h;
  const g = x.createLinearGradient(0, y - 46, 0, y + 4);
  g.addColorStop(0, "rgba(52,225,255,0)"); g.addColorStop(1, "rgba(52,225,255,.18)");
  x.fillStyle = g; x.fillRect(0, y - 46, w, 50);
  x.strokeStyle = "rgba(52,225,255,.45)"; x.lineWidth = 1; x.shadowColor = "#34e1ff"; x.shadowBlur = 6;
  x.beginPath(); x.moveTo(0, y); x.lineTo(w, y); x.stroke(); x.shadowBlur = 0;
}
function drawSkeleton(x, lm, w, h) {
  x.save(); x.strokeStyle = "rgba(31,221,138,.55)"; x.lineWidth = 2; x.shadowColor = "#1fdd8a"; x.shadowBlur = 4;
  x.beginPath();
  for (const [a, b] of BONES) { const p = lm[a], q = lm[b]; if (!p || !q || (p.visibility ?? 1) < 0.4 || (q.visibility ?? 1) < 0.4) continue; x.moveTo(p.x * w, p.y * h); x.lineTo(q.x * w, q.y * h); }
  x.stroke();
  x.fillStyle = "rgba(31,221,138,.85)";
  for (const i of [11, 12, 13, 14, 15, 16, 23, 24]) { const p = lm[i]; if (!p || (p.visibility ?? 1) < 0.4) continue; x.beginPath(); x.arc(p.x * w, p.y * h, 3, 0, 7); x.fill(); }
  x.restore();
}
function drawHands(x, hands, w, h) {
  x.save(); x.strokeStyle = "rgba(122,168,255,.7)"; x.lineWidth = 1.6; x.shadowColor = "#7aa8ff"; x.shadowBlur = 4;
  for (const hand of hands) {
    x.beginPath();
    for (const [a, b] of HAND_BONES) { x.moveTo(hand[a].x * w, hand[a].y * h); x.lineTo(hand[b].x * w, hand[b].y * h); }
    x.stroke();
    x.fillStyle = "rgba(122,168,255,.9)";
    for (const p of hand) { x.beginPath(); x.arc(p.x * w, p.y * h, 2, 0, 7); x.fill(); }
  }
  x.restore();
}
function bboxOf(lm, idx, w, h) { let x0 = 9, y0 = 9, x1 = -9, y1 = -9; for (const i of idx) { const p = lm[i]; if (!p) continue; x0 = Math.min(x0, p.x); y0 = Math.min(y0, p.y); x1 = Math.max(x1, p.x); y1 = Math.max(y1, p.y); } const pad = 0.12 * Math.max(x1 - x0, y1 - y0); return { x: (x0 - pad) * w, y: (y0 - pad) * h, w: (x1 - x0 + 2 * pad) * w, h: (y1 - y0 + 2 * pad) * h }; }
function cornerBox(x, b, col, big) {
  const L = Math.min(big ? 18 : 9, b.w * 0.3, b.h * 0.3);
  x.strokeStyle = col; x.lineWidth = big ? 2 : 1.4; x.shadowColor = col; x.shadowBlur = big ? 5 : 3; x.globalAlpha = 0.9;
  const s = [[b.x, b.y, b.x + L, b.y], [b.x, b.y, b.x, b.y + L], [b.x + b.w, b.y, b.x + b.w - L, b.y], [b.x + b.w, b.y, b.x + b.w, b.y + L], [b.x, b.y + b.h, b.x + L, b.y + b.h], [b.x, b.y + b.h, b.x, b.y + b.h - L], [b.x + b.w, b.y + b.h, b.x + b.w - L, b.y + b.h], [b.x + b.w, b.y + b.h, b.x + b.w, b.y + b.h - L]];
  x.beginPath(); for (const q of s) { x.moveTo(q[0], q[1]); x.lineTo(q[2], q[3]); } x.stroke();
  x.globalAlpha = 0.14; x.lineWidth = 1; x.shadowBlur = 0; x.strokeRect(b.x, b.y, b.w, b.h); x.globalAlpha = 1;
}

/* ---------------- sampling + colour stats ---------------- */
let scratch, sctx;
function ensureScratch() { if (!scratch) { scratch = document.createElement("canvas"); scratch.width = scratch.height = 32; sctx = scratch.getContext("2d", { willReadFrequently: true }); } }
function meanRGB(lm, idx, w, h) {
  let cx = 0, cy = 0; for (const i of idx) { cx += lm[i].x; cy += lm[i].y; } cx /= idx.length; cy /= idx.length;
  const s = patchStats(cx, cy, w, h, 0.03); return s ? [s.R, s.G, s.B] : [0, 0, 0];
}
/* mean colour + blue-ratio + saturation + spatial mottling over a small patch */
function patchStats(cx, cy, w, h, rad) {
  ensureScratch();
  const r = Math.max(8, w * rad);
  try {
    sctx.drawImage(proc, cx * w - r, cy * h - r, 2 * r, 2 * r, 0, 0, 32, 32);
    const d = sctx.getImageData(0, 0, 32, 32).data;
    let R = 0, G = 0, B = 0, n = 0; const rb = [];
    for (let i = 0; i < d.length; i += 4) { R += d[i]; G += d[i + 1]; B += d[i + 2]; rb.push(d[i] - d[i + 2]); n++; }
    R /= n; G /= n; B /= n;
    const blue = B / (R + G + B + 1e-6);
    const mx = Math.max(R, G, B), mn = Math.min(R, G, B), sat = (mx - mn) / (mx + 1e-6);
    const rbm = rb.reduce((a, b) => a + b, 0) / n; let v = 0; for (const z of rb) v += (z - rbm) ** 2; v /= n;
    return { R, G, B, blue, sat, lum: (R + G + B) / 3, mottle: Math.sqrt(v) };
  } catch { return null; }
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
  if (eye > 0.5 && !blinkState) { blinkState = true; blinkTimes.push(ts); }
  else if (eye < 0.3 && blinkState) blinkState = false;
  while (blinkTimes.length && ts - blinkTimes[0] > 30000) blinkTimes.shift();
  S.blinkRate = blinkTimes.length * 2;
}

/* ---------------- rPPG vitals (unchanged core) ---------------- */
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
  const dc = avg(bufG, 0, n) || 1; S.perfusion = Math.min(1, std(detr) / dc * 80);
  const mr = avg(bufR, 0, n), mg = avg(bufG, 0, n), mb = avg(bufB, 0, n);
  S.redness = mr / (mr + mg + mb + 1e-6); S.faceBlue = mb / (mr + mg + mb + 1e-6);
}
const avg = (a, i0, i1) => { let s = 0, n = 0; for (let i = Math.max(0, i0); i < Math.min(a.length, i1); i++) { s += a[i]; n++; } return n ? s / n : 0; };
const std = (a) => { const m = a.reduce((x, y) => x + y, 0) / a.length; return Math.sqrt(a.reduce((s, v) => s + (v - m) ** 2, 0) / a.length); };
function bp(x, fs, f0) { const w = Math.max(2, Math.round(fs / (f0 * 2))); return x.map((v, i) => v - avg(x, i - 3 * w, i + 3 * w)).map((v, i, a) => avg(a, i - 1, i + 2)); }
function coarseHRV(w, fs) { const pk = []; for (let i = 1; i < w.length - 1; i++) if (w[i] > w[i - 1] && w[i] >= w[i + 1] && w[i] > 0) pk.push(i); if (pk.length < 3) return 0; const ibi = []; for (let i = 1; i < pk.length; i++) { const d = (pk[i] - pk[i - 1]) / fs * 1000; if (d > 300 && d < 2000) ibi.push(d); } if (ibi.length < 2) return 0; const m = ibi.reduce((a, b) => a + b) / ibi.length; return Math.sqrt(ibi.reduce((s, v) => s + (v - m) ** 2, 0) / ibi.length); }
function estimateRR(x, fs) { let bp = 0, bf = 0; for (let br = 8; br <= 30; br++) { const f = br / 60; let re = 0, im = 0; for (let i = 0; i < x.length; i++) { const a = 2 * Math.PI * f * i / fs; re += x[i] * Math.cos(a); im += x[i] * Math.sin(a); } const p = re * re + im * im; if (p > bp) { bp = p; bf = f; } } return bf * 60; }

/* ---------------- body-derived vitals (shoulders / motion) ---------------- */
function computeBody() {
  // respiration from shoulder rise — independent of (and more robust than) facial RR
  if (shBuf.length > 20) {
    const ys = shBuf.map((b) => b.y), t0 = shBuf[0].ts;
    const fs = (shBuf.length - 1) / ((shBuf[shBuf.length - 1].ts - t0) / 1000 || 1);
    const m = ys.reduce((a, b) => a + b, 0) / ys.length;
    const det = ys.map((v) => v - m);
    let bp = 0, bf = 0;
    for (let br = 8; br <= 32; br++) { const f = br / 60; let re = 0, im = 0; for (let i = 0; i < det.length; i++) { const a = 2 * Math.PI * f * i / fs; re += det[i] * Math.cos(a); im += det[i] * Math.sin(a); } const p = re * re + im * im; if (p > bp) { bp = p; bf = f; } }
    S.rrBody = bf * 60;
    const shW = shBuf.reduce((a, b) => a + b.w, 0) / shBuf.length || 0.2;
    S.shExc = std(det) / shW;                  // breathing excursion, scale-invariant
    S.respEffort = clip01(S.shExc / 0.06);     // ~0.06 normalized excursion ≈ deep effort
  }
  // sustained whole-body motion -> agitation/restlessness
  if (poseMotionBuf.length > 4) S.agit = poseMotionBuf.slice(-12).reduce((a, b) => a + b.m, 0) / Math.min(12, poseMotionBuf.length);
  // hand jitter (tremor proxy) — normalized by hand size, high-passed
  if (handCenterBuf.length > 6) {
    let e = 0; for (let i = 1; i < handCenterBuf.length; i++) { const a = handCenterBuf[i], b = handCenterBuf[i - 1]; e += Math.hypot(a.x - b.x, a.y - b.y) / (a.s || 0.1); }
    S.handJit = e / (handCenterBuf.length - 1);
  }
  if (S.bodyTracking) S.posture = classifyBodyPosture();
}
function classifyBodyPosture() {
  if (S.forwardLean > 0.55) return "forward-lean";
  if (S.headGap < 0.55) return "slumped";
  return S.posture && S.posture !== "—" ? S.posture : "upright";
}
function classifyFacePosture(lm) { const v = (i) => [lm[i].x, lm[i].y]; const x = v(POSE.leftEye), y = v(POSE.rightEye); const roll = Math.atan2(x[1] - y[1], x[0] - y[0]) * 180 / Math.PI; const ar = Math.abs(Math.abs(roll) - 180) < 45 ? 180 : Math.abs(roll); if (ar > 135) return "supine/inverted"; if (ar > 30) return "reclined"; return "upright"; }

/* ============================================================
   PRECISION DETECTOR — adaptive baseline + persistence + hysteresis
   ============================================================ */
const Detector = (() => {
  const DEF = { hr: 75, hrv: 45, rrFace: 15, rrBody: 15, perf: 0.5, redness: 0.36,
    blink: 14, shExc: 0.02, agit: 0.015, handJit: 0.02, faceTremor: 0.05,
    handBlue: 0.33, neckBlue: 0.33, faceBlue: 0.33, headGap: 0.95, handMottle: 8 };
  let base = { ...DEF };
  const st = new Map();                          // id -> {level, active}
  let warm = 0; const WARMUP = 8;                // ~8 s calibration before any flag
  let sens = 0;                                  // sensitivity offset: − eager · + guarded
  const setSensitivity = (m) => { sens = m === "sensitive" ? -0.09 : m === "guarded" ? 0.09 : 0; };

  /* Persistence presets by evidence grade. Trustworthy physiology (validated /
     proxy) reacts FAST and on a low threshold; body proxies a little more
     guarded; facial EXPRESSIONS most guarded of all — they were the original
     false-positive source, so a frown must persist ~4 s before it ever counts.
     alpha = rise speed · on/off = hysteresis band. */
  const P = {
    validated:    { a: 0.50, on: 0.34, off: 0.20 },
    proxy:        { a: 0.42, on: 0.40, off: 0.24 },
    experimental: { a: 0.34, on: 0.46, off: 0.28 },
    expression:   { a: 0.22, on: 0.58, off: 0.36 },
  };
  const RW = { validated: 0.9, proxy: 0.7, experimental: 0.5, expression: 0.35 };  // risk weights
  const faceOK = (c) => c.tracking && c.quality > 0.4;
  const SIGNS = [
    // ---- cardiovascular / autonomic (face rPPG) ----
    { id: "tachy", name: "Tachycardia", cat: "cardiovascular", region: "heart", grade: "validated",
      gate: faceOK, score: (c, b) => ramp(c.hr, b.hr * 1.12, b.hr * 1.45) },
    { id: "brady", name: "Bradycardia", cat: "cardiovascular", region: "heart", grade: "validated",
      gate: (c) => faceOK(c) && c.hr > 30, score: (c, b) => ramp(b.hr * 0.85 - c.hr, 0, b.hr * 0.25) },
    { id: "hrv", name: "HRV suppression", cat: "autonomic", region: "heart", grade: "validated",
      gate: (c) => faceOK(c) && c.hrv > 0, score: (c, b) => ramp(1 - c.hrv / Math.max(b.hrv, 1e-6), 0.3, 0.62) },
    // ---- respiratory (face + shoulders) ----
    { id: "tachypnea", name: "Tachypnea", cat: "respiratory", region: "lungs", grade: "validated",
      gate: (c) => faceOK(c) || c.bodyTracking,
      score: (c, b) => { const rr = Math.max(faceOK(c) ? c.rr : 0, c.bodyTracking ? c.rrBody : 0); return ramp(rr, Math.max(22, b.rrFace * 1.35), 34); } },
    { id: "accessory", name: "Accessory-muscle / labored breathing", cat: "respiratory", region: "lungs", grade: "proxy",
      gate: (c) => c.bodyTracking, score: (c, b) => Math.min(ramp(c.shExc / Math.max(b.shExc, 1e-3), 1.9, 3.6), 1) * ramp(c.respEffort, 0.4, 0.9) },
    { id: "tripod", name: "Tripod / forward-lean posture", cat: "respiratory", region: "lungs", grade: "experimental",
      gate: (c) => c.bodyTracking, score: (c) => ramp(c.forwardLean, 0.55, 0.85) },
    { id: "mouth", name: "Open-mouth breathing", cat: "respiratory", region: "lungs", grade: "experimental",
      gate: faceOK, score: (c) => ramp(c.mouthOpen, 0.5, 0.85) },
    // ---- perfusion / integument (face, neck, hands) ----
    { id: "perf", name: "Reduced perfusion", cat: "perfusion", region: "skin", grade: "proxy",
      gate: (c) => faceOK(c) && c.perfusion > 0, score: (c, b) => ramp(1 - c.perfusion / Math.max(b.perf, 1e-6), 0.25, 0.6) },
    { id: "pallor", name: "Pallor", cat: "integument", region: "skin", grade: "proxy",
      gate: faceOK, score: (c, b) => ramp(1 - c.redness / Math.max(b.redness, 1e-6), 0.12, 0.34) },
    { id: "cyanosis", name: "Peripheral cyanosis", cat: "perfusion", region: "skin", grade: "experimental",
      gate: (c) => c.handTracking, score: (c, b) => ramp(c.handBlue / Math.max(b.handBlue, 1e-6), 1.12, 1.45) },
    { id: "mottle", name: "Skin mottling", cat: "perfusion", region: "skin", grade: "experimental",
      gate: (c) => c.handTracking, score: (c, b) => ramp(c.handMottle / Math.max(b.handMottle, 1e-6), 1.5, 3) },
    // ---- neurologic / mental status (face + posture) ----
    { id: "drowsy", name: "Drowsiness / ↓consciousness", cat: "neurologic", region: "brain", grade: "experimental",
      gate: faceOK, score: (c) => ramp(c.drowsy, 0.45, 0.9) },
    { id: "toneloss", name: "Head drop / postural tone loss", cat: "neurologic", region: "brain", grade: "experimental",
      gate: (c) => c.bodyTracking, score: (c, b) => ramp(1 - c.headGap / Math.max(b.headGap, 1e-6), 0.25, 0.55) },
    { id: "distress", name: "Facial distress", cat: "neurologic", region: "brain", grade: "expression",
      gate: faceOK, score: (c) => ramp(c.distress, 0.55, 0.9) },
    { id: "pain", name: "Pain expression", cat: "neurologic", region: "brain", grade: "expression",
      gate: faceOK, score: (c) => ramp(c.pain, 0.5, 0.85) },
    { id: "agitation", name: "Agitation / restlessness", cat: "neurologic", region: "brain", grade: "experimental",
      gate: (c) => c.bodyTracking, score: (c, b) => ramp(c.agit / Math.max(b.agit, 1e-4), 2.2, 4.5) },
    // ---- motor / autonomic ----
    { id: "rigors", name: "Rigors / shivering", cat: "autonomic", region: "skin", grade: "experimental",
      gate: (c) => faceOK(c) || c.handTracking, score: (c, b) => Math.max(ramp(c.tremor / Math.max(b.faceTremor, 1e-3), 1.8, 3.4), c.handTracking ? ramp(c.handJit / Math.max(b.handJit, 1e-3), 1.8, 3.4) : 0) },
    { id: "handtremor", name: "Hand tremor", cat: "autonomic", region: "skin", grade: "experimental",
      gate: (c) => c.handTracking, score: (c, b) => ramp(c.handJit / Math.max(b.handJit, 1e-3), 1.8, 3.6) },
  ];

  function baseSrc(c) {
    return { hr: c.hr, hrv: c.hrv, rrFace: c.rr, rrBody: c.rrBody, perf: c.perfusion, redness: c.redness,
      blink: c.blinkRate, shExc: c.shExc, agit: c.agit, handJit: c.handJit, faceTremor: c.tremor,
      handBlue: c.handBlue, neckBlue: c.neckBlue, faceBlue: c.faceBlue, headGap: c.headGap, handMottle: c.handMottle };
  }
  /* Learn the subject's resting normal — aggressively during warm-up, then only
     from good-quality, stable frames. The freeze gate watches the strongest
     single sign level (not just already-active ones), so the baseline stops
     adapting the moment ANY sign starts to build — it can never chase an
     emerging abnormality toward "normal". */
  function adapt(c, peak) {
    const warming = warm < WARMUP;
    if (!warming && (c.quality < 0.5 || peak > 0.2)) return;
    const src = baseSrc(c), lr = warming ? 0.16 : 0.03 * c.quality;
    for (const key in base) { const v = src[key]; if (v != null && isFinite(v) && v > 0) base[key] = (1 - lr) * base[key] + lr * v; }
  }

  function tick(c) {
    const cand = SIGNS.map((def) => {
      const p = P[def.grade], on = p.on + sens, off = Math.max(0.12, p.off + sens);
      const gated = warm >= WARMUP && def.gate(c);
      let raw = 0;
      if (gated) raw = clip01(def.score(c, base)) * clip01(c.quality + 0.3);
      const s = st.get(def.id) || { level: 0, active: false };
      s.level = (1 - p.a) * s.level + p.a * raw;
      if (s.level > on) s.active = true; else if (s.level < off) s.active = false;
      st.set(def.id, s);
      return { def, level: s.level, active: s.active, gated, on };
    });
    const peak = cand.reduce((m, x) => Math.max(m, x.level), 0);    // strongest emerging sign
    adapt(c, peak);
    warm = Math.min(WARMUP, warm + 1);
    const mk = (x) => ({ name: x.def.name, cat: x.def.cat, grade: x.def.grade, region: x.def.region,
      sev: x.level, conf: Math.round(x.level * 100),
      tier: x.level > 0.66 ? "crit" : x.level > 0.4 ? "dys" : "stress" });
    const signs = cand.filter((x) => x.active).map(mk).sort((a, b) => b.sev - a.sev);
    const emerging = cand.filter((x) => !x.active && x.gated && x.level > x.on * 0.5)
      .map(mk).sort((a, b) => b.sev - a.sev).slice(0, 3);
    // composite optical-risk index — saturating, grade-weighted blend of ALL
    // evidence (incl. sub-threshold) so the meter responds before any single
    // sign confirms.  risk = 1 − Π(1 − level·weight)
    let keep = 1; for (const x of cand) keep *= (1 - clip01(x.level) * RW[x.def.grade]);
    const risk = clip01(1 - keep);
    return { signs, emerging, risk, calibrating: warm < WARMUP, calibProgress: warm / WARMUP, baseline: { ...base } };
  }
  function reset() { st.clear(); warm = 0; base = { ...DEF }; sens = sens; }
  return { tick, reset, baseline: () => base, setSensitivity };
})();

function runDetector() {
  const out = Detector.tick(S);
  S.signs = out.signs; S.emerging = out.emerging; S.risk = out.risk;
  S.calibrating = out.calibrating; S.calibProgress = out.calibProgress;
  riskHist.push(out.risk); if (riskHist.length > 8) riskHist.shift();
  if (S.enabled) setState(out.calibrating ? "idle" : "live",
    out.calibrating ? `calibrating to subject… ${Math.round(out.calibProgress * 100)}%` : "optical feed active");
}
function riskTrend() {
  if (riskHist.length < 4) return "—";
  const d = riskHist[riskHist.length - 1] - riskHist[0];
  return d > 0.05 ? "▲ rising" : d < -0.05 ? "▼ falling" : "— stable";
}

/* ---------------- camera → twin coupling ---------------- */
function publishOptical() {
  const twin = window.AegisTwin; if (!twin || !twin.setOptical) return;
  const reg = {};
  if (S.enabled && (S.tracking || S.bodyTracking) && !S.calibrating) {
    for (const s of S.signs) {
      const k = s.region; if (!k) continue;
      const sev = Math.min(1, s.sev || 0);
      if (sev > (reg[k] ? reg[k].severity : 0))
        reg[k] = { monitored: true, severity: sev, optical: true,
          status: sev > 0.66 ? "CRITICAL" : sev > 0.33 ? "DYSFUNCTION" : "STRESS" };
    }
  }
  twin.setOptical(reg);
  const pill = $("#twin-optical");
  if (pill) pill.classList.toggle("hidden", !(S.enabled && (S.tracking || S.bodyTracking) && !S.calibrating));
}

/* ---------------- render all views + notes ---------------- */
function render() {
  paintApp(); paintMini(); publishOptical(); ingestToBackend();
  if (window.AegisNotes) window.AegisNotes.update(snapshot());
}

/* ---- push REAL readings to the backend → drives the live twin off the camera
   (taking over from the simulator) and persists to Supabase. Throttled. ---- */
let lastIngest = 0;
function ingestToBackend() {
  if (!S.enabled || S.calibrating || !S.tracking || !(S.hr > 0)) return;
  const now = performance.now();
  if (now - lastIngest < 2500) return;          // ≤1 real reading / 2.5 s
  lastIngest = now;
  const tel = { heart_rate: +S.hr.toFixed(1) };  // measured channels only (honest)
  if (S.hrv > 0) tel.hrv = +S.hrv.toFixed(1);
  const rr = Math.max(S.rr || 0, S.rrBody || 0);
  if (rr > 0) tel.respiratory_rate = +rr.toFixed(1);
  const cam = {
    perfusion_index: +(S.perfusion || 0).toFixed(3),
    respiratory_rate: rr || null,
    stress_vector: +Math.min(1, (S.distress || 0) * 0.6 + Math.min(1, (S.agit || 0) / 0.05) * 0.4).toFixed(3),
    quality: +(S.quality || 0).toFixed(3),
    signs: (S.signs || []).map((s) => ({ name: s.name, region: s.region, severity: +(s.sev || 0).toFixed(2), grade: s.grade })),
  };
  fetch(window.AEGIS.http("/aegis/ingest"), {
    method: "POST", headers: { "Content-Type": "application/json" }, keepalive: true,
    body: JSON.stringify({ telemetry: tel, source: "camera", signal_quality: +(S.quality || 0).toFixed(3), camera_vitals: cam }),
  }).catch(() => {});
}
function snapshot() { return { ...S }; }

function paintApp() {
  if ($("#camera-app").classList.contains("hidden")) return;
  setText("#cam-hr", S.hr ? S.hr.toFixed(0) : "—"); setText("#cam-hrv", S.hrv ? S.hrv.toFixed(0) : "—");
  const hrEl = $("#cam-hr");                                  // pulse the readout at the measured HR
  if (hrEl) hrEl.style.animation = S.hr > 35 ? `camPulse ${(60 / S.hr).toFixed(2)}s ease-in-out infinite` : "none";
  const rf = $("#cam-risk-fill");
  if (rf) {
    const pct = Math.round((S.risk || 0) * 100);
    rf.style.width = (S.calibrating ? 0 : pct) + "%";
    rf.className = S.risk > 0.6 ? "crit" : S.risk > 0.33 ? "dys" : "ok";
    setText("#cam-risk-val", S.calibrating ? "cal" : pct);
    setText("#cam-risk-trend", S.calibrating ? "—" : riskTrend());
  }
  const rr = Math.max(S.rr || 0, S.rrBody || 0);
  setText("#cam-rr", rr ? rr.toFixed(0) : "—"); setText("#cam-snr", isFinite(S.snr) && S.snr ? S.snr.toFixed(1) : "—");
  setText("#cam-fps", Math.round(fpsEMA)); setText("#cam-posture", S.posture);
  setText("#cam-track", S.tracking ? "face·" + (S.bodyTracking ? "body·" : "") + (S.handTracking ? "hands" : "") || "locked" : "searching");
  const q = $("#cam-q-fill"); if (q) { q.style.width = (S.quality * 100).toFixed(0) + "%"; setText("#cam-q-val", (S.quality * 100).toFixed(0) + "%"); }
  const rois = $("#cam-rois");
  if (rois) {
    const items = [
      ["face", S.tracking ? Math.round(S.quality * 100) : 0],
      ["neck", S.bodyTracking ? 78 : 0],
      ["torso / shoulders", S.bodyTracking ? 88 : 0],
      ["hands", S.handTracking ? 82 : 0],
    ];
    rois.innerHTML = items.map(([k, pct]) => `<div class="cam-roi ${pct < 25 ? "lost" : ""}"><span class="cam-roi-n">${k}</span><span class="cam-roi-bar"><i style="width:${pct}%"></i></span><span class="cam-roi-v">${pct || "—"}</span></div>`).join("");
  }
  const sc = $("#cam-sign-count"); if (sc) sc.textContent = S.calibrating ? "…" : S.signs.length;
  const sl = $("#cam-signs");
  if (sl) {
    if (S.calibrating) {
      sl.innerHTML = `<div class="muted">calibrating to subject — ${Math.round(S.calibProgress * 100)}% (learning your normal)…</div>`;
    } else {
      let html = S.signs.length
        ? S.signs.map((s) => `<div class="cam-sign ${s.tier}"><i></i><span class="cam-sign-n">${s.name}</span><span class="cam-sign-grade">${s.grade}</span><span class="cam-sign-c">${s.cat} · ${s.conf}%</span></div>`).join("")
        : '<div class="muted">no abnormal signs detected</div>';
      if (S.emerging && S.emerging.length)
        html += S.emerging.map((s) => `<div class="cam-sign emerging"><i></i><span class="cam-sign-n">${s.name}</span><span class="cam-sign-grade">building ${s.conf}%</span></div>`).join("");
      sl.innerHTML = html;
    }
  }
  paintWave("#cam-wave");
}
function paintMini() {
  setText("#mini-hr", S.hr ? S.hr.toFixed(0) : "—");
  const rr = Math.max(S.rr || 0, S.rrBody || 0); setText("#mini-rr", rr ? rr.toFixed(0) : "—");
  setText("#mini-q", S.enabled ? (S.calibrating ? "cal" : (S.quality * 100).toFixed(0) + "%") : "—");
  const sc = $("#mini-signs"); if (sc) sc.textContent = S.calibrating ? "…" : S.signs.filter((s) => s.cat !== "system").length;
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
window.AegisCamera = { open, close, enable, state: snapshot, setSensitivity: (m) => Detector.setSensitivity(m), _detector: Detector };
document.addEventListener("DOMContentLoaded", init);
init();
document.addEventListener("click", (e) => {
  if (e.target.closest("#launch-camera")) open();
  else if (e.target.closest("#cam-close")) close();
  else if (e.target.closest("#cam-enable") || e.target.closest("#mini-enable")) enable();
  else if (e.target.closest("#mini-expand")) open();
  else if (e.target.closest("#cam-sens button")) {
    const btn = e.target.closest("button");
    Detector.setSensitivity(btn.dataset.s);
    document.querySelectorAll("#cam-sens button").forEach((b) => b.classList.toggle("on", b === btn));
  }
});
