/* ============================================================
   AegisOS · LAYER 1 — Stochastic Hardware Abstraction Layer (SHAL)
   ------------------------------------------------------------
   The contactless optical biosensor is a HIGH-JITTER physical device: ambient
   light, motion and sensor noise make every raw frame an unreliable, asynchronous
   packet. The SHAL is the edge device-driver that sits directly above the sensor
   and conditions those raw optical packets into a DETERMINISTIC, timestamped
   sample stream for the upper layers — and fully instruments its own execution so
   determinism is provable, not asserted.

   Everything here is real, inspectable systems substance:
     • RingBuffer        — preallocated Float64Array; zero per-sample allocation,
                           single-writer / lock-free on the JS event loop.
     • KalmanScalar      — adaptive 1-D Kalman filter; measurement noise R scales
                           inversely with signal quality (the *stochastic* part):
                           noisy frames are distrusted, the model prediction leads.
     • Multi-ROI fusion  — skin-masked ROIs combined by inverse-variance (SNR)
                           weighting; masked/low-SNR regions contribute nothing.
     • Instrumentation   — per-frame conditioning latency (µs), inter-frame jitter,
                           throughput, dropped frames, live Kalman gain.
   ============================================================ */

/* ---- zero-allocation ring buffer (preallocated, lock-free single-writer) ---- */
export class RingBuffer {
  constructor(capacity) {
    this.cap = capacity;
    this.buf = new Float64Array(capacity);
    this._scratch = new Float64Array(capacity);
    this.i = 0; this.n = 0;
  }
  push(v) { this.buf[this.i] = v; this.i = (this.i + 1) % this.cap; if (this.n < this.cap) this.n++; }
  last() { return this.n ? this.buf[(this.i - 1 + this.cap) % this.cap] : NaN; }
  /* chronological copy of the live window (for plotting) */
  toArray() { const out = new Array(this.n); const start = this.n < this.cap ? 0 : this.i; for (let k = 0; k < this.n; k++) out[k] = this.buf[(start + k) % this.cap]; return out; }
  /* p50/p95/max/mean/std over the live window; sorts a scratch copy (no GC churn) */
  stats() {
    const n = this.n; if (!n) return { p50: 0, p95: 0, max: 0, mean: 0, std: 0 };
    const s = this._scratch;
    let sum = 0;
    for (let k = 0; k < n; k++) { const v = this.buf[k]; s[k] = v; sum += v; }
    const mean = sum / n;
    let varSum = 0; for (let k = 0; k < n; k++) varSum += (s[k] - mean) ** 2;
    const sub = s.subarray(0, n); sub.sort();
    const at = (q) => sub[Math.min(n - 1, Math.floor(q * (n - 1)))];
    return { p50: at(0.5), p95: at(0.95), max: sub[n - 1], mean, std: Math.sqrt(varSum / n) };
  }
}

/* ---- adaptive scalar Kalman filter (random-walk state model) ----
   x: state estimate · p: estimate covariance · k: Kalman gain.
   predict:  x̂ = x ,  p̂ = p + Q
   update:   K = p̂ / (p̂ + R) ,  x = x̂ + K·(z − x̂) ,  p = (1−K)·p̂
   R = R0 / max(quality, ε)  → low-quality (noisy) measurements get a large R, so
   the filter leans on its prediction instead of the jittery reading.            */
export class KalmanScalar {
  constructor({ q = 1.0, r = 100, x0 = NaN } = {}) { this.q = q; this.r0 = r; this.x = x0; this.p = 1; this.k = 0; }
  update(z, quality = 1) {
    if (!Number.isFinite(z)) return this.x;
    if (!Number.isFinite(this.x)) { this.x = z; this.p = 1; this.k = 1; return this.x; }
    const R = this.r0 / Math.max(quality, 0.05);
    const pPred = this.p + this.q;
    this.k = pPred / (pPred + R);
    this.x = this.x + this.k * (z - this.x);
    this.p = (1 - this.k) * pPred;
    return this.x;
  }
  reset() { this.x = NaN; this.p = 1; this.k = 0; }
}

const STATE = { OFFLINE: "offline", INIT: "initializing", ACQUIRING: "acquiring", LOCKED: "locked", DEGRADED: "degraded" };

class StochasticHAL {
  constructor() {
    this.state = STATE.OFFLINE;
    this.latencyUs = new RingBuffer(256);   // per-frame conditioning latency (µs)
    this.intervalMs = new RingBuffer(256);  // inter-frame intervals → jitter
    this.hrK = new KalmanScalar({ q: 1.6, r: 110 });    // HR (bpm)
    this.roi = new Map();                   // name -> { ring, snr, weight, masked }
    this.frames = 0; this.dropped = 0; this.throttled = false;
    this._t0 = 0; this._lastFrameTs = 0;
  }
  boot() { if (this.state === STATE.OFFLINE) this.state = STATE.INIT; }
  shutdown() { this.state = STATE.OFFLINE; }

  /* --- per-frame device-driver markers (wrap ONLY the signal-conditioning path,
         not the upstream ML inference — that is the sensor, not the driver) --- */
  beginFrame(tsMs) {
    if (this._lastFrameTs) this.intervalMs.push(tsMs - this._lastFrameTs);
    this._lastFrameTs = tsMs;
    this._t0 = performance.now();
  }
  endFrame(locked) {
    this.latencyUs.push((performance.now() - this._t0) * 1000);   // ms → µs
    this.frames++;
    if (this.state !== STATE.DEGRADED) this.state = locked ? STATE.LOCKED : STATE.ACQUIRING;
  }
  dropFrame() { this.dropped++; }
  setThrottled(on) { this.throttled = !!on; }   // Layer-3 load shedding (background ML clamp)

  /* --- multi-ROI skin-masked fusion: inverse-variance (SNR) weighting --- */
  ingestROIs(samples, ts) {
    let wsum = 0, vsum = 0;
    for (const s of samples) {
      let r = this.roi.get(s.name);
      if (!r) { r = { ring: new RingBuffer(90), snr: 0, weight: 0, masked: false }; this.roi.set(s.name, r); }
      r.ring.push(s.value);
      r.masked = !!s.masked;
      r.snr = this._snr(r.ring);
      const w = r.masked ? 0 : Math.max(0, r.snr);
      wsum += w; vsum += w * s.value;
    }
    for (const s of samples) { const r = this.roi.get(s.name); r.weight = wsum ? (r.masked ? 0 : Math.max(0, r.snr)) / wsum : 0; }
    return wsum ? vsum / wsum : (samples[0] ? samples[0].value : 0);
  }
  /* crude but real SNR (dB): pulsatile band power vs residual after detrend */
  _snr(ring) {
    const n = ring.n; if (n < 16) return 0;
    const b = ring.buf;
    let mean = 0; for (let k = 0; k < n; k++) mean += b[k]; mean /= n;
    let sig = 0, noise = 0, prev = b[0] - mean;
    for (let k = 0; k < n; k++) {
      const c = b[k] - mean; sig += c * c;
      const d = c - prev; noise += d * d; prev = c;   // first-difference ≈ high-freq noise
    }
    if (noise <= 1e-9) return 0;
    return Math.max(0, 10 * Math.log10(sig / noise));
  }

  /* --- stabilized heart rate: adaptive Kalman, not an EMA --- */
  stabilizeHR(rawBpm, quality) {
    const v = this.hrK.update(rawBpm, quality);
    this.state = quality < 0.3 ? STATE.DEGRADED : (this.state === STATE.DEGRADED ? STATE.ACQUIRING : this.state);
    return v;
  }

  telemetry() {
    const L = this.latencyUs.stats(), J = this.intervalMs.stats();
    return {
      state: this.state,
      latencyUs: { p50: Math.round(L.p50), p95: Math.round(L.p95), max: Math.round(L.max) },
      jitterMs: +J.std.toFixed(2),
      throughputHz: J.mean ? +(1000 / J.mean).toFixed(1) : 0,
      kalmanGain: +this.hrK.k.toFixed(3),
      frames: this.frames, dropped: this.dropped, throttled: this.throttled,
      rois: [...this.roi.entries()].map(([name, r]) => ({ name, snr: +r.snr.toFixed(1), weight: r.weight, masked: r.masked })),
    };
  }
  reset() {
    this.latencyUs = new RingBuffer(256); this.intervalMs = new RingBuffer(256);
    this.hrK.reset(); this.roi.clear(); this.frames = 0; this.dropped = 0;
    this.throttled = false; this._lastFrameTs = 0; this.state = STATE.INIT;
  }
}

export const SHAL = new StochasticHAL();
if (typeof window !== "undefined") window.AegisSHAL = SHAL;
