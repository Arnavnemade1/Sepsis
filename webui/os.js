/* ============================================================
   AEGIS-OS shell runtime — consumes the v1.2 streaming contract
   ============================================================ */
"use strict";
const $ = (s, r = document) => r.querySelector(s);
const TIER = { NORMAL: "normal", STRESS: "stress", DYSFUNCTION: "dysfunction", CRITICAL: "critical", UNMONITORED: "unmonitored" };
let HR = 72, SPO2 = 98, lastData = null;

/* ---------------- BOOT SEQUENCE ---------------- */
const BOOT_LINES = [
  ["Hardware Abstraction Layer", "device drivers · BLE + camera"],
  ["Hyperdimensional Kernel", "D=10000 · bind/bundle/similarity"],
  ["Digital Twin + What-if Scheduler", "live twin · predictive sim"],
  ["Neurosymbolic Reasoning Plane", "5 agents · debate→consensus"],
  ["Evolutionary Synthesis", "GA protocol optimizer"],
  ["Formal Verification", "Z3 SMT safety prover"],
  ["Persistence", "Supabase / SQLite"],
  ["Interface", "WS + REST snapshot v1.2"],
];
async function boot() {
  const log = $("#boot-log"), fill = $("#boot-fill");
  for (let i = 0; i < BOOT_LINES.length; i++) {
    const [n, d] = BOOT_LINES[i];
    const line = document.createElement("div");
    line.innerHTML = `<span class="ok">[ OK ]</span> layer ${i + 1}/8 · <b>${n}</b> <span style="color:#536084">— ${d}</span>`;
    log.appendChild(line);
    while (log.children.length > 7) log.removeChild(log.firstChild);
    fill.style.width = `${((i + 1) / 8) * 100}%`;
    await sleep(180 + Math.random() * 120);
  }
  await sleep(360);
  $("#boot").classList.add("gone");
  setTimeout(() => { $("#boot").remove(); $("#os").classList.remove("hidden"); window.AegisTwin?.resize(); }, 650);
  connect(); animate(); tickClock(); window.AegisPatient?.load();
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

/* ---------------- WEBSOCKET ---------------- */
let ws, retry = 0, paused = false;
function connect() {
  if (paused || document.hidden) return;   // don't stream to a backgrounded tab
  const url = window.AEGIS.ws();
  ws = new WebSocket(url);
  ws.onopen = () => { retry = 0; setConn(true); };
  ws.onclose = () => { setConn(false); if (paused || document.hidden) return; retry++; setTimeout(connect, Math.min(5000, 600 * retry)); };
  ws.onerror = () => ws.close();
  // Live frames drive the desktop only when viewing the LIVE patient; a selected
  // cohort bed is rendered from its recorded per-bed snapshot instead.
  ws.onmessage = e => { if (viewingBed) return; try { render(JSON.parse(e.data)); } catch (err) { console.error(err); } };
}

/* ---------------- DESKTOP PATIENT SCOPING ---------------- */
let viewingBed = null, bedPollTimer = null;
async function applyPatient(p) {
  if (!p || p.kind === "live") {
    viewingBed = null;
    if (bedPollTimer) { clearInterval(bedPollTimer); bedPollTimer = null; }
    if (!ws || ws.readyState > 1) connect();      // resume live stream to the desktop
    return;
  }
  viewingBed = p.id;
  await loadBedDesktop(p.id);
  if (bedPollTimer) clearInterval(bedPollTimer);
  bedPollTimer = setInterval(() => { if (viewingBed) loadBedDesktop(viewingBed); }, 4000);
}
async function loadBedDesktop(bed) {
  try {
    const d = await fetch(window.AEGIS.http("/aegis/patient/snapshot?bed=" + encodeURIComponent(bed))).then(r => r.json());
    if (d && !d.status) render(d);
  } catch (e) {}
}
document.addEventListener("aegis:patient", e => applyPatient(e.detail));
// Pause the live feed on hidden tabs (saves continuous backend egress), resume on return.
document.addEventListener("visibilitychange", () => {
  if (document.hidden) { paused = true; try { ws && ws.close(); } catch (e) {} }
  else { paused = false; if (!ws || ws.readyState > 1) connect(); }
});
function setConn(live) {
  const c = $("#sb-conn"); c.classList.toggle("live", live);
  c.lastChild.textContent = live ? "linked" : (retry >= 2 ? "waking backend…" : "connecting");
}

/* ---------------- MASTER RENDER ---------------- */
function render(d) {
  lastData = d;
  HR = num(d.vitals?.heart_rate, HR); SPO2 = num(d.vitals?.spo2, SPO2);
  renderSysbar(d); renderRisk(d); renderTwin(d); renderVitals(d);
  renderAgents(d); renderWhatif(d); renderProto(d); renderLayers(d);
  window.AegisStory?.onSnapshot(d);
  window.AegisConsensus?.onSnapshot(d);
  $("#dock-tick").textContent = `tick · ${d.source || "—"} · ${new Date((d.timestamp||0)*1000).toLocaleTimeString()}`;
}
const num = (v, f = 0) => (typeof v === "number" && !isNaN(v)) ? v : f;
const pct = v => `${Math.round(num(v) * 100)}%`;

/* ---------------- SYSTEM BAR ---------------- */
function renderSysbar(d) {
  const r = d.risk?.level || "LOW";
  const rb = $("#sb-risk"); rb.textContent = r; rb.className = "sysbar-risk " + r;
  const src = d.source || "idle";
  const sb = $("#sb-source");
  const live = src === "camera" || (src !== "simulator" && src !== "idle");
  sb.textContent = src === "camera" ? "LIVE · CAMERA" : src === "simulator" ? "SIMULATION · DEMO" : src.toUpperCase();
  sb.classList.toggle("live", live);
  $("#sb-ver").textContent = "v" + (d.system?.version || "0.8.0");
  // patient identity is owned by the AegisPatient selector (top-bar).
}

/* ---------------- RISK CORE ---------------- */
function renderRisk(d) {
  const prob = num(d.risk?.ml_probability ?? d.body_state?.sepsis_probability) || 0;
  const lvl = d.risk?.level || "LOW";
  countUpProb($("#risk-prob"), prob * 100, prob > 0);
  drawArc(prob, lvl);
  $("#risk-gauge")?.classList.toggle("is-high", lvl === "HIGH");
  const badge = $("#risk-tier-badge");
  if (badge) { badge.textContent = lvl; badge.className = "risk-tier-badge " + lvl; }
  $("#risk-bands")?.querySelectorAll(".rb-seg").forEach(s => s.classList.toggle("on", s.dataset.t === lvl));
  $("#rs-sirs").textContent = d.risk?.sirs_score ?? 0;
  $("#rs-qsofa").textContent = d.risk?.qsofa_score ?? 0;
  $("#rs-sq").textContent = pct(d.risk?.signal_quality);
  $("#twin-risk").textContent = lvl === "LOW" ? "monitoring" : lvl.toLowerCase() + " risk";
}
// Animated, well-formatted readout: large integer + smaller ".d%" decimal.
function countUpProb(el, to, has) {
  if (!el) return;
  const render = v => { el.innerHTML = has
    ? `<span class="rp-int">${Math.floor(v)}</span><span class="rp-dec">.${Math.round((v - Math.floor(v)) * 10)}%</span>`
    : `<span class="rp-int">—</span>`; };
  const from = parseFloat(el.dataset.val || "0");
  el.dataset.val = to;
  if (!has) { render(0); return; }
  if (Math.abs(from - to) < 0.05) { render(to); return; }
  const t0 = performance.now(), dur = 650;
  const tick = now => {
    const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
    render(from + (to - from) * e);
    if (k < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

// Tier palette — a clinical traffic-light read (green → amber → red), matched to
// the tier badge + band meter so the whole instrument agrees. In-theme, no pink.
const RISK_THEME = {
  LOW:    { a: "#39d9a2", b: "#3fb8c8", glow: "57,217,162" },
  MEDIUM: { a: "#f5c451", b: "#f59a4b", glow: "245,196,81" },
  HIGH:   { a: "#f2606e", b: "#ef5d8f", glow: "242,96,110" },
};
// 270° instrument sweep: a 90° gap centred at the bottom.
const ARC_START = -135, ARC_SWEEP = 270;
const polar = (cx, cy, r, deg) => {
  const a = (deg - 90) * Math.PI / 180;     // 0° at 12 o'clock, clockwise
  return [cx + r * Math.cos(a), cy + r * Math.sin(a)];
};
function arcPath(cx, cy, r, a0, a1) {
  const [x0, y0] = polar(cx, cy, r, a0), [x1, y1] = polar(cx, cy, r, a1);
  const large = (a1 - a0) > 180 ? 1 : 0;
  return `M ${x0.toFixed(2)} ${y0.toFixed(2)} A ${r} ${r} 0 ${large} 1 ${x1.toFixed(2)} ${y1.toFixed(2)}`;
}

const RG = { cx: 60, cy: 60, r: 50, built: false };
// Build the static gauge chrome ONCE; subsequent renders only nudge the live bits,
// so the fill sweep + dot glide animate smoothly instead of snapping each tick.
function buildGauge() {
  const svg = $("#risk-arc"); if (!svg) return;
  const { cx, cy, r } = RG;
  let ticks = "";
  for (let i = 0; i <= 30; i++) {
    const deg = ARC_START + (ARC_SWEEP / 30) * i, major = i % 5 === 0;
    const [x1, y1] = polar(cx, cy, 57, deg);
    const [x2, y2] = polar(cx, cy, major ? 52.5 : 54.5, deg);
    ticks += `<line x1="${x1.toFixed(2)}" y1="${y1.toFixed(2)}" x2="${x2.toFixed(2)}" y2="${y2.toFixed(2)}"
      stroke="rgba(150,175,215,${major ? .34 : .16})" stroke-width="${major ? 1.4 : 1}"/>`;
  }
  const marker = (frac, color) => {
    const [mx, my] = polar(cx, cy, r, ARC_START + ARC_SWEEP * frac);
    return `<circle cx="${mx.toFixed(2)}" cy="${my.toFixed(2)}" r="2.1" fill="${color}"
      stroke="rgba(6,8,13,.9)" stroke-width="1"/>`;
  };
  const fullPath = arcPath(cx, cy, r, ARC_START, ARC_START + ARC_SWEEP);
  svg.innerHTML = `
    <defs>
      <linearGradient id="rg" gradientUnits="userSpaceOnUse" x1="10" y1="100" x2="110" y2="20">
        <stop id="rg0" offset="0" stop-color="#5ccaf0"/><stop id="rg1" offset="1" stop-color="#6f8cf5"/></linearGradient>
      <radialGradient id="rgCore" cx="50%" cy="50%" r="55%">
        <stop id="rgc0" offset="0" stop-color="rgba(92,202,240,.16)"/>
        <stop offset="100%" stop-color="rgba(92,202,240,0)"/></radialGradient>
    </defs>
    <circle id="rg-core" cx="${cx}" cy="${cy}" r="46" fill="url(#rgCore)"/>
    <g class="risk-ticks">${ticks}</g>
    <path d="${fullPath}" fill="none" stroke="rgba(140,165,210,.12)" stroke-width="7" stroke-linecap="round"/>
    <path id="rg-fill" class="risk-arc-fill" d="${fullPath}" pathLength="1000" fill="none"
      stroke="url(#rg)" stroke-width="7" stroke-linecap="round"
      stroke-dasharray="1000" stroke-dashoffset="1000"/>
    ${marker(0.04, "#f5c451")}${marker(0.12, "#f2606e")}
    <circle id="rg-dot" class="risk-arc-dot" r="4.4" fill="#fff" opacity="0"/>`;
  RG.built = true;
}
function drawArc(p, lvl) {
  if (!RG.built) buildGauge();
  const fill = $("#rg-fill"), dot = $("#rg-dot"), core = $("#rgc0");
  if (!fill) return;
  const f = Math.max(0, Math.min(1, p));        // HONEST: fill tracks the probability
  const th = RISK_THEME[lvl] || RISK_THEME.LOW;
  $("#risk-gauge")?.style.setProperty("--halo", th.glow);
  fill.style.strokeDashoffset = String(1000 * (1 - f));
  fill.style.filter = `drop-shadow(0 0 7px rgba(${th.glow},.85))`;
  $("#rg0")?.setAttribute("stop-color", th.a);
  $("#rg1")?.setAttribute("stop-color", th.b);
  core?.setAttribute("stop-color", `rgba(${th.glow},.16)`);
  const [dx, dy] = polar(RG.cx, RG.cy, RG.r, ARC_START + ARC_SWEEP * f);
  dot.setAttribute("cx", dx.toFixed(2));
  dot.setAttribute("cy", dy.toFixed(2));
  dot.setAttribute("opacity", f > 0.003 ? "1" : "0");
  dot.style.filter = `drop-shadow(0 0 7px rgba(${th.glow},1))`;
}

/* ---------------- BODY TWIN (3D — see twin3d.js) ---------------- */
function renderTwin(d) {
  window.AegisTwin?.update(d);
}

/* ---------------- VITALS ---------------- */
function renderVitals(d) {
  $("#t-hr").textContent = Math.round(HR);
  $("#t-spo2").textContent = Math.round(SPO2);
  const ch = d.body_state?.channels || [];
  const SHORT = { heart_rate: "HR", hrv: "HRV", respiratory_rate: "Resp", map: "MAP",
    sweat_lactate: "Lactate", skin_temp: "Temp", spo2: "SpO₂", systolic_bp: "SBP" };
  const show = ["heart_rate", "hrv", "respiratory_rate", "map", "sweat_lactate", "skin_temp"];
  $("#vital-grid").innerHTML = ch.filter(c => show.includes(c.key)).map(c => `
    <div class="vchip ${c.status}">
      <span class="vk">${SHORT[c.key] || c.label}<div class="src">${c.source}</div></span>
      <span><span class="vv">${c.value ?? "—"}</span><span class="vu">${c.unit}</span></span>
    </div>`).join("");
}

/* ---------------- AGENTS ---------------- */
function renderAgents(d) {
  const r = d.reasoning; if (!r) return;
  $("#ag-consensus").textContent = `consensus ${(r.consensus * 100 | 0)}% · agree ${(r.agreement * 100 | 0)}%`;
  $("#agent-list").innerHTML = r.opinions.map(o => {
    const hot = o.concern >= 0.5;
    const ev = (o.evidence || []).map(e => `<span class="tag">${e}</span>`).join("");
    return `<div class="agent ${hot ? "hot" : ""}">
      <div class="agent-head"><span class="agent-name">${o.agent}</span>
        <span class="agent-conf">conf ${(o.confidence*100|0)}%</span></div>
      <div class="agent-bar"><div class="agent-fill" style="width:${(o.concern*100)|0}%"></div></div>
      <div class="agent-ev">${o.rationale}${ev ? "<br>" + ev : ""}</div></div>`;
  }).join("");
  const s = r.safety || {}; const crit = s.critical;
  $("#oracle").className = "oracle" + (crit ? " crit" : "");
  $("#oracle").innerHTML = `<h5>⛨ Safety Oracle ${s.escalate_to ? "→ escalate " + s.escalate_to : "· clear"}</h5>
    ${(s.flags || []).map(f => `<span class="flag">▲ ${f}</span>`).join("") || '<span class="muted">no hard danger criteria</span>'}`;
}

/* ---------------- WHAT-IF ---------------- */
function renderWhatif(d) {
  const w = d.whatif;
  const box = $("#whatif-list");
  if (!w || !w.candidates) { box.innerHTML = '<div class="muted">No active projection. Engages on HIGH risk.</div>'; return; }
  box.innerHTML = w.candidates.map(c => {
    const best = c.name === w.recommended, s = c.summary || {};
    return `<div class="cand ${best ? "best" : ""} ${c.safe ? "" : "unsafe"}">
      <div class="cand-head"><span class="cand-name">${c.name.replace(/_/g, " ")}</span>
        ${best ? '<span class="cand-badge rec">recommended</span>' :
          `<span class="cand-badge ${c.safe ? "safe" : "no"}">${c.safe ? "Z3 safe" : "rejected"}</span>`}</div>
      <div class="cand-spark"><canvas data-traj='${JSON.stringify((c.trajectory||[]).map(t=>t.projected_risk))}'></canvas></div>
      <div class="cand-meta"><span>final risk ${(num(s.final_risk)*100|0)}%</span><span>MAP→${s.final_map ?? "—"}</span></div>
    </div>`;
  }).join("");
  box.querySelectorAll("canvas[data-traj]").forEach(cv => sparkline(cv, JSON.parse(cv.dataset.traj)));
}
function sparkline(cv, arr) {
  const w = cv.clientWidth || 280, h = 30; cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d"); ctx.clearRect(0, 0, w, h); if (!arr.length) return;
  ctx.beginPath(); ctx.lineWidth = 1.6; ctx.strokeStyle = "#34e1ff";
  arr.forEach((v, i) => { const x = i / (arr.length - 1) * w, y = h - 2 - v * (h - 4); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke();
}

/* ---------------- PROTOCOL + PROOF ---------------- */
function renderProto(d) {
  const p = d.protocol, v = d.verification, box = $("#proto");
  if (!p) { box.innerHTML = '<div class="muted">Awaiting high-risk synthesis…</div>' + narr(d); return; }
  const t = p.protocol || {};
  const safe = v?.safe;
  box.innerHTML = `
    <div class="rx">
      <div>fluids<b>${t.fluid_rate ?? "—"}</b>mL/kg/h</div>
      <div>vaso<b>${t.vasopressor_dose ?? "—"}</b>µg/kg/m</div>
      <div>antibiotic<b>#${t.antibiotic_code ?? "—"}</b></div>
    </div>
    <div class="proof ${safe ? "" : "bad"}"><i>${safe ? "✓" : "✕"}</i>
      Z3 ${v?.z3_status || "—"} · ${safe ? "mathematically certified safe" : (v?.message || "rejected")}</div>
    ${portableProofLine(v?.portable_proof)}
    ${safe ? `<button class="dock-btn accent pulse-button" onclick="window.AegisTwin.triggerInfusion('fluids'); setTimeout(() => window.AegisTwin.triggerInfusion('antibiotic'), 1200); setTimeout(() => window.AegisTwin.triggerInfusion('vasopressor'), 2400);" style="width:100%;margin-top:8px;animation: pulse 2s infinite;cursor:pointer;background:#00d8ff;color:#0b1020;font-weight:bold;border:none;border-radius:4px;padding:8px 0;box-shadow: 0 0 10px rgba(0,216,255,0.4);">▲ APPROVE & INFUSE PROTOCOL</button>` : ""}
    ${narr(d)}`;
  wireProofVerify();
}

/* Compact portable-proof affordance for the live core-cage decision. */
function portableProofLine(pp) {
  if (!pp || !pp.available) return "";
  return `<div class="pproof" id="pproof">
    <div class="pproof-h">
      <span class="pproof-badge ${pp.all_satisfied ? "ok" : "bad"}">⚿ portable proof · ${pp.n_satisfied ?? pp.n_obligations}/${pp.n_obligations} invariants</span>
      <span class="pproof-key mono">Ed25519 ${pp.key_id || "—"}</span>
    </div>
    <div class="pproof-actions">
      <button class="pproof-btn" id="pproof-verify">⟳ re-verify independently</button>
      <a class="pproof-dl" href="${window.AEGIS.http('/aegis/proof/core/latest/bundle')}" download>⤓ audit bundle</a>
    </div>
    <div class="pproof-result" id="pproof-result"></div>
  </div>`;
}

function wireProofVerify() {
  const btn = document.querySelector("#pproof-verify");
  if (!btn) return;
  btn.onclick = async () => {
    const out = document.querySelector("#pproof-result");
    btn.disabled = true; btn.textContent = "re-running solver…";
    try {
      const latest = await fetch(window.AEGIS.http("/aegis/proof/core/latest")).then(r => r.json());
      if (latest.status !== "ok") { out.textContent = "no certificate to verify"; }
      else {
        const a = await fetch(window.AEGIS.http("/aegis/proof/verify"), {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ certificate: latest.certificate }) }).then(r => r.json());
        out.className = "pproof-result " + (a.trusted ? "ok" : "bad");
        out.innerHTML = `<b class="${a.trusted ? "ok" : "bad"}">${a.trusted ? "VERIFIED" : "REJECTED"}</b>`
          + ` · math ${a.math_ok ? "✓" : "✕"} · untampered ${a.integrity_ok ? "✓" : "✕"} · signature ${a.signature_ok ? "✓" : "✕"}`;
      }
    } catch (e) { out.className = "pproof-result bad"; out.textContent = "audit failed: " + e; }
    btn.disabled = false; btn.textContent = "⟳ re-verify independently";
  };
}
const narr = d => d.narrative ? `<div class="narr">${d.narrative}</div>` : "";


/* ---------------- LAYERS ---------------- */
function renderLayers(d) {
  const L = d.system?.layers; if (!L) return;
  $("#layers").innerHTML = L.map(l =>
    `<div class="layer"><i class="${l.ready ? "" : "off"}"></i><span class="lx">${l.index}</span>
      <b>${l.name}</b></div>`).join("");
}

/* ---------------- ANIMATION (ECG / PPG / heartbeat) ---------------- */
let ecgBuf = [], ppgBuf = [], ecgPhase = 0, ppgPhase = 0, last = performance.now();
function ecgY(t) { // t in [0,1)
  if (t < .12) return .12 * Math.sin(t / .12 * Math.PI);          // P
  if (t < .18) return 0;
  if (t < .21) return -.18 * ((t - .18) / .03);                   // Q
  if (t < .25) return 1.0 * ((t - .21) / .04);                    // R up
  if (t < .29) return 1.0 - 1.25 * ((t - .25) / .04);             // R down (to -.25)
  if (t < .33) return -.25 + .25 * ((t - .29) / .04);             // S back
  if (t < .5) return .28 * Math.sin((t - .33) / .17 * Math.PI);   // T
  return 0;
}
function animate() {
  const now = performance.now(), dt = Math.min(.05, (now - last) / 1000); last = now;
  const W = $("#ecg").clientWidth || 320;
  [["#ecg", ecgBuf, 74], ["#ppg", ppgBuf, 58]].forEach(([id]) => { const c = $(id); if (c) { c.width = W; } });
  // advance phases by heart rate
  const beats = HR / 60 * dt; ecgPhase = (ecgPhase + beats);
  ppgPhase = (ppgPhase + beats);
  const step = Math.max(2, Math.round(W / 260));
  for (let k = 0; k < step; k++) {
    ecgBuf.push(ecgY((ecgPhase + k * beats / step) % 1));
    const pp = (ppgPhase + k * beats / step) % 1;
    ppgBuf.push(0.5 * Math.sin(pp * 2 * Math.PI - 1.2) + 0.18 * Math.sin(pp * 4 * Math.PI));
  }
  while (ecgBuf.length > W) ecgBuf.shift();
  while (ppgBuf.length > W) ppgBuf.shift();
  drawTrace("#ecg", ecgBuf, "#5ccaf0", 74);
  drawTrace("#ppg", ppgBuf, "#39d9a2", 58, true);
  requestAnimationFrame(animate);
}
function drawTrace(id, buf, color, h) {
  const c = $(id); if (!c) return; const ctx = c.getContext("2d"), W = c.width;
  ctx.clearRect(0, 0, W, h); if (!buf.length) return;
  const yOf = v => h * 0.55 - v * h * 0.42;
  // gradient area fill under the waveform
  ctx.beginPath();
  buf.forEach((v, i) => { i ? ctx.lineTo(i, yOf(v)) : ctx.moveTo(i, yOf(v)); });
  ctx.lineTo(W, h); ctx.lineTo(0, h); ctx.closePath();
  const g = ctx.createLinearGradient(0, 0, 0, h);
  g.addColorStop(0, color + "30"); g.addColorStop(1, color + "00");
  ctx.fillStyle = g; ctx.fill();
  // glowing stroke on top
  ctx.beginPath(); ctx.lineWidth = 1.8; ctx.strokeStyle = color; ctx.shadowBlur = 7; ctx.shadowColor = color;
  buf.forEach((v, i) => { i ? ctx.lineTo(i, yOf(v)) : ctx.moveTo(i, yOf(v)); });
  ctx.stroke(); ctx.shadowBlur = 0;
}

/* ---------------- DOCK / CLOCK ---------------- */
document.addEventListener("click", e => {
  const b = e.target.closest(".dock-btn"); if (!b) return;
  if (b.dataset.stage) fetch(window.AEGIS.http(`/aegis/simulate/${b.dataset.stage}`), { method: "POST" }).catch(() => {});
  b.animate([{ transform: "scale(.92)" }, { transform: "scale(1)" }], { duration: 180 });
});
function tickClock() {
  $("#sb-clock").textContent = new Date().toLocaleTimeString("en-GB");
  setTimeout(tickClock, 1000);
}

boot();

/* ---------------- CLINICAL STORY (temporal causal graph) ---------------- */
/* ---------------- PATIENT SELECTOR (scopes the patient-facing views) ---------------- */
window.AegisPatient = (function () {
  const api = p => fetch(window.AEGIS.http(p)).then(r => r.json());
  const KEY = "aegis_patient";
  let list = [], cur = null;

  function current() { return cur; }
  function bed() { return cur ? cur.id : null; }     // the ?bed= scope param

  async function load() {
    try { const d = await api("/aegis/patients"); list = d.patients || []; }
    catch (e) { return; }
    const saved = localStorage.getItem(KEY);
    cur = list.find(p => p.id === saved)
       || list.find(p => p.kind === "cohort" && p.septic)
       || list[0] || null;
    renderTrigger(); renderMenu();
    if (cur) emit();
  }
  function set(id) {
    const p = list.find(x => x.id === id); if (!p) return;
    cur = p; localStorage.setItem(KEY, id);
    renderTrigger(); renderMenu(); close(); emit();
  }
  function emit() { document.dispatchEvent(new CustomEvent("aegis:patient", { detail: cur })); }

  function dotClass(p) { return p.kind === "live" ? "live" : (p.septic ? "septic" : "stable"); }
  function renderTrigger() {
    if (!cur) return;
    $("#patient-name").textContent = cur.name;
    $("#patient-meta").textContent = cur.kind === "live" ? "live monitor"
      : `bed ${cur.id.replace("bed_", "")}${cur.septic ? " · septic course" : " · stable course"}`;
    $("#patient-dot").className = "patient-dot " + dotClass(cur);
    // dynamic-island state: the capsule's ring/glow reflects the patient's state
    const trg = $("#patient-trigger");
    if (trg) { trg.className = "patient-trigger " + dotClass(cur); trg.classList.add("island-pop");
      setTimeout(() => trg.classList.remove("island-pop"), 480); }
  }
  function renderMenu() {
    $("#patient-menu").innerHTML = list.map(p => `
      <button class="patient-opt ${cur && cur.id === p.id ? "active" : ""}" data-pid="${p.id}" role="option">
        <span class="patient-opt-dot ${dotClass(p)}"></span>
        <span class="patient-opt-name">${p.name}</span>
        <span class="patient-opt-tag ${dotClass(p)}">${p.kind === "live" ? "live" : (p.septic ? "septic" : "stable")}</span>
      </button>`).join("");
  }
  function open() { $("#patient-menu").classList.remove("hidden"); $("#patient-trigger").setAttribute("aria-expanded", "true"); }
  function close() { $("#patient-menu").classList.add("hidden"); $("#patient-trigger").setAttribute("aria-expanded", "false"); }

  document.addEventListener("click", e => {
    if (e.target.closest("#patient-trigger")) {
      $("#patient-menu").classList.contains("hidden") ? open() : close();
      return;
    }
    const opt = e.target.closest(".patient-opt");
    if (opt) { set(opt.dataset.pid); return; }
    if (!e.target.closest("#patient-select")) close();
  });

  return { load, current, set, bed };
})();

window.AegisStory = (function () {
  const KIND = {
    vital_snapshot:   { i: "◇", c: "vital",   t: "vital" },
    lab_result:       { i: "🧪", c: "vital",   t: "lab" },
    drug_dose:        { i: "💉", c: "decision",t: "drug" },
    protocol_decision:{ i: "✚", c: "decision",t: "decision" },
    omission:         { i: "⚠", c: "omission", t: "omission" },
    alert:            { i: "▲", c: "alert",    t: "alert" },
    outcome:          { i: "◉", c: "outcome",  t: "outcome" },
  };
  const EV = { z3_proven: "proven", model_predicted: "predicted", observed_association: "association" };
  let lastGraph = null, reportNode = null, openFlag = false;
  // Epoch timestamps (live) render as clock time; synthetic small t renders relative.
  const fmtT = t => (t > 1e9)
    ? new Date(t * 1000).toLocaleTimeString("en-GB")
    : `t+${Math.round(t)}s`;

  const api = (p, opt) => fetch(window.AEGIS.http(p), opt).then(r => r.json());

  function open() {
    openFlag = true;
    document.querySelector("#story-app").classList.remove("hidden");
    refresh();
  }
  function close() { openFlag = false; document.querySelector("#story-app").classList.add("hidden"); }

  const _pt = () => window.AegisPatient?.current();
  const _isLive = () => { const p = _pt(); return !p || p.kind === "live"; };
  const _bedQ = (j = "?") => { const b = window.AegisPatient?.bed(); return b ? j + "bed=" + encodeURIComponent(b) : ""; };

  function onSnapshot(d) {
    const cg = d.causal_graph; if (!cg) return;
    const pill = document.querySelector("#story-count"); if (pill) pill.textContent = cg.nodes ?? 0;
    // The live snapshot only carries the live patient's graph; only auto-refresh
    // the open overlay when we're actually viewing that patient.
    if (openFlag && _isLive()) refresh();
  }

  function onPatient() {
    setHeader();
    reportNode = null;
    const ib = document.querySelector("#story-incident-out");
    const rb = document.querySelector("#story-replay-out");
    if (ib) ib.innerHTML = '<div class="muted">Trace back from the latest event to the earliest point an intervention could have changed the trajectory.</div>';
    if (rb) rb.innerHTML = '<div class="muted">Run an incident report first, then replay an alternative action.</div>';
    const rbtn = document.querySelector("#story-replay"); if (rbtn) rbtn.disabled = true;
    if (openFlag) refresh();
  }
  function setHeader() {
    const el = document.querySelector("#story-patient"); if (!el) return;
    const p = _pt();
    el.textContent = p ? p.name : "—";
    el.className = "story-patient " + (p ? (p.kind === "live" ? "live" : (p.septic ? "septic" : "stable")) : "");
  }

  async function refresh() {
    setHeader();
    let data; try { data = await api("/aegis/causal" + _bedQ()); } catch (e) { return; }
    if (data.status === "error") return;
    lastGraph = data.graph; renderLedger(data);
  }

  function renderLedger(data) {
    const g = data.graph, sum = data.summary || {};
    const intg = document.querySelector("#story-integrity");
    const ok = g.integrity_ok;
    intg.className = "cam-state" + (ok ? " live" : " bad");
    intg.lastChild.textContent = ok ? `chain verified · ${g.head}` : "INTEGRITY FAIL";
    document.querySelector("#story-ledger-sub").textContent =
      `${g.n_nodes} events · ${g.n_edges} edges`;

    const edgesByDst = {};
    (g.edges || []).forEach(e => { (edgesByDst[e.dst] ||= []).push(e); });
    const tl = document.querySelector("#story-timeline");
    if (!g.nodes || !g.nodes.length) {
      tl.innerHTML = '<div class="muted">No clinical events yet. The graph grows as risk evolves.</div>';
      return;
    }
    tl.innerHTML = g.nodes.map(n => {
      const k = KIND[n.kind] || { i: "•", c: "vital", t: n.kind };
      const inc = (edgesByDst[n.id] || []).map(e =>
        `<span class="story-edge ${e.evidence_type}">${e.mechanism} · ${EV[e.evidence_type] || e.evidence_type}` +
        (e.support ? ` (n=${e.support})` : "") + `</span>`).join("");
      const risk = Math.round((n.risk || 0) * 100);
      return `<div class="story-node ${k.c}" data-id="${n.id}">
        <div class="story-node-h"><span class="story-ico">${k.i}</span>
          <span class="story-kind">${k.t}</span>
          <span class="story-risk">risk ${risk}%</span>
          <span class="story-t">${fmtT(n.t)}</span></div>
        ${inc ? `<div class="story-edges">${inc}</div>` : ""}
        ${summaryLine(n)}
      </div>`;
    }).join("");
    // clicking a node makes it the incident anchor
    tl.querySelectorAll(".story-node").forEach(el =>
      el.onclick = () => runIncident(el.dataset.id));
  }

  function summaryLine(n) {
    const p = n.payload || {};
    if (n.kind === "vital_snapshot")
      return `<div class="story-pay">MAP ${p.map ?? "—"} · lactate ${p.lactate ?? p.sweat_lactate ?? "—"}</div>`;
    if (n.kind === "omission")
      return `<div class="story-pay">indicated: ${p.indicated || "treatment"} — not given</div>`;
    if (n.kind === "protocol_decision")
      return `<div class="story-pay">Z3 ${p.z3_status || "certified"} · protocol applied</div>`;
    if (n.kind === "outcome")
      return `<div class="story-pay">${p.state || "outcome"}</div>`;
    if (n.kind === "alert")
      return `<div class="story-pay">${p.risk_level || ""} ${(p.triggered_rules || []).slice(0,3).join(", ")}</div>`;
    return "";
  }

  async function runIncident(nodeId) {
    if (!lastGraph || !lastGraph.nodes?.length) return;
    // default anchor: an outcome if present, else the most recent node
    if (!nodeId) {
      const out = [...lastGraph.nodes].reverse().find(n => n.kind === "outcome");
      nodeId = out ? out.id : lastGraph.nodes[lastGraph.nodes.length - 1].id;
    }
    const box = document.querySelector("#story-incident-out");
    box.innerHTML = '<div class="muted">tracing…</div>';
    const rep = await api(`/aegis/causal/incident/${nodeId}` + _bedQ());
    if (rep.status === "error") { box.innerHTML = `<div class="muted">${rep.detail}</div>`; return; }
    reportNode = nodeId;
    const ea = rep.earliest_actionable;
    const mix = rep.evidence_mix || {};
    box.innerHTML = `
      <div class="story-line"><b>anchor</b> ${rep.outcome.kind} @ ${fmtT(rep.outcome.t)}</div>
      ${ea ? `<div class="story-flag">⚑ earliest actionable: <b>${ea.kind}</b> @ ${fmtT(ea.t)}
           · lead time <b>${fmtLead(rep.lead_time_s)}</b></div>` :
           `<div class="muted">no actionable node upstream</div>`}
      <div class="story-path">${(rep.decision_path || []).map(n =>
        `<span class="story-chip ${KIND[n.kind]?.c||""}">${n.kind}</span>`).join(" → ") || "—"}</div>
      <div class="story-mix">
        <span class="story-edge z3_proven">${mix.z3_proven||0} proven</span>
        <span class="story-edge model_predicted">${mix.model_predicted||0} predicted</span>
        <span class="story-edge observed_association">${mix.observed_association||0} assoc</span>
      </div>
      <div class="story-foot">${rep.integrity_ok ? "✓ chain verified" : "✕ integrity fail"} ·
        ${rep.events_considered} events traversed</div>`;
    const rb = document.querySelector("#story-replay");
    rb.disabled = !ea; rb.dataset.outcome = nodeId;
  }

  async function runReplay() {
    if (!reportNode) return;
    const box = document.querySelector("#story-replay-out");
    box.innerHTML = '<div class="muted">projecting…</div>';
    const res = await api("/aegis/causal/replay", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ outcome_id: reportNode, bed: window.AegisPatient?.bed() || null,
        treatment: { fluid_rate: 5.0, vasopressor_dose: 0.0, antibiotic_code: 3 } }),
    });
    if (res.status === "error") { box.innerHTML = `<div class="muted">${res.detail}</div>`; return; }
    const cf = res.counterfactual, fb = res.factual_baseline, dl = res.delta, mf = res.model_fidelity || {};
    const good = dl.diverges_toward_recovery;
    box.innerHTML = `
      <div class="story-line"><b>branch</b> ${res.branch.kind} @ ${fmtT(res.branch.t)}
        · MAP ${res.branch_vitals.map ?? "—"}</div>
      <div class="story-compare">
        <div class="story-col"><span>action taken</span>
          <b>risk ${(fb.final_risk*100|0)}%</b><i>MAP ${fb.final_map}</i></div>
        <div class="story-arrow ${good ? "good" : ""}">→</div>
        <div class="story-col cf"><span>+ antibiotics & fluids</span>
          <b>risk ${(cf.final_risk*100|0)}%</b><i>MAP ${cf.final_map}</i></div>
      </div>
      <div class="story-verdict ${good ? "good" : "neutral"}">
        ${good ? "▼ projected to diverge toward recovery" : "≈ no material divergence projected"}
        · Δrisk ${(dl.final_risk*100|0)}% · ΔMAP ${dl.final_map>0?"+":""}${dl.final_map}</div>
      <div class="story-conf">model confidence ${(cf.confidence*100|0)}% (decays with horizon)</div>
      <div class="story-caveat">${mf.note ? mf.note + " · " : ""}illustrative model projection — not a validated outcome forecast</div>`;
  }

  async function runMine() {
    const box = document.querySelector("#story-mine-out");
    box.innerHTML = '<div class="muted">mining cohort…</div>';
    const res = await api("/aegis/causal/patterns?min_support=3&min_lift=1.2");
    const rules = res.rules || [];
    box.innerHTML = `
      <div class="story-line">cohort <b>${res.cohort_size}</b> patients ·
        outcome: ${res.outcome} · ${res.candidates_tested} candidates tested</div>
      ${rules.length ? rules.map(r => `
        <div class="story-rule">
          <div class="story-rule-h"><span class="story-rule-a">${r.antecedent}</span>
            <span class="story-lift">lift ${r.lift}×</span></div>
          <div class="story-rule-bar">
            <div class="story-rule-fill" style="width:${Math.min(100, r.confidence*100)}%"></div>
            <div class="story-rule-base" style="left:${Math.min(100, r.baseline*100)}%"></div>
          </div>
          <div class="story-rule-m">${(r.confidence*100|0)}% (CI ${(r.ci_low*100|0)}–${(r.ci_high*100|0)}%)
            · n=${r.n_antecedent} · baseline ${(r.baseline*100|0)}%
            <span class="story-edge observed_association">association</span></div>
        </div>`).join("") :
        `<div class="muted">No associations above threshold yet — the cohort grows as
          more patient histories are recorded.</div>`}
      <div class="story-caveat">${res.disclaimer || ""}</div>`;
  }

  function fmtLead(s) {
    if (s == null) return "—";
    const m = Math.round(s / 60);
    return m >= 60 ? `${(m/60).toFixed(1)} h` : `${m} min`;
  }

  // wiring
  document.addEventListener("click", e => {
    if (e.target.closest("#launch-story")) open();
    if (e.target.closest("#story-close")) close();
    if (e.target.closest("#story-refresh")) refresh();
    if (e.target.closest("#story-incident")) runIncident(null);
    if (e.target.closest("#story-replay")) runReplay();
    if (e.target.closest("#story-mine")) runMine();
  });
  document.addEventListener("aegis:patient", onPatient);

  return { open, close, onSnapshot };
})();

/* ---------------- DISTRIBUTED CONSENSUS BRAIN (BFT specialist consensus) ---------------- */
window.AegisConsensus = (function () {
  const api = (p) => fetch(window.AEGIS.http(p)).then(r => r.json());
  const SPEC = { Intensivist: "🫁", Cardiology: "❤", Nephrology: "🫘", InfectiousDisease: "🦠" };

  const SPECNAME = { Intensivist: "Critical Care", Cardiology: "Cardiology",
                     Nephrology: "Nephrology", InfectiousDisease: "Infectious Disease" };
  const SPECFOCUS = { Intensivist: "hemodynamics & perfusion", Cardiology: "cardiac & pressor safety",
                      Nephrology: "renal & fluid balance", InfectiousDisease: "source control & antimicrobials" };
  let openFlag = false, manual = false;
  function open() {
    openFlag = true; manual = false;
    document.querySelector("#consensus-app").classList.remove("hidden");
    conveneLive();                       // review the live patient immediately, no clicks
  }
  function close() { openFlag = false; document.querySelector("#consensus-app").classList.add("hidden"); }

  // Convene the panel on the LIVE patient: no deterioration param → the backend
  // derives the real deadline from the patient's risk band; all sources trusted.
  async function conveneLive() {
    manual = false;
    const pt = window.AegisPatient?.current();
    document.querySelector("#cns-patient").textContent = pt?.name || lastData?.patient?.name || "this patient";
    const spec = document.querySelector("#cns-specialists");
    spec.innerHTML = '<div class="muted">convening the panel on the live patient…</div>';
    try {
      const bedQ = window.AegisPatient?.bed() ? `?bed=${encodeURIComponent(window.AegisPatient.bed())}` : "";
      render(await api(`/aegis/consensus${bedQ}`));
    } catch (e) { spec.innerHTML = '<div class="muted">consensus unavailable</div>'; }
  }

  // The live tick auto-convenes the panel on HIGH risk; reflect it without a click,
  // and badge the dock launcher so the operator sees a quorum/emergency was reached.
  function onSnapshot(d) {
    const c = d.consensus;
    const badge = document.querySelector("#launch-consensus");
    if (badge) {
      const tag = !c ? "" : c.path === "emergency_fallback" ? " ⏱" : " ✓";
      badge.dataset.path = c ? c.path : "";
      badge.firstChild && (badge.firstChild.textContent = "⚖ Consensus" + tag);
    }
    // While watching the live patient (not running a manual stress-test), mirror the
    // auto-convened round so the panel stays current with the patient's real state.
    if (c && openFlag && !manual) {
      document.querySelector("#cns-patient").textContent = d.patient?.name || "this patient";
      render(c);
    }
  }

  function detNote(v) {
    if (v >= 0.85) return "— patient crashing; debate may not converge in time";
    if (v >= 0.5) return "— deteriorating; tight deadline";
    return "— debate has time to converge";
  }

  async function run() {
    manual = true;                          // pin to the stress-test until reopened
    const det = parseFloat(document.querySelector("#cns-det").value);
    const faulty = document.querySelector("#cns-faulty").value;
    const spec = document.querySelector("#cns-specialists");
    spec.innerHTML = '<div class="muted">re-running the panel under stress…</div>';
    let r; try {
      const bedQ = window.AegisPatient?.bed() ? `&bed=${encodeURIComponent(window.AegisPatient.bed())}` : "";
      r = await api(`/aegis/consensus?deterioration=${det}` + (faulty ? `&faulty=${faulty}` : "") + bedQ);
    } catch (e) { spec.innerHTML = '<div class="muted">consensus unavailable</div>'; return; }
    render(r);
  }

  const ABX = { "0": "no antibiotic", "1": "Pip-Tazo", "2": "Vancomycin",
                "3": "Cefepime", "4": "Meropenem", "5": "Levofloxacin" };

  function render(r) {
    const cert = r.certificate, emergency = r.path === "emergency_fallback";
    const p = r.committed_plan || {};
    const N = cert.n, quorum = cert.quorum, acc = cert.n_signatures ?? (r.ledger || []).filter(v => v.accept).length;
    document.querySelector("#cns-quorum").textContent = `N=${N} · quorum=${quorum} · ${acc} agreed`;

    // ---- VERDICT HERO: what the panel concluded about THIS patient ---------
    const reached = acc >= quorum;
    const vbox = document.querySelector("#cns-verdict");
    vbox.className = "cns-verdict " + (emergency ? "emergency" : reached ? "ok" : "split");
    vbox.innerHTML = `
      <div class="cns-vd-top">
        <span class="cns-vd-icon">${emergency ? "⏱" : reached ? "✓" : "⚠"}</span>
        <div class="cns-vd-head">
          <div class="cns-vd-title">${emergency ? "EMERGENCY ORDER COMMITTED"
            : reached ? "PANEL CONSENSUS REACHED" : "NO CONSENSUS"}</div>
          <div class="cns-vd-sub">${acc} of ${N} specialists agree · quorum of ${quorum} ${reached ? "met" : "not met"}</div>
        </div>
        <div class="cns-vd-agree"><b>${Math.round(acc / N * 100)}%</b><span>agreement</span></div>
      </div>
      <div class="cns-vd-order">
        <span class="cns-vd-k">recommended order</span>
        <span class="cns-vd-rx">${ABX[p.antibiotic_code] ?? "abx " + p.antibiotic_code}
          · fluids ${p.fluid_rate} mL/kg/h · vaso ${p.vasopressor_dose} µg/kg/min</span>
      </div>
      ${emergency ? `<div class="cns-vd-note">The patient was deteriorating faster than the
        debate could converge, so a pre-signed, Z3-safe standing order was committed —
        safety and liveness both preserved.</div>` : ""}`;

    // ---- SPECIALIST OPINIONS (the hero): each specialist's call on this patient
    const finalView = Math.max(...(r.ledger || [{ view: 0 }]).map(v => v.view));
    const latest = {};
    (r.ledger || []).forEach(v => { if (v.view === finalView || !latest[v.specialty]) latest[v.specialty] = v; });
    const faulty = (r.panel || []).filter(a => a.byzantine).map(a => a.specialty);
    const opinions = Object.values(latest);
    document.querySelector("#cns-specialists").innerHTML = opinions.map(v => {
      const byz = faulty.includes(v.specialty);
      const conf = Math.round((v.confidence ?? 0) * 100);
      return `<div class="cns-op ${v.accept ? "accept" : "reject"} ${byz ? "byz" : ""}">
        <div class="cns-op-l"><span class="cns-op-ico">${SPEC[v.specialty] || "•"}</span></div>
        <div class="cns-op-body">
          <div class="cns-op-h">
            <span class="cns-op-name">${SPECNAME[v.specialty] || v.specialty}</span>
            <span class="cns-op-focus">${SPECFOCUS[v.specialty] || ""}</span>
            ${byz ? '<span class="cns-tag byz">compromised source</span>' : ""}
            <span class="cns-op-verdict ${v.accept ? "yes" : "no"}">${v.accept ? "✓ CONCUR" : "✕ OBJECT"}</span>
          </div>
          <div class="cns-op-why">${v.reasoning || "—"}</div>
          <div class="cns-op-conf"><div class="cns-op-bar"><i style="width:${conf}%"></i></div>
            <span>${conf}% confidence</span></div>
        </div>
      </div>`;
    }).join("");

    // ---- DISSENT: surface disagreement explicitly --------------------------
    const objectors = opinions.filter(v => !v.accept);
    const dbox = document.querySelector("#cns-dissent");
    if (objectors.length) {
      dbox.className = "cns-dissent show";
      dbox.innerHTML = `<span class="cns-dissent-h">⚠ dissent</span>
        ${objectors.map(v => `<span class="cns-dissent-item"><b>${SPECNAME[v.specialty] || v.specialty}</b> objected — ${v.reasoning}</span>`).join("")}
        <span class="cns-dissent-foot">The order still committed because a Byzantine-fault-tolerant
          quorum (${quorum} of ${N}) agreed; dissent is recorded, not silenced.</span>`;
    } else {
      dbox.className = "cns-dissent show unanimous";
      dbox.innerHTML = `<span class="cns-dissent-h ok">✓ no dissent</span>
        <span class="cns-dissent-item">All ${N} specialists independently concurred with the order.</span>`;
    }

    // ---- header state + committed certificate (trust evidence) -------------
    const hs = document.querySelector("#cns-cert");
    hs.className = "cam-state" + (r.safety_ok ? " live" : " bad");
    hs.lastChild.textContent = r.safety_ok ? `Z3 ${r.z3_status} · ${cert.cert_hash}` : "Z3 FAIL";

    document.querySelector("#cns-out").innerHTML = `
      <div class="cns-path ${emergency ? "emergency" : "quorum"}">
        ${emergency ? "⏱ PHYSIOLOGIC-LIVENESS PATH" : "✓ QUORUM COMMIT"}</div>
      <div class="cns-plan">
        <span>fluids <b>${p.fluid_rate}</b></span>
        <span>vaso <b>${p.vasopressor_dose}</b></span>
        <span>abx <b>${ABX[p.antibiotic_code] ?? p.antibiotic_code}</b></span></div>
      <div class="cns-cert-row">
        <span class="cns-edge ${r.safety_ok ? "z3_proven" : "omission"}">Z3 ${r.z3_status}</span>
        <span class="cns-edge model_predicted">${cert.n_signatures} signatures</span>
      </div>
      <div class="cns-signers"><span class="cns-vd-k">multi-signed by</span> ${(cert.signers || []).join(" · ")}</div>
      <div class="cns-clock">decision deadline ${r.deadline_s}s · ${r.views_used}/${r.max_views} view(s) used
        ${manual ? `· crash-speed ${r.deterioration}` : "· deadline from live risk"}</div>`;

    // ---- raw vote ledger (kept, but tucked into the collapsible) -----------
    const views = {};
    (r.ledger || []).forEach(v => { (views[v.view] ||= []).push(v); });
    document.querySelector("#cns-ledger-sub").textContent =
      `${Object.keys(views).length} view(s) · ${(r.ledger || []).length} signed votes`;
    document.querySelector("#cns-ledger").innerHTML = Object.keys(views)
      .sort((a, b) => a - b).map(vw => {
        const votes = views[vw], a = votes.filter(v => v.accept).length;
        return `<div class="cns-view">
          <div class="cns-view-h">VIEW ${vw}
            <span class="cns-view-tally ${a >= quorum ? "ok" : "no"}">${a}/${N} ${a >= quorum ? "✓ quorum" : "✗"}</span></div>
          ${votes.map(v => `
            <div class="cns-vote ${v.accept ? "yes" : "no"}">
              <span class="cns-v-spec">${SPEC[v.specialty] || "•"} ${v.specialty}</span>
              <span class="cns-v-verdict">${v.accept ? "ACCEPT" : "REJECT"}</span>
              <span class="cns-v-sig" title="HMAC signature">⚿ ${v.signature}</span>
            </div>`).join("")}
        </div>`;
      }).join("");
  }

  // wiring
  document.addEventListener("click", e => {
    if (e.target.closest("#launch-consensus")) open();
    if (e.target.closest("#cns-close")) close();
    if (e.target.closest("#cns-run")) run();
  });
  document.addEventListener("input", e => {
    if (e.target.id === "cns-det") {
      const v = parseFloat(e.target.value);
      document.querySelector("#cns-det-val").textContent = v.toFixed(2);
      document.querySelector("#cns-det-note").textContent = detNote(v);
    }
  });

  return { open, close, onSnapshot };
})();

/* ---------------- AUTONOMOUS ICU ORCHESTRATOR (the demo moment) ---------------- */
window.AegisOrchestrator = (function () {
  const api = (p, opt) => fetch(window.AEGIS.http(p), opt).then(r => r.json());
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const STRATA = [
    { id: "global", ico: "🌐", name: "GLOBAL", tag: "Hospital-network intelligence" },
    { id: "wing",   ico: "▦", name: "WING",   tag: "Multi-patient cognitive scheduler" },
    { id: "bed",    ico: "✚", name: "BED",    tag: "Microkernel · causal graph · protocol synth" },
    { id: "sensor", ico: "◇", name: "SENSOR", tag: "Verified driver sandbox · Byzantine consensus" },
  ];
  const LAYER_LABEL = { sensor: "SENSOR", bed: "BED", wing: "WING", global: "GLOBAL",
                        governance: "GOVERNANCE" };
  let openFlag = false, busy = false;

  function open() { openFlag = true; document.querySelector("#orch-app").classList.remove("hidden"); loadStatus(); }
  function close() { openFlag = false; document.querySelector("#orch-app").classList.add("hidden"); }

  async function loadStatus() {
    let s; try { s = await api("/aegis/orchestrator"); } catch (e) { return; }
    renderStrata(s);
    document.querySelector("#orch-actions").textContent = s.attestation.actions;
    document.querySelector("#orch-head").textContent = s.attestation.head;
    try { renderAttest(await api("/aegis/orchestrator/attest")); } catch (e) {}
  }

  function renderAttest(a) {
    const pill = document.querySelector("#orch-attest");
    pill.className = "orch-attest" + (a.ok ? " ok" : " bad");
    pill.lastChild.textContent = a.ok ? "INVARIANT ATTESTED" : "BREACH DETECTED";
  }

  function renderStrata(s) {
    const h = s.hierarchy;
    const body = {
      global: () => {
        const g = h.global;
        return metric([["epidemiological brain", g.epidemiological_brain],
                       ["multi-site consensus", g.multi_site_consensus]]);
      },
      wing: () => {
        const w = h.wing, sch = w.scheduler || {};
        const crit = (w.critical || []);
        return `
          <div class="orch-m-row"><span>beds</span><b>${w.beds}</b></div>
          <div class="orch-m-row"><span>lane budget</span><b>${sch.lane_budget ?? "—"}</b></div>
          <div class="orch-m-row"><span>critical</span>
            <b class="${crit.length ? "crit" : ""}">${crit.length}</b></div>
          ${crit.length ? `<div class="orch-chips">${crit.map(b =>
            `<span class="orch-chip crit">${b}</span>`).join("")}</div>` : ""}`;
      },
      bed: () => {
        const b = h.bed, graphs = b.causal_graphs || {};
        const total = Object.values(graphs).reduce((a, c) => a + c, 0);
        return `
          <div class="orch-m-row"><span>microkernel</span><b class="mono sm">/dev/bio · Z3 synth</b></div>
          <div class="orch-m-row"><span>causal-graph nodes</span><b>${total}</b></div>
          <div class="orch-m-row"><span>instrumented beds</span><b>${Object.keys(graphs).length}</b></div>`;
      },
      sensor: () => {
        const s2 = h.sensor;
        return metric([["eBPF sandbox", s2.ebpf_sandbox],
                       ["byzantine consensus", s2.byzantine_consensus],
                       ["beds instrumented", String(s2.beds_instrumented)]]);
      },
    };
    document.querySelector("#orch-strata").innerHTML = STRATA.map(st => `
      <div class="orch-stratum" data-layer="${st.id}">
        <div class="orch-node"><span class="orch-node-dot"></span></div>
        <div class="orch-stratum-card">
          <div class="orch-stratum-h">
            <span class="orch-stratum-ico">${st.ico}</span>
            <span class="orch-stratum-name">${st.name}</span>
            <span class="orch-stratum-tag">${st.tag}</span>
          </div>
          <div class="orch-stratum-body">${body[st.id]()}</div>
        </div>
      </div>`).join("");
  }
  const metric = rows => rows.map(([k, v]) =>
    `<div class="orch-m-line"><span>${k}</span><em>${v}</em></div>`).join("");

  function logLine(layer, text, cls = "") {
    const tag = LAYER_LABEL[layer] || layer.toUpperCase();
    const el = document.createElement("div");
    el.className = "orch-log-line " + cls;
    el.innerHTML = `<span class="orch-log-tag l-${layer}">${tag}</span>
      <span class="orch-log-text">${text}</span>`;
    document.querySelector("#orch-log").appendChild(el);
    el.scrollIntoView({ block: "nearest" });
    return el;
  }

  function glowStratum(layer) {
    document.querySelectorAll(".orch-stratum").forEach(s =>
      s.classList.toggle("active", s.dataset.layer === layer));
    // travel the spine pulse to the active stratum
    const el = document.querySelector(`.orch-stratum[data-layer="${layer}"] .orch-node`);
    const strata = document.querySelector("#orch-strata");
    if (el && strata) {
      const pulse = document.querySelector("#orch-pulse");
      pulse.style.opacity = "1";
      pulse.style.top = (el.offsetTop + el.offsetHeight / 2 - 6) + "px";
    }
  }

  function stepDetail(t) {
    if (t.layer === "bed" && t.synth_ms != null) return `${t.event} <em>· ${t.synth_ms} ms</em>`;
    if (t.layer === "global" && t.signatures) return `${t.event}: <em>${t.signatures.join(" · ") || "none"}</em>`;
    if (t.layer === "wing" && t.critical) return `${t.event} <em>· critical: ${t.critical.join(", ") || "—"}</em>`;
    if (t.layer === "bed" && t.fused_risk != null) return `${t.event} <em>· fused risk ${(t.fused_risk*100|0)}% (${t.tier||"—"})</em>`;
    return t.event;
  }

  async function runCascade() {
    if (busy) return; busy = true;
    const btn = document.querySelector("#orch-run");
    btn.classList.add("running"); btn.disabled = true;
    document.querySelector("#orch-log").innerHTML = "";
    document.querySelector("#orch-cert").innerHTML = '<div class="orch-cert-idle">synthesizing…</div>';
    document.querySelectorAll(".orch-stratum").forEach(s => s.classList.remove("active", "done"));

    let res; try { res = await api("/aegis/orchestrator/scenario", { method: "POST" }); }
    catch (e) { logLine("governance", "orchestrator unavailable", "blocked"); btn.classList.remove("running"); btn.disabled = false; busy = false; return; }

    document.querySelector("#orch-bed").textContent = "bed · " + res.bed;
    for (const t of res.timeline) {
      if (t.layer !== "governance") glowStratum(t.layer);
      const certified = t.layer === "governance" ? t.certified : null;
      logLine(t.layer, stepDetail(t), certified === false ? "blocked" : "");
      if (t.layer === "governance") { stampCertificate(res); renderPortableProof(res); }
      await sleep(620);
    }
    document.querySelectorAll(".orch-stratum.active").forEach(s => s.classList.add("done"));
    document.querySelector("#orch-pulse").style.opacity = "0";
    await loadStatus();
    btn.classList.remove("running"); btn.disabled = false; busy = false;
  }

  function stampCertificate(res) {
    const c = res.certificate, ok = c.ok;
    const p = res.committed_plan || {};
    const stage = document.querySelector("#orch-cert");
    stage.innerHTML = `
      <div class="orch-seal ${ok ? "ok" : "blocked"}">
        <svg viewBox="0 0 120 120" class="orch-seal-svg">
          <defs><radialGradient id="sealg" cx="50%" cy="40%" r="65%">
            <stop offset="0%" stop-color="${ok ? "#5ce0c0" : "#f2606e"}" stop-opacity=".35"/>
            <stop offset="100%" stop-color="${ok ? "#1f6f5e" : "#7a2530"}" stop-opacity=".05"/>
          </radialGradient></defs>
          <circle cx="60" cy="60" r="54" class="orch-seal-ring"/>
          <circle cx="60" cy="60" r="46" fill="url(#sealg)"/>
          <circle cx="60" cy="60" r="42" class="orch-seal-ring2"/>
          ${ok ? '<path d="M44 60 l11 12 l22 -26" class="orch-seal-check"/>'
               : '<path d="M46 46 l28 28 M74 46 l-28 28" class="orch-seal-x"/>'}
        </svg>
        <div class="orch-seal-z3">${ok ? "Z3 UNSAT" : "BLOCKED"}</div>
      </div>
      <div class="orch-cert-meta">
        <div class="orch-cert-verdict ${ok ? "ok" : "blocked"}">
          ${ok ? "ORDER CERTIFIED — PROVEN WITHIN SAFE ENVELOPE" : "ORDER QUARANTINED — INVARIANT VIOLATED"}</div>
        ${ok ? `<div class="orch-cert-plan">
          <span>fluids <b>${p.fluid_rate}</b></span>
          <span>vaso <b>${p.vasopressor_dose}</b></span>
          <span>abx <b>code ${p.antibiotic_code}</b></span></div>` :
          `<div class="orch-cert-viol">${(c.violations||[]).join(" · ")}</div>`}
        <div class="orch-cert-sig">
          <span class="orch-cert-k">⚿ cert</span><span class="mono">${c.hash}</span>
          <span class="orch-cert-k">prev</span><span class="mono">${c.prev_hash}</span></div>
        <div class="orch-cert-foot">sealed into the attestation ledger · immutable</div>
      </div>`;
    renderLedgerAppend(c);
  }

  function renderLedgerAppend(c) {
    const box = document.querySelector("#orch-ledger");
    if (box.querySelector(".orch-log-idle")) box.innerHTML = "";
    const el = document.createElement("div");
    el.className = "orch-led-row " + (c.ok ? "ok" : "blocked") + " just-sealed";
    el.innerHTML = `
      <span class="orch-led-seq">#${c.seq}</span>
      <span class="orch-led-act">${c.action}</span>
      <span class="orch-led-status ${c.ok ? "ok" : "blocked"}">${c.ok ? "✓ "+c.z3_status : "✕ blocked"}</span>
      <span class="orch-led-hash mono">${c.hash}</span>`;
    box.prepend(el);
  }

  function renderPortableProof(res) {
    const stage = document.querySelector("#orch-proof");
    if (!stage) return;
    const pc = res.proof_certificate, audit = res.proof_audit;
    if (!pc) {
      stage.innerHTML = `<div class="orch-cert-idle">proof export unavailable
        (signing key / crypto not present)</div>`;
      return;
    }
    const sig = pc.signature || {}, v = pc.verdict || {}, subj = pc.subject || {};
    const proto = subj.protocol || {};
    const seq = subj.ledger_seq;
    const obs = (pc.obligations || []).map(o => {
      const ok = o.expected_status === "unsat";
      return `<span class="orch-pf-ob ${ok ? "ok" : "bad"}">
        <i>${ok ? "✓" : "✕"}</i>${o.name}</span>`;
    }).join("");
    const trusted = audit && audit.trusted;
    stage.innerHTML = `
      <div class="orch-pf">
        <div class="orch-pf-top">
          <span class="orch-pf-badge ${v.all_satisfied ? "ok" : "bad"}">
            ${v.n_satisfied}/${v.n_obligations} invariants proven</span>
          <span class="orch-pf-drug">${proto.antibiotic_name || "—"} · fluids ${proto.fluid_rate} · vaso ${proto.vasopressor_dose}</span>
        </div>
        <div class="orch-pf-obs">${obs}</div>
        <div class="orch-pf-sig">
          <span class="orch-cert-k">⚿ Ed25519</span><span class="mono">${sig.key_id || "unsigned"}</span>
          <span class="orch-cert-k">SMT-LIB2</span><span class="mono">${(pc.solver||{}).name} ${(pc.solver||{}).version}</span>
        </div>
        <div class="orch-pf-actions">
          <button class="orch-pf-verify" id="orch-pf-verify">⟳ re-verify independently</button>
          <a class="orch-pf-dl" href="${window.AEGIS.http('/aegis/orchestrator/proof/' + seq + '/bundle')}"
             ${seq==null?'style="display:none"':''} download>⤓ audit bundle (.zip)</a>
        </div>
        <div class="orch-pf-audit ${trusted ? "ok" : "wait"}" id="orch-pf-audit">
          ${audit ? auditLine(audit) : "not yet audited"}
        </div>
      </div>`;
    const btn = stage.querySelector("#orch-pf-verify");
    if (btn) btn.onclick = async () => {
      btn.disabled = true; btn.textContent = "re-running solver…";
      const out = document.querySelector("#orch-pf-audit");
      try {
        const a = await api("/aegis/proof/verify", { method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ certificate: pc }) });
        out.className = "orch-pf-audit " + (a.trusted ? "ok" : "bad");
        out.innerHTML = auditLine(a);
      } catch (e) {
        out.className = "orch-pf-audit bad"; out.textContent = "audit failed: " + e;
      }
      btn.disabled = false; btn.textContent = "⟳ re-verify independently";
    };
  }

  function auditLine(a) {
    const f = (ok, label) => `<span class="orch-pf-flag ${ok ? "ok" : "bad"}">${ok ? "✓" : "✕"} ${label}</span>`;
    return `<span class="orch-pf-verdict ${a.trusted ? "ok" : "bad"}">${a.trusted ? "VERIFIED" : "REJECTED"}</span>`
      + f(a.math_ok, "math re-checked") + f(a.integrity_ok, "untampered")
      + f(a.signature_ok, "signature");
  }

  document.addEventListener("click", e => {
    if (e.target.closest("#launch-orch")) open();
    if (e.target.closest("#orch-close")) close();
    if (e.target.closest("#orch-run")) runCascade();
  });

  return { open, close };
})();

/* ---------------- FEDERATED INTELLIGENCE NETWORK (Layer 1) ---------------- */
window.AegisFederation = (function () {
  const api = (p) => fetch(window.AEGIS.http(p)).then(r => r.json());
  // AUROC bar scale: chance (0.5) → 0%, 0.85 → 100%, so real gaps are visible.
  const scale = v => Math.max(0, Math.min(100, (v - 0.5) / 0.35 * 100));
  let loaded = false;

  function open() { document.querySelector("#fed-app").classList.remove("hidden"); if (!loaded) run(); }
  function close() { document.querySelector("#fed-app").classList.add("hidden"); }

  function setStatus(txt, cls) {
    const p = document.querySelector("#fed-status");
    p.className = "orch-attest " + (cls || "");
    p.lastChild.textContent = txt;
  }

  function topology(nHosp, active) {
    const cx = 150, cy = 96, R = 78;
    let nodes = "", lines = "";
    for (let i = 0; i < nHosp; i++) {
      const a = Math.PI * (0.15 + 0.7 * (nHosp === 1 ? 0.5 : i / (nHosp - 1)));
      const x = cx - Math.cos(a) * R, y = cy - Math.sin(a) * R * 0.92;
      lines += `<line x1="${x}" y1="${y}" x2="${cx}" y2="${cy}"
        class="fed-link ${active ? "flow" : ""}" style="animation-delay:${i * 0.18}s"/>`;
      nodes += `<g class="fed-hosp"><circle cx="${x}" cy="${y}" r="13"/>
        <text x="${x}" y="${y + 3.5}">🏥</text>
        <text x="${x}" y="${y + 26}" class="fed-hosp-l">H${i + 1}</text></g>`;
    }
    return `<svg viewBox="0 0 300 200" class="fed-topo-svg">
      ${lines}${nodes}
      <g class="fed-server"><circle cx="${cx}" cy="${cy}" r="22"/>
        <text x="${cx}" y="${cy - 1}">⇄</text>
        <text x="${cx}" y="${cy + 13}" class="fed-server-l">AGG</text></g>
      <text x="${cx}" y="190" class="fed-ext">held-out hospital (setB) — never trained on</text>
    </svg>`;
  }

  function bar(label, v, cls, note) {
    return `<div class="fed-bar-row ${cls}">
      <div class="fed-bar-label">${label}</div>
      <div class="fed-bar-track">
        <div class="fed-bar-fill" style="width:${scale(v)}%"></div>
        <div class="fed-bar-val">${v.toFixed(3)}</div>
      </div>
      ${note ? `<div class="fed-bar-note">${note}</div>` : ""}</div>`;
  }

  async function run() {
    const btn = document.querySelector("#fed-run");
    btn.classList.add("running"); btn.disabled = true;
    setStatus("FEDERATING…", "");
    document.querySelector("#fed-topo").innerHTML = topology(4, true);
    document.querySelector("#fed-recovery").innerHTML =
      '<div class="orch-log-idle">training hospitals & aggregating…</div>';

    let d; try { d = await api("/aegis/fedlearn?limit=400"); }
    catch (e) { setStatus("UNAVAILABLE", "bad"); btn.classList.remove("running"); btn.disabled = false; return; }
    loaded = true;
    const ex = d.experiment, lo = ex.local_only, ce = ex.centralized, fe = ex.federated;

    document.querySelector("#fed-topo").innerHTML = topology(ex.n_hospitals, false);
    document.querySelector("#fed-recovery").innerHTML = `
      <div class="fed-bars-h">AUROC on the unseen hospital (setB)</div>
      ${bar("Local-only · one site", lo.setB_cross_site, "lo", "no collaboration")}
      ${bar("Centralized · data pooled", ce.setB_cross_site, "ce", "privacy-violating baseline")}
      ${bar("Federated · no data shared", fe.setB_cross_site, "fe win", "signed deltas only")}
      <div class="fed-scale">chance 0.5 ◂——————▸ 0.85</div>`;

    const rec = ex.headline.federated_recovers;
    document.querySelector("#fed-headline").innerHTML = `
      <span class="fed-up">▲ +${rec.toFixed(3)}</span> federation recovers cross-hospital
      AUROC over a lone site — <b>without any hospital sharing patient records.</b>`;
    setStatus("CONVERGED", "ok");

    // robustness
    const p = d.poisoning;
    document.querySelector("#fed-poison").innerHTML = `
      ${bar("FedAvg · clean", p.clean_fedavg, "ce")}
      ${bar("FedAvg · 1 poisoned", p.poisoned_fedavg, "lo bad", "mean is corruptible")}
      ${bar("Krum · 1 poisoned", p.poisoned_krum, "fe win", "tolerates the attacker")}
      ${bar("Median · 1 poisoned", p.poisoned_median, "fe win", "tolerates the attacker")}`;

    document.querySelector("#fed-caveats").innerHTML =
      (d.caveats || []).map(c => `<div class="fed-caveat">◆ ${c}</div>`).join("");

    btn.classList.remove("running"); btn.disabled = false;
  }

  document.addEventListener("click", e => {
    if (e.target.closest("#launch-fed")) open();
    if (e.target.closest("#fed-close")) close();
    if (e.target.closest("#fed-run")) run();
  });

  return { open, close };
})();

/* ---------------- CLINICAL RBAC — ROLE SHELLS ---------------- */
window.AegisRBAC = (function () {
  const api = (p, opt) => fetch(window.AEGIS.http(p), opt).then(r => r.json());
  const post = (p, body) => api(p, { method: "POST", headers: { "Content-Type": "application/json" },
                                     body: JSON.stringify(body) });
  const ROLES = [
    { id: "nurse", ico: "🩺", label: "Nurse", blurb: "observe · acknowledge alarms" },
    { id: "remote_specialist", ico: "🛰", label: "Remote Specialist", blurb: "read-only · annotate" },
    { id: "pharmacist", ico: "💊", label: "Pharmacist", blurb: "meds · countersign / veto" },
    { id: "attending", ico: "🧑‍⚕️", label: "Attending", blurb: "treat · order protocols" },
    { id: "chief", ico: "🏥", label: "Chief of Medicine", blurb: "whole-ICU oversight" },
  ];
  const CAP_LABEL = {
    "view:vitals": "view vitals", "view:twin": "view digital twin", "view:narrative": "view narrative",
    "view:meds": "view medications", "view:interactions": "drug interactions", "view:cohort": "ICU tactical sphere",
    "ack:alarm": "acknowledge alarm", "annotate": "annotate", "pharm:countersign": "countersign order",
    "pharm:veto": "veto medication", "order:treat": "order treatment", "order:escalate": "escalate patient",
    "order:clear": "clear / recover", "order:approve_protocol": "approve protocol",
    "drill:crisis": "run crisis drill", "override:alarm": "override alarm",
  };
  let policy = null, session = null, token = null, shell = null;

  function open() { document.querySelector("#rbac-app").classList.remove("hidden"); load(); }
  function close() { document.querySelector("#rbac-app").classList.add("hidden"); }

  async function load() {
    try { policy = await api("/aegis/rbac/policy"); } catch (e) { return; }
    renderRoles(); renderPolicy(); refreshAudit();
  }

  function renderRoles() {
    document.querySelector("#rbac-roles").innerHTML = ROLES.map(r => `
      <button class="rbac-role ${session && session.role === r.id ? "active" : ""}" data-role="${r.id}">
        <span class="rbac-role-ico">${r.ico}</span>
        <span class="rbac-role-name">${r.label}</span>
        <span class="rbac-role-ring">ring ${policy.rings[r.id]}</span>
        <span class="rbac-role-blurb">${r.blurb}</span>
      </button>`).join("");
  }

  function renderPolicy() {
    const s = policy.soundness;
    document.querySelector("#rbac-policy-sub").textContent = `${s.grants_checked} grants`;
    document.querySelector("#rbac-sound").innerHTML = `
      <div class="rbac-sound-badge ${s.sound ? "ok" : "bad"}">
        <span class="rbac-sound-ico">${s.sound ? "✓" : "✕"}</span>
        <div><b>${s.sound ? "POLICY PROVEN SOUND" : "POLICY UNSOUND"}</b>
          <span>Z3 ${s.z3_status} · no grant exceeds its ring</span></div></div>`;
    const ringNames = { 1: "observe", 2: "pharmacy", 3: "physician", 4: "oversight" };
    document.querySelector("#rbac-rings").innerHTML = [1, 2, 3, 4].map(r => `
      <div class="rbac-ring-row"><span class="rbac-ring-dot r${r}">${r}</span>
        <span>${ringNames[r]}</span></div>`).join("");
  }

  async function loginAs(role) {
    const res = await post("/aegis/rbac/login", { user: "demo_" + role, role });
    if (res.status === "error") return;
    session = res.session; token = res.token; shell = new Set(res.shell.capabilities);
    renderRoles(); renderShell(); refreshAudit();
    const pill = document.querySelector("#rbac-session");
    pill.className = "rbac-session active";
    pill.lastChild.textContent = `${session.user} · ${session.role} · ring ${session.ring}`;
  }

  function logout() {
    session = token = shell = null;
    document.querySelector("#rbac-shell").classList.add("hidden");
    const pill = document.querySelector("#rbac-session");
    pill.className = "rbac-session"; pill.lastChild.textContent = "no active session";
    renderRoles();
  }

  function renderShell() {
    document.querySelector("#rbac-shell").classList.remove("hidden");
    const role = ROLES.find(r => r.id === session.role);
    document.querySelector("#rbac-ring").innerHTML =
      `<span class="rbac-ring-num">${session.ring}</span>`;
    document.querySelector("#rbac-ring").className = "rbac-ring-badge r" + session.ring;
    document.querySelector("#rbac-shell-role").textContent = `${role.ico} ${role.label}`;
    const granted = [...shell].length;
    document.querySelector("#rbac-shell-sub").textContent =
      `${granted} capabilities granted · everything else denied by default`;
    // full capability taxonomy, sorted by ring, granted vs denied
    const caps = Object.keys(policy.cap_min_ring).sort(
      (a, b) => policy.cap_min_ring[a] - policy.cap_min_ring[b] || a.localeCompare(b));
    document.querySelector("#rbac-caps").innerHTML = caps.map(c => {
      const ok = shell.has(c);
      return `<button class="rbac-cap ${ok ? "granted" : "denied"}" data-cap="${c}">
        <span class="rbac-cap-dot"></span>
        <span class="rbac-cap-name">${CAP_LABEL[c] || c}</span>
        <span class="rbac-cap-ring">r${policy.cap_min_ring[c]}</span></button>`;
    }).join("");
  }

  async function attempt(cap) {
    if (!token) return;
    const el = document.querySelector(`.rbac-cap[data-cap="${cap}"]`);
    const res = await post("/aegis/rbac/authorize", { token, capability: cap });
    if (el) {
      el.classList.remove("flash-grant", "flash-deny");
      void el.offsetWidth;
      el.classList.add(res.ok ? "flash-grant" : "flash-deny");
    }
    refreshAudit();
  }

  async function refreshAudit() {
    let a; try { a = await api("/aegis/rbac/audit?limit=14"); } catch (e) { return; }
    document.querySelector("#rbac-audit-sub").textContent =
      `${a.count} decisions · ${a.integrity_ok ? "chain ✓" : "TAMPERED"}`;
    const box = document.querySelector("#rbac-audit");
    if (!a.entries.length) return;
    box.innerHTML = a.entries.slice().reverse().map(e => `
      <div class="rbac-aud-row ${e.decision === "GRANT" ? "grant" : "deny"}">
        <span class="rbac-aud-dec">${e.decision === "GRANT" ? "✓" : "✕"}</span>
        <span class="rbac-aud-who">${e.role}</span>
        <span class="rbac-aud-cap">${CAP_LABEL[e.capability] || e.capability}</span>
        <span class="rbac-aud-hash mono">${e.hash}</span></div>`).join("");
  }

  document.addEventListener("click", e => {
    if (e.target.closest("#launch-rbac")) open();
    if (e.target.closest("#rbac-close")) close();
    if (e.target.closest("#rbac-logout")) logout();
    const role = e.target.closest(".rbac-role");
    if (role) loginAs(role.dataset.role);
    const cap = e.target.closest(".rbac-cap");
    if (cap) attempt(cap.dataset.cap);
  });

  return { open, close };
})();

/* ---------------- CLINICAL SAFETY DSL (author rules → Z3 proof) ---------------- */
window.AegisDSL = (function () {
  const api = (p, opt) => fetch(window.AEGIS.http(p), opt).then(r => r.json());
  const post = (p, body) => api(p, { method: "POST", headers: { "Content-Type": "application/json" },
                                     body: JSON.stringify(body) });
  let loaded = false;
  // live order uses the cage/bundle antibiotic codebook; the DSL uses its own.
  // Translate by drug so the policy reasons about the SAME drug the patient gets.
  const CORE_TO_DSL = { 0: 0, 1: 2, 2: 3, 3: 1, 4: 4, 5: 6 };
  const DSL_ABX = { 0: "no antibiotic", 1: "Cefepime", 2: "Pip-Tazo", 3: "Vancomycin",
                    4: "Meropenem", 5: "Ceftriaxone", 6: "Ciprofloxacin" };

  function open() { document.querySelector("#dsl-app").classList.remove("hidden"); init(); }
  function close() { document.querySelector("#dsl-app").classList.add("hidden"); }

  async function init() {
    if (!loaded) {
      try { const d = await api("/aegis/dsl/default");
            document.querySelector("#dsl-editor").value = d.policy.trim(); loaded = true; }
      catch (e) { return; }
    }
    syncFromLive();      // mirror the live patient's comorbidities + proposed order
    prove();             // and immediately prove that order against the policy
  }

  // Pull the live patient's clinical context + proposed order into the controls so
  // the panel proves what's actually about to happen, not a hand-typed hypothetical.
  function syncFromLive() {
    const d = lastData; if (!d) return;
    const h = d.patient?.clinical_history || {};
    document.querySelector("#dsl-chf").checked = !!h.chf;
    document.querySelector("#dsl-pen").checked = !!h.penicillin_allergy;
    document.querySelector("#dsl-renal").checked = !!h.renal_failure;
    document.querySelector("#dsl-risk").value = d.risk?.level || "HIGH";
    const order = d.protocol?.protocol;
    if (order) {
      document.querySelector("#dsl-abx").value = String(CORE_TO_DSL[order.antibiotic_code] ?? 1);
      document.querySelector("#dsl-fluid").value = order.fluid_rate;
      document.querySelector("#dsl-vaso").value = order.vasopressor_dose;
      document.querySelector("#dsl-abxt").value = 0.5;
    }
  }

  function gather() {
    return {
      policy: document.querySelector("#dsl-editor").value,
      protocol: {
        antibiotic_code: parseInt(document.querySelector("#dsl-abx").value),
        antibiotic_time_hr: parseFloat(document.querySelector("#dsl-abxt").value),
        fluid_rate: parseFloat(document.querySelector("#dsl-fluid").value),
        vasopressor_dose: parseFloat(document.querySelector("#dsl-vaso").value),
      },
      context: {
        risk_level: document.querySelector("#dsl-risk").value,
        chf: document.querySelector("#dsl-chf").checked,
        penicillin_allergy: document.querySelector("#dsl-pen").checked,
        renal_failure: document.querySelector("#dsl-renal").checked,
      },
    };
  }

  // Plain-English statement of each default rule + why it is (or isn't) active.
  const RULE_PLAIN = {
    sepsis_antibiotic: "HIGH-risk sepsis must receive an antibiotic within the hour",
    allergy_guard: "a penicillin-allergic patient must not receive Pip-Tazo",
    fluid_cap: "CHF caps fluid resuscitation at ≤ 5 mL/kg/h",
    renal_vaso_cap: "renal failure caps the vasopressor dose at ≤ 0.2",
  };
  function ruleActive(name, text, ctx) {
    const t = (text || "").toLowerCase();
    if (name === "sepsis_antibiotic" || t.includes("risk = high")) return ctx.risk_level === "HIGH";
    if (name === "allergy_guard" || t.includes("penicillin")) return !!ctx.penicillin_allergy;
    if (name === "fluid_cap" || t.includes("chf")) return !!ctx.chf;
    if (name === "renal_vaso_cap" || t.includes("renal")) return !!ctx.renal_failure;
    return true;                              // unconditional rule → always binding
  }
  function activeReason(name, ctx) {
    return { sepsis_antibiotic: ctx.risk_level === "HIGH" ? "patient is HIGH risk" : "patient is not HIGH risk",
             allergy_guard: ctx.penicillin_allergy ? "penicillin allergy on file" : "no penicillin allergy",
             fluid_cap: ctx.chf ? "patient has CHF" : "no CHF",
             renal_vaso_cap: ctx.renal_failure ? "renal failure on file" : "no renal failure" }[name] || "";
  }

  function renderPatient(ctx, order, live) {
    const name = lastData?.patient?.name || "Patient";
    const como = [];
    if (ctx.chf) como.push("CHF");
    if (ctx.penicillin_allergy) como.push("penicillin allergy");
    if (ctx.renal_failure) como.push("renal failure");
    document.querySelector("#dsl-patient").innerHTML = `
      <div class="dsl-pt-h">
        <span class="dsl-pt-name">${name}</span>
        <span class="dsl-pt-risk ${ctx.risk_level}">${ctx.risk_level} risk</span>
      </div>
      <div class="dsl-pt-como">${como.length
        ? como.map(c => `<span class="dsl-chip2">${c}</span>`).join("")
        : '<span class="muted">no flagged comorbidities</span>'}</div>
      <div class="dsl-pt-order">
        <span class="dsl-pt-k">${live ? "● order under review (live)" : "candidate order"}</span>
        <span class="dsl-pt-rx">${DSL_ABX[order.antibiotic_code] ?? "abx " + order.antibiotic_code}
          · fluids ${order.fluid_rate} mL/kg/h · vaso ${order.vasopressor_dose} µg/kg/min</span>
      </div>`;
  }

  function renderActive(cert, ctx) {
    const nActive = cert.rules.filter(r => ruleActive(r.rule, r.text, ctx)).length;
    document.querySelector("#dsl-active-sub").textContent =
      `${nActive} of ${cert.n_rules} apply to this patient`;
    document.querySelector("#dsl-active").innerHTML = cert.rules.map(r => {
      const active = ruleActive(r.rule, r.text, ctx), why = activeReason(r.rule, ctx);
      const mark = !active ? "◦" : r.satisfied ? "✓" : "✕";
      const line = !active ? `dormant — ${why}`
        : r.satisfied ? `order complies — active because ${why}`
        : `⚠ ORDER VIOLATES THIS RULE — active because ${why}`;
      return `<div class="dsl-rule ${active ? "active" : "dormant"} ${active && !r.satisfied ? "fail" : ""}">
        <span class="dsl-rule-state">${mark}</span>
        <div class="dsl-rule-body">
          <div class="dsl-rule-name">${RULE_PLAIN[r.rule] || r.rule}
            <span class="dsl-rule-badge ${active ? "on" : "off"}">${active ? "ACTIVE" : "dormant"}</span></div>
          <div class="dsl-rule-reason">${line}</div>
        </div></div>`;
    }).join("");
  }

  function isLive() {
    const o = lastData?.protocol?.protocol;
    if (!o) return false;
    return parseInt(document.querySelector("#dsl-abx").value) === (CORE_TO_DSL[o.antibiotic_code] ?? -1)
      && parseFloat(document.querySelector("#dsl-fluid").value) === o.fluid_rate
      && parseFloat(document.querySelector("#dsl-vaso").value) === o.vasopressor_dose;
  }

  async function prove() {
    const g = gather();
    renderPatient(g.context, g.protocol, isLive());
    const btn = document.querySelector("#dsl-run");
    btn && (btn.classList.add("running"), btn.disabled = true);
    const box = document.querySelector("#dsl-results");
    box.innerHTML = '<div class="orch-log-idle">compiling to Z3…</div>';
    const res = await post("/aegis/dsl/verify", g);
    btn && (btn.classList.remove("running"), btn.disabled = false);

    const verdict = document.querySelector("#dsl-verdict");
    if (res.status === "error") {
      verdict.className = "rbac-session bad";
      verdict.lastChild.textContent = "compile error";
      box.innerHTML = `<div class="dsl-err">⚠ ${res.detail}</div>`;
      document.querySelector("#dsl-active").innerHTML = `<div class="dsl-err">⚠ ${res.detail}</div>`;
      return;
    }
    const cert = res.certificate;
    verdict.className = "rbac-session " + (cert.ok ? "active" : "bad");
    verdict.lastChild.textContent = cert.ok
      ? `ORDER PROVEN SAFE · ${cert.n_rules}/${cert.n_rules}` : `${cert.n_rules - cert.n_satisfied} VIOLATION(S)`;

    const con = res.consistency;
    document.querySelector("#dsl-consistency").textContent =
      con.consistent ? "policy consistent" : "⚠ policy self-contradictory";

    renderActive(cert, g.context);

    box.innerHTML = cert.rules.map(r => `
      <div class="dsl-cert ${r.satisfied ? "ok" : "bad"}">
        <div class="dsl-cert-h">
          <span class="dsl-cert-mark">${r.satisfied ? "✓" : "✕"}</span>
          <span class="dsl-cert-name">${r.rule}</span>
          <span class="dsl-cert-z3 ${r.satisfied ? "ok" : "bad"}">Z3 ${r.z3_status}</span></div>
        <div class="dsl-cert-text">${escapeHtml(r.text)}</div>
        ${!r.satisfied && r.counterexample && Object.keys(r.counterexample).length
          ? `<div class="dsl-cert-ce">counterexample: ${fmtCE(r.counterexample)}</div>` : ""}
        <details class="dsl-cert-formula"><summary>Z3 assertion</summary>
          <code>${escapeHtml(r.formula)}</code></details>
      </div>`).join("")
      + (con.consistent
          ? `<div class="dsl-note">✓ rule set is jointly satisfiable — a compliant protocol exists.</div>`
          : `<div class="dsl-note bad">⚠ no protocol can satisfy all rules for this patient (contradictory policy).</div>`);
  }

  const NICE = { "0": "NONE", "1": "Cefepime", "2": "PipTazo", "3": "Vancomycin",
                 "4": "Meropenem", "5": "Ceftriaxone" };
  function fmtCE(ce) {
    return Object.entries(ce).map(([k, v]) => {
      if (k === "antibiotic") v = NICE[v] || v;
      if (k === "risk") v = ["LOW", "MEDIUM", "HIGH"][parseInt(v)] ?? v;
      return `${k.replace("patient_", "")}=${v}`;
    }).join(", ");
  }
  function escapeHtml(s) {
    return String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  }

  document.addEventListener("click", e => {
    if (e.target.closest("#launch-dsl")) open();
    if (e.target.closest("#dsl-close")) close();
    if (e.target.closest("#dsl-run")) prove();
  });
  // editing the order/context in the advanced panel re-proves against the policy
  document.addEventListener("change", e => {
    if (e.target.closest("#dsl-app") && e.target.matches("#dsl-abx,#dsl-abxt,#dsl-fluid,#dsl-vaso,#dsl-risk,#dsl-chf,#dsl-pen,#dsl-renal")) prove();
  });

  return { open, close };
})();

/* ---------------- LIVING PHARMACOPOEIA (interaction graph → Z3) ---------------- */
window.AegisPharma = (function () {
  const api = (p, opt) => fetch(window.AEGIS.http(p), opt).then(r => r.json());
  const post = (p, body) => api(p, { method: "POST", headers: { "Content-Type": "application/json" },
                                     body: JSON.stringify(body) });
  const QUICK = ["warfarin", "fluconazole", "ciprofloxacin", "ondansetron", "amiodarone",
                 "vancomycin", "piperacillin", "cefepime", "meropenem", "linezolid"];
  const SEV_CLASS = { Major: "bad", Moderate: "warn", Minor: "warn" };
  let regimen = ["warfarin", "fluconazole", "ciprofloxacin", "ondansetron"], loaded = false;

  function open() { document.querySelector("#pharma-app").classList.remove("hidden"); load(); scopeToPatient(); }
  function close() { document.querySelector("#pharma-app").classList.add("hidden"); }

  function scopeToPatient() {
    // pre-fill the patient context (renal/hepatic) from the selected patient
    const p = window.AegisPatient?.current();
    const h = (p && p.history) || {};
    const renal = document.querySelector("#pharma-renal"), hep = document.querySelector("#pharma-hepatic");
    if (renal) renal.checked = !!h.renal_failure;
    if (hep) hep.checked = !!h.hepatic;
    const tag = document.querySelector("#pharma-patient");
    if (tag) tag.textContent = p ? `screening for ${p.name}` : "";
  }

  async function load() {
    renderRegimen(); renderQuick();
    if (loaded) return;
    let st; try { st = await api("/aegis/pharmacopoeia"); } catch (e) { return; }
    document.querySelector("#pharma-kb").textContent =
      `${st.ddinter.drugs.toLocaleString()} drugs · ${st.ddinter.edges.toLocaleString()} edges · ${st.cyp_enzymes.length} CYP enzymes · ${st.qt_known} QT drugs`;
    loaded = true;
  }

  function renderRegimen() {
    document.querySelector("#pharma-regimen").innerHTML = regimen.length
      ? regimen.map(d => `<span class="pharma-chip">${d}<button data-rm="${d}">✕</button></span>`).join("")
      : '<span class="orch-log-idle">no drugs yet</span>';
  }
  function renderQuick() {
    document.querySelector("#pharma-quick").innerHTML = "quick add: " +
      QUICK.map(d => `<button class="pharma-q" data-add="${d}">${d}</button>`).join("");
  }
  function add(d) { d = (d || "").trim().toLowerCase(); if (d && !regimen.includes(d)) { regimen.push(d); renderRegimen(); } }
  function remove(d) { regimen = regimen.filter(x => x !== d); renderRegimen(); }

  async function run() {
    const btn = document.querySelector("#pharma-run");
    btn.classList.add("running"); btn.disabled = true;
    const box = document.querySelector("#pharma-results");
    box.innerHTML = '<div class="orch-log-idle">traversing interaction graph · composing Z3…</div>';
    const res = await post("/aegis/pharmacopoeia/screen", {
      drugs: regimen,
      context: { renal_failure: document.querySelector("#pharma-renal").checked,
                 hepatic: document.querySelector("#pharma-hepatic").checked },
    });
    btn.classList.remove("running"); btn.disabled = false;

    const v = document.querySelector("#pharma-verdict");
    v.className = "rbac-session " + (res.safe ? "active" : "bad");
    v.lastChild.textContent = res.safe
      ? "REGIMEN PROVEN SAFE" : `${res.violations.length} MAJOR HAZARD(S)`;
    const sc = res.severity_counts;
    document.querySelector("#pharma-summary").textContent =
      `${res.n_interactions} interactions · ${sc.Major}M / ${sc.Moderate}Mod / ${sc.Minor}min`;

    const proof = res.proof_tree.map(p => `
      <div class="dsl-cert ${p.satisfied ? "ok" : "bad"}">
        <div class="dsl-cert-h">
          <span class="dsl-cert-mark">${p.satisfied ? "✓" : "✕"}</span>
          <span class="dsl-cert-name">${p.interaction}</span>
          <span class="pharma-src">${p.source}</span>
          <span class="dsl-cert-z3 ${p.satisfied ? "ok" : "bad"}">Z3 ${p.z3_status}</span></div>
        <div class="dsl-cert-text">${p.mechanism}</div>
        <details class="dsl-cert-formula"><summary>Z3 constraint</summary>
          <code>${escapeHtml(p.constraint)}</code></details>
      </div>`).join("");
    const warns = (res.warnings || []).map(w => `
      <div class="pharma-warn"><span class="pharma-warn-sev ${SEV_CLASS[w.severity]}">${w.severity}</span>
        <span>${w.a}${w.b ? " + " + w.b : ""} — ${w.mechanism}</span>
        <span class="pharma-src">${w.source}</span></div>`).join("");

    box.innerHTML =
      (res.proof_tree.length ? `<div class="pharma-grp">HARD CONSTRAINTS (Major · blocking)</div>${proof}` : "")
      + (warns ? `<div class="pharma-grp">MANAGED WARNINGS (Moderate / Minor)</div>${warns}` : "")
      + (!res.proof_tree.length && !warns
          ? `<div class="dsl-note">✓ no interactions found in the knowledge base for this regimen.</div>`
          : res.safe
            ? `<div class="dsl-note">✓ no Major hazard — regimen proven globally safe over the loaded KB.</div>`
            : `<div class="dsl-note bad">⚠ Major hazard present — regimen blocked. ${res.violations.join("; ")}</div>`);
  }

  function escapeHtml(s) {
    return String(s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  }

  document.addEventListener("click", e => {
    if (e.target.closest("#launch-pharma")) open();
    if (e.target.closest("#pharma-close")) close();
    if (e.target.closest("#pharma-run")) run();
    if (e.target.closest("#pharma-addbtn")) { add(document.querySelector("#pharma-input").value); document.querySelector("#pharma-input").value = ""; }
    const rm = e.target.closest("[data-rm]"); if (rm) remove(rm.dataset.rm);
    const q = e.target.closest("[data-add]"); if (q) add(q.dataset.add);
  });
  document.addEventListener("keydown", e => {
    if (e.key === "Enter" && e.target.id === "pharma-input") {
      add(e.target.value); e.target.value = "";
    }
  });

  return { open, close };
})();

/* ---------------- PHYSIOLOGIC IDENTITY GUARD ---------------- */
window.AegisIdentity = (function () {
  const api = (p, opt) => fetch(window.AEGIS.http(p), opt).then(r => r.json());
  const post = (p, body) => api(p, { method: "POST", headers: { "Content-Type": "application/json" },
                                     body: JSON.stringify(body || {}) });
  let loaded = false;

  function open() { document.querySelector("#identity-app").classList.remove("hidden"); load(); }
  function close() { document.querySelector("#identity-app").classList.add("hidden"); }

  async function load() {
    const box = document.querySelector("#idg-validation");
    if (!loaded) box.innerHTML = '<div class="orch-log-idle">enrolling cohort · measuring separability…</div>';
    let d; try { d = await api("/aegis/identity"); } catch (e) { return; }
    const ids = d.status.patients || [];
    const opts = ids.map(p => `<option value="${p}">${p}</option>`).join("");
    const tgt = document.querySelector("#idg-target"), src = document.querySelector("#idg-source");
    tgt.innerHTML = opts; src.innerHTML = opts;
    // default the monitored target to the currently-selected patient
    const sel = window.AegisPatient?.bed();
    if (sel && ids.includes(sel)) tgt.value = sel;
    if (src.options.length > 1) src.selectedIndex = (src.value === tgt.value) ? 0 : src.selectedIndex;
    if (src.value === tgt.value && src.options.length > 1)
      src.selectedIndex = [...src.options].findIndex(o => o.value !== tgt.value);
    renderValidation(d.validation);
    loaded = true;
  }

  function renderValidation(v) {
    document.querySelector("#idg-validation").innerHTML = `
      <div class="idg-val-grid">
        <div class="idg-val"><b>${v.auroc}</b><span>AUROC · same vs different</span></div>
        <div class="idg-val"><b>${(v.eer*100).toFixed(1)}%</b><span>equal-error rate</span></div>
        <div class="idg-val"><b>${v.same_median}</b><span>self distance (median)</span></div>
        <div class="idg-val"><b>${v.diff_median}</b><span>impostor distance (median)</span></div>
      </div>
      <div class="idg-caveat">${v.interpretation}. ${v.note_frr}</div>`;
  }

  function renderGauge(o) {
    const d = o.distance, thr = o.threshold;
    const pct = Math.min(100, (d / (thr * 3)) * 100);
    const mark = Math.min(100, (thr / (thr * 3)) * 100);
    document.querySelector("#idg-gauge").innerHTML = `
      <div class="idg-gauge-h">Mahalanobis distance
        <b class="${o.fault ? 'bad' : 'ok'}">${d}</b> / threshold ${thr}</div>
      <div class="idg-bar">
        <div class="idg-bar-fill ${o.fault ? 'bad' : 'ok'}" style="width:${pct}%"></div>
        <div class="idg-bar-thr" style="left:${mark}%"></div>
      </div>`;
  }

  function renderGate(o) {
    const g = o.gate, fault = o.fault;
    document.querySelector("#idg-state").className = "rbac-session " + (fault ? "bad" : "active");
    document.querySelector("#idg-state").lastChild.textContent =
      fault ? "PATIENT_IDENTITY_FAULT" : "identity verified";
    document.querySelector("#idg-gate").innerHTML = `
      <div class="idg-verdict ${g.dispense_allowed ? 'ok' : 'bad'}">
        <span class="idg-verdict-ico">${g.dispense_allowed ? '✓' : '⊘'}</span>
        <div><b>${g.dispense_allowed ? 'DISPENSE AUTHORIZED' : 'DISPENSE LOCKED'}</b>
          <span>Z3 ${g.z3_status} · ${g.message}</span></div></div>
      ${fault ? `<div class="idg-fault">
        <div class="idg-fault-h">⚠ ${ (o.events && o.events[0] && o.events[0].type) || 'PATIENT_IDENTITY_FAULT'}</div>
        <div class="idg-fault-act">feeds suspended · nurse shell alerted · dispensing locked</div></div>` : ""}
      <div class="idg-proof"><span>invariant</span><code>${g.proof}</code></div>`;
  }

  async function swap() {
    const target = document.querySelector("#idg-target").value;
    const source = document.querySelector("#idg-source").value;
    const btn = document.querySelector("#idg-swap"); btn.classList.add("running"); btn.disabled = true;
    const o = await post("/aegis/identity/swap", { target, source });
    btn.classList.remove("running"); btn.disabled = false;
    if (o.status === "error") return;
    renderGauge(o); renderGate(o);
  }

  async function selfCheck() {
    const target = document.querySelector("#idg-target").value;
    const o = await post("/aegis/identity/self_check", { target });
    if (o.status === "error") return;
    renderGauge(o); renderGate(o);
  }

  document.addEventListener("click", e => {
    if (e.target.closest("#launch-identity")) open();
    if (e.target.closest("#idg-close")) close();
    if (e.target.closest("#idg-swap")) swap();
    if (e.target.closest("#idg-self")) selfCheck();
  });

  return { open, close };
})();

/* ---------------- CLINICAL OBSERVATIONS (auto-written from camera) ---------------- */
window.AegisNotes = (function () {
  const seen = new Map();
  const PHRASE = {
    "Tachycardia": (s) => `Tachycardia — heart rate ${Math.round(s.hr)} bpm`,
    "Bradycardia": (s) => `Bradycardia — heart rate ${Math.round(s.hr)} bpm`,
    "HRV suppression": (s) => `Reduced heart-rate variability (${Math.round(s.hrv)} ms) — autonomic stress`,
    "Tachypnea": (s) => `Tachypnea — respiratory rate ${Math.round(s.rr)}/min`,
    "Reduced perfusion": () => `Reduced facial perfusion — possible peripheral vasoconstriction/shunting`,
    "Pallor": () => `Facial pallor noted`,
    "Drowsiness / ↓consciousness": () => `Reduced eye openness — drowsiness / possible altered mental status`,
    "Facial distress / grimacing": () => `Facial signs of distress / grimacing`,
    "Pain expression": () => `Facial expression consistent with pain`,
    "Labored breathing (mouth open)": () => `Sustained mouth-open posture — possible laboured breathing`,
    "Micro-tremor / rigors": () => `Facial micro-tremor — possible rigors/chills`,
    "Elevated blink rate": (s) => `Elevated blink rate (${Math.round(s.blinkRate)}/min)`,
    "Low signal quality": () => `Low optical signal quality — readings degraded`,
  };
  const now = () => new Date().toLocaleTimeString("en-GB");
  function addNote(text, tier) {
    const log = document.querySelector("#notes-log"); if (!log) return;
    const d = document.createElement("div"); d.className = "note " + (tier || "");
    d.innerHTML = `<span class="ni"></span><time>${now()}</time><span class="nt">${text}</span>`;
    log.prepend(d); while (log.children.length > 40) log.removeChild(log.lastChild);
  }
  function update(s) {
    const imp = document.querySelector("#notes-impression"); if (!imp) return;
    const notable = (s.signs || []);
    const active = new Map(notable.map((x) => [x.name, x]));
    for (const [name, x] of active) if (!seen.has(name)) {
      const ph = PHRASE[name] ? PHRASE[name](s) : name;
      addNote("Onset — " + ph, x.tier === "crit" ? "crit" : x.tier === "dys" ? "warn" : "");
    }
    for (const name of seen.keys()) if (!active.has(name) && PHRASE[name]) addNote("Resolved — " + name.replace(/ \(.*\)/, ""), "");
    seen.clear(); for (const [n, x] of active) seen.set(n, x);
    const worst = notable.reduce((a, b) => (b.sev > (a ? a.sev : 0) ? b : a), null);
    let txt, cls = "";
    if (!notable.length) {
      txt = s.hr ? `No acute distress. Optical vitals within normal limits — HR ${Math.round(s.hr)} bpm, RR ${Math.round(s.rr || 0)}/min. Posture: ${s.posture}.`
        : "Awaiting optical analysis…";
    } else {
      const lead = notable.slice(0, 3).map((x) => x.name.replace(/ \/.*| \(.*/, "").toLowerCase()).join(", ");
      txt = `Patient exhibits ${lead}${notable.length > 3 ? ` (+${notable.length - 3} more)` : ""}. `
        + `HR ${Math.round(s.hr || 0)} bpm, RR ${Math.round(s.rr || 0)}/min, perfusion ${Math.round((s.perfusion || 0) * 100)}%. Posture: ${s.posture}.`;
      cls = worst && worst.sev > 0.66 ? "crit" : "warn";
    }
    imp.className = "notes-impression " + cls; imp.textContent = txt;
    const st = document.querySelector("#notes-state"); if (st) st.textContent = notable.length ? (cls === "crit" ? "attention" : "observing") : "monitoring";
  }
  return { update };
})();
