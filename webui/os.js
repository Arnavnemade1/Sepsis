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
  setTimeout(() => { $("#boot").remove(); $("#os").classList.remove("hidden"); }, 650);
  buildBody(); connect(); animate(); tickClock();
}
const sleep = ms => new Promise(r => setTimeout(r, ms));

/* ---------------- WEBSOCKET ---------------- */
let ws, retry = 0;
function connect() {
  const url = `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/aegis/stream`;
  ws = new WebSocket(url);
  ws.onopen = () => { retry = 0; setConn(true); };
  ws.onclose = () => { setConn(false); retry++; setTimeout(connect, Math.min(5000, 600 * retry)); };
  ws.onerror = () => ws.close();
  ws.onmessage = e => { try { render(JSON.parse(e.data)); } catch (err) { console.error(err); } };
}
function setConn(live) {
  const c = $("#sb-conn"); c.classList.toggle("live", live);
  c.lastChild.textContent = live ? "linked" : "offline";
}

/* ---------------- MASTER RENDER ---------------- */
function render(d) {
  lastData = d;
  HR = num(d.vitals?.heart_rate, HR); SPO2 = num(d.vitals?.spo2, SPO2);
  renderSysbar(d); renderRisk(d); renderTwin(d); renderVitals(d);
  renderAgents(d); renderWhatif(d); renderProto(d); renderLayers(d);
  $("#dock-tick").textContent = `tick · ${d.source || "—"} · ${new Date((d.timestamp||0)*1000).toLocaleTimeString()}`;
}
const num = (v, f = 0) => (typeof v === "number" && !isNaN(v)) ? v : f;
const pct = v => `${Math.round(num(v) * 100)}%`;

/* ---------------- SYSTEM BAR ---------------- */
function renderSysbar(d) {
  const r = d.risk?.level || "LOW";
  const rb = $("#sb-risk"); rb.textContent = r; rb.className = "sysbar-risk " + r;
  $("#sb-source").textContent = d.source || "—";
  $("#sb-ver").textContent = "v" + (d.system?.version || "0.8.0");
  const p = d.patient || {};
  $("#sb-patient").innerHTML = `<b>${p.name || "Unknown"}</b> · ${p.age ?? "—"}${p.gender ? " · " + p.gender[0] : ""}`;
}

/* ---------------- RISK CORE ---------------- */
function renderRisk(d) {
  const prob = num(d.risk?.ml_probability ?? d.body_state?.sepsis_probability);
  $("#risk-prob").textContent = prob ? (prob * 100).toFixed(1) + "%" : "—";
  drawArc(prob);
  const lvl = d.risk?.level || "LOW";
  ["LOW", "MEDIUM", "HIGH"].forEach(t => $("#rt-" + t).classList.toggle("on", t === lvl));
  $("#rs-sirs").textContent = d.risk?.sirs_score ?? 0;
  $("#rs-qsofa").textContent = d.risk?.qsofa_score ?? 0;
  $("#rs-sq").textContent = pct(d.risk?.signal_quality);
  $("#twin-risk").textContent = lvl === "LOW" ? "monitoring" : lvl.toLowerCase() + " risk";
}
function drawArc(p) {
  const svg = $("#risk-arc"); const C = 2 * Math.PI * 52;
  const col = p >= 0.12 ? "#ff4d5e" : p >= 0.04 ? "#ffd23b" : "#34e1ff";
  svg.innerHTML =
    `<circle cx="60" cy="60" r="52" fill="none" stroke="rgba(120,160,220,.12)" stroke-width="9"/>
     <circle cx="60" cy="60" r="52" fill="none" stroke="${col}" stroke-width="9" stroke-linecap="round"
       stroke-dasharray="${C}" stroke-dashoffset="${C * (1 - Math.min(1, p * 3))}"
       style="filter:drop-shadow(0 0 8px ${col});transition:stroke-dashoffset .6s"/>`;
}

/* ---------------- BODY TWIN ---------------- */
function buildBody() {
  $("#twin-stage").innerHTML = `
  <svg viewBox="0 0 240 460" xmlns="http://www.w3.org/2000/svg">
    <!-- skin / silhouette -->
    <path id="organ-skin" class="organ unmonitored" d="M120 14c16 0 28 12 28 28 0 11-5 18-11 23 14 3 26 9 30 20 5 13 6 33 7 50 1 13-1 22-4 24-4 2-7-3-9-14-1-7-3-19-5-19s-2 10-2 22l3 96c1 14 3 40-1 46-3 5-11 5-13-1-3-8-6-30-9-52-1 22-4 44-7 52-2 6-10 6-13 1-4-6-2-32-1-46l3-96c0-12 0-22-2-22s-4 12-5 19c-2 11-5 16-9 14-3-2-5-11-4-24 1-17 2-37 7-50 4-11 16-17 30-20-6-5-11-12-11-23 0-16 12-28 28-28z"/>
    <!-- vascular (blood/coagulation) -->
    <path id="organ-blood" class="organ unmonitored" d="M120 96 V250" stroke-width="3" fill="none"/>
    <!-- brain -->
    <ellipse id="organ-brain" class="organ unmonitored" cx="120" cy="40" rx="17" ry="13"/>
    <!-- lungs -->
    <path id="organ-lungs" class="organ unmonitored" d="M104 116c-12 4-16 26-13 44 1 8 13 7 16-2 3-10 3-30 1-40-1-3-2-3-4-2z M136 116c12 4 16 26 13 44-1 8-13 7-16-2-3-10-3-30-1-40 1-3 2-3 4-2z"/>
    <!-- heart -->
    <path id="organ-heart" class="organ unmonitored heartbeat" d="M120 132c-5-9-20-9-20 3 0 10 14 18 20 24 6-6 20-14 20-24 0-12-15-12-20-3z"/>
    <!-- liver -->
    <path id="organ-liver" class="organ unmonitored" d="M104 168c14-3 30-2 38 2 4 2 3 12-2 16-10 7-30 8-37 2-4-4-3-18 1-20z"/>
    <!-- kidneys -->
    <path id="organ-kidneys" class="organ unmonitored" d="M101 196c5-1 8 4 8 11s-4 12-8 11c-5-1-6-7-6-11s1-10 6-11z M139 196c-5-1-8 4-8 11s4 12 8 11c5-1 6-7 6-11s-1-10-6-11z"/>
  </svg>`;
}
function renderTwin(d) {
  const regions = d.body_state?.regions || {};
  ["heart", "lungs", "brain", "skin", "kidneys", "liver", "blood"].forEach(reg => {
    const el = $("#organ-" + reg); if (!el) return;
    const st = regions[reg]?.status || "UNMONITORED";
    el.setAttribute("class", "organ " + TIER[st] + (reg === "heart" ? " heartbeat" : ""));
  });
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
    ${narr(d)}`;
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
  drawTrace("#ecg", ecgBuf, "#34e1ff", 74);
  drawTrace("#ppg", ppgBuf, "#2bd6ff", 58, true);
  // heartbeat scale
  const hb = $("#organ-heart");
  if (hb) { const s = 1 + 0.07 * Math.max(0, Math.sin(ecgPhase * 2 * Math.PI)); hb.style.transform = `scale(${s})`; }
  requestAnimationFrame(animate);
}
function drawTrace(id, buf, color, h, fill) {
  const c = $(id); if (!c) return; const ctx = c.getContext("2d"), W = c.width;
  ctx.clearRect(0, 0, W, h);
  ctx.beginPath(); ctx.lineWidth = 2; ctx.strokeStyle = color; ctx.shadowBlur = 8; ctx.shadowColor = color;
  buf.forEach((v, i) => { const x = i, y = h * 0.55 - v * h * 0.42; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
  ctx.stroke(); ctx.shadowBlur = 0;
}

/* ---------------- DOCK / CLOCK ---------------- */
document.addEventListener("click", e => {
  const b = e.target.closest(".dock-btn"); if (!b) return;
  fetch(`/aegis/simulate/${b.dataset.stage}`, { method: "POST" }).catch(() => {});
  b.animate([{ transform: "scale(.92)" }, { transform: "scale(1)" }], { duration: 180 });
});
function tickClock() {
  $("#sb-clock").textContent = new Date().toLocaleTimeString("en-GB");
  setTimeout(tickClock, 1000);
}

boot();
