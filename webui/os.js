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
    // 8 layers boot, so keep 8 — the old cap of 7 evicted layer 1 the moment layer 8
    // arrived, and the "8-layer cold boot" footer described a list nobody ever saw whole.
    while (log.children.length > 8) log.removeChild(log.firstChild);
    fill.style.width = `${((i + 1) / 8) * 100}%`;
    await sleep(180 + Math.random() * 120);
  }
  await sleep(360);
  $("#boot").classList.add("gone");
  setTimeout(() => { $("#boot").remove(); $("#os").classList.remove("hidden"); window.AegisTwin?.resize(); }, 650);
  connect(); animate(); tickClock(); window.AegisPatient?.load(); bindWhatifPK();
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
  renderOrganHud(d);
  window.AegisStory?.onSnapshot(d);
  window.AegisConsensus?.onSnapshot(d);
  window.AegisNotes?.updateFromBackend(d);   // physiologic observations when camera is idle
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

/* ---------------- MULTI-ORGAN DYSFUNCTION HUD (SOFA · qSOFA · MODS) ---------- */
function renderOrganHud(d) {
  const hud = $("#organ-hud"); if (!hud) return;
  const bs = d.body_state; const sofa = bs?.sofa, mods = bs?.mods, qs = bs?.qsofa, s3 = bs?.sepsis3;
  if (!bs || !sofa || sofa.organs_scored === 0) { hud.classList.add("hidden"); return; }
  hud.classList.remove("hidden");
  const modsTone = mods.count >= 2 ? "crit" : mods.count === 1 ? "warn" : "ok";
  const worsening = (mods.worsening || []);
  const eta = mods.soonest_eta_critical_min;
  const worseLine = worsening.length
    ? `<div class="ohud-worse"><span class="ohud-arrow">▲</span> worsening: <b>${worsening.join(", ")}</b>`
      + (eta != null ? ` · ETA→critical <b>${eta}m</b>` : "") + `</div>`
    : "";
  hud.innerHTML = `
    <div class="ohud-row">
      <span class="ohud-k">SOFA</span>
      <span class="ohud-sofa ${modsTone}">${sofa.total}</span>
      <span class="ohud-band">mortality ${sofa.mortality_band}</span>
      <span class="ohud-scored">${sofa.organs_scored}/6 organs</span>
    </div>
    <div class="ohud-row">
      <span class="ohud-pill ${qs.high_risk ? "crit" : "dim"}">qSOFA ${qs.score}/3</span>
      <span class="ohud-pill ${modsTone}">${mods.is_mods ? "MODS · " + mods.count + " organs" : mods.count === 1 ? "1 organ failing" : "no organ failure"}</span>
      ${s3?.flag ? `<span class="ohud-pill crit">SEPSIS-3 ✚</span>` : ""}
      ${mods.cascade_risk > 0 ? `<span class="ohud-cascade">cascade ${Math.round(mods.cascade_risk * 100)}%</span>` : ""}
    </div>
    ${worseLine}
    ${mods.onset_now ? `<div class="ohud-onset">⚡ MODS ONSET — second organ system entered failure</div>` : ""}`;
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
// Clinical-instrument palette. Each tier is ONE colour — the arc used to ramp between
// two hues (a->b), which made the stroke's colour depend on how full it was as well as
// on the tier, so a mid-range LOW and a low-range MEDIUM shaded into each other. A tier
// now reads the same at 1% as at 99%. Values mirror --low/--med/--high in os.css.
const RISK_THEME = {
  LOW:    { a: "#40b880", b: "#40b880", glow: "64,184,128" },
  MEDIUM: { a: "#e2a33c", b: "#e2a33c", glow: "226,163,60" },
  HIGH:   { a: "#e5544c", b: "#e5544c", glow: "229,84,76" },
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
        <stop id="rg0" offset="0" stop-color="#40b880"/><stop id="rg1" offset="1" stop-color="#40b880"/></linearGradient>
      <radialGradient id="rgCore" cx="50%" cy="50%" r="55%">
        <stop id="rgc0" offset="0" stop-color="rgba(64,184,128,0)"/>
        <stop offset="100%" stop-color="rgba(64,184,128,0)"/></radialGradient>
    </defs>
    <circle id="rg-core" cx="${cx}" cy="${cy}" r="46" fill="url(#rgCore)"/>
    <g class="risk-ticks">${ticks}</g>
    <path d="${fullPath}" fill="none" stroke="rgba(255,255,255,.09)" stroke-width="6" stroke-linecap="butt"/>
    <path id="rg-fill" class="risk-arc-fill" d="${fullPath}" pathLength="1000" fill="none"
      stroke="url(#rg)" stroke-width="6" stroke-linecap="butt"
      stroke-dasharray="1000" stroke-dashoffset="1000"/>
    ${marker(0.04, "#e2a33c")}${marker(0.12, "#e5544c")}
    <circle id="rg-dot" class="risk-arc-dot" r="3.4" fill="#f2f3f5" opacity="0"/>`;
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
  $("#rg0")?.setAttribute("stop-color", th.a);
  $("#rg1")?.setAttribute("stop-color", th.b);
  // The dial's inner core wash stays fully transparent: it was a tier-tinted radial
  // glow behind the number, i.e. coloured light directly under the one reading that
  // has to stay legible at a glance.
  core?.setAttribute("stop-color", `rgba(${th.glow},0)`);
  const [dx, dy] = polar(RG.cx, RG.cy, RG.r, ARC_START + ARC_SWEEP * f);
  dot.setAttribute("cx", dx.toFixed(2));
  dot.setAttribute("cy", dy.toFixed(2));
  dot.setAttribute("opacity", f > 0.003 ? "1" : "0");
  // No drop-shadow on arc or dot (rule 2). The arc's colour and length already carry
  // the tier and the magnitude; a halo only smears the edge you read the value off.
}

/* ---------------- BODY TWIN (3D — see twin3d.js) ---------------- */
function renderTwin(d) {
  window.AegisTwin?.update(d);
  _lastPatientName = d?.patient?.name || _lastPatientName;
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
      <span class="vk">${SHORT[c.key] || c.label}</span>
      <span class="vnum"><span class="vv">${c.value ?? "—"}</span><span class="vu">${c.unit}</span></span>
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
let _whatifState = null;   // {whatif, vitals, risk} kept for the drill-down modal
function renderWhatif(d) {
  const w = d.whatif;
  const box = $("#whatif-list");
  if (!w || !w.candidates) {
    _whatifState = null;
    const lvl = d.risk?.level || "LOW";
    box.innerHTML = `<div class="muted">${lvl === "LOW"
      ? "No active projection — patient stable. Treatment paths are projected automatically when risk becomes elevated."
      : "Projecting treatment trajectories…"}</div>`;
    return;
  }
  _whatifState = { whatif: w, vitals: d.vitals || {}, risk: d.risk || {} };
  const curRisk = num(d.risk?.ml_probability ?? d.body_state?.sepsis_probability);
  // display order: recommended pinned first, then safest→riskiest, rejected last —
  // but keep each candidate's ORIGINAL index so the modal opens the right path.
  const order = w.candidates.map((c, i) => ({ c, i })).sort((a, b) => {
    if ((a.c.name === w.recommended) !== (b.c.name === w.recommended)) return a.c.name === w.recommended ? -1 : 1;
    if (a.c.safe !== b.c.safe) return a.c.safe ? -1 : 1;
    return num(a.c.summary?.final_risk) - num(b.c.summary?.final_risk);
  });
  box.innerHTML =
    `<div class="whatif-hint">${w.candidates.length} paths · ${w.horizon_steps || "—"}-step horizon · <b>tap a row for the full projection</b></div>` +
    order.map(({ c, i }) => {
      const best = c.name === w.recommended, s = c.summary || {};
      const fr = num(s.final_risk), delta = fr - curRisk;
      const dCls = delta < -0.001 ? "down" : delta > 0.001 ? "up" : "flat";
      const dTxt = delta < -0.001 ? `▼${Math.abs(delta*100)|0}` : delta > 0.001 ? `▲${(delta*100)|0}` : "◦";
      return `<div class="cand ${best ? "best" : ""} ${c.safe ? "" : "unsafe"}" data-cand="${i}" role="button" tabindex="0">
      <span class="cand-dot ${c.safe ? (best ? "rec" : "safe") : "no"}"></span>
      <span class="cand-name">${c.name.replace(/_/g, " ")}</span>
      <span class="cand-spark"><canvas data-traj='${JSON.stringify((c.trajectory||[]).map(t=>t.projected_risk))}'></canvas></span>
      <span class="cand-risk"><span class="cand-delta ${dCls}">${dTxt}</span><b>${(fr*100)|0}%</b></span>
      <span class="cand-chev">›</span>
    </div>`;
    }).join("");
  box.querySelectorAll("canvas[data-traj]").forEach(cv => sparkline(cv, JSON.parse(cv.dataset.traj)));
  box.querySelectorAll(".cand[data-cand]").forEach(el => {
    // Tapping a path now opens the treatment twin AT that path: the body
    // progression, the exposures and that option's upsides/downsides. The old
    // trajectory modal is still reachable with shift-click for the step-by-step
    // vitals table, but the default is the thing you actually want to see.
    const cand = w.candidates[+el.dataset.cand];
    const go = (e) => {
      if (e && e.shiftKey) return openWhatifModal(+el.dataset.cand);
      if (!window.AegisPK) return openWhatifModal(+el.dataset.cand);
      window.AegisPK.open("session", { name: _lastPatientName || "Monitored patient",
                                       option: cand && cand.name });
    };
    el.addEventListener("click", go);
    el.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); go(e); } });
  });
}
function sparkline(cv, arr) {
  const w = cv.clientWidth || 90, h = 22, dpr = window.devicePixelRatio || 1;
  cv.width = w * dpr; cv.height = h * dpr;
  const ctx = cv.getContext("2d"); ctx.scale(dpr, dpr); ctx.clearRect(0, 0, w, h); if (!arr.length) return;
  const pad = 3, lo = Math.min(...arr, 0), hi = Math.max(...arr, 1);
  const px = i => i / Math.max(1, arr.length - 1) * (w - pad * 2) + pad;
  const py = v => h - pad - ((v - lo) / (hi - lo || 1)) * (h - pad * 2);
  // canvas cannot read CSS custom properties, so --accent (#7aa2f7) is duplicated here
  // by hand — as tactical.js and twin3d.js already do. The old literal was #5ccaf0, a
  // cyan left over from the retired theme and the last one on the desktop.
  const grad = ctx.createLinearGradient(0, 0, 0, h);
  grad.addColorStop(0, "rgba(122,162,247,.26)"); grad.addColorStop(1, "rgba(122,162,247,0)");
  ctx.beginPath(); ctx.moveTo(px(0), h - pad);
  arr.forEach((v, i) => ctx.lineTo(px(i), py(v))); ctx.lineTo(px(arr.length - 1), h - pad); ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();
  ctx.beginPath(); ctx.lineWidth = 1.5; ctx.strokeStyle = "#7aa2f7"; ctx.lineJoin = "round";
  arr.forEach((v, i) => i ? ctx.lineTo(px(i), py(v)) : ctx.moveTo(px(i), py(v)));
  ctx.stroke();
}

/* The pharmacokinetic twin, opened from the what-if card. It sits here rather
   than behind a protocol panel because "what would this treatment do" and "where
   would it actually go" are the same question asked one step apart, and the second
   one was previously three clicks deep in a different console. Unlike the what-if
   projection it does NOT require elevated risk — you can ask where a drug lands in
   a stable patient — so the button is always live. */
function bindWhatifPK() {
  const b = $("#whatif-pk");
  if (!b) return;
  b.addEventListener("click", () => {
    if (!window.AegisPK) return;
    window.AegisPK.open("session", { name: _lastPatientName || "Monitored patient" });
  });
}
let _lastPatientName = null;

/* ---------------- WHAT-IF DRILL-DOWN MODAL ---------------- */
const ANTIBIOTICS = { 0: "none", 1: "Piperacillin-Tazobactam", 2: "Meropenem", 3: "Cefepime", 4: "Vancomycin" };
function openWhatifModal(idx) {
  const st = _whatifState; if (!st) return;
  const c = st.whatif.candidates[idx]; if (!c) return;
  const modal = $("#whatif-modal");
  modal.innerHTML = whatifModalHTML(c, st);
  modal.classList.remove("hidden");
  requestAnimationFrame(() => modal.classList.add("show"));
  // wire close affordances
  const close = () => closeWhatifModal();
  modal.querySelector(".wm-close")?.addEventListener("click", close);
  modal.querySelector(".wm-scrim")?.addEventListener("click", close);
  document.addEventListener("keydown", _whatifEsc);
  // draw the multi-channel projection charts after layout settles
  requestAnimationFrame(() => drawWhatifCharts(c, st));
}
function _whatifEsc(e) { if (e.key === "Escape") closeWhatifModal(); }
function closeWhatifModal() {
  const modal = $("#whatif-modal");
  modal.classList.remove("show");
  document.removeEventListener("keydown", _whatifEsc);
  setTimeout(() => { modal.classList.add("hidden"); modal.innerHTML = ""; }, 260);
}

function whatifModalHTML(c, st) {
  const s = c.summary || {}, t = c.treatment || {}, v = st.vitals || {};
  const best = c.name === st.whatif.recommended;
  const curRisk = num(st.risk?.ml_probability), finRisk = num(s.final_risk);
  const dRisk = finRisk - curRisk;
  const curMap = num(v.map), finMap = num(s.final_map);
  const traj = c.trajectory || [];
  const first = traj[0] || {}, last = traj[traj.length - 1] || {};
  const z3 = c.verification || {};
  const { pros, cons } = whatifProsCons(c, st);

  const dose = (label, val, unit) =>
    `<div class="wm-dose"><span class="wm-dose-l">${label}</span><span class="wm-dose-v">${val}<i>${unit}</i></span></div>`;

  const compare = (label, cur, fin, unit, goodDir) => {
    const dv = fin - cur;
    const good = goodDir === 0 ? Math.abs(dv) < 1e-6 : (goodDir > 0 ? dv >= 0 : dv <= 0);
    const arrow = Math.abs(dv) < 1e-6 ? "◦" : dv > 0 ? "▲" : "▼";
    const cls = Math.abs(dv) < 1e-6 ? "flat" : good ? "good" : "bad";
    return `<div class="wm-cmp">
      <div class="wm-cmp-l">${label}</div>
      <div class="wm-cmp-row"><span class="wm-cmp-cur">${fmtNum(cur)}${unit}</span>
        <span class="wm-cmp-arrow ${cls}">${arrow}</span>
        <span class="wm-cmp-fin ${cls}">${fmtNum(fin)}${unit}</span></div>
    </div>`;
  };

  return `
  <div class="wm-scrim"></div>
  <div class="wm-panel ${best ? "best" : ""}" role="dialog" aria-modal="true" aria-label="Treatment projection">
    <div class="wm-glow"></div>
    <header class="wm-head">
      <div class="wm-head-l">
        <span class="wm-sigil">⑂</span>
        <div>
          <div class="wm-title">${c.name.replace(/_/g, " ")}</div>
          <div class="wm-sub">projected treatment twin · ${st.whatif.horizon_steps || traj.length} steps · severity ${st.whatif.severity ?? "—"}</div>
        </div>
      </div>
      <div class="wm-head-r">
        ${best ? '<span class="wm-badge rec">◈ recommended path</span>' : ""}
        <span class="wm-badge ${c.safe ? "safe" : "no"}">${c.safe ? "✓ Z3 certified safe" : "✕ rejected"}</span>
        <button class="wm-close" aria-label="close">✕</button>
      </div>
    </header>

    <div class="wm-body">
      <!-- headline outcome -->
      <section class="wm-hero">
        <div class="wm-hero-risk">
          <div class="wm-hero-label">projected sepsis risk</div>
          <div class="wm-hero-val ${finRisk >= .66 ? "crit" : finRisk >= .33 ? "warn" : "ok"}">${(finRisk*100)|0}<i>%</i></div>
          <div class="wm-hero-delta ${dRisk < -0.001 ? "good" : dRisk > 0.001 ? "bad" : "flat"}">
            ${dRisk < -0.001 ? `▼ ${Math.abs(dRisk*100)|0}% vs now` : dRisk > 0.001 ? `▲ ${(dRisk*100)|0}% vs now` : "◦ unchanged"}</div>
        </div>
        <div class="wm-hero-doses">
          <div class="wm-doses-h">treatment composition</div>
          ${dose("fluids", fmtNum(t.fluid_rate), "mL/kg/h")}
          ${dose("vasopressor", fmtNum(t.vasopressor_dose), "µg/kg/m")}
          ${dose("antibiotic", ANTIBIOTICS[t.antibiotic_code] || `#${t.antibiotic_code}`, "")}
        </div>
      </section>

      <!-- projection charts -->
      <section class="wm-charts">
        <div class="wm-chart-h">projected physiology over horizon <span class="wm-chart-note">now → step ${traj.length}</span></div>
        <div class="wm-chart-grid">
          ${chartCell("risk", first, last)}
          ${chartCell("map", first, last)}
          ${chartCell("heart_rate", first, last)}
          ${chartCell("sweat_lactate", first, last)}
        </div>
      </section>

      <!-- current → projected state -->
      <section class="wm-state">
        <div class="wm-sec-h">patient state · now → end of horizon</div>
        <div class="wm-cmp-grid">
          ${compare("sepsis risk", curRisk*100, finRisk*100, "%", -1)}
          ${compare("MAP", curMap, finMap, " mmHg", 1)}
          ${compare("min MAP (perfusion floor)", curMap, num(s.min_map), " mmHg", 1)}
          ${compare("mean risk (horizon)", curRisk*100, num(s.mean_risk)*100, "%", -1)}
        </div>
      </section>

      <!-- upsides / downsides -->
      <section class="wm-tradeoff">
        <div class="wm-trade-col up">
          <div class="wm-trade-h">▲ upsides</div>
          ${pros.length ? pros.map(p => `<div class="wm-trade-item good">${p}</div>`).join("") : '<div class="wm-trade-item muted">none identified</div>'}
        </div>
        <div class="wm-trade-col down">
          <div class="wm-trade-h">▼ downsides &amp; cautions</div>
          ${cons.length ? cons.map(p => `<div class="wm-trade-item bad">${p}</div>`).join("") : '<div class="wm-trade-item muted">no material downside projected</div>'}
        </div>
      </section>

      <!-- verification + honesty note -->
      <section class="wm-verify ${c.safe ? "" : "bad"}">
        <span class="wm-verify-i">${c.safe ? "✓" : "✕"}</span>
        <span>Z3 <b>${z3.z3_status || "—"}</b> · ${c.safe ? "all safety obligations satisfied" : (z3.message || "safety obligation violated")}</span>
      </section>
      <div class="wm-disclaimer">${st.whatif.note || "Illustrative pharmacodynamic projection for decision exploration — not a validated outcome forecast."}</div>
    </div>
  </div>`;
}

/* Clinical metric config: fixed physiologic Y-domains (honest scale, not tight
   autoscale), a reference threshold line, and whether rising is clinically good. */
const METRICS = {
  risk:          { key: "projected_risk", label: "sepsis risk", unit: "%", color: "#f2606e",
                   domain: [0, 100], scale: 100, refs: [{ v: 66, t: "HIGH" }, { v: 33, t: "MED" }], goodDir: -1 },
  map:           { key: "map", label: "MAP", unit: "mmHg", color: "#5ccaf0",
                   domain: [40, 120], scale: 1, refs: [{ v: 65, t: "perfusion floor" }], goodDir: 1 },
  heart_rate:    { key: "heart_rate", label: "heart rate", unit: "bpm", color: "#f5c451",
                   domain: [40, 150], scale: 1, refs: [{ v: 100, t: "tachy" }], goodDir: 0 },
  sweat_lactate: { key: "sweat_lactate", label: "lactate", unit: "mmol/L", color: "#9a8cf0",
                   domain: [0, 6], scale: 1, refs: [{ v: 2, t: "sepsis ≥2" }], goodDir: -1 },
};
function chartCell(key, first, last) {
  const m = METRICS[key];
  const a = num(first[m.key]) * m.scale, b = num(last[m.key]) * m.scale;
  const dv = b - a, eps = key === "risk" ? 0.5 : 0.1;
  const arrow = Math.abs(dv) < eps ? "◦" : dv > 0 ? "▲" : "▼";
  const trend = Math.abs(dv) < eps ? "flat" : (m.goodDir === 0 ? "neutral" : ((dv > 0) === (m.goodDir > 0) ? "good" : "bad"));
  return `<div class="wm-chart-cell" data-chart="${key}">
    <div class="wm-cc-top"><span class="wm-cc-label">${m.label}</span>
      <span class="wm-cc-end ${trend}">${fmtNum(b)}<i>${m.unit}</i> <span class="wm-cc-arrow">${arrow}</span></span></div>
    <canvas class="wm-cc-canvas"></canvas>
  </div>`;
}

/* Derive upsides/downsides from the projected trajectory + treatment. */
function whatifProsCons(c, st) {
  const s = c.summary || {}, t = c.treatment || {}, v = st.vitals || {};
  const pros = [], cons = [];
  const curRisk = num(st.risk?.ml_probability), finRisk = num(s.final_risk);
  const dRisk = finRisk - curRisk;
  const minMap = num(s.min_map), finMap = num(s.final_map);

  if (dRisk <= -0.05) pros.push(`Cuts projected sepsis risk by <b>${Math.abs(dRisk*100)|0}%</b> over the horizon.`);
  else if (dRisk < -0.001) pros.push(`Nudges projected risk down <b>${Math.abs(dRisk*100)|0}%</b>.`);
  else if (dRisk > 0.02) cons.push(`Projected risk <b>rises ${(dRisk*100)|0}%</b> — does not contain the trajectory.`);
  else pros.push("Holds projected risk roughly stable.");

  if (minMap >= 65) pros.push(`Perfusion floor stays adequate (min MAP <b>${fmtNum(minMap)}</b> ≥ 65 mmHg).`);
  else if (minMap > 0) cons.push(`Perfusion dips to <b>${fmtNum(minMap)} mmHg</b> — below the 65 mmHg target at some point.`);

  if (num(t.fluid_rate) > 0) {
    if (finMap - num(v.map) >= 3) pros.push(`Fluids raise MAP toward <b>${fmtNum(finMap)} mmHg</b>.`);
    if (num(t.fluid_rate) >= 5) cons.push(`Higher fluid rate (<b>${fmtNum(t.fluid_rate)} mL/kg/h</b>) — watch for volume overload / pulmonary edema.`);
  }
  if (num(t.vasopressor_dose) > 0) {
    pros.push(`Vasopressor support defends MAP under distributive shock.`);
    cons.push(`Vasopressor (<b>${fmtNum(t.vasopressor_dose)} µg/kg/m</b>) adds afterload / arrhythmia and end-organ ischemia risk.`);
  }
  if (num(t.antibiotic_code) > 0) pros.push(`Broad-spectrum antibiotic (<b>${ANTIBIOTICS[t.antibiotic_code] || "#"+t.antibiotic_code}</b>) covers the suspected source early.`);
  else cons.push("No antibiotic — leaves a suspected infection untreated.");

  if (num(t.fluid_rate) === 0 && num(t.vasopressor_dose) === 0 && num(t.antibiotic_code) === 0) {
    cons.push("No active intervention — relies on the patient self-correcting.");
  }
  if (!c.safe) cons.push("Z3 rejected this path — a hard safety obligation is violated.");
  return { pros, cons };
}

/* Draw the four clinical projection charts inside the modal. */
function drawWhatifCharts(c, st) {
  const traj = c.trajectory || [];
  const modal = $("#whatif-modal");
  modal.querySelectorAll(".wm-chart-cell").forEach(cell => {
    const m = METRICS[cell.dataset.chart]; if (!m) return;
    const arr = traj.map(p => num(p[m.key]) * m.scale);
    drawClinicalChart(cell.querySelector("canvas"), arr, m);
  });
}
/* A real monitor-style chart: fixed clinical Y-domain, gridlines with axis
   labels, a dashed clinical threshold line, area+line, and an endpoint marker. */
function drawClinicalChart(cv, arr, m) {
  if (!cv || !arr.length) return;
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth || 240, H = cv.clientHeight || 96;
  cv.width = W * dpr; cv.height = H * dpr;
  const ctx = cv.getContext("2d"); ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
  const padL = 30, padR = 10, padT = 8, padB = 15;         // room for y-labels + x-axis
  const x0 = padL, x1 = W - padR, y0 = padT, y1 = H - padB;

  // domain: clamp to the fixed clinical band, but grow to include the data
  let [lo, hi] = m.domain;
  lo = Math.min(lo, ...arr); hi = Math.max(hi, ...arr);
  if (hi - lo < 1e-6) hi = lo + 1;
  const px = i => x0 + (i / Math.max(1, arr.length - 1)) * (x1 - x0);
  const py = v => y1 - ((v - lo) / (hi - lo)) * (y1 - y0);

  ctx.font = "9px " + "ui-monospace,'JetBrains Mono',monospace";
  ctx.textBaseline = "middle";

  // horizontal gridlines + y-axis labels (4 divisions)
  ctx.strokeStyle = "rgba(150,170,205,.08)"; ctx.lineWidth = 1;
  ctx.fillStyle = "rgba(150,171,198,.5)"; ctx.textAlign = "right";
  for (let g = 0; g <= 3; g++) {
    const val = lo + (hi - lo) * (g / 3), gy = py(val);
    ctx.beginPath(); ctx.moveTo(x0, gy); ctx.lineTo(x1, gy); ctx.stroke();
    ctx.fillText(fmtNum(val), x0 - 5, gy);
  }

  // clinical threshold reference line(s) — dashed, labeled with a dark backing
  // pill so the text stays legible wherever the projected line crosses it
  (m.refs || []).forEach(r => {
    if (r.v < lo || r.v > hi) return;
    const ty = py(r.v);
    ctx.save(); ctx.setLineDash([3, 3]); ctx.strokeStyle = "rgba(245,154,75,.4)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, ty); ctx.lineTo(x1, ty); ctx.stroke();
    ctx.setLineDash([]); ctx.textAlign = "left"; ctx.textBaseline = "middle";
    const ly = ty - 7 < y0 ? ty + 7 : ty - 7, tw = ctx.measureText(r.t).width;
    ctx.fillStyle = "rgba(8,11,20,.72)";
    ctx.fillRect(x0 + 2, ly - 6, tw + 6, 12);
    ctx.fillStyle = "rgba(248,182,128,.9)"; ctx.fillText(r.t, x0 + 5, ly);
    ctx.restore();
  });

  // area fill under the curve
  const grad = ctx.createLinearGradient(0, y0, 0, y1);
  grad.addColorStop(0, m.color + "38"); grad.addColorStop(1, m.color + "05");
  ctx.beginPath(); ctx.moveTo(px(0), y1);
  arr.forEach((v, i) => ctx.lineTo(px(i), py(v)));
  ctx.lineTo(px(arr.length - 1), y1); ctx.closePath();
  ctx.fillStyle = grad; ctx.fill();

  // the projected line
  ctx.beginPath(); ctx.lineWidth = 2; ctx.strokeStyle = m.color; ctx.lineJoin = "round"; ctx.lineCap = "round";
  arr.forEach((v, i) => i ? ctx.lineTo(px(i), py(v)) : ctx.moveTo(px(i), py(v)));
  ctx.stroke();

  // start + end markers
  const sx = px(0), sy = py(arr[0]), ex = px(arr.length - 1), ey = py(arr[arr.length - 1]);
  ctx.fillStyle = "rgba(159,171,198,.7)"; ctx.beginPath(); ctx.arc(sx, sy, 2.4, 0, 7); ctx.fill();
  ctx.fillStyle = m.color; ctx.beginPath(); ctx.arc(ex, ey, 3.4, 0, 7); ctx.fill();
  ctx.strokeStyle = m.color + "55"; ctx.lineWidth = 1.4; ctx.beginPath(); ctx.arc(ex, ey, 6, 0, 7); ctx.stroke();

  // x-axis baseline + "now / step N" ticks
  ctx.strokeStyle = "rgba(150,170,205,.14)"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(x0, y1); ctx.lineTo(x1, y1); ctx.stroke();
  ctx.fillStyle = "rgba(150,171,198,.5)"; ctx.textBaseline = "top";
  ctx.textAlign = "left"; ctx.fillText("now", x0, y1 + 4);
  ctx.textAlign = "right"; ctx.fillText("step " + arr.length, x1, y1 + 4);
}
function fmtNum(v) {
  const n = Number(v);
  if (!isFinite(n)) return "—";
  return Math.abs(n) >= 100 ? Math.round(n) : (Math.round(n * 10) / 10).toFixed(1).replace(/\.0$/, "");
}

/* ---------------- PROTOCOL + PROOF ---------------- */
function renderProto(d) {
  const p = d.protocol, v = d.verification, box = $("#proto");
  if (!p) {
    // No synthesis yet — say why, honestly, per the current risk band.
    const lvl = d.risk?.level || "LOW";
    const msg = lvl === "LOW"
      ? "No treatment indicated — patient within normal limits. Synthesis engages automatically when risk becomes elevated."
      : "Synthesizing…";
    box.innerHTML = `<div class="muted">${msg}</div>` + narr(d);
    return;
  }
  const t = p.protocol || {};
  const safe = v?.safe;
  const advisory = !!p.advisory;   // MEDIUM: proven candidate, not an active order
  box.innerHTML = `
    ${advisory ? '<div class="proto-advisory">◇ ADVISORY · elevated risk — projected candidate, not yet indicated for administration</div>' : ""}
    <div class="rx">
      <div>fluids<b>${t.fluid_rate ?? "—"}</b>mL/kg/h</div>
      <div>vaso<b>${t.vasopressor_dose ?? "—"}</b>µg/kg/m</div>
      <div>antibiotic<b>#${t.antibiotic_code ?? "—"}</b></div>
    </div>
    <div class="proof ${safe ? "" : "bad"}"><i>${safe ? "✓" : "✕"}</i>
      Z3 ${v?.z3_status || "—"} · ${safe ? "mathematically certified safe" : (v?.message || "rejected")}</div>
    ${portableProofLine(v?.portable_proof)}
    ${(safe && !advisory) ? `<button class="dock-btn accent pulse-button" onclick="window.AegisTwin.triggerInfusion('fluids'); setTimeout(() => window.AegisTwin.triggerInfusion('antibiotic'), 1200); setTimeout(() => window.AegisTwin.triggerInfusion('vasopressor'), 2400);" style="width:100%;margin-top:8px;animation: pulse 2s infinite;cursor:pointer;background:#00d8ff;color:#0b1020;font-weight:bold;border:none;border-radius:4px;padding:8px 0;box-shadow: 0 0 10px rgba(0,216,255,0.4);">▲ APPROVE & INFUSE PROTOCOL</button>` : ""}
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
  drawTrace("#ecg", ecgBuf);
  drawTrace("#ppg", ppgBuf);
  requestAnimationFrame(animate);
}
/* A bedside trace, drawn like one.
   The previous version had three separate quality bugs stacked on top of each other:
   the canvas bitmap was sized in CSS pixels (so it resolved at half density on every
   retina display), its height attribute said 72 while its CSS said 30 (so the whole
   waveform was squashed to 40% and resampled), and it was stroked with a 7px coloured
   shadowBlur — a glow, which this identity does not use, and which at 30px tall was
   most of what you actually saw. It is now sized from its own CSS box at full device
   resolution, with a calibration grid behind it and a leading-edge dot, so it reads as
   an instrument sweep rather than a smear.
   The colour is neutral on purpose: saturated hue is reserved for clinical state, and
   the old traces were cyan and MINT-GREEN — the same green this system uses to mean
   "low risk", sitting permanently on screen regardless of the patient. */
function drawTrace(id, buf) {
  const c = $(id); if (!c) return;
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  // Bail rather than fall back to a guess. A `|| 320` default here is not a safety net:
  // on the frames before the panel is laid out it BAKES 320x52 into the backing store,
  // and the element then keeps that resolution for as long as nothing resizes it — which
  // on a throttled/background tab (where rAF stops firing) is forever.
  const W = c.clientWidth, h = c.clientHeight;
  if (!W || !h) return;
  if (c.width !== Math.round(W * dpr) || c.height !== Math.round(h * dpr)) {
    c.width = Math.round(W * dpr); c.height = Math.round(h * dpr);
  }
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, h);
  if (!buf.length) return;
  // Map the buffer ACROSS the canvas instead of assuming one sample per pixel. The
  // buffer is trimmed to a width `animate()` measures separately, so the moment the
  // two disagree — which they do on the very first frames, when the panel has not been
  // laid out and both fall back to different guesses — the sweep silently drew into
  // only the left fraction of its box and looked like a stub.
  const xOf = i => buf.length < 2 ? 0 : i / (buf.length - 1) * W;
  const yOf = v => h * 0.56 - v * h * 0.40;

  // calibration grid — the faint reference a monitor sweep is read against
  ctx.strokeStyle = "rgba(255,255,255,.045)"; ctx.lineWidth = 1;
  ctx.beginPath();
  for (let gx = 0; gx <= W; gx += 26) { ctx.moveTo(Math.round(gx) + .5, 0); ctx.lineTo(Math.round(gx) + .5, h); }
  for (let gy = h * .5 % 13; gy <= h; gy += 13) { ctx.moveTo(0, Math.round(gy) + .5); ctx.lineTo(W, Math.round(gy) + .5); }
  ctx.stroke();
  // baseline
  ctx.strokeStyle = "rgba(255,255,255,.10)";
  ctx.beginPath(); ctx.moveTo(0, Math.round(yOf(0)) + .5); ctx.lineTo(W, Math.round(yOf(0)) + .5); ctx.stroke();

  // the sweep itself
  ctx.beginPath();
  buf.forEach((v, i) => { i ? ctx.lineTo(xOf(i), yOf(v)) : ctx.moveTo(xOf(i), yOf(v)); });
  ctx.lineWidth = 1.4; ctx.lineJoin = "round"; ctx.lineCap = "round";
  ctx.strokeStyle = "rgba(242,243,245,.82)";
  ctx.stroke();

  // leading edge: where the sweep is right now
  const lx = xOf(buf.length - 1), ly = yOf(buf[buf.length - 1]);
  ctx.beginPath(); ctx.arc(lx, ly, 1.9, 0, Math.PI * 2);
  ctx.fillStyle = "#f2f3f5"; ctx.fill();
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
    lastGraph = data.graph; renderLedger(data); loadSynthesis();
  }

  const esc = s => String(s == null ? "" : s).replace(/[&<>]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));
  const riskCls = r => (r >= 0.7 ? "crit" : r >= 0.4 ? "warn" : "");
  // pretty labels for known payload keys so the audit reads like a chart
  const PKEY = {
    map: "MAP (mmHg)", lactate: "lactate (mmol/L)", sweat_lactate: "lactate (mmol/L)",
    hr: "heart rate (bpm)", heart_rate: "heart rate (bpm)", spo2: "SpO₂ (%)",
    resp_rate: "resp rate (/min)", respiratory_rate: "resp rate (/min)", temp: "temp (°C)",
    skin_temp: "temp (°C)", systolic_bp: "systolic BP (mmHg)", wbc: "WBC (10⁹/L)",
    risk_level: "risk band", triggered_rules: "rules fired", indicated: "indicated action",
    z3_status: "Z3 verdict", state: "outcome state", sign: "abnormal sign", type: "event type",
  };
  const fmtVal = v => {
    if (v == null) return "—";
    if (typeof v === "number") { const a = Math.abs(v); return a >= 100 ? v.toFixed(0) : (a >= 10 ? v.toFixed(1) : v.toFixed(2)).replace(/\.0+$/, ""); }
    if (Array.isArray(v)) return v.map(String).join(", ") || "—";
    if (typeof v === "object") return JSON.stringify(v);
    return String(v);
  };

  function edgeChip(e) {
    return `<span class="story-edge ${e.evidence_type}">${esc(e.mechanism)} · ${EV[e.evidence_type] || e.evidence_type}` +
      (e.support ? ` (n=${e.support})` : "") + `</span>`;
  }

  function renderVerdict(g) {
    const box = document.querySelector("#story-verdict"); if (!box) return;
    const nodes = g.nodes || [];
    if (!nodes.length) { box.className = "story-verdict-panel"; box.innerHTML = ""; return; }
    const last = nodes[nodes.length - 1], risk = last.risk || 0;
    const band = risk >= 0.7 ? ["CRITICAL", "crit"] : risk >= 0.4 ? ["ELEVATED", "warn"] : ["STABLE", "ok"];
    const counts = {}; nodes.forEach(n => counts[n.kind] = (counts[n.kind] || 0) + 1);
    const mix = { z3_proven: 0, model_predicted: 0, observed_association: 0 };
    (g.edges || []).forEach(e => { if (mix[e.evidence_type] != null) mix[e.evidence_type]++; });
    const named = { vital_snapshot: "vital snapshots", lab_result: "labs", alert: "alerts",
      protocol_decision: "decisions", omission: "omissions", drug_dose: "drug doses", outcome: "outcomes" };
    const parts = Object.keys(named).filter(k => counts[k]).map(k => `${counts[k]} ${named[k]}`);
    box.className = "story-verdict-panel show " + band[1];
    box.innerHTML = `
      <div class="verdict-hd">
        <span class="verdict-band ${band[1]}">${band[0]}</span>
        <span class="verdict-risk">risk ${Math.round(risk * 100)}%</span>
        <span class="verdict-t">as of ${fmtT(last.t)}</span>
      </div>
      <div class="verdict-derive">verdict derived from <b>${nodes.length}</b> logged events — ${parts.join(" · ") || "—"}</div>
      <div class="verdict-mix">
        <span class="story-edge z3_proven">${mix.z3_proven} Z3-proven</span>
        <span class="story-edge model_predicted">${mix.model_predicted} model-predicted</span>
        <span class="story-edge observed_association">${mix.observed_association} association</span>
      </div>
      <div class="verdict-foot">${g.integrity_ok
        ? "✓ hash-chain verified — every event below is tamper-evident"
        : "✕ INTEGRITY FAIL — the ledger has been altered"} · click any event to expand its measured values, evidence provenance &amp; ledger hash to cross-check the OS against the chart.</div>`;
  }

  function auditBlock(n, edges) {
    const p = n.payload || {};
    const keys = Object.keys(p);
    const rows = keys.length ? keys.map(key =>
      `<div class="aud-row"><span class="aud-k">${esc(PKEY[key] || key)}</span>` +
      `<span class="aud-v">${esc(fmtVal(p[key]))}</span></div>`).join("")
      : '<div class="muted">no measured fields on this event</div>';
    const prov = (edges && edges.length) ? edges.map(e => `
      <div class="aud-prov ${e.evidence_type}">
        <span class="story-edge ${e.evidence_type}">${EV[e.evidence_type] || e.evidence_type}</span>
        <span class="aud-mech">${esc(e.mechanism)}</span>
        ${e.support ? `<span class="aud-sup">cohort n=${e.support}</span>` : ""}
        ${e.proven_by ? `<span class="aud-cert" title="Z3 safety certificate hash">cert ${esc(e.proven_by)}</span>` : ""}
        <span class="aud-econf">edge conf ${Math.round((e.confidence || 0) * 100)}%</span>
      </div>`).join("") : '<div class="muted">root event — no upstream evidence edges</div>';
    return `
      <div class="aud-sec">MEASURED VALUES <span class="aud-hint">cross-check against the chart</span></div>
      <div class="aud-grid">${rows}</div>
      <div class="aud-sec">EVIDENCE PROVENANCE <span class="aud-hint">how the OS justified this</span></div>
      ${prov}
      <div class="aud-sec">LEDGER · tamper-evident</div>
      <div class="aud-chain">
        <span class="aud-chip">seq #${n.seq}</span>
        <span class="aud-chip">risk ${Math.round((n.risk || 0) * 100)}%</span>
        <span class="aud-chip">confidence ${Math.round((n.confidence || 0) * 100)}%</span>
        <div class="aud-hash">hash <code>${esc(n.hash || "—")}</code></div>
        <div class="aud-hash">prev <code>${esc(n.prev_hash || "—")}</code></div>
      </div>`;
  }

  function renderLedger(data) {
    const g = data.graph;
    const intg = document.querySelector("#story-integrity");
    const ok = g.integrity_ok;
    intg.className = "cam-state" + (ok ? " live" : " bad");
    intg.lastChild.textContent = ok ? `chain verified · ${g.head}` : "INTEGRITY FAIL";
    document.querySelector("#story-ledger-sub").textContent =
      `${g.n_nodes} events · ${g.n_edges} edges`;

    renderVerdict(g);

    const edgesByDst = {};
    (g.edges || []).forEach(e => { (edgesByDst[e.dst] ||= []).push(e); });
    const tl = document.querySelector("#story-timeline");
    if (!g.nodes || !g.nodes.length) {
      tl.innerHTML = '<div class="muted">No clinical events yet. The graph grows as risk evolves.</div>';
      return;
    }
    tl.innerHTML = g.nodes.map((n, i) => {
      const k = KIND[n.kind] || { i: "•", c: "vital", t: n.kind };
      const inc = (edgesByDst[n.id] || []).map(edgeChip).join("");
      const risk = Math.round((n.risk || 0) * 100), conf = Math.round((n.confidence || 0) * 100);
      return `<div class="story-node ${k.c}${allExpanded ? " open" : ""}" data-id="${n.id}">
        <div class="story-node-h" data-toggle>
          <span class="story-ico">${k.i}</span>
          <span class="story-kind">${k.t}</span>
          <span class="story-seq">#${n.seq ?? i}</span>
          <span class="story-risk ${riskCls(n.risk)}">risk ${risk}%</span>
          <span class="story-conf">conf ${conf}%</span>
          <span class="story-t">${fmtT(n.t)}</span>
          <span class="story-caret">▸</span></div>
        ${inc ? `<div class="story-edges">${inc}</div>` : ""}
        ${summaryLine(n)}
        <div class="story-audit"${allExpanded ? "" : " hidden"}>${auditBlock(n, edgesByDst[n.id] || [])}
          <div class="story-node-acts"><button class="story-mini" data-incident="${n.id}">⌖ trace incident from here</button></div>
        </div>
      </div>`;
    }).join("");
    // header click toggles the full audit; the trace button anchors an incident report
    tl.querySelectorAll(".story-node-h[data-toggle]").forEach(h => h.onclick = () => {
      const node = h.closest(".story-node"), aud = node.querySelector(".story-audit");
      const opening = aud.hasAttribute("hidden");
      opening ? aud.removeAttribute("hidden") : aud.setAttribute("hidden", "");
      node.classList.toggle("open", opening);
    });
    tl.querySelectorAll("[data-incident]").forEach(b => b.onclick = e => {
      e.stopPropagation(); runIncident(b.dataset.incident);
    });
  }

  function summaryLine(n) {
    const p = n.payload || {};
    if (n.kind === "vital_snapshot")
      return `<div class="story-pay">MAP ${fmtVal(p.map)} · lactate ${fmtVal(p.lactate ?? p.sweat_lactate)}` +
        (p.hr != null ? ` · HR ${fmtVal(p.hr)}` : "") + (p.spo2 != null ? ` · SpO₂ ${fmtVal(p.spo2)}` : "") +
        (p.resp_rate != null ? ` · RR ${fmtVal(p.resp_rate)}` : "") + `</div>`;
    if (n.kind === "omission")
      return `<div class="story-pay">indicated: ${esc(p.indicated || "treatment")} — not given</div>`;
    if (n.kind === "protocol_decision")
      return `<div class="story-pay">Z3 ${esc(p.z3_status || "certified")} · protocol applied</div>`;
    if (n.kind === "outcome")
      return `<div class="story-pay">${esc(p.state || "outcome")}</div>`;
    if (n.kind === "alert")
      return `<div class="story-pay">${esc(p.risk_level || "")} ${esc((p.triggered_rules || []).slice(0, 3).join(", "))}</div>`;
    if (n.kind === "lab_result")
      return `<div class="story-pay">${Object.keys(p).slice(0, 3).map(k => `${esc(k)} ${fmtVal(p[k])}`).join(" · ")}</div>`;
    return "";
  }

  async function loadSynthesis() {
    const box = document.querySelector("#story-synth-out"); if (!box) return;
    const b = window.AegisPatient?.bed();
    if (!b) { box.innerHTML = '<div class="muted">Select a patient to see their clinician synthesis log.</div>'; return; }
    let d; try { d = await api("/aegis/patient/" + encodeURIComponent(b) + "/synthesis"); } catch (e) { return; }
    const ann = (d && d.annotations) || [];
    if (!ann.length) {
      box.innerHTML = '<div class="muted">No clinician synthesis recorded for this patient yet. Record one from the patient terminal — it lands here, timestamped.</div>';
      return;
    }
    box.innerHTML = ann.slice().reverse().map(a => `
      <div class="synth-note${a.order_intent ? " flag" : ""}">
        <div class="synth-h"><span class="synth-author">${esc(a.author || "clinician")}</span>
          <span class="synth-t">${fmtT(a.t)}</span></div>
        <div class="synth-tx">${esc(a.text)}</div>
        ${(a.applied && a.applied.length) ? `<div class="synth-applied">✓ applied to monitoring: ${esc(a.applied.join("; "))}</div>` : ""}
        ${a.order_intent ? `<div class="synth-flag">⚠ treatment intent — NOT auto-executed; route via the certified order gate</div>` : ""}
      </div>`).join("");
  }

  let allExpanded = false;
  function toggleExpandAll() {
    allExpanded = !allExpanded;
    document.querySelectorAll("#story-timeline .story-node").forEach(node => {
      const aud = node.querySelector(".story-audit"); if (!aud) return;
      allExpanded ? aud.removeAttribute("hidden") : aud.setAttribute("hidden", "");
      node.classList.toggle("open", allExpanded);
    });
    const btn = document.querySelector("#story-expand");
    if (btn) btn.textContent = allExpanded ? "⊟ collapse all" : "⊞ expand all";
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
    if (e.target.closest("#story-expand")) toggleExpandAll();
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
    // The synthesis is already complete — reveal the certified result immediately
    // instead of gating it behind the full timeline walk, then play the cascade as
    // a quick visual flourish (was 7×620ms ≈ 4.3s of dead time before the seal).
    stampCertificate(res); renderPortableProof(res);
    for (const t of res.timeline) {
      if (t.layer !== "governance") glowStratum(t.layer);
      const certified = t.layer === "governance" ? t.certified : null;
      logLine(t.layer, stepDetail(t), certified === false ? "blocked" : "");
      await sleep(150);
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
        : "Acquiring optical signal…";
    } else {
      const lead = notable.slice(0, 3).map((x) => x.name.replace(/ \/.*| \(.*/, "").toLowerCase()).join(", ");
      txt = `Patient exhibits ${lead}${notable.length > 3 ? ` (+${notable.length - 3} more)` : ""}. `
        + `HR ${Math.round(s.hr || 0)} bpm, RR ${Math.round(s.rr || 0)}/min, perfusion ${Math.round((s.perfusion || 0) * 100)}%. Posture: ${s.posture}.`;
      cls = worst && worst.sev > 0.66 ? "crit" : "warn";
    }
    imp.className = "notes-impression " + cls; imp.textContent = txt;
    const st = document.querySelector("#notes-state"); if (st) st.textContent = notable.length ? (cls === "crit" ? "attention" : "observing") : "monitoring";
  }

  // Physiologic observations derived from the backend body-state (organ systems).
  // Drives the card whenever the camera is NOT the live source, so it is never a
  // dead "awaiting" panel. These are genuine physiologic findings (SOFA basis /
  // trend / drivers) — NOT fabricated optical readings.
  const seenBk = new Map();
  function cameraLive() {
    try { const c = window.AegisCamera?.state?.(); return !!(c && c.hr > 0 && c.enabled !== false); }
    catch (e) { return false; }
  }
  function updateFromBackend(d) {
    const imp = document.querySelector("#notes-impression"); if (!imp) return;
    if (cameraLive()) return;   // camera owns the card when it is producing signal
    const bs = d && d.body_state;
    const systems = (bs && bs.organ_systems) || [];
    const vit = (d && d.vitals) || {};
    // Notable = any organ system that has left NORMAL.
    const notable = systems
      .filter(o => o.status && o.status !== "NORMAL" && (o.severity || 0) > 0.05)
      .sort((a, b) => (b.severity || 0) - (a.severity || 0));
    const active = new Map(notable.map(o => [o.name, o]));
    for (const [name, o] of active) if (!seenBk.has(name)) {
      const drv = (o.drivers && o.drivers.length) ? " — " + o.drivers.slice(0, 2).join(", ") : (o.sofa_basis ? " — " + o.sofa_basis : "");
      const eta = (o.eta_critical_min != null) ? ` · ETA→critical ${o.eta_critical_min}m` : "";
      addNote(`Onset — ${o.name} ${o.status.toLowerCase()}${drv}${eta}`, o.severity > 0.66 ? "crit" : "warn");
    }
    for (const name of seenBk.keys()) if (!active.has(name)) addNote("Resolved — " + name + " back to normal", "");
    seenBk.clear(); for (const [n, o] of active) seenBk.set(n, o);

    const hr = Math.round(vit.heart_rate || 0), rr = Math.round(vit.respiratory_rate || 0);
    const map = Math.round(vit.map || 0), spo2 = Math.round(vit.spo2 || 0);
    let txt, cls = "";
    if (!notable.length) {
      txt = hr ? `No acute organ dysfunction. Physiologic vitals within normal limits — HR ${hr} bpm, RR ${rr}/min, MAP ${map} mmHg, SpO₂ ${spo2}%.`
        : "Monitoring — no live stream.";
    } else {
      const worst = notable[0];
      const lead = notable.slice(0, 3).map(o => o.name.toLowerCase()).join(", ");
      txt = `${worst.status === "CRITICAL" ? "Organ dysfunction" : "Physiologic stress"} in ${lead}${notable.length > 3 ? ` (+${notable.length - 3} more)` : ""}. `
        + `HR ${hr} bpm, RR ${rr}/min, MAP ${map} mmHg, SpO₂ ${spo2}%.`;
      cls = worst.severity > 0.66 ? "crit" : "warn";
    }
    imp.className = "notes-impression " + cls; imp.textContent = txt;
    const st = document.querySelector("#notes-state");
    if (st) st.textContent = notable.length ? (cls === "crit" ? "attention" : "observing") : "monitoring";
  }

  return { update, updateFromBackend };
})();
