/* ============================================================
   AEGIS-OS — 3D anatomical digital twin (Three.js r160, vendored)

   A medical hologram, not a coloured silhouette. A real rigged
   humanoid shell (translucent, with scanlines + a rising diagnostic
   scan-sweep + fresnel rim) houses DIGITIZED INTERNAL ORGANS — heart,
   lungs, brain, liver, kidneys, spleen, gut — each a low-poly wireframe
   volume that glows the colour of ITS system's severity. The heart
   beats at the patient's heart-rate; the lungs breathe at the
   respiratory rate. Organs whose system we don't actually monitor are
   dimmed to honest "no data" — never a confident healthy glow.

   A holo-platform (rings + radial grid), a drifting cellular particle
   field, and a slow diagnostic reticle complete the console look.

   Public API (unchanged):  window.AegisTwin = { start, update, resize, setOptical }
   Auto-mounts into #twin-stage. Spin / zoom via OrbitControls.
   ============================================================ */
import * as THREE from "three";
import { OrbitControls } from "./vendor/three/OrbitControls.js";
import { GLTFLoader } from "./vendor/three/GLTFLoader.js";

const H = 1.7;                                    // body height (feet ~0)
const BASE = new THREE.Color(0x2f7fd8);           // calm / healthy
const NODATA = new THREE.Color(0x3a4660);
const RIM = new THREE.Color(0x9fe6ff);
const TIER_HEX = { STRESS: 0xffd23b, DYSFUNCTION: 0xff9a3b, CRITICAL: 0xff4d5e };
const REGION_LABEL = {
  brain: "Neurological", heart: "Cardiovascular", lungs: "Respiratory",
  liver: "Hepatic", kidneys: "Renal", skin: "Peripheral", blood: "Immune",
};

/* Anatomical organ atlas — world Y is in body units (feet≈0, head≈1.62).
   region = which body_state region drives this organ's colour. */
const ORGANS = [
  { key: "brain",   region: "brain",   pos: [0.00, 1.560, 0.012], scale: [0.080, 0.074, 0.086], detail: 2, jit: 0.09 },
  { key: "lungL",   region: "lungs",   pos: [-0.076, 1.280, 0.004], scale: [0.044, 0.118, 0.050], detail: 1, jit: 0.10, breathe: true },
  { key: "lungR",   region: "lungs",   pos: [0.076, 1.280, 0.004], scale: [0.044, 0.118, 0.050], detail: 1, jit: 0.10, breathe: true },
  { key: "heart",   region: "heart",   pos: [-0.020, 1.225, 0.052], scale: [0.044, 0.052, 0.044], detail: 1, jit: 0.20, beat: true },
  { key: "liver",   region: "liver",   pos: [0.060, 1.060, 0.040], scale: [0.082, 0.046, 0.058], detail: 1, jit: 0.16 },
  { key: "spleen",  region: "blood",   pos: [-0.072, 1.055, 0.024], scale: [0.034, 0.040, 0.034], detail: 1, jit: 0.18 },
  { key: "kidneyL", region: "kidneys", pos: [-0.058, 0.985, -0.042], scale: [0.028, 0.046, 0.032], detail: 1, jit: 0.12 },
  { key: "kidneyR", region: "kidneys", pos: [0.058, 0.985, -0.042], scale: [0.028, 0.046, 0.032], detail: 1, jit: 0.12 },
  // anatomical context only — we don't monitor the GI tract, so it stays dim
  { key: "gut",     region: "_gut",    pos: [0.00, 0.940, 0.040], scale: [0.072, 0.060, 0.052], detail: 2, jit: 0.10 },
];

// shell zone colours (mutated in place → live in the shader)
const ZONE = {
  head: BASE.clone(), chest: BASE.clone(), abd: BASE.clone(),
  pelvis: BASE.clone(), limb: BASE.clone(), rim: RIM.clone(),
};

let renderer, scene, camera, controls, container, ro, clock;
let modelGroup, organGroup, organs = {}, ground, particles, pData, reticle;
let beacon, beaconLabel, wrist, wristBone, glowTex;
let shellU = null;                                // shell shader uniforms
let started = false, pendingData = null;
let hr = 72, resp = 15, overall = 0, tNow = 0;

/* ---------------- helpers ---------------- */
function glowTexture() {
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const x = c.getContext("2d"), g = x.createRadialGradient(64, 64, 1, 64, 64, 64);
  g.addColorStop(0, "rgba(255,255,255,1)"); g.addColorStop(.25, "rgba(255,255,255,.55)");
  g.addColorStop(.55, "rgba(255,255,255,.16)"); g.addColorStop(1, "rgba(255,255,255,0)");
  x.fillStyle = g; x.fillRect(0, 0, 128, 128); return new THREE.CanvasTexture(c);
}
function tierColor(region) {
  if (!region || !region.monitored) return NODATA;
  const sev = region.severity || 0;
  if (sev < 0.18 || region.status === "NORMAL") return BASE;
  return new THREE.Color(TIER_HEX[region.status] || 0xffd23b);
}
const worse = (a, b) => (((a && a.severity) || 0) >= ((b && b.severity) || 0) ? a : b);

/* ---------------- body shell material ----------------
   Translucent so the organs read through it; height-zoned colour, a
   fresnel rim, animated scanlines, and a rising diagnostic sweep band. */
function bodyMaterial() {
  const m = new THREE.MeshStandardMaterial({
    transparent: true, opacity: 0.22, depthWrite: false, side: THREE.DoubleSide,
    roughness: 1, metalness: 0,
  });
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uHead = { value: ZONE.head }; sh.uniforms.uChest = { value: ZONE.chest };
    sh.uniforms.uAbd = { value: ZONE.abd }; sh.uniforms.uPelvis = { value: ZONE.pelvis };
    sh.uniforms.uLimb = { value: ZONE.limb }; sh.uniforms.uRim = { value: ZONE.rim };
    sh.uniforms.uYmax = { value: H }; sh.uniforms.uTime = { value: 0 };
    sh.uniforms.uScan = { value: -1 }; sh.uniforms.uPulse = { value: 0 };
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nvarying float vWY; varying float vWX;")
      .replace("#include <project_vertex>",
        "#include <project_vertex>\n vec4 _wp = modelMatrix * vec4(transformed,1.0); vWY=_wp.y; vWX=abs(_wp.x);");
    sh.fragmentShader = sh.fragmentShader
      .replace("#include <common>", `#include <common>
        varying float vWY; varying float vWX;
        uniform vec3 uHead,uChest,uAbd,uPelvis,uLimb,uRim;
        uniform float uYmax,uTime,uScan,uPulse;`)
      .replace("#include <dithering_fragment>", `#include <dithering_fragment>
        float h = clamp(vWY/uYmax, 0.0, 1.0);
        vec3 col = uLimb;
        col = mix(col, uPelvis, smoothstep(0.44,0.50,h));
        col = mix(col, uAbd,    smoothstep(0.50,0.60,h));
        col = mix(col, uChest,  smoothstep(0.60,0.74,h));
        col = mix(col, uHead,   smoothstep(0.80,0.88,h));
        float arm = smoothstep(0.17,0.27,vWX) * smoothstep(0.55,0.62,h) * (1.0-smoothstep(0.80,0.86,h));
        col = mix(col, uLimb, arm);
        float fres = pow(1.0 - clamp(dot(normalize(normal), normalize(vViewPosition)),0.0,1.0), 1.5);
        // horizontal scanlines drifting downward
        float sl = 0.86 + 0.14*sin(vWY*150.0 - uTime*2.2);
        // rising diagnostic sweep band
        float sweep = exp(-pow((h - uScan)*8.0, 2.0));
        vec3 outc = col*(0.22 + 0.55*fres) + uRim*fres*0.6;
        outc *= sl;
        outc += col*sweep*0.9 + vec3(0.55,0.85,1.0)*sweep*0.5;
        outc *= (1.0 + 0.08*uPulse);
        float a = clamp(0.12 + fres*0.52 + sweep*0.35, 0.0, 0.92);
        gl_FragColor = vec4(outc, a);`);
    shellU = sh.uniforms;
  };
  return m;
}

/* ---------------- internal organs ---------------- */
function organGeo(detail, jit) {
  const g = new THREE.IcosahedronGeometry(1, detail);
  const p = g.attributes.position;
  for (let i = 0; i < p.count; i++) {
    const f = 1 + (Math.random() * 2 - 1) * jit;
    p.setXYZ(i, p.getX(i) * f, p.getY(i) * f, p.getZ(i) * f);
  }
  g.computeVertexNormals(); return g;
}
function buildOrgans() {
  organGroup = new THREE.Group(); scene.add(organGroup);
  ORGANS.forEach((cfg) => {
    const geo = organGeo(cfg.detail, cfg.jit);
    const core = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      color: BASE, transparent: true, opacity: 0.42, depthWrite: false }));
    const wire = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      color: BASE, wireframe: true, transparent: true, opacity: 0.55,
      depthWrite: false, blending: THREE.AdditiveBlending }));
    const glow = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex, color: BASE, transparent: true, opacity: 0,
      depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending }));
    const halo = new THREE.Mesh(new THREE.TorusGeometry(1, 0.045, 8, 44),
      new THREE.MeshBasicMaterial({ color: 0xff4d5e, transparent: true, opacity: 0,
        depthWrite: false, blending: THREE.AdditiveBlending }));
    halo.rotation.x = Math.PI / 2.4;
    const g = new THREE.Group(); g.position.set(cfg.pos[0], cfg.pos[1], cfg.pos[2]);
    core.scale.set(...cfg.scale); wire.scale.set(...cfg.scale);
    const gs = Math.max(...cfg.scale) * 3.4; glow.scale.set(gs, gs, 1);
    halo.scale.setScalar(Math.max(...cfg.scale) * 2.0);
    g.add(core, wire, glow, halo); g.renderOrder = 4; organGroup.add(g);
    organs[cfg.key] = { g, core, wire, glow, halo, cfg, base: cfg.scale.slice(),
      ph: Math.random() * 6.28, sev: 0, mon: false, target: 0, worst: false };
  });
}

/* ---------------- holo platform ---------------- */
function circle(r, segs) {
  const pts = [];
  for (let i = 0; i <= segs; i++) { const a = i / segs * Math.PI * 2; pts.push(new THREE.Vector3(Math.cos(a) * r, 0, Math.sin(a) * r)); }
  return new THREE.BufferGeometry().setFromPoints(pts);
}
function buildGround() {
  ground = new THREE.Group(); ground.position.y = 0.01; scene.add(ground);
  const disc = new THREE.Mesh(new THREE.CircleGeometry(0.52, 64),
    new THREE.MeshBasicMaterial({ color: 0x0a1830, transparent: true, opacity: 0.35, depthWrite: false }));
  disc.rotation.x = -Math.PI / 2; disc.position.y = -0.004; ground.add(disc);
  [0.18, 0.30, 0.42, 0.52].forEach((r, i) => {
    const ring = new THREE.LineLoop(circle(r, 80),
      new THREE.LineBasicMaterial({ color: 0x3f9fe0, transparent: true, opacity: i === 3 ? 0.5 : 0.22 }));
    ground.add(ring);
  });
  const spokes = new THREE.Group();
  for (let i = 0; i < 16; i++) {
    const a = i / 16 * Math.PI * 2;
    const g = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(Math.cos(a) * 0.18, 0, Math.sin(a) * 0.18),
      new THREE.Vector3(Math.cos(a) * 0.52, 0, Math.sin(a) * 0.52)]);
    spokes.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x274a72, transparent: true, opacity: 0.28 })));
  }
  ground.add(spokes);
  // rotating sweep wedge (two bright radials)
  const sweep = new THREE.Group(); sweep.name = "sweep";
  [0, 0.12].forEach((off) => {
    const a = off;
    const g = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0), new THREE.Vector3(Math.cos(a) * 0.52, 0, Math.sin(a) * 0.52)]);
    sweep.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x7fe8ff, transparent: true, opacity: 0.45 })));
  });
  ground.add(sweep);
}

/* ---------------- cellular particle field ---------------- */
function buildParticles() {
  const N = 300; const pos = new Float32Array(N * 3); const spd = new Float32Array(N);
  for (let i = 0; i < N; i++) {
    const a = Math.random() * 6.28, r = Math.sqrt(Math.random()) * 0.15;
    pos[i * 3] = Math.cos(a) * r; pos[i * 3 + 1] = Math.random() * H; pos[i * 3 + 2] = Math.sin(a) * r * 0.7;
    spd[i] = 0.06 + Math.random() * 0.12;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  particles = new THREE.Points(geo, new THREE.PointsMaterial({
    size: 0.016, map: glowTex, color: 0x6fd3ff, transparent: true, opacity: 0.5,
    depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true }));
  pData = { spd, N }; scene.add(particles);
}

/* Diagnostic scan ring — a thin disc that RISES up the body in lockstep with the
   shader's scan-sweep, like a CT scanner passing through the patient. */
function buildReticle() {
  reticle = new THREE.Group(); reticle.position.y = 0.1; scene.add(reticle);
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.22, 0.0025, 6, 90),
    new THREE.MeshBasicMaterial({ color: 0x8fe9ff, transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending }));
  ring.rotation.x = Math.PI / 2; reticle.add(ring);
  for (let i = 0; i < 4; i++) {                    // corner brackets
    const a = i / 4 * Math.PI * 2 + Math.PI / 4;
    const br = new THREE.Mesh(new THREE.TorusGeometry(0.235, 0.004, 6, 24, 0.5),
      new THREE.MeshBasicMaterial({ color: 0x7fe8ff, transparent: true, opacity: 0.6, depthWrite: false, blending: THREE.AdditiveBlending }));
    br.rotation.x = Math.PI / 2; br.rotation.z = a; reticle.add(br);
  }
}

/* ---------------- model ---------------- */
function loadBody() {
  new GLTFLoader().load("./vendor/models/Xbot.glb", (gltf) => {
    modelGroup = gltf.scene;
    const mat = bodyMaterial();
    modelGroup.traverse((o) => {
      if (!o.isMesh) return;
      if (/joint/i.test(o.name)) { o.visible = false; return; }
      o.material = mat; o.frustumCulled = false; o.renderOrder = 6;
    });
    modelGroup.traverse((o) => { if (o.isBone && !wristBone && /lefthand|leftwrist/i.test(o.name)) wristBone = o; });
    if (!wristBone) modelGroup.traverse((o) => { if (o.isBone && !wristBone && /hand/i.test(o.name)) wristBone = o; });
    scene.add(modelGroup);
    poseArms(modelGroup);
    normalizeBySkeleton(modelGroup);
    fitCamera(); buildWrist();
    if (pendingData) { update(pendingData); pendingData = null; }
  }, undefined, (e) => console.error("[twin] model load failed", e));
}

function normalizeBySkeleton(root) {
  const v = new THREE.Vector3();
  const bounds = () => {
    root.updateWorldMatrix(true, true);
    let lo = Infinity, hi = -Infinity, xl = Infinity, xh = -Infinity, zl = Infinity, zh = -Infinity;
    root.traverse((o) => { if (o.isBone) { o.getWorldPosition(v); lo = Math.min(lo, v.y); hi = Math.max(hi, v.y); xl = Math.min(xl, v.x); xh = Math.max(xh, v.x); zl = Math.min(zl, v.z); zh = Math.max(zh, v.z); } });
    return { lo, hi, cx: (xl + xh) / 2, cz: (zl + zh) / 2 };
  };
  root.scale.setScalar(1); root.position.set(0, 0, 0);
  let b = bounds();
  const realH = b.hi - b.lo || 1;
  const s = (H * 0.9) / realH;
  root.scale.setScalar(s); root.userData.baseScale = s;
  b = bounds();
  root.position.x -= b.cx; root.position.z -= b.cz;
  root.position.y -= b.lo - H * 0.05;
  root.updateWorldMatrix(true, true);
}

function poseArms(root) {
  const B = {};
  root.traverse((o) => { if (o.isBone) B[o.name.toLowerCase()] = o; });
  const find = (re) => Object.keys(B).find((k) => re.test(k));
  const set = (re, ax, v) => { const k = find(re); if (k) B[k].rotation[ax] += v; };
  set(/leftarm$/, "z", -1.32); set(/rightarm$/, "z", 1.32);
  set(/leftforearm$/, "z", -0.22); set(/rightforearm$/, "z", 0.22);
}

/* ---------------- wearable wrist marker ---------------- */
function buildWrist() {
  const c = document.createElement("canvas"); c.width = c.height = 128;
  drawWrist(c, false);
  wrist = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthTest: false, depthWrite: false }));
  wrist.scale.set(0.15, 0.15, 1); wrist.userData = { canvas: c, on: null }; wrist.visible = false;
  scene.add(wrist);
}
function drawWrist(c, on) {
  const x = c.getContext("2d"); x.clearRect(0, 0, 128, 128);
  const col = on ? "#1fdd8a" : "#5b6a8f";
  x.strokeStyle = col; x.fillStyle = on ? "rgba(31,221,138,.16)" : "rgba(91,106,143,.12)";
  x.lineWidth = 6; x.beginPath(); x.arc(64, 64, 40, 0, 7); x.fill(); x.stroke();
  x.strokeStyle = col; x.lineWidth = 5; x.lineJoin = "round"; x.beginPath();
  x.moveTo(40, 64); x.lineTo(54, 64); x.lineTo(60, 48); x.lineTo(70, 82); x.lineTo(76, 64); x.lineTo(90, 64); x.stroke();
  if (on) { x.shadowBlur = 16; x.shadowColor = col; x.stroke(); x.shadowBlur = 0; }
}
function setWrist(on) {
  if (!wrist || wrist.userData.on === on) return;
  wrist.userData.on = on; drawWrist(wrist.userData.canvas, on); wrist.material.map.needsUpdate = true;
}

function fitCamera() {
  const fov = camera.fov * Math.PI / 180;
  const dist = (H * 0.50) / Math.tan(fov / 2) * 1.08;
  controls.target.set(0, H * 0.50, 0);
  camera.position.set(0.10, H * 0.54, dist);
  controls.minDistance = dist * 0.45; controls.maxDistance = dist * 2.1;
  controls.update();
}

/* ---------------- threat beacon (label for the worst system) ---------------- */
function buildBeacon() {
  beacon = new THREE.Group(); beacon.visible = false;
  beaconLabel = makeLabel(); beaconLabel.position.y = 0.16; beacon.add(beaconLabel);
  scene.add(beacon);
}
function makeLabel() {
  const c = document.createElement("canvas"); c.width = 256; c.height = 64;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthTest: false }));
  sp.scale.set(0.42, 0.105, 1); sp.userData = { ctx: c.getContext("2d") }; return sp;
}
function setLabel(sprite, text) {
  const x = sprite.userData.ctx; x.clearRect(0, 0, 256, 64);
  x.fillStyle = "rgba(10,14,24,.85)"; rr(x, 6, 16, 244, 34, 8); x.fill();
  x.strokeStyle = "#ff4d5e"; x.lineWidth = 2; rr(x, 6, 16, 244, 34, 8); x.stroke();
  x.fillStyle = "#ffd2d6"; x.font = "600 18px ui-monospace,monospace"; x.textAlign = "center"; x.textBaseline = "middle";
  x.fillText(text, 128, 34); sprite.material.map.needsUpdate = true;
}
function rr(c, x, y, w, h, r) { c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath(); }

/* ---------------- lifecycle ---------------- */
function start(el) {
  if (started || !el) return;
  container = el; const w = el.clientWidth || 480, h = el.clientHeight || 560;
  scene = new THREE.Scene(); scene.fog = new THREE.FogExp2(0x05070d, 0.10);
  camera = new THREE.PerspectiveCamera(33, w / h, 0.1, 100);
  camera.position.set(0.2, H * 0.56, H * 1.9);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio)); renderer.setSize(w, h);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.18;
  el.querySelectorAll("canvas").forEach((c) => c.remove()); el.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.06; controls.enablePan = false;
  controls.target.set(0, H * 0.50, 0);
  controls.autoRotate = true; controls.autoRotateSpeed = 0.42;
  controls.minPolarAngle = Math.PI * 0.36; controls.maxPolarAngle = Math.PI * 0.60;
  scene.add(new THREE.AmbientLight(0xffffff, 1));

  glowTex = glowTexture();
  buildGround(); buildParticles(); buildReticle();
  loadBody(); buildOrgans(); buildBeacon();
  clock = new THREE.Clock(); started = true;
  ro = new ResizeObserver(resize); ro.observe(el); animate();
}

/* Camera→Twin coupling: optical biosensor publishes a region severity map;
   merged ESCALATION-ONLY (worse-of) so the webcam can light anatomy but never
   downgrade a server-detected critical state. */
let lastServerData = null, optical = {};
function mergedRegions() {
  const sr = (lastServerData && lastServerData.body_state && lastServerData.body_state.regions) || {};
  const out = {};
  for (const k of new Set([...Object.keys(sr), ...Object.keys(optical)])) out[k] = worse(sr[k], optical[k]);
  return out;
}
function readVital(data, key, dflt) {
  const ch = data && data.body_state && data.body_state.channels;
  if (Array.isArray(ch)) { const c = ch.find((c) => c.key === key); if (c && c.value != null) return c.value; }
  if (data && data.vitals && data.vitals[key] != null) return data.vitals[key];
  return dflt;
}
function update(data) {
  lastServerData = data || lastServerData;
  if (data) {
    hr = readVital(data, "heart_rate", hr) || 72;
    resp = readVital(data, "respiratory_rate", resp) || 15;
  }
  if (!started) { pendingData = data; return; }
  applyRegions();
}
function setOptical(reg) { optical = reg || {}; if (started) applyRegions(); }

function applyRegions() {
  const regions = mergedRegions();
  ZONE.head.copy(tierColor(regions.brain));
  ZONE.chest.copy(tierColor(worse(regions.heart, regions.lungs)));
  ZONE.abd.copy(tierColor(worse(worse(regions.liver, regions.kidneys), regions.blood)));
  ZONE.pelvis.copy(tierColor(regions.skin));   // pelvis follows peripheral, not immune
  ZONE.limb.copy(tierColor(regions.skin));

  // organs
  overall = 0; let worstKey = null, worstSev = 0.30, worstRegion = null;
  ORGANS.forEach((cfg) => {
    const o = organs[cfg.key]; if (!o) return;
    const reg = regions[cfg.region];
    const mon = !!(reg && reg.monitored);
    const sev = mon ? (reg.severity || 0) : 0;
    const col = tierColor(reg);
    o.core.material.color.copy(col); o.wire.material.color.copy(col); o.glow.material.color.copy(col);
    o.mon = mon; o.sev = sev;
    if (!mon) { o.core.material.opacity = 0.08; o.wire.material.opacity = 0.16; o.target = 0; }
    else {
      o.core.material.opacity = 0.36 + 0.30 * sev;
      o.wire.material.opacity = 0.45 + 0.42 * sev;
      o.target = cfg.mute ? Math.min(0.5, sev) : Math.min(0.95, 0.12 + sev * 1.05);
      if (mon) overall = Math.max(overall, sev);
    }
    o.worst = false;
    if (mon && reg.status !== "NORMAL" && sev > worstSev) { worstSev = sev; worstKey = cfg.key; worstRegion = reg; }
  });
  ORGANS.forEach((cfg) => { if (organs[cfg.key]) organs[cfg.key].worst = (cfg.key === worstKey); });

  // worst-system label beacon (at that organ)
  if (worstKey && beacon) {
    const p = organs[worstKey].cfg.pos;
    beacon.position.set(p[0], p[1] + 0.12, Math.max(p[2], 0) + 0.16); beacon.visible = true;
    const lbl = (REGION_LABEL[organs[worstKey].cfg.region] || worstKey).toUpperCase();
    setLabel(beaconLabel, lbl + (worstRegion && worstRegion.optical ? " ◉" : ""));
  } else if (beacon) beacon.visible = false;

  // particle tint by overall load
  if (particles) particles.material.color.copy(overall > 0.18 ? tierColor({ monitored: true, severity: overall, status: overall >= 0.75 ? "CRITICAL" : overall >= 0.45 ? "DYSFUNCTION" : "STRESS" }) : BASE);

  // wearable wrist marker
  const d = lastServerData || {};
  const live = ["wearable", "simulator", "camera"].includes(d.source) && (d.risk && d.risk.signal_quality != null ? d.risk.signal_quality : 1) > 0.15;
  if (wrist) { wrist.visible = !!wristBone; setWrist(live); }
}

function resize() {
  if (!renderer || !container) return;
  const w = container.clientWidth, h = container.clientHeight; if (!w || !h) return;
  camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h); fitCamera();
}

const _v = new THREE.Vector3();
function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta(); tNow += dt; const t = tNow;
  controls.update();

  // heartbeat (sharp) + breathing cycles, from live vitals
  const beat = Math.pow(0.5 + 0.5 * Math.sin(t * (hr / 60) * 6.2832), 6);
  const breathe = 0.5 - 0.5 * Math.cos(t * (resp / 60) * 6.2832);

  // body shell: gentle breathe + shader uniforms
  if (modelGroup && modelGroup.userData.baseScale)
    modelGroup.scale.setScalar(modelGroup.userData.baseScale * (1 + 0.010 * breathe));
  if (shellU) {
    shellU.uTime.value = t; shellU.uPulse.value = beat;
    shellU.uScan.value = ((t * 0.20) % 1.5) - 0.15;   // rising scan band, then repeat
  }

  // organs: per-system pulse, heart beat, lungs breathe, worst-system halo
  for (const k in organs) {
    const o = organs[k]; const b = o.base;
    let sx = b[0], sy = b[1], sz = b[2];
    if (o.cfg.beat) { const f = 1 + 0.17 * beat; sx *= f; sy *= f; sz *= f; }
    if (o.cfg.breathe) { sy *= 1 + 0.16 * breathe; sx *= 1 + 0.05 * breathe; sz *= 1 + 0.05 * breathe; }
    o.core.scale.set(sx, sy, sz); o.wire.scale.set(sx, sy, sz);
    const env = o.cfg.beat ? (0.35 + 0.65 * beat) : (0.55 + 0.45 * Math.sin(t * 1.8 + o.ph));
    o.glow.material.opacity = o.target * env;
    if (o.worst) {
      o.halo.material.opacity = 0.30 + 0.30 * Math.abs(Math.sin(t * 3.4));
      o.halo.scale.setScalar(Math.max(...b) * (1.9 + 0.18 * Math.sin(t * 3.4)));
      o.halo.rotation.z += dt * 0.6;
    } else o.halo.material.opacity = 0;
  }

  // particle field drifts upward, recycles at the feet
  if (particles) {
    const pos = particles.geometry.attributes.position;
    for (let i = 0; i < pData.N; i++) {
      let y = pos.getY(i) + pData.spd[i] * dt;
      if (y > H) { y = 0; const a = Math.random() * 6.28, r = Math.sqrt(Math.random()) * 0.15; pos.setX(i, Math.cos(a) * r); pos.setZ(i, Math.sin(a) * r * 0.7); }
      pos.setY(i, y);
    }
    pos.needsUpdate = true;
    particles.material.opacity = 0.32 + 0.22 * beat;
  }

  if (ground) { const sw = ground.getObjectByName("sweep"); if (sw) sw.rotation.y -= dt * 0.7; ground.rotation.y += dt * 0.02; }
  if (reticle && shellU) {                       // ride the scan band up the body
    const s = Math.max(0, Math.min(1, shellU.uScan.value));
    reticle.position.y = s * H * 0.96 + 0.06;
    reticle.rotation.y += dt * 0.3;
    const edge = Math.sin(s * Math.PI);          // fade at feet/head
    reticle.children.forEach((c) => { c.material.opacity = (c.geometry.parameters.arc ? 0.6 : 0.55) * edge; });
  }
  if (wrist && wristBone && wrist.visible) { wristBone.getWorldPosition(_v); wrist.position.copy(_v).add(new THREE.Vector3(0, 0.02, 0.04)); }

  renderer.render(scene, camera);
}

window.AegisTwin = { start, update, resize, setOptical };
const probe = setInterval(() => {
  const el = document.getElementById("twin-stage");
  if (el && el.clientWidth > 40 && el.clientHeight > 40) { clearInterval(probe); start(el); }
}, 250);
