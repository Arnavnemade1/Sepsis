/* ============================================================
   AEGIS-OS — 3D anatomical digital twin (Three.js r160, vendored)

   A medical hologram, not a coloured silhouette. A real rigged humanoid
   shell (translucent, scanlines + a rising diagnostic scan-sweep + fresnel
   rim) houses REAL ANATOMICAL ORGANS — heart, lungs, liver, kidneys, spleen
   — loaded from the HuBMAP CCF reference meshes (Visible Human, CC BY 4.0;
   see vendor/models/organs/ATTRIBUTION.md). The meshes are co-registered, so
   they assemble into a true torso; each glows the colour of ITS system's
   severity. The heart beats at the patient's heart-rate; the lungs breathe at
   the respiratory rate. Systems we don't actually monitor (renal/hepatic/
   neuro) are dimmed to honest "no data" — never a confident healthy glow.

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

/* Real anatomical organ atlas — HuBMAP CCF v1.2 reference meshes (Visible Human,
   CC BY 4.0, see vendor/models/organs/ATTRIBUTION.md). All six are co-registered
   in one coordinate space, so loading them together assembles a correct torso;
   we then scale/place the whole assembly inside the body shell.
   region = which body_state region drives this organ's colour. */
const ORGAN_FILES = [
  { key: "lung",    file: "VH_M_Lung.glb",     region: "lungs",   breathe: true },
  { key: "heart",   file: "VH_M_Heart.glb",    region: "heart",   beat: true },
  { key: "liver",   file: "VH_M_Liver.glb",    region: "liver" },
  { key: "spleen",  file: "VH_M_Spleen.glb",   region: "blood" },
  { key: "kidneyL", file: "VH_M_Kidney_L.glb", region: "kidneys" },
  { key: "kidneyR", file: "VH_M_Kidney_R.glb", region: "kidneys" },
];
// placement of the assembled organ atlas inside the body (tuned to the Xbot mesh)
let ORGAN_SPAN = 0.44;       // vertical extent of the organ stack in body units
let TORSO_Y = 1.19;          // world Y of the atlas centre (mid-torso)
let TORSO_Z = 0.0;           // forward offset
let ORGAN_FLIP = 1;          // liver→screen-left (patient's right), spleen→screen-right

// shell zone colours (mutated in place → live in the shader)
const ZONE = {
  head: BASE.clone(), chest: BASE.clone(), abd: BASE.clone(),
  pelvis: BASE.clone(), limb: BASE.clone(), rim: RIM.clone(),
};

let renderer, scene, camera, controls, container, ro, clock;
let modelGroup, atlasGroup, organs = {}, organLoadCount = 0, brainStub;
let ground, particles, pData, reticle;
let beacon, beaconLabel, wrist, wristBone, glowTex;
let infectFX, infLabel, lastInfection = null;     // suspected infection focus
const INFECT_COL = 0xff7a3b;                       // distinct "infection" orange
// suspected-focus region -> organ to light up (null = no single organ / systemic)
const FOCUS_ORGAN = { lungs: "lung", respiratory: "lung", kidneys: "kidneyL", renal: "kidneyL", liver: "liver", hepatic: "liver" };
const FOCUS_AT = { lungs: "lung", respiratory: "lung", kidneys: "kidneyL", renal: "kidneyL", liver: "liver", hepatic: "liver", systemic: "heart", bloodstream: "heart", core: "heart", heart: "heart" };
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
        vec3 outc = col*(0.30 + 0.6*fres) + uRim*fres*0.75;
        outc *= sl;
        outc += col*sweep*0.9 + vec3(0.55,0.85,1.0)*sweep*0.5;
        outc *= (1.0 + 0.08*uPulse);
        float a = clamp(0.17 + fres*0.6 + sweep*0.35, 0.0, 0.95);
        gl_FragColor = vec4(outc, a);`);
    shellU = sh.uniforms;
  };
  return m;
}

/* ---------------- internal organs (real anatomy) ----------------
   Holographic organ material: dark surface lit only by a settable emissive
   colour + a fresnel rim, so each organ glows its system's tier colour and the
   edges read crisply against the translucent shell. */
function makeOrganMat() {
  // translucent "wet tissue" look: a real shaded surface (key light gives form)
  // tinted by the system colour, plus a bright fresnel rim so the organ reads as
  // a glowing volume rather than a flat blob.
  const m = new THREE.MeshStandardMaterial({
    color: 0x222a38, emissive: BASE.clone(), emissiveIntensity: 0.5,
    transparent: true, opacity: 0.6, roughness: 0.42, metalness: 0.0,
    depthWrite: true, side: THREE.DoubleSide,
  });
  m.onBeforeCompile = (sh) => {
    sh.fragmentShader = sh.fragmentShader.replace("#include <dithering_fragment>", `#include <dithering_fragment>
      float fr = pow(1.0 - clamp(dot(normalize(normal), normalize(vViewPosition)),0.0,1.0), 1.7);
      gl_FragColor.rgb += emissive.rgb * (0.35 + 1.7 * fr);
      gl_FragColor.a = clamp(gl_FragColor.a + fr * 0.42, 0.0, 1.0);`);
  };
  return m;
}
function loadOrgans() {
  atlasGroup = new THREE.Group(); scene.add(atlasGroup);
  const loader = new GLTFLoader();
  ORGAN_FILES.forEach((cfg) => {
    loader.load("./vendor/models/organs/" + cfg.file, (g) => {
      const root = g.scene, mat = makeOrganMat();
      root.traverse((o) => { if (o.isMesh) { o.material = mat; o.frustumCulled = false; o.castShadow = o.receiveShadow = false; } });
      // wrap in a pivot at the organ's own centre so beat/breathe scale about it
      const box = new THREE.Box3().setFromObject(root), c = box.getCenter(new THREE.Vector3());
      const pivot = new THREE.Group(); pivot.position.copy(c); root.position.sub(c); pivot.add(root);
      atlasGroup.add(pivot);
      organs[cfg.key] = { cfg, pivot, mat, region: cfg.region, sev: 0, mon: false };
      if (++organLoadCount === ORGAN_FILES.length) calibrateAtlas();
    }, undefined, (e) => console.error("[twin] organ load failed", cfg.file, e));
  });
  // dim placeholder for the (unmonitored) brain so the head isn't empty
  const bg = new THREE.IcosahedronGeometry(0.072, 2);
  brainStub = new THREE.Mesh(bg, new THREE.MeshBasicMaterial({
    color: NODATA, wireframe: true, transparent: true, opacity: 0.22, depthWrite: false }));
  brainStub.position.set(0, 1.555, 0.01); brainStub.scale.set(1, 1.05, 1.12); scene.add(brainStub);
}
/* Assemble + fit the co-registered organ atlas inside the body shell. */
function calibrateAtlas() {
  atlasGroup.scale.set(1, 1, 1); atlasGroup.position.set(0, 0, 0); atlasGroup.updateWorldMatrix(true, true);
  const box = new THREE.Box3().setFromObject(atlasGroup);
  const c = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
  const s = ORGAN_SPAN / (size.y || 1);
  atlasGroup.scale.set(ORGAN_FLIP * s, s, s);
  atlasGroup.position.set(-ORGAN_FLIP * s * c.x, TORSO_Y - s * c.y, TORSO_Z - s * c.z);
  if (lastServerData || pendingData) applyRegions();
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
function setLabel(sprite, text, stroke, fill) {
  const x = sprite.userData.ctx; x.clearRect(0, 0, 256, 64);
  x.fillStyle = "rgba(10,14,24,.85)"; rr(x, 6, 16, 244, 34, 8); x.fill();
  x.strokeStyle = stroke || "#ff4d5e"; x.lineWidth = 2; rr(x, 6, 16, 244, 34, 8); x.stroke();
  x.fillStyle = fill || "#ffd2d6"; x.font = "600 17px ui-monospace,monospace"; x.textAlign = "center"; x.textBaseline = "middle";
  x.fillText(text, 128, 34); sprite.material.map.needsUpdate = true;
}
function rr(c, x, y, w, h, r) { c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath(); }

/* ---------------- suspected infection focus ----------------
   A distinct orange "hot zone" (pulsing bloom + expanding rings + label) placed
   at the inferred source of infection, with the organ at that site lit up. The
   site is INFERRED from the physiology upstream (clearly labelled "suspected"),
   not a microbiology result. */
function buildInfection() {
  infectFX = new THREE.Group(); infectFX.visible = false; scene.add(infectFX);
  const bloom = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: INFECT_COL,
    transparent: true, opacity: 0.8, depthWrite: false, depthTest: false, blending: THREE.AdditiveBlending }));
  bloom.scale.set(0.2, 0.2, 1); bloom.name = "bloom"; infectFX.add(bloom);
  for (let i = 0; i < 2; i++) {
    const ring = new THREE.Mesh(new THREE.TorusGeometry(1, 0.022, 8, 48),
      new THREE.MeshBasicMaterial({ color: INFECT_COL, transparent: true, opacity: 0, depthWrite: false, blending: THREE.AdditiveBlending }));
    ring.name = "ring" + i; ring.userData.ph = i * 0.5; infectFX.add(ring);
  }
  infLabel = makeLabel(); infLabel.position.y = 0.17; infLabel.scale.set(0.52, 0.105, 1); infectFX.add(infLabel);
}
function focusPos(region) {
  const k = FOCUS_AT[region];
  if (k && organs[k] && organs[k].pivot) { organs[k].pivot.getWorldPosition(_v); return _v.clone(); }
  if (region === "skin") return new THREE.Vector3(0.05, TORSO_Y - 0.02, 0.13);
  return new THREE.Vector3(0, TORSO_Y, 0.06);
}

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
  controls.autoRotate = true; controls.autoRotateSpeed = 0.28;
  controls.minPolarAngle = Math.PI * 0.38; controls.maxPolarAngle = Math.PI * 0.56;
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const key = new THREE.DirectionalLight(0xcfe4ff, 1.5); key.position.set(0.6, 1.4, 1.2); scene.add(key);
  const fill = new THREE.DirectionalLight(0x6080c0, 0.6); fill.position.set(-0.8, 0.6, -0.6); scene.add(fill);

  glowTex = glowTexture();
  buildGround(); buildParticles(); buildReticle();
  loadBody(); loadOrgans(); buildBeacon(); buildInfection();
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
    if ("infection" in data) lastInfection = data.infection;
  }
  if (!started) { pendingData = data; return; }
  applyRegions();
}
function setOptical(reg) { optical = reg || {}; if (started) applyRegions(); }

function applyRegions() {
  const regions = mergedRegions();

  // organs — recolour each real mesh by its system's severity
  overall = 0; let worstKey = null, worstSev = 0.30, worstRegion = null;
  for (const k in organs) {
    const o = organs[k]; const reg = regions[o.region];
    const mon = !!(reg && reg.monitored);
    const sev = mon ? (reg.severity || 0) : 0;
    o.mon = mon; o.sev = sev; o.worst = false;
    o.mat.emissive.copy(tierColor(reg));
    o.mat.emissiveIntensity = mon ? (0.65 + 0.95 * sev) : 0.32;
    o.mat.opacity = mon ? (0.62 + 0.26 * sev) : 0.34;
    if (mon) overall = Math.max(overall, sev);
    if (mon && reg.status !== "NORMAL" && sev > worstSev) { worstSev = sev; worstKey = k; worstRegion = reg; }
  }
  for (const k in organs) organs[k].worst = (k === worstKey);

  // body shell: a single clean translucent tint (neutral blue, washing gently
  // toward the overall tier colour in a crisis) — the diagnostic colour lives in
  // the organs, so the body always reads as a body, not a patchy heat-map.
  const tierHex = overall >= 0.75 ? 0xff4d5e : overall >= 0.45 ? 0xff9a3b : overall >= 0.2 ? 0xffd23b : null;
  const bc = BASE.clone(); if (tierHex) bc.lerp(new THREE.Color(tierHex), 0.4);
  ZONE.head.copy(bc); ZONE.chest.copy(bc); ZONE.abd.copy(bc); ZONE.pelvis.copy(bc); ZONE.limb.copy(bc);

  // suspected infection focus — light up the source organ + an orange hot zone
  for (const k in organs) organs[k].infected = false;
  if (lastInfection && lastInfection.site) {
    infectFX.position.copy(focusPos(lastInfection.region)); infectFX.visible = true;
    setLabel(infLabel, "⚠ FOCUS · " + lastInfection.site.toUpperCase(), "#ff7a3b", "#ffd8b0");
    // (systemic sites stay central; localised sites sit on the organ)
    const fk = FOCUS_ORGAN[lastInfection.region];
    [fk, lastInfection.region === "kidneys" ? "kidneyR" : null].forEach((key) => {
      const o = key && organs[key]; if (!o) return;
      o.infected = true; o.mat.emissive.lerp(new THREE.Color(INFECT_COL), 0.55);
      o.mat.opacity = Math.max(o.mat.opacity, 0.7);
    });
    if (beacon) beacon.visible = false;
  } else {
    if (infectFX) infectFX.visible = false;
    // fallback: worst-system label beacon
    if (worstKey && beacon && organs[worstKey].pivot) {
      organs[worstKey].pivot.getWorldPosition(_v);
      beacon.position.set(_v.x, _v.y + 0.14, _v.z + 0.16); beacon.visible = true;
      setLabel(beaconLabel, (REGION_LABEL[organs[worstKey].region] || worstKey).toUpperCase() + (worstRegion && worstRegion.optical ? " ◉" : ""));
    } else if (beacon) beacon.visible = false;
  }

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

  // organs: heart beats at HR, lungs breathe at the respiratory rate, the
  // worst-affected organ throbs; emissive brightness tracks the pulse.
  for (const k in organs) {
    const o = organs[k]; if (!o.pivot) continue;
    if (o.cfg.beat) o.pivot.scale.setScalar(1 + 0.11 * beat);
    else if (o.cfg.breathe) o.pivot.scale.set(1 + 0.035 * breathe, 1 + 0.075 * breathe, 1 + 0.035 * breathe);
    else if (o.worst) o.pivot.scale.setScalar(1 + 0.05 * Math.abs(Math.sin(t * 3.0)));
    else o.pivot.scale.setScalar(1);
    if (o.mon) {
      const env = o.cfg.beat ? (0.6 + 0.7 * beat) : (0.85 + 0.15 * Math.sin(t * 1.6 + (o.ph || 0)));
      o.mat.emissiveIntensity = (0.65 + 0.95 * o.sev) * env;
    }
    if (o.infected) {                              // suspected source — throb brighter
      const ip = 0.5 + 0.5 * Math.sin(t * 4.0);
      o.mat.emissiveIntensity = Math.max(o.mat.emissiveIntensity, 0.7) * (1.15 + 0.7 * ip);
      if (!o.cfg.beat && !o.cfg.breathe) o.pivot.scale.setScalar(1 + 0.05 * ip);
    }
  }

  // suspected infection "hot zone": pulsing bloom + expanding rings (billboarded)
  if (infectFX && infectFX.visible) {
    const ip = 0.5 + 0.5 * Math.sin(t * 4.0);
    const bloom = infectFX.getObjectByName("bloom");
    if (bloom) { bloom.material.opacity = 0.45 + 0.4 * ip; bloom.scale.setScalar(0.17 + 0.05 * ip); }
    for (let i = 0; i < 2; i++) {
      const r = infectFX.getObjectByName("ring" + i); if (!r) continue;
      const ph = (t * 0.55 + r.userData.ph) % 1;
      r.scale.setScalar(0.05 + ph * 0.17); r.material.opacity = 0.6 * (1 - ph); r.lookAt(camera.position);
    }
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

/* live calibration helpers (used to tune organ placement against the Xbot mesh) */
function _setAtlas(span, y, z, flip) {
  if (span != null) ORGAN_SPAN = span; if (y != null) TORSO_Y = y;
  if (z != null) TORSO_Z = z; if (flip != null) ORGAN_FLIP = flip;
  if (atlasGroup && organLoadCount === ORGAN_FILES.length) calibrateAtlas();
  return { ORGAN_SPAN, TORSO_Y, TORSO_Z, ORGAN_FLIP };
}
function _measure() {
  const out = {};
  if (modelGroup) { const b = new THREE.Box3().setFromObject(modelGroup); out.body = { min: b.min.toArray().map(n => +n.toFixed(3)), max: b.max.toArray().map(n => +n.toFixed(3)) }; }
  if (atlasGroup) { const b = new THREE.Box3().setFromObject(atlasGroup); out.atlas = { min: b.min.toArray().map(n => +n.toFixed(3)), max: b.max.toArray().map(n => +n.toFixed(3)) }; }
  for (const k in organs) { const o = organs[k]; if (o.pivot) { o.pivot.getWorldPosition(_v); out[k] = { p: _v.toArray().map(n => +n.toFixed(3)), col: "#" + o.mat.emissive.getHexString(), mon: o.mon, sev: +(o.sev || 0).toFixed(2), op: +o.mat.opacity.toFixed(2) }; } }
  return out;
}
window.AegisTwin = { start, update, resize, setOptical, _setAtlas, _measure };
const probe = setInterval(() => {
  const el = document.getElementById("twin-stage");
  if (el && el.clientWidth > 40 && el.clientHeight > 40) { clearInterval(probe); start(el); }
}, 250);
