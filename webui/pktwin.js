/* ============================================================
   AEGIS-OS — the PHARMACOKINETIC TWIN (Three.js r160, vendored)

   A second body, separate from the clinical bio-twin, that renders one thing the
   other one cannot: WHERE THE DRUG IS. The anatomy is the same real HuBMAP CCF
   atlas; what it displays is not the patient's severity but the projected
   concentration of an ordered drug, organ by organ, at a chosen moment in the
   treatment's own timeframe.

   Why a separate scene at all. The clinical twin's colour vocabulary is severity
   — amber stress, orange dysfunction, red critical — and it is load-bearing
   everywhere else in this console. Painting drug concentration onto the same body
   would overload that vocabulary: a brightly lit kidney would mean "failing" in
   one glance and "well perfused with antibiotic" in the next. So this twin has its
   own scene, its own palette (cool, deliberately outside the severity ramp), and
   its own claim: brightness here is DRUG, never danger.

   What is actually being drawn. Every value comes from the backend PBPK model
   (backend/aegis/pharmacokinetics.py) — a perfusion-limited compartment model on
   real organ blood flows. So the order in which the body lights up is not an
   animation choice: the lungs fill first because they take the entire cardiac
   output, the kidney runs hot because it is both well perfused and the elimination
   route, and muscle lags because its flow per litre of tissue is low. The render
   is a view onto that solution, not a decoration of it.

   Honesty, same rule as the clinical twin: an organ the model does not simulate
   (bladder, thymus) renders inert grey and says so. It never shows a confident
   drug level for a compartment that was never computed.

   Public API:  window.AegisPKTwin = { mount, setProjection, setArm, setTime,
                                       play, pause, resize, dispose, state }
   ============================================================ */
import * as THREE from "three";
import { OrbitControls } from "./vendor/three/OrbitControls.js";
import { GLTFLoader } from "./vendor/three/GLTFLoader.js";
import { EffectComposer } from "./vendor/three/postprocessing/EffectComposer.js";
import { RenderPass } from "./vendor/three/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "./vendor/three/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "./vendor/three/postprocessing/OutputPass.js";

const H = 1.7;

/* Drug palette. Chosen to sit OUTSIDE the severity ramp (amber/orange/red) and
   away from the healthy-green this console uses elsewhere, so a bright organ in
   this scene can never be misread as a bright organ in the clinical twin. Class,
   not identity: which specific antibiotic is running is written in the header,
   where it can be read, rather than encoded in a hue nobody can decode. */
const DRUG_COLOR = {
  antibiotic:  0x5fb3d4,   // cool cyan-steel
  vasopressor: 0xa78bfa,   // violet
  fluid:       0x8fa3b8,   // pale steel
};
const TISSUE = 0x191c20;          // inert tissue — the body when no drug is present
const INERT = 0x3d4147;           // an organ the model does not simulate

/* Which organ mesh shows which PBPK compartment. The backend ships the same map
   (ORGAN_MESH), but the atlas has meshes the model has no compartment for, and
   those must stay honestly dark rather than borrow a neighbour's number. */
const MESH_COMPARTMENT = {
  lung: "lung", heart: "heart", liver: "liver", spleen: "spleen",
  kidneyL: "kidney", kidneyR: "kidney", pancreas: "pancreas",
  intestine: "intestine", cord: "brain",
  // No compartment in the model — rendered inert, listed in the legend.
  bladder: null, thymus: null,
};
const UNMODELLED_LABEL = { bladder: "bladder", thymus: "thymus" };

const ORGAN_FILES = [
  { key: "lung", file: "VH_M_Lung.glb" },
  { key: "heart", file: "VH_M_Heart.glb" },
  { key: "liver", file: "VH_M_Liver.glb" },
  { key: "spleen", file: "VH_M_Spleen.glb" },
  { key: "kidneyL", file: "VH_M_Kidney_L.glb" },
  { key: "kidneyR", file: "VH_M_Kidney_R.glb" },
  { key: "pancreas", file: "VH_M_Pancreas.glb" },
  { key: "intestine", file: "VH_M_Small_Intestine.glb", calExclude: true },
  { key: "bladder", file: "VH_M_Urinary_Bladder.glb", calExclude: true },
  { key: "cord", file: "VH_M_Spinal_Cord.glb", calExclude: true },
  { key: "thymus", file: "VH_M_Thymus.glb" },
  { key: "vessels", file: "VH_M_Blood_Vasculature.glb", vascular: true, defer: true },
];

let renderer, scene, camera, controls, composer, bloomPass, container, ro, clock;
let started = false, disposed = false;
let bodyGroup, atlasGroup, shellMat = null, glowTex = null;
const organs = {};                       // meshKey -> { pivot, tissue, fill, halo, comp }
let vessels = null;

/* Projection state */
let projection = null;                   // the /aegis/pk payload
let arms = [], armIndex = 0, arm = null;
let tHours = 0, playing = false, playRate = 1.0;
let armNorm = null;                      // per-arm normalisation (see computeNorm)
let onFrame = null;                      // UI callback: (t_h, levels, arm) => void

/* ------------------------------------------------------------------ */
/* Materials                                                          */
/* ------------------------------------------------------------------ */
function makeTissueMat() {
  // The organ itself: dark, matte, barely there. It exists to give the drug
  // somewhere to be — a silhouette and a volume — not to carry information.
  return new THREE.MeshStandardMaterial({
    color: TISSUE, emissive: new THREE.Color(0x000000), emissiveIntensity: 1.0,
    transparent: true, opacity: 0.34, roughness: 0.55, metalness: 0.0,
    depthWrite: true, side: THREE.DoubleSide,
  });
}
function makeFillMat(hex) {
  // The drug. Additive, depth-tested but not depth-writing, so concentration
  // accumulates visually through the depth of the organ the way a real
  // volumetric load would — a thick liver reads denser than a thin pancreas at
  // the same concentration, which is exactly right.
  return new THREE.MeshBasicMaterial({
    color: new THREE.Color(hex), transparent: true, opacity: 0.0,
    blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide,
  });
}
function makeVesselMat(hex) {
  return new THREE.MeshBasicMaterial({
    color: new THREE.Color(hex), transparent: true, opacity: 0.12,
    blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide,
  });
}
function makeShellMat() {
  // A quieter shell than the clinical twin's: no scan sweep, no pulse. This scene
  // is read as a chart, and a moving body competes with a moving quantity.
  const m = new THREE.MeshBasicMaterial({
    color: new THREE.Color(0x2a2f36), transparent: true, opacity: 0.055,
    depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
  });
  return m;
}
function glowTexture() {
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const g = c.getContext("2d").createRadialGradient(64, 64, 0, 64, 64, 64);
  g.addColorStop(0, "rgba(255,255,255,1)");
  g.addColorStop(0.35, "rgba(255,255,255,0.35)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  const ctx = c.getContext("2d"); ctx.fillStyle = g; ctx.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

/* ------------------------------------------------------------------ */
/* Anatomy — borrowed from the clinical twin if it has already parsed it */
/* ------------------------------------------------------------------ */
function adoptOrgan(key, srcPivot) {
  const pivot = srcPivot.clone(true);          // shares BufferGeometry by reference
  const tissue = makeTissueMat();
  pivot.traverse((o) => { if (o.isMesh) { o.material = tissue; o.frustumCulled = false; } });

  const comp = MESH_COMPARTMENT[key];
  const vascular = key === "vessels";
  let fill = null, halo = null;
  if (vascular) {
    // ONE material for the whole tree — the vascular mesh is many sub-meshes, and
    // giving each its own material means only the first one ever gets updated.
    const vm = makeVesselMat(0x5fb3d4);
    pivot.traverse((o) => { if (o.isMesh) { o.material = vm; o.frustumCulled = false; } });
    vessels = { pivot, mat: vm };
  } else if (comp) {
    // The drug layer: the same geometry again, a hair larger so it reads as a
    // volume filling the organ rather than as a coat of paint on its surface.
    fill = pivot.clone(true);
    const fm = makeFillMat(0x5fb3d4);
    fill.traverse((o) => { if (o.isMesh) { o.material = fm; o.frustumCulled = false; } });
    fill.scale.multiplyScalar(1.006);
    fill.userData.mat = fm;
    halo = new THREE.Sprite(new THREE.SpriteMaterial({
      map: glowTex, color: new THREE.Color(0x5fb3d4), transparent: true, opacity: 0,
      depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending }));
    halo.scale.set(0.15, 0.15, 1); halo.visible = false;
  } else {
    // Unmodelled: force it inert and leave it that way.
    tissue.color.setHex(INERT); tissue.opacity = 0.16;
  }
  atlasGroup.add(pivot);
  if (fill) atlasGroup.add(fill);
  if (halo) scene.add(halo);
  organs[key] = { pivot, fill, halo, tissue, comp, vascular,
                  fillMat: fill ? fill.userData.mat : null };
}

function borrowAnatomy() {
  const lent = window.AegisTwin && window.AegisTwin._lendAnatomy
    ? window.AegisTwin._lendAnatomy() : null;
  if (!lent) return false;
  // Body shell — clone the already-posed, already-normalised humanoid so both
  // twins stand identically; only the material differs.
  bodyGroup = lent.body.clone(true);
  shellMat = makeShellMat();
  bodyGroup.traverse((o) => {
    if (!o.isMesh) return;
    if (/joint/i.test(o.name)) { o.visible = false; return; }
    o.material = shellMat; o.frustumCulled = false; o.renderOrder = 6;
  });
  scene.add(bodyGroup);
  // The lent atlas is already calibrated into the body; copying its transform
  // means this scene inherits the seating rather than re-deriving it.
  atlasGroup = new THREE.Group();
  atlasGroup.position.copy(lent.atlas.position);
  atlasGroup.scale.copy(lent.atlas.scale);
  atlasGroup.rotation.copy(lent.atlas.rotation);
  scene.add(atlasGroup);
  for (const k in lent.organs) adoptOrgan(k, lent.organs[k]);
  applyLevels();
  return true;
}

function loadAnatomy() {
  // Fallback path: the clinical twin has not finished parsing (or is not on this
  // page). Same files, so the browser serves them from cache — the fetch is warm,
  // the parse is not free, which is exactly why borrowing is preferred.
  atlasGroup = new THREE.Group(); scene.add(atlasGroup);
  const loader = new GLTFLoader();
  loader.load("./vendor/models/Xbot.glb", (gltf) => {
    bodyGroup = gltf.scene; shellMat = makeShellMat();
    bodyGroup.traverse((o) => {
      if (!o.isMesh) return;
      if (/joint/i.test(o.name)) { o.visible = false; return; }
      o.material = shellMat; o.frustumCulled = false; o.renderOrder = 6;
    });
    normalizeBySkeleton(bodyGroup); scene.add(bodyGroup); calibrate();
  }, undefined, () => {});
  let n = 0;
  const need = ORGAN_FILES.filter((c) => !c.defer && !c.calExclude).length;
  ORGAN_FILES.forEach((cfg) => {
    const go = () => loader.load("./vendor/models/organs/" + cfg.file, (g) => {
      const root = g.scene;
      const box = new THREE.Box3().setFromObject(root), c = box.getCenter(new THREE.Vector3());
      const pivot = new THREE.Group(); pivot.position.copy(c); root.position.sub(c); pivot.add(root);
      pivot.userData.calExclude = !!cfg.calExclude || !!cfg.vascular;
      adoptOrgan(cfg.key, pivot);                      // clones and parents it itself
      if (!cfg.defer && !cfg.calExclude && ++n >= need) calibrate();
      applyLevels();
    }, undefined, () => {});
    if (cfg.defer) setTimeout(go, 900); else go();
  });
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
  const s = (H * 0.9) / ((b.hi - b.lo) || 1);
  root.scale.setScalar(s); root.userData.baseScale = s;
  b = bounds();
  root.position.x -= b.cx; root.position.z -= b.cz; root.position.y -= b.lo - H * 0.05;
  root.updateWorldMatrix(true, true);
}

function calibrate() {
  const core = Object.values(organs).filter((o) => o.pivot && !o.vascular
    && !o.pivot.userData.calExclude);
  if (!atlasGroup || core.length < 6 || !bodyGroup) return;
  atlasGroup.scale.set(1, 1, 1); atlasGroup.position.set(0, 0, 0);
  atlasGroup.updateWorldMatrix(true, true);
  const box = new THREE.Box3(); core.forEach((o) => box.expandByObject(o.pivot));
  const c = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
  const bb = new THREE.Box3().setFromObject(bodyGroup);
  const lo = bb.min.y + (bb.max.y - bb.min.y) * 0.50;
  const hi = bb.min.y + (bb.max.y - bb.min.y) * 0.82;
  const torsoH = Math.max(0.1, hi - lo);
  const s = (torsoH * 0.78) / (size.y || 1);
  atlasGroup.scale.set(s, s, s);
  atlasGroup.position.set(-s * c.x, lo + torsoH * 0.56 - s * c.y, -s * c.z);
  // Keep every drug layer locked onto its organ after the atlas moves.
  for (const k in organs) {
    const o = organs[k];
    if (o.fill) { o.fill.position.copy(o.pivot.position); o.fill.scale.copy(o.pivot.scale).multiplyScalar(1.006); }
  }
}

/* ------------------------------------------------------------------ */
/* The projection → what each organ shows                             */
/* ------------------------------------------------------------------ */
/* Normalisation is the one honest-presentation decision in this file.

   Every compartment is scaled by the SAME number — the peak concentration
   reached by any compartment at any time in this arm — so the relative loading
   between organs is comparable: the kidney genuinely holds several times what
   muscle holds per litre, and the picture says so. Normalising each organ against
   its own maximum would make every organ peak at full brightness and destroy
   exactly the comparison the twin exists to make.

   The only concession to the eye is a display gamma: brightness is level^0.62,
   because a linear ramp leaves the low-concentration compartments visually
   indistinguishable from zero. It bends the BRIGHTNESS, never the number — the
   readouts alongside show true concentrations in the drug's own units. */
const DISPLAY_GAMMA = 0.62;

function computeNorm(a) {
  let peak = 0, vpeak = 0;
  for (const key in a.organs) {
    const series = a.organs[key];
    if (key === "arterial" || key === "venous") {
      for (let i = 0; i < series.length; i++) if (series[i] > vpeak) vpeak = series[i];
      continue;
    }
    for (let i = 0; i < series.length; i++) if (series[i] > peak) peak = series[i];
  }
  return { peak: peak || 1, vessel: vpeak || 1 };
}

function sampleAt(series, times, t) {
  if (!series || !series.length) return 0;
  if (t <= times[0]) return series[0];
  const last = times.length - 1;
  if (t >= times[last]) return series[last];
  // The time axis is uniform, so locate by arithmetic and interpolate.
  const step = (times[last] - times[0]) / last;
  const f = (t - times[0]) / step;
  const i = Math.min(last - 1, Math.max(0, Math.floor(f)));
  const w = f - i;
  return series[i] * (1 - w) + series[i + 1] * w;
}

/* The per-compartment concentration at the current instant, plus its display
   level. Returned to the UI so the readouts and the render can never disagree. */
function levelsNow() {
  const out = {};
  if (!arm) return out;
  const times = arm.times_h;
  for (const key in arm.organs) {
    const v = sampleAt(arm.organs[key], times, tHours);
    const denom = (key === "arterial" || key === "venous") ? armNorm.vessel : armNorm.peak;
    out[key] = { value: v, level: Math.max(0, Math.min(1, v / denom)) };
  }
  return out;
}

function applyLevels() {
  if (!arm || !armNorm) return;
  const lv = levelsNow();
  const hex = DRUG_COLOR[arm.drug_class] || DRUG_COLOR.antibiotic;
  const col = new THREE.Color(hex);

  for (const k in organs) {
    const o = organs[k];
    if (o.vascular) continue;
    if (!o.comp) continue;                          // unmodelled — stays inert
    const rec = lv[o.comp];
    const level = rec ? Math.pow(rec.level, DISPLAY_GAMMA) : 0;
    if (o.fillMat) {
      o.fillMat.color.copy(col);
      o.fillMat.opacity = 0.06 + 0.62 * level;
    }
    // The tissue itself lifts slightly so a loaded organ reads as solid rather
    // than as a glow hovering in an empty silhouette.
    o.tissue.opacity = 0.30 + 0.22 * level;
    o.tissue.emissive.copy(col).multiplyScalar(0.30 * level);
    if (o.halo) {
      o.halo.visible = level > 0.06;
      o.halo.material.color.copy(col);
      o.halo.material.opacity = 0.34 * level;
      const s = 0.13 + 0.16 * level;
      o.halo.scale.set(s, s, 1);
      o.pivot.getWorldPosition(o.halo.position);
    }
  }
  // The vascular tree carries the plasma channel — this is what makes the bolus
  // legible: the vessels light before any organ does, because that is the route.
  if (vessels && vessels.mat) {
    const art = lv.arterial ? Math.pow(lv.arterial.level, DISPLAY_GAMMA) : 0;
    vessels.mat.color.copy(col);
    vessels.mat.opacity = 0.06 + 0.40 * art;
  }
  // Muscle, skin, fat and the remainder have no mesh of their own — they are most
  // of the body by mass and hold most of the drug, so the shell carries them.
  if (shellMat) {
    const rest = ["muscle", "skin", "adipose", "rest"]
      .map((k) => (lv[k] ? lv[k].level : 0))
      .reduce((a, b) => Math.max(a, b), 0);
    const g = Math.pow(rest, DISPLAY_GAMMA);
    shellMat.color.copy(new THREE.Color(0x2a2f36)).lerp(col, 0.55 * g);
    shellMat.opacity = 0.055 + 0.075 * g;
  }
  if (onFrame) onFrame(tHours, lv, arm);
}

/* ------------------------------------------------------------------ */
/* Public API                                                         */
/* ------------------------------------------------------------------ */
function mount(el) {
  if (started || !el) return;
  container = el;
  const w = el.clientWidth || 520, h = el.clientHeight || 560;
  scene = new THREE.Scene(); scene.fog = new THREE.FogExp2(0x05070d, 0.11);
  camera = new THREE.PerspectiveCamera(33, w / h, 0.1, 100);
  camera.position.set(0.18, H * 0.56, H * 1.85);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio)); renderer.setSize(w, h);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.0;
  el.querySelectorAll("canvas").forEach((c) => c.remove());
  el.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.06; controls.enablePan = false;
  controls.target.set(0, H * 0.50, 0);
  controls.autoRotate = true; controls.autoRotateSpeed = 0.22;
  controls.minPolarAngle = Math.PI * 0.36; controls.maxPolarAngle = Math.PI * 0.58;

  // Darker key than the clinical twin: the drug layer is additive, so any
  // ambient lift on the tissue eats the contrast the concentration needs.
  scene.add(new THREE.AmbientLight(0xffffff, 0.26));
  const key = new THREE.DirectionalLight(0xffffff, 0.75); key.position.set(0.6, 1.4, 1.2); scene.add(key);
  const rim = new THREE.DirectionalLight(0xc8ced6, 0.55); rim.position.set(0, 0.5, -1.5); scene.add(rim);
  scene.add(new THREE.HemisphereLight(0x8a9099, 0x0c0e11, 0.30));

  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  bloomPass = new UnrealBloomPass(new THREE.Vector2(w, h), 0.46, 0.62, 0.72);
  composer.addPass(bloomPass);
  composer.addPass(new OutputPass());

  glowTex = glowTexture();
  if (!borrowAnatomy()) loadAnatomy();

  clock = new THREE.Clock(); started = true; disposed = false;
  ro = new ResizeObserver(resize); ro.observe(el);
  animate();
}

function setProjection(payload, opts) {
  projection = payload || null;
  arms = (payload && payload.arms) || [];
  armIndex = 0;
  onFrame = (opts && opts.onFrame) || onFrame;
  selectArm(0);
}

function selectArm(i) {
  if (!arms.length) { arm = null; armNorm = null; return; }
  armIndex = Math.max(0, Math.min(arms.length - 1, i | 0));
  arm = arms[armIndex];
  armNorm = computeNorm(arm);
  tHours = 0;
  applyLevels();
}

function setArm(i) { selectArm(i); }
function setTime(h) {
  if (!arm) return;
  tHours = Math.max(0, Math.min(arm.horizon_h, +h || 0));
  applyLevels();
}
function play() { playing = true; }
function pause() { playing = false; }
function setRate(r) { playRate = Math.max(0.05, +r || 1); }

function resize() {
  if (!renderer || !container) return;
  const w = container.clientWidth, h = container.clientHeight;
  if (!w || !h) return;
  camera.aspect = w / h; camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  if (composer) composer.setSize(w, h);
  if (bloomPass) bloomPass.setSize(w, h);
}

function animate() {
  if (disposed) return;
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  if (!renderer || !container || !container.offsetParent) return;   // overlay closed
  controls.update();
  if (playing && arm) {
    // One pass over the drug's OWN timeframe takes a fixed wall-clock duration,
    // so a 2 h vasopressor and a 24 h antibiotic both read at a watchable speed
    // and the header states the real hours either way.
    const secondsPerPass = 14.0 / playRate;
    tHours += dt * (arm.horizon_h / secondsPerPass);
    if (tHours >= arm.horizon_h) { tHours = arm.horizon_h; playing = false; }
    applyLevels();
  }
  if (composer) composer.render(); else renderer.render(scene, camera);
}

function dispose() {
  disposed = true; started = false; playing = false;
  if (ro && container) { try { ro.unobserve(container); } catch (e) {} }
  ro = null;
  if (renderer) {
    // Two WebGL contexts is one more than a laptop wants held open. The geometry
    // is BORROWED from the clinical twin and must survive, so only this scene's
    // own materials, its render targets and the context itself are released.
    const seen = new Set();
    scene && scene.traverse((o) => {
      if (!(o.isMesh || o.isSprite)) return;
      const list = Array.isArray(o.material) ? o.material : [o.material];
      list.forEach((m) => { if (m && !seen.has(m)) { seen.add(m); m.dispose && m.dispose(); } });
      // o.geometry is deliberately NOT disposed: it belongs to the clinical twin.
    });
    if (composer && composer.dispose) composer.dispose();
    renderer.dispose();
    if (renderer.forceContextLoss) renderer.forceContextLoss();
    renderer.domElement && renderer.domElement.remove();
  }
  renderer = scene = camera = controls = composer = bloomPass = null;
  bodyGroup = atlasGroup = vessels = shellMat = null;
  for (const k in organs) delete organs[k];
  container = null;
}

function state() {
  return {
    started, playing, t_h: tHours, arm: armIndex,
    arms: arms.map((a) => ({ label: a.label, drug_class: a.drug_class,
                             horizon_h: a.horizon_h, unit: a.unit, kind: a.kind })),
    unmodelled: Object.keys(UNMODELLED_LABEL).map((k) => UNMODELLED_LABEL[k]),
    levels: levelsNow(),
  };
}

window.AegisPKTwin = {
  mount, setProjection, setArm, setTime, play, pause, setRate, resize, dispose,
  state, levelsNow, isPlaying: () => playing, currentArm: () => arm,
  DRUG_COLOR, UNMODELLED: UNMODELLED_LABEL,
};
