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

/* The bloodstream is now drawn as blood, and the drug is drawn as something moving
   THROUGH it. Previously the vessels only existed when a drug was in them, which
   made the circulation look like a property of the dose rather than of the patient
   — at t=0 the body had no vasculature at all.

   Arterial red / venous blue is the one colour convention every anatomy text and
   every clinician already shares, so it costs no learning. Both are deliberately
   DESATURATED: the saturated end of the warm ramp belongs to the adverse-event
   rings, and a bloodstream that shouted as loudly as a life-threatening event
   would wreck that distinction. Blood sits back; the drug moves. */
const BLOOD_ARTERIAL = 0x8c3a44;
const BLOOD_VENOUS = 0x3c5570;

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
/* A repeating stripe run along the tube's length. TubeGeometry lays u down the
   axis, so scrolling this map's offset makes the drug appear to travel — bolus
   after bolus — instead of the whole vessel brightening at once. Soft-edged on
   purpose: hard bands read as a barber pole rather than as flow. */
function flowTexture() {
  const c = document.createElement("canvas"); c.width = 128; c.height = 1;
  const ctx = c.getContext("2d");
  const g = ctx.createLinearGradient(0, 0, 128, 0);
  g.addColorStop(0.00, "rgba(255,255,255,0.10)");
  g.addColorStop(0.34, "rgba(255,255,255,0.16)");
  g.addColorStop(0.52, "rgba(255,255,255,1.00)");
  g.addColorStop(0.68, "rgba(255,255,255,0.30)");
  g.addColorStop(1.00, "rgba(255,255,255,0.10)");
  ctx.fillStyle = g; ctx.fillRect(0, 0, 128, 1);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  return t;
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
const ORGAN_CFG = {};
ORGAN_FILES.forEach((c) => { ORGAN_CFG[c.key] = c; });

function adoptOrgan(key, srcPivot) {
  const pivot = srcPivot.clone(true);          // shares BufferGeometry by reference
  // Carry the calibration exclusions onto the clone. The clinical twin keeps
  // these on its own config record rather than on the pivot, so a borrowed
  // pivot arrives without them — and then the long spinal cord and the pelvic
  // bladder drag the fitted bounds down until the camera frames the hips.
  const cfg = ORGAN_CFG[key];
  pivot.userData.calExclude = !!(cfg && (cfg.calExclude || cfg.vascular));
  const tissue = makeTissueMat();
  pivot.traverse((o) => { if (o.isMesh) { o.material = tissue; o.frustumCulled = false; } });

  const comp = MESH_COMPARTMENT[key];
  const vascular = key === "vessels";
  let fill = null, halo = null;
  if (vascular) {
    // ONE material for the whole tree — the vascular mesh is many sub-meshes, and
    // giving each its own material means only the first one ever gets updated.
    // Blood-coloured at rest like the schematic tree, so the trunk vasculature and
    // the limb vasculature do not look like two different organs.
    const vm = makeVesselMat(BLOOD_ARTERIAL);
    vm.blending = THREE.NormalBlending;
    vm.opacity = 0.30;
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
  // The RISK layer — an outward-facing shell around the organ, dark until an
  // adverse event is selected. It is a separate channel from the drug fill on
  // purpose: the drug is cool and means "where the dose went", the risk shell is
  // warm and means "what this exposure threatens". Overloading one channel with
  // both is how a bright kidney would come to mean two opposite things.
  let risk = null;
  if (!vascular) {
    risk = pivot.clone(true);
    const rm = new THREE.MeshBasicMaterial({
      color: new THREE.Color(0xe5544c), transparent: true, opacity: 0.0,
      blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false,
      side: THREE.BackSide,          // BackSide → reads as a rim around the organ
    });
    risk.traverse((o) => { if (o.isMesh) { o.material = rm; o.frustumCulled = false; } });
    risk.scale.multiplyScalar(1.10);
    risk.userData.mat = rm;
    risk.visible = false;
  }
  atlasGroup.add(pivot);
  if (fill) atlasGroup.add(fill);
  if (risk) atlasGroup.add(risk);
  if (halo) scene.add(halo);
  organs[key] = { pivot, fill, halo, risk, tissue, comp, vascular,
                  fillMat: fill ? fill.userData.mat : null,
                  riskMat: risk ? risk.userData.mat : null };
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
  bodyGroup.updateWorldMatrix(true, true);
  buildVesselTree(bodyGroup);
  // The lent atlas is already calibrated into the body; copying its transform
  // means this scene inherits the seating rather than re-deriving it.
  atlasGroup = new THREE.Group();
  atlasGroup.position.copy(lent.atlas.position);
  atlasGroup.scale.copy(lent.atlas.scale);
  atlasGroup.rotation.copy(lent.atlas.rotation);
  scene.add(atlasGroup);
  for (const k in lent.organs) adoptOrgan(k, lent.organs[k]);
  buildLabels(); fitToAtlas(); applyLevels();
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
    normalizeBySkeleton(bodyGroup); scene.add(bodyGroup);
    bodyGroup.updateWorldMatrix(true, true); buildVesselTree(bodyGroup); calibrate();
  }, undefined, () => {});
  let n = 0;
  const need = ORGAN_FILES.filter((c) => !c.defer && !c.calExclude).length;
  ORGAN_FILES.forEach((cfg) => {
    const go = () => loader.load("./vendor/models/organs/" + cfg.file, (g) => {
      const root = g.scene;
      const box = new THREE.Box3().setFromObject(root), c = box.getCenter(new THREE.Vector3());
      const pivot = new THREE.Group(); pivot.position.copy(c); root.position.sub(c); pivot.add(root);
      adoptOrgan(cfg.key, pivot);                      // clones and parents it itself
      if (!cfg.defer && !cfg.calExclude && ++n >= need) {
        calibrate(); buildLabels(); fitToAtlas();
      }
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

/* Concentration → colour. Opacity alone was not enough to read: between a kidney
   at 30% and one at 80% the difference was a slightly less transparent shape, and
   the whole point of the view was getting lost. Loading now moves along a RAMP —
   a dim, desaturated version of the drug hue at trace levels, the full hue in the
   mid range, and a hot near-white core at the peak — which is how any thermal or
   uptake image is read, and which the eye resolves far better than alpha. */
const WHITE = new THREE.Color(0xffffff);
function ramp(base, level, out) {
  out.copy(base);
  if (level < 0.55) out.multiplyScalar(0.42 + 1.05 * level);  // dim → full hue
  else out.lerp(WHITE, (level - 0.55) * 0.80);                // full hue → hot core
  return out;
}
const _c1 = new THREE.Color(), _c2 = new THREE.Color();

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
    // First arrival is the single most informative instant for an organ — it is
    // the perfusion order made visible — and it used to slide past unmarked. The
    // organ now flashes as the drug reaches it, once, on the way up.
    if (level > 0.04 && (o.prev || 0) <= 0.04) o.flash = 1.0;
    o.prev = level;
    o.level = level;
    if (o.fillMat) {
      ramp(col, level, _c1);
      o.fillMat.color.copy(_c1);
      o.fillMat.opacity = 0.015 + 0.38 * level;
    }
    // The tissue itself lifts so a loaded organ reads as a solid, dense volume
    // rather than a glow hovering inside an empty silhouette.
    o.tissue.opacity = 0.26 + 0.32 * level;
    o.tissue.emissive.copy(_c1).multiplyScalar(0.42 * level);
    if (o.halo) {
      const boost = level + 0.55 * (o.flash || 0);
      o.halo.visible = boost > 0.04;
      o.halo.material.color.copy(_c1);
      o.halo.material.opacity = Math.min(0.5, 0.30 * boost);
      const s = 0.11 + 0.24 * boost;
      o.halo.scale.set(s, s, 1);
      o.pivot.getWorldPosition(o.halo.position);
    }
  }
  // The full-body tree. Each segment shows the arterial concentration as it was
  // one transit time AGO, so the front genuinely travels: out of the heart, down
  // the aorta, splitting at the iliacs, reaching the feet last. Sampling the same
  // curve at a delayed time is the whole mechanism — no keyframes, no easing.
  if (vesselTree) {
    const times = arm.times_h, series = arm.organs.arterial;
    vesselTree.segs.forEach((sg) => {
      const v = series ? sampleAt(series, times, tHours - sg.delay) : 0;
      const lvl = Math.pow(Math.max(0, Math.min(1, v / armNorm.vessel)), DISPLAY_GAMMA);
      sg.level = lvl;
      // The blood layer never goes away — it only brightens where the vessel is
      // carrying something, so the circulation stays legible at t=0 instead of the
      // patient appearing to have no vasculature until a drug is given.
      sg.bloodMat.opacity = (sg.kind === "a" ? 0.62 : 0.48) + 0.18 * lvl;
      // The drug rides ON the blood in its own colour rather than recolouring it:
      // at trace levels you see red vessels threaded with cyan, at peak the thread
      // dominates. Two questions — "where are the vessels" and "how much drug is
      // in them" — stop competing for one channel.
      ramp(col, lvl, _c2);
      sg.flowMat.color.copy(_c2);
      sg.flowMat.opacity = (sg.kind === "a" ? 0.62 : 0.42) * lvl;
      sg.flow.visible = lvl > 0.004;
    });
  }
  // The vascular tree carries the plasma channel — this is what makes the bolus
  // legible: the vessels light before any organ does, because that is the route.
  // A SURGE term on the rising edge is what turns a slow fade into an arrival:
  // while plasma concentration is climbing, the vasculature flares and then
  // settles, which is the delivery itself rather than a summary of it.
  if (vessels && vessels.mat) {
    const art = lv.arterial ? Math.pow(lv.arterial.level, DISPLAY_GAMMA) : 0;
    const rising = Math.max(0, art - (vesselPrev == null ? art : vesselPrev));
    vesselSurge = Math.max(vesselSurge * 0.90, Math.min(1, rising * 26));
    vesselPrev = art;
    // Blood at rest, lerping toward the drug hue as the trunk loads. Unlike the
    // schematic tree this mesh has no usable length-wise UVs, so it cannot carry
    // the scrolling stripe — it tints instead of flowing.
    ramp(col, Math.min(1, art + 0.5 * vesselSurge), _c2);
    _c1.setHex(BLOOD_ARTERIAL).lerp(_c2, Math.min(1, 0.35 + 0.65 * art));
    vessels.mat.color.copy(_c1);
    vessels.mat.opacity = 0.26 + 0.22 * art + 0.10 * vesselSurge;
  }
  // Muscle, skin, fat and the remainder have no mesh of their own — they are most
  // of the body by mass and hold most of the drug, so the shell carries them.
  // …unless an adverse event has claimed the shell, in which case the animation
  // loop owns it and writing the drug tint here every frame would erase the pulse.
  if (shellMat && !(riskFocus && riskFocus.shell)) {
    const rest = ["muscle", "skin", "adipose", "rest"]
      .map((k) => (lv[k] ? lv[k].level : 0))
      .reduce((a, b) => Math.max(a, b), 0);
    const g = Math.pow(rest, DISPLAY_GAMMA);
    shellMat.color.copy(SHELL_BASE).lerp(col, 0.60 * g);
    shellMat.opacity = 0.075 + 0.105 * g;
  }
  if (onFrame) onFrame(tHours, lv, arm);
}
const SHELL_BASE = new THREE.Color(0x2f353d);
let vesselPrev = null, vesselSurge = 0, labelTick = 0, riskPhase = 0;

/* ------------------------------------------------------------------ */
/* The circulation — a full-body vascular tree                        */
/* ------------------------------------------------------------------ */
/* The CCF vasculature mesh stops at the trunk, so the drug appeared to reach the
   organs and then stop existing: no delivery to the head, the arms or the legs,
   which between them are most of the body and most of the muscle compartment the
   model is actually solving for. This builds the missing circulation.

   WHAT IT IS. A schematic arterial tree and its venous return, routed along the
   body's OWN SKELETON — the bones are read out of the rigged mesh, so the vessels
   follow this body's actual shoulders, hips and limb lengths rather than a set of
   coordinates that would drift the moment the model changed. Aortic arch to
   carotids, subclavians down each arm, descending aorta to the iliacs and down
   each leg, with veins running alongside.

   WHAT IT IS NOT. It is not this patient's vasculature, and it is not a segmented
   angiogram: it is the major-vessel plan every anatomy text draws, at this body's
   proportions. It carries a real quantity — the model's arterial concentration —
   along a schematic route, and the twin says so rather than implying the geometry
   was measured.

   WHY IT EARNS ITS PLACE. Transit is not instant. Blood leaves the heart and
   reaches the foot several seconds later, and rendering that delay is what turns
   "the vessels are glowing" into "the drug is moving": on the first pass you watch
   the front travel out the aorta, split at the iliacs and run down both legs. The
   delay is computed from path length at a stated mean circulation velocity, so it
   is a physiological claim, not an animation curve. */
const VESSEL_SOURCE = "schematic major-vessel plan, routed along this body's skeleton";
// Mean flow velocity used to convert path length to arrival delay. Aortic flow is
// far faster than capillary flow; this is a whole-circuit average chosen so one
// complete pass takes about a minute, which is the textbook figure.
const CIRCULATION_M_PER_S = 0.22;

function boneMap(root) {
  const B = {};
  root.traverse((o) => { if (o.isBone) B[o.name.toLowerCase().replace(/^mixamorig:?/, "")] = o; });
  return B;
}
function bonePos(B, ...names) {
  for (const n of names) {
    const k = Object.keys(B).find((k) => k === n) || Object.keys(B).find((k) => k.includes(n));
    if (k) { const v = new THREE.Vector3(); B[k].getWorldPosition(v); return v; }
  }
  return null;
}

/* Each route: a name, the ordered skeleton anchors it threads through, and a
   radius. Arteries run slightly anterior, veins slightly posterior, so the two
   read as a pair rather than one doubled line. */
function vesselRoutes(B) {
  const P = (...n) => bonePos(B, ...n);
  const hips = P("hips", "pelvis"), spine = P("spine1", "spine"), chest = P("spine2", "spine1");
  const neck = P("neck"), head = P("head");
  if (!hips || !chest) return [];
  const heart = chest.clone().add(new THREE.Vector3(0.012, -0.02, 0.03));
  const routes = [];
  const side = (s) => (s === "l" ? "left" : "right");

  // Calibre is to scale on a 1.7 m body: a ~2 cm aorta, ~6 mm carotid, ~4 mm
  // brachial, ~7 mm femoral. The first pass used aorta-width tubes in every limb,
  // which read as plumbing rather than anatomy and drowned the organs behind them.
  const arch = chest.clone().lerp(neck || chest, 0.45);   // aortic arch / great vessels
  routes.push({ key: "aorta", pts: [heart, arch, chest, spine || chest, hips], r: 0.0105 });
  if (neck && head) {
    // Carotids run as a pair either side of the midline, not up the centre.
    [-1, 1].forEach((sgn) => routes.push({
      key: "carotid" + (sgn < 0 ? "L" : "R"), r: 0.0032,
      pts: [arch,
            neck.clone().add(new THREE.Vector3(sgn * 0.022, 0, 0)),
            head.clone().add(new THREE.Vector3(sgn * 0.016, 0, 0))] }));
  }
  ["l", "r"].forEach((s) => {
    const sh = P(side(s) + "arm", side(s) + "shoulder");
    const el = P(side(s) + "forearm");
    const hd = P(side(s) + "hand");
    if (sh && el && hd) {
      // Subclavian off the arch, then along the limb itself — routing it from the
      // heart made it bow out across the chest on its way to the shoulder.
      routes.push({ key: "arm" + s.toUpperCase(), r: 0.0028,
                    pts: [arch, sh.clone().lerp(arch, 0.28), sh, el, hd] });
    }
    const up = P(side(s) + "upleg", side(s) + "thigh");
    const lo = P(side(s) + "leg", side(s) + "shin");
    const ft = P(side(s) + "foot");
    if (up && lo && ft) {
      // Common iliac off the aortic bifurcation, then femoral down the leg.
      const iliac = hips.clone().lerp(up, 0.45);
      routes.push({ key: "leg" + s.toUpperCase(), r: 0.0042,
                    pts: [hips.clone(), iliac, up, lo, ft] });
    }
  });
  return routes;
}

let vesselTree = null;   // { group, segs:[{mesh, mat, delay, t0}] }

let flowTex = null;
function buildVesselTree(root) {
  if (!flowTex) flowTex = flowTexture();
  const B = boneMap(root);
  const routes = vesselRoutes(B);
  if (!routes.length) return false;
  const group = new THREE.Group();
  const segs = [];
  routes.forEach((rt) => {
    const pts = rt.pts.filter(Boolean);
    if (pts.length < 2) return;
    [["a", 0.010, 1.0], ["v", -0.012, 1.30]].forEach(([kind, off, rmul]) => {
      // Artery anterior, vein posterior and a little wider — the venous side of
      // the circulation genuinely carries more volume at lower pressure.
      const shifted = pts.map((p, i) =>
        p.clone().add(new THREE.Vector3(0, 0, i === 0 ? off * 0.4 : off)));
      const curve = new THREE.CatmullRomCurve3(shifted, false, "centripetal");
      const len = curve.getLength();
      const geo = new THREE.TubeGeometry(curve, Math.max(20, Math.round(len * 70)),
                                         rt.r * rmul, 6, false);
      // LAYER 1 — the blood itself. Always present, never animated, normal
      // blending so it reads as a filled vessel rather than a glow.
      const bloodMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(kind === "a" ? BLOOD_ARTERIAL : BLOOD_VENOUS),
        transparent: true, opacity: kind === "a" ? 0.62 : 0.48,
        depthWrite: false });
      const blood = new THREE.Mesh(geo, bloodMat);
      blood.frustumCulled = false;
      group.add(blood);

      // LAYER 2 — the drug moving through it. Same geometry, a hair wider so it
      // sheathes rather than z-fights, additive and scrolling.
      const tex = flowTex.clone();
      tex.needsUpdate = true;
      // Repeat by LENGTH, so a stripe is the same physical size in a carotid and
      // in a femoral. A shared repeat would make short vessels look fast.
      tex.repeat.set(Math.max(2, Math.round(len * 5)), 1);
      const flowMat = new THREE.MeshBasicMaterial({
        color: new THREE.Color(0x5fb3d4), map: tex, transparent: true,
        opacity: 0.0, blending: THREE.AdditiveBlending, depthWrite: false });
      const flowGeo = new THREE.TubeGeometry(
        curve, Math.max(20, Math.round(len * 70)), rt.r * rmul * 1.22, 6, false);
      const flow = new THREE.Mesh(flowGeo, flowMat);
      flow.frustumCulled = false;
      group.add(flow);

      // Arrival delay from path length at the stated mean velocity. Venous return
      // lags by the same journey again, which is why the veins light second.
      const delay = len / CIRCULATION_M_PER_S / 3600;      // hours
      segs.push({ blood, bloodMat, flow, flowMat, tex, len,
                  delay: kind === "a" ? delay : delay * 2.1, kind });
    });
  });
  scene.add(group);
  vesselTree = { group, segs };
  return true;
}

/* ------------------------------------------------------------------ */
/* Framing — the drug lives in the torso, so frame the torso           */
/* ------------------------------------------------------------------ */
/* Inherited framing from the clinical twin put the whole body head-to-feet in
   shot. That is right for a severity view, where the question is "which part of
   this patient", and wrong here, where every value lives inside one 40 cm band
   and the legs are a third of the frame carrying no information. Fitting to the
   organ atlas roughly triples the organs on screen, which is most of why the
   changes were hard to see at all. */
function fitToAtlas() {
  fitMode = "atlas";
  if (!camera || !controls || !atlasGroup) return;
  const core = Object.values(organs).filter(
    (o) => o.pivot && !o.vascular && o.comp && !o.pivot.userData.calExclude);
  if (core.length < 4) return;
  // Bounds come from world matrices, and nothing has rendered yet when this runs
  // straight after the organs are adopted — every matrixWorld is still identity,
  // so the box collapses toward the origin and the camera ends up framing the
  // feet. Force the update before measuring.
  atlasGroup.updateWorldMatrix(true, true);
  const box = new THREE.Box3();
  core.forEach((o) => box.expandByObject(o.pivot));
  if (box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  const extent = Math.max(size.y, size.x * 0.9, 0.12);
  const fov = camera.fov * Math.PI / 180;
  // 1.95 leaves room for the labels that sit outboard of the body.
  const dist = (extent * 1.95) / (2 * Math.tan(fov / 2));
  controls.target.copy(c);
  camera.position.set(c.x + dist * 0.16, c.y + extent * 0.10, c.z + dist);
  controls.minDistance = dist * 0.45;
  controls.maxDistance = dist * 2.6;
  camera.near = Math.max(0.01, dist * 0.02); camera.far = dist * 12;
  camera.updateProjectionMatrix();
  controls.update();
}

/* ------------------------------------------------------------------ */
/* Anatomical labels                                                  */
/* ------------------------------------------------------------------ */
/* A body with no names on it asks the viewer to already know the anatomy, and
   the numbers were only in the side rail — so reading "which organ is that bright
   one" meant looking away and matching by guesswork.

   Labels are placed in two OUTBOARD COLUMNS with leader lines back to the organ,
   the way an anatomical plate does it, rather than floating over the body. That
   is deliberate: labels pinned directly to their organ overlap each other
   constantly on a body this dense (the pancreas, spleen and kidney project within
   a few dozen pixels of one another), and an overlapping label is worse than
   none. Columns cannot overlap by construction, and the leader line carries the
   association instead. */
const LABEL_NAME = {
  lung: "Lungs", heart: "Heart", liver: "Liver", spleen: "Spleen",
  kidneyL: "Kidney", kidneyR: "Kidney", pancreas: "Pancreas",
  intestine: "Intestine", cord: "CNS", bladder: "Bladder", thymus: "Thymus",
};
let labelHost = null, labelSvg = null, labelItems = {}, labelsOn = true;
let labelUnit = "", labelFmt = null;

function buildLabels() {
  if (!container) return;
  labelHost = document.createElement("div");
  labelHost.className = "pkt-labels";
  labelSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  labelSvg.setAttribute("class", "pkt-leaders");
  labelHost.appendChild(labelSvg);
  container.appendChild(labelHost);
  labelItems = {};
  // One chip per labelled mesh. Both kidneys exist in the atlas but share a
  // compartment, so only one is labelled — two identical "Kidney 76.7" chips
  // would imply two independent measurements.
  const skip = { kidneyR: true };
  for (const k in organs) {
    if (skip[k] || !LABEL_NAME[k]) continue;
    const el = document.createElement("div");
    el.className = "pkt-label" + (organs[k].comp ? "" : " inert");
    el.innerHTML = '<b>' + LABEL_NAME[k] + "</b><span></span>";
    labelHost.appendChild(el);
    const line = document.createElementNS("http://www.w3.org/2000/svg", "polyline");
    line.setAttribute("class", "pkt-leader" + (organs[k].comp ? "" : " inert"));
    labelSvg.appendChild(line);
    labelItems[k] = { el, line, val: el.querySelector("span") };
  }
}

const _p = new THREE.Vector3();
function updateLabels() {
  if (!labelHost || !container || !camera) return;
  const w = container.clientWidth, h = container.clientHeight;
  if (!w || !h) return;
  if (!labelsOn) { labelHost.style.display = "none"; return; }
  labelHost.style.display = "";
  labelSvg.setAttribute("viewBox", "0 0 " + w + " " + h);

  // Pull the CURRENT values rather than reading a snapshot handed over by the
  // last render callback. A formatter that closes over a levels object goes stale
  // the moment the scrub moves without a repaint, and a label showing a different
  // number from the organ it points at is worse than no label at all.
  const lv = levelsNow();
  const rows = [];
  for (const k in labelItems) {
    const o = organs[k];
    if (!o || !o.pivot) { labelItems[k].el.style.display = "none"; continue; }
    o.pivot.getWorldPosition(_p);
    const anchor = _p.clone();
    _p.project(camera);
    // Behind the camera, or outside the frustum — drop it rather than clamp it to
    // an edge, where the leader line would point at the wrong thing.
    if (_p.z > 1) { labelItems[k].el.style.display = "none"; continue; }
    rows.push({ k, sx: (_p.x * 0.5 + 0.5) * w, sy: (-_p.y * 0.5 + 0.5) * h, anchor });
  }
  const left = rows.filter((r) => r.sx < w * 0.5).sort((a, b) => a.sy - b.sy);
  const right = rows.filter((r) => r.sx >= w * 0.5).sort((a, b) => a.sy - b.sy);

  const PITCH = 30, TOP = 84;   // clears the HUD block in the top-left
  function place(list, side) {
    // Walk down the column pushing each label below the previous one; a simple
    // monotone pass is enough because the list is already sorted by depth on
    // screen, and it keeps the leaders from crossing.
    let y = TOP;
    list.forEach((r) => {
      y = Math.max(y, Math.min(r.sy, h - 30));
      const it = labelItems[r.k];
      const o = organs[r.k];
      it.el.style.display = "";
      it.el.classList.toggle("hot", (o.level || 0) > 0.55);
      it.el.style.top = (y - 11) + "px";
      const x = side === "l" ? 10 : w - 10;
      it.el.style.left = side === "l" ? "10px" : "";
      it.el.style.right = side === "l" ? "" : "10px";
      const lw = it.el.offsetWidth || 96;
      const ex = side === "l" ? 10 + lw + 4 : w - 10 - lw - 4;
      const mx = side === "l" ? Math.min(r.sx, ex + 26) : Math.max(r.sx, ex - 26);
      it.line.setAttribute("points", ex + "," + y + " " + mx + "," + y + " " + r.sx + "," + r.sy);
      it.line.style.opacity = String(0.22 + 0.55 * (o.level || 0));
      if (it.val) {
        const rec = o.comp ? lv[o.comp] : null;
        it.val.textContent = o.comp
          ? (labelFmt && rec ? labelFmt(rec.value) : "")
          : "no compartment";
      }
      y += PITCH;
    });
  }
  place(left, "l"); place(right, "r");
}

/* Two framings, because the scene now has two scales of information. The torso
   view is where the organ numbers are legible; the body view is where the
   circulation reads as a circulation. Neither is right for both, so it is a
   toggle rather than a compromise that serves neither. */
function fitToBody() {
  fitMode = "body";
  if (!camera || !controls || !bodyGroup) return;
  bodyGroup.updateWorldMatrix(true, true);
  // Measure the BONES, not the mesh. Box3.setFromObject on a skinned mesh reads
  // the bind-pose geometry bounds, which ignore the skeleton this body is
  // actually posed with — the box comes out offset and oversized, and the camera
  // ends up aimed at empty space beside the patient. The bones are where the body
  // really is, which is why normalizeBySkeleton uses them too.
  const box = new THREE.Box3();
  const bv = new THREE.Vector3();
  let nb = 0;
  bodyGroup.traverse((o) => { if (o.isBone) { o.getWorldPosition(bv); box.expandByPoint(bv); nb++; } });
  if (nb < 4) box.setFromObject(bodyGroup);          // un-rigged fallback
  if (box.isEmpty()) return;
  const c = box.getCenter(new THREE.Vector3());
  const size = box.getSize(new THREE.Vector3());
  // Bones sit inside the flesh, so the skeleton box understates the silhouette.
  size.multiplyScalar(1.18);
  const fov = camera.fov * Math.PI / 180;
  const dist = (Math.max(size.y, size.x) * 1.22) / (2 * Math.tan(fov / 2));
  controls.target.copy(c);
  camera.position.set(c.x + dist * 0.10, c.y + size.y * 0.04, c.z + dist);
  controls.minDistance = dist * 0.25; controls.maxDistance = dist * 2.2;
  camera.near = Math.max(0.01, dist * 0.01); camera.far = dist * 12;
  camera.updateProjectionMatrix(); controls.update();
}

/* A narration beat names an organ; this points the eye at it. Reuses the same
   arrival-flash channel the drug already uses, so a beat cannot introduce a
   visual state the projection could not produce on its own. */
function pulseOrgan(comp) {
  for (const k in organs) {
    const o = organs[k];
    if (o.comp === comp) o.flash = 1.0;
  }
  applyLevels();
}

function setLabels(on) { labelsOn = !!on; updateLabels(); }
function setLabelFormat(fn) { labelFmt = fn; }

/* ------------------------------------------------------------------ */
/* Adverse-event focus                                                */
/* ------------------------------------------------------------------ */
/* Selecting a side effect puts the body into a second, warm channel: the organ
   the event lands in is ringed in its SEVERITY colour and breathes, while the
   drug channel keeps doing exactly what it was doing. The two never share a
   colour, so "this organ is well perfused with antibiotic" and "this organ is
   what the antibiotic threatens" can be true at once and still be told apart.

   The severity ramp is the console's own (amber → orange → red), because an
   adverse event IS the clinical-risk vocabulary the rest of the OS already
   speaks — this is the one place in this scene where that vocabulary belongs. */
const SEVERITY_COLOR = {
  "mild": 0x848a93, "moderate": 0xe2a33c,
  "severe": 0xde833a, "life-threatening": 0xe5544c,
};
// An adverse event names a compartment; the atlas may draw it with more than one
// mesh (two kidneys) or with none at all (skin and muscle live on the shell).
const MESH_FOR = {
  lung: ["lung"], heart: ["heart"], liver: ["liver"], spleen: ["spleen"],
  kidney: ["kidneyL", "kidneyR"], pancreas: ["pancreas"],
  intestine: ["intestine"], brain: ["cord"], vessels: ["vessels"], shell: [],
};
let riskFocus = null;          // { meshes:[], color, shell:bool }

function setRiskFocus(meshName, severity) {
  clearRiskFocus();
  if (!meshName) return;
  const keys = MESH_FOR[meshName] || [];
  const color = new THREE.Color(SEVERITY_COLOR[severity] || 0xe2a33c);
  riskFocus = { meshes: keys.filter((k) => organs[k]), color,
                shell: meshName === "shell" || keys.length === 0 };
  riskFocus.meshes.forEach((k) => {
    const o = organs[k];
    if (!o.risk) return;
    o.risk.visible = true;
    o.riskMat.color.copy(color);
  });
  applyLevels();
}

function clearRiskFocus() {
  for (const k in organs) {
    const o = organs[k];
    if (o.risk) { o.risk.visible = false; o.riskMat.opacity = 0; }
  }
  riskFocus = null;
  applyLevels();          // hands the shell back to the drug channel
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
  camera.position.set(0.18, H * 0.62, H * 0.95);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio)); renderer.setSize(w, h);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 0.92;
  el.querySelectorAll("canvas").forEach((c) => c.remove());
  el.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.06; controls.enablePan = false;
  controls.target.set(0, H * 0.62, 0);
  controls.autoRotate = true; controls.autoRotateSpeed = 0.22;
  controls.minPolarAngle = Math.PI * 0.30; controls.maxPolarAngle = Math.PI * 0.62;

  // Darker key than the clinical twin: the drug layer is additive, so any
  // ambient lift on the tissue eats the contrast the concentration needs.
  scene.add(new THREE.AmbientLight(0xffffff, 0.26));
  const key = new THREE.DirectionalLight(0xffffff, 0.75); key.position.set(0.6, 1.4, 1.2); scene.add(key);
  const rim = new THREE.DirectionalLight(0xc8ced6, 0.55); rim.position.set(0, 0.5, -1.5); scene.add(rim);
  scene.add(new THREE.HemisphereLight(0x8a9099, 0x0c0e11, 0.30));

  composer = new EffectComposer(renderer);
  composer.addPass(new RenderPass(scene, camera));
  bloomPass = new UnrealBloomPass(new THREE.Vector2(w, h), 0.26, 0.50, 0.86);
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

function setArm(i) { selectArm(i); updateLabels(); }
function setTime(h) {
  if (!arm) return;
  tHours = Math.max(0, Math.min(arm.horizon_h, +h || 0));
  applyLevels();
  // Refresh the labels HERE rather than waiting for the animation tick. A scrub
  // is a discrete jump: the body repaints on the next frame either way, but the
  // numbers on it must not be one tick behind the numbers beside it, and on a
  // paused twin there may not BE a next tick for a while.
  updateLabels();
}
function play() { playing = true; }
function pause() { playing = false; }
function setRate(r) { playRate = Math.max(0.05, +r || 1); }

let fitMode = "atlas";
function resize() {
  if (!renderer || !container) return;
  const w = container.clientWidth, h = container.clientHeight;
  if (!w || !h) return;
  const changed = Math.abs(w - lastW) > 1 || Math.abs(h - lastH) > 1;
  lastW = w; lastH = h;
  camera.aspect = w / h; camera.updateProjectionMatrix();
  renderer.setSize(w, h);
  if (composer) composer.setSize(w, h);
  if (bloomPass) bloomPass.setSize(w, h);
  // Re-apply the framing whenever the box actually changes. The overlay is
  // measured the instant it becomes visible, before layout has settled, so the
  // first mount fits a camera to a stage that is about to get wider — and the
  // body ends up half out of frame with no way back except a manual drag.
  if (changed) { if (fitMode === "body") fitToBody(); else fitToAtlas(); }
  updateLabels();
}
let lastW = 0, lastH = 0;

function animate() {
  if (disposed) return;
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  if (!renderer || !container || !container.offsetParent) return;   // overlay closed
  controls.update();
  // Scroll the drug stripes along every vessel. Arterial flow runs from the heart
  // outward and venous flow returns, so the two scroll in OPPOSITE directions —
  // which is what makes the picture read as a circuit rather than as two pipes
  // pointing the same way. Speed rises a little with concentration, so a loaded
  // vessel looks like it is moving more, not merely glowing more.
  if (vesselTree) {
    for (let i = 0; i < vesselTree.segs.length; i++) {
      const sg = vesselTree.segs[i];
      if (!sg.flow || !sg.flow.visible) continue;
      const speed = 0.16 + 0.34 * (sg.level || 0);
      sg.tex.offset.x += dt * speed * (sg.kind === "a" ? -1 : 1);
    }
  }
  // The selected adverse event breathes. A static ring gets read as part of the
  // anatomy within a few seconds; a slow pulse stays legible as an ALERT without
  // competing with the drug channel, which never moves on its own.
  if (riskFocus) {
    riskPhase += dt;
    const p = 0.5 - 0.5 * Math.cos(riskPhase * 2.4);
    riskFocus.meshes.forEach((k) => {
      const o = organs[k];
      if (o && o.riskMat) o.riskMat.opacity = 0.16 + 0.30 * p;
    });
    if (riskFocus.shell && shellMat) {
      // Skin, muscle and fat have no mesh of their own, so a risk that lands
      // there is shown on the body shell — which is anatomically what it is.
      shellMat.color.lerp(riskFocus.color, 0.10);
      shellMat.opacity = 0.075 + 0.10 * p;
    }
  }
  // Arrival flashes decay on wall-clock time, not model time, so they read the
  // same whether the scrub is at 1x or 3x.
  let flashing = false;
  for (const k in organs) {
    const o = organs[k];
    if (o.flash > 0) { o.flash = Math.max(0, o.flash - dt * 1.6); flashing = true; }
  }
  if (flashing && !playing) applyLevels();
  labelTick += dt;
  if (labelTick > 0.06) { labelTick = 0; updateLabels(); }
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
  setLabels, setLabelFormat, fitToAtlas, fitToBody, setRiskFocus, clearRiskFocus,
  pulseOrgan,
  hasVessels: () => !!vesselTree, VESSEL_SOURCE,
  state, levelsNow, isPlaying: () => playing, currentArm: () => arm,
  DRUG_COLOR, UNMODELLED: UNMODELLED_LABEL,
};
