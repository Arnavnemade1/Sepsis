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

   The vascular-tree mesh is intentionally absent — see ORGAN_FILES for the
   measurement that removed it. The `vascular` code path is kept: it is correct, and
   it costs nothing while no entry sets the flag.

   Public API (unchanged):  window.AegisTwin = { start, update, resize, setOptical }
   Auto-mounts into #twin-stage. Spin / zoom via OrbitControls.
   ============================================================ */
import * as THREE from "three";
import { OrbitControls } from "./vendor/three/OrbitControls.js";
import { GLTFLoader } from "./vendor/three/GLTFLoader.js";
// The render pipeline and the surface materials live in their own modules now.
// This file owns the anatomy, the animation and the clinical colour rules; how
// any of it is SHADED is a separate concern with its own defects to answer for
// (the composer was discarding multisampling, and there was no environment map).
import { createPipeline, createEnvironment, BLOOM_LAYER } from "./twin3d_pipeline.js";
import { makeOrganMaterial, makeShellMaterial } from "./twin3d_materials.js";
import { buildBody } from "./twin3d_body.js";
import { buildVessels, makeVesselMaterial } from "./twin3d_vessels.js";
let pipeline = null;

const H = 1.7;                                    // body height (feet ~0)
// Clinical-instrument palette — mirrors the --low/--med/--dys/--high tokens in os.css.
// A HEALTHY organ is neutral slate, not blue: rendering "calm" in saturated blue lit the
// whole torso up whenever nothing was wrong, so a genuinely stressed organ had to
// out-shout eleven glowing healthy ones. Colour on this body now means pathology.
const BASE = new THREE.Color(0x848a93);           // calm / healthy — neutral
const NODATA = new THREE.Color(0x4b4f56);
const RIM = new THREE.Color(0xc2c7cf);
const TIER_HEX = { STRESS: 0xe2a33c, DYSFUNCTION: 0xde833a, CRITICAL: 0xe5544c };
const REGION_LABEL = {
  brain: "Neurological", heart: "Cardiovascular", lungs: "Respiratory",
  liver: "Hepatic", kidneys: "Renal", skin: "Peripheral", blood: "Coagulation",
  metabolic: "Metabolic", immune: "Immune", vessels: "Circulatory",
};

/* Real anatomical organ atlas — HuBMAP CCF v1.2 reference meshes (Visible Human,
   CC BY 4.0, see vendor/models/organs/ATTRIBUTION.md). All six are co-registered
   in one coordinate space, so loading them together assembles a correct torso;
   we then scale/place the whole assembly inside the body shell.
   region = which body_state region drives this organ's colour. */
/* ANATOMICAL COLOUR — what each organ actually looks like.
   The twin used to render every organ the same neutral slate and let severity
   supply all the colour, which made a healthy body a uniform grey smear with no
   way to tell a liver from a spleen. These are the atlas colours; they are
   CONSTANT, so they carry no clinical signal, and every one of them is a
   desaturated earth tone. That is what keeps the rule intact: anatomy is fixed
   and muted, pathology is the only saturated, only CHANGING colour in frame. */
const TINT = {
  heart:     0xa04036,   // myocardium, deep red-brown
  lung:      0xb58186,   // dusky pink-grey
  liver:     0x7d3529,   // maroon
  spleen:    0x6e3040,   // purple-red
  kidney:    0x944735,   // red-brown
  pancreas:  0xc6a46d,   // tan
  intestine: 0xc28d6b,   // pink-tan
  bladder:   0xcbbc8a,   // pale yellow
  cord:      0xdad3c1,   // cream
  thymus:    0xd4a493,   // pale pink-tan
  brain:     0xc2b7ad,   // pale grey
};

const ORGAN_FILES = [
  { key: "lung",    file: "VH_M_Lung.glb",       region: "lungs",   breathe: true , tint: TINT.lung},
  { key: "heart",   file: "VH_M_Heart.glb",      region: "heart",   beat: true , tint: TINT.heart},
  { key: "liver",   file: "VH_M_Liver.glb",      region: "liver" , tint: TINT.liver},
  { key: "spleen",  file: "VH_M_Spleen.glb",     region: "blood" , tint: TINT.spleen},
  { key: "kidneyL", file: "VH_M_Kidney_L.glb",   region: "kidneys" , tint: TINT.kidney},
  { key: "kidneyR", file: "VH_M_Kidney_R.glb",   region: "kidneys" , tint: TINT.kidney},
  // expanded coverage — more organ systems light up on the same co-registered atlas
  { key: "pancreas",  file: "VH_M_Pancreas.glb",        region: "metabolic" , tint: TINT.pancreas},
  { key: "intestine", file: "VH_M_Small_Intestine.glb", region: "metabolic", calExclude: true , tint: TINT.intestine},
  { key: "bladder",   file: "VH_M_Urinary_Bladder.glb", region: "kidneys",   calExclude: true , tint: TINT.bladder},
  { key: "cord",      file: "VH_M_Spinal_Cord.glb",     region: "brain",     calExclude: true , tint: TINT.cord},
  { key: "thymus",    file: "VH_M_Thymus.glb",          region: "immune" , tint: TINT.thymus},
  // VH_M_Blood_Vasculature.glb is DELIBERATELY NOT LOADED. It was, and it was
  // described here as "a systemic vessel network" reaching the limbs. It is not:
  // rendered with a magenta probe at full opacity it resolves to a small cluster in
  // the pelvis — the CCF reference mesh covers abdominal/pelvic vessels, not a
  // whole-body tree. At its shipped settings (0.16 additive over dark grey) nothing
  // was visible at all, so 7.1 MiB — 27% of the twin's entire 25.9 MiB asset
  // payload, and a large share of the ~25s before the body appears — was downloaded
  // on every page load to draw nothing anyone could see.
  //
  // The file is still in vendor/; restoring the line below re-enables it. Doing so
  // is only worth it with a mesh that actually spans the body.
  // { key: "vessels", file: "VH_M_Blood_Vasculature.glb", region: "vessels", vascular: true, defer: true },
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
let modelGroup, atlasGroup, organs = {}, organLoadCount = 0, brainStub, bodyMetrics = null;
let focusFoci = [], focusRegions = {};       // the attending's per-patient focus (risk factor)
let ground, particles, pData, reticle;
let beacon, beaconLabel, wrist, wristBone, glowTex;
let vesselGroup, arterialMesh, venousMesh;
let infectFX, infLabel, lastInfection = null;     // suspected infection focus
const INFECT_COL = 0xde833a;                       // distinct "infection" orange (= --dys)
// suspected-focus region -> organ to light up (null = no single organ / systemic)
const FOCUS_ORGAN = { lungs: "lung", respiratory: "lung", kidneys: "kidneyL", renal: "kidneyL", liver: "liver", hepatic: "liver" };
const FOCUS_AT = { lungs: "lung", respiratory: "lung", kidneys: "kidneyL", renal: "kidneyL", liver: "liver", hepatic: "liver", systemic: "heart", bloodstream: "heart", core: "heart", heart: "heart" };

/* ---------------- Holographic Treatment Infusion ---------------- */
let infusionParticles = null;
let infusionActive = false;
let infusionProgress = 0.0;
const infusionColor = new THREE.Color(0x7aa2f7);   // = --accent (an action, not a state)
// The venous route (wrist → elbow → shoulder → SVC → heart) and the organ
// destinations are REBUILT from live world positions on each triggerInfusion()
// call — the organ atlas is skeleton-calibrated at runtime, so any hardcoded
// coords drift off the body. These literals are only a pre-load fallback.
let infusionPath = [
  new THREE.Vector3(0.20, 1.10, -0.05), // Arm entry (wrist)
  new THREE.Vector3(0.14, 1.18, -0.04), // Elbow
  new THREE.Vector3(0.10, 1.31, -0.02), // Shoulder
  new THREE.Vector3(0.04, 1.26, -0.02), // Superior Vena Cava
  new THREE.Vector3(0.011, 1.212, -0.02) // Heart
];
let organDestinations = {
  lung: new THREE.Vector3(0.002, 1.256, -0.043),
  liver: new THREE.Vector3(-0.026, 1.141, -0.038),
  kidneyL: new THREE.Vector3(0.046, 1.099, -0.063),
  kidneyR: new THREE.Vector3(-0.043, 1.087, -0.054),
  spleen: new THREE.Vector3(0.052, 1.142, -0.069)
};
const organTargetKeys = ["lung", "liver", "kidneyL", "kidneyR", "spleen"];


/* ---------------- click-to-inspect ---------------- */
const ORGAN_NAME = { heart: "HEART", lung: "LUNGS", liver: "LIVER", spleen: "SPLEEN", kidneyL: "KIDNEY", kidneyR: "KIDNEY" };
const STATUS_TXT = { NORMAL: "nominal", STRESS: "stress", DYSFUNCTION: "dysfunction", CRITICAL: "critical" };
// the live vital(s) that describe each organ in the inspect readout
const ORGAN_READ = {
  heart:  (d) => "HR " + iv(d, "heart_rate") + " · MAP " + iv(d, "map"),
  lung:   (d) => "RR " + iv(d, "respiratory_rate") + " · SpO₂ " + iv(d, "spo2"),
  liver:  (d) => "lactate " + iv(d, "lactate", 1),
  spleen: (d) => "WBC " + iv(d, "wbc", 1),
  kidneyL:(d) => "MAP " + iv(d, "map"),
  kidneyR:(d) => "MAP " + iv(d, "map"),
};
let raycaster = null, pointer = null, inspectKey = null, inspectLabel = null, downXY = null;
function iv(d, key, dec) {
  const v = readVital(d, key, null);
  if (v == null || isNaN(v)) return "—";
  return dec ? (+v).toFixed(dec) : String(Math.round(v));
}
let shellU = null;                                // shell shader uniforms
let started = false, pendingData = null;
let hr = 72, resp = 15, overall = 0, tNow = 0;

/* ---------------- helpers ---------------- */
/* Scene backdrop. The body used to stand in flat black, so its dark legs and the
   panel behind them were the same value and the silhouette simply stopped at the
   hips. This is a vertical gradient with a soft lift behind the figure — no hue
   beyond the interface accent's faintest trace at the floor, because a coloured
   backdrop would tint every organ colour judged against it.

   It lives in the SCENE and not in CSS: the renderer draws through an
   EffectComposer whose targets are opaque, so anything painted behind the canvas is
   covered by the composed frame. (Verified — a 45% red CSS gradient was invisible.) */
function backdropTexture() {
  const c = document.createElement("canvas");
  c.width = 2; c.height = 256;                       // 1-D gradient, stretched by three
  const x = c.getContext("2d");
  const g = x.createLinearGradient(0, 0, 0, 256);
  g.addColorStop(0.00, "#07080a");                   // top: near-black, keeps the head crisp
  g.addColorStop(0.42, "#101319");
  g.addColorStop(0.72, "#161a22");                   // horizon behind the torso
  g.addColorStop(0.93, "#0d1016");
  g.addColorStop(1.00, "#08090c");                   // floor, darker again
  x.fillStyle = g; x.fillRect(0, 0, 2, 256);
  const tex = new THREE.CanvasTexture(c);
  tex.colorSpace = THREE.SRGBColorSpace;
  tex.magFilter = THREE.LinearFilter; tex.minFilter = THREE.LinearFilter;
  return tex;
}

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
  return new THREE.Color(TIER_HEX[region.status] || 0xe2a33c);
}
const worse = (a, b) => (((a && a.severity) || 0) >= ((b && b.severity) || 0) ? a : b);

/* ---------------- body shell material ----------------
   Moved to twin3d_materials.js. Kept as a thin wrapper so the call site reads
   the same and so `shellU` — which the scan band, the reticle and the heartbeat
   all drive — still resolves to the live uniform block. */
function bodyMaterial() {
  const m = makeShellMaterial(ZONE, H);
  shellU = m.userData.aegis;
  shellMat = m;
  return m;
}
let shellMat = null;

/* ---------------- internal organs (real anatomy) ----------------
   Holographic organ material: dark surface lit only by a settable emissive
   colour + a fresnel rim, so each organ glows its system's tier colour and the
   edges read crisply against the translucent shell. */
function makeOrganMat(tint) { return makeOrganMaterial(tint); }

/* Vascular tree material — an additive glowing network that always reads as a
   faint circulatory web and floods with the tier colour under circulatory stress.
   MeshBasic so it self-illuminates (and blooms) regardless of scene lighting. */
function makeVascularMat() {
  return new THREE.MeshBasicMaterial({
    color: new THREE.Color(0x8f97a3), transparent: true, opacity: 0.34,
    blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide });
}
function loadOrgans() {
  atlasGroup = new THREE.Group(); scene.add(atlasGroup);
  const loader = new GLTFLoader();
  const N_CAL = ORGAN_FILES.filter((c) => !c.defer).length;   // vessels load lazily, don't gate calibration
  ORGAN_FILES.forEach((cfg) => {
    const load = () => loader.load("./vendor/models/organs/" + cfg.file, (g) => {
      const root = g.scene, vascular = !!cfg.vascular,
            mat = vascular ? makeVascularMat() : makeOrganMat(cfg.tint);
      // Organs are the things that are SUPPOSED to glow, so they go on the bloom
      // layer. Bloom used to be global and thresholded, which meant the body shell
      // — a large, softly lit surface — bloomed too, laying a wide low-frequency
      // haze directly over the anatomy and costing it every edge it had.
      root.traverse((o) => {
        if (o.isMesh) {
          o.material = mat; o.frustumCulled = false;
          o.castShadow = o.receiveShadow = false;
          o.layers.enable(BLOOM_LAYER);
        }
      });
      // wrap in a pivot at the organ's own centre so beat/breathe scale about it
      const box = new THREE.Box3().setFromObject(root), c = box.getCenter(new THREE.Vector3());
      const pivot = new THREE.Group(); pivot.position.copy(c); root.position.sub(c); pivot.add(root);
      atlasGroup.add(pivot);
      let halo = null;
      if (!vascular) {   // a soft additive glow halo so a stressed organ reads as "hot"
        halo = new THREE.Sprite(new THREE.SpriteMaterial({ map: glowTex, color: BASE.clone(),
          transparent: true, opacity: 0, depthTest: false, depthWrite: false, blending: THREE.AdditiveBlending }));
        halo.scale.set(0.16, 0.16, 1); halo.visible = false; scene.add(halo);
      }
      organs[cfg.key] = { cfg, pivot, mat, region: cfg.region, sev: 0, mon: false, halo, flash: 0, vascular };
      if (started) applyRegions();
      if (!cfg.defer && ++organLoadCount === N_CAL) calibrateAtlas();
    }, undefined, (e) => console.error("[twin] organ load failed", cfg.file, e));
    // defer the heavy vascular mesh until the core organs are up (keeps first paint fast)
    if (cfg.defer) setTimeout(load, 1400); else load();
  });
  // Dim placeholder for the (unmonitored) brain, so the head is not empty.
  //
  // It was a WIREFRAME icosahedron, which inside a translucent head rendered as a
  // geodesic ball floating in the skull — the single most artificial thing on the
  // body, and nothing like the smooth solids every other organ is. Solid now, and
  // dark: this system carries no data, and a no-data organ must never look like a
  // confident reading.
  const bg = new THREE.IcosahedronGeometry(0.070, 3);
  brainStub = new THREE.Mesh(bg, makeOrganMaterial(TINT.brain));
  // NO DATA, so it must not read as a reading. The wireframe version was
  // obviously artificial; the first solid replacement went the other way — put
  // on the bloom layer at 0.30 emissive it became the brightest object on the
  // whole body, a glowing ball in the skull for a system this cohort does not
  // monitor at all. Dim, and deliberately NOT on the bloom layer: on this body
  // a glow means a measurement.
  // NO DATA, handled the same way every unmonitored organ is: keep it
  // recognisable, drain it. Emissive stays black — on this body a glow means a
  // measurement, and the neurological system has none.
  {
    const hsl = {}; new THREE.Color(TINT.brain).getHSL(hsl);
    brainStub.material.color.setHSL(hsl.h, hsl.s * 0.22, hsl.l * 0.42);
    brainStub.material.emissive.setRGB(0, 0, 0);
    brainStub.material.envMapIntensity = 0.55;
  }
  brainStub.position.set(0, 1.585, -0.008);
  brainStub.scale.set(1.0, 1.06, 1.14);
  scene.add(brainStub);
}
/* Measure the loaded body's torso from its skeleton + mesh box, so the organ
   atlas can be seated INSIDE the real thorax/abdomen rather than at a guessed
   height. Returns torso Y span, lateral/depth centre and front-back thickness. */
function computeBodyMetrics(root) {
  // Exact, because the body generator placed these. The previous version read
  // them off a rig by regex and silently substituted hand-tuned constants when
  // a bone name did not match, so the organ atlas could be seated against
  // numbers that had nothing to do with the mesh on screen.
  const box = new THREE.Box3().setFromObject(root);
  const c = box.getCenter(new THREE.Vector3()), sz = box.getSize(new THREE.Vector3());
  if (bodyAnchors) {
    return {
      lo: bodyAnchors.torsoLo,      // pelvis floor of the torso
      hi: bodyAnchors.torsoHi,      // shoulders / base of neck
      cx: c.x, cz: c.z, depth: sz.z,
    };
  }
  return { lo: H * 0.50, hi: H * 0.82, cx: c.x, cz: c.z, depth: sz.z };
}

/* Assemble + seat the co-registered organ atlas inside the body. When the body is
   loaded we anchor to its real torso (skeleton-derived); otherwise we fall back to
   the hand-tuned constants. */
function calibrateAtlas() {
  // Scale/seat the atlas off the CORE thoraco-abdominal cluster only (heart, lungs,
  // liver, spleen, kidneys, pancreas, thymus). The spinal cord + pelvic organs +
  // vascular tree are co-registered in the SAME frame, so they inherit the
  // transform and land in their true anatomical positions — but they must NOT
  // enter the bounds, or the long cord/pelvis would shrink the whole atlas.
  const core = Object.values(organs).filter((o) => o.pivot && !o.vascular && !o.cfg.calExclude);
  if (!atlasGroup || core.length < 6) return;
  atlasGroup.scale.set(1, 1, 1); atlasGroup.position.set(0, 0, 0); atlasGroup.updateWorldMatrix(true, true);
  const box = new THREE.Box3(); core.forEach((o) => box.expandByObject(o.pivot));
  const c = box.getCenter(new THREE.Vector3()), size = box.getSize(new THREE.Vector3());
  let span = ORGAN_SPAN, ty = TORSO_Y, tz = TORSO_Z, cx = 0;
  if (bodyMetrics) {
    const torsoH = Math.max(0.1, bodyMetrics.hi - bodyMetrics.lo);
    span = torsoH * 0.78;                          // organs fill most of the torso height
    ty = bodyMetrics.lo + torsoH * 0.56;           // chest-biased (heart/lungs sit high)
    tz = bodyMetrics.cz + bodyMetrics.depth * 0.06; // a touch forward of the spine, well inside
    cx = bodyMetrics.cx;
  }
  const s = span / (size.y || 1);
  atlasGroup.scale.set(ORGAN_FLIP * s, s, s);
  atlasGroup.position.set(cx - ORGAN_FLIP * s * c.x, ty - s * c.y, tz - s * c.z);
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
    new THREE.MeshBasicMaterial({ color: 0x101113, transparent: true, opacity: 0.35, depthWrite: false }));
  disc.rotation.x = -Math.PI / 2; disc.position.y = -0.004; ground.add(disc);
  [0.18, 0.30, 0.42, 0.52].forEach((r, i) => {
    const ring = new THREE.LineLoop(circle(r, 80),
      new THREE.LineBasicMaterial({ color: 0x5b6068, transparent: true, opacity: i === 3 ? 0.5 : 0.22 }));
    ground.add(ring);
  });
  const spokes = new THREE.Group();
  for (let i = 0; i < 16; i++) {
    const a = i / 16 * Math.PI * 2;
    const g = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(Math.cos(a) * 0.18, 0, Math.sin(a) * 0.18),
      new THREE.Vector3(Math.cos(a) * 0.52, 0, Math.sin(a) * 0.52)]);
    spokes.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x44484f, transparent: true, opacity: 0.28 })));
  }
  ground.add(spokes);
  // rotating sweep wedge (two bright radials)
  const sweep = new THREE.Group(); sweep.name = "sweep";
  [0, 0.12].forEach((off) => {
    const a = off;
    const g = new THREE.BufferGeometry().setFromPoints([
      new THREE.Vector3(0, 0, 0), new THREE.Vector3(Math.cos(a) * 0.52, 0, Math.sin(a) * 0.52)]);
    sweep.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x9aa0a8, transparent: true, opacity: 0.45 })));
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
    size: 0.016, map: glowTex, color: 0x7d838b, transparent: true, opacity: 0.5,
    depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true }));
  pData = { spd, N }; scene.add(particles);
}

function buildInfusionParticles() {
  const N = 40;
  const pos = new Float32Array(N * 3);
  for (let i = 0; i < N * 3; i++) pos[i] = 999.0;
  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  infusionParticles = new THREE.Points(geo, new THREE.PointsMaterial({
    size: 0.025,
    map: glowTex,
    color: 0x7aa2f7,
    transparent: true,
    opacity: 0.0,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
    sizeAttenuation: true
  }));
  scene.add(infusionParticles);
}

function triggerInfusion(drugType) {
  infusionActive = true;
  infusionProgress = 0.0;
  // Every infusion renders in --accent, because an infusion is an ACTION the system is
  // taking, not a state the patient is in. The old per-drug hues broke that: antibiotic
  // was neon green — the same signal this UI uses for "healthy" — and vasopressor was
  // magenta, which encodes nothing clinical at all. Drug identity is carried by the
  // label in the protocol panel, where it can be read; hue is reserved for state.
  infusionColor.setHex(0x7aa2f7);
  if (infusionParticles) {
    infusionParticles.material.color.copy(infusionColor);
    infusionParticles.material.opacity = 0.9;
  }
  // Rebuild the route from the ACTUAL rendered anatomy so the stream lands on the
  // organs, not 6–8 cm off them. Heart + organ destinations come from each
  // pivot's live world position; the path starts at the wearable wrist anchor.
  const heart = new THREE.Vector3();
  if (organs.heart && organs.heart.pivot) organs.heart.pivot.getWorldPosition(heart);
  else heart.set(0.011, 1.212, -0.02);
  const wristW = new THREE.Vector3();
  if (wristBone) wristBone.getWorldPosition(wristW);
  else wristW.set(heart.x + 0.22, heart.y - 0.12, heart.z);
  const shoulder = new THREE.Vector3(heart.x + 0.11, heart.y + 0.10, heart.z); // out to the side, at clavicle height
  const elbow = new THREE.Vector3().lerpVectors(wristW, shoulder, 0.5);
  const svc = new THREE.Vector3().lerpVectors(shoulder, heart, 0.55);          // superior vena cava, into the heart
  infusionPath = [wristW, elbow, shoulder, svc, heart.clone()];
  organDestinations = {};
  organTargetKeys.forEach((k) => {
    const o = organs[k];
    if (o && o.pivot) { const p = new THREE.Vector3(); o.pivot.getWorldPosition(p); organDestinations[k] = p; }
    else organDestinations[k] = heart.clone();
  });
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
      new THREE.MeshBasicMaterial({ color: 0x9aa0a8, transparent: true, opacity: 0.6, depthWrite: false, blending: THREE.AdditiveBlending }));
    br.rotation.x = Math.PI / 2; br.rotation.z = a; reticle.add(br);
  }
}

/* ---------------- model ---------------- */
function loadBody() {
  // GENERATED, not loaded. This used to pull vendor/models/Xbot.glb — the
  // three.js example ROBOT — which is built as separate armour plates: a chest
  // piece, a pelvis piece, upper and lower arm segments, thigh and calf
  // segments, each a closed mesh with a visible gap at every joint. The twin
  // read as disjointed parts because it WAS disjointed parts, and no material
  // could have fixed that; a robot rendered as glass is a glass robot. It also
  // had no business being here at all: the organs are real anatomy from the
  // Visible Human and the thing housing them was a game character.
  //
  // twin3d_body.js builds the shell as a signed-distance field whose joints are
  // SMOOTH unions, so the surface is continuous by construction, and polygonises
  // it into one watertight mesh.
  //
  // 1.05 cm voxels. Measured across resolutions — holes at 0 throughout, and
  // pinches (an edge shared by more than two triangles, where the surface
  // self-touches inside one cell) only clearing at this step:
  //   1.30 cm  25,880 tris  0 holes  1 pinch
  //   1.15 cm  32,868 tris  0 holes  1 pinch
  //   1.05 cm  39,756 tris  0 holes  0 pinches
  const opts = { resolution: 0.0105 };

  // OFF THE MAIN THREAD. It takes ~1.1 s in a browser, and run inline that froze
  // the console while the twin came up — panels stopped painting, input stopped
  // responding. The fallback is the synchronous path, for anything that cannot
  // construct a module worker.
  let worker = null;
  try {
    worker = new Worker(new URL("./twin3d_body_worker.js", import.meta.url),
                        { type: "module" });
  } catch (err) {
    console.warn("[twin] no module worker; generating the body inline", err);
  }

  // Generate on the main thread. Slow and blocking, which is the whole reason
  // for the worker — but a frozen second is survivable and a missing body is not.
  const inline = (why) => {
    console.warn(`[twin] generating the body inline — ${why}`);
    try { adoptBody(buildBody(opts)); }
    catch (e) { console.error("[twin] body generation failed outright", e); }
  };

  if (worker) {
    let settled = false;
    const done = () => { if (settled) return true; settled = true; worker.terminate(); return false; };
    worker.onmessage = (ev) => {
      if (done()) return;
      // A worker that ran and REPORTED a failure used to log and return, leaving
      // no body at all — the organs would load into empty space and the shell
      // simply never appeared. That is how a ReferenceError inside buildBody
      // showed up as "the anatomy doesn't load" rather than as an error anyone
      // could act on. Any failure now falls through to the inline path.
      if (!ev.data || !ev.data.ok) {
        inline(`the worker reported: ${(ev.data && ev.data.error) || "no result"}`);
        return;
      }
      adoptBody(ev.data);
    };
    worker.onerror = (err) => {
      if (done()) return;
      inline(`the worker errored: ${err.message || err}`);
    };
    // A worker that never answers is a third failure mode, and it leaves exactly
    // the same empty stage as the other two.
    setTimeout(() => { if (!settled) { done(); inline("the worker did not answer in 20 s"); } },
               20000);
    worker.postMessage(opts);
  } else {
    inline("no module worker available");
  }
}

/** Turn the generator's raw arrays into the scene's body. */
function adoptBody(built) {
  console.info(`[twin] body generated: ${built.stats.triangles.toLocaleString()} tris, ` +
               `${built.stats.vertices.toLocaleString()} verts, ${built.stats.ms} ms`);

  const geo = new THREE.BufferGeometry();
  geo.setAttribute("position", new THREE.BufferAttribute(built.positions, 3));
  geo.setAttribute("normal", new THREE.BufferAttribute(built.normals, 3));
  geo.setIndex(new THREE.BufferAttribute(built.indices, 1));
  geo.computeBoundingSphere();

  const mat = bodyMaterial();
  const mesh = new THREE.Mesh(geo, mat);
  mesh.frustumCulled = false;
  mesh.renderOrder = 6;
  mesh.name = "bodyShell";

  modelGroup = new THREE.Group();
  modelGroup.add(mesh);

  // Anchors, placed rather than searched for. The old code hunted a third-party
  // rig for bones matching /hips|pelvis/ and /lefthand|leftwrist/ and fell back
  // to hand-tuned constants whenever the search missed.
  const a = built.anchors;
  wristBone = new THREE.Object3D();
  wristBone.position.set(a.wristL.x, a.wristL.y, a.wristL.z);
  modelGroup.add(wristBone);
  bodyAnchors = a;

  scene.add(modelGroup);

  // The body is the mesh the panel is empty without; the organs land after it and
  // are visible arriving, which is fine. This module owns #twin-stage, so it
  // clears its own placeholder rather than announcing to a listener that then has
  // to be kept in sync with it.
  const ld = document.getElementById("twin-loading");
  if (ld) { ld.classList.add("done"); setTimeout(() => ld.remove(), 600); }

  buildVesselTree();
  bodyMetrics = computeBodyMetrics(modelGroup);   // seat the organs in the REAL torso
  calibrateAtlas();                               // (re-runs once organs are loaded)
  fitCamera(); buildWrist();
  if (pendingData) { update(pendingData); pendingData = null; }
}
let bodyAnchors = null;

/* The vascular tree. Generated, not loaded — the CCF's
   VH_M_Blood_Vasculature.glb covers abdominal and pelvic vessels only and
   resolves to a cluster in the pelvis, which is why the atlas never loads it.
   See twin3d_vessels.js for what this is and, more importantly, what it is not. */
function buildVesselTree() {
  try {
    const V = buildVessels(H);
    vesselGroup = new THREE.Group();
    // Arterial: oxygenated, bright, flowing AWAY from the heart.
    arterialMesh = new THREE.Mesh(V.arterial,
      makeVesselMaterial(0xd4453c, { speed: 0.62, dir: 1.0, bright: 1.0 }));
    // Venous: deoxygenated, deeper and cooler, flowing BACK toward it. The
    // direction flip is the cheapest way to make the two trees legible as
    // different things at a glance.
    venousMesh = new THREE.Mesh(V.venous,
      makeVesselMaterial(0x3f6fb5, { speed: 0.40, dir: -1.0, bright: 0.78 }));
    for (const m of [arterialMesh, venousMesh]) {
      m.frustumCulled = false;
      m.renderOrder = 4;              // inside the shell, over the organs
      m.layers.enable(BLOOM_LAYER);   // vessels are emissive; they should bloom
      vesselGroup.add(m);
    }
    scene.add(vesselGroup);
    console.info(`[twin] vasculature: ${V.counts.arterial} arterial + ` +
                 `${V.counts.venous} venous segments`);
  } catch (err) {
    console.error("[twin] vessel generation failed", err);
  }
}

/* normalizeBySkeleton() and poseArms() lived here. Both existed only to wrestle
   the Xbot rig into shape — scaling the body by its bone bounds and rotating the
   arm bones down out of a T-pose. The generated shell is built at stature with
   its arms already where they belong, so neither has anything to act on. They
   are deleted rather than left dead: a function that traverses for `isBone` in a
   scene with no bones reads as if a skeleton is still involved. */

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
  const c = document.createElement("canvas"); c.width = 512; c.height = 172;   // retina-crisp
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c),
    transparent: true, depthTest: false, depthWrite: false }));
  sp.scale.set(0.47, 0.158, 1); sp.userData = { ctx: c.getContext("2d") }; return sp;
}
/* A sleek targeting tag: glass panel + tier-coloured accent bar, a glowing status
   pip, letter-spaced label, and a small pointer connecting down to the organ. */
function setLabel(sprite, text, stroke, fill) {
  const x = sprite.userData.ctx, W = 512, H = 172; x.clearRect(0, 0, W, H);
  const col = stroke || "#ff4d5e";
  const bx = 34, by = 30, bw = W - 68, bh = 78, r = 16;
  x.save();
  const g = x.createLinearGradient(0, by, 0, by + bh);
  g.addColorStop(0, "rgba(16,22,38,.94)"); g.addColorStop(1, "rgba(8,11,20,.9)");
  x.shadowColor = col; x.shadowBlur = 26; x.fillStyle = g; rr(x, bx, by, bw, bh, r); x.fill();
  x.restore();
  x.strokeStyle = col; x.lineWidth = 2.5; rr(x, bx, by, bw, bh, r); x.stroke();
  // left accent bar (clipped to the rounded panel)
  x.save(); rr(x, bx, by, bw, bh, r); x.clip();
  x.fillStyle = col; x.fillRect(bx, by, 8, bh); x.restore();
  // glowing status pip
  x.save(); x.shadowColor = col; x.shadowBlur = 16; x.fillStyle = col;
  x.beginPath(); x.arc(bx + 40, by + bh / 2, 8, 0, Math.PI * 2); x.fill(); x.restore();
  // label text, letter-spaced small-caps
  x.fillStyle = fill || "#eef4ff";
  x.font = "700 32px ui-monospace,'JetBrains Mono',monospace";
  x.textAlign = "left"; x.textBaseline = "middle";
  if ("letterSpacing" in x) x.letterSpacing = "3px";
  x.fillText(String(text).toUpperCase(), bx + 64, by + bh / 2 + 1);
  if ("letterSpacing" in x) x.letterSpacing = "0px";
  // downward pointer to the organ
  x.fillStyle = col; x.beginPath();
  x.moveTo(W / 2 - 13, by + bh - 1); x.lineTo(W / 2 + 13, by + bh - 1); x.lineTo(W / 2, by + bh + 20);
  x.closePath(); x.fill();
  sprite.material.map.needsUpdate = true;
}
function rr(c, x, y, w, h, r) { c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath(); }

/* ---------------- focus HUD (crisp DOM tag, immune to bloom) ---------------- */
let hudEl = null;
function ensureHud(el) {
  if (hudEl && hudEl.isConnected) return;
  const host = el || document.getElementById("twin-stage");
  if (!host) return;
  if (getComputedStyle(host).position === "static") host.style.position = "relative";
  hudEl = document.createElement("div");
  hudEl.className = "twin-hud"; hudEl.style.display = "none";
  host.appendChild(hudEl);
}
function esc(s) { return String(s == null ? "" : s).replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c])); }
function setHud(kicker, name, hex) {
  ensureHud();
  if (!hudEl) return;
  if (!name) { hudEl.style.display = "none"; return; }
  hudEl.style.display = "flex";
  hudEl.style.setProperty("--tc", hex || "#ff9a3b");
  hudEl.innerHTML =
    '<span class="twin-hud-pip"></span>' +
    '<span class="twin-hud-body"><span class="twin-hud-k">' + esc(kicker || "FOCUS") + '</span>' +
    '<b class="twin-hud-nm">' + esc(name) + '</b></span>';
}

/* ---------------- inspect readout (click an organ) ---------------- */
function buildInspect() {
  const c = document.createElement("canvas"); c.width = 300; c.height = 96;
  inspectLabel = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthTest: false }));
  inspectLabel.scale.set(0.5, 0.16, 1); inspectLabel.visible = false;
  inspectLabel.userData = { ctx: c.getContext("2d"), key: null }; scene.add(inspectLabel);
  raycaster = new THREE.Raycaster(); pointer = new THREE.Vector2();
}
function drawInspect(l1, l2, col) {
  const x = inspectLabel.userData.ctx; x.clearRect(0, 0, 300, 96);
  x.fillStyle = "rgba(8,12,22,.9)"; rr(x, 8, 12, 284, 72, 10); x.fill();
  x.strokeStyle = col; x.lineWidth = 2; rr(x, 8, 12, 284, 72, 10); x.stroke();
  x.fillStyle = col; x.fillRect(8, 12, 4, 72);                  // accent bar
  x.textAlign = "left"; x.textBaseline = "middle";
  x.fillStyle = "#eaf0ff"; x.font = "700 22px ui-monospace,monospace"; x.fillText(l1, 26, 38);
  x.fillStyle = "#9fb0d6"; x.font = "500 15px ui-monospace,monospace"; x.fillText(l2, 26, 66);
  inspectLabel.material.map.needsUpdate = true;
}
function inspectText(key) {
  const o = organs[key]; if (!o) return null;
  const reg = mergedRegions()[o.region];
  const mon = !!(reg && reg.monitored);
  const sys = REGION_LABEL[o.region] || o.region;
  let l2;
  if (mon) {
    const st = STATUS_TXT[reg.status] || String(reg.status || "").toLowerCase();
    const vit = ORGAN_READ[key] ? ORGAN_READ[key](lastServerData || {}) : "";
    l2 = sys + " · " + st + (vit ? " · " + vit : "");
  } else { l2 = sys + " · no data"; }
  return { l1: ORGAN_NAME[key] || key.toUpperCase(), l2: l2,
           col: mon ? "#" + tierColor(reg).getHexString() : "#7c89ad" };
}
function refreshInspect() {
  if (!inspectKey || !inspectLabel) return;
  const t = inspectText(inspectKey); if (!t) return;
  drawInspect(t.l1, t.l2, t.col);
}
function setInspect(key) {
  inspectKey = key;
  if (controls) controls.autoRotate = !key;        // pause the spin while inspecting an organ
  if (!key) { if (inspectLabel) inspectLabel.visible = false; return; }
  inspectLabel.userData.key = key; refreshInspect(); inspectLabel.visible = true;
}
function onDown(e) { downXY = [e.clientX, e.clientY]; }
function onUp(e) {
  if (!downXY) return;
  const moved = Math.hypot(e.clientX - downXY[0], e.clientY - downXY[1]); downXY = null;
  if (moved > 6) return;                                        // a drag, not a click
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((e.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((e.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);
  const meshes = [];
  for (const k in organs) organs[k].pivot && organs[k].pivot.traverse((o) => { if (o.isMesh) { o.userData._ok = k; meshes.push(o); } });
  const hit = raycaster.intersectObjects(meshes, false)[0];
  setInspect(hit ? hit.object.userData._ok : null);
}

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
  container = el; ensureHud(el); const w = el.clientWidth || 480, h = el.clientHeight || 560;
  scene = new THREE.Scene(); scene.fog = new THREE.FogExp2(0x05070d, 0.10);
  scene.background = backdropTexture();
  camera = new THREE.PerspectiveCamera(33, w / h, 0.1, 100);
  camera.position.set(0.2, H * 0.56, H * 1.9);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio)); renderer.setSize(w, h);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  // Left on the renderer, which is correct with a composer attached: three
  // applies NoToneMapping whenever it renders into a render target (see the
  // `_currentRenderTarget === null` guard in WebGLRenderer), so the scene reaches
  // the composer in linear light and OutputPass does the ACES + sRGB conversion
  // once, at the end. Setting this to NoToneMapping would not move tone mapping
  // to the output pass — OutputPass reads this very property — it would turn it
  // off altogether.
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.02;
  el.querySelectorAll("canvas").forEach((c) => c.remove()); el.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.06; controls.enablePan = false;
  controls.target.set(0, H * 0.50, 0);
  controls.autoRotate = true; controls.autoRotateSpeed = 0.28;
  controls.minPolarAngle = Math.PI * 0.38; controls.maxPolarAngle = Math.PI * 0.56;
  // DIRECTIONAL, not ambient. Every lamp was once tinted cool blue, which put a
  // brand cast back onto the tissue after the materials had been neutralised —
  // and a blue key shifts amber toward brown, weakening the exact signal the
  // organ colours exist to send. White key, neutral fill, achromatic hemisphere.
  //
  // The LEVELS changed with the body. A big ambient term is harmless on a
  // segmented robot, whose plate edges carry the form on their own; on one
  // continuous surface it is the thing that flattens it. Ambient and hemisphere
  // are cut hard and the key raised, so curvature — and the clavicles, sternum,
  // costal margin and iliac crests carved into the mesh — actually model.
  scene.add(new THREE.AmbientLight(0xffffff, 0.16));
  const key = new THREE.DirectionalLight(0xffffff, 2.35); key.position.set(0.7, 1.5, 1.15); scene.add(key);
  const fill = new THREE.DirectionalLight(0x9aa0a8, 0.42); fill.position.set(-0.9, 0.5, -0.5); scene.add(fill);
  const rim = new THREE.DirectionalLight(0xdfe3e8, 1.05); rim.position.set(0, 0.5, -1.5); scene.add(rim);   // back-rim → separates the body from the dark
  scene.add(new THREE.HemisphereLight(0x9298a1, 0x101113, 0.22));                                          // soft sky/ground gradient over the tissue

  // ---- image-based lighting ------------------------------------------------
  // Three directional lights and an ambient term are enough to shade a matte
  // solid and nowhere near enough for wet tissue, which is why the organs read
  // as flat glowing blobs. A PMREM-filtered procedural studio gives them a real
  // specular response and a plausible ambient, with no asset to fetch.
  scene.environment = createEnvironment(renderer);
  scene.environmentIntensity = 1.0;

  // ---- post-processing -----------------------------------------------------
  // Selective bloom (only BLOOM_LAYER glows), a multisampled target so the
  // composer stops throwing the renderer's antialiasing away, and a final grade.
  pipeline = createPipeline({
    renderer, scene, camera, width: w, height: h,
    pixelRatio: Math.min(2, devicePixelRatio),
  });
  // Says out loud when the GPU could not give us multisampling or a depth
  // texture, and switches the focus pass off rather than let it blur the whole
  // frame uniformly. Both of those failed silently in the previous pipeline.
  pipeline.selfTest();

  glowTex = glowTexture();
  buildGround(); buildParticles(); buildReticle(); buildInfusionParticles();
  loadBody(); loadOrgans(); buildBeacon(); buildInfection(); buildInspect();

  // Everything that self-illuminates joins the organs on the bloom layer:
  // additive FX are lights by construction, and a light that does not bloom in a
  // scene where lights bloom reads as a decal stuck on the glass. The shell, the
  // ground and the backdrop are deliberately left off it — they are surfaces.
  scene.traverse((o) => {
    if (!o.isMesh && !o.isPoints && !o.isLine) return;
    const m = o.material;
    if (!m) return;
    if (m.blending === THREE.AdditiveBlending || o.isPoints || o.isLine) {
      o.layers.enable(BLOOM_LAYER);
    }
  });
  renderer.domElement.addEventListener("pointerdown", onDown);
  renderer.domElement.addEventListener("pointerup", onUp);
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
    var tr = data.body_state && data.body_state.tracking;      // doctor's focus rides on the snapshot
    if (tr && tr.foci) setFocus(tr.foci);
  }
  if (!started) { pendingData = data; return; }
  applyRegions();
}
function setOptical(reg) { optical = reg || {}; if (started) applyRegions(); }

// The doctor's focus (organ-system names) → the regions those systems live in, so
// the twin can emphasise exactly what the supervising clinician flagged as the risk.
const _SYS_REGIONS = {
  "Cardiovascular": ["heart"], "Autonomic": ["heart"], "Respiratory": ["lungs"],
  "Renal": ["kidneys"], "Hepatic": ["liver"], "Neurological": ["brain"],
  "Immune": ["blood", "spleen", "thymus"], "Coagulation": ["blood"],
  "Metabolic": ["pancreas", "liver"], "Peripheral Perfusion": ["skin"],
};
function setFocus(foci) {
  focusFoci = (foci || []).slice();
  focusRegions = {};
  focusFoci.forEach((s) => (_SYS_REGIONS[s] || []).forEach((r) => (focusRegions[r] = 1)));
  if (started) applyRegions();
}

function applyRegions() {
  const regions = mergedRegions();

  // per-region trajectory: is any MONITORED system in this region trending toward
  // failure? (fed from body_state.organ_systems[].trend / eta_critical_min) — drives
  // the "worsening" alarm pulse so the twin shows an organ HEADING for failure.
  const rising = {}, etaByRegion = {};
  const systems = (lastServerData && lastServerData.body_state && lastServerData.body_state.organ_systems) || [];
  for (const s of systems) {
    if (s.monitored && s.trend === "RISING") {
      rising[s.region] = true;
      if (s.eta_critical_min != null)
        etaByRegion[s.region] = Math.min(etaByRegion[s.region] ?? Infinity, s.eta_critical_min);
    }
  }

  // organs — recolour each real mesh by its system's severity
  overall = 0; let worstKey = null, worstSev = 0.30, worstRegion = null;
  for (const k in organs) {
    const o = organs[k]; if (o.vascular) continue;       // vessels driven by overall, below
    const reg = regions[o.region];
    const mon = !!(reg && reg.monitored);
    const sev = mon ? (reg.severity || 0) : 0;
    o.mon = mon; o.sev = sev; o.worst = false;
    o.rising = !!rising[o.region]; o.eta = etaByRegion[o.region];
    // the attending's focus: emphasise flagged systems, recede everything else
    const anyFocus = focusFoci.length > 0;
    o.focused = !!focusRegions[o.region];
    const dim = anyFocus && !o.focused ? 0.42 : 1.0;         // non-focused organs fade back
    // ANATOMY IN `color`, PATHOLOGY IN `emissive`.
    //
    // A healthy organ emits NOTHING and is lit purely by the studio environment,
    // so it simply looks like a liver. Emissive rises only with severity, which
    // makes a sick organ the one glowing object in an otherwise still frame —
    // rather than the old arrangement, where every organ glowed and a failing
    // one had to out-shout eleven healthy ones.
    const base = new THREE.Color(o.cfg.tint || 0xb08070);
    if (!mon) {
      // NO DATA: keep the organ recognisable, drain the life out of it. A
      // desaturated, darkened liver still reads as a liver and could never be
      // mistaken for a confident healthy reading.
      const hsl = {}; base.getHSL(hsl);
      o.mat.color.setHSL(hsl.h, hsl.s * 0.22, hsl.l * 0.42);
      o.mat.emissive.setRGB(0, 0, 0);
      o.mat.emissiveIntensity = 1.0;
    } else {
      o.mat.color.copy(base);
      if (sev < 0.18 || (reg && reg.status === "NORMAL")) {
        o.mat.emissive.setRGB(0, 0, 0);            // healthy: no glow at all
      } else {
        // Ramp from the tier colour, scaled by severity. The floor keeps a
        // just-crossed threshold visible; the ceiling stops CRITICAL blowing
        // out into a white blob under bloom.
        o.mat.emissive.copy(tierColor(reg))
          .multiplyScalar(Math.min(0.85, 0.16 + 0.80 * sev) * dim);
      }
      o.mat.emissiveIntensity = 1.0;
    }
    o.mat.needsUpdate = false;
    // Focus dimming now works on the LIGHT response, not on opacity: the organs
    // are opaque, which is what gives them edges.
    o.mat.envMapIntensity = 1.55 * (o.focused ? 1.15 : dim);
    if (o.mat.userData.aegis) o.mat.userData.aegis.uRim.value = o.focused ? 1.25 : dim;
    if (o.halo) o.halo.material.color.copy(tierColor(reg));   // glow halo tracks the tier colour
    if (mon) overall = Math.max(overall, sev);
    if (mon && reg.status !== "NORMAL" && sev > worstSev) { worstSev = sev; worstKey = k; worstRegion = reg; }
  }
  for (const k in organs) organs[k].worst = (k === worstKey);

  // Vascular tree, if one is ever loaded again: floods with the overall tier colour
  // and brightens as circulatory stress rises. No ORGAN_FILES entry currently sets
  // `vascular`, so this loop is a no-op — kept because the path is correct and the
  // only thing wrong with it was the mesh it was pointed at.
  const vcol = overall >= 0.75 ? new THREE.Color(TIER_HEX.CRITICAL)
             : overall >= 0.45 ? new THREE.Color(TIER_HEX.DYSFUNCTION)
             : overall >= 0.2 ? new THREE.Color(TIER_HEX.STRESS) : new THREE.Color(0x8f97a3);
  for (const k in organs) {
    const o = organs[k]; if (!o.vascular || !o.mat) continue;
    o.mon = true; o.sev = overall;
    o.mat.color.copy(vcol);
    // The calm floor was 0.16 additive over a dark grey, which is invisible: this
    // block's own comment promised "a faint network when calm" and the 7.1 MiB mesh
    // that delivers it was being downloaded on every load to render nothing anyone
    // could see. It is the only anatomy that reaches the limbs, and the limbs are
    // where peripheral perfusion shows, so it should always be legible.
    o.mat.opacity = 0.34 + 0.34 * overall;
  }

  // Body shell: NEUTRAL, always. It used to wash 40% toward the tier colour from the
  // lowest threshold up (overall >= 0.2), which meant a merely-stressed patient turned
  // the entire mannequin — head, arms, legs, all of it — amber. That is the exact
  // inversion of "colour is signal": a body-wide amber localises nothing, and worse, it
  // removes the neutral ground the ONE genuinely amber organ needs to stand out against,
  // so the more the patient deteriorated the less you could see where. The shell stays
  // slate and the organs carry the diagnosis; only true CRITICAL gets a restrained tint,
  // because at that point the silhouette itself is the message.
  const bc = BASE.clone();
  if (overall >= 0.75) bc.lerp(new THREE.Color(TIER_HEX.CRITICAL), 0.16);
  ZONE.head.copy(bc); ZONE.chest.copy(bc); ZONE.abd.copy(bc); ZONE.pelvis.copy(bc); ZONE.limb.copy(bc);

  // suspected infection focus — light up the source organ + an orange hot zone.
  // The TEXT tag is rendered as a crisp DOM HUD (see setHud) instead of a 3D sprite,
  // so bloom can never wash it out and it sits cleanly in the corner, off the body.
  for (const k in organs) organs[k].infected = false;
  if (infLabel) infLabel.visible = false;
  if (beacon) beacon.visible = false;
  if (lastInfection && lastInfection.site) {
    infectFX.position.copy(focusPos(lastInfection.region)); infectFX.visible = true;
    const fk = FOCUS_ORGAN[lastInfection.region];
    [fk, lastInfection.region === "kidneys" ? "kidneyR" : null].forEach((key) => {
      const o = key && organs[key]; if (!o) return;
      o.infected = true; o.mat.emissive.lerp(new THREE.Color(INFECT_COL), 0.55);
      o.mat.opacity = Math.max(o.mat.opacity, 0.7);
    });
    setHud("SUSPECTED FOCUS", lastInfection.site.toUpperCase(), "#ff8a3b");
  } else {
    if (infectFX) infectFX.visible = false;
    if (worstKey && worstRegion) {
      const nm = (REGION_LABEL[organs[worstKey].region] || worstKey).toUpperCase();
      const hex = { CRITICAL: "#ff4d5e", DYSFUNCTION: "#ff9a3b", STRESS: "#ffd23b" }[worstRegion.status] || "#ff9a3b";
      setHud((worstRegion.status || "WATCH") + (worstRegion.optical ? " · OPTICAL" : ""), nm, hex);
    } else if (focusFoci.length) {                     // stable, but the clinician set a focus
      setHud("CLINICIAN FOCUS", focusFoci.join(" · ").toUpperCase(), "#48c4f2");
    } else {
      setHud(null);
    }
  }

  // particle tint by overall load
  if (particles) particles.material.color.copy(overall > 0.18 ? tierColor({ monitored: true, severity: overall, status: overall >= 0.75 ? "CRITICAL" : overall >= 0.45 ? "DYSFUNCTION" : "STRESS" }) : BASE);

  // wearable wrist marker
  const d = lastServerData || {};
  const live = ["wearable", "simulator", "camera"].includes(d.source) && (d.risk && d.risk.signal_quality != null ? d.risk.signal_quality : 1) > 0.15;
  if (wrist) { wrist.visible = !!wristBone; setWrist(live); }

  refreshInspect();                                 // keep the open organ readout current
}

function resize() {
  if (!renderer || !container) return;
  const w = container.clientWidth, h = container.clientHeight; if (!w || !h) return;
  camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h);
  if (pipeline) pipeline.resize(w, h);
  fitCamera();
}

const _v = new THREE.Vector3();
function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  // Parked = the treatment overlay is covering us. Keep the RAF alive (so we
  // resume instantly) but skip the simulation and the draw entirely.
  if (parked) return;
  tNow += dt; const t = tNow;
  controls.update();

  // heartbeat (sharp) + breathing cycles, from live vitals
  const beat = Math.pow(0.5 + 0.5 * Math.sin(t * (hr / 60) * 6.2832), 6);
  const breathe = 0.5 - 0.5 * Math.cos(t * (resp / 60) * 6.2832);

  // body shell: gentle breathe + shader uniforms
  if (modelGroup && modelGroup.userData.baseScale)
    modelGroup.scale.setScalar(modelGroup.userData.baseScale * (1 + 0.010 * breathe));
  // Vessels: the pulse travels along each tube, and PERFUSION gates brightness.
  // Falling MAP is the twin's most direct statement that a territory is losing
  // its supply, so the tree dims and slows rather than changing colour.
  if (arterialMesh) {
    const map = (lastServerData && lastServerData.vitals && lastServerData.vitals.map);
    const perf = map == null ? 1.0
               : Math.max(0.18, Math.min(1.0, (Number(map) - 40) / 30));
    for (const m of [arterialMesh, venousMesh]) {
      const u = m.material.userData.aegis;
      u.uTime.value = t;
      u.uPerfusion.value += (perf - u.uPerfusion.value) * 0.05;   // ease, no snap
    }
  }
  if (shellU) {
    shellU.uTime.value = t; shellU.uPulse.value = beat;
    shellU.uScan.value = ((t * 0.20) % 1.5) - 0.15;   // rising scan band, then repeat
  }

  // organs: heart beats at HR, lungs breathe at the respiratory rate, the
  // worst-affected organ throbs; emissive brightness tracks the pulse.
  for (const k in organs) {
    const o = organs[k]; if (!o.pivot) continue;
    if (o.vascular) {                                 // vessels: blood-flow shimmer on the heartbeat
      o.mat.opacity = (0.16 + 0.44 * (o.sev || 0)) * (0.82 + 0.34 * beat);
      o.pivot.scale.setScalar(1); continue;
    }
    if (o.cfg.beat) o.pivot.scale.setScalar(1 + 0.11 * beat);
    else if (o.cfg.breathe) o.pivot.scale.set(1 + 0.035 * breathe, 1 + 0.075 * breathe, 1 + 0.035 * breathe);
    else if (o.worst) o.pivot.scale.setScalar(1 + 0.05 * Math.abs(Math.sin(t * 3.0)));
    else o.pivot.scale.setScalar(1);
    const anyFocus = focusFoci.length > 0;
    const fdim = anyFocus && !o.focused ? 0.42 : 1.0;   // recede everything but the focus
    // The alarm pulse. Gated on severity inside the shader, so a healthy atlas
    // sits perfectly still — any movement in the frame is then information.
    if (o.mat.userData.aegis) {
      o.mat.userData.aegis.uAlarm.value =
        o.mon && o.sev > 0.18 ? (0.5 + 0.5 * Math.sin(t * (2.2 + 3.0 * o.sev))) : 0.0;
    }
    if (o.mon) {
      const env = o.cfg.beat ? (0.62 + 0.5 * beat) : (0.85 + 0.15 * Math.sin(t * 1.6 + (o.ph || 0)));
      o.mat.emissiveIntensity = (0.5 + 0.7 * o.sev) * env * fdim;
      // worsening alarm: a fast, sharp over-pulse (sharper + faster the sooner the
      // projected ETA to CRITICAL) so a deteriorating organ visibly throbs.
      if (o.rising) {
        const urg = o.eta != null ? Math.min(1, 15 / Math.max(1, o.eta)) : 0.5;
        const ap = Math.pow(0.5 + 0.5 * Math.sin(t * (3.0 + 5.0 * urg)), 3);
        o.mat.emissiveIntensity *= (1.0 + (0.3 + 0.5 * urg) * ap);
      }
    } else if (anyFocus && !o.focused) {
      o.mat.emissiveIntensity *= fdim;
    }
    // focused organ: a calm, steady breathing glow so the clinician's flagged
    // system is unmistakable even when it is not (yet) the most severe.
    if (o.focused) {
      o.mat.emissiveIntensity = Math.max(o.mat.emissiveIntensity, 0.55) * (1.12 + 0.16 * Math.sin(t * 2.2));
      if (o.halo) { o.halo.visible = true; o.pivot.getWorldPosition(_v); o.halo.position.copy(_v);
        o.halo.material.color.copy(RIM);
        const sc = 0.16 + 0.05 * Math.sin(t * 2.2); o.halo.scale.set(sc, sc, 1);
        o.halo.material.opacity = Math.max(o.halo.material.opacity || 0, 0.32); }
    }
    if (o.infected) {                              // suspected source — throb brighter
      const ip = 0.5 + 0.5 * Math.sin(t * 4.0);
      o.mat.emissiveIntensity = Math.max(o.mat.emissiveIntensity, 0.55) * (1.05 + 0.35 * ip);
      if (!o.cfg.beat && !o.cfg.breathe) o.pivot.scale.setScalar(1 + 0.05 * ip);
    }
    // Never blow out. This ceiling matters more since bloom became selective:
    // the organs are the only things on the bloom layer, so all of the glow
    // budget is theirs and an uncapped CRITICAL organ washes the torso flat.
    o.mat.emissiveIntensity = Math.min(o.mat.emissiveIntensity, 1.15);
    // glow halo: a hot organ blooms through the body, beating with the pulse
    if (o.halo) {
      if (o.mon && o.sev > 0.04) {
        o.halo.visible = true;
        o.pivot.getWorldPosition(_v); o.halo.position.copy(_v);
        const pul = o.cfg.beat ? (0.5 + 0.6 * beat) : (0.8 + 0.2 * Math.sin(t * 1.6 + (o.ph || 0)));
        const sc = (0.13 + 0.24 * o.sev) * (o.worst ? 1.3 : 1) * (0.9 + 0.2 * pul);
        o.halo.scale.set(sc, sc, 1);
        o.halo.material.opacity = (0.12 + 0.55 * o.sev) * pul * (o.worst ? 1.25 : 1);
        if (o.infected) o.halo.material.color.lerp(new THREE.Color(INFECT_COL), 0.5);
      } else o.halo.visible = false;
    }
    // therapeutic-response flash: an infusion pulse decays back to baseline (~0.5s),
    // so unmonitored organs no longer get stuck bright. Blooms the halo in the
    // infusion colour for the drug-arrival "lights up" effect.
    if (o.flash > 0) {
      o.flash = Math.max(0, o.flash - dt * 1.8);
      o.mat.emissiveIntensity = Math.min(1.35, o.mat.emissiveIntensity + o.flash * 1.2);
      if (o.halo) {
        o.halo.visible = true;
        o.pivot.getWorldPosition(_v); o.halo.position.copy(_v);
        o.halo.material.color.lerp(infusionColor, 0.6);
        const sc = 0.16 + 0.22 * o.flash;
        o.halo.scale.set(sc, sc, 1);
        o.halo.material.opacity = Math.max(o.halo.material.opacity, 0.5 * o.flash);
      }
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

  // Treatment infusion flow animation
  if (infusionActive && infusionParticles) {
    infusionProgress += dt * 0.7; // Speed of the flow
    if (infusionProgress > 1.6) {
      infusionActive = false;
      infusionParticles.material.opacity = 0;
      const pos = infusionParticles.geometry.attributes.position;
      for (let i = 0; i < 40; i++) { pos.setXYZ(i, 999, 999, 999); }
      pos.needsUpdate = true;
    } else {
      const pos = infusionParticles.geometry.attributes.position;
      const N = 40;
      for (let i = 0; i < N; i++) {
        const offset = i / N * 0.4;
        const p = infusionProgress - offset;
        if (p < 0) {
          pos.setXYZ(i, 999, 999, 999);
        } else if (p < 1.0) {
          // Arm to heart path
          const segments = infusionPath.length - 1;
          const idx = Math.min(segments - 1, Math.floor(p * segments));
          const localT = (p * segments) - idx;
          const p0 = infusionPath[idx];
          const p1 = infusionPath[idx + 1];
          const inter = new THREE.Vector3().lerpVectors(p0, p1, localT);
          const noise = 0.008;
          pos.setXYZ(i, 
            inter.x + (Math.random() - 0.5) * noise, 
            inter.y + (Math.random() - 0.5) * noise, 
            inter.z + (Math.random() - 0.5) * noise
          );
        } else if (p < 1.5) {
          // Heart to organs branch
          const target = organTargetKeys[i % organTargetKeys.length];
          const dest = organDestinations[target];
          const heart = infusionPath[infusionPath.length - 1];
          const localT = (p - 1.0) / 0.5;
          const inter = new THREE.Vector3().lerpVectors(heart, dest, localT);
          
          if (localT > 0.8 && organs[target]) organs[target].flash = 1.0;  // therapeutic-response flash (decays in animate)
          
          const noise = 0.015;
          pos.setXYZ(i, 
            inter.x + (Math.random() - 0.5) * noise, 
            inter.y + (Math.random() - 0.5) * noise, 
            inter.z + (Math.random() - 0.5) * noise
          );
        } else {
          pos.setXYZ(i, 999, 999, 999);
        }
      }
      pos.needsUpdate = true;
    }
  }

  if (ground) { const sw = ground.getObjectByName("sweep"); if (sw) sw.rotation.y -= dt * 0.7; ground.rotation.y += dt * 0.02; }
  if (reticle && shellU) {                       // ride the scan band up the body
    const s = Math.max(0, Math.min(1, shellU.uScan.value));
    reticle.position.y = s * H * 0.96 + 0.06;
    reticle.rotation.y += dt * (hyperOn ? 0.85 : 0.3);
    const edge = Math.sin(s * Math.PI);          // fade at feet/head
    reticle.children.forEach((c) => {
      if (c.userData.aura) {
        // each shell breathes on its own phase, so the aura pulses outward
        const ph = t * 2.1 - c.userData.i * 0.7;
        c.material.opacity = (0.3 - c.userData.i * 0.08) * edge * (0.55 + 0.45 * Math.sin(ph));
        const k = 1 + 0.06 * Math.sin(ph);
        c.scale.set(k, k, 1);
        return;
      }
      c.material.opacity = (c.geometry.parameters.arc ? 0.6 : 0.55)
        * edge * (hyperOn ? 1.5 : 1);
    });
  }
  _placeFlags();
  if (wrist && wristBone && wrist.visible) { wristBone.getWorldPosition(_v); wrist.position.copy(_v).add(new THREE.Vector3(0, 0.02, 0.04)); }

  // inspect readout tracks the clicked organ
  if (inspectKey && inspectLabel && inspectLabel.visible && organs[inspectKey] && organs[inspectKey].pivot) {
    organs[inspectKey].pivot.getWorldPosition(_v);
    inspectLabel.position.set(_v.x, _v.y + 0.13, _v.z + 0.14);
  }

  if (pipeline) {
    // Park the focus plane on the orbit target, which is the mid-torso — the
    // part of this body every clinically meaningful organ sits in. It has to be
    // updated per frame because the user can orbit and zoom, and a focus plane
    // fixed at start distance would drift off the anatomy the moment they did.
    pipeline.focusOn(controls.target, camera);
    pipeline.render();
  } else renderer.render(scene, camera);
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
/* Lend the parsed anatomy to a SECOND scene (the pharmacokinetic twin).
   The atlas is ~23 MB of GLB and takes 10-14 s to parse; the treatment twin needs
   the same bodies, so it clones these instead of re-parsing them. Object3D.clone()
   shares the underlying BufferGeometry by reference, so the clone costs no parse
   and no extra CPU-side geometry — only the second renderer's own GPU upload. The
   borrower must NOT mutate these objects: it re-materialises every clone. Returns
   null while the atlas is still loading, and the caller falls back to loading the
   files itself (the browser cache makes that a warm fetch, just not a free parse). */
function _lendAnatomy() {
  const core = Object.values(organs).filter((o) => o.pivot && !o.vascular && !o.cfg.calExclude);
  if (!modelGroup || core.length < 6) return null;
  const out = { body: modelGroup, atlas: atlasGroup, metrics: bodyMetrics, organs: {} };
  for (const k in organs) if (organs[k].pivot) out.organs[k] = organs[k].pivot;
  return out;
}
/* The main twin's render loop is pure overhead while the treatment overlay covers
   it — two WebGL contexts drawing 20 MB of anatomy each is a real cost on a
   laptop. The overlay parks it on open and wakes it on close. */
let parked = false;
function setParked(v) { parked = !!v; }

/* ---------------- HYPERSCAN mode ---------------- */
/* The idle sweep is a thin pale ring — deliberately quiet, because it runs all
   the time and must not compete with a stressed organ. A hyperscan is a discrete
   event with a beginning and an end, so it gets a louder instrument: the ring
   thickens, gains concentric aura shells that breathe, and shifts to the
   affordance accent.
   Accent, specifically, and not one of the clinical hues: on this body a
   saturated colour means pathology, and a scan that painted itself amber would
   be claiming something about the patient rather than about itself. */
const HYPER_COLOR = 0x7aa2f7;          // --accent; "the system is working", never "urgent"
let hyperOn = false, auraRings = [];

function setHyperscan(on) {
  hyperOn = !!on;
  if (!reticle) return;
  const base = reticle.children.filter((c) => !c.userData.aura);
  base.forEach((c) => {
    c.material.color.setHex(hyperOn ? HYPER_COLOR : (c.geometry.parameters.arc ? 0x9aa0a8 : 0x8fe9ff));
  });
  if (hyperOn && !auraRings.length) {
    // three shells at increasing radius and decreasing opacity — an aura is a
    // falloff, not a fatter line
    for (let i = 0; i < 3; i++) {
      const a = new THREE.Mesh(
        new THREE.TorusGeometry(0.24 + i * 0.05, 0.0016 + i * 0.001, 6, 90),
        new THREE.MeshBasicMaterial({ color: HYPER_COLOR, transparent: true,
          opacity: 0.3 - i * 0.08, depthWrite: false, blending: THREE.AdditiveBlending }));
      a.rotation.x = Math.PI / 2; a.userData = { aura: true, i };
      auraRings.push(a); reticle.add(a);
    }
  }
  auraRings.forEach((a) => { a.visible = hyperOn; });
}

/* Flags — a finding pinned to the organ it came from, as the sweep discovers it.
   DOM rather than sprites so the text stays crisp through the bloom pass, and
   projected each frame so they track the body as it rotates. */
let flagHost = null, flags = [];
function flag(regionOrKey, label, sub) {
  if (!renderer) return;
  const host = renderer.domElement.parentElement; if (!host) return;
  if (!flagHost) {
    flagHost = document.createElement("div");
    flagHost.className = "twin-flags";
    host.appendChild(flagHost);
  }
  // resolve a body REGION (what the backend speaks) to an organ pivot
  let pivot = (organs[regionOrKey] || {}).pivot;
  if (!pivot) {
    const k = Object.keys(organs).find((x) => organs[x].region === regionOrKey);
    pivot = k ? organs[k].pivot : null;
  }
  const e = document.createElement("div");
  e.className = "twin-flag";
  e.innerHTML = '<i></i><span class="tf-k"></span><span class="tf-v"></span>';
  e.querySelector(".tf-k").textContent = label || "";
  e.querySelector(".tf-v").textContent = sub || "";
  flagHost.appendChild(e);
  requestAnimationFrame(() => e.classList.add("in"));
  flags.push({ el: e, pivot, born: performance.now() });
  return e;
}
function clearFlags() {
  flags.forEach((f) => f.el.remove()); flags = [];
}
function _placeFlags() {
  if (!flags.length || !camera || !renderer) return;
  const w = renderer.domElement.clientWidth, h = renderer.domElement.clientHeight;
  const p = new THREE.Vector3();
  // Several stages hang off the same organ (brain takes history, causal AND
  // consensus), and distinct organs can project to nearly the same pixel from a
  // front-on camera — so anchoring alone stacked the flags into an unreadable
  // pile. Nudge each one down until it clears the ones already placed: the
  // anchor still says which organ it belongs to, the leader line does the rest.
  const placed = [];
  const ROW = 22;
  flags.forEach((f) => {
    if (!f.pivot) { f.el.style.display = "none"; return; }
    f.pivot.getWorldPosition(p); p.project(camera);
    if (p.z > 1) { f.el.style.display = "none"; return; }
    let x = (p.x * 0.5 + 0.5) * w, y = (-p.y * 0.5 + 0.5) * h;
    let guard = 0;
    while (guard++ < 12 && placed.some((q) => Math.abs(q.y - y) < ROW
                                             && Math.abs(q.x - x) < 190)) {
      y += ROW;
    }
    y = Math.max(6, Math.min(h - 26, y));
    placed.push({ x, y });
    f.el.style.display = "";
    f.el.style.transform = "translate(" + x.toFixed(1) + "px," + y.toFixed(1) + "px)";
  });
}

/* The render pipeline, exposed for tuning and for the checks in twin3d_verify.js.
   A shader whose effect is subtle is indistinguishable from a shader that is
   silently doing nothing, and the only way to tell them apart is to be able to
   turn the knob and watch the picture move. */
function _pipeline() { return pipeline; }
function _shell() { return shellMat; }

window.AegisTwin = { start, update, resize, setOptical, setFocus, _setAtlas, _measure,
  _pipeline, _shell,
                     triggerInfusion, _lendAnatomy, setParked,
                     setHyperscan, flag, clearFlags,
                     isParked: () => parked };
const probe = setInterval(() => {
  const el = document.getElementById("twin-stage");
  if (el && el.clientWidth > 40 && el.clientHeight > 40) { clearInterval(probe); start(el); }
}, 250);
