/* ============================================================
   AEGIS-OS — 3D anatomical digital twin (Three.js, vendored)
   A real rigged humanoid mesh rendered as a translucent medical
   hologram. Spin / zoom via OrbitControls. Internal organ glows
   + a 3D threat beacon are driven by BodyState region severity.
   ============================================================ */
import * as THREE from "three";
import { OrbitControls } from "./vendor/three/OrbitControls.js";
import { GLTFLoader } from "./vendor/three/GLTFLoader.js";

const H = 1.7;                         // target body height (feet at y=0)
const TIER_COLOR = { NORMAL: 0x36c6ff, STRESS: 0xffd23b, DYSFUNCTION: 0xff9a3b, CRITICAL: 0xff4d5e, UNMONITORED: 0x39456b };
const TIER_EMIT = { NORMAL: 0.3, STRESS: 0.75, DYSFUNCTION: 1.1, CRITICAL: 1.8, UNMONITORED: 0.06 };
const REGION_LABEL = {
  heart: "Cardiovascular", lungs: "Respiratory", brain: "Neurological",
  liver: "Hepatic", kidneys: "Renal", skin: "Peripheral", blood: "Coagulation",
};
let renderer, scene, camera, controls, container, ro, clock;
let organs = {}, bodyMats = [], beacon, beaconLabel, glowTex;
let started = false, pendingRegions = null, modelReady = false;

/* ---------------- helpers ---------------- */
function glowTexture() {
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const ctx = c.getContext("2d");
  const g = ctx.createRadialGradient(64, 64, 2, 64, 64, 64);
  g.addColorStop(0, "rgba(255,255,255,1)"); g.addColorStop(.3, "rgba(255,255,255,.45)");
  g.addColorStop(1, "rgba(255,255,255,0)");
  ctx.fillStyle = g; ctx.fillRect(0, 0, 128, 128);
  return new THREE.CanvasTexture(c);
}

/* hologram material for the body — MeshStandard (auto-skins) with a fresnel rim
   injected via onBeforeCompile so edges glow like a scan. */
function bodyMaterial() {
  const m = new THREE.MeshStandardMaterial({
    color: 0x113a5c, emissive: new THREE.Color(TIER_COLOR.NORMAL), emissiveIntensity: 0.55,
    transparent: true, opacity: 0.5, roughness: 0.4, metalness: 0.0,
    depthWrite: false, side: THREE.DoubleSide, blending: THREE.AdditiveBlending,
  });
  m.userData.rim = { value: new THREE.Color(TIER_COLOR.NORMAL) };
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uRim = m.userData.rim;
    // inject at the END of the fragment, where `normal` + `vViewPosition` exist
    sh.fragmentShader = "uniform vec3 uRim;\n" + sh.fragmentShader.replace(
      "#include <dithering_fragment>",
      `#include <dithering_fragment>
       float rim = pow(1.0 - clamp(dot(normalize(normal), normalize(vViewPosition)), 0.0, 1.0), 2.0);
       gl_FragColor.rgb += uRim * rim * 1.7;
       gl_FragColor.a = clamp(gl_FragColor.a + rim * 0.5, 0.0, 1.0);`
    );
  };
  bodyMats.push(m);
  return m;
}

function organMesh(geo, glowScale) {
  const mat = new THREE.MeshStandardMaterial({
    color: TIER_COLOR.NORMAL, emissive: new THREE.Color(TIER_COLOR.NORMAL), emissiveIntensity: 0.3,
    transparent: true, opacity: 0.95, roughness: 0.3, metalness: 0.1, depthWrite: false,
  });
  const mesh = new THREE.Mesh(geo, mat);
  const sprite = new THREE.Sprite(new THREE.SpriteMaterial({
    map: glowTex, color: TIER_COLOR.NORMAL, transparent: true,
    blending: THREE.AdditiveBlending, depthWrite: false, depthTest: false, opacity: 0,
  }));
  sprite.scale.setScalar(glowScale); mesh.add(sprite);
  mesh.userData = { mat, sprite };
  return mesh;
}

function wrapGroup(group) {
  group.userData = {
    isGroup: true, position: group.position,
    setTier(col, emit, glow) {
      group.children.forEach(ch => {
        if (!ch.userData.mat) return;
        ch.userData.mat.color.setHex(col); ch.userData.mat.emissive.setHex(col); ch.userData.mat.emissiveIntensity = emit;
        if (ch.userData.sprite) { ch.userData.sprite.material.color.setHex(col); ch.userData.sprite.material.opacity = glow; }
      });
    },
  };
  return group;
}

/* Place organs anchored to the REAL body bounds (robust to mesh scale/pose).
   Specs are normalized: f = height fraction from feet (0..1), x/z = fraction of
   body width/depth from centre, r = radius as fraction of body height. */
function placeOrgans(box) {
  const min = box.min, max = box.max;
  const cx = (min.x + max.x) / 2, cz = (min.z + max.z) / 2;
  const bw = max.x - min.x, bd = max.z - min.z, bh = max.y - min.y;
  const wx = (nx) => cx + nx * bw, wy = (f) => min.y + f * bh, wz = (nz) => cz + nz * bd, R = (r) => r * bh;
  const og = new THREE.Group(); scene.add(og);

  const brain = organMesh(new THREE.SphereGeometry(R(0.036), 22, 18), R(0.42));
  brain.scale.set(1.05, 0.95, 1.1); brain.position.set(wx(0), wy(0.94), wz(0.12)); og.add(brain); organs.brain = brain;

  const heart = organMesh(new THREE.SphereGeometry(R(0.03), 20, 16), R(0.44));
  heart.scale.set(1, 1.2, 0.9); heart.position.set(wx(-0.07), wy(0.71), wz(0.22)); og.add(heart); organs.heart = heart;

  const lungs = new THREE.Group();
  [-1, 1].forEach(s => { const L = organMesh(new THREE.SphereGeometry(R(0.03), 16, 14), R(0.38));
    L.scale.set(0.8, 1.7, 0.7); L.position.set(s * 0.12 * bw, 0, 0); lungs.add(L); });
  lungs.position.set(wx(0), wy(0.745), wz(0.16)); scene.add(lungs); organs.lungs = wrapGroup(lungs);

  const liver = organMesh(new THREE.SphereGeometry(R(0.035), 16, 12), R(0.42));
  liver.scale.set(1.7, 0.72, 0.85); liver.position.set(wx(0.1), wy(0.64), wz(0.18)); og.add(liver); organs.liver = liver;

  const kidneys = new THREE.Group();
  [-1, 1].forEach(s => { const K = organMesh(new THREE.SphereGeometry(R(0.016), 14, 12), R(0.26));
    K.scale.set(0.8, 1.5, 0.7); K.position.set(s * 0.1 * bw, 0, 0); kidneys.add(K); });
  kidneys.position.set(wx(0), wy(0.56), wz(-0.12)); scene.add(kidneys); organs.kidneys = wrapGroup(kidneys);

  const curve = new THREE.CatmullRomCurve3([new THREE.Vector3(wx(0), wy(0.82), wz(-0.1)), new THREE.Vector3(wx(0), wy(0.5), wz(-0.1))]);
  const blood = new THREE.Mesh(new THREE.TubeGeometry(curve, 8, R(0.007), 8),
    new THREE.MeshStandardMaterial({ color: TIER_COLOR.NORMAL, emissive: new THREE.Color(TIER_COLOR.NORMAL), emissiveIntensity: 0.3, transparent: true, opacity: 0.75, depthWrite: false }));
  blood.userData = { mat: blood.material, sprite: null }; scene.add(blood); organs.blood = blood;

  beacon && (beacon.userData.bh = bh);
}

function frameCamera() {
  // The body is reliably normalized to feet=0 / height=H, so frame on those
  // constants (robust to skinned-mesh bounding-box quirks).
  controls.target.set(0, H * 0.5, 0);
  const dist = H * 1.85;
  camera.position.set(0.18, H * 0.56, dist);
  controls.minDistance = H * 0.6; controls.maxDistance = H * 3.8;
  controls.update();
}

function setOrganTier(mesh, status) {
  const col = TIER_COLOR[status] ?? TIER_COLOR.UNMONITORED, emit = TIER_EMIT[status] ?? 0.06;
  const glow = status === "CRITICAL" ? 1.0 : status === "DYSFUNCTION" ? 0.6 : status === "STRESS" ? 0.32 : 0.0;
  if (mesh.userData.isGroup) return mesh.userData.setTier(col, emit, glow);
  mesh.userData.mat.color.setHex(col); mesh.userData.mat.emissive.setHex(col); mesh.userData.mat.emissiveIntensity = emit;
  if (mesh.userData.sprite) { mesh.userData.sprite.material.color.setHex(col); mesh.userData.sprite.material.opacity = glow; }
}

/* ---------------- threat beacon ---------------- */
function buildBeacon() {
  beacon = new THREE.Group(); beacon.visible = false;
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.12, 0.005, 8, 48),
    new THREE.MeshBasicMaterial({ color: 0xff4d5e, transparent: true, opacity: 0.9 }));
  ring.rotation.x = Math.PI / 2; ring.name = "ring"; beacon.add(ring);
  const beam = new THREE.Mesh(new THREE.CylinderGeometry(0.0032, 0.0032, 0.34, 8),
    new THREE.MeshBasicMaterial({ color: 0xff4d5e, transparent: true, opacity: 0.45 }));
  beam.position.y = 0.22; beacon.add(beam);
  beaconLabel = makeLabel("THREAT"); beaconLabel.position.y = 0.45; beacon.add(beaconLabel);
  scene.add(beacon);
}
function makeLabel() {
  const c = document.createElement("canvas"); c.width = 256; c.height = 64;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthTest: false }));
  sp.scale.set(0.42, 0.105, 1); sp.userData = { canvas: c, ctx: c.getContext("2d") };
  return sp;
}
function setLabel(sprite, text) {
  const ctx = sprite.userData.ctx; ctx.clearRect(0, 0, 256, 64);
  ctx.fillStyle = "rgba(8,12,22,.78)"; round(ctx, 6, 16, 244, 34, 9); ctx.fill();
  ctx.strokeStyle = "#ff4d5e"; ctx.lineWidth = 2; round(ctx, 6, 16, 244, 34, 9); ctx.stroke();
  ctx.fillStyle = "#ffd2d6"; ctx.font = "600 19px ui-monospace,monospace"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
  ctx.fillText(text, 128, 34); sprite.material.map.needsUpdate = true;
}
function round(c, x, y, w, h, r) { c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath(); }

/* ---------------- model load ---------------- */
// Xbot.glb already loads upright at ~human scale with feet near y=0, so the
// identity transform is correct — no bounding-box gymnastics needed.
const BODY_TUNE = { rotX: 0, rotZ: 0, scale: 1, cy: 0, y: 0 };
let bodyPivot = null, bodyModel = null;
function applyTune() {
  if (!bodyPivot) return;
  bodyModel.position.set(0, -BODY_TUNE.cy, 0);     // recentre vertically inside pivot
  bodyPivot.rotation.set(BODY_TUNE.rotX, 0, BODY_TUNE.rotZ);
  bodyPivot.scale.setScalar(BODY_TUNE.scale);
  bodyPivot.position.set(0, BODY_TUNE.y, 0);
}
function loadBody() {
  new GLTFLoader().load("./vendor/models/Xbot.glb", (gltf) => {
   try {
    const model = gltf.scene;
    model.traverse(o => { if (o.isMesh) { o.material = bodyMaterial(); o.frustumCulled = false; } });
    const pivot = new THREE.Group(); pivot.add(model); scene.add(pivot);
    bodyPivot = pivot; bodyModel = model;
    applyTune();
    const bodyBox = new THREE.Box3(new THREE.Vector3(-0.26, 0, -0.16), new THREE.Vector3(0.26, H, 0.16));
    placeOrgans(bodyBox);
    frameCamera();
    modelReady = true;
    if (pendingRegions) { update(pendingRegions); pendingRegions = null; }
   } catch (e) { console.error("[twin]", e); }
  }, undefined, (err) => console.error("[twin] model load failed", err));
}

/* ---------------- lifecycle ---------------- */
function start(el) {
  if (started || !el) return;
  container = el;
  const w = el.clientWidth || 480, h = el.clientHeight || 560;
  scene = new THREE.Scene(); scene.fog = new THREE.FogExp2(0x05070d, 0.14);
  camera = new THREE.PerspectiveCamera(36, w / h, 0.1, 100);
  camera.position.set(0.15, 1.08, 2.7);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio)); renderer.setSize(w, h);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  el.querySelectorAll("canvas").forEach(c => c.remove());
  el.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.06; controls.enablePan = false;
  controls.target.set(0, 1.02, 0); controls.minDistance = 1.1; controls.maxDistance = 5.5;
  controls.autoRotate = true; controls.autoRotateSpeed = 0.7;

  scene.add(new THREE.AmbientLight(0x4466aa, 0.7));
  const key = new THREE.PointLight(0x4fb6ff, 22, 14); key.position.set(2.4, 2.6, 3); scene.add(key);
  const rim = new THREE.PointLight(0x2b6cff, 16, 16); rim.position.set(-2.6, 1.4, -2.4); scene.add(rim);
  const fill = new THREE.DirectionalLight(0x88aaff, 0.5); fill.position.set(0, 1, 2); scene.add(fill);

  glowTex = glowTexture();
  buildBeacon(); loadBody();
  clock = new THREE.Clock(); started = true;
  ro = new ResizeObserver(resize); ro.observe(el);
  animate();
}

function update(regions) {
  if (!started) { pendingRegions = regions; return; }
  regions = regions || {};
  let worst = null, worstSev = 0.44;
  Object.keys(organs).forEach(key => {
    const r = regions[key] || {}, status = r.status || "UNMONITORED";
    setOrganTier(organs[key], status);
    if (r.monitored && (r.severity || 0) > worstSev && status !== "NORMAL") { worstSev = r.severity; worst = key; }
  });
  // body envelope colour follows skin region (or worst overall for ambient mood)
  const sk = regions.skin || {};
  const envCol = TIER_COLOR[sk.status] ?? TIER_COLOR.NORMAL;
  bodyMats.forEach(m => {
    m.emissive.setHex(envCol); if (m.userData.rim) m.userData.rim.value.setHex(envCol);
    m.opacity = (sk.status && sk.status !== "NORMAL") ? 0.3 : 0.26;
  });
  if (worst && organs[worst]) {
    const p = organs[worst].userData.isGroup ? organs[worst].position : organs[worst].position;
    beacon.position.set(p.x, p.y, p.z); beacon.visible = true;
    setLabel(beaconLabel, (REGION_LABEL[worst] || worst).toUpperCase());
  } else if (beacon) beacon.visible = false;
}

function resize() {
  if (!renderer || !container) return;
  const w = container.clientWidth, h = container.clientHeight; if (!w || !h) return;
  camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h);
}

function animate() {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime();
  controls.update();
  if (organs.heart) { const s = 1 + 0.07 * Math.max(0, Math.sin(t * 6.5)); organs.heart.scale.set(s, 1.18 * s, 0.92 * s); }
  if (beacon?.visible) {
    const ring = beacon.getObjectByName("ring");
    ring.scale.setScalar(1 + 0.16 * Math.sin(t * 4)); ring.material.opacity = 0.5 + 0.4 * Math.abs(Math.sin(t * 4));
  }
  renderer.render(scene, camera);
}

window.AegisTwin = { start, update, resize,
  tune: (o) => { Object.assign(BODY_TUNE, o); applyTune(); return BODY_TUNE; },
  autorot: (b) => { controls.autoRotate = b; } };
const probe = setInterval(() => {
  const el = document.getElementById("twin-stage");
  if (el && el.clientWidth > 40 && el.clientHeight > 40) { clearInterval(probe); start(el); }
}, 250);
