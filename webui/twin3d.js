/* ============================================================
   AEGIS-OS — 3D anatomical digital twin (Three.js, vendored)
   A real rigged humanoid mesh rendered as a translucent medical
   hologram. The BODY ITSELF is sectioned: each anatomical zone
   (head, chest, abdomen, pelvis, limbs) glows the colour of the
   compromised system in that region. A wrist marker shows when a
   wearable is streaming. Spin / zoom via OrbitControls.
   ============================================================ */
import * as THREE from "three";
import { OrbitControls } from "./vendor/three/OrbitControls.js";
import { GLTFLoader } from "./vendor/three/GLTFLoader.js";

const H = 1.7;                                   // body height (feet at y=0)
const BASE = 0x2f7fd8;                           // calm, healthy body colour
const TIER_COLOR = { STRESS: 0xffd23b, DYSFUNCTION: 0xff9a3b, CRITICAL: 0xff4d5e };
const REGION_LABEL = {
  brain: "Neurological", heart: "Cardiovascular", lungs: "Respiratory",
  liver: "Hepatic", kidneys: "Renal", skin: "Peripheral", blood: "Coagulation",
};
const ZONE_Y = { brain: 0.92, lungs: 0.72, heart: 0.71, liver: 0.58, kidneys: 0.56, skin: 0.42, blood: 0.6 };

// shared zone colours (uniform values — mutate in place to update the shader)
const ZONE = {
  head: new THREE.Color(BASE), chest: new THREE.Color(BASE), abd: new THREE.Color(BASE),
  pelvis: new THREE.Color(BASE), limb: new THREE.Color(BASE), rim: new THREE.Color(0x9fe6ff),
};

let renderer, scene, camera, controls, container, ro, clock, modelGroup;
let beacon, beaconLabel, wrist, wristBone, glowTex;
let started = false, pendingData = null;

/* ---------------- helpers ---------------- */
function glowTexture() {
  const c = document.createElement("canvas"); c.width = c.height = 128;
  const x = c.getContext("2d"), g = x.createRadialGradient(64, 64, 2, 64, 64, 64);
  g.addColorStop(0, "rgba(255,255,255,1)"); g.addColorStop(.3, "rgba(255,255,255,.4)"); g.addColorStop(1, "rgba(255,255,255,0)");
  x.fillStyle = g; x.fillRect(0, 0, 128, 128); return new THREE.CanvasTexture(c);
}

/* Body hologram material: standard (auto-skins) but the fragment colour is fully
   replaced by a height-zoned colour + fresnel rim, so we get clean sectioned
   glow with NO lighting hotspots and smooth transitions between zones. */
function bodyMaterial() {
  const m = new THREE.MeshStandardMaterial({
    transparent: true, opacity: 0.4, depthWrite: false, side: THREE.DoubleSide,
    roughness: 1, metalness: 0,
  });
  m.onBeforeCompile = (sh) => {
    sh.uniforms.uHead = { value: ZONE.head }; sh.uniforms.uChest = { value: ZONE.chest };
    sh.uniforms.uAbd = { value: ZONE.abd }; sh.uniforms.uPelvis = { value: ZONE.pelvis };
    sh.uniforms.uLimb = { value: ZONE.limb }; sh.uniforms.uRim = { value: ZONE.rim };
    sh.uniforms.uYmax = { value: H }; sh.uniforms.uPulse = { value: 0 };
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nvarying float vWY; varying float vWX;")
      .replace("#include <project_vertex>",
        "#include <project_vertex>\n vec4 _wp = modelMatrix * vec4(transformed,1.0); vWY=_wp.y; vWX=abs(_wp.x);");
    sh.fragmentShader = sh.fragmentShader
      .replace("#include <common>", `#include <common>
        varying float vWY; varying float vWX;
        uniform vec3 uHead,uChest,uAbd,uPelvis,uLimb,uRim; uniform float uYmax,uPulse;`)
      .replace("#include <dithering_fragment>", `#include <dithering_fragment>
        float h = clamp(vWY/uYmax, 0.0, 1.0);
        vec3 col = uLimb;                                   // legs
        col = mix(col, uPelvis, smoothstep(0.44,0.50,h));
        col = mix(col, uAbd,    smoothstep(0.50,0.60,h));
        col = mix(col, uChest,  smoothstep(0.60,0.74,h));
        col = mix(col, uHead,   smoothstep(0.80,0.88,h));
        float arm = smoothstep(0.17,0.27,vWX) * smoothstep(0.55,0.62,h) * (1.0-smoothstep(0.80,0.86,h));
        col = mix(col, uLimb, arm);
        float rim = pow(1.0 - clamp(dot(normalize(normal), normalize(vViewPosition)),0.0,1.0), 1.6);
        vec3 outc = col*(0.42+0.6*rim) + uRim*rim*0.55;
        outc *= (1.0 + 0.10*uPulse);
        gl_FragColor = vec4(outc, clamp(0.4 + rim*0.5, 0.0, 0.95));`);
    bodyMaterial._uPulse = sh.uniforms.uPulse;
  };
  return m;
}

/* ---------------- model ---------------- */
function loadBody() {
  new GLTFLoader().load("./vendor/models/Xbot.glb", (gltf) => {
    modelGroup = gltf.scene;
    const mat = bodyMaterial();
    modelGroup.traverse(o => {
      if (!o.isMesh) return;
      if (/joint/i.test(o.name)) { o.visible = false; return; }   // hide ball-joint helper mesh
      o.material = mat; o.frustumCulled = false;
    });
    // find a wrist bone for the wearable marker
    modelGroup.traverse(o => {
      if (o.isBone && !wristBone && /lefthand|leftwrist/i.test(o.name)) wristBone = o;
    });
    if (!wristBone) modelGroup.traverse(o => { if (o.isBone && !wristBone && /hand/i.test(o.name)) wristBone = o; });
    // normalize to height H, feet at y=0, centred
    let box = new THREE.Box3().setFromObject(modelGroup);
    const size = new THREE.Vector3(); box.getSize(size);
    modelGroup.scale.setScalar(H / size.y);
    modelGroup.userData.baseScale = H / size.y;
    box = new THREE.Box3().setFromObject(modelGroup);
    modelGroup.position.x -= (box.min.x + box.max.x) / 2;
    modelGroup.position.z -= (box.min.z + box.max.z) / 2;
    modelGroup.position.y -= box.min.y;
    poseArms(modelGroup);                          // T-pose -> natural standing
    scene.add(modelGroup);
    normalizeBySkeleton(modelGroup);               // scale/centre by REAL bone bounds
    fitCamera();                                   // frame the whole body
    buildWrist();
    if (pendingData) { update(pendingData); pendingData = null; }
  }, undefined, (e) => console.error("[twin] model load failed", e));
}

/* Scale + position the figure using actual SKELETON bounds. Box3.setFromObject
   reports a SkinnedMesh's bind-pose box, which for this rig is wildly wrong, so
   we measure the bones' world positions instead (robust + pose-aware). */
function normalizeBySkeleton(root) {
  const v = new THREE.Vector3();
  const bounds = () => {
    root.updateWorldMatrix(true, true);
    let lo = Infinity, hi = -Infinity, xl = Infinity, xh = -Infinity, zl = Infinity, zh = -Infinity;
    root.traverse(o => { if (o.isBone) { o.getWorldPosition(v); lo = Math.min(lo, v.y); hi = Math.max(hi, v.y); xl = Math.min(xl, v.x); xh = Math.max(xh, v.x); zl = Math.min(zl, v.z); zh = Math.max(zh, v.z); } });
    return { lo, hi, cx: (xl + xh) / 2, cz: (zl + zh) / 2 };
  };
  root.scale.setScalar(1); root.position.set(0, 0, 0);
  let b = bounds();
  const realH = b.hi - b.lo || 1;
  const s = (H * 0.9) / realH;                    // 0.9 leaves headroom above the head bone
  root.scale.setScalar(s);
  root.userData.baseScale = s;
  b = bounds();
  root.position.x -= b.cx; root.position.z -= b.cz;
  root.position.y -= b.lo - H * 0.05;             // feet just below 0
  root.updateWorldMatrix(true, true);
}

/* Rotate the shoulder/arm bones so the figure stands naturally (arms by the
   sides) instead of the model's default T-pose. */
function poseArms(root) {
  const B = {};
  root.traverse(o => { if (o.isBone) B[o.name.toLowerCase()] = o; });
  const find = (re) => Object.keys(B).find(k => re.test(k));
  const set = (re, ax, v) => { const k = find(re); if (k) B[k].rotation[ax] += v; };
  // bring upper arms down to the sides (negative z lowers the left arm)
  set(/leftarm$/, "z", -1.32); set(/rightarm$/, "z", 1.32);
  // slight elbow bend so hands rest naturally by the thighs
  set(/leftforearm$/, "z", -0.22); set(/rightforearm$/, "z", 0.22);
}

/* ---------------- wearable wrist marker ---------------- */
function buildWrist() {
  const c = document.createElement("canvas"); c.width = c.height = 128;
  drawWrist(c, false);
  wrist = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthTest: false, depthWrite: false }));
  wrist.scale.set(0.16, 0.16, 1); wrist.userData = { canvas: c, on: null }; wrist.visible = false;
  scene.add(wrist);
}
function drawWrist(c, on) {
  const x = c.getContext("2d"); x.clearRect(0, 0, 128, 128);
  const col = on ? "#1fdd8a" : "#5b6a8f";
  x.strokeStyle = col; x.fillStyle = on ? "rgba(31,221,138,.16)" : "rgba(91,106,143,.12)";
  x.lineWidth = 6; x.beginPath(); x.arc(64, 64, 40, 0, 7); x.fill(); x.stroke();
  // pulse glyph
  x.strokeStyle = col; x.lineWidth = 5; x.lineJoin = "round"; x.beginPath();
  x.moveTo(40, 64); x.lineTo(54, 64); x.lineTo(60, 48); x.lineTo(70, 82); x.lineTo(76, 64); x.lineTo(90, 64); x.stroke();
  if (on) { x.shadowBlur = 16; x.shadowColor = col; x.stroke(); x.shadowBlur = 0; }
}
function setWrist(on) {
  if (!wrist || wrist.userData.on === on) return;
  wrist.userData.on = on; drawWrist(wrist.userData.canvas, on); wrist.material.map.needsUpdate = true;
}

/* Frame the camera on the body. The figure is normalized to feet≈0 / height≈H,
   so we frame those constants (no unreliable skinned-mesh bounding box). */
function fitCamera() {
  const fov = camera.fov * Math.PI / 180;
  const dist = (H * 0.48) / Math.tan(fov / 2) * 1.12;
  controls.target.set(0, H * 0.46, 0);
  camera.position.set(0, H * 0.46, dist);
  controls.minDistance = dist * 0.5; controls.maxDistance = dist * 2.0;
  controls.update();
}

/* ---------------- threat beacon ---------------- */
function buildBeacon() {
  beacon = new THREE.Group(); beacon.visible = false;
  const ring = new THREE.Mesh(new THREE.TorusGeometry(0.13, 0.004, 8, 50),
    new THREE.MeshBasicMaterial({ color: 0xff4d5e, transparent: true, opacity: 0.9 }));
  ring.rotation.x = Math.PI / 2; ring.name = "ring"; beacon.add(ring);
  beaconLabel = makeLabel(); beaconLabel.position.y = 0.2; beacon.add(beaconLabel);
  scene.add(beacon);
}
function makeLabel() {
  const c = document.createElement("canvas"); c.width = 256; c.height = 64;
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthTest: false }));
  sp.scale.set(0.46, 0.115, 1); sp.userData = { ctx: c.getContext("2d") }; return sp;
}
function setLabel(sprite, text) {
  const x = sprite.userData.ctx; x.clearRect(0, 0, 256, 64);
  x.fillStyle = "rgba(10,14,24,.82)"; rr(x, 6, 16, 244, 34, 10); x.fill();
  x.strokeStyle = "#ff4d5e"; x.lineWidth = 2; rr(x, 6, 16, 244, 34, 10); x.stroke();
  x.fillStyle = "#ffd2d6"; x.font = "600 18px ui-monospace,monospace"; x.textAlign = "center"; x.textBaseline = "middle";
  x.fillText(text, 128, 34); sprite.material.map.needsUpdate = true;
}
function rr(c, x, y, w, h, r) { c.beginPath(); c.moveTo(x + r, y); c.arcTo(x + w, y, x + w, y + h, r); c.arcTo(x + w, y + h, x, y + h, r); c.arcTo(x, y + h, x, y, r); c.arcTo(x, y, x + w, y, r); c.closePath(); }

/* ---------------- lifecycle ---------------- */
function start(el) {
  if (started || !el) return;
  container = el; const w = el.clientWidth || 480, h = el.clientHeight || 560;
  scene = new THREE.Scene(); scene.fog = new THREE.FogExp2(0x05070d, 0.13);
  camera = new THREE.PerspectiveCamera(34, w / h, 0.1, 100);
  camera.position.set(0.2, H * 0.56, H * 1.9);
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(Math.min(2, devicePixelRatio)); renderer.setSize(w, h);
  renderer.outputColorSpace = THREE.SRGBColorSpace;
  el.querySelectorAll("canvas").forEach(c => c.remove()); el.appendChild(renderer.domElement);

  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true; controls.dampingFactor = 0.06; controls.enablePan = false;
  controls.target.set(0, H * 0.52, 0);
  controls.autoRotate = true; controls.autoRotateSpeed = 0.5;
  controls.minPolarAngle = Math.PI * 0.40; controls.maxPolarAngle = Math.PI * 0.56;  // stay near eye level
  scene.add(new THREE.AmbientLight(0xffffff, 1));   // colour is shader-driven; light is nominal

  glowTex = glowTexture();
  loadBody(); buildBeacon();
  clock = new THREE.Clock(); started = true;
  ro = new ResizeObserver(resize); ro.observe(el); animate();
}

function tierColor(region) {
  if (!region || !region.monitored) return BASE;
  const sev = region.severity || 0;
  if (sev < 0.18 || region.status === "NORMAL") return BASE;
  return TIER_COLOR[region.status] || BASE;
}
const worse = (a, b) => (((a && a.severity) || 0) >= ((b && b.severity) || 0) ? a : b);

/* Camera→Twin coupling: the optical biosensor (camera.js) publishes a region
   severity map here. We merge it with the streamed body_state, ESCALATION-ONLY
   (worse-of), so the webcam can light up the anatomy live but can never downgrade
   a server-detected critical state. */
let lastServerData = null, optical = {};
function mergedRegions() {
  const sr = (lastServerData && lastServerData.body_state && lastServerData.body_state.regions) || {};
  const out = {};
  for (const k of new Set([...Object.keys(sr), ...Object.keys(optical)])) out[k] = worse(sr[k], optical[k]);
  return out;
}

function update(data) {
  lastServerData = data || lastServerData;
  if (!started) { pendingData = data; return; }
  applyRegions();
}

/* Receive the optical region map from the camera engine (sign severities mapped
   to anatomical zones). Empty map clears the optical overlay. */
function setOptical(reg) {
  optical = reg || {};
  if (started) applyRegions();
}

function applyRegions() {
  const regions = mergedRegions();
  ZONE.head.setHex(tierColor(regions.brain));
  ZONE.chest.setHex(tierColor(worse(regions.heart, regions.lungs)));
  ZONE.abd.setHex(tierColor(worse(regions.liver, regions.kidneys)));
  ZONE.pelvis.setHex(tierColor(regions.blood));
  ZONE.limb.setHex(tierColor(regions.skin));

  // worst compromised region -> threat beacon (front of the relevant zone)
  let worst = null, ws = 0.44, worstR = null;
  Object.keys(ZONE_Y).forEach(k => {
    const r = regions[k]; if (r && r.monitored && (r.severity || 0) > ws && r.status !== "NORMAL") { ws = r.severity; worst = k; worstR = r; }
  });
  if (worst && beacon) {
    beacon.position.set(0, ZONE_Y[worst] * H, 0.22); beacon.visible = true;
    setLabel(beaconLabel, (REGION_LABEL[worst] || worst).toUpperCase() + (worstR && worstR.optical ? " ◉" : ""));
  } else if (beacon) beacon.visible = false;

  // wearable wrist marker
  const d = lastServerData || {};
  const live = ["wearable", "simulator", "camera"].includes(d.source) &&
    (d.risk?.signal_quality ?? 1) > 0.15;
  if (wrist) { wrist.visible = !!wristBone; setWrist(live); }
}

function resize() {
  if (!renderer || !container) return;
  const w = container.clientWidth, h = container.clientHeight; if (!w || !h) return;
  camera.aspect = w / h; camera.updateProjectionMatrix(); renderer.setSize(w, h);
  fitCamera();
}

const _v = new THREE.Vector3();
function animate() {
  requestAnimationFrame(animate);
  const t = clock.getElapsedTime(); controls.update();
  // gentle breathing of the whole body
  if (modelGroup?.userData.baseScale) modelGroup.scale.setScalar(modelGroup.userData.baseScale * (1 + 0.008 * Math.sin(t * 1.05)));
  if (bodyMaterial._uPulse) bodyMaterial._uPulse.value = 0.5 + 0.5 * Math.sin(t * 5.0);
  if (wrist && wristBone && wrist.visible) { wristBone.getWorldPosition(_v); wrist.position.copy(_v).add(new THREE.Vector3(0, 0.02, 0.04)); }
  if (beacon?.visible) {
    const ring = beacon.getObjectByName("ring");
    ring.scale.setScalar(1 + 0.14 * Math.sin(t * 4)); ring.material.opacity = 0.5 + 0.4 * Math.abs(Math.sin(t * 4));
  }
  renderer.render(scene, camera);
}

window.AegisTwin = { start, update, resize, setOptical };
const probe = setInterval(() => {
  const el = document.getElementById("twin-stage");
  if (el && el.clientWidth > 40 && el.clientHeight > 40) { clearInterval(probe); start(el); }
}, 250);
