/* ============================================================
   AEGIS-OS — the vascular tree, generated

   WHY NOT THE CCF MESH. `VH_M_Blood_Vasculature.glb` ships in the organ atlas
   and is deliberately not loaded: rendered with a magenta probe at full opacity
   it resolves to a small cluster in the PELVIS. The CCF reference covers
   abdominal and pelvic vessels, not a whole-body tree, and the twin described it
   in a comment as "a systemic vessel network reaching the limbs" for months
   while it was doing nothing of the sort.

   So the tree is built here instead, from anatomy, along the same landmark
   fractions the body field uses. It is a SCHEMATIC — named great vessels with
   plausible courses and calibres — and the docstring says so rather than
   implying a segmentation. What it is good for is exactly what the twin needs:
   showing perfusion reaching (or failing to reach) a territory.

   ARTERIAL AND VENOUS ARE SEPARATE TREES, because the clinical story the twin
   tells is about perfusion, and a single undifferentiated web cannot show a
   mean arterial pressure falling while venous return is unchanged.
   ============================================================ */
import * as THREE from "three";
import { mergeGeometries } from "./vendor/three/BufferGeometryUtils.js";

/** @typedef {{name:string, side:number, pts:number[][], r0:number, r1:number}} Vessel */

/**
 * The great vessels, as fractions of stature so they scale with the body.
 * Points are [x, y, z] with x already signed by `side`; radii in metres at a
 * 1.70 m stature, tapering r0 -> r1 along the path.
 *
 * Calibres are real: aorta ~12 mm radius at the root, common iliac ~5 mm,
 * femoral ~4 mm, brachial ~2 mm, radial ~1.3 mm. Getting these roughly right
 * matters — a tree of uniform tubes reads as a decoration, and one that tapers
 * correctly reads as a circulation.
 */
function arterialPlan(H, S) {
  const y = (f) => f * H, x = (m) => m * S, z = (m) => m * S;
  /** @type {Vessel[]} */
  const v = [];
  const both = (fn) => { fn(-1); fn(1); };

  // --- aorta: root -> arch -> descending -> bifurcation --------------------
  v.push({ name: "aortic root", side: 0, r0: 0.012, r1: 0.011, pts: [
    [x(0.010), y(0.712), z(-0.020)], [x(0.006), y(0.740), z(-0.028)],
    [x(-0.004), y(0.762), z(-0.034)]] });
  v.push({ name: "aortic arch", side: 0, r0: 0.011, r1: 0.010, pts: [
    [x(-0.004), y(0.762), z(-0.034)], [x(-0.014), y(0.775), z(-0.040)],
    [x(-0.022), y(0.770), z(-0.052)], [x(-0.018), y(0.752), z(-0.058)]] });
  v.push({ name: "descending aorta", side: 0, r0: 0.010, r1: 0.007, pts: [
    [x(-0.018), y(0.752), z(-0.058)], [x(-0.014), y(0.700), z(-0.062)],
    [x(-0.012), y(0.640), z(-0.060)], [x(-0.010), y(0.590), z(-0.056)],
    [x(-0.008), y(0.556), z(-0.050)]] });

  // --- head and neck: carotid + vertebral ----------------------------------
  both((s) => {
    v.push({ name: "common carotid", side: s, r0: 0.005, r1: 0.0035, pts: [
      [x(-0.010), y(0.772), z(-0.044)], [x(s * 0.020), y(0.800), z(-0.040)],
      [x(s * 0.026), y(0.830), z(-0.030)], [x(s * 0.026), y(0.862), z(-0.024)]] });
    v.push({ name: "vertebral", side: s, r0: 0.0022, r1: 0.0018, pts: [
      [x(s * 0.022), y(0.792), z(-0.026)], [x(s * 0.018), y(0.826), z(-0.012)],
      [x(s * 0.014), y(0.858), z(-0.008)]] });
  });

  // --- arms: subclavian -> axillary -> brachial -> radial ------------------
  both((s) => {
    v.push({ name: "subclavian", side: s, r0: 0.0055, r1: 0.0045, pts: [
      [x(-0.012), y(0.774), z(-0.042)], [x(s * 0.060), y(0.800), z(-0.034)],
      [x(s * 0.120), y(0.802), z(-0.020)], [x(s * 0.168), y(0.790), z(-0.008)]] });
    v.push({ name: "brachial", side: s, r0: 0.0045, r1: 0.0026, pts: [
      [x(s * 0.168), y(0.790), z(-0.008)], [x(s * 0.196), y(0.740), z(-0.004)],
      [x(s * 0.208), y(0.686), z(0.000)], [x(s * 0.214), y(0.634), z(0.004)]] });
    v.push({ name: "radial", side: s, r0: 0.0022, r1: 0.0013, pts: [
      [x(s * 0.214), y(0.634), z(0.004)], [x(s * 0.222), y(0.580), z(0.008)],
      [x(s * 0.228), y(0.524), z(0.012)], [x(s * 0.230), y(0.492), z(0.014)]] });
  });

  // --- viscera --------------------------------------------------------------
  v.push({ name: "celiac trunk", side: 0, r0: 0.004, r1: 0.003, pts: [
    [x(-0.012), y(0.664), z(-0.058)], [x(0.010), y(0.660), z(-0.040)],
    [x(0.034), y(0.656), z(-0.026)]] });
  v.push({ name: "superior mesenteric", side: 0, r0: 0.0038, r1: 0.0022, pts: [
    [x(-0.012), y(0.648), z(-0.056)], [x(-0.004), y(0.628), z(-0.030)],
    [x(0.004), y(0.606), z(-0.010)]] });
  both((s) => {
    v.push({ name: "renal", side: s, r0: 0.0032, r1: 0.0024, pts: [
      [x(-0.012), y(0.640), z(-0.056)], [x(s * 0.030), y(0.638), z(-0.050)],
      [x(s * 0.056), y(0.634), z(-0.044)]] });
  });

  // --- legs: iliac -> femoral -> popliteal -> tibial -----------------------
  both((s) => {
    v.push({ name: "common iliac", side: s, r0: 0.005, r1: 0.0042, pts: [
      [x(-0.008), y(0.556), z(-0.050)], [x(s * 0.036), y(0.534), z(-0.040)],
      [x(s * 0.062), y(0.508), z(-0.026)]] });
    v.push({ name: "femoral", side: s, r0: 0.0042, r1: 0.0030, pts: [
      [x(s * 0.062), y(0.508), z(-0.026)], [x(s * 0.078), y(0.452), z(-0.014)],
      [x(s * 0.084), y(0.380), z(-0.008)], [x(s * 0.086), y(0.312), z(-0.004)]] });
    v.push({ name: "popliteal", side: s, r0: 0.0030, r1: 0.0024, pts: [
      [x(s * 0.086), y(0.312), z(-0.004)], [x(s * 0.086), y(0.286), z(0.006)],
      [x(s * 0.084), y(0.258), z(0.004)]] });
    v.push({ name: "posterior tibial", side: s, r0: 0.0024, r1: 0.0013, pts: [
      [x(s * 0.084), y(0.258), z(0.004)], [x(s * 0.080), y(0.180), z(0.004)],
      [x(s * 0.078), y(0.100), z(0.006)], [x(s * 0.076), y(0.052), z(0.008)]] });
  });
  return v;
}

/**
 * Veins. Deliberately not a mirror of the arteries: the great veins run
 * anterior and to the RIGHT of the midline (the IVC), the venae cavae are wider
 * than the aorta at lower pressure, and the limb veins sit superficial to their
 * arteries. A mirrored tree would look like a rendering trick.
 */
function venousPlan(H, S) {
  const y = (f) => f * H, x = (m) => m * S, z = (m) => m * S;
  const v = [];
  const both = (fn) => { fn(-1); fn(1); };

  v.push({ name: "inferior vena cava", side: 0, r0: 0.008, r1: 0.013, pts: [
    [x(0.014), y(0.552), z(-0.046)], [x(0.018), y(0.606), z(-0.050)],
    [x(0.020), y(0.664), z(-0.052)], [x(0.018), y(0.712), z(-0.046)]] });
  v.push({ name: "superior vena cava", side: 0, r0: 0.009, r1: 0.011, pts: [
    [x(0.022), y(0.792), z(-0.038)], [x(0.020), y(0.760), z(-0.038)],
    [x(0.016), y(0.728), z(-0.034)]] });
  both((s) => {
    v.push({ name: "internal jugular", side: s, r0: 0.0045, r1: 0.0055, pts: [
      [x(s * 0.032), y(0.860), z(-0.020)], [x(s * 0.034), y(0.826), z(-0.026)],
      [x(s * 0.030), y(0.792), z(-0.034)]] });
    v.push({ name: "basilic", side: s, r0: 0.0016, r1: 0.0030, pts: [
      [x(s * 0.228), y(0.494), z(0.020)], [x(s * 0.216), y(0.586), z(0.014)],
      [x(s * 0.200), y(0.678), z(0.006)], [x(s * 0.176), y(0.762), z(-0.002)]] });
    v.push({ name: "femoral vein", side: s, r0: 0.0028, r1: 0.0046, pts: [
      [x(s * 0.082), y(0.256), z(0.014)], [x(s * 0.086), y(0.360), z(0.006)],
      [x(s * 0.078), y(0.462), z(-0.004)], [x(s * 0.058), y(0.520), z(-0.020)]] });
    v.push({ name: "great saphenous", side: s, r0: 0.0012, r1: 0.0022, pts: [
      [x(s * 0.072), y(0.070), z(0.016)], [x(s * 0.070), y(0.200), z(0.018)],
      [x(s * 0.076), y(0.360), z(0.014)], [x(s * 0.070), y(0.500), z(0.004)]] });
  });
  return v;
}

/** One tapering tube along a smooth curve through the given points. */
function tube(v, radial = 7) {
  const curve = new THREE.CatmullRomCurve3(
    v.pts.map((p) => new THREE.Vector3(p[0], p[1], p[2])), false, "catmullrom", 0.4);
  const segs = Math.max(12, Math.round(curve.getLength() * 90));
  const g = new THREE.TubeGeometry(curve, segs, 1.0, radial, false);

  // TubeGeometry cannot taper, so the radius is applied per-vertex afterwards:
  // each ring is pushed back to its own calibre along the ring normal. A tree of
  // uniform tubes reads as decoration; one that narrows from aorta to tibial
  // artery reads as a circulation.
  const pos = g.getAttribute("position"), nrm = g.getAttribute("normal");
  const rings = segs + 1, perRing = radial + 1;
  for (let i = 0; i < rings; i++) {
    const t = i / (rings - 1);
    const r = v.r0 + (v.r1 - v.r0) * t;
    const c = curve.getPointAt(Math.min(1, t));
    for (let j = 0; j < perRing; j++) {
      const k = i * perRing + j;
      pos.setXYZ(k,
        c.x + nrm.getX(k) * r, c.y + nrm.getY(k) * r, c.z + nrm.getZ(k) * r);
    }
  }
  pos.needsUpdate = true;

  // Flow coordinate: 0 at the proximal end, 1 at the distal. The shader animates
  // a pulse along it, so blood visibly travels OUT from the heart rather than
  // the whole tree blinking together.
  const flow = new Float32Array(pos.count);
  for (let i = 0; i < rings; i++) {
    for (let j = 0; j < perRing; j++) flow[i * perRing + j] = i / (rings - 1);
  }
  g.setAttribute("aFlow", new THREE.BufferAttribute(flow, 1));
  return g;
}

/**
 * Build the arterial and venous trees as two merged geometries.
 * @returns {{arterial: THREE.BufferGeometry, venous: THREE.BufferGeometry, names: string[]}}
 */
export function buildVessels(H = 1.7) {
  const S = H / 1.7;
  const a = arterialPlan(H, S), v = venousPlan(H, S);
  const merge = (list) => {
    const geos = list.map((x) => tube(x));
    const m = mergeGeometries(geos, false);
    geos.forEach((g) => g.dispose());
    return m;
  };
  return {
    arterial: merge(a),
    venous: merge(v),
    names: a.concat(v).map((x) => x.name),
    counts: { arterial: a.length, venous: v.length },
  };
}

/**
 * Vessel material. Emissive tubes with a pulse travelling along the flow
 * coordinate, so the tree reads as moving blood rather than as glowing wire.
 *
 * ARTERIAL AND VENOUS COLOUR IS ANATOMY, not signal — it is constant, and it is
 * the same convention every anatomy atlas uses. Perfusion FAILURE is what
 * changes: `uPerfusion` drops toward 0 and the affected tree dims and slows,
 * which is a legible way to show a territory losing its blood supply.
 */
export function makeVesselMaterial(color, opts = {}) {
  const m = new THREE.MeshBasicMaterial({
    color: new THREE.Color(color),
    transparent: true,
    opacity: 0.9,
    depthWrite: false,
    blending: THREE.AdditiveBlending,
  });
  const u = {
    uTime: { value: 0 },
    uPerfusion: { value: 1.0 },      // 1 = normal, 0 = no flow
    uSpeed: { value: opts.speed ?? 0.55 },
    uDir: { value: opts.dir ?? 1.0 },  // +1 away from the heart, -1 back toward it
    uBright: { value: opts.bright ?? 1.0 },
  };
  m.onBeforeCompile = (sh) => {
    Object.assign(sh.uniforms, u);
    sh.vertexShader = sh.vertexShader
      .replace("#include <common>", "#include <common>\nattribute float aFlow;\nvarying float vFlow;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvFlow = aFlow;");
    sh.fragmentShader = sh.fragmentShader
      .replace("#include <common>", `#include <common>
        varying float vFlow;
        uniform float uTime, uPerfusion, uSpeed, uDir, uBright;`)
      .replace("#include <dithering_fragment>", `#include <dithering_fragment>
        // A travelling pulse. Three waves at different phases so the flow reads
        // as continuous rather than as one dot chasing round a loop.
        float p = vFlow * uDir - uTime * uSpeed * (0.35 + 0.65 * uPerfusion);
        float wave = 0.0;
        wave += pow(0.5 + 0.5 * sin(6.2831 * (p * 3.0)), 6.0);
        wave += pow(0.5 + 0.5 * sin(6.2831 * (p * 3.0 + 0.33)), 6.0) * 0.6;
        wave += pow(0.5 + 0.5 * sin(6.2831 * (p * 3.0 + 0.66)), 6.0) * 0.35;

        // Perfusion gates BRIGHTNESS, not hue: a territory losing its supply
        // goes dark and slow, it does not change colour into something else.
        float lum = (0.16 + 0.84 * wave) * uPerfusion * uBright;
        gl_FragColor.rgb *= lum;
        gl_FragColor.a *= clamp(0.30 + 0.70 * lum, 0.0, 1.0);`);
  };
  m.userData.aegis = u;
  m.customProgramCacheKey = () => "aegis-vessel-v1";
  return m;
}
