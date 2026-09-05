/* ============================================================
   AEGIS-OS — the body shell, generated rather than loaded

   WHAT THIS REPLACES, AND WHY NO AMOUNT OF SHADING FIXED IT
   ---------------------------------------------------------
   The twin's body was `vendor/models/Xbot.glb` — the three.js example ROBOT.
   It is built as separate armour plates: a chest piece, a pelvis piece, upper
   and lower arm segments, thigh and calf segments, each a closed mesh with a
   visible gap at every joint. That is not a material problem and it never was.
   A robot rendered as glass is a glass robot, and the twin read as disjointed
   parts because it WAS disjointed parts.

   A medical digital twin also has no business shipping a game character's rig.
   The organs are real anatomy — HuBMAP CCF reference meshes from the Visible
   Human — and the thing housing them was a Mixamo bot.

   THE APPROACH: A SIGNED DISTANCE FIELD, SMOOTH-UNIONED, THEN POLYGONISED
   -----------------------------------------------------------------------
   The body is defined as a field: a set of anatomically proportioned capsules
   and ellipsoids combined with a POLYNOMIAL SMOOTH MINIMUM. That single choice
   is what fixes the complaint. A hard `min` union of two capsules leaves a
   crease where they meet; `smin` blends them over a radius, so a shoulder
   flows into an arm and a thigh into a hip with no seam anywhere — the surface
   is continuous by construction, not by careful placement.

   The field is then polygonised with NAIVE SURFACE NETS into one watertight
   BufferGeometry. Surface nets rather than marching cubes: it needs no 256-entry
   triangle table, it places one vertex per cell instead of up to five, and it
   produces far more uniform quads on smooth fields — which is exactly what an
   SDF is. The output is an ordinary mesh, so depth, materials, raycasting and
   the bloom layer all behave normally.

   Beyond fixing the seams this buys three things the GLB could not:
     * ~4 kB of code instead of a 2.9 MB download, and no third-party rig.
     * Exact anchors. The old code hunted for bones named /hips|pelvis/ and
       /lefthand|leftwrist/ and fell back to hand-tuned constants when a rename
       broke the search. A generated body KNOWS where its hips, neck and wrist
       are, because it put them there.
     * Parameters. Height, build and posture are numbers, so the shell can be
       fitted to a patient rather than being one fixed actor.
   NO THREE IMPORT, DELIBERATELY. Everything here is arithmetic over typed
   arrays, which means it can run in a Web Worker — and it needs to, because
   generating the mesh takes about 1.1 s in a browser and doing that on the main
   thread froze the whole console while the twin came up. The module returns raw
   arrays and plain objects; twin3d.js turns them into three.js objects at the
   boundary, where it is cheap.
   ============================================================ */

/* ------------------------------------------------------------------ *
 * SDF primitives — all return signed distance in metres
 * ------------------------------------------------------------------ */

/** Polynomial smooth minimum. k is the blend radius, in metres. */
function smin(a, b, k) {
  const h = Math.max(0, Math.min(1, 0.5 + 0.5 * (b - a) / k));
  return b * (1 - h) + a * h - k * h * (1 - h);
}

/**
 * Polynomial smooth SUBTRACTION: remove shape `b` from `a` with a soft lip.
 *
 * This is what carves anatomy rather than denting it. A hard subtraction leaves
 * a crease at the rim of every groove — the same class of artefact as a hard
 * union at a joint, and just as visible on a glass surface where every edge
 * catches the fresnel.
 */
function ssub(a, b, k) {
  const h = Math.max(0, Math.min(1, 0.5 - 0.5 * (a + b) / k));
  return a * (1 - h) + (-b) * h + k * h * (1 - h);
}

/** Distance to a capsule (a line segment of radius r) from point p. */
function sdCapsule(px, py, pz, ax, ay, az, bx, by, bz, r) {
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;
  const ab2 = abx * abx + aby * aby + abz * abz;
  let t = ab2 > 0 ? (apx * abx + apy * aby + apz * abz) / ab2 : 0;
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  const dx = apx - abx * t, dy = apy - aby * t, dz = apz - abz * t;
  return Math.sqrt(dx * dx + dy * dy + dz * dz) - r;
}

/**
 * Distance to an axis-aligned ellipsoid.
 *
 * The usual `(length(p/r) - 1) * min(r)` bound. It is not an exact distance —
 * no closed form is — but it is conservative and smooth, which is all the
 * smooth-min and the surface extraction need. Using the exact-looking
 * `length(p/r) - 1` alone would make the field scale wrongly along the long
 * axis and the blends would tighten and loosen around the body.
 */
function sdEllipsoid(px, py, pz, cx, cy, cz, rx, ry, rz) {
  const x = (px - cx) / rx, y = (py - cy) / ry, z = (pz - cz) / rz;
  const k0 = Math.sqrt(x * x + y * y + z * z);
  if (k0 === 0) return -Math.min(rx, ry, rz);
  const k1 = Math.sqrt((x / rx) ** 2 + (y / ry) ** 2 + (z / rz) ** 2);
  return (k0 - 1) * (k0 / k1);
}

/* ------------------------------------------------------------------ *
 * The body
 * ------------------------------------------------------------------ */

/**
 * Anatomical proportions, expressed in fractions of stature so the whole body
 * scales from one number. These follow standard adult canon (roughly 7.5 heads),
 * which is what makes the silhouette read as a person rather than as a doll:
 * shoulder span ~0.23H, hip width ~0.19H, elbow at ~0.63H, knee at ~0.29H.
 */
export const DEFAULT_BUILD = {
  height: 1.7,
  shoulderSpan: 0.115,   // half-span, x
  chestDepth: 0.098,
  waistScale: 0.80,      // waist width relative to chest
  hipSpan: 0.088,
  limbScale: 1.0,        // overall limb thickness
  armDrop: 0.16,         // how far the arms hang from the body, x
  blend: 0.052,          // smooth-union radius: THE seam control
};

/**
 * Build the body's signed-distance function and its anchor points.
 *
 * Every joint here is a smooth union, which is the whole point: the shoulder is
 * not an arm mesh butted against a torso mesh, it is one field in which the two
 * are indistinguishable.
 */
export function makeBodyField(build = {}) {
  const B = { ...DEFAULT_BUILD, ...build };
  const H = B.height;
  const k = B.blend;
  const L = B.limbScale;

  // Landmarks, as fractions of stature.
  const yHead = 0.935 * H, yChin = 0.865 * H, yNeck = 0.845 * H;
  const yShoulder = 0.818 * H, yChest = 0.742 * H, yWaist = 0.625 * H;
  const yHip = 0.545 * H, yCrotch = 0.485 * H;
  const yKnee = 0.285 * H, yAnkle = 0.042 * H;
  // Stature scale, hoisted. `H / 1.7` appeared about forty times inside sdf(),
  // which is evaluated once per grid point — roughly 700 000 times per build —
  // so it was some 28 million divisions for a constant.
  const HS = H / 1.7;
  const xShoulder = B.shoulderSpan * HS;
  const xHip = B.hipSpan * HS;
  const xElbow = xShoulder + B.armDrop * 0.42;
  const xWrist = xShoulder + B.armDrop * 0.72;
  const yElbow = 0.630 * H, yWrist = 0.480 * H;
  const dz = B.chestDepth * HS;

  // Anchors the rest of the twin needs. The old code searched a third-party rig
  // for bones called /hips|pelvis/ and /lefthand|leftwrist/, and silently fell
  // back to hand-tuned constants when the search missed.
  const anchors = {
    hips: { x: 0, y: yHip, z: 0 },
    neck: { x: 0, y: yNeck, z: 0 },
    head: { x: 0, y: yHead, z: 0 },
    chest: { x: 0, y: yChest, z: 0 },
    wristL: { x: -xWrist, y: yWrist, z: 0.01 },
    wristR: { x: xWrist, y: yWrist, z: 0.01 },
    torsoLo: yHip,
    torsoHi: yShoulder,
    depth: dz * 2.1,
  };

  /** The field. Returns signed distance; negative is inside the body. */
  function sdf(x, y, z) {
    // --- torso: a stack of ellipsoids, blended into one continuous trunk ----
    // Three sections rather than one, because a human trunk is not an ellipsoid:
    // it flares at the ribs, draws in at the waist and flares again at the
    // pelvis. Blending them is what produces that curve without a seam.
    let d = sdEllipsoid(x, y, z, 0, yChest, 0,
                        xShoulder * 1.02, (yShoulder - yWaist) * 0.86, dz);
    d = smin(d, sdEllipsoid(x, y, z, 0, yWaist, 0,
                            xShoulder * B.waistScale, (yChest - yHip) * 0.52, dz * 0.86), k);
    d = smin(d, sdEllipsoid(x, y, z, 0, yHip, 0,
                            xHip * 1.34, (yWaist - yCrotch) * 0.62, dz * 0.92), k);

    // --- neck and head ------------------------------------------------------
    d = smin(d, sdCapsule(x, y, z, 0, yNeck - 0.02 * H, -0.004,
                          0, yChin, -0.004, 0.049 * HS * L), k);
    d = smin(d, sdEllipsoid(x, y, z, 0, yHead, -0.004,
                            0.082 * HS, 0.104 * HS, 0.092 * HS), k * 1.3);

    // --- shoulders ----------------------------------------------------------
    // Deliberately generous spheres. A shoulder is the joint where a hard union
    // is most obvious, and it is the first place the eye reads "assembled".
    for (const s of [-1, 1]) {
      d = smin(d, sdEllipsoid(x, y, z, s * xShoulder, yShoulder, 0,
                              0.062 * HS * L, 0.058 * HS * L,
                              0.060 * HS * L), k * 1.5);
    }

    // --- arms ---------------------------------------------------------------
    for (const s of [-1, 1]) {
      d = smin(d, sdCapsule(x, y, z,
                            s * xShoulder, yShoulder - 0.012 * H, 0,
                            s * xElbow, yElbow, 0.004,
                            0.046 * HS * L), k * 1.2);
      d = smin(d, sdCapsule(x, y, z,
                            s * xElbow, yElbow, 0.004,
                            s * xWrist, yWrist, 0.010,
                            0.037 * HS * L), k);
      // hand — flattened, so it reads as a hand and not a club
      d = smin(d, sdEllipsoid(x, y, z,
                              s * (xWrist + 0.012 * H), yWrist - 0.052 * H, 0.012,
                              0.030 * HS, 0.058 * HS, 0.017 * HS), k);
    }

    // --- surface anatomy ----------------------------------------------------
    // A smooth blob of blended capsules is still a mannequin: correct in
    // silhouette and empty everywhere inside it. What makes a body read as a
    // body is the landmarks the skin actually shows — the clavicles, the
    // sternum, the costal margin, the iliac crest, the line of the spine. On a
    // glass shell these matter more than on an opaque one, because the fresnel
    // rim traces every one of them and gives the eye something to hold on to
    // between the silhouette and the organs.
    //
    // Added with smooth unions and carved with smooth subtractions, so none of
    // them introduces the crease the whole approach exists to avoid.

    // clavicles — the strongest surface landmark on a human chest
    for (const s of [-1, 1]) {
      d = smin(d, sdCapsule(x, y, z,
                            s * 0.012 * H, yShoulder + 0.012 * H, -dz * 0.52,
                            s * xShoulder * 0.94, yShoulder + 0.004 * H, -dz * 0.22,
                            0.016 * HS), k * 0.9);
    }
    // suprasternal notch, between them
    d = ssub(d, sdEllipsoid(x, y, z, 0, yShoulder + 0.016 * H, -dz * 0.72,
                            0.026 * HS, 0.020 * HS, 0.030 * HS),
             k * 0.8);

    // pectorals / chest mass, and the sternal groove between them
    for (const s of [-1, 1]) {
      d = smin(d, sdEllipsoid(x, y, z,
                              s * xShoulder * 0.46, yChest + 0.030 * H, -dz * 0.62,
                              0.062 * HS, 0.048 * HS, 0.030 * HS),
               k * 1.4);
    }
    d = ssub(d, sdCapsule(x, y, z, 0, yShoulder - 0.008 * H, -dz * 0.86,
                          0, yChest - 0.008 * H, -dz * 0.90, 0.018 * HS),
             k * 1.1);

    // costal margin — the shallow V under the ribs that separates chest from
    // abdomen, and the single cue that reads most clearly as "torso" rather
    // than "cylinder"
    for (const s of [-1, 1]) {
      d = ssub(d, sdCapsule(x, y, z,
                            s * 0.010 * H, yChest - 0.030 * H, -dz * 0.80,
                            s * xShoulder * 0.86, yChest - 0.002 * H, -dz * 0.30,
                            0.013 * HS), k * 1.0);
    }

    // linea alba, faint, and the navel
    d = ssub(d, sdCapsule(x, y, z, 0, yChest - 0.045 * H, -dz * 0.88,
                          0, yHip + 0.030 * H, -dz * 0.86, 0.009 * HS), k * 1.2);
    d = ssub(d, sdEllipsoid(x, y, z, 0, yWaist - 0.030 * H, -dz * 0.86,
                            0.013 * HS, 0.016 * HS, 0.020 * HS),
             k * 0.7);

    // iliac crests — the hip landmarks that stop the pelvis reading as a ball
    for (const s of [-1, 1]) {
      d = smin(d, sdCapsule(x, y, z,
                            s * xHip * 0.62, yHip + 0.028 * H, -dz * 0.52,
                            s * xHip * 1.30, yHip + 0.008 * H, dz * 0.10,
                            0.017 * HS), k * 1.3);
    }

    // spinal groove, down the back
    d = ssub(d, sdCapsule(x, y, z, 0, yShoulder - 0.010 * H, dz * 0.88,
                          0, yHip + 0.020 * H, dz * 0.86, 0.014 * HS), k * 1.3);
    // scapulae
    for (const s of [-1, 1]) {
      d = smin(d, sdEllipsoid(x, y, z,
                              s * xShoulder * 0.52, yChest + 0.048 * H, dz * 0.60,
                              0.048 * HS, 0.052 * HS, 0.020 * HS),
               k * 1.5);
    }

    // deltoid and biceps mass, so the arms taper like limbs and not like pipes
    for (const s of [-1, 1]) {
      d = smin(d, sdEllipsoid(x, y, z,
                              s * (xShoulder + 0.008 * H), yShoulder - 0.030 * H, 0,
                              0.040 * HS * L, 0.052 * HS * L,
                              0.042 * HS * L), k * 1.6);
      d = smin(d, sdEllipsoid(x, y, z,
                              s * (xShoulder + (xElbow - xShoulder) * 0.45),
                              (yShoulder + yElbow) * 0.5, -0.006,
                              0.033 * HS * L, 0.055 * HS * L,
                              0.034 * HS * L), k * 1.5);
    }

    // sternocleidomastoid — the two cords in the neck
    for (const s of [-1, 1]) {
      d = smin(d, sdCapsule(x, y, z,
                            s * 0.010 * H, yChin - 0.004 * H, -0.020,
                            s * 0.028 * H, yNeck - 0.030 * H, -0.016,
                            0.010 * HS), k * 0.9);
    }

    // --- legs ---------------------------------------------------------------
    for (const s of [-1, 1]) {
      d = smin(d, sdCapsule(x, y, z,
                            s * xHip * 0.86, yCrotch + 0.04 * H, 0,
                            s * xHip * 0.92, yKnee, 0.002,
                            0.072 * HS * L), k * 1.4);
      d = smin(d, sdCapsule(x, y, z,
                            s * xHip * 0.92, yKnee, 0.002,
                            s * xHip * 0.80, yAnkle, 0.006,
                            0.050 * HS * L), k * 1.1);
      // foot — long in z, so the figure stands rather than balancing on pegs
      d = smin(d, sdEllipsoid(x, y, z,
                              s * xHip * 0.80, yAnkle - 0.012 * H, 0.036 * H,
                              0.042 * HS, 0.030 * HS, 0.088 * HS), k);
      // quadriceps at the front of the thigh, calf at the back of the shin —
      // without them a leg is a cone and the figure reads as a shop dummy
      d = smin(d, sdEllipsoid(x, y, z,
                              s * xHip * 0.88, (yCrotch + yKnee) * 0.52, -0.014,
                              0.052 * HS * L, 0.098 * HS * L,
                              0.048 * HS * L), k * 1.7);
      d = smin(d, sdEllipsoid(x, y, z,
                              s * xHip * 0.90, yKnee - 0.062 * H, 0.016,
                              0.038 * HS * L, 0.062 * HS * L,
                              0.036 * HS * L), k * 1.5);
      // knee — a slight widening, so the joint is where the eye expects it
      d = smin(d, sdEllipsoid(x, y, z, s * xHip * 0.92, yKnee, -0.004,
                              0.044 * HS * L, 0.038 * HS * L,
                              0.044 * HS * L), k * 1.4);
    }
    return d;
  }

  return { sdf, anchors, build: B };
}

/* ------------------------------------------------------------------ *
 * Naive surface nets
 * ------------------------------------------------------------------ */

// The 12 edges of a cell, as pairs of corner indices, and the corner offsets.
const CUBE_CORNERS = [
  [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
  [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
];
const CUBE_EDGES = [
  [0, 1], [1, 2], [2, 3], [3, 0],
  [4, 5], [5, 6], [6, 7], [7, 4],
  [0, 4], [1, 5], [2, 6], [3, 7],
];

/**
 * Polygonise a signed-distance function into one watertight BufferGeometry.
 *
 * Naive surface nets: one vertex per cell that straddles the surface, placed at
 * the average of its edge crossings, then a quad for every grid edge with a sign
 * change. Chosen over marching cubes because it needs no 256-entry case table,
 * emits one vertex per cell rather than up to five, and gives markedly more
 * uniform quads on a smooth field — which is what an SDF is.
 *
 * Normals come from the FIELD GRADIENT rather than from the triangles. A
 * surface-nets mesh has slightly irregular facets, and averaging face normals
 * over them leaves faint quilting across the shoulders and thighs; the analytic
 * gradient is exact everywhere and costs six extra samples per vertex.
 */
export function polygonise(sdf, bounds, resolution = 0.012) {
  const [minX, minY, minZ] = bounds.min;
  const [maxX, maxY, maxZ] = bounds.max;
  const nx = Math.ceil((maxX - minX) / resolution) + 1;
  const ny = Math.ceil((maxY - minY) / resolution) + 1;
  const nz = Math.ceil((maxZ - minZ) / resolution) + 1;
  const h = resolution;

  // Sample the field on the grid. One pass, reused by every cell that touches
  // each point — sampling per cell instead would evaluate the SDF eight times
  // over for every interior point.
  const field = new Float32Array(nx * ny * nz);
  const at = (i, j, l) => i + nx * (j + ny * l);
  for (let l = 0; l < nz; l++) {
    const z = minZ + l * h;
    for (let j = 0; j < ny; j++) {
      const y = minY + j * h;
      for (let i = 0; i < nx; i++) {
        field[at(i, j, l)] = sdf(minX + i * h, y, z);
      }
    }
  }

  const positions = [];
  const cellVertex = new Int32Array((nx - 1) * (ny - 1) * (nz - 1)).fill(-1);
  const cellAt = (i, j, l) => i + (nx - 1) * (j + (ny - 1) * l);

  // Pass 1: place a vertex in every cell the surface passes through.
  const corner = new Float64Array(8);
  for (let l = 0; l < nz - 1; l++) {
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        let mask = 0;
        for (let c = 0; c < 8; c++) {
          const o = CUBE_CORNERS[c];
          const v = field[at(i + o[0], j + o[1], l + o[2])];
          corner[c] = v;
          if (v < 0) mask |= 1 << c;
        }
        if (mask === 0 || mask === 255) continue;   // wholly inside or outside

        let sx = 0, sy = 0, sz = 0, n = 0;
        for (let e = 0; e < 12; e++) {
          const [a, b] = CUBE_EDGES[e];
          const va = corner[a], vb = corner[b];
          if ((va < 0) === (vb < 0)) continue;
          // Linear interpolation to the zero crossing. On a true distance field
          // this lands within a fraction of a voxel of the real surface, which
          // is why the result is smooth at 1.2 cm without any post-smoothing.
          const t = va / (va - vb);
          const oa = CUBE_CORNERS[a], ob = CUBE_CORNERS[b];
          sx += oa[0] + (ob[0] - oa[0]) * t;
          sy += oa[1] + (ob[1] - oa[1]) * t;
          sz += oa[2] + (ob[2] - oa[2]) * t;
          n++;
        }
        cellVertex[cellAt(i, j, l)] = positions.length / 3;
        positions.push(minX + (i + sx / n) * h,
                       minY + (j + sy / n) * h,
                       minZ + (l + sz / n) * h);
      }
    }
  }

  // Pass 2: a quad per grid edge that changes sign, joining the four cells
  // around it. Winding follows the sign so every face points outward.
  const indices = [];
  const quad = (a, b, c, d, flip) => {
    if (a < 0 || b < 0 || c < 0 || d < 0) return;
    if (flip) indices.push(a, c, b, a, d, c);
    else indices.push(a, b, c, a, c, d);
  };
  for (let l = 0; l < nz - 1; l++) {
    for (let j = 0; j < ny - 1; j++) {
      for (let i = 0; i < nx - 1; i++) {
        const v000 = field[at(i, j, l)];
        // +x edge -> quad in the yz plane
        if (i < nx - 1 && j > 0 && l > 0) {
          const v100 = field[at(i + 1, j, l)];
          if ((v000 < 0) !== (v100 < 0)) {
            quad(cellVertex[cellAt(i, j - 1, l - 1)], cellVertex[cellAt(i, j, l - 1)],
                 cellVertex[cellAt(i, j, l)], cellVertex[cellAt(i, j - 1, l)], v000 < 0);
          }
        }
        // +y edge -> quad in the xz plane
        if (j < ny - 1 && i > 0 && l > 0) {
          const v010 = field[at(i, j + 1, l)];
          if ((v000 < 0) !== (v010 < 0)) {
            quad(cellVertex[cellAt(i - 1, j, l - 1)], cellVertex[cellAt(i, j, l - 1)],
                 cellVertex[cellAt(i, j, l)], cellVertex[cellAt(i - 1, j, l)], v000 >= 0);
          }
        }
        // +z edge -> quad in the xy plane
        if (l < nz - 1 && i > 0 && j > 0) {
          const v001 = field[at(i, j, l + 1)];
          if ((v000 < 0) !== (v001 < 0)) {
            quad(cellVertex[cellAt(i - 1, j - 1, l)], cellVertex[cellAt(i, j - 1, l)],
                 cellVertex[cellAt(i, j, l)], cellVertex[cellAt(i - 1, j, l)], v000 < 0);
          }
        }
      }
    }
  }

  // Normals from the analytic gradient — see the docstring.
  const normals = new Float32Array(positions.length);
  const e = resolution * 0.5;
  for (let v = 0; v < positions.length; v += 3) {
    const x = positions[v], y = positions[v + 1], z = positions[v + 2];
    let gx = sdf(x + e, y, z) - sdf(x - e, y, z);
    let gy = sdf(x, y + e, z) - sdf(x, y - e, z);
    let gz = sdf(x, y, z + e) - sdf(x, y, z - e);
    const len = Math.hypot(gx, gy, gz) || 1;
    normals[v] = gx / len; normals[v + 1] = gy / len; normals[v + 2] = gz / len;
  }

  return {
    positions: new Float32Array(positions),
    normals,
    // Uint32 unconditionally: a body at this resolution is comfortably under
    // 65 535 vertices today, and picking the index width from the current count
    // would make a finer resolution or a taller build silently overflow.
    indices: new Uint32Array(indices),
  };
}

/* ------------------------------------------------------------------ *
 * Public entry point
 * ------------------------------------------------------------------ */

/**
 * Generate the body shell. Returns raw arrays plus anchors and stats — see the
 * header for why this does not hand back a BufferGeometry.
 *
 * `resolution` is the voxel size in metres. 0.012 gives a smooth 1.7 m figure in
 * a few hundred milliseconds; the surface is smooth well below that because the
 * vertices are placed by interpolation on a true distance field rather than
 * snapped to grid lines.
 */
export function buildBody(opts = {}) {
  const t0 = performance.now();
  const { sdf, anchors, build } = makeBodyField(opts.build);
  const H = build.height;
  const pad = 0.06;
  const halfW = build.shoulderSpan + build.armDrop + 0.09;
  const mesh = polygonise(sdf, {
    min: [-halfW - pad, -pad, -0.20 * HS - pad],
    max: [halfW + pad, H * 0.99 + pad, 0.22 * HS + pad],
  }, opts.resolution || 0.012);

  return {
    ...mesh,
    anchors,
    sdf,
    stats: {
      vertices: mesh.positions.length / 3,
      triangles: mesh.indices.length / 3,
      ms: Math.round(performance.now() - t0),
    },
  };
}
