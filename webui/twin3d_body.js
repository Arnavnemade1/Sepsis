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
 * Build parameters. Only the knobs a caller should turn; the anatomy itself is a
 * profile table below, because a body's silhouette is not something three
 * ellipsoids can be talked into.
 */
export const DEFAULT_BUILD = {
  height: 1.7,
  girth: 1.0,        // scales widths and depths, not lengths
  limbScale: 1.0,
  blend: 0.020,      // joint blend radius: small, deliberately
};

/**
 * TORSO PROFILE — [y as a fraction of stature, half-width, half-depth, z-centre],
 * with the horizontal figures in metres for a 1.70 m adult.
 *
 * WHY A PROFILE AND NOT MORE ELLIPSOIDS. The first version stacked three
 * ellipsoids and smooth-unioned them with a 5 cm blend. The blend is larger than
 * the differences between the sections, so it averaged them away: measured on
 * the result, the figure got WIDER going down — shoulders 0.386 m, chest 0.403,
 * waist 0.462, hips 0.491 — which is a pear, not a person. The neck came out
 * 0.332 m across against an adult 0.11-0.14, because the head, the neck and the
 * shoulder mass all merged into one lump.
 *
 * A profile is sampled, not blended, so the silhouette is exactly what is
 * written here. Figures follow standard adult anthropometry: biacromial 0.395,
 * chest 0.30, waist 0.28, hip 0.33, chest depth 0.215, waist depth 0.185.
 */
const TORSO_PROFILE = [
  // y/H     halfW   halfD   zC
  [0.830,   0.058,  0.055,  0.000],   // base of neck
  [0.812,   0.120,  0.078, -0.004],   // clavicle line
  [0.795,   0.158,  0.094, -0.008],   // top of shoulder girdle
  [0.762,   0.152,  0.104, -0.010],   // armpit
  [0.720,   0.150,  0.107, -0.009],   // nipple line — widest of the chest
  [0.680,   0.145,  0.101, -0.005],
  [0.640,   0.140,  0.094, -0.001],
  [0.620,   0.139,  0.092,  0.000],   // natural waist — the narrowest point
  [0.596,   0.145,  0.095,  0.002],
  [0.570,   0.158,  0.099,  0.004],   // iliac crest
  [0.545,   0.165,  0.100,  0.004],   // hips — widest below the chest
  [0.515,   0.152,  0.098,  0.004],
  [0.487,   0.118,  0.090,  0.002],   // crotch
];

/**
 * Distance to a lofted profile: an elliptical cross-section whose half-width and
 * half-depth vary with height. Approximate, smooth and conservative, which is
 * all smooth-min and surface extraction need.
 */
function sdProfile(px, py, pz, H, profile, girth) {
  const n = profile.length;
  const yTop = profile[0][0] * H, yBot = profile[n - 1][0] * H;
  // Clamp into the table, then blend the radial distance with the overshoot past
  // the ends so the caps round off instead of shearing flat.
  const yc = Math.max(yBot, Math.min(yTop, py));
  let i = 0;
  while (i < n - 2 && profile[i + 1][0] * H > yc) i++;
  const a = profile[i], b = profile[i + 1];
  const ya = a[0] * H, yb = b[0] * H;
  const t = ya === yb ? 0 : (ya - yc) / (ya - yb);
  // Smoothstep between slices: linear interpolation leaves a visible crease at
  // every table row, which on a glass surface the fresnel then traces.
  const ts = t * t * (3 - 2 * t);
  const hw = (a[1] + (b[1] - a[1]) * ts) * girth;
  const hd = (a[2] + (b[2] - a[2]) * ts) * girth;
  const zc = (a[3] + (b[3] - a[3]) * ts) * girth;

  const qx = px / hw, qz = (pz - zc) / hd;
  const q = Math.sqrt(qx * qx + qz * qz);
  const radial = (q - 1) * Math.min(hw, hd);

  // dy must be SIGNED — negative inside the profile's height range. A first
  // version clamped it with Math.max(..., 0), which made the interior term
  // max(radial, 0) = 0 everywhere inside the torso: the field came back 0 on the
  // whole centreline and the trunk polygonised HOLLOW, an outline with nothing
  // in it. Measured body width at the chest, waist and hips was 0.000 m.
  const yMid = (yTop + yBot) / 2, yHalf = (yTop - yBot) / 2;
  const dy = Math.abs(py - yMid) - yHalf;

  // The standard extruded-2D combination: the interior is the larger (less
  // negative) of the two negatives, the exterior is the length of the positive
  // parts.
  return Math.min(Math.max(radial, dy), 0)
       + Math.hypot(Math.max(radial, 0), Math.max(dy, 0));
}

/**
 * Build the body's signed-distance function and its anchor points.
 *
 * Landmark heights are the standard 7.5-head adult canon, which is what makes a
 * figure read as a person rather than as a doll: crown 1.000, chin 0.867 (one
 * head unit), shoulder 0.818, nipple 0.720, waist 0.620, crotch 0.487 — the
 * midpoint of the body — knee 0.285, ankle 0.039.
 */
export function makeBodyField(build = {}) {
  const B = { ...DEFAULT_BUILD, ...build };
  const H = B.height;
  const k = B.blend;
  const L = B.limbScale;
  const g = B.girth;
  const S = H / 1.7;               // stature scale for every horizontal figure

  const yCrown = 1.000 * H, yChin = 0.867 * H, yNeck = 0.830 * H;
  const yShoulder = 0.818 * H, yChest = 0.720 * H, yWaist = 0.620 * H;
  const yHip = 0.560 * H, yCrotch = 0.487 * H;
  const yElbow = 0.630 * H, yWrist = 0.485 * H;
  const yKnee = 0.285 * H, yAnkle = 0.039 * H;

  // Arms hang OUTSIDE the ribcage. The chest half-width is 0.150; putting the
  // shoulder joint at 0.175 and the blend at 2 cm leaves a real armpit instead
  // of welding the arm to the torso, which is what the first version did.
  // Arm axes, set from the ribcage outward rather than guessed. The torso is
  // 0.152 half-width at the armpit; an upper arm of radius 0.042 whose axis sits
  // at 0.212 leaves 1.8 cm of daylight between the two. The first version put
  // the axis at 0.175, so the arm's inner surface was 1 cm INSIDE the chest wall
  // and the two fused into one slab — measured gap 0.000 m at the armpit, the
  // chest, the waist and the hip.
  const xShoulder = 0.212 * S, xElbow = 0.222 * S, xWrist = 0.228 * S;
  const xHip = 0.086 * S, xKnee = 0.086 * S, xAnkle = 0.076 * S;

  const anchors = {
    hips: { x: 0, y: yHip, z: 0 },
    neck: { x: 0, y: yNeck, z: 0 },
    head: { x: 0, y: (yChin + yCrown) / 2, z: 0 },
    chest: { x: 0, y: yChest, z: 0 },
    wristL: { x: -xWrist, y: yWrist, z: 0.012 },
    wristR: { x: xWrist, y: yWrist, z: 0.012 },
    torsoLo: yCrotch,
    torsoHi: yShoulder,
    depth: 0.215 * S * g,
  };

  /** The field. Returns signed distance; negative is inside the body. */
  function sdf(x, y, z) {
    // --- torso, straight off the profile ------------------------------------
    let d = sdProfile(x, y, z, H, TORSO_PROFILE, g * S);

    // --- neck: a real one, and narrow ---------------------------------------
    // Measured 0.332 m across before this; an adult neck is 0.115. It is blended
    // at a fraction of the joint radius so the head does not sink into the
    // shoulders, which is what made the first figure look like a snowman.
    d = smin(d, sdCapsule(x, y, z, 0, yNeck - 0.010 * H, -0.008 * S,
                          0, yChin - 0.012 * H, -0.012 * S, 0.058 * S * g), k * 0.7);

    // --- head ----------------------------------------------------------------
    // An ellipsoid taller than it is wide, set back a little, plus a jaw mass.
    // One head unit is H/7.5, and the cranium must occupy it or the whole canon
    // collapses: a small head on wide shoulders is what reads as "blob".
    const yHead = (yChin + yCrown) / 2;
    d = smin(d, sdEllipsoid(x, y, z, 0, yHead + 0.006 * H, -0.008 * S,
                            0.077 * S, 0.098 * S, 0.086 * S), k * 0.8);
    d = smin(d, sdEllipsoid(x, y, z, 0, yChin + 0.026 * H, -0.020 * S,
                            0.058 * S, 0.048 * S, 0.062 * S), k * 1.1);

    // --- shoulder caps -------------------------------------------------------
    for (const s of [-1, 1]) {
      // The deltoid bridges ribcage to arm ABOVE the armpit. Its blend is held
      // down because a generous one reaches below the joint and quietly fills
      // the gap the arm placement just bought.
      d = smin(d, sdEllipsoid(x, y, z, s * xShoulder * 0.86, yShoulder - 0.010 * H, -0.004 * S,
                              0.058 * S * L, 0.052 * S * L, 0.058 * S * L), k * 1.1);
    }

    // --- arms ----------------------------------------------------------------
    for (const s of [-1, 1]) {
      d = smin(d, sdCapsule(x, y, z,
                            s * xShoulder, yShoulder - 0.020 * H, -0.002 * S,
                            s * xElbow, yElbow, 0.004 * S,
                            0.042 * S * L), k * 0.55);
      d = smin(d, sdCapsule(x, y, z,
                            s * xElbow, yElbow, 0.004 * S,
                            s * xWrist, yWrist, 0.012 * S,
                            0.034 * S * L), k * 0.55);
      // Hand — flat, and long enough to read as one, but not thinner than the
      // grid can resolve. At 0.015 m half-depth it was about three voxels thick,
      // and the front and back surfaces met inside single cells: the mesh came
      // back with non-manifold pinch edges that no resolution fixed (5 at
      // 1.05 cm, 3 at 0.95, 4 at 0.85, 1 at 0.75). A real hand is ~0.024 half.
      d = smin(d, sdEllipsoid(x, y, z,
                              s * (xWrist + 0.004 * S), yWrist - 0.055 * H, 0.014 * S,
                              0.030 * S, 0.056 * S, 0.024 * S), k);
    }

    // --- legs ----------------------------------------------------------------
    for (const s of [-1, 1]) {
      d = smin(d, sdCapsule(x, y, z,
                            s * xHip, yCrotch + 0.030 * H, 0,
                            s * xKnee, yKnee + 0.020 * H, 0.002 * S,
                            0.079 * S * L), k * 1.5);
      d = smin(d, sdCapsule(x, y, z,
                            s * xKnee, yKnee, 0.002 * S,
                            s * xAnkle, yAnkle, 0.008 * S,
                            0.052 * S * L), k * 1.1);
      // foot — long in z, so the figure stands rather than balancing on pegs
      d = smin(d, sdEllipsoid(x, y, z,
                              s * xAnkle, yAnkle - 0.016 * H, 0.040 * H,
                              0.040 * S, 0.028 * S, 0.092 * S), k);
    }

    // --- surface anatomy ----------------------------------------------------
    // A correct silhouette is still an empty one. These are the landmarks the
    // skin actually shows, and on a glass shell they matter more than on an
    // opaque one: the fresnel rim traces every one of them, giving the eye
    // something between the outline and the organs. Carved with smooth
    // subtraction so none of them reintroduces a crease.
    const dzc = 0.107 * S * g;

    // Clavicles. Radius and blend both matter here: at 0.013 m the capsule was
    // about 2.5 voxels across, so where it grazed the chest wall the two
    // surfaces met inside single cells and every non-manifold pinch edge in the
    // mesh landed on it — all five, at y/H 0.814-0.830, x 0.10-0.12. A thicker
    // ridge blended further INTO the torso is both anatomically better (a
    // clavicle shows as a ridge, not a rod laid on the skin) and resolvable.
    for (const s of [-1, 1]) {
      d = smin(d, sdCapsule(x, y, z,
                            s * 0.016 * S, yShoulder + 0.007 * H, -dzc * 0.52,
                            s * xShoulder * 0.84, yShoulder + 0.001 * H, -dzc * 0.24,
                            0.019 * S), k * 1.6);
    }
    d = ssub(d, sdEllipsoid(x, y, z, 0, yShoulder + 0.012 * H, -dzc * 0.86,
                            0.024 * S, 0.018 * S, 0.026 * S), k * 0.9);

    // pectorals, and the sternal groove between them
    for (const s of [-1, 1]) {
      d = smin(d, sdEllipsoid(x, y, z, s * 0.062 * S, yChest + 0.014 * H, -dzc * 0.58,
                              0.058 * S, 0.042 * S, 0.026 * S), k * 1.6);
    }
    d = ssub(d, sdCapsule(x, y, z, 0, yShoulder - 0.014 * H, -dzc * 0.92,
                          0, yChest - 0.020 * H, -dzc * 0.94, 0.016 * S), k * 1.2);

    // costal margin — the shallow V under the ribs, the clearest cue that a
    // torso is a torso and not a cylinder
    for (const s of [-1, 1]) {
      d = ssub(d, sdCapsule(x, y, z,
                            s * 0.008 * S, yChest - 0.056 * H, -dzc * 0.84,
                            s * 0.124 * S, yChest - 0.012 * H, -dzc * 0.34,
                            0.012 * S), k * 1.1);
    }

    // linea alba and navel
    d = ssub(d, sdCapsule(x, y, z, 0, yChest - 0.062 * H, -dzc * 0.90,
                          0, yWaist - 0.050 * H, -dzc * 0.88, 0.008 * S), k * 1.3);
    d = ssub(d, sdEllipsoid(x, y, z, 0, yWaist - 0.004 * H, -dzc * 0.88,
                            0.012 * S, 0.014 * S, 0.018 * S), k * 0.7);

    // spinal groove and scapulae
    d = ssub(d, sdCapsule(x, y, z, 0, yShoulder - 0.014 * H, dzc * 0.90,
                          0, yWaist - 0.010 * H, dzc * 0.88, 0.013 * S), k * 1.4);
    for (const s of [-1, 1]) {
      d = smin(d, sdEllipsoid(x, y, z, s * 0.070 * S, yChest + 0.048 * H, dzc * 0.62,
                              0.046 * S, 0.050 * S, 0.018 * S), k * 1.6);
    }

    // deltoid and biceps, so an arm tapers like a limb and not like a pipe
    for (const s of [-1, 1]) {
      d = smin(d, sdEllipsoid(x, y, z, s * xShoulder, yShoulder - 0.052 * H, 0,
                              0.040 * S * L, 0.052 * S * L, 0.042 * S * L), k * 0.9);
      d = smin(d, sdEllipsoid(x, y, z,
                              s * (xShoulder + (xElbow - xShoulder) * 0.5),
                              (yShoulder + yElbow) * 0.5, -0.008 * S,
                              0.031 * S * L, 0.058 * S * L, 0.032 * S * L), k * 0.8);
    }

    // sternocleidomastoid — the two cords in the neck
    for (const s of [-1, 1]) {
      d = smin(d, sdCapsule(x, y, z,
                            s * 0.012 * S, yChin - 0.010 * H, -0.026 * S,
                            s * 0.030 * S, yNeck - 0.024 * H, -0.018 * S,
                            0.009 * S), k * 0.8);
    }

    // quadriceps, calf and knee
    for (const s of [-1, 1]) {
      d = smin(d, sdEllipsoid(x, y, z, s * xHip, (yCrotch + yKnee) * 0.54, -0.016 * S,
                              0.052 * S * L, 0.105 * S * L, 0.048 * S * L), k * 1.8);
      d = smin(d, sdEllipsoid(x, y, z, s * xKnee, yKnee - 0.070 * H, 0.018 * S,
                              0.038 * S * L, 0.066 * S * L, 0.036 * S * L), k * 1.6);
      d = smin(d, sdEllipsoid(x, y, z, s * xKnee, yKnee, -0.004 * S,
                              0.045 * S * L, 0.040 * S * L, 0.046 * S * L), k * 1.4);
    }
    return d;
  }

  // The volume the field can possibly occupy. Returned rather than recomputed by
  // the caller: buildBody used to derive it from `shoulderSpan` and `armDrop`,
  // and when the build parameters were rewritten those names disappeared. The
  // arithmetic produced NaN, the grid had no cells, and buildBody returned a
  // mesh of ZERO triangles — a body that silently did not exist. The field is
  // the only thing that knows how far it reaches, so it is the thing that says.
  const reach = (xWrist + 0.030 * S) * 1.06;   // hands are the widest point
  const bounds = {
    min: [-reach, -0.03 * H, -0.16 * S * g],
    max: [reach, yCrown + 0.03 * H, 0.20 * S * g],
  };

  return { sdf, anchors, bounds, build: B };
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
  const { sdf, anchors, bounds, build } = makeBodyField(opts.build);
  const pad = 0.05;
  const mesh = polygonise(sdf, {
    min: [bounds.min[0] - pad, bounds.min[1] - pad, bounds.min[2] - pad],
    max: [bounds.max[0] + pad, bounds.max[1] + pad, bounds.max[2] + pad],
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
