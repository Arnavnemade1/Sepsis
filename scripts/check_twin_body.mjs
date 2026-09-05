/* Run the body generator and assert the mesh is a body.
 *
 * WHY PARSING IS NOT ENOUGH: `make check-webui` catches syntax, and the defect
 * that actually shipped was a ReferenceError — a rename of `(H / 1.7)` to a
 * closure-scoped `HS` reached two lines in `buildBody`, where that name does not
 * exist. The file parsed perfectly. `buildBody` threw on every call, the worker
 * reported the failure, the caller logged it and returned, and the visible
 * symptom was that the body silently never appeared while the organs loaded into
 * empty space.
 *
 * So this EXECUTES the generator. It also checks the properties that make the
 * output a usable shell rather than merely an object: watertightness, manifold
 * edges, unit normals, and human proportions.
 */
import { buildBody, makeBodyField } from "../webui/twin3d_body.js";

let failed = 0;
const check = (name, ok, detail = "") => {
  console.log(`  [${ok ? "PASS" : "FAIL"}] ${name}${detail ? " — " + detail : ""}`);
  if (!ok) failed++;
};

console.log("twin body generator");

// Wrapped, because the defect this suite exists for made buildBody THROW. An
// uncaught throw is a failure, but it exits with a stack trace instead of a
// verdict, and a check that cannot say "FAIL" is hard to read in CI output.
let built = null, buildErr = null;
try { built = buildBody({ resolution: 0.0105 }); }
catch (e) { buildErr = e; }

check("buildBody() runs and returns a mesh",
      !!built && built.positions.length > 0 && built.indices.length > 0,
      built ? `${built.stats.triangles.toLocaleString()} tris, ${built.stats.ms} ms`
            : `threw: ${buildErr && buildErr.message}`);
if (!built) {
  console.log("\n1 check(s) FAILED — nothing else can be checked");
  process.exit(1);
}
const pos = built.positions, idx = built.indices, nrm = built.normals;

// Watertight + manifold: every edge of a closed surface is shared by exactly two
// triangles. This is the property the whole SDF approach exists to guarantee, and
// the one the user's complaint was about.
const edges = new Map();
for (let i = 0; i < idx.length; i += 3) {
  const t = [idx[i], idx[i + 1], idx[i + 2]];
  for (let e = 0; e < 3; e++) {
    const a = t[e], b = t[(e + 1) % 3];
    const key = a < b ? `${a}:${b}` : `${b}:${a}`;
    edges.set(key, (edges.get(key) || 0) + 1);
  }
}
let holes = 0, pinches = 0;
for (const c of edges.values()) { if (c === 1) holes++; else if (c !== 2) pinches++; }
check("the surface is watertight (no boundary edges)", holes === 0, `${holes} holes`);
check("...and manifold (no edge shared by 3+ triangles)", pinches === 0, `${pinches} pinches`);

let badN = 0;
for (let i = 0; i < nrm.length; i += 3) {
  if (Math.abs(Math.hypot(nrm[i], nrm[i + 1], nrm[i + 2]) - 1) > 1e-3) badN++;
}
check("every normal is unit length", badN === 0, `${badN} bad`);

const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
for (let i = 0; i < pos.length; i += 3) {
  for (let k = 0; k < 3; k++) {
    lo[k] = Math.min(lo[k], pos[i + k]); hi[k] = Math.max(hi[k], pos[i + k]);
  }
}
const h = hi[1] - lo[1], w = hi[0] - lo[0], d = hi[2] - lo[2];
check("it is the height it was asked for", Math.abs(h - 1.7) < 0.09, `${h.toFixed(3)} m`);
check("proportions are human, not a blob",
      w > 0.45 && w < 0.70 && d > 0.16 && d < 0.34,
      `${w.toFixed(3)} m across, ${d.toFixed(3)} m deep`);

// The field itself: inside must be negative, outside positive. A sign convention
// flipped somewhere would still polygonise, and would produce an inside-out body.
const { sdf, anchors } = makeBodyField();
check("the field is negative inside the chest",
      sdf(0, anchors.chest.y, 0) < 0, sdf(0, anchors.chest.y, 0).toFixed(4));
check("...and positive in free air beside the body",
      sdf(0.9, 1.0, 0) > 0, sdf(0.9, 1.0, 0).toFixed(4));
// The real requirement is that everything crossing the worker boundary survives
// structuredClone. A THREE.Vector3 does not, and the module's whole reason for
// having no three.js import is that it has to run in a worker.
let cloneable = true, cloneErr = "";
try { structuredClone({ anchors, stats: built.stats }); }
catch (e) { cloneable = false; cloneErr = String(e.message || e); }
check("anchors and stats survive structuredClone (worker-safe)",
      cloneable && typeof anchors.hips.x === "number",
      cloneable ? "plain objects only" : cloneErr);

// Parameters must actually parameterise. A build option that is read but never
// applied is the same class of defect as a correction that is documented and not
// applied — and this project has shipped that one three times.
const tall = buildBody({ resolution: 0.02, build: { height: 2.0 } });
let tlo = Infinity, thi = -Infinity;
for (let i = 1; i < tall.positions.length; i += 3) {
  tlo = Math.min(tlo, tall.positions[i]); thi = Math.max(thi, tall.positions[i]);
}
check("the height parameter changes the body", Math.abs((thi - tlo) - 2.0) < 0.11,
      `height 2.0 -> ${(thi - tlo).toFixed(3)} m`);

console.log(failed ? `\n${failed} check(s) FAILED` : "\nall checks passed");
process.exit(failed ? 1 : 0);
