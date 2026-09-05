/* ASCII silhouette of the body field — front and side, side by side.
 *
 * Proportions are the thing that was wrong, and checking them through the
 * browser costs a reload, a 15 s load and a screenshot per attempt. Sampling the
 * SDF directly costs about 40 ms, so the shape can be iterated properly instead
 * of guessed at twice and shipped.
 *
 *   node scripts/preview_body.mjs [rows]
 */
import { makeBodyField } from "../webui/twin3d_body.js";

const rows = Number(process.argv[2] || 56);
const { sdf, anchors, build } = makeBodyField();
const H = build.height;

// Character ramp by depth inside the surface, so the render shows mass and not
// just an outline — a silhouette alone hides that the waist never narrows.
const RAMP = " .:-=+*#%@";
const shade = (d) => {
  if (d > 0) return " ";
  const t = Math.min(1, -d / 0.06);
  return RAMP[Math.min(RAMP.length - 1, 1 + Math.floor(t * (RAMP.length - 2)))];
};

const yTop = H * 1.02, yBot = -0.02;
const halfW = 0.42;
// Terminal cells are about twice as tall as they are wide.
const cols = Math.round(rows * (halfW * 2) / (yTop - yBot) * 2.05);

const front = [], side = [];
for (let r = 0; r < rows; r++) {
  const y = yTop - (r + 0.5) * (yTop - yBot) / rows;
  let fr = "", sd = "";
  for (let c = 0; c < cols; c++) {
    const u = -halfW + (c + 0.5) * (halfW * 2) / cols;
    fr += shade(sdf(u, y, 0));
    sd += shade(sdf(0, y, u));
  }
  front.push(fr); side.push(sd);
}

const marks = {
  [anchors.head.y]: "head", [anchors.neck.y]: "neck",
  [anchors.torsoHi]: "shoulder", [anchors.chest.y]: "chest",
  [anchors.hips.y]: "hip", [anchors.wristL.y]: "wrist",
};
const label = (y) => {
  for (const [my, name] of Object.entries(marks)) {
    if (Math.abs(y - Number(my)) < (yTop - yBot) / rows / 2) return name;
  }
  return "";
};

console.log(`front${" ".repeat(Math.max(0, cols - 5))}  side`);
for (let r = 0; r < rows; r++) {
  const y = yTop - (r + 0.5) * (yTop - yBot) / rows;
  const lab = label(y);
  console.log(`${front[r]}  ${side[r]}  ${(y / H).toFixed(2)}${lab ? " <- " + lab : ""}`);
}

// Widths at the landmarks that decide whether a figure reads as human.
// The CENTRAL contiguous run, not the total extent. Measuring extent counts the
// arms, so a neck came back 0.407 m wide when the neck itself was fine and the
// deltoids either side of it were what the ruler was hitting.
const widthAt = (y) => {
  const N = 600, step = (halfW * 2) / N;
  if (sdf(0, y, 0) >= 0) return 0;                 // nothing on the centreline
  let lo = 0, hi = 0;
  for (let i = 1; i <= N / 2; i++) { const x = -i * step; if (sdf(x, y, 0) >= 0) break; lo = x; }
  for (let i = 1; i <= N / 2; i++) { const x = i * step; if (sdf(x, y, 0) >= 0) break; hi = x; }
  return hi - lo;
};
// How far out the arms reach, reported separately so the two are not confused.
const spanAt = (y) => {
  let lo = null, hi = null;
  for (let i = 0; i <= 600; i++) {
    const x = -halfW + i * (halfW * 2) / 600;
    if (sdf(x, y, 0) < 0) { if (lo === null) lo = x; hi = x; }
  }
  return lo === null ? 0 : hi - lo;
};
const heads = H / 7.5;
// Is there daylight between the ribcage and the arm? A body whose arms are
// welded to its sides reads as a gingerbread man however good the proportions
// are, and the total-width figures cannot show it — a fused arm just makes the
// "body" measurement wider.
const gapAt = (y) => {
  const N = 900, step = (halfW * 2) / N;
  let inside = sdf(0, y, 0) < 0, edges = [];
  for (let i = 1; i <= N / 2; i++) {
    const x = i * step, now = sdf(x, y, 0) < 0;
    if (now !== inside) { edges.push(x); inside = now; }
  }
  // torso edge, then arm inner edge: a gap exists when there are >= 2 crossings
  return edges.length >= 2 ? { gap: +(edges[1] - edges[0]).toFixed(3), torso: +edges[0].toFixed(3) }
                           : { gap: 0, torso: +(edges[0] || 0).toFixed(3) };
};
console.log("\narmpit / arm separation (half-body, +x side)");
for (const [name, yf] of [["armpit  ", 0.762], ["chest   ", 0.720], ["waist   ", 0.620], ["hip     ", 0.560]]) {
  const g2 = gapAt(yf * H);
  console.log(`  ${name} y/H ${yf}  torso edge ${g2.torso} m  gap to arm ${g2.gap} m` +
              (g2.gap > 0.008 ? "  ok" : "  <- FUSED"));
}

console.log(`\nstature ${H} m  ·  head unit ${heads.toFixed(3)} m  ·  ${(H / heads).toFixed(1)} heads tall`);
for (const [name, y, want] of [
  ["shoulders", anchors.torsoHi, "0.38-0.45"],
  ["chest    ", anchors.chest.y, "0.28-0.34"],
  ["waist    ", (anchors.chest.y + anchors.hips.y) / 2, "0.26-0.31"],
  ["hips     ", anchors.hips.y, "0.31-0.37"],
  ["neck     ", anchors.neck.y, "0.11-0.14"],
]) {
  console.log(`  ${name} y=${y.toFixed(3)}  body ${widthAt(y).toFixed(3)} m` +
              `  full span ${spanAt(y).toFixed(3)} m   (adult body ${want})`);
}
