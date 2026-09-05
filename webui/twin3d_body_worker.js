/* The body generator, off the main thread.
 *
 * Polygonising the field takes ~1.1 s in a browser. Run inline that froze the
 * whole console while the twin came up — the panels stopped painting and input
 * stopped responding, which on a clinical dashboard is the one thing a decorative
 * load must never do. The arrays come back as transferables, so handing them over
 * costs nothing beyond the message itself.
 */
import { buildBody } from "./twin3d_body.js";

self.onmessage = (ev) => {
  try {
    const built = buildBody(ev.data || {});
    self.postMessage({
      ok: true,
      positions: built.positions,
      normals: built.normals,
      indices: built.indices,
      anchors: built.anchors,
      stats: built.stats,
    }, [built.positions.buffer, built.normals.buffer, built.indices.buffer]);
  } catch (err) {
    self.postMessage({ ok: false, error: String((err && err.stack) || err) });
  }
};
