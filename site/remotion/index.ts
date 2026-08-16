import { registerRoot } from "remotion";

import { RemotionRoot } from "./Root";

/**
 * Remotion CLI entry point — `npm run video:studio` and `npm run video:render`.
 *
 * The Next.js app NEVER imports this file. Remotion's own bundler
 * (@remotion/bundler) does not work inside Next, so MP4 rendering can never be
 * part of `next build`; the two share only the plain-React composition
 * components, which is exactly the seam that keeps both working.
 */
registerRoot(RemotionRoot);
