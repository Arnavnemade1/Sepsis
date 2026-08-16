import { Composition } from "remotion";

import { Explainer } from "./Explainer";
import { FPS, W, H, DURATION } from "./constants";
import { headline, operatingPoint } from "@/lib/results";

/**
 * Registered compositions. Consumed by the Remotion CLI (studio / render);
 * the website imports <Explainer> directly and never touches this file.
 *
 * The default props come from the same data/results.json the research page
 * reads, so the video literally cannot state a number the evaluation does not.
 */
export function RemotionRoot() {
  return (
    <Composition
      id="explainer"
      component={Explainer}
      durationInFrames={DURATION}
      fps={FPS}
      width={W}
      height={H}
      defaultProps={{
        auroc: headline.production.auroc,
        ece: headline.production.ece,
        leadHours: headline.production.medianLeadHours,
        threshold: operatingPoint.high,
      }}
    />
  );
}
