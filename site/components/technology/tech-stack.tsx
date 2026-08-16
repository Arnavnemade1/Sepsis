"use client";

import { useEffect, useState } from "react";
import { useReducedMotion } from "motion/react";

import { LayerScroller, LayerList, type Layer } from "./layer-scroller";

/**
 * Chooses between the scroll-scrubbed display and a plain list.
 *
 * Falls back to the list when the visitor has asked for reduced motion, or is
 * on Save-Data / a genuinely slow link — the same bandwidth discipline the hero
 * uses. Renders the list on the server pass too, so the eight layers are in the
 * HTML for crawlers and for anyone whose JS never arrives.
 */
export function TechStack({ layers }: { layers: Layer[] }) {
  const reduced = useReducedMotion();
  const [scrub, setScrub] = useState(false);

  useEffect(() => {
    if (reduced) return;
    const conn = (
      navigator as Navigator & {
        connection?: { saveData?: boolean; effectiveType?: string };
      }
    ).connection;
    const tooSlow =
      conn?.saveData === true || /^(slow-2g|2g)$/.test(conn?.effectiveType ?? "");
    if (!tooSlow) setScrub(true);
  }, [reduced]);

  if (!scrub) return <LayerList layers={layers} />;
  return <LayerScroller layers={layers} />;
}
