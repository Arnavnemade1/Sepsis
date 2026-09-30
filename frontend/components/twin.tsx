"use client";

import { useEffect, useRef } from "react";

import type { Snapshot } from "@/lib/types";

type TwinModule = typeof import("@/engine/twin3d.js");

// One engine per page session: the organ meshes are ~35 MB, so the module (and
// its single WebGL renderer) is loaded once and re-parented between mounts.
let enginePromise: Promise<TwinModule> | null = null;
function engine(): Promise<TwinModule> {
  enginePromise ??= import("@/engine/twin3d.js");
  return enginePromise;
}

/**
 * The anatomical digital twin: real HuBMAP/Visible Human organ meshes (CC BY
 * 4.0) that glow the colour of their organ system's severity, a heart that
 * beats at the patient's heart rate and lungs that breathe at the respiratory
 * rate. Systems nobody is measuring are dimmed to "no data", never shown as a
 * confident healthy glow.
 */
export function BioTwin({ snapshot, className }: { snapshot: Snapshot | null; className?: string }) {
  const host = useRef<HTMLDivElement>(null);
  const mod = useRef<TwinModule | null>(null);

  useEffect(() => {
    let alive = true;
    const el = host.current;
    engine().then((m) => {
      if (!alive || !el) return;
      mod.current = m;
      m.start(el);
      if (snapshot) m.update(snapshot);
    });
    return () => {
      alive = false;
      mod.current?.detach(el);
    };
    // mount/unmount only; data flows through the effect below
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (snapshot) mod.current?.update(snapshot);
  }, [snapshot]);

  return <div ref={host} className={`twin-stage ${className ?? ""}`} aria-label="Anatomical digital twin" role="img" />;
}
