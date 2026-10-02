"use client";

import { Heart, RotateCcw, Wind } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";

import { engine } from "@/components/twin";
import { cx, Panel } from "@/components/ui";
import type { Body, RoomKind, SystemStatus } from "@/lib/rooms";

type TwinModule = Awaited<ReturnType<typeof engine>>;

const STATUS_HEX: Record<SystemStatus, string> = { CRITICAL: "#e5484d", DYSFUNCTION: "#f5762e", STRESS: "#eeb12a", NORMAL: "#868d97", NO_DATA: "#4b4f56" };
const STATUS_INK: Record<SystemStatus, string> = { CRITICAL: "#ff6b70", DYSFUNCTION: "#ff9357", STRESS: "#f4c55a", NORMAL: "#b1b7bf", NO_DATA: "#6c737e" };
const STATUS_LABEL: Record<SystemStatus, string> = { CRITICAL: "Critical", DYSFUNCTION: "Dysfunction", STRESS: "Stress", NORMAL: "Normal", NO_DATA: "Not measured" };
const RANK: Record<SystemStatus, number> = { CRITICAL: 4, DYSFUNCTION: 3, STRESS: 2, NORMAL: 1, NO_DATA: 0 };
// the order systems are read in, top of the body to the bottom
const ORDER = ["brain", "lungs", "heart", "immune", "liver", "metabolic", "kidneys", "blood"];

/**
 * The patient's body, lit by this room's own detections: every organ system glows the
 * colour of its worst finding, the heart beats at the room's pulse and the lungs breathe
 * at its respiratory rate. A system nothing in the room measures stays dark grey, "not
 * measured", never a confident healthy glow. Stressed systems are pinned with their
 * reading; hover a system to bring its organs forward, select it to see only its findings.
 */
export function RoomBody({ body, kind, system, onSystem }: {
  body: Body | null | undefined;
  kind: RoomKind | undefined;
  system: string | null;
  onSystem: (region: string | null) => void;
}) {
  const host = useRef<HTMLDivElement>(null);
  const mod = useRef<TwinModule | null>(null);
  const onSystemRef = useRef(onSystem);
  onSystemRef.current = onSystem;
  const [hover, setHover] = useState<string | null>(null);
  const [load, setLoad] = useState<{ organs: number; total: number; ready: boolean }>({ organs: 0, total: 11, ready: false });

  const regions = body?.regions ?? {};
  const pins = useMemo(
    () =>
      Object.entries(regions)
        .filter(([, r]) => r.monitored && RANK[r.status] >= 2)
        .sort(([, a], [, b]) => b.severity - a.severity)
        .slice(0, 5)
        .map(([region, r]) => ({ region, label: r.label, sub: r.reading ?? "", status: r.status as "CRITICAL" | "DYSFUNCTION" | "STRESS" })),
    [regions],
  );

  // one engine per page session (the organ meshes are ~18 MB), re-parented here
  useEffect(() => {
    let alive = true;
    const el = host.current;
    let poll: ReturnType<typeof setInterval> | undefined;
    engine().then((m) => {
      if (!alive || !el) return;
      mod.current = m;
      m.start(el);
      m.setHudEnabled(false);
      m.onHover((r) => setHover(r));
      m.onPin((r, how) => (how === "click" ? onSystemRef.current(r) : setHover(r)));
      m.onPick((r) => r && onSystemRef.current(r));
      poll = setInterval(() => {
        const p = m.progress();
        setLoad({ organs: p.organs, total: p.total, ready: p.ready });
        if (p.ready) clearInterval(poll);
      }, 400);
    });
    return () => {
      alive = false;
      clearInterval(poll);
      const m = mod.current;
      if (m) {
        m.onHover(null);
        m.onPin(null);
        m.onPick(null);
        m.setPins([]);
        m.setHighlight(null);
        m.resetView();                          // the engine is shared: leave it framed for the next page
        m.setHudEnabled(true);
        m.detach(el);
      }
    };
  }, []);

  // the room's state into the twin, every poll
  useEffect(() => {
    const m = mod.current;
    if (!m || !body) return;
    // the twin's wrist marker reads `source`: a synthetic room's feed is the simulator's
    m.update({ source: kind === "synthetic" ? "simulator" : kind === "live" ? "wearable" : kind, body_state: { regions: body.regions, channels: body.channels }, vitals: body.vitals, infection: body.infection });
    m.setPins(pins);
  }, [body, kind, pins, load.ready]);

  // a selected or hovered system brings its organs forward; a selected one also draws the camera in
  useEffect(() => {
    mod.current?.setHighlight(hover ?? system);
  }, [hover, system]);
  useEffect(() => {
    mod.current?.focusRegion(system);
  }, [system, load.ready]);

  const hr = body?.vitals.heart_rate ?? null;
  const rr = body?.vitals.respiratory_rate ?? null;
  const rows = ORDER.filter((r) => regions[r]).map((r) => [r, regions[r]] as const);
  const lit = rows.filter(([, r]) => RANK[r.status] >= 2).length;
  const measured = rows.filter(([, r]) => r.monitored).length;

  return (
    <Panel
      title="Body"
      meta={body ? `${lit ? `${lit} of ${measured} measured systems under stress` : `All ${measured} measured systems within range`} · ${rows.length - measured} not measured here` : "Organ systems lit by this room's detections"}
      flush
    >
      <div className="twin-viewport relative m-2 h-[460px] sm:h-[520px]">
        <div ref={host} className="twin-stage" aria-label="Anatomical twin of this patient" role="img" />

        {!load.ready && (
          <div className="pointer-events-none absolute inset-0 z-[7] grid place-items-center">
            <div className="rounded-md border border-line bg-s1/90 px-3 py-2 text-center">
              <div className="text-[12px] text-ink">Loading anatomy</div>
              <div className="num mt-0.5 font-mono text-[11px] text-dim">
                {load.organs} of {load.total} organs · HuBMAP CCF, CC BY 4.0
              </div>
            </div>
          </div>
        )}

        {body?.infection && (
          <div className="pointer-events-none absolute left-2.5 top-2.5 z-[6] max-w-[40%] rounded-md border border-dys/40 bg-[rgb(29_32_36/.88)] px-2.5 py-1.5 backdrop-blur-sm sm:max-w-[60%]">
            <div className="text-[10.5px] font-medium text-dys-ink">Suspected source</div>
            <div className="truncate text-[12px] font-medium text-ink">{body.infection.site}</div>
            <div className="hidden text-[10.5px] text-dim sm:block">from {body.infection.source}</div>
          </div>
        )}

        <div className="pointer-events-none absolute right-2.5 top-2.5 z-[6] flex flex-col items-end gap-1 text-[10.5px] text-dim">
          {(["CRITICAL", "DYSFUNCTION", "STRESS", "NO_DATA"] as const).map((s) => (
            <span key={s} className="inline-flex items-center gap-1.5">
              {STATUS_LABEL[s]}
              <span className="h-2 w-2 rounded-full" style={{ background: STATUS_HEX[s] }} />
            </span>
          ))}
        </div>

        <div className="pointer-events-none absolute bottom-2.5 left-2.5 z-[6] flex items-center gap-3 rounded-md border border-line bg-[rgb(29_32_36/.86)] px-2.5 py-1.5 backdrop-blur-sm">
          <span className="inline-flex items-center gap-1.5" title="The heart beats at the room's pulse">
            <Heart size={13} className={cx("text-high", hr != null && "tw-beat")} style={hr ? ({ "--beat": `${(60 / Math.max(30, hr)).toFixed(2)}s` } as React.CSSProperties) : undefined} fill="currentColor" />
            <span className="num font-mono text-[12px] text-ink">{hr != null ? Math.round(hr) : "—"}</span>
            <span className="text-[10.5px] text-dim">/min</span>
          </span>
          <span className="h-3 w-px bg-line-bright" />
          <span className="inline-flex items-center gap-1.5" title="The lungs breathe at the room's respiratory rate">
            <Wind size={13} className="text-muted" />
            <span className="num font-mono text-[12px] text-ink">{rr != null ? Math.round(rr) : "—"}</span>
            <span className="text-[10.5px] text-dim">/min</span>
          </span>
        </div>

        <div className="absolute bottom-2.5 right-2.5 z-[6] flex items-center gap-2">
          <span className="pointer-events-none hidden text-[10.5px] text-faint sm:inline">Drag to rotate · scroll to zoom</span>
          <button
            onClick={() => {
              mod.current?.resetView();
              onSystem(null);
            }}
            className="grid h-7 w-7 place-items-center rounded-md border border-line bg-[rgb(29_32_36/.86)] text-muted transition-colors hover:text-ink"
            aria-label="Reset the view"
            title="Reset the view"
          >
            <RotateCcw size={13} />
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-px border-t border-line bg-line sm:grid-cols-2">
        {rows.map(([region, r]) => {
          const on = system === region || hover === region;
          return (
            <button
              key={region}
              onMouseEnter={() => setHover(region)}
              onMouseLeave={() => setHover(null)}
              onClick={() => onSystem(system === region ? null : region)}
              className={cx("relative flex min-w-0 items-center gap-2.5 bg-s1 py-2 pl-3.5 pr-3 text-left transition-colors hover:bg-s2", on && "bg-s2")}
              aria-pressed={system === region}
            >
              <span className="absolute inset-y-0 left-0 w-[3px]" style={{ background: RANK[r.status] >= 2 ? STATUS_HEX[r.status] : "transparent" }} />
              <span className="h-2 w-2 shrink-0 rounded-full" style={{ background: STATUS_HEX[r.status], opacity: r.monitored ? 1 : 0.7 }} />
              <span className="min-w-0 flex-1">
                <span className="flex items-baseline gap-2">
                  <span className={cx("text-[12.5px]", r.monitored ? "text-ink" : "text-dim")}>{r.label}</span>
                  <span className="text-[11px]" style={{ color: STATUS_INK[r.status] }}>
                    {STATUS_LABEL[r.status]}
                  </span>
                </span>
                <span className="block truncate font-mono text-[11px] text-dim">{r.reading ?? (kind === "replay" ? "not in this dataset" : "nothing in this room measures it")}</span>
              </span>
              {r.findings.length > 0 && <span className="num shrink-0 font-mono text-[11px] text-dim">{r.findings.length}</span>}
            </button>
          );
        })}
      </div>
    </Panel>
  );
}
