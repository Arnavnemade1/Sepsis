"use client";

import { AnimatePresence, motion } from "motion/react";
import { useMemo, useState } from "react";

import { cx } from "@/components/ui";
import type { Clinician, Layout, Monitor, Occupant, Page } from "@/lib/paging";

import { P_INK, P_RGB, ST_RGB } from "./bits";

/* ------------------------------------------------------------------ */
/* Isometric projection                                                */
/* ------------------------------------------------------------------ */
// A flatter axonometric than true isometric: each floor reads as a wide, distinct
// layer and the stack overlaps less. The close-up floor uses a steeper angle.
const STACK_ANGLE = { fx: 0.93, fy: 0.36 };
const FLOOR_ANGLE = { fx: 0.88, fy: 0.46 };

type Pt = [number, number];
type Pos = { z: number; ox: number; fx?: number; fy?: number };
const proj = (x: number, y: number, s: number, p: Pos): Pt => [
  (x - y) * (p.fx ?? STACK_ANGLE.fx) * s + p.ox,
  (x + y) * (p.fy ?? STACK_ANGLE.fy) * s - p.z,
];
const pts = (a: Pt[]) => a.map(([x, y]) => `${x.toFixed(1)},${y.toFixed(1)}`).join(" ");

function quad(rect: number[], inset: number, s: number, p: Pos): Pt[] {
  const [x, y, w, h] = rect;
  const x0 = x + inset, y0 = y + inset, x1 = x + w - inset, y1 = y + h - inset;
  return [proj(x0, y0, s, p), proj(x1, y0, s, p), proj(x1, y1, s, p), proj(x0, y1, s, p)];
}
const center = (rect: number[], s: number, p: Pos): Pt => proj(rect[0] + rect[2] / 2, rect[1] + rect[3] / 2, s, p);

// structure is monochrome: colour on the plan means a page, a watch or a person.
// Critical-care and emergency units read a shade lighter so the eye finds them.
const ZONE_FILL = (critical: boolean) => (critical ? "rgb(255 255 255 / 0.05)" : "rgb(255 255 255 / 0.025)");

function hash(s: string): number {
  let h = 2166136261;
  for (const ch of s) h = Math.imul(h ^ ch.charCodeAt(0), 16777619);
  return (h >>> 0) / 4294967295;
}

type Props = {
  layout: Layout;
  clinicians: Record<string, Clinician>;
  roomPages: Record<string, Page>;
  pages: Record<string, Page>;
  monitor: Monitor;
  occupancy: Record<string, Occupant>;
  floor: string | "all";
  onFloor: (f: string | "all") => void;
  onRoom: (room: string) => void;
  onClinician?: (id: string) => void;
  selectedPage?: string | null;
};

type Tip = { x: number; y: number; title: string; lines: string[] } | null;
type Spots = Record<string, { floor: string; x: number; y: number }>;

export function FloorView(props: Props) {
  const { floor } = props;
  const [tip, setTip] = useState<Tip>(null);
  return (
    <div className="relative h-full w-full select-none" onMouseLeave={() => setTip(null)}>
      <AnimatePresence mode="wait">
        <motion.div
          key={floor}
          // in flow below lg so the drawing sets the height; mode="wait" means one view at a time
          className="relative lg:absolute lg:inset-0"
          initial={{ opacity: 0, scale: 0.985, y: 8 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          exit={{ opacity: 0, scale: 0.99, y: -6 }}
          transition={{ duration: 0.38, ease: [0.16, 1, 0.3, 1] }}
        >
          {floor === "all" ? <Stack {...props} setTip={setTip} /> : <Single {...props} fid={floor} setTip={setTip} />}
        </motion.div>
      </AnimatePresence>
      {tip && (
        <div className="pop pointer-events-none absolute z-20 max-w-[280px] px-3 py-2" style={{ left: tip.x + 14, top: tip.y + 14 }}>
          <div className="text-[12.5px] font-semibold text-ink">{tip.title}</div>
          {tip.lines.map((l) => (
            <div key={l} className="text-[11.5px] leading-snug text-muted">
              {l}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Where each clinician stands, in plan coordinates                    */
/* ------------------------------------------------------------------ */
function useStaffSpots(layout: Layout, clinicians: Record<string, Clinician>, pages: Record<string, Page>): Spots {
  return useMemo(() => {
    const onCase: Record<string, string> = {};
    for (const p of Object.values(pages)) if (p.status === "accepted" && p.assignee && p.room) onCase[p.assignee] = p.room;
    const byZone: Record<string, string[]> = {};
    for (const c of Object.values(clinicians)) {
      if (!layout.zones[c.zone] || c.zone === "remote") continue;
      (byZone[c.zone] ??= []).push(c.id);
    }
    const spots: Spots = {};
    for (const [zid, ids] of Object.entries(byZone)) {
      const z = layout.zones[zid];
      const [x, y, w, h] = z.rect;
      ids.sort().forEach((id, i) => {
        const room = onCase[id] ? layout.rooms[onCase[id]] : null;
        if (room && room.zone === zid) {
          const [rx, ry, rw, rh] = room.rect;
          spots[id] = { floor: z.floor, x: rx + rw * 0.74, y: ry + rh * 0.3 };
          return;
        }
        const jx = hash(id) * 0.5 + 0.25;
        const jy = hash(id + "y") * 0.4 + 0.3;
        spots[id] = { floor: z.floor, x: x + w * (((jx + i * 0.19) % 0.86) + 0.07), y: y + h * jy };
      });
    }
    return spots;
  }, [layout, clinicians, pages]);
}

/* ------------------------------------------------------------------ */
/* One floor slab                                                      */
/* ------------------------------------------------------------------ */
function Slab({ layout, clinicians, roomPages, monitor, occupancy, onRoom, onClinician, setTip, spots, fid, s, pos, detail, hover, summary }:
  Props & { setTip: (t: Tip) => void; spots: Spots; fid: string; s: number; pos: Pos; detail: boolean; hover?: boolean; summary?: { n: number; top: number | null } }) {
  const W = layout.plate.w, D = layout.plate.d;
  const T = detail ? 12 : 8;
  const drop: Pos = { ...pos, z: pos.z - T - (detail ? 16 : 10) };
  const low: Pos = { ...pos, z: pos.z - T };
  const top = [proj(0, 0, s, pos), proj(W, 0, s, pos), proj(W, D, s, pos), proj(0, D, s, pos)];
  const shadow = [proj(0, 0, s, drop), proj(W, 0, s, drop), proj(W, D, s, drop), proj(0, D, s, drop)];
  const front = [proj(0, D, s, pos), proj(W, D, s, pos), proj(W, D, s, low), proj(0, D, s, low)];
  const side = [proj(W, 0, s, pos), proj(W, D, s, pos), proj(W, D, s, low), proj(W, 0, s, low)];
  const floor = layout.floors.find((f) => f.id === fid)!;
  const tipAt = (e: React.MouseEvent, title: string, lines: string[]) => {
    const host = (e.currentTarget as SVGElement).closest("[data-floorhost]")?.parentElement?.parentElement as HTMLElement | null;
    const r = host?.getBoundingClientRect();
    setTip({ x: e.clientX - (r?.left ?? 0), y: e.clientY - (r?.top ?? 0), title, lines });
  };
  return (
    <g>
      <polygon points={pts(shadow)} fill="rgb(0 0 0 / .28)" style={{ filter: `blur(${detail ? 10 : 6}px)` }} />
      <polygon points={pts(front)} fill="#1d2024" />
      <polygon points={pts(side)} fill="#191b1f" />
      <polygon points={pts(top)} fill="url(#slab-top)" stroke={hover ? "rgb(110 170 255 / .8)" : "rgb(255 255 255 / .18)"} strokeWidth={hover ? 1.4 : 1} style={{ transition: "stroke .2s" }} />
      {floor.zones.map((zid) => {
        const zn = layout.zones[zid];
        return (
          <g key={zid}>
            <polygon points={pts(quad(zn.rect, 0.55, s, pos))} fill={ZONE_FILL(zn.critical)} stroke="rgb(255 255 255 / .18)" strokeWidth={0.7} />
            {zn.rooms.map((rid) => {
              const room = layout.rooms[rid];
              const pg = roomPages[rid];
              const mon = monitor[rid];
              const occ = occupancy[rid];
              let fill = occ ? "rgb(255 255 255 / .09)" : "rgb(255 255 255 / .025)";
              let stroke = "rgb(255 255 255 / .15)";
              let dash: string | undefined;
              let sw = 0.6;
              if (mon?.watch && !pg) {
                fill = "rgb(238 177 42 / .14)";
                stroke = "rgb(238 177 42 / .9)";
                dash = "3 2";
                sw = 1;
              }
              if (pg) {
                fill = `rgb(${P_RGB[pg.priority]} / ${pg.status === "accepted" ? 0.16 : 0.32})`;
                stroke = `rgb(${P_RGB[pg.priority]})`;
                sw = 1.2;
              }
              const q = quad(room.rect, 0.4, s, pos);
              const c = center(room.rect, s, pos);
              const urgent = pg && pg.priority <= 2 && (pg.status === "paging" || pg.status === "review" || pg.status === "unassigned");
              return (
                <g
                  key={rid}
                  className="cursor-pointer"
                  onClick={(e) => {
                    e.stopPropagation();
                    onRoom(rid);
                  }}
                  onMouseMove={(e) =>
                    tipAt(e, `Room ${rid} · ${zn.name}`, [
                      occ ? `${occ.name ?? "Patient"}${occ.age ? `, ${occ.age}${occ.sex ?? ""}` : ""} · ${occ.diagnosis}` : "Empty",
                      ...(mon ? [`Monitored · sepsis risk ${mon.risk != null ? mon.risk.toFixed(2) : "—"}${mon.watch ? " · on watch" : ""}`] : []),
                      ...(pg ? [`P${pg.priority} · ${pg.status === "paging" ? "ringing" : pg.status}: ${pg.text.slice(0, 64)}`] : []),
                      ...(occ?.synthetic ? ["synthetic demo record"] : occ?.monitored ? ["real replayed record"] : []),
                    ])
                  }
                  onMouseLeave={() => setTip(null)}
                >
                  {urgent && <polygon points={pts(q)} fill={`rgb(${P_RGB[pg!.priority]})`} opacity={0.55} filter="url(#room-glow)" className="pointer-events-none" />}
                  <polygon points={pts(q)} fill={fill} stroke={stroke} strokeWidth={sw} strokeDasharray={dash} className="transition-[fill] duration-300 hover:brightness-125" />
                  {pg && (pg.status === "paging" || pg.status === "review" || pg.status === "unassigned") && pg.priority <= 2 && (
                    <circle cx={c[0]} cy={c[1]} r={detail ? 6 : 3.5} fill={`rgb(${P_RGB[pg.priority]})`} className="pointer-events-none">
                      <animate attributeName="r" values={detail ? "4;16;4" : "2.5;10;2.5"} dur="1.8s" repeatCount="indefinite" />
                      <animate attributeName="opacity" values=".75;0;.75" dur="1.8s" repeatCount="indefinite" />
                    </circle>
                  )}
                  {mon && !pg && <circle cx={c[0] + (detail ? 10 : 4.5)} cy={c[1] - (detail ? 3.5 : 1.6)} r={detail ? 2.6 : 1.6} fill="none" stroke="rgb(255 255 255 / .7)" strokeWidth={detail ? 1.1 : 0.8} className="pointer-events-none" />}
                  {detail && (
                    <text x={c[0]} y={c[1] + 3.2} textAnchor="middle" fontSize={9} fontFamily="var(--font-geist-mono)" fill={pg ? P_INK[pg.priority] : "rgb(255 255 255 / .55)"} fontWeight={pg ? 700 : 500} className="pointer-events-none">
                      {rid}
                    </text>
                  )}
                  {detail && mon?.risk != null && (
                    <text x={c[0]} y={c[1] + 13} textAnchor="middle" fontSize={7.5} fontFamily="var(--font-geist-mono)" fill={mon.watch || pg ? "#f4c55a" : "rgb(255 255 255 / .38)"} className="pointer-events-none">
                      {mon.risk.toFixed(2)}
                    </text>
                  )}
                </g>
              );
            })}
            {detail &&
              (() => {
                const [x, y, w] = zn.rect;
                const p = proj(x + w / 2, y + 0.4, s, pos);
                return (
                  <text x={p[0]} y={p[1] - 6} textAnchor="middle" fontSize={9.5} letterSpacing="0.08em" fontFamily="var(--font-geist-mono)" fill="rgb(255 255 255 / .5)" className="pointer-events-none uppercase">
                    {zn.name}
                  </text>
                );
              })()}
          </g>
        );
      })}
      {Object.entries(spots)
        .filter(([, sp]) => sp.floor === fid)
        .map(([id, sp]) => {
          const c = clinicians[id];
          if (!c) return null;
          const [px, py] = proj(sp.x, sp.y, s, pos);
          const r = detail ? 5.8 : 4.3;
          return (
            <g
              key={id}
              className="cursor-pointer"
              onClick={(e) => {
                e.stopPropagation();
                onClinician?.(id);
              }}
              onMouseMove={(e) => tipAt(e, c.name, [`${c.specialty_label} · ${c.status.replace("_", " ")}`, ...(c.present ? ["pager open on a device"] : [])])}
              onMouseLeave={() => setTip(null)}
            >
              {c.status === "paging" && (
                <circle cx={px} cy={py} r={r} fill={`rgb(${ST_RGB[c.status]})`}>
                  <animate attributeName="r" values={`${r};${r * 3.2};${r}`} dur="1.4s" repeatCount="indefinite" />
                  <animate attributeName="opacity" values=".6;0;.6" dur="1.4s" repeatCount="indefinite" />
                </circle>
              )}
              <circle cx={px} cy={py + 1.2} r={r} fill="rgb(0 0 0 / .35)" />
              <circle cx={px} cy={py} r={r} fill={`rgb(${ST_RGB[c.status]})`} stroke="#1d2024" strokeWidth={detail ? 2 : 1.5} />
              {detail && (
                <text x={px + r + 3.5} y={py + 3} fontSize={9} fill="rgb(255 255 255 / .75)" fontWeight={500} className="pointer-events-none">
                  {c.name.replace(/^Dr\.\s*/, "").split(/[ ,]/)[0]}
                </text>
              )}
            </g>
          );
        })}
      {!detail &&
        (() => {
          const p = proj(0, D, s, pos);
          return (
            <g className="pointer-events-none">
              <text x={p[0] - 18} y={p[1] - 4} textAnchor="end" fontSize={10.5} fontFamily="var(--font-geist-mono)" fontWeight={600} letterSpacing="0.06em" fill="rgb(255 255 255 / .82)">
                {`FLOOR ${fid}`}
              </text>
              <text x={p[0] - 18} y={p[1] + 10} textAnchor="end" fontSize={10} fill="rgb(255 255 255 / .42)">
                {floor.label}
              </text>
              {summary && summary.n > 0 && summary.top != null && (
                <text x={p[0] - 18} y={p[1] + 24} textAnchor="end" fontSize={9.5} fontFamily="var(--font-geist-mono)" fontWeight={600} fill={P_INK[summary.top]}>
                  {`${summary.n} ACTIVE`}
                </text>
              )}
            </g>
          );
        })()}
    </g>
  );
}

/* ------------------------------------------------------------------ */
/* Dispatch arcs: from the clinician being paged to the room           */
/* ------------------------------------------------------------------ */
function Arcs({ layout, pages, spots, s, posOf, only, selected }: {
  layout: Layout;
  pages: Record<string, Page>;
  spots: Spots;
  s: number;
  posOf: (fid: string) => Pos | null;
  only?: string;
  selected?: string | null;
}) {
  const arcs = Object.values(pages).filter((p) => p.room && p.assignee && (p.status === "paging" || p.id === selected));
  return (
    <g className="pointer-events-none">
      {arcs.map((p) => {
        const room = layout.rooms[p.room!];
        const sp = spots[p.assignee!];
        if (!room || !sp) return null;
        if (only && (room.floor !== only || sp.floor !== only)) return null;
        const pr = posOf(room.floor), ps = posOf(sp.floor);
        if (!pr || !ps) return null;
        const a = proj(sp.x, sp.y, s, ps);
        const b = center(room.rect, s, pr);
        const lift = 30 + Math.abs(a[1] - b[1]) * 0.28 + Math.abs(a[0] - b[0]) * 0.08;
        const cx = (a[0] + b[0]) / 2, cy = Math.min(a[1], b[1]) - lift;
        const col = `rgb(${P_RGB[p.priority]})`;
        const d = `M${a[0].toFixed(1)},${a[1].toFixed(1)} Q${cx.toFixed(1)},${cy.toFixed(1)} ${b[0].toFixed(1)},${b[1].toFixed(1)}`;
        return (
          <g key={p.id}>
            <path d={d} fill="none" stroke={col} strokeOpacity={0.22} strokeWidth={5} strokeLinecap="round" />
            <path d={d} fill="none" stroke={col} strokeWidth={1.5} strokeDasharray="5 7" strokeLinecap="round" style={{ animation: "dash 0.9s linear infinite" }} />
            <circle cx={b[0]} cy={b[1]} r={2.8} fill={col} stroke="#1d2024" strokeWidth={1} />
          </g>
        );
      })}
    </g>
  );
}

/* ------------------------------------------------------------------ */
/* All floors, exploded and climbing diagonally                        */
/* ------------------------------------------------------------------ */
function Stack(props: Props & { setTip: (t: Tip) => void }) {
  const { layout, onFloor, clinicians, pages, selectedPage, roomPages } = props;
  const [hover, setHover] = useState<string | null>(null);
  const spots = useStaffSpots(layout, clinicians, pages);
  const S = 4.9, GAP = 66, DX = 54;
  const n = layout.floors.length;
  const { fx, fy } = STACK_ANGLE;
  const posOf = (fid: string): Pos | null => {
    const f = layout.floors.find((x) => x.id === fid);
    return f ? { z: f.index * GAP, ox: f.index * DX, fx, fy } : null;
  };
  const summaries = useMemo(() => {
    const out: Record<string, { n: number; top: number | null }> = {};
    for (const p of Object.values(roomPages)) {
      if (!p.floor) continue;
      const cur = (out[p.floor] ??= { n: 0, top: null });
      cur.n += 1;
      cur.top = cur.top == null ? p.priority : Math.min(cur.top, p.priority);
    }
    return out;
  }, [roomPages]);
  const W = layout.plate.w, D = layout.plate.d;
  const minX = -D * fx * S - 150;
  const maxX = W * fx * S + (n - 1) * DX + 24;
  const top = (n - 1) * GAP + 36;
  const bottom = (W + D) * fy * S + 34;
  return (
    <svg data-floorhost viewBox={`${minX.toFixed(0)} ${(-top).toFixed(0)} ${(maxX - minX).toFixed(0)} ${(top + bottom).toFixed(0)}`} className="h-auto w-full lg:h-full" preserveAspectRatio="xMidYMid meet">
      <defs>
        <linearGradient id="slab-top" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#3c4149" />
          <stop offset="1" stopColor="#2e3239" />
        </linearGradient>
        <filter id="room-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" />
        </filter>
      </defs>
      {layout.floors.map((f) => {
        const pos: Pos = { z: f.index * GAP, ox: f.index * DX, fx, fy };
        return (
          <g
            key={f.id}
            onMouseEnter={() => setHover(f.id)}
            onMouseLeave={() => setHover(null)}
            onClick={() => onFloor(f.id)}
            style={{ transform: hover === f.id ? "translateY(-6px)" : "none", transition: "transform .35s cubic-bezier(.16,1,.3,1)" }}
            className="cursor-pointer"
          >
            <Slab {...props} spots={spots} fid={f.id} s={S} pos={pos} detail={false} hover={hover === f.id} summary={summaries[f.id]} />
          </g>
        );
      })}
      <Arcs layout={layout} pages={pages} spots={spots} s={S} posOf={posOf} selected={selectedPage} />
    </svg>
  );
}

/* ------------------------------------------------------------------ */
/* One floor, up close                                                 */
/* ------------------------------------------------------------------ */
function Single(props: Props & { fid: string; setTip: (t: Tip) => void }) {
  const { layout, clinicians, pages, fid, selectedPage } = props;
  const spots = useStaffSpots(layout, clinicians, pages);
  const S = 8.8;
  const W = layout.plate.w, D = layout.plate.d;
  const { fx, fy } = FLOOR_ANGLE;
  const pos: Pos = { z: 0, ox: 0, fx, fy };
  const minX = -D * fx * S - 24, maxX = W * fx * S + 24;
  const minY = -64, maxY = (W + D) * fy * S + 40;
  return (
    <svg data-floorhost viewBox={`${minX.toFixed(0)} ${minY} ${(maxX - minX).toFixed(0)} ${(maxY - minY).toFixed(0)}`} className="h-auto w-full lg:h-full" preserveAspectRatio="xMidYMid meet">
      <defs>
        <linearGradient id="slab-top" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0" stopColor="#3c4149" />
          <stop offset="1" stopColor="#2e3239" />
        </linearGradient>
        <filter id="room-glow" x="-50%" y="-50%" width="200%" height="200%">
          <feGaussianBlur stdDeviation="4" />
        </filter>
      </defs>
      <Slab {...props} spots={spots} fid={fid} s={S} pos={pos} detail />
      <Arcs layout={layout} pages={pages} spots={spots} s={S} posOf={(f) => (f === fid ? pos : null)} only={fid} selected={selectedPage} />
    </svg>
  );
}

export function FloorChips({ layout, floor, onFloor, counts }: {
  layout: Layout;
  floor: string | "all";
  onFloor: (f: string | "all") => void;
  counts: Record<string, number>;
}) {
  const ids: (string | "all")[] = ["all", ...layout.floors.map((f) => f.id)];
  return (
    <div className="inline-flex items-center rounded-md border border-line bg-bg p-0.5" role="tablist" aria-label="Floor">
      {ids.map((id) => {
        const active = floor === id;
        const n = id === "all" ? 0 : counts[id] ?? 0;
        return (
          <button
            key={id}
            onClick={() => onFloor(id)}
            role="tab"
            aria-selected={active}
            className={cx(
              "relative grid h-6 min-w-7 place-items-center rounded-[4px] px-1.5 font-mono text-[11.5px] font-medium transition-colors",
              active ? "bg-s3 text-ink" : "text-dim hover:text-ink",
            )}
            title={id === "all" ? "All floors" : layout.floors.find((f) => f.id === id)?.label}
          >
            {id === "all" ? "All" : id}
            {n > 0 && !active && <span className="absolute right-0.5 top-0.5 h-1.5 w-1.5 rounded-full bg-p1" />}
          </button>
        );
      })}
    </div>
  );
}
