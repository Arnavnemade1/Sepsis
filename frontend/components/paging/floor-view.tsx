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

// calm unit tints: never a priority or status colour
const ZONE_TINT: Record<string, string> = {
  icu: "rgb(68 83 214 / 0.16)",
  ed: "rgb(28 126 240 / 0.13)",
  stepdown: "rgb(20 160 150 / 0.15)",
  ward: "rgb(100 116 139 / 0.11)",
  obs: "rgb(100 116 139 / 0.09)",
  or: "rgb(139 92 246 / 0.13)",
  pacu: "rgb(139 92 246 / 0.10)",
  support: "rgb(17 20 24 / 0.06)",
  imaging: "rgb(17 20 24 / 0.05)",
};

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
        <div className="pop pointer-events-none absolute z-20 max-w-[260px] px-3 py-2" style={{ left: tip.x + 14, top: tip.y + 14 }}>
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
      <polygon points={pts(shadow)} fill="rgb(17 20 24 / .05)" style={{ filter: `blur(${detail ? 10 : 6}px)` }} />
      <polygon points={pts(front)} fill="#d9d9d2" />
      <polygon points={pts(side)} fill="#c4c4bc" />
      <polygon points={pts(top)} fill="#ffffff" stroke={hover ? "rgb(68 83 214 / .7)" : "rgb(17 20 24 / .2)"} strokeWidth={hover ? 1.5 : 1} style={{ transition: "stroke .2s" }} />
      {floor.zones.map((zid) => {
        const zn = layout.zones[zid];
        return (
          <g key={zid}>
            <polygon points={pts(quad(zn.rect, 0.55, s, pos))} fill={ZONE_TINT[zn.kind] ?? ZONE_TINT.ward} stroke="rgb(17 20 24 / .12)" strokeWidth={0.7} />
            {zn.rooms.map((rid) => {
              const room = layout.rooms[rid];
              const pg = roomPages[rid];
              const mon = monitor[rid];
              const occ = occupancy[rid];
              let fill = occ ? "rgb(255 255 255 / .96)" : "rgb(255 255 255 / .42)";
              let stroke = "rgb(17 20 24 / .2)";
              let dash: string | undefined;
              let sw = 0.6;
              if (mon?.watch && !pg) {
                fill = "rgb(226 163 54 / .14)";
                stroke = "rgb(226 163 54 / .95)";
                dash = "3 2";
                sw = 1;
              }
              if (pg) {
                fill = `rgb(${P_RGB[pg.priority]} / ${pg.status === "accepted" ? 0.18 : 0.3})`;
                stroke = `rgb(${P_RGB[pg.priority]})`;
                sw = 1.2;
              }
              const q = quad(room.rect, 0.4, s, pos);
              const c = center(room.rect, s, pos);
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
                  <polygon points={pts(q)} fill={fill} stroke={stroke} strokeWidth={sw} strokeDasharray={dash} className="transition-[fill] duration-300 hover:brightness-95" />
                  {pg && (pg.status === "paging" || pg.status === "review" || pg.status === "unassigned") && pg.priority <= 2 && (
                    <circle cx={c[0]} cy={c[1]} r={detail ? 6 : 3.5} fill={`rgb(${P_RGB[pg.priority]})`} className="pointer-events-none">
                      <animate attributeName="r" values={detail ? "4;16;4" : "2.5;10;2.5"} dur="1.8s" repeatCount="indefinite" />
                      <animate attributeName="opacity" values=".75;0;.75" dur="1.8s" repeatCount="indefinite" />
                    </circle>
                  )}
                  {mon && !pg && <circle cx={c[0] + (detail ? 10 : 4.5)} cy={c[1] - (detail ? 3.5 : 1.6)} r={detail ? 2.4 : 1.5} fill="rgb(68 83 214 / .9)" className="pointer-events-none" />}
                  {detail && (
                    <text x={c[0]} y={c[1] + 3.2} textAnchor="middle" fontSize={9} fontFamily="var(--font-geist-mono)" fill={pg ? P_INK[pg.priority] : "rgb(17 20 24 / .6)"} fontWeight={pg ? 700 : 500} className="pointer-events-none">
                      {rid}
                    </text>
                  )}
                  {detail && mon?.risk != null && (
                    <text x={c[0]} y={c[1] + 13} textAnchor="middle" fontSize={7.5} fontFamily="var(--font-geist-mono)" fill={mon.watch || pg ? "#926000" : "rgb(17 20 24 / .42)"} className="pointer-events-none">
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
                  <text x={p[0]} y={p[1] - 6} textAnchor="middle" fontSize={9.5} letterSpacing="0.1em" fontFamily="var(--font-geist-mono)" fill="rgb(17 20 24 / .55)" className="pointer-events-none uppercase">
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
              <circle cx={px} cy={py + 1.2} r={r} fill="rgb(17 20 24 / .15)" />
              <circle cx={px} cy={py} r={r} fill={`rgb(${ST_RGB[c.status]})`} stroke="#fff" strokeWidth={detail ? 2 : 1.5} />
              {detail && (
                <text x={px + r + 3.5} y={py + 3} fontSize={9} fill="rgb(17 20 24 / .72)" fontWeight={500} className="pointer-events-none">
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
              <text x={p[0] - 18} y={p[1] - 4} textAnchor="end" fontSize={11} fontFamily="var(--font-geist-mono)" fontWeight={600} letterSpacing="0.08em" fill="rgb(17 20 24 / .78)">
                {`FLOOR ${fid}`}
              </text>
              <text x={p[0] - 18} y={p[1] + 10} textAnchor="end" fontSize={10.5} fill="rgb(17 20 24 / .46)">
                {floor.label}
              </text>
              {summary && summary.n > 0 && summary.top != null && (
                <g>
                  <rect x={p[0] - 18 - 56} y={p[1] + 16} width={56} height={15} rx={7.5} fill={`rgb(${P_RGB[summary.top]} / .14)`} />
                  <text x={p[0] - 18 - 28} y={p[1] + 26.5} textAnchor="middle" fontSize={9.5} fontWeight={600} fill={P_INK[summary.top]}>
                    {summary.n} open
                  </text>
                </g>
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
            <path d={d} fill="none" stroke={col} strokeOpacity={0.18} strokeWidth={5} strokeLinecap="round" />
            <path d={d} fill="none" stroke={col} strokeWidth={1.6} strokeDasharray="5 7" strokeLinecap="round" style={{ animation: "dash 0.9s linear infinite" }} />
            <circle cx={b[0]} cy={b[1]} r={2.8} fill={col} stroke="#fff" strokeWidth={1} />
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
    <div className="flex items-center gap-1">
      <span className="mr-1 font-mono text-[10.5px] uppercase tracking-[0.12em] text-dim">Floor</span>
      {ids.map((id) => {
        const active = floor === id;
        const n = id === "all" ? 0 : counts[id] ?? 0;
        return (
          <button
            key={id}
            onClick={() => onFloor(id)}
            className={cx(
              "relative grid h-7 min-w-7 place-items-center rounded-full px-2 font-mono text-[11.5px] font-medium transition",
              active ? "bg-ink text-white shadow-sm" : "text-muted hover:bg-s2 hover:text-ink",
            )}
            title={id === "all" ? "All floors" : layout.floors.find((f) => f.id === id)?.label}
          >
            {id === "all" ? "All" : id}
            {n > 0 && !active && <span className="absolute -right-0.5 -top-0.5 h-2 w-2 rounded-full border-2 border-bg bg-p1" />}
          </button>
        );
      })}
    </div>
  );
}
