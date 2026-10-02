"use client";

import { ArrowUpRight } from "lucide-react";
import Link from "next/link";
import { useMemo } from "react";

import { PriorityChip } from "@/components/paging/bits";
import { Button, CloseIcon, cx, Drawer, Empty, IconButton } from "@/components/ui";
import { PAGE_STATUS_LABEL, type Occupant, type Page, type PageStatus } from "@/lib/paging";
import { SEV_RANK, SEV_RGB, useUnit, type RoomStatus } from "@/lib/rooms";

import { AllergyChips, CodeChip, KindTag, News2Badge, SevDot, SevTag } from "./bits";

/* ------------------------------------------------------------------ */
/* Watchlist: the rooms whose monitor sees something, worst first       */
/* ------------------------------------------------------------------ */
export function Watchlist({ rooms, occupancy, roomPages, onRoom }: {
  rooms: Record<string, RoomStatus>;
  occupancy: Record<string, Occupant>;
  roomPages: Record<string, Page>;
  onRoom: (room: string) => void;
}) {
  const rows = useMemo(
    () =>
      Object.entries(rooms)
        .filter(([, s]) => s.sev && SEV_RANK[s.sev] >= 1)
        .sort(([, a], [, b]) => SEV_RANK[b.sev!] - SEV_RANK[a.sev!] || (b.news2 ?? 0) - (a.news2 ?? 0) || b.n - a.n),
    [rooms],
  );
  const monitored = Object.keys(rooms).length;
  if (!monitored) return <div className="px-3.5 py-6 text-center text-[12.5px] text-dim">Waiting for the room monitor.</div>;
  if (!rows.length)
    return <div className="px-3.5 py-6 text-center text-[12.5px] text-dim">All {monitored} occupied rooms are below the watch level.</div>;
  return (
    <ul>
      {rows.map(([room, s]) => {
        const occ = occupancy[room];
        const pg = roomPages[room];
        return (
          <li key={room}>
            <button onClick={() => onRoom(room)} className="relative flex w-full items-center gap-2.5 border-b border-line-soft py-2 pl-4 pr-3 text-left transition-colors last:border-b-0 hover:bg-s2">
              <span className="absolute inset-y-0 left-0 w-[3px]" style={{ background: `rgb(${SEV_RGB[s.sev!]})` }} />
              <span className="w-10 shrink-0 font-mono text-[12.5px] font-semibold text-ink">{room}</span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[12.5px] text-ink">{s.title}</span>
                <span className="block truncate text-[11px] text-dim">
                  {occ?.name ?? "Patient"}
                  {s.n > 1 && ` · ${s.n} findings`}
                </span>
              </span>
              {pg ? <PriorityChip p={pg.priority} size="sm" /> : s.news2 != null ? <News2Badge score={s.news2} band={s.band} size="sm" /> : null}
            </button>
          </li>
        );
      })}
    </ul>
  );
}

/* ------------------------------------------------------------------ */
/* Room drawer: the unit at a glance, from the floor plan               */
/* ------------------------------------------------------------------ */
export function RoomDrawer({ room, onClose, onRaise, onOpenPage }: {
  room: string | null;
  onClose: () => void;
  onRaise: (room: string) => void;
  onOpenPage: (id: string) => void;
}) {
  const { data: u } = useUnit(room, 4000);
  const cur = u && u.room === room ? u : null;
  const rec = cur?.record;
  const v = cur?.vitals ?? null;
  const active = (cur?.findings ?? []).filter((f) => f.severity !== "info");
  const n = cur?.scores?.news2;
  return (
    <Drawer open={!!room} onClose={onClose} width={460}>
      <header className="flex shrink-0 items-start gap-3 border-b border-line px-4 py-3.5">
        <div className="min-w-0 flex-1">
          <div className="text-[11.5px] text-dim">{cur ? `Floor ${cur.floor} · ${cur.zone_name}` : "Room"}</div>
          <div className="mt-0.5 flex flex-wrap items-baseline gap-x-2.5 gap-y-1">
            <h2 className="font-mono text-[18px] font-semibold text-ink">Room {room}</h2>
            {rec && <span className="text-[13.5px] font-medium text-ink">{rec.patient.name}</span>}
            {rec && (
              <span className="text-[12px] text-dim">
                {rec.patient.age != null ? Math.round(rec.patient.age) : ""}
                {rec.patient.sex ?? ""}
              </span>
            )}
          </div>
        </div>
        <IconButton label="Close" onClick={onClose}>
          <CloseIcon />
        </IconButton>
      </header>
      <div className="min-h-0 flex-1 space-y-3.5 overflow-y-auto p-4">
        {!cur ? (
          <div className="h-48 animate-pulse rounded-md bg-s2" />
        ) : !cur.occupied || !rec ? (
          <Empty>Unoccupied.</Empty>
        ) : (
          <>
            <div>
              <div className="text-[12.5px] text-muted">{rec.admission.reason}</div>
              <div className="mt-2 flex flex-wrap gap-1.5">
                <KindTag kind={cur.kind} />
                <AllergyChips allergies={rec.allergies} compact />
                <CodeChip code={rec.code_status} />
              </div>
            </div>

            <div className="grid grid-cols-4 gap-px overflow-hidden rounded-md border border-line bg-line">
              {[
                ["NEWS2", cur.kind === "replay" ? <span key="n" className="text-[12px] text-dim">ICU</span> : <News2Badge key="n" score={n?.score} band={n?.band} size="sm" />],
                ["qSOFA", cur.scores?.qsofa != null ? `${cur.scores.qsofa}/3` : "—"],
                ["Shock idx", cur.scores?.shock_index != null ? cur.scores.shock_index.toFixed(2) : "—"],
                ["Risk", cur.risk != null ? cur.risk.toFixed(2) : "—"],
              ].map(([k, val]) => (
                <div key={String(k)} className="bg-s1 px-2.5 py-2">
                  <div className="text-[10.5px] text-dim">{k}</div>
                  <div className="num mt-0.5 flex h-5 items-center font-mono text-[14px] font-medium text-ink">{val}</div>
                </div>
              ))}
            </div>

            {v && (
              <div className="grid grid-cols-3 gap-px overflow-hidden rounded-md border border-line bg-line">
                {[
                  ["Pulse", v.hr != null ? Math.round(v.hr) : "—", "/min"],
                  ["BP", v.sbp != null ? `${Math.round(v.sbp)}/${v.dbp != null ? Math.round(v.dbp) : "—"}` : v.map != null ? `MAP ${Math.round(v.map)}` : "—", "mmHg"],
                  ["Resp", v.rr != null ? Math.round(v.rr) : "—", "/min"],
                  ["SpO₂", v.spo2 != null ? Math.round(v.spo2) : "—", "%"],
                  ["Temp", v.temp_c != null ? v.temp_c.toFixed(1) : "—", "°C"],
                  ["O₂", v.o2_lpm ? `${v.o2_lpm} L` : cur.kind === "synthetic" ? "air" : "—", ""],
                ].map(([k, val, unit]) => (
                  <div key={String(k)} className="bg-s1 px-2.5 py-1.5">
                    <div className="text-[10.5px] text-dim">{k}</div>
                    <div className="num font-mono text-[13.5px] text-ink">
                      {val}
                      {unit && <span className="ml-0.5 text-[10px] text-dim">{unit}</span>}
                    </div>
                  </div>
                ))}
              </div>
            )}

            <div>
              <div className="mb-1.5 flex items-center justify-between text-[11.5px]">
                <span className="font-medium text-dim">Active findings</span>
                <span className="text-faint">{cur.monitoring?.detectors.length ?? 19} detectors running</span>
              </div>
              {active.length === 0 ? (
                <div className="rounded-md border border-line px-3 py-2.5 text-[12.5px] text-dim">Nothing above information level.</div>
              ) : (
                <ul className="overflow-hidden rounded-md border border-line">
                  {active.slice(0, 5).map((f) => (
                    <li key={f.key} className="relative border-b border-line-soft py-2 pl-3.5 pr-3 last:border-b-0">
                      <span className="absolute inset-y-0 left-0 w-[3px]" style={{ background: `rgb(${SEV_RGB[f.severity]})` }} />
                      <div className="flex items-start gap-2">
                        <span className="min-w-0 flex-1 text-[12.5px] leading-snug text-ink">{f.title}</span>
                        <SevTag sev={f.severity} />
                      </div>
                      <div className="mt-0.5 truncate text-[11px] text-dim">{f.source.startsWith(f.detector) ? f.source : `${f.detector} · ${f.source}`}</div>
                    </li>
                  ))}
                  {active.length > 5 && <li className="px-3.5 py-1.5 text-[11.5px] text-dim">and {active.length - 5} more</li>}
                </ul>
              )}
            </div>

            {(cur.pages ?? []).length > 0 && (
              <div>
                <div className="mb-1.5 text-[11.5px] font-medium text-dim">Pages</div>
                <div className="overflow-hidden rounded-md border border-line">
                  {(cur.pages ?? []).slice(0, 5).map((pg) => (
                    <button key={pg.id} onClick={() => onOpenPage(pg.id)} className="flex w-full items-center gap-2 border-b border-line-soft px-3 py-2 text-left transition-colors last:border-b-0 hover:bg-s2">
                      <PriorityChip p={pg.priority} size="sm" />
                      <span className="min-w-0 flex-1 truncate text-[12.5px] text-ink">{pg.text}</span>
                      <span className={cx("shrink-0 text-[11px]", pg.active ? "text-ink" : "text-dim")}>{PAGE_STATUS_LABEL[pg.status as PageStatus] ?? pg.status}</span>
                    </button>
                  ))}
                </div>
              </div>
            )}

            {cur.simulation && (
              <div className="flex items-center gap-2 rounded-md border border-dashed border-line-bright px-3 py-2 text-[11.5px] text-dim">
                <SevDot sev={null} />
                Simulated course: {cur.simulation.label.toLowerCase()} ({cur.simulation.phase})
              </div>
            )}
          </>
        )}
      </div>
      {room && (
        <footer className="flex shrink-0 items-center gap-2 border-t border-line bg-s1 px-4 py-3">
          <Link href={`/room/?id=${encodeURIComponent(room)}`} className="flex-1">
            <Button variant="primary" className="w-full">
              Open room {room} <ArrowUpRight size={14} />
            </Button>
          </Link>
          <Button onClick={() => onRaise(room)}>Raise an alert</Button>
        </footer>
      )}
    </Drawer>
  );
}
