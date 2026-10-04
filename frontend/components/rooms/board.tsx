"use client";

import { Search } from "lucide-react";
import Link from "next/link";
import { useMemo, useState } from "react";

import { PageHead } from "@/components/shell";
import { cx, Segmented, Skeleton } from "@/components/ui";
import { INDEX } from "@/lib/clinical";
import { SEV_LABEL, SEV_RANK, SEV_RGB, useBoard, type BoardRoom, type Severity } from "@/lib/rooms";

import { CodeChip, News2Badge, SevDot, Spark } from "./bits";

type Show = "attention" | "all" | "paged";

/** Every room as a unit: who is in it, what the monitor sees, whether anyone has been paged. */
export function WatchBoard() {
  const { data, error } = useBoard(5000);
  const [show, setShow] = useState<Show>("all");
  const [q, setQ] = useState("");
  const rooms = useMemo(() => (data?.rooms ?? []).filter((r) => r.kind), [data]);
  const counts = useMemo(() => {
    const c = { critical: 0, urgent: 0, watch: 0, paged: 0 };
    for (const r of rooms) {
      if (r.worst && r.worst !== "info") c[r.worst as "critical" | "urgent" | "watch"] += 1;
      if (r.episode?.open) c.paged += 1;
    }
    return c;
  }, [rooms]);
  const filtered = useMemo(() => {
    const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    return rooms.filter((r) => {
      if (show === "attention" && !(r.worst && SEV_RANK[r.worst] >= 1)) return false;
      if (show === "paged" && !r.episode?.open) return false;
      if (!terms.length) return true;
      const hay = `${r.room} ${r.name ?? ""} ${r.diagnosis ?? ""} ${r.zone_name} ${(r.findings ?? []).map((f) => f.title).join(" ")} ${(r.allergies ?? []).join(" ")}`.toLowerCase();
      return terms.every((t) => hay.includes(t));
    });
  }, [rooms, show, q]);
  // floors top to bottom as the building stands; units within a floor in plan order
  const groups = useMemo(() => {
    const byZone = new Map<string, { zone: string; name: string; floor: string; rooms: BoardRoom[] }>();
    for (const r of filtered) {
      const g = byZone.get(r.zone) ?? { zone: r.zone, name: r.zone_name, floor: r.floor, rooms: [] };
      g.rooms.push(r);
      byZone.set(r.zone, g);
    }
    const floorOrder = ["3", "4", "A", "2", "1", "5", "6"];
    return [...byZone.values()].sort((a, b) => floorOrder.indexOf(a.floor) - floorOrder.indexOf(b.floor) || a.name.localeCompare(b.name));
  }, [filtered]);
  const attention = useMemo(
    () => [...filtered].filter((r) => r.worst && SEV_RANK[r.worst] >= 1).sort((a, b) => SEV_RANK[b.worst!] - SEV_RANK[a.worst!] || (b.scores?.news2?.score ?? 0) - (a.scores?.news2?.score ?? 0)),
    [filtered],
  );

  return (
    <div>
      <PageHead title="Rooms">
        Every occupied room is watched as one unit: the patient&apos;s record, the vitals the room is producing and {data?.detectors.length ?? 19} detectors that read
        both. Synthetic rooms are paged by the room monitor; the ICU replay rooms and the live session by the sepsis detector.
      </PageHead>

      <div className="mb-3 grid grid-cols-2 gap-px overflow-hidden rounded-lg border border-line bg-line sm:grid-cols-3 xl:grid-cols-6">
        {[
          ["Rooms monitored", rooms.length, `${rooms.filter((r) => r.kind === "replay").length} replayed · ${rooms.filter((r) => r.kind === "live").length} live · ${rooms.filter((r) => r.kind === "interface").length} HL7-fed · ${rooms.filter((r) => r.kind === "synthetic").length} synthetic`],
          ["Critical", counts.critical, "a finding at the top band"],
          ["Urgent", counts.urgent, "review now"],
          ["Watch", counts.watch, "trending or borderline"],
          ["With an open episode", counts.paged, `${data?.counters.pages_opened ?? 0} pages raised by the monitor`],
          ["Evaluation", data?.eval_ms_p50 != null ? `${data.eval_ms_p50.toFixed(1)} ms` : "—", `every room, every ${data?.eval_period_s ?? 3} s`],
        ].map(([k, v, sub]) => (
          <div key={String(k)} className="min-w-0 bg-s1 px-3.5 py-2.5">
            <div className="truncate text-[11px] text-dim">{k}</div>
            <div className="num mt-0.5 flex items-center gap-2 font-mono text-[20px] font-medium text-ink">
              {(k === "Critical" || k === "Urgent" || k === "Watch") && <SevDot sev={k.toLowerCase() as Severity} size={8} />}
              {v}
            </div>
            <div className="truncate text-[11px] text-dim">{sub}</div>
          </div>
        ))}
      </div>

      <div className="mb-3 flex flex-wrap items-center gap-3">
        <label className="flex h-8 w-full items-center gap-1.5 rounded-md border border-line bg-bg px-2.5 text-dim focus-within:border-line-bright sm:w-72">
          <Search size={13} />
          <input value={q} onChange={(e) => setQ(e.target.value)} placeholder="Room, patient, finding, allergy" className="min-w-0 flex-1 bg-transparent text-[12.5px] text-ink outline-none placeholder:text-faint" aria-label="Search rooms" />
        </label>
        <Segmented
          value={show}
          onChange={setShow}
          options={[
            { value: "all", label: "All rooms" },
            { value: "attention", label: `Needs attention ${counts.critical + counts.urgent + counts.watch}` },
            { value: "paged", label: `Open episodes ${counts.paged}` },
          ]}
        />
        <div className="flex flex-wrap items-center gap-3 text-[11px] text-dim">
          {(["critical", "urgent", "watch"] as const).map((s) => (
            <span key={s} className="inline-flex items-center gap-1.5">
              <span className="h-2.5 w-[3px] rounded-full" style={{ background: `rgb(${SEV_RGB[s]})` }} />
              {SEV_LABEL[s]}
            </span>
          ))}
          <span className="inline-flex items-center gap-1.5">
            <News2Badge score={3} band="low" size="sm" /> NEWS2 (ward)
          </span>
          <span className="inline-flex items-center gap-1.5" title={`The sepsis detector's risk index; it pages at ${INDEX.page.toFixed(2)}`}>
            <span className="num grid h-4 place-items-center rounded-[4px] border border-line-bright bg-s2 px-1 font-mono text-[10px] font-semibold text-ink">0.42</span>
            Sepsis risk (ICU)
          </span>
        </div>
      </div>

      {!data && !error ? (
        <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 xl:grid-cols-4">
          {Array.from({ length: 8 }, (_, i) => (
            <Skeleton key={i} className="h-28" />
          ))}
        </div>
      ) : error && !data ? (
        <div className="rounded-md border border-line px-4 py-8 text-center text-[13px] text-dim">Could not load the rooms: {error}</div>
      ) : show === "attention" ? (
        attention.length === 0 ? (
          <div className="rounded-md border border-line px-4 py-10 text-center text-[13px] text-dim">No room has a finding above information level.</div>
        ) : (
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3 2xl:grid-cols-4">
            {attention.map((r) => (
              <Tile key={r.room} r={r} wide />
            ))}
          </div>
        )
      ) : (
        <div className="space-y-4">
          {groups.map((g) => (
            <section key={g.zone}>
              <div className="mb-2 flex items-baseline gap-2">
                <h2 className="text-[13px] font-semibold text-ink">{g.name}</h2>
                <span className="text-[11.5px] text-dim">
                  Floor {g.floor} · {g.rooms.length} {g.rooms.length === 1 ? "patient" : "patients"}
                </span>
              </div>
              {/* units hold six rooms: 3 + 3 on a tablet, one row of 6 on a wide screen */}
              <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 md:grid-cols-3 wide:grid-cols-6">
                {g.rooms.map((r) => (
                  <Tile key={r.room} r={r} />
                ))}
              </div>
            </section>
          ))}
        </div>
      )}
    </div>
  );
}

function Tile({ r, wide = false }: { r: BoardRoom; wide?: boolean }) {
  const n = r.scores?.news2;
  const worst = r.worst && r.worst !== "info" ? r.worst : null;
  const top = (r.findings ?? [])[0];
  const replay = r.kind === "replay";
  return (
    <Link
      href={`/room/?id=${encodeURIComponent(r.room)}`}
      className="group relative block overflow-hidden rounded-lg border border-line bg-s1 py-2.5 pl-3.5 pr-3 transition-colors hover:border-line-bright hover:bg-s2"
    >
      <span className="absolute inset-y-0 left-0 w-[3px]" style={{ background: worst ? `rgb(${SEV_RGB[worst]})` : "transparent" }} />
      <div className="flex items-start gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline gap-2">
            <span className="font-mono text-[14px] font-semibold text-ink">{r.room}</span>
            <span className="truncate text-[12.5px] text-ink/90">{r.name}</span>
          </div>
          <div className="truncate text-[11.5px] text-dim">
            {r.age != null ? `${Math.round(r.age)}${r.sex ?? ""} · ` : ""}
            {r.diagnosis}
          </div>
        </div>
        {replay || r.kind === "live" ? (
          <span
            className={cx("num grid h-6 min-w-6 place-items-center rounded-[5px] border px-1.5 font-mono text-[12px] font-semibold", (r.risk ?? 0) >= INDEX.page ? "border-high/50 bg-high/15 text-high-ink" : (r.risk ?? 0) >= INDEX.watch ? "border-med/50 bg-med/15 text-med-ink" : "border-line-bright bg-s2 text-ink")}
            title={`Sepsis risk index${r.probability != null ? ` (calibrated P ${(r.probability * 100).toFixed(1)}%)` : ""}; the detector pages at ${INDEX.page.toFixed(2)}`}
          >
            {r.risk != null ? r.risk.toFixed(2) : "—"}
          </span>
        ) : (
          <News2Badge score={n?.score} band={n?.band} />
        )}
      </div>
      <div className="mt-2 flex items-center gap-2">
        <SevDot sev={worst} />
        <span className={cx("min-w-0 flex-1 truncate text-[12px]", worst ? "text-ink" : "text-dim")}>{top ? top.title : "No active findings"}</span>
        {(r.counts?.critical ?? 0) + (r.counts?.urgent ?? 0) + (r.counts?.watch ?? 0) > 1 && (
          <span className="num shrink-0 font-mono text-[10.5px] text-dim">+{(r.counts!.critical + r.counts!.urgent + r.counts!.watch) - 1}</span>
        )}
      </div>
      <div className="mt-2 flex items-center gap-1.5">
        {r.trend && r.trend.length > 1 ? <Spark data={r.trend} w={wide ? 96 : 72} h={16} color="#868d97" domain={[0, Math.max(8, ...r.trend)]} /> : <span className="h-4 w-[72px]" />}
        <span className="ml-auto flex min-w-0 items-center gap-1">
          {(r.allergies ?? []).length > 0 && (
            <span className="h-4 truncate rounded-[3px] border border-high/40 bg-high/10 px-1 text-[10px] font-medium leading-[14px] text-high-ink" title={`Allergies: ${(r.allergies ?? []).join(", ")}`}>
              {(r.allergies ?? []).length === 1 ? (r.allergies ?? [])[0].split(" ")[0] : `${(r.allergies ?? []).length} allergies`}
            </span>
          )}
          {r.code_status && r.code_status !== "Full code" && (
            <span className="scale-90">
              <CodeChip code={r.code_status} />
            </span>
          )}
          {r.episode?.open && (
            <span className="h-4 rounded-[3px] border border-line-bright bg-s3 px-1 font-mono text-[10px] leading-[14px] text-ink" title="An episode is open: the room has a page">
              {r.episode.page ?? "paged"}
            </span>
          )}
          {r.kind === "replay" && (
            <span className="h-4 rounded-[3px] border border-accent/40 bg-accent-soft px-1 text-[10px] font-medium leading-[14px] text-accent-ink" title="PhysioNet/CinC 2019, de-identified, replayed">
              Real
            </span>
          )}
          {r.kind === "live" && (
            <span className="h-4 rounded-[3px] border border-accent/40 bg-accent-soft px-1 text-[10px] font-medium leading-[14px] text-accent-ink" title="Live monitored session">
              Live
            </span>
          )}
        </span>
      </div>
    </Link>
  );
}
