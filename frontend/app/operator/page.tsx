"use client";

import { AnimatePresence, motion } from "motion/react";
import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { LEGEND, PriorityChip, StatusDot } from "@/components/paging/bits";
import { Composer } from "@/components/paging/composer";
import { FloorChips, FloorView } from "@/components/paging/floor-view";
import { PageDrawer } from "@/components/paging/page-drawer";
import { CasesTable, Insights, KpiStrip, PolicyPanel, RoomDrawer, StaffBoard } from "@/components/paging/panels";
import { Feed, Queue } from "@/components/paging/queue";
import { AppBar } from "@/components/shell";
import { Button, cx, SectionLabel, Skeleton } from "@/components/ui";
import { readSession, revalidate, writeSession } from "@/lib/api";
import { useSession } from "@/lib/hooks";
import { activePages, useLayout, usePaging, useRoomPages, useServerNow, type Page } from "@/lib/paging";

type Tab = "queue" | "floor" | "cases" | "staff" | "insights" | "policy";

/** True at and above ``px``. Static HTML renders the desktop layout; a phone switches on mount. */
function useMinWidth(px: number): boolean {
  const [ok, setOk] = useState(true);
  useEffect(() => {
    const m = window.matchMedia(`(min-width: ${px}px)`);
    const f = () => setOk(m.matches);
    f();
    m.addEventListener("change", f);
    return () => m.removeEventListener("change", f);
  }, [px]);
  return ok;
}

export default function OperatorPage() {
  return (
    <Suspense>
      <Operator />
    </Suspense>
  );
}

function Operator() {
  const router = useRouter();
  const params = useSearchParams();
  // below lg there is no right rail, so the queue is a tab, and the one a phone opens on
  const wide = useMinWidth(1024);
  const asked = params.get("tab") as Tab | null;
  const tab: Tab = asked === "queue" && wide ? "floor" : asked || (wide ? "floor" : "queue");
  const setTab = (t: Tab) => router.replace(t === (wide ? "floor" : "queue") ? "/operator/" : `/operator/?tab=${t}`, { scroll: false });

  const session = useSession();
  const caps = new Set(session?.credential.capabilities ?? []);
  const canDispatch = caps.has("page:dispatch");
  const canRaise = caps.has("page:raise");
  const canConfigure = caps.has("page:configure");

  const { state, status, epoch } = usePaging();
  const layout = useLayout();
  // a reconnect may follow a backend restart that forgot every session: re-check ours
  useEffect(() => {
    const s = readSession();
    if (epoch < 2 || !s) return;
    revalidate(s)
      .then((v) => v.state === "renewed" ? writeSession(v.session) : v.state === "ended" && writeSession(null))
      .catch(() => {});
  }, [epoch]);
  const now = useServerNow(state.offsetMs);
  const [floor, setFloor] = useState<string | "all">("all");
  const [pageId, setPageId] = useState<string | null>(null);
  const [room, setRoom] = useState<string | null>(null);
  const [composer, setComposer] = useState<{ open: boolean; room?: string | null }>({ open: false });

  const open = useMemo(() => activePages(state.pages), [state.pages]);
  const roomPages = useRoomPages(state.pages);
  const recs = useMemo(() => Object.values(state.recs).sort((a, b) => (a.severity === "critical" ? -1 : 1) - (b.severity === "critical" ? -1 : 1) || b.updated_at - a.updated_at), [state.recs]);
  const floorCounts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const p of open) if (p.floor && p.priority <= 2 && p.status !== "accepted") c[p.floor] = (c[p.floor] ?? 0) + 1;
    return c;
  }, [open]);

  // ⌘K / N opens the composer; Esc is handled by the overlays themselves
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      const typing = (e.target as HTMLElement)?.closest("input, textarea, select, [contenteditable]");
      if ((e.key === "k" && (e.metaKey || e.ctrlKey)) || (e.key.toLowerCase() === "n" && !typing && !e.metaKey && !e.ctrlKey)) {
        e.preventDefault();
        setComposer({ open: true });
      }
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, []);

  const toasts = useToasts(state.pages, state.ready);
  const tabs: [Tab, string, number][] = [
    ["queue", "Queue", open.length],
    ["floor", "Floor", 0],
    ["cases", "Cases", open.length],
    ["staff", "Staff", 0],
    ["insights", "Insights", recs.length],
    ["policy", "Policy", 0],
  ];

  return (
    <div className="flex h-screen flex-col overflow-hidden">
      <AppBar
        title={
          <span>
            <span className="font-medium text-ink">{layout?.name ?? "Aegis Memorial"}</span>
            <span className="text-dim"> · Operator</span>
          </span>
        }
        center={
          <nav className="hidden items-center gap-0.5 rounded-xl bg-s2/80 p-0.5 lg:flex">
            {tabs.filter(([k]) => k !== "queue").map(([k, label, n]) => (
              <button
                key={k}
                onClick={() => setTab(k)}
                className={cx("relative flex h-8 items-center gap-1.5 rounded-[10px] px-3.5 text-[12.5px] font-medium transition", tab === k ? "text-ink" : "text-muted hover:text-ink")}
              >
                {tab === k && <motion.span layoutId="optab" className="absolute inset-0 rounded-[10px] bg-s1 shadow-[0_1px_2px_rgb(16_18_24/.1)]" transition={{ type: "spring", bounce: 0, duration: 0.35 }} />}
                <span className="relative">{label}</span>
                {n > 0 && <span className={cx("num relative rounded-full px-1.5 text-[10.5px] font-semibold", k === "insights" ? "bg-dys/15 text-dys-ink" : "bg-s3 text-muted")}>{n}</span>}
              </button>
            ))}
          </nav>
        }
        right={
          <Button variant="primary" onClick={() => setComposer({ open: true })}>
            <svg width="12" height="12" viewBox="0 0 12 12" aria-hidden><path d="M6 1v10M1 6h10" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" /></svg>
            <span className="hidden sm:inline">New alert</span>
            <span className="ml-1 hidden rounded bg-white/15 px-1 font-mono text-[10px] md:inline">⌘K</span>
          </Button>
        }
      />

      <div className="flex min-h-0 flex-1">
        <section className="flex min-w-0 flex-1 flex-col gap-3 overflow-y-auto p-4 lg:p-5">
          {status !== "live" && (
            <div className="rounded-lg border border-med/30 bg-med/10 px-3 py-2 text-[12.5px] text-med-ink">
              {status === "connecting" ? "Connecting to the hospital…" : "Reconnecting…"} A free-tier backend can take about 30 seconds to wake.
            </div>
          )}
          <KpiStrip stats={state.stats} pages={open} monitor={state.monitor} />

          {/* shrink-0: a flex child that scrolls on x has no min-height, and the column would squash it */}
          <nav className="-mx-1 flex shrink-0 gap-1 overflow-x-auto px-1 pb-0.5 lg:hidden" aria-label="Operator views">
            {tabs.map(([k, label, n]) => (
              <button
                key={k}
                onClick={() => setTab(k)}
                className={cx(
                  "flex h-8 shrink-0 items-center gap-1.5 rounded-full border px-3.5 text-[12.5px] font-medium transition",
                  tab === k ? "border-ink bg-ink text-white" : "border-line bg-s1 text-muted",
                )}
              >
                {label}
                {n > 0 && <span className={cx("num rounded-full px-1.5 text-[10.5px] font-semibold", tab === k ? "bg-white/20" : k === "insights" ? "bg-dys/15 text-dys-ink" : "bg-s3")}>{n}</span>}
              </button>
            ))}
          </nav>

          {tab === "queue" && (
            <div className="shrink-0 space-y-3 lg:hidden">
              <div className="plate p-3">
                <div className="px-1 pb-2">
                  <SectionLabel right={<span className="num font-mono">{open.length} open · {open.filter((p) => p.priority === 1).length} P1</span>}>Page queue</SectionLabel>
                </div>
                <Queue pages={open} clinicians={state.clinicians} now={now} onOpen={setPageId} canDispatch={canDispatch} selected={pageId} />
              </div>
              <div className="plate overflow-hidden">
                <div className="px-4 pb-1 pt-3">
                  <SectionLabel>Alert feed</SectionLabel>
                </div>
                <Feed items={state.alerts} now={now} onOpen={setPageId} />
              </div>
            </div>
          )}

          {tab === "floor" && (
            <div className="plate relative flex shrink-0 flex-col overflow-hidden lg:min-h-[560px] lg:flex-1 lg:shrink">
              <div className="flex flex-wrap items-center gap-3 border-b border-line-soft px-4 py-2.5">
                {layout ? <FloorChips layout={layout} floor={floor} onFloor={setFloor} counts={floorCounts} /> : <Skeleton className="h-7 w-80" />}
                <div className="ml-auto flex flex-wrap items-center gap-3">
                  {LEGEND.map((l) => (
                    <span key={l.status} className="inline-flex items-center gap-1.5 text-[11.5px] text-muted">
                      <StatusDot status={l.status} size={7} />
                      {l.label}
                    </span>
                  ))}
                </div>
              </div>
              <div className="relative bg-[radial-gradient(90%_70%_at_50%_40%,#fff_0%,#f6f6f3_70%)] py-4 lg:min-h-0 lg:flex-1 lg:py-0">
                {layout && state.ready ? (
                  <FloorView
                    layout={layout}
                    clinicians={state.clinicians}
                    roomPages={roomPages}
                    pages={state.pages}
                    monitor={state.monitor}
                    occupancy={state.occupancy}
                    floor={floor}
                    onFloor={setFloor}
                    onRoom={setRoom}
                    selectedPage={pageId}
                  />
                ) : (
                  <div className="grid h-full place-items-center text-[13px] text-dim">Drawing the building…</div>
                )}
              </div>
              <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-line-soft px-4 py-2 text-[11px] text-dim">
                <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm border border-p1 bg-p1/25" /> open page (colour = priority)</span>
                <span className="inline-flex items-center gap-1.5"><span className="h-2.5 w-2.5 rounded-sm border border-dashed border-med bg-med/10" /> sepsis watch, risk 0.60–0.70</span>
                <span className="inline-flex items-center gap-1.5"><span className="h-2 w-2 rounded-full bg-accent/85" /> monitored bed</span>
                <span className="inline-flex items-center gap-1.5"><svg width="18" height="6" aria-hidden><path d="M1 3h16" stroke="rgb(240 106 29)" strokeWidth="1.6" strokeDasharray="3 3" /></svg> page ringing</span>
                <span className="ml-auto max-w-[52%] truncate text-faint" title={layout?.note}>{layout?.note}</span>
              </div>
            </div>
          )}
          {tab === "cases" && <CasesTable pages={state.pages} clinicians={state.clinicians} now={now} onOpen={setPageId} />}
          {tab === "staff" && <StaffBoard clinicians={state.clinicians} pages={Object.values(state.pages)} layout={layout} canDispatch={canDispatch} now={now} />}
          {tab === "insights" && <Insights recs={recs} stats={state.stats} canDispatch={canDispatch} now={now} />}
          {tab === "policy" && <PolicyPanel settings={state.settings} invariants={state.invariants} canConfigure={canConfigure} stats={state.stats} />}
        </section>

        <aside className="hidden w-[400px] shrink-0 flex-col border-l border-line bg-s1/60 lg:flex">
          <div className="flex min-h-0 flex-[1.25] flex-col">
            <div className="px-4 pb-1 pt-4">
              <SectionLabel right={<span className="num font-mono">{open.length} open · {open.filter((p) => p.priority === 1).length} P1</span>}>Page queue</SectionLabel>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto px-3 pb-3">
              <Queue pages={open} clinicians={state.clinicians} now={now} onOpen={setPageId} canDispatch={canDispatch} selected={pageId} />
            </div>
          </div>
          <div className="flex min-h-0 flex-1 flex-col border-t border-line">
            <div className="px-4 pb-1 pt-3">
              <SectionLabel right={<span className="inline-flex items-center gap-1.5"><span className="live-dot h-1.5 w-1.5 rounded-full bg-low" style={{ color: "rgb(43 154 102 / .5)" }} />live</span>}>Alert feed</SectionLabel>
            </div>
            <div className="min-h-0 flex-1 overflow-y-auto">
              <Feed items={state.alerts} now={now} onOpen={setPageId} />
            </div>
          </div>
        </aside>
      </div>

      <PageDrawer page={pageId ? state.pages[pageId] ?? null : null} clinicians={state.clinicians} invariants={state.invariants} now={now} canDispatch={canDispatch} onClose={() => setPageId(null)} />
      <RoomDrawer room={room} onClose={() => setRoom(null)} now={now} onRaise={(r) => { setRoom(null); setComposer({ open: true, room: r }); }} onOpenPage={(id) => { setRoom(null); setPageId(id); }} />
      <Composer
        open={composer.open}
        presetRoom={composer.room}
        onClose={() => setComposer({ open: false })}
        layout={layout}
        occupancy={state.occupancy}
        clinicians={state.clinicians}
        onOpenPage={setPageId}
        canRaise={canRaise}
      />
      <Toasts toasts={toasts.list} onOpen={(id) => { toasts.dismiss(id); setPageId(id); }} onDismiss={toasts.dismiss} />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Toasts: only for things that need eyes now                          */
/* ------------------------------------------------------------------ */
type Toast = { id: string; page: Page; kind: "new" | "unassigned" };

function useToasts(pages: Record<string, Page>, ready: boolean) {
  const seen = useRef<Record<string, string>>({});
  const primed = useRef(false);
  const [list, setList] = useState<Toast[]>([]);
  const dismiss = useCallback((id: string) => setList((l) => l.filter((t) => t.id !== id)), []);
  useEffect(() => {
    if (!ready) return;
    const add: Toast[] = [];
    for (const p of Object.values(pages)) {
      const prev = seen.current[p.id];
      if (primed.current) {
        if (!prev && (p.priority <= 2 || p.source === "detector")) add.push({ id: p.id, page: p, kind: "new" });
        else if (prev && prev !== "unassigned" && p.status === "unassigned") add.push({ id: p.id, page: p, kind: "unassigned" });
      }
      seen.current[p.id] = p.status;
    }
    primed.current = true;
    if (add.length) {
      setList((l) => [...add, ...l.filter((t) => !add.some((a) => a.id === t.id))].slice(0, 4));
      for (const t of add) setTimeout(() => setList((l) => l.filter((x) => x.id !== t.id || x.kind !== t.kind)), 9000);
    }
  }, [pages, ready]);
  return { list, dismiss };
}

function Toasts({ toasts, onOpen, onDismiss }: { toasts: Toast[]; onOpen: (id: string) => void; onDismiss: (id: string) => void }) {
  return (
    <div className="pointer-events-none fixed bottom-4 left-4 z-[70] flex w-[min(360px,calc(100vw-32px))] flex-col gap-2">
      <AnimatePresence initial={false}>
        {toasts.map((t) => (
          <motion.div
            key={t.id + t.kind}
            layout
            initial={{ opacity: 0, y: 16, scale: 0.96 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, x: -20, transition: { duration: 0.2 } }}
            transition={{ type: "spring", bounce: 0, duration: 0.4 }}
            className="pop pointer-events-auto cursor-pointer p-3"
            onClick={() => onOpen(t.id)}
          >
            <div className="flex items-start gap-2.5">
              <PriorityChip p={t.page.priority} size="sm" />
              <div className="min-w-0 flex-1">
                <div className="text-[12.5px] font-semibold text-ink">
                  {t.kind === "unassigned" ? "Nobody took this page" : t.page.source === "detector" ? "Sepsis detector alert" : `New P${t.page.priority} alert`}
                  <span className="font-normal text-dim"> · room {t.page.room ?? "?"}</span>
                </div>
                <div className="line-clamp-2 text-[12px] text-muted">{t.page.text}</div>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDismiss(t.id);
                }}
                className="text-dim hover:text-ink"
                aria-label="Dismiss"
              >
                ×
              </button>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}
