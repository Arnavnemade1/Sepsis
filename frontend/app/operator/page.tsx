"use client";

import { AnimatePresence, motion } from "motion/react";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Plus } from "@/components/icons";
import { LEGEND, P_RGB, PriorityChip, StatusDot } from "@/components/paging/bits";
import { Composer } from "@/components/paging/composer";
import { FloorChips, FloorView } from "@/components/paging/floor-view";
import { PageDrawer } from "@/components/paging/page-drawer";
import { CasesTable, Insights, KpiStrip, PolicyPanel, StaffBoard } from "@/components/paging/panels";
import { Feed, Queue } from "@/components/paging/queue";
import { RoomDrawer, Watchlist } from "@/components/rooms/drawer";
import { AppShell, type NavKey } from "@/components/shell";
import { Button, CloseIcon, cx, Panel, Segmented, Skeleton } from "@/components/ui";
import { readSession, revalidate, writeSession } from "@/lib/api";
import { useSession } from "@/lib/hooks";
import { activePages, useLayout, usePaging, useRoomPages, useServerNow, type Page } from "@/lib/paging";
import { SEV_RANK, SEV_RGB } from "@/lib/rooms";

type View = "overview" | "pages" | "staff" | "sentinel" | "policy";
const VIEWS: View[] = ["overview", "pages", "staff", "sentinel", "policy"];
// links from before the console had these names
const LEGACY: Record<string, View> = { floor: "overview", queue: "overview", cases: "pages", insights: "sentinel" };

export default function OperatorPage() {
  return (
    <Suspense>
      <Operator />
    </Suspense>
  );
}

function Operator() {
  const params = useSearchParams();
  const raw = params.get("view") ?? params.get("tab") ?? "overview";
  const view: View = (VIEWS as string[]).includes(raw) ? (raw as View) : LEGACY[raw] ?? "overview";

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
      .then((v) => (v.state === "renewed" ? writeSession(v.session) : v.state === "ended" && writeSession(null)))
      .catch(() => {});
  }, [epoch]);
  const now = useServerNow(state.offsetMs);
  const [floor, setFloor] = useState<string | "all">("all");
  const [pageId, setPageId] = useState<string | null>(null);
  const [room, setRoom] = useState<string | null>(null);
  const [composer, setComposer] = useState<{ open: boolean; room?: string | null }>({ open: false });
  const [side, setSide] = useState<"watch" | "log">("watch");

  const open = useMemo(() => activePages(state.pages), [state.pages]);
  const roomPages = useRoomPages(state.pages);
  const recs = useMemo(
    () =>
      Object.values(state.recs).sort(
        (a, b) => (a.severity === "critical" ? -1 : 1) - (b.severity === "critical" ? -1 : 1) || b.updated_at - a.updated_at,
      ),
    [state.recs],
  );
  const floorCounts = useMemo(() => {
    const c: Record<string, number> = {};
    for (const p of open) if (p.floor && p.priority <= 2 && p.status !== "accepted") c[p.floor] = (c[p.floor] ?? 0) + 1;
    return c;
  }, [open]);

  // N raises an alert (⌘K is the command palette, which can also raise one); Esc closes overlays
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      const t = e.target as HTMLElement | null;
      const typing = typeof t?.closest === "function" && t.closest("input, textarea, select, [contenteditable]");
      if (e.key.toLowerCase() === "n" && !typing && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault();
        setComposer({ open: true });
      }
    };
    const compose = () => setComposer({ open: true });
    window.addEventListener("keydown", k);
    window.addEventListener("aegis:compose", compose);
    return () => {
      window.removeEventListener("keydown", k);
      window.removeEventListener("aegis:compose", compose);
    };
  }, []);
  // deep links: ?compose=1[&room=512] opens the composer, ?page=PG-… opens that page, ?room=512 that room
  const askCompose = params.get("compose");
  const askPage = params.get("page");
  const askRoom = params.get("room");
  useEffect(() => {
    if (askCompose) setComposer({ open: true, room: askRoom });
    else if (askRoom) setRoom(askRoom);
  }, [askCompose, askRoom]);
  useEffect(() => {
    if (askPage) setPageId(askPage);
  }, [askPage]);

  const toasts = useToasts(state.pages, state.ready);
  const activeRecs = recs.filter((r) => !r.acknowledged).length;
  const watching = Object.values(state.rooms).filter((r) => r.sev && SEV_RANK[r.sev] >= 1).length;
  const counts: Partial<Record<NavKey, number>> = { pages: open.length, sentinel: activeRecs };
  const alerts: Partial<Record<NavKey, boolean>> = { pages: open.some((p) => p.priority === 1), sentinel: recs.some((r) => r.severity === "critical" && !r.acknowledged) };

  return (
    <AppShell
      active={view}
      counts={counts}
      alerts={alerts}
      actions={
        <Button size="sm" variant="primary" onClick={() => setComposer({ open: true })} title="Raise an alert (N)">
          <Plus size={14} weight="bold" />
          <span className="hidden sm:inline">New alert</span>
          <span className="ml-1 hidden h-4 min-w-4 place-items-center rounded-[4px] bg-black/[0.08] px-1 text-[10px] font-semibold text-black/50 md:grid">N</span>
        </Button>
      }
    >
      {status !== "live" && (
        <div className="flex shrink-0 items-center gap-2 border-b border-line bg-med/10 px-4 py-1.5 text-xs text-med-ink">
          <span className="blink h-1.5 w-1.5 rounded-full bg-med" />
          {status === "connecting" ? "Connecting to the hospital" : "Reconnecting to the hospital"}. A sleeping free-tier backend takes 30–60 s to wake.
        </div>
      )}

      {view === "overview" ? (
        <main className="flex min-h-0 flex-1 flex-col gap-4 overflow-y-auto p-4 sm:p-5 lg:overflow-hidden">
          <KpiStrip stats={state.stats} pages={open} monitor={state.monitor} all={state.pages} now={now} />
          <div className="grid grid-cols-1 gap-4 lg:min-h-0 lg:flex-1 lg:grid-cols-[minmax(0,1fr)_400px]">
            <Panel
              title="Floor plan"
              meta={layout ? `${layout.name} · ${layout.floors.length} floors · ${Object.keys(layout.rooms).length} rooms` : undefined}
              right={layout ? <FloorChips layout={layout} floor={floor} onFloor={setFloor} counts={floorCounts} /> : <Skeleton className="h-7 w-64" />}
              flush
              className="order-2 lg:order-1 lg:min-h-0"
              bodyClass="flex flex-col lg:min-h-0"
            >
              <div
                className="relative py-3 lg:min-h-0 lg:flex-1 lg:py-0"
                style={{
                  // the building sits in a soft pool of light; no texture, nothing that competes with a page
                  backgroundImage: "radial-gradient(70% 65% at 50% 45%, #2b2e34 0%, #24272c 100%)",
                }}
              >
                {layout && state.ready ? (
                  <FloorView
                    layout={layout}
                    clinicians={state.clinicians}
                    roomPages={roomPages}
                    pages={state.pages}
                    monitor={state.monitor}
                    occupancy={state.occupancy}
                    rooms={state.rooms}
                    floor={floor}
                    onFloor={setFloor}
                    onRoom={setRoom}
                    selectedPage={pageId}
                  />
                ) : (
                  <div className="grid h-full min-h-[240px] place-items-center text-sm text-dim">Loading the building</div>
                )}
              </div>
              <div className="flex shrink-0 items-center gap-x-4 overflow-x-auto whitespace-nowrap border-t border-line px-4 py-2.5 text-xs text-dim [scrollbar-width:none]">
                {LEGEND.map((l) => (
                  <span key={l.status} className="inline-flex items-center gap-1.5">
                    <StatusDot status={l.status} size={7} still />
                    {l.label}
                  </span>
                ))}
                <span className="h-3 w-px shrink-0 bg-line-bright" />
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-[2px] border" style={{ borderColor: `rgb(${P_RGB[1]})`, background: `rgb(${P_RGB[1]} / .3)` }} /> Active page
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-2.5 w-2.5 rounded-[2px] border border-dashed border-med bg-med/15" /> Sepsis watch 0.60–0.70
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <svg width="10" height="10" viewBox="0 0 10 10" aria-hidden>
                    <path d="M0 0H8L0 8Z" fill={`rgb(${SEV_RGB.urgent})`} />
                  </svg>
                  Monitor finding
                </span>
                <span className="inline-flex items-center gap-1.5">
                  <span className="h-2 w-2 rounded-full border border-white/70" /> Monitored bed
                </span>
              </div>
            </Panel>

            <div className="order-1 flex flex-col gap-3 lg:order-2 lg:min-h-0">
              <Panel
                title="Active pages"
                right={
                  <span className="num text-xs text-dim">
                    {open.length} open{open.some((p) => p.priority === 1) ? <span className="text-high-ink"> · {open.filter((p) => p.priority === 1).length} P1</span> : null}
                  </span>
                }
                flush
                className="lg:min-h-0 lg:flex-[1.35]"
                bodyClass="lg:min-h-0 lg:overflow-y-auto"
              >
                <Queue pages={open} clinicians={state.clinicians} now={now} onOpen={setPageId} canDispatch={canDispatch} selected={pageId} />
              </Panel>
              <Panel
                title={side === "watch" ? "Watchlist" : "Event log"}
                meta={side === "watch" ? `${watching} of ${Object.keys(state.rooms).length} occupied rooms above the watch level` : undefined}
                right={
                  <Segmented
                    size="sm"
                    value={side}
                    onChange={setSide}
                    options={[
                      { value: "watch", label: "Rooms" },
                      { value: "log", label: "Log" },
                    ]}
                  />
                }
                flush
                className="lg:min-h-0 lg:flex-1"
                bodyClass="max-h-[420px] overflow-y-auto lg:max-h-none lg:min-h-0"
              >
                {side === "watch" ? (
                  <Watchlist rooms={state.rooms} occupancy={state.occupancy} roomPages={roomPages} onRoom={setRoom} />
                ) : (
                  <Feed items={state.alerts} now={now} onOpen={setPageId} />
                )}
              </Panel>
            </div>
          </div>
        </main>
      ) : (
        <main className="min-h-0 flex-1 overflow-y-auto p-4 sm:p-5">
          <div className="mx-auto flex max-w-[1600px] flex-col gap-4">
            {view === "pages" && <CasesTable pages={state.pages} clinicians={state.clinicians} now={now} onOpen={setPageId} />}
            {view === "staff" && <StaffBoard clinicians={state.clinicians} pages={Object.values(state.pages)} layout={layout} canDispatch={canDispatch} now={now} />}
            {view === "sentinel" && <Insights recs={recs} stats={state.stats} canDispatch={canDispatch} now={now} />}
            {view === "policy" && <PolicyPanel settings={state.settings} invariants={state.invariants} canConfigure={canConfigure} stats={state.stats} />}
          </div>
        </main>
      )}

      <PageDrawer page={pageId ? state.pages[pageId] ?? null : null} clinicians={state.clinicians} invariants={state.invariants} now={now} canDispatch={canDispatch} onClose={() => setPageId(null)} />
      <RoomDrawer
        room={room}
        onClose={() => setRoom(null)}
        onRaise={(r) => {
          setRoom(null);
          setComposer({ open: true, room: r });
        }}
        onOpenPage={(id) => {
          setRoom(null);
          setPageId(id);
        }}
      />
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
      <Toasts
        toasts={toasts.list}
        onOpen={(id) => {
          toasts.dismiss(id);
          setPageId(id);
        }}
        onDismiss={toasts.dismiss}
      />
    </AppShell>
  );
}

/* ------------------------------------------------------------------ */
/* Notifications: only for things that need eyes now                   */
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
        if (!prev && (p.priority <= 2 || p.source === "detector" || p.source === "monitor")) add.push({ id: p.id, page: p, kind: "new" });
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
    <div className="pointer-events-none fixed bottom-3 right-3 z-[45] flex w-[min(360px,calc(100vw-24px))] flex-col-reverse gap-2">
      <AnimatePresence initial={false}>
        {toasts.map((t) => (
          <motion.div
            key={t.id + t.kind}
            initial={{ opacity: 0, x: 12 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: 12, transition: { duration: 0.15 } }}
            transition={{ duration: 0.18 }}
            className="pop pointer-events-auto relative cursor-pointer overflow-hidden py-3 pl-3.5 pr-2.5"
            onClick={() => onOpen(t.id)}
          >
            <div className="flex items-start gap-2.5">
              <PriorityChip p={t.page.priority} size="sm" />
              <div className="min-w-0 flex-1">
                <div className="text-sm font-medium text-ink">
                  {t.kind === "unassigned"
                    ? "No one accepted this page"
                    : t.page.source === "detector"
                      ? "Sepsis detector alert"
                      : t.page.source === "monitor"
                        ? `Room monitor · P${t.page.priority}`
                        : `New P${t.page.priority} alert`}
                  <span className="font-normal text-dim"> · Room {t.page.room ?? "?"}</span>
                </div>
                <div className="mt-0.5 line-clamp-2 text-xs text-muted">{t.page.text}</div>
              </div>
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  onDismiss(t.id);
                }}
                className={cx("grid h-6 w-6 shrink-0 place-items-center rounded-md text-dim transition-colors hover:bg-s3 hover:text-ink")}
                aria-label="Dismiss"
              >
                <CloseIcon />
              </button>
            </div>
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

