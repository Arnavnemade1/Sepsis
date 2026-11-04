"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";

import { api } from "@/lib/api";
import { ROLE_LABEL } from "@/lib/ecosystem";
import { useSession } from "@/lib/hooks";
import { clock } from "@/lib/paging";
import { signInAs, signOut, useEnsureSession } from "@/lib/session";

import { CommandPalette } from "./command";
import { CaretUpDown, Close, Hospital, Menu, Search, SignOut, Sidebar } from "./icons";
import { GROUP_OF, initialsOf, NAV_GROUPS, NAV_ITEMS, SWITCH_USERS, type NavItem, type NavKey } from "./nav";
import { cx } from "./ui";

export type { NavKey } from "./nav";
export { signInAs, signOut, useEnsureSession } from "@/lib/session";

/* ------------------------------------------------------------------ */
/* Identity                                                            */
/* ------------------------------------------------------------------ */
export function Mark({ size = 16 }: { size?: number }) {
  return (
    <svg width={size * 0.9} height={size} viewBox="0 0 36 40" aria-hidden className="shrink-0">
      <path d="M18 0 36 6.6V22.4L18 40 0 22.4V6.6Z" fill="#e8eaed" />
      <path d="M18 0 36 6.6V22.4L18 40Z" fill="#9aa3ae" />
    </svg>
  );
}

export function Logo({ compact = false }: { compact?: boolean }) {
  return (
    <Link href="/operator/" className="flex min-w-0 items-center gap-2.5 rounded-md" aria-label="AegisOS">
      <span className="grid h-7 w-7 shrink-0 place-items-center">
        <Mark size={17} />
      </span>
      {!compact && <span className="truncate text-base font-semibold tracking-tight text-ink">AegisOS</span>}
    </Link>
  );
}

/* ------------------------------------------------------------------ */
/* System status                                                       */
/* ------------------------------------------------------------------ */
type Health = { status: string; source: string; prover?: { healthy: boolean; fail_closed: boolean } };
type Link_ = "connecting" | "online" | "waking" | "offline";

export function useHealth() {
  const [st, setSt] = useState<{ link: Link_; ms: number | null; prover: boolean | null; source: string | null }>({ link: "connecting", ms: null, prover: null, source: null });
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const run = async () => {
      const t0 = performance.now();
      try {
        const d = await api<Health>("/aegis/health", { timeoutMs: 60_000 });
        if (!alive) return;
        setSt({ link: "online", ms: Math.round(performance.now() - t0), prover: !!d.prover?.healthy && !d.prover?.fail_closed, source: d.source ?? null });
      } catch {
        if (alive) setSt((s) => ({ ...s, link: s.link === "online" ? "offline" : "waking" }));
      } finally {
        if (alive && !document.hidden) timer = setTimeout(run, 15_000);
      }
    };
    const onVis = () => {
      if (!document.hidden && alive) {
        clearTimeout(timer);
        run();
      }
    };
    document.addEventListener("visibilitychange", onVis);
    run();
    return () => {
      alive = false;
      clearTimeout(timer);
      document.removeEventListener("visibilitychange", onVis);
    };
  }, []);
  return st;
}

function Indicator({ color, label, value, title }: { color: string; label: string; value?: ReactNode; title?: string }) {
  return (
    <span className="flex items-center gap-1.5 whitespace-nowrap text-xs" title={title}>
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: color, boxShadow: `0 0 0 3px ${color}22` }} />
      <span className="text-muted">{label}</span>
      {value != null && <span className="num text-dim">{value}</span>}
    </span>
  );
}

export function StatusCluster({ h }: { h: ReturnType<typeof useHealth> }) {
  const link =
    h.link === "online"
      ? { c: "#2fb67c", l: "Online", t: "Backend reachable" }
      : h.link === "offline"
        ? { c: "#e5484d", l: "Offline", t: "The backend stopped answering; retrying" }
        : { c: "#eeb12a", l: h.link === "waking" ? "Waking" : "Connecting", t: "A sleeping free-tier backend takes 30–60 s to wake" };
  return (
    <div className="flex items-center gap-4">
      <Indicator color={link.c} label={link.l} value={h.link === "online" && h.ms != null ? `${h.ms} ms` : undefined} title={link.t} />
      {h.prover != null && (
        <span className="hidden md:block">
          <Indicator
            color={h.prover ? "#2fb67c" : "#e5484d"}
            label={h.prover ? "Prover healthy" : "Prover fail-closed"}
            title={h.prover ? "Z3 is healthy: every page and order is certified before it leaves" : "Z3 is unhealthy: nothing is sent until it recovers"}
          />
        </span>
      )}
    </div>
  );
}

export function Clock() {
  const [t, setT] = useState<string>("");
  useEffect(() => {
    const f = () => setT(clock(Date.now() / 1000));
    f();
    const i = setInterval(f, 1000);
    return () => clearInterval(i);
  }, []);
  return <span className="num hidden text-xs font-medium text-muted sm:inline">{t}</span>;
}

/* ------------------------------------------------------------------ */
/* User                                                                */
/* ------------------------------------------------------------------ */
function UserBlock({ collapsed }: { collapsed: boolean }) {
  const s = useSession();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const close = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", esc);
    };
  }, []);
  const c = s?.credential;
  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        className={cx("flex w-full items-center gap-2.5 rounded-lg p-1.5 text-left transition-colors hover:bg-white/[0.05]", collapsed && "justify-center")}
        aria-haspopup="menu"
        aria-expanded={open}
        title={collapsed ? (c ? c.display : "View only") : undefined}
      >
        <span className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-s4 text-2xs font-semibold text-ink shadow-[inset_0_0_0_1px_rgb(255_255_255/.06)]">{c ? initialsOf(c.display) : "–"}</span>
        {!collapsed && (
          <>
            <span className="min-w-0 flex-1 leading-tight">
              <span className="block truncate text-sm font-medium text-ink">{c ? c.display : "View only"}</span>
              <span className="block truncate text-xs text-dim">{c ? ROLE_LABEL[c.role] : "Not signed in"}</span>
            </span>
            <CaretUpDown size={14} className="shrink-0 text-dim" />
          </>
        )}
      </button>
      {open && (
        <div role="menu" className={cx("pop absolute bottom-full z-[70] mb-2 w-64 p-1", collapsed ? "left-0" : "left-0 right-0 w-auto")}>
          <div className="px-2.5 py-2">
            <div className="text-sm font-medium text-ink">{c ? c.display : "Not signed in"}</div>
            <div className="text-xs text-dim">
              {c ? `${ROLE_LABEL[c.role]} · privilege ring ${c.ring}` : "View only. Choose a user to act."}
            </div>
            {c && <div className="mt-0.5 font-mono text-2xs text-faint">{c.username}</div>}
          </div>
          <div className="my-1 h-px bg-white/[0.06]" />
          <div className="px-2.5 pb-1 pt-1.5 text-xs font-medium text-dim">Switch user</div>
          {SWITCH_USERS.map((x) => (
            <button
              key={x.u}
              role="menuitem"
              disabled={!!busy}
              onClick={async () => {
                setBusy(x.u);
                try {
                  await signInAs(x.u);
                  setOpen(false);
                } finally {
                  setBusy(null);
                }
              }}
              className="flex w-full items-center gap-2.5 rounded-md px-2.5 py-1.5 text-left transition-colors hover:bg-white/[0.06] disabled:opacity-50"
            >
              <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-s4 text-[10px] font-semibold text-muted">{initialsOf(x.n)}</span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-sm text-ink">{x.n}</span>
                <span className="block text-xs text-dim">{x.r}</span>
              </span>
              {c?.username === x.u ? <span className="text-xs text-dim">Current</span> : busy === x.u ? <span className="text-xs text-dim">…</span> : null}
            </button>
          ))}
          {c && (
            <>
              <div className="my-1 h-px bg-white/[0.06]" />
              <button
                role="menuitem"
                onClick={async () => {
                  await signOut();
                  setOpen(false);
                  router.refresh();
                }}
                className="flex w-full items-center gap-2 rounded-md px-2.5 py-1.5 text-left text-sm text-muted transition-colors hover:bg-white/[0.06] hover:text-ink"
              >
                <SignOut size={15} />
                Sign out
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Sidebar                                                             */
/* ------------------------------------------------------------------ */
function NavLink({ item, active, count, alert, collapsed, onNavigate }: {
  item: NavItem;
  active: boolean;
  count?: number;
  alert?: boolean;
  collapsed: boolean;
  onNavigate?: () => void;
}) {
  const Icon = item.icon;
  return (
    <Link
      href={item.href}
      onClick={onNavigate}
      title={collapsed ? `${item.label}${count ? ` (${count})` : ""}` : undefined}
      aria-current={active ? "page" : undefined}
      className={cx(
        "group relative flex h-8 items-center gap-2.5 rounded-md px-2 text-sm transition-colors duration-100",
        active ? "bg-white/[0.07] font-medium text-ink shadow-[inset_0_0_0_1px_rgb(255_255_255/.04)]" : "text-muted hover:bg-white/[0.04] hover:text-ink",
        collapsed && "justify-center px-0",
      )}
    >
      <Icon size={18} weight={active ? "fill" : "regular"} className={cx("shrink-0", active ? "text-ink" : "text-dim group-hover:text-muted")} />
      {!collapsed && <span className="min-w-0 flex-1 truncate">{item.label}</span>}
      {!collapsed && count ? (
        <span className={cx("num min-w-5 rounded-full px-1.5 text-center text-2xs font-medium leading-[18px]", alert ? "bg-high/[0.16] text-high-ink" : "text-dim")}>{count}</span>
      ) : null}
      {collapsed && count ? <span className={cx("absolute right-1.5 top-1.5 h-1.5 w-1.5 rounded-full", alert ? "bg-high" : "bg-muted")} /> : null}
    </Link>
  );
}

function SidebarBody({ active, counts, alerts, collapsed, onToggle, onSearch, onNavigate }: {
  active?: NavKey;
  counts: Partial<Record<NavKey, number>>;
  alerts: Partial<Record<NavKey, boolean>>;
  collapsed: boolean;
  onToggle?: () => void;
  onSearch: () => void;
  onNavigate?: () => void;
}) {
  return (
    <div className="flex h-full flex-col">
      <div className={cx("flex h-14 shrink-0 items-center gap-2 px-3.5", collapsed && "justify-center px-0")}>
        <Logo compact={collapsed} />
        {onToggle && !collapsed && (
          <button onClick={onToggle} className="ml-auto grid h-7 w-7 place-items-center rounded-md text-faint transition-colors hover:bg-white/[0.05] hover:text-ink" aria-label="Collapse sidebar" title="Collapse sidebar">
            <Sidebar size={17} />
          </button>
        )}
      </div>
      <div className={cx("px-3", collapsed && "px-2")}>
        {!collapsed ? (
          <div className="mb-2 flex items-center gap-2.5 rounded-lg px-2 py-1.5" title="ICU rooms 301–312 replay real de-identified PhysioNet/CinC 2019 stays; other patients and the nurse traffic are synthetic and labelled so.">
            <span className="grid h-7 w-7 shrink-0 place-items-center rounded-md bg-white/[0.06] text-muted shadow-[inset_0_0_0_1px_rgb(255_255_255/.05)]">
              <Hospital size={15} />
            </span>
            <div className="min-w-0 flex-1 leading-tight">
              <div className="truncate text-sm font-medium text-ink">Aegis Memorial</div>
              <div className="truncate text-xs text-dim">Demonstration hospital</div>
            </div>
          </div>
        ) : null}
        <button
          onClick={onSearch}
          className={cx(
            "flex h-8 w-full items-center gap-2 rounded-md bg-white/[0.04] px-2.5 text-sm text-dim shadow-[inset_0_0_0_1px_rgb(255_255_255/.05)] transition-colors hover:bg-white/[0.06] hover:text-muted",
            collapsed && "justify-center px-0",
          )}
          title="Search and commands (⌘K)"
        >
          <Search size={15} className="shrink-0" />
          {!collapsed && (
            <>
              <span className="flex-1 text-left">Search</span>
              <span className="kbd">⌘K</span>
            </>
          )}
        </button>
      </div>
      <nav className="mt-4 min-h-0 flex-1 overflow-y-auto px-3 pb-3 [scrollbar-width:none]" aria-label="Sections">
        {NAV_GROUPS.map((g) => (
          <div key={g.label} className="mb-4">
            {!collapsed ? <div className="px-2 pb-1.5 text-xs font-medium text-faint">{g.label}</div> : <div className="mx-2 mb-2 h-px bg-line" />}
            <div className="space-y-px">
              {g.items.map((it) => (
                <NavLink key={it.key} item={it} active={active === it.key} count={counts[it.key]} alert={alerts[it.key]} collapsed={collapsed} onNavigate={onNavigate} />
              ))}
            </div>
          </div>
        ))}
      </nav>
      <div className={cx("shrink-0 p-2.5", collapsed && "flex flex-col items-center gap-1")}>
        {collapsed && onToggle && (
          <button onClick={onToggle} className="grid h-8 w-8 place-items-center rounded-md text-dim transition-colors hover:bg-white/[0.05] hover:text-ink" aria-label="Expand sidebar" title="Expand sidebar">
            <Sidebar size={17} mirrored />
          </button>
        )}
        <UserBlock collapsed={collapsed} />
      </div>
    </div>
  );
}

const SIDEBAR_KEY = "aegis_sidebar";

/** While the backend wakes (a free-tier host sleeps when idle), the content area says
 *  so instead of showing empty panels. Shown only before the first successful contact. */
function ConnectGate({ h }: { h: ReturnType<typeof useHealth> }) {
  const [t0] = useState(() => Date.now());
  const [now, setNow] = useState(() => Date.now());
  const [seen, setSeen] = useState(false);
  useEffect(() => {
    if (h.link === "online") setSeen(true);
  }, [h.link]);
  useEffect(() => {
    if (seen) return;
    const i = setInterval(() => setNow(Date.now()), 500);
    return () => clearInterval(i);
  }, [seen]);
  const elapsed = Math.floor((now - t0) / 1000);
  if (seen || elapsed < 2) return null;
  return (
    <div className="absolute inset-x-0 bottom-0 top-12 z-[30] grid place-items-center bg-bg/95 px-6">
      <div className="w-full max-w-[420px] rounded-xl border border-line bg-s1 p-6">
        <div className="flex items-center gap-3">
          <span className="grid h-9 w-9 place-items-center">
            <Mark size={20} />
          </span>
          <div>
            <div className="text-base font-semibold text-ink">Connecting to Aegis Memorial</div>
            <div className="text-xs text-dim">The backend runs on a free tier and sleeps when idle.</div>
          </div>
        </div>
        <ol className="mt-4 space-y-2 text-sm">
          <li className="flex items-center gap-2.5">
            <span className="blink h-1.5 w-1.5 rounded-full bg-med" />
            <span className="text-ink">Waking the backend</span>
            <span className="num ml-auto text-xs text-dim">{elapsed}s</span>
          </li>
          <li className="flex items-center gap-2.5 text-dim">
            <span className="h-1.5 w-1.5 rounded-full border border-faint" />
            Signing in to the console
          </li>
          <li className="flex items-center gap-2.5 text-dim">
            <span className="h-1.5 w-1.5 rounded-full border border-faint" />
            Subscribing to the hospital stream
          </li>
        </ol>
        <div className="mt-4 h-1 overflow-hidden rounded-full bg-white/[0.07]">
          <div className="working h-full w-full" />
        </div>
        <p className="mt-3 text-xs leading-relaxed text-dim">
          {elapsed < 75 ? "Usually 30–60 seconds. Everything loads on its own once it answers." : "Still waking. If this persists, the backend may be redeploying; it retries on its own."}
        </p>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Frame                                                               */
/* ------------------------------------------------------------------ */
/** Every screen: sidebar, top bar, content; plus the command palette and phone navigation. */
export function AppShell({ active, counts = {}, alerts = {}, actions, children, title }: {
  active?: NavKey;
  counts?: Partial<Record<NavKey, number>>;
  alerts?: Partial<Record<NavKey, boolean>>;
  actions?: ReactNode;
  children: ReactNode;
  title?: ReactNode;
  fill?: boolean;
}) {
  useEnsureSession();
  const health = useHealth();
  const [collapsed, setCollapsed] = useState(false);
  const [drawer, setDrawer] = useState(false);
  const [palette, setPalette] = useState(false);
  const pathname = usePathname();

  useEffect(() => {
    try {
      setCollapsed(localStorage.getItem(SIDEBAR_KEY) === "collapsed");
    } catch {}
  }, []);
  const toggle = useCallback(() => {
    setCollapsed((c) => {
      try {
        localStorage.setItem(SIDEBAR_KEY, c ? "expanded" : "collapsed");
      } catch {}
      return !c;
    });
  }, []);
  // ⌘K anywhere opens the palette
  useEffect(() => {
    const k = (e: KeyboardEvent) => {
      if (e.key.toLowerCase() === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        setPalette((p) => !p);
      }
    };
    window.addEventListener("keydown", k);
    return () => window.removeEventListener("keydown", k);
  }, []);
  useEffect(() => setDrawer(false), [pathname]);

  const item = NAV_ITEMS.find((i) => i.key === active);
  return (
    <div className="flex h-[100dvh] overflow-hidden bg-canvas">
      <aside className={cx("hidden shrink-0 transition-[width] duration-200 ease-out lg:block", collapsed ? "w-[64px]" : "w-[240px]")}>
        <SidebarBody active={active} counts={counts} alerts={alerts} collapsed={collapsed} onToggle={toggle} onSearch={() => setPalette(true)} />
      </aside>

      {/* phone / tablet: the sidebar as a drawer */}
      {drawer && (
        <div className="fixed inset-0 z-[65] lg:hidden">
          <div className="absolute inset-0 bg-black/45" onClick={() => setDrawer(false)} />
          <aside className="absolute inset-y-0 left-0 w-[280px] border-r border-line bg-canvas shadow-[24px_0_48px_-24px_rgb(0_0_0/.6)]">
            <button onClick={() => setDrawer(false)} className="absolute right-3 top-3.5 z-10 grid h-7 w-7 place-items-center rounded-md text-dim hover:bg-white/[0.05] hover:text-ink" aria-label="Close menu">
              <Close size={16} />
            </button>
            <SidebarBody
              active={active}
              counts={counts}
              alerts={alerts}
              collapsed={false}
              onSearch={() => {
                setDrawer(false);
                setPalette(true);
              }}
              onNavigate={() => setDrawer(false)}
            />
          </aside>
        </div>
      )}

      {/* the sheet: on a desktop the working area is inset from the frame, like a document on a desk */}
      <div className="relative flex min-w-0 flex-1 flex-col overflow-hidden bg-bg lg:my-2 lg:mr-2 lg:rounded-xl lg:border lg:border-line">
        <ConnectGate h={health} />
        <header className="flex h-12 shrink-0 items-center gap-3 border-b border-line px-3 sm:px-5">
          <button onClick={() => setDrawer(true)} className="grid h-8 w-8 place-items-center rounded-md text-muted transition-colors hover:bg-white/[0.05] hover:text-ink lg:hidden" aria-label="Open menu">
            <Menu size={18} />
          </button>
          <span className="lg:hidden">
            <Logo compact />
          </span>
          <div className="flex min-w-0 items-center gap-2 text-sm">
            {item && <span className="hidden text-dim sm:inline">{GROUP_OF[item.key]}</span>}
            {item && (
              <svg width="6" height="10" viewBox="0 0 6 10" className="hidden shrink-0 text-faint sm:block" aria-hidden>
                <path d="M1 1l4 4-4 4" fill="none" stroke="currentColor" strokeWidth="1.3" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            )}
            <span className="truncate font-medium text-ink">{title ?? item?.label ?? ""}</span>
          </div>
          <div className="ml-auto flex items-center gap-3 sm:gap-5">
            <span className="hidden sm:block">
              <StatusCluster h={health} />
            </span>
            <Clock />
            {actions}
            <button onClick={() => setPalette(true)} className="grid h-8 w-8 place-items-center rounded-md text-muted transition-colors hover:bg-white/[0.05] hover:text-ink lg:hidden" aria-label="Search">
              <Search size={17} />
            </button>
          </div>
        </header>
        {children}
      </div>

      <CommandPalette open={palette} onClose={() => setPalette(false)} />
    </div>
  );
}

/** A drill-down screen. */
export function Page({ children, context, active }: { children: ReactNode; context?: ReactNode; active?: NavKey }) {
  const path = usePathname();
  const key: NavKey | undefined =
    active ??
    (path?.startsWith("/room")
      ? "rooms"
      : path?.startsWith("/patient")
        ? "patients"
        : path?.startsWith("/pharmacy")
          ? "pharmacy"
          : path?.startsWith("/lab")
            ? "evidence"
            : path?.startsWith("/integrations")
              ? "integrations"
            : path?.startsWith("/os")
              ? "architecture"
              : undefined);
  return (
    <AppShell active={key}>
      <main className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto w-full max-w-[1680px] px-4 pb-16 pt-4 sm:px-6 sm:pt-6">
          {context && <div className="mb-4 flex flex-wrap items-center gap-2">{context}</div>}
          {children}
        </div>
      </main>
    </AppShell>
  );
}

export function PageHead({ eyebrow, title, right, children }: { eyebrow?: ReactNode; title: ReactNode; right?: ReactNode; children?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end gap-x-4 gap-y-3">
      {/* the text claims a full row on a phone, so the tags wrap below it rather than
          squeezing the paragraph into a column a few words wide */}
      <div className="min-w-0 flex-1 basis-[min(100%,30rem)]">
        {eyebrow && <div className="mb-1.5 text-xs font-medium text-dim">{eyebrow}</div>}
        <h1 className="text-xl font-semibold leading-tight tracking-tighter text-ink">{title}</h1>
        {children && <p className="mt-2 max-w-[68ch] text-sm leading-relaxed text-muted">{children}</p>}
      </div>
      {right && <div className="flex flex-wrap items-center gap-2">{right}</div>}
    </div>
  );
}

// kept for screens that import them by name
export { UserBlock as UserMenu };
