"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState, type ReactNode } from "react";

import { api, readSession, revalidate, writeSession, type Credential } from "@/lib/api";
import { ROLE_LABEL } from "@/lib/ecosystem";
import { useSession } from "@/lib/hooks";
import { clock } from "@/lib/paging";

import { cx } from "./ui";

/* ------------------------------------------------------------------ */
/* Identity                                                            */
/* ------------------------------------------------------------------ */
export function Mark({ size = 16 }: { size?: number }) {
  return (
    <svg width={size * 0.9} height={size} viewBox="0 0 36 40" aria-hidden className="shrink-0">
      <path d="M18 0 36 6.6V22.4L18 40 0 22.4V6.6Z" fill="#e6e9ed" />
      <path d="M18 0 36 6.6V22.4L18 40Z" fill="#9aa3ae" />
    </svg>
  );
}

export function Logo() {
  return (
    <Link href="/operator/" className="flex items-center gap-2" aria-label="Aegis operations">
      <Mark />
      <span className="text-[14px] font-semibold tracking-[-0.01em] text-ink">Aegis</span>
    </Link>
  );
}

/* ------------------------------------------------------------------ */
/* Session                                                             */
/* ------------------------------------------------------------------ */
type LoginOut = { status: string; token: string; credential: Credential; detail?: string };
const SIGNED_OUT = "aegis_signed_out";
const DEFAULT_OPERATOR = "rn.patel";

async function signInAs(username: string): Promise<boolean> {
  const r = await api<LoginOut>("/aegis/sys/login", { body: { username }, token: null, timeoutMs: 60_000 });
  if (r.status !== "ok") return false;
  writeSession({ token: r.token, credential: r.credential });
  try {
    localStorage.removeItem(SIGNED_OUT);
  } catch {}
  return true;
}

/** The demo hospital opens on the charge nurse's console. Someone who signed out
 *  stays signed out (view only) until they choose a user. */
export function useEnsureSession() {
  useEffect(() => {
    let off = false;
    try {
      off = localStorage.getItem(SIGNED_OUT) === "1";
    } catch {}
    const s = readSession();
    if (s) {
      // the kernel's secret may be random per boot: renew a session it forgot, drop one it ended
      revalidate(s)
        .then((v) => (v.state === "renewed" ? writeSession(v.session) : v.state === "ended" && writeSession(null)))
        .catch(() => {});
      return;
    }
    if (!off) signInAs(DEFAULT_OPERATOR).catch(() => {});
  }, []);
}

/* ------------------------------------------------------------------ */
/* System status                                                       */
/* ------------------------------------------------------------------ */
type Health = { status: string; source: string; prover?: { healthy: boolean; fail_closed: boolean } };
type Link_ = "connecting" | "online" | "waking" | "offline";

function useHealth() {
  const [st, setSt] = useState<{ link: Link_; ms: number | null; prover: boolean | null }>({ link: "connecting", ms: null, prover: null });
  useEffect(() => {
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const run = async () => {
      const t0 = performance.now();
      try {
        const d = await api<Health>("/aegis/health", { timeoutMs: 60_000 });
        if (!alive) return;
        setSt({ link: "online", ms: Math.round(performance.now() - t0), prover: !!d.prover?.healthy && !d.prover?.fail_closed });
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
    <span className="flex items-center gap-1.5 whitespace-nowrap text-[11.5px]" title={title}>
      <span className="h-1.5 w-1.5 rounded-full" style={{ background: color }} />
      <span className="text-muted">{label}</span>
      {value != null && <span className="num font-mono text-[11px] text-dim">{value}</span>}
    </span>
  );
}

export function StatusCluster() {
  const h = useHealth();
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
            label={h.prover ? "Prover" : "Prover fail-closed"}
            value={h.prover ? "OK" : undefined}
            title={h.prover ? "Z3 is healthy: every page is certified before it is sent" : "Z3 is unhealthy: nothing is sent until it recovers"}
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
  return <span className="num hidden font-mono text-[12px] text-muted sm:inline">{t}</span>;
}

/* ------------------------------------------------------------------ */
/* User                                                                */
/* ------------------------------------------------------------------ */
const SWITCH = [
  { u: "rn.patel", n: "Priya Patel, RN", r: "Charge nurse" },
  { u: "dr.whitfield", n: "Dr. James Whitfield", r: "Chief of critical care" },
  { u: "guest", n: "Guest", r: "View only" },
];

function initialsOf(name: string) {
  return name.replace(/^Dr\.\s*/, "").split(/[\s,]+/).filter(Boolean).slice(0, 2).map((p) => p[0]).join("").toUpperCase();
}

export function UserMenu() {
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
  const switchTo = async (u: string) => {
    setBusy(u);
    try {
      await signInAs(u);
      setOpen(false);
    } finally {
      setBusy(null);
    }
  };
  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex h-7 items-center gap-2 rounded-md pl-0.5 pr-1.5 transition-colors hover:bg-s2"
        aria-haspopup="menu"
        aria-expanded={open}
      >
        <span className="grid h-6 w-6 place-items-center rounded-full bg-s4 text-[10px] font-semibold text-ink">{c ? initialsOf(c.display) : "–"}</span>
        <span className="hidden text-[12.5px] text-ink lg:inline">{c ? c.display : "View only"}</span>
        <svg width="10" height="10" viewBox="0 0 10 10" className="text-dim" aria-hidden>
          <path d="M2 3.5 5 6.5 8 3.5" stroke="currentColor" fill="none" strokeWidth="1.4" strokeLinecap="round" />
        </svg>
      </button>
      {open && (
        <div role="menu" className="pop absolute right-0 top-9 z-[70] w-72 py-1">
          <div className="px-3 py-2">
            <div className="text-[13px] font-medium text-ink">{c ? c.display : "Not signed in"}</div>
            <div className="text-[11.5px] text-dim">
              {c ? `${ROLE_LABEL[c.role]} · privilege ring ${c.ring}` : "View only. Choose a user to act."}
              {c && <span className="ml-1.5 font-mono text-[11px] text-faint">{c.username}</span>}
            </div>
          </div>
          <div className="my-1 h-px bg-line" />
          <div className="px-3 pb-1 pt-1.5 text-[11px] font-medium text-dim">Switch user</div>
          {SWITCH.map((x) => (
            <button
              key={x.u}
              role="menuitem"
              disabled={!!busy}
              onClick={() => switchTo(x.u)}
              className="flex w-full items-center gap-2.5 px-3 py-1.5 text-left transition-colors hover:bg-s2 disabled:opacity-50"
            >
              <span className="grid h-5 w-5 shrink-0 place-items-center rounded-full bg-s3 text-[9px] font-semibold text-muted">{initialsOf(x.n)}</span>
              <span className="min-w-0 flex-1">
                <span className="block truncate text-[12.5px] text-ink">{x.n}</span>
                <span className="block text-[11px] text-dim">{x.r}</span>
              </span>
              {c?.username === x.u ? <span className="text-[11px] text-dim">current</span> : busy === x.u ? <span className="text-[11px] text-dim">…</span> : null}
            </button>
          ))}
          <div className="my-1 h-px bg-line" />
          <Link href="/clinician/" onClick={() => setOpen(false)} role="menuitem" className="block px-3 py-1.5 text-[12.5px] text-ink transition-colors hover:bg-s2">
            Open a clinician&apos;s pager
          </Link>
          {c && (
            <button
              role="menuitem"
              onClick={async () => {
                await api("/aegis/sys/logout", { method: "POST" }).catch(() => {});
                writeSession(null);
                try {
                  localStorage.setItem(SIGNED_OUT, "1");
                } catch {}
                setOpen(false);
                router.refresh();
              }}
              className="block w-full px-3 py-1.5 text-left text-[12.5px] text-muted transition-colors hover:bg-s2 hover:text-ink"
            >
              Sign out
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Navigation                                                          */
/* ------------------------------------------------------------------ */
export type NavKey = "overview" | "pages" | "staff" | "sentinel" | "policy" | "patients" | "pharmacy" | "evidence" | "system";

const NAV: { key: NavKey; label: string; href: string; group: number }[] = [
  { key: "overview", label: "Overview", href: "/operator/", group: 0 },
  { key: "pages", label: "Pages", href: "/operator/?view=pages", group: 0 },
  { key: "staff", label: "Staff", href: "/operator/?view=staff", group: 0 },
  { key: "sentinel", label: "Sentinel", href: "/operator/?view=sentinel", group: 0 },
  { key: "policy", label: "Policy", href: "/operator/?view=policy", group: 0 },
  { key: "patients", label: "Patients", href: "/patient/", group: 1 },
  { key: "pharmacy", label: "Pharmacy", href: "/pharmacy/", group: 1 },
  { key: "evidence", label: "Evidence", href: "/lab/", group: 2 },
  { key: "system", label: "System", href: "/os/", group: 2 },
];

function keyFromPath(path: string | null): NavKey | undefined {
  if (!path) return undefined;
  if (path.startsWith("/patient")) return "patients";
  if (path.startsWith("/pharmacy")) return "pharmacy";
  if (path.startsWith("/lab")) return "evidence";
  if (path.startsWith("/os")) return "system";
  return undefined;
}

export function NavBar({ active, counts = {}, alerts = {}, actions }: {
  active?: NavKey;
  counts?: Partial<Record<NavKey, number>>;
  alerts?: Partial<Record<NavKey, boolean>>;
  actions?: ReactNode;
}) {
  return (
    <nav className="flex h-10 shrink-0 items-stretch border-b border-line bg-bg pl-1 pr-2 sm:pl-2 sm:pr-3" aria-label="Sections">
      <div className="flex min-w-0 flex-1 items-stretch overflow-x-auto [scrollbar-width:none] max-md:[mask-image:linear-gradient(to_right,#000_82%,transparent)]">
        {NAV.map((n, i) => (
          <div key={n.key} className="flex shrink-0 items-stretch">
            {i > 0 && NAV[i - 1].group !== n.group && <span className="mx-1.5 my-3 w-px bg-line-bright" aria-hidden />}
            <Link
              href={n.href}
              className={cx(
                "relative flex items-center gap-1.5 px-2.5 text-[12.5px] font-medium transition-colors",
                active === n.key ? "text-ink" : "text-dim hover:text-ink",
              )}
              aria-current={active === n.key ? "page" : undefined}
            >
              {n.label}
              {counts[n.key] ? (
                <span className={cx("num rounded-[3px] px-1 font-mono text-[10.5px]", alerts[n.key] ? "bg-high/15 text-high-ink" : "bg-s3 text-muted")}>{counts[n.key]}</span>
              ) : null}
              {active === n.key && <span className="absolute inset-x-2.5 bottom-0 h-[2px] rounded-t-sm bg-ink" />}
            </Link>
          </div>
        ))}
      </div>
      {actions && <div className="flex shrink-0 items-center gap-2 pl-2">{actions}</div>}
    </nav>
  );
}

/* ------------------------------------------------------------------ */
/* Frames                                                              */
/* ------------------------------------------------------------------ */
export function SystemBar() {
  return (
    <header className="flex h-11 shrink-0 items-center gap-3 border-b border-line bg-bg px-3 sm:px-4">
      <Logo />
      <span className="hidden h-4 w-px bg-line-bright sm:block" aria-hidden />
      <span className="hidden text-[12.5px] text-muted sm:inline">Aegis Memorial</span>
      <span
        className="hidden rounded-[3px] border border-line-bright px-1 py-px font-mono text-[10px] tracking-wide text-dim sm:inline"
        title="Demonstration hospital. ICU rooms 301–312 replay real de-identified PhysioNet/CinC 2019 stays; other patients and nurse traffic are synthetic and labelled so."
      >
        DEMO
      </span>
      <div className="ml-auto flex items-center gap-4">
        <StatusCluster />
        <Clock />
        <UserMenu />
      </div>
    </header>
  );
}

/** Every screen: the system bar, the section bar, then the content. */
export function AppShell({ active, counts, alerts, actions, children, fill = false }: {
  active?: NavKey;
  counts?: Partial<Record<NavKey, number>>;
  alerts?: Partial<Record<NavKey, boolean>>;
  actions?: ReactNode;
  children: ReactNode;
  fill?: boolean;
}) {
  useEnsureSession();
  return (
    <div className={cx("flex flex-col", fill ? "h-[100dvh] overflow-hidden" : "min-h-[100dvh]")}>
      <SystemBar />
      <NavBar active={active} counts={counts} alerts={alerts} actions={actions} />
      {children}
    </div>
  );
}

/** A drill-down screen (patients, pharmacy, evidence, system). */
export function Page({ children, context }: { children: ReactNode; context?: ReactNode }) {
  const path = usePathname();
  return (
    <AppShell active={keyFromPath(path)}>
      <main className="mx-auto w-full max-w-[1680px] flex-1 px-3 pb-12 pt-3 sm:px-4 sm:pt-4">
        {context && <div className="mb-3 flex flex-wrap items-center gap-2">{context}</div>}
        {children}
      </main>
    </AppShell>
  );
}

export function PageHead({ eyebrow, title, right, children }: { eyebrow?: ReactNode; title: ReactNode; right?: ReactNode; children?: ReactNode }) {
  return (
    <div className="mb-4 flex flex-wrap items-end gap-3 border-b border-line pb-4">
      <div className="min-w-0 flex-1">
        {eyebrow && <div className="mb-0.5 text-[11.5px] font-medium text-dim">{eyebrow}</div>}
        <h1 className="text-[18px] font-semibold tracking-[-0.01em] text-ink">{title}</h1>
        {children && <p className="mt-1 max-w-3xl text-[12.5px] leading-relaxed text-muted">{children}</p>}
      </div>
      {right && <div className="flex flex-wrap items-center gap-2">{right}</div>}
    </div>
  );
}
