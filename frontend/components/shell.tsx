"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState, type ReactNode } from "react";

import { api, revalidate, writeSession } from "@/lib/api";
import { ROLE_LABEL } from "@/lib/ecosystem";
import { usePoll, useSession } from "@/lib/hooks";
import { clock } from "@/lib/paging";

import { cx, Pill } from "./ui";

export function Mark({ size = 26 }: { size?: number }) {
  return (
    <span className="grid shrink-0 place-items-center rounded-[9px] bg-ink shadow-[inset_0_1px_0_rgb(255_255_255/.14),0_1px_2px_rgb(0_0_0/.2)]" style={{ width: size, height: size }} aria-hidden>
      <svg width={size * 0.6} height={size * 0.6} viewBox="0 0 24 24" fill="none">
        <path d="M12 2.8 4.6 5.8v5.6c0 4.6 3.1 8.3 7.4 9.8 4.3-1.5 7.4-5.2 7.4-9.8V5.8L12 2.8Z" stroke="url(#mk)" strokeWidth="1.9" strokeLinejoin="round" />
        <path d="M7.6 12.4h2.5l1.5-3.1 2.1 6 1.4-2.9h1.5" stroke="#fff" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
        <defs>
          <linearGradient id="mk" x1="4" y1="3" x2="20" y2="21">
            <stop stopColor="#8b9bff" />
            <stop offset="1" stopColor="#c9a8ff" />
          </linearGradient>
        </defs>
      </svg>
    </span>
  );
}

export function Wordmark() {
  return (
    <Link href="/" className="flex items-center gap-2.5">
      <Mark />
      <span className="text-[15px] font-semibold tracking-tight text-ink">AegisOS</span>
    </Link>
  );
}

type Health = { status: string; source: string; prover: { healthy: boolean; fail_closed: boolean } };

export function BackendPill() {
  const { data, error } = usePoll<Health>("/aegis/health", 20_000);
  if (error && !data) return <Pill tone="med" dot>Waking backend…</Pill>;
  if (!data) return <Pill tone="nodata" dot>Connecting</Pill>;
  const ok = data.prover?.healthy && !data.prover?.fail_closed;
  return (
    <Pill tone={ok ? "low" : "high"} dot live={ok}>
      {ok ? "Live" : "Prover fail-closed"}
    </Pill>
  );
}

export function Clock() {
  const [t, setT] = useState<string>("");
  useEffect(() => {
    const f = () => setT(clock(Date.now() / 1000, false));
    f();
    const i = setInterval(f, 10_000);
    return () => clearInterval(i);
  }, []);
  return <span className="num hidden font-mono text-[12px] text-dim md:inline">{t}</span>;
}

const MORE = [
  { href: "/operator/", label: "Operator console", hint: "Floor, queue, staff, policy" },
  { href: "/patient/", label: "Patients", hint: "Twin, risk trend, orders" },
  { href: "/pharmacy/", label: "Pharmacy", hint: "Treatment paths, interactions" },
  { href: "/lab/", label: "Lab", hint: "The evidence behind the detector" },
  { href: "/os/", label: "System", hint: "Kernel, journal, shell" },
];

export function UserMenu() {
  const s = useSession();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const close = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);
  // the kernel's session secret may be random per boot: renew a session it forgot,
  // drop one it ended
  useEffect(() => {
    if (!s) return;
    revalidate(s)
      .then((v) => v.state === "renewed" ? writeSession(v.session) : v.state === "ended" && writeSession(null))
      .catch(() => {});
  }, [s]);
  const c = s?.credential;
  const initials = c ? c.display.replace(/^Dr\.\s*/, "").split(/[\s,]+/).filter(Boolean).slice(0, 2).map((p) => p[0]).join("") : "?";
  return (
    <div className="relative" ref={ref}>
      <button onClick={() => setOpen((o) => !o)} className="flex items-center gap-2 rounded-full border border-line bg-s1 py-1 pl-1 pr-2.5 shadow-[0_1px_1px_rgb(16_18_24/.04)] hover:border-line-bright">
        <span className="grid h-6 w-6 place-items-center rounded-full bg-s3 text-[10.5px] font-semibold text-ink">{initials}</span>
        <span className="hidden text-left leading-tight sm:block">
          <span className="block text-[12px] font-medium text-ink">{c ? c.display : "Not signed in"}</span>
          <span className="block text-[10.5px] text-dim">{c ? `${ROLE_LABEL[c.role]} · ring ${c.ring}` : "view only"}</span>
        </span>
        <svg width="10" height="10" viewBox="0 0 10 10" className="text-dim" aria-hidden><path d="M2 3.5 5 6.5 8 3.5" stroke="currentColor" fill="none" strokeWidth="1.4" strokeLinecap="round" /></svg>
      </button>
      {open && (
        <div className="pop absolute right-0 top-11 z-50 w-72 p-1.5">
          {c && c.username !== "guest" && c.username !== "rn.patel" && c.role !== "chief" && (
            <Link href={`/clinician/?id=${c.username}`} onClick={() => setOpen(false)} className="mb-1 flex items-center gap-2 rounded-lg bg-accent-soft px-3 py-2 text-[12.5px] font-medium text-accent-ink">
              Open my pager →
            </Link>
          )}
          {MORE.map((m) => (
            <Link key={m.href} href={m.href} onClick={() => setOpen(false)} className="block rounded-lg px-3 py-1.5 hover:bg-s2">
              <span className="block text-[12.5px] font-medium text-ink">{m.label}</span>
              <span className="block text-[11px] text-dim">{m.hint}</span>
            </Link>
          ))}
          <div className="my-1 h-px bg-line-soft" />
          <Link href="/" onClick={() => setOpen(false)} className="block rounded-lg px-3 py-1.5 text-[12.5px] text-muted hover:bg-s2 hover:text-ink">
            Switch staff member
          </Link>
          {c && (
            <button
              onClick={async () => {
                await api("/aegis/sys/logout", { method: "POST" }).catch(() => {});
                writeSession(null);
                setOpen(false);
                router.push("/");
              }}
              className="block w-full rounded-lg px-3 py-1.5 text-left text-[12.5px] text-muted hover:bg-s2 hover:text-ink"
            >
              Sign out
            </button>
          )}
        </div>
      )}
    </div>
  );
}

/** The bar every screen shares. */
export function AppBar({ title, center, right }: { title?: ReactNode; center?: ReactNode; right?: ReactNode }) {
  return (
    <header className="plate-float sticky top-0 z-40 border-x-0 border-t-0">
      <div className="mx-auto flex h-14 max-w-[1760px] items-center gap-4 px-4 lg:px-6">
        <Wordmark />
        {title && (
          <>
            <span className="hidden h-5 w-px bg-line md:block" />
            <div className="hidden min-w-0 truncate text-[13px] text-muted md:block">{title}</div>
          </>
        )}
        <div className="flex min-w-0 flex-1 items-center justify-center">{center}</div>
        {right}
        <span className="hidden sm:contents">
          <BackendPill />
        </span>
        <Clock />
        <UserMenu />
      </div>
    </header>
  );
}

/** Secondary screens (patients, pharmacy, lab, system): a bar with the way back. */
export function Page({ children, context }: { children: ReactNode; context?: ReactNode }) {
  const path = usePathname();
  const links = MORE.filter((m) => m.href !== "/operator/");
  return (
    <>
      <AppBar
        title={
          <nav className="flex items-center gap-0.5">
            <Link href="/operator/" className="rounded-md px-2 py-1 text-[12.5px] text-muted hover:bg-s2 hover:text-ink">
              ← Operator
            </Link>
            {links.map((l) => (
              <Link
                key={l.href}
                href={l.href}
                className={cx("rounded-md px-2 py-1 text-[12.5px] transition", path?.startsWith(l.href.replace(/\/$/, "")) ? "bg-s2 font-medium text-ink" : "text-muted hover:bg-s2 hover:text-ink")}
              >
                {l.label}
              </Link>
            ))}
          </nav>
        }
        center={context}
      />
      <main className="mx-auto w-full max-w-[1680px] px-4 pb-16 pt-6 lg:px-6">{children}</main>
    </>
  );
}

export function PageHead({ eyebrow, title, right, children }: { eyebrow?: ReactNode; title: ReactNode; right?: ReactNode; children?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-end gap-4">
      <div className="min-w-0 flex-1">
        {eyebrow && <div className="mb-1 font-mono text-[10.5px] uppercase tracking-[0.12em] text-dim">{eyebrow}</div>}
        <h1 className="serif text-[34px] leading-[1.05] text-ink">{title}</h1>
        {children && <div className="mt-2 max-w-3xl text-[13.5px] leading-relaxed text-muted">{children}</div>}
      </div>
      {right && <div className="flex flex-wrap items-center gap-2">{right}</div>}
    </div>
  );
}
