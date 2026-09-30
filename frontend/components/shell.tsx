"use client";

import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useRef, useState, type ReactNode } from "react";

import { api, writeSession } from "@/lib/api";
import { ROLE_LABEL, WORKSPACES } from "@/lib/ecosystem";
import { usePoll, useSession } from "@/lib/hooks";

import { cx, Pill } from "./ui";

type Health = {
  status: string;
  source: string;
  mode: string;
  persistence: boolean;
  model_loaded: boolean;
  prover: { healthy: boolean; fail_closed: boolean };
};

export function Mark({ size = 26 }: { size?: number }) {
  return (
    <span
      className="brand-fill grid shrink-0 place-items-center rounded-[8px] shadow-[inset_0_1px_0_rgb(255_255_255/.35)]"
      style={{ width: size, height: size }}
      aria-hidden
    >
      <svg width={size * 0.56} height={size * 0.56} viewBox="0 0 24 24" fill="none">
        <path d="M12 3 4.5 6v5.6c0 4.6 3.1 8.1 7.5 9.4 4.4-1.3 7.5-4.8 7.5-9.4V6L12 3Z" stroke="#050609" strokeWidth="2" strokeLinejoin="round" />
        <path d="M7.5 12.5h2.6l1.4-3 2.2 6 1.4-3h1.4" stroke="#050609" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
      </svg>
    </span>
  );
}

function BackendStatus() {
  const { data, error } = usePoll<Health>("/aegis/health", 15_000);
  if (error && !data)
    return (
      <Pill tone="med" dot>
        Backend waking…
      </Pill>
    );
  if (!data) return <Pill tone="nodata" dot>Connecting</Pill>;
  const proverOk = data.prover?.healthy && !data.prover?.fail_closed;
  return (
    <div className="hidden items-center gap-2 md:flex">
      <Pill tone={proverOk ? "neutral" : "high"} dot live={proverOk}>
        {proverOk ? "Prover healthy" : "Prover fail-closed"}
      </Pill>
      <span className="text-[11.5px] text-dim" title="Where the live session's telemetry comes from">
        {data.source === "simulator" ? "Demo stream" : `Live · ${data.source}`}
      </span>
    </div>
  );
}

function UserChip() {
  const s = useSession();
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const close = (e: MouseEvent) => ref.current && !ref.current.contains(e.target as Node) && setOpen(false);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  // The kernel's session secret is random per boot, so a backend restart signs
  // everyone out. Ask the kernel who we are; if it no longer knows this token,
  // drop it rather than keep showing a name the gate treats as a guest.
  useEffect(() => {
    if (!s) return;
    api<{ credential: { username: string } }>("/aegis/sys/whoami")
      .then((w) => {
        if (w.credential.username !== s.credential.username) writeSession(null);
      })
      .catch(() => {});
  }, [s]);

  if (!s)
    return (
      <Link href="/" className="rounded-md border border-line bg-s2 px-3 py-1.5 text-[12.5px] text-ink hover:bg-s3">
        Sign in
      </Link>
    );
  const c = s.credential;
  const initials = c.display.replace(/^Dr\.\s*/, "").split(/[\s,]+/).filter(Boolean).slice(0, 2).map((p) => p[0]).join("");
  const signOut = async () => {
    try {
      await api("/aegis/sys/logout", { method: "POST" });
    } catch {}
    writeSession(null);
    router.push("/");
  };
  return (
    <div className="relative" ref={ref}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-2 rounded-full border border-line bg-s2 py-1 pl-1 pr-3 hover:border-line-bright"
      >
        <span className="grid h-6 w-6 place-items-center rounded-full bg-s3 text-[10.5px] font-medium text-ink">{initials}</span>
        <span className="hidden text-left leading-tight sm:block">
          <span className="block text-[12px] text-ink">{c.display}</span>
          <span className="block text-[10.5px] text-dim">
            {ROLE_LABEL[c.role]} · ring {c.ring}
          </span>
        </span>
      </button>
      {open && (
        <div className="plate-float absolute right-0 top-11 z-50 w-64 rounded-md p-1.5 shadow-2xl">
          <div className="px-2.5 py-2">
            <div className="text-[13px] text-ink">{c.display}</div>
            <div className="text-[11.5px] text-dim">
              {c.title ?? ROLE_LABEL[c.role]} · uid {c.uid}
            </div>
          </div>
          <div className="my-1 h-px bg-line-soft" />
          <Link href="/" onClick={() => setOpen(false)} className="block rounded px-2.5 py-1.5 text-[12.5px] text-muted hover:bg-s2 hover:text-ink">
            Switch staff member
          </Link>
          <button onClick={signOut} className="block w-full rounded px-2.5 py-1.5 text-left text-[12.5px] text-muted hover:bg-s2 hover:text-ink">
            Sign out
          </button>
        </div>
      )}
    </div>
  );
}

export function TopBar({ context }: { context?: ReactNode }) {
  const path = usePathname();
  return (
    <header className="plate-float sticky top-0 z-40 border-x-0 border-t-0">
      <div className="mx-auto flex h-14 max-w-[1680px] items-center gap-4 px-4 lg:px-6">
        <Link href="/" className="flex items-center gap-2.5">
          <Mark />
          <span className="text-[15px] font-semibold tracking-tight">
            Aegis<span className="brand-text">OS</span>
          </span>
        </Link>
        <nav className="flex min-w-0 items-center gap-0.5 overflow-x-auto">
          {WORKSPACES.map((w) => {
            const active = path?.startsWith(w.href.replace(/\/$/, ""));
            return (
              <Link
                key={w.key}
                href={w.href}
                className={cx(
                  "whitespace-nowrap rounded-md px-3 py-1.5 text-[13px] transition",
                  active ? "bg-s3 text-ink" : "text-muted hover:bg-s2 hover:text-ink",
                )}
              >
                {w.label}
              </Link>
            );
          })}
        </nav>
        <div className="flex min-w-0 flex-1 items-center justify-center">{context}</div>
        <BackendStatus />
        <UserChip />
      </div>
    </header>
  );
}

export function Page({ children, context }: { children: ReactNode; context?: ReactNode }) {
  return (
    <>
      <TopBar context={context} />
      <main className="mx-auto w-full max-w-[1680px] px-4 pb-16 pt-5 lg:px-6">{children}</main>
    </>
  );
}

export function PageHead({ eyebrow, title, right, children }: {
  eyebrow?: ReactNode;
  title: ReactNode;
  right?: ReactNode;
  children?: ReactNode;
}) {
  return (
    <div className="mb-5 flex flex-wrap items-end gap-4">
      <div className="min-w-0 flex-1">
        {eyebrow && <div className="mb-1 text-[12px] text-dim">{eyebrow}</div>}
        <h1 className="text-[26px] font-semibold tracking-tight text-ink">{title}</h1>
        {children && <div className="mt-1 max-w-3xl text-[13.5px] text-muted">{children}</div>}
      </div>
      {right && <div className="flex flex-wrap items-center gap-2">{right}</div>}
    </div>
  );
}
