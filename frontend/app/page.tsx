"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useState } from "react";

import { Mark } from "@/components/shell";
import { cx, Pill, Skeleton } from "@/components/ui";
import { api, writeSession, type Credential } from "@/lib/api";
import { HOME, ROLE_LABEL, WORKSPACES, workspaceFor } from "@/lib/ecosystem";
import { usePoll, useSession } from "@/lib/hooks";
import type { Account } from "@/lib/types";

type Accounts = { accounts: Account[]; demo_login: boolean; note?: string };
type LoginOut = { status: string; token: string; credential: Credential; detail?: string };

export default function Launcher() {
  const router = useRouter();
  const session = useSession();
  const accounts = usePoll<Accounts>("/aegis/sys/accounts");
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const signIn = async (a: Account) => {
    setBusy(a.username);
    setErr(null);
    try {
      const out = await api<LoginOut>("/aegis/sys/login", { body: { username: a.username } });
      if (out.status !== "ok") throw new Error(out.detail || "sign-in refused");
      writeSession({ token: out.token, credential: { ...out.credential, title: a.title } });
      router.push(workspaceFor(HOME[out.credential.role]).href);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };

  return (
    <main className="mx-auto flex min-h-screen w-full max-w-[1180px] flex-col px-5 py-10 lg:py-16">
      <div className="flex items-center gap-3">
        <Mark size={34} />
        <div className="text-[20px] font-semibold tracking-tight">
          Aegis<span className="brand-text">OS</span>
        </div>
        <span className="ml-auto">
          <Pill tone="nodata">Research use only · not a medical device</Pill>
        </span>
      </div>

      <section className="mt-12 max-w-3xl rise">
        <h1 className="text-[40px] font-semibold leading-[1.08] tracking-tight lg:text-[52px]">
          One operating system for the whole ICU.
        </h1>
        <p className="mt-4 text-[16px] leading-relaxed text-muted">
          A wearable, a camera and the chart feed one running model of every patient: a calibrated sepsis risk,
          an organ-by-organ body state, and treatment options that each carry a formal proof of safety. Each role
          works in its own workspace, and all of them read the same live system.
        </p>
      </section>

      <section className="mt-12">
        <div className="mb-3 flex items-baseline gap-3">
          <h2 className="text-[15px] font-medium">Sign in as</h2>
          <span className="text-[12.5px] text-dim">Demo staff. Each role is a privilege ring the kernel gate enforces.</span>
        </div>
        {err && <div className="mb-3 rounded-md border border-high/30 bg-high/10 px-3 py-2 text-[13px] text-high-ink">{err}</div>}
        {accounts.error && !accounts.data && (
          <div className="plate px-4 py-3 text-[13px] text-muted">
            The backend is waking up (a free-tier service sleeps when idle; this takes about 30 seconds).
            This page retries on its own.
          </div>
        )}
        <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
          {!accounts.data &&
            Array.from({ length: 6 }).map((_, i) => <Skeleton key={i} className="h-[92px] rounded-lg" />)}
          {accounts.data?.accounts
            .filter((a) => a.demo)
            .map((a) => {
              const home = workspaceFor(HOME[a.role]);
              const current = session?.credential.username === a.username;
              return (
                <button
                  key={a.username}
                  disabled={!!busy}
                  onClick={() => signIn(a)}
                  className={cx(
                    "plate group flex items-start gap-3 p-4 text-left transition duration-200 hover:-translate-y-px hover:border-line-bright disabled:opacity-60",
                    current && "border-accent/40",
                  )}
                >
                  <span className="grid h-9 w-9 shrink-0 place-items-center rounded-full bg-s3 text-[12px] font-medium">
                    {a.display.replace(/^Dr\.\s*/, "").split(/[\s,]+/).filter(Boolean).slice(0, 2).map((p) => p[0]).join("")}
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="block truncate text-[14px] text-ink">{a.display}</span>
                    <span className="block truncate text-[12px] text-muted">{a.title}</span>
                    <span className="mt-2 flex items-center gap-2 text-[11.5px] text-dim">
                      <span>
                        {ROLE_LABEL[a.role]} · ring {a.ring}
                      </span>
                      <span className="ml-auto text-muted transition group-hover:text-ink">
                        {busy === a.username ? "Signing in…" : current ? `Continue in ${home.label} →` : `${home.label} →`}
                      </span>
                    </span>
                  </span>
                </button>
              );
            })}
        </div>
      </section>

      <section className="mt-14">
        <h2 className="mb-3 text-[15px] font-medium">Workspaces</h2>
        <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-5">
          {WORKSPACES.map((w, i) => (
            <Link
              key={w.key}
              href={w.href}
              className="plate rise flex flex-col p-4 transition hover:border-line-bright"
              style={{ animationDelay: `${i * 50}ms` }}
            >
              <span className="text-[14px] font-medium text-ink">{w.label}</span>
              <span className="mt-0.5 text-[11.5px] text-dim">{w.who}</span>
              <span className="mt-3 text-[12.5px] leading-relaxed text-muted">{w.blurb}</span>
            </Link>
          ))}
        </div>
      </section>

      <footer className="mt-auto pt-16 text-[12px] leading-relaxed text-dim">
        AegisOS is a research system. It is not a medical device and must not be used for clinical decisions. The
        ward is replayed de-identified PhysioNet/CinC 2019 ICU records; the live session is a camera or wearable
        feed, or a simulator when neither is connected, and every screen says which.
      </footer>
    </main>
  );
}
