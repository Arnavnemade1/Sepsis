"use client";

import { AnimatePresence, motion } from "motion/react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Avatar, Countdown, P_INK, P_RGB, PriorityChip, SourceTag } from "@/components/paging/bits";
import { Mark } from "@/components/shell";
import { Button, cx } from "@/components/ui";
import { api, readDeviceSession, revalidate, writeDeviceSession, type Credential } from "@/lib/api";
import { CATEGORY_LABEL, clock, PAGE_STATUS_LABEL, PRIORITY_LABEL, shortName, since, STATUS_LABEL, usePaging, useServerNow, type Clinician, type Page, type StaffStatus } from "@/lib/paging";

export default function ClinicianPage() {
  return (
    <Suspense>
      <Device />
    </Suspense>
  );
}

type Login = { status: string; token: string; credential: Credential; detail?: string };

/** The device's own session for clinician ``id``. ``renew`` signs in again when the
 *  kernel has forgotten the token (a backend restart), and returns the token to use. */
function useDeviceToken(id: string | null) {
  const [token, setToken] = useState<string | null | undefined>(undefined);
  const [err, setErr] = useState<string | null>(null);
  const tokenRef = useRef<string | null>(null);
  const login = useCallback(async (who: string) => {
    const r = await api<Login>("/aegis/sys/login", { body: { username: who }, token: null, timeoutMs: 60_000 });
    if (r.status !== "ok") throw new Error(r.detail ?? "sign-in refused");
    writeDeviceSession(who, { token: r.token, credential: r.credential });
    tokenRef.current = r.token;
    setToken(r.token);
    return r.token;
  }, []);
  const renew = useCallback(async (): Promise<string | null> => {
    if (!id) return null;
    const cached = readDeviceSession(id);
    try {
      if (cached) {
        const v = await revalidate(cached);
        const good = v.state === "ok" ? cached : v.state === "renewed" ? v.session : null;
        if (good) {
          writeDeviceSession(id, good);
          tokenRef.current = good.token;
          setToken(good.token);
          return good.token;
        }
      }
      // no session yet, or the kernel ended it: opening a clinician's pager signs them in
      return await login(id);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
      return tokenRef.current;
    }
  }, [id, login]);
  useEffect(() => {
    if (!id) return;
    let alive = true;
    const start = async () => {
      const t = await renew();
      if (!alive) return;
      if (t) setErr(null);
      else setTimeout(start, 5000);
    };
    start();
    return () => {
      alive = false;
    };
  }, [id, renew]);
  return { token, err, renew };
}

function Device() {
  const params = useSearchParams();
  const id = params.get("id");
  const { token, err, renew } = useDeviceToken(id);
  const { state, status, epoch } = usePaging({ clinician: token ? id : null, token: token ?? null });
  // a reconnect may follow a backend restart that forgot this device's session
  useEffect(() => {
    if (epoch > 1) renew();
  }, [epoch, renew]);
  // every write from the device goes AS its clinician; a forgotten session is renewed once
  const call = useCallback(
    async <T extends { status: string; errno?: string; detail?: string }>(path: string, body: Record<string, unknown>): Promise<T> => {
      const r = await api<T>(path, { body, token: token ?? null });
      if (r.status === "ok" || !["EPERM", "EAUTH"].includes(r.errno ?? "")) return r;
      const fresh = await renew();
      return fresh && fresh !== token ? api<T>(path, { body, token: fresh }) : r;
    },
    [token, renew],
  );
  const now = useServerNow(state.offsetMs);
  const me = id ? state.clinicians[id] : undefined;

  if (!id) return <Picker />;
  return (
    <div className="min-h-screen bg-bg sm:bg-[radial-gradient(120%_80%_at_50%_0%,#ffffff_0%,#f6f6f3_55%,#ecece7_100%)]">
      {/* on a phone the page IS the pager; on anything wider it is shown as a device, with the why beside it */}
      <div className="mx-auto flex max-w-[1180px] flex-col items-center sm:gap-10 sm:px-5 sm:py-8 lg:flex-row lg:items-center lg:justify-between lg:py-12">
        <aside className="hidden max-w-md sm:block lg:pt-10">
          <Link href="/" className="inline-flex items-center gap-2.5">
            <Mark />
            <span className="text-[15px] font-semibold tracking-tight">AegisOS</span>
          </Link>
          <h1 className="serif mt-8 text-[44px] leading-[1.02] text-ink">
            {me ? me.name.split(",")[0] : "Pager"}
            <span className="block text-dim">on call, in your pocket.</span>
          </h1>
          <p className="mt-4 text-[15px] leading-relaxed text-muted">
            When a nurse raises an alert or the sepsis detector crosses its line, AegisOS decides who should go. If it is you, this rings. Accept and a brief is waiting; decline or let it ring out and the next person is paged automatically.
          </p>
          <div className="mt-6 rounded-xl border border-line bg-s1/80 p-4 text-[13px] leading-relaxed text-muted">
            <div className="mb-1 font-medium text-ink">Try it</div>
            Open the{" "}
            <Link href="/operator/" target="_blank" className="font-medium text-accent-ink underline decoration-accent/30 underline-offset-2">
              operator console
            </Link>{" "}
            in another window and raise an alert this clinician covers
            {me ? ` (${me.specialty_label.toLowerCase()})` : ""}. While this page is open, nobody answers for {me ? shortName(me.name) : "them"}.
          </div>
          {err && <p className="mt-3 text-[12.5px] text-high-ink">Signing in: {err}</p>}
        </aside>
        <Phone>
          {!state.ready || !me ? (
            <div className="grid h-full place-items-center text-[13px] text-dim">{status === "live" ? "Loading…" : "Connecting to the hospital…"}</div>
          ) : (
            <Screen me={me} pages={state.pages} clinicians={state.clinicians} now={now} call={call} notifications={state.notifications} />
          )}
        </Phone>
      </div>
    </div>
  );
}

function Phone({ children }: { children: React.ReactNode }) {
  return (
    <div className="relative w-full shrink-0 sm:w-auto sm:rounded-[52px] sm:bg-[#16181d] sm:p-[11px] sm:shadow-[0_50px_100px_-30px_rgb(16_18_24/.55),inset_0_0_0_1.5px_rgb(255_255_255/.08)]">
      <div className="relative h-[100dvh] w-full overflow-hidden bg-bg sm:h-[780px] sm:w-[370px] sm:rounded-[42px]">
        <div className="absolute left-1/2 top-2.5 z-30 hidden h-[30px] w-[110px] -translate-x-1/2 rounded-full bg-[#0c0d10] sm:block" />
        {children}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The device screen                                                   */
/* ------------------------------------------------------------------ */
type Call = <T extends { status: string; errno?: string; detail?: string }>(path: string, body: Record<string, unknown>) => Promise<T>;

function Screen({ me, pages, clinicians, now, call, notifications }: {
  me: Clinician;
  pages: Record<string, Page>;
  clinicians: Record<string, Clinician>;
  now: number;
  call: Call;
  notifications: { to: string | null; page: string; kind: string; text: string; ts: number; priority: number; room: string | null }[];
}) {
  const ringing = useMemo(
    () =>
      Object.values(pages)
        .filter((p) => p.status === "paging" && p.attempts.at(-1)?.clinician === me.id)
        .sort((a, b) => a.priority - b.priority)[0] ?? null,
    [pages, me.id],
  );
  const accepted = useMemo(
    () => Object.values(pages).filter((p) => p.status === "accepted" && p.assignee === me.id).sort((a, b) => a.priority - b.priority),
    [pages, me.id],
  );
  const history = useMemo(
    () =>
      Object.values(pages)
        .filter((p) => p.attempts.some((a) => a.clinician === me.id) && !(p.status === "accepted" && p.assignee === me.id) && p.id !== ringing?.id)
        .sort((a, b) => b.created_at - a.created_at)
        .slice(0, 8),
    [pages, me.id, ringing],
  );
  const fyi = notifications.filter((n) => n.to === me.id && n.kind === "fyi").slice(-3).reverse();
  const [open, setOpen] = useState<string | null>(null);
  useAlertTone(ringing?.id ?? null, ringing?.priority ?? 4);

  const setStatus = async (s: StaffStatus) => {
    await call(`/aegis/paging/clinicians/${me.id}/status`, { status: s }).catch(() => {});
  };

  const detail = open ? pages[open] : accepted[0] ?? null;
  return (
    <div className="relative flex h-full flex-col">
      <div className="hidden h-11 items-end justify-between px-7 pb-1 font-mono text-[12px] font-semibold text-ink sm:flex">
        <span>{clock(now, false)}</span>
        <span className="inline-flex items-center gap-1 text-[11px]">
          <span className="h-1.5 w-1.5 rounded-full bg-low" />
          AegisOS
        </span>
      </div>
      <div className="flex items-center gap-3 px-5 pb-3 pt-[max(1rem,env(safe-area-inset-top))] sm:pt-3">
        <Avatar name={me.name} status={me.status} size={40} present />
        <div className="min-w-0 flex-1">
          <div className="truncate text-[15px] font-semibold text-ink">{me.name}</div>
          <div className="truncate text-[12px] text-dim">
            {me.specialty_label} · {me.on_call ? "on call" : me.title}
          </div>
        </div>
      </div>
      <div className="mx-5 mb-3 grid grid-cols-3 gap-1 rounded-xl bg-s2 p-1">
        {(["available", "in_procedure", "off_shift"] as StaffStatus[]).map((s) => {
          const active = me.status === s || (s === "available" && (me.status === "paging" || me.status === "on_case"));
          return (
            <button key={s} onClick={() => setStatus(s)} className={cx("h-8 rounded-lg text-[11.5px] font-medium transition", active ? "bg-s1 text-ink shadow-sm" : "text-dim hover:text-ink")}>
              {s === "available" ? (me.status === "on_case" ? "On a case" : "Available") : STATUS_LABEL[s]}
            </button>
          );
        })}
      </div>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-4 pb-6">
        {detail && !open && accepted.length > 0 && <CaseCard page={detail} now={now} call={call} />}
        {open && pages[open] && (
          <div>
            <button onClick={() => setOpen(null)} className="mb-2 text-[12px] text-accent-ink">← Back</button>
            <CaseCard page={pages[open]} now={now} call={call} readOnly={pages[open].assignee !== me.id || pages[open].status !== "accepted"} />
          </div>
        )}
        {!open && fyi.length > 0 && (
          <div className="space-y-1.5">
            {fyi.map((n) => (
              <div key={n.ts + n.page} className="rounded-xl border border-accent/20 bg-accent-soft/70 px-3 py-2">
                <div className="text-[11px] font-medium uppercase tracking-wide text-accent-ink">FYI · room {n.room}</div>
                <div className="text-[12.5px] text-ink">{n.text}</div>
              </div>
            ))}
          </div>
        )}
        {!open && accepted.length === 0 && !ringing && (
          <div className="grid place-items-center rounded-2xl border border-dashed border-line-bright px-6 py-12 text-center">
            <svg width="30" height="30" viewBox="0 0 24 24" fill="none" className="text-faint">
              <path d="M6 8a6 6 0 1 1 12 0c0 7 3 9 3 9H3s3-2 3-9Z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
              <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            <div className="mt-2 text-[14px] font-medium text-ink">Quiet for now</div>
            <div className="mt-1 text-[12.5px] text-dim">Pages for you ring here.</div>
          </div>
        )}
        {!open && history.length > 0 && (
          <div>
            <div className="px-1 pb-1.5 pt-2 font-mono text-[10.5px] uppercase tracking-[0.12em] text-dim">Recent</div>
            <div className="divide-y divide-line-soft overflow-hidden rounded-xl border border-line bg-s1">
              {history.map((p) => {
                const mine = p.attempts.filter((a) => a.clinician === me.id).at(-1);
                return (
                  <button key={p.id} onClick={() => setOpen(p.id)} className="flex w-full items-center gap-2.5 px-3 py-2.5 text-left hover:bg-s2">
                    <PriorityChip p={p.priority} size="sm" />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-[12.5px] text-ink">{p.room} · {p.text}</span>
                      <span className="block text-[11px] text-dim">
                        {mine?.outcome ? (mine.outcome === "timeout" ? "you didn't answer" : `you ${mine.outcome}`) : PAGE_STATUS_LABEL[p.status]} · {since(p.created_at, now)}
                      </span>
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </div>

      <AnimatePresence>{ringing && <Incoming key={ringing.id} page={ringing} me={me} now={now} call={call} clinicians={clinicians} />}</AnimatePresence>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* A page ringing                                                      */
/* ------------------------------------------------------------------ */
function Incoming({ page, me, now, call, clinicians }: { page: Page; me: Clinician; now: number; call: Call; clinicians: Record<string, Clinician> }) {
  const att = page.attempts.at(-1)!;
  const [busy, setBusy] = useState<string | null>(null);
  const [declining, setDeclining] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const respond = async (action: "accept" | "decline", reason?: string) => {
    setBusy(action);
    setErr(null);
    try {
      const r = await call<{ status: string; errno?: string; detail?: string }>(`/aegis/paging/pages/${page.id}/respond`, { action, reason, clinician: me.id });
      if (r.status !== "ok") setErr(r.detail ?? r.status);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };
  const tint = P_RGB[page.priority];
  const prev = page.attempts.length > 1 ? page.attempts[page.attempts.length - 2] : null;
  return (
    <motion.div
      className="absolute inset-0 z-20 flex flex-col"
      initial={{ y: "100%" }}
      animate={{ y: 0 }}
      exit={{ y: "100%", transition: { duration: 0.3 } }}
      transition={{ type: "spring", bounce: 0.05, duration: 0.55 }}
      style={{ backgroundColor: "#fff", backgroundImage: `linear-gradient(180deg, rgb(${tint} / .16) 0%, rgb(${tint} / 0) 42%)` }}
    >
      <div className="h-[env(safe-area-inset-top)] sm:h-11" />
      <div className="flex flex-1 flex-col px-6 pt-4">
        <div className="flex items-center gap-2">
          <span className="relative flex h-3 w-3">
            <span className="absolute inset-0 rounded-full" style={{ background: `rgb(${tint})`, animation: "ring 1.4s cubic-bezier(.16,1,.3,1) infinite" }} />
            <span className="relative h-3 w-3 rounded-full" style={{ background: `rgb(${tint})` }} />
          </span>
          <span className="text-[12px] font-semibold uppercase tracking-[0.14em]" style={{ color: P_INK[page.priority] }}>
            Incoming page
          </span>
          <span className="ml-auto">
            <Countdown remaining={att.deadline - now} total={att.deadline - att.sent_at} priority={page.priority} size={44} />
          </span>
        </div>
        <div className="mt-6 flex items-center gap-2">
          <PriorityChip p={page.priority} size="lg" label />
          <span className="text-[12.5px] text-dim">{CATEGORY_LABEL[page.category]}</span>
        </div>
        <div className="serif mt-4 text-[54px] leading-none text-ink">Room {page.room}</div>
        <div className="mt-1 text-[14px] text-muted">
          {page.zone_name}
          {page.patient?.name ? ` · ${page.patient.name}` : ""}
          {page.patient?.age ? `, ${page.patient.age}${page.patient.sex ?? ""}` : ""}
        </div>
        <p className="mt-5 text-[17px] leading-relaxed text-ink">{page.text}</p>
        <div className="mt-3 flex items-center gap-2 text-[12px] text-dim">
          <SourceTag source={page.source} synthetic={page.synthetic} />
          {prev && <span>· {prev.name} {prev.outcome === "timeout" ? "didn't answer" : prev.outcome}</span>}
        </div>
        {page.triage.red_flags.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-1.5">
            {page.triage.red_flags.slice(0, 3).map((f) => (
              <span key={f} className="rounded-md bg-s1/80 px-2 py-0.5 text-[11.5px] font-medium shadow-[inset_0_0_0_1px_rgb(17_20_24/.08)]" style={{ color: P_INK[page.priority] }}>
                {f}
              </span>
            ))}
          </div>
        )}
        <div className="mt-auto pb-[max(2rem,env(safe-area-inset-bottom))]">
          {err && <div className="mb-2 text-[12px] text-high-ink">{err}</div>}
          {!declining ? (
            <div className="grid grid-cols-[1fr_1.6fr] gap-3">
              <button disabled={!!busy} onClick={() => setDeclining(true)} className="h-14 rounded-2xl border border-line bg-s1 text-[15px] font-semibold text-ink shadow-sm active:translate-y-px">
                Decline
              </button>
              <button
                disabled={!!busy}
                onClick={() => respond("accept")}
                className="h-14 rounded-2xl text-[15px] font-semibold text-white shadow-[0_10px_24px_-10px_rgb(43_154_102/.8)] active:translate-y-px"
                style={{ background: "linear-gradient(180deg,#34b27a,#2b9a66)" }}
              >
                {busy === "accept" ? "Accepting…" : "Accept"}
              </button>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="text-center text-[12px] text-dim">Why? The next clinician is paged at once.</div>
              {["In a procedure", "With another patient", "Not my service"].map((r) => (
                <button key={r} disabled={!!busy} onClick={() => respond("decline", r.toLowerCase())} className="h-11 w-full rounded-xl border border-line bg-s1 text-[14px] font-medium text-ink">
                  {r}
                </button>
              ))}
              <button onClick={() => setDeclining(false)} className="h-9 w-full text-[13px] text-dim">
                Back
              </button>
            </div>
          )}
          <div className="mt-3 text-center text-[11px] text-dim">
            Escalates to {page.backups[0] ? clinicians[page.backups[0]]?.name ?? "the next clinician" : "the next clinician"} if unanswered
          </div>
        </div>
      </div>
    </motion.div>
  );
}

/* ------------------------------------------------------------------ */
/* An accepted case: the brief                                         */
/* ------------------------------------------------------------------ */
function CaseCard({ page, now, call, readOnly }: { page: Page; now: number; call: Call; readOnly?: boolean }) {
  const [busy, setBusy] = useState(false);
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="overflow-hidden rounded-2xl border border-line bg-s1 shadow-[var(--shadow-card)]">
      <div className="px-4 pb-3 pt-4" style={{ background: `linear-gradient(180deg, rgb(${P_RGB[page.priority]} / .08), transparent)` }}>
        <div className="flex items-center gap-2">
          <PriorityChip p={page.priority} />
          <span className="text-[11.5px] text-dim">{PRIORITY_LABEL[page.priority]} · {CATEGORY_LABEL[page.category]}</span>
          <span className="ml-auto text-[11px] text-dim">{page.status === "accepted" ? `accepted ${since(page.accepted_at, now)}` : PAGE_STATUS_LABEL[page.status]}</span>
        </div>
        <div className="mt-2 text-[22px] font-semibold tracking-tight text-ink">Room {page.room}</div>
        <div className="text-[12.5px] text-muted">
          {page.zone_name}
          {page.patient?.name ? ` · ${page.patient.name}` : ""}
        </div>
      </div>
      <div className="border-t border-line-soft px-4 py-3">
        <div className="mb-2 flex items-center gap-2">
          <span className="font-mono text-[10.5px] uppercase tracking-[0.12em] text-dim">Brief</span>
          <span className="ml-auto text-[10.5px] text-dim">
            {page.sbar_refining ? <span className="shimmer rounded px-1 text-accent-ink">model writing</span> : page.sbar ? (page.sbar.source === "llm" ? "written by the model" : "from the record") : ""}
          </span>
        </div>
        {page.sbar ? (
          <dl className="space-y-2.5">
            {(["situation", "background", "assessment", "request"] as const).map((k) => (
              <div key={k} className="grid grid-cols-[20px_1fr] gap-2">
                <dt className="grid h-5 w-5 place-items-center rounded-md bg-ink font-mono text-[10px] font-semibold text-white">{k[0].toUpperCase()}</dt>
                <dd className={cx("text-[13px] leading-relaxed", k === "request" ? "font-medium text-ink" : "text-ink/90")}>{page.sbar![k]}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <div className="shimmer h-24 rounded-lg" />
        )}
      </div>
      {!readOnly && (
        <div className="flex gap-2 border-t border-line-soft p-3">
          {page.bed && (
            <Link href={`/patient/?id=${encodeURIComponent(page.bed)}`} className="flex h-10 flex-1 items-center justify-center rounded-xl border border-line text-[13px] font-medium text-ink">
              Open patient
            </Link>
          )}
          <Button
            variant="primary"
            className="h-10 flex-1 rounded-xl text-[13px]"
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              await call(`/aegis/paging/pages/${page.id}/resolve`, {}).catch(() => {});
              setBusy(false);
            }}
          >
            {busy ? "Resolving…" : "Mark resolved"}
          </Button>
        </div>
      )}
    </motion.div>
  );
}

/* ------------------------------------------------------------------ */
/* Sound and vibration when a page arrives                             */
/* ------------------------------------------------------------------ */
function useAlertTone(pageId: string | null, priority: number) {
  const last = useRef<string | null>(null);
  useEffect(() => {
    if (!pageId || last.current === pageId) return;
    last.current = pageId;
    try {
      navigator.vibrate?.(priority <= 2 ? [180, 90, 180, 90, 320] : [160]);
    } catch {}
    try {
      const Ctx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
      const ctx = new Ctx();
      const beeps = priority <= 2 ? [0, 0.22, 0.44] : [0];
      for (const t of beeps) {
        const o = ctx.createOscillator();
        const g = ctx.createGain();
        o.frequency.value = priority === 1 ? 880 : 740;
        o.type = "sine";
        g.gain.setValueAtTime(0.0001, ctx.currentTime + t);
        g.gain.exponentialRampToValueAtTime(0.18, ctx.currentTime + t + 0.02);
        g.gain.exponentialRampToValueAtTime(0.0001, ctx.currentTime + t + 0.18);
        o.connect(g).connect(ctx.destination);
        o.start(ctx.currentTime + t);
        o.stop(ctx.currentTime + t + 0.2);
      }
      setTimeout(() => ctx.close().catch(() => {}), 1500);
    } catch {}
  }, [pageId, priority]);
}

/* ------------------------------------------------------------------ */
/* No id: choose whose pager                                           */
/* ------------------------------------------------------------------ */
function Picker() {
  const { state } = usePaging();
  const staff = Object.values(state.clinicians).filter((c) => c.pageable).sort((a, b) => a.specialty_label.localeCompare(b.specialty_label));
  return (
    <main className="mx-auto max-w-[980px] px-5 py-12">
      <Link href="/" className="inline-flex items-center gap-2.5">
        <Mark />
        <span className="text-[15px] font-semibold tracking-tight">AegisOS</span>
      </Link>
      <h1 className="serif mt-10 text-[40px] text-ink">Whose pager?</h1>
      <p className="mt-2 text-[14px] text-muted">Every clinician on the roster carries one. Open theirs and pages meant for them ring here.</p>
      <div className="mt-8 grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {staff.map((c) => (
          <Link key={c.id} href={`/clinician/?id=${c.id}`} className="plate flex items-center gap-3 p-3.5 transition hover:-translate-y-px hover:border-line-bright">
            <Avatar name={c.name} status={c.status} size={36} present={c.present} />
            <span className="min-w-0">
              <span className="block truncate text-[13.5px] font-medium text-ink">{c.name}</span>
              <span className="block truncate text-[12px] text-dim">{c.specialty_label} · {STATUS_LABEL[c.status]}</span>
            </span>
          </Link>
        ))}
      </div>
    </main>
  );
}
