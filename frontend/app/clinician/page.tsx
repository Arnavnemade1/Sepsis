"use client";

import { AnimatePresence, motion } from "motion/react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";

import { Avatar, Countdown, P_RGB, PriorityChip, SourceTag, StatusText, TimerBar } from "@/components/paging/bits";
import { AppShell, Logo } from "@/components/shell";
import { Button, cx, Panel, Tag } from "@/components/ui";
import { api, readDeviceSession, revalidate, writeDeviceSession, type Credential } from "@/lib/api";
import {
  CATEGORY_LABEL,
  clock,
  PAGE_STATUS_LABEL,
  PRIORITY_LABEL,
  shortName,
  usePaging,
  useServerNow,
  type Clinician,
  type Page,
  type StaffStatus,
} from "@/lib/paging";

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

type Call = <T extends { status: string; errno?: string; detail?: string }>(path: string, body: Record<string, unknown>) => Promise<T>;

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

  if (!id) return <Directory />;
  return (
    <div className="min-h-[100dvh] bg-bg">
      {/* on a phone the page IS the pager; on anything wider it is framed as a device */}
      <header className="hidden h-11 items-center gap-3 border-b border-line px-4 sm:flex">
        <Logo />
        <span className="h-4 w-px bg-line-bright" aria-hidden />
        <span className="text-[12.5px] text-muted">Pager</span>
        {me && <span className="text-[12.5px] text-ink">{me.name}</span>}
        <nav className="ml-auto flex items-center gap-1 text-[12.5px]">
          <Link href="/clinician/" className="rounded-md px-2.5 py-1 text-dim transition-colors hover:bg-s2 hover:text-ink">
            All pagers
          </Link>
          <Link href="/operator/" className="rounded-md px-2.5 py-1 text-dim transition-colors hover:bg-s2 hover:text-ink">
            Operator console
          </Link>
        </nav>
      </header>
      <div className="sm:flex sm:flex-col sm:items-center sm:py-8">
        <div className="relative h-[100dvh] w-full overflow-hidden bg-bg sm:h-[760px] sm:w-[400px] sm:rounded-xl sm:border sm:border-line-bright sm:bg-s1 sm:shadow-[0_32px_80px_-32px_rgb(0_0_0/.6)]">
          {!state.ready || !me ? (
            <div className="grid h-full place-items-center px-8 text-center text-[12.5px] text-dim">
              {err ? `Signing in: ${err}` : status === "live" ? "Loading…" : "Connecting to the hospital…"}
            </div>
          ) : (
            <Screen me={me} pages={state.pages} clinicians={state.clinicians} now={now} call={call} notifications={state.notifications} live={status === "live"} />
          )}
        </div>
        {me && (
          <p className="mt-4 hidden max-w-[400px] text-center text-[11.5px] leading-relaxed text-dim sm:block">
            While this pager is open, pages for {shortName(me.name)} ring here and the demo&apos;s simulated responders stand down for them.
          </p>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The device screen                                                   */
/* ------------------------------------------------------------------ */
function Screen({ me, pages, clinicians, now, call, notifications, live }: {
  me: Clinician;
  pages: Record<string, Page>;
  clinicians: Record<string, Clinician>;
  now: number;
  call: Call;
  notifications: { to: string | null; page: string; kind: string; text: string; ts: number; priority: number; room: string | null }[];
  live: boolean;
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
      <div className="flex items-center gap-3 border-b border-line px-4 pb-3 pt-[max(0.875rem,env(safe-area-inset-top))]">
        <Avatar name={me.name} size={34} />
        <div className="min-w-0 flex-1">
          <div className="truncate text-[14px] font-semibold text-ink">{me.name}</div>
          <div className="truncate text-[11.5px] text-dim">
            {me.specialty_label} · {me.on_call ? "on call" : me.title}
          </div>
        </div>
        <span className="flex items-center gap-1.5 text-[11px] text-dim" title={live ? "Connected" : "Reconnecting"}>
          <span className={cx("h-1.5 w-1.5 rounded-full", live ? "bg-low" : "blink bg-med")} />
          {live ? "Connected" : "Reconnecting"}
        </span>
      </div>
      <div className="px-4 py-3">
        <div className="grid grid-cols-3 gap-0.5 rounded-md border border-line bg-bg p-0.5">
          {(["available", "in_procedure", "off_shift"] as StaffStatus[]).map((s) => {
            const active = me.status === s || (s === "available" && (me.status === "paging" || me.status === "on_case"));
            return (
              <button
                key={s}
                onClick={() => setStatus(s)}
                className={cx("h-8 rounded-[4px] text-[12px] font-medium transition-colors", active ? "bg-s3 text-ink" : "text-dim hover:text-ink")}
              >
                {s === "available" ? (me.status === "on_case" ? "On a case" : "Available") : s === "in_procedure" ? "In procedure" : "Off shift"}
              </button>
            );
          })}
        </div>
      </div>

      <div className="min-h-0 flex-1 space-y-3 overflow-y-auto px-4 pb-6">
        {detail && !open && accepted.length > 0 && <CaseCard page={detail} call={call} />}
        {open && pages[open] && (
          <div>
            <button onClick={() => setOpen(null)} className="mb-2 text-[12px] text-accent-ink">
              ← Back
            </button>
            <CaseCard page={pages[open]} call={call} readOnly={pages[open].assignee !== me.id || pages[open].status !== "accepted"} />
          </div>
        )}
        {!open && fyi.length > 0 && (
          <div className="overflow-hidden rounded-md border border-line">
            {fyi.map((n) => (
              <div key={n.ts + n.page} className="relative border-b border-line-soft px-3 py-2 last:border-b-0">
                <span className="absolute inset-y-0 left-0 w-[2px] bg-accent" />
                <div className="text-[11px] font-medium text-accent-ink">For your information · Room {n.room}</div>
                <div className="text-[12.5px] text-ink/90">{n.text}</div>
              </div>
            ))}
          </div>
        )}
        {!open && accepted.length === 0 && !ringing && (
          <div className="rounded-md border border-dashed border-line-bright px-6 py-10 text-center">
            <div className="text-[13px] font-medium text-ink">No active pages</div>
            <div className="mt-1 text-[12px] text-dim">Pages for you will ring here.</div>
          </div>
        )}
        {!open && history.length > 0 && (
          <div>
            <div className="pb-1.5 pt-1 text-[11.5px] font-medium text-dim">Recent</div>
            <div className="overflow-hidden rounded-md border border-line">
              {history.map((p) => {
                const mine = p.attempts.filter((a) => a.clinician === me.id).at(-1);
                return (
                  <button key={p.id} onClick={() => setOpen(p.id)} className="flex w-full items-center gap-2.5 border-b border-line-soft px-3 py-2.5 text-left transition-colors last:border-b-0 hover:bg-s2">
                    <PriorityChip p={p.priority} size="sm" />
                    <span className="min-w-0 flex-1">
                      <span className="block truncate text-[12.5px] text-ink">
                        <span className="font-mono">{p.room}</span> · {p.text}
                      </span>
                      <span className="block text-[11px] text-dim">
                        {mine?.outcome ? (mine.outcome === "timeout" ? "Not answered" : `You ${mine.outcome}`) : PAGE_STATUS_LABEL[p.status]} · {clock(p.created_at, false)}
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
  const prev = page.attempts.length > 1 ? page.attempts[page.attempts.length - 2] : null;
  const remaining = att.deadline - now;
  return (
    <motion.div
      className="absolute inset-0 z-20 flex flex-col bg-s1"
      initial={{ y: "100%" }}
      animate={{ y: 0 }}
      exit={{ y: "100%", transition: { duration: 0.22 } }}
      transition={{ duration: 0.28, ease: [0.2, 0.8, 0.2, 1] }}
    >
      <span className="h-[3px] w-full shrink-0" style={{ background: `rgb(${P_RGB[page.priority]})` }} />
      <div className="flex items-center gap-2 px-5 pb-2 pt-[max(1rem,env(safe-area-inset-top))]">
        <span className="blink h-2 w-2 rounded-full" style={{ background: `rgb(${P_RGB[page.priority]})` }} />
        <span className="text-[12px] font-semibold text-ink">Incoming page</span>
        <span className="font-mono text-[11px] text-dim">{page.id}</span>
        <span className="ml-auto text-right">
          <Countdown remaining={remaining} className="text-[18px]" />
        </span>
      </div>
      <TimerBar remaining={remaining} total={att.deadline - att.sent_at} priority={page.priority} className="mx-5" />
      <div className="flex min-h-0 flex-1 flex-col overflow-y-auto px-5 pt-5">
        <div className="flex items-center gap-2">
          <PriorityChip p={page.priority} size="lg" label />
          <span className="text-[12.5px] text-dim">{CATEGORY_LABEL[page.category]}</span>
        </div>
        <div className="mt-4 font-mono text-[44px] font-semibold leading-none tracking-[-0.02em] text-ink">{page.room ?? "—"}</div>
        <div className="mt-1.5 text-[13.5px] text-muted">
          {page.zone_name}
          {page.patient?.name ? ` · ${page.patient.name}` : ""}
          {page.patient?.age ? `, ${page.patient.age}${page.patient.sex ?? ""}` : ""}
        </div>
        <p className="mt-5 text-[16px] leading-relaxed text-ink">{page.text}</p>
        <div className="mt-3 flex flex-wrap items-center gap-2 text-[12px] text-dim">
          <SourceTag source={page.source} synthetic={page.synthetic} />
          {prev && (
            <span>
              · {prev.name} {prev.outcome === "timeout" ? "did not answer" : prev.outcome}
            </span>
          )}
        </div>
        {page.triage.red_flags.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-1.5">
            {page.triage.red_flags.slice(0, 3).map((f) => (
              <Tag key={f} tone="high">
                {f}
              </Tag>
            ))}
          </div>
        )}
        <div className="mt-auto pb-[max(1.5rem,env(safe-area-inset-bottom))] pt-6">
          {err && <div className="mb-2 text-[12px] text-high-ink">{err}</div>}
          {!declining ? (
            <div className="grid grid-cols-[1fr_1.6fr] gap-2.5">
              <Button size="lg" variant="ghost" disabled={!!busy} onClick={() => setDeclining(true)} className="h-12">
                Decline
              </Button>
              <Button size="lg" variant="success" disabled={!!busy} onClick={() => respond("accept")} className="h-12 text-[14px]">
                {busy === "accept" ? "Accepting…" : "Accept"}
              </Button>
            </div>
          ) : (
            <div className="space-y-1.5">
              <div className="pb-1 text-center text-[12px] text-dim">Reason. The next clinician is paged at once.</div>
              {["In a procedure", "With another patient", "Not my service"].map((r) => (
                <Button key={r} size="lg" variant="ghost" disabled={!!busy} onClick={() => respond("decline", r.toLowerCase())} className="h-11 w-full">
                  {r}
                </Button>
              ))}
              <button onClick={() => setDeclining(false)} className="h-9 w-full text-[12.5px] text-dim">
                Back
              </button>
            </div>
          )}
          <div className="mt-3 text-center text-[11.5px] text-dim">
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
function CaseCard({ page, call, readOnly }: { page: Page; call: Call; readOnly?: boolean }) {
  const [busy, setBusy] = useState(false);
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="relative overflow-hidden rounded-md border border-line bg-s1">
      <span className="absolute inset-y-0 left-0 w-[2px]" style={{ background: `rgb(${P_RGB[page.priority]})` }} />
      <div className="px-3.5 pb-3 pt-3">
        <div className="flex items-center gap-2">
          <PriorityChip p={page.priority} size="sm" />
          <span className="text-[11.5px] text-dim">
            {PRIORITY_LABEL[page.priority]} · {CATEGORY_LABEL[page.category]}
          </span>
          <span className="ml-auto text-[11px] text-dim">
            {page.status === "accepted" && page.accepted_at ? `Accepted ${clock(page.accepted_at, false)}` : PAGE_STATUS_LABEL[page.status]}
          </span>
        </div>
        <div className="mt-2 font-mono text-[22px] font-semibold tracking-[-0.01em] text-ink">Room {page.room}</div>
        <div className="text-[12px] text-muted">
          {page.zone_name}
          {page.patient?.name ? ` · ${page.patient.name}` : ""}
        </div>
      </div>
      <div className="border-t border-line px-3.5 py-3">
        <div className="mb-2 flex items-center gap-2">
          <span className="text-[11.5px] font-semibold text-muted">Brief</span>
          <span className="ml-auto text-[10.5px] text-dim">
            {page.sbar_refining ? <span className="working rounded-[3px] px-1 text-accent-ink">model writing</span> : page.sbar ? (page.sbar.source === "llm" ? "Written by the model" : "From the record") : ""}
          </span>
        </div>
        {page.sbar ? (
          <dl className="space-y-2">
            {(["situation", "background", "assessment", "request"] as const).map((k) => (
              <div key={k} className="grid grid-cols-[20px_minmax(0,1fr)] gap-2.5">
                <dt className="grid h-5 w-5 place-items-center rounded-[3px] bg-s3 font-mono text-[10px] font-semibold text-muted">{k[0].toUpperCase()}</dt>
                <dd className={cx("text-[12.5px] leading-relaxed", k === "request" ? "font-medium text-ink" : "text-ink/85")}>{page.sbar![k]}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <div className="working h-20 rounded-md" />
        )}
      </div>
      {!readOnly && (
        <div className="flex gap-2 border-t border-line p-3">
          {page.bed && (
            <Link href={`/patient/?id=${encodeURIComponent(page.bed)}`} className="flex h-9 flex-1 items-center justify-center rounded-md border border-line-bright bg-s2 text-[12.5px] font-medium text-ink transition-colors hover:bg-s3">
              Open patient
            </Link>
          )}
          <Button
            variant="primary"
            className="h-9 flex-1"
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
/* No id: the pager directory                                          */
/* ------------------------------------------------------------------ */
function Directory() {
  const { state } = usePaging();
  const staff = Object.values(state.clinicians)
    .filter((c) => c.pageable)
    .sort((a, b) => a.specialty_label.localeCompare(b.specialty_label) || a.name.localeCompare(b.name));
  return (
    <AppShell>
      <main className="mx-auto w-full max-w-[1100px] flex-1 px-3 pb-12 pt-4 sm:px-4">
        <div className="mb-4 border-b border-line pb-4">
          <h1 className="text-[18px] font-semibold tracking-[-0.01em] text-ink">Pagers</h1>
          <p className="mt-1 max-w-2xl text-[12.5px] leading-relaxed text-muted">
            Every clinician on the roster carries a pager. Open one on this device and pages routed to that clinician ring here; while it is open, the
            simulated responder stands down for them.
          </p>
        </div>
        <Panel title="Roster" meta={`${staff.length} pageable clinicians`} flush>
          <div className="overflow-x-auto">
            <table className="w-full min-w-[640px] text-[12.5px]">
              <thead>
                <tr className="border-b border-line text-left text-[11px] text-dim">
                  <th className="px-3.5 py-2 font-medium">Clinician</th>
                  <th className="px-3 py-2 font-medium">Service</th>
                  <th className="px-3 py-2 font-medium">Status</th>
                  <th className="px-3.5 py-2" />
                </tr>
              </thead>
              <tbody>
                {staff.length === 0 &&
                  [0, 1, 2, 3].map((i) => (
                    <tr key={i} className="border-b border-line-soft">
                      <td colSpan={4} className="px-3.5 py-3">
                        <div className="h-4 w-1/2 animate-pulse rounded bg-s2" />
                      </td>
                    </tr>
                  ))}
                {staff.map((c) => (
                  <tr key={c.id} className="border-b border-line-soft transition-colors last:border-b-0 hover:bg-s2">
                    <td className="px-3.5 py-2">
                      <div className="flex items-center gap-2.5">
                        <Avatar name={c.name} size={24} present={c.present} />
                        <div>
                          <div className="text-ink">{c.name}</div>
                          <div className="text-[11px] text-dim">{c.title}</div>
                        </div>
                      </div>
                    </td>
                    <td className="px-3 py-2 text-muted">{c.specialty_label}</td>
                    <td className="px-3 py-2">
                      <StatusText status={c.status} />
                    </td>
                    <td className="px-3.5 py-2 text-right">
                      <Link href={`/clinician/?id=${c.id}`} className="inline-flex h-7 items-center rounded-md border border-line-bright bg-s2 px-2.5 text-[12px] font-medium text-ink transition-colors hover:bg-s3">
                        Open pager
                      </Link>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </Panel>
      </main>
    </AppShell>
  );
}
