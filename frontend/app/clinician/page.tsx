"use client";

import { AnimatePresence, motion } from "motion/react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import {
  Suspense,
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import {
  Avatar,
  Countdown,
  P_RGB,
  PriorityChip,
  SourceTag,
  StatusText,
  TimerBar,
} from "@/components/paging/bits";
import { Bell, CaretDown, CaretLeft, CaretRight, Check, Checks, Clock, Close, Inbox, Offline, Search, Signal, Spinner } from "@/components/icons";
import { Logo, Page as ShellPage } from "@/components/shell";
import { cx, Tag } from "@/components/ui";
import {
  api,
  readDeviceSession,
  revalidate,
  writeDeviceSession,
  type Credential,
} from "@/lib/api";
import {
  CATEGORY_LABEL,
  clock,
  PAGE_STATUS_LABEL,
  usePaging,
  useServerNow,
  type Clinician,
  type Page,
  type StaffStatus,
} from "@/lib/paging";

import "./pager.css";

export default function ClinicianPage() {
  return (
    <Suspense>
      <Device />
    </Suspense>
  );
}

type Login = {
  status: string;
  token: string;
  credential: Credential;
  detail?: string;
};

/** The device's own session for clinician ``id``. ``renew`` signs in again when the
 *  kernel has forgotten the token (a backend restart), and returns the token to use. */
function useDeviceToken(id: string | null) {
  const [token, setToken] = useState<string | null | undefined>(undefined);
  const [err, setErr] = useState<string | null>(null);
  const tokenRef = useRef<string | null>(null);
  const login = useCallback(async (who: string) => {
    const r = await api<Login>("/aegis/sys/login", {
      body: { username: who },
      token: null,
      timeoutMs: 60_000,
    });
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
        const good =
          v.state === "ok" ? cached : v.state === "renewed" ? v.session : null;
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

type Call = <T extends { status: string; errno?: string; detail?: string }>(
  path: string,
  body: Record<string, unknown>,
) => Promise<T>;

function Device() {
  const params = useSearchParams();
  const id = params.get("id");
  const { token, err, renew } = useDeviceToken(id);
  const { state, status, epoch } = usePaging({
    clinician: token ? id : null,
    token: token ?? null,
  });
  // a reconnect may follow a backend restart that forgot this device's session
  useEffect(() => {
    if (epoch > 1) renew();
  }, [epoch, renew]);
  // every write from the device goes AS its clinician; a forgotten session is renewed once
  const call = useCallback(
    async <T extends { status: string; errno?: string; detail?: string }>(
      path: string,
      body: Record<string, unknown>,
    ): Promise<T> => {
      const r = await api<T>(path, { body, token: token ?? null });
      if (r.status === "ok" || !["EPERM", "EAUTH"].includes(r.errno ?? ""))
        return r;
      const fresh = await renew();
      return fresh && fresh !== token
        ? api<T>(path, { body, token: fresh })
        : r;
    },
    [token, renew],
  );
  const now = useServerNow(state.offsetMs);
  const me = id ? state.clinicians[id] : undefined;

  if (!id) return <Directory />;
  return (
    <div className="pager-stage">
      <header className="pager-desktop-header">
        <Logo />
        <nav aria-label="Pager navigation">
          <Link href="/clinician/">All pagers</Link>
          <Link href="/operator/">
            Operator console <CaretRight size={14} />
          </Link>
        </nav>
      </header>
      <div className="pager-stage-main">
        <div className="pager-device">
          <span className="pager-side-key pager-side-key-left" aria-hidden />
          <span className="pager-side-key pager-side-key-right" aria-hidden />
          <div className="pager-display">
            <div className="pager-system-bar">
              <span className="num">{clock(now, false)}</span>
              <span className="pager-camera" aria-hidden />
              <span
                title={
                  status === "live"
                    ? "Connected to hospital"
                    : "Reconnecting to hospital"
                }
              >
                {status === "live" ? (
                  <Signal size={17} aria-hidden />
                ) : (
                  <Offline size={17} aria-hidden />
                )}
                <span className="sr-only">
                  {status === "live" ? "Connected" : "Reconnecting"}
                </span>
              </span>
            </div>
            {!state.ready || !me ? (
              <div className="pager-loading" role="status">
                <div className="pager-app-icon">
                  <Bell size={26} />
                </div>
                <h1>Opening the pager</h1>
                <p>
                  {err
                    ? `Signing in: ${err}`
                    : status === "live"
                      ? "Loading your pages"
                      : "Connecting to the hospital"}
                </p>
                <Spinner size={18} className="animate-spin" aria-hidden />
                <Link href="/clinician/">Back to all pagers</Link>
              </div>
            ) : (
              <Screen
                me={me}
                pages={state.pages}
                clinicians={state.clinicians}
                now={now}
                call={call}
                notifications={state.notifications}
                live={status === "live"}
              />
            )}
            <div className="pager-home-area" aria-hidden>
              <span />
            </div>
          </div>
        </div>
        <p className="pager-device-caption">
          <span /> {me ? `${me.name}'s pager, live from the hospital` : "Connecting the pager"}
        </p>
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The device screen                                                   */
/* ------------------------------------------------------------------ */
function Screen({
  me,
  pages,
  clinicians,
  now,
  call,
  notifications,
  live,
}: {
  me: Clinician;
  pages: Record<string, Page>;
  clinicians: Record<string, Clinician>;
  now: number;
  call: Call;
  notifications: {
    to: string | null;
    page: string;
    kind: string;
    text: string;
    ts: number;
    priority: number;
    room: string | null;
  }[];
  live: boolean;
}) {
  const ringing = useMemo(
    () =>
      Object.values(pages)
        .filter(
          (p) =>
            p.status === "paging" && p.attempts.at(-1)?.clinician === me.id,
        )
        .sort((a, b) => a.priority - b.priority)[0] ?? null,
    [pages, me.id],
  );
  const accepted = useMemo(
    () =>
      Object.values(pages)
        .filter((p) => p.status === "accepted" && p.assignee === me.id)
        .sort((a, b) => a.priority - b.priority),
    [pages, me.id],
  );
  const history = useMemo(
    () =>
      Object.values(pages)
        .filter(
          (p) =>
            p.attempts.some((a) => a.clinician === me.id) &&
            !(p.status === "accepted" && p.assignee === me.id) &&
            p.id !== ringing?.id,
        )
        .sort((a, b) => b.created_at - a.created_at)
        .slice(0, 8),
    [pages, me.id, ringing],
  );
  const fyi = notifications
    .filter((n) => n.to === me.id && n.kind === "fyi")
    .slice(-3)
    .reverse();
  const [open, setOpen] = useState<string | null>(null);
  const [tab, setTab] = useState<"pages" | "recent">("pages");
  const [statusBusy, setStatusBusy] = useState(false);
  const [statusError, setStatusError] = useState<string | null>(null);
  useAlertTone(ringing?.id ?? null, ringing?.priority ?? 4);

  const setStatus = async (s: StaffStatus) => {
    setStatusBusy(true);
    setStatusError(null);
    try {
      const r = await call(`/aegis/paging/clinicians/${me.id}/status`, {
        status: s,
      });
      if (r.status !== "ok")
        setStatusError(r.detail ?? "Could not update availability. Try again.");
    } catch {
      setStatusError("Could not update availability. Try again.");
    } finally {
      setStatusBusy(false);
    }
  };
  const availability = ["paging", "on_case"].includes(me.status)
    ? "available"
    : me.status;
  const selected = open ? pages[open] : null;

  return (
    <div className="pager-screen">
      <div className="pager-screen-content" inert={!!ringing}>
        <header className="pager-profile">
          <Link
            href="/clinician/"
            className="pager-back"
            aria-label="All pagers"
          >
            <CaretLeft size={21} />
          </Link>
          <div className="pager-profile-name">
            <strong>{me.name}</strong>
            <span>
              {me.specialty_label}
              {me.on_call ? " · On call" : ""}
            </span>
          </div>
          <Avatar name={me.name} size={38} />
        </header>
        <div className="pager-heading">
          <h1>
            {selected
              ? "Page details"
              : tab === "recent"
                ? "Recent pages"
                : "Your pages"}
          </h1>
          <div className="pager-availability" data-status={availability}>
            <span className="pager-status-dot" />
            <div className="pager-availability-control">
              <select
                aria-label="Your availability"
                value={availability}
                disabled={statusBusy}
                onChange={(e) => setStatus(e.target.value as StaffStatus)}
              >
                <option value="available">
                  {me.status === "on_case"
                    ? "On a case"
                    : "Available for pages"}
                </option>
                <option value="in_procedure">In a procedure</option>
                <option value="off_shift">Off shift</option>
              </select>
              <CaretDown size={13} aria-hidden />
            </div>
            {statusBusy && (
              <Spinner
                size={13}
                className="animate-spin"
                aria-label="Updating availability"
              />
            )}
          </div>
          {statusError && (
            <p className="pager-error" role="alert">
              {statusError}
            </p>
          )}
        </div>

        <main className="pager-content">
          {selected ? (
            <>
              <button
                onClick={() => setOpen(null)}
                className="pager-text-button"
              >
                <CaretLeft size={16} /> Back to{" "}
                {tab === "recent" ? "recent pages" : "your pages"}
              </button>
              <CaseCard
                page={selected}
                call={call}
                readOnly={
                  selected.assignee !== me.id || selected.status !== "accepted"
                }
              />
            </>
          ) : tab === "pages" ? (
            <>
              {accepted.length > 0 && (
                <div className="pager-section-label">
                  In your care <span>{accepted.length}</span>
                </div>
              )}
              {accepted.map((p) => (
                <CaseCard key={p.id} page={p} call={call} />
              ))}
              {accepted.length === 0 && !ringing && (
                <div className="pager-empty">
                  <div className="pager-empty-icon">
                    <Checks size={28} />
                  </div>
                  <h2>No open pages</h2>
                  <p>A new page rings this device and appears here.</p>
                </div>
              )}
              {fyi.length > 0 && (
                <section
                  className="pager-notifications"
                  aria-label="For your information"
                >
                  <div className="pager-section-label">
                    For your information
                  </div>
                  <div className="pager-list">
                    {fyi.map((n) => (
                      <div key={n.ts + n.page} className="pager-notification">
                        <span>Room {n.room ?? "—"}</span>
                        <p>{n.text}</p>
                      </div>
                    ))}
                  </div>
                </section>
              )}
            </>
          ) : history.length > 0 ? (
            <>
              <div className="pager-section-label">
                Latest activity <span>{history.length}</span>
              </div>
              <div className="pager-list">
                {history.map((p) => {
                  const mine = p.attempts
                    .filter((a) => a.clinician === me.id)
                    .at(-1);
                  return (
                    <button
                      key={p.id}
                      onClick={() => setOpen(p.id)}
                      className="pager-history-row"
                    >
                      <span
                        className="pager-history-priority"
                        style={{
                          background: `rgb(${P_RGB[p.priority]} / .12)`,
                          color: `rgb(${P_RGB[p.priority]})`,
                        }}
                      >
                        P{p.priority}
                      </span>
                      <span className="pager-history-copy">
                        <strong>
                          Room {p.room ?? "—"}
                          <time>{clock(p.created_at, false)}</time>
                        </strong>
                        <span>{p.text}</span>
                        <small>
                          {mine?.outcome
                            ? mine.outcome === "timeout"
                              ? "Not answered"
                              : `You ${mine.outcome}`
                            : PAGE_STATUS_LABEL[p.status]}
                        </small>
                      </span>
                      <CaretRight size={15} aria-hidden />
                    </button>
                  );
                })}
              </div>
            </>
          ) : (
            <div className="pager-empty">
              <div className="pager-empty-icon">
                <Clock size={28} />
              </div>
              <h2>No recent pages</h2>
              <p>Pages you accept or decline are kept here for the shift.</p>
            </div>
          )}
        </main>
        <div className="pager-connection" role="status">
          <span className={cx("pager-status-dot", !live && "reconnecting")} />
          {live ? "Connected to Aegis Memorial" : "Reconnecting to Aegis Memorial"}
        </div>
        <nav className="pager-tab-bar" aria-label="Pager views">
          <button
            aria-current={tab === "pages" ? "page" : undefined}
            onClick={() => {
              setTab("pages");
              setOpen(null);
            }}
          >
            <span>
              <Inbox size={21} />
              {accepted.length > 0 && <i>{accepted.length}</i>}
            </span>
            Pages
          </button>
          <button
            aria-current={tab === "recent" ? "page" : undefined}
            onClick={() => {
              setTab("recent");
              setOpen(null);
            }}
          >
            <Clock size={21} />
            Recent
          </button>
        </nav>
      </div>
      <AnimatePresence>
        {ringing && (
          <Incoming
            key={ringing.id}
            page={ringing}
            me={me}
            now={now}
            call={call}
            clinicians={clinicians}
          />
        )}
      </AnimatePresence>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* A page ringing                                                      */
/* ------------------------------------------------------------------ */
function Incoming({
  page,
  me,
  now,
  call,
  clinicians,
}: {
  page: Page;
  me: Clinician;
  now: number;
  call: Call;
  clinicians: Record<string, Clinician>;
}) {
  const dialogRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const previous = document.activeElement as HTMLElement | null;
    dialogRef.current?.focus();
    return () => {
      if (previous?.isConnected) previous.focus();
    };
  }, []);
  const att = page.attempts.at(-1)!;
  const [busy, setBusy] = useState<string | null>(null);
  const [declining, setDeclining] = useState(false);
  useEffect(() => {
    dialogRef.current?.focus();
  }, [declining]);
  const [err, setErr] = useState<string | null>(null);
  const respond = async (action: "accept" | "decline", reason?: string) => {
    setBusy(action);
    setErr(null);
    try {
      const r = await call<{ status: string; errno?: string; detail?: string }>(
        `/aegis/paging/pages/${page.id}/respond`,
        { action, reason, clinician: me.id },
      );
      if (r.status !== "ok") setErr(r.detail ?? r.status);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };
  const prev =
    page.attempts.length > 1 ? page.attempts[page.attempts.length - 2] : null;
  const remaining = att.deadline - now;
  return (
    <motion.div
      ref={dialogRef}
      tabIndex={-1}
      onKeyDown={(event) => {
        if (event.key !== "Tab") return;
        const controls = Array.from(
          event.currentTarget.querySelectorAll<HTMLButtonElement>(
            "button:not(:disabled)",
          ),
        );
        const first = controls[0];
        const last = controls.at(-1);
        if (
          event.shiftKey &&
          (document.activeElement === first ||
            document.activeElement === event.currentTarget)
        ) {
          event.preventDefault();
          last?.focus();
        } else if (
          !event.shiftKey &&
          (document.activeElement === last ||
            document.activeElement === event.currentTarget)
        ) {
          event.preventDefault();
          first?.focus();
        }
      }}
      className="pager-incoming"
      initial={{ y: "100%" }}
      animate={{ y: 0 }}
      exit={{ y: "100%", transition: { duration: 0.2 } }}
      transition={{ duration: 0.28, ease: [0.2, 0.8, 0.2, 1] }}
      role="dialog"
      aria-modal="true"
      aria-labelledby="incoming-title"
    >
      <div className="pager-incoming-header">
        <Bell size={17} />
        <span id="incoming-title">Incoming page</span>
        <PriorityChip p={page.priority} size="sm" label />
      </div>
      <div className="pager-incoming-body">
        <div
          className="pager-call-symbol"
          style={{
            color: `rgb(${P_RGB[page.priority]})`,
            borderColor: `rgb(${P_RGB[page.priority]} / .3)`,
          }}
        >
          <Bell size={30} />
        </div>
        <p className="pager-incoming-category">
          {CATEGORY_LABEL[page.category]}
        </p>
        <h2>
          <span>Room</span>
          {page.room ?? "—"}
        </h2>
        <p className="pager-incoming-location">
          {page.zone_name}
          {page.patient?.name ? ` · ${page.patient.name}` : ""}
          {page.patient?.age
            ? `, ${page.patient.age}${page.patient.sex ?? ""}`
            : ""}
        </p>
        <p className="pager-incoming-message">{page.text}</p>
        <div className="pager-incoming-source">
          <SourceTag source={page.source} synthetic={page.synthetic} />
          {prev && (
            <span>
              {prev.name}{" "}
              {prev.outcome === "timeout" ? "did not answer" : prev.outcome}
            </span>
          )}
        </div>
        {page.triage.red_flags.length > 0 && (
          <div className="pager-red-flags">
            {page.triage.red_flags.slice(0, 3).map((f) => (
              <Tag key={f} tone="high">
                {f}
              </Tag>
            ))}
          </div>
        )}
      </div>
      <div className="pager-incoming-actions">
        <div className="pager-deadline">
          <span>Time to respond</span>
          <Countdown remaining={remaining} />
        </div>
        <TimerBar
          remaining={remaining}
          total={att.deadline - att.sent_at}
          priority={page.priority}
        />
        {err && (
          <p className="pager-error" role="alert">
            {err}
          </p>
        )}
        {!declining ? (
          <div className="pager-response-buttons">
            <button
              className="pager-action pager-action-secondary"
              disabled={!!busy}
              onClick={() => setDeclining(true)}
            >
              <Close size={18} />
              Decline
            </button>
            <button
              className="pager-action pager-action-accept"
              disabled={!!busy}
              onClick={() => respond("accept")}
            >
              <Check size={19} />
              {busy === "accept" ? "Accepting…" : "Accept page"}
            </button>
          </div>
        ) : (
          <div className="pager-decline-reasons">
            <p>Choose a reason to page the next clinician.</p>
            {["In a procedure", "With another patient", "Not my service"].map(
              (r) => (
                <button
                  className="pager-action pager-action-secondary"
                  key={r}
                  disabled={!!busy}
                  onClick={() => respond("decline", r.toLowerCase())}
                >
                  {busy === "decline" ? "Declining…" : r}
                </button>
              ),
            )}
            <button
              disabled={!!busy}
              onClick={() => setDeclining(false)}
              className="pager-text-button"
            >
              Go back
            </button>
          </div>
        )}
        <p className="pager-escalation">
          If unanswered, routes to{" "}
          {page.backups[0]
            ? (clinicians[page.backups[0]]?.name ?? "the next clinician")
            : "the next clinician"}
          .
        </p>
      </div>
    </motion.div>
  );
}

/* An accepted case: the brief */
function CaseCard({
  page,
  call,
  readOnly,
}: {
  page: Page;
  call: Call;
  readOnly?: boolean;
}) {
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  return (
    <motion.article
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      className="pager-case"
    >
      <div className="pager-case-header">
        <div className="pager-case-meta">
          <PriorityChip p={page.priority} size="sm" label />
          <span>
            {page.status === "accepted" && page.accepted_at
              ? `Accepted ${clock(page.accepted_at, false)}`
              : PAGE_STATUS_LABEL[page.status]}
          </span>
        </div>
        <h2>Room {page.room ?? "—"}</h2>
        <p>
          {page.zone_name}
          {page.patient?.name ? ` · ${page.patient.name}` : ""}
        </p>
        <div className="pager-case-source">
          <span>{CATEGORY_LABEL[page.category]}</span>
          <SourceTag source={page.source} synthetic={page.synthetic} />
        </div>
      </div>
      <div className="pager-brief">
        <div className="pager-section-label">
          Clinical brief{" "}
          <span>
            {page.sbar_refining
              ? "Updating…"
              : page.sbar?.source === "llm"
                ? "Model generated"
                : page.sbar
                  ? "From the record"
                  : ""}
          </span>
        </div>
        {page.sbar ? (
          <dl>
            {(
              ["situation", "background", "assessment", "request"] as const
            ).map((k) => (
              <div key={k}>
                <dt>{k}</dt>
                <dd>{page.sbar![k]}</dd>
              </div>
            ))}
          </dl>
        ) : (
          <p className="pager-brief-pending">
            {page.text}
            <span>Preparing the clinical brief…</span>
          </p>
        )}
      </div>
      {!readOnly && (
        <div className="pager-case-actions">
          {err && (
            <p className="pager-error" role="alert">
              {err}
            </p>
          )}
          {page.bed && (
            <Link
              href={`/patient/?id=${encodeURIComponent(page.bed)}`}
              className="pager-action pager-action-secondary"
            >
              Open patient <CaretRight size={16} />
            </Link>
          )}
          <button
            className="pager-action pager-action-primary"
            disabled={busy}
            onClick={async () => {
              setBusy(true);
              setErr(null);
              try {
                const r = await call(
                  `/aegis/paging/pages/${page.id}/resolve`,
                  {},
                );
                if (r.status !== "ok")
                  setErr(r.detail ?? "Could not resolve this page. Try again.");
              } catch {
                setErr("Could not resolve this page. Try again.");
              } finally {
                setBusy(false);
              }
            }}
          >
            <Check size={17} />
            {busy ? "Resolving…" : "Mark resolved"}
          </button>
        </div>
      )}
    </motion.article>
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
      const Ctx =
        window.AudioContext ||
        (window as unknown as { webkitAudioContext: typeof AudioContext })
          .webkitAudioContext;
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
  const [query, setQuery] = useState("");
  const staff = Object.values(state.clinicians)
    .filter((c) => c.pageable)
    .sort(
      (a, b) =>
        a.specialty_label.localeCompare(b.specialty_label) ||
        a.name.localeCompare(b.name),
    );
  const filtered = staff.filter((c) =>
    `${c.name} ${c.specialty_label} ${c.title}`
      .toLowerCase()
      .includes(query.trim().toLowerCase()),
  );
  const groups = [...new Set(filtered.map((c) => c.specialty_label))];
  return (
    <ShellPage active="pagers">
      <main className="pager-directory">
        <header className="pager-directory-header">
          <div>
            <h1>Pagers</h1>
            <p>Every clinician&apos;s pager, as it appears on their phone.</p>
          </div>
          <span className="pager-directory-count">
            {staff.length} clinicians
          </span>
        </header>
        <div className="pager-directory-toolbar">
          <label className="pager-search">
            <Search size={17} aria-hidden />
            <input
              type="search"
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              placeholder="Find a clinician or service"
              aria-label="Find a clinician or service"
            />
          </label>
          <p>Choose a clinician to open their pager.</p>
        </div>
        {!state.ready ? (
          <div className="pager-directory-loading" role="status">
            <Spinner size={18} className="animate-spin" />
            Connecting to the care team…
          </div>
        ) : filtered.length === 0 ? (
          <div className="pager-directory-loading">
            {query
              ? `No clinicians found for “${query}”.`
              : "No pageable clinicians are on the roster."}
          </div>
        ) : (
          <div className="pager-directory-services">
            {groups.map((group) => (
              <section key={group} className="pager-directory-service">
                <h2>
                  {group}
                  <span>
                    {filtered.filter((c) => c.specialty_label === group).length}
                  </span>
                </h2>
                <div>
                  {filtered
                    .filter((c) => c.specialty_label === group)
                    .map((c) => (
                      <Link
                        key={c.id}
                        href={`/clinician/?id=${encodeURIComponent(c.id)}`}
                        className="pager-contact"
                        aria-label={`Open pager for ${c.name}`}
                      >
                        <Avatar name={c.name} size={42} present={c.present} />
                        <span className="pager-contact-name">
                          <strong>{c.name}</strong>
                          <span>
                            {c.title}
                            {c.on_call ? " · On call" : ""}
                          </span>
                        </span>
                        <span className="pager-contact-status">
                          <StatusText status={c.status} />
                        </span>
                        <span className="pager-contact-open">
                          Open pager <CaretRight size={16} />
                        </span>
                      </Link>
                    ))}
                </div>
              </section>
            ))}
          </div>
        )}
        <p className="pager-directory-note">
          While a pager is open, its pages ring on this device and simulated
          responses pause for that clinician.
        </p>
      </main>
    </ShellPage>
  );
}
