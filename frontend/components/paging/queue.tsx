"use client";

import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";

import { Button, cx, Empty } from "@/components/ui";
import { api } from "@/lib/api";
import {
  BLOCK_LABEL,
  CATEGORY_LABEL,
  clock,
  PAGE_STATUS_LABEL,
  type Clinician,
  type FeedItem,
  type Page,
} from "@/lib/paging";

import { Countdown, P_RGB, PriorityChip, SourceTag, StatusDot, TimerBar } from "./bits";

export async function pageAction(id: string, action: "approve" | "escalate" | "cancel" | "resolve", body: Record<string, unknown> = {}) {
  return api<{ status: string; detail?: string }>(`/aegis/paging/pages/${id}/${action}`, { body });
}

export function age(ts: number, now: number): string {
  const s = Math.max(0, now - ts);
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  return `${Math.floor(s / 3600)}h`;
}

/** What a page is waiting on, in one line. */
export function statusLine(page: Page, now: number): { text: string; tone: string } {
  const bits: string[] = [PAGE_STATUS_LABEL[page.status]];
  if (page.status === "review" && page.blocked.length) bits.push(`blocked: ${page.blocked.map((b) => BLOCK_LABEL[b] ?? b).join(", ")}`);
  if (page.status === "review" && page.review_deadline)
    bits.push(`${page.blocked.length ? "re-routes" : "sends"} in ${Math.max(0, Math.round(page.review_deadline - now))}s`);
  if (page.status === "held" && page.hold_until) bits.push(`sends in ${Math.max(1, Math.round((page.hold_until - now) / 60))} min`);
  if (page.escalations > 0) bits.push(`escalated ×${page.escalations}`);
  const tone =
    page.status === "unassigned" ? "text-high-ink" : page.status === "review" ? "text-dys-ink" : page.status === "paging" ? "text-accent-ink" : page.status === "accepted" ? "text-low-ink" : "text-dim";
  return { text: bits.join(" · "), tone };
}

/* ------------------------------------------------------------------ */
/* One page, one row                                                   */
/* ------------------------------------------------------------------ */
export function PageRow({ page, clinicians, now, onOpen, canDispatch, selected }: {
  page: Page;
  clinicians: Record<string, Clinician>;
  now: number;
  onOpen: (id: string) => void;
  canDispatch: boolean;
  selected?: boolean;
}) {
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const who = page.assignee ?? page.planned;
  const c = who ? clinicians[who] : null;
  const att = page.attempts[page.attempts.length - 1];
  const ringing = page.status === "paging" && att;
  const st = statusLine(page, now);
  const act = async (a: "approve" | "escalate" | "cancel") => {
    setBusy(a);
    setErr(null);
    try {
      const r = await pageAction(page.id, a);
      if (r.status !== "ok") setErr(r.detail ?? r.status);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(null);
    }
  };
  const actionable = canDispatch && ["review", "held", "unassigned", "paging"].includes(page.status);
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0, transition: { duration: 0.12 } }}
      transition={{ duration: 0.18 }}
      onClick={() => onOpen(page.id)}
      className={cx("group relative cursor-pointer border-b border-line px-3.5 py-2.5 transition-colors last:border-b-0", selected ? "bg-s3" : "hover:bg-s2")}
    >
      <span className="absolute inset-y-0 left-0 w-[2px]" style={{ background: `rgb(${P_RGB[page.priority]})` }} />
      <div className="flex items-center gap-2">
        <PriorityChip p={page.priority} size="sm" />
        <span className="font-mono text-[12px] font-semibold text-ink">{page.room ?? "—"}</span>
        <span className="min-w-0 flex-1 truncate text-[11.5px] text-dim">
          {page.zone_name ?? "Unknown unit"} · {CATEGORY_LABEL[page.category] ?? page.category}
        </span>
        {ringing ? <Countdown remaining={att.deadline - now} className="text-[12px]" /> : <span className="num font-mono text-[11.5px] text-dim">{age(page.created_at, now)}</span>}
      </div>
      <p className="mt-1 line-clamp-2 text-[12.5px] leading-snug text-ink/90">{page.text}</p>
      <div className="mt-1.5 flex items-center gap-2">
        {c ? <StatusDot status={c.status} size={7} /> : <span className="h-[7px] w-[7px] rounded-full border border-dashed border-faint" />}
        <span className="min-w-0 flex-1 truncate text-[11.5px]">
          <span className="text-muted">{c ? c.name : "Unassigned"}</span>
          {c?.present && <span className="text-accent-ink" title="Pager open on a device"> · pager</span>}
          <span className="text-faint"> · </span>
          <span className={st.tone}>{st.text}</span>
        </span>
        <SourceTag source={page.source} />
      </div>
      {actionable && (
        <div className="mt-2 flex items-center gap-1.5" onClick={(e) => e.stopPropagation()}>
          {(page.status === "review" || page.status === "held") && (
            <Button size="sm" variant="primary" disabled={!!busy} onClick={() => act("approve")}>
              {busy === "approve" ? "Paging…" : page.status === "held" ? "Send now" : "Approve & page"}
            </Button>
          )}
          {(page.status === "paging" || page.status === "unassigned") && (
            <Button size="sm" variant="ghost" disabled={!!busy} onClick={() => act("escalate")}>
              {busy === "escalate" ? "Escalating…" : page.status === "unassigned" ? "Find someone" : "Escalate"}
            </Button>
          )}
          <Button size="sm" variant="quiet" disabled={!!busy} onClick={() => act("cancel")}>
            Cancel
          </Button>
          {page.refining && <span className="working ml-auto rounded-[3px] px-1.5 text-[11px] text-accent-ink">model reviewing</span>}
        </div>
      )}
      {err && <div className="mt-1.5 text-[11.5px] text-high-ink">{err}</div>}
      {ringing && <TimerBar remaining={att.deadline - now} total={att.deadline - att.sent_at} priority={page.priority} className="absolute inset-x-0 bottom-0" />}
    </motion.div>
  );
}

export function Queue({ pages, clinicians, now, onOpen, canDispatch, selected }: {
  pages: Page[];
  clinicians: Record<string, Clinician>;
  now: number;
  onOpen: (id: string) => void;
  canDispatch: boolean;
  selected?: string | null;
}) {
  if (!pages.length) return <Empty>No active pages.</Empty>;
  return (
    <div>
      <AnimatePresence initial={false}>
        {pages.map((p) => (
          <PageRow key={p.id} page={p} clinicians={clinicians} now={now} onOpen={onOpen} canDispatch={canDispatch} selected={selected === p.id} />
        ))}
      </AnimatePresence>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The event log                                                       */
/* ------------------------------------------------------------------ */
const FEED_TONE: Record<string, string> = {
  review: "text-dys-ink",
  paging: "text-accent-ink",
  accepted: "text-low-ink",
  unassigned: "text-high-ink",
};

export function Feed({ items, now, onOpen }: { items: FeedItem[]; now: number; onOpen: (pageId: string) => void }) {
  const list = [...items].sort((a, b) => b.created_at - a.created_at).slice(0, 80);
  if (!list.length) return <Empty>No events yet.</Empty>;
  return (
    <ul>
      <AnimatePresence initial={false}>
        {list.map((a) => (
          <motion.li
            key={a.id}
            initial={{ opacity: 0, backgroundColor: "rgba(61,142,245,0.10)" }}
            animate={{ opacity: 1, backgroundColor: "rgba(61,142,245,0)" }}
            transition={{ duration: 1.2 }}
            onClick={() => a.page_id && onOpen(a.page_id)}
            className={cx("grid grid-cols-[52px_minmax(0,1fr)] gap-2 border-b border-line-soft px-3.5 py-2 last:border-b-0", a.page_id && "cursor-pointer hover:bg-s2")}
          >
            <span className="num pt-px font-mono text-[11px] text-dim" title={`${Math.round(now - a.created_at)} s ago`}>
              {clock(a.created_at, false)}
            </span>
            <div className="min-w-0">
              <div className="flex items-start gap-2">
                {a.priority ? <PriorityChip p={a.priority} size="sm" /> : <span className="mt-[5px] h-1.5 w-1.5 shrink-0 rounded-full bg-faint" />}
                <p className={cx("min-w-0 flex-1 text-[12px] leading-snug", a.kind === "note" ? "text-muted" : "text-ink/90")}>{a.text}</p>
              </div>
              <div className="mt-0.5 flex items-center gap-1.5 text-[11px]">
                <span className={FEED_TONE[a.status] ?? "text-dim"}>{a.kind === "note" ? "Note" : PAGE_STATUS_LABEL[a.status as Page["status"]] ?? a.status}</span>
                <span className="text-faint">·</span>
                <SourceTag source={a.source} synthetic={a.synthetic} />
              </div>
            </div>
          </motion.li>
        ))}
      </AnimatePresence>
    </ul>
  );
}
