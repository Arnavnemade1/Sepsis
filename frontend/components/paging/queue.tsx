"use client";

import { AnimatePresence, motion } from "motion/react";
import { useState } from "react";

import { Button, cx, Empty } from "@/components/ui";
import { api } from "@/lib/api";
import {
  BLOCK_LABEL,
  CATEGORY_LABEL,
  PAGE_STATUS_LABEL,
  since,
  type Clinician,
  type FeedItem,
  type Page,
} from "@/lib/paging";

import { Avatar, Countdown, P_RGB, PriorityChip, SourceTag } from "./bits";

export async function pageAction(id: string, action: "approve" | "escalate" | "cancel" | "resolve", body: Record<string, unknown> = {}) {
  return api<{ status: string; detail?: string }>(`/aegis/paging/pages/${id}/${action}`, { body });
}

/* ------------------------------------------------------------------ */
/* The page queue                                                      */
/* ------------------------------------------------------------------ */
export function QueueCard({ page, clinicians, now, onOpen, canDispatch, selected }: {
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
  const ringing = page.status === "paging" && att;
  return (
    <motion.div
      layout
      initial={{ opacity: 0, y: -8, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, x: 20, transition: { duration: 0.2 } }}
      transition={{ type: "spring", bounce: 0, duration: 0.45 }}
      onClick={() => onOpen(page.id)}
      className={cx(
        "group relative cursor-pointer overflow-hidden rounded-xl border bg-s1 p-3 transition hover:border-line-bright hover:shadow-[0_6px_20px_-12px_rgb(16_18_24/.25)]",
        selected ? "border-accent/50 ring-2 ring-accent/15" : "border-line",
      )}
    >
      <span className="absolute inset-y-0 left-0 w-[3px]" style={{ background: `rgb(${P_RGB[page.priority]})` }} />
      <div className="flex items-start gap-2.5">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2">
            <PriorityChip p={page.priority} size="sm" />
            <span className="font-mono text-[11.5px] font-medium text-ink">{page.room ?? "room ?"}</span>
            <span className="truncate text-[11.5px] text-dim">
              {page.zone_name ?? ""} · {CATEGORY_LABEL[page.category] ?? page.category}
            </span>
          </div>
          <p className="mt-1.5 line-clamp-2 text-[13px] leading-snug text-ink">{page.text}</p>
        </div>
        {ringing ? (
          <Countdown remaining={att.deadline - now} total={att.deadline - att.sent_at} priority={page.priority} />
        ) : (
          <span className="num shrink-0 font-mono text-[11px] text-dim">{since(page.created_at, now).replace(" ago", "")}</span>
        )}
      </div>

      <div className="mt-2.5 flex items-center gap-2">
        {c ? <Avatar name={c.name} status={c.status} size={22} present={c.present} /> : <span className="h-[22px] w-[22px] rounded-full border border-dashed border-line-bright" />}
        <div className="min-w-0 flex-1 leading-tight">
          <div className="truncate text-[12px] font-medium text-ink">{c ? c.name : "No one yet"}</div>
          <div className={cx("truncate text-[11px]", page.status === "unassigned" || page.status === "review" ? "text-dys-ink" : "text-dim")}>
            {PAGE_STATUS_LABEL[page.status]}
            {page.status === "held" && page.hold_until ? ` · sends in ${Math.max(1, Math.round((page.hold_until - now) / 60))} min` : ""}
            {page.status === "review" && page.blocked.length > 0 ? ` · blocked: ${page.blocked.map((b) => BLOCK_LABEL[b] ?? b).join(", ")}` : ""}
            {page.status === "review" && page.review_deadline
              ? ` · ${page.blocked.length > 0 ? "re-routes" : "auto-sends"} in ${Math.max(0, Math.round(page.review_deadline - now))}s`
              : ""}
            {page.status === "accepted" && page.accepted_at ? ` ${since(page.accepted_at, now)}` : ""}
            {page.escalations > 0 ? ` · escalated ×${page.escalations}` : ""}
          </div>
        </div>
        <SourceTag source={page.source} />
      </div>

      {canDispatch && ["review", "held", "unassigned", "paging"].includes(page.status) && (
        <div className="mt-2.5 flex items-center gap-1.5" onClick={(e) => e.stopPropagation()}>
          {(page.status === "review" || page.status === "held") && (
            <Button size="sm" variant="primary" disabled={!!busy} onClick={() => act("approve")}>
              {busy === "approve" ? "Paging…" : page.status === "held" ? "Send now" : "Approve & page"}
            </Button>
          )}
          {(page.status === "paging" || page.status === "unassigned") && (
            <Button size="sm" variant="ghost" disabled={!!busy} onClick={() => act("escalate")}>
              {busy === "escalate" ? "Escalating…" : page.status === "unassigned" ? "Find someone" : "Escalate now"}
            </Button>
          )}
          <Button size="sm" variant="quiet" disabled={!!busy} onClick={() => act("cancel")}>
            Cancel
          </Button>
          {page.refining && <span className="shimmer ml-auto rounded px-1.5 text-[10.5px] text-accent-ink">model refining</span>}
        </div>
      )}
      {err && <div className="mt-2 text-[11.5px] text-high-ink">{err}</div>}
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
  if (!pages.length)
    return (
      <Empty
        icon={
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" className="text-faint">
            <path d="M6 8a6 6 0 1 1 12 0c0 7 3 9 3 9H3s3-2 3-9Z" stroke="currentColor" strokeWidth="1.5" strokeLinejoin="round" />
            <path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
          </svg>
        }
      >
        No open pages. The queue is clear.
      </Empty>
    );
  return (
    <div className="space-y-2">
      <AnimatePresence initial={false}>
        {pages.map((p) => (
          <QueueCard key={p.id} page={p} clinicians={clinicians} now={now} onOpen={onOpen} canDispatch={canDispatch} selected={selected === p.id} />
        ))}
      </AnimatePresence>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* The alert feed                                                      */
/* ------------------------------------------------------------------ */
const FEED_STATUS: Record<string, { label: string; cls: string }> = {
  review: { label: "Awaiting review", cls: "text-dys-ink" },
  held: { label: "Queued", cls: "text-muted" },
  paging: { label: "Ringing", cls: "text-accent-ink" },
  accepted: { label: "Accepted", cls: "text-low-ink" },
  resolved: { label: "Resolved", cls: "text-dim" },
  cancelled: { label: "Cancelled", cls: "text-dim" },
  declined: { label: "Declined", cls: "text-dim" },
  unassigned: { label: "Needs an owner", cls: "text-high-ink" },
  info: { label: "Detector", cls: "text-dim" },
  new: { label: "New", cls: "text-muted" },
};

export function Feed({ items, now, onOpen }: { items: FeedItem[]; now: number; onOpen: (pageId: string) => void }) {
  const list = [...items].sort((a, b) => b.created_at - a.created_at).slice(0, 60);
  if (!list.length) return <Empty>No alerts yet.</Empty>;
  return (
    <ul className="divide-y divide-line-soft">
      <AnimatePresence initial={false}>
        {list.map((a) => {
          const st = FEED_STATUS[a.status] ?? { label: a.status, cls: "text-dim" };
          return (
            <motion.li
              key={a.id}
              layout="position"
              initial={{ opacity: 0, backgroundColor: "rgb(68 83 214 / 0.06)" }}
              animate={{ opacity: 1, backgroundColor: "rgb(68 83 214 / 0)" }}
              transition={{ duration: 1.2 }}
              onClick={() => a.page_id && onOpen(a.page_id)}
              className={cx("px-4 py-3", a.page_id && "cursor-pointer hover:bg-s2/70")}
            >
              <div className="flex items-start gap-2.5">
                <p className={cx("min-w-0 flex-1 text-[13px] leading-snug", a.kind === "note" ? "text-muted" : "text-ink")}>{a.text}</p>
                {a.priority ? <PriorityChip p={a.priority} size="sm" /> : null}
              </div>
              <div className="mt-1 flex items-center gap-2 text-[11px]">
                <span className={cx("font-medium", st.cls)}>{st.label}</span>
                <span className="text-faint">·</span>
                <span className="text-dim">{since(a.created_at, now)}</span>
                <span className="ml-auto">
                  <SourceTag source={a.source} synthetic={a.synthetic} />
                </span>
              </div>
            </motion.li>
          );
        })}
      </AnimatePresence>
    </ul>
  );
}
