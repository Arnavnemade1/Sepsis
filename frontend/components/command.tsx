"use client";

import { AnimatePresence, motion } from "motion/react";
import { usePathname, useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

import { Bed, BellRinging, Enter, Forward, Heartbeat, type Icon, Pager, Plus, Search, SignOut, User } from "@/components/icons";
import { api } from "@/lib/api";
import { signInAs, signOut } from "@/lib/session";

import { NAV_ITEMS, SWITCH_USERS } from "./nav";
import { cx } from "./ui";

type Cmd = { id: string; group: string; label: string; hint?: string; icon: Icon; keywords?: string; run: () => void; tone?: string };

type Snapshot = {
  pages: { id: string; priority: number; room: string | null; zone_name: string | null; text: string; status: string }[];
  clinicians: { id: string; name: string; specialty_label: string; pageable: boolean }[];
};
type Roster = { patients: { id: string; name: string; kind: string }[] };
type Rooms = { rooms: { room: string; zone_name: string; kind: string | null; name?: string | null; diagnosis?: string | null; worst?: string | null; scores?: { news2?: { score: number } }; findings?: { title: string }[] }[] };

const ACTIVE = new Set(["review", "held", "paging", "accepted", "unassigned"]);
const GROUP_ORDER = ["Actions", "Go to", "Active pages", "Rooms", "Patients", "Pagers", "Account"];

/** ⌘K: jump anywhere, act on anything. */
export function CommandPalette({ open, onClose }: { open: boolean; onClose: () => void }) {
  const router = useRouter();
  const path = usePathname();
  const [q, setQ] = useState("");
  const [idx, setIdx] = useState(0);
  const [snap, setSnap] = useState<Snapshot | null>(null);
  const [roster, setRoster] = useState<Roster | null>(null);
  const [rooms, setRooms] = useState<Rooms | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const list = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    setQ("");
    setIdx(0);
    setTimeout(() => input.current?.focus(), 30);
    api<Snapshot>("/aegis/paging/snapshot", { timeoutMs: 20_000 }).then(setSnap).catch(() => {});
    api<Roster>("/aegis/patients", { timeoutMs: 20_000 }).then(setRoster).catch(() => {});
    api<Rooms>("/aegis/rooms", { timeoutMs: 20_000 }).then(setRooms).catch(() => {});
  }, [open]);

  const go = (href: string) => {
    onClose();
    router.push(href);
  };

  const cmds: Cmd[] = useMemo(() => {
    const out: Cmd[] = [
      {
        id: "a:alert",
        group: "Actions",
        label: "Raise an alert",
        hint: "Text or voice; triaged, routed and certified",
        icon: Plus,
        keywords: "new page nurse compose",
        run: () => {
          onClose();
          if (path?.startsWith("/operator")) window.dispatchEvent(new CustomEvent("aegis:compose"));
          else router.push("/operator/?compose=1");
        },
      },
      ...NAV_ITEMS.map((n) => ({ id: `n:${n.key}`, group: "Go to", label: n.label, hint: n.hint, icon: n.icon, keywords: n.key, run: () => go(n.href) })),
    ];
    for (const p of (snap?.pages ?? []).filter((p) => ACTIVE.has(p.status)).sort((a, b) => a.priority - b.priority)) {
      out.push({
        id: `p:${p.id}`,
        group: "Active pages",
        label: `${p.id} · Room ${p.room ?? "?"}`,
        hint: p.text,
        icon: BellRinging,
        tone: p.priority === 1 ? "text-high-ink" : p.priority === 2 ? "text-dys-ink" : undefined,
        keywords: `p${p.priority} ${p.zone_name ?? ""} ${p.status}`,
        run: () => go(`/operator/?page=${encodeURIComponent(p.id)}`),
      });
    }
    for (const r of (rooms?.rooms ?? []).filter((r) => r.kind)) {
      const top = r.findings?.[0]?.title;
      out.push({
        id: `r:${r.room}`,
        group: "Rooms",
        label: `Room ${r.room} · ${r.name ?? "Patient"}`,
        hint: top ?? `${r.zone_name} · ${r.diagnosis ?? ""}`,
        icon: Bed,
        tone: r.worst === "critical" ? "text-high-ink" : r.worst === "urgent" ? "text-dys-ink" : undefined,
        keywords: `${r.room} ${r.zone_name} ${r.diagnosis ?? ""} news2 ${r.scores?.news2?.score ?? ""} ${(r.findings ?? []).map((f) => f.title).join(" ")}`,
        run: () => go(`/room/?id=${encodeURIComponent(r.room)}`),
      });
    }
    for (const p of roster?.patients ?? []) {
      out.push({
        id: `pt:${p.id}`,
        group: "Patients",
        label: p.name,
        hint: p.kind === "live" ? "Live monitored session" : p.id.replace("bed_", "Bed "),
        icon: Heartbeat,
        keywords: p.id,
        run: () => go(`/patient/?id=${encodeURIComponent(p.id)}`),
      });
    }
    for (const c of (snap?.clinicians ?? []).filter((c) => c.pageable)) {
      out.push({ id: `c:${c.id}`, group: "Pagers", label: c.name, hint: `Open pager · ${c.specialty_label}`, icon: Pager, keywords: c.id, run: () => go(`/clinician/?id=${encodeURIComponent(c.id)}`) });
    }
    for (const u of SWITCH_USERS) {
      out.push({
        id: `u:${u.u}`,
        group: "Account",
        label: `Switch to ${u.n}`,
        hint: u.r,
        icon: User,
        keywords: `sign in user ${u.u}`,
        run: () => {
          onClose();
          signInAs(u.u).catch(() => {});
        },
      });
    }
    out.push({ id: "u:out", group: "Account", label: "Sign out", hint: "View only until you choose a user", icon: SignOut, run: () => { onClose(); signOut().then(() => router.refresh()); } });
    return out;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [snap, roster, rooms, path]);

  const shown = useMemo(() => {
    const terms = q.toLowerCase().split(/\s+/).filter(Boolean);
    const match = (c: Cmd) => {
      const hay = `${c.label} ${c.hint ?? ""} ${c.keywords ?? ""}`.toLowerCase();
      return terms.every((t) => hay.includes(t));
    };
    const pool = terms.length ? cmds.filter(match) : cmds.filter((c) => c.group === "Actions" || c.group === "Go to" || c.group === "Active pages");
    const groups: { group: string; items: Cmd[] }[] = [];
    for (const g of GROUP_ORDER) {
      const items = pool.filter((c) => c.group === g).slice(0, terms.length ? 6 : g === "Active pages" ? 5 : 20);
      if (items.length) groups.push({ group: g, items });
    }
    return groups;
  }, [cmds, q]);
  const flat = shown.flatMap((g) => g.items);

  useEffect(() => setIdx(0), [q]);
  useEffect(() => {
    list.current?.querySelector(`[data-idx="${idx}"]`)?.scrollIntoView({ block: "nearest" });
  }, [idx]);

  const onKey = (e: React.KeyboardEvent) => {
    if (e.key === "ArrowDown") {
      e.preventDefault();
      setIdx((i) => Math.min(flat.length - 1, i + 1));
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      setIdx((i) => Math.max(0, i - 1));
    } else if (e.key === "Enter") {
      e.preventDefault();
      flat[idx]?.run();
    } else if (e.key === "Escape") {
      e.preventDefault();
      onClose();
    }
  };

  let n = -1;
  return (
    <AnimatePresence>
      {open && (
        <motion.div
          className="fixed inset-0 z-[80] flex items-start justify-center bg-black/50 px-3 pt-[12vh]"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.12 }}
          onMouseDown={(e) => e.target === e.currentTarget && onClose()}
        >
          <motion.div
            className="pop w-full max-w-[620px] overflow-hidden"
            initial={{ y: 8, opacity: 0, scale: 0.99 }}
            animate={{ y: 0, opacity: 1, scale: 1 }}
            exit={{ y: 4, opacity: 0 }}
            transition={{ duration: 0.15, ease: [0.2, 0.8, 0.2, 1] }}
            role="dialog"
            aria-label="Command palette"
          >
            <div className="flex h-[52px] items-center gap-3 border-b border-white/[0.06] px-4">
              <Search size={18} className="shrink-0 text-dim" />
              <input
                ref={input}
                value={q}
                onChange={(e) => setQ(e.target.value)}
                onKeyDown={onKey}
                placeholder="Search pages, patients, pagers, or jump to a section"
                className="min-w-0 flex-1 bg-transparent text-base text-ink outline-none placeholder:text-faint"
                aria-label="Search"
              />
              <span className="kbd">esc</span>
            </div>
            <div ref={list} className="max-h-[52vh] overflow-y-auto py-1.5">
              {flat.length === 0 && <div className="px-4 py-8 text-center text-sm text-dim">No results for “{q}”.</div>}
              {shown.map((g) => (
                <div key={g.group} className="pb-1">
                  <div className="px-4 pb-1 pt-2 text-xs font-medium text-faint">{g.group}</div>
                  {g.items.map((c) => {
                    n += 1;
                    const i = n;
                    const Icon = c.icon;
                    return (
                      <button
                        key={c.id}
                        data-idx={i}
                        onMouseMove={() => idx !== i && setIdx(i)}
                        onClick={() => c.run()}
                        className={cx("mx-1.5 flex w-[calc(100%-12px)] items-center gap-3 rounded-md px-2.5 py-2 text-left", i === idx ? "bg-white/[0.07]" : "")}
                      >
                        <Icon size={16} className={cx("shrink-0", c.tone ?? (i === idx ? "text-ink" : "text-dim"))} />
                        <span className={cx("shrink-0 text-sm", i === idx ? "text-ink" : "text-ink/90")}>{c.label}</span>
                        {c.hint && <span className="min-w-0 flex-1 truncate text-xs text-dim">{c.hint}</span>}
                        {i === idx && <Forward size={14} className="ml-auto shrink-0 text-dim" />}
                      </button>
                    );
                  })}
                </div>
              ))}
            </div>
            <div className="flex items-center gap-4 border-t border-white/[0.06] bg-black/[0.12] px-4 py-2.5 text-xs text-dim">
              <span className="flex items-center gap-1.5">
                <span className="kbd">↑</span>
                <span className="kbd">↓</span> navigate
              </span>
              <span className="flex items-center gap-1.5">
                <span className="kbd">
                  <Enter size={10} />
                </span>
                open
              </span>
              <span className="ml-auto text-faint">AegisOS</span>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
