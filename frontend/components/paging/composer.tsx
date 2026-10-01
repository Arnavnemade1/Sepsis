"use client";

import { motion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";

import { Button, cx, Kbd, Modal } from "@/components/ui";
import { api } from "@/lib/api";
import { CATEGORY_LABEL, PAGE_STATUS_LABEL, type Clinician, type Layout, type Occupant, type Page } from "@/lib/paging";

import { Avatar, PriorityChip } from "./bits";
import { pageAction } from "./queue";

const EXAMPLES = [
  "Room 512: BP 84/52, HR 126, temp 39.3, new confusion",
  "Chest pain in 409, diaphoretic, history of CABG",
  "Fall in 503, no head strike, alert and oriented",
  "603 new facial droop, last known well 30 min ago",
  "605 desaturating to 84% on room air, RR 30",
];

// the same room patterns the backend uses, so the preview matches the pipeline
function detectRoom(text: string, rooms: Record<string, unknown>): string | null {
  const rx = [/\b(?:room|rm\.?|bed|bay)\s*#?\s*([A-Za-z]{0,2}\d{1,3})\b/gi, /\b(OR\s?\d|T\d|A\d{1,2})\b/gi, /\b(\d{3})\b/g];
  for (let i = 0; i < rx.length; i++) {
    for (const m of text.matchAll(rx[i])) {
      const tok = m[1].toUpperCase().replace(/\s/g, "");
      if (!(tok in rooms)) continue;
      if (i === rx.length - 1) {
        const before = text.slice(Math.max(0, (m.index ?? 0) - 12), m.index ?? 0);
        const after = text.slice((m.index ?? 0) + m[0].length, (m.index ?? 0) + m[0].length + 1);
        if (/(hr|bp|rr|map|sats?|spo2|o2|temp|glucose|bg|lactate|pulse|gcs)\s*[:=]?\s*$/i.test(before) || ["/", "%", "."].includes(after)) continue;
      }
      return tok;
    }
  }
  return null;
}

type SpeechRec = { start: () => void; stop: () => void; onresult: ((e: { results: ArrayLike<ArrayLike<{ transcript: string }>> }) => void) | null; onend: (() => void) | null; interimResults: boolean; continuous: boolean; lang: string };

export function Composer({ open, onClose, layout, occupancy, clinicians, presetRoom, onOpenPage, canRaise }: {
  open: boolean;
  onClose: () => void;
  layout: Layout | null;
  occupancy: Record<string, Occupant>;
  clinicians: Record<string, Clinician>;
  presetRoom?: string | null;
  onOpenPage: (id: string) => void;
  canRaise: boolean;
}) {
  const [text, setText] = useState("");
  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);
  const [result, setResult] = useState<Page | null>(null);
  const [listening, setListening] = useState(false);
  const [voiceUsed, setVoiceUsed] = useState(false);
  const rec = useRef<SpeechRec | null>(null);
  const field = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!open) return;
    setText(presetRoom ? `Room ${presetRoom}: ` : "");
    setResult(null);
    setErr(null);
    setVoiceUsed(false);
    setTimeout(() => field.current?.focus(), 50);
  }, [open, presetRoom]);

  const room = useMemo(() => (layout ? detectRoom(text, layout.rooms) : null), [text, layout]);
  const zone = room && layout ? layout.zones[layout.rooms[room].zone] : null;
  const occ = room ? occupancy[room] : null;
  const speechOK = typeof window !== "undefined" && ("webkitSpeechRecognition" in window || "SpeechRecognition" in window);

  const toggleVoice = () => {
    if (listening) {
      rec.current?.stop();
      return;
    }
    const W = window as unknown as { webkitSpeechRecognition?: new () => SpeechRec; SpeechRecognition?: new () => SpeechRec };
    const Ctor = W.SpeechRecognition ?? W.webkitSpeechRecognition;
    if (!Ctor) return;
    const r = new Ctor();
    r.lang = "en-US";
    r.interimResults = true;
    r.continuous = true;
    const base = text ? text.trimEnd() + " " : "";
    r.onresult = (e) => {
      let t = "";
      for (let i = 0; i < e.results.length; i++) t += e.results[i][0].transcript;
      setText(base + t);
    };
    r.onend = () => setListening(false);
    rec.current = r;
    setVoiceUsed(true);
    setListening(true);
    r.start();
  };

  const submit = async () => {
    if (!text.trim() || busy) return;
    rec.current?.stop();
    setBusy(true);
    setErr(null);
    try {
      const r = await api<{ status: string; page?: Page; detail?: string }>("/aegis/paging/alerts", {
        body: { text, room: room ?? undefined, source: voiceUsed ? "voice" : "nurse" },
        timeoutMs: 60_000,
      });
      if (r.status !== "ok" || !r.page) throw new Error(r.detail ?? "the alert was refused");
      setResult(r.page);
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Modal open={open} onClose={onClose} width={640}>
      {!result ? (
        <div>
          <div className="flex items-start gap-3 px-4 pb-2 pt-4">
            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" className="mt-[3px] shrink-0 text-dim" aria-hidden>
              <path d="M8 1.5v13M1.5 8h13" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            </svg>
            <textarea
              ref={field}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
              }}
              placeholder="Describe the situation: room, what you see, any readings"
              rows={3}
              className="min-w-0 flex-1 resize-none bg-transparent text-[15px] leading-relaxed text-ink outline-none placeholder:text-faint"
              aria-label="Alert"
            />
          </div>
          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t border-line px-4 py-2.5 text-[12px]">
            <span className="text-dim">Room</span>
            {room && zone ? (
              <span className="flex min-w-0 items-center gap-1.5">
                <span className="font-mono font-semibold text-ink">{room}</span>
                <span className="text-muted">{zone.name}</span>
                {occ && <span className="truncate text-dim">· {occ.name}</span>}
                {occ?.monitored && <span className="text-accent-ink">· monitored</span>}
              </span>
            ) : (
              <span className="text-faint">{text.trim() ? "not detected: the operator will be asked" : "mention a room, e.g. 512 or A4"}</span>
            )}
            {listening && (
              <span className="ml-auto inline-flex items-center gap-1.5 text-high-ink">
                <span className="blink h-1.5 w-1.5 rounded-full bg-high" />
                Listening
              </span>
            )}
          </div>
          {!text.trim() && (
            <div className="border-t border-line px-2 py-1.5">
              {EXAMPLES.slice(0, 3).map((x) => (
                <button key={x} onClick={() => setText(x)} className="block w-full truncate rounded-md px-2 py-1.5 text-left text-[12.5px] text-dim transition-colors hover:bg-s2 hover:text-ink">
                  {x}
                </button>
              ))}
            </div>
          )}
          {err && <div className="border-t border-line bg-high/10 px-4 py-2 text-[12.5px] text-high-ink">{err}</div>}
          <div className="flex items-center gap-2 border-t border-line bg-s2/60 px-3 py-2">
            {speechOK && (
              <Button size="sm" variant={listening ? "danger" : "quiet"} onClick={toggleVoice} title="Dictate the alert">
                <svg width="13" height="13" viewBox="0 0 16 16" fill="none" aria-hidden>
                  <rect x="5.5" y="1.5" width="5" height="8.5" rx="2.5" stroke="currentColor" strokeWidth="1.5" />
                  <path d="M3 8a5 5 0 0 0 10 0M8 13v2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
                {listening ? "Stop" : "Dictate"}
              </Button>
            )}
            {!canRaise && <span className="text-[12px] text-dys-ink">View only. Switch to a staff user to raise alerts.</span>}
            <div className="ml-auto flex items-center gap-3">
              <span className="hidden items-center gap-1 text-[11.5px] text-dim sm:flex">
                <Kbd>esc</Kbd> close
              </span>
              <span className="hidden items-center gap-1 text-[11.5px] text-dim sm:flex">
                <Kbd>⌘</Kbd>
                <Kbd>↵</Kbd> send
              </span>
              <Button size="sm" variant="primary" onClick={submit} disabled={!text.trim() || busy}>
                {busy ? "Dispatching…" : "Send alert"}
              </Button>
            </div>
          </div>
        </div>
      ) : (
        <Result page={result} clinicians={clinicians} onClose={onClose} onOpenPage={onOpenPage} onAnother={() => { setResult(null); setText(""); }} />
      )}
    </Modal>
  );
}

function Result({ page, clinicians, onClose, onOpenPage, onAnother }: {
  page: Page;
  clinicians: Record<string, Clinician>;
  onClose: () => void;
  onOpenPage: (id: string) => void;
  onAnother: () => void;
}) {
  const [busy, setBusy] = useState(false);
  const [status, setStatus] = useState(page.status);
  const who = page.assignee ?? page.planned;
  const c = who ? clinicians[who] : null;
  const steps = [
    { k: "Triage", v: `P${page.priority} · ${CATEGORY_LABEL[page.category]}${page.triage.red_flags[0] ? ` · ${page.triage.red_flags[0]}` : ""}`, ms: page.timings.triage_ms, ok: true },
    { k: "Locate", v: page.room ? `Room ${page.room} · ${page.zone_name}` : "Room not identified", ms: page.timings.locate_ms, ok: !!page.room },
    { k: "Route", v: c ? `${c.name} · ${c.specialty_label}` : "No eligible clinician", ms: page.timings.route_ms, ok: !!c },
    {
      k: "Verify",
      v: page.certificate ? (page.certificate.certified ? "Z3: every invariant holds" : `Z3 refused: ${page.certificate.violations.join(", ")}`) : "Held for the operator's review",
      ms: page.timings.certify_ms,
      ok: page.certificate ? page.certificate.certified : true,
    },
    { k: "Page", v: PAGE_STATUS_LABEL[status], ms: undefined, ok: true },
  ];
  return (
    <div>
      <div className="flex items-center gap-2 border-b border-line px-4 py-3">
        <PriorityChip p={page.priority} label />
        <span className="font-mono text-[11.5px] text-dim">{page.id}</span>
        {page.timings.total_ms != null && <span className="num ml-auto font-mono text-[11.5px] text-dim">dispatched in {page.timings.total_ms.toFixed(1)} ms</span>}
      </div>
      <ol className="divide-y divide-line-soft">
        {steps.map((s, i) => (
          <motion.li
            key={s.k}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.06 * i, duration: 0.15 }}
            className="grid grid-cols-[14px_64px_minmax(0,1fr)_auto] items-center gap-3 px-4 py-2"
          >
            <span className={cx("h-1.5 w-1.5 justify-self-center rounded-full", s.ok ? "bg-low" : "bg-high")} />
            <span className="text-[12.5px] font-medium text-ink">{s.k}</span>
            <span className="truncate text-[12.5px] text-muted">{s.v}</span>
            <span className="num font-mono text-[11px] text-dim">{s.ms != null ? (s.ms < 1 ? "<1 ms" : `${s.ms.toFixed(1)} ms`) : ""}</span>
          </motion.li>
        ))}
      </ol>
      {c && (
        <div className="flex items-center gap-3 border-t border-line px-4 py-3">
          <Avatar name={c.name} status={c.status} size={28} />
          <div className="min-w-0 flex-1">
            <div className="text-[13px] text-ink">
              {status === "paging" ? "Paging " : status === "review" ? "Ready to page " : status === "held" ? "Queued for " : ""}
              <span className="font-medium">{c.name}</span>
            </div>
            <div className="truncate text-[11.5px] text-dim">{page.decision.reasoning}</div>
          </div>
          {status === "review" && (
            <Button
              size="sm"
              variant="primary"
              disabled={busy}
              onClick={async () => {
                setBusy(true);
                const r = await pageAction(page.id, "approve").catch(() => null);
                if (r?.status === "ok") setStatus("paging");
                setBusy(false);
              }}
            >
              {busy ? "Paging…" : "Approve & page"}
            </Button>
          )}
        </div>
      )}
      {page.review_reason && status === "review" && <p className="border-t border-line px-4 py-2 text-[12px] text-dys-ink">{page.review_reason}</p>}
      {page.refining && <p className="working border-t border-line px-4 py-2 text-[12px] text-accent-ink">Sent on the rules. A model is reviewing it and may raise the priority.</p>}
      <div className="flex items-center justify-end gap-2 border-t border-line bg-s2/60 px-3 py-2">
        <Button size="sm" variant="quiet" onClick={onAnother}>
          New alert
        </Button>
        <Button
          size="sm"
          variant="ghost"
          onClick={() => {
            onOpenPage(page.id);
            onClose();
          }}
        >
          Open page
        </Button>
        <Button size="sm" variant="primary" onClick={onClose}>
          Done
        </Button>
      </div>
    </div>
  );
}
