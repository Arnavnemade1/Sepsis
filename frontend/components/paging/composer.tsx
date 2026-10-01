"use client";

import { motion } from "motion/react";
import { useEffect, useMemo, useRef, useState } from "react";

import { Button, Modal } from "@/components/ui";
import { api } from "@/lib/api";
import { CATEGORY_LABEL, PAGE_STATUS_LABEL, type Clinician, type Layout, type Occupant, type Page } from "@/lib/paging";

import { Avatar, PriorityChip } from "./bits";
import { pageAction } from "./queue";

const EXAMPLES = [
  "Room 512: BP 84/52, HR 126, temp 39.3, new confusion",
  "Chest pain in 409, diaphoretic, history of CABG",
  "Fall in 503, no head strike, alert and oriented",
  "603 new facial droop, last known well 30 min ago",
  "Pt in 611 asking for something to help sleep",
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
  const [placeholder, setPlaceholder] = useState(EXAMPLES[0]);
  const rec = useRef<SpeechRec | null>(null);
  const field = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (!open) return;
    setText(presetRoom ? `Room ${presetRoom}: ` : "");
    setResult(null);
    setErr(null);
    setVoiceUsed(false);
    setTimeout(() => field.current?.focus(), 60);
    let i = 0;
    const t = setInterval(() => setPlaceholder(EXAMPLES[++i % EXAMPLES.length]), 3200);
    return () => clearInterval(t);
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
    <Modal open={open} onClose={onClose} width={680}>
      {!result ? (
        <div>
          <div className="flex items-center gap-2 border-b border-line-soft px-5 py-3">
            <span className="text-[13px] font-semibold text-ink">New alert</span>
            <span className="text-[12px] text-dim">Describe what is happening. Triage, routing and the page are automatic.</span>
            <span className="kbd ml-auto">esc</span>
          </div>
          <div className="px-5 pt-4">
            <textarea
              ref={field}
              value={text}
              onChange={(e) => setText(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && (e.metaKey || e.ctrlKey)) submit();
              }}
              placeholder={placeholder}
              rows={4}
              className="w-full resize-none bg-transparent text-[18px] leading-relaxed text-ink outline-none placeholder:text-faint"
            />
          </div>
          <div className="flex flex-wrap items-center gap-2 px-5 pb-4">
            {room && zone ? (
              <span className="inline-flex items-center gap-1.5 rounded-full border border-line bg-s2 px-2.5 py-1 text-[12px] text-ink">
                <span className="font-mono font-semibold">{room}</span>
                <span className="text-dim">{zone.name}</span>
                {occ && <span className="text-muted">· {occ.name}</span>}
                {occ?.monitored && <span className="text-accent-ink">· monitored</span>}
              </span>
            ) : (
              <span className="rounded-full border border-dashed border-line-bright px-2.5 py-1 text-[12px] text-dim">
                {text.trim() ? "No room detected yet (the operator will be asked)" : "Mention a room, e.g. 512 or A4"}
              </span>
            )}
            {listening && (
              <span className="inline-flex items-center gap-1.5 text-[12px] text-high-ink">
                <span className="live-dot h-2 w-2 rounded-full bg-high" style={{ color: "rgb(229 72 77 / .5)" }} />
                Listening…
              </span>
            )}
          </div>
          {err && <div className="mx-5 mb-3 rounded-lg bg-high/8 px-3 py-2 text-[12.5px] text-high-ink">{err}</div>}
          <div className="flex items-center gap-2 border-t border-line-soft bg-s2/50 px-5 py-3">
            {speechOK && (
              <Button variant={listening ? "danger" : "ghost"} onClick={toggleVoice} title="Dictate the alert">
                <svg width="14" height="14" viewBox="0 0 16 16" fill="none">
                  <rect x="5.5" y="1.5" width="5" height="8.5" rx="2.5" stroke="currentColor" strokeWidth="1.5" />
                  <path d="M3 8a5 5 0 0 0 10 0M8 13v2" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                </svg>
                {listening ? "Stop" : "Voice"}
              </Button>
            )}
            {!canRaise && <span className="text-[12px] text-dys-ink">Sign in as staff to raise alerts; guests are view-only.</span>}
            <div className="ml-auto flex items-center gap-2">
              <span className="hidden text-[11.5px] text-dim sm:inline">
                <span className="kbd">⌘</span> <span className="kbd">↵</span> to send
              </span>
              <Button variant="primary" onClick={submit} disabled={!text.trim() || busy}>
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
    { k: "Triage", v: `P${page.priority} · ${CATEGORY_LABEL[page.category]}${page.triage.red_flags[0] ? ` · ${page.triage.red_flags[0]}` : ""}`, ms: page.timings.triage_ms },
    { k: "Locate", v: page.room ? `Room ${page.room} · ${page.zone_name}` : "room not identified", ms: page.timings.locate_ms },
    { k: "Route", v: c ? `${c.name} · ${c.specialty_label}` : "no eligible clinician", ms: page.timings.route_ms },
    { k: "Verify", v: page.certificate ? (page.certificate.certified ? "Z3: every invariant holds" : `Z3 refused: ${page.certificate.violations.join(", ")}`) : "held for the operator's review", ms: page.timings.certify_ms },
    { k: "Page", v: PAGE_STATUS_LABEL[status], ms: undefined },
  ];
  return (
    <div className="p-6">
      <div className="flex items-center gap-2">
        <PriorityChip p={page.priority} size="lg" label />
        <span className="font-mono text-[12px] text-dim">{page.id}</span>
        {page.timings.total_ms != null && <span className="num ml-auto font-mono text-[12px] text-dim">dispatched in {page.timings.total_ms.toFixed(1)} ms</span>}
      </div>
      <ol className="mt-5 space-y-2.5">
        {steps.map((s, i) => (
          <motion.li
            key={s.k}
            initial={{ opacity: 0, x: -6 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.12 * i, duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="flex items-center gap-3"
          >
            <span className="grid h-6 w-6 place-items-center rounded-full bg-low/12 text-low-ink">
              <svg width="11" height="11" viewBox="0 0 8 8"><path d="M1.5 4.2 3.2 6 6.5 2" stroke="currentColor" strokeWidth="1.3" fill="none" strokeLinecap="round" /></svg>
            </span>
            <span className="w-14 text-[13px] font-medium text-ink">{s.k}</span>
            <span className="min-w-0 flex-1 truncate text-[13px] text-muted">{s.v}</span>
            {s.ms != null && <span className="num font-mono text-[11px] text-dim">{s.ms < 1 ? "<1" : s.ms.toFixed(1)} ms</span>}
          </motion.li>
        ))}
      </ol>
      {c && (
        <motion.div initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} transition={{ delay: 0.7 }} className="mt-5 flex items-center gap-3 rounded-xl border border-line p-3">
          <Avatar name={c.name} status={c.status} size={36} />
          <div className="min-w-0 flex-1">
            <div className="text-[14px] font-medium text-ink">
              {status === "paging" ? "Paging " : status === "review" ? "Ready to page " : status === "held" ? "Queued for " : ""}
              {c.name}
            </div>
            <div className="truncate text-[12px] text-dim">{page.decision.reasoning}</div>
          </div>
          {status === "review" && (
            <Button
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
        </motion.div>
      )}
      {page.review_reason && status === "review" && <p className="mt-3 text-[12.5px] text-dys-ink">{page.review_reason}</p>}
      {page.refining && <p className="shimmer mt-3 rounded-lg px-3 py-2 text-[12px] text-accent-ink">Sent on the rules. The model is reviewing it now and may raise the priority.</p>}
      <div className="mt-6 flex items-center justify-end gap-2">
        <Button variant="quiet" onClick={onAnother}>
          New alert
        </Button>
        <Button
          variant="ghost"
          onClick={() => {
            onOpenPage(page.id);
            onClose();
          }}
        >
          Open the page
        </Button>
        <Button variant="primary" onClick={onClose}>
          Done
        </Button>
      </div>
    </div>
  );
}

