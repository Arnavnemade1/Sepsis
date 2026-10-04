"use client";

import { Check, RotateCcw, X } from "lucide-react";
import { useState } from "react";

import { Button, CloseIcon, cx, Drawer, Empty, IconButton, Kv, Restricted, SectionLabel, Skeleton, Tag } from "@/components/ui";
import { api } from "@/lib/api";
import { useCaps, usePoll } from "@/lib/hooks";
import { ACK_TONE, ago, CHECK_TONE, fmtNum, proposerLabel, STATUS_LABEL, STATUS_TONE, type Check as CheckT, type MappingDetail, type MsgDetail } from "@/lib/interop";

import { EvidenceChart } from "./evidence-chart";

/* ------------------------------------------------------------------ */
/* One HL7 message: the raw ER7, what the engine did with each result   */
/* ------------------------------------------------------------------ */
function Er7({ raw }: { raw: string }) {
  const segs = raw.split(/\r\n|\r|\n/).filter(Boolean);
  return (
    <div className="max-h-72 overflow-auto rounded-md border border-line bg-bg p-2.5 font-mono text-[11.5px] leading-[1.55]">
      {segs.map((s, i) => {
        const name = s.slice(0, 3);
        const rest = s.slice(3);
        return (
          <div key={i} className="whitespace-pre-wrap break-all">
            <span className={cx("font-semibold", name === "MSH" ? "text-accent-ink" : name === "OBX" ? "text-ink" : "text-muted")}>{name}</span>
            {rest.split("|").map((f, j) => (
              <span key={j}>
                {j > 0 && <span className="text-faint">|</span>}
                <span className="text-muted">{f}</span>
              </span>
            ))}
          </div>
        );
      })}
    </div>
  );
}

export function MessageDrawer({ id, onClose, onMapping }: { id: string | null; onClose: () => void; onMapping: (id: string) => void }) {
  const { data: m } = usePoll<MsgDetail>(id ? `/aegis/interop/messages/${id}` : null);
  return (
    <Drawer open={!!id} onClose={onClose} width={640}>
      <div className="flex items-center gap-2 border-b border-line px-4 py-3">
        <div className="min-w-0 flex-1">
          <div className="text-[11.5px] text-dim">{m?.interface_name ?? "Message"}</div>
          <div className="truncate text-[14px] font-semibold text-ink">
            {m?.type ?? "…"} <span className="font-mono text-[12px] font-normal text-dim">{m?.control_id}</span>
          </div>
        </div>
        {m && (
          <Tag tone={ACK_TONE[m.ack]} dot>
            ACK {m.ack}
          </Tag>
        )}
        <IconButton label="Close" onClick={onClose}>
          <CloseIcon />
        </IconButton>
      </div>
      <div className="flex-1 space-y-4 overflow-y-auto p-4">
        {!m ? (
          <Skeleton className="h-64" />
        ) : (
          <>
            <p className="text-[12.5px] leading-relaxed text-muted">{m.summary}</p>
            <div className="grid grid-cols-2 gap-x-6 sm:grid-cols-3">
              <Kv k="Received" v={ago(m.at)} />
              <Kv k="Transport" v={m.transport.toUpperCase()} />
              <Kv k="Processed in" v={`${(m.latency_us / 1000).toFixed(2)} ms`} />
              <Kv k="HL7 version" v={m.version ?? "—"} />
              <Kv k="Bytes" v={m.bytes ?? "—"} />
              <Kv k="Resent" v={m.duplicate ? "yes, not reprocessed" : "no"} />
            </div>
            {m.error && (
              <div className="rounded-md border border-high/30 bg-high/[0.07] px-3 py-2 text-[12px] leading-relaxed text-muted">
                <span className="font-mono text-high-ink">ERR {m.error.code}</span> {m.error.text}
                {m.error.where && <span className="font-mono text-dim"> at {m.error.where}</span>}: {m.error.message}
              </div>
            )}
            <div>
              <SectionLabel>As received (ER7)</SectionLabel>
              <Er7 raw={m.raw} />
            </div>
            {m.outcomes.length > 0 && (
              <div>
                <SectionLabel>Each result, through its mapping</SectionLabel>
                <div className="overflow-x-auto rounded-md border border-line">
                  <table className="w-full min-w-[520px] text-[12px]">
                    <thead className="bg-s1 text-left text-[11px] text-dim">
                      <tr>
                        <th className="px-2.5 py-1.5 font-medium">Code</th>
                        <th className="px-2.5 py-1.5 font-medium">Sent</th>
                        <th className="px-2.5 py-1.5 font-medium">AegisOS read</th>
                        <th className="px-2.5 py-1.5 font-medium">Outcome</th>
                      </tr>
                    </thead>
                    <tbody>
                      {m.outcomes.map((o, i) => (
                        <tr key={i} className="border-t border-line">
                          <td className="px-2.5 py-1.5">
                            <button className="text-left hover:underline" onClick={() => o.mapping && onMapping(o.mapping)} disabled={!o.mapping}>
                              <div className="font-mono text-[11.5px] text-ink">{o.code}</div>
                              <div className="max-w-[180px] truncate text-[11px] text-dim">{o.text}</div>
                            </button>
                          </td>
                          <td className="num px-2.5 py-1.5 font-mono text-[11.5px] text-muted">
                            {o.value} {o.unit}
                          </td>
                          <td className="num px-2.5 py-1.5 font-mono text-[11.5px] text-ink">{o.concept ? `${o.concept} ${o.canonical != null ? fmtNum(o.canonical) : ""}` : "—"}</td>
                          <td className="px-2.5 py-1.5">
                            <Tag tone={o.status === "delivered" ? "low" : o.status.startsWith("withheld") ? "high" : "neutral"}>{o.status}</Tag>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </>
        )}
      </div>
    </Drawer>
  );
}

/* ------------------------------------------------------------------ */
/* One mapping: the claim, the evidence, the decision                   */
/* ------------------------------------------------------------------ */
function Checks({ items, empty }: { items: CheckT[]; empty?: string }) {
  if (!items.length) return <div className="text-[12px] text-dim">{empty ?? "None run."}</div>;
  return (
    <ul className="space-y-1.5">
      {items.map((c, i) => (
        <li key={i} className="flex items-start gap-2 text-[12px] leading-snug">
          <Tag tone={CHECK_TONE[c.status]} className="mt-px shrink-0">
            {c.status}
          </Tag>
          <span className="min-w-0 text-muted">
            <span className="font-mono text-[11px] text-dim">{c.name}</span> {c.detail}
          </span>
        </li>
      ))}
    </ul>
  );
}

export function MappingDrawer({ id, onClose }: { id: string | null; onClose: () => void }) {
  const { data: m, reload } = usePoll<MappingDetail>(id ? `/aegis/interop/mappings/${id}` : null, 3000);
  const { caps, role } = useCaps();
  const may = caps.has("interop:configure");
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const act = async (path: string, body?: unknown) => {
    setBusy(true);
    setMsg(null);
    try {
      const r = await api<{ status: string; detail?: string; against_evidence?: boolean }>(path, { body: body ?? {} });
      setMsg(r.status === "ok" ? (r.against_evidence ? "Adopted against the evidence; the record says so." : "Done.") : r.detail ?? "Refused.");
      reload();
    } catch (e) {
      setMsg(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };
  const viable = (m?.candidates ?? []).filter((c) => c.viable || c.conversion);
  const conv = m?.conversion ?? (m?.candidates?.[0]?.conversion ?? null);
  return (
    <Drawer open={!!id} onClose={onClose} width={680}>
      <div className="flex items-center gap-2 border-b border-line px-4 py-3">
        <div className="min-w-0 flex-1">
          <div className="truncate text-[11.5px] text-dim">
            {m?.interface} · <span className="font-mono">{m?.code}</span>
          </div>
          <div className="truncate text-[14px] font-semibold text-ink">
            {m?.text || m?.code} {m?.concept_info && <span className="font-normal text-muted">→ {m.concept_info.label}</span>}
          </div>
        </div>
        {m && (
          <Tag tone={STATUS_TONE[m.status]} dot>
            {STATUS_LABEL[m.status]}
          </Tag>
        )}
        <IconButton label="Close" onClick={onClose}>
          <CloseIcon />
        </IconButton>
      </div>
      <div className="flex-1 space-y-5 overflow-y-auto p-4">
        {!m ? (
          <Skeleton className="h-80" />
        ) : (
          <>
            <p className="text-[12.5px] leading-relaxed text-muted first-letter:uppercase">{m.reason}</p>
            <div className="grid grid-cols-2 gap-x-6">
              <Kv k="Unit written" v={m.unit_written || "none"} mono />
              <Kv k="Read as" v={m.conversion?.label ?? "—"} mono />
              <Kv k="Proposed by" v={proposerLabel(m.proposer)} />
              <Kv k="Concept (LOINC)" v={m.concept_info ? `${m.concept_info.label} · ${m.concept_info.loinc}` : "—"} />
              <Kv k="Values seen" v={m.counts.seen} />
              <Kv k="Delivered / withheld" v={`${m.counts.delivered + m.counts.backfilled} / ${m.counts.withheld}`} />
              <Kv k="Withdrawn" v={m.counts.retracted} />
              <Kv k="Live checks run" v={m.checks_run} />
            </div>

            {m.fault && (
              <div className="rounded-md border border-high/30 bg-high/[0.07] px-3 py-2.5 text-[12.5px] leading-relaxed">
                <div className="font-medium text-high-ink">Quarantined {ago(m.fault.at)}</div>
                <div className="mt-0.5 text-muted">{m.fault.cause}</div>
                <div className="mt-1 text-dim">
                  {m.fault.withdrawn} delivered value{m.fault.withdrawn === 1 ? "" : "s"} withdrawn from rooms.
                  {m.fault.suggested
                    ? ` The evidence's fix: read the values as ${m.fault.suggested.label}.`
                    : m.fault.swap
                      ? ` Read the other way round, the identities hold again: ${m.code} and ${m.fault.swap.code} look exchanged at the sender.`
                      : m.fault.by_identity
                        ? " An identity across fields failed and no exchange explains it, so every field in it is withheld."
                        : " No single reading of the values fits."}
                </div>
              </div>
            )}

            {m.concept_info && (
              <div>
                <SectionLabel right={conv ? `read as ${conv.label}` : undefined}>The values against the {m.concept_info.label.toLowerCase()} reference</SectionLabel>
                <div className="rounded-md border border-line bg-bg px-2 py-1.5">
                  <EvidenceChart m={m} factor={conv?.factor ?? 1} offset={conv?.offset ?? 0} />
                </div>
                <div className="mt-1 text-[11px] text-dim">
                  Blue: delivered. Red: not delivered (learning, withheld or quarantined). Reference: {m.reference?.source}.
                </div>
              </div>
            )}

            {m.evidence?.live && (
              <div>
                <SectionLabel right={ago(m.evidence.live.at)}>Live evidence (last {m.evidence.live.n} values)</SectionLabel>
                <Checks items={[...m.evidence.live.checks, ...m.evidence.live.invariants]} />
                {m.evidence.live.liveness && (
                  <div className="mt-1.5 text-[11.5px] text-dim">
                    Longest identical run {m.evidence.live.liveness.longest_identical_run} (limit {m.evidence.live.liveness.limit}).
                  </div>
                )}
              </div>
            )}

            <div>
              <SectionLabel>Proposals and the evidence for each</SectionLabel>
              {!m.candidates.length ? (
                <Empty>No proposals yet: the code is still collecting values.</Empty>
              ) : (
                <div className="space-y-2.5">
                  {m.candidates.map((c, i) => {
                    const fit = c.inference.declared ?? c.inference.best;
                    return (
                      <div key={i} className={cx("rounded-md border px-3 py-2.5", c.viable ? "border-low/30 bg-low/[0.05]" : "border-line bg-s1")}>
                        <div className="flex flex-wrap items-center gap-2">
                          <span className="text-[13px] font-medium text-ink">{c.label}</span>
                          <Tag tone={c.viable ? "low" : c.decision === "insufficient" ? "neutral" : "high"}>{c.viable ? `fits (${c.decision})` : c.decision}</Tag>
                          <span className="ml-auto text-[11.5px] text-dim">
                            {proposerLabel(c.proposal.proposer)} · {Math.round(c.proposal.score * 100)}%
                          </span>
                        </div>
                        <div className="mt-1 text-[11.5px] text-dim">{c.proposal.rationale}</div>
                        {fit && (
                          <div className="mt-2">
                            <Checks items={[...fit.checks, ...c.joint]} />
                          </div>
                        )}
                        {may && (m.status === "review" || m.status === "unmapped") && (
                          <div className="mt-2 flex flex-wrap gap-2">
                            {c.conversion && (
                              <Button size="sm" variant={c.viable ? "success" : "ghost"} disabled={busy} onClick={() => act(`/aegis/interop/mappings/${m.id}/adopt`, { concept: c.concept, conversion: c.conversion })}>
                                <Check size={13} /> Adopt as {c.label} ({c.conversion.label})
                              </Button>
                            )}
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>

            <div>
              <SectionLabel>Decide</SectionLabel>
              {!may ? (
                <Restricted cap="interop:configure" role={role}>
                  Deciding what a local code means changes what every detector reads, so it is the chief&apos;s. The evidence above is visible to everyone who can read the feed.
                </Restricted>
              ) : (
                <div className="flex flex-wrap gap-2">
                  {m.status === "quarantined" && m.fault?.suggested && (
                    <Button variant="success" disabled={busy} onClick={() => act(`/aegis/interop/mappings/${m.id}/adopt`, { conversion: "suggested" })}>
                      <Check size={13} /> Accept the fix ({m.fault.suggested.label})
                    </Button>
                  )}
                  {m.status === "quarantined" && m.fault?.swap && (
                    <Button variant="success" disabled={busy} onClick={() => act(`/aegis/interop/mappings/${m.id}/adopt`, { conversion: "swap" })}>
                      <Check size={13} /> Accept the exchange ({m.code} ↔ {m.fault.swap.code})
                    </Button>
                  )}
                  {m.status === "quarantined" && (
                    <Button disabled={busy} onClick={() => act(`/aegis/interop/mappings/${m.id}/restore`)}>
                      <RotateCcw size={13} /> Restore (the sender fixed it)
                    </Button>
                  )}
                  {(m.status === "review" || m.status === "unmapped") && !viable.length && <span className="text-[12px] text-dim">No proposal the data supports. Map it by hand through the API, or reject it.</span>}
                  {m.status !== "rejected" && m.status !== "adopted" && (
                    <Button variant="danger" disabled={busy} onClick={() => act(`/aegis/interop/mappings/${m.id}/reject`, { reason: "rejected from the console" })}>
                      <X size={13} /> Reject
                    </Button>
                  )}
                  <Button variant="quiet" disabled={busy} onClick={() => act(`/aegis/interop/mappings/${m.id}/reevaluate`)}>
                    Re-run the evidence
                  </Button>
                </div>
              )}
              {msg && <div className="mt-2 text-[12px] text-muted">{msg}</div>}
            </div>

            <div>
              <SectionLabel>History</SectionLabel>
              <ol className="space-y-1.5 border-l border-line pl-3">
                {[...m.history].reverse().map((h, i) => (
                  <li key={i} className="text-[12px] leading-snug">
                    <span className="text-dim">{ago(h.at)}</span> <span className="text-ink">{STATUS_LABEL[h.to] ?? h.to}</span> <span className="text-dim">· {h.by}</span>
                    <div className="text-muted first-letter:uppercase">{h.why}</div>
                  </li>
                ))}
              </ol>
            </div>
          </>
        )}
      </div>
    </Drawer>
  );
}
