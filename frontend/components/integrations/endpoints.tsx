"use client";

import { useState } from "react";

import { Send, Webhook } from "@/components/icons";
import { Button, cx, Empty, Kv, Panel, Restricted, Skeleton, Tag } from "@/components/ui";
import { api, backendBase } from "@/lib/api";
import { useCaps, usePoll } from "@/lib/hooks";
import { ACK_TONE, ago, type CdsCard, type Webhooks } from "@/lib/interop";

const SAMPLE = [
  "MSH|^~\\&|LABTEST|DEMOSITE|AEGISOS|AEGISMEM|20261004120000||ORU^R01^ORU_R01|TEST0001|P|2.5.1",
  "PID|1||E000000^^^DEMOSITE^MR||Test^Pat",
  "OBX|1|NM|2823-3^Potassium^LN||4.2|mmol/L|||||F",
].join("\r");

const IND_TONE = { info: "neutral", warning: "med", critical: "high" } as const;

export function EndpointsView() {
  const { caps, role } = useCaps();
  const base = backendBase();
  return (
    <div className="grid grid-cols-1 gap-4 xl:grid-cols-2">
      <div className="min-w-0 space-y-3">
        <Hl7Console canSend={caps.has("interop:ingest")} role={role} base={base} />
        <FhirPanel base={base} />
      </div>
      <div className="min-w-0 space-y-3">
        <CdsPanel base={base} />
        <WebhooksPanel canConfigure={caps.has("interop:configure")} role={role} />
      </div>
    </div>
  );
}

function Code({ children }: { children: string }) {
  return <code className="break-all rounded bg-bg px-1 py-px font-mono text-xs text-ink">{children}</code>;
}

function Hl7Console({ canSend, role, base }: { canSend: boolean; role: string | null; base: string }) {
  const [text, setText] = useState(SAMPLE.replace(/\r/g, "\n"));
  const [res, setRes] = useState<{ ack: string; record: { ack: string; summary: string; error: { code: number; message: string } | null } } | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const send = async () => {
    setErr(null);
    try {
      setRes(await api("/aegis/interop/hl7", { body: { message: text.replace(/\n/g, "\r") } }));
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };
  return (
    <Panel title="HL7 v2 inbound" meta="ADT, ORU^R01 results and ORU^R40 device alarms; every message acknowledged">
      <div className="space-y-2.5 text-sm leading-relaxed text-muted">
        <div className="grid grid-cols-1 gap-x-6 sm:grid-cols-2">
          <Kv k="HTTP" v={<Code>{`POST ${base}/aegis/interop/hl7`}</Code>} />
          <Kv k="MLLP (on premises)" v={<Code>AEGIS_MLLP_PORT</Code>} />
          <Kv k="Body" v="ER7 text, or JSON {message}" />
          <Kv k="Capability" v={<Code>interop:ingest</Code>} />
        </div>
        <p>
          The ACK carries the HL7 table-0357 code when a message is refused. A resent control id is answered with the same ACK and not processed again. An unknown sender is registered and its codes are learned from its traffic.
        </p>
        {!canSend && (
          <Restricted cap="interop:ingest" role={role}>
            Sending HL7 is the interface engine&apos;s (service account <span className="font-mono">svc.hl7</span>).
          </Restricted>
        )}
        <textarea value={text} onChange={(e) => setText(e.target.value)} rows={5} spellCheck={false} className="w-full field p-2.5 font-mono text-xs leading-[1.55] text-ink outline-none focus:border-line-bright" aria-label="HL7 message" />
        <div className="flex items-center gap-2">
          <Button size="sm" disabled={!canSend} onClick={send}>
            <Send size={13} /> Send
          </Button>
          {res && (
            <>
              <Tag tone={ACK_TONE[res.record.ack]} dot>
                ACK {res.record.ack}
              </Tag>
              <span className="min-w-0 truncate text-xs text-muted">{res.record.error ? `ERR ${res.record.error.code}: ${res.record.error.message}` : res.record.summary}</span>
            </>
          )}
          {err && <span className="text-xs text-high-ink">{err}</span>}
        </div>
        {res && <pre className="max-h-32 overflow-auto well p-2.5 font-mono text-xs text-muted">{res.ack.replace(/\r/g, "\n")}</pre>}
      </div>
    </Panel>
  );
}

function FhirPanel({ base }: { base: string }) {
  const { data: cap } = usePoll<{ fhirVersion: string; rest: { resource: { type: string; interaction: { code: string }[]; searchParam: { name: string }[] }[] }[] }>("/aegis/fhir/R4/metadata");
  return (
    <Panel title="FHIR R4" meta="What AegisOS sees, in the resources an EHR already reads">
      <div className="space-y-2.5 text-sm leading-relaxed text-muted">
        <Kv k="Base URL" v={<Code>{`${base}/aegis/fhir/R4`}</Code>} />
        <p>
          A read facade over the live rooms, plus inbound Observations. A posted Observation or transaction Bundle goes through the same verified mapping path as an HL7 result. Synthetic demo patients carry the <span className="font-mono">HTEST</span> security label.
        </p>
        {!cap ? (
          <Skeleton className="h-28" />
        ) : (
          <div className="overflow-x-auto rounded-md border border-line">
            <table className="w-full min-w-[420px] text-xs">
              <thead className="bg-s1 text-left text-xs text-dim">
                <tr>
                  <th className="px-2.5 py-1.5 font-medium">Resource</th>
                  <th className="px-2.5 py-1.5 font-medium">Interactions</th>
                  <th className="px-2.5 py-1.5 font-medium">Search by</th>
                </tr>
              </thead>
              <tbody>
                {cap.rest[0].resource.map((r) => (
                  <tr key={r.type} className="border-t border-line">
                    <td className="px-2.5 py-1.5 font-mono text-xs text-ink">{r.type}</td>
                    <td className="px-2.5 py-1.5 text-muted">{r.interaction.map((i) => i.code).join(", ")}</td>
                    <td className="px-2.5 py-1.5 font-mono text-xs text-dim">{r.searchParam.map((p) => p.name).join(" ")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
        <p className="text-xs text-dim">
          Example: <Code>{`GET ${base}/aegis/fhir/R4/Observation?patient=room-512&category=laboratory`}</Code>. A read facade, not a conformant FHIR server: no history, no _include, no conditional writes.
        </p>
      </div>
    </Panel>
  );
}

function CdsPanel({ base }: { base: string }) {
  const { data: disc } = usePoll<{ services: { id: string; hook: string; title: string; description: string }[] }>("/aegis/cds-services");
  const { data: board } = usePoll<{ rooms: { room: string; name: string; kind: string | null }[] }>("/aegis/rooms", 15000);
  const [room, setRoom] = useState("");
  const [cards, setCards] = useState<CdsCard[] | null>(null);
  const [drug, setDrug] = useState("levofloxacin 750 MG IV");
  const occupied = (board?.rooms ?? []).filter((r) => r.kind);
  const call = async (service: string) => {
    if (!room) return;
    const body: Record<string, unknown> = {
      hook: service === "aegis-patient-view" ? "patient-view" : "order-sign",
      hookInstance: crypto.randomUUID?.() ?? String(Date.now()),
      context: { userId: "Practitioner/dr.alvarez", patientId: `room-${room}` },
    };
    if (service === "aegis-order-sign") {
      (body.context as Record<string, unknown>).draftOrders = {
        resourceType: "Bundle",
        entry: [{ resource: { resourceType: "MedicationRequest", id: "draft-1", status: "draft", intent: "order", medicationCodeableConcept: { text: drug } } }],
      };
    }
    setCards(null);
    try {
      const r = await api<{ cards: CdsCard[] }>(`/aegis/cds-services/${service}`, { body });
      setCards(r.cards ?? []);
    } catch {
      setCards([]);
    }
  };
  return (
    <Panel title="CDS Hooks 2.0" meta="AegisOS's findings inside the EHR, when a chart opens or an order is signed">
      <div className="space-y-2.5 text-sm leading-relaxed text-muted">
        <Kv k="Discovery" v={<Code>{`GET ${base}/aegis/cds-services`}</Code>} />
        <ul className="space-y-1.5">
          {(disc?.services ?? []).map((s) => (
            <li key={s.id}>
              <span className="font-mono text-xs text-ink">{s.hook}</span> <span className="text-dim">· {s.id}</span>
              <div className="text-xs text-muted">{s.description}</div>
            </li>
          ))}
        </ul>
        <div className="flex flex-wrap items-center gap-2 border-t border-line pt-2.5">
          <select value={room} onChange={(e) => setRoom(e.target.value)} className="h-7 field px-2 text-xs text-ink" aria-label="Room">
            <option value="">Choose a room</option>
            {occupied.map((r) => (
              <option key={r.room} value={r.room}>
                {r.room} · {r.name}
              </option>
            ))}
          </select>
          <Button size="sm" disabled={!room} onClick={() => call("aegis-patient-view")}>
            Open the chart (patient-view)
          </Button>
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <input value={drug} onChange={(e) => setDrug(e.target.value)} className="h-7 min-w-0 flex-1 field px-2 text-xs text-ink" aria-label="Draft medication order" />
          <Button size="sm" disabled={!room} onClick={() => call("aegis-order-sign")}>
            Sign this order (order-sign)
          </Button>
        </div>
        {cards && (
          <div className="space-y-2">
            {!cards.length && <Empty>No cards: nothing to flag. A CDS service answers with an empty list when it has nothing to say.</Empty>}
            {cards.map((c) => (
              <div key={c.uuid} className={cx("rounded-md border-l-2 bg-s1 px-3 py-2", c.indicator === "critical" ? "border-high" : c.indicator === "warning" ? "border-med" : "border-line-bright")}>
                <div className="flex items-center gap-2">
                  <Tag tone={IND_TONE[c.indicator]}>{c.indicator}</Tag>
                  <span className="text-sm font-medium text-ink">{c.summary}</span>
                </div>
                {c.detail && <div className="mt-1 whitespace-pre-line text-xs text-muted">{c.detail.replace(/\*\*/g, "").replace(/_/g, "")}</div>}
                {c.suggestions?.map((s) => (
                  <div key={s.uuid} className="mt-1.5 text-xs text-low-ink">
                    Suggestion: {s.label}
                  </div>
                ))}
                <div className="mt-1 text-2xs text-dim">{c.source.label}</div>
              </div>
            ))}
          </div>
        )}
      </div>
    </Panel>
  );
}

function WebhooksPanel({ canConfigure, role }: { canConfigure: boolean; role: string | null }) {
  const { data, reload } = usePoll<Webhooks>("/aegis/interop/webhooks", 3000);
  const [url, setUrl] = useState("internal://echo");
  const [secret, setSecret] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const subscribe = async () => {
    setErr(null);
    try {
      const r = await api<{ status: string; detail?: string; subscription?: { secret: string } }>("/aegis/interop/webhooks", { body: { url, events: [] } });
      if (r.status !== "ok") setErr(r.detail ?? "refused");
      else setSecret(r.subscription?.secret ?? null);
      reload();
    } catch (e) {
      setErr(e instanceof Error ? e.message : String(e));
    }
  };
  const act = async (path: string) => {
    await api(path, { body: {} });
    reload();
  };
  return (
    <Panel title="Outbound webhooks" meta="Pages and feed events to the hospital's other systems, signed (Standard Webhooks)">
      {!data ? (
        <Skeleton className="h-40" />
      ) : (
        <div className="space-y-3 text-sm leading-relaxed text-muted">
          <p>
            Each delivery carries <span className="font-mono text-xs">webhook-id</span>, <span className="font-mono text-xs">webhook-timestamp</span> and an HMAC-SHA256 <span className="font-mono text-xs">webhook-signature</span>. Failed deliveries retry with backoff (5 s to 30 min) and then go to the dead-letter list. Private and loopback addresses are refused; <span className="font-mono text-xs">internal://echo</span> is a built-in receiver that verifies each signature.
          </p>
          {canConfigure ? (
            <div className="flex flex-wrap items-center gap-2">
              <input value={url} onChange={(e) => setUrl(e.target.value)} className="h-7 min-w-0 flex-1 field px-2 font-mono text-xs text-ink" aria-label="Webhook URL" />
              <Button size="sm" onClick={subscribe}>
                <Webhook size={13} /> Subscribe
              </Button>
            </div>
          ) : (
            <Restricted cap="interop:configure" role={role} />
          )}
          {err && <div className="text-xs text-high-ink">{err}</div>}
          {secret && (
            <div className="rounded-md bg-med/[0.08] px-3 py-2 text-xs">
              Signing secret, shown once: <span className="break-all font-mono text-xs text-ink">{secret}</span>
            </div>
          )}
          {data.subscriptions.length > 0 && (
            <ul className="divide-y divide-line rounded-md border border-line">
              {data.subscriptions.map((s) => (
                <li key={s.id} className="flex flex-wrap items-center gap-2 px-3 py-2">
                  <span className="min-w-0 flex-1 truncate font-mono text-xs text-ink">{s.url}</span>
                  <span className="text-xs text-dim">
                    {s.delivered} delivered · {s.failed} dead
                  </span>
                  {canConfigure && (
                    <>
                      <Button size="sm" variant="quiet" onClick={() => act(`/aegis/interop/webhooks/${s.id}/test`)}>
                        Ping
                      </Button>
                      <Button size="sm" variant="quiet" onClick={() => act(`/aegis/interop/webhooks/${s.id}/delete`)}>
                        Remove
                      </Button>
                    </>
                  )}
                </li>
              ))}
            </ul>
          )}
          <div>
            <div className="mb-1 text-xs font-medium text-dim">
              Recent deliveries · {data.counters.delivered ?? 0} delivered, {data.counters.retries ?? 0} retries, {data.counters.dead ?? 0} dead · {data.queued} queued
            </div>
            {!data.recent.length ? (
              <div className="text-xs text-dim">No deliveries yet. Subscribe, then raise a page or run a drill.</div>
            ) : (
              <ul className="max-h-48 space-y-1 overflow-auto">
                {data.recent.map((d) => (
                  <li key={d.id} className="flex items-center gap-2 text-xs">
                    <Tag tone={d.status === "delivered" ? "low" : d.status === "dead" ? "high" : "med"}>{d.last_code ?? "—"}</Tag>
                    <span className="font-mono text-xs text-ink">{d.event}</span>
                    <span className="ml-auto text-xs text-dim">
                      {d.latency_ms != null ? `${d.latency_ms} ms · ` : ""}
                      {ago(d.delivered_at ?? d.created_at)}
                    </span>
                  </li>
                ))}
              </ul>
            )}
          </div>
          {data.echo.length > 0 && (
            <div>
              <div className="mb-1 text-xs font-medium text-dim">Built-in receiver (internal://echo)</div>
              <ul className="max-h-32 space-y-1 overflow-auto">
                {data.echo.map((e) => (
                  <li key={e.id} className="flex items-center gap-2 text-xs">
                    <Tag tone={e.verified ? "low" : "high"}>{e.verified ? "signature ok" : "bad signature"}</Tag>
                    <span className="font-mono text-xs text-ink">{e.type}</span>
                    <span className="ml-auto text-xs text-dim">{ago(e.at)}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </Panel>
  );
}
