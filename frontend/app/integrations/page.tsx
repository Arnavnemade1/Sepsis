"use client";

import { useRouter, useSearchParams } from "next/navigation";
import { Suspense, useState } from "react";

import { MappingDrawer, MessageDrawer } from "@/components/integrations/drawers";
import { EndpointsView } from "@/components/integrations/endpoints";
import { MappingsView } from "@/components/integrations/mappings";
import { IntegrationsOverview } from "@/components/integrations/overview";
import { Page, PageHead } from "@/components/shell";
import { Segmented, Tag } from "@/components/ui";
import { usePoll } from "@/lib/hooks";
import type { Overview } from "@/lib/interop";

type View = "overview" | "mappings" | "endpoints";
const VIEWS: View[] = ["overview", "mappings", "endpoints"];

export default function IntegrationsPage() {
  return (
    <Suspense>
      <Integrations />
    </Suspense>
  );
}

function Integrations() {
  const params = useSearchParams();
  const router = useRouter();
  const raw = params.get("view") ?? "overview";
  const view: View = (VIEWS as string[]).includes(raw) ? (raw as View) : "overview";
  const [msg, setMsg] = useState<string | null>(null);
  const [mapping, setMapping] = useState<string | null>(params.get("mapping"));
  const { data: ov } = usePoll<Overview>("/aegis/interop", 5000);
  const q = ov?.mappings.by_status.quarantined ?? 0;
  const r = ov?.mappings.by_status.review ?? 0;
  return (
    <Page active="integrations">
      <PageHead
        title="Integrations"
        right={
          ov && (
            <>
              <Tag tone={q ? "high" : "low"} dot>
                {q ? `${q} feed${q === 1 ? "" : "s"} quarantined` : "Every adopted feed passes"}
              </Tag>
              {r > 0 && <Tag tone="med">{r} waiting for review</Tag>}
            </>
          )
        }
      >
        Every system that feeds the hospital&apos;s rooms (admissions, monitors, laboratories) speaks HL7 v2 or FHIR in its own dialect. Each local code is mapped onto an AegisOS concept only when its own data agrees, and keeps being checked: a feed that drifts is quarantined before a wrong value reaches a detector.
      </PageHead>
      <div className="mb-3">
        <Segmented
          value={view}
          onChange={(v) => router.replace(v === "overview" ? "/integrations/" : `/integrations/?view=${v}`)}
          options={[
            { value: "overview", label: "Overview" },
            { value: "mappings", label: `Mappings${ov ? ` ${ov.mappings.total}` : ""}` },
            { value: "endpoints", label: "Standards & endpoints" },
          ]}
        />
      </div>
      {view === "overview" && <IntegrationsOverview onMessage={setMsg} onMapping={setMapping} />}
      {view === "mappings" && <MappingsView onMapping={setMapping} />}
      {view === "endpoints" && <EndpointsView />}
      <MessageDrawer id={msg} onClose={() => setMsg(null)} onMapping={(id) => { setMsg(null); setMapping(id); }} />
      <MappingDrawer id={mapping} onClose={() => setMapping(null)} />
    </Page>
  );
}
