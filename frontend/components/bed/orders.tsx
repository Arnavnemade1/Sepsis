"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { Button, cx, Empty, Pill } from "@/components/ui";
import { api } from "@/lib/api";
import { ago } from "@/lib/clinical";
import { useSession } from "@/lib/hooks";

type Result = {
  sid: number | null;
  syscall: string;
  bed: string;
  allowed: boolean;
  errno_name?: string;
  z3_status: string;
  message: string;
  violations?: string[];
  result?: { proof?: string; band?: number; target_severity?: number; pin?: number | null };
  projection?: { href: string };
};
type AuditRow = { sid: number; syscall: string; bed: string; allowed: boolean; message: string; ts: number; z3_status: string };

/**
 * The write side of the bedside: the four syscalls a clinician can issue on a
 * bed. Each one is applied to the running kernel only if the Z3 egress cage
 * certifies it; a refusal changes nothing and comes back with the reason.
 *
 * The buttons mirror the RBAC policy (backend/aegis/rbac.py): a role that lacks
 * the capability sees why, instead of a button. The gate is still the
 * authority — in `audit` mode it records such a call as one it would refuse.
 */
const ACTIONS: { syscall: string; cap: string; label: string; hint: string; who: string; tone?: "primary" }[] = [
  { syscall: "treat", cap: "order:treat", label: "Order the certified bundle", hint: "Dispatches the Hour-1 bundle only if Z3 proves it safe for this patient", who: "attending", tone: "primary" },
  { syscall: "escalate", cap: "order:escalate", label: "Escalate", hint: "Raises monitoring priority one band; denied if the risk does not justify it", who: "attending · chief" },
  { syscall: "clear", cap: "order:clear", label: "Clear interventions", hint: "Releases the priority pin and clears this bed's intervention state", who: "attending" },
  { syscall: "crisis", cap: "drill:crisis", label: "Crisis drill", hint: "Injects a simulated deterioration into this bed's feed (training)", who: "attending" },
];

export function Orders({ bed }: { bed: string }) {
  const session = useSession();
  const caps = new Set(session?.credential.capabilities ?? []);
  const [busy, setBusy] = useState<string | null>(null);
  const [last, setLast] = useState<Result | null>(null);
  const [log, setLog] = useState<AuditRow[]>([]);

  const loadLog = useCallback(async () => {
    try {
      const r = await api<{ log: AuditRow[] }>("/aegis/syscall/audit?limit=64");
      setLog(r.log.filter((x) => x.bed === bed).slice(-5).reverse());
    } catch {}
  }, [bed]);
  useEffect(() => {
    setLast(null);
    loadLog();
  }, [bed, loadLog]);

  const run = async (syscall: string) => {
    setBusy(syscall);
    try {
      const r = await api<Result>("/aegis/syscall", { body: { syscall, bed, args: {} } });
      setLast(r);
    } catch (e) {
      setLast({ sid: null, syscall, bed, allowed: false, z3_status: "n/a", message: e instanceof Error ? e.message : String(e) });
    } finally {
      setBusy(null);
      loadLog();
    }
  };

  return (
    <div>
      {!session && <div className="mb-3 text-[12px] text-dim">Sign in as a clinician to issue orders.</div>}
      <div className="grid gap-2">
        {ACTIONS.map((a) => {
          const can = caps.has(a.cap);
          return (
            <div key={a.syscall} className="flex items-center gap-3">
              <Button
                variant={a.tone === "primary" ? "primary" : "ghost"}
                disabled={!can || busy !== null}
                onClick={() => run(a.syscall)}
                className="w-[210px] shrink-0 justify-start"
              >
                {busy === a.syscall ? "Proving…" : a.label}
              </Button>
              <span className="min-w-0 text-[11.5px] leading-snug text-dim">
                {can ? a.hint : `Requires ${a.cap} (${a.who})`}
              </span>
            </div>
          );
        })}
      </div>

      {last && (
        <div
          className={cx(
            "mt-4 rounded-md border p-3 text-[12.5px]",
            last.allowed ? "border-line bg-s2/60" : "border-high/30 bg-high/10",
          )}
        >
          <div className="flex items-center gap-2">
            <Pill tone={last.allowed ? "neutral" : "high"} dot>
              {last.allowed ? "Certified and applied" : last.errno_name === "EPERM" && !last.sid ? "Not permitted" : "Refused by the prover"}
            </Pill>
            <span className="font-mono text-[11px] text-dim">
              {last.syscall} · z3 {last.z3_status}
              {last.sid != null && ` · sid ${last.sid}`}
            </span>
          </div>
          <div className={cx("mt-2", last.allowed ? "text-ink" : "text-high-ink")}>{last.message}</div>
          {last.violations && last.violations.length > 0 && <div className="mt-1 text-muted">{last.violations.join(" · ")}</div>}
          {last.result?.proof && (
            <details className="mt-2">
              <summary className="cursor-pointer text-[12px] text-dim hover:text-ink">Proof</summary>
              <p className="mt-1 break-words font-mono text-[11px] leading-relaxed text-muted">{last.result.proof}</p>
            </details>
          )}
          {last.allowed && last.syscall === "treat" && (
            <Link href={`/pharmacy/?id=${encodeURIComponent(bed)}`} className="mt-2 inline-block text-[12px] text-accent-ink hover:underline">
              See where this order goes in the body →
            </Link>
          )}
        </div>
      )}

      <div className="mt-4 border-t border-line-soft pt-3">
        <div className="mb-1.5 text-[11.5px] text-dim">Audit log for this bed</div>
        {log.length === 0 ? (
          <Empty>No orders on this bed yet.</Empty>
        ) : (
          <ul className="space-y-1">
            {log.map((r) => (
              <li key={r.sid} className="flex items-baseline gap-2 text-[12px]">
                <span className={cx("h-1.5 w-1.5 shrink-0 rounded-full", r.allowed ? "bg-faint" : "bg-high")} />
                <span className="font-mono text-[11.5px] text-ink">{r.syscall}</span>
                <span className="min-w-0 truncate text-muted">{r.message}</span>
                <span className="ml-auto shrink-0 text-[11px] text-dim">
                  z3 {r.z3_status} · {ago(r.ts)}
                </span>
              </li>
            ))}
          </ul>
        )}
      </div>
    </div>
  );
}
