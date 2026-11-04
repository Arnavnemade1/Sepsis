"use client";

import { useEffect, useRef, useState, type KeyboardEvent } from "react";

import { api, readSession } from "@/lib/api";
import { useSession } from "@/lib/hooks";
import { clock } from "@/lib/paging";

/**
 * A shell on the AegisOS kernel. Every verb is a real syscall through the gate
 * (the same REST surface the workspaces read), so what you can `cat` here is
 * exactly what your role may read — a refusal is the kernel's, not the UI's.
 */
type Line = { k: "in" | "out" | "err" | "dim"; t: string };

type FsNode = { path: string; name: string; kind: string; size: number; mode: string; note?: string };

const HELP = `AegisOS shell — every command is a syscall through the gate.

  ls [path]        list a directory          cat <path>     read a file
  cd <path>        change directory          stat <path>    file metadata
  pwd              print working directory   uname -a       the running kernel
  ps               process table             top            processes by CPU
  dmesg [n] [fac]  kernel journal tail       boot           boot record
  whoami / who     identity, sessions        gate           syscall gate status
  abi [filter]     the syscall table         clear          clear the screen

  Try:  ls /proc   ·   cat /etc/os-release   ·   ls /patients   ·   dmesg 20 clinical`;

function norm(cwd: string, p?: string): string {
  if (!p) return cwd;
  const parts = (p.startsWith("/") ? p : `${cwd}/${p}`).split("/");
  const out: string[] = [];
  for (const s of parts) {
    if (!s || s === ".") continue;
    if (s === "..") out.pop();
    else out.push(s);
  }
  return "/" + out.join("/");
}

/** The kernel answers a refused or missing path with {status:"error", errno, detail}. */
function fsErr(r: { status?: string; errno?: string; detail?: string }): string | null {
  return r.status === "error" ? `${r.errno ?? "EIO"}: ${r.detail ?? "error"}` : null;
}

function pad(s: string | number, n: number): string {
  const v = String(s);
  return v.length >= n ? v.slice(0, n) : v + " ".repeat(n - v.length);
}

export function Terminal({ className = "h-[440px]", greet = false }: { className?: string; greet?: boolean }) {
  const session = useSession();
  // a full-page shell opens on the command reference rather than an empty screen
  const [lines, setLines] = useState<Line[]>(() =>
    greet ? HELP.split("\n").map((t) => ({ k: "dim" as const, t })) : [{ k: "dim", t: 'AegisOS shell. Type "help".' }],
  );
  const [cwd, setCwd] = useState("/");
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const hist = useRef<string[]>([]);
  const hi = useRef(-1);
  const box = useRef<HTMLDivElement>(null);
  const field = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (box.current) box.current.scrollTop = box.current.scrollHeight;
  }, [lines]);

  const print = (t: string, k: Line["k"] = "out") => setLines((l) => [...l, ...t.split("\n").map((x) => ({ k, t: x }))]);
  // other panels can type a command in: window.dispatchEvent(new CustomEvent("aegis:shell", { detail: "ps" }))
  const runRef = useRef<(cmd: string) => void>(() => {});
  useEffect(() => {
    const h = (e: Event) => runRef.current(String((e as CustomEvent).detail ?? ""));
    window.addEventListener("aegis:shell", h);
    return () => window.removeEventListener("aegis:shell", h);
  }, []);

  const run = async (raw: string) => {
    const cmd = raw.trim();
    const user = readSession()?.credential.username ?? "guest";
    setLines((l) => [...l, { k: "in", t: `${user}@aegis:${cwd}$ ${cmd}` }]);
    if (!cmd) return;
    hist.current = [cmd, ...hist.current.slice(0, 99)];
    hi.current = -1;
    const [verb, ...args] = cmd.split(/\s+/);
    setBusy(true);
    try {
      switch (verb) {
        case "help":
          print(HELP);
          break;
        case "clear":
          setLines([]);
          break;
        case "pwd":
          print(cwd);
          break;
        case "cd": {
          const p = norm(cwd, args[0] ?? "/");
          const st = await api<{ kind?: string; status?: string; errno?: string; detail?: string }>(`/aegis/fs/stat?path=${encodeURIComponent(p)}`);
          if (fsErr(st)) print(`cd: ${fsErr(st)}`, "err");
          else if (st.kind !== "dir") print(`cd: ${p}: not a directory`, "err");
          else setCwd(p);
          break;
        }
        case "ls": {
          const p = norm(cwd, args[0]);
          const r = await api<{ entries?: FsNode[]; status?: string; errno?: string; detail?: string }>(`/aegis/fs/ls?path=${encodeURIComponent(p)}`);
          if (fsErr(r)) print(`ls: ${fsErr(r)}`, "err");
          else
            print(
              (r.entries ?? [])
                .map((e) => `${pad(e.mode, 5)} ${pad(e.kind === "dir" ? e.name + "/" : e.name, 26)} ${e.note ?? ""}`)
                .join("\n") || "(empty)",
            );
          break;
        }
        case "cat": {
          if (!args[0]) {
            print("cat: missing path", "err");
            break;
          }
          const p = norm(cwd, args[0]);
          const r = await api<{ content?: unknown; kind?: string; status?: string; errno?: string; detail?: string }>(`/aegis/fs/read?path=${encodeURIComponent(p)}`);
          if (fsErr(r)) print(`cat: ${fsErr(r)}`, "err");
          else print(typeof r.content === "string" ? r.content.replace(/\n$/, "") : JSON.stringify(r.content, null, 2));
          break;
        }
        case "stat": {
          const p = norm(cwd, args[0]);
          const r = await api<Record<string, unknown> & { status?: string; errno?: string; detail?: string }>(`/aegis/fs/stat?path=${encodeURIComponent(p)}`);
          if (fsErr(r)) print(`stat: ${fsErr(r)}`, "err");
          else print(JSON.stringify(r, null, 2));
          break;
        }
        case "uname": {
          const r = await api<{ content?: string }>(`/aegis/fs/read?path=/etc/os-release`);
          const kv = Object.fromEntries(
            (r.content ?? "").split("\n").filter(Boolean).map((l) => {
              const i = l.indexOf("=");
              return [l.slice(0, i), l.slice(i + 1).replace(/^"|"$/g, "")];
            }),
          );
          print(args.includes("-a") ? `${kv.PRETTY_NAME} · ${kv.KERNEL} · boot ${kv.BOOT_ID} · data ${kv.DATA_MODE} · gate ${kv.GATE_MODE}` : kv.NAME ?? "AegisOS");
          break;
        }
        case "ps":
        case "top": {
          const r = await api<{ processes: { pid: number; name: string; kind: string; uid: string; state: string; cpu_ms: number; duty: number }[] }>(
            "/aegis/sys/ps",
          );
          const ps = verb === "top" ? [...r.processes].sort((a, b) => b.cpu_ms - a.cpu_ms).slice(0, 15) : r.processes;
          print(`${pad("PID", 5)} ${pad("NAME", 24)} ${pad("KIND", 9)} ${pad("UID", 10)} ${pad("STATE", 9)} ${pad("CPU ms", 9)}`, "dim");
          print(ps.map((p) => `${pad(p.pid, 5)} ${pad(p.name, 24)} ${pad(p.kind, 9)} ${pad(p.uid, 10)} ${pad(p.state, 9)} ${pad(p.cpu_ms.toFixed(1), 9)}`).join("\n"));
          break;
        }
        case "dmesg":
        case "journal": {
          const n = Number(args.find((a) => /^\d+$/.test(a)) ?? 30);
          const fac = args.find((a) => !/^\d+$/.test(a)) ?? "";
          const r = await api<{ entries: { seq: number; ts: number; facility: string; severity_name: string; event: string; uid: string; detail: unknown }[] }>(
            `/aegis/sys/journal?limit=${n}${fac ? `&facility=${encodeURIComponent(fac)}` : ""}`,
          );
          print(
            r.entries
              .map((e) => `[${clock(e.ts)}] ${pad(e.facility, 8)} ${pad(e.severity_name, 7)} ${pad(e.event, 18)} ${JSON.stringify(e.detail).slice(0, 90)}`)
              .join("\n") || "(no entries)",
          );
          break;
        }
        case "boot": {
          const r = await api<{ boot_seconds: number; phases: number; failed: unknown[]; log: { phase: string; summary: string; ms: number; ok: boolean }[] }>("/aegis/sys/boot");
          print(`booted in ${r.boot_seconds}s · ${r.phases} phases · ${r.failed.length} failed`, "dim");
          print(r.log.map((p) => `${p.ok ? "  ok " : "FAIL "} ${pad(p.phase, 12)} ${pad(`${p.ms.toFixed(1)} ms`, 11)} ${p.summary}`).join("\n"));
          break;
        }
        case "whoami": {
          const r = await api<{ credential: { username: string; role: string; ring: number; capabilities: string[] }; gate: string }>("/aegis/sys/whoami");
          print(`${r.credential.username} · role ${r.credential.role} · ring ${r.credential.ring} · gate ${r.gate}\ncapabilities: ${r.credential.capabilities.join(" ")}`);
          break;
        }
        case "who": {
          const r = await api<{ sessions: { username: string; role: string; client?: string; sid?: string }[] }>("/aegis/sys/who");
          print(r.sessions.map((s) => `${pad(s.username, 16)} ${pad(s.role, 18)} ${s.client ?? ""}`).join("\n") || "(no sessions)");
          break;
        }
        case "gate": {
          const r = await api<{ mode: string; calls: number; denied: number; would_deny: number; overhead_us: { p50: number; p99: number }; abi: { syscalls: number } }>("/aegis/sys/gate");
          print(`mode ${r.mode} · ${r.abi.syscalls} syscalls · ${r.calls} calls · ${r.denied} denied · ${r.would_deny} would deny under enforce · overhead p50 ${r.overhead_us.p50}µs p99 ${r.overhead_us.p99}µs`);
          break;
        }
        case "abi": {
          const r = await api<{ syscalls: { nr: number; name: string; method: string; path: string; class: string; cap: string }[] }>("/aegis/sys/abi");
          const f = (args[0] ?? "").toLowerCase();
          const rows = r.syscalls.filter((s) => !f || s.name.includes(f) || s.path.includes(f) || s.class === f || s.cap.includes(f));
          print(rows.map((s) => `${pad(s.nr, 4)} ${pad(s.name, 30)} ${pad(s.class, 8)} ${pad(s.cap, 22)} ${s.method} ${s.path}`).join("\n") || "(no match)");
          break;
        }
        default:
          print(`${verb}: command not found. Type "help".`, "err");
      }
    } catch (e) {
      print(`${verb}: ${e instanceof Error ? e.message : String(e)}`, "err");
    } finally {
      setBusy(false);
    }
  };

  runRef.current = (cmd: string) => {
    if (!busy) run(cmd);
  };

  const onKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter" && !busy) {
      run(input);
      setInput("");
    } else if (e.key === "ArrowUp") {
      e.preventDefault();
      hi.current = Math.min(hi.current + 1, hist.current.length - 1);
      setInput(hist.current[hi.current] ?? input);
    } else if (e.key === "ArrowDown") {
      e.preventDefault();
      hi.current = Math.max(hi.current - 1, -1);
      setInput(hi.current < 0 ? "" : hist.current[hi.current]);
    } else if (e.key === "l" && e.ctrlKey) {
      e.preventDefault();
      setLines([]);
    }
  };

  // from the session hook, not localStorage during render: the static HTML says "guest"
  // and the first client render must match it
  const user = session?.credential.username ?? "guest";
  // a terminal stays a terminal on a light page: dark glass, literal colours (the theme's
  // ink tokens are for paper)
  return (
    <div
      className={`flex flex-col overflow-hidden rounded-lg border border-line-bright bg-night font-mono text-xs leading-[1.55] ${className}`}
      onClick={() => field.current?.focus()}
    >
      <div className="flex items-center gap-1.5 border-b border-line px-3 py-2">
        <span className="text-xs text-[#868d97]">aegis-sh · {user}@aegis · every command is a syscall through the gate</span>
      </div>
      <div ref={box} className="min-h-0 flex-1 overflow-auto whitespace-pre p-3">
        {lines.map((l, i) => (
          <div
            key={i}
            className={l.k === "in" ? "text-[#a5b0ff]" : l.k === "err" ? "text-[#ff8a8f]" : l.k === "dim" ? "text-[#6b717c]" : "text-[#c9cdd4]"}
          >
            {l.t || " "}
          </div>
        ))}
      </div>
      <div className="flex items-center gap-2 border-t border-line px-3 py-2">
        <span className="shrink-0 text-[#7ee2a8]">
          {user}@aegis:{cwd}$
        </span>
        <input
          ref={field}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={onKey}
          spellCheck={false}
          autoCapitalize="off"
          autoComplete="off"
          aria-label="Shell command"
          className="min-w-0 flex-1 bg-transparent text-[#eef0f3] caret-[#a5b0ff] outline-none"
        />
        {busy && <span className="text-[#6b717c]">…</span>}
      </div>
    </div>
  );
}
