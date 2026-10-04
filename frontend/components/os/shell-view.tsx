"use client";

import { CornerDownLeft } from "lucide-react";

import { PageHead } from "@/components/shell";
import { Terminal } from "@/components/terminal";
import { Panel } from "@/components/ui";

const COMMANDS: { cmd: string; about: string }[] = [
  { cmd: "ls /proc", about: "Kernel state as files" },
  { cmd: "cat /etc/os-release", about: "The running system" },
  { cmd: "ls /patients", about: "Every bed as a directory" },
  { cmd: "ps", about: "Process table" },
  { cmd: "top", about: "Processes by CPU" },
  { cmd: "dmesg 20 paging", about: "Journal tail for one facility" },
  { cmd: "boot", about: "The measured boot record" },
  { cmd: "gate", about: "Syscall gate status" },
  { cmd: "abi paging", about: "Syscalls matching a filter" },
  { cmd: "whoami", about: "Your identity and capabilities" },
  { cmd: "who", about: "Open sessions" },
  { cmd: "uname -a", about: "Kernel identification" },
];

export function ShellView() {
  return (
    <div>
      <PageHead title="Shell">
        A shell on the kernel. Every command is a syscall through the gate, so what you can read here is exactly what your role may read: a refusal
        comes from the kernel, not from the interface.
      </PageHead>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[minmax(0,1fr)_320px]">
        <Terminal greet className="h-[calc(100dvh-230px)] min-h-[420px]" />
        <Panel title="Commands" meta="Select one to run it" flush>
          <ul>
            {COMMANDS.map((c) => (
              <li key={c.cmd}>
                <button
                  onClick={() => window.dispatchEvent(new CustomEvent("aegis:shell", { detail: c.cmd }))}
                  className="group flex w-full items-center gap-3 border-b border-line-soft px-3.5 py-2 text-left transition-colors last:border-b-0 hover:bg-s2"
                >
                  <span className="min-w-0 flex-1">
                    <span className="block font-mono text-[12px] text-ink">{c.cmd}</span>
                    <span className="block text-[11.5px] text-dim">{c.about}</span>
                  </span>
                  <CornerDownLeft size={13} className="shrink-0 text-faint group-hover:text-muted" />
                </button>
              </li>
            ))}
          </ul>
        </Panel>
      </div>
    </div>
  );
}
