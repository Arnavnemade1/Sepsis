"use client";

import { useSearchParams } from "next/navigation";
import { Suspense } from "react";

import { Architecture } from "@/components/os/architecture";
import { GateView } from "@/components/os/gate";
import { JournalView } from "@/components/os/journal";
import { Kernel } from "@/components/os/kernel";
import { ShellView } from "@/components/os/shell-view";
import { Page } from "@/components/shell";

type View = "architecture" | "kernel" | "journal" | "gate" | "shell";
const VIEWS: View[] = ["architecture", "kernel", "journal", "gate", "shell"];

export default function OsPage() {
  return (
    <Suspense>
      <Platform />
    </Suspense>
  );
}

function Platform() {
  const raw = useSearchParams().get("view") ?? "architecture";
  const view: View = (VIEWS as string[]).includes(raw) ? (raw as View) : "architecture";
  return (
    <Page active={view}>
      {view === "architecture" && <Architecture />}
      {view === "kernel" && <Kernel />}
      {view === "journal" && <JournalView />}
      {view === "gate" && <GateView />}
      {view === "shell" && <ShellView />}
    </Page>
  );
}
