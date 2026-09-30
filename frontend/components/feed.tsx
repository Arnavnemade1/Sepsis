"use client";

import { useEffect, useState } from "react";

import type { FeedStatus } from "@/lib/hooks";

import { Pill } from "./ui";

/** The connection state of a live feed, with the age of the last frame. */
export function FeedPill({ status, lastAt, label }: { status: FeedStatus; lastAt: number | null; label: string }) {
  const [, tick] = useState(0);
  useEffect(() => {
    const t = setInterval(() => tick((n) => n + 1), 1000);
    return () => clearInterval(t);
  }, []);
  const age = lastAt ? Math.round((Date.now() - lastAt) / 1000) : null;
  const stale = age != null && age > 12;
  if (status !== "live")
    return (
      <Pill tone="med" dot>
        {label} · {status === "connecting" ? "connecting" : "reconnecting"}
      </Pill>
    );
  return (
    <Pill tone={stale ? "med" : "neutral"} dot live={!stale}>
      {label} · {age == null ? "live" : stale ? `stale ${age}s` : "live"}
    </Pill>
  );
}
