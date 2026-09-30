"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { api, readSession, wsProtocols, wsUrl, type StoredSession } from "./api";

/**
 * A WebSocket feed with reconnect. The backend pushes a full payload on
 * connect, then one per tick, so the last message IS the state — no merging.
 * Backoff doubles to 15 s; a Render free-tier cold start takes ~30 s, and
 * hammering it while it boots only delays it.
 */
export function useFeed<T>(path: string | null): { data: T | null; status: FeedStatus; lastAt: number | null } {
  const [data, setData] = useState<T | null>(null);
  const [status, setStatus] = useState<FeedStatus>("connecting");
  const [lastAt, setLastAt] = useState<number | null>(null);

  useEffect(() => {
    if (!path) return;
    let ws: WebSocket | null = null;
    let closed = false;
    let backoff = 1000;
    let timer: ReturnType<typeof setTimeout> | undefined;
    let ping: ReturnType<typeof setInterval> | undefined;

    const open = () => {
      setStatus("connecting");
      try {
        ws = new WebSocket(wsUrl(path), wsProtocols());
      } catch {
        schedule();
        return;
      }
      ws.onopen = () => {
        backoff = 1000;
        setStatus("live");
        ping = setInterval(() => ws?.readyState === 1 && ws.send("ping"), 20_000);
      };
      ws.onmessage = (ev) => {
        if (ev.data === "pong") return;
        try {
          setData(JSON.parse(ev.data));
          setLastAt(Date.now());
        } catch {}
      };
      ws.onclose = () => {
        clearInterval(ping);
        if (!closed) schedule();
      };
      ws.onerror = () => ws?.close();
    };
    const schedule = () => {
      setStatus("reconnecting");
      timer = setTimeout(open, backoff);
      backoff = Math.min(backoff * 2, 15_000);
    };
    // A backgrounded tab must not keep draining the backend's bandwidth (Render
    // free tier): close the socket while hidden, reopen when visible again.
    const onVis = () => {
      if (document.hidden) {
        closed = true;
        clearTimeout(timer);
        clearInterval(ping);
        ws?.close();
      } else if (closed) {
        closed = false;
        backoff = 1000;
        open();
      }
    };
    document.addEventListener("visibilitychange", onVis);
    if (!document.hidden) open();
    else closed = true;
    return () => {
      closed = true;
      document.removeEventListener("visibilitychange", onVis);
      clearTimeout(timer);
      clearInterval(ping);
      ws?.close();
    };
  }, [path]);

  return { data, status, lastAt };
}

export type FeedStatus = "connecting" | "live" | "reconnecting";

/** Poll a REST resource. `path=null` pauses. */
export function usePoll<T>(path: string | null, everyMs = 0): {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
} {
  const [data, setData] = useState<T | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [nonce, setNonce] = useState(0);
  const reload = useCallback(() => setNonce((n) => n + 1), []);
  const pathRef = useRef(path);
  pathRef.current = path;

  useEffect(() => {
    if (!path) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout> | undefined;
    const run = async () => {
      setLoading(true);
      try {
        const d = await api<T>(path, { timeoutMs: 60_000 });
        if (alive && pathRef.current === path) {
          setData(d);
          setError(null);
        }
      } catch (e) {
        if (alive) setError(e instanceof Error ? e.message : String(e));
      } finally {
        if (alive) setLoading(false);
        if (alive && everyMs > 0 && !document.hidden) timer = setTimeout(run, everyMs);
      }
    };
    // Polling stops while the tab is hidden and refreshes once on return.
    const onVis = () => {
      if (!document.hidden && alive && everyMs > 0) {
        clearTimeout(timer);
        run();
      }
    };
    document.addEventListener("visibilitychange", onVis);
    run();
    return () => {
      alive = false;
      document.removeEventListener("visibilitychange", onVis);
      clearTimeout(timer);
    };
  }, [path, everyMs, nonce]);

  return { data, error, loading, reload };
}

/** The signed-in staff member, kept in step across tabs and components. */
export function useSession(): StoredSession | null {
  const [s, setS] = useState<StoredSession | null>(null);
  useEffect(() => {
    const sync = () => setS(readSession());
    sync();
    window.addEventListener("aegis:session", sync);
    window.addEventListener("storage", sync);
    return () => {
      window.removeEventListener("aegis:session", sync);
      window.removeEventListener("storage", sync);
    };
  }, []);
  return s;
}
