/**
 * The one door to the AegisOS backend.
 *
 * Every request goes through `api()` so the gate token rides on it. The kernel
 * gate (backend/aegis/kernel/gate.py) reads a Bearer token over HTTP and an
 * `aegis.session.<token>` WebSocket subprotocol on sockets — never the query
 * string, because URLs land in access logs.
 *
 * Backend resolution, most specific first:
 *   1. localStorage "aegis_backend"   — a live override, no redeploy needed
 *   2. NEXT_PUBLIC_AEGIS_BACKEND      — the build-time default (Render in prod)
 *   3. same origin                    — when the backend serves this app itself
 */

const STORAGE_BACKEND = "aegis_backend";
const STORAGE_SESSION = "aegis_session";

export function backendBase(): string {
  if (typeof window !== "undefined") {
    try {
      const o = window.localStorage.getItem(STORAGE_BACKEND);
      if (o) return o.replace(/\/+$/, "");
    } catch {}
  }
  const env = (process.env.NEXT_PUBLIC_AEGIS_BACKEND || "").replace(/\/+$/, "");
  if (env) return env;
  return typeof window !== "undefined" ? window.location.origin : "";
}

export function wsUrl(path: string): string {
  return backendBase().replace(/^http/, "ws") + path;
}

export type StoredSession = {
  token: string;
  credential: Credential;
};

export type Credential = {
  uid: number;
  username: string;
  display: string;
  role: Role;
  ring: number;
  kind: string;
  capabilities?: string[];
  title?: string;
};

export type Role = "guest" | "nurse" | "remote_specialist" | "pharmacist" | "attending" | "chief";

export function readSession(): StoredSession | null {
  if (typeof window === "undefined") return null;
  try {
    const raw = window.localStorage.getItem(STORAGE_SESSION);
    return raw ? (JSON.parse(raw) as StoredSession) : null;
  } catch {
    return null;
  }
}

export function writeSession(s: StoredSession | null): void {
  try {
    if (s) window.localStorage.setItem(STORAGE_SESSION, JSON.stringify(s));
    else window.localStorage.removeItem(STORAGE_SESSION);
  } catch {}
  window.dispatchEvent(new Event("aegis:session"));
}

export class ApiError extends Error {
  constructor(
    message: string,
    readonly status: number,
    readonly body?: unknown,
  ) {
    super(message);
  }
}

type ApiInit = Omit<RequestInit, "body"> & { body?: unknown; timeoutMs?: number };

export async function api<T = unknown>(path: string, init: ApiInit = {}): Promise<T> {
  const { body, timeoutMs = 30_000, headers, ...rest } = init;
  const s = readSession();
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  try {
    const res = await fetch(backendBase() + path, {
      ...rest,
      method: rest.method ?? (body === undefined ? "GET" : "POST"),
      signal: rest.signal ?? ctrl.signal,
      headers: {
        ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
        ...(s?.token ? { Authorization: `Bearer ${s.token}` } : {}),
        ...headers,
      },
      body: body === undefined ? undefined : JSON.stringify(body),
    });
    const text = await res.text();
    const data = text ? JSON.parse(text) : null;
    if (!res.ok) {
      const detail = (data && (data.detail || data.error)) || res.statusText;
      throw new ApiError(typeof detail === "string" ? detail : JSON.stringify(detail), res.status, data);
    }
    return data as T;
  } finally {
    clearTimeout(timer);
  }
}

/** WebSocket subprotocols that carry the session to the gate. */
export function wsProtocols(): string[] {
  const s = readSession();
  return s?.token ? ["aegis", `aegis.session.${s.token}`] : ["aegis"];
}
