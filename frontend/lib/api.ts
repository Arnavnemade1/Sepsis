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

type ApiInit = Omit<RequestInit, "body"> & { body?: unknown; timeoutMs?: number; token?: string | null; renewed?: boolean };

export async function api<T = unknown>(path: string, init: ApiInit = {}): Promise<T> {
  const { body, timeoutMs = 30_000, headers, token, renewed, ...rest } = init;
  // a pager device carries its own clinician's token so it cannot collide with an
  // operator signed in on another tab of the same browser
  const stored = token === undefined ? readSession() : null;
  const s = token !== undefined ? (token ? { token } : null) : stored;
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs);
  let data: unknown;
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
    data = text ? JSON.parse(text) : null;
    if (!res.ok) {
      const d = data as { detail?: unknown; error?: unknown; reason?: unknown } | null;
      const detail = (d && (d.detail || d.reason || d.error)) || res.statusText;
      throw new ApiError(typeof detail === "string" ? detail : JSON.stringify(detail), res.status, data);
    }
  } catch (e) {
    // the signed-in operator's own session refused: renew it once if the kernel forgot it
    if (stored && !renewed && e instanceof ApiError && refused(e.body) && (await renewStored(stored))) return api<T>(path, { ...init, renewed: true });
    throw e;
  } finally {
    clearTimeout(timer);
  }
  if (stored && !renewed && refused(data) && (await renewStored(stored))) return api<T>(path, { ...init, renewed: true });
  return data as T;
}

const refused = (d: unknown) => !!d && typeof d === "object" && ["EPERM", "EAUTH"].includes(String((d as { errno?: unknown }).errno));

async function renewStored(s: StoredSession): Promise<boolean> {
  try {
    const v = await revalidate(s);
    if (v.state === "renewed") writeSession(v.session);
    else if (v.state === "ended") writeSession(null);
    return v.state === "renewed";
  } catch {
    return false;
  }
}

const FORGOTTEN = "invalid or expired session token";

/**
 * Is this session still one the kernel knows?
 *
 * The kernel signs sessions with a per-boot secret unless AEGIS_SESSION_SECRET is
 * set, so a backend restart forgets every token, and a forgotten token is served as
 * guest (audit mode) or refused (enforce). Demo staff sign in by name, so a session
 * the kernel forgot is renewed for the same person. A session that was ENDED (signed
 * out, revoked) is not renewed: that is the kernel's decision, and it stands.
 */
export async function revalidate(s: StoredSession): Promise<{ state: "ok" } | { state: "renewed"; session: StoredSession } | { state: "ended" }> {
  let why = "";
  try {
    const w = await api<{ credential: { username: string; auth_error?: string | null } }>("/aegis/sys/whoami", { token: s.token, renewed: true });
    if (w.credential.username === s.credential.username) return { state: "ok" };
    why = w.credential.auth_error ?? "";
  } catch (e) {
    if (!(e instanceof ApiError) || e.status !== 403) throw e;
    why = JSON.stringify(e.body ?? e.message);
  }
  if (why && !why.includes(FORGOTTEN)) return { state: "ended" };
  const r = await api<{ status: string; token: string; credential: Credential }>("/aegis/sys/login", { body: { username: s.credential.username }, token: null, timeoutMs: 60_000 });
  return r.status === "ok" ? { state: "renewed", session: { token: r.token, credential: r.credential } } : { state: "ended" };
}

/** WebSocket subprotocols that carry the session to the gate. */
export function wsProtocols(token?: string | null): string[] {
  const t = token !== undefined ? token : readSession()?.token;
  return t ? ["aegis", `aegis.session.${t}`] : ["aegis"];
}

/** A clinician device's own session, kept apart from the operator's. */
const deviceKey = (id: string) => `aegis_device_${id}`;
export function readDeviceSession(id: string): StoredSession | null {
  try {
    const raw = window.localStorage.getItem(deviceKey(id));
    return raw ? (JSON.parse(raw) as StoredSession) : null;
  } catch {
    return null;
  }
}
export function writeDeviceSession(id: string, s: StoredSession | null): void {
  try {
    if (s) window.localStorage.setItem(deviceKey(id), JSON.stringify(s));
    else window.localStorage.removeItem(deviceKey(id));
  } catch {}
}
