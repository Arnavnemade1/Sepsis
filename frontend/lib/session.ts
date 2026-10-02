"use client";

import { useEffect } from "react";

import { api, readSession, revalidate, writeSession, type Credential } from "./api";

type LoginOut = { status: string; token: string; credential: Credential; detail?: string };
const SIGNED_OUT = "aegis_signed_out";
const DEFAULT_OPERATOR = "rn.patel";

export async function signInAs(username: string): Promise<boolean> {
  const r = await api<LoginOut>("/aegis/sys/login", { body: { username }, token: null, timeoutMs: 60_000 });
  if (r.status !== "ok") return false;
  writeSession({ token: r.token, credential: r.credential });
  try {
    localStorage.removeItem(SIGNED_OUT);
  } catch {}
  return true;
}

export async function signOut() {
  await api("/aegis/sys/logout", { method: "POST" }).catch(() => {});
  writeSession(null);
  try {
    localStorage.setItem(SIGNED_OUT, "1");
  } catch {}
}

/** The demo hospital opens on the charge nurse's console. Someone who signed out
 *  stays signed out (view only) until they choose a user. */
export function useEnsureSession() {
  useEffect(() => {
    let off = false;
    try {
      off = localStorage.getItem(SIGNED_OUT) === "1";
    } catch {}
    const s = readSession();
    if (s) {
      // the kernel's secret may be random per boot: renew a session it forgot, drop one it ended
      revalidate(s)
        .then((v) => (v.state === "renewed" ? writeSession(v.session) : v.state === "ended" && writeSession(null)))
        .catch(() => {});
      return;
    }
    if (!off) signInAs(DEFAULT_OPERATOR).catch(() => {});
  }, []);
}

