"""Verify the real server's OS surface: the namespace, the policy file, and a loop that
stays responsive while the prover is busy.

    .venv/bin/python -m backend.aegis.verify_os_surface

Runs the server WITH its lifespan (the loops start). Supabase is switched off
before import so a test run can never write the production database.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from typing import List, Tuple

os.environ["SUPABASE_URL"] = ""                   # never the production database
os.environ.setdefault("AEGIS_JOURNAL_DB", os.path.join(tempfile.mkdtemp(), "surface.db"))

#: (name, ok, detail) — the container the falsification engine reads.
_results: List[Tuple[str, bool, str]] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"    [{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    _results.append((label, bool(cond), detail))


def main() -> int:
    _results.clear()
    print("=" * 64)
    print("  THE OS SURFACE OF THE REAL SERVER")
    print("=" * 64)
    import backend.aegis.server as S
    from fastapi.testclient import TestClient
    from backend.aegis.z3lock import Z3_LOCK

    with TestClient(S.app) as client:
        def ls(p, h=None):
            return client.get("/aegis/fs/ls", params={"path": p}, headers=h or {}).json()

        def cat(p, h=None):
            return client.get("/aegis/fs/read", params={"path": p}, headers=h or {}).json()

        # -- 1. the namespace over live objects ----------------------------------- #
        root = [e["name"] for e in ls("/")["entries"]]
        want = {"proc", "dev", "patients", "proofs", "var", "etc", "models", "apps",
                "artifacts", "home", "tmp"}
        check("every mount is present at the root", want <= set(root), str(root))
        bed = S.cluster.node_ids[0]
        sample = json.loads(cat(f"/dev/bio/{bed}/optical_feed").get("content", "{}"))
        check("/dev/bio serves the LIVE ring's latest sample, seqlock-checked",
              sample.get("writes", 0) > 0 and "values" in sample, str(sample)[:80])
        pid1 = json.loads(cat("/proc/1").get("content", "{}"))
        check("/proc/<pid> resolves the real process table", pid1.get("pid") == 1, str(pid1)[:60])
        check("a guest is refused /var/log/auth (EACCES)", cat("/var/log/auth").get("errno") == "EACCES")

        # -- 2. /etc/policy.dsl constrains the bundles the server certifies ------ #
        chief = client.post("/aegis/sys/login", json={"username": "dr.whitfield"}).json()["token"]
        HC = {"Authorization": f"Bearer {chief}"}
        nurse = client.post("/aegis/sys/login", json={"username": "rn.patel"}).json()["token"]
        HN = {"Authorization": f"Bearer {nurse}"}
        before = S.POLICY.version
        r = client.post("/aegis/fs/write", headers=HN,
                        json={"path": "/etc/policy.dsl", "content": "RULE x: ALWAYS (fluid_rate <= 1.0)\n"}).json()
        check("a nurse cannot write the policy (EACCES), whatever the gate mode",
              r.get("errno") == "EACCES" and S.POLICY.version == before, str(r)[:80])
        bundle = json.loads(cat(f"/patients/{bed}/bundle").get("content", "{}"))
        drug = bundle.get("protocol", {}).get("antibiotic")
        from backend.aegis.kernel.bundle import ANTIBIOTICS
        from backend.aegis.kernel.policyfile import CAGE_TO_DSL
        from backend.aegis import clinical_dsl as dsl
        code = next(k for k, n in ANTIBIOTICS.items() if n == drug)
        dsl_name = next(n for n, v in dsl.ENUMS["antibiotic"].items() if v == CAGE_TO_DSL[code])
        r = client.post("/aegis/fs/write", headers=HC, json={
            "path": "/etc/policy.dsl",
            "content": S.POLICY.text + f"\nRULE surface_ban: ALWAYS (antibiotic != {dsl_name})\n"}).json()
        check("the chief writes a policy that forbids the drug this bed's bundle uses",
              r.get("status") == "ok" and S.POLICY.version == before + 1, str(r)[:100])
        after = json.loads(cat(f"/patients/{bed}/bundle").get("content", "{}"))
        check("THE CLAIM: the next certified bundle is proven against it and is NOT all_verified",
              after.get("policy", {}).get("version") == S.POLICY.version
              and after.get("policy", {}).get("ok") is False and after.get("all_verified") is False,
              f"drug={drug}; policy={str(after.get('policy'))[:90]}")
        client.post("/aegis/fs/write", headers=HC, json={"path": "/etc/policy.dsl",
                                                          "content": dsl.DEFAULT_POLICY})

        # -- 3. the event loop stays responsive while the prover is held -------- #
        with client.websocket_connect("/aegis/os/stream") as ws:
            ws.receive_json()                                    # hello
            release = threading.Event()

            def hog() -> None:
                with Z3_LOCK:
                    release.wait(3.0)

            threading.Thread(target=hog, daemon=True).start()
            time.sleep(0.1)
            rtts = []
            t_end = time.time() + 2.6
            while time.time() < t_end:
                t0 = time.perf_counter()
                ws.send_json({"op": "ping"})
                while ws.receive_json().get("type") != "pong":
                    pass
                rtts.append(time.perf_counter() - t0)
                time.sleep(0.3)
            release.set()
            worst = max(rtts) if rtts else 99.0
            check("THE CLAIM: while another thread holds Z3 for 3 s the OS socket still answers "
                  "(the ward loop's prover work is off the event loop)",
                  len(rtts) >= 5 and worst < 0.6, f"{len(rtts)} pings, worst {worst * 1000:.0f} ms")

    failed = [n for n, ok, _ in _results if not ok]
    print("\n" + "=" * 64)
    if failed:
        print(f"  FAILED: {len(failed)} check(s): {', '.join(failed)}")
        print("=" * 64)
        return 1
    print(f"  ALL {len(_results)} CHECKS PASSED — the namespace is live and the policy file is law")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
