"""Verify the filesystem and /etc/policy.dsl — permissions, quotas, persistence, and a
policy file that actually constrains the orders the kernel certifies.

    .venv/bin/python -m backend.aegis.kernel.verify_vfs
"""
from __future__ import annotations

import os
import sys
import tempfile
from typing import List, Tuple

from backend.aegis.kernel.journal import Journal
from backend.aegis.kernel.policyfile import PolicyRejected, PolicyStore
from backend.aegis.kernel.proc import ProcessTable
from backend.aegis.kernel.users import GUEST, UserDB
from backend.aegis.kernel.vfs import (HOME_FILE_MAX, FileStore, HomeFS, Leaf, TmpFS, VFS,
                                      VFSError, VirtualFS)
from backend.aegis.rbac import RBAC

#: (name, ok, detail) — the container the falsification engine reads.
_results: List[Tuple[str, bool, str]] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"    [{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    _results.append((label, bool(cond), detail))


def _errno(fn) -> str:
    try:
        fn()
        return "OK"
    except VFSError as exc:
        return exc.errno


def main() -> int:
    _results.clear()
    print("=" * 64)
    print("  VFS — one namespace, enforced permissions, a policy file that bites")
    print("=" * 64)
    tmpdir = tempfile.mkdtemp()
    db = os.path.join(tmpdir, "files.db")
    j = Journal(ring=2048)
    users = UserDB(RBAC(), procs=ProcessTable(journal=j), journal=j)
    _, _, att = users.login("dr.alvarez")
    _, _, nurse = users.login("rn.patel")
    _, _, chief = users.login("dr.whitfield")

    v = VFS(journal=j)
    notes: List[str] = []
    v.mount("/etc", VirtualFS(lambda: {
        "os-release": Leaf(lambda c: "AegisOS"),
        "shadow": Leaf(lambda c: "secret", cap_read="view:audit"),
    }, "/etc"))
    v.mount("/patients", VirtualFS(lambda: {"bed_0": {
        "notes": Leaf(lambda c: "\n".join(notes), write=lambda c, d, a: notes.append(d),
                      cap_write="annotate", append_only=True),
        "vitals": Leaf(lambda c: {"hr": 88}),
    }}, "/patients"))
    v.mount("/var/log", VirtualFS(lambda: {"kern": Leaf(lambda c: "boot")}, "/var/log"))
    store = FileStore(sqlite_path=db, use_supabase=False, journal=j)
    v.mount("/home", HomeFS(store, lambda: ["dr.alvarez", "rn.patel", "dr.whitfield"]))
    v.mount("/tmp", TmpFS())

    # -- 1. the namespace ------------------------------------------------------ #
    check("'..' cannot climb above the root", v.normalize("/../../etc/../../proc") == "/proc")
    check("relative paths resolve against the working directory",
          v.normalize("notes", "/patients/bed_0") == "/patients/bed_0/notes")
    check("the root lists every mount, including nested ones (/var)",
          [n.name for n in v.ls(GUEST, "/")] == ["etc", "home", "patients", "tmp", "var"])
    check("an intermediate directory of a nested mount lists its child",
          [n.name for n in v.ls(GUEST, "/var")] == ["log"])
    check("a missing path is ENOENT", _errno(lambda: v.stat(GUEST, "/etc/nope")) == "ENOENT")
    check("reading a directory is EISDIR", _errno(lambda: v.read(GUEST, "/etc")) == "EISDIR")

    # -- 2. read and write capabilities, in every gate mode ------------------- #
    check("THE CLAIM: a guest cannot read a view:audit file (EACCES)",
          _errno(lambda: v.read(GUEST, "/etc/shadow")) == "EACCES")
    check("…and the refusal is journalled",
          any(e.event == "read.denied" and e.uid == "guest" for e in j.entries))
    check("CONTROL: the attending, who holds view:audit, reads it",
          v.read(att, "/etc/shadow")["content"] == "secret")
    check("a read-only file refuses a write (EROFS)",
          _errno(lambda: v.write(chief, "/etc/os-release", "x")) == "EROFS")
    check("a nurse (who holds annotate) may append a note",
          _errno(lambda: v.write(nurse, "/patients/bed_0/notes", "reassessed", append=True)) == "OK")
    check("a guest may not (EACCES)",
          _errno(lambda: v.write(GUEST, "/patients/bed_0/notes", "x", append=True)) == "EACCES")
    check("an append-only file refuses an overwrite (EPERM)",
          _errno(lambda: v.write(nurse, "/patients/bed_0/notes", "rewrite")) == "EPERM")
    check("JSON leaves read back as JSON", v.read(GUEST, "/patients/bed_0/vitals")["kind"] == "json")

    # -- 3. /home: owner-only, quotas, persistence --------------------------- #
    check("an owner writes a file in her home",
          _errno(lambda: v.write(att, "/home/dr.alvarez/rounds.sh", "ls /patients\n")) == "OK")
    check("another user cannot read it (EACCES)",
          _errno(lambda: v.read(nurse, "/home/dr.alvarez/rounds.sh")) == "EACCES")
    check("…nor overwrite it (EACCES)",
          _errno(lambda: v.write(nurse, "/home/dr.alvarez/rounds.sh", "rm -r /")) == "EACCES")
    check("the shared guest identity has no writable home (EROFS)",
          _errno(lambda: v.write(GUEST, "/home/guest/x", "y")) == "EROFS")
    check("a file above the per-file limit is refused (EFBIG)",
          _errno(lambda: v.write(att, "/home/dr.alvarez/big", "x" * (HOME_FILE_MAX + 1))) == "EFBIG")
    # 16 files of HOME_FILE_MAX-10 bytes plus rounds.sh is 1,048,429 bytes — still under
    # the 1 MiB quota (a first version of this check wrote only 15 and "expected" the
    # 16th to overflow; it fit). The 17th cannot.
    from backend.aegis.kernel.vfs import HOME_USER_MAX
    for i in range(16):
        v.write(att, f"/home/dr.alvarez/chunk{i}", "y" * (HOME_FILE_MAX - 10))
    used = sum(r["size"] for r in store.files("dr.alvarez").values())
    check("the quota is not yet exceeded (the boundary is where it should be)",
          used <= HOME_USER_MAX < used + HOME_FILE_MAX - 10, f"{used} bytes used")
    check("the per-user quota refuses what would exceed it (EDQUOT)",
          _errno(lambda: v.write(att, "/home/dr.alvarez/chunk99", "y" * (HOME_FILE_MAX - 10))) == "EDQUOT")
    check("a file in a missing directory is ENOENT (mkdir first)",
          _errno(lambda: v.write(att, "/home/dr.alvarez/scripts/a.sh", "x")) == "ENOENT")
    v.mkdir(att, "/home/dr.alvarez/scripts")
    v.write(att, "/home/dr.alvarez/scripts/a.sh", "echo hi")
    check("a non-empty directory cannot be removed (ENOTEMPTY)",
          _errno(lambda: v.rm(att, "/home/dr.alvarez/scripts")) == "ENOTEMPTY")
    reborn = FileStore(sqlite_path=db, use_supabase=False, journal=j)     # "restart"
    check("THE CLAIM: a file survives a restart of the store",
          (reborn.files("dr.alvarez").get("rounds.sh") or {}).get("content") == "ls /patients\n")

    # -- 4. /tmp: shared, sticky, bounded ----------------------------------- #
    check("a guest may write scratch in /tmp", _errno(lambda: v.write(GUEST, "/tmp/scratch", "hi")) == "OK")
    check("/tmp is sticky: nobody overwrites another's file (EACCES)",
          _errno(lambda: v.write(att, "/tmp/scratch", "mine now")) == "EACCES")
    check("/tmp is bounded (ENOSPC)",
          _errno(lambda: v.write(nurse, "/tmp/flood", "z" * (300 * 1024))) == "ENOSPC")

    # -- 5. /etc/policy.dsl -------------------------------------------------- #
    ps = PolicyStore(journal=j, store=store)
    v0 = ps.version

    def rejected(text: str) -> Tuple[bool, str]:
        try:
            ps.write(chief, text)
            return False, "accepted"
        except PolicyRejected as exc:
            return True, str(exc)[:60]

    ok, why = rejected("RULE broken: ALWAYS (risk = = HIGH")
    check("a policy that does not parse is refused", ok, why)
    ok, why = rejected("RULE a: ALWAYS (risk = HIGH -> antibiotic != NONE)\n"
                       "RULE b: ALWAYS (risk = HIGH -> antibiotic = NONE)\n")
    check("THE CLAIM: a self-contradictory policy is refused", ok, why)
    check("…and the refusal left the active policy untouched", ps.version == v0)
    allergic_pip = {"antibiotic_code": 1, "fluid_rate": 4.0, "vasopressor_dose": 0.0}
    cert = ps.certify(allergic_pip, "HIGH", {"penicillin_allergy": True})
    check("the default policy refuses pip-tazo for a penicillin-allergic patient",
          not cert["ok"] and cert["violated"][0]["rule"] == "allergy_guard", str(cert["violated"])[:80])
    ps.write(chief, "RULE levo_ban: ALWAYS (antibiotic != Levofloxacin)\n")
    check("a valid policy activates (version increments)", ps.version == v0 + 1)
    check("the cage's Levofloxacin (code 5) is proven as Levofloxacin",
          not ps.certify({"antibiotic_code": 5}, "LOW", {})["ok"])
    ps.write(chief, "RULE cipro_ban: ALWAYS (antibiotic != Ciprofloxacin)\n")
    check("CONTROL: a Ciprofloxacin ban does NOT catch Levofloxacin (no silent substitution)",
          ps.certify({"antibiotic_code": 5}, "LOW", {})["ok"])
    persisted = PolicyStore(journal=j, store=FileStore(sqlite_path=db, use_supabase=False))
    check("the active policy persists through the file store",
          "cipro_ban" in persisted.text and [r.name for r in persisted.rules] == ["cipro_ban"])

    # -- 6. the bundle the kernel certifies is proven against the file --------- #
    from backend.aegis.kernel.bundle import SepsisBundleEngine
    from backend.aegis.kernel.scheduler import BioCluster
    cl = BioCluster(n_nodes=2, lane_budget=32, seed=4)
    cl.run(10)
    cl.inject_crisis(cl.node_ids[0], 1.0)
    for _ in range(40):
        cl.tick()
    prof = cl.profiles[cl.node_ids[0]]
    eng = SepsisBundleEngine(policy=PolicyStore(journal=j))
    b = eng.generate(prof, {"penicillin_allergy": True})
    drug = b["protocol"]["antibiotic"]
    check("a certified bundle carries the policy certificate",
          isinstance(b.get("policy"), dict) and b["policy"]["ok"] is True, str(b.get("policy"))[:80])
    from backend.aegis.kernel.policyfile import CAGE_TO_DSL
    from backend.aegis import clinical_dsl as dsl
    code = next(k for k, name in __import__("backend.aegis.kernel.bundle", fromlist=["ANTIBIOTICS"])
                .ANTIBIOTICS.items() if name == drug)
    dsl_name = next(n for n, val in dsl.ENUMS["antibiotic"].items() if val == CAGE_TO_DSL[code])
    ban = PolicyStore(journal=j, initial=f"RULE ban_it: ALWAYS (antibiotic != {dsl_name})\n")
    b2 = SepsisBundleEngine(policy=ban).generate(prof, {"penicillin_allergy": True})
    check("THE CLAIM: a policy forbidding the bundle's own drug makes it NOT all_verified",
          b2["all_verified"] is False and b2["policy"]["ok"] is False
          and b2["protocol"]["certified"] is True,
          f"drug={drug} ({dsl_name}); cage still certifies it: {b2['protocol']['certified']}")

    failed = [n for n, ok, _ in _results if not ok]
    print("\n" + "=" * 64)
    if failed:
        print(f"  FAILED: {len(failed)} check(s): {', '.join(failed)}")
        print("=" * 64)
        return 1
    print(f"  ALL {len(_results)} CHECKS PASSED — permissions hold, and the policy file constrains orders")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
