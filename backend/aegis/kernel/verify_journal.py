"""Verify the kernel journal — and prove its tamper detection can FAIL.

    .venv/bin/python -m backend.aegis.kernel.verify_journal

The claim under test is "tamper-evident". A hash chain that is never shown
rejecting a tampered entry is decoration, so check 3 edits a stored entry and
requires ``verify()`` to go red and to name the entry it broke at.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
from typing import List, Tuple

from backend.aegis.kernel.journal import (CRIT, DEBUG, INFO, WARN, Journal,
                                          JournalStore, GENESIS, default_sqlite_path)

#: (name, ok, detail) — the container the falsification engine reads.
_results: List[Tuple[str, bool, str]] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"    [{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    _results.append((label, bool(cond), detail))


def main() -> int:
    _results.clear()
    print("=" * 64)
    print("  KERNEL JOURNAL — ordering, integrity, durability")
    print("=" * 64)
    tmp = os.path.join(tempfile.mkdtemp(), "journal_test.db")

    # -- 1. ordering across facilities ------------------------------------- #
    j = Journal(ring=64)
    j.append("syscall", "treat", uid="okafor", pid=7, bed="bed_3")
    j.append("sched", "preempt", severity=WARN, owner="bed_3")
    j.append("auth", "login", uid="okafor", role="attending")
    seqs = [e.seq for e in j.entries]
    check("one monotonic sequence across facilities", seqs == [0, 1, 2], str(seqs))
    check("per-facility counts", j.counts == {"syscall": 1, "sched": 1, "auth": 1},
          str(j.counts))

    # -- 2. the chain verifies ---------------------------------------------- #
    v = j.verify()
    check("the intact chain verifies", v["ok"] and v["window"] == 3, str(v))
    check("the head advances with the last entry", j.head() == j.entries[-1].hash)

    # -- 3. THE CLAIM: a tampered entry is detected -------------------------- #
    victim = j.entries[1]
    original = victim.detail
    victim.detail = {"owner": "bed_9"}          # rewrite history, keep the hash
    v_bad = j.verify()
    check("an altered entry breaks the chain", not v_bad["ok"], str(v_bad))
    check("verify() names where it broke", v_bad.get("broken_at") == 1,
          f"broken_at={v_bad.get('broken_at')}")
    victim.detail = original
    check("CONTROL: restoring the entry restores the chain (the check is not stuck red)",
          j.verify()["ok"])

    # -- 4. severity + facility filtering ----------------------------------- #
    j2 = Journal(ring=64)
    j2.append("kern", "boot", severity=INFO)
    j2.append("irq", "high alarm", severity=CRIT, bed="bed_1")
    j2.append("vfs", "write", severity=DEBUG)
    crit_only = j2.tail(min_severity=WARN)
    check("severity filter keeps only the serious lines",
          len(crit_only) == 1 and crit_only[0]["event"] == "high alarm", str(crit_only))
    check("facility filter selects one facility",
          [e["event"] for e in j2.tail(facility="kern")] == ["boot"])

    # -- 5. the ring is bounded but the sequence is not --------------------- #
    j3 = Journal(ring=8)
    for i in range(50):
        j3.append("kern", f"tick {i}")
    check("the ring holds only its window", len(j3.entries) == 8, str(len(j3.entries)))
    check("the sequence keeps counting past the window",
          j3.entries[-1].seq == 49, str(j3.entries[-1].seq))
    check("the retained window still verifies", j3.verify()["ok"])

    # -- 6. durability: a reboot CONTINUES the chain ------------------------ #
    store = JournalStore(sqlite_path=tmp, use_supabase=False)
    boot1 = Journal(ring=64, store=store)
    for i in range(5):
        boot1.append("boot", f"phase {i}")
    store.flush()
    head1 = boot1.head()
    persisted = store.load(limit=100)
    check("entries reach the store", len(persisted) == 5, str(len(persisted)))
    check("the persisted run verifies from genesis",
          Journal.verify_chain(persisted, GENESIS))

    boot2 = Journal(ring=64, store=store)        # a fresh kernel, same database
    rehydrated = boot2.rehydrate()
    check("a rebooted kernel resumes the persisted head",
          rehydrated is not None and boot2.head() == head1, str(rehydrated))
    boot2.append("boot", "phase 5")
    store.flush()
    all_rows = store.load(limit=100)
    check("the chain spans the reboot (one chain, not two)",
          len(all_rows) == 6 and Journal.verify_chain(all_rows, GENESIS),
          f"{len(all_rows)} rows")
    check("sequence continues across the reboot", all_rows[-1].seq == 5,
          str(all_rows[-1].seq))

    # -- 7. a full queue drops and counts, it never blocks ------------------ #
    small = JournalStore(sqlite_path=tmp, use_supabase=False, queue_max=4)
    j4 = Journal(ring=64, store=small)
    for i in range(20):
        j4.append("kern", f"flood {i}")          # no flusher running: queue fills
    check("a full queue drops rather than blocking the kernel", small.dropped == 16,
          f"dropped={small.dropped}")
    check("dropped entries are still in the live chain",
          j4.verify()["ok"] and len(j4.entries) == 20)

    # -- 8. reads race appends from other threads ---------------------------- #
    # The loops, the executor and the training threads all append while the API
    # reads. Iterating the ring unlocked raised "deque mutated during iteration",
    # and reading the head after copying the ring made a concurrent append look
    # like a broken chain — a false tamper alarm.
    jc = Journal(ring=256)
    stop = threading.Event()

    def _writer() -> None:
        i = 0
        while not stop.is_set():
            jc.append("sched", f"decision {i}")
            i += 1

    th = threading.Thread(target=_writer, daemon=True)
    th.start()
    errors, false_alarms = 0, 0
    try:
        for _ in range(300):
            try:
                if not jc.verify()["ok"]:
                    false_alarms += 1
                jc.tail(limit=20)
            except RuntimeError:
                errors += 1
    finally:
        stop.set()
        th.join(2.0)
    check("verify/tail never raise while another thread appends", errors == 0,
          f"{errors} RuntimeErrors")
    check("a concurrent append is never reported as tampering", false_alarms == 0,
          f"{false_alarms} false alarms in 300 verifies")

    # -- 9. the default store is the gitignored backend database ------------ #
    path = default_sqlite_path()
    check("the default store is backend/sepsis_twin.db (gitignored), not inside the package",
          path.replace(os.sep, "/").endswith("/backend/sepsis_twin.db")
          or bool(os.getenv("AEGIS_JOURNAL_DB")), path)

    failed = [n for n, ok, _ in _results if not ok]
    print("\n" + "=" * 64)
    if failed:
        print(f"  FAILED: {len(failed)} check(s): {', '.join(failed)}")
        print("=" * 64)
        return 1
    print("  ALL CHECKS PASSED — one ordered, tamper-evident, restart-spanning record")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
