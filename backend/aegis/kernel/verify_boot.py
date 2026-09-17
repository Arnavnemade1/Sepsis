"""Verify the boot sequencer — real durations, and failures that stay visible.

    .venv/bin/python -m backend.aegis.kernel.verify_boot

The point of this record is that when a deploy hangs or a subsystem dies at
startup, the log says which phase and why. So the check that matters is the
failing one: a phase that raises must be recorded as failed AND must still
propagate, because a boot recorder that swallows exceptions converts a broken
deploy into a silently degraded one.
"""
from __future__ import annotations

import sys
import time
from typing import List, Tuple

from backend.aegis.kernel.boot import BootSequencer
from backend.aegis.kernel.journal import Journal

#: (name, ok, detail) — the container the falsification engine reads.
_results: List[Tuple[str, bool, str]] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"    [{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    _results.append((label, bool(cond), detail))


def main() -> int:
    _results.clear()
    print("=" * 64)
    print("  BOOT SEQUENCER — real phases, real durations")
    print("=" * 64)
    j = Journal(ring=256)
    b = BootSequencer(journal=j)

    with b.phase("detector", "load the calibrated model") as p:
        time.sleep(0.02)
        p.detail(artifact="gbm_wearable.joblib", mb=7.7)
    with b.phase("devbio", "mount the patient buses") as p:
        time.sleep(0.01)
        p.detail(beds=12, drivers=3)

    log = b.log()
    check("phases are recorded in order",
          [x["phase"] for x in log] == ["detector", "devbio"], str([x["phase"] for x in log]))
    check("durations are measured, not declared", log[0]["ms"] >= 18.0, f"{log[0]['ms']}ms")
    check("a phase records what it produced", log[1]["detail"] == {"beds": 12, "drivers": 3},
          str(log[1]["detail"]))
    check("the slowest phase is identified",
          b.summary()["slowest"]["phase"] == "detector", str(b.summary()["slowest"]))

    # -- THE CLAIM: a failing phase is recorded AND still raises ------------ #
    raised = False
    try:
        with b.phase("services", "start aegisd"):
            raise RuntimeError("planted: dependency cycle")
    except RuntimeError:
        raised = True
    check("a failing phase propagates (boot is observed, not swallowed)", raised)
    failed = [x for x in b.log() if not x["ok"]]
    check("the failure is recorded with its cause",
          len(failed) == 1 and "planted" in (failed[0]["error"] or ""), str(failed))
    check("summary names the failed phase", b.summary()["failed"] == ["services"],
          str(b.summary()["failed"]))
    # CONTROL: the two good phases did NOT get marked failed, so "failed" is
    # selective rather than a flag that trips for everything after an error.
    check("CONTROL: successful phases stay successful",
          [x["ok"] for x in b.log()] == [True, True, False])

    # -- checkpoints: module-level construction without re-indenting it ---- #
    c = BootSequencer(journal=Journal(ring=64))
    c.checkpoint("session", "load the detector")
    time.sleep(0.01)
    c.note(model="stack+xgb@utility")
    c.checkpoint("wing", "mount the beds")        # closes "session"
    time.sleep(0.005)
    c.ready()                                    # closes "wing"
    cl = c.log()
    check("a checkpoint closes the previous one with a measured duration",
          [x["phase"] for x in cl] == ["session", "wing"] and cl[0]["ms"] >= 9.0,
          str([(x["phase"], x["ms"]) for x in cl]))
    check("note() attaches detail to the open checkpoint",
          cl[0]["detail"] == {"model": "stack+xgb@utility"}, str(cl[0]["detail"]))
    check("ready() closes the last checkpoint", all(x["ok"] for x in cl) and c.summary()["booted"])

    # A process that dies mid-boot (import error before uvicorn listens) must
    # still leave the phase it died in, marked failed — the atexit path.
    d = BootSequencer(journal=Journal(ring=64))
    d.checkpoint("detector", "load the model")
    died_in = d.abandon("planted: ImportError during load")
    dl = d.log()
    check("an unfinished checkpoint is recorded as failed at exit",
          died_in == "detector" and dl[0]["ok"] is False and "planted" in (dl[0]["error"] or ""),
          str(dl))
    # CONTROL: abandon() after a clean boot marks nothing, so the failure above
    # is about the unfinished phase rather than abandon() failing everything.
    check("CONTROL: abandon() after ready() marks nothing", c.abandon() is None)

    total = b.ready()
    check("ready() reports a real elapsed boot", total >= 0.03, f"{total:.3f}s")
    check("summary flips to booted", b.summary()["booted"])

    events = [(e.facility, e.event) for e in j.entries]
    check("every phase reached the journal",
          events.count(("boot", "phase")) == 2 and ("boot", "phase.failed") in events,
          str(events))
    check("boot ready is journalled", ("boot", "ready") in events)
    check("the boot journal chain verifies", j.verify()["ok"])

    failed = [n for n, ok, _ in _results if not ok]
    print("\n" + "=" * 64)
    if failed:
        print(f"  FAILED: {len(failed)} check(s): {', '.join(failed)}")
        print("=" * 64)
        return 1
    print("  ALL CHECKS PASSED — the boot log is the boot, not an animation")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
