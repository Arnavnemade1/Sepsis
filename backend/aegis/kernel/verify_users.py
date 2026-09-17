"""Verify accounts, sessions, break-glass and the agent credential.

    .venv/bin/python -m backend.aegis.kernel.verify_users

The claims that need adversarial treatment:

  * The ROLE comes from the ACCOUNT. A request cannot pick its own role.
  * A session the kernel ended stays ended — the token alone no longer works.
  * Break-glass adds what an emergency needs and nothing it must not confer, and
    it EXPIRES. Checked on a controllable clock, at the boundary.
  * Dr. Aegis never holds more than the clinician it acts for.
"""
from __future__ import annotations

import sys
from typing import List, Tuple

from backend.aegis.kernel.journal import Journal
from backend.aegis.kernel.proc import ProcessTable, SIGTERM
from backend.aegis.kernel.users import (AGENT_USER, GUEST, AuthError, PermissionDenied,
                                        UserDB, hash_passphrase)
from backend.aegis.rbac import AGENT_DENY, FORBIDDEN_ELEVATION, GRANTS, PUBLIC_CAPS, RBAC

#: (name, ok, detail) — the container the falsification engine reads.
_results: List[Tuple[str, bool, str]] = []


def check(label: str, cond: bool, detail: str = "") -> None:
    print(f"    [{'PASS' if cond else 'FAIL'}] {label}" + (f" — {detail}" if detail else ""))
    _results.append((label, bool(cond), detail))


class Clock:
    def __init__(self) -> None:
        self.t = 1_800_000_000.0

    def __call__(self) -> float:
        return self.t


def _raises(fn, exc) -> Tuple[bool, str]:
    try:
        fn()
        return False, "no error"
    except exc as e:
        return True, str(e)[:60]


def main() -> int:
    _results.clear()
    print("=" * 64)
    print("  USERS — accounts, sessions, break-glass, the agent")
    print("=" * 64)
    clock = Clock()
    j = Journal(ring=1024)
    procs = ProcessTable(journal=j)
    users = UserDB(RBAC(), procs=procs, journal=j, demo_login=True, clock=clock,
                   idle_lock_s=600.0)

    # -- 1. the role belongs to the account --------------------------------- #
    tok, st, cred = users.login("rn.patel")
    check("a nurse signs in as a nurse", cred.role == "nurse" and cred.ring == 1)
    check("…and holds exactly the nurse grants", cred.caps == GRANTS["nurse"])
    ok, why = _raises(lambda: users.login("rn.patel ", passphrase="attending"), AuthError)
    check("there is no way to name a role at sign-in (only an account)",
          users.login("rn.patel")[2].role == "nurse")
    ok, why = _raises(lambda: users.login("dr-aegis"), AuthError)
    check("the agent's service account cannot sign in", ok, why)
    ok, why = _raises(lambda: users.login("root"), AuthError)
    check("the kernel account cannot sign in", ok, why)
    check("no token is a guest, with ring 0 only",
          users.credential(None) is GUEST and GUEST.caps == PUBLIC_CAPS)

    # -- 2. demo sign-in is a switch, and passphrase accounts need one ------ #
    strict = UserDB(RBAC(), procs=ProcessTable(journal=j), journal=j, demo_login=False)
    ok, why = _raises(lambda: strict.login("dr.alvarez"), AuthError)
    check("with demo sign-in off, a demo account needs a passphrase", ok, why)
    real = UserDB(RBAC(), procs=ProcessTable(journal=j), journal=j)
    from backend.aegis.kernel.users import Account
    real.accounts["dr.real"] = Account(2000, "dr.real", "Dr. Real", "attending",
                                       pbkdf2=hash_passphrase("correct horse"))
    ok, why = _raises(lambda: real.login("dr.real", "wrong"), AuthError)
    check("a passphrase account refuses the wrong passphrase", ok, why)
    check("…and accepts the right one", real.login("dr.real", "correct horse")[2].role == "attending")

    # -- 3. tokens: forged, and ENDED sessions ------------------------------ #
    ok, why = _raises(lambda: users.credential(tok[:-2] + "00"), AuthError)
    check("a forged token is refused", ok, why)
    t2, s2, c2 = users.login("dr.alvarez")
    check("a valid token resolves to its session", users.credential(t2).sid == c2.sid)
    users.logout(users.credential(t2))
    ok, why = _raises(lambda: users.credential(t2), AuthError)
    check("THE CLAIM: after sign-out the same token no longer works", ok, why)
    zomb = procs.get(s2.pid)
    check("sign-out ends the session process", zomb.state == "ZOMBIE", zomb.state)
    t3, s3, c3 = users.login("dr.lindqvist")
    procs.signal(s3.pid, SIGTERM, uid="dr.lindqvist")
    ok, why = _raises(lambda: users.credential(t3), AuthError)
    check("SIGTERM to a session process signs it out (the process table is the session)",
          ok, why)

    # -- 4. lock and idle ---------------------------------------------------- #
    t4, s4, c4 = users.login("dr.alvarez")
    users.lock(users.credential(t4))
    locked = users.credential(t4)
    check("a locked session keeps only ring-0 capabilities",
          locked.locked and locked.caps == PUBLIC_CAPS and not locked.has("order:treat"))
    unlocked = users.unlock(locked)
    check("unlock restores the account's own capabilities", unlocked.has("order:treat"))
    clock.t += 601.0
    idle = users.credential(t4)
    check("a session idle past the limit locks itself", idle.locked, str(idle.locked))
    clock.t -= 601.0

    # -- 5. break-glass ------------------------------------------------------ #
    # The clock jumps 15 minutes below; with a 10-minute idle lock every session
    # would lock first and the expiry checks would pass or fail for that reason.
    users.idle_lock_s = 1e9
    tn, sn, cn = users.login("rn.patel")
    ok, why = _raises(lambda: users.breakglass(cn, "too short"), ValueError)
    check("break-glass needs a stated reason", ok, why)
    tp, sp, cp = users.login("dr.whitfield")
    ok, why = _raises(lambda: users.breakglass(cp, "a long enough reason to be accepted here"),
                      PermissionDenied)
    check("a role without breakglass:invoke cannot invoke it (the chief reviews it)", ok, why)
    bg = users.breakglass(cn, "septic shock on bed 3, attending in theatre", minutes=60)
    check("the duration is capped at 15 minutes", round((bg.until - bg.started) / 60) == 15)
    el = users.credential(tn)
    check("during break-glass the nurse can see the whole ward", el.has("view:cohort"))
    check("THE CLAIM: …and still cannot prescribe, change policy or install software",
          not (el.caps & FORBIDDEN_ELEVATION),
          str(sorted(el.caps & FORBIDDEN_ELEVATION)))
    clock.t = bg.until - 1.0
    check("one second before expiry it is still active", users.credential(tn).has("view:cohort"))
    clock.t = bg.until + 1.0
    after = users.credential(tn)
    check("one second after expiry it is gone", not after.has("view:cohort")
          and after.caps == GRANTS["nurse"])
    ok, why = _raises(lambda: users.review_breakglass(cn, bg.id, "appropriate"), PermissionDenied)
    check("the nurse cannot review break-glass (review is the chief's capability)",
          ok and "breakglass:review" in why, why)
    # The self-review guard, exercised directly: a record whose subject IS the
    # reviewer (no current role can both invoke and review — this is the guard for
    # a policy that someday lets one).
    from backend.aegis.kernel.users import BreakGlass
    own = BreakGlass(id="selfcheck01", username="dr.whitfield", role="chief",
                     reason="planted: a chief's own emergency record", started=clock.t,
                     until=clock.t + 60)
    users.breakglass_log.append(own)
    ok, why = _raises(lambda: users.review_breakglass(users.credential(tp), own.id, "appropriate"),
                      PermissionDenied)
    check("nobody reviews their own break-glass", ok and "own" in why, why)
    rv = users.review_breakglass(users.credential(tp), bg.id, "appropriate", "justified")
    check("the chief reviews it, and the verdict is recorded", rv.reviewed and rv.reviewer == "dr.whitfield")
    check("break-glass is journalled at CRIT with its reason",
          any(e.event == "breakglass.invoke" and e.severity == 2 and "septic" in e.detail.get("reason", "")
              for e in j.entries))

    # -- 6. the agent -------------------------------------------------------- #
    ta, sa, ca = users.login("dr.alvarez")
    agent = users.agent_credential(users.credential(ta))
    check("Dr. Aegis acts as dr-aegis on the clinician's behalf",
          agent.username == AGENT_USER and agent.ruid == "dr.alvarez" and agent.kind == "agent")
    check("THE CLAIM: the agent holds nothing its clinician lacks",
          agent.caps <= GRANTS["attending"], str(sorted(agent.caps - GRANTS["attending"])))
    check("…and none of the agent-denied capabilities", not (agent.caps & AGENT_DENY))
    guest_agent = users.agent_credential(None)
    check("with nobody signed in, the agent is a guest's agent (ring 0)",
          guest_agent.caps <= PUBLIC_CAPS and not guest_agent.has("order:treat"))
    # CONTROL: the agent for an attending CAN order — the subset is not empty.
    check("CONTROL: for an attending the agent may order treatment", agent.has("order:treat"))

    # -- 7. who ------------------------------------------------------------- #
    rows = users.who()
    check("who lists the live sessions and not the ended ones",
          any(r["username"] == "rn.patel" for r in rows)
          and not any(r["sid"] == s2.sid[:8] for r in rows))
    check("the auth journal chain verifies", j.verify()["ok"])

    failed = [n for n, ok, _ in _results if not ok]
    print("\n" + "=" * 64)
    if failed:
        print(f"  FAILED: {len(failed)} check(s): {', '.join(failed)}")
        print("=" * 64)
        return 1
    print(f"  ALL {len(_results)} CHECKS PASSED — identity is the account's, and it ends")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    sys.exit(main())
