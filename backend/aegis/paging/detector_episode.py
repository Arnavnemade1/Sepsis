"""The detector's paging episode, as a pure state machine.

A monitored bed's risk index moves every hour. Paging on every hour above the line
would page the same patient over and over, so the paging OS keeps one *episode* per
room:

  * the index reaches the alarm line (``DETECTOR_ALARM``) -> an episode opens and a
    page goes out, unless the previous episode's page is still being handled, in
    which case the re-crossing is written on that page (one patient, one page);
  * Sepsis-3 shock criteria while an episode is open raise it to P1: the open page
    is upgraded, or a new page goes out if the old one is already closed;
  * the episode closes only after three consecutive readings below the clear line
    (``DETECTOR_CLEAR``): hysteresis, so a score hovering at the line does not
    open and close an episode every hour.

This module holds only the decision. ``PagingOS._episode`` executes it on live
pages, and ``backend.ml.system_replay`` executes it on held-out patients. Both
import this one function, so the replay measures the rule the product runs, not a
copy of it that can drift.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

from backend.aegis.paging.triage import DETECTOR_ALARM, DETECTOR_CLEAR

#: consecutive readings below the clear line before an episode closes
CLEAR_READINGS = 3

# actions the caller executes
PAGE = "page"                         # open a new detector page
RECROSS = "recross"                   # note the re-crossing on the still-active page
RECROSS_UPGRADE = "recross_upgrade"   # ... and raise that page to the new priority
UPGRADE = "upgrade"                   # raise the episode's open page
REPAGE = "repage"                     # higher priority, but the old page is closed: page again
CLOSE = "close"                       # the episode ended
ACTIONS = (PAGE, RECROSS, RECROSS_UPGRADE, UPGRADE, REPAGE, CLOSE)


def new_episode() -> Dict[str, Any]:
    return {"open": False, "priority": None, "page": None, "below": 0}


def step(ep: Optional[Dict[str, Any]], risk: float, priority: int,
         prior_active_priority: Callable[[], Optional[int]],
         page_active: Callable[[], bool]) -> Tuple[Dict[str, Any], Optional[str]]:
    """Advance one room's episode by one reading.

    ``priority`` is what the detector triage would give this reading (1 = shock
    criteria met, else 2); it is ignored below the alarm line.
    ``prior_active_priority()`` returns the priority of this room's detector page
    that is still being handled, or None. It is called only when a closed episode
    re-crosses the line, so the caller's lookup costs nothing otherwise.
    ``page_active()`` says whether the open episode's own page is still being
    handled; it is called only on a priority upgrade.

    Returns the next episode (a new dict; the input is not mutated) and the action
    for the caller to execute, or None. The caller fills in ``page`` and
    ``opened_at``, which are about its pages, not about this decision.
    """
    ep = dict(ep) if ep else new_episode()
    action: Optional[str] = None
    if risk >= DETECTOR_ALARM:
        if not ep["open"]:
            prior = prior_active_priority()
            if prior is not None:
                action = RECROSS_UPGRADE if priority < prior else RECROSS
                ep = {"open": True, "priority": min(priority, prior), "page": ep.get("page"), "below": 0}
            else:
                action = PAGE
                ep = {"open": True, "priority": priority, "page": None, "below": 0}
        elif priority < (ep.get("priority") or 4):
            ep["priority"] = priority
            action = UPGRADE if page_active() else REPAGE
        ep["below"] = 0
    elif ep["open"] and risk < DETECTOR_CLEAR:
        ep["below"] = ep.get("below", 0) + 1
        if ep["below"] >= CLEAR_READINGS:
            ep["open"] = False
            action = CLOSE
    return ep, action
