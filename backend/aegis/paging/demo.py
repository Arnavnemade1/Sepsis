"""Demo traffic: synthetic nurse alerts so an empty ward still shows the system work.

Everything produced here is marked ``synthetic`` and the UI labels it. It exists
because a paging system with no one paging is indistinguishable from a broken
one on a screen. The detector's alerts are NOT here: those come from the real
replayed CinC stays crossing the alarm line.
"""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

from backend.aegis.paging.hospital import ROOMS, ZONES

# (template, zone kinds or unit ids it fits[, orders the room must have]). {room} is filled with
# an occupied room. Texts state findings, not diagnoses, so they cannot contradict the room's
# record; one that names a drug only lands in a room whose record has that order.
_IV_ABX = ("ceftriaxone", "cefazolin", "piperacillin-tazobactam", "cefepime", "azithromycin")
SCENARIOS: List[Tuple] = [
    ("Room {room}: temp 38.9, HR 118, RR 24, new confusion.", ("ward", "obs")),
    ("{room} c/o chest pain 7/10, diaphoretic, hx CAD. BP 148/92, HR 104.", ("stepdown", "ward", "obs")),
    ("Fall in {room}, no head strike, alert and oriented. Requesting eval.", ("ward", "obs")),
    ("Pt in {room} requesting something to help sleep.", ("ward", "obs", "stepdown")),
    ("{room} SpO2 87% on 2L, RR 28, crackles bilaterally.", ("ward", "stepdown")),
    ("{room} post-op pain 9/10 despite PRN hydromorphone.", ("pacu", "ward"), ("hydromorphone PCA",)),
    ("Infusion reaction in {room}: hives and itching during the IV antibiotic. BP 124/80.", ("ward", "obs"), _IV_ABX),
    ("{room} new afib RVR, HR 148, BP 102/64, asymptomatic.", ("stepdown",)),
    ("{room} blood glucose 42, diaphoretic, responding to voice.", ("ward", "stepdown", "obs", "ed"),
     ("insulin glargine", "insulin infusion")),
    ("{room} BP 86/50, HR 124, temp 39.4, urine output dropping.", ("ward", "obs")),
    ("Family in {room} asking to speak with the physician about discharge plans.", ("ward", "obs", "stepdown")),
    ("{room} potassium 5.9 on morning labs, no ECG changes yet.", ("ward", "stepdown")),
    ("{room} wound dehiscence, small amount of serous drainage.", ("pacu", "ward")),
    ("{room} new facial droop and slurred speech, last known well 40 min ago.", ("neuro", "stepdown", "5e", "5w")),
    ("{room} melena x2, HR 112, BP 104/66.", ("ward", "obs", "stepdown"),
     ("enoxaparin", "heparin infusion", "apixaban", "aspirin")),
    ("{room} febrile neutropenia, temp 38.6, ANC 300. Awaiting cultures.", ("onc",), ("cefepime",)),
    ("{room} medication question: patient refusing evening metoprolol.", ("stepdown", "ward"), ("metoprolol",)),
    ("{room} laceration to forearm from fall, 4 cm, bleeding controlled.", ("ed",)),
    ("{room} abdominal pain 8/10, rigid abdomen, guarding.", ("ed", "obs")),
    ("{room} seizure lasting 2 minutes, now post-ictal, GCS 13.", ("ed", "ward")),
]

# history for the feed at boot: (minutes ago, text, status)
HISTORY: List[Tuple[int, str, str]] = [
    (118, "Perianal abscess in {room}, needs I&D in the next 2 h.", "declined"),
    (96, "Grade 1 infusion reaction in {room}. Diphenhydramine given, resolved.", "resolved"),
    (74, "Single syncopal episode in {room}, history of vasovagal.", "resolved"),
    (51, "6 cm laceration in {room}, no neurovascular deficit. Needs sutures and tetanus.", "resolved"),
    (33, "Post-op nausea in {room}, ondansetron not effective.", "resolved"),
    (19, "{room}: family requesting update on MRI results.", "resolved"),
]


def pick_scenario(rng: random.Random, occupied: Dict[str, Dict]) -> Optional[Tuple[str, str]]:
    """(text, room) for the next synthetic alert, placed in an occupied room that fits."""
    for _ in range(12):
        template, kinds, *needs = rng.choice(SCENARIOS)
        orders = needs[0] if needs else ()
        # a scenario names zone kinds ("ward") or specific units ("onc"), and sometimes an order
        rooms = [r for r in occupied if (ZONES[ROOMS[r].zone].kind in kinds or ROOMS[r].zone in kinds)
                 and (not orders or set(orders) & set(occupied[r].get("medications") or []))]
        if rooms:
            room = rng.choice(rooms)
            return template.format(room=room), room
    return None
