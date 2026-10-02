"""Room units: every occupied room tracks its patient as one unit.

A room unit joins four things the rest of the OS kept apart: the patient's prior
record (problems, medications, allergies, code status, earlier encounters and the
baselines measured there), the vitals the room is producing now, a library of
published early-warning and safety detectors that read both, and the pages the
findings raise. See ``monitor.RoomMonitor``.
"""
from backend.aegis.rooms.monitor import RoomMonitor

__all__ = ["RoomMonitor"]
