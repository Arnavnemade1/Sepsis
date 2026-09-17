"""Device Pairing & Management — the registry behind the hardware hub.

This is the software side of the (future) AegisOS sensor fleet: wrist wearables,
optical biosensor units, and bedside gateways. It models device identity,
lifecycle, pairing, patient assignment, health (battery / signal / firmware), a
calibration gate, OTA, and an audit log.

IMPORTANT — this runs in SIMULATED mode until real hardware exists. Every device
here is a mock, flagged ``simulated: True``. The point of building it now is to
nail the UX and lock the exact contract the firmware must implement; we never
present a simulated device as a connected physical one. When real telemetry
arrives (via /aegis/ingest), a device only drives detection once it is paired,
assigned, AND passes the calibration / signal-quality gate — otherwise its
channels stay honest "no data", consistent with the rest of AegisOS.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional

# Lifecycle states (ordinal-ish; the UI renders each distinctly).
STATES = ["unprovisioned", "pairing", "provisioned", "assigned",
          "streaming", "degraded", "offline", "decommissioned"]

# Device classes and what each one measures.
TYPES = {
    "wearable": {"model": "Aegis Band", "channels": ["hr", "hrv", "spo2", "skin_temp", "resp_rate", "motion"],
                 "mains": False},
    "optical":  {"model": "Aegis Optical", "channels": ["hr", "resp_rate", "perfusion_index", "stress_vector"],
                 "mains": False},
    "gateway":  {"model": "Aegis Gateway", "channels": ["uplink"], "mains": True},
}
FW_LATEST = "1.5.0"
_FW_OLD = ["1.3.4", "1.4.1", "1.4.2", "1.5.0"]


def _hexid(prefix: str) -> str:
    return f"AEG-{prefix}-{''.join(random.choice('0123456789ABCDEF') for _ in range(4))}"


@dataclass
class Device:
    id: str
    type: str
    model: str
    serial: str
    firmware: str
    channels: List[str]
    mains: bool
    status: str = "provisioned"
    battery: Optional[int] = 100          # None for mains-powered
    signal_quality: float = 0.0           # 0..1 (live link quality)
    rssi: int = -60                       # dBm
    calibrated: bool = False
    calibration_quality: float = 0.0
    assigned_bed: Optional[str] = None
    assigned_label: Optional[str] = None  # patient display name (frontend-resolved)
    last_seen: float = field(default_factory=time.time)
    paired_at: Optional[float] = None
    assigned_at: Optional[float] = None
    simulated: bool = True

    def health(self) -> str:
        """Roll-up health for the fleet grid."""
        if self.status in ("offline", "decommissioned"):
            return "offline"
        if self.status == "degraded":
            return "degraded"
        if self.battery is not None and self.battery <= 15:
            return "degraded"
        if self.status == "streaming" and self.signal_quality < 0.4:
            return "degraded"
        if self.status in ("streaming", "assigned", "provisioned", "pairing"):
            return "ok"
        return "idle"

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["health"] = self.health()
        d["fw_latest"] = FW_LATEST
        d["fw_outdated"] = self.firmware != FW_LATEST
        d["streams"] = self.status == "streaming" and self.calibrated and self.signal_quality >= 0.4
        return d


class DeviceRegistry:
    """In-memory fleet of simulated devices + a discoverable (unpaired) pool."""

    def __init__(self, seed: int = 11) -> None:
        self._rng = random.Random(seed)
        random.seed(seed)
        self.devices: Dict[str, Device] = {}
        self.discoverable: Dict[str, Device] = {}   # nearby, not yet paired
        self.audit: List[Dict[str, Any]] = []
        self._last_tick = time.time()
        self._seed_fleet()

    # ---- seeding -------------------------------------------------------- #
    def _mk(self, dtype: str, fw: Optional[str] = None) -> Device:
        spec = TYPES[dtype]
        pfx = {"wearable": "W", "optical": "O", "gateway": "G"}[dtype]
        return Device(
            id=_hexid(pfx), type=dtype, model=spec["model"],
            serial=f"{pfx}{self._rng.randint(10**7, 10**8 - 1)}",
            firmware=fw or self._rng.choice(_FW_OLD),
            channels=list(spec["channels"]), mains=spec["mains"],
            battery=None if spec["mains"] else self._rng.randint(35, 100),
        )

    def _seed_fleet(self) -> None:
        # a couple of bedside gateways, already provisioned
        for _ in range(2):
            d = self._mk("gateway", FW_LATEST)
            d.status, d.signal_quality, d.calibrated = "streaming", 0.97, True
            d.paired_at = time.time() - self._rng.randint(3600, 86400)
            self.devices[d.id] = d
        # a spread of wearables / optical units in assorted states
        plan = [("wearable", "streaming"), ("wearable", "streaming"), ("wearable", "provisioned"),
                ("optical", "streaming"), ("optical", "degraded"), ("wearable", "offline"),
                ("optical", "provisioned")]
        for dtype, st in plan:
            d = self._mk(dtype)
            d.status = st
            d.paired_at = time.time() - self._rng.randint(600, 172800)
            if st in ("streaming", "degraded"):
                d.calibrated = True
                d.calibration_quality = round(self._rng.uniform(0.8, 0.99), 2)
                d.signal_quality = round(self._rng.uniform(0.45, 0.95) if st == "streaming"
                                         else self._rng.uniform(0.18, 0.38), 2)
            self.devices[d.id] = d
        # nearby unpaired devices, ready to discover + pair
        for dtype in ("wearable", "optical", "wearable"):
            d = self._mk(dtype)
            d.status = "unprovisioned"
            d.rssi = self._rng.randint(-82, -52)
            self.discoverable[d.id] = d

    # ---- audit ---------------------------------------------------------- #
    def _log(self, event: str, device: str, detail: str = "", actor: str = "operator") -> None:
        self.audit.append({"ts": time.time(), "event": event, "device": device,
                           "detail": detail, "actor": actor})
        self.audit = self.audit[-200:]
        from backend.aegis.kernel.journal import KJOURNAL, INFO
        # The fleet is SIMULATED (see the module docstring) and says so in the
        # record, so a reader never mistakes a modelled dropout for a real one.
        KJOURNAL.append("device", event, severity=INFO, uid=str(actor),
                        device=device, detail=detail, simulated=True)

    # ---- liveness simulation ------------------------------------------- #
    def tick(self) -> None:
        """Advance the simulated fleet: battery drain, signal jitter, the odd
        dropout/recovery. Called lazily on each snapshot from elapsed wall-time."""
        now = time.time()
        dt = now - self._last_tick
        if dt < 0.5:
            return
        self._last_tick = now
        for d in self.devices.values():
            if d.status in ("offline", "decommissioned", "unprovisioned"):
                continue
            if d.battery is not None:
                # ~1% per few minutes; streaming drains faster
                drain = dt / 240.0 * (1.6 if d.status == "streaming" else 0.7)
                d.battery = max(0, int(round(d.battery - drain)))
            if d.status in ("streaming", "degraded"):
                d.signal_quality = round(min(1.0, max(0.05,
                    d.signal_quality + self._rng.uniform(-0.04, 0.04))), 2)
                d.rssi = max(-92, min(-40, d.rssi + self._rng.randint(-2, 2)))
                d.last_seen = now
                # promote/demote between streaming and degraded on link quality
                if d.status == "streaming" and (d.signal_quality < 0.34 or (d.battery or 100) <= 8):
                    d.status = "degraded"
                    self._log("degraded", d.id, "signal/battery below gate")
                elif d.status == "degraded" and d.signal_quality > 0.5 and (d.battery or 100) > 12:
                    d.status = "streaming"
                    self._log("recovered", d.id, "link restored")

    # ---- queries -------------------------------------------------------- #
    def fleet(self) -> Dict[str, Any]:
        self.tick()
        items = [d.to_dict() for d in self.devices.values()]
        order = {s: i for i, s in enumerate(["streaming", "degraded", "assigned",
                                             "provisioned", "pairing", "offline", "decommissioned"])}
        items.sort(key=lambda x: (order.get(x["status"], 9), x["type"]))
        counts = {"total": len(items),
                  "streaming": sum(1 for d in items if d["status"] == "streaming"),
                  "degraded": sum(1 for d in items if d["health"] == "degraded"),
                  "offline": sum(1 for d in items if d["status"] == "offline"),
                  "unassigned": sum(1 for d in items if d["assigned_bed"] is None
                                    and d["status"] not in ("offline", "decommissioned"))}
        return {"simulated": True, "counts": counts, "devices": items,
                "fw_latest": FW_LATEST}

    def discover(self) -> Dict[str, Any]:
        """Nearby unpaired devices (a fresh scan may surface a new one)."""
        self.tick()
        if self._rng.random() < 0.25 and len(self.discoverable) < 5:
            d = self._mk(self._rng.choice(["wearable", "optical"]))
            d.status = "unprovisioned"; d.rssi = self._rng.randint(-84, -50)
            self.discoverable[d.id] = d
        items = sorted((d.to_dict() for d in self.discoverable.values()),
                       key=lambda x: -x["rssi"])
        return {"simulated": True, "devices": items}

    def get(self, did: str) -> Optional[Device]:
        return self.devices.get(did) or self.discoverable.get(did)

    # ---- actions -------------------------------------------------------- #
    def pair(self, did: str) -> Dict[str, Any]:
        d = self.discoverable.pop(did, None)
        if not d:
            return {"status": "error", "detail": "device not in range / already paired"}
        d.status = "provisioned"; d.firmware = d.firmware or FW_LATEST
        d.paired_at = time.time(); d.last_seen = time.time()
        self.devices[d.id] = d
        self._log("paired", d.id, f"{d.model} provisioned")
        return {"status": "ok", "device": d.to_dict()}

    def assign(self, did: str, bed: str, label: str = "") -> Dict[str, Any]:
        d = self.devices.get(did)
        if not d:
            return {"status": "error", "detail": "unknown device"}
        d.assigned_bed = bed; d.assigned_label = label or bed
        d.assigned_at = time.time()
        d.status = "streaming" if d.calibrated and d.signal_quality >= 0.4 else "assigned"
        if d.status == "assigned" and d.signal_quality == 0:
            d.signal_quality = round(self._rng.uniform(0.5, 0.9), 2)
        self._log("assigned", d.id, f"-> {label or bed}")
        return {"status": "ok", "device": d.to_dict()}

    def unassign(self, did: str) -> Dict[str, Any]:
        d = self.devices.get(did)
        if not d:
            return {"status": "error", "detail": "unknown device"}
        prev = d.assigned_label
        d.assigned_bed = d.assigned_label = d.assigned_at = None
        d.status = "provisioned"
        self._log("unassigned", d.id, f"from {prev}")
        return {"status": "ok", "device": d.to_dict()}

    def action(self, did: str, kind: str) -> Dict[str, Any]:
        d = self.devices.get(did)
        if not d:
            return {"status": "error", "detail": "unknown device"}
        kind = kind.lower()
        if kind == "calibrate":
            d.calibrated = True
            d.calibration_quality = round(self._rng.uniform(0.85, 0.99), 2)
            if d.assigned_bed and d.signal_quality >= 0.4:
                d.status = "streaming"
            self._log("calibrated", d.id, f"q={d.calibration_quality}")
        elif kind == "reboot":
            d.status = "offline"; d.signal_quality = 0.0
            self._log("reboot", d.id, "rebooting")
            # comes back online on the next assign/calibrate or manual wake
            d.status = "provisioned" if d.assigned_bed is None else "assigned"
        elif kind == "ota":
            d.firmware = FW_LATEST
            self._log("ota", d.id, f"updated to {FW_LATEST}")
        elif kind == "wake":
            d.status = "provisioned" if d.assigned_bed is None else "assigned"
            d.last_seen = time.time(); d.signal_quality = round(self._rng.uniform(0.5, 0.9), 2)
            self._log("wake", d.id, "back online")
        elif kind == "decommission":
            d.status = "decommissioned"; d.assigned_bed = d.assigned_label = None
            self._log("decommission", d.id, "retired")
        else:
            return {"status": "error", "detail": f"unknown action '{kind}'"}
        return {"status": "ok", "device": d.to_dict()}

    def audit_log(self, limit: int = 40) -> List[Dict[str, Any]]:
        return list(reversed(self.audit[-limit:]))
