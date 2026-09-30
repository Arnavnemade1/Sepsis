"""The sensor fleet (pair, assign, act) and the PPG morphology pathway.

Routes only. Shared runtime state lives in backend/aegis/server.py, the
composition root that includes this router."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from backend.api.schemas import (
    DeviceActionIn,
    DeviceAssignIn,
    VasodilationIn,
)
from backend.aegis.server import (
    device_registry,
    ppg_monitor,
)

router = APIRouter()


@router.get("/aegis/devices")
def devices_fleet() -> Dict[str, Any]:
    """The simulated sensor fleet — every device, health-rolled-up for the hub."""
    return device_registry.fleet()


@router.get("/aegis/devices/discover")
def devices_discover() -> Dict[str, Any]:
    """Nearby unpaired devices (simulated BLE scan) available to pair."""
    return device_registry.discover()


@router.get("/aegis/devices/audit")
def devices_audit() -> Dict[str, Any]:
    """Pairing / assignment audit trail (device↔patient binding is PHI)."""
    return {"events": device_registry.audit_log()}


@router.get("/aegis/devices/{device_id}")
def device_detail(device_id: str) -> Dict[str, Any]:
    d = device_registry.get(device_id)
    return d.to_dict() if d else {"status": "error", "detail": "unknown device"}


@router.post("/aegis/devices/{device_id}/pair")
def device_pair(device_id: str) -> Dict[str, Any]:
    return device_registry.pair(device_id)


@router.post("/aegis/devices/{device_id}/assign")
def device_assign(device_id: str, payload: DeviceAssignIn) -> Dict[str, Any]:
    return device_registry.assign(device_id, payload.bed_id, payload.label)


@router.post("/aegis/devices/{device_id}/unassign")
def device_unassign(device_id: str) -> Dict[str, Any]:
    return device_registry.unassign(device_id)


@router.post("/aegis/devices/{device_id}/action")
def device_action(device_id: str, payload: DeviceActionIn) -> Dict[str, Any]:
    return device_registry.action(device_id, payload.action)


@router.get("/aegis/ppg/morphology")
def ppg_morphology() -> Dict[str, Any]:
    """The live PPG-morphology vascular-tone monitor: current waveform window +
    dicrotic-notch / b-a features extracted in real time from a real BIDMC PPG."""
    if ppg_monitor is None:
        return {"available": False}
    return ppg_monitor.state()


@router.post("/aegis/ppg/vasodilation")
async def ppg_vasodilation(payload: VasodilationIn) -> Dict[str, Any]:
    """Dial the vasodilation model and watch the dicrotic notch flatten + the
    vascular-tone metrics collapse in real time (the distributive-shock signature)."""
    if ppg_monitor is None:
        return {"available": False}
    ppg_monitor.set_vasodilation(payload.alpha)
    return ppg_monitor.state()

