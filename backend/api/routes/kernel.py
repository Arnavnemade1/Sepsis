"""The kernel plane: the eBPF-verified device drivers, aegisd services, packages, /proc and HDKernel.

Routes only. Shared runtime state lives in backend/aegis/server.py, the
composition root that includes this router."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter

from backend.api.schemas import (
    HDKernelRunIn,
    KernelFaultIn,
    PkgIn,
    VerifyIn,
)
from backend.aegis.server import (
    AVAILABLE_PKGS,
    EXEC,
    PROCS,
    SERVICE_BED,
    _KPOLICIES,
    _KWORKLOADS,
    _broadcast_cluster,
    _cluster_payload,
    _hdk_last,
    _hdk_state,
    _kernel_plane,
    _pkg_catalogue,
    _resolve_ebpf_bed,
    _run_kworkload,
    aegisd,
    cluster,
    kernel_devices,
    pkg_registry,
)

router = APIRouter()


@router.get("/aegis/kernel/devices")
def kernel_device_status() -> Dict[str, Any]:
    """The verified device plane: the real eBPF-driven cohort beds + their
    supervisor state, the live verifier counters, and the driver catalogue."""
    return _kernel_plane(full=True)


@router.post("/aegis/kernel/verify")
def kernel_verify(payload: VerifyIn) -> Dict[str, Any]:
    """Submit a driver to the LIVE static verifier. A catalogue ``name`` or raw
    ``source`` is statically analyzed; the verdict (ACCEPT + proof, or REJECT +
    the exact offending instruction and violated invariant) is computed now."""
    if payload.name:
        return kernel_devices.verify_named(payload.name)
    if payload.source:
        return kernel_devices.verify_source(payload.source, payload.win_len)
    return {"ok": False, "stage": "input", "error": "provide 'name' or 'source'"}


@router.post("/aegis/kernel/fault")
async def kernel_fault(payload: KernelFaultIn) -> Dict[str, Any]:
    """Kill a REAL patient bed's optical sensor → watch the supervisor quarantine
    the verified driver, fail over to its backup, and keep the patient monitored
    while every other bed keeps ticking."""
    bid = _resolve_ebpf_bed(payload.slot)
    if bid is None:
        return {"status": "error",
                "detail": f"{payload.slot!r} is not an eBPF-driven bed "
                          f"(verified beds: {', '.join(getattr(cluster, 'ebpf_beds', {}))})"}
    res = EXEC.call(cluster.inject_driver_fault, bid, payload.reason)
    res = {"status": "ok", "bed": bid, "name": cluster.names.get(bid, bid), **(res or {})}
    await _broadcast_cluster(_cluster_payload())
    return res


@router.get("/aegis/services")
def services_status() -> Dict[str, Any]:
    """The managed app/service plane: every service's capabilities, health, and
    the supervisor's event log."""
    s = aegisd.status()
    s["bed"] = SERVICE_BED
    return s


@router.post("/aegis/services/{name}/{action}")
async def service_control(name: str, action: str) -> Dict[str, Any]:
    """systemctl-style control: start | stop | restart a service."""
    act = action.lower()
    ok = (aegisd.stop(name) if act == "stop" else
          aegisd.restart(name) if act in ("restart", "start") else None)
    if ok is None:
        return {"status": "error", "detail": f"unknown action {action!r}"}
    if not ok:
        return {"status": "error", "detail": f"unknown service {name!r}"}
    await _broadcast_cluster(_cluster_payload())
    return {"status": "ok", "service": name, "action": act,
            "state": aegisd.services[name].state}


@router.get("/aegis/pkg")
def pkg_list() -> Dict[str, Any]:
    """Installed packages + the installable catalogue + the install audit log."""
    lst = pkg_registry.list()
    lst["available"] = _pkg_catalogue()
    return lst


@router.post("/aegis/pkg/install")
async def pkg_install(payload: PkgIn) -> Dict[str, Any]:
    """Install a catalogue package through the verify-then-register gate. On
    success it becomes a live, supervised service."""
    factory = AVAILABLE_PKGS.get(payload.name)
    if factory is None:
        return {"ok": False, "stage": "lookup",
                "reason": f"no such package {payload.name!r}",
                "available": list(AVAILABLE_PKGS)}
    res = pkg_registry.install(factory()).to_dict()
    await _broadcast_cluster(_cluster_payload())
    return res


@router.post("/aegis/pkg/uninstall")
async def pkg_uninstall(payload: PkgIn) -> Dict[str, Any]:
    ok = pkg_registry.uninstall(payload.name)
    await _broadcast_cluster(_cluster_payload())
    return {"ok": ok, "name": payload.name}


@router.get("/aegis/hdkernel")
def hdkernel_state() -> Dict[str, Any]:
    """The live process plane (Layer 3c): last run's summary/ps/top + the menu."""
    return {**_hdk_last(),
            "workloads": list(_KWORKLOADS), "policies": list(_KPOLICIES)}


@router.post("/aegis/hdkernel/run")
def hdkernel_run(payload: HDKernelRunIn) -> Dict[str, Any]:
    """Run a named workload under a chosen scheduling policy on the live kernel and
    return its process table — a genuine 'launch a program on the OS and inspect
    it'. Swapping ``policy`` visibly changes turnaround / context switches."""
    wl = payload.workload if payload.workload in _KWORKLOADS else "mixed"
    pol = payload.policy if payload.policy in _KPOLICIES else "HDSched"
    # A bounded program run as a JOB in the real process table; the synthetic
    # tasks it schedules live inside its own HDKernel, not in this table.
    with PROCS.job(f"hdkernel {wl} ({pol})", uid="operator", workload=wl, policy=pol):
        _hdk_state["last"] = _run_kworkload(wl, pol)
    return {**_hdk_state["last"],
            "workloads": list(_KWORKLOADS), "policies": list(_KPOLICIES)}


@router.post("/aegis/kernel/recover")
async def kernel_recover(payload: KernelFaultIn) -> Dict[str, Any]:
    """Recover a quarantined bed: re-verify and re-arm its primary optical driver."""
    bid = _resolve_ebpf_bed(payload.slot)
    if bid is None:
        return {"status": "error", "detail": f"{payload.slot!r} is not an eBPF-driven bed"}
    res = EXEC.call(cluster.recover_driver, bid)
    res = {"status": "ok", "bed": bid, "name": cluster.names.get(bid, bid), **(res or {})}
    await _broadcast_cluster(_cluster_payload())
    return res

