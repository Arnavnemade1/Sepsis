"""Aegis bio-kernel — the bare-metal layer stack of AegisOS.

This subpackage hosts the systems-software substrate that the higher-level
clinical runtime (``backend/aegis/core.py``, ``os.py``) is reframed on top of
for the ISEF Systems Software track. Each layer is a real, inspectable artifact:

    Layer 1  devbio       /dev/bio unified device-driver bus + inline Kalman
    Layer 2  memfabric    zero-copy lockless ring-buffer memory fabric
    Layer 3  scheduler    multi-tenant cognitive cost-function scheduler + cluster
    Layer 4  neurosym     formally-verified neurosymbolic egress cage (Z3)

Built incrementally — see the systems-architecture-pivot direction.
"""
from backend.aegis.kernel.devbio import (
    BioBlock,
    BioDevice,
    ChannelSpec,
    DevBio,
    DeviceDescriptor,
    KalmanScalar,
    OpticalRPPGDriver,
    ThermalGridDriver,
    BedsideTelemetryDriver,
    LabsDriver,
    DRIVER_REGISTRY,
    provision_node,
)
from backend.aegis.kernel.fusion import BedProfile, FusionEngine
from backend.aegis.kernel.bundle import SepsisBundleEngine
from backend.aegis.kernel.memfabric import (
    FabricPump,
    LocklessRing,
    MemFabric,
    MemoryArena,
    ReadCursor,
    Region,
)
from backend.aegis.kernel.scheduler import (
    BioCluster,
    CognitiveScheduler,
    NodeState,
    ScheduleDecision,
    risk_proxy,
)
from backend.aegis.kernel.neurosym import (
    ACTUATOR,
    ALERT,
    NOTE,
    PROTOCOL,
    NeuroSymbolicRuntime,
    SafetyCertificate,
)
from backend.aegis.kernel.ebpf import (
    Instr,
    Interval,
    VerdictReport,
    Verifier,
    Violation,
    VM,
    assemble,
)
from backend.aegis.kernel.ebpfdrv import (
    DriverFault,
    DriverRejected,
    EbpfDevice,
    load_ebpf_driver,
    scatter_program,
)
from backend.aegis.kernel.supervisor import (
    DriverSupervisor,
    FaultRecord,
    SupervisedDevice,
)

__all__ = [
    # Layer 1
    "BioBlock", "BioDevice", "ChannelSpec", "DevBio", "DeviceDescriptor",
    "KalmanScalar", "OpticalRPPGDriver", "ThermalGridDriver",
    "BedsideTelemetryDriver", "LabsDriver", "DRIVER_REGISTRY", "provision_node",
    # Fusion (Sepsis Ontology) + Hour-1 bundle
    "FusionEngine", "BedProfile", "SepsisBundleEngine",
    # Layer 2
    "MemFabric", "MemoryArena", "LocklessRing", "ReadCursor", "Region",
    "FabricPump",
    # Layer 3
    "CognitiveScheduler", "NodeState", "ScheduleDecision", "BioCluster",
    "risk_proxy",
    # Layer 4
    "NeuroSymbolicRuntime", "SafetyCertificate", "ALERT", "PROTOCOL",
    "ACTUATOR", "NOTE",
    # Aegis-eBPF: untrusted-driver static verifier + sandbox (input safety)
    "Verifier", "VerdictReport", "Violation", "Interval", "Instr", "VM", "assemble",
    "EbpfDevice", "load_ebpf_driver", "scatter_program",
    "DriverRejected", "DriverFault",
    "DriverSupervisor", "SupervisedDevice", "FaultRecord",
]
