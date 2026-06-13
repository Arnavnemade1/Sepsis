"""Aegis bio-kernel — the bare-metal layer stack of AegisOS.

This subpackage hosts the systems-software substrate that the higher-level
clinical runtime (``backend/aegis/core.py``, ``os.py``) is reframed on top of
for the ISEF Systems Software track. Each layer is a real, inspectable artifact:

    Layer 1  devbio       /dev/bio unified device-driver bus + inline Kalman
    Layer 2  memfabric    (planned) zero-copy lockless ring-buffer memory fabric
    Layer 3  scheduler    (planned) multi-tenant cognitive cost-function scheduler
    Layer 4  neurosym     (planned) formally-verified neurosymbolic safety runtime

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
    DRIVER_REGISTRY,
    provision_node,
)

__all__ = [
    # Layer 1
    "BioBlock", "BioDevice", "ChannelSpec", "DevBio", "DeviceDescriptor",
    "KalmanScalar", "OpticalRPPGDriver", "ThermalGridDriver",
    "BedsideTelemetryDriver", "DRIVER_REGISTRY", "provision_node",
]
