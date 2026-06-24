"""AegisOS applications — capability-confined diagnostics running on the kernel.

The SDK (:mod:`backend.aegis.apps.sdk`) defines the contract; each ``*_app.py`` is
a real app that wraps an existing engine behind a declared capability manifest.
"""
from backend.aegis.apps.sdk import (
    AegisApp,
    AppContext,
    AppManifest,
    CapabilityViolation,
    EGRESS_CLASSES,
    LANE_CLASSES,
    TickResult,
    tick_app,
)

__all__ = [
    "AegisApp", "AppContext", "AppManifest", "CapabilityViolation",
    "EGRESS_CLASSES", "LANE_CLASSES", "TickResult", "tick_app",
]
