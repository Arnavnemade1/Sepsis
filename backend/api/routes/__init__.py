"""AegisOS REST routers, one per domain. Order is the order they are mounted."""
from backend.api.routes import (  # noqa: F401
    health,
    research,
    live,
    patients,
    causal,
    pharmacy,
    governance,
    devices,
    kernel,
    system,
    ai,
)

ROUTERS = [health, research, live, patients, causal, pharmacy, governance, devices, kernel, system, ai]
