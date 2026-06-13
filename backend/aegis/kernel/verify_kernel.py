"""Run every Aegis bio-kernel layer verification in one shot.

    .venv/bin/python -m backend.aegis.kernel.verify_kernel
"""
from __future__ import annotations

from backend.aegis.kernel import verify_devbio, verify_memfabric
from backend.aegis.kernel import verify_neurosym, verify_scheduler, verify_fusion


def main() -> None:
    print("#" * 64)
    print("#  AEGIS BIO-KERNEL — full layer-stack verification (L1–L4)")
    print("#" * 64)
    verify_devbio.main()       # L1 /dev/bio device-driver bus
    print()
    verify_memfabric.main()    # L2 zero-copy memory fabric
    print()
    verify_scheduler.main()    # L3 multi-tenant cognitive scheduler
    print()
    verify_fusion.main()       # Core Data Fusion Engine (Sepsis Ontology)
    print()
    verify_neurosym.main()     # L4 neurosymbolic egress cage
    print("\n" + "#" * 64)
    print("#  BIO-KERNEL L1–L4: ALL LAYERS VERIFIED")
    print("#" * 64)


if __name__ == "__main__":
    main()
