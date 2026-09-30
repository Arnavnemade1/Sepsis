"""The AegisOS paging OS: alerts from nurses and from the sepsis detector, triaged,
routed to the right clinician, certified against the paging policy by Z3, and
escalated automatically. See engine.py for the pipeline."""
from backend.aegis.paging.engine import PagingOS

__all__ = ["PagingOS"]
