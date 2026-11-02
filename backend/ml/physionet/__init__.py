"""The PhysioNet data platform: credentialed data in, nothing identifiable out.

``catalog``   what exists, at which version, under which access class
``auth``      where credentials come from (never argv, never disk, never a log)
``fetch``     verified transport: resume, SHA-256, a login page is not a file
``lake``      csv.gz -> bucketed Parquet, streamed, published only after its hash checks
``guard``     the Data Use Agreement as code: posture checks and an egress scanner
``pipeline``  the stages, their gates, and the one place it stops for a human

``python -m backend.ml.physionet`` is the command line; docs/PHYSIONET.md is the manual.
"""
