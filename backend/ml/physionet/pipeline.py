"""The whole path from an empty disk to a pre-registration, as stages with gates.

    doctor -> access -> capacity -> ingest -> contract -> canary -> parity -> plan -> dev_run
           -> release -> PRE-REGISTRATION (stops here, for a human)

Each stage records a fingerprint of what it read. A re-run skips a stage whose inputs have not
changed and redoes one whose inputs have, so a crash costs only the stage it hit, and a new lake
or new harness code cannot leave an old result looking current. Every stage is written to a
hash-chained journal beside the lake.

**It never spends the lock.** The last automatic stage releases a dev-fold report. Choosing claims
and calling ``trust.confirm`` is a human decision the code offers no shortcut to. And one heavy
job runs at a time on this machine: the lake's file lock refuses a second one.

Two routes reach the same store:

* ``lake``: the tables stream from physionet.org into local Parquet. Works anywhere there's disk.
* ``bigquery``: Google bins the events server-side, and only the hourly aggregates come down.
  Needs a linked Google Cloud project and has no local tables, so no contract stage: the SQL is
  verified offline against the demo (``sqlcheck``), then against the real service (``parity``).
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from backend.ml import config
from backend.ml.physionet import auth, catalog, fetch, guard
from backend.ml.physionet import lake as PL


class Refused(RuntimeError):
    """A gate said no. The message says what to do about it."""


class Hold(RuntimeError):
    """The pipeline has reached a decision that belongs to a person."""


#: full / demo row ratios for sizing only: the largest table of each module, demo rows as counted
#: here against the full database's approximate published size (chartevents ~430M vs 669k,
#: labevents ~160M vs 108k). The real sizes replace these the moment the tables are ingested.
SCALE_UP = {"icu": 650.0, "hosp": 1470.0}
#: compression gets worse as a bucket mixes more stays than the demo's few: a planning margin
LAKE_MARGIN = 3.0
FULL_HOURS_PER_STAY = 75.0     # the demo's mean ICU hours per stay, after the 336-h cap
STORE_CHANNELS = 155
RESERVE_GB = 2.0


@dataclass
class Ctx:
    dataset: catalog.Dataset
    root: str
    route: str = "lake"
    project: str = ""
    base_url: Optional[str] = None             # a mirror, or the test server
    from_dir: Optional[str] = None             # tables already on disk
    client: Optional[fetch.Client] = None
    hours: float = 12.0
    scale_name: Optional[str] = None           # "cloud" on a dedicated machine; else by access
    parallel: int = 1                          # >1: each table over that many ranges, staged (deploy/physionet)
    runner: Optional[Callable] = None          # runs a modelling step; the verifier plants one
    repo: str = config.REPO_ROOT
    demo_root: Optional[str] = None
    log: Callable[[str], None] = field(default=lambda s: print(s, flush=True))

    @property
    def key(self) -> str:
        return self.dataset.key

    @property
    def url(self) -> str:
        return (self.base_url or self.dataset.url).rstrip("/") + "/"

    @property
    def lake(self) -> str:
        return os.path.join(self.root, self.key, "lake")

    @property
    def demo_lake(self) -> str:
        return os.path.join(self.root, "mimic-iv-demo-baseline", "lake")

    @property
    def sums_path(self) -> str:
        return os.path.join(self.root, self.key, "SHA256SUMS.txt")

    @property
    def cache(self) -> str:
        return os.path.join(self.root, "cache")

    @property
    def reports(self) -> str:
        return os.path.join(self.root, "reports")

    @property
    def canary_path(self) -> str:
        return os.path.join(self.root, "canary.json")

    @property
    def state_path(self) -> str:
        return os.path.join(self.root, "state", f"{self.key}-{self.route}.json")

    @property
    def scale(self) -> str:
        if self.scale_name:
            return self.scale_name
        return "demo" if self.dataset.access == "open" else "full"

    @property
    def demo_csv(self) -> str:
        if self.demo_root:
            return self.demo_root
        from backend.ml.mimic import extract as E
        if os.path.isdir(E.DEMO_ROOT):
            return E.DEMO_ROOT
        return fetch_demo(os.path.join(self.root, "mimic-iv-demo-csv"))

    def http(self) -> fetch.Client:
        if self.client is None:
            creds = auth.resolve() if self.dataset.access != "open" else None
            self.client = fetch.Client(creds)
        return self.client


def fetch_demo(dest: str) -> str:
    """A machine with no local demo (a fresh cloud box) fetches it: open, anonymous, verified."""
    d = catalog.get("mimic-iv-demo")
    client = fetch.Client()
    sums = fetch.parse_sums(client.get_text(d.url + "SHA256SUMS.txt"))
    for spec in _specs():
        fetch.fetch_file(client, d.url + spec.source, os.path.join(dest, spec.source), sums[spec.source])
    return dest


# --------------------------------------------------------------------------- #
# Stages
# --------------------------------------------------------------------------- #
def _sha(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def _specs() -> List[PL.TableSpec]:
    from backend.ml.mimic import lake as ML
    return ML.project()


def _stage_doctor(ctx: Ctx) -> Dict:
    guard.init_lake(ctx.root)
    need = ctx.dataset.access != "open" and ctx.route == "lake"
    rep = guard.doctor(ctx.root, ctx.repo, creds_needed=need)
    if not rep["ok"]:
        lines = [f"  {c['check']}: {c['detail']}" + (f"\n    -> {c['fix']}" if c.get("fix") else "")
                 for c in rep["checks"] if c["ok"] is False and c["severity"] == "fail"]
        raise Refused("this machine may not hold the data yet:\n" + "\n".join(lines))
    return {"checks": len(rep["checks"]), "warnings": [c["check"] for c in rep["checks"] if c["ok"] is False]}


def _stage_access(ctx: Ctx) -> Dict:
    """Can this account read it, and is it the release we pinned the first time?"""
    text = ctx.http().get_text(ctx.url + "SHA256SUMS.txt")
    sums = fetch.parse_sums(text)
    if os.path.exists(ctx.sums_path):
        with open(ctx.sums_path) as fh:
            pinned = fh.read()
        if pinned != text:
            raise Refused(f"{ctx.key}'s published hashes changed since they were pinned. Nothing was overwritten. "
                          f"Read the release notes, then delete {ctx.sums_path} to accept the new contents.")
    else:
        os.makedirs(os.path.dirname(ctx.sums_path), exist_ok=True)
        with open(ctx.sums_path, "w") as fh:
            fh.write(text)
    missing = [s.source for s in _specs() if s.source not in sums]
    if missing:
        raise Refused(f"{ctx.key} does not publish tables the pipeline reads: {missing}")
    return {"files_published": len(sums), "sha256sums": hashlib.sha256(text.encode()).hexdigest()}


def build_demo_baseline(ctx: Ctx) -> None:
    """The open demo as a lake: the contract's baseline, and the size model's reference."""
    for spec in _specs():
        if PL.is_current(ctx.demo_lake, spec, None):
            continue
        stream = fetch.LocalStream(os.path.join(ctx.demo_csv, spec.source))
        try:
            PL.ingest(spec, stream, ctx.demo_lake, None)
        finally:
            stream.close()


def project_size(ctx: Ctx) -> Dict:
    """Disk needed, projected from the demo as actually built. An estimate, stated as one."""
    build_demo_baseline(ctx)
    tables, total, raw = {}, 0.0, 0.0
    factor = (lambda s: 1.0) if ctx.dataset.access == "open" else (lambda s: SCALE_UP[s.module])
    margin = 1.0 if ctx.dataset.access == "open" else LAKE_MARGIN
    for spec in _specs():
        m = PL.read_manifest(ctx.demo_lake, spec)
        gb = m["bytes"] * factor(spec) * margin / 1e9
        tables[spec.rel] = round(gb, 2)
        total += gb
        raw += m["source_bytes"] * factor(spec) / 1e9
    from backend.ml.mimic.lake import LakeSource
    stays = len(LakeSource(ctx.demo_lake).icustays()) * (1.0 if ctx.dataset.access == "open" else SCALE_UP["icu"])
    store_gb = stays * FULL_HOURS_PER_STAY * STORE_CHANNELS * 4 / 1e9
    return {"lake_gb": round(total, 1), "raw_csv_gz_gb": round(raw, 1), "store_gb": round(store_gb, 1),
            "by_table_gb": tables, "stays_projected": int(stays)}


def _free_gb(path: str) -> float:
    while not os.path.exists(path):
        path = os.path.dirname(path)
    return shutil.disk_usage(path).free / 1e9


def _stage_capacity(ctx: Ctx) -> Dict:
    p = project_size(ctx)
    need = p["lake_gb"] + p["store_gb"] + RESERVE_GB
    free = _free_gb(ctx.root)
    p.update(free_gb=round(free, 1), needed_through_store_gb=round(need, 1))
    have = sum(PL.read_manifest(ctx.lake, s)["bytes"] for s in _specs() if PL.read_manifest(ctx.lake, s)) / 1e9
    if free + have < need:
        raise Refused(f"needs ~{need:.0f} GB through the store (lake {p['lake_gb']}, grid {p['store_gb']}, reserve "
                      f"{RESERVE_GB:.0f}); {free:.0f} GB free at {ctx.root}. Options: an external encrypted drive "
                      f"(PHYSIONET_LAKE=/Volumes/<drive>/physionet-lake), --route bigquery, or deploy/physionet.")
    return p


def _stage_ingest(ctx: Ctx) -> Dict:
    # before the first credentialed byte can exist here. A copy with no repository (the cloud image)
    # can't commit anything, so there is nothing to hook; a checkout always gets it.
    if ctx.dataset.access != "open" and os.path.exists(os.path.join(ctx.repo, ".git")):
        guard.install_hook(ctx.repo)
    with open(ctx.sums_path) as fh:
        sums = fetch.parse_sums(fh.read())
    done = {}
    for spec in _specs():
        sha = sums[spec.source]
        if PL.is_current(ctx.lake, spec, sha):
            done[spec.rel] = "kept"
            continue
        check = lambda: fetch.require_free(ctx.lake, 0, int(RESERVE_GB * 1e9))
        check()
        t0 = time.time()
        ctx.log(f"  {spec.rel} ...")
        staged = None
        if ctx.from_dir:
            stream = fetch.LocalStream(os.path.join(ctx.from_dir, spec.source))
        else:
            stream = None
            if ctx.parallel > 1:              # the raw file lands on the lake's disk, briefly, then goes
                staged = os.path.join(ctx.root, "staging", os.path.basename(spec.source))
                try:
                    fetch.fetch_parallel(ctx.http(), ctx.url + spec.source, staged, sha, parts=ctx.parallel)
                    stream = fetch.LocalStream(staged)
                except fetch.RangeIgnored:
                    ctx.log("    ranges not served: streaming instead")
            if stream is None:
                stream = fetch.RemoteStream(ctx.http(), ctx.url + spec.source, on_retry=lambda w: ctx.log(f"    retrying: {w}"))
        try:
            m = PL.ingest(spec, stream, ctx.lake, sha, disk_check=check)
        finally:
            stream.close()
            if staged and os.path.exists(staged):
                os.remove(staged)
        done[spec.rel] = {k: m[k] for k in ("rows_in", "rows_kept", "bytes", "source_bytes", "reconnects")}
        ctx.log(f"    {m['rows_kept']:,} of {m['rows_in']:,} rows, {m['bytes'] / 1e6:,.0f} MB, "
                f"{time.time() - t0:,.0f}s, sha256 verified")
    return {"tables": done, "fingerprint": PL.fingerprint(ctx.lake, _specs())}


def _stage_contract(ctx: Ctx) -> Dict:
    from backend.ml.mimic import contracts as CT
    from backend.ml.mimic.lake import LakeSource
    build_demo_baseline(ctx)
    rep = CT.check(LakeSource(ctx.lake), LakeSource(ctx.demo_lake), min_stays=ctx.dataset.expected_stays)
    if not rep["passed"]:
        details = [f"  {c['check']}: {c['detail']}" for c in rep["checks"] if not c["ok"] and c["severity"] == "fail"]
        raise Refused("the data does not match its contract:\n" + "\n".join(details))
    return rep


def _stage_canary(ctx: Ctx) -> Dict:
    from backend.ml.mimic.lake import LakeSource
    src = LakeSource(ctx.lake)
    st, ad = src.icustays(), src.admissions()
    pa = src.patients()
    ids = list(st.stay_id) + list(ad.hadm_id) + list(pa.subject_id)
    opened = []
    if ctx.dataset.access != "open":
        build_demo_baseline(ctx)
        demo = LakeSource(ctx.demo_lake)
        opened = list(demo.icustays().stay_id) + list(demo.admissions().hadm_id) + list(demo.patients().subject_id)
    can = guard.Canary.build(ids, opened)
    can.save(ctx.canary_path)
    return {"identifiers": len(can.hashes), "open_excluded": len(set(opened))}


def _stage_parity(ctx: Ctx) -> Dict:
    """The demo's patients, as they read in the lake, against the CSV reference."""
    from backend.ml.mimic import extract as E
    from backend.ml.mimic import parity as PA
    from backend.ml.mimic.lake import LakeSource
    from backend.ml.mimic.sources import CsvSource
    ref = CsvSource(ctx.demo_csv)
    stays = E.cohort(ref)[["stay_id", "hadm_id", "intime", "outtime"]]
    exact = ctx.dataset.access == "open"
    rep = PA.parity(ref, LakeSource(ctx.lake), stays, agreement=1.0 if exact else PA.AGREEMENT,
                    across_releases=not exact)       # a credentialed release against the 2.2 demo
    if not rep["verdict"]["passed"]:
        worst = {k: round(v["agreement"], 4) for k, v in rep.items() if k in rep["verdict"]["failing"]}
        raise Refused(f"the lake does not reproduce the reference on the demo's patients: {worst}"
                      + (f"; {rep['revised']['stays']} of {rep['revised']['of']} stays revised by the release"
                         if "revised" in rep else ""))
    out = {k: v["agreement"] for k, v in rep.items() if k != "verdict"}
    if "revised" in rep:
        out["revised_stays"] = f"{rep['revised']['stays']} of {rep['revised']['of']}"
    return out


def _stage_sqlcheck(ctx: Ctx) -> Dict:
    from backend.ml.mimic import extract as E
    from backend.ml.mimic import sqlcheck as SC
    from backend.ml.mimic.sources import CsvSource
    stays = E.cohort(CsvSource(ctx.demo_csv))[["stay_id", "hadm_id", "intime", "outtime"]]
    rep = SC.check(ctx.demo_csv, stays)
    if not rep["verdict"]["passed"]:
        raise Refused(f"the SQL does not reproduce the reference, offline: {rep['verdict']['failing']}")
    return {k: v["agreement"] for k, v in rep.items() if k != "verdict"}


def _stage_bq_parity(ctx: Ctx) -> Dict:  # pragma: no cover - needs a linked Google Cloud project
    if not ctx.project:
        raise Refused("--route bigquery needs --project <gcp-project> linked on the MIMIC-IV page")
    from backend.ml.mimic import extract as E
    from backend.ml.mimic import parity as PA
    from backend.ml.mimic.sources import BigQuerySource, CsvSource
    ref = CsvSource(ctx.demo_csv)
    stays = E.cohort(ref)[["stay_id", "hadm_id", "intime", "outtime"]]
    rep = PA.parity(ref, BigQuerySource(ctx.project), stays)
    if not rep["verdict"]["passed"]:
        raise Refused(f"BigQuery does not reproduce the reference: {rep['verdict']['failing']}")
    return {k: v["agreement"] for k, v in rep.items() if k != "verdict"}


def tag(ctx: Ctx) -> str:
    if ctx.route == "bigquery":
        return f"bq-{ctx.key}"
    return "lake-" + PL.fingerprint(ctx.lake, _specs())[:10]


def run_cmd(ctx: Ctx, plan_only: bool) -> List[str]:
    cmd = [sys.executable, "-m", "backend.ml.mimic.run", "--scale", ctx.scale, "--tag", tag(ctx), "--hours", str(ctx.hours)]
    cmd += ["--source", "bigquery", "--project", ctx.project] if ctx.route == "bigquery" else ["--source", "lake", "--lake", ctx.lake]
    cmd += ["--checkpoint", os.path.join(ctx.cache, "checkpoints")]
    if plan_only:
        cmd.append("--plan-only")
    if platform.system() == "Darwin":
        cmd = ["caffeinate", "-ims"] + cmd       # a sleeping laptop stops nothing but the clock
    return cmd


def _run_step(ctx: Ctx, name: str, plan_only: bool) -> None:
    """A modelling step, in its own process: its output goes to a log in the lake, never to a
    terminal, a transcript or an AI assistant. Only the report comes back, after the scan."""
    if ctx.runner is not None:
        ctx.runner(ctx, name, plan_only)
        return
    os.makedirs(os.path.join(ctx.root, "logs"), exist_ok=True)
    path = os.path.join(ctx.root, "logs", f"{tag(ctx)}-{name}.log")
    env = dict(os.environ, MIMIC_CACHE_DIR=ctx.cache, MIMIC_REPORT_DIR=ctx.reports, PYTHONPATH=ctx.repo)
    ctx.log(f"  running; progress in {path}")
    with open(path, "a") as log:
        code = subprocess.run(run_cmd(ctx, plan_only), cwd=ctx.repo, env=env, stdout=log, stderr=subprocess.STDOUT).returncode
    if code != 0:
        raise Refused(f"{name} failed with exit code {code}: see {path}. Re-running resumes from its checkpoints.")


def _stage_plan(ctx: Ctx) -> Dict:
    _run_step(ctx, "plan", True)
    path = os.path.join(ctx.reports, f"{tag(ctx)}_plan.json")
    if not os.path.exists(path):
        return {"plan": "not needed at this scale"}
    with open(path) as fh:
        return json.load(fh)


def _stage_dev_run(ctx: Ctx) -> Dict:
    _run_step(ctx, "dev", False)
    path = os.path.join(ctx.reports, f"{tag(ctx)}_report.json")
    if not os.path.exists(path):
        raise Refused(f"the run finished without writing {path}")
    return {"report": path, "bytes": os.path.getsize(path)}


def _stage_release(ctx: Ctx) -> Dict:
    """A report leaves the lake only through the scanner."""
    src = os.path.join(ctx.reports, f"{tag(ctx)}_report.json")
    found = guard.scan_paths([src], guard.Canary.load(ctx.canary_path))
    blocks = [f for f in found if f.severity == "block"]
    if blocks:
        raise Refused("the report may not leave the lake:\n" + "\n".join("  " + f.text() for f in blocks))
    dest = os.path.join(ctx.repo, "artifacts", "mimic", f"{tag(ctx)}_report.json")
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    shutil.copyfile(src, dest)
    return {"released": os.path.relpath(dest, ctx.repo), "warnings": [f.text() for f in found]}


@dataclass(frozen=True)
class Stage:
    name: str
    run: Callable[[Ctx], Dict]
    #: what its result depends on, beyond the stages before it
    inputs: Callable[[Ctx], object] = lambda ctx: None


def _code(ctx: Ctx) -> str:
    from backend.ml.mimic import trust as TR
    return TR.code_fingerprint(ctx.repo)["sha256"]


def _lake_fp(ctx: Ctx) -> str:
    return PL.fingerprint(ctx.lake, _specs())


def _spec_hashes(ctx: Ctx) -> List[str]:
    return [s.spec_hash() for s in _specs()]


def _tables_present(ctx: Ctx) -> List[bool]:
    """A lost or stale table must reopen the ingest stage, even though its inputs are unchanged."""
    return [PL.is_current(ctx.lake, s, None) for s in _specs()]


def _sums(ctx: Ctx) -> str:
    with open(ctx.sums_path) as fh:
        return hashlib.sha256(fh.read().encode()).hexdigest()


def stages(route: str) -> List[Stage]:
    start = [Stage("doctor", _stage_doctor, lambda c: time.strftime("%Y-%m-%d"))]   # posture is re-checked daily
    finish = [Stage("plan", _stage_plan, _code), Stage("dev_run", _stage_dev_run, _code), Stage("release", _stage_release, _code)]
    if route == "bigquery":
        return start + [Stage("sqlcheck", _stage_sqlcheck, _code), Stage("bq_parity", _stage_bq_parity, _code)] + finish
    return start + [
        Stage("access", _stage_access),
        Stage("capacity", _stage_capacity, _spec_hashes),
        Stage("ingest", _stage_ingest, lambda c: [_sums(c), _spec_hashes(c), _tables_present(c)]),
        Stage("contract", _stage_contract, lambda c: [_lake_fp(c), _code(c)]),
        Stage("canary", _stage_canary, _lake_fp),
        Stage("parity", _stage_parity, lambda c: [_lake_fp(c), _code(c)]),
    ] + finish


# --------------------------------------------------------------------------- #
# The runner
# --------------------------------------------------------------------------- #
def _load_state(ctx: Ctx) -> Dict:
    if os.path.exists(ctx.state_path):
        with open(ctx.state_path) as fh:
            return json.load(fh)
    return {}


def _save_state(ctx: Ctx, state: Dict) -> None:
    os.makedirs(os.path.dirname(ctx.state_path), exist_ok=True)
    tmp = ctx.state_path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(state, fh, indent=1, default=str)
    os.replace(tmp, ctx.state_path)


class _Lock:
    def __init__(self, root: str):
        self.path = os.path.join(root, ".lock")

    def __enter__(self):
        self.fh = open(self.path, "w")
        try:
            fcntl.flock(self.fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.fh.close()
            raise Refused(f"another PhysioNet job holds {self.path}: one heavy job at a time") from None
        return self

    def __exit__(self, *exc):
        fcntl.flock(self.fh, fcntl.LOCK_UN)
        self.fh.close()


def journal(ctx: Ctx, kind: str, body: Dict) -> None:
    from backend.ml.mimic import trust as TR
    TR.Chain(os.path.join(ctx.root, "journal.jsonl")).append(kind, dict(body, dataset=ctx.key, route=ctx.route))


def up(ctx: Ctx, until: Optional[str] = None) -> Dict:
    """Run every stage not already current, in order, stopping at a gate or at ``until``."""
    guard.init_lake(ctx.root)
    with _Lock(ctx.root):
        state = _load_state(ctx)
        for st in stages(ctx.route):
            try:
                fp = _sha(st.inputs(ctx))
            except FileNotFoundError:
                fp = "unknown"
            prev = state.get(st.name)
            if prev and prev.get("status") == "done" and prev.get("fp") == fp:
                ctx.log(f"= {st.name}: current")
            else:
                ctx.log(f"> {st.name}")
                t0 = time.time()
                try:
                    result = st.run(ctx)
                except (Refused, fetch.FetchError) as err:
                    state[st.name] = {"status": "refused", "fp": fp, "reason": str(err), "finished": time.strftime("%F %T")}
                    _save_state(ctx, state)
                    journal(ctx, "refused", {"stage": st.name, "reason": str(err)[:500]})
                    raise
                try:
                    fp = _sha(st.inputs(ctx))
                except FileNotFoundError:
                    pass
                state[st.name] = {"status": "done", "fp": fp, "result": result, "seconds": round(time.time() - t0, 1),
                                  "finished": time.strftime("%F %T")}
                _save_state(ctx, state)
                journal(ctx, "stage", {"stage": st.name, "fp": fp, "seconds": state[st.name]["seconds"]})
            if st.name == until:
                return state
        raise Hold(next_steps(ctx))


def next_steps(ctx: Ctx) -> str:
    return ("Every automatic stage is done. The lock has not been read. What's left is a person's decision: "
            "read the released report, choose the claims, pre-register them (trust.preregister), commit the chain, "
            "and only then confirm. docs/MIMIC.md section 8.")


def progress(ctx: Ctx) -> Optional[Dict]:
    """The modelling run's own journal (``mimic.progress``): fits done, memory, an ETA. Numbers only."""
    from backend.ml.mimic import progress as PR
    return PR.summary(os.path.join(ctx.cache, "checkpoints", PR.JOURNAL))


def status(ctx: Ctx) -> List[Dict]:
    state = _load_state(ctx)
    return [{"stage": st.name, **{k: state.get(st.name, {}).get(k, "") for k in ("status", "finished", "seconds", "reason")}}
            for st in stages(ctx.route)]
