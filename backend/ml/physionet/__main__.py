"""``python -m backend.ml.physionet <command>``. The manual is docs/PHYSIONET.md."""
from __future__ import annotations

import argparse
import fnmatch
import json
import os
import sys

from backend.ml import config
from backend.ml.physionet import auth, catalog, fetch, guard, pipeline


def _ctx(a) -> pipeline.Ctx:
    return pipeline.Ctx(dataset=catalog.get(a.dataset), root=a.lake, route=a.route, project=a.project,
                        base_url=a.mirror, from_dir=a.from_dir, hours=a.hours,
                        scale_name=a.scale, parallel=a.parallel)


def cmd_datasets(a) -> int:
    for d in catalog.DATASETS.values():
        print(f"{d.key:<24} {d.access:<13} {d.use}")
    return 0


def cmd_login(a) -> int:
    print("Run this yourself; it prompts on your terminal, so the password goes nowhere else:\n")
    print("  " + auth.login_command(a.username or "<your-physionet-username>"))
    return 0


def cmd_doctor(a) -> int:
    rep = guard.doctor(a.lake, creds_needed=True)
    for c in rep["checks"]:
        mark = {True: "ok  ", False: "FAIL" if c["severity"] == "fail" else "warn", None: "n/a "}[c["ok"]]
        print(f"[{mark}] {c['check']}: {c['detail']}")
        if c["ok"] is False and c["fix"]:
            print(f"       -> {c['fix']}")
    return 0 if rep["ok"] else 1


def cmd_access(a) -> int:
    """What this account can actually read, measured against each project's hash file."""
    client = fetch.Client(auth.resolve(), retries=2)
    print(f"credentials: {client.creds.source if client.creds else 'none'}")
    for d in catalog.DATASETS.values():
        try:
            n = len(fetch.parse_sums(client.get_text(d.url + "SHA256SUMS.txt")))
            print(f"  OPEN TO YOU  {d.key:<24} {n} files")
        except fetch.AuthRejected:
            print(f"  REJECTED     {d.key:<24} username or password")
        except fetch.NoAccess:
            print(f"  NO ACCESS    {d.key:<24} check the username and password, then this dataset's agreement on physionet.org")
        except Exception as err:
            print(f"  ERROR        {d.key:<24} {type(err).__name__}")
    return 0


def cmd_plan(a) -> int:
    ctx = _ctx(a)
    guard.init_lake(ctx.root)
    p = pipeline.project_size(ctx)
    free = pipeline._free_gb(ctx.root)
    need = p["lake_gb"] + p["store_gb"] + pipeline.RESERVE_GB
    print(f"{ctx.key}, projected from the demo as built (an estimate, with a {pipeline.LAKE_MARGIN:.0f}x margin):")
    print(f"  published csv.gz, had they all been downloaded   {p['raw_csv_gz_gb']:>7} GB")
    print(f"  the lake (only what the pipeline reads)          {p['lake_gb']:>7} GB")
    print(f"  the hourly grid                                  {p['store_gb']:>7} GB  (uncompressed)")
    print(f"  through the grid, with the reserve               {need:>7.1f} GB")
    print(f"  free at {ctx.root}: {free:.1f} GB -> {'fits' if free >= need else 'DOES NOT FIT'}")
    print("  features add ~6 GB beyond this; `make physionet-up` runs the exact planner once the grid exists")
    return 0


def cmd_up(a) -> int:
    ctx = _ctx(a)
    try:
        pipeline.up(ctx, a.until)
    except pipeline.Hold as held:
        print(f"\nHOLD: {held}")
        return 0
    except (pipeline.Refused, fetch.FetchError) as err:
        print(f"\nREFUSED: {err}")
        return 1
    print("\ndone up to " + (a.until or "the end"))
    return 0


def cmd_status(a) -> int:
    for row in pipeline.status(_ctx(a)):
        s = row["status"] or "pending"
        print(f"  {row['stage']:<10} {s:<8} {row['finished']}" + (f"  {row['seconds']}s" if row["seconds"] != "" else ""))
        if row["reason"]:
            print("    " + str(row["reason"]).replace("\n", "\n    "))
    p = pipeline.progress(_ctx(a))
    if p:
        print(progress_line(p))
    return 0


def progress_line(p: dict) -> str:
    import time
    mem = (f"{p['rss_gb']:.1f} GB RAM + {p.get('swap_gb', 0):.1f} GB swap" if "rss_gb" in p
           else f"peak {p.get('peak_gb', 0):.1f} GB")
    share = f"{100 * p['share']:.0f}% of the planned fitting" if p["share"] is not None else "fitting"
    out = (f"  modelling: {share} ({p['fits']} fits, {p['resumed']} resumed), {p['elapsed_h']:.1f} h in, "
           f"last fit {p['since_last_s'] / 60:.0f} min ago, {mem}")
    if p["eta_h"] is not None:
        eta = time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(p["eta_t"]))
        out += (f"\n             about {p['eta_h']:.1f} h to go{'' if p['twin_done'] else ' (twin included)'}, "
                f"around {eta}")
    return out


def cmd_fetch(a) -> int:
    """Whole files, verified: for what the lake does not model (waveforms, notes)."""
    d = catalog.get(a.dataset)
    client = fetch.Client(auth.resolve() if d.access != "open" else None)
    sums = fetch.parse_sums(client.get_text(d.url + "SHA256SUMS.txt"))
    chosen = sorted(p for p in sums if any(fnmatch.fnmatch(p, g) for g in a.include))
    if a.list or not chosen:
        print(f"{len(sums)} files published; {len(chosen)} match {a.include}")
        for p in chosen[:40]:
            print(" ", p)
        return 0 if chosen or a.list else 1
    guard.init_lake(a.lake)
    if d.access != "open":
        guard.install_hook()
    for p in chosen:
        dest = os.path.join(a.lake, d.key, "raw", fetch.safe_relpath(p))
        _, size = fetch.fetch_file(client, d.url + p, dest, sums[p])
        print(f"  {p}: {size / 1e6:,.1f} MB, sha256 verified")
    return 0


def cmd_scan(a) -> int:
    canary = guard.Canary.load(os.path.join(a.lake, "canary.json"))
    found = guard.scan_staged(config.REPO_ROOT, canary) if a.staged else guard.scan_paths(a.paths, canary, a.lake)
    for f in found:
        print(f.text(), file=sys.stderr)
    blocked = any(f.severity == "block" for f in found)
    if blocked:
        print("blocked by the PhysioNet egress scan (docs/PHYSIONET.md). If this is a false alarm, "
              "look at the file yourself before overriding.", file=sys.stderr)
    return 1 if blocked else 0


def cmd_hook(a) -> int:
    print(guard.install_hook())
    return 0


def cmd_sqlcheck(a) -> int:
    ctx = _ctx(a)
    print(json.dumps(pipeline._stage_sqlcheck(ctx), indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="physionet", description=__doc__)
    ap.add_argument("--lake", default=guard.default_lake())
    ap.add_argument("--dataset", default="mimiciv")
    ap.add_argument("--route", choices=["lake", "bigquery"], default="lake")
    ap.add_argument("--project", default="", help="the Google Cloud project linked on PhysioNet")
    ap.add_argument("--mirror", default=None, help="another base URL for the dataset")
    ap.add_argument("--from-dir", default=None, help="tables already on disk (a local copy of the csv.gz tree)")
    ap.add_argument("--hours", type=float, default=12.0)
    ap.add_argument("--parallel", type=int, default=1,
                    help="fetch each table over this many byte ranges, staged on the lake's disk (cloud: 4)")
    ap.add_argument("--scale", choices=["demo", "full", "cloud"], default=None,
                    help="default: demo for open data, full for credentialed; cloud on a dedicated machine")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name, fn in [("datasets", cmd_datasets), ("doctor", cmd_doctor), ("access", cmd_access), ("plan", cmd_plan),
                     ("status", cmd_status), ("hook", cmd_hook), ("sqlcheck", cmd_sqlcheck)]:
        sub.add_parser(name).set_defaults(fn=fn)
    p = sub.add_parser("login")
    p.add_argument("username", nargs="?")
    p.set_defaults(fn=cmd_login)
    p = sub.add_parser("up")
    p.add_argument("--until", default=None)
    p.set_defaults(fn=cmd_up)
    p = sub.add_parser("fetch")
    p.add_argument("include", nargs="*", default=["*"])
    p.add_argument("--list", action="store_true")
    p.set_defaults(fn=cmd_fetch)
    p = sub.add_parser("scan")
    p.add_argument("paths", nargs="*")
    p.add_argument("--staged", action="store_true")
    p.set_defaults(fn=cmd_scan)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
