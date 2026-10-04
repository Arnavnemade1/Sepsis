"""Screenshot every screen of the clinical app at the widths the design rule names.

    .venv/bin/python scripts/screens.py                     # all screens, 1512 / 820 / 390
    .venv/bin/python scripts/screens.py --widths 390 --only rooms,room-301
    .venv/bin/python scripts/screens.py --out /tmp/shots --base http://localhost:3000

Headless Chrome over the DevTools protocol, in real time (no --virtual-time-budget, which
hangs on the live WebSocket), with SwiftShader so the WebGL twin renders. Each shot also
reports console errors and uncaught exceptions. A phone width is emulated as a mobile
device. Needs the backend and `npm run dev` (or a static build) running, and `websockets`
(in the dev venv).
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import subprocess
import tempfile
import time
import urllib.request

import websockets

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
PORT = 9333
HEIGHT = {1512: 945, 820: 1180, 390: 844}

SCREENS = [
    ("op-overview", "/operator/"), ("op-pages", "/operator/?view=pages"), ("op-staff", "/operator/?view=staff"),
    ("op-sentinel", "/operator/?view=sentinel"), ("op-policy", "/operator/?view=policy"),
    ("rooms", "/room/"), ("room-301", "/room/?id=301"), ("room-414", "/room/?id=414"), ("room-402", "/room/?id=402"),
    ("patients", "/patient/"), ("patient-bed3", "/patient/?id=bed_3"), ("patient-live", "/patient/?id=aegis-demo-patient"),
    ("pharmacy", "/pharmacy/"), ("lab", "/lab/"),
    ("os-arch", "/os/"), ("os-kernel", "/os/?view=kernel"), ("os-journal", "/os/?view=journal"),
    ("os-gate", "/os/?view=gate"), ("os-shell", "/os/?view=shell"),
    ("pagers", "/clinician/"), ("pager-alvarez", "/clinician/?id=dr.alvarez"),
    ("integrations", "/integrations/"), ("integrations-mappings", "/integrations/?view=mappings"),
    ("integrations-endpoints", "/integrations/?view=endpoints"),
    ("room-hl7", "/room/?id=512"),       # the simulated hospital's first HL7-fed admission (seeded)
]


async def shoot(shots, wait: float) -> int:
    prof = tempfile.mkdtemp(prefix="aegis-shots-")
    proc = subprocess.Popen([CHROME, "--headless=new", "--hide-scrollbars", "--no-first-run",
                             "--no-default-browser-check", "--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                             f"--user-data-dir={prof}", f"--remote-debugging-port={PORT}", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    problems = 0
    try:
        page = None
        for _ in range(80):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json"))
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.25)
        async with websockets.connect(page["webSocketDebuggerUrl"], max_size=2 ** 28) as ws:
            pending, events, n = {}, [], 0

            async def reader():
                async for raw in ws:
                    m = json.loads(raw)
                    if "id" in m and m["id"] in pending:
                        pending.pop(m["id"]).set_result(m)
                    elif m.get("method") in ("Runtime.exceptionThrown", "Runtime.consoleAPICalled"):
                        events.append(m)

            task = asyncio.create_task(reader())

            async def call(method, **params):
                nonlocal n
                n += 1
                fut = asyncio.get_event_loop().create_future()
                pending[n] = fut
                await ws.send(json.dumps({"id": n, "method": method, "params": params}))
                m = await asyncio.wait_for(fut, 60)
                if "error" in m:
                    raise RuntimeError(f"{method}: {m['error']}")
                return m.get("result", {})

            await call("Page.enable")
            await call("Runtime.enable")
            for url, out, w in shots:
                await call("Emulation.setDeviceMetricsOverride", width=w, height=HEIGHT.get(w, 900),
                           deviceScaleFactor=1, mobile=w < 768)
                events.clear()
                await call("Page.navigate", url=url)
                await asyncio.sleep(wait)
                shot = await call("Page.captureScreenshot", format="png")
                with open(out, "wb") as fh:
                    fh.write(base64.b64decode(shot["data"]))
                errs = [e for e in events if e["method"] == "Runtime.exceptionThrown"
                        or e["params"].get("type") == "error"]
                problems += len(errs)
                print(f"{os.path.basename(out):32} {'ok' if not errs else f'{len(errs)} console error(s)'}")
                for e in errs[:3]:
                    p = e["params"]
                    text = (p.get("exceptionDetails", {}).get("exception", {}).get("description")
                            or " ".join(str(a.get("value", "")) for a in p.get("args", [])))
                    print("    ", text.replace("\n", " ")[:300])
            task.cancel()
    finally:
        proc.terminate()
    return problems


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:3000")
    ap.add_argument("--widths", default="1512,820,390")
    ap.add_argument("--only", default="", help="comma-separated screen names")
    ap.add_argument("--out", default=os.path.join(tempfile.gettempdir(), "aegis-screens"))
    ap.add_argument("--wait", type=float, default=7.0, help="seconds per screen (the app waits on live data)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    only = {s for s in args.only.split(",") if s}
    shots = [(args.base + path, os.path.join(args.out, f"w{w}-{name}.png"), int(w))
             for w in args.widths.split(",") for name, path in SCREENS if not only or name in only]
    bad = asyncio.run(shoot(shots, args.wait))
    print(f"{len(shots)} screens -> {args.out}; {bad} console error(s)")


if __name__ == "__main__":
    main()
