"""MLLP: HL7 v2 over TCP, the way interface engines actually connect.

Every message arrives framed ``<VT> message <FS><CR>`` on a long-lived socket, and the
sender waits for the ACK before sending the next one. This listener de-frames with
``hl7.MLLPDecoder`` (partial reads, several messages per read, noise between frames),
hands each message to the engine off the event loop, and writes the framed ACK back.

It is off unless ``AEGIS_MLLP_PORT`` is set: a hosted deployment that only exposes
HTTP (Render's free tier does) uses ``POST /aegis/interop/hl7`` instead, which runs the
same ``Integration.ingest_hl7``. An on-premises deployment points the hospital's
interface engine at this port, on the hospital network.
"""
from __future__ import annotations

import asyncio
from typing import Any, Callable, Optional, Tuple

from backend.aegis.interop.hl7 import MLLPDecoder, frame

Handler = Callable[[str, str, str], Tuple[str, Any]]


async def _serve_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter, handler: Handler,
                        principal: str) -> None:
    dec = MLLPDecoder()
    peer = writer.get_extra_info("peername")
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            for msg in dec.feed(data):
                ack, _rec = await asyncio.to_thread(handler, msg, "mllp", f"{principal}@{peer[0] if peer else '?'}")
                writer.write(frame(ack))
                await writer.drain()
    except (ConnectionError, asyncio.IncompleteReadError):
        pass
    finally:
        try:
            writer.close()
        except Exception:
            pass


async def start(port: int, handler: Handler, host: str = "0.0.0.0", principal: str = "mllp") -> Optional[asyncio.AbstractServer]:
    """Start listening; returns the server (or None if the port cannot be bound)."""
    try:
        return await asyncio.start_server(lambda r, w: _serve_client(r, w, handler, principal), host, port)
    except OSError as exc:
        print(f"[aegis] MLLP listener could not bind {host}:{port}: {exc}")
        return None
