"""HL7 version 2: the wire format nearly every hospital feed still speaks.

Admissions (ADT), monitor vitals and lab results (ORU), device alarms (ORU^R40):
in a real hospital they arrive as pipe-delimited ER7 text, framed for TCP by MLLP,
from an interface engine that expects an acknowledgement for every message. This is
that codec, written out rather than imported: the server runs under a 512 MB cap
and the subset a hospital feed needs is small.

    MSH|^~\\&|LIS|AEGISMEM|AEGISOS|AEGISMEM|20261004101500||ORU^R01^ORU_R01|LIS0001|P|2.5.1
    PID|1||S123456^^^AEGISMEM^MR||Morgan^A||19610402|F
    OBX|1|NM|CREA^Creatinine^L||132|umol/L|62-106|H|||F|||20261004095100

What it does:

* **parse** ER7 into segments, with the separators read from MSH-1/MSH-2 (a feed may
  choose its own), escape sequences (``\\F\\ \\S\\ \\T\\ \\R\\ \\E\\ \\Xhh\\ \\.br\\``)
  decoded, CR / LF / CRLF segment endings all accepted;
* **build** messages and segments with the values escaped;
* **acknowledge**: ``AA`` / ``AE`` / ``AR`` with an ``ERR`` segment carrying the HL7
  table 0357 condition code, so the sender's interface engine knows whether to resend;
* **MLLP**: frame (``<VT> message <FS><CR>``) and a streaming de-framer that survives
  partial reads, several messages in one read and noise between frames.

What it does not do: validate against a conformance profile, or cover message
structures beyond the ones the integration engine routes (ADT^A01/A02/A03/A04/A08,
ORU^R01, ORU^R40). A structure it does not route is acknowledged ``AR`` with code 200
(unsupported message type), which is the correct answer, not a crash.
"""
from __future__ import annotations

import calendar
import itertools
import re
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, List, Optional, Sequence, Tuple, Union

# HL7 table 0357 — message error condition codes (the ones this engine can raise)
E_ACCEPTED = 0
E_SEGMENT_SEQUENCE = 100
E_REQUIRED_FIELD = 101
E_DATA_TYPE = 102
E_TABLE_VALUE = 103
E_UNSUPPORTED_TYPE = 200
E_UNSUPPORTED_EVENT = 201
E_UNSUPPORTED_PROCESSING = 202
E_UNSUPPORTED_VERSION = 203
E_UNKNOWN_KEY = 204
E_DUPLICATE_KEY = 205
E_INTERNAL = 207
ERROR_TEXT = {
    0: "Message accepted", 100: "Segment sequence error", 101: "Required field missing",
    102: "Data type error", 103: "Table value not found", 200: "Unsupported message type",
    201: "Unsupported event code", 202: "Unsupported processing id", 203: "Unsupported version id",
    204: "Unknown key identifier", 205: "Duplicate key identifier", 207: "Application internal error",
}

VT, FS, CR = b"\x0b", b"\x1c", b"\x0d"
SUPPORTED_VERSIONS = ("2.3", "2.3.1", "2.4", "2.5", "2.5.1", "2.6", "2.7", "2.7.1", "2.8")


class HL7Error(Exception):
    """A message the codec cannot accept. ``code`` is from HL7 table 0357; ``where`` is
    an ERR-2 style location (segment^sequence^field)."""

    def __init__(self, message: str, code: int = E_SEGMENT_SEQUENCE, where: str = ""):
        super().__init__(message)
        self.code = code
        self.where = where


@dataclass(frozen=True)
class Encoding:
    field: str = "|"
    component: str = "^"
    repetition: str = "~"
    escape: str = "\\"
    subcomponent: str = "&"

    @property
    def msh2(self) -> str:
        return self.component + self.repetition + self.escape + self.subcomponent

    # -- escaping ---------------------------------------------------------- #
    def escape_text(self, s: Any) -> str:
        s = "" if s is None else str(s)
        if not s:
            return s
        e = self.escape
        out = []
        for ch in s:
            if ch == e:
                out.append(f"{e}E{e}")
            elif ch == self.field:
                out.append(f"{e}F{e}")
            elif ch == self.component:
                out.append(f"{e}S{e}")
            elif ch == self.subcomponent:
                out.append(f"{e}T{e}")
            elif ch == self.repetition:
                out.append(f"{e}R{e}")
            elif ch in "\r\n":
                out.append(f"{e}.br{e}")
            else:
                out.append(ch)
        return "".join(out)

    def unescape_text(self, s: str) -> str:
        e = self.escape
        if not s or e not in s:
            return s
        out, i, n = [], 0, len(s)
        while i < n:
            ch = s[i]
            if ch != e:
                out.append(ch)
                i += 1
                continue
            j = s.find(e, i + 1)
            if j < 0:                      # a lone escape character: keep it literally
                out.append(s[i:])
                break
            seq = s[i + 1:j]
            if seq == "F":
                out.append(self.field)
            elif seq == "S":
                out.append(self.component)
            elif seq == "T":
                out.append(self.subcomponent)
            elif seq == "R":
                out.append(self.repetition)
            elif seq == "E":
                out.append(e)
            elif seq == ".br":
                out.append("\n")
            elif seq[:1] == "X" and len(seq) > 1 and len(seq) % 2 == 1:
                try:
                    out.append(bytes.fromhex(seq[1:]).decode("latin-1"))
                except ValueError:
                    out.append(s[i:j + 1])
            else:                          # formatting commands we do not render (\H\, \N\, ...)
                pass
            i = j + 1
        return "".join(out)


DEFAULT = Encoding()


@dataclass
class Segment:
    """One segment. ``fields[0]`` is the segment name and ``fields[n]`` is HL7 field n,
    still escaped. For MSH, ``fields[1]`` is the field separator itself and
    ``fields[2]`` the encoding characters, exactly as the standard numbers them."""
    fields: List[str]
    enc: Encoding = DEFAULT

    @property
    def name(self) -> str:
        return self.fields[0]

    def raw(self, n: int) -> str:
        return self.fields[n] if 0 <= n < len(self.fields) else ""

    def reps(self, n: int) -> List[str]:
        r = self.raw(n)
        if self.name == "MSH" and n <= 2:
            return [r] if r else []
        return r.split(self.enc.repetition) if r else []

    def components(self, n: int, rep: int = 1) -> List[str]:
        reps = self.reps(n)
        if rep - 1 >= len(reps):
            return []
        if self.name == "MSH" and n <= 2:
            return [reps[rep - 1]]
        return [self.enc.unescape_text(c) for c in reps[rep - 1].split(self.enc.component)]

    def get(self, n: int, comp: int = 1, rep: int = 1, sub: int = 1) -> str:
        """The unescaped value at field ``n``, component ``comp``, repetition ``rep``,
        subcomponent ``sub`` (all 1-based, HL7 numbering); "" when absent."""
        reps = self.reps(n)
        if rep - 1 >= len(reps):
            return ""
        if self.name == "MSH" and n <= 2:
            return reps[0] if comp == 1 else ""
        comps = reps[rep - 1].split(self.enc.component)
        if comp - 1 >= len(comps):
            return ""
        subs = comps[comp - 1].split(self.enc.subcomponent)
        if sub - 1 >= len(subs):
            return ""
        return self.enc.unescape_text(subs[sub - 1])

    def set(self, n: int, value: Any) -> None:
        while len(self.fields) <= n:
            self.fields.append("")
        self.fields[n] = encode_field(value, self.enc)

    def to_er7(self) -> str:
        if self.name == "MSH":
            return "MSH" + self.enc.field + self.enc.field.join(self.fields[2:])
        return self.enc.field.join(self.fields)


FieldValue = Union[None, str, int, float, Sequence[Any]]


def encode_field(value: FieldValue, enc: Encoding = DEFAULT) -> str:
    """A field from a Python value: a scalar is one escaped value; a tuple is components
    (each a scalar, or a tuple of subcomponents); a list is repetitions of either."""
    def comp(c: Any) -> str:
        if isinstance(c, tuple):
            return enc.subcomponent.join(enc.escape_text(x) for x in c)
        return enc.escape_text(c)

    def one(v: Any) -> str:
        if isinstance(v, tuple):
            return enc.component.join(comp(c) for c in v).rstrip(enc.component)
        if isinstance(v, float):
            return format_number(v)
        return enc.escape_text(v)

    if value is None:
        return ""
    if isinstance(value, list):
        return enc.repetition.join(one(v) for v in value)
    return one(value)


def segment(name: str, *fields: FieldValue, enc: Encoding = DEFAULT) -> Segment:
    """Build a segment; ``fields`` start at field 1. Trailing empty fields are dropped."""
    vals = [encode_field(f, enc) for f in fields]
    while vals and vals[-1] == "":
        vals.pop()
    if name == "MSH":
        return Segment(["MSH", enc.field, enc.msh2] + vals, enc)
    return Segment([name] + vals, enc)


@dataclass
class Message:
    segments: List[Segment]
    enc: Encoding = DEFAULT
    raw_text: str = ""

    # -- parsing ------------------------------------------------------------ #
    @classmethod
    def parse(cls, text: Union[str, bytes]) -> "Message":
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="replace")
        body = text.strip().strip("\x0b\x1c")
        if not body.startswith("MSH"):
            # some senders put a BOM or whitespace first; anything else is not HL7 v2
            body = body.lstrip("﻿ \t")
            if not body.startswith("MSH"):
                raise HL7Error("message does not start with an MSH segment", E_SEGMENT_SEQUENCE, "MSH^1")
        if len(body) < 8:
            raise HL7Error("MSH segment is truncated", E_REQUIRED_FIELD, "MSH^1^2")
        fs = body[3]
        if fs.isalnum() or fs in " \r\n":
            raise HL7Error("MSH-1 field separator is invalid", E_DATA_TYPE, "MSH^1^1")
        end = body.find(fs, 4)
        msh2 = body[4:end] if end > 0 else body[4:]
        if len(msh2) < 4 or len(set(msh2[:4] + fs)) < 5:
            raise HL7Error("MSH-2 encoding characters are invalid", E_DATA_TYPE, "MSH^1^2")
        enc = Encoding(field=fs, component=msh2[0], repetition=msh2[1], escape=msh2[2], subcomponent=msh2[3])
        lines = [ln for ln in re.split(r"\r\n|\r|\n", body) if ln.strip()]
        segs: List[Segment] = []
        for i, ln in enumerate(lines):
            if len(ln) < 3 or not re.match(r"^[A-Z][A-Z0-9]{2}$", ln[:3]) or (len(ln) > 3 and ln[3] != fs):
                raise HL7Error(f"line {i + 1} is not a segment ({ln[:12]!r})", E_SEGMENT_SEQUENCE, f"{ln[:3]}^{i + 1}")
            if ln.startswith("MSH"):
                if i:
                    raise HL7Error("a second MSH inside one message (batch files are not accepted here)",
                                   E_SEGMENT_SEQUENCE, f"MSH^{i + 1}")
                parts = ln.split(fs)
                segs.append(Segment(["MSH", fs] + parts[1:], enc))
            else:
                segs.append(Segment(ln.split(fs), enc))
        msg = cls(segs, enc, body)
        msg._check_header()
        return msg

    def _check_header(self) -> None:
        msh = self.segments[0]
        if not msh.get(9, 1):
            raise HL7Error("MSH-9 message type is missing", E_REQUIRED_FIELD, "MSH^1^9")
        if not msh.get(10):
            raise HL7Error("MSH-10 message control id is missing", E_REQUIRED_FIELD, "MSH^1^10")
        if not msh.get(11, 1):
            raise HL7Error("MSH-11 processing id is missing", E_REQUIRED_FIELD, "MSH^1^11")
        if not msh.get(12, 1):
            raise HL7Error("MSH-12 version id is missing", E_REQUIRED_FIELD, "MSH^1^12")

    # -- the header --------------------------------------------------------- #
    @property
    def msh(self) -> Segment:
        return self.segments[0]

    @property
    def message_type(self) -> Tuple[str, str, str]:
        m = self.msh
        return (m.get(9, 1).upper(), m.get(9, 2).upper(), m.get(9, 3).upper())

    @property
    def type_name(self) -> str:
        code, event, _ = self.message_type
        return f"{code}^{event}" if event else code

    @property
    def control_id(self) -> str:
        return self.msh.get(10)

    @property
    def processing_id(self) -> str:
        return self.msh.get(11, 1).upper()

    @property
    def version(self) -> str:
        return self.msh.get(12, 1)

    @property
    def sending_app(self) -> str:
        return self.msh.get(3, 1)

    @property
    def sending_facility(self) -> str:
        return self.msh.get(4, 1)

    @property
    def receiving_app(self) -> str:
        return self.msh.get(5, 1)

    @property
    def receiving_facility(self) -> str:
        return self.msh.get(6, 1)

    @property
    def timestamp(self) -> Optional[float]:
        return parse_ts(self.msh.get(7, 1))

    def first(self, name: str) -> Optional[Segment]:
        for s in self.segments:
            if s.name == name:
                return s
        return None

    def all(self, name: str) -> List[Segment]:
        return [s for s in self.segments if s.name == name]

    def groups(self, head: str, members: Iterable[str]) -> List[Tuple[Segment, List[Segment]]]:
        """Segments grouped under each ``head`` (e.g. each OBR with the OBX/NTE that follow it)."""
        ms = set(members)
        out: List[Tuple[Segment, List[Segment]]] = []
        for s in self.segments:
            if s.name == head:
                out.append((s, []))
            elif s.name in ms and out:
                out[-1][1].append(s)
        return out

    def to_er7(self) -> str:
        return "\r".join(s.to_er7() for s in self.segments) + "\r"


# --------------------------------------------------------------------------- #
# Timestamps (DTM: YYYY[MM[DD[HH[MM[SS[.S...]]]]]][+/-ZZZZ])
# --------------------------------------------------------------------------- #
_TS = re.compile(r"^(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\.\d{1,4})?([+-]\d{4})?$")


def parse_ts(s: str) -> Optional[float]:
    """Epoch seconds, or None for an empty or malformed value. Without an offset the
    time is taken as UTC (a deployment would set the facility's zone)."""
    s = (s or "").strip()
    if not s:
        return None
    m = _TS.match(s)
    if not m:
        return None
    y, mo, d, hh, mi, ss, frac, off = m.groups()
    try:
        t = calendar.timegm((int(y), int(mo or 1), int(d or 1), int(hh or 0), int(mi or 0), int(ss or 0), 0, 0, 0))
    except (ValueError, OverflowError):
        return None
    if frac:
        t += float(frac)
    if off:
        sign = 1 if off[0] == "+" else -1
        t -= sign * (int(off[1:3]) * 3600 + int(off[3:5]) * 60)
    return float(t)


def format_ts(t: Optional[float] = None, precision: str = "second") -> str:
    t = time.time() if t is None else t
    g = time.gmtime(t)
    s = time.strftime("%Y%m%d%H%M%S", g)
    if precision == "minute":
        s = s[:12]
    return s + "+0000"


# --------------------------------------------------------------------------- #
# Building messages and acknowledgements
# --------------------------------------------------------------------------- #
_ctl = itertools.count(1)


def control_id(prefix: str = "AEG") -> str:
    return f"{prefix}{int(time.time()) % 100000:05d}{next(_ctl):06d}"


def msh(sending_app: str, sending_facility: str, receiving_app: str, receiving_facility: str,
        msg_type: Tuple[str, str, str], ctl: Optional[str] = None, ts: Optional[float] = None,
        processing: str = "P", version: str = "2.5.1", enc: Encoding = DEFAULT) -> Segment:
    return segment("MSH", sending_app, sending_facility, receiving_app, receiving_facility, format_ts(ts), "",
                   msg_type, ctl or control_id(), processing, version, enc=enc)


def build(segments: Sequence[Segment]) -> str:
    return "\r".join(s.to_er7() for s in segments) + "\r"


def ack(msg: Optional[Message], code: str = "AA", text: str = "", error: Optional[HL7Error] = None,
        app: str = "AEGISOS", facility: str = "AEGIS", ts: Optional[float] = None,
        severity: str = "E") -> str:
    """The acknowledgement an interface engine waits for.

    ``AA`` accepted · ``AE`` error (the engine may correct and resend) · ``AR`` rejected
    (do not resend unchanged: unsupported type, wrong version, unknown sender). With an
    error the ERR segment carries the table-0357 code and where it was found."""
    if msg is not None:
        event = msg.message_type[1]
        to_app, to_fac, ctl_in = msg.sending_app, msg.sending_facility, msg.control_id
        version = msg.version if msg.version in SUPPORTED_VERSIONS else "2.5.1"
        enc = msg.enc
    else:
        event, to_app, to_fac, ctl_in, version, enc = "", "", "", "", "2.5.1", DEFAULT
    segs = [msh(app, facility, to_app, to_fac, ("ACK", event, "ACK"), ts=ts, version=version, enc=enc),
            segment("MSA", code, ctl_in, text[:80], enc=enc)]
    if error is not None:
        where = tuple(error.where.split("^")) if error.where else ""
        segs.append(segment("ERR", "", where,
                            (str(error.code), ERROR_TEXT.get(error.code, "Error"), "HL70357"),
                            severity, "", "", "", str(error)[:250], enc=enc))
    return build(segs)


def parse_ack(text: str) -> dict:
    """{code, control_id, text, error} from an ACK — for the sender side and the tests."""
    m = Message.parse(text)
    msa = m.first("MSA")
    err = m.first("ERR")
    return {"code": msa.get(1) if msa else "", "control_id": msa.get(2) if msa else "",
            "text": msa.get(3) if msa else "",
            "error": ({"code": int(err.get(3, 1) or 0), "text": err.get(3, 2), "where": err.raw(2),
                       "severity": err.get(4), "message": err.get(8)} if err else None)}


# --------------------------------------------------------------------------- #
# MLLP framing
# --------------------------------------------------------------------------- #
def frame(message: Union[str, bytes]) -> bytes:
    b = message.encode("utf-8") if isinstance(message, str) else message
    return VT + b + FS + CR


@dataclass
class MLLPDecoder:
    """Streaming de-framer. ``feed`` returns every complete message in the bytes so far
    and keeps the remainder; bytes outside a frame are counted, not fatal."""
    max_frame: int = 1 << 20
    _buf: bytearray = field(default_factory=bytearray)
    noise_bytes: int = 0
    oversize: int = 0

    def feed(self, data: bytes) -> List[str]:
        self._buf.extend(data)
        out: List[str] = []
        while True:
            start = self._buf.find(VT)
            if start < 0:
                self.noise_bytes += len(self._buf)
                self._buf.clear()
                break
            if start:
                self.noise_bytes += start
                del self._buf[:start]
            end = self._buf.find(FS + CR, 1)
            if end < 0:
                if len(self._buf) > self.max_frame:
                    self.oversize += 1
                    self._buf.clear()
                break
            out.append(bytes(self._buf[1:end]).decode("utf-8", errors="replace"))
            del self._buf[:end + 2]
        return out


# --------------------------------------------------------------------------- #
# Small readers for the segments the engine routes
# --------------------------------------------------------------------------- #
def format_number(v: float) -> str:
    """An NM value without exponent notation (``1e-05`` is not a valid HL7 number)."""
    if v != v:
        return ""
    if v == int(v) and abs(v) < 1e15:
        return str(int(v))
    return ("%.6f" % v).rstrip("0").rstrip(".")


def number(s: str) -> Optional[float]:
    """An NM value: a decimal number with optional sign. Comparators (``<0.01``) and
    ranges are not numbers here; the caller decides what to do with them."""
    s = (s or "").strip()
    if not s:
        return None
    try:
        v = float(s)
    except ValueError:
        return None
    if v != v or v in (float("inf"), float("-inf")):
        return None
    return v
